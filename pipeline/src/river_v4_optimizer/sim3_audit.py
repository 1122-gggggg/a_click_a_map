from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * (values @ self.rotation.T) + self.translation


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Estimate target = scale * source @ rotation.T + translation."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("at least three Sim3 correspondences are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= 1e-15:
        raise ValueError("source Sim3 correspondences have zero spatial variance")
    covariance = centered_target.T @ centered_source / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("estimated Sim3 scale is not positive and finite")
    translation = target_mean - scale * (rotation @ source_mean)
    return SimilarityTransform(scale, rotation, translation)


def _robust_span(points: np.ndarray) -> float:
    values = np.asarray(points, dtype=np.float64)
    center = np.median(values, axis=0)
    span = 2.0 * float(np.percentile(np.linalg.norm(values - center, axis=1), 95))
    if not math.isfinite(span) or span <= 1e-12:
        raise ValueError("shared anchor cameras have zero robust spatial span")
    return span


def estimate_robust_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_error: float,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[SimilarityTransform, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("robust Sim3 requires matching source/target points and N >= 3")
    if not math.isfinite(maximum_error) or maximum_error <= 0:
        raise ValueError("robust Sim3 maximum_error must be positive and finite")
    random = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_key = (-1, -math.inf)
    for _ in range(iterations):
        indexes = random.choice(len(source), size=3, replace=False)
        try:
            candidate = estimate_similarity_transform(source[indexes], target[indexes])
        except (ValueError, np.linalg.LinAlgError):
            continue
        errors = np.linalg.norm(candidate.apply(source) - target, axis=1)
        mask = errors <= maximum_error
        count = int(mask.sum())
        median = float(np.median(errors[mask])) if count else math.inf
        key = (count, -median)
        if key > best_key:
            best_key = key
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < 3:
        raise ValueError("robust Sim3 found fewer than three inliers")
    for _ in range(5):
        transform = estimate_similarity_transform(source[best_mask], target[best_mask])
        updated = np.linalg.norm(transform.apply(source) - target, axis=1) <= maximum_error
        if np.array_equal(updated, best_mask) or int(updated.sum()) < 3:
            break
        best_mask = updated
    return estimate_similarity_transform(source[best_mask], target[best_mask]), best_mask


def audit_shared_camera_sim3(
    source_centers: Mapping[str, np.ndarray],
    target_centers: Mapping[str, np.ndarray],
    *,
    holdout_stride: int = 4,
    ransac_max_error_normalized: float = 0.02,
    maximum_holdout_p90_normalized: float = 0.02,
    maximum_holdout_max_normalized: float = 0.05,
) -> dict[str, Any]:
    """Fit Sim3 on deterministic anchors and score untouched shared cameras."""

    if holdout_stride < 2:
        raise ValueError("holdout_stride must be at least two")
    names = sorted(set(source_centers) & set(target_centers))
    holdout_names = names[::holdout_stride]
    holdout = set(holdout_names)
    training_names = [name for name in names if name not in holdout]
    if len(training_names) < 3 or len(holdout_names) < 3:
        raise ValueError("Sim3 audit requires at least three training and three holdout anchors")
    span = _robust_span(np.asarray([target_centers[name] for name in names]))
    source_train = np.asarray([source_centers[name] for name in training_names])
    target_train = np.asarray([target_centers[name] for name in training_names])
    transform, training_mask = estimate_robust_similarity_transform(
        source_train,
        target_train,
        maximum_error=ransac_max_error_normalized * span,
    )

    def errors(selected: list[str]) -> np.ndarray:
        source = np.asarray([source_centers[name] for name in selected])
        target = np.asarray([target_centers[name] for name in selected])
        return np.linalg.norm(transform.apply(source) - target, axis=1) / span

    train_errors = errors(training_names)
    holdout_errors = errors(holdout_names)
    holdout_p90 = float(np.percentile(holdout_errors, 90))
    holdout_max = float(holdout_errors.max())
    passed = (
        holdout_p90 <= maximum_holdout_p90_normalized
        and holdout_max <= maximum_holdout_max_normalized
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "common_anchors": len(names),
        "training_anchors": len(training_names),
        "training_inliers": int(training_mask.sum()),
        "training_inlier_fraction": float(training_mask.mean()),
        "holdout_anchors": len(holdout_names),
        "holdout_names": holdout_names,
        "target_robust_span": span,
        "sim3": {
            "scale": transform.scale,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
        },
        "training_error_normalized": {
            "p50": float(np.percentile(train_errors, 50)),
            "p90": float(np.percentile(train_errors, 90)),
            "max": float(train_errors.max()),
        },
        "holdout_error_normalized": {
            "p50": float(np.percentile(holdout_errors, 50)),
            "p90": holdout_p90,
            "max": holdout_max,
        },
        "thresholds": {
            "ransac_max_error_normalized": ransac_max_error_normalized,
            "holdout_p90_normalized": maximum_holdout_p90_normalized,
            "holdout_max_normalized": maximum_holdout_max_normalized,
        },
    }


def _rotation_error_degrees(predicted: np.ndarray, measured: np.ndarray) -> float:
    predicted_rotation = Rotation.from_matrix(predicted)
    values = []
    for candidate in (measured, measured.T):
        values.append(
            math.degrees(
                (Rotation.from_matrix(candidate).inv() * predicted_rotation).magnitude()
            )
        )
    return float(min(values))


def relative_pose_residual(
    rotation_cw_i: np.ndarray,
    center_i: np.ndarray,
    rotation_cw_j: np.ndarray,
    center_j: np.ndarray,
    measured_rotation: np.ndarray,
    measured_translation_direction: np.ndarray,
) -> dict[str, float]:
    """Compare a measured essential pose with camera poses, convention safely."""

    rotation_i = np.asarray(rotation_cw_i, dtype=np.float64).reshape(3, 3)
    rotation_j = np.asarray(rotation_cw_j, dtype=np.float64).reshape(3, 3)
    predicted_rotation = rotation_j @ rotation_i.T
    direction_world = np.asarray(center_j, dtype=np.float64) - np.asarray(
        center_i, dtype=np.float64
    )
    norm = float(np.linalg.norm(direction_world))
    if norm <= 1e-12:
        raise ValueError("relative-pose camera centers coincide")
    predicted_translation = rotation_i @ (direction_world / norm)
    measured_translation = np.asarray(measured_translation_direction, dtype=np.float64).reshape(3)
    measured_norm = float(np.linalg.norm(measured_translation))
    if measured_norm <= 1e-12:
        raise ValueError("measured translation direction has zero norm")
    measured_translation /= measured_norm
    cosine = float(np.clip(abs(predicted_translation @ measured_translation), -1.0, 1.0))
    return {
        "rotation_deg": _rotation_error_degrees(
            predicted_rotation, np.asarray(measured_rotation, dtype=np.float64).reshape(3, 3)
        ),
        "translation_axis_deg": math.degrees(math.acos(cosine)),
    }


def transform_camera_pose_to_target_gauge(
    rotation_cw: np.ndarray,
    center: np.ndarray,
    transform: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(rotation_cw, dtype=np.float64) @ transform.rotation.T,
        transform.apply(np.asarray(center, dtype=np.float64)),
    )


__all__ = [
    "SimilarityTransform",
    "audit_shared_camera_sim3",
    "estimate_similarity_transform",
    "estimate_robust_similarity_transform",
    "relative_pose_residual",
    "transform_camera_pose_to_target_gauge",
]
