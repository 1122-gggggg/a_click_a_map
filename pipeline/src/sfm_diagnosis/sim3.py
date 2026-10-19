"""Numerically checked 3D similarity transforms for cross-map validation.

Fit and validation never share the same anchors.  Residuals are relative to
the target-anchor span so neither reconstruction is treated as metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    """A transform from source-world coordinates to target-world coordinates."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform one or more points with shape ``(N, 3)``."""

        values = _points(points, name="points", minimum_count=1)
        return self.scale * (self.rotation @ values.T).T + self.translation


def _points(values: np.ndarray, *, name: str, minimum_count: int) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_count:
        raise ValueError(f"{name} must have shape (N, 3) with N >= {minimum_count}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def _assert_disjoint_anchors(
    support_source: np.ndarray,
    support_target: np.ndarray,
    holdout_source: np.ndarray,
    holdout_target: np.ndarray,
) -> None:
    """Refuse to validate a correspondence that was also used to fit."""

    support = np.hstack((support_source, support_target))
    holdout = np.hstack((holdout_source, holdout_target))
    for row in holdout:
        if np.any(np.all(np.isclose(support, row, rtol=0.0, atol=0.0), axis=1)):
            raise ValueError(
                "holdout anchors must be disjoint from support; "
                "never fit and validate on the same points"
            )


def estimate_similarity_3d(source: np.ndarray, target: np.ndarray) -> SimilarityTransform:
    """Estimate target = scale * rotation * source + translation.

    The estimate uses the least-squares Umeyama solution.  It deliberately
    rejects collinear centers because they do not constrain a unique 3D
    rotation and would make a cross-map admission result ambiguous.
    """

    source_points = _points(source, name="source", minimum_count=3)
    target_points = _points(target, name="target", minimum_count=3)
    if source_points.shape != target_points.shape:
        raise ValueError("source and target must have identical shapes")

    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("anchor centers must be non-collinear")

    covariance = target_centered.T @ source_centered / len(source_points)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transpose) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transpose
    source_variance = float(np.sum(source_centered**2) / len(source_points))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("source anchor centers have zero variance")
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("estimated similarity scale must be finite and positive")
    translation = target_center - scale * rotation @ source_center
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def evaluate_holdout_similarity(
    support_source: np.ndarray,
    support_target: np.ndarray,
    holdout_source: np.ndarray,
    holdout_target: np.ndarray,
    *,
    max_relative_residual: float,
) -> dict[str, object]:
    """Fit from support anchors and validate against independent holdouts.

    Residuals are normalized by the maximum target-anchor separation, avoiding
    a claim that either reconstruction's coordinates are metric.
    """

    if not np.isfinite(max_relative_residual) or max_relative_residual <= 0.0:
        raise ValueError("max_relative_residual must be finite and positive")
    support = _points(support_source, name="support_source", minimum_count=3)
    support_targets = _points(support_target, name="support_target", minimum_count=3)
    heldout = _points(holdout_source, name="holdout_source", minimum_count=1)
    heldout_targets = _points(holdout_target, name="holdout_target", minimum_count=1)
    if support.shape != support_targets.shape:
        raise ValueError("support source and target must have identical shapes")
    if heldout.shape != heldout_targets.shape:
        raise ValueError("holdout source and target must have identical shapes")
    _assert_disjoint_anchors(support, support_targets, heldout, heldout_targets)

    transform = estimate_similarity_3d(support, support_targets)
    residuals = np.linalg.norm(transform.apply(heldout) - heldout_targets, axis=1)
    all_targets = np.vstack((support_targets, heldout_targets))
    separations = np.linalg.norm(
        all_targets[:, np.newaxis, :] - all_targets[np.newaxis, :, :],
        axis=2,
    )
    target_span = float(np.max(separations))
    if target_span <= np.finfo(np.float64).eps:
        raise ValueError("target anchor centers have zero span")
    relative_residuals = residuals / target_span
    return {
        "transform": transform,
        "holdout_residuals": residuals.tolist(),
        "holdout_relative_residuals": relative_residuals.tolist(),
        "target_anchor_span": target_span,
        "max_relative_residual": max_relative_residual,
        "consistent": bool(np.all(relative_residuals <= max_relative_residual)),
    }


def evaluate_leave_one_out_similarity(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_relative_residual: float,
) -> dict[str, object]:
    """Fit every anchor from the remaining anchors and validate that holdout."""

    source_points = _points(source, name="source", minimum_count=4)
    target_points = _points(target, name="target", minimum_count=4)
    if source_points.shape != target_points.shape:
        raise ValueError("source and target must have identical shapes")

    folds: list[dict[str, object]] = []
    for heldout_index in range(len(source_points)):
        support_mask = np.arange(len(source_points)) != heldout_index
        try:
            evaluation = evaluate_holdout_similarity(
                source_points[support_mask],
                target_points[support_mask],
                source_points[~support_mask],
                target_points[~support_mask],
                max_relative_residual=max_relative_residual,
            )
        except ValueError as error:
            folds.append(
                {
                    "consistent": False,
                    "error": str(error),
                    "heldout_index": heldout_index,
                }
            )
        else:
            folds.append({"heldout_index": heldout_index, **evaluation})

    return {
        "consistent": all(bool(fold["consistent"]) for fold in folds),
        "folds": folds,
        "max_relative_residual": max_relative_residual,
    }


__all__ = [
    "SimilarityTransform",
    "estimate_similarity_3d",
    "evaluate_holdout_similarity",
    "evaluate_leave_one_out_similarity",
]
