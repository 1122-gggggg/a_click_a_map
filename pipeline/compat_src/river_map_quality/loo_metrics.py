"""Pure NumPy leave-one-out pose-quality diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from river_map_quality.pose_information import (
    FIM_JACOBIAN_CONVENTION,
    FIM_STATE_PARAMETERIZATION,
    decompose_pose_information,
    fim_scaling_convention,
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hull_area(points: np.ndarray) -> float:
    unique = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(unique) < 3:
        return 0.0

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon = lower[:-1] + upper[:-1]
    return (
        abs(
            sum(
                polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
                - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
                for index in range(len(polygon))
            )
        )
        / 2.0
    )


def _invalid() -> dict[str, Any]:
    return {"status": "invalid", "position_error": None, "rotation_error_deg": None}
def _effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.asarray(eigenvalues, dtype=float)
    positive = positive[positive > 0]
    if not len(positive):
        return 0.0
    probabilities = positive / positive.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _information_summary(
    *,
    state: str,
    rank: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    trace: float | None = None,
    effective_rank: float | None = None,
    condition_number: float | None = None,
    log_determinant: float | None = None,
    covariance_diag: list[float | None] | None = None,
    covariance: list[list[float | None]] | None = None,
    scene_scale: float | None = None,
    pixel_sigma: float | None = None,
) -> dict[str, Any]:
    """Serialize the one canonical visual-information result without recomputation."""

    if state != "COMPUTED":
        return {"state": state}
    if scene_scale is None or pixel_sigma is None:
        raise ValueError("computed information summary requires scale and pixel sigma")
    return {
        "state": "COMPUTED",
        "parameterization": FIM_STATE_PARAMETERIZATION,
        "jacobian_convention": FIM_JACOBIAN_CONVENTION,
        "scaling": fim_scaling_convention(
            scene_scale=scene_scale,
            pixel_sigma=pixel_sigma,
        ),
        "rank": rank,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "trace": trace,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "log_determinant": log_determinant,
        "covariance_diag": covariance_diag,
        "covariance": covariance,
        "degenerate": rank is None or rank < 6,
    }


def compute_loo_metrics(
    world_points: np.ndarray,
    image_points: np.ndarray,
    inlier_mask: np.ndarray,
    intrinsics: tuple[float, float, float, float] | dict[str, float],
    estimated_pose: np.ndarray,
    ground_truth_pose: np.ndarray,
    reference_sources: np.ndarray | Sequence[Any] | None = None,
    *,
    image_size: tuple[int, int],
    pixel_sigma: float = 1.0,
    fim_scene_scale: float | None = None,
) -> dict[str, Any]:
    """Measure one PnP result using world-to-camera 4x4 pose matrices.

    ``pose_information`` is the authoritative visual/prior/posterior decomposition.
    The existing ``scaled_*`` scalar fields are its visual summary projection, retained
    for current report consumers rather than a separate posterior estimate.
    """

    try:
        world = np.asarray(world_points, dtype=float)
        image = np.asarray(image_points, dtype=float)
        mask = np.asarray(inlier_mask, dtype=bool)
        estimated = np.asarray(estimated_pose, dtype=float)
        ground_truth = np.asarray(ground_truth_pose, dtype=float)
        if world.ndim != 2 or world.shape[1] != 3:
            raise ValueError
        if image.shape != (len(world), 2) or mask.shape != (len(world),):
            raise ValueError
        if estimated.shape != (4, 4) or ground_truth.shape != (4, 4):
            raise ValueError
        if not np.isfinite(estimated).all() or not np.isfinite(ground_truth).all():
            raise ValueError
        if isinstance(intrinsics, dict):
            fx, fy, cx, cy = (float(intrinsics[key]) for key in ("fx", "fy", "cx", "cy"))
        else:
            fx, fy, cx, cy = map(float, intrinsics)
        width, height = map(int, image_size)
        if fx <= 0 or fy <= 0 or pixel_sigma <= 0 or width <= 0 or height <= 0:
            raise ValueError
        if reference_sources is not None and len(reference_sources) != len(world):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return _invalid()

    valid = np.isfinite(world).all(axis=1) & np.isfinite(image).all(axis=1)
    selected = valid & mask
    estimated_rotation = estimated[:3, :3]
    estimated_translation = estimated[:3, 3]
    truth_rotation = ground_truth[:3, :3]
    truth_translation = ground_truth[:3, 3]
    estimated_center = -estimated_rotation.T @ estimated_translation
    truth_center = -truth_rotation.T @ truth_translation
    rotation_delta = estimated_rotation @ truth_rotation.T

    output: dict[str, Any] = {
        "status": "ok" if selected.any() else "failed",
        "inlier_ratio": _finite(np.count_nonzero(selected) / len(world)) if len(world) else None,
        "position_error": _finite(np.linalg.norm(estimated_center - truth_center)),
        "rotation_error_deg": _finite(
            np.degrees(np.arccos(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)))
        ),
        "reprojection_p50": None,
        "reprojection_median": None,
        "reprojection_p90": None,
        "reprojection_p95": None,
        "reprojection_max": None,
        "reprojection_mean": None,
        "positive_depth_ratio": None,
        "convex_hull_coverage": None,
        "occupancy_4x4": None,
        "scaled_fim_rank": None,
        "scaled_fim_lambda_min": None,
        "scaled_fim_lambda_max": None,
        "scaled_fim_trace": None,
        "scaled_fim_effective_rank": None,
        "scaled_fim_condition": None,
        "scaled_fim_logdet": None,
        "scaled_pose_covariance_diag": None,
        "scaled_pose_covariance": None,
        "scaled_fim_degenerate": None,
        "scene_scale": None,
        "pose_information": {
            "visual": _information_summary(state="NOT_COMPUTED_INSUFFICIENT_VISUAL_SUPPORT"),
            "prior": {
                "state": "ABSENT",
                "coordinate_requirement": FIM_STATE_PARAMETERIZATION,
            },
            "posterior": {
                "state": "NOT_COMPUTED_INSUFFICIENT_VISUAL_SUPPORT",
                "matrix_source": None,
            },
        },
        "reference_independent_count": None,
        "reference_top1_share": None,
        "reference_entropy": None,
        "reference_entropy_normalized": None,
        "inliers_by_source": None,
    }
    if not selected.any():
        return output

    camera_points = (estimated_rotation @ world.T).T + estimated_translation
    selected_camera = camera_points[selected]
    positive = selected_camera[:, 2] > 1e-9
    output["positive_depth_ratio"] = _finite(np.mean(positive))

    selected_image = image[selected]
    if positive.any():
        positive_camera = selected_camera[positive]
        positive_image = selected_image[positive]
        projection = np.column_stack(
            (
                fx * positive_camera[:, 0] / positive_camera[:, 2] + cx,
                fy * positive_camera[:, 1] / positive_camera[:, 2] + cy,
            )
        )
        errors = np.linalg.norm(projection - positive_image, axis=1)
        output["reprojection_p50"] = _finite(np.percentile(errors, 50))
        output["reprojection_median"] = output["reprojection_p50"]
        output["reprojection_p90"] = _finite(np.percentile(errors, 90))
        output["reprojection_p95"] = _finite(np.percentile(errors, 95))
        output["reprojection_max"] = _finite(np.max(errors))
        output["reprojection_mean"] = _finite(np.mean(errors))

    in_bounds = (
        (selected_image[:, 0] >= 0)
        & (selected_image[:, 0] < width)
        & (selected_image[:, 1] >= 0)
        & (selected_image[:, 1] < height)
    )
    bounded = selected_image[in_bounds]
    output["convex_hull_coverage"] = float(_hull_area(bounded) / (width * height))
    if len(bounded):
        cell_x = np.clip((bounded[:, 0] * 4 / width).astype(int), 0, 3)
        cell_y = np.clip((bounded[:, 1] * 4 / height).astype(int), 0, 3)
        output["occupancy_4x4"] = len(set(zip(cell_x.tolist(), cell_y.tolist(), strict=True)))
    else:
        output["occupancy_4x4"] = 0

    try:
        decomposition = decompose_pose_information(
            selected_camera,
            fx=fx,
            fy=fy,
            pixel_sigma=pixel_sigma,
            scene_scale=fim_scene_scale,
        )
        visual = decomposition.visual
        covariance = (
            [[_finite(value) for value in row] for row in visual.covariance]
            if visual.rank == 6
            else None
        )
        visual_summary = _information_summary(
            state="COMPUTED",
            rank=visual.rank,
            lambda_min=_finite(visual.lambda_min),
            lambda_max=_finite(visual.lambda_max),
            trace=_finite(np.trace(visual.matrix)),
            effective_rank=_finite(_effective_rank(visual.eigenvalues)),
            condition_number=_finite(visual.condition_number),
            log_determinant=_finite(visual.log_determinant),
            covariance_diag=[_finite(value) for value in np.diag(visual.covariance)],
            covariance=covariance,
            scene_scale=_finite(visual.scene_scale),
            pixel_sigma=pixel_sigma,
        )
        output.update(
            {
                "scaled_fim_rank": visual.rank,
                "scaled_fim_lambda_min": _finite(visual.lambda_min),
                "scaled_fim_lambda_max": _finite(visual.lambda_max),
                "scaled_fim_trace": _finite(np.trace(visual.matrix)),
                "scaled_fim_effective_rank": _finite(_effective_rank(visual.eigenvalues)),
                "scaled_fim_condition": _finite(visual.condition_number),
                "scaled_fim_logdet": _finite(visual.log_determinant),
                "scaled_pose_covariance_diag": [
                    _finite(value) for value in np.diag(visual.covariance)
                ],
                "scaled_pose_covariance": covariance,
                "scaled_fim_degenerate": visual.rank < 6,
                "scene_scale": _finite(visual.scene_scale),
                "pose_information": {
                    "visual": visual_summary,
                    "prior": {
                        "state": "ABSENT",
                        "coordinate_requirement": FIM_STATE_PARAMETERIZATION,
                    },
                    "posterior": {
                        "state": "IDENTICAL_TO_VISUAL_NO_PRIOR",
                        "matrix_source": "visual",
                    },
                },
            }
        )
    except ValueError:
        output["scaled_fim_degenerate"] = True
        output["pose_information"] = {
            "visual": _information_summary(state="NOT_COMPUTED_INVALID_VISUAL_GEOMETRY"),
            "prior": {
                "state": "ABSENT",
                "coordinate_requirement": FIM_STATE_PARAMETERIZATION,
            },
            "posterior": {
                "state": "NOT_COMPUTED_INVALID_VISUAL_GEOMETRY",
                "matrix_source": None,
            },
        }

    if reference_sources is not None:
        selected_sources = [
            str(source) for source, keep in zip(reference_sources, selected, strict=True) if keep
        ]
        counts = Counter(selected_sources)
        count = len(selected_sources)
        if count:
            probabilities = np.asarray(list(counts.values()), dtype=float) / count
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
            output["reference_independent_count"] = len(counts)
            output["reference_top1_share"] = float(max(counts.values()) / count)
            output["reference_entropy"] = entropy
            output["reference_entropy_normalized"] = (
                float(entropy / math.log(len(counts))) if len(counts) > 1 else 0.0
            )
            output["inliers_by_source"] = dict(sorted(counts.items()))
    return output


loo_pose_metrics = compute_loo_metrics
