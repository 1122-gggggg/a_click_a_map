"""Bearing-space rotation-only and derotated-parallax diagnostics.

Homography support is auxiliary evidence: planar translation can also be
homography-dominant.  Rotation-dominated classification is therefore based on
angular residuals over a sufficient temporal baseline and their spatial spread.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _camera_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("K must be a finite 3x3 camera matrix")
    determinant = float(np.linalg.det(matrix))
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("K must be invertible")
    return matrix


def _pixels(value: Any) -> np.ndarray:
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] not in {2, 3}:
        raise ValueError("pixels must have shape (N,2) or (N,3)")
    if not np.isfinite(points).all():
        raise ValueError("pixels must be finite")
    if points.shape[1] == 2:
        points = np.column_stack((points, np.ones(len(points), dtype=float)))
    elif np.any(np.abs(points[:, 2]) < 1e-12):
        raise ValueError("homogeneous pixels must have non-zero scale")
    return points


def bearing_vectors(pixels: Any, K: np.ndarray) -> np.ndarray:
    """Convert pixels to unit camera-frame bearing vectors."""

    matrix = _camera_matrix(K)
    homogeneous = _pixels(pixels)
    rays = np.linalg.solve(matrix, homogeneous.T).T
    norms = np.linalg.norm(rays, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise FloatingPointError("pixel normalization produced an invalid bearing")
    return rays / norms[:, None]


def inlier_correspondences(
    points_a: Any,
    points_b: Any,
    inlier_mask: Any,
    *,
    minimum_count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select one verified correspondence subset without silently reshaping masks."""

    first = np.asarray(points_a, dtype=float)
    second = np.asarray(points_b, dtype=float)
    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2:
        raise ValueError("matched points must have equal shape (N,2)")
    if len(mask) != len(first):
        raise ValueError("inlier mask length differs from correspondences")
    if int(mask.sum()) < max(1, int(minimum_count)):
        raise ValueError("insufficient verified correspondences")
    return first[mask], second[mask]


def _wahba_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) < 3:
        raise ValueError("rotation-only fit requires at least three correspondences")
    covariance = target.T @ source
    u, _singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rotation = u @ correction @ vt
    if not np.isfinite(rotation).all() or abs(float(np.linalg.det(rotation)) - 1.0) > 1e-6:
        raise FloatingPointError("rotation-only fit produced an invalid rotation")
    return rotation


def _angular_errors_deg(source: np.ndarray, target: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    predicted = (rotation @ source.T).T
    cosine = np.clip(np.sum(target * predicted, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def estimate_rotation_only(
    source_bearings: np.ndarray,
    target_bearings: np.ndarray,
    *,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic robust Wahba fit and its retained inlier mask."""

    source = np.asarray(source_bearings, dtype=float)
    target = np.asarray(target_bearings, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("bearing arrays must have matching shape (N,3)")
    mask = np.ones(len(source), dtype=bool)
    rotation = _wahba_rotation(source, target)
    for _ in range(max(0, int(robust_iterations))):
        errors = _angular_errors_deg(source, target, rotation)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = median + max(3.0 * 1.4826 * mad, 0.05)
        candidate = errors <= threshold
        if int(candidate.sum()) < 3 or np.array_equal(candidate, mask):
            mask = candidate if int(candidate.sum()) >= 3 else mask
            break
        mask = candidate
        rotation = _wahba_rotation(source[mask], target[mask])
    return rotation, mask


def derotated_parallax_errors(
    points_a: Any,
    points_b: Any,
    K: np.ndarray,
    rotation_ab: np.ndarray,
) -> np.ndarray:
    """Evaluate unseen correspondences after rotating target bearings back."""

    bearings_a = bearing_vectors(points_a, K)
    bearings_b = bearing_vectors(points_b, K)
    rotation = np.asarray(rotation_ab, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_ab must be a finite 3x3 matrix")
    return _angular_errors_deg(bearings_a, bearings_b, rotation)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile))


def rotation_only_diagnostics(
    points_a: Any,
    points_b: Any,
    K: np.ndarray,
    image_size: tuple[int, int],
    translation_threshold_deg: float = 1.0,
) -> dict[str, Any]:
    """Fit ``f_b ~= R f_a`` and measure rotation-compensated angular motion."""

    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    threshold = float(translation_threshold_deg)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("translation_threshold_deg must be positive and finite")
    pixels_a = np.asarray(points_a, dtype=float)
    pixels_b = np.asarray(points_b, dtype=float)
    if pixels_a.shape != pixels_b.shape or len(pixels_a) < 3:
        raise ValueError("matched pixel arrays must have matching length >= 3")
    bearings_a = bearing_vectors(pixels_a, K)
    bearings_b = bearing_vectors(pixels_b, K)
    rotation, fit_mask = estimate_rotation_only(bearings_a, bearings_b)
    errors = _angular_errors_deg(bearings_a, bearings_b, rotation)
    # angle(f_b, R f_a) == angle(f_a, R.T f_b); retain both names for
    # physical interpretation but never double-count them in a score.
    beta = errors.copy()

    grid_values: dict[tuple[int, int], list[float]] = defaultdict(list)
    for point, value in zip(pixels_a[:, :2], beta, strict=True):
        col = int(np.clip(math.floor(float(point[0]) / width * 4.0), 0, 3))
        row = int(np.clip(math.floor(float(point[1]) / height * 4.0), 0, 3))
        grid_values[(row, col)].append(float(value))
    grid: list[list[float | None]] = [[None for _ in range(4)] for _ in range(4)]
    for (row, col), values in grid_values.items():
        grid[row][col] = float(np.median(np.asarray(values, dtype=float)))
    supported_cells = [value for row in grid for value in row if value is not None]
    high_cells = [value for value in supported_cells if value > threshold]
    high_fraction = len(high_cells) / len(supported_cells) if supported_cells else 0.0
    ratio_above = float(np.mean(beta > threshold))
    rotation_cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_magnitude = float(np.degrees(np.arccos(rotation_cosine)))

    if _percentile(beta, 90) > threshold or ratio_above >= 0.2 or high_fraction > 0.1:
        classification = "TRANSLATION_EVIDENCE"
    elif _percentile(beta, 50) <= 0.5 * threshold:
        classification = "PURE_ROTATION"
    else:
        classification = "AMBIGUOUS_LOW_PARALLAX"
    return {
        "rotation_matrix": rotation.tolist(),
        "correspondence_count": len(beta),
        "rotation_fit_inlier_count": int(fit_mask.sum()),
        "rotation_magnitude_deg": rotation_magnitude,
        "rotation_residual_p50_deg": _percentile(errors, 50),
        "rotation_residual_p90_deg": _percentile(errors, 90),
        "derotated_beta_p10_deg": _percentile(beta, 10),
        "derotated_beta_p50_deg": _percentile(beta, 50),
        "derotated_beta_p90_deg": _percentile(beta, 90),
        "beta_above_threshold_ratio": ratio_above,
        "translation_threshold_deg": threshold,
        "grid_beta_p50_deg": grid,
        "high_beta_grid_fraction": high_fraction,
        "classification": classification,
        "homography_dominant": False,
        "residual_identity": "rotation_residual_equals_derotated_beta_angle",
    }


def aggregate_temporal_rotation_diagnostics(
    rows: Iterable[Mapping[str, Any]],
    fps: float,
    min_baseline_seconds: float = 0.5,
    translation_threshold_deg: float = 1.0,
) -> dict[str, Any]:
    """Require a non-adjacent temporal baseline and retain spatial beta evidence."""

    rate = float(fps)
    minimum = float(min_baseline_seconds)
    threshold = float(translation_threshold_deg)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("fps must be positive and finite")
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("min_baseline_seconds must be positive and finite")
    long_rows = []
    for raw in rows:
        gap = float(raw.get("gap", raw.get("frame_gap", 0.0)) or 0.0)
        row_rate = float(raw.get("fps", rate) or rate)
        baseline = float(raw.get("baseline_seconds", gap / row_rate))
        if baseline + 1e-12 >= minimum:
            long_rows.append(dict(raw))
    if not long_rows:
        return {
            "classification": "INSUFFICIENT_TEMPORAL_BASELINE",
            "long_baseline_count": 0,
            "global_beta_p50_deg": None,
            "global_beta_p90_deg": None,
            "high_beta_grid_fraction": 0.0,
            "rotation_rate_p50_deg_s": None,
            "rotation_dominated_duration_seconds": 0.0,
        }

    beta_samples: list[float] = []
    p90_samples: list[float] = []
    grid: dict[tuple[int, int], list[float]] = defaultdict(list)
    direct_high_fractions: list[float] = []
    rotation_rates: list[float] = []
    baseline_durations: list[float] = []
    for row in long_rows:
        gap = float(row.get("gap", row.get("frame_gap", 0.0)) or 0.0)
        row_rate = float(row.get("fps", rate) or rate)
        baseline = float(row.get("baseline_seconds", gap / row_rate))
        baseline_durations.append(baseline)
        rotation_degrees = row.get(
            "rotation_magnitude_deg", row.get("rotation_degrees")
        )
        if rotation_degrees is not None and baseline > 0.0:
            value = float(rotation_degrees) / baseline
            if math.isfinite(value):
                rotation_rates.append(value)
        if row.get("high_beta_grid_fraction") is not None:
            value = float(row["high_beta_grid_fraction"])
            if math.isfinite(value):
                direct_high_fractions.append(max(0.0, min(1.0, value)))
        if row.get("derotated_beta_deg") is not None:
            value = float(row["derotated_beta_deg"])
            if math.isfinite(value):
                beta_samples.append(value)
                if row.get("grid_row") is not None and row.get("grid_col") is not None:
                    grid[(int(row["grid_row"]), int(row["grid_col"]))].append(value)
        elif row.get("derotated_beta_p50_deg") is not None:
            value = float(row["derotated_beta_p50_deg"])
            if math.isfinite(value):
                beta_samples.append(value)
        if row.get("derotated_beta_p90_deg") is not None:
            value = float(row["derotated_beta_p90_deg"])
            if math.isfinite(value):
                p90_samples.append(value)
    if not beta_samples:
        return {
            "classification": "UNAVAILABLE",
            "long_baseline_count": len(long_rows),
            "global_beta_p50_deg": None,
            "global_beta_p90_deg": None,
            "high_beta_grid_fraction": 0.0,
            "rotation_rate_p50_deg_s": (
                float(np.median(rotation_rates)) if rotation_rates else None
            ),
            "rotation_dominated_duration_seconds": 0.0,
        }
    cell_medians = [float(np.median(values)) for values in grid.values() if values]
    high_fraction = (
        sum(value > threshold for value in cell_medians) / len(cell_medians)
        if cell_medians
        else 0.0
    )
    if direct_high_fractions:
        high_fraction = max(high_fraction, max(direct_high_fractions))
    global_p50 = float(np.percentile(np.asarray(beta_samples), 50))
    global_p90 = (
        float(np.percentile(np.asarray(p90_samples), 90))
        if p90_samples
        else float(np.percentile(np.asarray(beta_samples), 90))
    )
    if high_fraction > 0.1 or global_p50 > threshold or global_p90 > threshold:
        classification = "TRANSLATION_EVIDENCE"
    elif global_p50 <= 0.5 * threshold:
        classification = "ROTATION_DOMINATED"
    else:
        classification = "AMBIGUOUS_LOW_PARALLAX"
    return {
        "classification": classification,
        "long_baseline_count": len(long_rows),
        "global_beta_p50_deg": global_p50,
        "global_beta_p90_deg": global_p90,
        "high_beta_grid_fraction": high_fraction,
        "rotation_rate_p50_deg_s": (
            float(np.median(rotation_rates)) if rotation_rates else None
        ),
        "rotation_dominated_duration_seconds": (
            max(baseline_durations)
            if classification == "ROTATION_DOMINATED" and baseline_durations
            else 0.0
        ),
    }
