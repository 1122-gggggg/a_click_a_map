"""Relative-motion arbitration and calibration-residual summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RelativePoseTrial:
    seed: int
    threshold_px: float
    inliers: int
    recovered_rotation_deg: float
    m0_rotation_error_deg: float
    edm_rotation_error_deg: float
    m0_translation_direction_error_deg: float | None
    edm_translation_direction_error_deg: float | None


@dataclass(frozen=True)
class SequentialSupport:
    support: str
    trial_count: int
    m0_rotation_median_deg: float
    edm_rotation_median_deg: float
    m0_rotation_ci95_deg: tuple[float, float]
    edm_rotation_ci95_deg: tuple[float, float]
    rotation_stability_p90_deg: float


def _median_bootstrap_ci(values: np.ndarray, *, samples: int = 1000) -> tuple[float, float]:
    generator = np.random.default_rng(0)
    indices = generator.integers(0, len(values), size=(samples, len(values)))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, [2.5, 97.5])
    return float(lower), float(upper)


def classify_sequential_support(
    trials: Sequence[RelativePoseTrial],
    *,
    preferred_max_error_deg: float = 2.0,
    alternative_min_error_deg: float = 5.0,
    maximum_rotation_instability_deg: float = 1.0,
) -> SequentialSupport:
    """Apply the conservative M0-vs-EDM relative-rotation decision gate."""

    if not trials:
        raise ValueError("at least one relative-pose trial is required")
    m0 = np.asarray([trial.m0_rotation_error_deg for trial in trials], dtype=float)
    edm = np.asarray([trial.edm_rotation_error_deg for trial in trials], dtype=float)
    stability = np.asarray([trial.recovered_rotation_deg for trial in trials], dtype=float)
    if not np.isfinite(m0).all() or not np.isfinite(edm).all() or not np.isfinite(stability).all():
        raise ValueError("relative-pose errors must be finite")
    m0_median = float(np.median(m0))
    edm_median = float(np.median(edm))
    m0_ci = _median_bootstrap_ci(m0)
    edm_ci = _median_bootstrap_ci(edm)
    stability_p90 = float(np.percentile(stability, 90))
    stable = stability_p90 <= maximum_rotation_instability_deg
    if (
        stable
        and edm_median <= preferred_max_error_deg
        and m0_median >= alternative_min_error_deg
        and edm_ci[1] < m0_ci[0]
    ):
        support = "EDM"
    elif (
        stable
        and m0_median <= preferred_max_error_deg
        and edm_median >= alternative_min_error_deg
        and m0_ci[1] < edm_ci[0]
    ):
        support = "M0"
    else:
        support = "INDETERMINATE"
    return SequentialSupport(
        support=support,
        trial_count=len(trials),
        m0_rotation_median_deg=m0_median,
        edm_rotation_median_deg=edm_median,
        m0_rotation_ci95_deg=m0_ci,
        edm_rotation_ci95_deg=edm_ci,
        rotation_stability_p90_deg=stability_p90,
    )


def _equal_count_bins(
    coordinate: np.ndarray,
    values: dict[str, np.ndarray],
    count: int,
) -> list[dict[str, float | int]]:
    order = np.argsort(coordinate, kind="stable")
    bins = []
    for indices in np.array_split(order, count):
        if not len(indices):
            continue
        row: dict[str, float | int] = {
            "count": len(indices),
            "coordinate_p50": float(np.median(coordinate[indices])),
        }
        for name, array in values.items():
            row[f"{name}_p50"] = float(np.median(array[indices]))
            row[f"{name}_p90"] = float(np.percentile(array[indices], 90))
        bins.append(row)
    return bins


def residual_structure(
    image_points: np.ndarray,
    residual_vectors: np.ndarray,
    *,
    intrinsics: tuple[float, float, float, float],
    image_size: tuple[int, int],
    radius_bins: int = 10,
    row_bins: int = 16,
) -> dict[str, Any]:
    """Summarize signed residual trends against normalized radius and image row."""

    points = np.asarray(image_points, dtype=float)
    residuals = np.asarray(residual_vectors, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or residuals.shape != points.shape:
        raise ValueError("image points and residual vectors must both have shape (N, 2)")
    if not len(points) or not np.isfinite(points).all() or not np.isfinite(residuals).all():
        raise ValueError("residual diagnostics require finite observations")
    fx, fy, cx, cy = map(float, intrinsics)
    width, height = map(int, image_size)
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0 or radius_bins <= 0 or row_bins <= 0:
        raise ValueError("camera and bin dimensions must be positive")
    normalized = np.column_stack(((points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy))
    radius = np.linalg.norm(normalized, axis=1)
    unit = np.divide(
        normalized,
        radius[:, None],
        out=np.zeros_like(normalized),
        where=radius[:, None] > np.finfo(float).eps,
    )
    radial = np.sum(residuals * unit, axis=1)
    magnitude = np.linalg.norm(residuals, axis=1)
    normalized_row = points[:, 1] / height
    row_slope_x = (
        float(np.polyfit(normalized_row, residuals[:, 0], 1)[0]) if len(points) > 1 else 0.0
    )
    row_slope_y = (
        float(np.polyfit(normalized_row, residuals[:, 1], 1)[0]) if len(points) > 1 else 0.0
    )
    radius_slope = float(np.polyfit(radius, radial, 1)[0]) if np.ptp(radius) > 0 else 0.0
    return {
        "radius_bins": _equal_count_bins(
            radius,
            {"signed_radial": radial, "magnitude": magnitude},
            radius_bins,
        ),
        "row_bins": _equal_count_bins(
            normalized_row,
            {"residual_x": residuals[:, 0], "residual_y": residuals[:, 1], "magnitude": magnitude},
            row_bins,
        ),
        "radius_slope_signed_radial": radius_slope,
        "row_slope_x": row_slope_x,
        "row_slope_y": row_slope_y,
    }
