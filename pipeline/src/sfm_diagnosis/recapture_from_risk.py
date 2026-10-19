"""Convert occupied weak/dead risk voxels into recapture planner pose cells."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .edm_risk.schema import RiskClass, SpatialDiagnostic


def pose_cells_from_risk_rows(
    rows: Sequence[SpatialDiagnostic],
    *,
    localizer: str = "edm-holdout",
    map_producer: str = "gluemap",
    route_tangent: tuple[float, float, float] | None = None,
    map_up_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[SpatialDiagnostic]] = defaultdict(list)
    for row in rows:
        grouped[tuple(float(value) for value in row.position)].append(row)

    cells: list[dict[str, Any]] = []
    for position, group in grouped.items():
        occupied = [row for row in group if int(row.visible_landmarks) > 0]
        if not occupied:
            continue
        risk_class = occupied[0].risk_class
        if risk_class not in {RiskClass.WEAK, RiskClass.DEAD_ZONE, RiskClass.VIEW_DIRECTION_SENSITIVE}:
            continue
        best = min(
            occupied,
            key=lambda row: float(row.failure_probability)
            if row.failure_probability is not None
            else 1.0,
        )
        worst = max(
            occupied,
            key=lambda row: float(row.failure_probability)
            if row.failure_probability is not None
            else 0.0,
        )
        healths = [
            1.0 - float(row.failure_probability)
            for row in occupied
            if row.failure_probability is not None
        ]
        region = f"{risk_class.value}:{position[0]:.3f},{position[1]:.3f},{position[2]:.3f}"
        for row in occupied:
            failure = (
                float(row.failure_probability)
                if row.failure_probability is not None
                else None
            )
            health = None if failure is None else 1.0 - failure
            cells.append(
                {
                    "cell_id": (
                        f"{region}:{row.yaw_deg:.1f}:{row.pitch_deg:.1f}"
                    ),
                    "region_id": region,
                    "position": list(position),
                    "yaw_deg": float(row.yaw_deg),
                    "pitch_deg": float(row.pitch_deg),
                    "localizer": localizer,
                    "map_producer": map_producer,
                    "root_causes": list(row.primary_failure_causes),
                    "directional_health": health,
                    "position_best_health": max(healths) if healths else None,
                    "position_mean_health": (
                        float(np.mean(healths)) if healths else None
                    ),
                    "position_worst_health": min(healths) if healths else None,
                    "route_tangent": list(route_tangent) if route_tangent else None,
                    "map_up_vector": list(map_up_vector),
                    "metrics": _metrics(row, best=best, worst=worst),
                }
            )
    return cells


def _metric(value: Any) -> dict[str, Any]:
    if value is None:
        return {"value": None, "status": "unavailable"}
    return {"value": value, "status": "available"}


def _metrics(
    row: SpatialDiagnostic,
    *,
    best: SpatialDiagnostic,
    worst: SpatialDiagnostic,
) -> dict[str, dict[str, Any]]:
    del best, worst
    fim_rank = 6 if row.fim_lambda_min not in {None, 0.0} else 0
    return {
        "coordinate_scale_status": _metric("map_units"),
        "camera_intrinsics_valid": _metric(True),
        "frame_transform_valid": _metric(True),
        "handedness_valid": _metric(True),
        "visible_landmark_count": _metric(row.visible_landmarks),
        "inlier_convex_hull_coverage": _metric(row.convex_hull_coverage),
        "grid_occupancy_count": _metric(row.grid_occupancy),
        "positive_depth_ratio": _metric(row.positive_depth_ratio),
        "fim_rank": _metric(fim_rank),
        "fim_lambda_min": _metric(row.fim_lambda_min),
        "fim_condition_number": _metric(row.fim_condition),
        "triangulation_angle_p10_deg": _metric(row.parallax_p10_deg),
        "view_direction_entropy": _metric(row.view_entropy),
        "attempt_count": _metric(1),
        "localization_success_rate": _metric(
            None
            if row.failure_probability is None
            else 1.0 - float(row.failure_probability)
        ),
        "holdout_query_coverage": _metric(row.edm_loo_success_rate),
        "track_length_p50": _metric(row.median_track_length),
        "reprojection_error_p90_px": _metric(row.reprojection_error_p90),
    }
