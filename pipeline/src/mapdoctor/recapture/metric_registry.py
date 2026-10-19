from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .profiles import (
    COMMON_DIAGNOSTIC,
    COMMON_INTEGRITY,
    COMMON_RECAPTURE_AUTHORIZATION,
    COMMON_RECOMMENDED,
    DEFAULT_PROFILE,
)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    category: str = "diagnostic"
    description: str = ""
    hard_gate: bool = False
    minimum: float | None = None
    maximum: float | None = None
    allow_estimated: bool = True
    aliases: tuple[str, ...] = ()


# Domain checks are deliberately strict only for evidence that can participate
# in recapture authorization. Threshold pass/fail belongs to planner policy,
# not to this schema-level validation layer.
_HARD: dict[str, tuple[float | None, float | None, bool]] = {
    "visible_landmark_count": (0.0, None, False),
    "inlier_convex_hull_coverage": (0.0, 1.0, False),
    "grid_occupancy_count": (0.0, None, False),
    "positive_depth_ratio": (0.0, 1.0, False),
    "fim_rank": (0.0, 6.0, False),
    "fim_lambda_min": (0.0, None, False),
    "fim_condition_number": (1.0, None, False),
    "triangulation_angle_p10_deg": (0.0, None, False),
    "view_direction_entropy": (0.0, 1.0, False),
    "attempt_count": (0.0, None, False),
    "localization_success_rate": (0.0, 1.0, False),
    "holdout_query_coverage": (0.0, 1.0, False),
    "existing_data_repairability": (0.0, 1.0, False),
}

_SPECIAL = {
    "coordinate_scale_status": MetricSpec(
        "coordinate_scale_status",
        "integrity",
        "explicit metric/map-unit declaration",
        True,
        allow_estimated=False,
    ),
    "camera_intrinsics_valid": MetricSpec(
        "camera_intrinsics_valid",
        "integrity",
        "camera intrinsics validated",
        True,
        allow_estimated=False,
    ),
    "frame_transform_valid": MetricSpec(
        "frame_transform_valid",
        "integrity",
        "runtime/map frame transform validated",
        True,
        allow_estimated=False,
    ),
    "handedness_valid": MetricSpec(
        "handedness_valid",
        "integrity",
        "coordinate handedness validated",
        True,
        allow_estimated=False,
    ),
    "existing_data_counterfactual_complete": MetricSpec(
        "existing_data_counterfactual_complete",
        "authorization",
        "all predeclared existing-data repair counterfactuals were evaluated on frozen weak/stable holdouts",
        True,
        allow_estimated=False,
    ),
}

_names = set(COMMON_INTEGRITY + COMMON_RECAPTURE_AUTHORIZATION + COMMON_DIAGNOSTIC + COMMON_RECOMMENDED)
_names.update({"existing_data_repairability", "existing_data_counterfactual_complete"})
_names.update(DEFAULT_PROFILE.diagnostic_metrics)
_names.update(DEFAULT_PROFILE.recommended_metrics)

_specs: list[MetricSpec] = []
for name in sorted(_names):
    if name in _SPECIAL:
        _specs.append(_SPECIAL[name])
    elif name in _HARD:
        minimum, maximum, allow_estimated = _HARD[name]
        _specs.append(
            MetricSpec(
                name,
                "authorization",
                name.replace("_", " "),
                True,
                minimum,
                maximum,
                allow_estimated,
            )
        )
    else:
        _specs.append(MetricSpec(name, "diagnostic", name.replace("_", " ")))

METRICS = tuple(_specs)
METRIC_BY_NAME = {spec.name: spec for spec in METRICS}
ALIASES = {
    "convex_hull_coverage": "inlier_convex_hull_coverage",
    "hull_coverage": "inlier_convex_hull_coverage",
    "reprojection_error_p50": "reprojection_error_p50_px",
    "reprojection_error_p90": "reprojection_error_p90_px",
    "track_length_median": "track_length_p50",
    "reprojection_error_median_px": "reprojection_error_p50_px",
    "largest_covisibility_component_ratio": "largest_component_ratio",
}


def canonical_metric_name(name: str) -> str:
    return ALIASES.get(name, name)


def required_metrics(
    localizer: str | None = None,
    *,
    hard_only: bool = False,
    map_producer: str | None = None,
) -> tuple[MetricSpec, ...]:
    """Return method-agnostic metrics and optional map-producer diagnostics.

    GLUEMAP-only diagnostics are not reported as missing for COLMAP/GLOMAP
    maps. This function intentionally distinguishes map producer from the
    downstream localization method. ``localizer`` is provenance only.
    """
    profile = DEFAULT_PROFILE
    names = set(
        profile.integrity_metrics
        + profile.recapture_authorization_metrics
        + profile.diagnostic_metrics
        + profile.recommended_metrics
    )
    names.update({"existing_data_repairability", "existing_data_counterfactual_complete"})
    producer = (map_producer or "unknown").strip().lower()
    if producer not in {"gluemap", "glue_map"}:
        names = {name for name in names if not name.startswith("gluemap_")}
    specs = tuple(METRIC_BY_NAME[name] for name in sorted(names) if name in METRIC_BY_NAME)
    if hard_only:
        return tuple(spec for spec in specs if spec.hard_gate)
    return specs


def categories(specs: Iterable[MetricSpec] = METRICS) -> tuple[str, ...]:
    return tuple(sorted({spec.category for spec in specs}))
