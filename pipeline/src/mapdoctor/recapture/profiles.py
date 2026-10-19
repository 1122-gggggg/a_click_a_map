from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class PlannerThresholds:
    """Project defaults, not universal SfM constants.

    Every deployment should calibrate these thresholds on a frozen, condition-
    matched holdout set. The defaults preserve the conservative gates discussed
    in this project and are intentionally separated from metric domain checks.
    """

    min_attempt_count: int = 8
    min_holdout_coverage: float = 0.75
    healthy_success_rate: float = 0.95
    weak_success_rate: float = 0.60
    navigation_best_health: float = 0.75
    navigation_mean_gap: float = 0.25
    min_visible_landmarks: int = 80
    min_effective_landmarks: float = 50.0
    min_hull_coverage: float = 0.15
    min_grid_occupancy: int = 6
    min_positive_depth_ratio: float = 1.0
    min_fim_rank: int = 6
    min_fim_effective_rank: float = 4.5
    max_fim_condition_number: float = 1_000.0
    min_triangulation_angle_p10_deg: float = 1.0
    min_view_direction_entropy: float = 0.35
    max_reprojection_p90_px: float = 3.0
    min_track_length_p50: float = 3.0
    min_condition_match_score: float = 0.45
    min_illumination_ratio: float = 0.35
    min_existing_data_repairability: float = 0.55
    weak_structural_health: float = 0.52
    healthy_structural_health: float = 0.68
    max_alias_risk: float = 0.45


@dataclass(frozen=True)
class CaptureGeometry:
    """Map-frame capture shell used only to create planning hypotheses."""

    anchor_extension: float = 2.0
    path_half_length: float = 3.0
    sample_step: float = 1.0
    lateral_offset: float = 2.0
    height_offset: float = 1.5
    max_passes: int = 6
    expected_overlap: float = 0.70
    map_units: str = "map_units"


@dataclass(frozen=True)
class LocalizerProfile:
    integrity_metrics: tuple[str, ...]
    recapture_authorization_metrics: tuple[str, ...]
    diagnostic_metrics: tuple[str, ...]
    recommended_metrics: tuple[str, ...]
    earliest_stage_order: tuple[str, ...]
    repair_actions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


COMMON_INTEGRITY = (
    "coordinate_scale_status",
    "camera_intrinsics_valid",
    "frame_transform_valid",
    "handedness_valid",
)

COMMON_RECAPTURE_AUTHORIZATION = (
    "visible_landmark_count",
    "inlier_convex_hull_coverage",
    "grid_occupancy_count",
    "positive_depth_ratio",
    "fim_rank",
    "fim_lambda_min",
    "fim_condition_number",
    "triangulation_angle_p10_deg",
    "view_direction_entropy",
    "attempt_count",
    "localization_success_rate",
    "holdout_query_coverage",
)

COMMON_DIAGNOSTIC = (
    "effective_landmark_count",
    "localizer_usable_landmark_ratio",
    "view_support_score",
    "image_spatial_entropy",
    "bearing_isotropy",
    "depth_p10",
    "depth_p50",
    "depth_p90",
    "depth_iqr_ratio",
    "depth_entropy",
    "fim_effective_rank",
    "fim_logdet",
    "fim_trace_inverse",
    "fim_weakest_mode",
    "crlb_translation_trace",
    "crlb_rotation_trace",
    "track_length_p50",
    "track_length_p10",
    "track_observation_count_p50",
    "reprojection_error_p50_px",
    "reprojection_error_p90_px",
    "triangulation_angle_p50_deg",
    "baseline_diversity",
    "covisibility_neighbor_count",
    "independent_view_neighbor_count",
    "graph_component_count",
    "articulation_camera_ratio",
    "bridge_edge_ratio",
    "cycle_supported_edge_ratio",
    "algebraic_connectivity",
    "single_camera_removal_fragmentation",
    "repeated_structure_alias_risk",
    "pose_mode_count",
    "multi_reference_rotation_disagreement_deg",
    "illumination_direct_visibility_ratio",
    "condition_match_score",
    "blur_score",
    "exposure_clipping_ratio",
    "temporal_stability_probability",
    "dynamic_landmark_ratio",
    "directional_sample_count",
    "directional_angular_coverage",
    "directional_best_health",
    "directional_mean_health",
    "directional_worst_health",
    "directional_health_variance",
    "operational_direction_success_rate",
    "orientation_recovery_gain",
    "learned_localization_success_probability",
    "learned_localizability_ood_score",
    "learned_localizability_calibration_error",
    "planarity_ratio",
    "collinearity_ratio",
    "landmark_spread_3d",
    "triangulation_uncertainty_p90",
    "landmark_position_uncertainty_p90",
    "viewpoint_extrapolation_score",
    "nearest_reference_distance",
    "detected_feature_count_p50",
    "raw_match_count_p50",
    "verified_match_count_p50",
    "verified_match_inlier_ratio",
    "match_convex_hull_coverage",
    "lifted_2d3d_count_p50",
    "lifted_unique_landmark_count_p50",
    "lifted_convex_hull_coverage",
    "pnp_covariance_trace",
    "pnp_covariance_condition_number",
    "gluemap_retrieval_degree_p50",
    "gluemap_retrieval_margin_p50",
    "gluemap_twoview_valid_probability_p10",
    "gluemap_star_neighbor_count_p10",
    "gluemap_star_confidence_p10",
    "gluemap_rotation_averaging_residual_p90_deg",
    "gluemap_similarity_rotation_residual_p90_deg",
    "gluemap_similarity_log_scale_residual_p90",
    "gluemap_intrinsics_relative_deviation_p90",
    "gluemap_snap_acceptance_ratio",
    "gluemap_snap_observation_gain",
    "gluemap_coarse_reprojection_p90_px",
    "gluemap_refined_reprojection_p90_px",
    "false_acceptance_rate",
    "false_rejection_rate",
    "pnp_unique_inliers_p50",
    "pnp_unique_inliers_p05",
    "pnp_inlier_ratio_p50",
    "pnp_reprojection_p90_px",
    "pose_translation_error_p90",
    "pose_rotation_error_p90_deg",
    "time_to_relocalize_p90_s",
    "consecutive_failure_duration_p95_s",
    "latency_p95_ms",
)

COMMON_RECOMMENDED = (
    "registered_image_ratio",
    "largest_component_ratio",
    "multi_session_support_ratio",
    "unstable_landmark_ratio",
    "condition_strata_coverage",
    "weak_region_success_rate",
    "stable_region_success_rate",
    "common_success_inlier_drop_ratio",
    "common_success_reprojection_increase_ratio",
    "safe_zone_decision_accuracy",
    "query_set_hash_present",
    "config_hash_present",
    "metric_provenance_complete",
    "route_direction_coverage",
    "yaw_pitch_bin_coverage",
    "root_cause_precision",
    "root_cause_recall",
    "root_cause_calibration_error",
    "intervention_rescue_precision",
    "query_map_leakage_count",
    "map_version_hash_present",
    "reference_eligibility_hash_present",
)


DEFAULT_PROFILE = LocalizerProfile(
    integrity_metrics=COMMON_INTEGRITY,
    recapture_authorization_metrics=COMMON_RECAPTURE_AUTHORIZATION,
    diagnostic_metrics=COMMON_DIAGNOSTIC,
    recommended_metrics=COMMON_RECOMMENDED,
    earliest_stage_order=("map_support", "localizer", "geometry"),
    repair_actions={
        "map_support": ("rebuild active reference set", "repair track payload"),
        "localizer": (
            "trace the localizer's own stages on the frozen query set",
            "repair the earliest measured failing stage",
        ),
        "geometry": ("targeted rematching", "retriangulation", "anchored local BA"),
    },
)


def profile_for(localizer: str | None = None) -> LocalizerProfile:
    """Return one method-agnostic contract; ``localizer`` is provenance only."""
    return DEFAULT_PROFILE


def _replace_dataclass(instance: Any, values: Mapping[str, Any], section: str) -> Any:
    allowed = {field.name for field in instance.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = set(values) - allowed
    if extra:
        raise ValueError(f"Unknown {section} settings: {', '.join(sorted(extra))}")
    return replace(instance, **values)


def load_config(path: str | Path | None) -> tuple[PlannerThresholds, CaptureGeometry]:
    thresholds = PlannerThresholds()
    capture = CaptureGeometry()
    if path is None:
        return thresholds, capture
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("configuration root must be a JSON object")
    known = {"thresholds", "capture"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown top-level settings: {', '.join(sorted(unknown))}")
    threshold_data = data.get("thresholds", {})
    capture_data = data.get("capture", {})
    if not isinstance(threshold_data, Mapping):
        raise ValueError("thresholds must be a JSON object")
    if not isinstance(capture_data, Mapping):
        raise ValueError("capture must be a JSON object")
    thresholds = _replace_dataclass(thresholds, threshold_data, "thresholds")
    capture = _replace_dataclass(capture, capture_data, "capture")
    return thresholds, capture
