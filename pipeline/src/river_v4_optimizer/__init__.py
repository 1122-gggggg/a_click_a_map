"""Connectivity-first optimization tools for the River V4 all-video map."""

from .integration import (
    OptimizationRecipe,
    build_candidate_plan,
    prunable_loser_models,
    select_winner,
)
from .cleanup import prune_observation_free_registered_images
from .forced_pairs import build_forced_cross_video_pairs, merge_pair_geometry
from .detector_free_injection import (
    DetectorFreeInjectionConfig,
    DetectorFreeMatch,
    PlannedDetectorFreeTrack,
    cluster_detector_free_matches,
    inject_planned_tracks,
    plan_detector_free_tracks,
    validate_planned_track,
)
from .metrics import (
    analyze_model,
    component_summary,
    evaluate_objective_improvement,
    focus_session_track_summary,
    rank_variants,
)
from .sim3_audit import (
    SimilarityTransform,
    audit_shared_camera_sim3,
    estimate_similarity_transform,
    estimate_robust_similarity_transform,
    relative_pose_residual,
    transform_camera_pose_to_target_gauge,
)
from .selection import build_connected_submap_selection, rewire_selection

__all__ = [
    "OptimizationRecipe",
    "DetectorFreeMatch",
    "DetectorFreeInjectionConfig",
    "PlannedDetectorFreeTrack",
    "SimilarityTransform",
    "analyze_model",
    "audit_shared_camera_sim3",
    "build_forced_cross_video_pairs",
    "build_candidate_plan",
    "build_connected_submap_selection",
    "component_summary",
    "cluster_detector_free_matches",
    "evaluate_objective_improvement",
    "estimate_similarity_transform",
    "estimate_robust_similarity_transform",
    "focus_session_track_summary",
    "merge_pair_geometry",
    "inject_planned_tracks",
    "plan_detector_free_tracks",
    "validate_planned_track",
    "prunable_loser_models",
    "prune_observation_free_registered_images",
    "rank_variants",
    "relative_pose_residual",
    "rewire_selection",
    "select_winner",
    "transform_camera_pose_to_target_gauge",
]
