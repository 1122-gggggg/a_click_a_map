"""Public session-selection types. Names are a contract — keep them stable."""

from __future__ import annotations

from dataclasses import dataclass, field


ROLES = (
    "BASE_CORE",
    "BASE_SUPPORT",
    "APPEARANCE_REF",
    "GEOMETRY_REINFORCEMENT",
    "UPDATE_CANDIDATE",
    "NEW_SUBMAP",
    "QUARANTINE",
    "REJECT",
    "VALIDATION_ONLY",
)

INTERNAL_STATUSES = ("STRONG", "USABLE", "WEAK", "INCONSISTENT", "REJECT")
EDGE_STATUSES = ("STRONG", "USABLE", "WEAK", "AMBIGUOUS", "REJECT")


@dataclass
class SessionQuality:
    session_id: str
    timestamp: str | None = None
    num_frames: int = 0
    num_keyframes: int = 0
    registered_ratio: float | None = None
    sharpness_median: float | None = None
    sharpness_p10: float | None = None
    underexposed_ratio: float | None = None
    overexposed_ratio: float | None = None
    near_duplicate_ratio: float | None = None
    exposure_mean: float | None = None
    parallax_ratio: float | None = None
    low_parallax_ratio: float | None = None
    hover_ratio: float | None = None
    pure_rotation_ratio: float | None = None
    fast_motion_ratio: float | None = None
    unproven_ratio: float | None = None
    epipolar_outlier_ratio_median: float | None = None
    essential_inlier_ratio_median: float | None = None
    flow_median_px: float | None = None
    motion_parallax_median_px: float | None = None
    num_tracks: int | None = None
    num_observations: int | None = None
    median_track_length: float | None = None
    long_track_ratio: float | None = None
    reprojection_rmse: float | None = None
    reprojection_p90: float | None = None
    parallax_median_deg: float | None = None
    parallax_p10_deg: float | None = None
    positive_depth_ratio: float | None = None
    convex_hull_coverage: float | None = None
    grid_occupancy_4x4: float | None = None
    fim_condition_number: float | None = None
    fim_logdet: float | None = None
    rotation_cycle_error: float | None = None
    translation_consistency: float | None = None
    connected_components: int | None = None
    average_degree: float | None = None
    num_bridges: int | None = None
    num_articulation_points: int | None = None
    fiedler_value: float | None = None
    internal_quality_score: float = 0.0
    internal_status: str = "REJECT"
    reasons: tuple[str, ...] = ()
    hard_valid: bool = True
    hard_failures: tuple[str, ...] = ()
    metric_coverage: float = 0.0
    soft_rank: float = 0.5
    candidate_tier: str = "UNSCORED"
    admission_state: str = "UNASSESSED"
    evaluation_role: str | None = None
    evidence_provenance: tuple[str, ...] = ()
    coverage_cells: tuple[str, ...] = ()
    degeneracy_flags: tuple[str, ...] = ()
    session_graph_degree: float | None = None
    component_size: int | None = None
    is_isolated: bool = False
    incident_bridges: int | None = None
    is_articulation: bool = False
    graph_radius: int | None = None


@dataclass
class SessionEdgeQuality:
    session_a: str
    session_b: str
    num_candidate_pairs: int = 0
    num_verified_pairs: int = 0
    num_cross_session_tracks: int | None = None
    num_cross_session_observations: int | None = None
    independent_bridge_groups: int = 0
    inlier_count: int | None = None
    inlier_ratio: float | None = None
    rotation_consensus_deg: float | None = None
    translation_direction_consensus_deg: float | None = None
    scale_consensus: float | None = None
    cross_session_reprojection_error: float | None = None
    spatial_coverage: float | None = None
    cycle_support: int | None = None
    cycle_error: float | None = None
    edge_quality_score: float = 0.0
    is_bridge: bool = False
    is_critical_bridge: bool = False
    status: str = "REJECT"
    reasons: tuple[str, ...] = ()
    independent_artifact: bool = False
    evidence_scope: str = "unknown"
    geometry_artifact: str | None = None
    geometry_artifact_sha256: str | None = None
    fit_evidence_ids: tuple[str, ...] = ()
    holdout_evidence_ids: tuple[str, ...] = ()
    support_count: int | None = None
    holdout_count: int | None = None
    holdout_inlier_ratio: float | None = None
    holdout_residual: float | None = None
    bridge_group_ids: tuple[str, ...] = ()
    group_holdout_disjoint: bool = False
    bridge_diversity_axes: tuple[str, ...] = ()
    degeneracy_flags: tuple[str, ...] = ()
    geometry_complete: bool = False
    parallax_deg: float | None = None
    edge_positive_depth_ratio: float | None = None


@dataclass
class SessionInfluence:
    session_id: str
    median_rotation_shift_deg: float | None = None
    p90_rotation_shift_deg: float | None = None
    median_position_shift_normalized: float | None = None
    p90_position_shift_normalized: float | None = None
    reprojection_delta: float | None = None
    FIM_delta: float | None = None
    coverage_delta: float | None = None
    high_influence: bool = False
    reasons: tuple[str, ...] = ()


@dataclass
class SessionRecord:
    """Discovered video + joined artifacts. Not a public scoring type."""

    session_id: str
    video_path: str
    sha256: str
    timestamp: str | None = None
    duration_seconds: float | None = None
    num_frames: int = 0
    width: int | None = None
    height: int | None = None
    keyframes: tuple[dict, ...] = ()
    image_dirs: tuple[str, ...] = ()
    map_sources: tuple[dict, ...] = ()
    motion_rows: tuple[dict, ...] = ()


@dataclass
class RoleAssignment:
    session_id: str
    role: str
    reason: str
    scores: dict = field(default_factory=dict)


def edge_is_vpr_only(edge: SessionEdgeQuality) -> bool:
    """True when an edge has retrieval candidates but no geometric support.

    VPR / retrieval is never a geometric edge. Status alone does not override
    explicit zero-geometry + nonzero candidate pairs.
    """

    no_tracks = not edge.num_cross_session_tracks
    no_verified = not edge.num_verified_pairs
    no_bridges = not edge.independent_bridge_groups
    return bool(edge.num_candidate_pairs) and no_tracks and no_verified and no_bridges
