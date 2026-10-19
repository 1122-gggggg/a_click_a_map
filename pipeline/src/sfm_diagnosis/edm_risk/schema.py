from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RiskClass(str, Enum):
    GOOD = "GOOD"
    VIEW_DIRECTION_SENSITIVE = "VIEW_DIRECTION_SENSITIVE"
    WEAK = "WEAK"
    DEAD_ZONE = "DEAD_ZONE"
    UNKNOWN = "UNKNOWN"


@dataclass
class SpatialDiagnostic:
    """One normalized diagnosis row for ``(x, y, z, yaw, pitch)``.

    ``None`` means unavailable evidence, never an implicit zero.  This keeps
    fast/map-only outputs honest when EDM, ActLoc, or calibration artifacts are
    absent and lets later phases enrich the same schema without rewriting it.
    """

    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float

    actloc_score: float | None = None
    actloc_rank: float | None = None
    viewpoint_entropy: float | None = None
    actloc_best_yaw_deg: float | None = None
    actloc_best_pitch_deg: float | None = None
    actloc_worst_yaw_deg: float | None = None
    actloc_worst_pitch_deg: float | None = None
    actloc_source: str = "disabled"

    visible_landmarks: int = 0
    raw_visible_landmarks: int | None = None
    occluded_count: int = 0
    occlusion_uncertain_count: int = 0
    effective_landmarks: float = 0.0
    visibility_source: str = "frustum_only"
    occlusion_verified: bool = False
    occlusion_proxy_applied: bool = False
    matchability_source: str = "heuristic"

    median_track_length: float | None = None
    track_ge_3_ratio: float | None = None
    track_ge_5_ratio: float | None = None
    covisibility: float | None = None
    independent_observers: int = 0

    parallax_p10_deg: float | None = None
    parallax_p25_deg: float | None = None
    parallax_median_deg: float | None = None
    mapping_view_yaw_std_deg: float | None = None
    mapping_view_pitch_std_deg: float | None = None
    view_entropy: float | None = None

    convex_hull_coverage: float = 0.0
    grid_occupancy: int = 0
    landmark_spatial_entropy: float | None = None
    median_reprojection_error: float | None = None
    reprojection_error_p90: float | None = None
    positive_depth_ratio: float | None = None
    landmark_pca_eigenvalues: list[float] | None = None
    mapping_camera_pca_eigenvalues: list[float] | None = None
    landmark_linearity: float | None = None
    landmark_planarity: float | None = None
    landmark_scattering: float | None = None
    mapping_camera_linearity: float | None = None
    mapping_camera_planarity: float | None = None
    mapping_camera_scattering: float | None = None

    fim_lambda_min: float | None = None
    fim_lambda_max: float | None = None
    fim_condition: float | None = None
    fim_logdet: float | None = None
    fim_trace: float | None = None
    fim_a_opt: float | None = None
    fim_d_opt: float | None = None
    fim_e_opt: float | None = None
    fim_translation_min_eigenvalue: float | None = None
    fim_rotation_min_eigenvalue: float | None = None
    weakest_eigenvector: list[float] | None = None

    edm_loo_success_rate: float | None = None
    edm_inliers_median: float | None = None
    edm_inlier_ratio_median: float | None = None
    edm_reprojection_median: float | None = None
    edm_loo_mode: str | None = None

    num_pose_modes: int | None = None
    mode_support_margin: float | None = None
    mode_translation_separation: float | None = None
    mode_rotation_separation_deg: float | None = None
    mode_entropy: float | None = None
    loo_translation_jump: float | None = None
    loo_rotation_jump_deg: float | None = None

    consecutive_failure_probability: float | None = None
    max_consecutive_failures: int | None = None
    recovery_time: float | None = None

    degeneracy_flags: list[str] = field(default_factory=list)
    failure_probability: float | None = None
    risk_class: RiskClass = RiskClass.UNKNOWN
    primary_failure_causes: list[str] = field(default_factory=list)
    recommended_action: str | None = None
    recommended_capture_position: list[float] | None = None
    recommended_yaw_deg: float | None = None
    recommended_pitch_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["position"] = [float(value) for value in self.position]
        payload["risk_class"] = self.risk_class.value
        return payload
