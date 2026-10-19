from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from .actloc import LocalizabilityPredictor, StructuralLocalizabilityProxy
from .fisher import (
    FisherMetrics,
    PoseUncertaintyMetrics,
    compute_fisher_metrics,
    compute_pose_uncertainty,
    weighted_bearing_fim,
)
from .geometry_stats import QueryGeometryStats, compute_query_geometry_stats
from .logs import HistoricalStats, LocalizationHistory
from .matchability import (
    LandmarkMatchability,
    MatchabilityConfig,
    QueryMatchabilityMetrics,
    query_matchability,
)
from .models import CameraIntrinsics, MapData, Pose
from .view_support import ViewSupportConfig, ViewSupportMetrics, compute_view_support
from .visibility import (
    IlluminationModel,
    OcclusionModel,
    image_grid_occupancy,
    normalized_convex_hull_area,
    visible_points,
)


class DiagnosisCode(str, Enum):
    HEALTHY = "HEALTHY"
    DATA_SPARSE = "DATA_SPARSE"
    GEOMETRY_WEAK = "GEOMETRY_WEAK"
    QUERY_PARALLAX_WEAK = "QUERY_PARALLAX_WEAK"
    OBSERVATION_SCALE_WEAK = "OBSERVATION_SCALE_WEAK"
    VIEW_COVERAGE_WEAK = "VIEW_COVERAGE_WEAK"
    ILLUMINATION_WEAK = "ILLUMINATION_WEAK"
    RETRIEVAL_WEAK = "RETRIEVAL_WEAK"
    LANDMARK_MATCHABILITY_WEAK = "LANDMARK_MATCHABILITY_WEAK"
    MATCHING_WEAK = "MATCHING_WEAK"
    REFERENCE_EVIDENCE_INSUFFICIENT = "REFERENCE_EVIDENCE_INSUFFICIENT"
    REFERENCE_DISAGREEMENT = "REFERENCE_DISAGREEMENT"
    REFERENCE_OBSERVABILITY_WEAK = "REFERENCE_OBSERVABILITY_WEAK"
    PERCEPTUAL_ALIASING_SUSPECTED = "PERCEPTUAL_ALIASING_SUSPECTED"
    PNP_DEGENERATE = "PNP_DEGENERATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class DiagnosticThresholds:
    min_visible_points: int = 40
    min_effective_points: float = 25.0
    min_grid_occupancy: int = 6
    min_hull_coverage: float = 0.15
    min_fim_rank: int = 6
    min_fim_isotropy: float = 1e-3
    max_fim_condition: float = 1e8
    # Optional calibrated absolute uncertainty gates. Disabled by default because
    # FIM covariance depends on assumed bearing noise and map scale.
    max_fim_translation_sigma_m: float | None = None
    max_fim_rotation_sigma_deg: float | None = None
    min_track_diversity: float = 0.015
    min_nearby_mapping_cameras: int = 3
    mapping_camera_radius_m: float = 8.0
    min_view_support_fraction: float = 0.10
    min_effective_redetectable_points: float = 15.0
    min_illumination_ratio: float = 0.45
    min_history_success: float = 0.80
    min_pnp_inliers: float = 25.0
    min_inlier_ratio: float = 0.25
    max_reproj_p90: float = 3.0
    min_positive_depth_ratio: float = 0.999
    min_retrieval_score: float = 0.2
    min_registration_confidence: float = 0.5
    min_unique_tracks: float = 25.0
    min_reference_count: float = 2.0
    max_pose_consensus_translation_m: float = 0.5
    max_pose_consensus_rotation_deg: float = 5.0
    max_reference_dispersion_m: float = 0.5
    max_reference_consensus_sigma_m: float = 0.5
    max_reference_rotation_dispersion_deg: float = 5.0
    min_reference_covariance_eligible_ratio: float = 0.5
    min_query_triangulation_angle_deg: float = 2.0
    max_low_parallax_visible_fraction: float = 0.60
    max_scale_ratio: float = 2.0
    min_scale_extrapolated_fraction: float = 0.50
    min_matchability_evidence_fraction: float = 0.30
    min_mean_matchability: float = 0.25
    min_effective_matchable_points: float = 15.0


@dataclass
class Diagnosis:
    primary: DiagnosisCode
    codes: list[DiagnosisCode]
    recommendations: list[str]
    visible_points: int
    effective_points: float
    illumination_ratio: float
    grid_occupancy: int
    hull_coverage: float
    track_diversity: float
    nearby_mapping_cameras: int
    fim: FisherMetrics
    fim_uncertainty: PoseUncertaintyMetrics
    fim_isotropy: float
    view_support: ViewSupportMetrics
    query_geometry: QueryGeometryStats
    structural_localizability: float | None
    history: HistoricalStats
    matchability: QueryMatchabilityMetrics | None
    matchability_fim: FisherMetrics | None
    point_indices: np.ndarray
    point_weights: np.ndarray

    def as_dict(self, include_point_indices: bool = False) -> dict:
        data = {
            "primary": self.primary.value,
            "codes": [c.value for c in self.codes],
            "recommendations": self.recommendations,
            "visible_points": self.visible_points,
            "effective_points": self.effective_points,
            "illumination_ratio": self.illumination_ratio,
            "grid_occupancy": self.grid_occupancy,
            "hull_coverage": self.hull_coverage,
            "track_diversity": self.track_diversity,
            "nearby_mapping_cameras": self.nearby_mapping_cameras,
            "fim": self.fim.as_dict(),
            "fim_uncertainty": self.fim_uncertainty.as_dict(),
            "fim_isotropy": self.fim_isotropy,
            "view_support": self.view_support.as_dict(),
            "query_geometry": self.query_geometry.as_dict(),
            "structural_localizability": self.structural_localizability,
            "history": self.history.as_dict(),
            "matchability": None if self.matchability is None else self.matchability.as_dict(),
            "matchability_fim": None
            if self.matchability_fim is None
            else self.matchability_fim.as_dict(),
        }
        if include_point_indices:
            data["point_indices"] = self.point_indices.tolist()
            data["point_weights"] = self.point_weights.tolist()
        return data


def diagnose_pose(
    map_data: MapData,
    pose: Pose,
    *,
    intrinsics: CameraIntrinsics | None = None,
    thresholds: DiagnosticThresholds | None = None,
    history: LocalizationHistory | None = None,
    predictor: LocalizabilityPredictor | None = None,
    illumination: IlluminationModel | None = None,
    occlusion: OcclusionModel | None = None,
    matchability: LandmarkMatchability | None = None,
    matchability_config: MatchabilityConfig | None = None,
    max_landmark_distance_m: float | None = 30.0,
    bearing_sigma_rad: float = 0.002,
    translation_scale_m: float = 1.0,
) -> Diagnosis:
    t = thresholds or DiagnosticThresholds()
    intr = intrinsics or map_data.median_intrinsics
    vis = visible_points(
        map_data,
        pose,
        intr,
        max_distance=max_landmark_distance_m,
        occlusion=occlusion,
    )
    idx = vis.point_indices
    map_quality = map_data.point_quality_weights()[idx] if len(idx) else np.zeros(0)
    illumination_w = (
        np.asarray(illumination.weights(map_data.points_xyz[idx]), dtype=float)
        if illumination is not None and len(idx)
        else np.ones(len(idx), dtype=float)
    )
    illumination_w = np.clip(illumination_w, 0.0, 1.0)
    weights = map_quality * illumination_w

    fim = weighted_bearing_fim(
        vis.camera_points,
        weights,
        bearing_sigma_rad=bearing_sigma_rad,
        translation_scale_m=translation_scale_m,
    )
    fm = compute_fisher_metrics(fim)
    uncertainty = compute_pose_uncertainty(
        fim,
        translation_scale_m=translation_scale_m,
    )
    fim_isotropy = float(6.0 * fm.lambda_min / max(fm.trace, 1e-12))
    occupancy = image_grid_occupancy(vis.uv, intr)
    hull = normalized_convex_hull_area(vis.uv, intr)
    diversity = (
        float(np.mean(map_data.observation_direction_diversity(idx))) if len(idx) else 0.0
    )
    illumination_ratio = float(np.mean(illumination_w)) if len(idx) else 0.0
    view_support = compute_view_support(map_data, pose, idx)
    query_geometry = compute_query_geometry_stats(
        map_data,
        pose,
        idx,
        range_expansion=ViewSupportConfig.range_expansion,
        parallax_angle_deg=t.min_query_triangulation_angle_deg,
    )
    match_cfg = matchability_config or MatchabilityConfig()
    match_metrics = (
        query_matchability(matchability, pose, idx, map_data, config=match_cfg)
        if matchability is not None
        else None
    )
    match_fim = None
    if (
        matchability is not None
        and match_cfg.reweight_fim
        and match_metrics is not None
        and len(idx)
    ):
        match_w = np.where(
            np.isfinite(match_metrics.point_p),
            match_metrics.point_p,
            map_quality,
        )
        match_fim = compute_fisher_metrics(
            weighted_bearing_fim(
                vis.camera_points,
                match_w * illumination_w,
                bearing_sigma_rad=bearing_sigma_rad,
                translation_scale_m=translation_scale_m,
            )
        )

    if map_data.num_images:
        nearby_mapping = len(
            map_data.image_tree().query_ball_point(pose.center_w, t.mapping_camera_radius_m)
        )
    else:
        nearby_mapping = 0

    hist = history.query(pose) if history is not None else HistoricalStats()
    if predictor is None:
        predictor = StructuralLocalizabilityProxy()
    structural = predictor.predict(map_data, pose) if predictor is not None else None

    codes: list[DiagnosisCode] = []
    recommendations: list[str] = []

    if len(idx) < t.min_visible_points or float(np.sum(weights)) < t.min_effective_points:
        codes.append(DiagnosisCode.DATA_SPARSE)
        recommendations.append(
            "Add mapping coverage for this view; prioritize images that overlap the weak "
            "frustum and create new triangulatable tracks."
        )

    if (
        illumination is not None
        and len(idx) >= t.min_visible_points
        and illumination_ratio < t.min_illumination_ratio
    ):
        codes.append(DiagnosisCode.ILLUMINATION_WEAK)
        recommendations.append(
            "Geometry is visible but direct illumination support is weak; add appearance "
            "references under this lighting/time-of-day or choose a better-lit viewpoint."
        )

    calibrated_uncertainty_bad = (
        t.max_fim_translation_sigma_m is not None
        and uncertainty.sigma_translation_worst_m > t.max_fim_translation_sigma_m
    ) or (
        t.max_fim_rotation_sigma_deg is not None
        and uncertainty.sigma_rotation_worst_deg > t.max_fim_rotation_sigma_deg
    )
    geometry_weak = (
        fm.rank < t.min_fim_rank
        or fim_isotropy < t.min_fim_isotropy
        or fm.condition_number > t.max_fim_condition
        or occupancy < t.min_grid_occupancy
        or hull < t.min_hull_coverage
        or calibrated_uncertainty_bad
    )
    if geometry_weak:
        codes.append(DiagnosisCode.GEOMETRY_WEAK)
        recommendations.append(
            "Strengthen geometry with lateral/oblique views and a larger useful baseline; "
            "optimize for broader image support and the weakest FIM eigen-direction, not "
            "raw point count."
        )

    view_support_bad = (
        view_support.weighted_visible_support_fraction < t.min_view_support_fraction
        or view_support.effective_redetectable_points < t.min_effective_redetectable_points
    )
    scale_ratio = query_geometry.scale_ratio_p50
    scale_weak = len(idx) >= t.min_visible_points and (
        query_geometry.scale_extrapolated_fraction >= t.min_scale_extrapolated_fraction
        or (
            scale_ratio is not None
            and (
                scale_ratio > t.max_scale_ratio
                or scale_ratio < 1.0 / t.max_scale_ratio
            )
        )
    )
    if scale_weak:
        codes.append(DiagnosisCode.OBSERVATION_SCALE_WEAK)
        recommendations.append(
            "Deployment range/scale is outside the mapping observation envelope; change "
            "standoff, zoom, or altitude rather than rebuilding geometry."
        )

    parallax_p50 = query_geometry.triangulation_angle_p50_deg
    parallax_weak = len(idx) >= t.min_visible_points and (
        (
            parallax_p50 is not None
            and parallax_p50 < t.min_query_triangulation_angle_deg
        )
        or query_geometry.low_parallax_visible_fraction
        >= t.max_low_parallax_visible_fraction
    )
    if parallax_weak:
        codes.append(DiagnosisCode.QUERY_PARALLAX_WEAK)
        recommendations.append(
            "Visible landmarks lack a useful independent baseline; search existing "
            "long-baseline/cross-route anchors, then recapture a lateral/oblique view."
        )

    if (
        len(idx) >= t.min_visible_points
        and (
            diversity < t.min_track_diversity
            or nearby_mapping < t.min_nearby_mapping_cameras
            or (view_support_bad and not scale_weak)
        )
    ):
        codes.append(DiagnosisCode.VIEW_COVERAGE_WEAK)
        recommendations.append(
            "Add reference observations from missing approach directions; preserve overlap "
            "while increasing viewpoint diversity. If FIM is already healthy, prioritize "
            "redetectability/view support before rebuilding geometry."
        )

    if match_metrics is not None and (
        match_metrics.evidenced_visible_fraction >= t.min_matchability_evidence_fraction
        and (
            (
                match_metrics.mean_matchability is not None
                and match_metrics.mean_matchability < t.min_mean_matchability
            )
            or match_metrics.effective_matchable < t.min_effective_matchable_points
        )
    ):
        codes.append(DiagnosisCode.LANDMARK_MATCHABILITY_WEAK)
        recommendations.append(
            "Mapped landmarks are visible but historically unmatched; add appearance "
            "references or stronger matching and drop dead landmarks from retrieval. "
            "Do not recapture for geometry."
        )

    if hist.count:
        retrieval_bad = (
            hist.retrieval_score is not None
            and hist.retrieval_score < t.min_retrieval_score
        ) or (
            hist.registration_confidence is not None
            and hist.registration_confidence < t.min_registration_confidence
        )
        matching_support_bad = (
            hist.unique_tracks is not None and hist.unique_tracks < t.min_unique_tracks
        ) or (
            hist.reference_count is not None
            and hist.reference_count < t.min_reference_count
        )
        consensus_bad = (
            hist.pose_consensus_translation_m is not None
            and hist.pose_consensus_translation_m > t.max_pose_consensus_translation_m
        ) or (
            hist.pose_consensus_rotation_deg is not None
            and hist.pose_consensus_rotation_deg > t.max_pose_consensus_rotation_deg
        )
        pnp_bad = (
            hist.pnp_inliers is not None and hist.pnp_inliers < t.min_pnp_inliers
        ) or (
            hist.inlier_ratio is not None and hist.inlier_ratio < t.min_inlier_ratio
        ) or (
            hist.reproj_p90 is not None and hist.reproj_p90 > t.max_reproj_p90
        )
        coverage_bad = (
            hist.grid_occupancy is not None
            and hist.grid_occupancy < t.min_grid_occupancy
        ) or (
            hist.hull_coverage is not None
            and hist.hull_coverage < t.min_hull_coverage
        )
        depth_bad = (
            hist.positive_depth_ratio is not None
            and hist.positive_depth_ratio < t.min_positive_depth_ratio
        )
        actual_bad = hist.success_rate is not None and hist.success_rate < t.min_history_success

        reference_evidence_insufficient = (
            hist.reference_hypothesis_count is not None
            and hist.reference_hypothesis_count < t.min_reference_count
        )
        reference_disagreement = (
            hist.reference_dispersion_m is not None
            and hist.reference_dispersion_m > t.max_reference_dispersion_m
        ) or (
            hist.reference_rotation_dispersion_deg is not None
            and hist.reference_rotation_dispersion_deg
            > t.max_reference_rotation_dispersion_deg
        )
        reference_observability_bad = (
            hist.reference_consensus_sigma_m is not None
            and hist.reference_consensus_sigma_m > t.max_reference_consensus_sigma_m
        ) or (
            hist.reference_covariance_eligible_ratio is not None
            and hist.reference_covariance_eligible_ratio
            < t.min_reference_covariance_eligible_ratio
        )
        covariance_evidence_available = (
            hist.reference_consensus_sigma_m is not None
            and hist.reference_covariance_eligible_ratio is not None
        )

        if reference_evidence_insufficient:
            codes.append(DiagnosisCode.REFERENCE_EVIDENCE_INSUFFICIENT)
            recommendations.append(
                "Too few active per-reference pose hypotheses are available to assess "
                "cross-reference consensus. Preserve at least two independent reference "
                "hypotheses before interpreting dispersion as a reliability signal."
            )

        if reference_disagreement:
            codes.append(DiagnosisCode.REFERENCE_DISAGREEMENT)
            recommendations.append(
                "Per-reference pose hypotheses disagree. Inspect repeated structures, wrong "
                "retrievals, reference-pose consistency, scale/alignment failure, and model "
                "failure before averaging the hypotheses into one pose."
            )

            strong_correspondence_evidence = (
                not retrieval_bad
                and not matching_support_bad
                and not reference_evidence_insufficient
                and hist.retrieval_score is not None
                and hist.retrieval_score >= t.min_retrieval_score
                and hist.unique_tracks is not None
                and hist.unique_tracks >= t.min_unique_tracks
                and hist.reference_count is not None
                and hist.reference_count >= t.min_reference_count
            )
            if (
                not geometry_weak
                and covariance_evidence_available
                and not reference_observability_bad
                and strong_correspondence_evidence
            ):
                codes.append(DiagnosisCode.PERCEPTUAL_ALIASING_SUSPECTED)
                recommendations.append(
                    "Map geometry, covariance observability, retrieval, and track support are "
                    "adequate but references induce incompatible global poses. Treat repeated-"
                    "structure/perceptual aliasing as a primary suspect; add discriminative "
                    "references, stronger global verification, or sequence/temporal anchoring."
                )

        if reference_observability_bad:
            codes.append(DiagnosisCode.REFERENCE_OBSERVABILITY_WEAK)
            recommendations.append(
                "Reference hypotheses may agree yet remain weakly constrained. Inspect the "
                "per-reference information eigenvalues/covariances and add lateral/oblique "
                "geometry or spatially broader tracks; more matches alone may not help."
            )

        if retrieval_bad and not geometry_weak:
            codes.append(DiagnosisCode.RETRIEVAL_WEAK)
            recommendations.append(
                "Geometry is adequate but retrieval is weak; improve reference retrieval/"
                "indexing or add appearance-diverse reference images before rebuilding geometry."
            )

        reference_failure_explains_result = (
            reference_disagreement or reference_observability_bad
        )
        matching_failure = pnp_bad or coverage_bad or matching_support_bad
        if (
            (matching_failure or (actual_bad and not reference_failure_explains_result))
            and not geometry_weak
            and not retrieval_bad
        ):
            codes.append(DiagnosisCode.MATCHING_WEAK)
            recommendations.append(
                "Geometry is adequate but actual correspondence support is weak; try targeted "
                "stronger matching, reference expansion, and appearance-specific observations."
            )

        if depth_bad or consensus_bad:
            codes.append(DiagnosisCode.PNP_DEGENERATE)
            recommendations.append(
                "Positive-depth or final PnP pose consensus is failing; inspect 2D-3D "
                "correspondences, calibration, repeated structure, and pose-hypothesis degeneracy."
            )

    codes = _dedupe(codes)
    recommendations = _dedupe(recommendations)
    if not codes:
        codes = [DiagnosisCode.HEALTHY]
        recommendations = [
            "No configured failure mode was triggered. Validate with held-out localization "
            "queries before declaring the region production-ready."
        ]

    primary = _primary_code(codes)
    return Diagnosis(
        primary=primary,
        codes=codes,
        recommendations=recommendations,
        visible_points=len(idx),
        effective_points=float(np.sum(weights)),
        illumination_ratio=illumination_ratio,
        grid_occupancy=occupancy,
        hull_coverage=hull,
        track_diversity=diversity,
        nearby_mapping_cameras=int(nearby_mapping),
        fim=fm,
        fim_uncertainty=uncertainty,
        fim_isotropy=fim_isotropy,
        view_support=view_support,
        query_geometry=query_geometry,
        structural_localizability=structural,
        history=hist,
        matchability=match_metrics,
        matchability_fim=match_fim,
        point_indices=idx,
        point_weights=weights,
    )


def thresholds_from_dict(data: dict) -> DiagnosticThresholds:
    allowed = set(asdict(DiagnosticThresholds()))
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown diagnostic thresholds: {sorted(unknown)}")
    return DiagnosticThresholds(**data)


def _primary_code(codes: list[DiagnosisCode]) -> DiagnosisCode:
    priority = [
        DiagnosisCode.DATA_SPARSE,
        DiagnosisCode.GEOMETRY_WEAK,
        DiagnosisCode.QUERY_PARALLAX_WEAK,
        DiagnosisCode.OBSERVATION_SCALE_WEAK,
        DiagnosisCode.REFERENCE_OBSERVABILITY_WEAK,
        DiagnosisCode.PERCEPTUAL_ALIASING_SUSPECTED,
        DiagnosisCode.REFERENCE_DISAGREEMENT,
        DiagnosisCode.REFERENCE_EVIDENCE_INSUFFICIENT,
        DiagnosisCode.ILLUMINATION_WEAK,
        DiagnosisCode.PNP_DEGENERATE,
        DiagnosisCode.RETRIEVAL_WEAK,
        DiagnosisCode.LANDMARK_MATCHABILITY_WEAK,
        DiagnosisCode.MATCHING_WEAK,
        DiagnosisCode.VIEW_COVERAGE_WEAK,
        DiagnosisCode.HEALTHY,
        DiagnosisCode.UNKNOWN,
    ]
    for code in priority:
        if code in codes:
            return code
    return DiagnosisCode.UNKNOWN


def _dedupe(items: list):
    return list(dict.fromkeys(items))
