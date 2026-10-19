"""Adapter-based SfM map diagnostics."""

from .diagnose import Diagnosis, DiagnosisCode, diagnose_pose
from .evidence import BuildEvidence, load_build_evidence
from .fisher import (
    FisherMetrics,
    PoseUncertaintyMetrics,
    compute_fisher_metrics,
    compute_pose_uncertainty,
    weighted_bearing_fim,
)
from .geometry_stats import QueryGeometryStats, compute_query_geometry_stats
from .io import load_gluemap
from .matchability import (
    LandmarkMatchability,
    MatchabilityConfig,
    QueryMatchabilityMetrics,
    build_landmark_matchability,
    query_matchability,
)
from .models import CameraIntrinsics, MapData, Pose
from .pair_geometry import pair_model_flags, reconstructed_relative_error
from .reference_consensus import (
    ReferenceConsensusAssessment,
    ReferenceConsensusMetrics,
    ReferenceConsensusStatus,
    ReferenceHypothesis,
    SelectiveNormalization,
    analyze_reference_hypotheses,
    assess_reference_consensus,
    compute_reference_consensus,
    compute_sigma_joint,
    fit_selective_normalization,
    load_reference_hypotheses,
)
from .route import (
    MonotonicRiskCalibrator,
    RouteAuditConfig,
    RouteAuditResult,
    audit_route,
    fit_monotonic_risk_calibrator,
    save_route_audit,
)
from .view_support import ViewSupportConfig, ViewSupportMetrics, compute_view_support
from .risk_ply import write_risk_ply
from .weak_regions import (
    WeakRegionAnalysis,
    WeakRegionCause,
    analyze_weak_regions,
)

__all__ = [
    "BuildEvidence",
    "CameraIntrinsics",
    "Diagnosis",
    "DiagnosisCode",
    "FisherMetrics",
    "LandmarkMatchability",
    "MapData",
    "MatchabilityConfig",
    "MonotonicRiskCalibrator",
    "Pose",
    "PoseUncertaintyMetrics",
    "QueryGeometryStats",
    "QueryMatchabilityMetrics",
    "ReferenceConsensusAssessment",
    "ReferenceConsensusMetrics",
    "ReferenceConsensusStatus",
    "ReferenceHypothesis",
    "SelectiveNormalization",
    "RouteAuditConfig",
    "RouteAuditResult",
    "ViewSupportConfig",
    "ViewSupportMetrics",
    "WeakRegionAnalysis",
    "WeakRegionCause",
    "analyze_reference_hypotheses",
    "analyze_weak_regions",
    "assess_reference_consensus",
    "audit_route",
    "build_landmark_matchability",
    "compute_fisher_metrics",
    "compute_pose_uncertainty",
    "compute_query_geometry_stats",
    "compute_reference_consensus",
    "compute_sigma_joint",
    "compute_view_support",
    "fit_selective_normalization",
    "diagnose_pose",
    "fit_monotonic_risk_calibrator",
    "load_build_evidence",
    "load_gluemap",
    "load_reference_hypotheses",
    "pair_model_flags",
    "query_matchability",
    "reconstructed_relative_error",
    "save_route_audit",
    "weighted_bearing_fim",
    "write_risk_ply",
]

__version__ = "0.5.0"
