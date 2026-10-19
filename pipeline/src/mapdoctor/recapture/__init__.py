"""Targeted weak-region repair and recapture planning for MapDoctor."""

from .api import analyze_pose_cells, attach_recapture_analysis
from .audit import audit_by_region, audit_cells, audit_source_repository
from .bridge import enrich_pose_cells_from_model, map_producer_from_model
from .compute import bearing_fisher_information, compute_metric_bundle, fim_summary
from .counterfactual import (
    ExistingDataCounterfactualSummary,
    ExistingDataRepairTrial,
    summarize_existing_data_counterfactual,
)
from .planner import plan_regions
from .profiles import CaptureGeometry, PlannerThresholds, profile_for
from .types import (
    Availability,
    CaptureMode,
    CapturePass,
    DecisionStatus,
    MetricValue,
    PoseDirectionCell,
    RecaptureDecision,
    normalize_localizer,
)

__all__ = [
    "Availability",
    "CaptureGeometry",
    "CaptureMode",
    "CapturePass",
    "DecisionStatus",
    "ExistingDataCounterfactualSummary",
    "ExistingDataRepairTrial",
    "MetricValue",
    "PlannerThresholds",
    "PoseDirectionCell",
    "RecaptureDecision",
    "analyze_pose_cells",
    "attach_recapture_analysis",
    "audit_by_region",
    "audit_cells",
    "audit_source_repository",
    "bearing_fisher_information",
    "compute_metric_bundle",
    "enrich_pose_cells_from_model",
    "fim_summary",
    "map_producer_from_model",
    "normalize_localizer",
    "plan_regions",
    "profile_for",
    "summarize_existing_data_counterfactual",
]
