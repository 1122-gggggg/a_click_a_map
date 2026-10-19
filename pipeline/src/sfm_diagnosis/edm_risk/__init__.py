"""Unified post-map EDM localization-failure diagnosis.

The external seam is :class:`EDMRiskDiagnosis`.  Provider-specific localizers,
ActLoc implementations, calibration models, and visualization remain internal
adapters so the map-first fast path has no heavyweight dependency.
"""

from .grid import SpatialGridConfig, SpatialPoseGrid, SpatialPoseSample
from .pipeline import DiagnosisRun, EDMRiskDiagnosis
from .schema import RiskClass, SpatialDiagnostic

__all__ = [
    "DiagnosisRun",
    "EDMRiskDiagnosis",
    "RiskClass",
    "SpatialDiagnostic",
    "SpatialGridConfig",
    "SpatialPoseGrid",
    "SpatialPoseSample",
]
