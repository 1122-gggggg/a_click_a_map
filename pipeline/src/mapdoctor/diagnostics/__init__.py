"""Statistical, calibrated-risk, selective-risk, and graph diagnostics."""

from .calibration import (
    BetaFailureCalibrator,
    CrossFittedCalibrationReport,
    IdentityFailureCalibrator,
    IsotonicFailureCalibrator,
    apply_failure_calibrator,
    calibration_metrics,
    cross_fit_failure_calibration,
    fit_beta_failure_calibrator,
    failure_calibrator_from_dict,
    fit_isotonic_failure_calibrator,
    spatial_block_groups,
)
from .graph import (
    CovisibilityFragilityReport,
    SpectralConnectivity,
    ThresholdSensitivityPoint,
    analyze_covisibility_fragility,
)
from .regions import (
    RegionDiagnosisConfig,
    RegionDiagnosticsReport,
    diagnose_regions,
)
from .risk_coverage import RiskCoverageReport, evaluate_risk_coverage
from .statistics import (
    ProportionInterval,
    beta_posterior_mean,
    empirical_bayes_prior,
    wilson_interval,
)

__all__ = [
    "BetaFailureCalibrator",
    "CovisibilityFragilityReport",
    "CrossFittedCalibrationReport",
    "IdentityFailureCalibrator",
    "IsotonicFailureCalibrator",
    "ProportionInterval",
    "RegionDiagnosisConfig",
    "RegionDiagnosticsReport",
    "RiskCoverageReport",
    "SpectralConnectivity",
    "ThresholdSensitivityPoint",
    "analyze_covisibility_fragility",
    "apply_failure_calibrator",
    "beta_posterior_mean",
    "calibration_metrics",
    "cross_fit_failure_calibration",
    "diagnose_regions",
    "empirical_bayes_prior",
    "evaluate_risk_coverage",
    "failure_calibrator_from_dict",
    "fit_beta_failure_calibrator",
    "fit_isotonic_failure_calibrator",
    "spatial_block_groups",
    "wilson_interval",
]
