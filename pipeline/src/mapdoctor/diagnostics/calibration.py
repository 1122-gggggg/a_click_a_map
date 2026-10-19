"""Leakage-resistant calibration of localization-failure risk scores."""

from ._calibration_crossfit import (
    CalibrationCandidateResult,
    CrossFittedCalibrationReport,
    calibration_metrics,
    cross_fit_failure_calibration,
    spatial_block_groups,
)
from ._calibration_fit import (
    fit_beta_failure_calibrator,
    fit_isotonic_failure_calibrator,
)
from ._calibration_models import (
    BetaFailureCalibrator,
    FailureCalibrator,
    IdentityFailureCalibrator,
    IsotonicFailureCalibrator,
    apply_failure_calibrator,
    failure_calibrator_from_dict,
)

__all__ = [
    "BetaFailureCalibrator",
    "CalibrationCandidateResult",
    "CrossFittedCalibrationReport",
    "FailureCalibrator",
    "IdentityFailureCalibrator",
    "IsotonicFailureCalibrator",
    "apply_failure_calibrator",
    "calibration_metrics",
    "cross_fit_failure_calibration",
    "failure_calibrator_from_dict",
    "fit_beta_failure_calibrator",
    "fit_isotonic_failure_calibrator",
    "spatial_block_groups",
]
