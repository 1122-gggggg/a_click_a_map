from __future__ import annotations

from dataclasses import dataclass

from .schema import RiskClass, SpatialDiagnostic


@dataclass(frozen=True)
class RiskThresholds:
    good_failure_max: float = 0.20
    direction_sensitive_best_max: float = 0.20
    direction_sensitive_worst_min: float = 0.60
    dead_zone_best_min: float = 0.80
    low_effective_landmarks: float = 50.0
    low_fim_lambda_min: float = 1e-4
    high_fim_condition: float = 1e8


def classify_spatial_risk(
    rows: list[SpatialDiagnostic], thresholds: RiskThresholds | None = None
) -> None:
    """Classify positions after calibrated per-orientation probabilities exist."""

    limits = thresholds or RiskThresholds()
    grouped: dict[tuple[float, float, float], list[SpatialDiagnostic]] = {}
    for row in rows:
        grouped.setdefault(row.position, []).append(row)
    for group in grouped.values():
        if max(int(row.visible_landmarks) for row in group) <= 0:
            for row in group:
                row.risk_class = RiskClass.UNKNOWN
                row.primary_failure_causes = ["UNOCCUPIED"]
                row.recommended_action = None
                row.recommended_capture_position = None
            continue
        available = [
            row
            for row in group
            if row.failure_probability is not None and int(row.visible_landmarks) > 0
        ]
        unoccupied = [row for row in group if int(row.visible_landmarks) <= 0]
        for row in unoccupied:
            row.risk_class = RiskClass.UNKNOWN
            row.primary_failure_causes = ["UNOCCUPIED"]
            row.recommended_action = None
            row.recommended_capture_position = None
        occupied = [row for row in group if int(row.visible_landmarks) > 0]
        if not occupied:
            continue
        if not available:
            _classify_map_only_group(occupied, limits)
            continue
        if len(available) != len(occupied):
            for row in occupied:
                row.risk_class = RiskClass.UNKNOWN
            continue
        best = min(available, key=lambda row: float(row.failure_probability))
        best_failure = float(best.failure_probability)
        worst_failure = max(float(row.failure_probability) for row in available)
        if best_failure >= limits.dead_zone_best_min:
            risk_class = RiskClass.DEAD_ZONE
            action = "SUPPLEMENTAL_CAPTURE"
        elif (
            best_failure <= limits.direction_sensitive_best_max
            and worst_failure >= limits.direction_sensitive_worst_min
        ):
            risk_class = RiskClass.VIEW_DIRECTION_SENSITIVE
            action = "REORIENT_CAMERA"
        elif worst_failure <= limits.good_failure_max:
            risk_class = RiskClass.GOOD
            action = None
        else:
            risk_class = RiskClass.WEAK
            action = "CONDITIONAL_CAPTURE_OR_REORIENT"
        for row in occupied:
            row.risk_class = risk_class
            row.primary_failure_causes = _causes(row, limits)
            row.recommended_action = action
            if action is not None:
                row.recommended_yaw_deg = best.yaw_deg
                row.recommended_pitch_deg = best.pitch_deg
            if action == "SUPPLEMENTAL_CAPTURE":
                row.recommended_capture_position = list(row.position)


def _orientation_supported(row: SpatialDiagnostic, limits: RiskThresholds) -> bool:
    if int(row.visible_landmarks) < 10:
        return False
    if float(row.effective_landmarks) < limits.low_effective_landmarks:
        return False
    if row.parallax_p10_deg is not None and row.parallax_p10_deg < 1.0:
        return False
    flags = set(row.degeneracy_flags)
    if "LOW_PARALLAX" in flags or "ONE_SIDED_OBSERVATION" in flags:
        return False
    return True


def _classify_map_only_group(
    group: list[SpatialDiagnostic], limits: RiskThresholds
) -> None:
    """Geometry screen when failure probabilities are unavailable.

    Corridor collinearity alone is not a dead zone. Classification uses the
    best occupied orientation so a linear path is not marked dead everywhere.
    """

    occupied = [row for row in group if int(row.visible_landmarks) > 0]
    if not occupied:
        for row in group:
            row.risk_class = RiskClass.UNKNOWN
            row.primary_failure_causes = ["UNOCCUPIED"]
            row.recommended_action = None
            row.recommended_capture_position = None
        return
    supported = [row for row in occupied if _orientation_supported(row, limits)]
    best = max(occupied, key=lambda row: int(row.visible_landmarks))
    if not supported:
        if max(int(row.visible_landmarks) for row in occupied) < 10:
            risk_class = RiskClass.DEAD_ZONE
            action = "SUPPLEMENTAL_CAPTURE"
        else:
            risk_class = RiskClass.WEAK
            action = "CONDITIONAL_CAPTURE_OR_REORIENT"
    elif len(supported) != len(occupied):
        risk_class = RiskClass.VIEW_DIRECTION_SENSITIVE
        action = "REORIENT_CAMERA"
        best = max(supported, key=lambda row: int(row.visible_landmarks))
    else:
        risk_class = RiskClass.GOOD
        action = None
    for row in group:
        if int(row.visible_landmarks) <= 0:
            row.risk_class = RiskClass.UNKNOWN
            row.primary_failure_causes = ["UNOCCUPIED"]
            row.recommended_action = None
            continue
        row.risk_class = risk_class
        row.primary_failure_causes = _causes(row, limits)
        row.recommended_action = action
        if action == "SUPPLEMENTAL_CAPTURE":
            row.recommended_capture_position = list(row.position)
            row.recommended_yaw_deg = best.yaw_deg
            row.recommended_pitch_deg = best.pitch_deg
        elif action is not None:
            row.recommended_yaw_deg = best.yaw_deg
            row.recommended_pitch_deg = best.pitch_deg


def _causes(row: SpatialDiagnostic, limits: RiskThresholds) -> list[str]:
    causes: list[str] = []
    if row.edm_loo_success_rate is not None and row.edm_loo_success_rate < 0.5:
        causes.append("EDM_EMPIRICAL_FAILURE_HIGH")
    if (
        row.num_pose_modes is not None
        and row.num_pose_modes >= 2
        and row.mode_support_margin is not None
        and row.mode_support_margin < 0.2
    ):
        causes.append("PERCEPTUAL_ALIASING_MULTIMODAL")
    if (
        row.fim_lambda_min is not None
        and row.fim_lambda_min < limits.low_fim_lambda_min
    ) or (
        row.fim_condition is not None
        and row.fim_condition > limits.high_fim_condition
    ):
        causes.append("LOW_FIM_EIGENVALUE")
    if row.effective_landmarks < limits.low_effective_landmarks:
        causes.append("WEAK_LANDMARK_SUPPORT")
    if row.parallax_p10_deg is not None and row.parallax_p10_deg < 1.0:
        causes.append("LOW_PARALLAX")
    if "ONE_SIDED_OBSERVATION" in row.degeneracy_flags:
        causes.append("ONE_SIDED_OBSERVATION")
    if (
        row.consecutive_failure_probability is not None
        and row.consecutive_failure_probability > 0.1
    ):
        causes.append("TEMPORAL_FAILURE_BURST")
    return causes
