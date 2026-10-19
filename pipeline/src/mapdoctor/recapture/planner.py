from __future__ import annotations

from collections import Counter, defaultdict
import math
from statistics import median
from typing import Mapping, Sequence

from .audit import MetricAuditReport, audit_by_region
from .profiles import CaptureGeometry, PlannerThresholds, profile_for
from .types import (
    Availability,
    CaptureMode,
    CapturePass,
    CapturePose,
    DecisionStatus,
    MetricValue,
    PoseDirectionCell,
    RecaptureDecision,
    normalize_localizer,
)


def _numbers(cells: Sequence[PoseDirectionCell], name: str) -> list[float]:
    return [value for cell in cells if (value := cell.metric(name).finite_number()) is not None]


def _num(cells: Sequence[PoseDirectionCell], name: str) -> float | None:
    values = _numbers(cells, name)
    return float(median(values)) if values else None


def _min_num(cells: Sequence[PoseDirectionCell], name: str) -> float | None:
    values = _numbers(cells, name)
    return min(values) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _weighted_success(cells: Sequence[PoseDirectionCell]) -> float | None:
    weighted = 0.0
    total = 0.0
    fallback: list[float] = []
    for cell in cells:
        success = cell.metric("localization_success_rate").finite_number()
        attempts = cell.metric("attempt_count").finite_number()
        if success is None:
            continue
        fallback.append(success)
        if attempts is not None and attempts > 0:
            weighted += success * attempts
            total += attempts
    if total > 0:
        return weighted / total
    return float(median(fallback)) if fallback else None


def _directional(
    cells: Sequence[PoseDirectionCell],
) -> tuple[float | None, float | None, float | None, float | None]:
    grouped: dict[tuple[float, float, float], list[PoseDirectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[tuple(round(value, 6) for value in cell.position)].append(cell)

    per_position: list[tuple[float, float, float]] = []
    for group in grouped.values():
        external = next(
            (
                (
                    cell.position_best_health,
                    cell.position_mean_health,
                    cell.position_worst_health,
                )
                for cell in group
                if cell.position_best_health is not None
                and cell.position_mean_health is not None
                and cell.position_worst_health is not None
            ),
            None,
        )
        if external is not None:
            per_position.append((float(external[0]), float(external[1]), float(external[2])))
            continue
        raw = [cell.directional_health for cell in group if cell.directional_health is not None]
        if raw:
            per_position.append((max(raw), sum(raw) / len(raw), min(raw)))

    if not per_position:
        return None, None, None, None
    best = _mean([item[0] for item in per_position])
    mean = _mean([item[1] for item in per_position])
    worst = _mean([item[2] for item in per_position])
    sensitivity = best - mean if best is not None and mean is not None else None
    return best, mean, worst, sensitivity


def _structural_health(cells: Sequence[PoseDirectionCell], thresholds: PlannerThresholds) -> float | None:
    """Return structural health only when the complete core evidence set exists."""
    required = {
        "visible_landmark_count": _num(cells, "visible_landmark_count"),
        "inlier_convex_hull_coverage": _num(cells, "inlier_convex_hull_coverage"),
        "grid_occupancy_count": _num(cells, "grid_occupancy_count"),
        "positive_depth_ratio": _num(cells, "positive_depth_ratio"),
        "fim_rank": _num(cells, "fim_rank"),
        "fim_condition_number": _num(cells, "fim_condition_number"),
        "triangulation_angle_p10_deg": _num(cells, "triangulation_angle_p10_deg"),
        "view_direction_entropy": _num(cells, "view_direction_entropy"),
    }
    if any(value is None for value in required.values()):
        return None

    components = [
        min(1.0, required["visible_landmark_count"] / max(thresholds.min_visible_landmarks, 1)),
        min(1.0, required["inlier_convex_hull_coverage"] / max(thresholds.min_hull_coverage, 1e-9)),
        min(1.0, required["grid_occupancy_count"] / max(thresholds.min_grid_occupancy, 1)),
        min(1.0, required["positive_depth_ratio"] / max(thresholds.min_positive_depth_ratio, 1e-9)),
        min(1.0, required["fim_rank"] / max(thresholds.min_fim_rank, 1)),
        min(1.0, thresholds.max_fim_condition_number / max(required["fim_condition_number"], 1.0)),
        min(
            1.0,
            required["triangulation_angle_p10_deg"]
            / max(thresholds.min_triangulation_angle_p10_deg, 1e-9),
        ),
        min(1.0, required["view_direction_entropy"] / max(thresholds.min_view_direction_entropy, 1e-9)),
    ]

    effective_rank = _num(cells, "fim_effective_rank")
    if effective_rank is not None:
        components.append(min(1.0, effective_rank / max(thresholds.min_fim_effective_rank, 1e-9)))
    track_length = _num(cells, "track_length_p50")
    if track_length is not None:
        components.append(min(1.0, track_length / max(thresholds.min_track_length_p50, 1e-9)))
    reprojection = _num(cells, "reprojection_error_p90_px")
    if reprojection is not None:
        components.append(min(1.0, thresholds.max_reprojection_p90_px / max(reprojection, 1e-9)))
    return sum(max(0.0, min(1.0, component)) for component in components) / len(components)


def _repairability(
    cells: Sequence[PoseDirectionCell],
) -> tuple[float | None, str | None]:
    """Use only measured counterfactual evidence, independent of localizer type."""
    return _num(cells, "existing_data_repairability"), None


def _repair_actions(stage: str | None) -> tuple[str, ...]:
    profile = profile_for()
    if stage and stage in profile.repair_actions:
        return profile.repair_actions[stage]
    return (
        "run targeted rematching on weak/anchor/cross-route pairs",
        "merge fragmented tracks and retriangulate existing observations",
        "run anchored local BA, then rerun the frozen holdout evaluation",
        "emit measured existing_data_repairability from that counterfactual",
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float] | None:
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        return None
    return tuple(value / norm for value in vector)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[index] * b[index] for index in range(3))


def _route_basis(
    cells: Sequence[PoseDirectionCell],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
    tangents = [cell.route_tangent for cell in cells if cell.route_tangent is not None]
    up_vectors = [cell.map_up_vector for cell in cells if cell.map_up_vector is not None]
    if not tangents or not up_vectors:
        return None
    forward = _normalize(tuple(sum(vector[index] for vector in tangents) for index in range(3)))
    up = _normalize(tuple(sum(vector[index] for vector in up_vectors) for index in range(3)))
    if forward is None or up is None:
        return None
    lateral = _normalize(_cross(up, forward))
    if lateral is None:
        # A route parallel to the map-up vector has no unambiguous left/right
        # without an additional body/reference axis, so fail closed.
        return None
    vertical = _normalize(_cross(forward, lateral))
    if vertical is None:
        return None
    if _dot(vertical, up) < 0:
        vertical = tuple(-value for value in vertical)
        lateral = tuple(-value for value in lateral)
    return forward, lateral, vertical


def _offset(
    center: tuple[float, float, float],
    axis: tuple[float, float, float],
    distance: float,
) -> tuple[float, float, float]:
    return tuple(center[index] + axis[index] * distance for index in range(3))


def _poses(
    center: tuple[float, float, float],
    mode: CaptureMode,
    geometry: CaptureGeometry,
    yaw: float,
    pitch: float,
    basis: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[CapturePose, ...]:
    forward, lateral, vertical = basis
    seed_reason = "operational yaw/pitch reused from measured pose-cell evidence"
    look_reason = "exact Euler angles depend on the deployment frame/camera convention; aim at look_at target"
    if mode == CaptureMode.ANCHOR_BRIDGE:
        return (
            CapturePose(
                _offset(center, forward, -geometry.anchor_extension),
                yaw,
                pitch,
                "healthy_anchor_in",
                look_at=center,
                orientation_reason=seed_reason,
            ),
            CapturePose(center, yaw, pitch, "weak_center", look_at=None, orientation_reason=seed_reason),
            CapturePose(
                _offset(center, forward, geometry.anchor_extension),
                yaw,
                pitch,
                "healthy_anchor_out",
                look_at=center,
                orientation_reason=seed_reason,
            ),
        )
    if mode == CaptureMode.LATERAL_OBLIQUE_LEFT:
        return (
            CapturePose(
                _offset(center, lateral, geometry.lateral_offset),
                None,
                None,
                "left_oblique",
                look_at=center,
                orientation_reason=look_reason,
            ),
        )
    if mode == CaptureMode.LATERAL_OBLIQUE_RIGHT:
        return (
            CapturePose(
                _offset(center, lateral, -geometry.lateral_offset),
                None,
                None,
                "right_oblique",
                look_at=center,
                orientation_reason=look_reason,
            ),
        )
    if mode == CaptureMode.HEIGHT_OBLIQUE_HIGH:
        return (
            CapturePose(
                _offset(center, vertical, geometry.height_offset),
                None,
                None,
                "high_oblique",
                look_at=center,
                orientation_reason=look_reason,
            ),
        )
    if mode == CaptureMode.HEIGHT_OBLIQUE_LOW:
        return (
            CapturePose(
                _offset(center, vertical, -geometry.height_offset),
                None,
                None,
                "low_oblique",
                look_at=center,
                orientation_reason=look_reason,
            ),
        )
    if mode == CaptureMode.OPERATIONAL_REVERSE:
        return (
            CapturePose(
                center,
                None,
                None,
                "operational_reverse",
                look_at=_offset(center, forward, -geometry.path_half_length),
                orientation_reason=look_reason,
            ),
        )
    return (
        CapturePose(center, yaw, pitch, mode.value.lower(), look_at=None, orientation_reason=seed_reason),
    )


def _weakest_mode(cells: Sequence[PoseDirectionCell]) -> str:
    modes: list[str] = []
    for cell in cells:
        raw = cell.metric("fim_weakest_mode").value
        if isinstance(raw, Mapping):
            token = raw.get("dominant_dof")
            if token:
                modes.append(str(token).lower())
        elif raw:
            modes.append(str(raw).lower())
    return Counter(modes).most_common(1)[0][0] if modes else ""


def _passes(
    cells: Sequence[PoseDirectionCell],
    geometry: CaptureGeometry,
) -> tuple[CapturePass, ...]:
    basis = _route_basis(cells)
    if basis is None:
        return ()
    center = tuple(sum(cell.position[index] for cell in cells) / len(cells) for index in range(3))
    yaw = float(median([cell.yaw_deg for cell in cells]))
    pitch = float(median([cell.pitch_deg for cell in cells]))
    weakest = _weakest_mode(cells)
    causes = {cause.lower() for cell in cells for cause in cell.root_causes}

    modes = [CaptureMode.ANCHOR_BRIDGE, CaptureMode.OPERATIONAL_FORWARD]
    if any("reverse" in cause for cause in causes):
        modes.append(CaptureMode.OPERATIONAL_REVERSE)
    modes.extend([CaptureMode.LATERAL_OBLIQUE_LEFT, CaptureMode.LATERAL_OBLIQUE_RIGHT])
    if weakest in {"tz", "rz"} or any(
        token in cause for cause in causes for token in ("vertical", "height", "pitch")
    ):
        modes.extend([CaptureMode.HEIGHT_OBLIQUE_HIGH, CaptureMode.HEIGHT_OBLIQUE_LOW])

    result: list[CapturePass] = []
    for index, mode in enumerate(modes[: geometry.max_passes]):
        expected_gain = {
            "delta_fim_lambda_min": MetricValue(
                status=Availability.UNAVAILABLE,
                reason="must be recomputed from candidate visibility/FIM; no scale-dependent constant is invented",
                source="recapture.planner",
            )
        }
        result.append(
            CapturePass(
                f"{cells[0].region_id}:{index + 1}",
                cells[0].region_id,
                mode,
                max(0.0, 1.0 - index * 0.05),
                0.65,
                _poses(center, mode, geometry, yaw, pitch, basis),
                ("increase missing view/baseline support while preserving a bridge to the healthy map",),
                expected_gain,
                _repair_actions(None),
                (
                    "B1 must improve the frozen weak-region holdout",
                    "stable-region regression must not worsen",
                ),
                map_units=geometry.map_units,
            )
        )
    return tuple(result)


def decide_region(
    cells: Sequence[PoseDirectionCell],
    audit: MetricAuditReport,
    localizer: str,
    thresholds: PlannerThresholds,
    geometry: CaptureGeometry,
) -> RecaptureDecision:
    region = cells[0].region_id
    success = _weighted_success(cells)
    attempts = _min_num(cells, "attempt_count")
    coverage = _min_num(cells, "holdout_query_coverage")
    structural = _structural_health(cells, thresholds)
    repairability, stage = _repairability(cells)
    best, mean, _, sensitivity = _directional(cells)
    base = dict(
        region_id=region,
        localizer=localizer,
        existing_data_repairability=repairability,
        structural_health=structural,
        directional_sensitivity=sensitivity,
    )

    if not audit.integrity_ready:
        return RecaptureDecision(
            **base,
            status=DecisionStatus.BLOCKED_METRIC_AUDIT,
            recapture_required=False,
            confidence=0.95,
            reasons=("coordinate/intrinsic integrity gates failed",),
            non_capture_actions=("fix map/runtime frame and intrinsics before diagnosis",),
            blocked_by=audit.blocking_metrics,
            capture_passes=(),
        )
    if (
        attempts is None
        or coverage is None
        or attempts < thresholds.min_attempt_count
        or coverage < thresholds.min_holdout_coverage
        or success is None
    ):
        return RecaptureDecision(
            **base,
            status=DecisionStatus.EVIDENCE_CAPTURE_ONLY,
            recapture_required=False,
            confidence=0.4,
            reasons=("holdout localization evidence is insufficient",),
            non_capture_actions=("collect query-only evidence without adding it to the map",),
            blocked_by=tuple(
                name
                for name in audit.blocking_metrics
                if name in {"attempt_count", "localization_success_rate", "holdout_query_coverage"}
            ),
            capture_passes=(),
        )
    if best is not None and mean is not None and best >= thresholds.navigation_best_health and best - mean >= thresholds.navigation_mean_gap:
        return RecaptureDecision(
            **base,
            status=DecisionStatus.NAVIGATION_POLICY_ONLY,
            recapture_required=False,
            confidence=0.8,
            reasons=("weakness is strongly orientation-dependent",),
            non_capture_actions=("turn camera/body toward the best LocMap direction before entering the weak segment",),
            blocked_by=(),
            capture_passes=(),
        )
    if success >= thresholds.healthy_success_rate and (
        structural is None or structural >= thresholds.weak_structural_health
    ):
        return RecaptureDecision(
            **base,
            status=DecisionStatus.KEEP_BASELINE,
            recapture_required=False,
            confidence=0.8,
            reasons=("held-out localization is healthy",),
            non_capture_actions=("keep B0 immutable and monitor condition strata",),
            blocked_by=(),
            capture_passes=(),
        )

    illumination = _num(cells, "illumination_direct_visibility_ratio")
    condition = _num(cells, "condition_match_score")
    if structural is not None and structural >= thresholds.weak_structural_health and (
        (illumination is not None and illumination < thresholds.min_illumination_ratio)
        or (condition is not None and condition < thresholds.min_condition_match_score)
    ):
        return RecaptureDecision(
            **base,
            status=DecisionStatus.CONDITION_OR_SCHEDULE_REPAIR,
            recapture_required=False,
            confidence=0.7,
            reasons=("geometry is adequate but condition/illumination support is weak",),
            non_capture_actions=("change flight time/lighting or add condition-matched appearance references",),
            blocked_by=(),
            capture_passes=(),
        )

    if repairability is None:
        return RecaptureDecision(
            **base,
            status=DecisionStatus.EXISTING_DATA_REPAIR_FIRST,
            recapture_required=False,
            confidence=0.55,
            reasons=("existing-data repairability is unknown; unknown is not evidence for recapture",),
            non_capture_actions=_repair_actions(stage),
            blocked_by=("existing_data_repairability",),
            capture_passes=(),
        )
    if repairability >= thresholds.min_existing_data_repairability:
        return RecaptureDecision(
            **base,
            status=DecisionStatus.EXISTING_DATA_REPAIR_FIRST,
            recapture_required=False,
            confidence=0.75,
            reasons=("existing images contain recoverable support",),
            non_capture_actions=_repair_actions(stage),
            blocked_by=(),
            capture_passes=(),
        )

    if success < thresholds.weak_success_rate and structural is not None and structural < thresholds.weak_structural_health:
        if not audit.authorization_ready:
            return RecaptureDecision(
                **base,
                status=DecisionStatus.EVIDENCE_CAPTURE_ONLY,
                recapture_required=False,
                confidence=0.5,
                reasons=("recapture is plausible but hard authorization evidence is incomplete",),
                non_capture_actions=("complete hard metrics and rerun existing-data counterfactual repairs",),
                blocked_by=audit.blocking_metrics
                + (() if audit.directional.ready else ("directional_pose_lattice",)),
                capture_passes=(),
            )
        capture_passes = _passes(cells, geometry)
        if not capture_passes:
            missing_geometry = []
            if not any(cell.route_tangent is not None for cell in cells):
                missing_geometry.append("route_tangent")
            if not any(cell.map_up_vector is not None for cell in cells):
                missing_geometry.append("map_up_vector")
            if not missing_geometry:
                missing_geometry.append("nondegenerate_route_frame")
            return RecaptureDecision(
                **base,
                status=DecisionStatus.TARGETED_RECAPTURE_REQUIRED,
                recapture_required=True,
                confidence=0.7,
                reasons=(
                    "repeatable localization failure coincides with intrinsic structural weakness and low existing-data repairability",
                    "capture geometry cannot be instantiated without an explicit nondegenerate route frame",
                ),
                non_capture_actions=("supply validated route_tangent and map_up_vector in the map frame",),
                blocked_by=tuple(missing_geometry),
                capture_passes=(),
            )
        return RecaptureDecision(
            **base,
            status=DecisionStatus.TARGETED_RECAPTURE_REQUIRED,
            recapture_required=True,
            confidence=0.85,
            reasons=("repeatable localization failure coincides with intrinsic structural weakness and low existing-data repairability",),
            non_capture_actions=("freeze B0/query/config hashes before capture",),
            blocked_by=(),
            capture_passes=capture_passes,
        )

    return RecaptureDecision(
        **base,
        status=DecisionStatus.EXISTING_DATA_REPAIR_FIRST,
        recapture_required=False,
        confidence=0.6,
        reasons=("physical information deficit is not proven",),
        non_capture_actions=_repair_actions(stage),
        blocked_by=(),
        capture_passes=(),
    )


def plan_regions(
    cells: Sequence[PoseDirectionCell],
    localizer: str = "unspecified",
    *,
    thresholds: PlannerThresholds | None = None,
    capture: CaptureGeometry | None = None,
) -> tuple[tuple[RecaptureDecision, ...], Mapping[str, MetricAuditReport]]:
    localizer = normalize_localizer(localizer)
    thresholds = thresholds or PlannerThresholds()
    capture = capture or CaptureGeometry()
    grouped: dict[str, list[PoseDirectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.region_id].append(cell)
    audits = audit_by_region(cells, localizer)
    decisions = tuple(
        decide_region(group, audits[region], localizer, thresholds, capture)
        for region, group in sorted(grouped.items())
    )
    return decisions, audits
