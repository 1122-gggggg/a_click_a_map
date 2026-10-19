from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class Availability(str, Enum):
    """Evidence state. Missing information is never silently converted to zero."""

    AVAILABLE = "available"
    DERIVED = "derived"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    NOT_APPLICABLE = "not_applicable"


def normalize_localizer(value: str | None) -> str:
    """Keep a localizer name as provenance, never as a logic selector."""
    token = str(value or "unspecified").strip()
    return token or "unspecified"


class DecisionStatus(str, Enum):
    BLOCKED_METRIC_AUDIT = "BLOCKED_METRIC_AUDIT"
    KEEP_BASELINE = "KEEP_BASELINE"
    NAVIGATION_POLICY_ONLY = "NAVIGATION_POLICY_ONLY"
    EXISTING_DATA_REPAIR_FIRST = "EXISTING_DATA_REPAIR_FIRST"
    CONDITION_OR_SCHEDULE_REPAIR = "CONDITION_OR_SCHEDULE_REPAIR"
    EVIDENCE_CAPTURE_ONLY = "EVIDENCE_CAPTURE_ONLY"
    TARGETED_RECAPTURE_REQUIRED = "TARGETED_RECAPTURE_REQUIRED"


class CaptureMode(str, Enum):
    ANCHOR_BRIDGE = "ANCHOR_BRIDGE"
    OPERATIONAL_FORWARD = "OPERATIONAL_FORWARD"
    OPERATIONAL_REVERSE = "OPERATIONAL_REVERSE"
    LATERAL_OBLIQUE_LEFT = "LATERAL_OBLIQUE_LEFT"
    LATERAL_OBLIQUE_RIGHT = "LATERAL_OBLIQUE_RIGHT"
    HEIGHT_OBLIQUE_HIGH = "HEIGHT_OBLIQUE_HIGH"
    HEIGHT_OBLIQUE_LOW = "HEIGHT_OBLIQUE_LOW"
    CONDITION_REFERENCE = "CONDITION_REFERENCE"
    APPEARANCE_REFERENCE_ONLY = "APPEARANCE_REFERENCE_ONLY"
    HOLDOUT_EVIDENCE = "HOLDOUT_EVIDENCE"


def _finite_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field_name} must be boolean")


def _optional_vector3(raw: Any, field_name: str) -> tuple[float, float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers")
    vector = tuple(_finite_float(value, f"{field_name}[{index}]") for index, value in enumerate(raw))
    if math.sqrt(sum(value * value for value in vector)) <= 1e-12:
        raise ValueError(f"{field_name} must be non-zero")
    return vector


@dataclass(frozen=True)
class MetricValue:
    value: Any = None
    status: Availability = Availability.UNAVAILABLE
    reason: str = ""
    source: str = ""
    formula_version: str = ""
    confidence: float | None = None

    @classmethod
    def coerce(cls, raw: Any, *, source: str = "input") -> "MetricValue":
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, Mapping):
            status_raw = str(raw.get("status", "available" if "value" in raw else "unavailable")).lower()
            try:
                status = Availability(status_raw)
            except ValueError:
                status = Availability.INVALID
            confidence = raw.get("confidence")
            try:
                confidence_value = None if confidence is None else float(confidence)
            except (TypeError, ValueError):
                confidence_value = None
                status = Availability.INVALID
            if confidence_value is not None and (
                not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0
            ):
                status = Availability.INVALID
            return cls(
                value=raw.get("value"),
                status=status,
                reason=str(raw.get("reason", "")),
                source=str(raw.get("source", source)),
                formula_version=str(raw.get("formula_version", "")),
                confidence=confidence_value,
            )
        if raw is None:
            return cls(status=Availability.UNAVAILABLE, reason="missing", source=source)
        return cls(value=raw, status=Availability.AVAILABLE, source=source)

    @property
    def usable(self) -> bool:
        return self.status in {Availability.AVAILABLE, Availability.DERIVED, Availability.ESTIMATED}

    @property
    def gate_usable(self) -> bool:
        return self.status in {Availability.AVAILABLE, Availability.DERIVED}

    def finite_number(self) -> float | None:
        if not self.usable or isinstance(self.value, bool):
            return None
        try:
            value = float(self.value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status.value,
            "reason": self.reason,
            "source": self.source,
            "formula_version": self.formula_version,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PoseDirectionCell:
    cell_id: str
    region_id: str
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    localizer: str
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
    root_causes: tuple[str, ...] = ()
    condition: str = "default"
    directional_health: float | None = None
    position_best_health: float | None = None
    position_mean_health: float | None = None
    position_worst_health: float | None = None
    route_tangent: tuple[float, float, float] | None = None
    map_up_vector: tuple[float, float, float] | None = None
    operational_direction: bool = True
    map_producer: str = "unknown"

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_localizer: str = "unspecified",
    ) -> "PoseDirectionCell":
        if not isinstance(data, Mapping):
            raise ValueError("pose-direction record must be an object")

        position_raw = data.get("position")
        if position_raw is None:
            if not all(name in data and data.get(name) is not None for name in ("x", "y", "z")):
                raise ValueError("position or explicit x/y/z is required; missing geometry is never replaced with origin")
            position_raw = (data["x"], data["y"], data["z"])
        if isinstance(position_raw, (str, bytes)) or not isinstance(position_raw, Sequence) or len(position_raw) != 3:
            raise ValueError("position must contain exactly three numbers")
        position = tuple(_finite_float(value, f"position[{index}]") for index, value in enumerate(position_raw))

        yaw_raw = data.get("yaw_deg", data.get("yaw"))
        pitch_raw = data.get("pitch_deg", data.get("pitch"))
        if yaw_raw is None or pitch_raw is None:
            raise ValueError("yaw_deg/yaw and pitch_deg/pitch are required for every pose-direction record")
        yaw_deg = _finite_float(yaw_raw, "yaw_deg")
        pitch_deg = _finite_float(pitch_raw, "pitch_deg")

        metrics_raw = data.get("metrics", {})
        if not isinstance(metrics_raw, Mapping):
            raise ValueError("metrics must be an object")
        metrics = {str(name): MetricValue.coerce(value) for name, value in metrics_raw.items()}

        causes_raw = data.get("root_causes", data.get("codes", ()))
        if isinstance(causes_raw, str):
            causes = tuple(token.strip() for token in causes_raw.replace(",", "|").split("|") if token.strip())
        else:
            causes = tuple(str(token).strip() for token in causes_raw or () if str(token).strip())

        tangent = _optional_vector3(data.get("route_tangent"), "route_tangent")
        map_up = _optional_vector3(
            data.get("map_up_vector", data.get("map_up")),
            "map_up_vector",
        )
        localizer = normalize_localizer(
            data.get("localizer", data.get("backend", default_localizer))
        )

        def optional_float(*names: str) -> float | None:
            for name in names:
                value = data.get(name)
                if value is not None:
                    try:
                        return _finite_float(value, name)
                    except ValueError:
                        return None
            return None

        region_id = str(data.get("region_id") or data.get("weak_region_id") or "unassigned")
        cell_id = str(
            data.get("cell_id")
            or data.get("pose_id")
            or f"{region_id}:{position}:{yaw_deg}:{pitch_deg}"
        )
        producer = str(data.get("map_producer", data.get("producer", "unknown"))).strip().lower() or "unknown"
        operational_direction = _parse_bool(data.get("operational_direction", True), "operational_direction")

        return cls(
            cell_id=cell_id,
            region_id=region_id,
            position=position,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            localizer=localizer,
            metrics=metrics,
            root_causes=causes,
            condition=str(data.get("condition", "default")),
            directional_health=optional_float("directional_health", "directional_success_probability", "success_probability"),
            position_best_health=optional_float("position_best_health", "position_best_success_probability", "best_view_score"),
            position_mean_health=optional_float("position_mean_health", "position_mean_success_probability", "mean_view_score"),
            position_worst_health=optional_float("position_worst_health", "position_worst_success_probability", "worst_view_score"),
            route_tangent=tangent,
            map_up_vector=map_up,
            operational_direction=operational_direction,
            map_producer=producer,
        )

    def metric(self, name: str) -> MetricValue:
        return self.metrics.get(
            name,
            MetricValue(status=Availability.UNAVAILABLE, reason="not emitted", source="pose_cell"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "region_id": self.region_id,
            "position": list(self.position),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "localizer": self.localizer,
            "map_producer": self.map_producer,
            "condition": self.condition,
            "root_causes": list(self.root_causes),
            "directional_health": self.directional_health,
            "position_best_health": self.position_best_health,
            "position_mean_health": self.position_mean_health,
            "position_worst_health": self.position_worst_health,
            "route_tangent": None if self.route_tangent is None else list(self.route_tangent),
            "map_up_vector": None if self.map_up_vector is None else list(self.map_up_vector),
            "operational_direction": self.operational_direction,
            "metrics": {name: value.as_dict() for name, value in sorted(self.metrics.items())},
        }


@dataclass(frozen=True)
class CapturePose:
    position: tuple[float, float, float]
    yaw_deg: float | None
    pitch_deg: float | None
    role: str
    look_at: tuple[float, float, float] | None = None
    orientation_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "role": self.role,
            "look_at": None if self.look_at is None else list(self.look_at),
            "orientation_reason": self.orientation_reason,
        }


@dataclass(frozen=True)
class CapturePass:
    pass_id: str
    region_id: str
    mode: CaptureMode
    score: float
    confidence: float
    poses: tuple[CapturePose, ...]
    rationale: tuple[str, ...]
    expected_gain: Mapping[str, MetricValue]
    localizer_actions: tuple[str, ...] = ()
    acceptance_gates: tuple[str, ...] = ()
    safety_status: Availability = Availability.UNAVAILABLE
    safety_reason: str = "collision/geofence/dynamics not checked"
    map_units: str = "map_units"

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "region_id": self.region_id,
            "mode": self.mode.value,
            "score": self.score,
            "confidence": self.confidence,
            "poses": [pose.as_dict() for pose in self.poses],
            "rationale": list(self.rationale),
            "expected_gain": {name: value.as_dict() for name, value in sorted(self.expected_gain.items())},
            "localizer_actions": list(self.localizer_actions),
            "acceptance_gates": list(self.acceptance_gates),
            "safety_status": self.safety_status.value,
            "safety_reason": self.safety_reason,
            "map_units": self.map_units,
        }


@dataclass(frozen=True)
class RecaptureDecision:
    region_id: str
    localizer: str
    status: DecisionStatus
    recapture_required: bool
    confidence: float
    reasons: tuple[str, ...]
    non_capture_actions: tuple[str, ...]
    blocked_by: tuple[str, ...]
    capture_passes: tuple[CapturePass, ...]
    existing_data_repairability: float | None
    structural_health: float | None
    directional_sensitivity: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "localizer": self.localizer,
            "status": self.status.value,
            "recapture_required": self.recapture_required,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "non_capture_actions": list(self.non_capture_actions),
            "blocked_by": list(self.blocked_by),
            "existing_data_repairability": self.existing_data_repairability,
            "structural_health": self.structural_health,
            "directional_sensitivity": self.directional_sensitivity,
            "capture_passes": [item.as_dict() for item in self.capture_passes],
        }
