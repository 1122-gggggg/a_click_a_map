from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import math
from pathlib import Path
from typing import Any, Sequence

from .metric_registry import (
    METRIC_BY_NAME,
    METRICS,
    MetricSpec,
    canonical_metric_name,
    required_metrics,
)
from .profiles import profile_for
from .types import Availability, MetricValue, PoseDirectionCell, normalize_localizer


class AuditStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    INVALID = "INVALID"
    ESTIMATED_ONLY = "ESTIMATED_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class MetricAuditItem:
    metric: str
    category: str
    description: str
    requirement: str
    status: AuditStatus
    record_count: int
    emitted_count: int
    gate_usable_count: int
    estimated_count: int
    unavailable_count: int
    invalid_count: int
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "status": self.status.value, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class DirectionalAudit:
    position_count: int
    position_direction_pairs: int
    positions_with_multiple_directions: int
    positions_with_external_direction_summary: int
    ready: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "warnings": list(self.warnings)}


@dataclass(frozen=True)
class MetricAuditReport:
    localizer: str
    scope: str
    record_count: int
    authorization_ready: bool
    integrity_ready: bool
    diagnostic_complete: bool
    directional: DirectionalAudit
    items: tuple[MetricAuditItem, ...]
    blocking_metrics: tuple[str, ...]
    missing_diagnostic_metrics: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def item(self, metric: str) -> MetricAuditItem | None:
        metric = canonical_metric_name(metric)
        return next((item for item in self.items if item.metric == metric), None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "localizer": self.localizer,
            "scope": self.scope,
            "record_count": self.record_count,
            "authorization_ready": self.authorization_ready,
            "integrity_ready": self.integrity_ready,
            "diagnostic_complete": self.diagnostic_complete,
            "directional": self.directional.as_dict(),
            "blocking_metrics": list(self.blocking_metrics),
            "missing_diagnostic_metrics": list(self.missing_diagnostic_metrics),
            "notes": list(self.notes),
            "items": [item.as_dict() for item in self.items],
        }


@dataclass(frozen=True)
class SourceAuditReport:
    repository: str
    localizer: str
    source_files_scanned: int
    test_files_scanned: int
    metrics_with_source_evidence: tuple[str, ...]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "metrics_with_source_evidence": list(self.metrics_with_source_evidence),
            "notes": list(self.notes),
        }


def _directional(cells: Sequence[PoseDirectionCell]) -> DirectionalAudit:
    grouped: dict[tuple[float, float, float], list[PoseDirectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[tuple(round(v, 6) for v in cell.position)].append(cell)
    multi = external = 0
    warnings: list[str] = []
    for pos, group in grouped.items():
        orientations = {(round(c.yaw_deg, 3), round(c.pitch_deg, 3)) for c in group}
        ext = any(
            c.position_best_health is not None
            and c.position_mean_health is not None
            and c.position_worst_health is not None
            for c in group
        )
        multi += int(len(orientations) >= 2)
        external += int(ext)
        if len(orientations) < 2 and not ext:
            warnings.append(f"position {pos} has one direction and no best/mean/worst summary")
    ready = bool(grouped) and all(
        len({(round(c.yaw_deg, 3), round(c.pitch_deg, 3)) for c in group}) >= 2
        or any(
            c.position_best_health is not None
            and c.position_mean_health is not None
            and c.position_worst_health is not None
            for c in group
        )
        for group in grouped.values()
    )
    return DirectionalAudit(len(grouped), len(cells), multi, external, ready, tuple(warnings))


def _validate(name: str, value: MetricValue) -> tuple[AuditStatus, str]:
    spec = METRIC_BY_NAME.get(name, MetricSpec(name, description=name.replace("_", " ")))
    if value.status == Availability.UNAVAILABLE:
        return AuditStatus.MISSING, value.reason or "unavailable"
    if value.status == Availability.INVALID:
        return AuditStatus.INVALID, value.reason or "invalid"
    if value.status == Availability.NOT_APPLICABLE:
        return AuditStatus.NOT_APPLICABLE, value.reason or "not applicable"
    if value.status == Availability.ESTIMATED and (spec.hard_gate or not spec.allow_estimated):
        return AuditStatus.ESTIMATED_ONLY, "estimated evidence cannot satisfy this gate"
    if (
        value.usable
        and isinstance(value.value, (int, float))
        and not isinstance(value.value, bool)
        and not math.isfinite(float(value.value))
    ):
        return AuditStatus.INVALID, "numeric value is not finite"
    if name in {
        "camera_intrinsics_valid",
        "frame_transform_valid",
        "handedness_valid",
        "existing_data_counterfactual_complete",
    }:
        return (AuditStatus.PASS, "") if value.value is True else (AuditStatus.INVALID, "required boolean is not true")
    if name == "coordinate_scale_status":
        token = str(value.value).lower().strip()
        accepted = {"metric", "meters", "metres", "m", "map_units", "arbitrary_sim3"}
        return (AuditStatus.PASS, "") if token in accepted else (AuditStatus.INVALID, "scale status must be explicit")
    if spec.minimum is not None or spec.maximum is not None:
        number = value.finite_number()
        if number is None:
            return AuditStatus.INVALID, "value is not finite"
        if spec.minimum is not None and number < spec.minimum:
            return AuditStatus.INVALID, "below domain minimum"
        if spec.maximum is not None and number > spec.maximum:
            return AuditStatus.INVALID, "above domain maximum"
    return AuditStatus.PASS, ""


def _producer_for(cells: Sequence[PoseDirectionCell]) -> str | None:
    producers = {cell.map_producer for cell in cells if cell.map_producer not in {"", "unknown"}}
    return next(iter(producers)) if len(producers) == 1 else None


def audit_cells(
    cells: Sequence[PoseDirectionCell],
    localizer: str = "unspecified",
    *,
    scope: str = "all",
) -> MetricAuditReport:
    localizer = normalize_localizer(localizer)
    profile = profile_for(localizer)
    producer = _producer_for(cells)

    integrity_names = set(profile.integrity_metrics)
    authorization_names = set(profile.recapture_authorization_metrics)
    authorization_names.update(
        {"existing_data_repairability", "existing_data_counterfactual_complete"}
    )
    diagnostic_names = set(profile.diagnostic_metrics)
    recommended_names = set(profile.recommended_metrics)
    if producer not in {"gluemap", "glue_map"}:
        diagnostic_names = {name for name in diagnostic_names if not name.startswith("gluemap_")}

    relevant_specs = required_metrics(localizer, map_producer=producer)
    required_names = {spec.name for spec in relevant_specs}
    required_names.update(authorization_names)
    required_names.update(
        canonical_metric_name(name)
        for cell in cells
        for name in cell.metrics
    )

    items: list[MetricAuditItem] = []
    blocking: list[str] = []
    missing_diag: list[str] = []
    for name in sorted(required_names):
        spec = METRIC_BY_NAME.get(
            name,
            MetricSpec(name, description=name.replace("_", " ")),
        )
        statuses: list[AuditStatus] = []
        reasons: list[str] = []
        emitted = gate_usable = estimated = unavailable = invalid = 0
        for cell in cells:
            value = cell.metric(name)
            status, reason = _validate(name, value)
            statuses.append(status)
            emitted += int(value.status not in {Availability.UNAVAILABLE, Availability.NOT_APPLICABLE})
            gate_usable += int(status == AuditStatus.PASS and value.gate_usable)
            estimated += int(value.status == Availability.ESTIMATED)
            unavailable += int(status == AuditStatus.MISSING)
            invalid += int(status == AuditStatus.INVALID)
            if reason:
                reasons.append(reason)
        if not cells or all(status == AuditStatus.MISSING for status in statuses):
            summary = AuditStatus.MISSING
        elif any(status == AuditStatus.INVALID for status in statuses):
            summary = AuditStatus.INVALID
        elif all(status == AuditStatus.ESTIMATED_ONLY for status in statuses):
            summary = AuditStatus.ESTIMATED_ONLY
        elif all(status == AuditStatus.PASS for status in statuses):
            summary = AuditStatus.PASS
        else:
            summary = AuditStatus.PARTIAL

        if name in integrity_names:
            requirement = "INTEGRITY"
        elif name in authorization_names:
            requirement = "RECAPTURE_AUTHORIZATION"
        elif name in diagnostic_names:
            requirement = "DIAGNOSTIC"
        elif name in recommended_names:
            requirement = "RECOMMENDED"
        else:
            requirement = "DIAGNOSTIC"

        item = MetricAuditItem(
            name,
            spec.category,
            spec.description,
            requirement,
            summary,
            len(cells),
            emitted,
            gate_usable,
            estimated,
            unavailable,
            invalid,
            tuple(sorted(set(reasons))),
        )
        items.append(item)
        if requirement in {"INTEGRITY", "RECAPTURE_AUTHORIZATION"} and summary != AuditStatus.PASS:
            blocking.append(name)
        if requirement == "DIAGNOSTIC" and summary != AuditStatus.PASS:
            missing_diag.append(name)

    directional = _directional(cells)
    integrity_ready = not any(name in integrity_names for name in blocking)
    authorization_ready = (
        integrity_ready
        and not any(name in authorization_names for name in blocking)
        and directional.ready
    )
    notes: list[str] = []
    if producer is None:
        notes.append("map producer is unknown or mixed; producer-specific diagnostics were not required")
    elif producer in {"gluemap", "glue_map"}:
        notes.append("GLUEMAP producer-specific diagnostics are included")
    return MetricAuditReport(
        localizer,
        scope,
        len(cells),
        authorization_ready,
        integrity_ready,
        not missing_diag,
        directional,
        tuple(items),
        tuple(blocking),
        tuple(missing_diag),
        tuple(notes),
    )


def audit_by_region(
    cells: Sequence[PoseDirectionCell],
    localizer: str = "unspecified",
) -> dict[str, MetricAuditReport]:
    grouped: dict[str, list[PoseDirectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.region_id].append(cell)
    return {
        region: audit_cells(group, localizer, scope=region)
        for region, group in grouped.items()
    }


def audit_source_repository(
    path: str | Path,
    localizer: str = "unspecified",
) -> SourceAuditReport:
    """Conservative static reference scan.

    Registry/profile declarations and tests are excluded from implementation
    evidence; otherwise every registered metric would trivially appear in the
    source and the audit would produce false confidence. Runtime audit remains
    authoritative.
    """
    root = Path(path)
    localizer = normalize_localizer(localizer)
    all_python = [path for path in root.rglob("*.py") if ".git" not in path.parts]
    test_files = [
        path for path in all_python if "test" in path.name.lower() or "tests" in path.parts
    ]
    excluded_names = {"metric_registry.py", "profiles.py"}
    source_files = [
        path
        for path in all_python
        if path not in test_files and path.name not in excluded_names
    ]
    contents = {
        path: path.read_text(encoding="utf-8", errors="ignore") for path in source_files
    }
    evidence = tuple(
        spec.name
        for spec in METRICS
        if any(spec.name in text for text in contents.values())
    )
    return SourceAuditReport(
        str(root),
        localizer,
        len(source_files),
        len(test_files),
        evidence,
        (
            "static implementation-reference scan only; runtime metric audit is authoritative",
            "metric registry/profile declarations and tests are excluded from source evidence",
        ),
    )
