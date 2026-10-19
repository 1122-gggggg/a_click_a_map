from __future__ import annotations

from typing import Any, Mapping, Sequence

from mapdoctor.model import MapModel

from .bridge import enrich_pose_cells_from_model
from .planner import plan_regions
from .profiles import CaptureGeometry, PlannerThresholds
from .types import PoseDirectionCell, normalize_localizer


def analyze_pose_cells(
    records: Sequence[PoseDirectionCell | Mapping[str, Any]],
    *,
    localizer: str = "unspecified",
    thresholds: PlannerThresholds | None = None,
    capture: CaptureGeometry | None = None,
    model: MapModel | None = None,
) -> dict[str, Any]:
    """In-process integration API for weak-region repair planning.

    The caller remains responsible for computing/instrumenting the metrics in
    each pose-direction record. When ``model`` is supplied, only producer
    provenance is inherited; absent pose-local evidence is never replaced with
    sparse-map-wide proxy values.
    """
    localizer = normalize_localizer(localizer)
    cells = tuple(
        record
        if isinstance(record, PoseDirectionCell)
        else PoseDirectionCell.from_dict(record, default_localizer=localizer)
        for record in records
    )
    if model is not None:
        cells = enrich_pose_cells_from_model(cells, model)
    decisions, audits = plan_regions(
        cells,
        localizer,
        thresholds=thresholds,
        capture=capture,
    )
    return {
        "schema_version": 2,
        "localizer": localizer,
        "decisions": [decision.as_dict() for decision in decisions],
        "metric_audit_by_region": {
            region_id: report.as_dict() for region_id, report in audits.items()
        },
    }


def attach_recapture_analysis(
    report: Mapping[str, Any],
    *,
    localizer: str = "unspecified",
    pose_cells_key: str = "pose_cells",
    output_key: str = "recapture_analysis",
    thresholds: PlannerThresholds | None = None,
    capture: CaptureGeometry | None = None,
    model: MapModel | None = None,
) -> dict[str, Any]:
    """Return a copy of an existing report with audited recapture analysis attached."""
    raw = report.get(pose_cells_key)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"report[{pose_cells_key!r}] must be a sequence of pose-direction records")
    result = dict(report)
    result[output_key] = analyze_pose_cells(
        raw,
        localizer=localizer,
        thresholds=thresholds,
        capture=capture,
        model=model,
    )
    return result
