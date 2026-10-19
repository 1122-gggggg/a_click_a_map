"""Role-correct historical experiment reports and integrity receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from river_map_quality.historical_experiment import HistoricalExperimentError
from river_map_quality.historical_stages import STAGE_ORDER
from river_map_quality.river_mvroma_contracts import (
    BASE_VIDEO_NAMES,
    UPDATE_VIDEO_NAMES,
    VALIDATION_VIDEO_NAMES,
    write_new_json,
)

HISTORICAL_REPORT_SCHEMA = "RIVER_HISTORICAL_EDM_UPDATE_REPORT_V1"
INTEGRITY_RECEIPT_SCHEMA = "RIVER_HISTORICAL_INTEGRITY_RECEIPT_V1"


class HistoricalReportError(HistoricalExperimentError):
    """Raised when a report would mix diagnostic historical geometry into B0."""


def _source_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("source_role") or "unknown") for row in rows))


def render_historical_report(
    *,
    run_root,
    stage_results: Mapping[str, Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    selected_stage: str,
    b0_receipt: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> dict[str, object]:
    if selected_stage not in STAGE_ORDER:
        raise HistoricalReportError(f"selected stage is not an E0-E5 stage: {selected_stage}")
    if any(row.get("merged_into_b0") is True for row in reference_rows):
        raise HistoricalReportError("historical-only diagnostic geometry cannot merge into B0")
    artifact = {
        "schema_version": 1,
        "artifact_type": HISTORICAL_REPORT_SCHEMA,
        "roles": {
            "base": list(BASE_VIDEO_NAMES),
            "update": list(UPDATE_VIDEO_NAMES),
            "validation": list(VALIDATION_VIDEO_NAMES),
        },
        "selected_stage": selected_stage,
        "stage_results": dict(stage_results),
        "reference_source_distribution": _source_distribution(reference_rows),
        "visual_contract": {
            "b0_rgb_cloud_immutable": True,
            "historical_frusta_separate": True,
            "historical_only_points_are_diagnostic": True,
            "never_visually_merge_historical_only_geometry": True,
        },
        "b0": {
            "pose_table_sha256": b0_receipt["pose_table_sha256"],
            "point_xyz_sha256": b0_receipt["point_xyz_sha256"],
        },
        "integrity": dict(integrity),
    }
    write_new_json(run_root, "reports/FINAL_REPORT.json", artifact)
    markdown = [
        "# River historical-view MegaLoc + EDM update",
        "",
        f"- Selected stage: **{selected_stage}**.",
        "- Current map remains immutable B0; historical geometry is derivative/diagnostic.",
        "- P157 is validation-only and never entered candidate selection.",
        f"- Reference sources: `{artifact['reference_source_distribution']}`.",
        "",
    ]
    markdown_path = Path(run_root) / "reports" / "FINAL_REPORT.md"
    if markdown_path.exists() or markdown_path.is_symlink():
        raise FileExistsError(f"artifact already exists: {markdown_path}")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return artifact


def write_integrity_receipt(
    *,
    run_root,
    hashes: Mapping[str, Any],
) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "artifact_type": INTEGRITY_RECEIPT_SCHEMA,
        "b0": hashes.get("b0"),
        "source_videos": hashes.get("source_videos"),
        "runtime_assets": hashes.get("runtime_assets"),
        "code_config": hashes.get("code_config"),
        "bundles": hashes.get("bundles"),
        "reports": hashes.get("reports"),
        "selected_stage_export": hashes.get("selected_stage_export"),
    }
    write_new_json(run_root, "reports/integrity_receipt.json", payload)
    return payload
