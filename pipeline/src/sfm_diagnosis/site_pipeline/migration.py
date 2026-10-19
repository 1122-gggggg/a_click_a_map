"""Read-only migration of legacy site-run manifests into v2 evidence."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from sfm_diagnosis.io import write_json


def backfill_legacy_run(legacy_run: str | Path, v2_run: str | Path) -> Path:
    source = Path(legacy_run).expanduser().resolve()
    target = Path(v2_run).expanduser().resolve()
    frame_path = source / "provenance/selection/frame_manifest.json"
    query_path = source / "provenance/selection/query_manifest.json"
    if not frame_path.is_file():
        raise FileNotFoundError(frame_path)
    frame_payload = json.loads(frame_path.read_text(encoding="utf-8"))
    frames = list(frame_payload.get("frames") or frame_payload.get("keyframes") or ())
    segments, keyframes = project_legacy_frames(frames)
    output = target / "migration"
    _write_jsonl(output / "legacy_segments.jsonl", segments)
    _write_jsonl(output / "legacy_keyframes.jsonl", keyframes)
    holdouts = []
    if query_path.is_file():
        query_payload = json.loads(query_path.read_text(encoding="utf-8"))
        holdouts = list(query_payload.get("queries") or ())
        _write_jsonl(output / "legacy_holdouts.jsonl", holdouts)
        canonical_queries = [
            {
                "query_id": str(row.get("query_id") or row.get("output_name")),
                "session_id": str(row.get("session_id") or row.get("session")),
                "timestamp": float(row.get("timestamp") or row.get("source_pts_seconds") or 0.0),
                "image_path": str(
                    row.get("image_path")
                    or source / "input/heldout_queries" / str(row.get("output_name"))
                ),
                "pose_provenance": "MAPPING_DISJOINT_HISTORICALLY_OBSERVED",
            }
            for row in holdouts
        ]
        _write_jsonl(output / "holdout_queries_v2.jsonl", canonical_queries)
    evidence_candidates = (
        "diagnostics/map_qa/report.json",
        "diagnostics/map_localization_qa/report.json",
        "localization/heldout_edm/edm_loo_results.json",
        "FINAL_WORKFLOW_SUMMARY.json",
    )
    evidence = [
        {
            "artifact": relative,
            "path": str(source / relative),
            "sha256": _sha256(source / relative),
        }
        for relative in evidence_candidates
        if (source / relative).is_file()
    ]
    write_json(
        output / "legacy_evidence.json",
        {
            "schema_version": 2,
            "artifact_type": "LEGACY_FROZEN_EVIDENCE_REFERENCES",
            "artifacts": evidence,
        },
    )
    receipt = output / "BACKFILL_RECEIPT.json"
    write_json(
        receipt,
        {
            "schema_version": 2,
            "artifact_type": "LEGACY_SITE_RUN_BACKFILL",
            "legacy_run": str(source),
            "read_only": True,
            "source_artifacts": {
                str(frame_path): _sha256(frame_path),
                **({str(query_path): _sha256(query_path)} if query_path.is_file() else {}),
            },
            "segments": len(segments),
            "keyframes": len(keyframes),
            "holdout_queries": len(holdouts),
            "holdout_provenance": "MAPPING_DISJOINT_HISTORICALLY_OBSERVED",
            "frozen_evidence_artifacts": len(evidence),
            "role_note": "legacy roles are pre-SfM evidence only, never final CORE/BRIDGE authority",
        },
    )
    return receipt


def project_legacy_frames(
    frames: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frames:
        video = str(row.get("video") or row.get("video_id") or row.get("session") or "unknown")
        by_video[video].append(row)
    segment_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    for video, rows in sorted(by_video.items()):
        rows.sort(
            key=lambda row: (
                float(row.get("source_pts_seconds") or 0.0),
                int(row.get("source_frame_index") or 0),
            )
        )
        segment_index = 0
        previous_class = None
        current_segment = None
        for row in rows:
            motion_class = str(row.get("motion_class") or "unproven")
            declared = row.get("segment_id")
            if declared is not None:
                segment_id = f"legacy:{video}:{declared}"
            elif current_segment is None or motion_class != previous_class:
                segment_index += 1
                segment_id = f"legacy:{video}:S{segment_index:03d}"
            else:
                segment_id = current_segment
            if segment_id != current_segment:
                segment_rows.append(
                    {
                        "segment_id": segment_id,
                        "video_id": video,
                        "session_id": str(row.get("session") or video),
                        "motion_class": motion_class,
                        "data_status": "CANDIDATE",
                        "pre_sfm_role": _legacy_pre_role(str(row.get("role") or "")),
                        "provenance": "LEGACY_BACKFILL",
                    }
                )
            frame_index = int(row.get("source_frame_index") or 0)
            keyframe_rows.append(
                {
                    **dict(row),
                    "keyframe_id": str(row.get("frame_id") or f"{video}:{frame_index:08d}"),
                    "video_id": video,
                    "segment_id": segment_id,
                    "data_status": "CANDIDATE",
                    "pre_sfm_role": _legacy_pre_role(str(row.get("role") or "")),
                    "mapping_mode": "POSE_ONLY"
                    if row.get("role") == "bridge_only"
                    else "TRIANGULATE",
                    "provenance": "LEGACY_BACKFILL",
                }
            )
            current_segment = segment_id
            previous_class = motion_class
    return segment_rows, keyframe_rows


def _legacy_pre_role(role: str) -> str:
    if role == "bridge_only":
        return "BRIDGE_CANDIDATE"
    if role == "triangulation":
        return "CORE_CANDIDATE"
    return "UNDECIDED"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["backfill_legacy_run", "project_legacy_frames"]
