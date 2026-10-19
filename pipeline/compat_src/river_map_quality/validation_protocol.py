"""P157-only validation split, E0 MegaLoc alignment, and freeze invariants.

P157 is evidence only.  It never enters candidate selection, mask calibration, map
construction, or threshold fitting.  Alignment anchors are E0 MegaLoc+EDM estimates
inside reserved windows; if those fail, pose-accuracy fields stay explicit nulls.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from river_map_quality.historical_experiment import HistoricalExperimentError
from river_map_quality.p157_sim3 import (
    evaluate_holdout_similarity,
    evaluate_leave_one_out_similarity,
)
from river_map_quality.pose_attribution import camera_center
from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    VALIDATION_VIDEO_NAMES,
    write_new_json,
)
from river_map_quality.verify_protocol import (
    VerifyProtocolError,
    build_temporal_protocol,
    run_verify_protocol,
)

VALIDATION_PROTOCOL_SCHEMA = "RIVER_P157_VALIDATION_PROTOCOL_V1"
E0_ALIGNMENT_SCHEMA = "RIVER_P157_E0_MEGALOC_ALIGNMENT_V1"
VALIDATION_FREEZE_SCHEMA = "RIVER_P157_VALIDATION_FREEZE_V1"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ValidationProtocolError(HistoricalExperimentError):
    """Raised when the held-out P157 protocol would leak into map construction."""


def _require_p157(video_name: str) -> str:
    if video_name not in VALIDATION_VIDEO_NAMES:
        raise ValidationProtocolError(f"validation corpus is P157-only: {video_name}")
    return video_name


def locked_validation_videos(source_lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    locked = source_lock.get("VALIDATION") or source_lock.get("VERIFY")
    if tuple(locked or ()) != VALIDATION_VIDEO_NAMES:
        raise ValidationProtocolError("locked validation corpus membership or ordering changed")
    return dict(locked)


def build_p157_temporal_protocol(duration_seconds: float) -> dict[str, object]:
    """Build the one deterministic 15 s P157 split before any candidate experiment."""

    return build_temporal_protocol(VALIDATION_VIDEO_NAMES[0], duration_seconds)


def materialize_p157_protocol(*, run_root: Path, raw_root: Path) -> dict[str, object]:
    """Extract P157 evidence streams into the isolated historical validation tree."""

    run_root = run_root.resolve(strict=True)
    raw_root = raw_root.resolve(strict=True)
    lock_path = run_root / "input_lock/source_videos.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    locked = locked_validation_videos(source_lock)
    duration = float(locked[VALIDATION_VIDEO_NAMES[0]]["ffprobe"]["duration_seconds"])
    protocol = build_p157_temporal_protocol(duration)
    artifact = run_verify_protocol(
        run_root=run_root,
        raw_root=raw_root,
        output_dirname="validation",
    )
    if tuple(artifact["videos"]) != VALIDATION_VIDEO_NAMES:
        raise ValidationProtocolError("VERIFY extraction escaped the P157-only corpus")
    summary = {
        "schema_version": 1,
        "artifact_type": VALIDATION_PROTOCOL_SCHEMA,
        "validation_videos": list(VALIDATION_VIDEO_NAMES),
        "candidate_selection_forbidden": True,
        "mask_calibration_forbidden": True,
        "map_construction_forbidden": True,
        "threshold_fitting_forbidden": True,
        "protocol": protocol,
        "source_lock": fingerprint_file(
            run_root / "input_lock/source_videos.json", sha256=True
        ).as_dict(),
        "partition_manifest": fingerprint_file(
            run_root / "validation/partition_manifest.json", sha256=True
        ).as_dict(),
    }
    write_new_json(run_root, "validation/protocol.json", summary)
    return summary


def _anchor_centers(
    anchors: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    source: list[np.ndarray] = []
    target: list[np.ndarray] = []
    names: list[str] = []
    for row in anchors:
        if row.get("status") != "E0_PNP_SOLVED":
            continue
        source_pose = np.asarray(row["pseudo_gt_pose"], dtype=float)
        target_pose = np.asarray(row["e0_pose"], dtype=float)
        if source_pose.shape != (4, 4) or target_pose.shape != (4, 4):
            raise ValidationProtocolError("alignment anchors must carry 4x4 poses")
        source.append(camera_center(source_pose))
        target.append(camera_center(target_pose))
        names.append(str(row["query_name"]))
    if not names:
        return (
            np.empty((0, 3), dtype=float),
            np.empty((0, 3), dtype=float),
            [],
        )
    return np.asarray(source, dtype=float), np.asarray(target, dtype=float), names


def _spatial_separation(centers: np.ndarray, *, minimum_span: float) -> dict[str, object]:
    if len(centers) < 3:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "reason": "MISSING_ALIGNMENT_CLUSTER",
            "span": None,
        }
    separations = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    span = float(np.max(separations))
    if not np.isfinite(span) or span < minimum_span:
        return {
            "status": INSUFFICIENT_EVIDENCE,
            "reason": "ANCHORS_NOT_SPATIALLY_SEPARATED",
            "span": span,
        }
    return {"status": "PASS", "reason": "THREE_SPATIALLY_SEPARATED_E0_ANCHORS", "span": span}


def align_p157_with_e0_anchors(
    *,
    run_root: Path,
    anchors: Sequence[Mapping[str, Any]],
    minimum_span: float = 5.0,
    max_relative_residual: float = 0.05,
) -> dict[str, object]:
    """Fit B0-relative Sim3 from reserved-window E0 MegaLoc+EDM anchors only."""

    run_root = run_root.resolve(strict=True)
    if any(str(row.get("source_role", "e0")) != "e0" for row in anchors):
        raise ValidationProtocolError("P157 alignment may use only E0 MegaLoc+EDM anchors")
    source, target, names = _anchor_centers(anchors)
    separation = _spatial_separation(target, minimum_span=minimum_span)
    pose_accuracy_available = False
    sim3: dict[str, object] | None = None
    loo: dict[str, object] | None = None
    status = INSUFFICIENT_EVIDENCE
    if separation["status"] == "PASS" and len(names) >= 4:
        try:
            loo = evaluate_leave_one_out_similarity(
                source,
                target,
                max_relative_residual=max_relative_residual,
            )
            holdout = evaluate_holdout_similarity(
                source[:-1],
                target[:-1],
                source[-1:],
                target[-1:],
                max_relative_residual=max_relative_residual,
            )
        except ValueError as error:
            loo = {"consistent": False, "error": str(error)}
            holdout = {"consistent": False, "error": str(error)}
        if bool(loo.get("consistent")) and bool(holdout.get("consistent")):
            transform = holdout["transform"]
            sim3 = {
                "scale": float(transform.scale),
                "rotation": np.asarray(transform.rotation, dtype=float).tolist(),
                "translation": np.asarray(transform.translation, dtype=float).tolist(),
                "holdout_relative_residuals": holdout["holdout_relative_residuals"],
            }
            pose_accuracy_available = True
            status = "ALIGNED"
    artifact = {
        "schema_version": 1,
        "artifact_type": E0_ALIGNMENT_SCHEMA,
        "validation_videos": list(VALIDATION_VIDEO_NAMES),
        "status": status,
        "anchors": [dict(row) for row in anchors],
        "anchor_names": names,
        "spatial_separation": separation,
        "leave_one_anchor_cluster_out": loo,
        "sim3": sim3,
        "pose_accuracy_available": pose_accuracy_available,
        "recovery_gate": "AVAILABLE" if pose_accuracy_available else INSUFFICIENT_EVIDENCE,
        "e0_only_anchors": True,
        "historical_anchors_forbidden": True,
    }
    write_new_json(run_root, "validation/alignments/P1570157.json", artifact)
    return artifact


def freeze_validation_protocol(
    *,
    run_root: Path,
    code_sha256: str,
    config_sha256: str,
    runtime_sha256: str,
    bundle_sha256: Mapping[str, str],
) -> dict[str, object]:
    """Seal the P157 split/alignment after development choices freeze."""

    run_root = run_root.resolve(strict=True)
    protocol_path = run_root / "validation/protocol.json"
    alignment_path = run_root / "validation/alignments/P1570157.json"
    if not protocol_path.is_file():
        raise ValidationProtocolError("cannot freeze before the P157 protocol exists")
    if not alignment_path.is_file():
        raise ValidationProtocolError("cannot freeze before the E0 alignment receipt exists")
    payload = {
        "schema_version": 1,
        "artifact_type": VALIDATION_FREEZE_SCHEMA,
        "code_sha256": code_sha256,
        "config_sha256": config_sha256,
        "runtime_sha256": runtime_sha256,
        "bundle_sha256": dict(bundle_sha256),
        "protocol": fingerprint_file(protocol_path, sha256=True).as_dict(),
        "alignment": fingerprint_file(alignment_path, sha256=True).as_dict(),
        "final_replay_permitted": True,
        "development_retuning_permitted": False,
    }
    write_new_json(run_root, "validation/freeze.json", payload)
    return payload


def assert_validation_split_unchanged(run_root: Path, freeze: Mapping[str, Any]) -> None:
    """Fail closed if the frozen P157 split or alignment receipt changed."""

    run_root = run_root.resolve(strict=True)
    protocol = fingerprint_file(run_root / "validation/protocol.json", sha256=True).as_dict()
    alignment = fingerprint_file(
        run_root / "validation/alignments/P1570157.json", sha256=True
    ).as_dict()
    if protocol != freeze.get("protocol") or alignment != freeze.get("alignment"):
        raise ValidationProtocolError("frozen P157 protocol or alignment changed")


__all__ = [
    "E0_ALIGNMENT_SCHEMA",
    "INSUFFICIENT_EVIDENCE",
    "VALIDATION_FREEZE_SCHEMA",
    "VALIDATION_PROTOCOL_SCHEMA",
    "ValidationProtocolError",
    "VerifyProtocolError",
    "align_p157_with_e0_anchors",
    "assert_validation_split_unchanged",
    "build_p157_temporal_protocol",
    "freeze_validation_protocol",
    "locked_validation_videos",
    "materialize_p157_protocol",
]
