from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .edm_loo import EDMQueryResult


def filter_edm_results(
    results: Sequence[EDMQueryResult],
    *,
    include_sessions: Iterable[str] | None = None,
    exclude_sessions: Iterable[str] | None = None,
) -> list[EDMQueryResult]:
    """Split leftover coverage queries from mapping-session calibration labels."""

    include = None if include_sessions is None else {str(item) for item in include_sessions}
    exclude = {str(item) for item in (exclude_sessions or ())}
    filtered: list[EDMQueryResult] = []
    for result in results:
        if include is not None and result.session_id not in include:
            continue
        if result.session_id in exclude:
            continue
        filtered.append(result)
    return filtered


def load_edm_results(
    path: str | Path,
    *,
    evidence_mode: str,
    artifact_format: str = "auto",
) -> list[EDMQueryResult]:
    """Load the reusable canonical schema or the legacy river JSONL adapter."""

    if artifact_format not in {"auto", "canonical", "river"}:
        raise ValueError("artifact_format must be auto, canonical, or river")
    root = Path(path)
    if artifact_format == "river":
        return load_river_edm_results(root, evidence_mode=evidence_mode)
    rows = _artifact_rows(root)
    if not rows:
        raise ValueError(f"EDM artifact contains no result rows: {root}")
    detected = artifact_format
    if detected == "auto":
        detected = "canonical" if {"query_id", "session_id", "success"} <= set(rows[0]) else "river"
    if detected == "river":
        return sorted(
            [_river_row(row, evidence_mode=evidence_mode) for row in rows],
            key=lambda row: (row.session_id, row.timestamp, row.query_id),
        )
    results = []
    for row in rows:
        result = EDMQueryResult.from_dict(row)
        if result.loo_mode is None:
            result = replace(result, loo_mode=evidence_mode)
        results.append(result)
    return sorted(results, key=lambda row: (row.session_id, row.timestamp, row.query_id))


def load_river_edm_results(
    path: str | Path,
    *,
    evidence_mode: str,
) -> list[EDMQueryResult]:
    """Load river official-EDM JSONL without overstating its GT/LOO provenance."""

    root = Path(path)
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(root)
    results: list[EDMQueryResult] = []
    for file in files:
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{file}:{line_number} must be a JSON object")
            results.append(_river_row(payload, evidence_mode=evidence_mode))
    return sorted(results, key=lambda row: (row.session_id, row.timestamp, row.query_id))


def _artifact_rows(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.rglob("*.jsonl")) + sorted(path.rglob("*.ndjson"))
        if not files:
            canonical = path / "edm_loo_results.json"
            files = [canonical] if canonical.is_file() else []
    else:
        files = [path]
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for file in files:
        if file.suffix.lower() in {".jsonl", ".ndjson"}:
            payloads = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            payload = json.loads(file.read_text(encoding="utf-8"))
            payloads = payload.get("results", []) if isinstance(payload, dict) else payload
        for payload in payloads:
            if not isinstance(payload, dict):
                raise ValueError(f"{file} result rows must be JSON objects")
            rows.append(payload)
    return rows


def _river_row(row: dict[str, Any], *, evidence_mode: str) -> EDMQueryResult:
    query_id = str(row.get("query_name") or row.get("query") or "")
    session_id = str(row.get("video") or query_id.split("/", 1)[0])
    if not query_id or not session_id:
        raise ValueError("river EDM row requires query_name and session identity")
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
    success = (
        str(row.get("status")) == "DIRECT_STRONG"
        and str(decision.get("status")) == "ACCEPT"
    )
    observations = row.get("inlier_observations")
    observations = observations if isinstance(observations, list) else []
    point_ids = tuple(
        dict.fromkeys(
            int(observation["point3d_id"])
            for observation in observations
            if isinstance(observation, dict) and observation.get("point3d_id") is not None
        )
    )
    references = row.get("retrieved_names")
    references = tuple(str(value) for value in references) if isinstance(references, list) else ()
    pose = _pose_fields(row.get("pose"))
    registration_success = pose is not None and str(metrics.get("status") or "ok") == "ok"
    return EDMQueryResult(
        query_id=query_id,
        session_id=session_id,
        timestamp=float(row.get("source_second") or 0.0),
        success=success,
        registration_success=registration_success,
        raw_matches=_integer(row.get("raw_matches")),
        valid_2d3d=_integer(
            row.get("lifted_correspondences") or row.get("unique_point3d_count")
        ),
        ransac_inliers=len(observations),
        inlier_ratio=_number(metrics.get("inlier_ratio")),
        reprojection_mean=_number(metrics.get("reprojection_mean")),
        reprojection_median=_number(
            metrics.get("reprojection_median") or metrics.get("reprojection_p50")
        ),
        reprojection_p90=_number(metrics.get("reprojection_p90")),
        positive_depth_ratio=_number(metrics.get("positive_depth_ratio")),
        pose_translation_error=None,
        pose_rotation_error_deg=None,
        reference_ids=references,
        point_ids=point_ids,
        estimated_position=None if pose is None else pose[0],
        estimated_R_wc=None if pose is None else pose[1],
        estimated_yaw_deg=None if pose is None else pose[2],
        estimated_pitch_deg=None if pose is None else pose[3],
        ground_truth_source=(
            None if pose is None else "SELF_FIT_NO_ABSOLUTE_GT"
        ),
        loo_mode=evidence_mode,
    )


def _pose_fields(value: Any):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise ValueError("EDM pose must be a finite 4x4 camera-from-world matrix")
    rotation_cw = matrix[:3, :3]
    translation = matrix[:3, 3]
    rotation_wc = rotation_cw.T
    center = -rotation_wc @ translation
    forward = rotation_wc[:, 2]
    yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0]))) % 360.0
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    return (
        tuple(float(value) for value in center),
        tuple(tuple(float(value) for value in row) for row in rotation_wc),
        yaw,
        pitch,
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)
