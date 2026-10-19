"""Map-set audits that do not mutate a frozen reconstruction."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .edm_risk.edm_artifacts import filter_edm_results
from .edm_risk.edm_loo import EDMQueryResult
from .session_holdout import session_from_image_name

_FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def frame_index_from_name(name: str) -> int | None:
    match = _FRAME_RE.search(str(name))
    return int(match.group(1)) if match else None


def names_in_frame_range(
    names: Sequence[str],
    *,
    session: str,
    lo: int,
    hi: int,
) -> list[str]:
    selected: list[str] = []
    for name in names:
        if session_from_image_name(name) != session:
            continue
        index = frame_index_from_name(name)
        if index is None:
            continue
        if int(lo) <= index <= int(hi):
            selected.append(name)
    return selected


def classify_registered_cameras(
    *,
    registered: Sequence[str],
    bridge_only: Sequence[str],
    selected: Sequence[str],
) -> dict[str, Any]:
    registered_set = {str(name) for name in registered}
    bridge_set = {str(name) for name in bridge_only}
    selected_set = {str(name) for name in selected}
    unknown_bridge = sorted(bridge_set - registered_set)
    triangulating = registered_set - bridge_set
    unregistered = selected_set - registered_set
    by_session: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "selected": 0,
            "registered": 0,
            "bridge_only": 0,
            "selected_bridge_only": 0,
            "triangulating": 0,
            "unregistered": 0,
        }
    )
    for name in selected_set:
        bucket = by_session[session_from_image_name(name)]
        bucket["selected"] += 1
        if name in bridge_set:
            bucket["selected_bridge_only"] += 1
    for name in registered_set:
        bucket = by_session[session_from_image_name(name)]
        bucket["registered"] += 1
        if name in bridge_set:
            bucket["bridge_only"] += 1
        else:
            bucket["triangulating"] += 1
    for name in unregistered:
        by_session[session_from_image_name(name)]["unregistered"] += 1
    return {
        "schema_version": 1,
        "artifact_type": "REGISTERED_CAMERA_ROLES",
        "selected": len(selected_set),
        "registered": len(registered_set),
        "bridge_only": len(registered_set & bridge_set),
        "selected_bridge_only": len(selected_set & bridge_set),
        "triangulating": len(triangulating),
        "selected_unregistered": len(unregistered),
        "unknown_bridge_names": unknown_bridge,
        "by_session": dict(by_session),
        "triangulating_names": sorted(triangulating),
        "bridge_only_names": sorted(registered_set & bridge_set),
        "unregistered_names": sorted(unregistered),
    }


def parse_gluemap_conversion_log(text: str) -> dict[str, Any]:
    patterns = {
        "input_images": r"Number of images:\s+(\d+)",
        "virtual_points": r"valid virtual points after selection:\s+(\d+)",
        "established_tracks": r"Established\s+(\d+)\s+tracks from",
        "established_observations": r"Established\s+\d+\s+tracks from\s+(\d+)\s+observations",
        "final_real_tracks": r"Final:\s+(\d+)\s+real tracks",
        "final_real_observations": r"Final:\s+\d+\s+real tracks\s+\((\d+)\s+obs\)",
    }
    values: dict[str, Any] = {"schema_version": 1, "artifact_type": "GLUEMAP_CONVERSION_RATES"}
    for key, pattern in patterns.items():
        matches = list(re.finditer(pattern, text))
        values[key] = int(matches[-1].group(1)) if matches else None
    images = values["input_images"]
    tracks = values["established_tracks"]
    final_tracks = values["final_real_tracks"]
    values["established_tracks_per_image"] = (
        None if images in {None, 0} or tracks is None else tracks / images
    )
    values["final_tracks_per_established"] = (
        None
        if tracks in {None, 0} or final_tracks is None
        else final_tracks / tracks
    )
    return values


def consecutive_rotation_segments(
    names: Sequence[str],
    centers: np.ndarray,
    rotations: np.ndarray,
    *,
    min_rotation_deg: float = 10.0,
    max_translation: float = 0.05,
) -> list[dict[str, Any]]:
    if len(names) != len(centers) or len(names) != len(rotations):
        raise ValueError("rotation audit arrays must be the same length")
    if len(names) < 2:
        return []
    flagged: list[dict[str, Any]] = []
    for index in range(len(names) - 1):
        relative = np.asarray(rotations[index + 1], dtype=float).T @ np.asarray(
            rotations[index], dtype=float
        )
        trace = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        rotation_deg = float(np.degrees(math.acos(trace)))
        translation = float(
            np.linalg.norm(
                np.asarray(centers[index + 1], dtype=float)
                - np.asarray(centers[index], dtype=float)
            )
        )
        if rotation_deg >= float(min_rotation_deg) and translation <= float(max_translation):
            flagged.append(
                {
                    "first": str(names[index]),
                    "second": str(names[index + 1]),
                    "rotation_deg": rotation_deg,
                    "translation": translation,
                }
            )
    return flagged


def leftover_coverage_report(
    results: Sequence[EDMQueryResult],
    *,
    mapping_sessions: Sequence[str],
    roles: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    leftover = filter_edm_results(results, exclude_sessions=mapping_sessions)
    by_session: dict[str, dict[str, Any]] = {}
    for row in leftover:
        bucket = by_session.setdefault(
            row.session_id,
            {
                "queries": 0,
                "successes": 0,
                "success_rate": 0.0,
                "role": (roles or {}).get(row.session_id),
            },
        )
        bucket["queries"] += 1
        bucket["successes"] += int(bool(row.success))
    for bucket in by_session.values():
        bucket["success_rate"] = (
            bucket["successes"] / bucket["queries"] if bucket["queries"] else None
        )
    return {
        "schema_version": 1,
        "artifact_type": "LEFTOVER_COVERAGE_ONLY",
        "usable_as_failure_probability_labels": False,
        "reason": (
            "Leftover sessions were excluded from the frozen map. Their failures "
            "measure coverage/overlap, not in-map localizability."
        ),
        "queries": len(leftover),
        "successes": sum(int(bool(row.success)) for row in leftover),
        "by_session": by_session,
    }


def exclude_from_triangulation(
    *,
    seam_names: Sequence[str],
    rotation_pairs: Sequence[Mapping[str, Any]],
) -> list[str]:
    names = {str(name) for name in seam_names}
    for pair in rotation_pairs:
        names.add(str(pair["first"]))
        names.add(str(pair["second"]))
    return sorted(names)
