"""Fail-closed planning of existing video frames for weak-zone recovery.

This module only proposes frames and pairs.  It does not claim geometric
verification and deliberately rejects frames whose pose cannot be bracketed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _center(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row.get("center", row.get("center_w")), dtype=float).reshape(3)


def _rotation(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row.get("rotation", row.get("R_wc", np.eye(3))), dtype=float).reshape(3, 3)


def _forward(row: Mapping[str, Any]) -> np.ndarray:
    """Return world forward under explicit R_wc (camera-to-world) convention."""
    value = _rotation(row) @ np.array([0.0, 0.0, 1.0])
    return value / max(float(np.linalg.norm(value)), 1e-12)


def _interpolate(
    a: Mapping[str, Any], b: Mapping[str, Any], frame_index: float
) -> tuple[np.ndarray, np.ndarray]:
    t = (frame_index - float(a["frame_index"])) / (
        float(b["frame_index"]) - float(a["frame_index"])
    )
    center = (1.0 - t) * _center(a) + t * _center(b)
    rotations = Rotation.from_matrix(np.stack([_rotation(a), _rotation(b)]))
    rot = Slerp([0.0, 1.0], rotations)([t]).as_matrix()[0]
    return center, rot


def _as_zones(
    weak_zones: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[tuple[np.ndarray, float]]:
    rows = weak_zones.get("zones", []) if isinstance(weak_zones, Mapping) else weak_zones
    result = []
    for row in rows:
        result.append(
            (
                np.asarray(row.get("center", row.get("xyz")), dtype=float).reshape(3),
                float(row.get("radius", 1.0)),
            )
        )
    return result


def plan_weak_zone_recovery(
    frames: Sequence[Mapping[str, Any]],
    canonical_cameras: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    weak_zones: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    max_frame_gap: int = 12,
    max_candidates: int = 12,
    min_candidates: int = 8,
    pair_radius: float = 8.0,
    min_frame_spacing: int = 0,
    references_per_candidate: int = 3,
    fail_closed: bool = True,
    minimum_forward_novelty_deg: float = 0.0,
    minimum_lateral_distance: float = 0.0,
) -> dict[str, Any]:
    """Return deterministic candidate/rejection/pair records for pre-match use."""
    if (
        max_frame_gap < 1
        or max_candidates < 1
        or min_candidates < 1
        or references_per_candidate < 1
    ):
        raise ValueError("gap and candidate limits must be positive")
    cameras = (
        list(canonical_cameras.values())
        if isinstance(canonical_cameras, Mapping)
        else list(canonical_cameras)
    )
    zones = _as_zones(weak_zones)
    selected_names = {
        str(r.get("name")) for r in frames if r.get("selected") and r.get("registered", True)
    }
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frames:
        if str(row.get("name")) in selected_names:
            by_video[str(row.get("video", row.get("session", "unknown")))].append(row)
    for values in by_video.values():
        values.sort(key=lambda r: (int(r["frame_index"]), str(r.get("name"))))

    rejected = []
    candidates = []
    for row in sorted(
        frames,
        key=lambda r: (
            str(r.get("video", r.get("session", "unknown"))),
            int(r.get("frame_index", -1)),
            str(r.get("name")),
        ),
    ):
        name = str(row.get("name"))
        if name in selected_names:
            continue
        if fail_closed and row.get("segment_geometry_valid") is not True:
            rejected.append(
                {
                    "name": name,
                    "video": str(row.get("video", row.get("session", "unknown"))),
                    "reason": "SEGMENT_GEOMETRY_INVALID_OR_UNKNOWN",
                }
            )
            continue
        video = str(row.get("video", row.get("session", "unknown")))
        idx = int(row.get("frame_index", -1))
        brackets = (
            [r for r in by_video.get(video, []) if int(r["frame_index"]) < idx],
            [r for r in by_video.get(video, []) if int(r["frame_index"]) > idx],
        )
        if not brackets[0] or not brackets[1]:
            rejected.append(
                {"name": name, "video": video, "reason": "UNBRACKETED_OR_GAP_TOO_LARGE"}
            )
            continue
        before, after = brackets[0][-1], brackets[1][0]
        left_gap, right_gap = idx - int(before["frame_index"]), int(after["frame_index"]) - idx
        if left_gap > max_frame_gap or right_gap > max_frame_gap:
            rejected.append(
                {
                    "name": name,
                    "video": video,
                    "reason": "UNBRACKETED_OR_GAP_TOO_LARGE",
                    "left_gap": left_gap,
                    "right_gap": right_gap,
                }
            )
            continue
        center, rotation = _interpolate(before, after, idx)
        zone_distance = min(
            (max(0.0, float(np.linalg.norm(center - z)) - radius) for z, radius in zones),
            default=0.0,
        )
        nearby = sorted(
            ((float(np.linalg.norm(center - _center(c))), c) for c in cameras),
            key=lambda x: (x[0], str(x[1].get("name", ""))),
        )
        novelty = min((d for d, _ in nearby), default=0.0)
        angles = []
        lateral = []
        for _, camera in nearby:
            dot = float(
                np.clip(np.dot(rotation @ np.array([0.0, 0.0, 1.0]), _forward(camera)), -1.0, 1.0)
            )
            angles.append(float(np.degrees(np.arccos(dot))))
            lateral.append(float(np.linalg.norm(center - _center(camera))))
        angle_novelty = min(angles, default=180.0)
        lateral_distance = min(lateral, default=0.0)
        if angle_novelty < minimum_forward_novelty_deg:
            rejected.append(
                {
                    "name": name,
                    "video": video,
                    "reason": "INSUFFICIENT_FORWARD_NOVELTY",
                    "value": angle_novelty,
                }
            )
            continue
        if lateral_distance < minimum_lateral_distance:
            rejected.append(
                {
                    "name": name,
                    "video": video,
                    "reason": "INSUFFICIENT_LATERAL_DISTANCE",
                    "value": lateral_distance,
                }
            )
            continue
        candidates.append(
            {
                "name": name,
                "video": video,
                "frame_index": idx,
                "center": center.tolist(),
                "rotation": rotation.tolist(),
                "interpolation": {
                    "before": str(before.get("name")),
                    "after": str(after.get("name")),
                    "gap": int(after["frame_index"]) - int(before["frame_index"]),
                    "left_gap": left_gap,
                    "right_gap": right_gap,
                    "uncertainty": float(left_gap + right_gap),
                },
                "score_components": {
                    "weak_zone_proximity": 1.0 / (1.0 + zone_distance),
                    "view_novelty": novelty,
                    "forward_angle_novelty_deg": angle_novelty,
                    "lateral_center_distance": lateral_distance,
                },
                "_sort": (zone_distance, -angle_novelty, -novelty, video, idx, name),
            }
        )
    candidates.sort(key=lambda r: r["_sort"])
    # Round-robin sessions first, then score order: deterministic and balanced.
    chosen = []
    remaining = list(candidates)
    videos = sorted({r["video"] for r in remaining})
    while remaining and len(chosen) < max_candidates:
        progressed = False
        for video in videos:
            hit = next((r for r in remaining if r["video"] == video), None)
            if hit is not None and all(
                abs(hit["frame_index"] - r["frame_index"]) >= min_frame_spacing
                for r in chosen
                if r["video"] == video
            ):
                chosen.append(hit)
                remaining.remove(hit)
                progressed = True
                if len(chosen) >= max_candidates:
                    break
        if not progressed:
            break
    while remaining and len(chosen) < max_candidates:
        hit = next(
            (
                r
                for r in remaining
                if all(
                    abs(r["frame_index"] - old["frame_index"]) >= min_frame_spacing
                    for old in chosen
                    if old["video"] == r["video"]
                )
            ),
            None,
        )
        if hit is None:
            break
        chosen.append(hit)
        remaining.remove(hit)
    chosen.sort(key=lambda r: (r["video"], r["frame_index"], r["name"]))
    for row in chosen:
        row.pop("_sort", None)
    chosen_names = {r["name"] for r in chosen}
    rejected.extend(
        {"name": r["name"], "video": r["video"], "reason": "LOWER_PRIORITY_OR_LIMIT"}
        for r in candidates
        if r["name"] not in chosen_names
    )
    pairs = []
    for candidate in chosen:
        ranked = sorted(
            (
                (float(np.linalg.norm(_center(candidate) - _center(c))), c)
                for c in cameras
                if str(c.get("name")) != candidate["name"]
            ),
            key=lambda x: (x[0], str(x[1].get("name", ""))),
        )
        preferred = [
            item
            for item in ranked
            if str(item[1].get("video", item[1].get("session", ""))) != candidate["video"]
        ]
        ordered = preferred[:references_per_candidate] + [
            item for item in ranked if item not in preferred[:references_per_candidate]
        ]
        for distance, reference in ordered[:references_per_candidate]:
            pairs.append(
                {
                    "candidate": candidate["name"],
                    "canonical": str(reference.get("name")),
                    "canonical_video": str(
                        reference.get("video", reference.get("session", "unknown"))
                    ),
                    "distance": distance,
                    "kind": "candidate_to_canonical",
                }
            )
    cross_session = [r for r in chosen]
    for a, b in zip(cross_session, cross_session[1:]):
        if a["video"] == b["video"]:
            continue
        distance = float(np.linalg.norm(_center(a) - _center(b)))
        if distance <= pair_radius:
            pairs.append(
                {
                    "candidate": a["name"],
                    "canonical": b["name"],
                    "candidate_video": a["video"],
                    "canonical_video": b["video"],
                    "distance": distance,
                    "kind": "candidate_to_candidate",
                }
            )
    sessions = sorted({r["video"] for r in chosen})
    ready = len(chosen) >= min_candidates and len(sessions) >= 3
    return {
        "status": "PRE_MATCH_ESTIMATE",
        "fail_closed": fail_closed,
        "candidates": chosen,
        "rejected": sorted(rejected, key=lambda r: (r["reason"], r["name"])),
        "pairs": pairs,
        "summary": {
            "candidate_count": len(chosen),
            "available_count": len(candidates),
            "rejected_count": len(rejected),
            "sessions": sessions,
            "minimum_target": min_candidates,
            "minimum_target_met": len(chosen) >= min_candidates,
            "readiness": "READY_FOR_GEOMETRY" if ready else "INSUFFICIENT_PREMATCH_CANDIDATES",
        },
    }
