from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np


def _track_length(point: Any) -> int:
    return len(point.track.elements)


def focus_session_track_summary(
    *,
    names_by_id: Mapping[int, str],
    track_image_ids: Iterable[Iterable[int]],
    focus_video: str,
) -> dict[str, Any]:
    """Summarize how one video's tracks connect to the other mapping videos.

    A track is counted once in ``cross_session_tracks`` even when it touches
    several other videos. The per-video counters intentionally record every
    other video touched by that track, so their sum may exceed the cross-track
    total.
    """

    per_other_video: Counter[str] = Counter()
    focus_tracks = 0
    cross_session_tracks = 0
    for raw_image_ids in track_image_ids:
        videos = {
            str(names_by_id[int(image_id)]).split("/", 1)[0]
            for image_id in raw_image_ids
            if int(image_id) in names_by_id
        }
        if focus_video not in videos:
            continue
        focus_tracks += 1
        other_videos = videos - {focus_video}
        if not other_videos:
            continue
        cross_session_tracks += 1
        per_other_video.update(other_videos)

    ordered = sorted(per_other_video.items(), key=lambda row: (-row[1], row[0]))
    dominant_video, dominant_tracks = ordered[0] if ordered else (None, 0)
    return {
        "focus_video": focus_video,
        "focus_tracks": focus_tracks,
        "cross_session_tracks": cross_session_tracks,
        "connected_videos": sorted(per_other_video),
        "per_other_video_tracks": dict(sorted(per_other_video.items())),
        "dominant_other_video": dominant_video,
        "dominant_other_video_tracks": dominant_tracks,
        "dominant_share": (
            dominant_tracks / cross_session_tracks if cross_session_tracks else None
        ),
    }


def component_summary(
    nodes: set[str], shared_landmarks: Mapping[tuple[str, str], int], *, threshold: int
) -> dict[str, Any]:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(
        tuple(sorted(pair)) for pair, count in shared_landmarks.items() if count >= threshold
    )
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    return {
        "threshold": threshold,
        "component_sizes": [len(component) for component in components],
        "largest_component_ratio": len(components[0]) / len(nodes) if nodes else 0.0,
        "articulation_images": sorted(nx.articulation_points(graph)),
        "bridge_edges": sorted([list(sorted(edge)) for edge in nx.bridges(graph)]),
        "isolated_images": sorted(nx.isolates(graph)),
        "components": [
            sorted(str(value) for value in component)  # ty: ignore[not-iterable]
            for component in components
        ],
    }


def analyze_model(
    model: Path,
    *,
    focus_video: str | None = None,
    weak_observation_threshold: int = 15,
) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model.resolve(strict=True)))
    names_by_id = {
        int(image.image_id): str(image.name)
        for image in reconstruction.images.values()
        if image.has_pose
    }
    observations: Counter[int] = Counter()
    shared: Counter[tuple[str, str]] = Counter()
    track_lengths = []
    track_image_ids: list[list[int]] = []
    errors = []
    for point in reconstruction.points3D.values():
        image_ids = sorted(
            {
                int(element.image_id)
                for element in point.track.elements
                if int(element.image_id) in names_by_id
            }
        )
        track_image_ids.append(image_ids)
        track_lengths.append(_track_length(point))
        errors.append(float(point.error))
        for image_id in image_ids:
            observations[image_id] += 1
        for offset, left in enumerate(image_ids):
            for right in image_ids[offset + 1 :]:
                first, second = sorted((names_by_id[left], names_by_id[right]))
                shared[(first, second)] += 1
    tracks = np.asarray(track_lengths, dtype=int)
    reprojection = np.asarray(errors, dtype=float)
    per_video = {}
    for video in sorted({name.split("/", 1)[0] for name in names_by_id.values()}):
        image_ids = [
            image_id for image_id, name in names_by_id.items() if name.split("/", 1)[0] == video
        ]
        values = np.asarray([observations[image_id] for image_id in image_ids], dtype=int)
        per_video[video] = {
            "images": len(image_ids),
            "observation_min": int(values.min()),
            "observation_p10": float(np.percentile(values, 10)),
            "observation_p50": float(np.percentile(values, 50)),
            "zero_observation_images": int(np.count_nonzero(values == 0)),
        }
    graph = {
        str(value): component_summary(set(names_by_id.values()), shared, threshold=value)
        for value in (3, 7, 15)
    }
    primary = graph["15"]
    low_observation_images = sorted(
        name
        for image_id, name in names_by_id.items()
        if observations[image_id] < weak_observation_threshold
    )
    result = {
        "model": str(model.resolve()),
        "registered_images": reconstruction.num_reg_images(),
        "points3D": reconstruction.num_points3D(),
        "observations": int(reconstruction.compute_num_observations()),
        "track_length_min": int(tracks.min()) if len(tracks) else 0,
        "track_length_p50": float(np.percentile(tracks, 50)) if len(tracks) else None,
        "track_length_p90": float(np.percentile(tracks, 90)) if len(tracks) else None,
        "multi_view_track_ratio_ge5": (
            float(np.count_nonzero(tracks >= 5) / len(tracks)) if len(tracks) else 0.0
        ),
        "reprojection_p50_px": float(np.percentile(reprojection, 50))
        if len(reprojection)
        else None,
        "reprojection_p90_px": float(np.percentile(reprojection, 90))
        if len(reprojection)
        else None,
        "reprojection_p99_px": float(np.percentile(reprojection, 99))
        if len(reprojection)
        else None,
        "largest_component_ratio": primary["largest_component_ratio"],
        "component_sizes": primary["component_sizes"],
        "articulation_images": primary["articulation_images"],
        "bridge_edges": primary["bridge_edges"],
        "articulation_count": len(primary["articulation_images"]),
        "bridge_count": len(primary["bridge_edges"]),
        "low_observation_threshold": weak_observation_threshold,
        "low_observation_images": low_observation_images,
        "per_video": per_video,
        "graphs": graph,
    }
    if focus_video is not None:
        result["focus_session"] = focus_session_track_summary(
            names_by_id=names_by_id,
            track_image_ids=track_image_ids,
            focus_video=focus_video,
        )
    return result


def _passes(row: Mapping[str, Any]) -> bool:
    return bool(
        float(row.get("largest_component_ratio") or 0.0) >= 0.99
        and int(row.get("articulation_count") or 0) == 0
        and int(row.get("bridge_count") or 0) == 0
        and float(row.get("reprojection_p90_px") or np.inf) <= 2.0
        and float(row.get("reprojection_p99_px") or np.inf) <= 3.0
        and int(row.get("track_length_min") or 3) >= 3
    )


def evaluate_objective_improvement(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed unless a candidate satisfies the River V4 improvement goals."""

    baseline_focus = dict(baseline.get("focus_session") or {})
    candidate_focus = dict(candidate.get("focus_session") or {})
    baseline_connected = set(baseline_focus.get("connected_videos") or ())
    candidate_connected = set(candidate_focus.get("connected_videos") or ())
    baseline_dominant = baseline_focus.get("dominant_share")
    candidate_dominant = candidate_focus.get("dominant_share")
    checks = {
        "geometry_gate": _passes(candidate),
        "weak_frames_cleaned": not candidate.get("low_observation_images"),
        "largest_component_non_regression": float(
            candidate.get("largest_component_ratio") or 0.0
        )
        >= float(baseline.get("largest_component_ratio") or 0.0),
        "multi_view_track_non_regression": float(
            candidate.get("multi_view_track_ratio_ge5") or 0.0
        )
        >= float(baseline.get("multi_view_track_ratio_ge5") or 0.0),
        "focus_cross_session_tracks_non_regression": int(
            candidate_focus.get("cross_session_tracks") or 0
        )
        >= int(baseline_focus.get("cross_session_tracks") or 0),
        "focus_connected_videos_non_regression": len(candidate_connected)
        >= len(baseline_connected),
        "focus_dominance_non_regression": (
            candidate_dominant is not None
            and baseline_dominant is not None
            and float(candidate_dominant) <= float(baseline_dominant)
        ),
        "reprojection_p90_within_five_percent": float(
            candidate.get("reprojection_p90_px") or np.inf
        )
        <= 1.05 * float(baseline.get("reprojection_p90_px") or np.inf),
        "reprojection_p99_within_five_percent": float(
            candidate.get("reprojection_p99_px") or np.inf
        )
        <= 1.05 * float(baseline.get("reprojection_p99_px") or np.inf),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "deltas": {
            "low_observation_images": len(candidate.get("low_observation_images") or ())
            - len(baseline.get("low_observation_images") or ()),
            "multi_view_track_ratio_ge5": float(
                candidate.get("multi_view_track_ratio_ge5") or 0.0
            )
            - float(baseline.get("multi_view_track_ratio_ge5") or 0.0),
            "focus_cross_session_tracks": int(
                candidate_focus.get("cross_session_tracks") or 0
            )
            - int(baseline_focus.get("cross_session_tracks") or 0),
            "focus_connected_videos": len(candidate_connected) - len(baseline_connected),
            "focus_dominant_share": (
                float(candidate_dominant) - float(baseline_dominant)
                if candidate_dominant is not None and baseline_dominant is not None
                else None
            ),
        },
    }


def rank_variants(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row, geometry_gate_pass=_passes(row)) for row in rows]
    return sorted(
        result,
        key=lambda row: (
            bool(row["geometry_gate_pass"]),
            float(row.get("largest_component_ratio") or 0.0),
            -int(row.get("articulation_count") or 0),
            -int(row.get("bridge_count") or 0),
            -float(row.get("reprojection_p99_px") or np.inf),
            int(row.get("points3D") or 0),
        ),
        reverse=True,
    )
