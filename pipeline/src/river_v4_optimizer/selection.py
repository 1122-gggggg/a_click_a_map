from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable, Mapping


def _component_sizes(nodes: set[str], pairs: Iterable[Mapping[str, Any]]) -> list[int]:
    return [len(component) for component in _connected_components(nodes, pairs)]


def _connected_components(
    nodes: set[str], pairs: Iterable[Mapping[str, Any]]
) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        left, right = str(pair["image_i"]), str(pair["image_j"])
        if left in nodes and right in nodes and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(nodes)
    components = []
    while remaining:
        root = remaining.pop()
        queue = deque([root])
        component = {root}
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return sorted(components, key=lambda values: (-len(values), sorted(values)))


def build_connected_submap_selection(
    keyframes: Mapping[str, Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
    *,
    allowed_keyframes: set[str],
    required_videos: set[str],
    minimum_images_per_video: int,
    profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep the deterministic largest VERIFIED component for an isolated submap."""

    if minimum_images_per_video < 1:
        raise ValueError("minimum_images_per_video must be positive")
    unknown = allowed_keyframes - set(keyframes)
    if unknown:
        raise ValueError(f"unknown keyframes: {sorted(unknown)}")
    rows = [
        dict(row)
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in allowed_keyframes
        and str(row.get("image_j")) in allowed_keyframes
    ]
    components = _connected_components(set(allowed_keyframes), rows)
    if not components:
        raise ValueError("submap selection has no keyframes")
    selected = components[0]
    video_counts: dict[str, int] = defaultdict(int)
    for keyframe_id in selected:
        video_counts[keyframe_id.split(":", 1)[0]] += 1
    missing = sorted(
        video
        for video in required_videos
        if video_counts.get(video, 0) < minimum_images_per_video
    )
    if missing:
        raise ValueError(f"largest submap component lacks required video support: {missing}")
    admitted = [
        row
        for row in rows
        if str(row["image_i"]) in selected and str(row["image_j"]) in selected
    ]
    segments = sorted({str(keyframes[key]["segment_id"]) for key in selected})
    result = {
        "schema_version": 2,
        "artifact_type": "SUBMAP_SELECTION",
        "selection_profile": profile,
        "selected_keyframes": sorted(selected),
        "active_segments": segments,
        "admitted_pairs": admitted,
        "mapping_modes": {segment: "TRIANGULATE" for segment in segments},
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "CONNECTED_SUBMAP_SELECTION_RECEIPT",
        "profile": profile,
        "allowed_keyframes": len(allowed_keyframes),
        "component_sizes_before": [len(component) for component in components],
        "selected_keyframes": len(selected),
        "admitted_pairs": len(admitted),
        "per_video_images": dict(sorted(video_counts.items())),
        "dropped_keyframes": sorted(allowed_keyframes - selected),
    }
    return result, receipt


def rewire_selection(
    selection: Mapping[str, Any],
    keyframes: Mapping[str, Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
    *,
    remove: set[str],
    add: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_before = {str(value) for value in selection.get("selected_keyframes") or ()}
    unknown = (remove | add) - set(keyframes)
    if unknown:
        raise ValueError(f"unknown keyframes: {sorted(unknown)}")
    selected = (selected_before - remove) | add
    admitted = [
        dict(row)
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in selected
        and str(row.get("image_j")) in selected
    ]
    components = _component_sizes(selected, admitted)
    if components != [len(selected)]:
        raise ValueError(f"rewired selection is disconnected: {components}")
    segments = {str(keyframes[key]["segment_id"]) for key in selected}
    result = dict(selection)
    result["selected_keyframes"] = sorted(selected)
    result["active_segments"] = sorted(segments)
    result["admitted_pairs"] = admitted
    result["mapping_modes"] = {segment: "TRIANGULATE" for segment in sorted(segments)}
    result["selection_profile"] = "RIVER_V4_DIRECT_MAIN_CONNECTOR_REWIRE"
    receipt = {
        "artifact_type": "RIVER_V4_CONNECTOR_REWIRE",
        "selected_before": len(selected_before),
        "selected_after": len(selected),
        "removed_keyframes": sorted(remove),
        "added_keyframes": sorted(add),
        "admitted_pairs": len(admitted),
        "component_sizes_after": components,
    }
    return result, receipt
