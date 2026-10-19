"""Balanced, localization-dense selection over immutable verified pair evidence."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BalancedSelectionResult:
    selected_keyframes: frozenset[str]
    admitted_pairs: tuple[dict[str, Any], ...]
    selected_segments: tuple[str, ...]
    connector_keyframes: tuple[str, ...]
    per_video_counts: dict[str, int]
    components: int
    connected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "BALANCED_LOCALIZATION_DENSE_SELECTION",
            "selected_keyframes": sorted(self.selected_keyframes),
            "selected_segments": list(self.selected_segments),
            "connector_keyframes": list(self.connector_keyframes),
            "per_video_counts": dict(sorted(self.per_video_counts.items())),
            "admitted_pairs": list(self.admitted_pairs),
            "components": self.components,
            "connected": self.connected,
        }


def build_balanced_selection(
    keyframes: Iterable[Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int],
) -> BalancedSelectionResult:
    """Select temporally distributed high-support views, then connect them exactly.

    Quotas apply to ordinary candidates. Shortest paths may add extra connector
    keyframes from the verified graph, but no unverified pair can enter the output.
    """

    rows = [dict(row) for row in keyframes]
    by_id = {str(row["keyframe_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("keyframe IDs must be unique")
    normalized_quotas = {str(video): int(value) for video, value in quotas.items()}
    if not normalized_quotas or any(value <= 0 for value in normalized_quotas.values()):
        raise ValueError("every video quota must be positive")

    verified_rows: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = defaultdict(set)
    for raw in geometry:
        if raw.get("admission") != "VERIFIED":
            continue
        row = dict(raw)
        left, right = str(row.get("image_i")), str(row.get("image_j"))
        if left not in by_id or right not in by_id:
            raise ValueError("verified geometry references an unknown keyframe")
        if left == right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        verified_rows.append(row)

    selected: set[str] = set()
    for video, quota in sorted(normalized_quotas.items()):
        candidates = sorted(
            (
                row
                for row in rows
                if str(row.get("video_id")) == video
                and row.get("selection_eligible", True) is not False
                and str(row["keyframe_id"]) in adjacency
            ),
            key=lambda row: (
                float(row.get("timestamp") or row.get("source_pts_seconds") or 0.0),
                str(row["keyframe_id"]),
            ),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"video quota cannot be met for {video}: requested={quota}, "
                f"eligible={len(candidates)}"
            )
        for index in range(quota):
            start = round(index * len(candidates) / quota)
            stop = max(start + 1, round((index + 1) * len(candidates) / quota))
            bucket = candidates[start:stop]
            choice = max(
                bucket,
                key=lambda row: _candidate_score(row, by_id=by_id, adjacency=adjacency),
            )
            selected.add(str(choice["keyframe_id"]))

    initial = set(selected)
    components = _components(selected, adjacency)
    if not components:
        raise ValueError("balanced selection is empty")
    connected = set(components[0])
    for component in components[1:]:
        path = _shortest_path_to_set(component, connected, adjacency)
        if path is None:
            raise ValueError("quota selections are disconnected in the verified graph")
        selected.update(path)
        connected.update(component)
        connected.update(path)

    final_components = _components(selected, adjacency)
    selected_segments = tuple(
        sorted({str(by_id[key].get("segment_id") or "") for key in selected})
    )
    admitted_pairs = tuple(
        row
        for row in verified_rows
        if str(row["image_i"]) in selected and str(row["image_j"]) in selected
    )
    if not admitted_pairs:
        raise ValueError("balanced selection contains no verified pair")
    return BalancedSelectionResult(
        selected_keyframes=frozenset(selected),
        admitted_pairs=admitted_pairs,
        selected_segments=selected_segments,
        connector_keyframes=tuple(sorted(selected - initial)),
        per_video_counts={
            video: sum(str(by_id[key].get("video_id")) == video for key in selected)
            for video in sorted(normalized_quotas)
        },
        components=len(final_components),
        connected=len(final_components) == 1,
    )


def _candidate_score(
    row: Mapping[str, Any],
    *,
    by_id: Mapping[str, Mapping[str, Any]],
    adjacency: Mapping[str, set[str]],
) -> tuple[int, int, float, str]:
    keyframe_id = str(row["keyframe_id"])
    video = str(row.get("video_id"))
    neighbors = adjacency.get(keyframe_id, set())
    cross_video = sum(str(by_id[value].get("video_id")) != video for value in neighbors)
    return (
        cross_video,
        len(neighbors),
        float(row.get("blur_score") or row.get("blur_variance") or 0.0),
        keyframe_id,
    )


def _components(nodes: set[str], adjacency: Mapping[str, set[str]]) -> list[set[str]]:
    remaining = set(nodes)
    output: list[set[str]] = []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component = {root}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, set()) & remaining):
                remaining.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        output.append(component)
    return sorted(output, key=lambda value: (-len(value), min(value)))


def _shortest_path_to_set(
    sources: Sequence[str] | set[str],
    targets: set[str],
    adjacency: Mapping[str, set[str]],
) -> tuple[str, ...] | None:
    queue = deque(sorted(sources))
    previous: dict[str, str | None] = {source: None for source in sources}
    meeting: str | None = None
    while queue and meeting is None:
        current = queue.popleft()
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in targets:
                previous[neighbor] = current
                meeting = neighbor
                break
            if neighbor not in previous:
                previous[neighbor] = current
                queue.append(neighbor)
    if meeting is None:
        return None
    path = []
    current: str | None = meeting
    while current is not None:
        path.append(current)
        current = previous[current]
    return tuple(path)


__all__ = ["BalancedSelectionResult", "build_balanced_selection"]
