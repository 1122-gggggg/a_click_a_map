from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class MVGroup:
    group_id: str
    source_id: str
    source_name: str
    source_path: Path
    target_ids: tuple[str, ...]
    target_names: tuple[str, ...]
    target_paths: tuple[Path, ...]
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_path": str(self.source_path),
            "target_ids": list(self.target_ids),
            "target_names": list(self.target_names),
            "target_paths": [str(path) for path in self.target_paths],
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class MVGroupConfig:
    scope: str
    sparse_video_ids: tuple[str, ...] = ()
    weak_keyframe_ids: tuple[str, ...] = ()
    targets_per_source: int = 3
    maximum_bridge_hops: int = 3

    def __post_init__(self) -> None:
        if self.scope not in {"targeted", "full"}:
            raise ValueError("MV-RoMa scope must be targeted or full")
        if self.targets_per_source < 2:
            raise ValueError("MV-RoMa groups require at least two targets")


def _video(keyframe_id: str) -> str:
    return keyframe_id.split(":", 1)[0]


def _output_name(keyframe_id: str) -> str:
    video, frame = keyframe_id.split(":", 1)
    return f"{video}/frame_{frame}.jpg"


def _reject_p168(values: Sequence[str]) -> None:
    forbidden = [value for value in values if "P168" in value.upper() or "vid_540e7" in value]
    if forbidden:
        raise ValueError(f"P168 is forbidden from MV-RoMa mapping groups: {forbidden[0]}")


def _adjacency(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    graph: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        if str(row.get("admission") or "") != "VERIFIED":
            continue
        left, right = str(row["image_i"]), str(row["image_j"])
        score = float(row.get("parallax_p50_deg") or 0.0)
        graph.setdefault(left, []).append((right, score))
        graph.setdefault(right, []).append((left, score))
    for node, neighbors in graph.items():
        graph[node] = sorted(neighbors, key=lambda item: (-item[1], item[0]))
    return graph


def _path_to_other_video(
    source: str,
    graph: Mapping[str, Sequence[tuple[str, float]]],
    *,
    maximum_hops: int,
) -> tuple[str, ...]:
    source_video = _video(source)
    queue: deque[tuple[str, tuple[str, ...]]] = deque([(source, ())])
    visited = {source}
    while queue:
        node, path = queue.popleft()
        if len(path) >= maximum_hops:
            continue
        for neighbor, _ in graph.get(node, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            candidate = (*path, neighbor)
            if _video(neighbor) != source_video:
                return candidate
            queue.append((neighbor, candidate))
    return ()


def _multihop_context(
    source: str,
    graph: Mapping[str, Sequence[tuple[str, float]]],
    *,
    maximum_hops: int,
) -> tuple[str, ...]:
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    visited = {source}
    context: list[str] = []
    while queue:
        node, depth = queue.popleft()
        if depth >= maximum_hops:
            continue
        for neighbor, _ in graph.get(node, ()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            context.append(neighbor)
            queue.append((neighbor, depth + 1))
    return tuple(context)


def _targets_for_source(
    source: str,
    graph: Mapping[str, Sequence[tuple[str, float]]],
    config: MVGroupConfig,
) -> tuple[str, ...]:
    neighbors = list(graph.get(source, ()))
    cross = [name for name, _ in neighbors if _video(name) != _video(source)]
    same = [name for name, _ in neighbors if _video(name) == _video(source)]
    selected: list[str] = []
    rationale_path: tuple[str, ...] = ()
    if cross:
        selected.append(cross[0])
    else:
        rationale_path = _path_to_other_video(
            source, graph, maximum_hops=config.maximum_bridge_hops
        )
        selected.extend(rationale_path)
    context = _multihop_context(source, graph, maximum_hops=config.maximum_bridge_hops)
    for candidate in (*same, *context, *(name for name, _ in neighbors), *rationale_path):
        if candidate != source and candidate not in selected:
            selected.append(candidate)
        if len(selected) >= config.targets_per_source:
            break
    return tuple(selected[: config.targets_per_source])


def plan_groups(
    selection: Mapping[str, Any],
    *,
    images_root: Path,
    config: MVGroupConfig,
) -> tuple[MVGroup, ...]:
    selected = tuple(sorted(str(value) for value in selection.get("selected_keyframes") or ()))
    rows = tuple(dict(row) for row in selection.get("admitted_pairs") or ())
    _reject_p168(
        [
            *selected,
            *(str(row.get(key) or "") for row in rows for key in ("image_i", "image_j")),
        ]
    )
    graph = _adjacency(rows)
    if config.scope == "full":
        sources = selected
    else:
        sparse = set(config.sparse_video_ids)
        weak = set(config.weak_keyframe_ids)
        sources = tuple(value for value in selected if _video(value) in sparse or value in weak)
    groups = []
    for source in sources:
        targets = _targets_for_source(source, graph, config)
        if len(targets) < 2:
            continue
        source_name = _output_name(source)
        target_names = tuple(_output_name(value) for value in targets)
        groups.append(
            MVGroup(
                group_id=f"mvroma_{source.replace(':', '_')}",
                source_id=source,
                source_name=source_name,
                source_path=images_root / source_name,
                target_ids=targets,
                target_names=target_names,
                target_paths=tuple(images_root / name for name in target_names),
                rationale=("sparse_or_weak_rescue" if config.scope == "targeted" else "full_graph"),
            )
        )
    return tuple(sorted(groups, key=lambda group: group.source_id))


__all__ = ["MVGroup", "MVGroupConfig", "plan_groups"]
