"""Deterministic reference partitions for pose-source attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np

from river_map_quality.pose_attribution import camera_center


@dataclass(frozen=True)
class ReferenceSubset:
    kind: str
    label: str
    references: tuple[str, ...]


def _route(name: str) -> str:
    parts = PurePosixPath(name).parts
    if len(parts) < 2:
        raise ValueError(f"reference name does not contain a route: {name!r}")
    return parts[0]


def _frame(name: str) -> int:
    try:
        return int(PurePosixPath(name).stem)
    except ValueError as exc:
        raise ValueError(f"reference filename does not contain an integer frame: {name!r}") from exc


def _view_direction(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("reference poses must have shape (4, 4)")
    direction = matrix[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    norm = np.linalg.norm(direction)
    if not np.isfinite(direction).all() or norm <= np.finfo(float).eps:
        raise ValueError("reference viewing direction is invalid")
    return direction / norm


def _connected_components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    output: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        output.append(sorted(component))
    return output


def build_reference_subsets(
    references: Sequence[str],
    poses_by_name: Mapping[str, np.ndarray],
    *,
    camera_spacing: float,
    position_factor: float = 5.0,
    maximum_view_angle_deg: float = 20.0,
    maximum_frame_gap: int = 3,
) -> tuple[ReferenceSubset, ...]:
    """Build single, route, sequence-block, and pose-cluster subsets."""

    names = tuple(str(reference) for reference in references)
    if not names or len(set(names)) != len(names):
        raise ValueError("references must be a non-empty sequence of unique names")
    if camera_spacing <= 0 or position_factor <= 0 or maximum_view_angle_deg <= 0:
        raise ValueError("reference clustering thresholds must be positive")
    if maximum_frame_gap < 0:
        raise ValueError("maximum_frame_gap cannot be negative")
    missing = set(names) - set(poses_by_name)
    if missing:
        raise KeyError(f"reference poses are missing: {sorted(missing)!r}")
    centers = [camera_center(poses_by_name[name]) for name in names]
    directions = [_view_direction(poses_by_name[name]) for name in names]
    output = [ReferenceSubset("single", f"single:{name}", (name,)) for name in names]

    by_route: dict[str, list[str]] = {}
    for name in names:
        by_route.setdefault(_route(name), []).append(name)
    for route, route_names in sorted(by_route.items()):
        output.append(ReferenceSubset("route", f"route:{route}", tuple(route_names)))
        ordered = sorted(route_names, key=_frame)
        block = [ordered[0]]
        for name in ordered[1:]:
            if _frame(name) - _frame(block[-1]) <= maximum_frame_gap:
                block.append(name)
            else:
                output.append(
                    ReferenceSubset(
                        "block",
                        f"block:{route}:{_frame(block[0]):06d}-{_frame(block[-1]):06d}",
                        tuple(block),
                    )
                )
                block = [name]
        output.append(
            ReferenceSubset(
                "block",
                f"block:{route}:{_frame(block[0]):06d}-{_frame(block[-1]):06d}",
                tuple(block),
            )
        )

    adjacency = [set() for _ in names]
    maximum_distance = position_factor * camera_spacing
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            distance = np.linalg.norm(centers[first] - centers[second])
            cosine = np.clip(np.dot(directions[first], directions[second]), -1.0, 1.0)
            view_angle = float(np.degrees(np.arccos(cosine)))
            if distance <= maximum_distance and view_angle <= maximum_view_angle_deg:
                adjacency[first].add(second)
                adjacency[second].add(first)
    for cluster_index, component in enumerate(_connected_components(adjacency), 1):
        output.append(
            ReferenceSubset(
                "pose_cluster",
                f"pose_cluster:{cluster_index}",
                tuple(names[index] for index in component),
            )
        )
    output.append(ReferenceSubset("aggregate", "aggregate:all", names))
    return tuple(output)
