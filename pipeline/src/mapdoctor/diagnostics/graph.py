from __future__ import annotations

import sys
from typing import Iterable

from mapdoctor.model import MapModel

from ._graph_sparse import (
    _connected_components,
    _graph_at_threshold,
    _shared_landmark_counts,
    _spectral_metrics,
    _sensitivity_thresholds,
    _threshold_sensitivity,
)
from ._graph_types import (
    ArticulationImage,
    BridgeEdge,
    CovisibilityFragilityReport,
    SpectralConnectivity,
    ThresholdSensitivityPoint,
)

__all__ = [
    "ArticulationImage",
    "BridgeEdge",
    "CovisibilityFragilityReport",
    "SpectralConnectivity",
    "ThresholdSensitivityPoint",
    "analyze_covisibility_fragility",
]


def analyze_covisibility_fragility(
    model: MapModel,
    *,
    minimum_shared_landmarks: int = 15,
    route_image_names: Iterable[str] | None = None,
) -> CovisibilityFragilityReport:
    """Find hard cuts, soft bottlenecks, and support-threshold instability."""

    if minimum_shared_landmarks < 1:
        raise ValueError("minimum_shared_landmarks must be >= 1")

    image_names = {
        image_id: image.name for image_id, image in model.images.items()
    }
    sensitivity_thresholds = _sensitivity_thresholds(minimum_shared_landmarks)
    minimum_retained_support = min(sensitivity_thresholds)
    (
        shared,
        track_observations,
        estimated_pair_expansions,
        shared_landmark_block_rows,
    ) = _shared_landmark_counts(
        model,
        minimum_support=minimum_retained_support,
    )
    adjacency, edge_support = _graph_at_threshold(
        model.images,
        shared,
        minimum_shared_landmarks,
    )

    components, component_index = _connected_components(adjacency)
    component_sizes = [len(component) for component in components]
    route_names = {str(name) for name in (route_image_names or [])}
    known_names = set(image_names.values())
    unknown_route_names = sorted(route_names - known_names)
    if unknown_route_names:
        raise ValueError(
            "route image manifest contains unknown images: "
            + ", ".join(unknown_route_names)
        )
    route_nodes = {
        image_id
        for image_id, name in image_names.items()
        if name in route_names
    }
    component_route_counts = [
        len(component & route_nodes) for component in components
    ]

    node_count = len(adjacency)
    old_recursion_limit = sys.getrecursionlimit()
    if node_count * 2 + 100 > old_recursion_limit:
        sys.setrecursionlimit(node_count * 2 + 100)

    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    subtree_size: dict[int, int] = {}
    subtree_route_count: dict[int, int] = {}
    separated_children: dict[int, list[int]] = {}
    bridge_children: list[tuple[int, int]] = []
    timer = 0

    def dfs(node: int) -> None:
        nonlocal timer
        timer += 1
        discovery[node] = timer
        low[node] = timer
        subtree_size[node] = 1
        subtree_route_count[node] = int(node in route_nodes)

        for neighbor in adjacency[node]:
            if neighbor not in discovery:
                parent[neighbor] = node
                dfs(neighbor)
                subtree_size[node] += subtree_size[neighbor]
                subtree_route_count[node] += subtree_route_count[neighbor]
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] >= discovery[node]:
                    separated_children.setdefault(node, []).append(neighbor)
                if low[neighbor] > discovery[node]:
                    bridge_children.append((node, neighbor))
            elif neighbor != parent.get(node):
                low[node] = min(low[node], discovery[neighbor])

    try:
        for node in adjacency:
            if node not in discovery:
                parent[node] = None
                dfs(node)
    finally:
        if sys.getrecursionlimit() != old_recursion_limit:
            sys.setrecursionlimit(old_recursion_limit)

    articulation_rows: list[ArticulationImage] = []
    for node, children in separated_children.items():
        is_root = parent[node] is None
        if is_root and len(children) <= 1:
            continue
        if not is_root and not children:
            continue

        component_id = component_index[node]
        component_size = component_sizes[component_id]
        component_route_count = component_route_counts[component_id]
        parts = [
            (subtree_size[child], subtree_route_count[child])
            for child in children
        ]
        if not is_root:
            separated_size = sum(size for size, _ in parts)
            separated_route = sum(route_count for _, route_count in parts)
            remainder_size = component_size - 1 - separated_size
            remainder_route = (
                component_route_count
                - int(node in route_nodes)
                - separated_route
            )
            if remainder_size > 0:
                parts.append((remainder_size, remainder_route))

        sizes = tuple(sorted((size for size, _ in parts), reverse=True))
        if not sizes:
            continue
        largest_index = max(
            range(len(parts)),
            key=lambda index: (parts[index][0], parts[index][1]),
        )
        separated_from_largest = (
            component_size - 1 - parts[largest_index][0]
        )
        route_separated = sum(
            route_count
            for index, (_, route_count) in enumerate(parts)
            if index != largest_index
        )
        articulation_rows.append(
            ArticulationImage(
                image_id=node,
                image_name=image_names[node],
                degree=len(adjacency[node]),
                component_size=component_size,
                component_sizes_after_removal=sizes,
                separated_from_largest=separated_from_largest,
                separated_fraction=(
                    separated_from_largest / (component_size - 1)
                    if component_size > 1
                    else 0.0
                ),
                route_images_separated=route_separated,
            )
        )

    bridge_rows: list[BridgeEdge] = []
    for parent_node, child in bridge_children:
        component_id = component_index[parent_node]
        component_size = component_sizes[component_id]
        child_side = subtree_size[child]
        other_side = component_size - child_side
        child_route = subtree_route_count[child]
        other_route = component_route_counts[component_id] - child_route
        if child_side <= other_side:
            smaller_size = child_side
            route_on_smaller = child_route
        else:
            smaller_size = other_side
            route_on_smaller = other_route
        edge = tuple(sorted((parent_node, child)))
        bridge_rows.append(
            BridgeEdge(
                image_id_a=edge[0],
                image_name_a=image_names[edge[0]],
                image_id_b=edge[1],
                image_name_b=image_names[edge[1]],
                shared_landmarks=edge_support[edge],
                component_size=component_size,
                side_sizes=tuple(
                    sorted((child_side, other_side), reverse=True)
                ),
                smaller_side_size=smaller_size,
                smaller_side_fraction=smaller_size / component_size,
                splits_route=child_route > 0 and other_route > 0,
                route_images_on_smaller_side=route_on_smaller,
            )
        )

    articulation_rows.sort(
        key=lambda row: (
            -row.route_images_separated,
            -row.separated_fraction,
            -row.degree,
            row.image_name,
        )
    )
    bridge_rows.sort(
        key=lambda row: (
            not row.splits_route,
            -row.smaller_side_fraction,
            row.shared_landmarks,
            row.image_name_a,
            row.image_name_b,
        )
    )
    isolated = tuple(
        {
            "image_id": image_id,
            "image_name": image_names[image_id],
        }
        for image_id in sorted(
            (
                node
                for node, neighbors in adjacency.items()
                if not neighbors
            ),
            key=lambda item: image_names[item],
        )
    )
    sorted_component_sizes = tuple(sorted(component_sizes, reverse=True))
    return CovisibilityFragilityReport(
        minimum_shared_landmarks=minimum_shared_landmarks,
        node_count=node_count,
        edge_count=len(edge_support),
        component_count=len(components),
        component_sizes=sorted_component_sizes,
        largest_component_ratio=(
            sorted_component_sizes[0] / node_count
            if node_count and sorted_component_sizes
            else 0.0
        ),
        isolated_images=isolated,
        articulation_images=tuple(articulation_rows),
        bridge_edges=tuple(bridge_rows),
        shared_landmark_backend="scipy_sparse_block_incidence_product",
        shared_landmark_block_rows=shared_landmark_block_rows,
        minimum_retained_support=minimum_retained_support,
        track_observations=track_observations,
        estimated_pair_expansions=estimated_pair_expansions,
        spectral_connectivity=_spectral_metrics(
            adjacency, edge_support, components
        ),
        threshold_sensitivity=_threshold_sensitivity(
            tuple(model.images),
            shared,
            minimum_shared_landmarks,
        ),
    )
