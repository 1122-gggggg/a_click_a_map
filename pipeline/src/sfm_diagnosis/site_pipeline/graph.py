"""Deterministic graph diagnostics shared by all four Pre-SfM graph levels."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping


Node = Hashable
Edge = tuple[Node, Node]


def _key(value: object) -> str:
    return repr(value)


def canonical_edge(left: Node, right: Node) -> Edge:
    if left == right:
        raise ValueError("self edges are not supported")
    return (left, right) if _key(left) <= _key(right) else (right, left)


def normalize_edges(edges: Iterable[Edge]) -> tuple[Edge, ...]:
    return tuple(sorted({canonical_edge(*edge) for edge in edges}, key=_key))


def adjacency(nodes: Iterable[Node], edges: Iterable[Edge]) -> dict[Node, set[Node]]:
    graph = {node: set() for node in nodes}
    for left, right in normalize_edges(edges):
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def connected_components(
    nodes: Iterable[Node], edges: Iterable[Edge]
) -> tuple[tuple[Node, ...], ...]:
    graph = adjacency(nodes, edges)
    seen: set[Node] = set()
    result: list[tuple[Node, ...]] = []
    for root in sorted(graph, key=_key):
        if root in seen:
            continue
        queue = [root]
        component: list[Node] = []
        while queue:
            node = queue.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            queue.extend(sorted(graph[node] - seen, key=_key, reverse=True))
        result.append(tuple(sorted(component, key=_key)))
    return tuple(sorted(result, key=lambda part: (-len(part), tuple(map(_key, part)))))


def largest_component_ratio(nodes: Iterable[Node], edges: Iterable[Edge]) -> float:
    material = tuple(dict.fromkeys(nodes))
    if not material:
        return 0.0
    components = connected_components(material, edges)
    return len(components[0]) / len(material)


def _tarjan(edges: Iterable[Edge]) -> tuple[set[Node], set[Edge]]:
    normalized = normalize_edges(edges)
    graph = adjacency((), normalized)
    discovery: dict[Node, int] = {}
    low: dict[Node, int] = {}
    parent: dict[Node, Node] = {}
    articulations: set[Node] = set()
    bridges: set[Edge] = set()
    clock = 0

    def visit(node: Node) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = low[node] = clock
        children = 0
        for neighbor in sorted(graph[node], key=_key):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if node not in parent and children > 1:
                    articulations.add(node)
                if node in parent and low[neighbor] >= discovery[node]:
                    articulations.add(node)
                if low[neighbor] > discovery[node]:
                    bridges.add(canonical_edge(node, neighbor))
            elif parent.get(node) != neighbor:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(graph, key=_key):
        if node not in discovery:
            visit(node)
    return articulations, bridges


def articulation_points(edges: Iterable[Edge]) -> set[Node]:
    return _tarjan(edges)[0]


def bridge_edges(edges: Iterable[Edge]) -> set[Edge]:
    return _tarjan(edges)[1]


def replacement_paths(edges: Iterable[Edge], removed: Edge) -> int:
    """Return the number of edge-disjoint alternative paths for ``removed``.

    The removed edge is excluded before running an integral unit-capacity
    Edmonds-Karp max flow. This is the plan's authoritative redundancy ``R_e``.
    """

    source, target = canonical_edge(*removed)
    material = [edge for edge in normalize_edges(edges) if edge != (source, target)]
    nodes = {node for edge in material for node in edge} | {source, target}
    capacity: dict[Node, dict[Node, int]] = {node: defaultdict(int) for node in nodes}
    for left, right in material:
        capacity[left][right] += 1
        capacity[right][left] += 1
    residual = {node: defaultdict(int, neighbors) for node, neighbors in capacity.items()}
    flow = 0
    while True:
        parent: dict[Node, Node | None] = {source: None}
        queue: deque[Node] = deque([source])
        while queue and target not in parent:
            node = queue.popleft()
            for neighbor in sorted(residual[node], key=_key):
                if residual[node][neighbor] > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)
        if target not in parent:
            return flow
        flow += 1
        node = target
        while parent[node] is not None:
            previous = parent[node]
            residual[previous][node] -= 1
            residual[node][previous] += 1
            node = previous


def cycle_basis_membership(
    nodes: Iterable[Node], edges: Iterable[Edge]
) -> tuple[frozenset[Edge], ...]:
    """Return a deterministic fundamental cycle basis.

    Cycle-basis membership is diagnostic and basis-dependent; ``R_e`` remains
    the primary alternative-support measure.
    """

    graph = adjacency(nodes, edges)
    parent: dict[Node, Node | None] = {}
    depth: dict[Node, int] = {}
    tree_edges: set[Edge] = set()
    for root in sorted(graph, key=_key):
        if root in parent:
            continue
        parent[root] = None
        depth[root] = 0
        queue: deque[Node] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph[node], key=_key):
                if neighbor in parent:
                    continue
                parent[neighbor] = node
                depth[neighbor] = depth[node] + 1
                tree_edges.add(canonical_edge(node, neighbor))
                queue.append(neighbor)

    cycles: list[frozenset[Edge]] = []
    for chord in normalize_edges(edges):
        if chord in tree_edges:
            continue
        left, right = chord
        cycle: set[Edge] = {chord}
        a, b = left, right
        while depth[a] > depth[b]:
            previous = parent[a]
            if previous is None:
                break
            cycle.add(canonical_edge(a, previous))
            a = previous
        while depth[b] > depth[a]:
            previous = parent[b]
            if previous is None:
                break
            cycle.add(canonical_edge(b, previous))
            b = previous
        while a != b:
            parent_a, parent_b = parent[a], parent[b]
            if parent_a is None or parent_b is None:
                break
            cycle.add(canonical_edge(a, parent_a))
            cycle.add(canonical_edge(b, parent_b))
            a, b = parent_a, parent_b
        cycles.append(frozenset(cycle))
    return tuple(sorted(cycles, key=lambda cycle: tuple(sorted(map(_key, cycle)))))


def cycle_support(nodes: Iterable[Node], edges: Iterable[Edge]) -> dict[Edge, int]:
    support = {edge: 0 for edge in normalize_edges(edges)}
    for cycle in cycle_basis_membership(nodes, edges):
        for edge in cycle:
            support[edge] += 1
    return support


def star_overlap(graph: Mapping[Node, Iterable[Node]], left: Node, right: Node) -> int:
    return len(set(graph.get(left, ())) & set(graph.get(right, ())))


@dataclass(frozen=True)
class GraphDiagnostics:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    components: tuple[tuple[Node, ...], ...]
    largest_component_ratio: float
    articulation_nodes: tuple[Node, ...]
    bridges: tuple[Edge, ...]
    replacement_path_counts: Mapping[Edge, int]
    cycle_support: Mapping[Edge, int]


def diagnose_graph(nodes: Iterable[Node], edges: Iterable[Edge]) -> GraphDiagnostics:
    node_rows = tuple(sorted(dict.fromkeys(nodes), key=_key))
    edge_rows = normalize_edges(edges)
    components = connected_components(node_rows, edge_rows)
    bridges = tuple(sorted(bridge_edges(edge_rows), key=_key))
    return GraphDiagnostics(
        nodes=node_rows,
        edges=edge_rows,
        components=components,
        largest_component_ratio=largest_component_ratio(node_rows, edge_rows),
        articulation_nodes=tuple(sorted(articulation_points(edge_rows), key=_key)),
        bridges=bridges,
        replacement_path_counts={edge: replacement_paths(edge_rows, edge) for edge in edge_rows},
        cycle_support=cycle_support(node_rows, edge_rows),
    )


__all__ = [
    "GraphDiagnostics",
    "adjacency",
    "articulation_points",
    "bridge_edges",
    "canonical_edge",
    "connected_components",
    "cycle_basis_membership",
    "cycle_support",
    "diagnose_graph",
    "largest_component_ratio",
    "normalize_edges",
    "replacement_paths",
    "star_overlap",
]
