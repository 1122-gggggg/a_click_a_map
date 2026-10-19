"""Session-graph connectivity: Laplacian λ2, Tarjan bridges, CRITICAL_BRIDGE.

Image-graph Tarjan is a different object. Isolated / single-node graphs have λ2 = 0.
VPR / retrieval edges are not geometric and must not appear as STRONG/USABLE.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .types import SessionEdgeQuality, edge_is_vpr_only

_USABLE = frozenset({"STRONG", "USABLE"})


def _edge_endpoints(edge: Any) -> tuple[str, str]:
    if isinstance(edge, SessionEdgeQuality):
        return edge.session_a, edge.session_b
    if isinstance(edge, Mapping):
        return str(edge["session_a"]), str(edge["session_b"])
    if isinstance(edge, Sequence) and len(edge) >= 2 and not isinstance(edge, (str, bytes)):
        return str(edge[0]), str(edge[1])
    raise TypeError("edge must be SessionEdgeQuality, mapping, or (a, b, ...)")


def _edge_status(edge: Any) -> str:
    if isinstance(edge, SessionEdgeQuality):
        return edge.status
    if isinstance(edge, Mapping):
        return str(edge.get("status") or "USABLE")
    return "USABLE"


def _edge_weight(edge: Any) -> float:
    if isinstance(edge, SessionEdgeQuality):
        return float(edge.edge_quality_score)
    if isinstance(edge, Mapping):
        value = edge.get("edge_quality_score", edge.get("weight", 1.0))
        return float(value if value is not None else 1.0)
    if isinstance(edge, Sequence) and len(edge) >= 3:
        return float(edge[2])
    return 1.0


def _is_vpr_only(edge: Any) -> bool:
    if isinstance(edge, SessionEdgeQuality):
        return edge_is_vpr_only(edge)
    if isinstance(edge, Mapping):
        try:
            return edge_is_vpr_only(SessionEdgeQuality(**edge))  # type: ignore[arg-type]
        except TypeError:
            return False
    return False


def _usable_undirected(
    edges: Iterable[Any],
    *,
    require_usable: bool,
) -> dict[tuple[str, str], float]:
    weights: dict[tuple[str, str], float] = {}
    for edge in edges:
        left, right = _edge_endpoints(edge)
        if not left or not right or left == right:
            continue
        if require_usable and _edge_status(edge) not in _USABLE:
            continue
        if require_usable and _is_vpr_only(edge):
            continue
        key = (left, right) if left <= right else (right, left)
        weight = max(0.0, _edge_weight(edge))
        weights[key] = max(weights.get(key, 0.0), weight)
    return weights


def _ordered_nodes(nodes: Iterable[str], edges: Mapping[tuple[str, str], float]) -> list[str]:
    named = {str(node) for node in nodes}
    for left, right in edges:
        named.add(left)
        named.add(right)
    return sorted(named)


def adjacency_laplacian(nodes: Iterable[str], edges: Iterable[Any]) -> np.ndarray:
    """Return L = D − A over STRONG/USABLE session edges (weight = edge_quality_score)."""

    usable = _usable_undirected(edges, require_usable=True)
    names = _ordered_nodes(nodes, usable)
    size = len(names)
    laplacian = np.zeros((size, size), dtype=float)
    if size == 0:
        return laplacian
    index = {name: i for i, name in enumerate(names)}
    for (left, right), weight in usable.items():
        i, j = index[left], index[right]
        laplacian[i, i] += weight
        laplacian[j, j] += weight
        laplacian[i, j] -= weight
        laplacian[j, i] -= weight
    return laplacian


def fiedler_value(nodes: Iterable[str], edges: Iterable[Any]) -> float:
    """Second-smallest Laplacian eigenvalue. Isolated or single node → 0.0."""

    names = list(nodes)
    usable = _usable_undirected(edges, require_usable=True)
    names = _ordered_nodes(names, usable)
    if len(names) <= 1:
        return 0.0
    laplacian = adjacency_laplacian(names, edges)
    values = np.linalg.eigvalsh(laplacian)
    values = np.sort(np.real(values))
    if len(values) < 2:
        return 0.0
    lambda2 = float(max(0.0, values[1]))
    if not np.isfinite(lambda2):
        return 0.0
    return lambda2


def _adjacency(nodes: Iterable[str], edges: Iterable[Any], *, require_usable: bool) -> dict[str, set[str]]:
    usable = _usable_undirected(edges, require_usable=require_usable)
    names = _ordered_nodes(nodes, usable)
    adj: dict[str, set[str]] = {name: set() for name in names}
    for left, right in usable:
        adj[left].add(right)
        adj[right].add(left)
    return adj


def session_tarjan_bridges(nodes: Iterable[str], edges: Iterable[Any]) -> list[tuple[str, str]]:
    """Undirected bridges on the STRONG/USABLE session adjacency."""

    adj = _adjacency(nodes, edges, require_usable=True)
    tin: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    bridges: list[tuple[str, str]] = []
    tick = 0

    def visit(node: str) -> None:
        nonlocal tick
        tin[node] = low[node] = tick
        tick += 1
        for neighbor in sorted(adj[node]):
            if neighbor not in tin:
                parent[neighbor] = node
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > tin[node]:
                    pair = (node, neighbor) if node <= neighbor else (neighbor, node)
                    bridges.append(pair)
            elif neighbor != parent.get(node):
                low[node] = min(low[node], tin[neighbor])

    for name in adj:
        if name not in tin:
            parent[name] = None
            visit(name)
    bridges.sort()
    return bridges


def session_articulation_points(nodes: Iterable[str], edges: Iterable[Any]) -> list[str]:
    """Articulation sessions on the STRONG/USABLE adjacency."""

    adj = _adjacency(nodes, edges, require_usable=True)
    tin: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    arts: set[str] = set()
    tick = 0

    def visit(node: str) -> None:
        nonlocal tick
        tin[node] = low[node] = tick
        tick += 1
        children = 0
        for neighbor in sorted(adj[node]):
            if neighbor not in tin:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent.get(node) is None and children > 1:
                    arts.add(node)
                if parent.get(node) is not None and low[neighbor] >= tin[node]:
                    arts.add(node)
            elif neighbor != parent.get(node):
                low[node] = min(low[node], tin[neighbor])

    for name in adj:
        if name not in tin:
            parent[name] = None
            visit(name)
    return sorted(arts)


def _components(adj: Mapping[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    groups: list[list[str]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        members = [start]
        while stack:
            node = stack.pop()
            for neighbor in adj[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
                    members.append(neighbor)
        groups.append(sorted(members))
    return groups


def classify_critical_bridges(nodes: Iterable[str], edges: Iterable[Any]) -> list[dict[str, Any]]:
    """A session bridge is CRITICAL when it is the unique connector of two groups ≥2."""

    adj = _adjacency(nodes, edges, require_usable=True)
    bridges = session_tarjan_bridges(nodes, edges)
    rows: list[dict[str, Any]] = []
    for left, right in bridges:
        reduced = {node: set(neighbors) for node, neighbors in adj.items()}
        reduced[left].discard(right)
        reduced[right].discard(left)
        groups = _components(reduced)
        side_a = next((group for group in groups if left in group), [left])
        side_b = next((group for group in groups if right in group), [right])
        is_critical = len(side_a) >= 2 and len(side_b) >= 2
        rows.append(
            {
                "session_a": left,
                "session_b": right,
                "is_bridge": True,
                "is_critical_bridge": is_critical,
                "group_a": side_a,
                "group_b": side_b,
                "status": "CRITICAL_BRIDGE" if is_critical else "BRIDGE",
            }
        )
    return rows


def edge_connectivity(nodes: Iterable[str], edges: Iterable[Any]) -> int:
    """Small-graph edge connectivity: 0 if disconnected, 1 if a bridge exists, else min degree."""

    adj = _adjacency(nodes, edges, require_usable=True)
    if not adj:
        return 0
    comps = _components(adj)
    if len(comps) != 1:
        return 0
    if len(adj) <= 1:
        return 0
    degrees = [len(adj[name]) for name in adj]
    if min(degrees) == 0:
        return 0
    if session_tarjan_bridges(nodes, edges):
        return 1
    return int(min(degrees))


def session_graph_diagnostics(nodes: Iterable[str], edges: Iterable[Any]) -> dict[str, Any]:
    """Bundle λ2, bridges, articulations, and critical bridges."""

    usable = _usable_undirected(edges, require_usable=True)
    names = _ordered_nodes(nodes, usable)
    adj = _adjacency(names, edges, require_usable=True)
    degrees = [len(adj[name]) for name in names] if names else []
    components = _components(adj) if adj else [[name] for name in names]
    critical = classify_critical_bridges(names, edges)
    bridges = session_tarjan_bridges(names, edges)
    arts = session_articulation_points(names, edges)
    component_of: dict[str, list[str]] = {}
    for group in components:
        for member in group:
            component_of[member] = group
    per_node = {}
    for name in names:
        incident = sum(1 for a, b in bridges if name in {a, b})
        per_node[name] = {
            "degree": len(adj.get(name, ())),
            "incident_bridges": incident,
            "is_articulation": name in arts,
            "component_size": len(component_of.get(name) or [name]),
        }
    return {
        "nodes": names,
        "fiedler_value": fiedler_value(names, edges),
        "bridges": bridges,
        "articulation_points": arts,
        "critical_bridges": critical,
        "connected_components": len(components) if names else 0,
        "connected": bool(names) and len(components) == 1,
        "average_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "min_degree": int(min(degrees)) if degrees else 0,
        "edge_connectivity": edge_connectivity(names, edges),
        "component_sizes": [len(group) for group in components],
        "per_node": per_node,
        "usable_edges": len(usable),
    }
