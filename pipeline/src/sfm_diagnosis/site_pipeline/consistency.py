"""Independent pair-cycle and LOO submap-consistency diagnostics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .graph import canonical_edge, cycle_basis_membership


def pair_rotation_cycles(
    keyframes: Iterable[Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    known = {str(row["keyframe_id"]) for row in keyframes}
    rotations: dict[tuple[str, str], np.ndarray] = {}
    edges = []
    for row in geometry:
        if row.get("admission") != "VERIFIED" or row.get("relative_rotation") is None:
            continue
        left, right = str(row["image_i"]), str(row["image_j"])
        if left not in known or right not in known or left == right:
            continue
        rotation = np.asarray(row["relative_rotation"], dtype=float)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            continue
        rotations[(left, right)] = rotation
        rotations[(right, left)] = rotation.T
        edges.append(canonical_edge(left, right))
    cycles = cycle_basis_membership(known, edges)
    results = []
    for cycle_edges in cycles:
        order = _ordered_cycle(cycle_edges)
        product = np.eye(3)
        complete = True
        for left, right in zip(order, order[1:] + order[:1], strict=True):
            rotation = rotations.get((left, right))
            if rotation is None:
                complete = False
                break
            product = rotation @ product
        if complete:
            results.append(
                {
                    "nodes": order,
                    "rotation_residual_deg": _rotation_angle_deg(product),
                }
            )
    return results


def loo_alignment_modes(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [row for row in results if row.get("status") == "OK"]
    scale_logs = [
        float(np.log(float(row["sim3_scale"])))
        for row in successful
        if row.get("sim3_scale") not in (None, 0)
    ]
    rotations = [
        float(row["rotation_p90_deg"])
        for row in successful
        if row.get("rotation_p90_deg") is not None
    ]
    positions = [
        float(row["position_p90_normalized"])
        for row in successful
        if row.get("position_p90_normalized") is not None
    ]
    scale_spread = max(scale_logs) - min(scale_logs) if len(scale_logs) >= 2 else 0.0
    rotation_spread = max(rotations) - min(rotations) if len(rotations) >= 2 else 0.0
    position_max = max(positions, default=0.0)
    multimodal = bool(
        len(successful) >= 2
        and (
            scale_spread > abs(float(np.log(1.15))) or rotation_spread > 2.0 or position_max > 0.02
        )
    )
    return {
        "evaluated_groups": len(successful),
        "scale_log_spread": scale_spread,
        "rotation_residual_spread_deg": rotation_spread,
        "max_position_residual_normalized": position_max,
        "multimodal_alignment": multimodal,
    }


def _ordered_cycle(edges) -> list[str]:
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        graph[str(left)].add(str(right))
        graph[str(right)].add(str(left))
    start = min(graph)
    order = [start]
    previous = None
    current = start
    while True:
        choices = sorted(graph[current] - ({previous} if previous is not None else set()))
        if not choices:
            break
        following = choices[0]
        if following == start:
            break
        order.append(following)
        previous, current = current, following
        if len(order) > len(graph):
            break
    return order


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


__all__ = ["loo_alignment_modes", "pair_rotation_cycles"]
