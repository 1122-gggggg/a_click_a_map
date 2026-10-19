"""Session-level rotation cycle error. This is not a pixel reprojection cycle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def _as_rotation(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    return matrix


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Geodesic angle of a rotation matrix, in degrees."""

    matrix = _as_rotation(rotation)
    trace = float(np.trace(matrix))
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_cycle_error_deg(R_ab: np.ndarray, R_bc: np.ndarray, R_ca: np.ndarray) -> float:
    """Angle of R_ab @ R_bc @ R_ca. Identity cycle → 0."""

    composed = _as_rotation(R_ab) @ _as_rotation(R_bc) @ _as_rotation(R_ca)
    return rotation_angle_deg(composed)


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _parse_relative_poses(
    relative_poses: Iterable[Any],
) -> dict[tuple[str, str], np.ndarray]:
    """Accept (a, b, R) tuples or mappings with session_a/session_b/R keys."""

    stored: dict[tuple[str, str], np.ndarray] = {}
    for item in relative_poses:
        rotation: Any
        if isinstance(item, Mapping):
            left = str(item.get("session_a") or item.get("a"))
            right = str(item.get("session_b") or item.get("b"))
            rotation = item.get("R", item.get("rotation", item.get("R_ab")))
        elif isinstance(item, Sequence) and len(item) >= 3:
            left, right, rotation = str(item[0]), str(item[1]), item[2]
        else:
            raise TypeError("relative_poses items must be (a, b, R) or a mapping")
        if not left or not right or left == right:
            continue
        matrix = _as_rotation(rotation)
        stored[(left, right)] = matrix
        stored[(right, left)] = matrix.T
    return stored


def tag_suspicious_edges(
    relative_poses: Iterable[Any],
    cycle_error_threshold_deg: float,
) -> list[dict[str, Any]]:
    """Tag undirected edges that participate in a 3-cycle above the heuristic threshold.

    Returns one record per evaluated directed cycle plus a rolled-up undirected
    edge tag. Edges with cycle error > threshold get status ``SUSPICIOUS_EDGE``.
    """

    if not np.isfinite(cycle_error_threshold_deg) or cycle_error_threshold_deg <= 0.0:
        raise ValueError("cycle_error_threshold_deg must be a positive finite heuristic")

    rotations = _parse_relative_poses(relative_poses)
    nodes = sorted({name for pair in rotations for name in pair})
    undirected = sorted({_pair_key(a, b) for a, b in rotations})
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in undirected:
        adjacency[left].add(right)
        adjacency[right].add(left)

    cycle_rows: list[dict[str, Any]] = []
    worst_error: dict[tuple[str, str], float] = {edge: 0.0 for edge in undirected}

    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes[i + 1 :], start=i + 1):
            if b not in adjacency[a]:
                continue
            for c in nodes[j + 1 :]:
                if c not in adjacency[b] or a not in adjacency[c]:
                    continue
                r_ab = rotations.get((a, b))
                r_bc = rotations.get((b, c))
                r_ca = rotations.get((c, a))
                if r_ab is None or r_bc is None or r_ca is None:
                    continue
                error = rotation_cycle_error_deg(r_ab, r_bc, r_ca)
                suspicious = error > cycle_error_threshold_deg
                cycle = (a, b, c)
                cycle_rows.append(
                    {
                        "cycle": cycle,
                        "error_deg": error,
                        "status": "SUSPICIOUS_EDGE" if suspicious else "OK",
                        "threshold_deg": cycle_error_threshold_deg,
                        "threshold_provenance": "heuristic",
                    }
                )
                for edge in (_pair_key(a, b), _pair_key(b, c), _pair_key(c, a)):
                    worst_error[edge] = max(worst_error[edge], error)

    tagged: list[dict[str, Any]] = []
    for left, right in undirected:
        error = worst_error.get((left, right), 0.0)
        status = "SUSPICIOUS_EDGE" if error > cycle_error_threshold_deg else "OK"
        tagged.append(
            {
                "session_a": left,
                "session_b": right,
                "status": status,
                "cycle_error_deg": error,
                "threshold_deg": cycle_error_threshold_deg,
                "threshold_provenance": "heuristic",
            }
        )
    tagged.extend(cycle_rows)
    return tagged
