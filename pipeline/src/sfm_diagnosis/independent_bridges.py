"""Fail-closed primitives for cross-session overlap diagnostics.

These helpers only organize retrieval and geometric observations.  They never
modify a frozen base map or treat a retrieval/PnP result as authority to fuse.
VPR/retrieval is a candidate generator, not a geometric edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MutualPair:
    """One reciprocal query/reference retrieval candidate."""

    query_index: int
    reference_index: int
    score: float


@dataclass(frozen=True)
class BridgeObservation:
    """A geometrically verified query-to-reference candidate used only for clustering."""

    query_index: int
    reference_name: str
    score: float
    reference_center: tuple[float, float, float]


def mutual_topk_pairs(scores: np.ndarray, *, topk: int) -> list[MutualPair]:
    """Return only candidates ranked in each other's top-``k`` lists.

    ``scores`` has one row per query and one column per reference.  A stable
    sort makes equal scores deterministic rather than silently creating
    arbitrary bridge candidates.
    """

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("scores must be a non-empty 2D array")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not 1 <= topk <= min(values.shape):
        raise ValueError("topk must be within both score dimensions")

    query_ranked = np.argsort(-values, axis=1, kind="stable")[:, :topk]
    reference_ranked = np.argsort(-values, axis=0, kind="stable")[:topk, :]
    reciprocal: list[MutualPair] = []
    for query_index, references in enumerate(query_ranked):
        for reference_index in references:
            if query_index in reference_ranked[:, reference_index]:
                reciprocal.append(
                    MutualPair(
                        query_index=query_index,
                        reference_index=int(reference_index),
                        score=float(values[query_index, reference_index]),
                    )
                )
    return sorted(reciprocal, key=lambda pair: (pair.query_index, pair.reference_index))


def grid_cells_occupied(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    columns: int = 4,
    rows: int = 4,
) -> int:
    """Count distinct grid cells containing finite image points."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if min(width, height, columns, rows) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    if not len(values):
        return 0

    finite = values[np.isfinite(values).all(axis=1)]
    if not len(finite):
        return 0
    x = np.clip((finite[:, 0] * columns / width).astype(int), 0, columns - 1)
    y = np.clip((finite[:, 1] * rows / height).astype(int), 0, rows - 1)
    return len({(int(cell_x), int(cell_y)) for cell_x, cell_y in zip(x, y, strict=True)})


def cluster_bridge_observations(
    observations: list[BridgeObservation],
    *,
    query_frame_separation: int,
    reference_center_separation: float,
) -> list[list[BridgeObservation]]:
    """Cluster nearby temporal and reference-camera observations into one bridge.

    Pairs are in the same bridge only when both their query-frame distance
    and reference camera-center distance are below the supplied independence
    scales.  Sequential close pairs therefore collapse into one group.
    """

    if query_frame_separation <= 0 or reference_center_separation <= 0:
        raise ValueError("independence separations must be positive")
    ordered = sorted(observations, key=lambda item: (item.query_index, item.reference_name))
    parent = list(range(len(ordered)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, first in enumerate(ordered):
        first_center = np.asarray(first.reference_center, dtype=np.float64)
        for right in range(left + 1, len(ordered)):
            second = ordered[right]
            if abs(second.query_index - first.query_index) >= query_frame_separation:
                continue
            second_center = np.asarray(second.reference_center, dtype=np.float64)
            if float(np.linalg.norm(first_center - second_center)) < reference_center_separation:
                join(left, right)

    grouped: dict[int, list[BridgeObservation]] = {}
    for index, observation in enumerate(ordered):
        grouped.setdefault(root(index), []).append(observation)
    return list(grouped.values())


def decide_overlap_admission(
    *,
    independent_bridge_count: int,
    pnp_anchor_count: int,
    independent_source_model_available: bool,
    sim3_consistent: bool,
) -> dict[str, object]:
    """Return an evidence-only fusion decision that fails closed by default.

    Fewer than two independent bridges, or an inconsistent / unavailable Sim3,
    is ``NO_GO``.  ``map_fusion_authorized`` is always ``False`` here; fusion
    is a later explicit step, never implied by this helper.
    """

    failures: list[str] = []
    if independent_bridge_count < 2:
        failures.append("INSUFFICIENT_INDEPENDENT_BRIDGES")
    if pnp_anchor_count < 2:
        failures.append("INSUFFICIENT_INDEPENDENT_PNP_ANCHORS")
    if not independent_source_model_available:
        failures.append("SIM3_UNAVAILABLE_NO_INDEPENDENT_SOURCE_MODEL")
    if not sim3_consistent:
        failures.append("SIM3_INCONSISTENT_OR_UNAVAILABLE")
    return {
        "status": "NO_GO" if failures else "ADMITTED_FOR_SHADOW_ONLY",
        "failure_reasons": failures,
        "independent_bridge_count": independent_bridge_count,
        "pnp_anchor_count": pnp_anchor_count,
        "independent_source_model_available": independent_source_model_available,
        "sim3_consistent": sim3_consistent,
        "map_fusion_authorized": False,
    }


__all__ = [
    "BridgeObservation",
    "MutualPair",
    "cluster_bridge_observations",
    "decide_overlap_admission",
    "grid_cells_occupied",
    "mutual_topk_pairs",
]
