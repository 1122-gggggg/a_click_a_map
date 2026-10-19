from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.sparse import csr_matrix

from .models import MapData


@dataclass(frozen=True)
class CovisibilityGraph:
    image_support: np.ndarray
    pair_counts: dict[tuple[int, int], int]
    adjacency: tuple[tuple[int, ...], ...]
    degrees: np.ndarray
    strong_edges: int
    components: tuple[tuple[int, ...], ...]
    support_mode: str = "exact"
    omitted_long_track_count: int = 0

    def shared_points(self, a: int, b: int) -> int:
        """Return retained shared-point support for an image pair.

        In the default exact mode, pair support is computed exactly but only
        entries meeting the graph's ``min_shared_points`` threshold are
        retained in ``pair_counts`` to bound output memory.  Consequently, a
        zero result can mean either no shared points or sub-threshold support.
        Legacy approximation mode retains the historical unthresholded pair
        counts.
        """
        if a > b:
            a, b = b, a
        return int(self.pair_counts.get((a, b), 0))


def build_covisibility_graph(
    map_data: MapData,
    *,
    min_shared_points: int = 15,
    max_track_for_pair_expansion: int | None = None,
) -> CovisibilityGraph:
    """Build a registered-image covisibility graph from SfM tracks.

    Nodes are registered images. An undirected edge is considered strong when
    at least ``min_shared_points`` reconstructed landmarks are observed by both
    images. By default, exact shared-point counts are computed from a sparse
    image-track incidence product in row blocks. Only supports at least
    ``min_shared_points`` are retained in ``pair_counts`` and exposed through
    ``shared_points`` in exact mode. Supplying
    ``max_track_for_pair_expansion`` explicitly selects the legacy pair
    expansion approximation for compatibility; tracks longer than that cap
    are omitted from pair counts but still contribute to image support.
    """
    image_lookup = map_data.image_index()
    image_support = np.zeros(map_data.num_images, dtype=int)
    pair_counts: dict[tuple[int, int], int]
    omitted_long_track_count = 0
    support_mode = "exact"

    incidence_rows: list[int] = []
    incidence_columns: list[int] = []
    legacy_pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for point_index, obs in enumerate(map_data.track_image_ids):
        indices = sorted({image_lookup[int(i)] for i in obs if int(i) in image_lookup})
        for i in indices:
            image_support[i] += 1

        if max_track_for_pair_expansion is None:
            incidence_rows.extend(indices)
            incidence_columns.extend([point_index] * len(indices))
        else:
            if len(indices) > max_track_for_pair_expansion:
                omitted_long_track_count += 1
            if 2 <= len(indices) <= max_track_for_pair_expansion:
                for a, b in combinations(indices, 2):
                    legacy_pair_counts[(a, b)] += 1

    if max_track_for_pair_expansion is None:
        support_mode = "exact"
        pair_counts = _exact_pair_counts(
            incidence_rows,
            incidence_columns,
            num_images=map_data.num_images,
            num_points=map_data.num_points,
            min_shared_points=min_shared_points,
        )
    else:
        support_mode = "legacy_approximation"
        pair_counts = dict(legacy_pair_counts)

    adjacency_lists: list[list[int]] = [[] for _ in range(map_data.num_images)]
    strong_edges = 0
    for (a, b), shared in pair_counts.items():
        if shared >= min_shared_points:
            adjacency_lists[a].append(b)
            adjacency_lists[b].append(a)
            strong_edges += 1

    adjacency = tuple(tuple(sorted(v)) for v in adjacency_lists)
    degrees = np.asarray([len(v) for v in adjacency], dtype=int)
    components = connected_components(adjacency)
    return CovisibilityGraph(
        image_support=image_support,
        pair_counts=dict(pair_counts),
        adjacency=adjacency,
        degrees=degrees,
        strong_edges=strong_edges,
        components=components,
        support_mode=support_mode,
        omitted_long_track_count=omitted_long_track_count,
    )


def _exact_pair_counts(
    incidence_rows: list[int],
    incidence_columns: list[int],
    *,
    num_images: int,
    num_points: int,
    min_shared_points: int,
) -> dict[tuple[int, int], int]:
    """Return threshold-retained exact pair support from sparse incidence.

    The row-block product keeps the temporary ``block_rows x num_images``
    product near the eight-million-cell memory policy used by MapDoctor. The
    output itself is intentionally threshold-retained because an exact dense
    support map can have quadratic cardinality.
    """
    if num_images < 2 or num_points == 0 or not incidence_rows:
        return {}

    incidence = csr_matrix(
        (
            np.ones(len(incidence_rows), dtype=np.int32),
            (
                np.asarray(incidence_rows, dtype=np.int64),
                np.asarray(incidence_columns, dtype=np.int64),
            ),
        ),
        shape=(num_images, num_points),
        dtype=np.int32,
    )
    block_rows = max(1, min(1024, 8_000_000 // max(num_images, 1)))
    pair_counts: dict[tuple[int, int], int] = {}
    for start in range(0, num_images, block_rows):
        stop = min(num_images, start + block_rows)
        product = (incidence[start:stop] @ incidence.T).tocoo()
        global_rows = product.row.astype(np.int64, copy=False) + start
        supports = product.data.astype(np.int64, copy=False)
        mask = (global_rows < product.col) & (supports >= min_shared_points)
        for row, column, support in zip(
            global_rows[mask],
            product.col[mask],
            supports[mask],
            strict=True,
        ):
            pair_counts[(int(row), int(column))] = int(support)
    return pair_counts


def connected_components(adjacency: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    seen: set[int] = set()
    components: list[tuple[int, ...]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            u = stack.pop()
            component.append(u)
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(tuple(sorted(component)))
    return tuple(components)


def nearest_neighbor_distances(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(xyz) < 2:
        return np.asarray([], dtype=float)
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xyz).query(xyz, k=2)
    return np.asarray(distances[:, 1], dtype=float)
