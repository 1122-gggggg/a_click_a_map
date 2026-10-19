"""Pure contracts shared by the production-EDM reference-LOO runner."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rank_reference_indices(
    reference_descriptors: np.ndarray,
    *,
    query_index: int,
    excluded_indices: Sequence[int],
    topk: int,
) -> tuple[list[int], list[float]]:
    """Rank cosine-equivalent bundled descriptors after applying LOO exclusions."""

    descriptors = np.asarray(reference_descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or not 0 <= query_index < len(descriptors):
        raise ValueError("invalid descriptor matrix or query index")
    if topk <= 0:
        raise ValueError("topk must be positive")
    excluded = {int(index) for index in excluded_indices}
    if any(index < 0 or index >= len(descriptors) for index in excluded):
        raise ValueError("excluded index is outside descriptor rows")
    available = len(descriptors) - len(excluded)
    if topk > available:
        raise ValueError(f"topk={topk} exceeds {available} available references")

    similarities = descriptors @ descriptors[query_index]
    if excluded:
        similarities[np.asarray(sorted(excluded), dtype=int)] = -np.inf
    order = np.argsort(-similarities, kind="stable")[:topk]
    return order.astype(int).tolist(), [float(similarities[index]) for index in order]


def spatial_cap_indices(
    points2d: np.ndarray,
    *,
    max_total: int,
    width: int,
    height: int,
    grid: int = 8,
) -> np.ndarray:
    """Return the deterministic indices used by production EDM's spatial cap."""

    points = np.asarray(points2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points2d must have shape (N, 2)")
    if max_total <= 0 or width <= 0 or height <= 0 or grid <= 0:
        raise ValueError("spatial cap dimensions must be positive")
    if len(points) <= max_total:
        return np.arange(len(points), dtype=np.int64)

    grid_x = np.clip((points[:, 0] / width * grid).astype(int), 0, grid - 1)
    grid_y = np.clip((points[:, 1] / height * grid).astype(int), 0, grid - 1)
    cells = grid_y * grid + grid_x
    per_cell = max(1, max_total // (grid * grid))
    generator = np.random.default_rng(0)
    kept: list[np.ndarray] = []
    for cell in np.unique(cells):
        indices = np.flatnonzero(cells == cell)
        if len(indices) > per_cell:
            indices = generator.choice(indices, per_cell, replace=False)
        kept.append(np.asarray(indices, dtype=np.int64))
    selected = np.concatenate(kept)
    if len(selected) > max_total:
        selected = generator.choice(selected, max_total, replace=False)
    return np.asarray(selected, dtype=np.int64)
