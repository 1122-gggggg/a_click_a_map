from __future__ import annotations

from typing import Sequence

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sample_multiview_tracks(
    *,
    source_name: str,
    target_names: Sequence[str],
    flow: np.ndarray,
    certainty_logits: np.ndarray,
    native_size: tuple[int, int],
    certainty_threshold: float,
    minimum_target_observations: int = 2,
    grid_rows: int = 27,
    grid_cols: int = 48,
    max_samples: int = 1024,
) -> list[dict[str, tuple[float, float]]]:
    warps = np.asarray(flow, dtype=np.float32)
    logits = np.asarray(certainty_logits, dtype=np.float32)
    if warps.ndim != 4 or warps.shape[1] != 2:
        raise ValueError("MV-RoMa flow must have shape (T,2,H,W)")
    if logits.shape != (warps.shape[0], warps.shape[2], warps.shape[3]):
        raise ValueError("MV-RoMa certainty shape does not match flow")
    if len(target_names) != warps.shape[0]:
        raise ValueError("MV-RoMa target name count does not match flow")
    if not 0.0 < certainty_threshold < 1.0:
        raise ValueError("certainty threshold must be in (0,1)")
    if minimum_target_observations < 2:
        raise ValueError("MV-RoMa tracks require at least two target observations")
    target_count, _, height, width = warps.shape
    if target_count < minimum_target_observations:
        return []
    certainty = _sigmoid(logits)
    required_score = np.partition(certainty, -minimum_target_observations, axis=0)[
        -minimum_target_observations
    ]
    candidates: list[tuple[float, int, int]] = []
    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)
    for row in range(grid_rows):
        for col in range(grid_cols):
            y0, y1 = row_edges[row], row_edges[row + 1]
            x0, x1 = col_edges[col], col_edges[col + 1]
            if y1 <= y0 or x1 <= x0:
                continue
            tile = required_score[y0:y1, x0:x1]
            offset = int(np.argmax(tile))
            local_y, local_x = np.unravel_index(offset, tile.shape)
            y, x = y0 + int(local_y), x0 + int(local_x)
            score = float(required_score[y, x])
            if score >= certainty_threshold:
                candidates.append((score, y, x))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    native_height, native_width = native_size
    tracks = []
    for _, y, x in candidates[:max_samples]:
        observations: dict[str, tuple[float, float]] = {
            source_name: (
                x / max(width - 1, 1) * (native_width - 1),
                y / max(height - 1, 1) * (native_height - 1),
            )
        }
        for target_index, target_name in enumerate(target_names):
            if certainty[target_index, y, x] < certainty_threshold:
                continue
            normalized_x = float(warps[target_index, 0, y, x])
            normalized_y = float(warps[target_index, 1, y, x])
            target_x = (normalized_x + 1.0) / 2.0 * (native_width - 1)
            target_y = (normalized_y + 1.0) / 2.0 * (native_height - 1)
            if not (0.0 <= target_x < native_width and 0.0 <= target_y < native_height):
                continue
            observations[str(target_name)] = (target_x, target_y)
        if len(observations) >= minimum_target_observations + 1:
            tracks.append(observations)
    return tracks


__all__ = ["sample_multiview_tracks"]
