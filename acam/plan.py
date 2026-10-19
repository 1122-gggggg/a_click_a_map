"""Choose `num_track_per_img` from the corpus size and host RAM, and classify OOM exits.

The only measured data behind the budget (RTX 5090, 30 GiB RAM, MemoryMax 29 GiB):

    images  num_track_per_img  GlueMap track points   outcome
    1389    1024               22.36 M                completed (all6)
    2043    1024               32.74 M                OOM-killed in track establishment at 21.3 GiB
    2043     512               15.04 M                completed, cgroup peak 25.2 GiB (BA)

Track points per image is ~16.1k at K=1024 and ~7.4k at K=512. The budget below is the
largest point count that is *known* to have finished on 30 GiB, scaled by (RAM-6)/24 GiB.
It is a heuristic from those three runs, not a model: the retry loop in `runner.py` is the
safety net.
"""
from __future__ import annotations

from .hw import GIB

POINTS_PER_IMAGE_K1024 = 16_050.0
POINTS_PER_IMAGE_K512 = 7_361.0
SAFE_POINTS_AT_30_GIB = 15_040_000.0
MIN_TRACKS = 256
MAX_TRACKS = 1024
STEP = 64


def points_per_image(tracks: int) -> float:
    """Interpolate between the two measured points; below 512 scale proportionally (conservative)."""

    if tracks >= 512:
        slope = (POINTS_PER_IMAGE_K1024 - POINTS_PER_IMAGE_K512) / 512.0
        return POINTS_PER_IMAGE_K512 + slope * (tracks - 512)
    return POINTS_PER_IMAGE_K512 * tracks / 512.0


def point_budget(mem_total: int) -> float:
    return SAFE_POINTS_AT_30_GIB * max(mem_total / GIB - 6.0, 1.0) / 24.0


def choose_tracks(n_images: int, mem_total: int, *, max_tracks: int = MAX_TRACKS) -> tuple[int, float, bool]:
    """Return (num_track_per_img, estimated track points, fits_budget)."""

    budget = point_budget(mem_total)
    best = MIN_TRACKS
    for tracks in range(max_tracks, MIN_TRACKS - 1, -STEP):
        if n_images * points_per_image(tracks) <= budget:
            best = tracks
            break
    estimate = n_images * points_per_image(best)
    return best, estimate, estimate <= budget


def reduce_tracks(tracks: int) -> int | None:
    lower = (int(tracks * 0.75) // STEP) * STEP
    lower = max(lower, MIN_TRACKS)
    return lower if lower < tracks else None


def memory_limits(mem_total: int, swap_free: int) -> tuple[int, int]:
    """MemoryMax / MemorySwapMax for the mapping scope: leave 1.5 GiB to the rest of the host."""

    return int(mem_total - 1.5 * GIB), int(min(swap_free, 6 * GIB))


def is_oom_exit(returncode: int, log_tail: str) -> bool:
    """SIGKILL (kernel/cgroup OOM killer) or an allocator failure reported in the log."""

    if returncode in (-9, 137):
        return True
    needles = ("std::bad_alloc", "MemoryError", "out of memory", "Out of memory")
    return returncode != 0 and any(needle in log_tail for needle in needles)
