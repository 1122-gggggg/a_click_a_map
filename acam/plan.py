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

from dataclasses import dataclass, field

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


# --------------------------------------------------------------------------- corpus-size budget
KF_PER_SECOND = 1.15      # measured keyframes per second of video: 1.21 (all6), 1.10 (all9), 1.16 (smoke)
MIN_SCALE = 0.5           # never thin the sampling below half of the validated density
SECONDS_PER_KEYFRAME = 6.5  # wall clock on an RTX 5090 (all9: 13.5 ks / 2043; smoke: 6 s). Other GPUs differ.


@dataclass(frozen=True)
class VideoBudget:
    name: str
    duration: float
    score: float = 1.0     # screening score, higher = more valuable when a subset must be chosen


@dataclass(frozen=True)
class BudgetPlan:
    keep: list[str]
    dropped: list[tuple[str, str]]
    sampling_scale: float
    tracks: int
    est_keyframes: int
    est_points: float
    fits: bool
    notes: list[str] = field(default_factory=list)


def keyframe_cap(mem_total: int, max_keyframes: int | None = None) -> int:
    """Most keyframes the validated setting (K=512) is expected to hold on this host."""

    cap = int(point_budget(mem_total) / points_per_image(512))
    return min(cap, max_keyframes) if max_keyframes else cap


def estimate_keyframes(videos: list[VideoBudget], scale: float = 1.0) -> float:
    return sum(v.duration for v in videos) * KF_PER_SECOND * scale


def plan_budget(videos: list[VideoBudget], mem_total: int, *, max_keyframes: int | None = None) -> BudgetPlan:
    """Fit the corpus to this machine, in this order of preference:

    1. everything at full sampling, K chosen from the budget (K >= 512, the validated region);
    2. thin the sampling (keyframe spacing) down to MIN_SCALE with K=512;
    3. drop the lowest-scoring videos until MIN_SCALE fits.

    K below 512 is never chosen here: it is untested at this scale. `choose_tracks` can still
    reach it through the OOM retry loop.
    """

    cap = keyframe_cap(mem_total, max_keyframes)
    notes: list[str] = []
    total = estimate_keyframes(videos)
    keep, dropped, scale = list(videos), [], 1.0
    if total <= cap * 1.05:  # the estimate is good to a few percent; K=512 is validated up to the cap
        tracks = max(512, choose_tracks(int(total), mem_total)[0])
    else:
        tracks = 512
        scale = min(cap / total, 1.0)
        if scale >= MIN_SCALE:
            notes.append(f"keyframe sampling thinned to {scale:.2f}x to fit the memory budget")
        else:
            scale = MIN_SCALE
            ranked = sorted(videos, key=lambda v: (-v.score, -v.duration, v.name))
            keep = []
            for video in ranked:
                if estimate_keyframes(keep + [video], scale) <= cap:
                    keep.append(video)
                else:
                    dropped.append((video.name, "memory budget"))
            if not keep:  # even one video is too long: thin that one further instead of refusing
                keep = ranked[:1]
                dropped = [(v.name, "memory budget") for v in ranked[1:]]
                scale = cap / estimate_keyframes(keep)
                notes.append("a single video exceeds the budget; sampling thinned below the validated density")
            order = {v.name: i for i, v in enumerate(videos)}
            keep.sort(key=lambda v: order[v.name])
            notes.append(f"{len(dropped)} video(s) dropped (lowest screening score first) and sampling at {scale:.2f}x")
    estimate = estimate_keyframes(keep, scale)
    points = estimate * points_per_image(tracks)
    return BudgetPlan([v.name for v in keep], dropped, round(scale, 3), tracks, int(estimate), points,
                      points <= point_budget(mem_total) * 1.05, notes)


def rescale_after_preprocess(actual_keyframes: int, current_scale: float, mem_total: int,
                             max_keyframes: int | None = None, tolerance: float = 1.05) -> float | None:
    """None if the real keyframe count fits; otherwise the sampling scale to retry with."""

    cap = keyframe_cap(mem_total, max_keyframes)
    if actual_keyframes <= cap * tolerance:
        return None
    return round(current_scale * cap / actual_keyframes * 0.95, 3)


def eta_hours(keyframes: int) -> float:
    return keyframes * SECONDS_PER_KEYFRAME / 3600.0


def is_cuda_oom(log_tail: str) -> bool:
    return "CUDA out of memory" in log_tail or "cudaErrorMemoryAllocation" in log_tail
