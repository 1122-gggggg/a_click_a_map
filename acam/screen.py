"""Video-level quality screening (engine environment: numpy + OpenCV).

    python -m acam.screen VIDEO [VIDEO ...] [--out screening.json] [--workers 4]

The mapping engine already discards individual garbage frames (corrupt, severe blur, fully
black/white, duplicates). This stage answers a different question *before* the expensive
run: is the whole video worth feeding in? It samples frames evenly across each video
(never decodes the full stream), measures sharpness, exposure and texture, and returns
OK / WARN / REJECT with reasons. REJECT videos are excluded by default; WARN ones are kept
and listed in the report.

Thresholds are conservative on purpose (they target videos that cannot map, not
mediocre ones) and were checked against the Wulai/River corpora; see `tests/test_screen.py`.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

SAMPLES = 48            # frames sampled across the video
WIDTH = 640             # analysis width: metrics are comparable across source resolutions
MIN_DURATION_S = 5.0
MIN_SHORT_SIDE = 480
MIN_FPS = 8.0
LAP_BAD = 20.0          # Laplacian variance at 640 px wide below this = unusable blur
LUMA_DARK, LUMA_BRIGHT = 25.0, 235.0
FAST_BAD = 150          # FAST corners at 640 px below this = nothing to match (sky, water, blank wall)
REJECT_BAD_FRACTION = 0.60
WARN_BAD_FRACTION = 0.25
STATIC_DIFF = 0.8       # median mean-abs-diff (0-255) between frames 0.5 s apart


@dataclass
class FrameStats:
    lap_var: float
    luma: float
    dark_fraction: float
    saturated_fraction: float
    corners: int

    @property
    def bad_reasons(self) -> list[str]:
        reasons = []
        if self.lap_var < LAP_BAD:
            reasons.append("blur")
        if self.luma < LUMA_DARK or self.dark_fraction > 0.7:
            reasons.append("dark")
        if self.luma > LUMA_BRIGHT or self.saturated_fraction > 0.7:
            reasons.append("overexposed")
        if self.corners < FAST_BAD:
            reasons.append("low_texture")
        return reasons


@dataclass
class Verdict:
    path: str
    verdict: str                      # OK | WARN | REJECT
    reasons: list[str] = field(default_factory=list)
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0
    frames_sampled: int = 0
    unreadable_samples: int = 0
    bad_fraction: float = 0.0
    bad_breakdown: dict[str, float] = field(default_factory=dict)
    median_lap_var: float = 0.0
    median_corners: float = 0.0
    median_luma: float = 0.0
    static_motion: float | None = None
    score: float = 0.0                # 0..1, higher = better; used to prioritise when memory forces a subset


def frame_stats(gray) -> FrameStats:
    import cv2
    import numpy as np

    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    corners = len(cv2.FastFeatureDetector_create(threshold=20).detect(gray))
    return FrameStats(
        lap_var=lap,
        luma=float(gray.mean()),
        dark_fraction=float(np.mean(gray < 16)),
        saturated_fraction=float(np.mean(gray > 240)),
        corners=int(corners),
    )


def _resize_gray(frame):
    import cv2

    height, width = frame.shape[:2]
    scale = WIDTH / float(width)
    frame = cv2.resize(frame, (WIDTH, max(int(round(height * scale)), 1)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def judge(stats: list[FrameStats], *, unreadable: int, width: int, height: int, fps: float, duration: float,
          static_motion: float | None, path: str = "") -> Verdict:
    """Pure decision function (unit-tested): turn measurements into a verdict."""

    verdict = Verdict(path=path, verdict="OK", width=width, height=height, fps=fps, duration=duration,
                      frames_sampled=len(stats), unreadable_samples=unreadable, static_motion=static_motion)
    reject: list[str] = []
    warn: list[str] = []
    if min(width, height) < MIN_SHORT_SIDE:
        reject.append(f"resolution {width}x{height} below {MIN_SHORT_SIDE}p")
    if duration < MIN_DURATION_S:
        reject.append(f"duration {duration:.1f}s below {MIN_DURATION_S:.0f}s")
    total = len(stats) + unreadable
    if total == 0 or len(stats) == 0:
        reject.append("no decodable frames")
    elif unreadable / total > 0.25:
        reject.append(f"{unreadable}/{total} sampled positions unreadable (corrupt or truncated)")
    elif unreadable / total > 0.10:  # one or two failed seeks near the end of a cut file are normal
        warn.append(f"{unreadable}/{total} sampled positions unreadable")
    if fps and fps < MIN_FPS:
        warn.append(f"low frame rate {fps:.1f} fps")
    if stats:
        counts = {"blur": 0, "dark": 0, "overexposed": 0, "low_texture": 0}
        bad = 0
        for item in stats:
            reasons = item.bad_reasons
            bad += bool(reasons)
            for reason in reasons:
                counts[reason] += 1
        verdict.bad_fraction = bad / len(stats)
        verdict.bad_breakdown = {key: round(value / len(stats), 3) for key, value in counts.items() if value}
        verdict.median_lap_var = statistics.median(s.lap_var for s in stats)
        verdict.median_corners = statistics.median(s.corners for s in stats)
        verdict.median_luma = statistics.median(s.luma for s in stats)
        detail = ", ".join(f"{k} {v:.0%}" for k, v in verdict.bad_breakdown.items())
        if verdict.bad_fraction >= REJECT_BAD_FRACTION:
            reject.append(f"{verdict.bad_fraction:.0%} of sampled frames unusable ({detail})")
        elif verdict.bad_fraction >= WARN_BAD_FRACTION:
            warn.append(f"{verdict.bad_fraction:.0%} of sampled frames poor ({detail})")
    if static_motion is not None and static_motion < STATIC_DIFF:
        warn.append("camera looks static: little parallax, few keyframes will be kept")
    verdict.reasons = reject + warn
    verdict.verdict = "REJECT" if reject else ("WARN" if warn else "OK")
    verdict.score = 0.0 if reject else round(max(0.0, 1.0 - verdict.bad_fraction) * (0.6 if static_motion is not None and static_motion < STATIC_DIFF else 1.0), 3)
    return verdict


def screen_video(path: str) -> dict:
    import cv2
    import numpy as np

    cv2.setNumThreads(1)  # screening runs one process per video; avoid oversubscribing the CPU
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return asdict(Verdict(path=path, verdict="REJECT", reasons=["cannot open video"]))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps > 0 and count > 0 else 0.0
    stats: list[FrameStats] = []
    unreadable = 0
    diffs: list[float] = []
    positions = np.linspace(0, max(count - 2, 0), SAMPLES + 2)[1:-1].astype(int) if count > 0 else []
    gap = max(int(round(fps * 0.5)), 1) if fps > 0 else 12
    for index, position in enumerate(positions):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
        ok, frame = capture.read()
        if not ok or frame is None:
            unreadable += 1
            continue
        gray = _resize_gray(frame)
        stats.append(frame_stats(gray))
        if index % 8 == 0:  # camera-motion probe: second frame 0.5 s later
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position) + gap)
            ok2, later = capture.read()
            if ok2 and later is not None:
                diffs.append(float(np.mean(cv2.absdiff(gray, _resize_gray(later)))))
    capture.release()
    motion = statistics.median(diffs) if diffs else None
    return asdict(judge(stats, unreadable=unreadable, width=width, height=height, fps=fps, duration=duration,
                        static_motion=motion, path=path))


def screen_all(paths: list[str], workers: int = 4) -> list[dict]:
    if workers <= 1 or len(paths) == 1:
        return [screen_video(p) for p in paths]
    with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
        return list(pool.map(screen_video, paths))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    results = screen_all(args.videos, args.workers)
    text = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
