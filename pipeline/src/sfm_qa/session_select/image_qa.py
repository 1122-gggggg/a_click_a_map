"""Intra-video image QA: Laplacian sharpness, exposure, near-duplicates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime
    cv2 = None


def _null_result(*reasons: str) -> dict[str, Any]:
    return {
        "sharpness_median": None,
        "sharpness_p10": None,
        "underexposed_ratio": None,
        "overexposed_ratio": None,
        "near_duplicate_ratio": None,
        "sampled": 0,
        "exposure_mean": None,
        "reasons": tuple(reasons),
    }


def laplacian_variance(image: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for Laplacian sharpness")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(image: np.ndarray) -> float:
    if cv2 is not None and image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    return float(image.mean())


def histogram_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for histogram correlation")
    if first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    if second.ndim == 3:
        second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    hist_a = cv2.calcHist([first], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([second], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _even_indexes(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if limit <= 0 or count <= limit:
        return list(range(count))
    raw = np.linspace(0, count - 1, num=limit)
    return list(dict.fromkeys(int(round(value)) for value in raw.tolist()))


def _sample_video_frames(path: Path, sample_limit: int) -> list[np.ndarray]:
    if cv2 is None:
        return []
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if reported > 0 and sample_limit > 0:
            targets = set(_even_indexes(reported, sample_limit))
            index = 0
            max_index = max(targets)
            while index <= max_index:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in targets and frame is not None:
                    frames.append(frame)
                index += 1
        else:
            decoded: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is not None:
                    decoded.append(frame)
            for index in _even_indexes(len(decoded), sample_limit):
                frames.append(decoded[index])
    finally:
        capture.release()
    return frames


def evaluate_video(
    path: str | Path,
    sample_limit: int = 24,
    *,
    underexposure_mean: float = 20.0,
    overexposure_mean: float = 235.0,
    near_duplicate_hist_corr: float = 0.995,
    blur_variance_reject: float = 25.0,
) -> dict[str, Any]:
    """Sample frames from a video with explicitly supplied heuristic gates.

    The caller is responsible for passing the values from ``defaults.yaml`` so
    the documented configuration and the executed QA cannot drift apart.
    """

    video = Path(path)
    if cv2 is None:
        return _null_result("missing_opencv")
    if not video.is_file():
        return _null_result("missing_video")

    limit = int(sample_limit) if sample_limit else 24
    under = float(underexposure_mean)
    over = float(overexposure_mean)
    dup_thr = float(near_duplicate_hist_corr)
    blur_thr = float(blur_variance_reject)
    frames = _sample_video_frames(video, limit)
    if not frames:
        return _null_result("unreadable_video")

    sharp: list[float] = []
    means: list[float] = []
    for frame in frames:
        try:
            sharp.append(laplacian_variance(frame))
            means.append(brightness_mean(frame))
        except Exception:
            continue
    if not sharp:
        return _null_result("unreadable_video")

    dups = 0
    comparisons = 0
    for prev, curr in zip(frames, frames[1:]):
        comparisons += 1
        try:
            if histogram_correlation(prev, curr) >= dup_thr:
                dups += 1
        except Exception:
            continue
    array = np.asarray(sharp, dtype=float)
    mean_array = np.asarray(means, dtype=float)
    reasons: list[str] = []
    p10 = float(np.percentile(array, 10))
    if p10 < blur_thr:
        reasons.append(f"low_sharpness_p10_heuristic_{blur_thr:g}")
    return {
        "sharpness_median": float(np.median(array)),
        "sharpness_p10": p10,
        "underexposed_ratio": float(np.mean(mean_array < under)),
        "overexposed_ratio": float(np.mean(mean_array > over)),
        "near_duplicate_ratio": float(dups / comparisons) if comparisons else 0.0,
        "sampled": len(sharp),
        "exposure_mean": float(np.mean(mean_array)),
        "reasons": tuple(reasons),
        "thresholds": {
            "underexposure_mean": under,
            "overexposure_mean": over,
            "near_duplicate_hist_corr": dup_thr,
            "blur_variance_reject": blur_thr,
        },
        "threshold_provenance": "heuristic_config",
    }


__all__ = [
    "brightness_mean",
    "evaluate_video",
    "histogram_correlation",
    "laplacian_variance",
]
