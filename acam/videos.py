"""Inventory the input videos and check them against the calibration's aspect ratio."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


@dataclass(frozen=True)
class Video:
    path: Path
    width: int
    height: int
    fps: float
    frames: int
    duration: float


def find_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def probe(path: Path) -> Video:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of", "json", str(path),
        ],
        text=True,
    )
    data = json.loads(raw)
    stream = data["streams"][0]
    num, _, den = str(stream.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    frames = int(stream.get("nb_frames") or round(fps * duration))
    return Video(path, int(stream["width"]), int(stream["height"]), fps, frames, duration)


def aspect_mismatches(videos: list[Video], calibration: dict) -> list[Video]:
    """Videos whose aspect ratio differs from the calibration (K can only be rescaled, not re-cropped)."""

    cw, ch = int(calibration["image_width"]), int(calibration["image_height"])
    return [v for v in videos if v.width * ch != v.height * cw]
