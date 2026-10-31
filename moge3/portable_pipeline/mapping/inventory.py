"""Video/image-source inventory for the portable mapping stage.

Discovers mapping sources (video files or image-sequence directories),
probes their resolution/fps with ffprobe (falling back to OpenCV), and
fingerprints each source with SHA-256. No site-specific names, no holdout
logic: every discovered source is a mapping input.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_utils import sha256_file


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    kind: str  # "video" | "image_dir"
    width: int
    height: int
    fps: float
    num_frames: int | None
    duration_seconds: float | None
    sha256: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "video_id": self.source_id,
            "source_kind": self.kind,
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "sha256": self.sha256,
        }


def discover_sources(
    inputs: list[str | Path],
    *,
    ffprobe: str = "ffprobe",
) -> list[SourceSpec]:
    """Discover mapping sources from files and directories.

    Each input may be a video file, an image-sequence directory, or a
    directory holding videos and/or image-sequence subdirectories.
    """
    candidates: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"mapping input does not exist: {raw}")
        if path.is_file():
            candidates.append(path)
        else:
            children = sorted(p for p in path.iterdir() if not p.name.startswith("."))
            videos = [p for p in children if p.is_file()]
            dirs = [p for p in children if p.is_dir()]
            if videos and _looks_like_images(path):
                # A bare directory of images is one image-sequence source.
                candidates.append(path)
            else:
                candidates.extend(videos)
                candidates.extend(d for d in dirs if _looks_like_images(d))
                for child in dirs:
                    if not _looks_like_images(child):
                        nested = [
                            p
                            for p in sorted(child.iterdir())
                            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                        ]
                        candidates.extend(nested)
    seen: set[str] = set()
    specs: list[SourceSpec] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        spec = _probe_source(resolved, ffprobe=ffprobe)
        if spec is not None:
            specs.append(spec)
    if not specs:
        raise ValueError("no mapping sources found (videos or image directories)")
    return sorted(specs, key=lambda spec: spec.source_id)


def _looks_like_images(path: Path) -> bool:
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    images = [
        p for p in children if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return len(images) >= 2


def _source_id(path: Path) -> str:
    stem = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in path.stem
    ).strip("._-")
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
    stem = stem or "source"
    return f"vid_{stem}_{digest}"


def _probe_source(path: Path, *, ffprobe: str) -> SourceSpec | None:
    if path.is_dir():
        images = sorted(
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            return None
        width, height = _probe_image_size(images[0])
        fingerprint = _manifest_fingerprint(
            [(p.name, p.stat().st_size, sha256_file(p)) for p in images]
        )
        return SourceSpec(
            source_id=_source_id(path),
            path=path,
            kind="image_dir",
            width=width,
            height=height,
            fps=0.0,
            num_frames=len(images),
            duration_seconds=None,
            sha256=fingerprint,
        )
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        width, height = _probe_image_size(path)
        return SourceSpec(
            source_id=_source_id(path),
            path=path,
            kind="image_dir",
            width=width,
            height=height,
            fps=0.0,
            num_frames=1,
            duration_seconds=None,
            sha256=sha256_file(path),
        )
    if suffix not in VIDEO_SUFFIXES:
        return None
    info = _ffprobe_stream(path, ffprobe=ffprobe)
    return SourceSpec(
        source_id=_source_id(path),
        path=path,
        kind="video",
        width=int(info["width"]),
        height=int(info["height"]),
        fps=float(info["fps"]),
        num_frames=info["num_frames"],
        duration_seconds=info["duration_seconds"],
        sha256=sha256_file(path),
    )


def _probe_image_size(path: Path) -> tuple[int, int]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv is required to probe image sources") from error
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return int(image.shape[1]), int(image.shape[0])


def _ffprobe_stream(path: Path, *, ffprobe: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,"
                "nb_frames,duration",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout or "{}")
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                fps = _parse_rate(
                    stream.get("avg_frame_rate") or stream.get("r_frame_rate")
                )
                num_frames = _parse_int(stream.get("nb_frames"))
                duration = _parse_float(
                    stream.get("duration")
                    or (payload.get("format") or {}).get("duration")
                )
                if min(width, height) > 0 and fps > 0:
                    return {
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "num_frames": num_frames,
                        "duration_seconds": duration,
                    }
        except (ValueError, KeyError, TypeError):
            pass
    return _opencv_stream(path)


def _opencv_stream(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "ffprobe failed and opencv is unavailable to probe video"
        ) from error
    capture = cv2.VideoCapture(str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if min(width, height) <= 0 or fps <= 0:
        raise ValueError(f"cannot probe video stream: {path}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "num_frames": count or None,
        "duration_seconds": (count / fps) if count else None,
    }


def _parse_rate(value: Any) -> float:
    try:
        text = str(value).strip()
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)
    except (ValueError, TypeError):
        return 0.0


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return None


def _parse_float(value: Any) -> float | None:
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return None


def _manifest_fingerprint(rows: list[tuple[str, int, str]]) -> str:
    encoded = json.dumps(sorted(rows), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IMAGE_SUFFIXES",
    "SourceSpec",
    "VIDEO_SUFFIXES",
    "discover_sources",
]
