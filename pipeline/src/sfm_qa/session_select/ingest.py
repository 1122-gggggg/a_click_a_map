"""Discover one session per video file. Session id is the file stem."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = {".mp4", ".mov"}


try:
    from sfm_qa.session_select.types import SessionRecord as _ImportedSessionRecord
except ImportError:  # sibling types may land later
    _ImportedSessionRecord = None  # type: ignore[misc, assignment]


@dataclass
class _LocalSessionRecord:
    """Fallback record when types.SessionRecord is not importable."""

    session_id: str
    video_path: str
    sha256: str
    timestamp: str | None
    duration_seconds: float | None
    num_frames: int
    width: int | None = None
    height: int | None = None
    keyframes: tuple[dict, ...] = ()
    image_dirs: tuple[str, ...] = ()
    map_sources: tuple[dict, ...] = ()
    motion_rows: tuple[dict, ...] = ()


def _record_cls() -> type:
    return _ImportedSessionRecord or _LocalSessionRecord


def _construct(cls: type, **kwargs: Any) -> Any:
    if is_dataclass(cls):
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in kwargs.items() if key in allowed})
    return cls(**kwargs)


def session_id_from_name(name: str) -> str:
    """Map a path or filename to a site-agnostic session id (file stem)."""

    text = str(name).replace("\\", "/")
    return Path(text.split("/")[-1]).stem


def file_sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def light_video_probe(path: Path) -> dict[str, Any]:
    """Duration / size / creation_time without dumping every PTS."""

    empty = {
        "width": None,
        "height": None,
        "avg_frame_rate": "",
        "frame_count": 0,
        "duration_seconds": None,
        "creation_time": None,
    }
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_frames:format=duration:format_tags=creation_time",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return empty
    if completed.returncode:
        return empty
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return empty
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    tags = fmt.get("tags") or {}
    duration = fmt.get("duration")
    nb_frames = stream.get("nb_frames")
    try:
        frame_count = int(nb_frames) if nb_frames not in (None, "N/A") else 0
    except (TypeError, ValueError):
        frame_count = 0
    rate = str(stream.get("avg_frame_rate") or "")
    if frame_count <= 0 and duration not in (None, "N/A") and rate and rate not in {"0/0", "N/A"}:
        try:
            if "/" in rate:
                num, den = rate.split("/", 1)
                fps = float(num) / float(den) if float(den) else 0.0
            else:
                fps = float(rate)
            if fps > 0:
                frame_count = int(round(float(duration) * fps))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return {
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "avg_frame_rate": rate,
        "frame_count": frame_count,
        "duration_seconds": float(duration) if duration not in (None, "N/A") else None,
        "creation_time": tags.get("creation_time"),
    }


def _iter_videos(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in VIDEO_SUFFIXES:
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(root)
    found: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES:
            found.append(child)
    seen: set[str] = set()
    unique: list[Path] = []
    for video in found:
        key = video.name.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(video)
    return unique


def discover_sessions(
    video_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> list[Any]:
    """One session per ``.mp4`` / ``.mov`` file. ``session_id`` is the stem."""

    del config
    root = Path(video_dir)
    cls = _record_cls()
    records = []
    seen_ids: set[str] = set()
    for video in _iter_videos(root):
        sid = session_id_from_name(video.name)
        if not sid or sid in seen_ids:
            continue
        seen_ids.add(sid)
        probe = light_video_probe(video)
        try:
            digest = file_sha256(video)
        except OSError:
            digest = ""
        records.append(
            _construct(
                cls,
                session_id=sid,
                video_path=str(video),
                sha256=digest,
                timestamp=probe.get("creation_time"),
                duration_seconds=probe.get("duration_seconds"),
                num_frames=int(probe.get("frame_count") or 0),
                width=probe.get("width"),
                height=probe.get("height"),
                keyframes=(),
                image_dirs=(),
                map_sources=(),
                motion_rows=(),
            )
        )
    return records


__all__ = [
    "VIDEO_SUFFIXES",
    "discover_sessions",
    "file_sha256",
    "light_video_probe",
    "session_id_from_name",
]
