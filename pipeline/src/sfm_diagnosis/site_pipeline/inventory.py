from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Mapping

from sfm_diagnosis.io import write_json

from .config import PipelineConfig


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ARCHIVE_SUFFIXES = {".zip"}

METADATA_COLUMNS = (
    "source_id",
    "source_kind",
    "path",
    "evaluation_role",
    "video_id",
    "session_id",
    "capture_date",
    "width",
    "height",
    "fps",
    "camera_mode",
    "crop_state",
    "stabilization_state",
    "intrinsics_group",
    "route",
    "direction",
    "height_m",
    "pitch_deg",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(rows: Iterable[tuple[str, int, str]]) -> str:
    encoded = json.dumps(sorted(rows), separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def image_sequence_fingerprint(path: Path, *, hash_contents: bool) -> tuple[str, int]:
    images = sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )
    rows = [
        (
            child.name,
            child.stat().st_size,
            sha256_file(child) if hash_contents else str(child.stat().st_mtime_ns),
        )
        for child in images
    ]
    return _manifest_fingerprint(rows), len(rows)


def archive_image_fingerprint(path: Path, *, hash_contents: bool) -> tuple[str | None, int]:
    if path.suffix.lower() != ".zip":
        return None, 0
    rows: list[tuple[str, int, str]] = []
    try:
        with zipfile.ZipFile(path) as bundle:
            members = sorted(
                (
                    member
                    for member in bundle.infolist()
                    if not member.is_dir()
                    and Path(member.filename).suffix.lower() in IMAGE_SUFFIXES
                ),
                key=lambda member: member.filename,
            )
            for member in members:
                digest = (
                    hashlib.sha256(bundle.read(member)).hexdigest()
                    if hash_contents
                    else str(member.CRC)
                )
                rows.append((Path(member.filename).name, member.file_size, digest))
    except (OSError, zipfile.BadZipFile):
        return None, 0
    return (_manifest_fingerprint(rows), len(rows)) if rows else (None, 0)


def ffprobe_media(path: Path, executable: str = "ffprobe") -> dict[str, object]:
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames:stream_tags=creation_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return {"probe_status": "FFPROBE_UNAVAILABLE"}
    if completed.returncode != 0:
        return {"probe_status": "UNREADABLE", "probe_error": completed.stderr.strip()}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"probe_status": "INVALID_OUTPUT"}
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    rate = str(stream.get("avg_frame_rate") or "")
    fps = None
    if "/" in rate:
        left, right = rate.split("/", 1)
        try:
            fps = float(left) / float(right) if float(right) else None
        except ValueError:
            fps = None
    return {
        "probe_status": "OK",
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "num_frames": _int_or_none(stream.get("nb_frames")),
        "duration_seconds": _float_or_none(fmt.get("duration")),
        "capture_date": (stream.get("tags") or {}).get("creation_time"),
    }


def _float_or_none(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _matches_holdout(path: Path, root: Path, patterns: tuple[str, ...]) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(relative, pattern)
        for pattern in patterns
    )


def discover_corpus(
    root: str | Path,
    config: PipelineConfig,
    *,
    probe: Callable[[Path], Mapping[str, object]] = ffprobe_media,
) -> dict[str, object]:
    corpus = Path(root).expanduser().resolve()
    if not corpus.is_dir():
        raise FileNotFoundError(corpus)
    rows: list[dict[str, object]] = []
    sequence_fingerprints: dict[str, str] = {}

    videos = sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    for path in videos:
        _assert_safe_source(path, corpus)
        content_sha = sha256_file(path) if config.hash_contents else _stat_fingerprint(path)
        metadata = dict(probe(path))
        source_id = f"vid_{content_sha[:16]}"
        rows.append(
            {
                "source_id": source_id,
                "video_id": source_id,
                "source_kind": "video",
                "path": str(path),
                "relative_path": path.relative_to(corpus).as_posix(),
                "size": path.stat().st_size,
                "sha256": content_sha if config.hash_contents else None,
                "evaluation_role": (
                    "HOLDOUT"
                    if _matches_holdout(path, corpus, config.heldout_patterns)
                    else "MAPPING"
                ),
                "holdout_provenance": (
                    config.holdout_provenance
                    if _matches_holdout(path, corpus, config.heldout_patterns)
                    else None
                ),
                "duplicate_of": None,
                **metadata,
            }
        )

    image_parents = sorted(
        {
            path.parent
            for path in corpus.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
    )
    for path in image_parents:
        _assert_safe_source(path, corpus, allow_directory=True)
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES:
                _assert_safe_source(child, corpus)
        fingerprint, count = image_sequence_fingerprint(path, hash_contents=config.hash_contents)
        source_id = f"seq_{fingerprint[:16]}"
        sequence_fingerprints[fingerprint] = source_id
        sample = next(
            child for child in sorted(path.iterdir()) if child.suffix.lower() in IMAGE_SUFFIXES
        )
        metadata = dict(probe(sample))
        rows.append(
            {
                "source_id": source_id,
                "video_id": source_id,
                "source_kind": "image_sequence",
                "path": str(path),
                "relative_path": path.relative_to(corpus).as_posix(),
                "size": sum(
                    child.stat().st_size
                    for child in path.iterdir()
                    if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
                ),
                "sha256": fingerprint,
                "file_count": count,
                "evaluation_role": (
                    "HOLDOUT"
                    if _matches_holdout(path, corpus, config.heldout_patterns)
                    else "MAPPING"
                ),
                "holdout_provenance": (
                    config.holdout_provenance
                    if _matches_holdout(path, corpus, config.heldout_patterns)
                    else None
                ),
                "duplicate_of": None,
                **metadata,
            }
        )

    archives = sorted(
        path
        for path in corpus.rglob("*")
        if path.is_file() and path.suffix.lower() in ARCHIVE_SUFFIXES
    )
    for path in archives:
        _assert_safe_source(path, corpus)
        archive_sha = sha256_file(path) if config.hash_contents else _stat_fingerprint(path)
        content_fingerprint, count = archive_image_fingerprint(
            path, hash_contents=config.hash_contents
        )
        rows.append(
            {
                "source_id": f"arc_{archive_sha[:16]}",
                "video_id": "",
                "source_kind": "archive",
                "path": str(path),
                "relative_path": path.relative_to(corpus).as_posix(),
                "size": path.stat().st_size,
                "sha256": archive_sha if config.hash_contents else None,
                "file_count": count,
                "content_fingerprint": content_fingerprint,
                "evaluation_role": "ARCHIVE_ONLY",
                "duplicate_of": sequence_fingerprints.get(content_fingerprint or ""),
            }
        )

    if not any(row["source_kind"] != "archive" for row in rows):
        raise ValueError("corpus contains no videos or image sequences")
    return {
        "schema_version": 2,
        "artifact_type": "SITE_CORPUS_MANIFEST",
        "site_name": config.site_name,
        "corpus_root": str(corpus),
        "sources": rows,
    }


def _assert_safe_source(path: Path, root: Path, *, allow_directory: bool = False) -> None:
    if path.is_symlink():
        raise ValueError(f"corpus sources must not be symlinks: {path}")
    resolved = path.resolve()
    corpus = root.resolve()
    if resolved != corpus and corpus not in resolved.parents:
        raise ValueError(f"corpus source escapes root: {path}")
    if allow_directory and not path.is_dir():
        raise ValueError(f"expected image-sequence directory: {path}")


def _stat_fingerprint(path: Path) -> str:
    stat = path.stat()
    return hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()


def write_metadata_template(path: str | Path, sources: Iterable[Mapping[str, object]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "source_id": source.get("source_id", ""),
                    "source_kind": source.get("source_kind", ""),
                    "path": source.get("path", ""),
                    "evaluation_role": source.get("evaluation_role", ""),
                    "video_id": source.get("video_id", ""),
                    "capture_date": source.get("capture_date", ""),
                    "width": source.get("width", ""),
                    "height": source.get("height", ""),
                    "fps": source.get("fps", ""),
                }
            )
    return output


def write_corpus_manifest(path: str | Path, manifest: Mapping[str, object]) -> Path:
    output = Path(path)
    write_json(output, dict(manifest))
    return output


__all__ = [
    "METADATA_COLUMNS",
    "archive_image_fingerprint",
    "discover_corpus",
    "ffprobe_media",
    "image_sequence_fingerprint",
    "sha256_file",
    "write_corpus_manifest",
    "write_metadata_template",
]
