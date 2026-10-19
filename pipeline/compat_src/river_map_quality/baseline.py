"""Create a content-addressed, physically independent M0 baseline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from river_map_quality.provenance import FileFingerprint, fingerprint_file


def _safe_destination(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe baseline destination: {value!r}")
    return Path(*path.parts)


def _source_files(source: Path) -> list[tuple[Path, Path]]:
    if source.is_file():
        return [(source, Path())]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return [
        (path, path.relative_to(source)) for path in sorted(source.rglob("*")) if path.is_file()
    ]


def _same_content(source: FileFingerprint, destination: FileFingerprint) -> bool:
    return source.size == destination.size and source.sha256 == destination.sha256


def freeze_baseline(target: Path, sources: Mapping[str, Path]) -> dict[str, object]:
    """Copy inputs into an atomic, read-only M0 directory and bind every byte by SHA-256.

    Destination keys are relative paths below ``target``. Directory sources copy their
    contents below that path; file sources copy to that exact path. Sources are hashed
    both before and after copying so concurrent writers cannot silently enter M0.
    """

    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"baseline target already exists: {target}")
    if not sources:
        raise ValueError("at least one baseline source is required")
    target.parent.mkdir(parents=True, exist_ok=True)

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    records: list[dict[str, object]] = []
    try:
        for destination_name, source_value in sorted(sources.items()):
            destination_root = _safe_destination(destination_name)
            source = Path(source_value).resolve(strict=True)
            source_is_file = source.is_file()
            for source_file, source_relative in _source_files(source):
                destination_relative = (
                    destination_root if source_is_file else destination_root / source_relative
                )
                before = fingerprint_file(source_file, sha256=True)
                destination = temp / destination_relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination, follow_symlinks=True)
                copied = fingerprint_file(destination, sha256=True)
                after = fingerprint_file(source_file, sha256=True)
                if before != after or not _same_content(before, copied):
                    raise RuntimeError(f"source changed while freezing baseline: {source_file}")
                records.append(
                    {
                        "baseline_path": destination_relative.as_posix(),
                        "source_path": before.path,
                        "size": before.size,
                        "source_mtime_ns": before.mtime_ns,
                        "sha256": before.sha256,
                    }
                )

        manifest: dict[str, object] = {
            "baseline_id": "M0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "copy_semantics": "independent byte copy; no symlinks or hardlinks",
            "files": records,
        }
        manifest_path = temp / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for file_path in sorted(temp.rglob("*")):
            if file_path.is_file():
                file_path.chmod(0o444)
        for directory in sorted(
            (path for path in temp.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temp.chmod(0o555)
        os.replace(temp, target)
        return manifest
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise


def verify_frozen_baseline(root: Path) -> int:
    """Re-hash a frozen baseline and reject missing, extra, linked, or changed files."""

    root = root.resolve(strict=True)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest.get("files", []))
    expected = {str(record["baseline_path"]) for record in records}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        raise RuntimeError(
            f"baseline file set mismatch: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )

    for record in records:
        relative = _safe_destination(str(record["baseline_path"]))
        path = root / relative
        if path.is_symlink():
            raise RuntimeError(f"baseline file must not be a symlink: {relative}")
        current = fingerprint_file(path, sha256=True)
        if current.size != int(record["size"]) or current.sha256 != record["sha256"]:
            raise RuntimeError(f"baseline content mismatch: {relative}")
    return len(records)
