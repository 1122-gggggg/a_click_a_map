"""Immutable corpus and artifact contracts for River map experiments.

The current base map, historical update corpus, and held-out validation corpus have
distinct roles.  This module owns the fail-closed boundary shared by every stage.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from river_map_quality.provenance import fingerprint_file

BASE_VIDEO_NAMES = ("P1180118.MP4", "P1190119.MP4", "P1200120.MP4")
UPDATE_VIDEO_NAMES = ("P1160116.MP4", "P1170117.MP4")
VALIDATION_VIDEO_NAMES = ("P1570157.MP4",)
NATIVE_WIDTH = 2688
NATIVE_HEIGHT = 1512

# Conditional paths are deliberately absent.  B2-2V and B3 only materialize after
# their evidence gates, rather than being mistaken for available implementations.
_RUN_DIRECTORIES = (
    "input_lock/environment_receipts",
    "sampling/images",
    "maps",
    "mvroma/raw_shards",
    "edm/b0/fixed_roster",
    "edm/b0/deployment_roster",
    "edm/b1/fixed_roster",
    "edm/b1/deployment_roster",
    "edm/b2/fixed_roster",
    "edm/b2/deployment_roster",
    "verify/pseudo_gt",
    "verify/independent_2hz",
    "verify/sequential_24fps",
    "reports",
)

_HISTORICAL_UPDATE_RUN_DIRECTORIES = (
    "input_lock/runtime",
    "historical/images",
    "historical/descriptors",
    "historical/direct_registration",
    "historical/bridge_graph",
    "historical/bridge_submaps",
    "historical/change_masks",
    "historical/selection",
    *(f"bundles/e{stage}/{roster}" for stage in range(6) for roster in ("fixed_roster", "deployment_roster")),
    *(
        f"experiments/e{stage}/{partition}"
        for stage in range(6)
        for partition in (
            "base_loo",
            "update_loo",
            "validation_development",
            "validation_final",
            "sequential",
            "ablations",
        )
    ),
    "validation/pseudo_gt",
    "validation/alignments",
    "validation/independent_2hz",
    "validation/sequential_24fps",
    "reports/visualizations",
)


VideoProbe = Callable[[Path], Mapping[str, object]]
ImageWriter = Callable[[str, np.ndarray], bool]


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    return Path(*path.parts)


def _reject_symlinked_parents(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"artifact parent must not be a symlink: {current}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _create_layout(
    root: Path,
    *,
    directories: tuple[str, ...],
    top_level: tuple[str, ...],
) -> dict[str, Path]:
    """Atomically create one append-only artifact root with declared directories."""

    root = root.absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"run root already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        for relative in directories:
            (temporary / relative).mkdir(parents=True, exist_ok=False)
        os.replace(temporary, root)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
        raise
    return {"root": root, **{name: root / name for name in top_level}}


def create_run_layout(root: Path) -> dict[str, Path]:
    """Create the legacy base-only run layout without changing its semantics."""

    return _create_layout(
        root,
        directories=_RUN_DIRECTORIES,
        top_level=("input_lock", "sampling", "maps", "mvroma", "edm", "verify", "reports"),
    )


def create_historical_update_layout(root: Path) -> dict[str, Path]:
    """Create the isolated E0--E5 historical-update artifact tree once."""

    return _create_layout(
        root,
        directories=_HISTORICAL_UPDATE_RUN_DIRECTORIES,
        top_level=("input_lock", "historical", "bundles", "experiments", "validation", "reports"),
    )


def write_new_json(root: Path, relative: str, payload: Mapping[str, Any]) -> Path:
    """Atomically create one JSON artifact; replacing a receipt is forbidden."""

    root = root.resolve(strict=True)
    relative_path = _safe_relative(relative)
    _reject_symlinked_parents(root, relative_path)
    target = root / relative_path
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_parents(root, relative_path)
    _atomic_json(target, payload)
    return target


def load_native_pinhole_intrinsics(accepted_intrinsics: Path) -> dict[str, object]:
    """Lock the accepted native PINHOLE K while intentionally discarding distortion.

    The new B0 consumes native decoded pixels.  Its fixed zero distortion is a map
    contract, not an invitation to undistort source video again.
    """

    accepted_intrinsics = accepted_intrinsics.resolve(strict=True)
    payload = json.loads(accepted_intrinsics.read_text(encoding="utf-8"))
    if payload.get("camera_model") != "PINHOLE":
        raise ValueError("accepted intrinsics must use the PINHOLE camera model")
    width = int(payload.get("image_width", 0))
    height = int(payload.get("image_height", 0))
    if (width, height) != (NATIVE_WIDTH, NATIVE_HEIGHT):
        raise ValueError(
            f"accepted intrinsics must be {NATIVE_WIDTH}x{NATIVE_HEIGHT}, got {width}x{height}"
        )
    params = [float(value) for value in payload.get("params", [])]
    if len(params) != 4 or not np.isfinite(params).all() or params[0] <= 0 or params[1] <= 0:
        raise ValueError("accepted PINHOLE parameters must be four finite positive focal values")
    return {
        "schema_version": 1,
        "source_intrinsics": fingerprint_file(accepted_intrinsics, sha256=True).as_dict(),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
        "params": params,
        "K": [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]],
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "native_pixels_only": True,
        "resize_permitted": False,
        "undistort_permitted": False,
    }


def probe_video_pts(video: Path) -> dict[str, object]:
    """Inspect decoded-frame PTS with ffprobe rather than assuming a 24 Hz clock."""

    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-show_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration:frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode:
        raise ValueError(f"ffprobe failed for {video}: {completed.stderr.strip()}")
    try:
        decoded = json.loads(completed.stdout)
        stream = next(iter(decoded["streams"]))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe returned no video stream for {video}") from exc
    pts = [
        float(row["best_effort_timestamp_time"])
        for row in decoded.get("frames", [])
        if row.get("best_effort_timestamp_time") not in (None, "N/A")
    ]
    strictly_increasing = bool(pts) and all(later > earlier for earlier, later in pairwise(pts))
    duration_value = decoded.get("format", {}).get("duration")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": str(stream.get("avg_frame_rate") or ""),
        "frame_count": len(pts),
        "stream_nb_frames": stream.get("nb_frames"),
        "duration_seconds": float(duration_value) if duration_value not in (None, "N/A") else None,
        "first_pts_time": pts[0] if pts else None,
        "last_pts_time": pts[-1] if pts else None,
        "pts_count": len(pts),
        "pts_strictly_increasing": strictly_increasing,
    }


def _validate_partition(
    root: Path,
    *,
    partition: str,
    expected_names: tuple[str, ...],
    probe_video: VideoProbe,
    source_directory: str | None = None,
) -> dict[str, dict[str, object]]:
    directory = root / (source_directory or partition)
    if not directory.is_dir() or directory.is_symlink():
        raise FileNotFoundError(directory)
    actual_names = tuple(sorted(path.name for path in directory.iterdir() if path.is_file()))
    expected_set = set(expected_names)
    actual_set = set(actual_names)
    if actual_set != expected_set:
        raise ValueError(
            f"unexpected source videos in {partition}: "
            f"missing={sorted(expected_set - actual_set)!r}, "
            f"extra={sorted(actual_set - expected_set)!r}"
        )
    result: dict[str, dict[str, object]] = {}
    for name in expected_names:
        video = directory / name
        if video.is_symlink() or video.parent.resolve(strict=True) != directory.resolve(strict=True):
            raise ValueError(f"source role path must not resolve outside {partition}: {video}")
        probe = dict(probe_video(video))
        if (int(probe.get("width", 0)), int(probe.get("height", 0))) != (
            NATIVE_WIDTH,
            NATIVE_HEIGHT,
        ):
            raise ValueError(f"source video is not native {NATIVE_WIDTH}x{NATIVE_HEIGHT}: {video}")
        if int(probe.get("pts_count", 0)) < 2 or probe.get("pts_strictly_increasing") is not True:
            raise ValueError(f"source video lacks decodable monotonic PTS: {video}")
        result[name] = {
            "basename": name,
            "partition": partition,
            "source_directory": directory.name,
            **fingerprint_file(video, sha256=True).as_dict(),
            "ffprobe": probe,
        }
    return result


def validate_source_corpus(
    raw_root: Path, *, probe_video: VideoProbe = probe_video_pts
) -> dict[str, dict[str, dict[str, object]]]:
    """Hash and validate role-disjoint base, historical-update, and validation videos."""

    raw_root = raw_root.resolve(strict=True)
    return {
        "base": _validate_partition(
            raw_root,
            partition="base",
            expected_names=BASE_VIDEO_NAMES,
            probe_video=probe_video,
        ),
        "update": _validate_partition(
            raw_root,
            partition="update",
            expected_names=UPDATE_VIDEO_NAMES,
            probe_video=probe_video,
        ),
        "VALIDATION": _validate_partition(
            raw_root,
            partition="VALIDATION",
            source_directory="VERIFY",
            expected_names=VALIDATION_VIDEO_NAMES,
            probe_video=probe_video,
        ),
    }


def write_native_frame(
    pixels: np.ndarray,
    destination: Path,
    *,
    width: int,
    height: int,
    writer: ImageWriter,
) -> dict[str, object]:
    """Write exactly the decoded native pixel buffer, with no resize or undistort hook."""

    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[:2] != (height, width):
        raise ValueError(
            f"decoded pixels must retain native dimensions {width}x{height}, got {array.shape[:2]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not writer(str(destination), pixels):
        raise RuntimeError(f"image writer failed for {destination}")
    if not destination.is_file():
        raise RuntimeError(f"image writer returned success without creating {destination}")
    return {
        "path": str(destination),
        "width": width,
        "height": height,
        "image_sha256": fingerprint_file(destination, sha256=True).sha256,
    }
