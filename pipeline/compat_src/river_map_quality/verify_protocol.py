"""Create the immutable VERIFY temporal split and native-pixel query corpus.

VERIFY images are evidence only.  This module never reads a base-map image or writes
into a mapping/MV-RoMa artifact; its only input is the locked VERIFY source corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2

from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    VALIDATION_VIDEO_NAMES,
    write_native_frame,
    write_new_json,
)

_BLOCK_SECONDS = 15.0
_BOUNDARY_GUARD_SECONDS = 5.0
_ALIGNMENT_WINDOW_SECONDS = 4.0
_INDEPENDENT_INTERVAL_SECONDS = 0.5
_PSEUDO_GT_INTERVAL_SECONDS = 1.0


class VerifyProtocolError(RuntimeError):
    """Raised when a VERIFY evidence boundary or temporal contract is violated."""


def _intervals_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _block_assignments(video_name: str, duration_seconds: float) -> list[dict[str, object]]:
    """Assign whole 15 s blocks deterministically toward a 40/60 dev/final split."""

    if duration_seconds <= 0:
        raise ValueError("VERIFY duration must be positive")
    block_count = max(1, int((duration_seconds + _BLOCK_SECONDS - 1e-9) // _BLOCK_SECONDS))
    rows = [
        {
            "block_index": index,
            "start_seconds": index * _BLOCK_SECONDS,
            "end_seconds": min((index + 1) * _BLOCK_SECONDS, duration_seconds),
            "partition": "final",
        }
        for index in range(block_count)
    ]
    development_count = min(block_count - 1, max(1, round(block_count * 0.4)))
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"VERIFY_SPLIT_V1:{video_name}:{row['block_index']}".encode()
            ).hexdigest(),
            int(row["block_index"]),
        ),
    )
    for row in ranked[:development_count]:
        row["partition"] = "development"
    return rows


def _guard_intervals(blocks: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Exclude five seconds centered at every development/final block boundary."""

    ordered = sorted(blocks, key=lambda row: int(row["block_index"]))
    intervals: list[dict[str, object]] = []
    for left, right in pairwise(ordered):
        if left["partition"] == right["partition"]:
            continue
        boundary = float(left["end_seconds"])
        intervals.append(
            {
                "start_seconds": boundary - _BOUNDARY_GUARD_SECONDS / 2.0,
                "end_seconds": boundary + _BOUNDARY_GUARD_SECONDS / 2.0,
                "left_block_index": int(left["block_index"]),
                "right_block_index": int(right["block_index"]),
            }
        )
    return intervals


def _alignment_windows(
    blocks: Iterable[Mapping[str, object]],
    guards: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Reserve three disjoint, temporally separated windows for later Sim3 alignment."""

    guard_intervals = [(float(row["start_seconds"]), float(row["end_seconds"])) for row in guards]
    candidates: list[dict[str, object]] = []
    for block in sorted(blocks, key=lambda row: int(row["block_index"])):
        start = float(block["start_seconds"])
        end = float(block["end_seconds"])
        half = _ALIGNMENT_WINDOW_SECONDS / 2.0
        if end - start < _ALIGNMENT_WINDOW_SECONDS:
            continue
        center = (start + end) / 2.0
        interval = (center - half, center + half)
        if any(_intervals_overlap(interval, guard) for guard in guard_intervals):
            continue
        candidates.append(
            {
                "block_index": int(block["block_index"]),
                "start_seconds": interval[0],
                "end_seconds": interval[1],
            }
        )
    if len(candidates) < 3:
        raise VerifyProtocolError("VERIFY video lacks three guard-disjoint alignment windows")
    selected_positions = (0, (len(candidates) - 1) // 2, len(candidates) - 1)
    windows: list[dict[str, object]] = []
    for anchor_index, position in enumerate(selected_positions):
        windows.append(
            {
                "anchor_index": anchor_index,
                **candidates[position],
                "spatial_separation_status": "PENDING_INDEPENDENT_TRAJECTORY_CLUSTER_CHECK",
            }
        )
    return windows


def build_temporal_protocol(video_name: str, duration_seconds: float) -> dict[str, object]:
    """Build one fully deterministic temporal split before extracting any query pixels."""

    blocks = _block_assignments(video_name, duration_seconds)
    guards = _guard_intervals(blocks)
    alignment = _alignment_windows(blocks, guards)
    for window in alignment:
        interval = (float(window["start_seconds"]), float(window["end_seconds"]))
        if any(
            _intervals_overlap(
                interval,
                (float(guard["start_seconds"]), float(guard["end_seconds"])),
            )
            for guard in guards
        ):
            raise VerifyProtocolError(
                "alignment window overlaps a development/final guard interval"
            )
    return {
        "block_seconds": _BLOCK_SECONDS,
        "boundary_guard_seconds": _BOUNDARY_GUARD_SECONDS,
        "alignment_window_seconds": _ALIGNMENT_WINDOW_SECONDS,
        "blocks": blocks,
        "boundary_guard_intervals": guards,
        "alignment_windows": alignment,
    }


def _label_time(protocol: Mapping[str, object], pts_seconds: float) -> str:
    for window in protocol["alignment_windows"]:
        if float(window["start_seconds"]) <= pts_seconds < float(window["end_seconds"]):
            return "alignment"
    for guard in protocol["boundary_guard_intervals"]:
        if float(guard["start_seconds"]) <= pts_seconds < float(guard["end_seconds"]):
            return "excluded_guard"
    for block in protocol["blocks"]:
        if float(block["start_seconds"]) <= pts_seconds < float(block["end_seconds"]):
            return str(block["partition"])
    # Video decoders can return the final PTS exactly equal to duration.
    return str(protocol["blocks"][-1]["partition"])


def _selected(
    next_seconds: float, pts_seconds: float, interval_seconds: float
) -> tuple[bool, float]:
    if pts_seconds + 1e-6 < next_seconds:
        return False, next_seconds
    while next_seconds <= pts_seconds + 1e-6:
        next_seconds += interval_seconds
    return True, next_seconds


def _write_frame(
    *,
    pixels: Any,
    destination: Path,
    video_sha256: str,
    source_frame_index: int,
    pts_seconds: float,
    partition: str,
    stream: str,
) -> dict[str, object]:
    record = write_native_frame(
        pixels,
        destination,
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        writer=lambda path, image: cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]),
    )
    return {
        "stream": stream,
        "partition": partition,
        "video_sha256": video_sha256,
        "source_pts_us": round(pts_seconds * 1_000_000.0),
        "source_pts_seconds": pts_seconds,
        "source_frame_index": source_frame_index,
        "output_name": destination.as_posix(),
        "image_sha256": record["image_sha256"],
        "width": NATIVE_WIDTH,
        "height": NATIVE_HEIGHT,
    }


def _extract_video(
    *,
    video: Path,
    video_sha256: str,
    protocol: Mapping[str, object],
    stage: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    independent: list[dict[str, object]] = []
    sequential: list[dict[str, object]] = []
    pseudo_gt: list[dict[str, object]] = []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise VerifyProtocolError(f"cannot decode locked VERIFY video: {video}")
    next_independent = 0.0
    next_pseudo_gt = 0.0
    frame_index = 0
    stem = video.stem
    try:
        while True:
            ok, pixels = capture.read()
            if not ok:
                break
            if pixels is None or pixels.shape[:2] != (NATIVE_HEIGHT, NATIVE_WIDTH):
                raise VerifyProtocolError(
                    f"VERIFY decoder changed native pixels for {video}:{frame_index}"
                )
            pts_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            partition = _label_time(protocol, pts_seconds)
            pts_us = round(pts_seconds * 1_000_000.0)
            filename = f"pts_{pts_us:012d}_frame_{frame_index:06d}.jpg"
            if partition in {"development", "final"}:
                sequential_record = _write_frame(
                    pixels=pixels,
                    destination=stage / "sequential_24fps" / partition / stem / filename,
                    video_sha256=video_sha256,
                    source_frame_index=frame_index,
                    pts_seconds=pts_seconds,
                    partition=partition,
                    stream="sequential_24fps",
                )
                sequential_record["output_name"] = str(
                    Path("verify") / Path(str(sequential_record["output_name"])).relative_to(stage)
                )
                sequential.append(sequential_record)
            take_independent, next_independent = _selected(
                next_independent, pts_seconds, _INDEPENDENT_INTERVAL_SECONDS
            )
            if take_independent and partition in {"development", "final"}:
                independent_record = _write_frame(
                    pixels=pixels,
                    destination=stage / "independent_2hz" / partition / stem / filename,
                    video_sha256=video_sha256,
                    source_frame_index=frame_index,
                    pts_seconds=pts_seconds,
                    partition=partition,
                    stream="independent_2hz",
                )
                independent_record["output_name"] = str(
                    Path("verify") / Path(str(independent_record["output_name"])).relative_to(stage)
                )
                independent.append(independent_record)
            take_pseudo_gt, next_pseudo_gt = _selected(
                next_pseudo_gt, pts_seconds, _PSEUDO_GT_INTERVAL_SECONDS
            )
            if take_pseudo_gt:
                pseudo_record = _write_frame(
                    pixels=pixels,
                    destination=stage / "pseudo_gt" / stem / "images" / filename,
                    video_sha256=video_sha256,
                    source_frame_index=frame_index,
                    pts_seconds=pts_seconds,
                    partition=partition,
                    stream="pseudo_gt_1hz_continuous",
                )
                pseudo_record["output_name"] = str(
                    Path("verify") / Path(str(pseudo_record["output_name"])).relative_to(stage)
                )
                pseudo_gt.append(pseudo_record)
            frame_index += 1
    finally:
        capture.release()
    if frame_index < 2 or not independent or not pseudo_gt:
        raise VerifyProtocolError(f"insufficient decoded VERIFY evidence from {video}")
    return independent, sequential, pseudo_gt


def _stage_directory(run_root: Path) -> Path:
    return Path(tempfile.mkdtemp(prefix=".verify-protocol.", dir=run_root))


def _publish_directory(stage: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"VERIFY artifact directory already populated: {destination}")
    if destination.exists():
        destination.rmdir()
    os.replace(stage, destination)


def run_verify_protocol(
    *,
    run_root: Path,
    raw_root: Path,
    output_dirname: str = "verify",
) -> dict[str, object]:
    """Extract fixed VERIFY evidence streams without allowing them into base-map inputs."""

    run_root = run_root.resolve(strict=True)
    raw_root = raw_root.resolve(strict=True)
    source_lock_path = run_root / "input_lock/source_videos.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    locked_verify = source_lock.get("VALIDATION") or source_lock.get("VERIFY")
    if tuple(locked_verify or ()) != VALIDATION_VIDEO_NAMES:
        raise VerifyProtocolError("locked validation corpus membership or ordering changed")
    verify_root = run_root / output_dirname
    verify_root.mkdir(parents=True, exist_ok=True)
    output_manifest = verify_root / "partition_manifest.json"
    if output_manifest.exists():
        raise FileExistsError(f"VERIFY partition manifest already exists: {output_manifest}")
    stage = _stage_directory(run_root)
    try:
        protocols: dict[str, dict[str, object]] = {}
        independent: list[dict[str, object]] = []
        sequential: list[dict[str, object]] = []
        pseudo_gt: list[dict[str, object]] = []
        for name in VALIDATION_VIDEO_NAMES:
            source = raw_root / "VERIFY" / name
            current = fingerprint_file(source, sha256=True)
            expected = locked_verify[name]
            if (
                current.sha256 != expected["sha256"]
                or current.size != expected["size"]
                or str(source) != expected["path"]
            ):
                raise VerifyProtocolError(f"locked VERIFY source changed: {source}")
            duration = float(expected["ffprobe"]["duration_seconds"])
            protocol = build_temporal_protocol(name, duration)
            protocols[name] = protocol
            rows = _extract_video(
                video=source,
                video_sha256=current.sha256,
                protocol=protocol,
                stage=stage,
            )
            independent.extend(rows[0])
            sequential.extend(rows[1])
            pseudo_gt.extend(rows[2])
        artifact = {
            "schema_version": 1,
            "artifact_type": "VERIFY_TEMPORAL_PARTITION_AND_NATIVE_QUERY_CORPUS",
            "base_map_input_forbidden": True,
            "mvroma_map_group_input_forbidden": True,
            "calibration_input_forbidden": True,
            "threshold_input_forbidden": True,
            "source_lock": fingerprint_file(source_lock_path, sha256=True).as_dict(),
            "query_protocol": {
                "independent_rate_hz": 1.0 / _INDEPENDENT_INTERVAL_SECONDS,
                "sequential_rate_hz": "decoded_native_stream",
                "pseudo_gt_rate_hz": 1.0 / _PSEUDO_GT_INTERVAL_SECONDS,
                "native_pixels_only": True,
                "jpeg_quality": 95,
            },
            "videos": protocols,
            "independent_2hz_frames": independent,
            "sequential_24fps_frames": sequential,
            "pseudo_gt_continuous_frames": pseudo_gt,
        }
        _publish_directory(stage / "independent_2hz", verify_root / "independent_2hz")
        _publish_directory(stage / "sequential_24fps", verify_root / "sequential_24fps")
        _publish_directory(stage / "pseudo_gt", verify_root / "pseudo_gt")
        write_new_json(run_root, f"{output_dirname}/partition_manifest.json", artifact)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_verify_protocol(**vars(_parser().parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
