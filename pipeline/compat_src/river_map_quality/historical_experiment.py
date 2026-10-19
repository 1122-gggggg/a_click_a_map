"""Append-only orchestration for the role-separated historical EDM experiment.

The module deliberately separates initialization and immutable-input verification from
expensive extraction/replay stages.  Every stage reuses :func:`assert_b0_unchanged`
before publishing output; no function in this file writes below the frozen B0 root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from river_map_quality.baseline import verify_frozen_baseline
from river_map_quality.colmap_binary import (
    iter_binary_image_observations,
    read_binary_image_poses,
    read_binary_points3d,
)
from river_map_quality.colmap_tracks import read_point_track_index
from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    BASE_VIDEO_NAMES,
    UPDATE_VIDEO_NAMES,
    VALIDATION_VIDEO_NAMES,
    create_historical_update_layout,
    validate_source_corpus,
    write_new_json,
)

HISTORICAL_EXPERIMENT_CONFIG_SCHEMA = "RIVER_HISTORICAL_EDM_UPDATE_CONFIG_V1"
B0_IMMUTABILITY_SCHEMA = "RIVER_B0_IMMUTABILITY_RECEIPT_V1"
RUNTIME_RECEIPT_SCHEMA = "RIVER_HISTORICAL_RUNTIME_RECEIPT_V1"

AUDITED_TORCH_SITE_PACKAGES = Path("/home/cihcilab/.local/lib/python3.12/site-packages")


class HistoricalExperimentError(RuntimeError):
    """Raised when an experiment would cross a role or immutable-map boundary."""


@dataclass(frozen=True)
class FrozenB0:
    """Validated immutable B0 location and exact COLMAP model directory."""

    root: Path
    model: Path


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def resolve_b0_model_root(baseline_root: Path) -> FrozenB0:
    """Resolve an immutable B0 model accepting only ``model`` or ``reconstruction``."""

    root = baseline_root.resolve(strict=True)
    if root.is_symlink():
        raise HistoricalExperimentError(f"B0 root must not be a symlink: {root}")
    manifest = root / "MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    candidates = [root / name for name in ("model", "reconstruction")]
    binaries = ("cameras.bin", "images.bin", "points3D.bin")
    models = [
        candidate
        for candidate in candidates
        if candidate.is_dir() and all((candidate / name).is_file() for name in binaries)
    ]
    if len(models) != 1:
        raise HistoricalExperimentError(
            "B0 must contain exactly one complete COLMAP model directory "
            "named model or reconstruction"
        )
    return FrozenB0(root=root, model=models[0])


def _bridge_observation_invariants(model: Path, bridge_list: Path) -> dict[str, object]:
    observations = {row.name: row for row in iter_binary_image_observations(model / "images.bin")}
    zero_observation_names = tuple(
        sorted(
            name
            for name, row in observations.items()
            if not np.any(np.asarray(row.point3d_ids, dtype=np.int64) >= 0)
        )
    )
    if not bridge_list.is_file():
        return {
            "bridge_list_present": False,
            "bridge_image_count": 0,
            "zero_observation_count": len(zero_observation_names),
            "zero_observation_names_sha256": _canonical_json_sha256(zero_observation_names),
        }
    bridge_names = tuple(
        line.strip()
        for line in bridge_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(bridge_names) != len(set(bridge_names)):
        raise HistoricalExperimentError("B0 bridge-only image list contains duplicate names")
    missing = sorted(set(bridge_names) - set(observations))
    if missing:
        raise HistoricalExperimentError(
            f"B0 bridge-only image list names absent from model: {missing!r}"
        )
    counts = {
        name: int(np.count_nonzero(observations[name].point3d_ids >= 0)) for name in bridge_names
    }
    nonzero = {name: count for name, count in counts.items() if count != 0}
    if nonzero:
        raise HistoricalExperimentError(
            f"B0 bridge-only images gained Point3D observations: {sorted(nonzero.items())!r}"
        )
    return {
        "bridge_list_present": True,
        "bridge_image_count": len(bridge_names),
        "listed_bridge_zero_observation_count": len(counts),
        "zero_observation_count": len(zero_observation_names),
        "zero_observation_names_sha256": _canonical_json_sha256(zero_observation_names),
        "unlisted_zero_observation_count": len(set(zero_observation_names) - set(bridge_names)),
        "nonzero_observation_images": [],
        "all_bridge_images_pose_only": True,
    }


def b0_immutability_receipt(baseline_root: Path) -> dict[str, object]:
    """Hash B0 bytes plus semantic pose/track invariants before a historical stage.

    A raw binary hash catches byte mutation.  Canonical semantic hashes identify which
    protected element changed when a binary is regenerated with different ordering.
    """

    frozen = resolve_b0_model_root(baseline_root)
    immutable_file_count = verify_frozen_baseline(frozen.root)
    model = frozen.model
    files = {
        name: fingerprint_file(model / name, sha256=True).as_dict()
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    manifest = fingerprint_file(frozen.root / "MANIFEST.json", sha256=True).as_dict()
    poses = read_binary_image_poses(model / "images.bin", strict_quaternion=True)
    pose_rows = [
        {
            "image_id": int(row.image_id),
            "name": row.name,
            "camera_id": int(row.camera_id),
            "qvec": np.asarray(row.qvec, dtype=np.float64).tolist(),
            "tvec": np.asarray(row.tvec, dtype=np.float64).tolist(),
        }
        for _, row in sorted(poses.items())
    ]
    observations = list(iter_binary_image_observations(model / "images.bin"))
    observation_rows = [
        {
            "image_id": int(row.image_id),
            "name": row.name,
            "xy_sha256": _array_sha256(np.asarray(row.xy, dtype=np.float64)),
            "point3d_ids_sha256": _array_sha256(np.asarray(row.point3d_ids, dtype=np.int64)),
            "point3d_observation_count": int(np.count_nonzero(row.point3d_ids >= 0)),
        }
        for row in sorted(observations, key=lambda row: row.image_id)
    ]
    points = read_binary_points3d(model / "points3D.bin")
    point_order = np.argsort(points["point_id"], kind="stable")
    sorted_points = points[point_order]
    tracks = read_point_track_index(model / "points3D.bin")
    track_rows = [
        {
            "point_id": int(point_id),
            "xyz": list(track.xyz),
            "error": float(track.error),
            "track_length": int(track.track_length),
            "image_ids": list(track.image_ids),
        }
        for point_id, track in sorted(tracks.items())
    ]
    intrinsics_path = frozen.root / "intrinsics.json"
    intrinsics = None
    if intrinsics_path.is_file():
        intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    bridge = _bridge_observation_invariants(model, frozen.root / "bridge_only_images.txt")
    return {
        "schema_version": 1,
        "artifact_type": B0_IMMUTABILITY_SCHEMA,
        "baseline_root": str(frozen.root),
        "model_directory": model.name,
        "manifest": manifest,
        "model_files": files,
        "immutable_file_count": immutable_file_count,
        "image_count": len(poses),
        "point3d_count": len(sorted_points),
        "pose_table_sha256": _canonical_json_sha256(pose_rows),
        "intrinsics_sha256": _canonical_json_sha256(intrinsics),
        "point_xyz_sha256": _array_sha256(np.asarray(sorted_points["xyz"], dtype=np.float64)),
        "point_track_table_sha256": _canonical_json_sha256(track_rows),
        "original_observation_table_sha256": _canonical_json_sha256(observation_rows),
        "bridge_only_invariants": bridge,
    }


def assert_b0_unchanged(
    baseline_root: Path, frozen_receipt: Mapping[str, object]
) -> dict[str, object]:
    """Recompute every protected B0 identity and fail closed on any difference."""

    current = b0_immutability_receipt(baseline_root)
    protected = (
        "baseline_root",
        "model_directory",
        "manifest",
        "model_files",
        "pose_table_sha256",
        "intrinsics_sha256",
        "point_xyz_sha256",
        "point_track_table_sha256",
        "original_observation_table_sha256",
        "bridge_only_invariants",
    )
    mismatches = [name for name in protected if current.get(name) != frozen_receipt.get(name)]
    if mismatches:
        raise HistoricalExperimentError(f"immutable B0 receipt changed: {mismatches!r}")
    return current


def activate_audited_torch() -> Path:
    """Append the audited CUDA Torch location without shadowing canonical PyCOLMAP."""

    path = AUDITED_TORCH_SITE_PACKAGES.resolve(strict=True)
    if "torch" in sys.modules:
        loaded = Path(str(getattr(sys.modules["torch"], "__file__", ""))).resolve()
        if path not in loaded.parents:
            raise HistoricalExperimentError(
                f"torch was loaded from an unapproved location before runtime admission: {loaded}"
            )
    if str(path) not in sys.path:
        sys.path.append(str(path))
    return path


def _module_location(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve())


def runtime_receipt(runtime_assets: Mapping[str, Path | str]) -> dict[str, object]:
    """Hash offline model/runtime inputs and record the executing Python/CUDA stack."""

    torch_site_packages = activate_audited_torch()

    assets: dict[str, dict[str, object]] = {}
    for name, value in sorted(runtime_assets.items()):
        path = Path(value).resolve(strict=True)
        if path.is_dir():
            files = [
                {
                    "relative_path": child.relative_to(path).as_posix(),
                    **fingerprint_file(child, sha256=True).as_dict(),
                }
                for child in sorted(path.rglob("*"))
                if child.is_file()
            ]
            if not files:
                raise HistoricalExperimentError(f"runtime asset directory is empty: {path}")
            assets[name] = {
                "kind": "directory",
                "path": str(path),
                "file_count": len(files),
                "files": files,
                "tree_sha256": _canonical_json_sha256(files),
            }
        elif path.is_file():
            assets[name] = {"kind": "file", **fingerprint_file(path, sha256=True).as_dict()}
        else:
            raise FileNotFoundError(path)
    package_locations = {
        name: _module_location(name) for name in ("numpy", "cv2", "pycolmap", "torch")
    }
    torch_payload: dict[str, object] = {"available": False}
    try:
        import torch

        torch_payload = {
            "available": True,
            "version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            torch_payload.update(
                {
                    "cuda_device": int(device),
                    "cuda_device_name": str(torch.cuda.get_device_name(device)),
                    "cuda_capability": list(torch.cuda.get_device_capability(device)),
                }
            )
    except Exception as exc:  # Receipt remains useful on CPU-only test hosts.
        torch_payload = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "schema_version": 1,
        "artifact_type": RUNTIME_RECEIPT_SCHEMA,
        "network_model_load_permitted": False,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "audited_torch_site_packages": str(torch_site_packages),
        "package_locations": package_locations,
        "torch": torch_payload,
        "assets": assets,
    }


def initialize_historical_run(
    *,
    run_root: Path,
    raw_root: Path,
    b0_root: Path,
    runtime_assets: Mapping[str, Path | str],
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Create the fresh run root after all immutable source inputs have been verified."""

    if config.get("schema") != HISTORICAL_EXPERIMENT_CONFIG_SCHEMA:
        raise HistoricalExperimentError("historical configuration schema is not recognized")
    source_videos = validate_source_corpus(raw_root)
    if tuple(source_videos["base"]) != BASE_VIDEO_NAMES:
        raise HistoricalExperimentError("base corpus membership changed")
    if tuple(source_videos["update"]) != UPDATE_VIDEO_NAMES:
        raise HistoricalExperimentError("historical update corpus membership changed")
    if tuple(source_videos["VALIDATION"]) != VALIDATION_VIDEO_NAMES:
        raise HistoricalExperimentError("validation corpus membership changed")
    b0 = b0_immutability_receipt(b0_root)
    runtime = runtime_receipt(runtime_assets)
    layout = create_historical_update_layout(run_root)
    write_new_json(layout["root"], "input_lock/source_videos.json", source_videos)
    write_new_json(layout["root"], "input_lock/b0.json", b0)
    write_new_json(layout["root"], "input_lock/runtime/runtime.json", runtime)
    write_new_json(layout["root"], "input_lock/config.json", dict(config))
    return {
        "run_root": str(layout["root"]),
        "source_videos": source_videos,
        "b0": b0,
        "runtime": runtime,
    }


def _load_config(path: Path) -> dict[str, Any]:
    """Read the JSON-compatible YAML config used by the one historical CLI."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HistoricalExperimentError(
            "config must be JSON-compatible YAML to avoid an unpinned YAML parser"
        ) from exc
    if not isinstance(value, dict):
        raise HistoricalExperimentError("historical config must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("initialize", "all", "extend", "append-loo", *(f"e{index}" for index in range(6))),
        required=True,
    )
    parser.add_argument("--run-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = _load_config(arguments.config)
    run_root = arguments.run_root or Path(str(config["run_root"]))
    if arguments.stage in {"initialize", "all"}:
        if not (run_root / "input_lock/source_videos.json").is_file():
            initialize_historical_run(
                run_root=run_root,
                raw_root=Path(str(config["raw_root"])),
                b0_root=Path(str(config["b0_root"])),
                runtime_assets={
                    str(key): Path(str(value))
                    for key, value in dict(config["runtime_assets"]).items()
                },
                config=config,
            )
        if arguments.stage == "initialize":
            return 0
    from river_map_quality.historical_pipeline import (
        execute_append_loo,
        execute_historical_pipeline,
        execute_map_extend,
    )

    if arguments.stage == "all":
        execute_historical_pipeline(config, run_root)
        return 0
    if arguments.stage == "extend":
        execute_map_extend(config, run_root)
        return 0
    if arguments.stage == "append-loo":
        execute_append_loo(config, run_root)
        return 0
    from river_map_quality.historical_stages import STAGE_ORDER

    requested = arguments.stage.upper()
    if requested not in STAGE_ORDER:
        raise HistoricalExperimentError(f"unknown stage: {requested}")
    execute_historical_pipeline(config, run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
