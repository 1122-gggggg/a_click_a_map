"""Offline MegaLoc reference catalogs for immutable B0 and historical images.

This module never calls Torch Hub or Hugging Face download APIs.  It imports the
pinned local MegaLoc source directly, loads the explicit local safetensors checkpoint,
and serializes immutable descriptor/observation bundles with tamper checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np

from river_map_quality.colmap_binary import (
    ColmapImageObservations,
    iter_binary_image_observations,
    read_binary_image_poses,
    read_binary_points3d,
)
from river_map_quality.historical_experiment import (
    FrozenB0,
    HistoricalExperimentError,
    activate_audited_torch,
    assert_b0_unchanged,
    resolve_b0_model_root,
)
from river_map_quality.provenance import fingerprint_file

CATALOG_SCHEMA = "RIVER_MEGALOC_EDM_CATALOG_V1"
MEGALOC_DESCRIPTOR_DIMENSION = 8448
MEGALOC_INPUT_SIZE = 322
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class MegaLocCatalogError(HistoricalExperimentError):
    """Raised when a descriptor, catalog, or offline runtime violates its contract."""


@dataclass(frozen=True)
class MegaLocRuntime:
    """Pinned offline MegaLoc model and its immutable source/checkpoint identities."""

    model: Any
    torch: Any
    device: str
    source: Path
    checkpoint: Path
    source_sha256: str
    checkpoint_sha256: str
    descriptor_dimension: int = MEGALOC_DESCRIPTOR_DIMENSION
    input_size: int = MEGALOC_INPUT_SIZE


@dataclass(frozen=True)
class FrozenReferenceCatalog:
    """Read-only retrieval rows and lifted B0 observation tables."""

    root: Path
    names: tuple[str, ...]
    descriptors: np.ndarray
    metadata: Mapping[str, Any]
    observation_offsets: np.ndarray
    observation_xy: np.ndarray
    observation_point3d_ids: np.ndarray
    point_ids: np.ndarray
    point_xyz: np.ndarray

    def observation_slice(self, reference_name: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            index = self.names.index(reference_name)
        except ValueError as exc:
            raise MegaLocCatalogError(f"reference is absent from catalog: {reference_name}") from exc
        start, end = (int(self.observation_offsets[index]), int(self.observation_offsets[index + 1]))
        return self.observation_xy[start:end], self.observation_point3d_ids[start:end]

    def point_xyz_for_ids(self, point3d_ids: np.ndarray) -> np.ndarray:
        requested = np.asarray(point3d_ids, dtype=np.int64)
        indices = np.searchsorted(self.point_ids, requested)
        if np.any(indices >= len(self.point_ids)) or not np.array_equal(
            self.point_ids[indices], requested
        ):
            raise MegaLocCatalogError("catalog does not contain a lifted Point3D ID")
        return self.point_xyz[indices]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    files = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "size": fingerprint_file(path, sha256=True).size,
            "sha256": fingerprint_file(path, sha256=True).sha256,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    if not files:
        raise MegaLocCatalogError(f"MegaLoc source directory is empty: {root}")
    return _canonical_sha256(files)


def _import_megaloc_module(source: Path) -> ModuleType:
    module_path = source / "megaloc_model.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    module_name = f"_river_megaloc_{hashlib.sha256(str(module_path).encode()).hexdigest()[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise MegaLocCatalogError(f"cannot import pinned MegaLoc source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_offline_megaloc_runtime(
    *,
    source: Path,
    checkpoint: Path,
    device: str = "cuda",
) -> MegaLocRuntime:
    """Load exactly the local MegaLoc source and explicit cached checkpoint.

    This intentionally avoids ``torch.hub.load`` and ``hf_hub_download`` because both
    can turn a cache miss into an implicit network model load.
    """

    activate_audited_torch()
    import torch

    source = source.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if source.is_symlink() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise MegaLocCatalogError("MegaLoc source and checkpoint must be non-symlink local inputs")
    if device == "cuda" and not torch.cuda.is_available():
        raise MegaLocCatalogError("CUDA was requested for MegaLoc but is unavailable")
    if device not in {"cpu", "cuda"}:
        raise MegaLocCatalogError(f"unsupported MegaLoc device: {device}")
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise MegaLocCatalogError("pinned MegaLoc runtime requires safetensors") from exc
    module = _import_megaloc_module(source)
    model = module.MegaLoc()
    state = load_file(str(checkpoint), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise MegaLocCatalogError(
            f"MegaLoc checkpoint disagrees with pinned source: missing={missing!r}, unexpected={unexpected!r}"
        )
    model.eval().to(device)
    return MegaLocRuntime(
        model=model,
        torch=torch,
        device=device,
        source=source,
        checkpoint=checkpoint,
        source_sha256=_tree_sha256(source),
        checkpoint_sha256=str(fingerprint_file(checkpoint, sha256=True).sha256),
    )


def _load_native_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim != 3 or image.shape[2] != 3:
        raise MegaLocCatalogError(f"MegaLoc image is not a native RGB-like image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _batch_tensor(runtime: MegaLocRuntime, paths: Sequence[Path]) -> Any:
    images = [_load_native_rgb(path) for path in paths]
    torch = runtime.torch
    rows = []
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    for image in images:
        row = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
        row = torch.nn.functional.interpolate(
            row.unsqueeze(0),
            size=(runtime.input_size, runtime.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)
        rows.append((row - mean) / std)
    return torch.stack(rows, dim=0).to(runtime.device, non_blocking=runtime.device == "cuda")


def extract_megaloc_descriptors(
    runtime: MegaLocRuntime,
    image_paths: Sequence[Path],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract finite normalized 8,448-D descriptors from native source pixels."""

    if batch_size <= 0:
        raise ValueError("MegaLoc batch_size must be positive")
    paths = tuple(Path(path).resolve(strict=True) for path in image_paths)
    if not paths:
        raise MegaLocCatalogError("cannot extract an empty MegaLoc catalog")
    output: list[np.ndarray] = []
    torch = runtime.torch
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            descriptor = runtime.model(_batch_tensor(runtime, paths[start : start + batch_size]))
            array = descriptor.detach().float().cpu().numpy()
            if array.ndim != 2 or array.shape[1] != runtime.descriptor_dimension:
                raise MegaLocCatalogError(
                    f"MegaLoc descriptor shape {array.shape!r} violates {runtime.descriptor_dimension}-D contract"
                )
            if not np.isfinite(array).all():
                raise MegaLocCatalogError("MegaLoc emitted a non-finite descriptor")
            norms = np.linalg.norm(array, axis=1)
            if np.any(norms <= 1e-12) or not np.isfinite(norms).all():
                raise MegaLocCatalogError("MegaLoc emitted a zero-norm descriptor")
            output.append(np.ascontiguousarray(array / norms[:, None], dtype=np.float32))
    return np.ascontiguousarray(np.concatenate(output, axis=0), dtype=np.float32)


def _atomic_numpy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact_fingerprint(path: Path) -> dict[str, object]:
    """Fingerprint staged content with a stable bundle-relative path identity."""

    fingerprint = fingerprint_file(path, sha256=True).as_dict()
    return {**fingerprint, "path": path.name}


def _write_fingerprint_sidecar(path: Path) -> None:
    _atomic_json(
        path.with_name(f"{path.name}.fingerprint.json"),
        {
            "artifact_path": path.name,
            "fingerprint": _artifact_fingerprint(path),
            "sidecar_schema": 1,
        },
    )


def _read_frame_records(frame_manifest: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(frame_manifest.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise MegaLocCatalogError("B0 frame manifest lacks frames")
    records: dict[str, dict[str, Any]] = {}
    for raw in frames:
        if not isinstance(raw, dict):
            raise MegaLocCatalogError("B0 frame manifest contains a non-object frame")
        name = str(raw.get("output_name", ""))
        image_hash = str(raw.get("image_sha256", ""))
        if not name or len(image_hash) != 64 or name in records:
            raise MegaLocCatalogError("B0 frame manifest has duplicate or unhashed image identity")
        records[name] = dict(raw)
    return records


def _pack_observations(
    names: Sequence[str], observations: Mapping[str, ColmapImageObservations], points: np.ndarray
) -> dict[str, np.ndarray]:
    offsets = [0]
    xy_parts: list[np.ndarray] = []
    id_parts: list[np.ndarray] = []
    for name in names:
        row = observations[name]
        xy = np.asarray(row.xy, dtype=np.float64)
        ids = np.asarray(row.point3d_ids, dtype=np.int64)
        if xy.shape != (len(ids), 2):
            raise MegaLocCatalogError(f"B0 observation table is malformed for {name}")
        xy_parts.append(xy)
        id_parts.append(ids)
        offsets.append(offsets[-1] + len(ids))
    order = np.argsort(points["point_id"], kind="stable")
    sorted_points = points[order]
    ids = np.asarray(sorted_points["point_id"], dtype=np.int64)
    if len(ids) != len(np.unique(ids)):
        raise MegaLocCatalogError("B0 Point3D IDs are not unique")
    return {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "xy": np.concatenate(xy_parts, axis=0) if xy_parts else np.empty((0, 2), dtype=np.float64),
        "observation_point3d_ids": (
            np.concatenate(id_parts, axis=0) if id_parts else np.empty(0, dtype=np.int64)
        ),
        "point_ids": ids,
        "point_xyz": np.asarray(sorted_points["xyz"], dtype=np.float64),
    }


def _catalog_contract_sha256(
    *,
    names: Sequence[str],
    descriptors: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    observations: Mapping[str, np.ndarray],
    model_identity: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(CATALOG_SCHEMA.encode("ascii"))
    digest.update(_canonical_sha256({"names": list(names), "records": list(records), "model": model_identity}).encode())
    digest.update(_array_sha256(descriptors).encode())
    for name in ("offsets", "xy", "observation_point3d_ids", "point_ids", "point_xyz"):
        digest.update(name.encode("ascii"))
        digest.update(_array_sha256(observations[name]).encode())
    return digest.hexdigest()


def build_b0_reference_catalog(
    *,
    b0_root: Path,
    image_root: Path,
    output_directory: Path,
    runtime: MegaLocRuntime,
    b0_receipt: Mapping[str, Any],
    batch_size: int = 8,
) -> FrozenReferenceCatalog:
    """Build an experimental, immutable MegaLoc+EDM catalog over unchanged B0 geometry."""

    frozen: FrozenB0 = resolve_b0_model_root(b0_root)
    assert_b0_unchanged(frozen.root, b0_receipt)
    image_root = image_root.resolve(strict=True)
    if image_root.is_symlink() or not image_root.is_dir():
        raise MegaLocCatalogError(f"B0 image root must be a non-symlink directory: {image_root}")
    output_directory = output_directory.absolute()
    if output_directory.exists() or output_directory.is_symlink():
        raise FileExistsError(f"catalog output already exists: {output_directory}")
    frame_manifest = frozen.root / "frame_manifest.json"
    records_by_name = _read_frame_records(frame_manifest)
    observations = {row.name: row for row in iter_binary_image_observations(frozen.model / "images.bin")}
    poses = read_binary_image_poses(frozen.model / "images.bin", strict_quaternion=True)
    names = tuple(sorted(observations))
    if set(names) != set(records_by_name):
        raise MegaLocCatalogError("B0 frame manifest and model image identities disagree")
    if {row.name for row in poses.values()} != set(names):
        raise MegaLocCatalogError("B0 pose table and observation table image identities disagree")
    image_paths = []
    records = []
    for name in names:
        path = image_root / name
        if path.is_symlink() or not path.is_file():
            raise MegaLocCatalogError(f"B0 catalog image is unavailable: {path}")
        current = fingerprint_file(path, sha256=True)
        record = records_by_name[name]
        if current.sha256 != record["image_sha256"]:
            raise MegaLocCatalogError(f"B0 catalog image changed since frame manifest: {name}")
        image_paths.append(path)
        records.append(
            {
                "name": name,
                "image_sha256": current.sha256,
                "source_video_sha256": str(record["video_sha256"]),
                "source_pts": int(record["source_pts"]),
                "source_frame_index": int(record["source_frame_index"]),
                "motion_class": str(record["motion_class"]),
                "sampling_role": str(record["role"]),
                "source_role": "current",
            }
        )
    descriptors = extract_megaloc_descriptors(runtime, image_paths, batch_size=batch_size)
    if descriptors.shape != (len(names), runtime.descriptor_dimension):
        raise MegaLocCatalogError("B0 MegaLoc descriptors do not align with reference names")
    packed = _pack_observations(names, observations, read_binary_points3d(frozen.model / "points3D.bin"))
    model_identity = {
        "runtime_label": "REPRODUCED_EXPERIMENTAL_RUNTIME",
        "model": "MegaLoc",
        "source_sha256": runtime.source_sha256,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "descriptor_dimension": runtime.descriptor_dimension,
        "input_size": runtime.input_size,
        "input_preprocess": "native RGB -> ImageNet normalize -> bilinear 322x322",
        "network_model_load_permitted": False,
    }
    contract_sha = _catalog_contract_sha256(
        names=names,
        descriptors=descriptors,
        records=records,
        observations=packed,
        model_identity=model_identity,
    )
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=parent))
    try:
        descriptor_path = stage / "descriptors.npy"
        observation_path = stage / "observations.npz"
        _atomic_numpy(descriptor_path, descriptors)
        _atomic_npz(
            observation_path,
            names=np.asarray(names, dtype=np.str_),
            offsets=packed["offsets"],
            xy=packed["xy"],
            observation_point3d_ids=packed["observation_point3d_ids"],
            point_ids=packed["point_ids"],
            point_xyz=packed["point_xyz"],
        )
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": CATALOG_SCHEMA,
            "runtime_label": "REPRODUCED_EXPERIMENTAL_RUNTIME",
            "source_role": "current",
            "reference_count": len(names),
            "reference_names": list(names),
            "reference_records": records,
            "descriptor": {
                "path": descriptor_path.name,
                "shape": list(descriptors.shape),
                "dtype": str(descriptors.dtype),
                "sha256": _array_sha256(descriptors),
            },
            "observations": {
                "path": observation_path.name,
                "sha256": _canonical_sha256(
                    {name: _array_sha256(values) for name, values in packed.items()}
                ),
                "reference_count": len(names),
                "point3d_count": len(packed["point_ids"]),
                "pose_only_reference_count": int(
                    sum(
                        packed["offsets"][index] == packed["offsets"][index + 1]
                        or not np.any(
                            packed["observation_point3d_ids"][
                                packed["offsets"][index] : packed["offsets"][index + 1]
                            ]
                            >= 0
                        )
                        for index in range(len(names))
                    )
                ),
                "zero_lift_from_pose_only_references": True,
            },
            "b0": {
                "manifest_sha256": b0_receipt["manifest"]["sha256"],
                "model_files": b0_receipt["model_files"],
                "pose_table_sha256": b0_receipt["pose_table_sha256"],
                "point_xyz_sha256": b0_receipt["point_xyz_sha256"],
                "original_observation_table_sha256": b0_receipt["original_observation_table_sha256"],
                "bridge_only_invariants": b0_receipt["bridge_only_invariants"],
            },
            "model": model_identity,
            "catalog_contract_sha256": contract_sha,
        }
        metadata_path = stage / "catalog.json"
        _atomic_json(metadata_path, metadata)
        for path in (descriptor_path, observation_path, metadata_path):
            _write_fingerprint_sidecar(path)
        os.replace(stage, output_directory)
    except Exception:
        if stage.exists():
            for path in sorted(stage.rglob("*"), reverse=True):
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            stage.rmdir()
        raise
    assert_b0_unchanged(frozen.root, b0_receipt)
    return load_reference_catalog(output_directory)


def _validate_sidecar(path: Path) -> None:
    sidecar_path = path.with_name(f"{path.name}.fingerprint.json")
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise MegaLocCatalogError(f"catalog fingerprint sidecar is missing: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = payload.get("fingerprint") if isinstance(payload, dict) else None
    if payload.get("artifact_path") != path.name or expected != _artifact_fingerprint(path):
        raise MegaLocCatalogError(f"catalog artifact fingerprint mismatch: {path}")


def load_reference_catalog(directory: Path) -> FrozenReferenceCatalog:
    """Load and validate a serialized catalog before it enters retrieval or PnP."""

    root = directory.resolve(strict=True)
    if root.is_symlink():
        raise MegaLocCatalogError(f"catalog directory must not be a symlink: {root}")
    descriptor_path = root / "descriptors.npy"
    observation_path = root / "observations.npz"
    metadata_path = root / "catalog.json"
    for path in (descriptor_path, observation_path, metadata_path):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        _validate_sidecar(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("artifact_type") != CATALOG_SCHEMA:
        raise MegaLocCatalogError("unsupported MegaLoc catalog metadata")
    names_value = metadata.get("reference_names")
    if not isinstance(names_value, list) or not names_value or not all(
        isinstance(name, str) and name for name in names_value
    ):
        raise MegaLocCatalogError("catalog reference names are invalid")
    names = tuple(names_value)
    if len(names) != len(set(names)) or len(names) != int(metadata.get("reference_count", -1)):
        raise MegaLocCatalogError("catalog reference identities are duplicated or miscounted")
    descriptors = np.load(descriptor_path, allow_pickle=False)
    if descriptors.dtype != np.float32 or descriptors.shape != (len(names), MEGALOC_DESCRIPTOR_DIMENSION):
        raise MegaLocCatalogError("catalog descriptor shape or dtype violates the MegaLoc contract")
    if not np.isfinite(descriptors).all():
        raise MegaLocCatalogError("catalog descriptors are non-finite")
    norms = np.linalg.norm(descriptors, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-4):
        raise MegaLocCatalogError("catalog descriptors are not normalized")
    if not secrets.compare_digest(str(metadata["descriptor"]["sha256"]), _array_sha256(descriptors)):
        raise MegaLocCatalogError("catalog descriptor payload was tampered")
    with np.load(observation_path, allow_pickle=False) as payload:
        required = ("names", "offsets", "xy", "observation_point3d_ids", "point_ids", "point_xyz")
        if tuple(sorted(payload.files)) != tuple(sorted(required)):
            raise MegaLocCatalogError("catalog observation payload has unexpected arrays")
        packed = {name: np.array(payload[name], copy=True) for name in required}
    if tuple(str(name) for name in packed["names"].tolist()) != names:
        raise MegaLocCatalogError("catalog observation names disagree with descriptor names")
    offsets = np.asarray(packed["offsets"], dtype=np.int64)
    xy = np.asarray(packed["xy"], dtype=np.float64)
    observation_ids = np.asarray(packed["observation_point3d_ids"], dtype=np.int64)
    point_ids = np.asarray(packed["point_ids"], dtype=np.int64)
    point_xyz = np.asarray(packed["point_xyz"], dtype=np.float64)
    if (
        offsets.shape != (len(names) + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] != len(xy)
        or xy.shape != (len(observation_ids), 2)
        or point_xyz.shape != (len(point_ids), 3)
        or len(point_ids) != len(np.unique(point_ids))
        or not np.isfinite(xy).all()
        or not np.isfinite(point_xyz).all()
    ):
        raise MegaLocCatalogError("catalog observation packing is malformed")
    observed_sha = _canonical_sha256(
        {
            "offsets": _array_sha256(offsets),
            "xy": _array_sha256(xy),
            "observation_point3d_ids": _array_sha256(observation_ids),
            "point_ids": _array_sha256(point_ids),
            "point_xyz": _array_sha256(point_xyz),
        }
    )
    if not secrets.compare_digest(str(metadata["observations"]["sha256"]), observed_sha):
        raise MegaLocCatalogError("catalog observation payload was tampered")
    records = metadata.get("reference_records")
    if not isinstance(records, list) or [row.get("name") for row in records] != list(names):
        raise MegaLocCatalogError("catalog reference records are not name-aligned")
    expected_contract = _catalog_contract_sha256(
        names=names,
        descriptors=descriptors,
        records=records,
        observations={
            "offsets": offsets,
            "xy": xy,
            "observation_point3d_ids": observation_ids,
            "point_ids": point_ids,
            "point_xyz": point_xyz,
        },
        model_identity=metadata.get("model", {}),
    )
    if not secrets.compare_digest(str(metadata.get("catalog_contract_sha256", "")), expected_contract):
        raise MegaLocCatalogError("catalog contract checksum was tampered")
    return FrozenReferenceCatalog(
        root=root,
        names=names,
        descriptors=np.ascontiguousarray(descriptors),
        metadata=metadata,
        observation_offsets=offsets,
        observation_xy=xy,
        observation_point3d_ids=observation_ids,
        point_ids=point_ids,
        point_xyz=point_xyz,
    )
