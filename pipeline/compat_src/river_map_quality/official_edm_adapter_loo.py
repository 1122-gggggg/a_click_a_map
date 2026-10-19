"""Run a separately labelled official-EDM + MegaLoc + M0-COLMAP LOO adapter.

This is a diagnostic adapter over fixed M0 landmarks.  It is not the deleted
production-EDM localizer and must not be used as production HL evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np

from river_map_quality.baseline import verify_frozen_baseline
from river_map_quality.colmap_binary import (
    ColmapImageObservations,
    iter_binary_image_observations,
    read_binary_image_poses,
    read_binary_points3d,
)
from river_map_quality.edm_loo import rank_reference_indices, spatial_cap_indices
from river_map_quality.exclusion import loo_exclusion_indices, parse_reference_name
from river_map_quality.loo_metrics import compute_loo_metrics
from river_map_quality.historical_experiment import activate_audited_torch
from river_map_quality.official_edm_adapter import (
    ADAPTER_EVIDENCE_SCOPE,
    OFFICIAL_EDM_ADAPTER_LINEAGE,
    AdapterEvidence,
    AdapterThresholds,
    assess_adapter_evidence,
    deduplicate_lifted_matches,
    lift_reference_matches,
)
from river_map_quality.provenance import FileFingerprint, fingerprint_file

ADAPTER_LOCAL_SOURCE_FILES = (
    "__init__.py",
    "baseline.py",
    "colmap_binary.py",
    "edm_loo.py",
    "exclusion.py",
    "loo_metrics.py",
    "official_edm_adapter.py",
    "official_edm_adapter_loo.py",
    "pose_information.py",
    "provenance.py",
)

AUDITED_EDM_SITE_PACKAGES = Path(
    "/media/cihcilab/新增磁碟區1/sfm_system/EDM定位測試/"
    "env/edm_eval_py312/lib/python3.12/site-packages"
)
AUDITED_SYSTEM_PYTHON_PACKAGES = Path("/usr/lib/python3/dist-packages")


@dataclass(frozen=True)
class AdapterReferenceSelection:
    indices: tuple[int, ...]
    names: tuple[str, ...]
    scores: tuple[float, ...]
    excluded_indices: tuple[int, ...]


@dataclass(frozen=True)
class SourceTreeFingerprint:
    """Hash-lock either every source file or an explicit runtime dependency closure."""

    path: str
    files: tuple[str, ...]
    sha256: str
    scope: str

    @property
    def file_count(self) -> int:
        return len(self.files)

    def as_dict(self) -> dict[str, str | int | list[str]]:
        return {
            "kind": f"python_source_{self.scope}",
            "path": self.path,
            "files": list(self.files),
            "file_count": self.file_count,
            "sha256": self.sha256,
        }

    def assert_unchanged(self) -> None:
        root = Path(self.path)
        if self.scope == "tree":
            current = fingerprint_python_source_tree(root)
        elif self.scope == "closure":
            current = fingerprint_python_source_files(root, self.files)
        else:
            raise ValueError(f"unknown official EDM source fingerprint scope: {self.scope}")
        if current != self:
            raise ValueError(f"official EDM Python sources changed: {self.path}")


@dataclass(frozen=True)
class B0Geometry:
    """Exact immutable B0 geometry required for EDM 2D→3D lifting."""

    baseline_root: Path
    model_root: Path
    manifest_sha256: str
    width: int
    height: int
    camera_params: tuple[float, float, float, float]
    pnp_camera: Any
    raw_poses_by_name: Mapping[str, Any]
    observations_by_name: Mapping[str, ColmapImageObservations]
    point_ids: np.ndarray
    point_xyz: np.ndarray


@dataclass(frozen=True)
class PreparedMegadepthImage:
    pixels: np.ndarray
    coarse_mask: np.ndarray
    scale: tuple[float, float]
    content_width: int
    content_height: int


@dataclass(frozen=True)
class OfficialEdmRuntime:
    """The single public offline EDM matcher runtime shared by every experiment."""

    matcher: Any
    torch: Any
    device: str
    input_width: int
    input_height: int
    image_resize: int
    image_divisor: int
    coarse_scale: int
    match_threshold: float
    coarse_topk: int

    def prepare_image(
        self,
        image_root: Path,
        image_name: str,
        *,
        native_width: int,
        native_height: int,
    ) -> PreparedMegadepthImage:
        return prepare_official_megadepth_image(
            image_root,
            image_name,
            self,
            native_width=native_width,
            native_height=native_height,
        )

    def match(
        self,
        query_image: PreparedMegadepthImage,
        reference_image: PreparedMegadepthImage,
        *,
        native_width: int,
        native_height: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return match_official_edm(
            self,
            query_image,
            reference_image,
            native_width=native_width,
            native_height=native_height,
        )




def select_adapter_references(
    *,
    reference_names: Sequence[str],
    reference_descriptors: np.ndarray,
    query_name: str,
    near_k: int,
    topk: int,
    exclude_query_route: bool = False,
) -> AdapterReferenceSelection:
    """Rank cached MegaLoc references after explicit reference-set exclusion.

    Default mode removes the query and near same-route frames.  Route-held-out mode
    removes *every* reference in the query route, but deliberately retains fixed M0
    landmarks; it is a reference-route ablation, not reconstruction-level LOO.
    """

    names = tuple(str(name) for name in reference_names)
    descriptors = np.asarray(reference_descriptors, dtype=np.float32)
    if len(names) != len(set(names)):
        raise ValueError("MegaLoc reference names must be unique")
    if descriptors.ndim != 2 or descriptors.shape[0] != len(names):
        raise ValueError("MegaLoc descriptor rows must align with reference names")
    try:
        query_index = names.index(query_name)
    except ValueError as exc:
        raise ValueError(f"query is absent from the MegaLoc catalog: {query_name}") from exc
    if exclude_query_route:
        query_route = parse_reference_name(query_name).sequence
        if not query_route:
            raise ValueError("route-held-out selection requires a route-qualified query name")
        excluded = tuple(
            index
            for index, name in enumerate(names)
            if parse_reference_name(name).sequence == query_route
        )
    else:
        excluded = loo_exclusion_indices(names, query_index=query_index, near_k=near_k)
    indices, scores = rank_reference_indices(
        descriptors,
        query_index=query_index,
        excluded_indices=excluded,
        topk=topk,
    )
    return AdapterReferenceSelection(
        indices=tuple(indices),
        names=tuple(names[index] for index in indices),
        scores=tuple(scores),
        excluded_indices=excluded,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("evidence_lineage") != OFFICIAL_EDM_ADAPTER_LINEAGE:
                raise ValueError(f"output contains a foreign evidence lineage: {path}")
            query_name = row.get("query_name")
            if not isinstance(query_name, str):
                raise ValueError(f"output record has no query_name: {path}")
            completed.add(query_name)
    return completed


def _pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(translation, dtype=float)
    return pose


def _camera_center_from_pose(world_to_camera: np.ndarray) -> np.ndarray:
    pose = np.asarray(world_to_camera, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("world-to-camera pose must be a finite (4, 4) matrix")
    return -pose[:3, :3].T @ pose[:3, 3]


def _megaloc_descriptor_sha256(descriptors: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(descriptors, dtype=np.float32)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _megaloc_contract_sha256(
    descriptors: np.ndarray,
    names: Sequence[str],
    metadata: Mapping[str, Any],
) -> str:
    identity = {
        "schema_name": metadata.get("schema_name"),
        "schema_version": metadata.get("schema_version"),
        "model": metadata.get("model"),
        "input_size": metadata.get("input_size"),
        "descriptor_dim": int(descriptors.shape[1]),
        "refs": len(names),
        "ref_names": list(names),
    }
    encoded_identity = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(b"sfm_system.megaloc_cache.contract.v1\0")
    digest.update(encoded_identity)
    digest.update(b"\0")
    digest.update(memoryview(np.ascontiguousarray(descriptors)).cast("B"))
    return digest.hexdigest()


def _load_reference_catalog(
    descriptor_path: Path,
    metadata_path: Path,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    """Validate stored MegaLoc payload before deriving normalized retrieval rows."""

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("MegaLoc metadata must be a JSON object")
    names_value = metadata.get("ref_names")
    if not isinstance(names_value, list) or not all(isinstance(name, str) for name in names_value):
        raise ValueError("MegaLoc metadata lacks string ref_names")
    names = tuple(names_value)
    if not names or len(names) != len(set(names)):
        raise ValueError("MegaLoc catalog names are empty or duplicated")
    expected_dim = metadata.get("descriptor_dim")
    if isinstance(expected_dim, bool) or not isinstance(expected_dim, int) or expected_dim <= 0:
        raise ValueError("MegaLoc metadata descriptor_dim must be a positive integer")
    if metadata.get("schema_name") != "sfm_system.megaloc_cache":
        raise ValueError("unsupported MegaLoc cache schema_name")
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported MegaLoc cache schema_version")
    if str(metadata.get("model", "")).lower() != "megaloc":
        raise ValueError("MegaLoc metadata model is not MegaLoc")
    input_size = metadata.get("input_size")
    if isinstance(input_size, bool) or not isinstance(input_size, int):
        raise ValueError("MegaLoc metadata input_size must be an integer")
    if input_size <= 0 or metadata.get("refs") != len(names):
        raise ValueError("MegaLoc metadata input_size or refs is invalid")
    descriptors = np.load(descriptor_path, allow_pickle=False)
    if descriptors.dtype != np.float32 or descriptors.ndim != 2:
        raise ValueError("MegaLoc descriptors must be a float32 matrix")
    if descriptors.shape != (len(names), expected_dim):
        raise ValueError("MegaLoc descriptor shape disagrees with its metadata")
    if not np.isfinite(descriptors).all():
        raise ValueError("MegaLoc descriptors contain non-finite values")
    norms = np.linalg.norm(descriptors, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("MegaLoc descriptors contain a zero or non-finite norm")
    expected_descriptor_sha256 = metadata.get("descriptor_sha256")
    if not isinstance(expected_descriptor_sha256, str) or len(expected_descriptor_sha256) != 64:
        raise ValueError("MegaLoc descriptor checksum metadata is invalid")
    if not secrets.compare_digest(
        _megaloc_descriptor_sha256(descriptors),
        expected_descriptor_sha256.lower(),
    ):
        raise ValueError("MegaLoc descriptor payload checksum does not match its sidecar")
    expected_contract_sha256 = metadata.get("contract_sha256")
    if not isinstance(expected_contract_sha256, str) or len(expected_contract_sha256) != 64:
        raise ValueError("MegaLoc contract checksum metadata is invalid")
    if not secrets.compare_digest(
        _megaloc_contract_sha256(descriptors, names, metadata),
        expected_contract_sha256.lower(),
    ):
        raise ValueError("MegaLoc contract checksum does not match its sidecar")
    normalized = descriptors / norms[:, None]
    return names, np.ascontiguousarray(normalized, dtype=np.float32), metadata


def load_b0_geometry(baseline: Path) -> B0Geometry:
    """Load only one manifest-backed ``model`` or ``reconstruction`` B0 geometry."""

    import pycolmap

    baseline = baseline.resolve(strict=True)
    manifest_path = baseline / "MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(manifest_path)
    verify_frozen_baseline(baseline)
    model_candidates = [
        baseline / name
        for name in ("model", "reconstruction")
        if (baseline / name).is_dir()
        and all(
            (baseline / name / binary).is_file() and not (baseline / name / binary).is_symlink()
            for binary in ("cameras.bin", "images.bin", "points3D.bin")
        )
    ]
    if len(model_candidates) != 1:
        raise ValueError(
            "official EDM adapter requires exactly one complete B0 model/ or reconstruction/ directory"
        )
    model_root = model_candidates[0]
    reconstruction = pycolmap.Reconstruction(str(model_root))
    if len(reconstruction.cameras) != 1:
        raise ValueError("official EDM adapter requires exactly one B0 camera")
    camera = next(iter(reconstruction.cameras.values()))
    if camera.model_name != "PINHOLE" or len(camera.params) != 4:
        raise ValueError("official EDM adapter requires a 4-parameter PINHOLE camera")
    raw_poses = read_binary_image_poses(model_root / "images.bin", strict_quaternion=False)
    raw_poses_by_name = {pose.name: pose for pose in raw_poses.values()}
    if len(raw_poses_by_name) != len(raw_poses):
        raise ValueError("B0 images.bin contains duplicate names")
    observations = {row.name: row for row in iter_binary_image_observations(model_root / "images.bin")}
    if set(observations) != set(raw_poses_by_name):
        raise ValueError("B0 observations and poses have different image identities")
    points = read_binary_points3d(model_root / "points3D.bin")
    order = np.argsort(points["point_id"], kind="stable")
    point_ids = np.asarray(points["point_id"][order], dtype=np.uint64)
    if len(point_ids) != len(np.unique(point_ids)):
        raise ValueError("B0 points3D.bin contains duplicate Point3D IDs")
    return B0Geometry(
        baseline_root=baseline,
        model_root=model_root,
        manifest_sha256=str(fingerprint_file(manifest_path, sha256=True).sha256),
        width=int(camera.width),
        height=int(camera.height),
        camera_params=tuple(float(value) for value in camera.params),
        pnp_camera=pycolmap.Camera(
            model=camera.model_name,
            width=int(camera.width),
            height=int(camera.height),
            params=list(camera.params),
        ),
        raw_poses_by_name=raw_poses_by_name,
        observations_by_name=observations,
        point_ids=point_ids,
        point_xyz=np.asarray(points["xyz"][order], dtype=np.float64),
    )


def _lookup_point_xyz(geometry: B0Geometry, point_ids: np.ndarray) -> np.ndarray:
    requested = np.asarray(point_ids, dtype=np.uint64)
    indices = np.searchsorted(geometry.point_ids, requested)
    if np.any(indices >= len(geometry.point_ids)) or not np.array_equal(
        geometry.point_ids[indices], requested
    ):
        raise ValueError("lifted Point3D ID is absent from frozen M0 points3D.bin")
    return geometry.point_xyz[indices]


def _official_megadepth_shape(
    native_width: int,
    native_height: int,
    *,
    long_edge: int,
    divisor: int,
) -> tuple[int, int, int]:
    """Mirror EDM's long-edge resize, divisible floor, and square-pad policy."""

    if min(native_width, native_height, long_edge, divisor) <= 0:
        raise ValueError("EDM image dimensions and divisor must be positive")
    scale = long_edge / max(native_width, native_height)
    content_width = round(native_width * scale) // divisor * divisor
    content_height = round(native_height * scale) // divisor * divisor
    if min(content_width, content_height) <= 0:
        raise ValueError("official EDM resize produced an empty image")
    return content_width, content_height, max(content_width, content_height)

def _activate_official_edm_dependencies() -> tuple[Path, Path]:
    """Expose audited EDM config dependencies without shadowing Torch/PyCOLMAP."""

    paths = (
        AUDITED_EDM_SITE_PACKAGES.resolve(strict=True),
        AUDITED_SYSTEM_PYTHON_PACKAGES.resolve(strict=True),
    )
    for path in paths:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return paths

def official_edm_dependency_receipt() -> dict[str, object]:
    """Record every non-canonical package path used by the one public EDM runtime."""

    activate_audited_torch()
    edm_packages, system_packages = _activate_official_edm_dependencies()
    import torch
    import yacs
    import yaml

    package_modules = {"torch": torch, "yacs": yacs, "yaml": yaml}
    packages: dict[str, dict[str, object]] = {}
    for name, module in package_modules.items():
        location = Path(str(module.__file__)).resolve()
        packages[name] = {
            "path": str(location),
            "sha256": fingerprint_file(location, sha256=True).sha256,
            "version": getattr(module, "__version__", None),
        }
    return {
        "schema_version": 1,
        "artifact_type": "OFFICIAL_EDM_RUNTIME_DEPENDENCY_RECEIPT_V1",
        "edm_site_packages": str(edm_packages),
        "system_python_packages": str(system_packages),
        "packages": packages,
    }


def load_official_edm_runtime(
    *,
    edm_repo: Path,
    checkpoint: Path,
    config_path: Path,
    data_config_path: Path,
    device: str,
) -> OfficialEdmRuntime:
    activate_audited_torch()
    _activate_official_edm_dependencies()
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if str(edm_repo) not in sys.path:
        sys.path.insert(0, str(edm_repo))
    from src.config.default import get_cfg_defaults
    from src.edm.edm import EDM
    from src.utils.misc import lower_config

    config = get_cfg_defaults()
    config.merge_from_file(str(config_path))
    config.merge_from_file(str(data_config_path))
    image_resize = int(config.EDM.TEST_RES_W)
    image_divisor = int(config.DATASET.MGDPT_DF)
    coarse_scale = int(config.EDM.LOCAL_RESOLUTION)
    if not config.DATASET.MGDPT_IMG_PAD:
        raise ValueError("official MegaDepth EDM contract requires square image padding")
    if image_resize != config.EDM.TEST_RES_H:
        raise ValueError("official MegaDepth EDM adapter requires a square test resolution")
    if image_resize % image_divisor or image_resize % coarse_scale:
        raise ValueError("official MegaDepth EDM resolution disagrees with its divisors")
    matcher = EDM(config=lower_config(config)["edm"]).eval().to(device)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    matcher.load_state_dict(state_dict)
    return OfficialEdmRuntime(
        matcher=matcher,
        torch=torch,
        device=device,
        input_width=image_resize,
        input_height=image_resize,
        image_resize=image_resize,
        image_divisor=image_divisor,
        coarse_scale=coarse_scale,
        match_threshold=float(config.EDM.COARSE.MCONF_THR),
        coarse_topk=int(config.EDM.COARSE.TOPK),
    )


def prepare_official_megadepth_image(
    image_root: Path,
    image_name: str,
    runtime: OfficialEdmRuntime,
    *,
    native_width: int,
    native_height: int,
) -> PreparedMegadepthImage:
    """Apply the packaged EDM MegaDepth preprocessing without aspect distortion."""

    path = image_root / image_name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.shape != (native_height, native_width):
        raise ValueError(f"{path} does not match the frozen M0 camera dimensions")
    content_width, content_height, pad_size = _official_megadepth_shape(
        native_width,
        native_height,
        long_edge=runtime.image_resize,
        divisor=runtime.image_divisor,
    )
    if (pad_size, pad_size) != (runtime.input_width, runtime.input_height):
        raise ValueError(
            "official EDM square padding disagrees with its configured test resolution"
        )
    resized = cv2.resize(
        image,
        (content_width, content_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pixels = np.zeros((pad_size, pad_size), dtype=resized.dtype)
    pixels[:content_height, :content_width] = resized
    coarse_mask = np.zeros(
        (pad_size // runtime.coarse_scale, pad_size // runtime.coarse_scale),
        dtype=bool,
    )
    coarse_mask[
        : content_height // runtime.coarse_scale,
        : content_width // runtime.coarse_scale,
    ] = True
    return PreparedMegadepthImage(
        pixels=pixels,
        coarse_mask=coarse_mask,
        scale=(native_width / content_width, native_height / content_height),
        content_width=content_width,
        content_height=content_height,
    )


def match_official_edm(
    runtime: OfficialEdmRuntime,
    query_image: PreparedMegadepthImage,
    reference_image: PreparedMegadepthImage,
    *,
    native_width: int,
    native_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if query_image.pixels.shape != (runtime.input_height, runtime.input_width):
        raise ValueError("query image disagrees with official EDM input contract")
    if reference_image.pixels.shape != query_image.pixels.shape:
        raise ValueError("official EDM requires equal padded query/reference dimensions")
    torch = runtime.torch
    batch = {
        "image0": torch.from_numpy(np.ascontiguousarray(query_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "image1": torch.from_numpy(np.ascontiguousarray(reference_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "mask0": torch.from_numpy(query_image.coarse_mask)[None].to(runtime.device),
        "mask1": torch.from_numpy(reference_image.coarse_mask)[None].to(runtime.device),
        "scale0": torch.tensor([query_image.scale], dtype=torch.float32, device=runtime.device),
        "scale1": torch.tensor(
            [reference_image.scale],
            dtype=torch.float32,
            device=runtime.device,
        ),
    }
    with torch.inference_mode():
        runtime.matcher(batch)
    query_points = np.asarray(batch["mkpts0_f"].detach().cpu().numpy(), dtype=np.float32)
    reference_points = np.asarray(batch["mkpts1_f"].detach().cpu().numpy(), dtype=np.float32)
    confidences = np.asarray(batch["mconf"].detach().cpu().numpy(), dtype=np.float32)
    if query_points.shape != reference_points.shape or query_points.shape != (len(confidences), 2):
        raise RuntimeError("official EDM emitted inconsistent correspondence arrays")
    for points in (query_points, reference_points):
        if not np.isfinite(points).all() or np.any(points[:, 0] < -1) or np.any(points[:, 1] < -1):
            raise RuntimeError("official EDM emitted invalid native-coordinate points")
        if np.any(points[:, 0] > native_width) or np.any(points[:, 1] > native_height):
            raise RuntimeError("official EDM emitted points outside the M0 camera contract")
    if not np.isfinite(confidences).all():
        raise RuntimeError("official EDM emitted non-finite confidences")
    return query_points, reference_points, confidences


def _pair_digest(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    confidences: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (query_points, reference_points, confidences):
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
def _estimate_absolute_pose(
    pycolmap: Any,
    image_points: np.ndarray,
    world_points: np.ndarray,
    camera: Any,
    arguments: argparse.Namespace,
) -> dict[str, Any] | None:
    if len(world_points) < 6:
        return None
    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = arguments.pnp_max_error
    options.ransac.random_seed = arguments.ransac_seed
    options.ransac.num_threads = 1
    return pycolmap.estimate_and_refine_absolute_pose(
        np.asarray(image_points, dtype=float),
        np.asarray(world_points, dtype=float),
        camera,
        options,
        return_covariance=True,
    )


_PNP_INLIER_GEOMETRY_FIELDS = (
    "status",
    "inlier_ratio",
    "reprojection_p50",
    "reprojection_p90",
    "reprojection_mean",
    "positive_depth_ratio",
    "convex_hull_coverage",
    "occupancy_4x4",
    "scaled_fim_rank",
    "scaled_fim_lambda_min",
    "scaled_fim_condition",
    "scaled_fim_logdet",
    "scaled_pose_covariance_diag",
    "scaled_fim_degenerate",
    "scene_scale",
)


def _pnp_inlier_geometry(
    world_points: np.ndarray,
    image_points: np.ndarray,
    inlier_mask: np.ndarray,
    estimated_pose: np.ndarray,
    *,
    camera_params: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    """Measure only the PnP inlier geometry, without an M0 query-pose comparison."""

    metrics = compute_loo_metrics(
        world_points,
        image_points,
        inlier_mask,
        camera_params,
        estimated_pose,
        estimated_pose,
        image_size=image_size,
    )
    return {field: metrics[field] for field in _PNP_INLIER_GEOMETRY_FIELDS}


def _summarize_reference_pose_consensus(
    poses: Sequence[np.ndarray],
    *,
    scene_scale: float | None,
    thresholds: AdapterThresholds,
) -> dict[str, Any]:
    """Compare independently estimated reference poses without using the M0 query pose."""

    valid_poses = [
        np.asarray(pose, dtype=float)
        for pose in poses
        if np.asarray(pose).shape == (4, 4) and np.isfinite(pose).all()
    ]
    if len(valid_poses) < 2:
        return {
            "classification": "INSUFFICIENT_INDEPENDENT_REFERENCE_POSES",
            "eligible_reference_count": len(valid_poses),
            "pair_count": 0,
            "position_disagreement_p50": None,
            "position_disagreement_p90": None,
            "normalized_position_disagreement_p90": None,
            "rotation_error_deg_p50": None,
            "rotation_error_deg_p90": None,
        }

    position_disagreements: list[float] = []
    rotation_disagreements: list[float] = []
    for index, first in enumerate(valid_poses):
        first_center = _camera_center_from_pose(first)
        for second in valid_poses[index + 1 :]:
            second_center = _camera_center_from_pose(second)
            position_disagreements.append(float(np.linalg.norm(first_center - second_center)))
            rotation_delta = first[:3, :3] @ second[:3, :3].T
            rotation_disagreements.append(
                float(
                    np.degrees(
                        np.arccos(
                            np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
                        )
                    )
                )
            )
    position_p90 = float(np.quantile(position_disagreements, 0.9))
    rotation_p90 = float(np.quantile(rotation_disagreements, 0.9))
    normalized_position_p90 = (
        position_p90 / scene_scale
        if isinstance(scene_scale, (int, float)) and scene_scale > 0
        else None
    )
    classification = "SCENE_SCALE_UNAVAILABLE"
    if normalized_position_p90 is not None:
        classification = (
            "REFERENCE_POSES_CONSISTENT"
            if normalized_position_p90 <= thresholds.max_normalized_position_error
            and rotation_p90 <= thresholds.max_rotation_error_deg
            else "REFERENCE_POSES_DISAGREE"
        )
    return {
        "classification": classification,
        "eligible_reference_count": len(valid_poses),
        "pair_count": len(position_disagreements),
        "position_disagreement_p50": float(np.median(position_disagreements)),
        "position_disagreement_p90": position_p90,
        "normalized_position_disagreement_p90": normalized_position_p90,
        "rotation_error_deg_p50": float(np.median(rotation_disagreements)),
        "rotation_error_deg_p90": rotation_p90,
    }


def _reference_pose_probes(
    *,
    pycolmap: Any,
    image_points: np.ndarray,
    world_points: np.ndarray,
    reference_sources: np.ndarray,
    selected_reference_names: Sequence[str],
    geometry: B0Geometry,
    arguments: argparse.Namespace,
    ground_truth_pose: np.ndarray | None,
    scene_scale: float | None,
) -> dict[str, Any]:
    """Attribute PnP stability to each retrieved reference; diagnostic evidence only."""

    thresholds = AdapterThresholds(min_pnp_inliers=arguments.min_inliers)
    probes: list[dict[str, Any]] = []
    eligible_poses: list[np.ndarray] = []
    for reference_name in selected_reference_names:
        indices = np.flatnonzero(reference_sources == reference_name)
        if not len(indices):
            probes.append(
                {
                    "reference_name": reference_name,
                    "pnp_correspondences": 0,
                    "pnp_succeeded": False,
                    "pnp_inliers": 0,
                    "eligible_for_consensus": False,
                    "map_relative_pose": {
                        "status": "NOT_PROBED_AFTER_AGGREGATE_CORRESPONDENCE_CAP"
                    },
                    "pnp_inlier_geometry": {
                        "status": "NOT_PROBED_AFTER_AGGREGATE_CORRESPONDENCE_CAP"
                    },
                }
            )
            continue
        estimate = _estimate_absolute_pose(
            pycolmap,
            image_points[indices],
            world_points[indices],
            geometry.pnp_camera,
            arguments,
        )
        if estimate is None:
            probes.append(
                {
                    "reference_name": reference_name,
                    "pnp_correspondences": len(indices),
                    "pnp_succeeded": False,
                    "pnp_inliers": 0,
                    "eligible_for_consensus": False,
                    "map_relative_pose": {"status": "PNP_UNSOLVED"},
                    "pnp_inlier_geometry": {"status": "PNP_UNSOLVED"},
                }
            )
            continue
        estimated_pose = _pose_matrix(
            estimate["cam_from_world"].rotation.matrix(),
            estimate["cam_from_world"].translation,
        )
        inlier_mask = np.asarray(estimate["inlier_mask"], dtype=bool)
        num_inliers = int(estimate["num_inliers"])
        pnp_inlier_geometry = _pnp_inlier_geometry(
            world_points[indices],
            image_points[indices],
            inlier_mask,
            estimated_pose,
            camera_params=geometry.camera_params,
            image_size=(geometry.width, geometry.height),
        )
        map_relative_pose: dict[str, Any] = {"status": "GROUND_TRUTH_UNAVAILABLE"}
        if ground_truth_pose is not None:
            metrics = compute_loo_metrics(
                world_points[indices],
                image_points[indices],
                inlier_mask,
                geometry.camera_params,
                estimated_pose,
                ground_truth_pose,
                np.full(len(indices), reference_name, dtype=object),
                image_size=(geometry.width, geometry.height),
            )
            position_error = metrics.get("position_error")
            probe_scene_scale = metrics.get("scene_scale")
            normalized_position_error = (
                float(position_error / probe_scene_scale)
                if isinstance(position_error, (int, float))
                and isinstance(probe_scene_scale, (int, float))
                and probe_scene_scale > 0
                else None
            )
            rotation_error_deg = metrics.get("rotation_error_deg")
            map_relative_pose = {
                "status": (
                    "MAP_RELATIVE_SUPPORTED"
                    if num_inliers >= thresholds.min_pnp_inliers
                    and isinstance(normalized_position_error, float)
                    and normalized_position_error <= thresholds.max_normalized_position_error
                    and isinstance(rotation_error_deg, (int, float))
                    and rotation_error_deg <= thresholds.max_rotation_error_deg
                    else "MAP_RELATIVE_UNSUPPORTED"
                ),
                "normalized_position_error": normalized_position_error,
                "rotation_error_deg": rotation_error_deg,
            }
        eligible = num_inliers >= thresholds.min_pnp_inliers
        if eligible:
            eligible_poses.append(estimated_pose)
        probes.append(
            {
                "reference_name": reference_name,
                "pnp_correspondences": len(indices),
                "pnp_succeeded": True,
                "pnp_inliers": num_inliers,
                "eligible_for_consensus": eligible,
                "map_relative_pose": map_relative_pose,
                "pnp_inlier_geometry": pnp_inlier_geometry,
                "estimated_map_pose": {
                    "camera_center": _camera_center_from_pose(estimated_pose).tolist(),
                    "world_to_camera_rotation": estimated_pose[:3, :3].tolist(),
                },
            }
        )
    return {
        "evidence_scope": ADAPTER_EVIDENCE_SCOPE,
        "comparison_standard": "independent_per_reference_pnp_pose_consensus",
        "per_reference": probes,
        "consensus": _summarize_reference_pose_consensus(
            eligible_poses,
            scene_scale=scene_scale,
            thresholds=thresholds,
        ),
    }



def _fingerprint_python_paths(
    root: Path,
    relative_paths: Sequence[str],
    *,
    scope: str,
) -> SourceTreeFingerprint:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"official EDM source root is not a directory: {resolved}")
    files = tuple(sorted(relative_paths))
    if not files:
        raise ValueError(f"official EDM source set contains no Python files: {resolved}")
    if len(files) != len(set(files)):
        raise ValueError("official EDM source set contains duplicate paths")
    digest = hashlib.sha256(
        f"river_map_quality.official_edm.python_{scope}.v1\0".encode("ascii")
    )
    for relative_path in files:
        path = resolved / relative_path
        if (
            Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            or path.suffix != ".py"
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(f"invalid official EDM Python source: {path}")
        fingerprint = fingerprint_file(path, sha256=True)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(fingerprint.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(fingerprint.sha256.encode("ascii"))
        digest.update(b"\0")
    return SourceTreeFingerprint(
        path=str(resolved),
        files=files,
        sha256=digest.hexdigest(),
        scope=scope,
    )


def fingerprint_python_source_tree(root: Path) -> SourceTreeFingerprint:
    """Fingerprint every importable Python source in an official EDM checkout."""

    resolved = root.resolve(strict=True)
    relative_paths = [
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
    ]
    return _fingerprint_python_paths(resolved, relative_paths, scope="tree")


def fingerprint_python_source_files(
    root: Path,
    relative_paths: Sequence[str],
) -> SourceTreeFingerprint:
    """Fingerprint the reviewed local adapter dependency closure, not its package."""

    return _fingerprint_python_paths(root, relative_paths, scope="closure")

def _verified_adapter_source_commit(
    adapter_source_root: Path,
    relative_paths: Sequence[str],
) -> str:
    """Require the fingerprinted local closure to match one Git commit."""

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(adapter_source_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    repository = git("rev-parse", "--show-toplevel")
    if repository.returncode != 0:
        raise ValueError(f"adapter source is not in a Git repository: {adapter_source_root}")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--", *relative_paths)
    if status.returncode != 0:
        raise ValueError(f"cannot inspect adapter source status: {status.stderr.strip()}")
    if status.stdout:
        raise ValueError(
            "fingerprinted adapter source closure must be clean before execution: "
            f"{status.stdout.strip()}"
        )
    revision = git("rev-parse", "--verify", "HEAD")
    if revision.returncode != 0 or not revision.stdout.strip():
        raise ValueError(f"cannot resolve adapter source commit: {adapter_source_root}")
    return revision.stdout.strip()


def _source_fingerprints(
    *,
    descriptor_path: Path,
    metadata_path: Path,
    checkpoint: Path,
    config_path: Path,
    data_config_path: Path,
    edm_repo: Path,
    adapter_source_root: Path,
) -> dict[str, FileFingerprint | SourceTreeFingerprint]:
    return {
        "megaloc_descriptors": fingerprint_file(descriptor_path, sha256=True),
        "megaloc_metadata": fingerprint_file(metadata_path, sha256=True),
        "official_edm_checkpoint": fingerprint_file(checkpoint, sha256=True),
        "official_edm_config": fingerprint_file(config_path, sha256=True),
        "official_edm_data_config": fingerprint_file(data_config_path, sha256=True),
        "official_edm_python_source_tree": fingerprint_python_source_tree(edm_repo),
        "adapter_python_source_tree": fingerprint_python_source_files(
            adapter_source_root,
            ADAPTER_LOCAL_SOURCE_FILES,
        ),
    }


def _manifest_payload(
    *,
    baseline: Path,
    source_fingerprints: Mapping[str, FileFingerprint | SourceTreeFingerprint],
    adapter_source_commit: str,
    metadata: Mapping[str, Any],
    arguments: argparse.Namespace,
    runtime: OfficialEdmRuntime,
    queries: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "evidence_lineage": OFFICIAL_EDM_ADAPTER_LINEAGE,
        "evidence_scope": ADAPTER_EVIDENCE_SCOPE,
        "is_production_hl": False,
        "baseline": str(baseline),
        "source_fingerprints": {
            name: fingerprint.as_dict() for name, fingerprint in source_fingerprints.items()
        },
        "adapter_source_commit": adapter_source_commit,
        "megaloc_contract": {
            "model": metadata.get("model"),
            "input_size": metadata.get("input_size"),
            "descriptor_dim": metadata.get("descriptor_dim"),
            "descriptor_sha256": metadata.get("descriptor_sha256"),
            "contract_sha256": metadata.get("contract_sha256"),
        },
        "adapter_config": {
            "topk": arguments.topk,
            "near_k": arguments.near_k,
            "reference_exclusion_mode": (
                "WHOLE_QUERY_ROUTE"
                if arguments.exclude_query_route
                else "QUERY_AND_NEAR_SAME_ROUTE"
            ),
            "lift_distance_px": arguments.lift_distance_px,
            "query_conflict_distance_px": arguments.query_conflict_distance_px,
            "max_correspondences": arguments.max_correspondences,
            "cap_grid": arguments.cap_grid,
            "pnp_max_error": arguments.pnp_max_error,
            "ransac_seed": arguments.ransac_seed,
            "min_inliers": arguments.min_inliers,
            "reference_pose_probes": arguments.reference_pose_probes,
            "official_edm_preprocess": {
                "mode": "megadepth_long_edge_divisible_then_square_pad",
                "resize_long_edge": runtime.image_resize,
                "resize_divisor": runtime.image_divisor,
                "padded_input_size": [runtime.input_width, runtime.input_height],
                "coarse_mask_scale": runtime.coarse_scale,
                "interpolation": "cv2.INTER_LINEAR",
                "match_threshold": runtime.match_threshold,
                "coarse_topk": runtime.coarse_topk,
            },
        },
        "queries": list(queries),
        "fixed_m0_landmarks": True,
        "observation_level_reconstruction_loo": False,
        "map_relative_pose_only": True,
    }


def _queries(arguments: argparse.Namespace, names: Sequence[str]) -> tuple[str, ...]:
    requested: list[str] = list(arguments.query or [])
    if arguments.queries_file is not None:
        for line in arguments.queries_file.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                requested.append(value)
    if arguments.all_queries:
        requested.extend(names)
    if not requested:
        raise ValueError("provide --query, --queries-file, or explicit --all-queries")
    unique = tuple(dict.fromkeys(requested))
    unknown = sorted(set(unique) - set(names))
    if unknown:
        raise ValueError(f"queries absent from MegaLoc catalog: {unknown!r}")
    return unique


def _record_query(
    *,
    query_name: str,
    reference_names: Sequence[str],
    reference_descriptors: np.ndarray,
    geometry: B0Geometry,
    runtime: OfficialEdmRuntime,
    image_root: Path,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    import pycolmap

    started = time.perf_counter()
    selection = select_adapter_references(
        reference_names=reference_names,
        reference_descriptors=reference_descriptors,
        query_name=query_name,
        near_k=arguments.near_k,
        topk=arguments.topk,
        exclude_query_route=arguments.exclude_query_route,
    )
    query_image = runtime.prepare_image(
        image_root,
        query_name,
        native_width=geometry.width,
        native_height=geometry.height,
    )
    raw_match_count = 0
    unmapped_match_count = 0
    lifted_matches = []
    per_reference: list[dict[str, Any]] = []
    for reference_name, retrieval_score in zip(selection.names, selection.scores, strict=True):
        reference_image = runtime.prepare_image(
            image_root,
            reference_name,
            native_width=geometry.width,
            native_height=geometry.height,
        )
        query_points, reference_points, confidences = runtime.match(
            query_image,
            reference_image,
            native_width=geometry.width,
            native_height=geometry.height,
        )
        raw_match_count += len(query_points)
        observations = geometry.observations_by_name.get(reference_name)
        if observations is None:
            unmapped_match_count += len(query_points)
            per_reference.append(
                {
                    "reference_name": reference_name,
                    "retrieval_score": retrieval_score,
                    "raw_matches": len(query_points),
                    "lifted_matches": 0,
                    "unmapped_matches": len(query_points),
                    "m0_observation_status": "MISSING",
                    "match_sha256": _pair_digest(query_points, reference_points, confidences),
                }
            )
            continue
        lifted = lift_reference_matches(
            query_points=query_points,
            reference_points=reference_points,
            observation_points=observations.xy,
            observation_point3d_ids=observations.point3d_ids,
            confidences=confidences,
            maximum_distance_px=arguments.lift_distance_px,
            reference_name=reference_name,
        )
        lifted_matches.extend(lifted.matches)
        unmapped_match_count += lifted.unmapped_match_count
        per_reference.append(
            {
                "reference_name": reference_name,
                "retrieval_score": retrieval_score,
                "raw_matches": len(query_points),
                "lifted_matches": len(lifted.matches),
                "unmapped_matches": lifted.unmapped_match_count,
                "m0_observation_status": "AVAILABLE",
                "match_sha256": _pair_digest(query_points, reference_points, confidences),
            }
        )

    deduplicated = deduplicate_lifted_matches(
        lifted_matches,
        query_conflict_distance_px=arguments.query_conflict_distance_px,
    )
    all_query_points = np.asarray(
        [match.query_xy for match in deduplicated.matches],
        dtype=float,
    ).reshape(-1, 2)
    all_point_ids = np.asarray(
        [match.point3d_id for match in deduplicated.matches],
        dtype=np.uint64,
    )
    selected = spatial_cap_indices(
        all_query_points,
        max_total=arguments.max_correspondences,
        width=geometry.width,
        height=geometry.height,
        grid=arguments.cap_grid,
    )
    pnp_points2d = all_query_points[selected]
    pnp_point_ids = all_point_ids[selected]
    pnp_points3d = _lookup_point_xyz(geometry, pnp_point_ids)
    pnp_sources = np.asarray(
        [deduplicated.matches[index].reference_name for index in selected],
        dtype=object,
    )

    estimate = _estimate_absolute_pose(
        pycolmap,
        pnp_points2d,
        pnp_points3d,
        geometry.pnp_camera,
        arguments,
    )
    raw_pose = geometry.raw_poses_by_name.get(query_name)
    ground_truth_valid = bool(
        raw_pose and raw_pose.valid_rigid_transform and raw_pose.rotation is not None
    )
    ground_truth_pose = (
        _pose_matrix(raw_pose.rotation, raw_pose.tvec) if ground_truth_valid else None
    )
    pose_metrics: dict[str, Any] | None = None
    estimated_map_pose: dict[str, Any] | None = None
    num_inliers = 0
    if estimate is not None:
        transform = estimate["cam_from_world"]
        estimated_pose = _pose_matrix(transform.rotation.matrix(), transform.translation)
        estimated_map_pose = {
            "camera_center": _camera_center_from_pose(estimated_pose).tolist(),
            "world_to_camera_rotation": estimated_pose[:3, :3].tolist(),
        }
        inlier_mask = np.asarray(estimate["inlier_mask"], dtype=bool)
        num_inliers = int(estimate["num_inliers"])
        truth = ground_truth_pose if ground_truth_pose is not None else estimated_pose
        pose_metrics = compute_loo_metrics(
            pnp_points3d,
            pnp_points2d,
            inlier_mask,
            geometry.camera_params,
            estimated_pose,
            truth,
            pnp_sources,
            image_size=(geometry.width, geometry.height),
        )
        if not ground_truth_valid:
            pose_metrics["position_error"] = None
            pose_metrics["rotation_error_deg"] = None
            pose_metrics["ground_truth_pose_evaluation"] = "UNAVAILABLE"
    normalized_position_error = None
    rotation_error_deg = None
    scene_scale: float | None = None
    if pose_metrics is not None:
        value = pose_metrics.get("scene_scale")
        if isinstance(value, (int, float)):
            scene_scale = float(value)
        position_error = pose_metrics.get("position_error")
        if (
            scene_scale is not None
            and isinstance(position_error, (int, float))
            and scene_scale > 0
        ):
            normalized_position_error = float(position_error / scene_scale)
        value = pose_metrics.get("rotation_error_deg")
        if isinstance(value, (int, float)):
            rotation_error_deg = float(value)
    thresholds = AdapterThresholds(min_pnp_inliers=arguments.min_inliers)
    assessment = assess_adapter_evidence(
        AdapterEvidence(
            completed=True,
            retrieved_refs=len(selection.names),
            raw_matches=raw_match_count,
            lifted_correspondences=len(pnp_points3d),
            pnp_succeeded=estimate is not None,
            pnp_inliers=num_inliers,
            ground_truth_pose_valid=ground_truth_valid,
            normalized_position_error=normalized_position_error,
            rotation_error_deg=rotation_error_deg,
        ),
        thresholds,
    )
    reference_pose_probes = (
        _reference_pose_probes(
            pycolmap=pycolmap,
            image_points=pnp_points2d,
            world_points=pnp_points3d,
            reference_sources=pnp_sources,
            selected_reference_names=selection.names,
            geometry=geometry,
            arguments=arguments,
            ground_truth_pose=ground_truth_pose,
            scene_scale=scene_scale,
        )
        if arguments.reference_pose_probes
        else None
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "evidence_lineage": OFFICIAL_EDM_ADAPTER_LINEAGE,
        "evidence_scope": ADAPTER_EVIDENCE_SCOPE,
        "is_production_hl": False,
        "confirms_production_localization": False,
        "query_name": query_name,
        "query_index": reference_names.index(query_name),
        "query_descriptor_source": "MEGALOC_REFERENCE_CACHE_ROW",
        "leave_one_out": {
            "near_k": arguments.near_k,
            "reference_exclusion_mode": (
                "WHOLE_QUERY_ROUTE"
                if arguments.exclude_query_route
                else "QUERY_AND_NEAR_SAME_ROUTE"
            ),
            "excluded_route": (
                parse_reference_name(query_name).sequence
                if arguments.exclude_query_route
                else None
            ),
            "excluded_indices": list(selection.excluded_indices),
            "excluded_names": [reference_names[index] for index in selection.excluded_indices],
            "retrieved_indices": list(selection.indices),
            "retrieved_names": list(selection.names),
            "retrieval_scores": list(selection.scores),
            "fixed_m0_landmarks": True,
            "observation_level_reconstruction_loo": False,
            "route_loo_limitation": (
                "reference-route exclusion only; M0 landmark tracks remain fixed"
                if arguments.exclude_query_route
                else None
            ),
        },
        "official_edm": {
            "preprocess": "megadepth_long_edge_divisible_then_square_pad",
            "model_input_size": [runtime.input_width, runtime.input_height],
            "resized_content_size": [
                query_image.content_width,
                query_image.content_height,
            ],
            "coarse_mask_size": list(query_image.coarse_mask.shape[::-1]),
            "native_camera_size": [geometry.width, geometry.height],
            "native_coordinate_scale": list(query_image.scale),
            "interpolation": "cv2.INTER_LINEAR",
            "official_config_match_threshold": runtime.match_threshold,
            "official_config_coarse_topk": runtime.coarse_topk,
        },
        "adapter_raw_matches": raw_match_count,
        "adapter_lifted_matches": len(lifted_matches),
        "adapter_unmapped_matches": unmapped_match_count,
        "adapter_conflicting_query_matches": deduplicated.conflicting_query_match_count,
        "adapter_duplicate_point3d_matches": deduplicated.duplicate_point3d_match_count,
        "adapter_pnp_correspondences_before_cap": len(deduplicated.matches),
        "adapter_pnp_correspondences": len(pnp_points3d),
        "adapter_pnp_succeeded": estimate is not None,
        "adapter_pnp_inliers": num_inliers,
        "adapter_pose_metrics": pose_metrics,
        "adapter_estimated_map_pose": estimated_map_pose,
        "ground_truth_pose_valid": ground_truth_valid,
        "adapter_assessment": asdict(assessment),
        "per_reference": per_reference,
        "runtime_seconds": time.perf_counter() - started,
    }
    if reference_pose_probes is not None:
        record["reference_pose_probes"] = reference_pose_probes
    return record


def run(arguments: argparse.Namespace) -> int:
    baseline = arguments.baseline.resolve(strict=True)
    descriptor_path = arguments.megaloc_descriptors.resolve(strict=True)
    metadata_path = arguments.megaloc_metadata.resolve(strict=True)
    edm_repo = arguments.edm_repo.resolve(strict=True)
    checkpoint = arguments.edm_checkpoint.resolve(strict=True)
    config_path = (edm_repo / arguments.edm_config).resolve(strict=True)
    data_config_path = (edm_repo / arguments.edm_data_config).resolve(strict=True)
    image_root = (arguments.images_root or baseline / "images").resolve(strict=True)
    if not (edm_repo / "src/edm/edm.py").is_file():
        raise ValueError("edm_repo must contain the official src/edm/edm.py")
    if not image_root.is_dir():
        raise ValueError("images_root must be a directory")
    verify_frozen_baseline(baseline)
    adapter_source_root = Path(__file__).resolve().parent
    source_fingerprints = _source_fingerprints(
        descriptor_path=descriptor_path,
        metadata_path=metadata_path,
        checkpoint=checkpoint,
        config_path=config_path,
        data_config_path=data_config_path,
        edm_repo=edm_repo,
        adapter_source_root=adapter_source_root,
    )
    adapter_source_commit = _verified_adapter_source_commit(
        adapter_source_root,
        ADAPTER_LOCAL_SOURCE_FILES,
    )
    reference_names, descriptors, metadata = _load_reference_catalog(
        descriptor_path,
        metadata_path,
    )
    queries = _queries(arguments, reference_names)
    missing_images = [name for name in queries if not (image_root / name).is_file()]
    if missing_images:
        raise FileNotFoundError(f"query images are missing: {missing_images!r}")
    missing_reference_images = [
        name for name in reference_names if not (image_root / name).is_file()
    ]
    if missing_reference_images:
        raise FileNotFoundError(
            f"MegaLoc reference images are missing: {missing_reference_images[:3]!r}"
        )
    geometry = load_b0_geometry(baseline)
    runtime = load_official_edm_runtime(
        edm_repo=edm_repo,
        checkpoint=checkpoint,
        config_path=config_path,
        data_config_path=data_config_path,
        device=arguments.device,
    )
    output_dir = arguments.output_dir
    manifest_path = output_dir / "manifest.json"
    records_path = output_dir / "records.jsonl"
    manifest = _manifest_payload(
        baseline=baseline,
        adapter_source_commit=adapter_source_commit,
        source_fingerprints=source_fingerprints,
        metadata=metadata,
        arguments=arguments,
        runtime=runtime,
        queries=queries,
    )
    if manifest_path.exists():
        if not arguments.resume:
            raise FileExistsError(f"adapter output already exists: {output_dir}; pass --resume")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("adapter output manifest does not match the current input contract")
    else:
        if records_path.exists():
            raise ValueError("adapter records exist without a manifest")
        _atomic_json(manifest_path, manifest)
    completed = _read_completed(records_path) if arguments.resume else set()
    for index, query_name in enumerate(queries, 1):
        if query_name in completed:
            continue
        record = _record_query(
            query_name=query_name,
            reference_names=reference_names,
            reference_descriptors=descriptors,
            geometry=geometry,
            runtime=runtime,
            image_root=image_root,
            arguments=arguments,
        )
        _append_jsonl(records_path, record)
        print(
            f"[{index}/{len(queries)}] {query_name} "
            f"status={record['adapter_assessment']['status']} "
            f"inliers={record['adapter_pnp_inliers']}",
            flush=True,
        )
        if arguments.device == "cuda":
            runtime.torch.cuda.empty_cache()
    verify_frozen_baseline(baseline)
    for fingerprint in source_fingerprints.values():
        fingerprint.assert_unchanged()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an official-EDM + MegaLoc + M0-COLMAP diagnostic adapter; "
            "not a production-EDM localizer."
        )
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--megaloc-descriptors", required=True, type=Path)
    parser.add_argument("--megaloc-metadata", required=True, type=Path)
    parser.add_argument("--edm-repo", required=True, type=Path)
    parser.add_argument("--edm-checkpoint", required=True, type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query", action="append")
    parser.add_argument("--queries-file", type=Path)
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--near-k", type=int, default=3)
    parser.add_argument(
        "--exclude-query-route",
        action="store_true",
        help=(
            "exclude every reference from each query route; this is a fixed-landmark "
            "reference-route ablation, not reconstruction-level leave-one-out"
        ),
    )
    parser.add_argument("--lift-distance-px", type=float, default=1.5)
    parser.add_argument("--query-conflict-distance-px", type=float, default=1.0)
    parser.add_argument("--max-correspondences", type=int, default=1200)
    parser.add_argument("--cap-grid", type=int, default=8)
    parser.add_argument("--pnp-max-error", type=float, default=5.0)
    parser.add_argument("--ransac-seed", type=int, default=0)
    parser.add_argument("--min-inliers", type=int, default=80)
    parser.add_argument(
        "--reference-pose-probes",
        action="store_true",
        help="emit per-reference PnP consensus diagnostics; never production evidence",
    )
    parser.add_argument("--edm-config", type=Path, default=Path("configs/edm/outdoor/edm_base.py"))
    parser.add_argument(
        "--edm-data-config",
        type=Path,
        default=Path("configs/data/megadepth_test_1500.py"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.topk <= 0 or arguments.near_k < 0:
        raise ValueError("topk must be positive and near-k must be non-negative")
    if min(arguments.max_correspondences, arguments.cap_grid, arguments.min_inliers) <= 0:
        raise ValueError("cap and inlier counts must be positive")
    if (
        min(
            arguments.lift_distance_px,
            arguments.query_conflict_distance_px,
            arguments.pnp_max_error,
        )
        <= 0
    ):
        raise ValueError("adapter distance thresholds must be positive")
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
