"""Hash-bound, offline-only COLMAP stereo using an immutable camera solution.

Only image_undistorter, the two PatchMatch passes and optional stereo_fusion
are executable stages. Pixel-convention conversion is a representation change,
not calibration: poses, world points and native rays are never optimized.
Artifacts are supplemental map-relative measurements, never external truth or
strong-localization geometry. A failed/partial staging directory is not readable.

COLMAP formats/conventions: https://colmap.github.io/format.html and
https://colmap.github.io/faq.html#using-calibration-from-opencv-kalibr-or-other-tools
The installed 4.0.4 mvs/mat.h stores planar channels, row-major within a plane;
this is equivalent to the official Python Fortran reshape + transpose.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters import ExclusiveResourceLease
from .geometry_contract import validate_mapping_record, validate_pinhole, validate_pose
from .inventory import sha256_file

REQUEST_SCHEMA = "offline-fixed-pose-mvs-request/v1"
RECEIPT_SCHEMA = "offline-fixed-pose-mvs-receipt/v1"
SUPPORTED_COLMAP_VERSION = "4.0.4"
COLMAP_SOURCE_REVISION = "9c23f6942fe69962e06030905e77067c8673382f"
MAPPING_SCOPE = "REGISTERED_NATIVE_MAPPING_ALLOWLIST"
MAPPING_HOLDOUT_SCOPE = "MAPPING_INTERNAL_BLOCKED_VIEW_CELL_SESSION"
OPENCV_NATIVE_CENTER = "OPENCV_NATIVE_PIXEL_CENTER_0"
COLMAP_CORNER = "COLMAP_TOP_LEFT_CORNER_PIXEL_CENTER_0.5"
UNKNOWN, AGREE, CONFLICT, OCCLUDED = 0, 1, 2, 3
MVS_FAMILY = "FIXED_POSE_COLMAP_MVS"
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
OPTIONAL_MODEL_FILES = ("rigs.bin", "frames.bin")
SHIFT_TO_COLMAP = np.array([[1., 0., .5], [0., 1., .5], [0., 0., 1.]])
SHIFT_TO_NATIVE = np.array([[1., 0., -.5], [0., 1., -.5], [0., 0., 1.]])


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_relative_name(name: str) -> Path:
    relative = Path(name)
    if not relative.parts:
        raise ValueError(f"unsafe artifact/image name: {name!r}")
    if (not name or relative.is_absolute() or relative.as_posix() != name
            or any(part in (".", "..") for part in relative.parts)
            or any(char in name for char in "\r\n\0,")):
        raise ValueError(f"unsafe artifact/image name: {name!r}")
    return relative


def frozen_model_hashes(model_dir: Path | str) -> dict[str, str]:
    root = Path(model_dir).resolve(strict=True)
    for name in MODEL_FILES:
        if not (root / name).is_file():
            raise ValueError(f"incomplete frozen binary model: {name}")
    return {name: sha256_file(root / name) for name in (*MODEL_FILES, *OPTIONAL_MODEL_FILES)
            if (root / name).is_file()}


def pixel_center_intrinsics(k: np.ndarray, *, from_convention: str,
                            to_convention: str) -> np.ndarray:
    matrix = np.asarray(k, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.array_equal(matrix[2], [0., 0., 1.])
            or matrix[0, 1] != 0 or matrix[1, 0] != 0):
        raise ValueError("expected finite fixed PINHOLE intrinsics")
    conventions = {OPENCV_NATIVE_CENTER, COLMAP_CORNER}
    if from_convention not in conventions or to_convention not in conventions:
        raise ValueError("pixel convention must be explicitly declared, never inferred")
    if from_convention == to_convention:
        return matrix.copy()
    return (SHIFT_TO_COLMAP if to_convention == COLMAP_CORNER else SHIFT_TO_NATIVE) @ matrix


@dataclass(frozen=True)
class FrozenCapture:
    image_id: int
    name: str
    image_path: str
    source_image_uri: str
    image_sha256: str
    session_id: str
    frame_index: int
    evaluation_role: str
    camera_id: int
    width: int
    height: int
    K_frozen: list[list[float]]
    K_native: list[list[float]]
    K_colmap: list[list[float]]
    cam_from_world: list[list[float]]

    @property
    def center(self) -> np.ndarray:
        pose = np.asarray(self.cam_from_world)
        return -pose[:, :3].T @ pose[:, 3]


def validate_mapping_capture(capture: FrozenCapture) -> None:
    """Check native MAPPING identity; FrozenScene separately binds its allowlist."""
    safe_relative_name(capture.name)
    validate_mapping_record({"name": capture.name, "session_id": capture.session_id,
                             "frame_index": capture.frame_index,
                             "image_sha256": capture.image_sha256,
                             "evaluation_role": capture.evaluation_role})
    for k in (capture.K_frozen, capture.K_native, capture.K_colmap):
        validate_pinhole(k, capture.width, capture.height)
    pose = np.asarray(capture.cam_from_world, dtype=np.float64)
    if pose.shape != (3, 4):
        raise ValueError("capture must record its proper frozen 3x4 world-to-camera pose")
    validate_pose(pose)


@dataclass(frozen=True)
class FrozenScene:
    model_dir: Path
    keyframes_manifest: Path
    image_root: Path
    model_hashes: Mapping[str, str]
    keyframes_sha256: str
    input_pixel_convention: str
    captures: Mapping[int, FrozenCapture]


def validate_mapping_source(scene: FrozenScene, capture: FrozenCapture) -> None:
    """Require a validated scene member at its exact native allowlist JPEG path."""
    validate_mapping_capture(capture)
    source = Path(capture.image_path)
    if (scene.captures.get(capture.image_id) != capture
            or scene.image_root.resolve(strict=True) != (scene.keyframes_manifest.parent / "images").resolve(strict=True)
            or source.is_symlink() or not source.resolve(strict=True).is_relative_to(scene.image_root)
            or source.resolve(strict=True) !=
            (scene.image_root / safe_relative_name(capture.name)).resolve(strict=True)):
        raise ValueError("SIFT source is not in the validated registered MAPPING JPEG allowlist")
    if (sha256_file(scene.keyframes_manifest) != scene.keyframes_sha256
            or frozen_model_hashes(scene.model_dir) != dict(scene.model_hashes)):
        raise ValueError("registered MAPPING allowlist changed before SIFT")
    entries = {}
    for line in scene.keyframes_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row["output_name"])
        if name in entries or row.get("evaluation_role") != "MAPPING":
            raise ValueError("SIFT needs an exact, all-MAPPING registered manifest")
        entries[name] = row
    registered = colmap_image_order(scene.model_dir)
    if (set(entries) != {name for _, name in registered}
            or (capture.image_id, capture.name) not in registered):
        raise ValueError("SIFT source is not a registered MAPPING manifest capture")
    row = entries[capture.name]
    if (row["image_uri"] != capture.source_image_uri
            or Path(str(row["image_uri"])).resolve(strict=True) != Path(capture.image_path).resolve(strict=True)
            or row.get("image_sha256") != capture.image_sha256
            or row.get("session_id") != capture.session_id
            or int(row["frame_index"]) != capture.frame_index):
        raise ValueError("SIFT source path/hash is not the authoritative MAPPING manifest row")


def load_frozen_scene(model_dir: Path | str, keyframes_manifest: Path | str,
                      image_root: Path | str, *, input_pixel_convention: str,
                      image_ids: Sequence[int] | None = None) -> FrozenScene:
    """Read native MAPPING captures; no query, resized surrogate or depth input."""
    import cv2
    import pycolmap

    model = Path(model_dir).resolve(strict=True)
    manifest = Path(keyframes_manifest).resolve(strict=True)
    images = Path(image_root).resolve(strict=True)
    if images != (manifest.parent / "images").resolve(strict=True):
        raise ValueError("only the manifest's exact native images directory is allowed")
    hashes = frozen_model_hashes(model)
    manifest_sha = sha256_file(manifest)
    entries: dict[str, dict[str, Any]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        name = str(row["output_name"])
        safe_relative_name(name)
        validate_mapping_record({**row, "name": name})
        if name in entries:
            raise ValueError(f"duplicate keyframe: {name}")
        session = str(row["session_id"])
        if row.get("evaluation_role") != "MAPPING" or not session:
            raise ValueError(f"query/non-MAPPING capture cannot be an offline input: {name}")
        source_path = Path(str(row["image_uri"]))
        source = source_path.resolve(strict=True)
        if (source_path.is_symlink() or not source.is_relative_to(images)
                or source != (images / safe_relative_name(name)).resolve(strict=True)
                or source.suffix.lower() not in {".jpg", ".jpeg"}):
            raise ValueError(f"capture is outside the exact native MAPPING JPEG allowlist: {name}")
        entries[name] = row
    reconstruction = pycolmap.Reconstruction(str(model))
    if (not reconstruction.images
            or reconstruction.num_reg_images() != len(reconstruction.images)
            or set(entries) != {image.name for image in reconstruction.images.values()}):
        raise ValueError("keyframes must exactly describe the frozen registered mapping model")
    wanted = set(reconstruction.images) if image_ids is None else {int(i) for i in image_ids}
    if not wanted or not wanted <= set(reconstruction.images):
        raise ValueError("subset contains absent frozen image IDs")
    captures: dict[int, FrozenCapture] = {}
    for image_id in sorted(wanted):
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        if str(camera.model).split(".")[-1] != "PINHOLE":
            raise ValueError(f"native image is not fixed undistorted PINHOLE: {image.name}")
        row = entries[image.name]
        path = Path(str(row["image_uri"])).resolve(strict=True)
        if path != (images / safe_relative_name(image.name)).resolve(strict=True):
            raise ValueError(f"keyframe is not the native manifest JPEG: {image.name}")
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError("offline evidence requires the original native MAPPING JPEGs")
        digest = sha256_file(path)
        probe = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if (digest != row["image_sha256"] or probe is None
                or probe.shape != (camera.height, camera.width)):
            raise ValueError(f"changed/unreadable/native-resolution mismatch: {image.name}")
        k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        native = pixel_center_intrinsics(k, from_convention=input_pixel_convention,
                                         to_convention=OPENCV_NATIVE_CENTER)
        colmap = pixel_center_intrinsics(k, from_convention=input_pixel_convention,
                                         to_convention=COLMAP_CORNER)
        pose = validate_pose(np.asarray(image.cam_from_world().matrix()))
        captures[image_id] = FrozenCapture(
            image_id, image.name, str(path), str(row["image_uri"]), digest, str(row["session_id"]),
            int(row["frame_index"]), str(row["evaluation_role"]), int(image.camera_id),
            int(camera.width), int(camera.height),
            k.tolist(), native.tolist(), colmap.tolist(), pose.tolist())
        validate_mapping_capture(captures[image_id])
    if frozen_model_hashes(model) != hashes or sha256_file(manifest) != manifest_sha:
        raise ValueError("frozen geometry changed during input validation")
    return FrozenScene(model, manifest, images, hashes, manifest_sha,
                       input_pixel_convention, captures)


def _assert_scene_unchanged(scene: FrozenScene) -> None:
    if (frozen_model_hashes(scene.model_dir) != dict(scene.model_hashes)
            or sha256_file(scene.keyframes_manifest) != scene.keyframes_sha256):
        raise ValueError("frozen model/keyframes changed during offline evidence work")
    for capture in scene.captures.values():
        if sha256_file(Path(capture.image_path)) != capture.image_sha256:
            raise ValueError(f"native JPEG changed: {capture.name}")


def verify_frozen_capture_snapshot(root: Path, captures: Mapping[int, FrozenCapture], *,
                                    input_pixel_convention: str) -> float:
    """Bind snapshots and return the full registered model's correlation-cell width."""
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(root / "frozen_model"))
    entries = {}
    for line in (root / "keyframes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            name = str(row["output_name"])
            validate_mapping_record({**row, "name": name})
            if name in entries:
                raise ValueError("duplicate frozen keyframe snapshot")
            if row.get("evaluation_role") != "MAPPING" or not row.get("session_id"):
                raise ValueError("every frozen registered capture must declare its MAPPING role")
            safe_relative_name(name)
            entries[name] = row
    if (reconstruction.num_reg_images() != len(reconstruction.images)
            or set(entries) != {image.name for image in reconstruction.images.values()}):
        raise ValueError("keyframe snapshot is not the exact frozen mapping model")
    for image_id, capture in captures.items():
        validate_mapping_capture(capture)
        safe_relative_name(capture.name)
        if image_id not in reconstruction.images or capture.name not in entries:
            raise ValueError("artifact camera is absent from the original frozen model")
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        frame = entries[capture.name]
        k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        native = pixel_center_intrinsics(k, from_convention=input_pixel_convention,
                                         to_convention=OPENCV_NATIVE_CENTER)
        colmap = pixel_center_intrinsics(k, from_convention=input_pixel_convention,
                                         to_convention=COLMAP_CORNER)
        if (str(camera.model).split(".")[-1] != "PINHOLE" or image.name != capture.name
                or int(image.camera_id) != capture.camera_id
                or (camera.width, camera.height) != (capture.width, capture.height)
                or not np.array_equal(k, capture.K_frozen)
                or not np.array_equal(native, capture.K_native)
                or not np.array_equal(colmap, capture.K_colmap)
                or not np.allclose(validate_pose(np.asarray(image.cam_from_world().matrix())),
                                   validate_pose(np.asarray(capture.cam_from_world)), atol=1e-12, rtol=0)
                or frame.get("evaluation_role") != "MAPPING"
                or frame.get("session_id") != capture.session_id
                or int(frame["frame_index"]) != capture.frame_index
                or frame.get("image_uri") != capture.source_image_uri
                or frame.get("image_sha256") != capture.image_sha256
                or sha256_file(root / "native_images" / safe_relative_name(capture.name)) != capture.image_sha256):
            raise ValueError("artifact changed a frozen pose/intrinsic/native mapping capture")
    centers = []
    for image in reconstruction.images.values():
        try:
            pose = validate_pose(image.cam_from_world().matrix())
        except ValueError:
            # Unsupported frozen cameras cannot define correlation geometry.
            # Selected captures already passed the strict pose check above.
            continue
        centers.append(-pose[:, :3].T @ pose[:, 3])
    return max(float(np.linalg.norm(np.ptp(np.asarray(centers), axis=0))) * .05, 1e-8)


def _pair_sparse_observability(reconstruction: Any, a: FrozenCapture, b: FrozenCapture,
                               *, min_baseline_deg: float,
                               mapping_image_ids: frozenset[int]) -> tuple[int, float]:
    from .metric_anchors import valid_track_observations
    ids_a = {int(p.point3D_id) for p in reconstruction.images[a.image_id].points2D
             if p.has_point3D()}
    ids_b = {int(p.point3D_id) for p in reconstruction.images[b.image_id].points2D
             if p.has_point3D()}
    angles = []
    for point_id in sorted(ids_a & ids_b):
        if point_id not in reconstruction.points3D:
            continue
        point = reconstruction.points3D[point_id]
        valid_images = {image_id for image_id, _, _ in
                        valid_track_observations(reconstruction, point_id,
                                                 mapping_image_ids=mapping_image_ids)}
        if not {a.image_id, b.image_id} <= valid_images:
            continue
        if not np.isfinite(point.xyz).all():
            continue
        xyz = np.asarray(point.xyz, dtype=np.float64)
        pa, pb = np.asarray(a.cam_from_world), np.asarray(b.cam_from_world)
        if (pa[:, :3] @ xyz + pa[:, 3])[2] <= 0 or (pb[:, :3] @ xyz + pb[:, 3])[2] <= 0:
            continue
        ra, rb = xyz - a.center, xyz - b.center
        denominator = np.linalg.norm(ra) * np.linalg.norm(rb)
        if denominator <= 1e-12:
            continue
        angle = math.degrees(math.acos(float(np.clip(ra @ rb / denominator, -1., 1.))))
        if angle >= min_baseline_deg:
            angles.append(angle)
    return len(angles), float(np.median(angles)) if angles else 0.


def select_source_lists(scene: FrozenScene, reference_image_ids: Sequence[int], *,
                        max_sources: int = 8, min_shared_tracks: int = 8,
                        min_baseline_deg: float = 2.) -> dict[int, list[int]]:
    """Select sparse-covisible, baseline- and session-diverse sources, never MoGe.

    Only references and their needed sources enter the workspace. Source views
    must themselves have at least two usable sources inside this subset; a poor
    canary is rejected instead of silently expanding to all mapping captures.
    """
    import pycolmap

    if min_baseline_deg < 2 or max_sources < 2 or min_shared_tracks < 3:
        raise ValueError("source selection needs two-degree baseline and three usable sparse points")
    references = sorted({int(i) for i in reference_image_ids})
    if not references or not set(references) <= set(scene.captures):
        raise ValueError("reference images are absent from the frozen scene")
    reconstruction = pycolmap.Reconstruction(str(scene.model_dir))

    def choose(reference: int, allowed: set[int]) -> list[int]:
        capture = scene.captures[reference]
        by_session: dict[str, list[tuple[int, float, int]]] = {}
        for image_id in sorted(allowed - {reference}):
            neighbor = scene.captures[image_id]
            count, angle = _pair_sparse_observability(
                reconstruction, capture, neighbor, min_baseline_deg=min_baseline_deg,
                mapping_image_ids=frozenset(scene.captures))
            if count < min_shared_tracks:
                continue
            by_session.setdefault(neighbor.session_id, []).append((count, angle, image_id))
        for candidates in by_session.values():
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        sessions = sorted(by_session, key=lambda s: (s == capture.session_id,
                          -by_session[s][0][0], -by_session[s][0][1], s))
        selected: list[int] = []
        while len(selected) < max_sources:
            added = False
            for session in sessions:
                if by_session[session]:
                    selected.append(by_session[session].pop(0)[2])
                    added = True
                    if len(selected) == max_sources:
                        break
            if not added:
                break
        if len(selected) < 2:
            raise ValueError(f"canary source graph lacks two observable sources for {capture.name}")
        return selected

    primary = {image_id: choose(image_id, set(scene.captures)) for image_id in references}
    needed = set(references) | {source for sources in primary.values() for source in sources}
    return {image_id: (primary[image_id] if image_id in primary else choose(image_id, needed))
            for image_id in sorted(needed)}


@dataclass(frozen=True)
class MvsParameters:
    max_image_size: int = -1
    gpu_index: str = "0"
    window_radius: int = 5
    window_step: int = 1
    num_samples: int = 15
    num_iterations: int = 5
    sigma_spatial: float = -1.
    sigma_color: float = .2
    ncc_sigma: float = .6
    min_triangulation_angle: float = 2.
    incident_angle_sigma: float = .9
    geom_consistency_regularizer: float = .3
    geom_consistency_max_cost: float = 3.
    filter_min_ncc: float = .1
    filter_min_triangulation_angle: float = 3.
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.
    cache_size: float = 4.
    stereo_fusion: bool = False

    def validate(self) -> None:
        numeric = [v for v in asdict(self).values() if isinstance(v, (float, int))]
        if (not all(math.isfinite(v) for v in numeric)
                or (self.max_image_size != -1 and self.max_image_size < 2)
                or self.min_triangulation_angle < 2 or self.filter_min_triangulation_angle < 2
                or self.filter_min_num_consistent < 2 or not 0 < self.cache_size <= 8
                or self.window_radius < 1 or self.window_step < 1
                or self.num_samples < 1 or self.num_iterations < 1
                or not re.fullmatch(r"\d+", self.gpu_index)):
            raise ValueError("invalid or gate-relaxing fixed-pose MVS parameters")

    def patch_match_options(self, *, geometric: bool) -> dict[str, Any]:
        self.validate()
        options = {key: value for key, value in asdict(self).items() if key != "stereo_fusion"}
        options.update(depth_min=-1., depth_max=-1., num_threads=8,
                       geom_consistency=geometric, filter=geometric, allow_missing_files=False,
                       write_consistency_graph=True)
        return options


@dataclass(frozen=True)
class MvsResourceLimits:
    cpu_affinity: tuple[int, ...] = tuple(range(2, 10))
    nice: int = 10
    cpu_weight: int = 20
    memory_high_bytes: int = 20 * 1024 ** 3
    memory_max_bytes: int = 24 * 1024 ** 3
    swap_max_bytes: int = 1024 ** 3

    def validate(self) -> None:
        if (not self.cpu_affinity or len(set(self.cpu_affinity)) != len(self.cpu_affinity)
                or len(self.cpu_affinity) > 8 or min(self.cpu_affinity) < 0
                or not 10 <= self.nice <= 19 or not 1 <= self.cpu_weight <= 20
                or not 0 < self.memory_high_bytes <= 20 * 1024 ** 3
                or not self.memory_high_bytes <= self.memory_max_bytes <= 24 * 1024 ** 3
                or not 0 <= self.swap_max_bytes <= 1024 ** 3):
            raise ValueError("offline resource envelope must remain at most eight CPUs/24 GiB")


def pin_colmap_executable(executable: Path | str, *, executable_sha256: str,
                          cuda_build_receipt: Path | str,
                          version: str = SUPPORTED_COLMAP_VERSION) -> dict[str, Any]:
    """Static pinning only; actual version/CUDA banner is checked by the runner."""
    binary = Path(executable).resolve(strict=True)
    build_path = Path(cuda_build_receipt).resolve(strict=True)
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if (version != SUPPORTED_COLMAP_VERSION or build.get("name") != "colmap"
            or build.get("version") != version
            or not str(build.get("build", "")).startswith("cuda_")
            or not any(str(dep).startswith("cuda-cudart ") for dep in build.get("depends", []))
            or sha256_file(binary) != executable_sha256 or not os.access(binary, os.X_OK)):
        raise ValueError("requires pinned COLMAP 4.0.4 CUDA executable; CPU-only 3.9.1 is forbidden")
    return {"path": str(binary), "sha256": executable_sha256, "version": version,
            "cuda_build_receipt_path": str(build_path), "cuda_build_receipt_sha256": sha256_file(build_path),
            "build": str(build["build"]), "cuda_dependencies": [dep for dep in build["depends"]
                                      if "cuda" in str(dep).lower()]}


def _source_config(captures: Mapping[int, FrozenCapture], sources: Mapping[int, Sequence[int]]) -> str:
    return "".join(f"{captures[i].name}\n" + ", ".join(captures[j].name for j in sources[i]) + "\n"
                   for i in sorted(sources))


def _validate_source_lists(sources: Mapping[int, Sequence[int]], references: Sequence[int], *,
                           min_support: int) -> None:
    """A capture votes once, with a complete fixed-camera dependency graph."""
    if (min_support < 2 or not references or len(references) != len(set(references))
            or not set(references) <= set(sources)):
        raise ValueError("MVS requires an explicit unique reference subset")
    for image_id, neighbors in sources.items():
        if (len(neighbors) < min_support or len(neighbors) != len(set(neighbors))
                or image_id in neighbors or not set(neighbors) <= set(sources)):
            raise ValueError("every needed source requires an explicit self-free complete source list")


def estimate_mvs_resources(captures: Mapping[int, FrozenCapture], sources: Mapping[int, Sequence[int]],
                            parameters: MvsParameters, *, model_bytes: int = 0) -> dict[str, Any]:
    parameters.validate()
    image_bytes = sum(Path(c.image_path).stat().st_size for c in captures.values())
    pixels: dict[int, int] = {}
    for image_id, capture in captures.items():
        factor = (min(1., parameters.max_image_size / max(capture.width, capture.height))
                  if parameters.max_image_size > 0 else 1.)
        pixels[image_id] = math.ceil(capture.width * factor) * math.ceil(capture.height * factor)
    # Two passes: depth + three-channel normal, plus worst-case per-pixel graph.
    map_bytes = sum(pixels[i] * (32 + 2 * (12 + 4 * len(sources[i]))) for i in captures)
    fusion_bytes = sum(pixels.values()) * 32 if parameters.stereo_fusion else 0
    upper = math.ceil((2 * image_bytes + map_bytes + fusion_bytes + 6 * model_bytes) * 1.25)
    return {"workspace_images": len(captures), "reference_pixels": sum(pixels.values()),
            "disk_upper_estimate_bytes": upper, "disk_estimate_includes_safety_factor": 1.25,
            "host_cache_bytes": int(parameters.cache_size * 1024 ** 3),
            "cpu_threads": 8, "memory_high_bytes": 20 * 1024 ** 3,
            "memory_max_bytes": 24 * 1024 ** 3, "swap_max_bytes": 1024 ** 3,
            "cuda_memory_fraction": None,
            "cuda_memory_note": "COLMAP has no PyTorch memory-fraction allocator; parent schedules GPU lease"}


def prepare_offline_mvs_request(model_dir: Path | str, keyframes_manifest: Path | str,
                               image_root: Path | str, source_lists: Mapping[int, Sequence[int]],
                               request_dir: Path | str, *, reference_image_ids: Sequence[int],
                               input_pixel_convention: str, executable: Path | str,
                               executable_sha256: str, cuda_build_receipt: Path | str,
                               map_revision_id: str, coordinate_frame_id: str,
                               parameters: MvsParameters = MvsParameters(),
                               resources: MvsResourceLimits = MvsResourceLimits()) -> Path:
    """Prepare a genuine canary/full request without launching COLMAP or GPU work."""
    parameters.validate()
    resources.validate()
    sources = {int(i): [int(j) for j in neighbors] for i, neighbors in source_lists.items()}
    references = sorted({int(i) for i in reference_image_ids})
    if not map_revision_id.strip() or not coordinate_frame_id.strip():
        raise ValueError("MVS requires explicit coordinate/map identity")
    _validate_source_lists(sources, references, min_support=parameters.filter_min_num_consistent)
    scene = load_frozen_scene(model_dir, keyframes_manifest, image_root,
                              input_pixel_convention=input_pixel_convention, image_ids=sorted(sources))
    import pycolmap
    reconstruction = pycolmap.Reconstruction(str(scene.model_dir))
    source_pair_statistics = {}
    for image_id, neighbors in sources.items():
        for neighbor in neighbors:
            count, angle = _pair_sparse_observability(reconstruction, scene.captures[image_id],
                          scene.captures[neighbor], min_baseline_deg=parameters.min_triangulation_angle,
                          mapping_image_ids=frozenset(scene.captures))
            if count < 3:
                raise ValueError("source pairs need at least three valid sparse observations at two-degree baseline")
            source_pair_statistics[f"{image_id}:{neighbor}"] = {
                "shared_usable_tracks": count, "median_angle_deg": angle,
                "reference_session": scene.captures[image_id].session_id,
                "source_session": scene.captures[neighbor].session_id}
    config = _source_config(scene.captures, sources)
    pin = pin_colmap_executable(executable, executable_sha256=executable_sha256,
                               cuda_build_receipt=cuda_build_receipt)
    model_bytes = sum((scene.model_dir / name).stat().st_size for name in scene.model_hashes)
    request = {"schema": REQUEST_SCHEMA, "geometry_origin": "FROZEN_CAMERA_RGB_STEREO",
               "runtime_depth_inference": False, "source_selection": "SPARSE_COVISIBILITY_BASELINE_SESSION",
               "source_scope": MAPPING_SCOPE, "holdout_scope": MAPPING_HOLDOUT_SCOPE,
               "colmap_source_revision": COLMAP_SOURCE_REVISION,
               "canonical_model_pixel_convention": COLMAP_CORNER,
               "workspace_pixel_convention": OPENCV_NATIVE_CENTER,
               "array_pixel_convention": OPENCV_NATIVE_CENTER,
               "model_dir": str(scene.model_dir), "model_hashes": dict(scene.model_hashes),
               "keyframes_manifest": str(scene.keyframes_manifest), "keyframes_sha256": scene.keyframes_sha256,
               "image_root": str(scene.image_root), "input_pixel_convention": input_pixel_convention,
               "native_to_colmap_pixel_transform": SHIFT_TO_COLMAP.tolist(),
               "map_revision_id": map_revision_id, "coordinate_frame_id": coordinate_frame_id,
               "reference_image_ids": references, "images": [asdict(scene.captures[i]) for i in sorted(sources)],
               "source_lists": {str(i): sources[i] for i in sorted(sources)},
               "source_list_sha256": hashlib.sha256(config.encode()).hexdigest(),
               "source_pair_statistics": source_pair_statistics,
               "parameters": asdict(parameters), "resources": asdict(resources), "executable": pin,
               "implementation_sha256": sha256_file(Path(__file__)),
               "estimates": estimate_mvs_resources(scene.captures, sources, parameters, model_bytes=model_bytes)}
    request["request_sha256"] = canonical_sha256(request)
    output = Path(request_dir).absolute()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "request.json"
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != request:
            raise ValueError("immutable MVS request directory already has different inputs")
        return target
    _write_json_atomic(target, request)
    return target


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".partial")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _write_fixed_subset(scene: FrozenScene, output: Path, *, target_pixel_convention: str) -> None:
    """Serialize existing tracks in a declared origin; never solve/triangulate."""
    import pycolmap
    if any(any(char.isspace() for char in capture.name) for capture in scene.captures.values()):
        raise ValueError("COLMAP text workspace image names cannot contain whitespace")
    reconstruction = pycolmap.Reconstruction(str(scene.model_dir))
    output.mkdir()
    wanted = set(scene.captures)
    retained: dict[int, list[tuple[int, int]]] = {}
    for point_id, point in reconstruction.points3D.items():
        observations = [(int(e.image_id), int(e.point2D_idx)) for e in point.track.elements
                        if e.image_id in wanted]
        if observations:
            retained[int(point_id)] = observations
    camera_rows: dict[int, FrozenCapture] = {}
    for capture in scene.captures.values():
        camera_rows[capture.camera_id] = capture
    with (output / "cameras.txt").open("w", encoding="utf-8") as stream:
        for camera_id, capture in sorted(camera_rows.items()):
            k = pixel_center_intrinsics(np.asarray(capture.K_frozen),
                         from_convention=scene.input_pixel_convention, to_convention=target_pixel_convention)
            stream.write(f"{camera_id} PINHOLE {capture.width} {capture.height} "
                         + " ".join(format(v, ".17g") for v in (k[0, 0], k[1, 1], k[0, 2], k[1, 2])) + "\n")
    with (output / "images.txt").open("w", encoding="utf-8") as stream:
        for image_id, capture in sorted(scene.captures.items()):
            image = reconstruction.images[image_id]
            pose = image.cam_from_world()
            quaternion = np.asarray(pose.rotation.quat)[[3, 0, 1, 2]]
            values = np.r_[quaternion, np.asarray(pose.translation)]
            stream.write(f"{image_id} " + " ".join(format(v, ".17g") for v in values)
                         + f" {capture.camera_id} {capture.name}\n")
            entries = []
            origins = {OPENCV_NATIVE_CENTER: 0., COLMAP_CORNER: .5}
            offset = origins[target_pixel_convention] - origins[scene.input_pixel_convention]
            for point in image.points2D:
                x, y = np.asarray(point.xy) + offset
                point_id = int(point.point3D_id) if point.has_point3D() and int(point.point3D_id) in retained else -1
                entries.append(f"{x:.17g} {y:.17g} {point_id}")
            stream.write(" ".join(entries) + "\n")
    with (output / "points3D.txt").open("w", encoding="utf-8") as stream:
        for point_id, observations in sorted(retained.items()):
            point = reconstruction.points3D[point_id]
            stream.write(f"{point_id} " + " ".join(format(float(v), ".17g") for v in point.xyz)
                         + " " + " ".join(str(int(v)) for v in point.color)
                         + f" {float(point.error):.17g} "
                         + " ".join(f"{i} {j}" for i, j in observations) + "\n")


def colmap_image_order(model_dir: Path | str) -> list[tuple[int, str]]:
    """Read actual on-disk ordering, not pycolmap dictionary order or image IDs."""
    root = Path(model_dir)
    binary = root / "images.bin"
    if binary.is_file():
        result = []
        size = binary.stat().st_size
        with binary.open("rb") as stream:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("partial COLMAP image-order header")
            count, = struct.unpack("<Q", header)
            if count > size // 73:
                raise ValueError("invalid COLMAP image count")
            for _ in range(count):
                row = stream.read(64)
                if len(row) != 64:
                    raise ValueError("partial COLMAP image pose")
                image_id, = struct.unpack("<I", row[:4])
                name = bytearray()
                while True:
                    char = stream.read(1)
                    if not char or len(name) > 16384:
                        raise ValueError("partial/oversized COLMAP image name")
                    if char == b"\0":
                        break
                    name.extend(char)
                num_points = stream.read(8)
                if len(num_points) != 8:
                    raise ValueError("partial COLMAP observations")
                num_points, = struct.unpack("<Q", num_points)
                if num_points > (size - stream.tell()) // 24:
                    raise ValueError("partial COLMAP observation payload")
                stream.seek(num_points * 24, 1)
                result.append((int(image_id), name.decode("utf-8")))
            if stream.tell() != size:
                raise ValueError("extra COLMAP binary image bytes")
    else:
        result = []
        with (root / "images.txt").open(encoding="utf-8") as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.strip() or line.startswith("#"):
                    continue
                row = line.rstrip("\n").split(maxsplit=9)
                if len(row) != 10:
                    raise ValueError("invalid COLMAP text image header")
                result.append((int(row[0]), row[9]))
                if not stream.readline():
                    raise ValueError("partial COLMAP text observations")
    if len({i for i, _ in result}) != len(result) or len({n for _, n in result}) != len(result):
        raise ValueError("duplicate COLMAP image-order identities")
    return result


def _dense_header(path: Path) -> tuple[int, int, int, int]:
    with path.open("rb") as stream:
        tokens = []
        token = bytearray()
        for _ in range(128):
            char = stream.read(1)
            if not char:
                raise ValueError("partial COLMAP dense header")
            if char == b"&":
                if not token or not token.isdigit():
                    raise ValueError("invalid COLMAP dense dimensions")
                tokens.append(int(token))
                token.clear()
                if len(tokens) == 3:
                    if min(tokens) <= 0:
                        raise ValueError("COLMAP dense dimensions must be positive")
                    return *tokens, stream.tell()
            else:
                token.extend(char)
    raise ValueError("oversized COLMAP dense header")


def read_colmap_dense_array(path: Path | str, *, channels: int | None = None) -> np.ndarray:
    """Strict little-endian float32 dense reader, preserving Fortran-layout ABI."""
    file = Path(path).resolve(strict=True)
    width, height, actual_channels, offset = _dense_header(file)
    if channels is not None and actual_channels != channels:
        raise ValueError("wrong COLMAP depth/normal channel count")
    if actual_channels not in (1, 3) or file.stat().st_size != offset + 4 * width * height * actual_channels:
        raise ValueError("partial or oversized COLMAP dense binary payload")
    payload = np.fromfile(file, dtype="<f4", count=width * height * actual_channels, offset=offset)
    array = payload.reshape((width, height, actual_channels), order="F").transpose(1, 0, 2)
    return array[..., 0] if actual_channels == 1 else array


@dataclass(frozen=True)
class ConsistencyGraph:
    width: int
    height: int
    pixel_index: np.ndarray
    offsets: np.ndarray
    image_indices: np.ndarray

    def support_at(self, uv: np.ndarray, *, min_support: int = 2) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(coordinates).all(axis=1)
        rounded = np.zeros(coordinates.shape, dtype=np.int64)
        rounded[finite] = np.rint(coordinates[finite]).astype(np.int64)
        inside = (finite & (rounded[:, 0] >= 0) & (rounded[:, 0] < self.width)
                  & (rounded[:, 1] >= 0) & (rounded[:, 1] < self.height))
        pixels = rounded[:, 1] * self.width + rounded[:, 0]
        rows = np.searchsorted(self.pixel_index, pixels)
        present = inside & (rows < len(self.pixel_index))
        valid_rows = np.flatnonzero(present)
        present[valid_rows] &= self.pixel_index[rows[valid_rows]] == pixels[valid_rows]
        counts = np.zeros(len(coordinates), dtype=np.int32)
        valid_rows = np.flatnonzero(present)
        counts[valid_rows] = self.offsets[rows[valid_rows] + 1] - self.offsets[rows[valid_rows]]
        rows[~present] = -1
        return counts >= min_support, rows


def read_colmap_consistency_graph(path: Path | str, *, num_images: int,
                                   allowed_image_indices: Sequence[int] | None = None) -> ConsistencyGraph:
    file = Path(path).resolve(strict=True)
    width, height, channels, offset = _dense_header(file)
    if channels != 1 or (file.stat().st_size - offset) % 4:
        raise ValueError("partial/invalid COLMAP consistency graph payload")
    count = (file.stat().st_size - offset) // 4
    data = np.fromfile(file, dtype="<i4", count=count, offset=offset)
    pixels, starts, stops = [], [], []
    cursor = 0
    allowed = set(range(num_images)) if allowed_image_indices is None else set(allowed_image_indices)
    if num_images < 1 or not allowed <= set(range(num_images)):
        raise ValueError("invalid graph source-image identity table")
    while cursor < len(data):
        if len(data) - cursor < 3:
            raise ValueError("partial consistency graph record")
        # COLMAP 4.0.4 ConsistencyGraph::InitializeMap reads column, row, count.
        column, row, n_sources = map(int, data[cursor:cursor + 3])
        cursor += 3
        if (not 0 <= row < height or not 0 <= column < width or n_sources < 0
                or n_sources > num_images or n_sources > len(data) - cursor):
            raise ValueError("invalid/truncated consistency graph record")
        identities = data[cursor:cursor + n_sources]
        if len(set(map(int, identities))) != n_sources or not set(map(int, identities)) <= allowed:
            raise ValueError("consistency graph contains undeclared/duplicate source views")
        pixels.append(row * width + column)
        starts.append(cursor)
        stops.append(cursor + n_sources)
        cursor += n_sources
    order = np.argsort(pixels, kind="stable")
    pixel_index = np.asarray(pixels, dtype=np.int64)[order]
    if len(pixel_index) > 1 and np.any(pixel_index[1:] == pixel_index[:-1]):
        raise ValueError("duplicate consistency graph pixel record")
    counts = np.asarray(stops, dtype=np.int64) - np.asarray(starts, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts[order], dtype=np.int64)]
    indices = np.empty(int(offsets[-1]), dtype=np.int32)
    for destination, original in enumerate(order):
        indices[offsets[destination]:offsets[destination + 1]] = data[starts[original]:stops[original]]
    for array in (pixel_index, offsets, indices):
        array.flags.writeable = False
    return ConsistencyGraph(width, height, pixel_index, offsets, indices)


def signed_depth_verdict(candidate_z: np.ndarray, measured_z: np.ndarray, *,
                          usable: np.ndarray | None = None, max_log_residual: float = .08) -> dict[str, np.ndarray]:
    if not 0 < max_log_residual <= .08:
        raise ValueError("signed checker cannot relax the frozen 0.08 log-depth gate")
    candidate, measured = np.broadcast_arrays(np.asarray(candidate_z, dtype=np.float64),
                                               np.asarray(measured_z, dtype=np.float64))
    valid = np.isfinite(candidate) & np.isfinite(measured) & (candidate > 0) & (measured > 0)
    if usable is not None:
        valid &= np.asarray(usable, dtype=bool)
    residual = np.full(candidate.shape, np.nan)
    residual[valid] = np.log(candidate[valid]) - np.log(measured[valid])
    states = np.full(candidate.shape, UNKNOWN, dtype=np.uint8)
    states[valid & (np.abs(residual) <= max_log_residual)] = AGREE
    states[valid & (residual < -max_log_residual)] = CONFLICT
    states[valid & (residual > max_log_residual)] = OCCLUDED
    return {"signed_log_residual": residual, "states": states}


def coherent_depth_sample(depth: np.ndarray, uv: np.ndarray, *, max_log_span: float = .08) -> tuple[np.ndarray, np.ndarray]:
    """One-layer four taps (.08), plus the separate 20 percent 3x3 continuity gate."""
    if not 0 < max_log_span <= .08:
        raise ValueError("four-tap interpolation cannot exceed the frozen 0.08 log span")
    array = np.asarray(depth)
    if array.ndim != 2:
        raise ValueError("depth must be a two-dimensional optical-Z map")
    coordinates = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    valid = np.isfinite(coordinates).all(axis=1)
    height, width = array.shape
    valid &= ((coordinates[:, 0] >= 1) & (coordinates[:, 0] < width - 2)
              & (coordinates[:, 1] >= 1) & (coordinates[:, 1] < height - 2))
    result = np.full(len(coordinates), np.nan)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return result, valid
    xy = coordinates[ids]
    x, y = np.floor(xy).astype(np.int64).T
    nearest_x, nearest_y = np.rint(xy).astype(np.int64).T
    patches = np.stack([array[nearest_y + dy, nearest_x + dx] for dy in (-1, 0, 1)
                        for dx in (-1, 0, 1)])
    taps = np.stack((array[y, x], array[y, x + 1], array[y + 1, x], array[y + 1, x + 1]))
    good = (np.isfinite(patches).all(axis=0) & (patches > 0).all(axis=0)
            & np.isfinite(taps).all(axis=0) & (taps > 0).all(axis=0))
    selected = np.flatnonzero(good)
    if len(selected):
        good[selected] &= (np.log(patches[:, selected].max(axis=0) / patches[:, selected].min(axis=0)) <= math.log(1.2))
        good[selected] &= (np.log(taps[:, selected].max(axis=0) / taps[:, selected].min(axis=0)) <= max_log_span)
    valid[ids] = good
    wx, wy = xy[:, 0] - x, xy[:, 1] - y
    samples = (array[y, x] * (1 - wx) * (1 - wy) + array[y, x + 1] * wx * (1 - wy)
               + array[y + 1, x] * (1 - wx) * wy + array[y + 1, x + 1] * wx * wy)
    result[ids[good]] = samples[good]
    return result, valid


def _options_arguments(options: Mapping[str, Any], prefix: str = "") -> list[str]:
    result = []
    for key, value in sorted(options.items()):
        result.extend(["--" + prefix + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return result


def _validate_colmap_banner(banner: str) -> None:
    if (not re.search(r"COLMAP\s+" + re.escape(SUPPORTED_COLMAP_VERSION) + r"\b", banner)
            or not re.search(r"(?:with|enabled)\s+CUDA|CUDA\s+enabled", banner, re.IGNORECASE)
            or re.search(r"without\s+CUDA|CUDA\s+disabled", banner, re.IGNORECASE)):
        raise ValueError("actual COLMAP executable is not the pinned CUDA 4.0.4 build")


def _require_parent_resource_envelope(limits: MvsResourceLimits) -> dict[str, Any]:
    """Require nonempty actual affinity within the declared <=8-core cgroup envelope."""
    limits.validate()
    affinity = sorted(os.sched_getaffinity(0))
    if (not affinity or not set(affinity) <= set(limits.cpu_affinity)
            or os.getpriority(os.PRIO_PROCESS, 0) < limits.nice):
        raise ValueError("parent must launch inside the declared CPU affinity/nice envelope")
    relative = None
    for row in Path("/proc/self/cgroup").read_text().splitlines():
        if row.startswith("0::"):
            relative = row[3:]
    if relative is None:
        raise ValueError("parent must provide an enforced cgroup-v2 memory envelope")
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    requirements = {"memory.high": limits.memory_high_bytes, "memory.max": limits.memory_max_bytes,
                    "memory.swap.max": limits.swap_max_bytes, "cpu.weight": limits.cpu_weight}
    observed = {}
    for name, maximum in requirements.items():
        value = (root / name).read_text().strip()
        if value == "max" or int(value) > maximum:
            raise ValueError(f"parent cgroup does not enforce {name} <= {maximum}")
        observed[name] = int(value)
    return {"cgroup": relative, "cpu_affinity": affinity,
            "nice": os.getpriority(os.PRIO_PROCESS, 0), **observed}


def _load_request(path: Path) -> tuple[dict[str, Any], FrozenScene, dict[int, list[int]], MvsParameters]:
    request = json.loads(path.read_text(encoding="utf-8"))
    identity = {k: v for k, v in request.items() if k != "request_sha256"}
    if request.get("schema") != REQUEST_SCHEMA or canonical_sha256(identity) != request.get("request_sha256"):
        raise ValueError("invalid/hash-mismatched MVS request")
    if request.get("implementation_sha256") != sha256_file(Path(__file__)):
        raise ValueError("MVS request was prepared by another implementation")
    _validate_request_policy(request)
    sources = {int(i): [int(j) for j in neighbors] for i, neighbors in request["source_lists"].items()}
    parameters = MvsParameters(**request["parameters"])
    parameters.validate()
    _validate_source_lists(sources, request["reference_image_ids"],
                           min_support=parameters.filter_min_num_consistent)
    scene = load_frozen_scene(request["model_dir"], request["keyframes_manifest"], request["image_root"],
                              input_pixel_convention=request["input_pixel_convention"], image_ids=sorted(sources))
    if (dict(scene.model_hashes) != request["model_hashes"] or scene.keyframes_sha256 != request["keyframes_sha256"]
            or [asdict(scene.captures[i]) for i in sorted(sources)] != request["images"]
            or hashlib.sha256(_source_config(scene.captures, sources).encode()).hexdigest() != request["source_list_sha256"]):
        raise ValueError("frozen MVS inputs/source lists changed")
    pin = request["executable"]
    if pin_colmap_executable(pin["path"], executable_sha256=pin["sha256"],
                             cuda_build_receipt=pin["cuda_build_receipt_path"], version=pin["version"]) != pin:
        raise ValueError("pinned executable/build receipt changed")
    return request, scene, sources, parameters


def _validate_request_policy(request: Mapping[str, Any]) -> None:
    expected = {"source_scope": MAPPING_SCOPE, "holdout_scope": MAPPING_HOLDOUT_SCOPE,
                "source_selection": "SPARSE_COVISIBILITY_BASELINE_SESSION",
                "colmap_source_revision": COLMAP_SOURCE_REVISION,
                "canonical_model_pixel_convention": COLMAP_CORNER,
                "workspace_pixel_convention": OPENCV_NATIVE_CENTER,
                "array_pixel_convention": OPENCV_NATIVE_CENTER,
                "runtime_depth_inference": False, "geometry_origin": "FROZEN_CAMERA_RGB_STEREO",
                "native_to_colmap_pixel_transform": SHIFT_TO_COLMAP.tolist()}
    if (any(request.get(key) != value for key, value in expected.items())
            or not str(request.get("map_revision_id", "")).strip()
            or not str(request.get("coordinate_frame_id", "")).strip()):
        raise ValueError("MVS request changed the registered MAPPING/backend pixel-origin contract")


def _canonical_dense_captures(model_dir: Path, workspace: Path, scene: FrozenScene) -> dict[int, FrozenCapture]:
    """Validate the undistorter representation, including unchanged world points."""
    import pycolmap
    canonical = pycolmap.Reconstruction(str(model_dir))
    original = pycolmap.Reconstruction(str(scene.model_dir))
    if ({int(i) for i in canonical.reg_image_ids()} != set(scene.captures)
            or len(canonical.images) != len(scene.captures)):
        raise ValueError("undistorter changed the frozen subset image identities")
    for point_id, point in canonical.points3D.items():
        if point_id not in original.points3D or not np.array_equal(point.xyz, original.points3D[point_id].xyz):
            raise ValueError("undistorter created/refit a frozen world point")
    captures = {}
    for image_id, capture in scene.captures.items():
        image = canonical.images[image_id]
        camera = canonical.cameras[image.camera_id]
        pose = validate_pose(np.asarray(image.cam_from_world().matrix()))
        k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        scale = np.diag([camera.width / capture.width, camera.height / capture.height, 1.])
        if (image.name != capture.name or str(camera.model).split(".")[-1] != "PINHOLE"
                or not np.allclose(pose, capture.cam_from_world, atol=1e-12, rtol=0)
                or not np.allclose(k, scale @ np.asarray(capture.K_colmap), atol=1e-10, rtol=1e-12)):
            raise ValueError("undistorter refit/cropped a frozen camera/pose instead of declared rescaling")
        jpeg = workspace / "images" / safe_relative_name(capture.name)
        captures[image_id] = FrozenCapture(
            image_id, capture.name, str(jpeg), capture.source_image_uri, sha256_file(jpeg), capture.session_id,
            capture.frame_index, capture.evaluation_role, int(image.camera_id),
            int(camera.width), int(camera.height),
            k.tolist(), (SHIFT_TO_NATIVE @ k).tolist(), k.tolist(), pose.tolist())
    return captures


def prepare_patch_match_workspace(workspace: Path, scene: FrozenScene) -> None:
    """Adapt COLMAP's corner-origin undistorter to its 4.0.4 integer-pixel MVS.

    Pinned upstream model.cc copies K unchanged; patch_match_cuda.cu uses
    integer row/col in ComputePointAtDepth. Consequently its dense workspace
    must contain K_native, not K_colmap. Preserve the canonical undistorter
    model separately and explicitly shift K AND sparse observations by -0.5.
    This is exact ray-preserving representation transport, never camera refit.
    """
    canonical = workspace / "sparse_colmap"
    if canonical.exists():
        raise ValueError("PatchMatch workspace convention was already adapted")
    captures = _canonical_dense_captures(workspace / "sparse", workspace, scene)
    os.rename(workspace / "sparse", canonical)
    dense_scene = FrozenScene(canonical, scene.keyframes_manifest, workspace / "images",
                              scene.model_hashes, scene.keyframes_sha256, COLMAP_CORNER, captures)
    _write_fixed_subset(dense_scene, workspace / "sparse", target_pixel_convention=OPENCV_NATIVE_CENTER)


def _workspace_views(workspace: Path, scene: FrozenScene, sources: Mapping[int, Sequence[int]]) -> tuple[list[dict[str, Any]], list[int]]:
    import cv2
    import pycolmap
    output_model = workspace / "sparse"
    reconstruction = pycolmap.Reconstruction(str(output_model))
    # The native text model deliberately has no rigs/frames; text order equals
    # COLMAP4.0.4 Reconstruction::RegImageIds() order, used by mvs::Model.
    order = colmap_image_order(output_model)
    image_order = [int(i) for i in reconstruction.reg_image_ids()]
    if (set(order) != {(i, c.name) for i, c in scene.captures.items()}
            or image_order != [i for i, _ in order]):
        raise ValueError("dense graph image ordering differs from the registered native text model")
    canonical_captures = _canonical_dense_captures(workspace / "sparse_colmap", workspace, scene)
    canonical_model = pycolmap.Reconstruction(str(workspace / "sparse_colmap"))
    for point_id, point in reconstruction.points3D.items():
        if (point_id not in canonical_model.points3D
                or not np.array_equal(point.xyz, canonical_model.points3D[point_id].xyz)):
            raise ValueError("native PatchMatch workspace created/refit a world point")
    index_of = {image_id: index for index, image_id in enumerate(image_order)}
    rows = []
    for image_id in image_order:
        capture = scene.captures[image_id]
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        if str(camera.model).split(".")[-1] != "PINHOLE":
            raise ValueError("undistorter produced an unsupported camera")
        pose = validate_pose(np.asarray(image.cam_from_world().matrix()))
        if not np.allclose(pose, capture.cam_from_world, atol=1e-12, rtol=0):
            raise ValueError("undistorter changed a frozen pose")
        k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        scale = np.diag([camera.width / capture.width, camera.height / capture.height, 1.])
        expected_native = SHIFT_TO_NATIVE @ scale @ np.asarray(capture.K_colmap)
        if (not np.allclose(k, expected_native, atol=1e-10, rtol=1e-12)
                or not np.allclose(k, canonical_captures[image_id].K_native, atol=1e-10, rtol=1e-12)):
            raise ValueError("PatchMatch model does not carry explicitly transported native-center K")
        jpeg = workspace / "images" / safe_relative_name(capture.name)
        probe = cv2.imread(str(jpeg), cv2.IMREAD_GRAYSCALE)
        if probe is None or probe.shape != (camera.height, camera.width):
            raise ValueError("partial workspace undistorted image")
        row = {"image_id": image_id, "name": capture.name, "session_id": capture.session_id,
               "width": int(camera.width), "height": int(camera.height),
               "K_colmap": (SHIFT_TO_COLMAP @ k).tolist(),
               "K_native": k.tolist(), "cam_from_world": pose.tolist(),
               "native_source_to_workspace": (k @ np.linalg.inv(np.asarray(capture.K_native))).tolist(),
               "workspace_image_sha256": sha256_file(jpeg), "source_image_ids": list(sources[image_id])}
        for kind in ("photometric", "geometric"):
            for folder, channels in (("depth_maps", 1), ("normal_maps", 3)):
                path = workspace / "stereo" / folder / (capture.name + f".{kind}.bin")
                data = read_colmap_dense_array(path, channels=channels)
                if data.shape[:2] != (camera.height, camera.width):
                    raise ValueError("partial/wrong-resolution MVS depth or normal map")
            graph = read_colmap_consistency_graph(workspace / "stereo/consistency_graphs" /
                        (capture.name + f".{kind}.bin"), num_images=len(image_order),
                        allowed_image_indices=[index_of[j] for j in sources[image_id]])
            if (graph.height, graph.width) != (camera.height, camera.width):
                raise ValueError("consistency graph resolution differs from its depth")
        rows.append(row)
    return rows, image_order


def artifact_members(root: Path, *, exclude: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("immutable offline artifacts cannot contain symlinks")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in exclude:
                result[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return result


def fsync_artifact_tree(root: Path) -> None:
    """Make every staged file and directory durable before atomic publication."""
    directories = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("immutable offline artifacts cannot contain symlinks")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted([root, *directories], key=lambda p: len(p.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_artifact_members(root: Path, members: Mapping[str, Mapping[str, Any]], *,
                             exclude: Sequence[str] = ()) -> None:
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("immutable offline artifacts cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual - set(exclude) != set(members):
        raise ValueError("partial/extra offline artifact workspace members")
    for name, expected in members.items():
        file = root / safe_relative_name(name)
        if (file.is_symlink() or file.resolve(strict=True).is_relative_to(root.resolve()) is False
                or file.stat().st_size != expected["size_bytes"] or sha256_file(file) != expected["sha256"]):
            raise ValueError(f"changed/partial offline artifact member: {name}")


def run_offline_mvs(request_path: Path | str, output_dir: Path | str, *,
                    heavy_resource_lock: Path | str) -> Path:
    """Parent-only launch. Staging remains private until every required output verifies."""
    request_file = Path(request_path).resolve(strict=True)
    request, scene, sources, parameters = _load_request(request_file)
    limits_data = dict(request["resources"])
    limits_data["cpu_affinity"] = tuple(limits_data["cpu_affinity"])
    limits = MvsResourceLimits(**limits_data)
    envelope = _require_parent_resource_envelope(limits)
    target = Path(output_dir).absolute()
    if target.exists():
        raise ValueError("MVS output must be a new independent immutable directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free < request["estimates"]["disk_upper_estimate_bytes"]:
        raise ValueError("insufficient disk for the declared complete MVS workspace estimate")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    workspace = staging / "workspace"
    logs = staging / "logs"
    logs.mkdir()
    native = staging / "native_images"
    native.mkdir()
    commands = []
    started = time.monotonic()
    environment = dict(os.environ, OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", QT_QPA_PLATFORM="offscreen")
    executable = request["executable"]["path"]

    def invoke(command: str, arguments: Sequence[str], log_name: str) -> str:
        if command not in {"-h", "image_undistorter", "patch_match_stereo", "stereo_fusion"}:
            raise ValueError("non-dense/refitting COLMAP command is forbidden")
        args = ["taskset", "--cpu-list", ",".join(map(str, limits.cpu_affinity)), executable, command, *arguments]
        began = time.monotonic()
        path = logs / log_name
        with path.open("wb") as stream:
            completed = subprocess.run(args, cwd=staging, env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, check=False)
            stream.flush()
            os.fsync(stream.fileno())
        commands.append({"command": command, "argv": args, "returncode": completed.returncode,
                         "elapsed_seconds": time.monotonic() - began, "log": path.relative_to(staging).as_posix()})
        if completed.returncode:
            raise RuntimeError(f"MVS stage failed ({completed.returncode}); private staging: {staging}")
        return path.read_text(encoding="utf-8", errors="replace") if command == "-h" else ""

    try:
        with ExclusiveResourceLease(heavy_resource_lock, "gpu_heavy"):
            banner = invoke("-h", [], "colmap-version.log")
            _validate_colmap_banner(banner)
            shutil.copyfile(request_file, staging / "request.json")
            shutil.copyfile(scene.keyframes_manifest, staging / "keyframes.jsonl")
            shutil.copyfile(request["executable"]["cuda_build_receipt_path"], staging / "colmap-build.json")
            frozen = staging / "frozen_model"
            frozen.mkdir()
            for name, digest in scene.model_hashes.items():
                shutil.copyfile(scene.model_dir / name, frozen / name)
                if sha256_file(frozen / name) != digest:
                    raise ValueError("frozen model changed while snapshotting")
            for capture in scene.captures.values():
                dest = native / safe_relative_name(capture.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(capture.image_path, dest)
                if sha256_file(dest) != capture.image_sha256:
                    raise ValueError("native image changed while snapshotting")
            subset = staging / "input_model"
            _write_fixed_subset(scene, subset, target_pixel_convention=COLMAP_CORNER)
            undistorter = {"image_path": str(native), "input_path": str(subset), "output_path": str(workspace),
                           "output_type": "COLMAP", "copy_policy": "copy", "num_threads": 8,
                           "jpeg_quality": 100, "num_patch_match_src_images": 20,
                           "max_image_size": parameters.max_image_size, "blank_pixels": 0.,
                           "min_scale": 1., "max_scale": 1., "roi_min_x": 0., "roi_min_y": 0.,
                           "roi_max_x": 1., "roi_max_y": 1.}
            invoke("image_undistorter", _options_arguments(undistorter), "undistorter.log")
            prepare_patch_match_workspace(workspace, scene)
            config = _source_config(scene.captures, sources)
            (workspace / "stereo/patch-match.cfg").write_text(config, encoding="utf-8")
            (workspace / "stereo/fusion.cfg").write_text("".join(scene.captures[i].name + "\n" for i in sorted(sources)), encoding="utf-8")
            base = ["--workspace_path", str(workspace), "--workspace_format", "COLMAP"]
            for geometric in (False, True):
                invoke("patch_match_stereo", [*base, *_options_arguments(
                    parameters.patch_match_options(geometric=geometric), "PatchMatchStereo.")],
                    "geometric.log" if geometric else "photometric.log")
            if parameters.stereo_fusion:
                fusion = {"num_threads": 8, "max_image_size": parameters.max_image_size,
                          "min_num_pixels": 5, "max_num_pixels": 10000, "max_traversal_depth": 100,
                          "max_reproj_error": 2., "max_depth_error": .01, "max_normal_error": 10.,
                          "check_num_images": 50, "use_cache": True, "cache_size": parameters.cache_size}
                invoke("stereo_fusion", [*base, "--input_type", "geometric", "--output_path", str(workspace / "fused.ply"),
                       *_options_arguments(fusion, "StereoFusion.")], "fusion.log")
                if not (workspace / "fused.ply").is_file() or (workspace / "fused.ply").stat().st_size == 0:
                    raise ValueError("partial stereo-fusion output")
            views, order = _workspace_views(workspace, scene, sources)
            _assert_scene_unchanged(scene)
            if sha256_file(Path(executable)) != request["executable"]["sha256"]:
                raise ValueError("COLMAP executable changed while running")
            receipt = {"schema": RECEIPT_SCHEMA, "status": "COMPLETE", "evidence_family": MVS_FAMILY,
                       "geometry_origin": "MVS_ONLY", "authority": "SUPPLEMENTAL_DISPLAY_ONLY",
                       "deployment_authorized": False, "external_absolute_truth": "UNAVAILABLE",
                       "source_scope": MAPPING_SCOPE, "holdout_scope": MAPPING_HOLDOUT_SCOPE,
                       "holdout_status": "PENDING_MAPPING_INTERNAL_BLOCKED_HOLDOUT",
                       "request_sha256": request["request_sha256"], "model_hashes": dict(scene.model_hashes),
                       "keyframes_sha256": scene.keyframes_sha256, "map_revision_id": request["map_revision_id"],
                       "coordinate_frame_id": request["coordinate_frame_id"], "source_list_sha256": request["source_list_sha256"],
                       "input_pixel_convention": scene.input_pixel_convention,
                       "canonical_model_pixel_convention": COLMAP_CORNER,
                       "workspace_pixel_convention": OPENCV_NATIVE_CENTER,
                       "array_pixel_convention": OPENCV_NATIVE_CENTER, "image_index_to_image_id": order,
                       "views": views, "commands": commands, "resources": envelope,
                       "elapsed_seconds": time.monotonic() - started,
                       "members": artifact_members(staging)}
            _write_json_atomic(staging / "receipt.json", receipt)
            read_offline_mvs(staging / "receipt.json", expected_model_hashes=scene.model_hashes,
                             expected_keyframes_sha256=scene.keyframes_sha256)
            fsync_artifact_tree(staging)
            if target.exists():
                raise ValueError("MVS output was concurrently published")
            os.rename(staging, target)
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except Exception:
        # Retain private logs for diagnosis; no receipt is accepted without a
        # complete set, and the requested final directory remains unpublished.
        raise
    return target / "receipt.json"


class OfflineMvsArtifact:
    evidence_family = "fixed_pose_mvs"

    def __init__(self, root: Path, receipt: dict[str, Any], request: dict[str, Any], manifest_sha256: str) -> None:
        self.root, self.receipt, self.request = root, receipt, request
        self.manifest_sha256 = manifest_sha256
        self.views = {int(row["image_id"]): row for row in receipt["views"]}
        self.captures = {int(row["image_id"]): FrozenCapture(**row) for row in request["images"]}
        centers = np.asarray([capture.center for capture in self.captures.values()])
        self._camera_cell_width = max(float(np.linalg.norm(np.ptp(centers, axis=0))) * .05, 1e-8)
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, ConsistencyGraph]] = OrderedDict()

    def _view_data(self, image_id: int) -> tuple[np.ndarray, np.ndarray, ConsistencyGraph]:
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
            return self._cache[image_id]
        row = self.views[image_id]
        name = row["name"] + ".geometric.bin"
        stereo = self.root / "workspace/stereo"
        depth = read_colmap_dense_array(stereo / "depth_maps" / name, channels=1)
        normals = read_colmap_dense_array(stereo / "normal_maps" / name, channels=3)
        order = self.receipt["image_index_to_image_id"]
        graph = read_colmap_consistency_graph(stereo / "consistency_graphs" / name, num_images=len(order),
                    allowed_image_indices=[order.index(i) for i in row["source_image_ids"]])
        self._cache[image_id] = depth, normals, graph
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return depth, normals, graph

    def compare_candidates(self, source_image_id: int, xyz: np.ndarray, *,
                            max_log_residual: float = .08) -> dict[str, np.ndarray]:
        """Compare optical-Z in each named neighbor's fixed measurement camera.

        support_image_ids identifies actual captures, not additional modalities.
        residual_image_ids repeats that measurement camera for known entries;
        UNKNOWN has no residual or evidence identity.
        """
        points = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
        source = self.captures.get(int(source_image_id))
        if source is None:
            return _empty_evidence(len(points))
        neighbors = [int(i) for i in self.request["source_lists"][str(source_image_id)]]
        residuals = np.full((len(neighbors), len(points)), np.nan)
        states = np.zeros(residuals.shape, dtype=np.uint8)
        evidence_ids = np.full(residuals.shape, -1, dtype=np.int64)
        source_pose = np.asarray(source.cam_from_world)
        source_camera = points @ source_pose[:, :3].T + source_pose[:, 3]
        source_front = np.isfinite(source_camera).all(axis=1) & (source_camera[:, 2] > 0)
        for view_index, image_id in enumerate(neighbors):
            row = self.views[image_id]
            pose = np.asarray(row["cam_from_world"])
            camera = points @ pose[:, :3].T + pose[:, 3]
            uv = np.full((len(points), 2), np.nan)
            front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
            projected = camera[front] @ np.asarray(row["K_native"]).T
            uv[front] = projected[:, :2] / projected[:, 2, None]
            depth, normals, graph = self._view_data(image_id)
            measured, usable = coherent_depth_sample(depth, uv)
            graph_usable, graph_rows = graph.support_at(uv, min_support=int(self.request["parameters"]["filter_min_num_consistent"]))
            rays_a, rays_b = points - source.center, points - self.captures[image_id].center
            denominator = np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1)
            cosine = np.ones(len(points))
            np.divide(np.einsum("ij,ij->i", rays_a, rays_b), denominator, out=cosine, where=denominator > 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
            usable &= source_front & front & graph_usable & (angle >= 2.)
            indices = np.flatnonzero(usable)
            if len(indices):
                xy = np.rint(uv[indices]).astype(np.int64)
                normal = normals[xy[:, 1], xy[:, 0]]
                normal_length = np.linalg.norm(normal, axis=1)
                usable[indices] &= np.isfinite(normal).all(axis=1) & (np.abs(normal_length - 1.) <= .05)
            verdict = signed_depth_verdict(camera[:, 2], measured, usable=usable,
                                           max_log_residual=max_log_residual)
            residuals[view_index], states[view_index] = verdict["signed_log_residual"], verdict["states"]
            evidence_ids[view_index, usable] = graph_rows[usable]
        support_ids = np.asarray(neighbors, dtype=np.int64)
        residual_ids = np.where(states != UNKNOWN, support_ids[:, None], -1)
        return {"signed_log_residual": residuals, "states": states,
                "support_image_ids": support_ids, "evidence_ids": evidence_ids,
                "residual_image_ids": residual_ids}

    def iter_surface_candidates(self, *, stride: int = 4, min_support: int = 2,
                                max_log_residual: float = .08, chunk_points: int = 65536):
        """Yield bounded, real MVS-only display candidates; never fill/publish.

        Two distinct other capture/ray groups, coherent patches and no usable
        free-space contradiction are required. Repeated cameras in one session's
        32-frame segment or camera cell, or rays within two degrees, cannot multiply support.
        Whole workspaces can be consumed without materializing every surface.
        """
        from .metric_dense import aggregate_view_evidence

        if stride < 1 or min_support < 2 or not 1 <= chunk_points <= 65536:
            raise ValueError("MVS-only candidates require bounded chunks and two other capture groups")
        metadata = {"geometry_origin": "MVS_ONLY", "source_algorithm": MVS_FAMILY,
                    "authority": "SUPPLEMENTAL_DISPLAY_ONLY", "deployment_authorized": False,
                    "source_scope": MAPPING_SCOPE, "holdout_scope": MAPPING_HOLDOUT_SCOPE,
                    "holdout_status": "PENDING_MAPPING_INTERNAL_BLOCKED_HOLDOUT", "coordinate_units": "MAP_UNITS",
                    "map_revision_id": self.request["map_revision_id"],
                    "coordinate_frame_id": self.request["coordinate_frame_id"],
                    "model_hashes": self.request["model_hashes"],
                    "mvs_manifest_sha256": self.manifest_sha256}
        for image_id in self.request["reference_image_ids"]:
            row = self.views[image_id]
            depth, normals, graph = self._view_data(image_id)
            ys, xs = np.mgrid[2:row["height"] - 2:stride, 2:row["width"] - 2:stride]
            grid_uv = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float64)
            for start in range(0, len(grid_uv), chunk_points):
                uv = grid_uv[start:start + chunk_points]
                z, usable = coherent_depth_sample(depth, uv)
                graph_usable, _ = graph.support_at(uv, min_support=min_support)
                usable &= graph_usable
                xy = uv.astype(np.int64)
                source_normals = normals[xy[:, 1], xy[:, 0]]
                usable &= (np.isfinite(source_normals).all(axis=1)
                           & (np.abs(np.linalg.norm(source_normals, axis=1) - 1.) <= .05))
                uv, z, source_normals = uv[usable], z[usable], source_normals[usable]
                if not len(z):
                    continue
                homogeneous = np.column_stack((uv, np.ones(len(uv))))
                camera = (homogeneous @ np.linalg.inv(np.asarray(row["K_native"])).T) * z[:, None]
                pose = np.asarray(row["cam_from_world"])
                world = (camera - pose[:, 3]) @ pose[:, :3]
                evidence = self.compare_candidates(image_id, world, max_log_residual=max_log_residual)
                agree = evidence["states"] == AGREE
                neighbor_ids = evidence["support_image_ids"]
                rays = np.asarray([world - self.captures[int(i)].center for i in neighbor_ids])
                view_keys = [{"family": "fixed_pose_mvs", "view_id": int(i),
                              "session": self.captures[int(i)].session_id,
                              "temporal_segment": self.captures[int(i)].frame_index // 32,
                              "camera_cell": tuple(np.floor(self.captures[int(i)].center /
                                                           self._camera_cell_width).astype(int))}
                             for i in neighbor_ids]
                grouped = aggregate_view_evidence(evidence["signed_log_residual"], rays, np.zeros_like(agree),
                    tolerance=max_log_residual, separation_deg=2., view_keys=view_keys, detailed=True)
                support = grouped.geometric_support
                selected = np.flatnonzero((support >= min_support) & (grouped.conflict == 0))
                if not len(selected):
                    continue
                offsets, observers = [0], []
                for point_index in selected:
                    actual = neighbor_ids[agree[:, point_index]].tolist()
                    observers.extend([image_id, *actual])
                    offsets.append(offsets[-1] + len(actual) + 1)
                world_normals = source_normals[selected] @ pose[:, :3]
                world_normals /= np.linalg.norm(world_normals, axis=1)[:, None]
                yield {**metadata, "xyz": world[selected],
                       "normals": world_normals, "normal_valid": np.ones(len(selected), dtype=bool),
                       "source_image_id": np.full(len(selected), image_id, dtype=np.int64),
                       "support_count": support[selected].astype(np.uint16),
                       "observation_offsets": np.asarray(offsets, dtype=np.int64),
                       "observation_image_id": np.asarray(observers, dtype=np.int64)}

    def surface_candidates(self, *, stride: int = 4, min_support: int = 2,
                            max_log_residual: float = .08,
                            max_materialized_bytes: int = 1024 ** 3) -> dict[str, Any]:
        """Materialize a canary layer; full runs use iter_surface_candidates.

        Exceeding the caller's explicit memory budget raises rather than dropping
        points or publishing a partial surface. This never authorizes deployment.
        """
        if not 0 < max_materialized_bytes <= 4 * 1024 ** 3:
            raise ValueError("materialized MVS layer budget must be at most four GiB")
        chunks = []
        byte_count = 0
        for chunk in self.iter_surface_candidates(stride=stride, min_support=min_support,
                                                  max_log_residual=max_log_residual):
            byte_count += sum(value.nbytes for value in chunk.values() if isinstance(value, np.ndarray))
            if 2 * byte_count > max_materialized_bytes:
                raise ValueError("MVS layer exceeds memory budget; consume iter_surface_candidates")
            chunks.append(chunk)
        point_fields = ("xyz", "normals", "normal_valid", "source_image_id", "support_count")
        result = {field: np.concatenate([chunk[field] for chunk in chunks]) for field in point_fields} if chunks else {
            "xyz": np.empty((0, 3)), "normals": np.empty((0, 3)), "normal_valid": np.empty(0, bool),
            "source_image_id": np.empty(0, np.int64), "support_count": np.empty(0, np.uint16)}
        offsets = [np.array([0], dtype=np.int64)]
        total = 0
        for chunk in chunks:
            offsets.append(chunk["observation_offsets"][1:] + total)
            total += int(chunk["observation_offsets"][-1])
        result["observation_offsets"] = np.concatenate(offsets)
        result["observation_image_id"] = (np.concatenate([chunk["observation_image_id"] for chunk in chunks])
                                          if chunks else np.empty(0, np.int64))
        result.update(geometry_origin="MVS_ONLY", source_algorithm=MVS_FAMILY,
                      authority="SUPPLEMENTAL_DISPLAY_ONLY", deployment_authorized=False,
                      source_scope=MAPPING_SCOPE, holdout_scope=MAPPING_HOLDOUT_SCOPE,
                      holdout_status="PENDING_MAPPING_INTERNAL_BLOCKED_HOLDOUT", coordinate_units="MAP_UNITS",
                      map_revision_id=self.request["map_revision_id"],
                      coordinate_frame_id=self.request["coordinate_frame_id"],
                      model_hashes=self.request["model_hashes"], mvs_manifest_sha256=self.manifest_sha256)
        return result


def _empty_evidence(n_points: int) -> dict[str, np.ndarray]:
    return {"signed_log_residual": np.empty((0, n_points), dtype=np.float64),
            "states": np.empty((0, n_points), dtype=np.uint8),
            "support_image_ids": np.empty(0, dtype=np.int64),
            "evidence_ids": np.empty((0, n_points), dtype=np.int64),
            "residual_image_ids": np.empty((0, n_points), dtype=np.int64)}


def read_offline_mvs(receipt_path: Path | str, *, expected_model_hashes: Mapping[str, str],
                      expected_keyframes_sha256: str, expected_manifest_sha256: str | None = None) -> OfflineMvsArtifact:
    file = Path(receipt_path).resolve(strict=True)
    root = file.parent
    digest = sha256_file(file)
    receipt = json.loads(file.read_text(encoding="utf-8"))
    if (receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "COMPLETE"
            or receipt.get("evidence_family") != MVS_FAMILY
            or receipt.get("model_hashes") != dict(expected_model_hashes)
            or receipt.get("keyframes_sha256") != expected_keyframes_sha256
            or receipt.get("geometry_origin") != "MVS_ONLY"
            or receipt.get("authority") != "SUPPLEMENTAL_DISPLAY_ONLY"
            or receipt.get("deployment_authorized") is not False
            or receipt.get("source_scope") != MAPPING_SCOPE
            or receipt.get("holdout_scope") != MAPPING_HOLDOUT_SCOPE
            or receipt.get("array_pixel_convention") != OPENCV_NATIVE_CENTER
            or receipt.get("workspace_pixel_convention") != OPENCV_NATIVE_CENTER
            or receipt.get("canonical_model_pixel_convention") != COLMAP_CORNER
            or (expected_manifest_sha256 is not None and digest != expected_manifest_sha256)):
        raise ValueError("unsupported/mismatched/partial MVS receipt")
    verify_artifact_members(root, receipt["members"], exclude=[file.name])
    request = json.loads((root / "request.json").read_text(encoding="utf-8"))
    if (request.get("schema") != REQUEST_SCHEMA
            or canonical_sha256({k: v for k, v in request.items() if k != "request_sha256"}) != request.get("request_sha256")
            or request.get("implementation_sha256") != sha256_file(Path(__file__))
            or request["request_sha256"] != receipt["request_sha256"]
            or request["model_hashes"] != receipt["model_hashes"]
            or request["keyframes_sha256"] != receipt["keyframes_sha256"]
            or request["source_list_sha256"] != receipt["source_list_sha256"]
            or request["map_revision_id"] != receipt["map_revision_id"]
            or request["coordinate_frame_id"] != receipt["coordinate_frame_id"]
            or request["input_pixel_convention"] != receipt["input_pixel_convention"]):
        raise ValueError("MVS request/receipt identity mismatch")
    _validate_request_policy(request)
    parameters = MvsParameters(**request["parameters"])
    parameters.validate()
    sources = {int(i): list(map(int, neighbors)) for i, neighbors in request["source_lists"].items()}
    _validate_source_lists(sources, request["reference_image_ids"],
                           min_support=parameters.filter_min_num_consistent)
    limits = MvsResourceLimits(**request["resources"])
    limits.validate()
    envelope = receipt["resources"]
    affinity = envelope["cpu_affinity"]
    maxima = {"memory.high": limits.memory_high_bytes, "memory.max": limits.memory_max_bytes,
              "memory.swap.max": limits.swap_max_bytes, "cpu.weight": limits.cpu_weight}
    if (not affinity or len(affinity) != len(set(affinity))
            or not set(affinity) <= set(limits.cpu_affinity) or envelope["nice"] < limits.nice
            or not envelope.get("cgroup")
            or any(type(envelope.get(name)) is not int or not 0 <= envelope[name] <= maximum
                   for name, maximum in maxima.items())):
        raise ValueError("MVS receipt does not preserve the enforced parent resource envelope")
    captures = {int(row["image_id"]): FrozenCapture(**row) for row in request["images"]}
    if set(captures) != set(sources) or len(captures) != len(request["images"]):
        raise ValueError("incomplete/duplicate MVS capture set")
    if frozen_model_hashes(root / "frozen_model") != request["model_hashes"]:
        raise ValueError("MVS frozen model snapshot differs from its request")
    if sha256_file(root / "keyframes.jsonl") != request["keyframes_sha256"]:
        raise ValueError("MVS keyframe snapshot differs from its request")
    cell_width = verify_frozen_capture_snapshot(root, captures,
                                                input_pixel_convention=request["input_pixel_convention"])
    config_path = root / "workspace/stereo/patch-match.cfg"
    if (hashlib.sha256(_source_config(captures, sources).encode()).hexdigest() != request["source_list_sha256"]
            or sha256_file(config_path) != request["source_list_sha256"]):
        raise ValueError("MVS source-list content is not the explicitly pinned list")
    scene = FrozenScene(root / "frozen_model", root / "keyframes.jsonl", root / "native_images",
                        request["model_hashes"], request["keyframes_sha256"], request["input_pixel_convention"], captures)
    views, order = _workspace_views(root / "workspace", scene, sources)
    if views != receipt["views"] or order != receipt["image_index_to_image_id"]:
        raise ValueError("MVS output K/pose/resolution/graph origins differ from the receipt")
    allowed_commands = ["-h", "image_undistorter", "patch_match_stereo", "patch_match_stereo"]
    if request["parameters"]["stereo_fusion"]:
        allowed_commands.append("stereo_fusion")
    if [row["command"] for row in receipt["commands"]] != allowed_commands or any(row["returncode"] for row in receipt["commands"]):
        raise ValueError("MVS receipt includes missing, failed or refitting commands")
    for row in receipt["commands"]:
        argv = row["argv"]
        if (len(argv) < 5 or argv[:2] != ["taskset", "--cpu-list"]
                or argv[2] != ",".join(map(str, limits.cpu_affinity))
                or argv[3] != request["executable"]["path"] or argv[4] != row["command"]):
            raise ValueError("MVS executed argv is not the declared fixed-pose stage")
    banner_path = root / safe_relative_name(receipt["commands"][0]["log"])
    _validate_colmap_banner(banner_path.read_text(encoding="utf-8", errors="replace"))
    build_path = root / "colmap-build.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    if (sha256_file(build_path) != request["executable"]["cuda_build_receipt_sha256"]
            or build.get("name") != "colmap"
            or build.get("version") != SUPPORTED_COLMAP_VERSION
            or build.get("build") != request["executable"]["build"]
            or not str(build.get("build", "")).startswith("cuda_")
            or not any(str(dep).startswith("cuda-cudart ") for dep in build.get("depends", []))):
        raise ValueError("MVS receipt does not bind a CUDA COLMAP build")
    artifact = OfflineMvsArtifact(root, receipt, request, digest)
    artifact._camera_cell_width = cell_width
    return artifact
