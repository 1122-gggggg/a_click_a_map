"""Native-SIFT reciprocal RGB geometry with fixed K/R/t, offline only.

Match selection, epipolar filtering and DLT triangulation have no depth-model
inputs. Association to a MoGe/surface candidate happens only after this immutable
artifact exists. The original sparse reconstruction is only read, never edited,
retriangulated or bundle-adjusted. These points share the camera gauge and are
supplemental evidence, not independent metric ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .geometry_contract import validate_pose
from .inventory import sha256_file
from .offline_mvs import (
    AGREE, UNKNOWN, MAPPING_SCOPE, MAPPING_HOLDOUT_SCOPE, OPENCV_NATIVE_CENTER,
    FrozenCapture, FrozenScene,
    _assert_scene_unchanged, _empty_evidence, _write_json_atomic,
    artifact_members, canonical_sha256, frozen_model_hashes, fsync_artifact_tree, load_frozen_scene,
    pixel_center_intrinsics,
    safe_relative_name, signed_depth_verdict, validate_mapping_source,
    verify_artifact_members, verify_frozen_capture_snapshot,
)

PHOTO_SCHEMA = "offline-reciprocal-photographic-evidence/v1"
PAIR_SCHEMA = "offline-reciprocal-photographic-pairs/v1"
SIFT_SCHEMA = "native-sift-cache/v1"
SIFT_PREPROCESSING = "NATIVE_HASHED_JPEG_CV2_IMDECODE_GRAYSCALE_NO_RESIZE_NO_DEPTH"
PHOTO_FAMILY = "RECIPROCAL_NATIVE_SIFT_TRIANGULATION"
ARRAY_DTYPES = {"source_image_id": "<i8", "target_image_id": "<i8", "uv_source": "<f8",
                "uv_target": "<f8", "xyz": "<f8", "triangulation_angle_deg": "<f8",
                "reprojection_error_px": "<f8", "descriptor_ratio": "<f8", "match_id": "<i8",
                "source_feature_index": "<i8", "target_feature_index": "<i8"}


@dataclass(frozen=True)
class PhotoParameters:
    max_features: int = 8192
    octave_layers: int = 3
    contrast_threshold: float = .04
    edge_threshold: float = 10.
    sigma: float = 1.6
    ratio_threshold: float = .8
    max_epipolar_error_px: float = 2.
    min_triangulation_angle_deg: float = 2.
    max_reprojection_error_px: float = 2.

    def validate(self) -> None:
        values = asdict(self)
        if (not all(math.isfinite(value) for value in values.values())
                or self.max_features < 2 or self.octave_layers < 1
                or self.contrast_threshold <= 0 or self.edge_threshold <= 0 or self.sigma <= 0
                or not 0 < self.ratio_threshold < 1
                or not 0 < self.max_epipolar_error_px <= 2
                or self.min_triangulation_angle_deg < 2
                or not 0 < self.max_reprojection_error_px <= 2):
            raise ValueError("photographic geometry cannot relax the two-degree/two-pixel gates")


def fixed_fundamental_matrix(k_a: np.ndarray, pose_a: np.ndarray,
                              k_b: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Known-pose fundamental matrix; never fit a new F, pose or intrinsics."""
    a, b = validate_pose(pose_a), validate_pose(pose_b)
    ka = pixel_center_intrinsics(k_a, from_convention=OPENCV_NATIVE_CENTER,
                                to_convention=OPENCV_NATIVE_CENTER)
    kb = pixel_center_intrinsics(k_b, from_convention=OPENCV_NATIVE_CENTER,
                                to_convention=OPENCV_NATIVE_CENTER)
    rotation = b[:, :3] @ a[:, :3].T
    translation = b[:, 3] - rotation @ a[:, 3]
    if np.linalg.norm(translation) <= 1e-12:
        raise ValueError("zero-baseline/pure-rotation photographs cannot triangulate")
    x, y, z = translation
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    fundamental = np.linalg.inv(kb).T @ skew @ rotation @ np.linalg.inv(ka)
    return fundamental / np.linalg.norm(fundamental)


def known_epipolar_inliers(uv_a: np.ndarray, uv_b: np.ndarray, fundamental: np.ndarray,
                            *, max_error_px: float = 2.) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < max_error_px <= 2:
        raise ValueError("epipolar gate cannot exceed two native pixels")
    a, b = np.asarray(uv_a, dtype=np.float64), np.asarray(uv_b, dtype=np.float64)
    f = np.asarray(fundamental, dtype=np.float64)
    if (a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape
            or f.shape != (3, 3) or not np.isfinite(f).all()):
        raise ValueError("invalid photographic epipolar inputs")
    xa, xb = np.column_stack((a, np.ones(len(a)))), np.column_stack((b, np.ones(len(b))))
    line_b, line_a = xa @ f.T, xb @ f
    numerator = np.abs(np.einsum("ij,ij->i", xb, line_b))
    denominator_a, denominator_b = np.linalg.norm(line_a[:, :2], axis=1), np.linalg.norm(line_b[:, :2], axis=1)
    errors = np.full((len(a), 2), np.inf)
    np.divide(numerator, denominator_a, out=errors[:, 0], where=denominator_a > 1e-12)
    np.divide(numerator, denominator_b, out=errors[:, 1], where=denominator_b > 1e-12)
    return np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (errors.max(axis=1) <= max_error_px), errors


def reciprocal_match_indices(forward_indices: np.ndarray, forward_distances: np.ndarray,
                               reverse_indices: np.ndarray, reverse_distances: np.ndarray, *,
                               ratio_threshold: float = .8) -> tuple[np.ndarray, np.ndarray]:
    """Require mutual nearest neighbors and an unambiguous ratio in BOTH directions."""
    if not 0 < ratio_threshold < 1:
        raise ValueError("descriptor ratio must be between zero and one")
    fi, fd = np.asarray(forward_indices), np.asarray(forward_distances, dtype=np.float64)
    ri, rd = np.asarray(reverse_indices), np.asarray(reverse_distances, dtype=np.float64)
    if (fi.ndim != 2 or fi.shape[1] != 2 or fd.shape != fi.shape
            or ri.ndim != 2 or ri.shape[1] != 2 or rd.shape != ri.shape
            or fi.dtype.kind not in "iu" or ri.dtype.kind not in "iu"):
        raise ValueError("reciprocal matcher needs two neighbor indices/distances per descriptor")
    forward_ratio = np.full(len(fi), np.inf)
    reverse_ratio = np.full(len(ri), np.inf)
    np.divide(fd[:, 0], fd[:, 1], out=forward_ratio,
              where=np.isfinite(fd).all(axis=1) & (fd[:, 1] > 0))
    np.divide(rd[:, 0], rd[:, 1], out=reverse_ratio,
              where=np.isfinite(rd).all(axis=1) & (rd[:, 1] > 0))
    forward_ok = ((fi >= 0).all(axis=1) & (fi < len(ri)).all(axis=1)
                  & (fi[:, 0] != fi[:, 1]) & np.isfinite(fd).all(axis=1)
                  & (fd >= 0).all(axis=1) & (fd[:, 0] <= fd[:, 1]) & (forward_ratio < ratio_threshold))
    reverse_ok = ((ri >= 0).all(axis=1) & (ri < len(fi)).all(axis=1)
                  & (ri[:, 0] != ri[:, 1]) & np.isfinite(rd).all(axis=1) & (rd >= 0).all(axis=1)
                  & (rd[:, 0] <= rd[:, 1]) & (reverse_ratio < ratio_threshold))
    candidates = np.flatnonzero(forward_ok)
    targets = fi[candidates, 0].astype(np.int64)
    keep = reverse_ok[targets] & (ri[targets, 0] == candidates)
    matches = np.column_stack((candidates[keep], targets[keep])).astype(np.int64)
    ratios = np.column_stack((forward_ratio[candidates[keep]], reverse_ratio[targets[keep]]))
    return matches, ratios


def reciprocal_sift_matches(descriptors_a: np.ndarray, descriptors_b: np.ndarray, *,
                              ratio_threshold: float = .8) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    a, b = np.asarray(descriptors_a), np.asarray(descriptors_b)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != 128 or b.shape[1] != 128
            or a.dtype != np.float32 or b.dtype != np.float32
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("requires actual finite float32 native SIFT descriptors")
    if len(a) < 2 or len(b) < 2:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 2), dtype=np.float64)
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def neighbors(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.full((len(source), 2), -1, dtype=np.int64)
        distances = np.full((len(source), 2), np.inf)
        for matches in matcher.knnMatch(source, target, k=2):
            if len(matches) == 2:
                query = matches[0].queryIdx
                indices[query] = [match.trainIdx for match in matches]
                distances[query] = [match.distance for match in matches]
        return indices, distances

    forward, forward_distance = neighbors(a, b)
    reverse, reverse_distance = neighbors(b, a)
    return reciprocal_match_indices(forward, forward_distance, reverse, reverse_distance,
                                     ratio_threshold=ratio_threshold)


@dataclass(frozen=True)
class TriangulationResult:
    xyz: np.ndarray
    valid: np.ndarray
    triangulation_angle_deg: np.ndarray
    reprojection_error_px: np.ndarray
    positive_depth: np.ndarray
    epipolar_inlier: np.ndarray


def triangulate_fixed_pair(uv_a: np.ndarray, uv_b: np.ndarray, k_a: np.ndarray, pose_a: np.ndarray,
                            k_b: np.ndarray, pose_b: np.ndarray, *,
                            min_angle_deg: float = 2., max_reprojection_error_px: float = 2.,
                            max_epipolar_error_px: float = 2.) -> TriangulationResult:
    """Batched DLT, cheirality, native-pixel reprojection and useful-baseline gates."""
    if (not math.isfinite(min_angle_deg) or min_angle_deg < 2
            or not 0 < max_reprojection_error_px <= 2):
        raise ValueError("fixed triangulation cannot relax the two-degree/two-pixel gates")
    a, b = np.asarray(uv_a, dtype=np.float64), np.asarray(uv_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or b.shape != a.shape:
        raise ValueError("triangulation requires aligned native 2D observations")
    pa, pb = validate_pose(pose_a), validate_pose(pose_b)
    ka, kb = np.asarray(k_a, dtype=np.float64), np.asarray(k_b, dtype=np.float64)
    n = len(a)
    xyz = np.full((n, 3), np.nan)
    errors = np.full((n, 2), np.inf)
    positive = np.zeros((n, 2), dtype=bool)
    angles = np.zeros(n)
    try:
        fundamental = fixed_fundamental_matrix(ka, pa, kb, pb)
    except ValueError as error:
        if "zero-baseline" not in str(error):
            raise
        return TriangulationResult(xyz, np.zeros(n, bool), angles, errors, positive, np.zeros(n, bool))
    epipolar, _ = known_epipolar_inliers(a, b, fundamental, max_error_px=max_epipolar_error_px)
    selected = np.flatnonzero(epipolar)
    if len(selected):
        homogeneous_a = np.column_stack((a[selected], np.ones(len(selected)))) @ np.linalg.inv(ka).T
        homogeneous_b = np.column_stack((b[selected], np.ones(len(selected)))) @ np.linalg.inv(kb).T
        normalized_a = homogeneous_a[:, :2] / homogeneous_a[:, 2, None]
        normalized_b = homogeneous_b[:, :2] / homogeneous_b[:, 2, None]
        equations = np.stack((normalized_a[:, 0, None] * pa[2] - pa[0],
                              normalized_a[:, 1, None] * pa[2] - pa[1],
                              normalized_b[:, 0, None] * pb[2] - pb[0],
                              normalized_b[:, 1, None] * pb[2] - pb[1]), axis=1)
        _, _, vh = np.linalg.svd(equations)
        solution = vh[:, -1]
        finite = np.isfinite(solution).all(axis=1) & (np.abs(solution[:, 3]) > 1e-12)
        xyz[selected[finite]] = solution[finite, :3] / solution[finite, 3, None]
        for column, (pose, k, observed) in enumerate(((pa, ka, a), (pb, kb, b))):
            camera = xyz[selected] @ pose[:, :3].T + pose[:, 3]
            positive[selected, column] = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
            usable = positive[selected, column]
            projected = camera[usable] @ k.T
            errors[selected[usable], column] = np.linalg.norm(
                projected[:, :2] / projected[:, 2, None] - observed[selected[usable]], axis=1)
        center_a, center_b = -pa[:, :3].T @ pa[:, 3], -pb[:, :3].T @ pb[:, 3]
        ray_a, ray_b = xyz[selected] - center_a, xyz[selected] - center_b
        denominator = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        cosine = np.ones(len(selected))
        np.divide(np.einsum("ij,ij->i", ray_a, ray_b), denominator, out=cosine, where=denominator > 1e-12)
        angles[selected] = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    valid = (epipolar & positive.all(axis=1) & np.isfinite(xyz).all(axis=1)
             & (angles >= min_angle_deg) & (errors.max(axis=1) <= max_reprojection_error_px))
    xyz[~valid] = np.nan
    return TriangulationResult(xyz, valid, angles, errors, positive, epipolar)


def _opencv_identity() -> dict[str, str]:
    import cv2
    return {"version": cv2.__version__, "build_sha256": canonical_sha256(cv2.getBuildInformation())}


def cached_native_sift(capture: FrozenCapture, cache_dir: Path | str, *,
                        scene: FrozenScene,
                        parameters: PhotoParameters = PhotoParameters()) -> tuple[np.ndarray, np.ndarray, Path]:
    """Content-addressed real native MAPPING SIFT, no resize/depth/query guidance."""
    parameters.validate()
    validate_mapping_source(scene, capture)
    source = Path(capture.image_path)
    native_jpeg = source.read_bytes()
    if hashlib.sha256(native_jpeg).hexdigest() != capture.image_sha256:
        raise ValueError("native SIFT input JPEG changed")
    import cv2
    extraction = {key: value for key, value in asdict(parameters).items()
                  if key in {"max_features", "octave_layers", "contrast_threshold", "edge_threshold", "sigma"}}
    identity = {"schema": SIFT_SCHEMA, "image_sha256": capture.image_sha256,
                "width": capture.width, "height": capture.height, "pixel_convention": OPENCV_NATIVE_CENTER,
                "parameters": extraction, "opencv": _opencv_identity(),
                "preprocessing": SIFT_PREPROCESSING}
    key = canonical_sha256(identity)
    root = Path(cache_dir).absolute()
    root.mkdir(parents=True, exist_ok=True)
    target = root / key
    if not target.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.partial-", dir=root))
        try:
            gray = cv2.imdecode(np.frombuffer(native_jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if gray is None or gray.shape != (capture.height, capture.width):
                raise ValueError("SIFT input is not the exact native mapping resolution")
            sift = cv2.SIFT_create(nfeatures=parameters.max_features, nOctaveLayers=parameters.octave_layers,
                                   contrastThreshold=parameters.contrast_threshold,
                                   edgeThreshold=parameters.edge_threshold, sigma=parameters.sigma)
            features, descriptors = sift.detectAndCompute(gray, None)
            keypoints = np.asarray([[*kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id]
                                    for kp in features], dtype="<f8").reshape(-1, 7)
            if descriptors is None:
                descriptors = np.empty((0, 128), dtype="<f4")
            else:
                descriptors = np.asarray(descriptors, dtype="<f4")
            np.save(staging / "keypoints.npy", keypoints, allow_pickle=False)
            np.save(staging / "descriptors.npy", descriptors, allow_pickle=False)
            _write_json_atomic(staging / "manifest.json", {"identity": identity, "cache_sha256": key,
                               "members": artifact_members(staging)})
            if sha256_file(source) != capture.image_sha256:
                raise ValueError("native JPEG changed while SIFT was computing")
            fsync_artifact_tree(staging)
            try:
                os.rename(staging, target)
            except FileExistsError:
                shutil.rmtree(staging)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity or manifest.get("cache_sha256") != key:
        raise ValueError("native SIFT cache content/implementation identity mismatch")
    verify_artifact_members(target, manifest["members"], exclude=["manifest.json"])
    keypoints = np.load(target / "keypoints.npy", allow_pickle=False)
    descriptors = np.load(target / "descriptors.npy", allow_pickle=False)
    if (keypoints.dtype.str != "<f8" or keypoints.ndim != 2 or keypoints.shape[1] != 7
            or descriptors.dtype.str != "<f4" or descriptors.shape != (len(keypoints), 128)
            or not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all()
            or np.any(keypoints[:, 0] < 0) or np.any(keypoints[:, 0] >= capture.width)
            or np.any(keypoints[:, 1] < 0) or np.any(keypoints[:, 1] >= capture.height)):
        raise ValueError("malformed native SIFT keypoint/descriptor cache")
    return keypoints, descriptors, manifest_path


def build_photographic_evidence(model_dir: Path | str, keyframes_manifest: Path | str,
                                 image_root: Path | str, pairs: Sequence[tuple[int, int]],
                                 output_dir: Path | str, *, map_revision_id: str,
                                 coordinate_frame_id: str, input_pixel_convention: str,
                                 cache_dir: Path | str | None = None,
                                 parameters: PhotoParameters = PhotoParameters()) -> Path:
    """Build independent pair geometry before any candidate/depth association."""
    import cv2
    parameters.validate()
    normalized_pairs = sorted({tuple(sorted((int(a), int(b)))) for a, b in pairs})
    if not normalized_pairs or any(a == b for a, b in normalized_pairs):
        raise ValueError("photographic pair manifest requires real distinct-camera pairs")
    if not map_revision_id.strip() or not coordinate_frame_id.strip():
        raise ValueError("photographic geometry requires explicit coordinate/map identity")
    image_ids = sorted({i for pair in normalized_pairs for i in pair})
    scene = load_frozen_scene(model_dir, keyframes_manifest, image_root,
                              input_pixel_convention=input_pixel_convention, image_ids=image_ids)
    target = Path(output_dir).absolute()
    if target.exists():
        raise ValueError("photographic evidence output must be a new immutable directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    sift_cache = Path(cache_dir).absolute() if cache_dir is not None else staging / "sift_cache"
    pair_manifest = {"schema": PAIR_SCHEMA, "evidence_family": PHOTO_FAMILY,
                     "model_hashes": dict(scene.model_hashes), "keyframes_sha256": scene.keyframes_sha256,
                     "map_revision_id": map_revision_id, "coordinate_frame_id": coordinate_frame_id,
                     "input_pixel_convention": input_pixel_convention, "native_pixel_convention": OPENCV_NATIVE_CENTER,
                     "images": [asdict(scene.captures[i]) for i in image_ids],
                     "pairs": [list(pair) for pair in normalized_pairs], "parameters": asdict(parameters),
                     "opencv": _opencv_identity(), "implementation_sha256": sha256_file(Path(__file__)),
                     "match_selection": "NATIVE_SIFT_MUTUAL_BIDIRECTIONAL_RATIO_FIXED_F_EPIPOLAR_DLT"}
    pair_manifest["pair_manifest_sha256"] = canonical_sha256(pair_manifest)
    buffers: dict[str, list[np.ndarray]] = {name: [] for name in ARRAY_DTYPES}
    cache_receipts: dict[str, dict[str, str]] = {}
    diagnostics = []
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(min(8, max(1, previous_threads)))
    try:
        _write_json_atomic(staging / "pair_manifest.json", pair_manifest)
        shutil.copyfile(scene.keyframes_manifest, staging / "keyframes.jsonl")
        frozen = staging / "frozen_model"
        frozen.mkdir()
        for name, expected in scene.model_hashes.items():
            shutil.copyfile(scene.model_dir / name, frozen / name)
            if sha256_file(frozen / name) != expected:
                raise ValueError("frozen photographic model changed while snapshotting")
        for capture in scene.captures.values():
            destination = staging / "native_images" / safe_relative_name(capture.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(capture.image_path, destination)
            if sha256_file(destination) != capture.image_sha256:
                raise ValueError("native photographic JPEG changed while snapshotting")
        point_count = 0
        for image_a, image_b in normalized_pairs:
            a, b = scene.captures[image_a], scene.captures[image_b]
            keypoints_a, descriptors_a, cache_a = cached_native_sift(a, sift_cache, scene=scene, parameters=parameters)
            keypoints_b, descriptors_b, cache_b = cached_native_sift(b, sift_cache, scene=scene, parameters=parameters)
            for image_id, path in ((image_a, cache_a), (image_b, cache_b)):
                snapshot = staging / "sift_cache" / path.parent.name
                if snapshot != path.parent and not snapshot.exists():
                    snapshot.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(path.parent, snapshot)
                cache_receipts[str(image_id)] = {"cache_sha256": path.parent.name,
                                                "manifest_sha256": sha256_file(path),
                                                "path": snapshot.relative_to(staging).as_posix()}
            matches, ratios = reciprocal_sift_matches(descriptors_a, descriptors_b,
                                                      ratio_threshold=parameters.ratio_threshold)
            uv_a, uv_b = np.asarray(keypoints_a[matches[:, 0], :2]), np.asarray(keypoints_b[matches[:, 1], :2])
            result = triangulate_fixed_pair(uv_a, uv_b, np.asarray(a.K_native), np.asarray(a.cam_from_world),
                                            np.asarray(b.K_native), np.asarray(b.cam_from_world),
                                            min_angle_deg=parameters.min_triangulation_angle_deg,
                                            max_reprojection_error_px=parameters.max_reprojection_error_px,
                                            max_epipolar_error_px=parameters.max_epipolar_error_px)
            valid = result.valid
            count = int(valid.sum())
            rows = {"source_image_id": np.full(count, image_a), "target_image_id": np.full(count, image_b),
                    "uv_source": uv_a[valid], "uv_target": uv_b[valid], "xyz": result.xyz[valid],
                    "triangulation_angle_deg": result.triangulation_angle_deg[valid],
                    "reprojection_error_px": result.reprojection_error_px[valid], "descriptor_ratio": ratios[valid],
                    "source_feature_index": matches[valid, 0], "target_feature_index": matches[valid, 1],
                    "match_id": np.arange(point_count, point_count + count)}
            for name, values in rows.items():
                buffers[name].append(np.asarray(values, dtype=ARRAY_DTYPES[name]))
            point_count += count
            diagnostics.append({"image_a": image_a, "image_b": image_b,
                                "reciprocal_ratio_matches": len(matches), "epipolar_inliers": int(result.epipolar_inlier.sum()),
                                "positive_depth_both": int(result.positive_depth.all(axis=1).sum()),
                                "accepted_fixed_pose_points": count})
        arrays_metadata = {}
        for name, chunks in buffers.items():
            values = np.concatenate(chunks, axis=0)
            np.save(staging / (name + ".npy"), values, allow_pickle=False)
            arrays_metadata[name] = {"path": name + ".npy", "dtype": ARRAY_DTYPES[name], "shape": list(values.shape)}
        _assert_scene_unchanged(scene)
        manifest = {"schema": PHOTO_SCHEMA, "status": "COMPLETE", "evidence_family": PHOTO_FAMILY,
                    "geometry_origin": "PHOTO_TRIANGULATED", "authority": "SUPPLEMENTAL_DISPLAY_ONLY",
                    "external_absolute_truth": "UNAVAILABLE", "runtime_depth_inference": False,
                    "source_scope": MAPPING_SCOPE, "holdout_scope": MAPPING_HOLDOUT_SCOPE,
                    "holdout_status": "PENDING_MAPPING_INTERNAL_BLOCKED_HOLDOUT", "deployment_authorized": False,
                    "model_hashes": dict(scene.model_hashes), "keyframes_sha256": scene.keyframes_sha256,
                    "map_revision_id": map_revision_id, "coordinate_frame_id": coordinate_frame_id,
                    "pair_manifest_sha256": pair_manifest["pair_manifest_sha256"], "point_count": point_count,
                    "arrays": arrays_metadata, "sift_cache_receipts": cache_receipts,
                    "pair_diagnostics": diagnostics, "members": artifact_members(staging)}
        _write_json_atomic(staging / "manifest.json", manifest)
        read_photographic_evidence(staging / "manifest.json", expected_model_hashes=scene.model_hashes,
                                    expected_keyframes_sha256=scene.keyframes_sha256)
        fsync_artifact_tree(staging)
        if target.exists():
            raise ValueError("photographic evidence output was concurrently published")
        os.rename(staging, target)
        descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        cv2.setNumThreads(previous_threads)
    return target / "manifest.json"


class PhotoEvidenceArtifact:
    evidence_family = "reciprocal_photo"

    def __init__(self, root: Path, manifest: dict[str, Any], pairs: dict[str, Any],
                 arrays: Mapping[str, np.ndarray], manifest_sha256: str) -> None:
        self.root, self.manifest, self.pair_manifest = root, manifest, pairs
        self.arrays, self.manifest_sha256 = arrays, manifest_sha256
        self.captures = {int(row["image_id"]): FrozenCapture(**row) for row in pairs["images"]}
        # The pair points are independent of candidates. Cache one source's
        # geometric index, replacing rather than accumulating trees per image.
        self._source_id: int | None = None
        self._indices: dict[int, tuple[np.ndarray, np.ndarray, cKDTree, bool]] = {}

    def _source_indices(self, image_id: int) -> Mapping[int, tuple[np.ndarray, np.ndarray, cKDTree, bool]]:
        if self._source_id == image_id:
            return self._indices
        index = {}
        for reversed_pair in (False, True):
            source_key, target_key = ("target_image_id", "source_image_id") if reversed_pair else ("source_image_id", "target_image_id")
            uv_key = "uv_target" if reversed_pair else "uv_source"
            source_rows = np.flatnonzero(self.arrays[source_key] == image_id)
            for target in sorted(set(map(int, self.arrays[target_key][source_rows]))):
                rows = source_rows[self.arrays[target_key][source_rows] == target]
                # Stable match IDs resolve equal pixel-distance ties; candidate
                # depth/residual never chooses the photographic association.
                rows = rows[np.argsort(self.arrays["match_id"][rows], kind="stable")]
                uv = np.asarray(self.arrays[uv_key][rows], dtype=np.float64)
                index[target] = rows, uv, cKDTree(uv), reversed_pair
        self._source_id, self._indices = image_id, index
        return index

    def compare_candidates(self, source_image_id: int, uv: np.ndarray, xyz: np.ndarray, *,
                            pixel_tolerance: float = 2., max_log_residual: float = .08) -> dict[str, np.ndarray]:
        """Associate only by native source pixels, then compare fixed photo geometry.

        Each row identifies the other actual photograph of a triangulated pair.
        Agreement also requires candidate reprojection near that real target
        observation. Signed conflicts/occlusions retain the associated measured
        source-ray surface; they are not chosen by minimizing a depth residual.
        residual_image_ids names the source optical-Z camera, not the supporting
        photograph; UNKNOWN entries carry no residual/evidence camera identity.
        """
        if not 0 < pixel_tolerance <= 2:
            raise ValueError("photo association cannot exceed two native pixels")
        pixels = np.asarray(uv, dtype=np.float64)
        points = np.asarray(xyz, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or points.shape != (len(pixels), 3):
            raise ValueError("candidate source pixels/XYZ must align")
        source = self.captures.get(int(source_image_id))
        if source is None:
            return _empty_evidence(len(points))
        groups = self._source_indices(int(source_image_id))
        neighbors = sorted(groups)
        residuals = np.full((len(neighbors), len(points)), np.nan)
        states = np.full(residuals.shape, UNKNOWN, dtype=np.uint8)
        evidence_ids = np.full(residuals.shape, -1, dtype=np.int64)
        source_pose = np.asarray(source.cam_from_world)
        candidate_source = points @ source_pose[:, :3].T + source_pose[:, 3]
        finite = np.isfinite(pixels).all(axis=1) & np.isfinite(points).all(axis=1) & (candidate_source[:, 2] > 0)
        source_uv_projected = np.full(pixels.shape, np.nan)
        if finite.any():
            projected = candidate_source[finite] @ np.asarray(source.K_native).T
            source_uv_projected[finite] = projected[:, :2] / projected[:, 2, None]
        finite &= np.linalg.norm(source_uv_projected - pixels, axis=1) <= pixel_tolerance
        for view_index, target_image_id in enumerate(neighbors):
            rows, photo_uv, tree, reversed_pair = groups[target_image_id]
            candidates = np.flatnonzero(finite)
            if not len(candidates):
                continue
            # All within-radius matches are compared only in pixel space; select
            # shortest distance, then smallest immutable match ID.
            selected_points, selected_rows = [], []
            for candidate_index, local_rows in zip(candidates, tree.query_ball_point(pixels[candidates], pixel_tolerance)):
                if not local_rows:
                    continue
                local = np.asarray(local_rows, dtype=np.int64)
                distances = np.sum((photo_uv[local] - pixels[candidate_index]) ** 2, axis=1)
                order = np.lexsort((self.arrays["match_id"][rows[local]], distances))
                selected_points.append(int(candidate_index))
                selected_rows.append(int(rows[local[order[0]]]))
            if not selected_points:
                continue
            ids, photo_rows = np.asarray(selected_points), np.asarray(selected_rows)
            measured = self.arrays["xyz"][photo_rows]
            target = self.captures[target_image_id]
            target_pose = np.asarray(target.cam_from_world)
            candidate_target = points[ids] @ target_pose[:, :3].T + target_pose[:, 3]
            measured_source = measured @ source_pose[:, :3].T + source_pose[:, 3]
            source_ray, target_ray = points[ids] - source.center, points[ids] - target.center
            denominator = np.linalg.norm(source_ray, axis=1) * np.linalg.norm(target_ray, axis=1)
            cosine = np.ones(len(ids))
            np.divide(np.einsum("ij,ij->i", source_ray, target_ray), denominator,
                      out=cosine, where=denominator > 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
            usable = (np.isfinite(candidate_target).all(axis=1) & (candidate_target[:, 2] > 0)
                      & (angle >= 2.))
            verdict = signed_depth_verdict(candidate_source[ids, 2], measured_source[:, 2],
                                           usable=usable, max_log_residual=max_log_residual)
            target_uv = self.arrays["uv_source" if reversed_pair else "uv_target"][photo_rows]
            projected = candidate_target[usable] @ np.asarray(target.K_native).T
            reprojection = np.full(len(ids), np.inf)
            reprojection[usable] = np.linalg.norm(projected[:, :2] / projected[:, 2, None] - target_uv[usable], axis=1)
            false_agreement = (verdict["states"] == AGREE) & (reprojection > pixel_tolerance)
            verdict["states"][false_agreement] = UNKNOWN
            verdict["signed_log_residual"][false_agreement] = np.nan
            residuals[view_index, ids] = verdict["signed_log_residual"]
            states[view_index, ids] = verdict["states"]
            known = verdict["states"] != UNKNOWN
            evidence_ids[view_index, ids[known]] = self.arrays["match_id"][photo_rows[known]]
        return {"signed_log_residual": residuals, "states": states,
                "support_image_ids": np.asarray(neighbors, dtype=np.int64), "evidence_ids": evidence_ids,
                "residual_image_ids": np.where(states != UNKNOWN, np.int64(source_image_id), -1)}


def _bound_sift_snapshot(root: Path, capture: FrozenCapture, binding: Mapping[str, str],
                          pairs: Mapping[str, Any], members: Mapping[str, Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    snapshot = root / safe_relative_name(binding["path"])
    file = snapshot / "manifest.json"
    if sha256_file(file) != binding["manifest_sha256"]:
        raise ValueError("photographic SIFT snapshot receipt changed")
    cache = json.loads(file.read_text(encoding="utf-8"))
    identity = cache["identity"]
    extraction = {key: pairs["parameters"][key] for key in
                  ("max_features", "octave_layers", "contrast_threshold", "edge_threshold", "sigma")}
    if (cache.get("cache_sha256") != binding["cache_sha256"]
            or canonical_sha256(identity) != binding["cache_sha256"]
            or identity.get("schema") != SIFT_SCHEMA
            or identity.get("image_sha256") != capture.image_sha256
            or (identity.get("width"), identity.get("height")) != (capture.width, capture.height)
            or identity.get("pixel_convention") != OPENCV_NATIVE_CENTER
            or identity.get("preprocessing") != SIFT_PREPROCESSING
            or identity.get("parameters") != extraction or identity.get("opencv") != pairs["opencv"]
            or set(cache["members"]) != {"keypoints.npy", "descriptors.npy"}):
        raise ValueError("photographic SIFT snapshot is not bound to the native source/parameters")
    for name, info in cache["members"].items():
        relative = (snapshot / safe_relative_name(name)).relative_to(root).as_posix()
        if members.get(relative) != info:
            raise ValueError("photographic SIFT snapshot members differ from the verified artifact")
    keypoints = np.load(snapshot / "keypoints.npy", allow_pickle=False)
    descriptors = np.load(snapshot / "descriptors.npy", allow_pickle=False)
    if (keypoints.dtype.str != "<f8" or keypoints.ndim != 2 or keypoints.shape[1] != 7
            or descriptors.dtype.str != "<f4" or descriptors.shape != (len(keypoints), 128)
            or not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all()
            or np.any(keypoints[:, 0] < 0) or np.any(keypoints[:, 0] >= capture.width)
            or np.any(keypoints[:, 1] < 0) or np.any(keypoints[:, 1] >= capture.height)):
        raise ValueError("partial/invalid bound photographic SIFT data")
    return keypoints, descriptors


def read_photographic_evidence(manifest_path: Path | str, *, expected_model_hashes: Mapping[str, str],
                                expected_keyframes_sha256: str,
                                expected_manifest_sha256: str | None = None) -> PhotoEvidenceArtifact:
    file = Path(manifest_path).resolve(strict=True)
    root, digest = file.parent, sha256_file(file)
    manifest = json.loads(file.read_text(encoding="utf-8"))
    if (manifest.get("schema") != PHOTO_SCHEMA or manifest.get("status") != "COMPLETE"
            or manifest.get("evidence_family") != PHOTO_FAMILY
            or manifest.get("model_hashes") != dict(expected_model_hashes)
            or manifest.get("runtime_depth_inference") is not False
            or manifest.get("geometry_origin") != "PHOTO_TRIANGULATED"
            or manifest.get("keyframes_sha256") != expected_keyframes_sha256
            or manifest.get("authority") != "SUPPLEMENTAL_DISPLAY_ONLY"
            or manifest.get("source_scope") != MAPPING_SCOPE
            or manifest.get("holdout_scope") != MAPPING_HOLDOUT_SCOPE
            or manifest.get("deployment_authorized") is not False
            or (expected_manifest_sha256 is not None and digest != expected_manifest_sha256)):
        raise ValueError("unsupported/hash-mismatched/partial photographic evidence")
    verify_artifact_members(root, manifest["members"], exclude=[file.name])
    pairs = json.loads((root / "pair_manifest.json").read_text(encoding="utf-8"))
    if (pairs.get("schema") != PAIR_SCHEMA
            or canonical_sha256({k: v for k, v in pairs.items() if k != "pair_manifest_sha256"}) != pairs.get("pair_manifest_sha256")
            or pairs.get("evidence_family") != PHOTO_FAMILY
            or pairs.get("implementation_sha256") != sha256_file(Path(__file__))
            or pairs.get("match_selection") != "NATIVE_SIFT_MUTUAL_BIDIRECTIONAL_RATIO_FIXED_F_EPIPOLAR_DLT"
            or pairs["pair_manifest_sha256"] != manifest["pair_manifest_sha256"]
            or pairs["model_hashes"] != manifest["model_hashes"]
            or pairs["keyframes_sha256"] != manifest["keyframes_sha256"]
            or pairs["map_revision_id"] != manifest["map_revision_id"]
            or pairs["coordinate_frame_id"] != manifest["coordinate_frame_id"]
            or pairs["native_pixel_convention"] != OPENCV_NATIVE_CENTER
            or not str(pairs["map_revision_id"]).strip()
            or not str(pairs["coordinate_frame_id"]).strip()
            or sha256_file(root / "keyframes.jsonl") != expected_keyframes_sha256):
        raise ValueError("photographic pair/source/pose identity differs from its receipt")
    parameters = PhotoParameters(**pairs["parameters"])
    parameters.validate()
    if pairs["opencv"] != _opencv_identity():
        raise ValueError("photographic revalidation requires the pinned OpenCV version/build")
    captures = {int(row["image_id"]): FrozenCapture(**row) for row in pairs["images"]}
    declared_pairs = {tuple(map(int, pair)) for pair in pairs["pairs"]}
    if (len(captures) != len(pairs["images"]) or len(declared_pairs) != len(pairs["pairs"])
            or not declared_pairs or any(a >= b or a not in captures or b not in captures for a, b in declared_pairs)):
        raise ValueError("invalid immutable photographic capture/pair set")
    if set(manifest["sift_cache_receipts"]) != {str(i) for i in captures}:
        raise ValueError("photographic evidence is missing native SIFT source receipts")
    if frozen_model_hashes(root / "frozen_model") != pairs["model_hashes"]:
        raise ValueError("photographic frozen model snapshot differs from its pair manifest")
    verify_frozen_capture_snapshot(root, captures, input_pixel_convention=pairs["input_pixel_convention"])
    if set(manifest["arrays"]) != set(ARRAY_DTYPES):
        raise ValueError("incomplete photographic geometry arrays")
    arrays = {}
    n = int(manifest["point_count"])
    if n < 0:
        raise ValueError("invalid photographic point count")
    for name, dtype in ARRAY_DTYPES.items():
        info = manifest["arrays"][name]
        path = root / safe_relative_name(info["path"])
        array = np.load(path, allow_pickle=False)
        width = 3 if name == "xyz" else 2 if name in {"uv_source", "uv_target", "descriptor_ratio", "reprojection_error_px"} else None
        shape = (n, width) if width else (n,)
        if (array.dtype.str != dtype or info["dtype"] != dtype or array.shape != shape
                or info["shape"] != list(shape) or not np.isfinite(array).all()):
            raise ValueError(f"malformed/truncated photographic array: {name}")
        arrays[name] = array
    if (not np.array_equal(arrays["match_id"], np.arange(n, dtype=np.int64))
            or np.any(arrays["descriptor_ratio"] < 0)
            or np.any(arrays["descriptor_ratio"] >= parameters.ratio_threshold)
            or np.any(arrays["triangulation_angle_deg"] < parameters.min_triangulation_angle_deg)
            or np.any(arrays["reprojection_error_px"] < 0)
            or np.any(arrays["reprojection_error_px"] > parameters.max_reprojection_error_px)):
        raise ValueError("photographic points do not satisfy frozen geometry/matching gates")
    actual_pairs = set(zip(map(int, arrays["source_image_id"]), map(int, arrays["target_image_id"])))
    if not actual_pairs <= declared_pairs:
        raise ValueError("photographic points originate from undeclared pairs")
    for a_id, b_id in sorted(declared_pairs):
        selected = np.flatnonzero((arrays["source_image_id"] == a_id) & (arrays["target_image_id"] == b_id))
        a, b = captures[a_id], captures[b_id]
        keypoints_a, descriptors_a = _bound_sift_snapshot(
            root, a, manifest["sift_cache_receipts"][str(a_id)], pairs, manifest["members"])
        keypoints_b, descriptors_b = _bound_sift_snapshot(
            root, b, manifest["sift_cache_receipts"][str(b_id)], pairs, manifest["members"])
        matches, ratios = reciprocal_sift_matches(descriptors_a, descriptors_b,
                                                  ratio_threshold=parameters.ratio_threshold)
        uv_a, uv_b = keypoints_a[matches[:, 0], :2], keypoints_b[matches[:, 1], :2]
        result = triangulate_fixed_pair(uv_a, uv_b, np.asarray(a.K_native), np.asarray(a.cam_from_world),
                                        np.asarray(b.K_native), np.asarray(b.cam_from_world),
                                        min_angle_deg=parameters.min_triangulation_angle_deg,
                                        max_reprojection_error_px=parameters.max_reprojection_error_px,
                                        max_epipolar_error_px=parameters.max_epipolar_error_px)
        accepted = result.valid
        if (len(selected) != int(accepted.sum())
                or not np.array_equal(matches[accepted, 0], arrays["source_feature_index"][selected])
                or not np.array_equal(matches[accepted, 1], arrays["target_feature_index"][selected])
                or not np.array_equal(uv_a[accepted], arrays["uv_source"][selected])
                or not np.array_equal(uv_b[accepted], arrays["uv_target"][selected])
                or not np.allclose(ratios[accepted], arrays["descriptor_ratio"][selected], atol=1e-8, rtol=1e-8)
                or not np.allclose(result.xyz[accepted], arrays["xyz"][selected], atol=1e-10, rtol=1e-10)
                or not np.allclose(result.reprojection_error_px[accepted], arrays["reprojection_error_px"][selected], atol=1e-8, rtol=1e-8)
                or not np.allclose(result.triangulation_angle_deg[accepted], arrays["triangulation_angle_deg"][selected], atol=1e-8, rtol=1e-8)):
            raise ValueError("photographic evidence is partial or does not reproduce reciprocal fixed-pose geometry")
    return PhotoEvidenceArtifact(root, manifest, pairs, arrays, digest)
