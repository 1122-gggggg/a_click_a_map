"""Offline multiview depth verification and selective surface fusion in GlueMap units.

This module consumes MoGe metric depth predictions (depth_m, mask, K_work, source_to_work),
per-image scale calibration factors, and frozen COLMAP/GlueMap reconstruction geometry.
Depths are corrected offline using the frozen calibration artifact, divided by
per-image scale, and checked across separated but model-correlated mapping views.

Visibility and Evidence Principles:
- Occluded points (Z_proj > Z_meas * (1 + tol)): camera line-of-sight is blocked by a
  foreground surface -> NO EVIDENCE.
- Outside view (behind camera or outside sensor boundaries) -> NO EVIDENCE.
- Invalid depth (NaN, zero, unmasked) -> NO EVIDENCE.
- Low parallax (baseline angle between rays < min_baseline_deg) -> NO EVIDENCE.
- Multiview agreement (|log Z_proj - log Z_meas| <= tol) is correlated corroboration.
- Free-space violation (negative signed log residual) is retained as conflict;
  a positive residual denotes occlusion, not a negative vote against a hidden surface.
- Sparse tracks, real reciprocal photo triangulation and hash-verified fixed-pose
  MVS have separate ledger modalities. None supplies external metric truth.
- Capture/session/temporal/camera-cell/ray groups cap repeated evidence before
  frozen support gates; two modalities from one capture are not two views.
- Only VERIFIED candidates are retained; unsupported samples remain unknown.
  Unverified/unsupported points remain unknown and are never fabricated.
- Fail closed: missing/corrupted depth caches fail immediately with ValueError. If zero
  points survive verification, the pipeline fails closed instead of publishing empty maps.

Persisted Artifacts:
- dense_map.npz: rich canonical offline geometry/evidence/observation archive.
  It retains decomposed accuracy evidence alongside precomputed CPU geometry.
- dense_map.ply: binary little-endian PLY with confidence-coded vertex RGB.
- dense_summary.json: diagnostics and verification rates.

Rendering:
- `render_depth` is the offline CPU reference renderer for arbitrary proper
  camera poses. Metre output refuses rejected calibration; no neural inference
  is performed here; online consumers never access raw per-image depth caches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from sfm_diagnosis.io import write_json
from .geometry_contract import (
    LEDGER_DTYPES, LEDGER_SCHEMA, SCORE_SEMANTICS, SAME_MODEL, RECIPROCAL_PHOTO,
    SPARSE_ANCHOR, FIXED_POSE_MVS, OBS_SOURCE, OBS_DEPTH_AGREE, OBS_PHOTO_AGREE,
    OBS_MVS_AGREE, OBS_SPARSE_AGREE, OBS_CONFLICT, OBS_OCCLUDED,
    apply_depth_calibration, calibration_contract, calibration_quality,
    depth_patch_quality, validate_calibration_contract, validate_depth_calibration,
    validate_evidence_ledger, validate_mapping_record, validate_pose, validate_pinhole,
)

REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
VALIDATION_VERSION = "correlation-visibility-ledger/v4"


@dataclass(frozen=True)
class ViewEvidence:
    same_model_support: np.ndarray
    photo_support: np.ndarray
    mvs_support: np.ndarray
    geometric_support: np.ndarray
    conflict: np.ndarray
    occluded: np.ndarray
    mean_abs_residual: np.ndarray
    p90_abs_residual: np.ndarray
    min_angle: np.ndarray
    median_angle: np.ndarray
    max_angle: np.ndarray
    session_count: np.ndarray
    source_group_count: np.ndarray
    observation_flags: np.ndarray
    assignments: np.ndarray
    raw_same_model_count: np.ndarray
    raw_conflict_count: np.ndarray
    unknown_count: np.ndarray


def aggregate_view_evidence(
    log_ratios: np.ndarray, rays: np.ndarray, photographic: np.ndarray,
    *, tolerance: float, separation_deg: float,
    view_keys: Sequence[Mapping[str, Any]] | None = None,
    angles: np.ndarray | None = None, detailed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | ViewEvidence:
    """Deduplicate captures before gates, retaining signed contradictions.

    Family labels never manufacture independent views. A capture group carries
    one geometric vote and separately labeled modalities. Duplicate rows do
    not erase a conflict/occlusion, even when another member agrees.
    """
    residuals = np.asarray(log_ratios, dtype=np.float64)
    directions = np.asarray(rays, dtype=np.float64)
    photos = np.asarray(photographic, dtype=bool)
    if (residuals.ndim != 2 or directions.shape != (*residuals.shape, 3) or
            photos.shape != residuals.shape or not 0 < tolerance <= .08 or
            not 2. <= separation_deg <= 180.):
        raise ValueError("malformed or relaxed view-evidence inputs")
    ray_length = np.linalg.norm(directions, axis=2)
    usable = (np.isfinite(residuals) & np.isfinite(directions).all(axis=2) &
              np.isfinite(ray_length) & (ray_length > 0))
    angle_array = np.asarray(angles) if angles is not None else np.zeros(residuals.shape)
    if angle_array.shape != residuals.shape:
        raise ValueError("usable angle ledger is misaligned")
    if angles is not None:
        usable &= np.isfinite(angle_array) & (angle_array >= separation_deg) & (angle_array <= 180.)
    directions = np.divide(directions, ray_length[..., None], out=np.zeros_like(directions),
                           where=(ray_length > 0)[..., None] & np.isfinite(ray_length[..., None]))
    n_views, n_points = residuals.shape
    keys = list(view_keys) if view_keys is not None else [
        {"family": "same_model", "session": str(i), "temporal_segment": i,
         "camera_cell": (i,), "view_id": i} for i in range(n_views)]
    if len(keys) != n_views:
        raise ValueError("view correlation metadata is misaligned")
    families = [str(key["family"]) for key in keys]
    if any(family not in ("same_model", "reciprocal_photo", "fixed_pose_mvs") for family in families):
        raise ValueError("unknown view evidence family")
    captures = [(str(k["session"]), int(k["temporal_segment"]), tuple(k["camera_cell"])) for k in keys]
    # A stable metadata/ray sort is independent of input neighbor ordering.
    order = sorted(range(n_views), key=lambda i: (captures[i], int(keys[i]["view_id"]), families[i],
                    tuple(np.nan_to_num(directions[i, 0])) if n_points else ()))
    assignments = np.full((n_views, n_points), -1, dtype=np.int16)
    cosine_limit = math.cos(math.radians(separation_deg))
    priority = np.empty(n_views, dtype=int)
    priority[order] = np.arange(n_views)
    for ordinal, i in enumerate(order):
        valid = usable[i]
        assignments[i, valid] = i
        for j in order[:ordinal]:
            cosine = np.einsum("ij,ij->i", directions[i], directions[j])
            same_capture = captures[i][0] == captures[j][0] and (
                captures[i][1] == captures[j][1] or captures[i][2] == captures[j][2])
            merge = valid & (assignments[j] >= 0) & ((cosine >= cosine_limit) | same_capture)
            columns = np.flatnonzero(merge)
            if not len(columns):
                continue
            a, b = assignments[i, columns], assignments[j, columns]
            winner = np.where(priority[a] <= priority[b], a, b)
            loser = np.where(priority[a] <= priority[b], b, a)
            current = assignments[:, columns]
            assignments[:, columns] = np.where(current == loser[None, :], winner[None, :], current)
    same, photo, mvs, support, conflict, occluded = [np.zeros(n_points, np.int32) for _ in range(6)]
    flags = np.zeros((n_views, n_points), np.uint8)
    grouped_residuals = np.full((n_views, n_points), np.nan)
    grouped_angles = np.full((n_views, n_points), np.nan)
    agreement_sessions = np.zeros(n_points, np.uint16)
    session_presence = {str(k["session"]): np.zeros(n_points, bool) for k in keys}
    for representative in order:
        members = assignments == representative
        present = members.any(axis=0)
        if not present.any():
            continue
        agreeing = members & (np.abs(residuals) <= tolerance)
        contradicting = members & (residuals < -tolerance)
        hidden = members & (residuals > tolerance)
        active = np.flatnonzero(present)
        # Deduplicate exact residual values without a per-point Python loop.
        values = np.sort(np.where(members[:, active], residuals[:, active], np.nan), axis=0)
        unique = np.isfinite(values)
        unique[1:] &= values[1:] != values[:-1]
        ranks = np.cumsum(unique, axis=0)
        counts = unique.sum(axis=0)
        lower, upper = (counts - 1) // 2 + 1, counts // 2 + 1
        lo = np.sum(np.where(unique & (ranks == lower), values, 0.), axis=0)
        hi = np.sum(np.where(unique & (ranks == upper), values, 0.), axis=0)
        grouped_residuals[representative, active] = (lo + hi) * .5
        actual = agreeing.any(axis=0)
        conflict += contradicting.any(axis=0)
        occluded += hidden.any(axis=0)
        # A conflict and a positive observation cannot cancel into a fabricated
        # surface. Actual agreement can coexist with a retained contradiction.
        supporting = actual & (np.abs(grouped_residuals[representative]) <= tolerance)
        support += supporting
        for family, counter, bit in (("same_model", same, OBS_DEPTH_AGREE),
                                     ("reciprocal_photo", photo, OBS_PHOTO_AGREE),
                                     ("fixed_pose_mvs", mvs, OBS_MVS_AGREE)):
            family_rows = np.asarray([f == family for f in families])[:, None]
            family_agree = agreeing & family_rows
            counter += family_agree.any(axis=0) & supporting
            flags[family_agree] |= bit
        # The historical photographic input is sparse-track corroboration.
        sparse_rows = agreeing & photos & supporting[None, :]
        if view_keys is None and not detailed:
            photo += sparse_rows.any(axis=0)
        for verdict, bit in ((contradicting, OBS_CONFLICT), (hidden, OBS_OCCLUDED)):
            flags[verdict] |= bit
        valid_angles = np.where(agreeing & supporting[None, :], angle_array, -np.inf)
        grouped_angles[representative, supporting] = np.max(valid_angles[:, supporting], axis=0)
        choices = np.where(agreeing & supporting[None, :], np.abs(residuals), np.inf)[order]
        chosen_member = np.asarray(order)[np.argmin(choices, axis=0)]
        for i in order:
            session_presence[str(keys[i]["session"])] |= supporting & (chosen_member == i)
    for present in session_presence.values():
        agreement_sessions += present
    mean, p90, minimum, median, maximum = [np.zeros(n_points, np.float32) for _ in range(5)]
    columns = np.flatnonzero(np.isfinite(grouped_residuals).any(axis=0))
    if len(columns):
        values = np.abs(grouped_residuals[:, columns])
        mean[columns] = np.nansum(values, axis=0) / np.isfinite(values).sum(axis=0)
        p90[columns] = np.nanquantile(values, .9, axis=0)
    columns = np.flatnonzero(np.isfinite(grouped_angles).any(axis=0))
    if len(columns):
        values = grouped_angles[:, columns]
        minimum[columns], median[columns], maximum[columns] = (
            np.nanmin(values, axis=0), np.nanmedian(values, axis=0), np.nanmax(values, axis=0))
    result = ViewEvidence(same, photo, mvs, support, conflict, occluded, mean, p90,
                          minimum, median, maximum, agreement_sessions, support.astype(np.uint16),
                          flags, assignments,
                          np.sum(usable & (np.abs(residuals) <= tolerance) &
                                 np.asarray([f == "same_model" for f in families])[:, None], axis=0),
                          np.sum(usable & (residuals < -tolerance), axis=0),
                          np.sum(~usable, axis=0))
    if detailed:
        return result
    # Legacy offline diagnostics preserve the four-value API, not an online
    # geometry authority. Sparse corroboration is not photo triangulation.
    summed = np.sum(np.where((np.abs(grouped_residuals) <= tolerance) &
                    np.isfinite(grouped_residuals), np.abs(grouped_residuals), 0.), axis=0)
    return same, conflict, photo, summed


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA256 of a file."""
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def model_file_hashes(model_dir: Path | str) -> dict[str, str]:
    """Compute SHA256 of required COLMAP model binaries."""
    md = Path(model_dir).resolve(strict=True)
    hashes: dict[str, str] = {}
    for filename in REQUIRED_MODEL_FILES:
        fp = md / filename
        if not fp.is_file():
            raise ValueError(f"missing map model file: {fp}")
        hashes[filename] = sha256_file(fp)
    for opt in ("frames.bin", "rigs.bin"):
        fp = md / opt
        if fp.is_file():
            hashes[opt] = sha256_file(fp)
    return hashes


def safe_image_name(name: str) -> Path:
    """Validate relative image name and guard against path traversal."""
    if not name:
        raise ValueError("image name cannot be empty")
    rel = Path(name)
    if rel.is_absolute() or any(part in (".", "..") for part in rel.parts) or str(rel) != name:
        raise ValueError(f"unsafe map image name: {name!r}")
    return rel


def compute_work_intrinsics(source: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute centered work intrinsics and homography from source intrinsics."""
    k = np.asarray(source, dtype=np.float64)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or width <= 1
        or height <= 1
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not (0.0 < k[0, 2] < width - 1)
        or not (0.0 < k[1, 2] < height - 1)
        or not np.allclose(k, [[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]])
    ):
        raise ValueError("expected positive undistorted PINHOLE intrinsics with interior principal point")
    cxw = (width - 1) / 2.0
    cyw = (height - 1) / 2.0
    focal = max(
        k[0, 0] * cxw / min(k[0, 2], width - 1 - k[0, 2]),
        k[1, 1] * cyw / min(k[1, 2], height - 1 - k[1, 2]),
    )
    target = np.array([[focal, 0.0, cxw], [0.0, focal, cyw], [0.0, 0.0, 1.0]], dtype=np.float64)
    homography = target @ np.linalg.inv(k)
    return target, homography


def sample_depth_bilinear(
    depth: np.ndarray,
    uv: np.ndarray,
    native_wh: tuple[int, int],
    mask: np.ndarray | None = None,
    *, max_bilinear_log_span: float = .08,
) -> np.ndarray:
    """Sample only coherent contributing taps; an observed integer pixel survives."""
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.ndim == 3:
        depth_arr = depth_arr[..., 0]
    if depth_arr.ndim != 2:
        raise ValueError("depth must have shape (H, W)")

    uv_arr = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    native_w, native_h = int(native_wh[0]), int(native_wh[1])
    if native_w <= 0 or native_h <= 0:
        raise ValueError("native_wh must be positive")
    if not 0 < max_bilinear_log_span <= .08:
        raise ValueError("bilinear depth-layer gate may not exceed .08 log-depth")
    mask_arr = None if mask is None else np.asarray(mask, dtype=bool)
    if mask_arr is not None and mask_arr.shape != depth_arr.shape:
        raise ValueError("sampling mask is not aligned with depth")

    height, width = depth_arr.shape
    out = np.full(len(uv_arr), np.nan, dtype=np.float64)
    if len(uv_arr) == 0:
        return out

    x = uv_arr[:, 0] * (width / native_w)
    y = uv_arr[:, 1] * (height / native_h)
    finite_coords = np.isfinite(x) & np.isfinite(y)
    in_bounds = finite_coords & (x >= 0.0) & (y >= 0.0) & (x <= width - 1.0) & (y <= height - 1.0)
    if not in_bounds.any():
        return out

    xs = x[in_bounds]
    ys = y[in_bounds]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = xs - x0
    wy = ys - y0

    v00 = depth_arr[y0, x0]
    v10 = depth_arr[y0, x1]
    v01 = depth_arr[y1, x0]
    v11 = depth_arr[y1, x1]

    corners = np.stack((v00, v10, v01, v11))
    weights = np.stack(((1. - wx) * (1. - wy), wx * (1. - wy),
                        (1. - wx) * wy, wx * wy))
    contributing = weights > 0.
    valid_taps = np.isfinite(corners) & (corners > 0.)
    if mask_arr is not None:
        valid_taps &= np.stack((mask_arr[y0, x0], mask_arr[y0, x1],
                               mask_arr[y1, x0], mask_arr[y1, x1]))
    valid = np.all(~contributing | valid_taps, axis=0)
    low = np.min(np.where(contributing & valid_taps, corners, np.inf), axis=0)
    high = np.max(np.where(contributing & valid_taps, corners, -np.inf), axis=0)
    span = np.log(np.maximum(high, 1e-30)) - np.log(np.maximum(low, 1e-30))
    valid &= span <= max_bilinear_log_span
    sampled = np.sum(np.where(contributing & valid_taps, corners, 0.) * weights, axis=0)
    out[in_bounds] = np.where(valid, sampled, np.nan)
    return out


def _extract_cam_from_world(cam_from_world_or_image: Any) -> np.ndarray:
    """Extract 3x4 [R|t] camera pose matrix from diverse camera pose representations."""
    if hasattr(cam_from_world_or_image, "cam_from_world"):
        cfw = cam_from_world_or_image.cam_from_world()
        if hasattr(cfw, "matrix"):
            mat = np.asarray(cfw.matrix(), dtype=np.float64)
        else:
            mat = np.asarray(cfw, dtype=np.float64)
    elif hasattr(cam_from_world_or_image, "matrix"):
        mat = np.asarray(cam_from_world_or_image.matrix(), dtype=np.float64)
    else:
        mat = np.asarray(cam_from_world_or_image, dtype=np.float64)

    return validate_pose(mat)


def _camera_center(pose: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates: C = -R.T @ t."""
    return -pose[:3, :3].T @ pose[:3, 3]


def _load_raw_npz(path: Path, row: Mapping[str, Any], expected_digest: str | None = None) -> dict[str, Any]:
    """Read the original six-field cache with original hash/request provenance."""
    actual = Path(row.get("raw_depth_path", path))
    if not actual.is_file():
        raise FileNotFoundError(f"missing metric depth cache: {actual}")
    if row.get("raw_depth_sha256") and sha256_file(actual) != row["raw_depth_sha256"]:
        raise ValueError("raw depth archive hash mismatch")
    origin_request = row.get("raw_request_sha256", expected_digest)
    required = {"depth_m", "mask", "K_work", "source_to_work", "request_sha256", "image_sha256"}
    with np.load(actual, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError(f"invalid original metric depth cache fields: {actual}")
        data = {key: archive[key] for key in required}
    width, height = int(row["width"]), int(row["height"])
    for name, shape, dtype in (("depth_m", (height, width), np.dtype("<f4")),
                              ("mask", (height, width), np.dtype("bool")),
                              ("K_work", (3, 3), np.dtype("<f8")),
                              ("source_to_work", (3, 3), np.dtype("<f8"))):
        if data[name].shape != shape or data[name].dtype != dtype:
            raise ValueError(f"invalid metric depth cache shape/dtype {name}: {actual}")
    expected_k, expected_h = compute_work_intrinsics(np.asarray(row["K_source"]), width, height)
    if (not np.allclose(data["K_work"], expected_k, atol=1e-6) or
            not np.allclose(data["source_to_work"], expected_h, atol=1e-8)):
        raise ValueError("metric depth cache differs from frozen PINHOLE K/transform")
    if origin_request is not None and str(data["request_sha256"]) != origin_request:
        raise ValueError("original raw request SHA256 mismatch")
    if str(data["image_sha256"]) != str(row["image_sha256"]):
        raise ValueError("raw depth image SHA256 mismatch")
    depth, mask = data["depth_m"], data["mask"]
    if (np.any(mask & (~np.isfinite(depth) | (depth <= 0))) or
            np.any(~mask & ~np.isnan(depth))):
        raise ValueError("corrupt metric depth cache mask/depth values")
    data["request_sha256"], data["image_sha256"] = str(data["request_sha256"]), str(data["image_sha256"])
    return data


def build_covisibility_graph(
    reconstruction: Any,
    image_names: Sequence[str],
    min_shared_points: int = 4,
) -> dict[str, list[str]]:
    """Construct an image co-visibility adjacency list from sparse reconstruction tracks.

    Falls back to spatial proximity and viewing direction overlap when sparse tracks are sparse.
    """
    image_name_set = set(image_names)
    rec_images = getattr(reconstruction, "images", {})
    if isinstance(rec_images, Sequence):
        rec_images = {i: im for i, im in enumerate(rec_images)}

    id_to_name: dict[int, str] = {}
    centers: dict[str, np.ndarray] = {}
    look_vectors: dict[str, np.ndarray] = {}

    for img_id, im in rec_images.items():
        name = getattr(im, "name", str(img_id))
        if name in image_name_set:
            id_to_name[img_id] = name
            try:
                pose = _extract_cam_from_world(im)
                center = _camera_center(pose)
                centers[name] = center
                # Optical axis in world frame is R.T @ e3 = pose[2, :3]
                look_vectors[name] = pose[2, :3]
            except Exception:
                pass

    shared_point_counts: dict[str, dict[str, int]] = {name: {} for name in image_names}
    rec_points = getattr(reconstruction, "points3D", {})
    if isinstance(rec_points, Sequence):
        rec_points = {i: pt for i, pt in enumerate(rec_points)}

    from .metric_anchors import valid_track_observations
    mapping_image_ids = frozenset(id_to_name)
    for point_id in rec_points:
        if not mapping_image_ids:
            break
        observed_names = [id_to_name[image_id] for image_id, _, _ in valid_track_observations(
            reconstruction, point_id, mapping_image_ids=mapping_image_ids)]
        if len(observed_names) < 2:
            continue
        unique_names = sorted(set(observed_names))
        for i, name_a in enumerate(unique_names):
            for name_b in unique_names[i + 1 :]:
                shared_point_counts[name_a][name_b] = shared_point_counts[name_a].get(name_b, 0) + 1
                shared_point_counts[name_b][name_a] = shared_point_counts[name_b].get(name_a, 0) + 1

    graph: dict[str, list[str]] = {}
    for name in image_names:
        shared = shared_point_counts.get(name, {})
        valid_shared = [(nbr, cnt) for nbr, cnt in shared.items() if cnt >= min_shared_points and nbr != name]
        valid_shared.sort(key=lambda item: item[1], reverse=True)
        neighbors = [nbr for nbr, _ in valid_shared]

        # Spatial fallback if sparse track overlap is insufficient
        if len(neighbors) < 6 and name in centers:
            c_ref = centers[name]
            v_ref = look_vectors.get(name)
            candidates = []
            for other in image_names:
                if other == name or other in neighbors or other not in centers:
                    continue
                dist = float(np.linalg.norm(c_ref - centers[other]))
                cos_dir = 1.0
                if v_ref is not None and other in look_vectors:
                    cos_dir = float(np.dot(v_ref, look_vectors[other]))
                if cos_dir > 0.3:
                    candidates.append((other, dist))
            candidates.sort(key=lambda item: item[1])
            neighbors.extend(nbr for nbr, _ in candidates[: 10 - len(neighbors)])

        graph[name] = neighbors[:12]

    return graph


def _compute_normals_and_confidences(
    X_world: np.ndarray, u_work: np.ndarray, v_work: np.ndarray,
    depth_work_map: np.ndarray, K_work: np.ndarray, pose: np.ndarray,
    n_agree: np.ndarray, n_conflict: np.ndarray, n_photo: np.ndarray,
    mean_rel_diff: np.ndarray, scale_m_per_unit: float,
    *, mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full 3D tangent normals; invalid patches remain the zero sentinel."""
    n = len(X_world)
    normals = np.zeros((n, 3), np.float32)
    normal_valid = np.zeros(n, bool)
    z = np.asarray(depth_work_map, dtype=np.float64) / scale_m_per_unit
    valid, boundary, _ = depth_patch_quality(z, np.isfinite(z) if mask is None else mask, radius=2)
    height, width = z.shape
    u, v = np.asarray(u_work, int), np.asarray(v_work, int)
    interior = (u >= 2) & (v >= 2) & (u < width - 2) & (v < height - 2)
    indices = np.flatnonzero(interior)
    indices = indices[valid[v[indices], u[indices]] & ~boundary[v[indices], u[indices]]]
    if len(indices):
        ui, vi = u[indices], v[indices]
        fx, fy, cx, cy = K_work[0, 0], K_work[1, 1], K_work[0, 2], K_work[1, 2]
        def backproject(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            d = z[y, x]
            return np.column_stack(((x - cx) * d / fx, (y - cy) * d / fy, d))
        tx = backproject(ui + 2, vi) - backproject(ui - 2, vi)
        ty = backproject(ui, vi + 2) - backproject(ui, vi - 2)
        cross = np.cross(tx, ty)
        lengths = np.linalg.norm(cross, axis=1)
        good = np.isfinite(cross).all(axis=1) & (lengths > 1e-12)
        world = (cross[good] / lengths[good, None]) @ pose[:, :3]
        chosen = indices[good]
        toward_camera = _camera_center(pose) - X_world[chosen]
        world[np.einsum("ij,ij->i", world, toward_camera) < 0] *= -1
        normals[chosen] = world
        normal_valid[chosen] = True
    confidences = np.clip(
        (1. - np.exp(-.75 * n_agree)) * np.exp(-1.8 * n_conflict) *
        np.maximum(.5, 1. - np.asarray(mean_rel_diff) / .16) *
        (1. + .15 * np.minimum(n_photo, 3)), .05, 1.).astype(np.float32)
    return normals, confidences, normal_valid


def write_dense_ply(
    path: Path | str,
    xyz: np.ndarray,
    confidence: np.ndarray,
    normals: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    comments: Sequence[str] = (),
) -> Path:
    """Write packed little-endian binary PLY with confidence-shaded colors."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(points)

    if rgb is not None:
        colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    else:
        # Confidence color gradient: high confidence green, moderate yellow, low orange
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        colors = np.zeros((n, 3), dtype=np.uint8)
        high = conf >= 0.85
        med_high = (conf >= 0.70) & ~high
        med = (conf >= 0.50) & (conf < 0.70)
        low = conf < 0.50

        colors[high] = [40, 200, 50]
        colors[med_high] = [140, 210, 40]
        colors[med] = [230, 200, 30]
        colors[low] = [235, 100, 30]

    header = [
        "ply",
        "format binary_little_endian 1.0",
    ]
    for c in comments:
        header.append(f"comment {c}")
    header.extend(
        [
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float confidence",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    )
    if normals is not None:
        header.extend(
            [
                "property float nx",
                "property float ny",
                "property float nz",
            ]
        )
    header.append("end_header\n")

    dtype_fields = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("confidence", "<f4"),
        ("r", "u1"),
        ("g", "u1"),
        ("b", "u1"),
    ]
    if normals is not None:
        dtype_fields.extend([("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])

    vertex_dt = np.dtype(dtype_fields)
    records = np.empty(n, dtype=vertex_dt)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["confidence"] = np.asarray(confidence, dtype=np.float32)
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]

    if normals is not None:
        norms = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        records["nx"] = norms[:, 0]
        records["ny"] = norms[:, 1]
        records["nz"] = norms[:, 2]

    temp_path = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    try:
        with temp_path.open("wb") as f:
            f.write("\n".join(header[:-1]).encode("ascii") + b"\nend_header\n")
            f.write(records.tobytes(order="C"))
        temp_path.replace(dest)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return dest


@dataclass(frozen=True)
class DenseMap:
    """In-memory compact dense map representation for fast CPU queries and depth rendering."""

    xyz: np.ndarray
    confidence: np.ndarray
    support_count: np.ndarray
    normals: np.ndarray | None = None
    source_image_id: np.ndarray | None = None
    meters_per_map_unit: float = 1.0
    scale_status: str = "ESTIMATED_UNVALIDATED"
    session_disagreement: float | None = None
    model_hashes: dict[str, str] | None = None
    request_sha256: str | None = None
    metric_scale_sha256: str | None = None
    metadata: dict[str, Any] | None = None
    evidence: Mapping[str, np.ndarray] | None = None
    observations: Mapping[str, np.ndarray] | None = None
    reference_observations: Mapping[str, np.ndarray] | None = None

    @property
    def normal_valid(self) -> np.ndarray | None:
        return None if self.evidence is None else self.evidence["normal_valid"]

    @property
    def surface_id(self) -> np.ndarray | None:
        return None if self.evidence is None else self.evidence["surface_id"]

    def __len__(self) -> int:
        return len(self.xyz)


def load_dense_map(target: DenseMap | Path | str | Mapping[str, Any]) -> DenseMap:
    """Load canonical offline NPZ, preserving and validating every v4 field.

    Historical v3 archives remain readable for frozen offline A/B only; they
    do not acquire an invented evidence ledger or observation CSR.
    """
    if isinstance(target, DenseMap):
        return target
    if isinstance(target, Mapping):
        arrays = dict(target)
    else:
        path = Path(target)
        if not path.is_file():
            raise FileNotFoundError(f"dense map file not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
    def decoded(name: str) -> Any:
        value = arrays.get(name)
        if isinstance(value, (str, np.ndarray)):
            return json.loads(str(value)) if str(value) else None
        return value
    meta = decoded("metadata") or {}
    n = len(arrays["xyz"])
    for name, shape, dtype in (("xyz", (n, 3), np.dtype("<f4")),
                                ("confidence", (n,), np.dtype("<f4")),
                                ("support_count", (n,), np.dtype("<u2"))):
        value = np.asarray(arrays[name])
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f"malformed canonical dense array {name}")
    if np.any((arrays["confidence"] < 0) | (arrays["confidence"] > 1)):
        raise ValueError("dense ranking heuristic is outside [0,1]")
    rich = meta.get("validation_version") == VALIDATION_VERSION
    if rich or "ledger_schema" in meta:
        if not rich:
            raise ValueError("unsupported evidence-grade dense validation version")
        validate_evidence_ledger(arrays, n, meta)
    declarations = meta.get("ledger_fields", {})
    for name, spec in declarations.items():
        if (name not in arrays or np.asarray(arrays[name]).dtype.str != spec["dtype"] or
                np.asarray(arrays[name]).shape != (n, *spec["shape"])):
            raise ValueError(f"malformed declared dense ledger member {name}")
    disagree = arrays.get("session_disagreement")
    disagreement = float(disagree) if disagree is not None and np.isfinite(disagree) else None
    observation_names = tuple(name for name in arrays if name.startswith(("view_", "observation_")))
    return DenseMap(
        xyz=arrays["xyz"], confidence=arrays["confidence"], support_count=arrays["support_count"],
        normals=arrays.get("normals"), source_image_id=arrays.get("source_image_id"),
        meters_per_map_unit=float(arrays.get("meters_per_map_unit", 1.)),
        scale_status=str(arrays.get("scale_status", "ESTIMATED_UNVALIDATED")),
        session_disagreement=disagreement, model_hashes=decoded("model_hashes"),
        request_sha256=str(arrays.get("request_sha256", "")),
        metric_scale_sha256=str(arrays.get("metric_scale_sha256", "")), metadata=meta,
        evidence={name: arrays[name] for name in arrays if name in LEDGER_DTYPES or name in declarations} if rich else None,
        observations={name: arrays[name] for name in observation_names} if rich else None,
        reference_observations={name: arrays[name] for name in
                                ("reference_surface_id", "reference_image_id", "reference_uv")} if rich else None,
    )


def render_depth(
    dense_map: DenseMap | Path | str | Mapping[str, Any],
    cam_from_world: np.ndarray | Any,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    meters_per_map_unit: float | None = None,
    splat_radius: int = 1,
    max_depth_m: float | None = None,
    mode: str = "optical_z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Offline CPU reference projective / ray depth rendering with z-buffering.

    Metre rendering refuses rejected scale status or session disagreement >20%.

    Args:
        dense_map: Loaded DenseMap object, path to dense_map.npz, or mapping.
        cam_from_world: 3x4 or 4x4 camera pose matrix [R|t].
        K: 3x3 camera intrinsic matrix.
        width: Output image width.
        height: Output image height.
        meters_per_map_unit: Conversion factor from map units to meters.
        splat_radius: Pixel splat radius (0 for exact point, 1 for 3x3 neighborhood).
        max_depth_m: Optional maximum depth cutoff in meters.
        mode: "optical_z" (depth along optical axis in meters), "ray" (Euclidean range in meters),
              or "map_units" (unscaled map optical z).

    Returns:
        depth_m: (H, W) float32 array (NaN for unrendered pixels).
        valid: (H, W) bool array indicating rendered surface coverage.
        confidence: (H, W) float32 array in [0, 1].
    """
    dense = load_dense_map(dense_map)

    # Integrity guard: fail closed if metric calibration status indicates unresolved disagreement
    if mode in ("optical_z", "ray"):
        if dense.scale_status == "SCALE_DISAGREEMENT_UNVALIDATED":
            raise ValueError(
                "Refusing metre rendering: metric scale status is SCALE_DISAGREEMENT_UNVALIDATED"
            )
        if dense.session_disagreement is not None and dense.session_disagreement > .20 + 1e-12:
            raise ValueError(
                f"Refusing metre rendering: session disagreement {dense.session_disagreement:.1%} exceeds 20%"
            )

    empty_depth = np.full((height, width), np.nan, dtype=np.float32)
    empty_valid = np.zeros((height, width), dtype=bool)
    empty_conf = np.zeros((height, width), dtype=np.float32)

    if len(dense) == 0:
        return empty_depth, empty_valid, empty_conf

    pose = _extract_cam_from_world(cam_from_world)
    k_mat = np.asarray(K, dtype=np.float64)
    if k_mat.shape != (3, 3):
        raise ValueError(f"expected 3x3 intrinsics matrix, got shape {k_mat.shape}")

    if mode == "map_units":
        scale = 1.0
    else:
        scale = float(meters_per_map_unit) if meters_per_map_unit is not None else dense.meters_per_map_unit
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid meters_per_map_unit: {scale}")

    # World to camera transformation: X_cam = X_world @ R.T + t
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_cam = dense.xyz @ R.T + t

    xc = points_cam[:, 0]
    yc = points_cam[:, 1]
    zc = points_cam[:, 2]

    # Filter points in front of the camera
    front_mask = zc > 1e-4
    if max_depth_m is not None and max_depth_m > 0 and mode != "map_units":
        max_zc_map = max_depth_m / scale
        front_mask = front_mask & (zc <= max_zc_map)

    if not np.any(front_mask):
        return empty_depth, empty_valid, empty_conf

    xc = xc[front_mask]
    yc = yc[front_mask]
    zc = zc[front_mask]
    conf = dense.confidence[front_mask]

    # Pinhole projection
    fx, fy = k_mat[0, 0], k_mat[1, 1]
    cx, cy = k_mat[0, 2], k_mat[1, 2]
    u = fx * (xc / zc) + cx
    v = fy * (yc / zc) + cy

    # Boundary filter considering splat radius
    pad = max(0, int(splat_radius))
    in_view = (u >= -pad) & (u < width + pad) & (v >= -pad) & (v < height + pad)
    if not np.any(in_view):
        return empty_depth, empty_valid, empty_conf

    u = u[in_view]
    v = v[in_view]
    zc = zc[in_view]
    conf = conf[in_view]

    if mode == "ray":
        rendered_val = np.sqrt(xc[in_view] ** 2 + yc[in_view] ** 2 + zc ** 2) * scale
    else:
        rendered_val = zc * scale

    u_center = np.rint(u).astype(np.int64)
    v_center = np.rint(v).astype(np.int64)

    all_u, all_v, all_z, all_val, all_conf = [], [], [], [], []
    if pad == 0:
        in_bounds = (u_center >= 0) & (u_center < width) & (v_center >= 0) & (v_center < height)
        all_u = u_center[in_bounds]
        all_v = v_center[in_bounds]
        all_z = zc[in_bounds]
        all_val = rendered_val[in_bounds]
        all_conf = conf[in_bounds]
    else:
        offsets = [(du, dv) for du in range(-pad, pad + 1) for dv in range(-pad, pad + 1)]
        for du, dv in offsets:
            uk = u_center + du
            vk = v_center + dv
            in_b = (uk >= 0) & (uk < width) & (vk >= 0) & (vk < height)
            if np.any(in_b):
                all_u.append(uk[in_b])
                all_v.append(vk[in_b])
                all_z.append(zc[in_b])
                all_val.append(rendered_val[in_b])
                all_conf.append(conf[in_b])
        if all_u:
            all_u = np.concatenate(all_u)
            all_v = np.concatenate(all_v)
            all_z = np.concatenate(all_z)
            all_val = np.concatenate(all_val)
            all_conf = np.concatenate(all_conf)
        else:
            return empty_depth, empty_valid, empty_conf

    # Vectorized z-buffering: sort by descending zc so closest points overwrite farthest
    order = np.argsort(-all_z)
    u_sorted = all_u[order]
    v_sorted = all_v[order]
    val_sorted = all_val[order]
    conf_sorted = all_conf[order]

    depth_buf = np.full((height, width), np.nan, dtype=np.float32)
    conf_buf = np.zeros((height, width), dtype=np.float32)

    depth_buf[v_sorted, u_sorted] = val_sorted.astype(np.float32)
    conf_buf[v_sorted, u_sorted] = conf_sorted.astype(np.float32)

    valid = np.isfinite(depth_buf) & (depth_buf > 0)
    depth_m = np.where(valid, depth_buf, np.nan).astype(np.float32)
    confidence_out = np.where(valid, conf_buf, 0.0).astype(np.float32)

    return depth_m, valid, confidence_out


def _stable_surface_id(xyz: np.ndarray, identity: Any) -> int:
    material = np.asarray(xyz, dtype="<f4").tobytes() + json.dumps(identity, sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little") & ((1 << 63) - 1)


def consolidate_surfaces(arrays: Mapping[str, np.ndarray], *, voxel_size_map: float,
                         view_keys: Sequence[Mapping[str, Any]],
                         view_poses: np.ndarray | None = None,
                         view_intrinsics: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Deterministic visibility-compatible layers with an observed medoid.

    Every reference pixel survives and is remapped to the shared stable surface.
    Boundary/invalid-normal samples can consolidate only exact duplicate XYZ;
    there is no smoothing or interpolation across unknown layers.
    """
    if not math.isfinite(voxel_size_map) or voxel_size_map <= 0:
        raise ValueError("voxel size must be finite and positive")
    result = {name: np.asarray(value) for name, value in arrays.items()}
    if len(view_keys) != len(result["view_image_id"]):
        raise ValueError("surface correlation view metadata is misaligned")
    if view_poses is not None:
        view_poses = np.asarray([validate_pose(pose) for pose in view_poses])
        if view_poses.shape != (len(view_keys), 3, 4):
            raise ValueError("surface optical-Z camera poses are misaligned")
    xyz = result["xyz"]
    count = len(xyz)
    grid = np.floor(xyz / voxel_size_map).astype(np.int64)
    order = sorted(range(count), key=lambda i: (tuple(grid[i]), tuple(xyz[i]),
                                                int(result["surface_id"][i])))
    groups: list[list[int]] = []
    voxels: dict[tuple[int, ...], list[int]] = {}
    def compatible(a: int, b: int) -> bool:
        delta = xyz[a].astype(float) - xyz[b]
        if np.linalg.norm(delta) <= 1e-8:
            return not (result["normal_valid"][a] and result["normal_valid"][b] and
                        np.dot(result["normals"][a], result["normals"][b]) < 0)
        if (result["source_boundary"][a] or result["source_boundary"][b] or
                not result["normal_valid"][a] or not result["normal_valid"][b]):
            return False
        normals = result["normals"][[a, b]].astype(float)
        if np.dot(normals[0], normals[1]) < math.cos(math.radians(15.)):
            return False
        if np.max(np.abs(normals @ delta)) > voxel_size_map * .02:
            return False
        # The representative must remain a valid observation in every original
        # source, not merely share a voxel with another point.
        if view_poses is None or view_intrinsics is None:
            return False
        for source, target in ((a, b), (b, a)):
            start, stop = result["observation_offsets"][source:source + 2].astype(int)
            for record in range(start, stop):
                residual_pose = view_poses[int(result["observation_residual_view_index"][record])]
                original_z = residual_pose[2, :3] @ xyz[source] + residual_pose[2, 3]
                proposed_z = residual_pose[2, :3] @ xyz[target] + residual_pose[2, 3]
                if original_z <= 0 or proposed_z <= 0:
                    return False
                residual = float(result["observation_residual"][record]) + math.log(proposed_z / original_z)
                flag = int(result["observation_flags"][record])
                if ((flag & (OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE | OBS_SPARSE_AGREE) and abs(residual) > .08) or
                        (flag & OBS_CONFLICT and residual >= -.08) or
                        (flag & OBS_OCCLUDED and residual <= .08)):
                    return False
                if not flag & OBS_SOURCE:
                    continue
                view = int(result["observation_view_index"][record])
                pose, k = view_poses[view], view_intrinsics[view]
                original = pose[:, :3] @ xyz[source] + pose[:, 3]
                proposed = pose[:, :3] @ xyz[target] + pose[:, 3]
                if original[2] <= 0 or proposed[2] <= 0 or abs(math.log(proposed[2] / original[2])) > .08:
                    return False
                p = k[:2, :2] @ (original[:2] / original[2]) + k[:2, 2]
                q = k[:2, :2] @ (proposed[:2] / proposed[2]) + k[:2, 2]
                if np.linalg.norm(p - q) > 2.:
                    return False
        return True
    for index in order:
        key = tuple(grid[index])
        matches = [g for g in voxels.get(key, []) if all(compatible(index, other) for other in groups[g])]
        if matches:
            groups[matches[0]].append(index)
        else:
            voxels.setdefault(key, []).append(len(groups))
            groups.append([index])
    point_fields = [name for name, value in result.items() if len(value.shape) and
                    value.shape[0] == count and not name.startswith(("view_", "observation_", "reference_"))]
    rows = {name: [] for name in point_fields}
    obs_indices, obs_flags, obs_residual, obs_group, obs_family, obs_evidence, obs_residual_view = [], [], [], [], [], [], []
    offsets = [0]
    remapping = {}
    for members in groups:
        points = xyz[members].astype(float)
        # Duplicate rows cannot weight the location; medoid is an actual
        # observed point, with deterministic geometric/ID tie-breaking.
        unique = np.unique(points, axis=0)
        cost = np.sum(np.linalg.norm(points[:, None] - unique[None, :], axis=2), axis=1)
        chosen = members[min(range(len(members)), key=lambda j: (cost[j], tuple(points[j]),
                                                                int(result["surface_id"][members[j]])))]
        identities = sorted({tuple(float(v) for v in xyz[i]) for i in members})
        surface_id = _stable_surface_id(xyz[chosen], [identities, bool(result["normal_valid"][chosen]),
                                                     result["normals"][chosen].tolist()])
        for index in members:
            remapping[int(result["surface_id"][index])] = surface_id
        center = result["view_camera_center"]
        source_view = int(np.flatnonzero(result["view_image_id"] == result["source_image_id"][chosen])[0])
        observations: set[tuple[int, int, int, int, int, float]] = set()
        for index in members:
            start, stop = result["observation_offsets"][index:index + 2].astype(int)
            for view, flag, residual, family, evidence_id, residual_view in zip(
                    result["observation_view_index"][start:stop], result["observation_flags"][start:stop],
                    result["observation_residual"][start:stop], result["observation_family_mask"][start:stop],
                    result["observation_evidence_id"][start:stop], result["observation_residual_view_index"][start:stop]):
                # Transport only the candidate numerator in the ACTUAL optical-Z
                # camera. The measured depth, state and capture identity stay real.
                if view_poses is not None:
                    pose = view_poses[int(residual_view)]
                    old_z = pose[2, :3] @ xyz[index] + pose[2, 3]
                    new_z = pose[2, :3] @ xyz[chosen] + pose[2, 3]
                    residual = float(residual) + math.log(new_z / old_z)
                residual = float(np.float32(residual))
                flag = int(flag)
                if flag & OBS_SOURCE and int(view) != source_view:
                    flag |= OBS_DEPTH_AGREE
                observations.add((int(view), flag, int(family), int(evidence_id), int(residual_view), residual))
        retained = sorted(observations)
        evidence_rows, directions, metadata, angle_values, evidence_observation = [], [], [], [], []
        source_direction = xyz[chosen] - center[source_view]
        source_direction /= max(np.linalg.norm(source_direction), 1e-12)
        for observation, (view, flag, family_mask, _, _, residual) in enumerate(retained):
            ray = xyz[chosen] - center[view]
            ray /= max(np.linalg.norm(ray), 1e-12)
            angle = math.degrees(math.acos(float(np.clip(np.dot(source_direction, ray), -1., 1.))))
            for kind, bit, family in ((SAME_MODEL, OBS_DEPTH_AGREE, "same_model"),
                                     (RECIPROCAL_PHOTO, OBS_PHOTO_AGREE, "reciprocal_photo"),
                                     (FIXED_POSE_MVS, OBS_MVS_AGREE, "fixed_pose_mvs")):
                if family_mask & kind and flag & (bit | OBS_CONFLICT | OBS_OCCLUDED):
                    evidence_rows.append([residual])
                    directions.append([ray])
                    metadata.append({**view_keys[view], "family": family})
                    angle_values.append([angle])
                    evidence_observation.append(observation)
        aggregate = aggregate_view_evidence(np.asarray(evidence_rows).reshape(-1, 1),
                    np.asarray(directions).reshape(-1, 1, 3), np.zeros((len(evidence_rows), 1), bool),
                    tolerance=.08, separation_deg=2., view_keys=metadata,
                    angles=np.asarray(angle_values).reshape(-1, 1), detailed=True)
        observation_groups = np.zeros(len(retained), np.uint32)
        for row, observation in enumerate(evidence_observation):
            if aggregate.assignments[row, 0] >= 0:
                observation_groups[observation] = int(aggregate.assignments[row, 0]) + 1
        for name in point_fields:
            rows[name].append(result[name][chosen])
        rows["surface_id"][-1] = surface_id
        for name, value in (("effective_same_model_support", aggregate.same_model_support[0]),
                            ("reciprocal_photo_support", aggregate.photo_support[0]),
                            ("mvs_support", aggregate.mvs_support[0]), ("support_count", aggregate.geometric_support[0]),
                            ("conflict_count", aggregate.conflict[0]), ("occluded_count", aggregate.occluded[0]),
                            ("mean_abs_log_residual", aggregate.mean_abs_residual[0]),
                            ("p90_abs_log_residual", aggregate.p90_abs_residual[0]),
                            ("independent_session_count", aggregate.session_count[0]),
                            ("independent_source_group_count", aggregate.source_group_count[0])):
            rows[name][-1] = value
        for name, value in (("min_usable_angle_deg", aggregate.min_angle[0]),
                            ("median_usable_angle_deg", aggregate.median_angle[0]),
                            ("max_usable_angle_deg", aggregate.max_angle[0]),
                            ("raw_same_model_count", aggregate.raw_same_model_count[0]),
                            ("raw_conflict_count", aggregate.raw_conflict_count[0]),
                            ("unknown_count", aggregate.unknown_count[0])):
            rows[name][-1] = value
        rows["sparse_anchor_support"][-1] = len({evidence_id for _, flag, _, evidence_id, _, _ in retained
                                                if flag & OBS_SPARSE_AGREE and evidence_id >= 0})
        rows["sparse_anchor_conflict"][-1] = len({evidence_id for _, flag, family, evidence_id, _, _ in retained
            if family & SPARSE_ANCHOR and flag & (OBS_CONFLICT | OBS_OCCLUDED) and evidence_id >= 0})
        rows["conflict_count"][-1] += rows["sparse_anchor_conflict"][-1]
        rows["edge_quality"][-1] = min(result["edge_quality"][members])
        rows["source_boundary"][-1] = bool(np.any(result["source_boundary"][members]))
        rows["local_calibration_quality"][-1] = min(result["local_calibration_quality"][members])
        rows["confidence"][-1] = np.clip((1. - math.exp(-.75 * rows["support_count"][-1])) *
            math.exp(-1.8 * rows["conflict_count"][-1]) *
            max(.5, 1. - rows["mean_abs_log_residual"][-1] / .16) *
            (1. + .15 * min(rows["reciprocal_photo_support"][-1], 3)), .05, 1.)
        kind = 0
        for field, bit in (("effective_same_model_support", SAME_MODEL), ("reciprocal_photo_support", RECIPROCAL_PHOTO),
                           ("sparse_anchor_support", SPARSE_ANCHOR), ("mvs_support", FIXED_POSE_MVS)):
            if rows[field][-1]:
                kind |= bit
        rows["evidence_kind_mask"][-1] = kind
        for observation, (view, flag, family, evidence_id, residual_view, residual) in enumerate(retained):
            obs_indices.append(view)
            obs_flags.append(flag)
            obs_residual.append(residual)
            obs_group.append(observation_groups[observation])
            obs_family.append(family)
            obs_evidence.append(evidence_id)
            obs_residual_view.append(residual_view)
        offsets.append(len(obs_indices))
    for name in point_fields:
        result[name] = np.asarray(rows[name], dtype=result[name].dtype).reshape((-1, *result[name].shape[1:]))
    result.update(observation_offsets=np.asarray(offsets, np.uint64),
                  observation_view_index=np.asarray(obs_indices, np.uint32),
                  observation_flags=np.asarray(obs_flags, np.uint8),
                  observation_residual=np.asarray(obs_residual, np.float32),
                  observation_residual_view_index=np.asarray(obs_residual_view, np.uint32),
                  observation_family_mask=np.asarray(obs_family, np.uint8),
                  observation_evidence_id=np.asarray(obs_evidence, np.int64),
                  observation_group_index=np.asarray(obs_group, np.uint32))
    result["reference_surface_id"] = np.asarray([remapping[int(i)] for i in result["reference_surface_id"]], np.int64)
    reference_order = np.lexsort((result["reference_uv"][:, 1], result["reference_uv"][:, 0],
                                 result["reference_image_id"], result["reference_surface_id"]))
    for name in ("reference_surface_id", "reference_image_id", "reference_uv"):
        result[name] = result[name][reference_order]
    unique_reference = np.ones(len(result["reference_surface_id"]), bool)
    unique_reference[1:] = ((result["reference_surface_id"][1:] != result["reference_surface_id"][:-1]) |
                            (result["reference_image_id"][1:] != result["reference_image_id"][:-1]) |
                            np.any(result["reference_uv"][1:] != result["reference_uv"][:-1], axis=1))
    for name in ("reference_surface_id", "reference_image_id", "reference_uv"):
        result[name] = result[name][unique_reference]
    return result


def build_dense_map(
    *, reconstruction: Any,
    image_rows: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    image_scales: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    raw_depth_dir: Path | str, output_dir: Path | str, request_sha256: str | None = None,
    meters_per_map_unit: float = 1., min_support: int = 2, min_baseline_deg: float = 2.,
    max_rel_depth_diff: float = .08, max_points_per_image: int = 15000,
    subsample_stride: int = 4, voxel_size_m: float | None = None,
    require_independent_views: bool = True, model_hashes: Mapping[str, str] | None = None,
    metric_scale_sha256: str | None = None, scale_status: str = "ESTIMATED_UNVALIDATED",
    session_disagreement: float | None = None, geometry_anchor_ids: Mapping[int, float] | None = None,
    calibration_contract_data: Mapping[str, Any] | None = None,
    keyframes_sha256: str | None = None, photo_evidence_path: Path | str | None = None,
    mvs_evidence_path: Path | str | None = None,
    geometry_run_sha256: str | None = None, raw_depth_manifest_sha256: str | None = None,
    inference_provenance: Mapping[str, Any] | None = None,
    request_file_sha256: str | None = None,
    photo_evidence_sha256: str | None = None, mvs_evidence_sha256: str | None = None,
) -> dict[str, Any]:
    """Build canonical offline surfaces with real decomposed evidence and CSR."""
    if (not math.isfinite(meters_per_map_unit) or meters_per_map_unit <= 0 or min_support < 2 or
            min_baseline_deg < 2. or not 0 < max_rel_depth_diff <= .08 or
            not require_independent_views or max_points_per_image < 1 or subsample_stride < 1):
        raise ValueError("invalid or relaxed frozen dense geometry gates")
    row_values = list(image_rows.values()) if isinstance(image_rows, Mapping) else list(image_rows)
    scale_values = list(image_scales.values()) if isinstance(image_scales, Mapping) else list(image_scales)
    for row in (*row_values, *scale_values):
        validate_mapping_record(row)
    rows = {row["name"]: row for row in row_values}
    scales = {row["name"]: row for row in scale_values}
    if len(rows) != len(row_values) or len(scales) != len(scale_values) or set(rows) != set(scales):
        raise ValueError("dense input must exactly cover unique registered MAPPING image identities")
    for name, row in rows.items():
        if (scales[name]["image_sha256"] != row["image_sha256"] or
                scales[name]["session_id"] != row["session_id"] or
                scales[name]["frame_index"] != row["frame_index"] or
                scales[name]["width"] != row["width"] or scales[name]["height"] != row["height"] or
                not np.array_equal(scales[name]["K_source"], row["K_source"])):
            raise ValueError("dense calibration differs from the native MAPPING capture")
        path = Path(row["image_path"]).resolve(strict=True)
        if path.suffix.lower() not in (".jpg", ".jpeg") or sha256_file(path) != row["image_sha256"]:
            raise ValueError("native MAPPING JPEG changed before geometry artifact processing")
    if calibration_contract_data is not None:
        validate_calibration_contract(calibration_contract_data)
    contract = calibration_contract()
    raw_root, out_dir = Path(raw_depth_dir).resolve(strict=True), Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes = dict(model_hashes) if model_hashes is not None else None
    if isinstance(reconstruction, (str, Path)):
        rec_path = Path(reconstruction).resolve(strict=True)
        hashes = model_file_hashes(rec_path) if hashes is None else hashes
        import pycolmap
        reconstruction = pycolmap.Reconstruction(str(rec_path))
    images = {image.name: image for image in reconstruction.images.values()}
    ids = {image.name: int(i) for i, image in reconstruction.images.items()}
    if set(rows) != set(images) or len(images) != len(reconstruction.images):
        raise ValueError("dense allowlist differs from the exact frozen registered MAPPING model")
    mapping_image_ids = frozenset(ids.values())
    registered_names = sorted(rows)
    registered_captures = [
        {"image_id": ids[name], **{key: rows[name][key] for key in
            ("name", "session_id", "frame_index", "evaluation_role", "image_sha256")}}
        for name in registered_names]
    view_names, proper_poses, unsupported_sources = [], [], []
    for name, identity in zip(registered_names, registered_captures):
        try:
            pose = _extract_cam_from_world(images[name])
        except ValueError:
            unsupported_sources.append({**identity, "status": "INVALID_CAMERA_POSE",
                "reason": "IMPROPER_FROZEN_WORLD_TO_CAMERA_POSE"})
            continue
        view_names.append(name)
        proper_poses.append(pose)
    valid_names = [name for name in view_names if scales[name].get("status") == "SCALE_OK"]
    for name in valid_names:
        validate_depth_calibration(scales[name].get("depth_calibration"))
    graph = build_covisibility_graph(reconstruction, valid_names)
    view_index = {name: i for i, name in enumerate(view_names)}
    name_by_id = {ids[name]: name for name in view_names}
    poses = np.asarray(proper_poses, dtype=np.float64).reshape(-1, 3, 4)
    centers = np.asarray([_camera_center(pose) for pose in poses], dtype=np.float64).reshape(-1, 3)
    native_ks = np.asarray([rows[name]["K_source"] for name in view_names], dtype=float)
    for name, k in zip(view_names, native_ks):
        row, image = rows[name], images[name]
        camera = reconstruction.cameras[image.camera_id]
        validate_pinhole(k, int(row["width"]), int(row["height"]))
        if (str(camera.model).split(".")[-1] != "PINHOLE" or
                int(camera.width) != int(row["width"]) or int(camera.height) != int(row["height"]) or
                not np.allclose(camera.calibration_matrix(), k, atol=1e-8, rtol=0)):
            raise ValueError("dense source differs from frozen PINHOLE camera")
        if not np.array_equal(scales[name]["cam_from_world"], poses[view_index[name]]):
            raise ValueError("dense calibration differs from its frozen camera pose")
    cell_width = max(float(np.linalg.norm(np.ptp(centers, axis=0))) * .05, 1e-8) if len(centers) else 1.
    sessions = sorted({str(rows[name]["session_id"]) for name in view_names})
    capture_keys = []
    for name, center in zip(view_names, centers):
        row = rows[name]
        capture_keys.append({"session": str(row["session_id"]), "temporal_segment": int(row["frame_index"]) // 32,
                             "camera_cell": tuple(np.floor(center / cell_width).astype(int).tolist()),
                             "view_id": ids[name]})
    sources = sorted({json.dumps({k: v for k, v in key.items() if k != "view_id"}, sort_keys=True)
                      for key in capture_keys})
    source_indices = [sources.index(json.dumps({k: v for k, v in key.items() if k != "view_id"}, sort_keys=True))
                      for key in capture_keys]
    artifacts = []
    if photo_evidence_path is not None or mvs_evidence_path is not None:
        if hashes is None or keyframes_sha256 is None:
            raise ValueError("independent geometry artifacts require fixed model and keyframe hashes")
        if photo_evidence_path is not None:
            from .photographic_evidence import read_photographic_evidence
            artifacts.append(read_photographic_evidence(photo_evidence_path, expected_model_hashes=hashes,
                expected_keyframes_sha256=keyframes_sha256, expected_manifest_sha256=photo_evidence_sha256))
        if mvs_evidence_path is not None:
            from .offline_mvs import read_offline_mvs
            artifacts.append(read_offline_mvs(mvs_evidence_path, expected_model_hashes=hashes,
                expected_keyframes_sha256=keyframes_sha256, expected_manifest_sha256=mvs_evidence_sha256))
    for artifact in artifacts:
        for image_id, capture in artifact.captures.items():
            name = name_by_id.get(image_id)
            if name is None:
                raise ValueError("geometry artifact source is absent from the registered MAPPING allowlist")
            registered, image = rows[name], images[name]
            if (capture.name != name or capture.evaluation_role != "MAPPING" or
                    capture.session_id != registered["session_id"] or
                    capture.frame_index != registered["frame_index"] or
                    capture.image_sha256 != registered["image_sha256"] or
                    Path(capture.image_path).resolve(strict=True) != Path(registered["image_path"]).resolve(strict=True) or
                    capture.camera_id != image.camera_id or capture.width != registered["width"] or
                    capture.height != registered["height"] or
                    not np.allclose(capture.K_frozen, registered["K_source"], atol=1e-8, rtol=0) or
                    not np.allclose(capture.cam_from_world, poses[view_index[name]], atol=1e-8, rtol=0)):
                raise ValueError("geometry artifact differs from the exact registered MAPPING capture")
    from .metric_anchors import valid_track_observations
    # Sparse support uses exactly the same all-observation validation as anchors.
    sparse = {name: {} for name in valid_names}
    if geometry_anchor_ids is not None:
        for point_id in sorted(geometry_anchor_ids):
            for image_id, uv, _ in valid_track_observations(
                    reconstruction, point_id, mapping_image_ids=mapping_image_ids):
                name = name_by_id.get(image_id)
                if name in sparse:
                    sparse[name][point_id] = uv
    cache: dict[str, dict[str, Any]] = {}
    def get_depth(name: str) -> dict[str, Any]:
        if name in cache:
            return cache[name]
        row = rows[name]
        path = Path(row.get("raw_depth_path", raw_root / f"{safe_image_name(name).as_posix()}.npz"))
        raw = _load_raw_npz(path, row, request_sha256)
        z = raw["depth_m"]
        # Work-to-native affine PINHOLE transform: avoid materializing 3D grids.
        inverse = np.linalg.inv(raw["source_to_work"])
        h, w = z.shape
        uu = np.arange(w)[None, :] * inverse[0, 0] + inverse[0, 2]
        vv = np.arange(h)[:, None] * inverse[1, 1] + inverse[1, 2]
        corrected = apply_depth_calibration(z, uu, vv, scales[name]["depth_calibration"],
                                            width=int(row["width"]), height=int(row["height"]))
        valid, boundary, quality = depth_patch_quality(corrected, raw["mask"])
        value = {**raw, "corrected_depth": corrected, "source_valid": valid,
                 "source_boundary": boundary, "edge_quality": quality}
        if len(cache) > 24:
            del cache[next(iter(cache))]
        cache[name] = value
        return value
    chunks: list[dict[str, np.ndarray]] = []
    obs_chunks = []
    total_candidates = total_verified = total_conflicted = total_unknown = images_verified = 0
    for name in valid_names:
        row, scale_info = rows[name], scales[name]
        raw = get_depth(name)
        z, mask, k = raw["corrected_depth"], raw["mask"], raw["K_work"]
        pose = poses[view_index[name]]
        center = centers[view_index[name]]
        scale = math.exp(float(scale_info["log_scale"]))
        height, width = z.shape
        yy, xx = np.mgrid[0:height:subsample_stride, 0:width:subsample_stride]
        u, v = xx.ravel(), yy.ravel()
        keep = raw["source_valid"][v, u]
        u, v = u[keep], v[keep]
        if len(u) > max_points_per_image:
            selection = np.linspace(0, len(u) - 1, max_points_per_image, dtype=int)
            u, v = u[selection], v[selection]
        if not len(u):
            continue
        zmap = z[v, u] / scale
        camera_xyz = np.column_stack(((u - k[0, 2]) * zmap / k[0, 0],
                                     (v - k[1, 2]) * zmap / k[1, 1], zmap))
        world = camera_xyz @ pose[:, :3] + center
        inverse = np.linalg.inv(raw["source_to_work"])
        homogeneous = np.column_stack((u, v, np.ones(len(u)))) @ inverse.T
        native_uv = homogeneous[:, :2] / homogeneous[:, 2:3]
        m = len(world)
        total_candidates += m
        source_ray = world - center
        source_ray /= np.linalg.norm(source_ray, axis=1)[:, None]
        sparse_support, sparse_conflict = np.zeros(m, bool), np.zeros(m, bool)
        sparse_point_id = np.full(m, -1, np.int64)
        sparse_residual = np.zeros(m, np.float32)
        if sparse[name]:
            point_ids = sorted(sparse[name])
            observations = np.asarray([sparse[name][i] for i in point_ids])
            distances, indices = cKDTree(observations).query(native_uv, distance_upper_bound=2.)
            for candidate in np.flatnonzero(np.isfinite(distances)):
                sparse_point_id[candidate] = point_ids[indices[candidate]]
                xyz = reconstruction.points3D[point_ids[indices[candidate]]].xyz
                actual_z = pose[2, :3] @ xyz + pose[2, 3]
                residual = math.log(zmap[candidate] / actual_z)
                sparse_residual[candidate] = residual
                sparse_support[candidate] = abs(residual) <= max_rel_depth_diff
                sparse_conflict[candidate] = abs(residual) > max_rel_depth_diff
        residuals, rays, angles, keys, views, evidence_ids, residual_views = [], [], [], [], [], [], []
        def append_view(neighbor: str, values: np.ndarray, family: str,
                        observed_ids: np.ndarray | None = None,
                        residual_camera_ids: np.ndarray | None = None) -> None:
            neighbor_center = centers[view_index[neighbor]]
            ray = world - neighbor_center
            length = np.linalg.norm(ray, axis=1)
            direction = ray / np.maximum(length[:, None], 1e-12)
            angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", source_ray, direction), -1., 1.)))
            values = np.asarray(values, dtype=np.float32).copy()
            values[angle < min_baseline_deg] = np.nan
            nbr_row = rows[neighbor]
            if row["session_id"] == nbr_row["session_id"] and (
                    abs(int(row["frame_index"]) - int(nbr_row["frame_index"])) < 4 or
                    np.linalg.norm(center - neighbor_center) < .02):
                values[:] = np.nan
            residuals.append(values)
            rays.append(direction)
            angles.append(angle)
            keys.append({**capture_keys[view_index[neighbor]], "family": family})
            views.append(view_index[neighbor])
            evidence_ids.append(np.full(m, -1, np.int64) if observed_ids is None else np.asarray(observed_ids, np.int64))
            if residual_camera_ids is None:
                residual_views.append(np.full(m, view_index[neighbor], np.uint32))
            else:
                camera_ids = np.asarray(residual_camera_ids)
                if camera_ids.shape != (m,) or camera_ids.dtype.kind not in "iu":
                    raise ValueError("geometry artifact residual cameras are misaligned")
                valid_residual = np.isfinite(values)
                if not np.isin(camera_ids[valid_residual], list(name_by_id)).all():
                    raise ValueError("geometry artifact residual refers to an unknown optical-Z camera")
                residual_views.append(np.asarray([
                    view_index[name_by_id[int(camera)]] if valid else view_index[name]
                    for camera, valid in zip(camera_ids, valid_residual)], np.uint32))
        for neighbor in sorted(graph.get(name, [])):
            nbr = get_depth(neighbor)
            nbr_pose, nbr_k = poses[view_index[neighbor]], nbr["K_work"]
            xyz_cam = world @ nbr_pose[:, :3].T + nbr_pose[:, 3]
            depth_camera = xyz_cam[:, 2]
            projected = np.full((m, 2), np.nan)
            front = depth_camera > 1e-4
            projected[front] = xyz_cam[front, :2] / depth_camera[front, None]
            projected = projected @ nbr_k[:2, :2].T + nbr_k[:2, 2]
            sampled = sample_depth_bilinear(nbr["corrected_depth"], projected,
                         (nbr["corrected_depth"].shape[1], nbr["corrected_depth"].shape[0]), nbr["mask"])
            measured = sampled / math.exp(float(scales[neighbor]["log_scale"]))
            values = np.full(m, np.nan)
            known = front & np.isfinite(measured) & (measured > 0)
            values[known] = np.log(depth_camera[known] / measured[known])
            append_view(neighbor, values, "same_model")
        for artifact in artifacts:
            if artifact.evidence_family == "reciprocal_photo":
                capture = artifact.captures.get(ids[name])
                if capture is None:
                    continue
                # Pure coordinate-origin transport from the artifact's explicit
                # native convention, never an intrinsic/camera refit.
                photo_transform = np.asarray(capture.K_native) @ np.linalg.inv(np.asarray(row["K_source"]))
                photo_pixels = np.column_stack((native_uv, np.ones(m))) @ photo_transform.T
                photo_uv = photo_pixels[:, :2] / photo_pixels[:, 2:3]
                compared = artifact.compare_candidates(ids[name], photo_uv, world,
                                  pixel_tolerance=2., max_log_residual=max_rel_depth_diff)
            else:
                compared = artifact.compare_candidates(ids[name], world, max_log_residual=max_rel_depth_diff)
            shape = (len(compared["support_image_ids"]), m)
            if any(np.shape(compared[field]) != shape for field in
                   ("signed_log_residual", "evidence_ids", "residual_image_ids", "states")):
                raise ValueError("geometry artifact candidate comparison arrays are misaligned")
            for image_id, values, real_ids, camera_ids in zip(
                    compared["support_image_ids"], compared["signed_log_residual"],
                    compared["evidence_ids"], compared["residual_image_ids"]):
                neighbor = name_by_id.get(int(image_id))
                if neighbor is None:
                    raise ValueError("verified geometry artifact references an unknown capture")
                if neighbor != name:
                    append_view(neighbor, values, artifact.evidence_family, real_ids, camera_ids)
        aggregated = aggregate_view_evidence(np.asarray(residuals).reshape(-1, m),
                        np.asarray(rays).reshape(-1, m, 3), np.zeros((len(residuals), m), bool),
                        tolerance=max_rel_depth_diff, separation_deg=min_baseline_deg,
                        view_keys=keys, angles=np.asarray(angles).reshape(-1, m), detailed=True)
        # MVS is check-only: it cannot relax existing MoGe support admission.
        conflicted = sparse_conflict | ((aggregated.conflict > 0) &
                                      (aggregated.conflict >= aggregated.geometric_support))
        admission_rows = np.asarray([key["family"] != "fixed_pose_mvs" for key in keys], bool)
        if admission_rows.all():
            admission = aggregated
        else:
            admission = aggregate_view_evidence(
                np.asarray(residuals).reshape(-1, m)[admission_rows],
                np.asarray(rays).reshape(-1, m, 3)[admission_rows],
                np.zeros((int(admission_rows.sum()), m), bool),
                tolerance=max_rel_depth_diff, separation_deg=min_baseline_deg,
                view_keys=[key for key, allowed in zip(keys, admission_rows) if allowed],
                angles=np.asarray(angles).reshape(-1, m)[admission_rows], detailed=True)
        verified = ~conflicted & (admission.same_model_support >= min_support)
        # Photo may preserve thin boundaries at the same independent-view gate;
        # MVS remains check-only and cannot supply a missing admission vote.
        boundary = raw["source_boundary"][v, u]
        photo_supported = (admission.photo_support > 0) & (admission.geometric_support >= min_support)
        verified |= ~conflicted & boundary & photo_supported
        verified &= aggregated.geometric_support >= min_support
        total_verified += int(verified.sum())
        total_conflicted += int(conflicted.sum())
        total_unknown += int((~verified & ~conflicted).sum())
        selected = np.flatnonzero(verified)
        if not len(selected):
            continue
        images_verified += 1
        normals, confidence, normal_valid = _compute_normals_and_confidences(
            world[selected], u[selected], v[selected], z, k, pose,
            aggregated.geometric_support[selected], aggregated.conflict[selected],
            aggregated.photo_support[selected], aggregated.mean_abs_residual[selected], scale, mask=mask)
        surface_ids = np.asarray([_stable_surface_id(world[i], [ids[name], float(native_uv[i, 0]),
                                                              float(native_uv[i, 1])]) for i in selected], np.int64)
        kind = (np.where(aggregated.same_model_support[selected] > 0, SAME_MODEL, 0) |
                np.where(aggregated.photo_support[selected] > 0, RECIPROCAL_PHOTO, 0) |
                np.where(sparse_support[selected], SPARSE_ANCHOR, 0) |
                np.where(aggregated.mvs_support[selected] > 0, FIXED_POSE_MVS, 0))
        chunk = {
            "xyz": world[selected].astype(np.float32), "confidence": confidence,
            "support_count": aggregated.geometric_support[selected].astype(np.uint16),
            "normals": normals, "source_image_id": np.full(len(selected), ids[name], np.int32),
            "evidence_kind_mask": kind, "effective_same_model_support": aggregated.same_model_support[selected],
            "reciprocal_photo_support": aggregated.photo_support[selected], "sparse_anchor_support": sparse_support[selected],
            "sparse_anchor_conflict": sparse_conflict[selected], "conflict_count": aggregated.conflict[selected],
            "mean_abs_log_residual": aggregated.mean_abs_residual[selected],
            "p90_abs_log_residual": aggregated.p90_abs_residual[selected],
            "min_usable_angle_deg": aggregated.min_angle[selected], "median_usable_angle_deg": aggregated.median_angle[selected],
            "max_usable_angle_deg": aggregated.max_angle[selected], "independent_session_count": aggregated.session_count[selected],
            "independent_source_group_count": aggregated.source_group_count[selected],
            "edge_quality": raw["edge_quality"][v[selected], u[selected]],
            "local_calibration_quality": calibration_quality(native_uv[selected, 0], native_uv[selected, 1],
                raw["depth_m"][v[selected], u[selected]], scale_info["depth_calibration"],
                width=int(row["width"]), height=int(row["height"])),
            "normal_valid": normal_valid, "surface_id": surface_ids, "source_boundary": boundary[selected],
            "occluded_count": aggregated.occluded[selected], "mvs_support": aggregated.mvs_support[selected],
            "raw_same_model_count": aggregated.raw_same_model_count[selected],
            "raw_conflict_count": aggregated.raw_conflict_count[selected], "unknown_count": aggregated.unknown_count[selected],
        }
        for field, dtype in LEDGER_DTYPES.items():
            chunk[field] = np.asarray(chunk[field], dtype=dtype)
        for field in ("occluded_count", "mvs_support", "raw_same_model_count", "raw_conflict_count", "unknown_count"):
            chunk[field] = np.asarray(chunk[field], dtype=np.uint16)
        chunks.append(chunk)
        source_flags = np.full(len(selected), OBS_SOURCE, np.uint8)
        sparse_flags = np.where(sparse_support[selected], OBS_SPARSE_AGREE, 0).astype(np.uint8)
        flags = np.column_stack((source_flags, sparse_flags, aggregated.observation_flags[:, selected].T))
        rr, cc = np.nonzero(flags)
        counts = np.bincount(rr, minlength=len(selected))
        view_list = np.asarray([view_index[name], view_index[name], *views], np.uint32)
        signed = np.column_stack((np.zeros(len(selected)), sparse_residual[selected],
                                  np.asarray(residuals).reshape(-1, m)[:, selected].T))
        groups = np.column_stack((np.zeros((len(selected), 2), int), aggregated.assignments[:, selected].T + 1))
        family_list = np.asarray([SAME_MODEL, SPARSE_ANCHOR, *[{
            "same_model": SAME_MODEL, "reciprocal_photo": RECIPROCAL_PHOTO,
            "fixed_pose_mvs": FIXED_POSE_MVS}[key["family"]] for key in keys]], np.uint8)
        real_ids = np.column_stack((np.full(len(selected), -1, np.int64), sparse_point_id[selected],
                                    np.asarray(evidence_ids).reshape(-1, m)[:, selected].T))
        residual_view_list = np.column_stack((
            np.full((len(selected), 2), view_index[name], np.uint32),
            np.asarray(residual_views).reshape(-1, m)[:, selected].T))
        obs_chunks.append((np.cumsum(counts, dtype=np.uint64), view_list[cc], flags[rr, cc],
                           signed[rr, cc].astype(np.float32), groups[rr, cc].astype(np.uint32),
                           surface_ids, np.full(len(selected), ids[name], np.int32), native_uv[selected].astype(np.float32),
                           family_list[cc], real_ids[rr, cc], residual_view_list[rr, cc]))
    if not chunks:
        raise ValueError("Multiview depth verification rejected all candidate surfaces: 0 verified points")
    arrays = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    arrays.update(view_image_id=np.asarray([ids[name] for name in view_names], np.int32),
                  view_camera_center=centers.astype(np.float64),
                  view_session_index=np.asarray([sessions.index(str(rows[name]["session_id"])) for name in view_names], np.uint32),
                  view_source_index=np.asarray(source_indices, np.uint32))
    offsets, base = [np.zeros(1, np.uint64)], 0
    for chunk in obs_chunks:
        offsets.append(chunk[0] + base)
        base += len(chunk[1])
    arrays.update(observation_offsets=np.concatenate(offsets),
                  observation_view_index=np.concatenate([c[1] for c in obs_chunks]),
                  observation_flags=np.concatenate([c[2] for c in obs_chunks]),
                  observation_residual=np.concatenate([c[3] for c in obs_chunks]),
                  observation_group_index=np.concatenate([c[4] for c in obs_chunks]),
                  reference_surface_id=np.concatenate([c[5] for c in obs_chunks]),
                  reference_image_id=np.concatenate([c[6] for c in obs_chunks]),
                  reference_uv=np.concatenate([c[7] for c in obs_chunks]),
                  observation_family_mask=np.concatenate([c[8] for c in obs_chunks]),
                  observation_evidence_id=np.concatenate([c[9] for c in obs_chunks]),
                  observation_residual_view_index=np.concatenate([c[10] for c in obs_chunks]))
    if voxel_size_m is not None:
        arrays = consolidate_surfaces(arrays, voxel_size_map=voxel_size_m / meters_per_map_unit,
                                       view_keys=capture_keys, view_poses=poses, view_intrinsics=native_ks)
    n = len(arrays["xyz"])
    metadata = {
        "validation_version": VALIDATION_VERSION, "ledger_schema": LEDGER_SCHEMA,
        "score_semantics": SCORE_SEMANTICS, "calibration_contract": contract,
        "validator_sha256": sha256_file(Path(__file__)), "view_names": view_names,
        "view_sessions": sessions, "view_sources": sources, "camera_cell_width": cell_width,
        "view_capture_identity": [
            {"image_id": ids[name], **{key: rows[name][key] for key in
                ("name", "session_id", "frame_index", "evaluation_role", "image_sha256")}}
            for name in view_names],
        "registered_capture_identity": registered_captures,
        "unsupported_sources": unsupported_sources,
        "geometry_run_sha256": geometry_run_sha256, "raw_depth_manifest_sha256": raw_depth_manifest_sha256,
        "keyframes_sha256": keyframes_sha256,
        "request_file_sha256": request_file_sha256,
        "inference_provenance": dict(inference_provenance) if inference_provenance is not None else None,
        "ledger_fields": {name: {"dtype": value.dtype.str, "shape": list(value.shape[1:]), "meaning": name}
                          for name, value in arrays.items() if value.shape[0] == n and
                          not name.startswith(("view_", "observation_", "reference_"))},
        "independent_artifacts": [{"family": a.evidence_family, "manifest_sha256": a.manifest_sha256} for a in artifacts],
        "depth_calibration_digests": {name: scales[name]["depth_calibration"]["digest"] for name in valid_names},
        "per_image_log_scales": {name: float(scales[name]["log_scale"]) for name in valid_names},
        "total_candidate_points": total_candidates, "total_verified_points": total_verified,
        "total_conflicted_points": total_conflicted, "total_unverified_points": total_unknown,
        "min_support": min_support, "min_baseline_deg": min_baseline_deg,
        "max_rel_depth_diff": max_rel_depth_diff, "subsample_stride": subsample_stride,
        "voxel_size_m": voxel_size_m, "num_fused_points": n,
    }
    validate_evidence_ledger(arrays, n, metadata)
    npz_path, ply_path, summary_path = out_dir / "dense_map.npz", out_dir / "dense_map.ply", out_dir / "dense_summary.json"
    temp = npz_path.with_name(f"{npz_path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez_compressed(temp, **arrays, meters_per_map_unit=np.float64(meters_per_map_unit),
            scale_status=np.array(scale_status), session_disagreement=np.float64(session_disagreement if session_disagreement is not None else np.nan),
            model_hashes=np.array(json.dumps(hashes, sort_keys=True) if hashes else ""), request_sha256=np.array(request_sha256 or ""),
            metric_scale_sha256=np.array(metric_scale_sha256 or ""), metadata=np.array(json.dumps(metadata, allow_nan=False)))
        temp.replace(npz_path)
    finally:
        temp.unlink(missing_ok=True)
    write_dense_ply(ply_path, arrays["xyz"], arrays["confidence"], arrays["normals"],
                    comments=("Offline supplemental verified surface; ranking score is not probability",))
    summary = {"status": "DENSE_MAP_BUILT", "dense_map_npz": str(npz_path), "dense_map_ply": str(ply_path),
               "dense_summary_json": str(summary_path), "num_verified_points": n, "num_candidate_points": total_candidates,
               "num_conflicted_points": total_conflicted, "num_unverified_points": total_unknown,
               "verification_rate": n / max(total_candidates, 1), "conflict_rate": total_conflicted / max(total_candidates, 1),
               "unverified_rate": total_unknown / max(total_candidates, 1), "num_images_processed": len(valid_names),
               "num_images_with_verified_points": images_verified, "meters_per_map_unit": meters_per_map_unit,
               "scale_status": scale_status, "session_disagreement": session_disagreement, "metadata": metadata}
    write_json(summary_path, summary)
    return summary
