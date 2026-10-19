"""Replace GlueMap SIFT observations with official EDM on frozen poses.

Adopted mapping stage after all-sequence GlueMap: covisibility + temporal pairs,
official EDM matching, Sampson/angle inliers, 2 px cell quantization, then
fixed-intrinsics triangulation. Poses never move.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sfm_diagnosis.io import write_json

CELL_PX = 2.0
COVIS_MIN_SHARED = 15
COVIS_TOP_K = 20
TEMPORAL_ORDINAL = 8
# Guardrail, not a policy: selection yields ~13.4 pairs/image (18,599 for 1,389
# images; 27,225 for 2,043), so the cap must scale with corpus size.
MAX_PAIRS = 40_000
SAMPSON_MAX_PX = 3.0
MIN_TRIANGULATION_ANGLE_DEG = 1.0
MIN_INLIERS = 30
KEYPOINT_CAP = 50_000
KEYPOINT_CAP_OOM = 20_000
LIFT_DISTANCE_PX = 4.0
TEST_RES = 640
EDM_TOPK = 2240

Matcher = Callable[[Path, Path], tuple[np.ndarray, np.ndarray, np.ndarray]]


def quantize_xy(
    uv: np.ndarray | tuple[float, float], cell: float = CELL_PX
) -> tuple[int, int]:
    u, v = float(uv[0]), float(uv[1])
    return (int(math.floor(u / cell)), int(math.floor(v / cell)))


def pair_id(name_a: str, name_b: str) -> str:
    return hashlib.sha256((name_a + "\0" + name_b).encode("utf-8")).hexdigest()[:16]


def _skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = (float(t[0]), float(t[1]), float(t[2]))
    return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)


def w2c_4x4(image: Any) -> np.ndarray:
    matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = matrix
        return out
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"unexpected cam_from_world matrix shape {matrix.shape}")


def k_from_camera(camera: Any) -> np.ndarray:
    model = str(getattr(camera.model, "name", camera.model)).rsplit(".", 1)[-1]
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model != "PINHOLE" or params.size < 4:
        raise ValueError(f"expected PINHOLE camera, got {model} params={params}")
    fx, fy, cx, cy = (float(params[0]), float(params[1]), float(params[2]), float(params[3]))
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def sampson_pixels(
    uv1: np.ndarray, uv2: np.ndarray, essential: np.ndarray, k1: np.ndarray, k2: np.ndarray
) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    ones = np.ones((len(points1), 1), dtype=np.float64)
    x1 = np.concatenate([points1, ones], axis=1)
    x2 = np.concatenate([points2, ones], axis=1)
    fx1 = x1 @ fundamental.T
    ftx2 = x2 @ fundamental
    num = np.sum(x2 * fx1, axis=1) ** 2
    den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    return np.sqrt(np.maximum(num / np.maximum(den, 1e-18), 0.0))


def triangulate_linear(
    uv1: np.ndarray, uv2: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    n = len(points1)
    design = np.empty((n, 4, 4), dtype=np.float64)
    design[:, 0] = points1[:, 0:1] * p1[2] - p1[0]
    design[:, 1] = points1[:, 1:2] * p1[2] - p1[1]
    design[:, 2] = points2[:, 0:1] * p2[2] - p2[0]
    design[:, 3] = points2[:, 1:2] * p2[2] - p2[1]
    _, _, vh = np.linalg.svd(design)
    homogeneous = vh[:, -1, :]
    w = homogeneous[:, 3:4]
    return homogeneous[:, :3] / np.where(np.abs(w) < 1e-12, np.nan, w)


def camera_centers(t_w2c: np.ndarray) -> np.ndarray:
    rotation = t_w2c[:3, :3]
    translation = t_w2c[:3, 3]
    return -rotation.T @ translation


def triangulation_angles_deg(
    xyz: np.ndarray, center_a: np.ndarray, center_b: np.ndarray
) -> np.ndarray:
    ray_a = xyz - center_a
    ray_b = xyz - center_b
    norm_a = np.linalg.norm(ray_a, axis=1)
    norm_b = np.linalg.norm(ray_b, axis=1)
    denom = np.maximum(norm_a * norm_b, 1e-18)
    cosine = np.clip(np.sum(ray_a * ray_b, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def frozen_pose_inliers(
    uv1: np.ndarray,
    uv2: np.ndarray,
    t_i: np.ndarray,
    t_j: np.ndarray,
    k_i: np.ndarray,
    k_j: np.ndarray,
    *,
    sampson_max_px: float = SAMPSON_MAX_PX,
    min_angle_deg: float = MIN_TRIANGULATION_ANGLE_DEG,
) -> np.ndarray:
    points1 = np.asarray(uv1, dtype=np.float64).reshape(-1, 2)
    points2 = np.asarray(uv2, dtype=np.float64).reshape(-1, 2)
    if len(points1) == 0:
        return np.zeros(0, dtype=bool)
    relative = t_j @ np.linalg.inv(t_i)
    rotation = relative[:3, :3]
    translation = relative[:3, 3]
    essential = _skew(translation) @ rotation
    sampson = sampson_pixels(points1, points2, essential, k_i, k_j)
    p1 = k_i @ t_i[:3]
    p2 = k_j @ t_j[:3]
    xyz = triangulate_linear(points1, points2, p1, p2)
    cam_z_i = (t_i[:3, :3] @ xyz.T + t_i[:3, 3:4])[2]
    cam_z_j = (t_j[:3, :3] @ xyz.T + t_j[:3, 3:4])[2]
    angles = triangulation_angles_deg(xyz, camera_centers(t_i), camera_centers(t_j))
    return (
        np.isfinite(sampson)
        & np.isfinite(xyz).all(axis=1)
        & np.isfinite(cam_z_i)
        & np.isfinite(cam_z_j)
        & np.isfinite(angles)
        & (sampson <= sampson_max_px)
        & (cam_z_i > 0.0)
        & (cam_z_j > 0.0)
        & (angles >= min_angle_deg)
    )


def pose_only_and_tracks(
    reconstruction: Any, keyframes: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], dict[str, set[int]], dict[str, int]]:
    pose_only_mode = {
        name
        for name, row in keyframes.items()
        if str(row.get("mapping_mode") or "") == "POSE_ONLY"
    }
    pose_only: set[str] = set()
    tracks: dict[str, set[int]] = {}
    image_ids: dict[str, int] = {}
    for image_id, image in reconstruction.images.items():
        name = str(image.name)
        image_ids[name] = int(image_id)
        if name in pose_only_mode:
            pose_only.add(name)
            continue
        tracks[name] = {
            int(point.point3D_id) for point in image.points2D if point.has_point3D()
        }
    return pose_only, tracks, image_ids


def select_retriangulation_pairs(
    tracks: Mapping[str, set[int]],
    keyframes: Mapping[str, Mapping[str, Any]],
    *,
    covis_min_shared: int = COVIS_MIN_SHARED,
    covis_top_k: int = COVIS_TOP_K,
    temporal_ordinal: int = TEMPORAL_ORDINAL,
    max_pairs: int = MAX_PAIRS,
) -> list[dict[str, Any]]:
    point_to_images: dict[int, list[str]] = defaultdict(list)
    for name, pids in tracks.items():
        for pid in pids:
            point_to_images[pid].append(name)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for names in point_to_images.values():
        unique = sorted(set(names))
        for i, left in enumerate(unique):
            for right in unique[i + 1 :]:
                shared[(left, right)] += 1

    neighbors: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (left, right), count in shared.items():
        if count < covis_min_shared:
            continue
        neighbors[left].append((count, right))
        neighbors[right].append((count, left))

    covis: dict[tuple[str, str], int] = {}
    for name, items in neighbors.items():
        items.sort(key=lambda row: (-row[0], row[1]))
        for count, other in items[:covis_top_k]:
            pair = (name, other) if name < other else (other, name)
            covis[pair] = shared[pair]

    temporal: set[tuple[str, str]] = set()
    by_video: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for name in tracks:
        row = keyframes[name]
        by_video[name.split("/")[0]].append((int(row["frame_index"]), name))
    for rows in by_video.values():
        rows.sort()
        ordered = [name for _index, name in rows]
        for i, left in enumerate(ordered):
            for delta in range(1, temporal_ordinal + 1):
                j = i + delta
                if j >= len(ordered):
                    break
                right = ordered[j]
                pair = (left, right) if left < right else (right, left)
                temporal.add(pair)

    all_pairs = set(covis) | temporal
    if len(all_pairs) > max_pairs:
        raise ValueError(f"n_pairs={len(all_pairs)} > {max_pairs}")

    rows: list[dict[str, Any]] = []
    for name_a, name_b in sorted(all_pairs):
        in_covis = (name_a, name_b) in covis
        in_temporal = (name_a, name_b) in temporal
        if in_covis and in_temporal:
            reason = "both"
        elif in_covis:
            reason = "covisibility"
        else:
            reason = "temporal"
        rows.append(
            {
                "name_a": name_a,
                "name_b": name_b,
                "reason": reason,
                "shared_points": int(covis.get((name_a, name_b), 0)),
            }
        )
    return rows


def accumulate_cell_keypoints(
    inlier_pairs: Iterable[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
    pose_only: set[str],
    *,
    cell_px: float = CELL_PX,
    keypoint_cap: int = KEYPOINT_CAP,
) -> tuple[dict[str, np.ndarray], dict[str, dict[tuple[int, int], int]], int]:
    cells: dict[str, dict[tuple[int, int], tuple[float, float, float]]] = defaultdict(dict)
    for name_a, name_b, mkpts0, mkpts1, mconf in inlier_pairs:
        if name_a in pose_only or name_b in pose_only:
            continue
        for xy, conf in zip(mkpts0, mconf, strict=True):
            key = quantize_xy(xy, cell=cell_px)
            prev = cells[name_a].get(key)
            if prev is None or float(conf) > prev[0]:
                cells[name_a][key] = (float(conf), float(xy[0]), float(xy[1]))
        for xy, conf in zip(mkpts1, mconf, strict=True):
            key = quantize_xy(xy, cell=cell_px)
            prev = cells[name_b].get(key)
            if prev is None or float(conf) > prev[0]:
                cells[name_b][key] = (float(conf), float(xy[0]), float(xy[1]))

    discarded_cells = 0
    keypoints: dict[str, np.ndarray] = {}
    index_of: dict[str, dict[tuple[int, int], int]] = {}
    for name, grid in cells.items():
        items = list(grid.items())
        if len(items) > keypoint_cap:
            items.sort(key=lambda item: -item[1][0])
            discarded_cells += len(items) - keypoint_cap
            items = items[:keypoint_cap]
        items.sort(key=lambda item: (item[0][1], item[0][0]))
        xy = np.asarray([[row[1][1], row[1][2]] for row in items], dtype=np.float32)
        keypoints[name] = xy
        index_of[name] = {cell: i for i, (cell, _value) in enumerate(items)}
    return keypoints, index_of, discarded_cells


def match_rows_for_pair(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    index_a: Mapping[tuple[int, int], int],
    index_b: Mapping[tuple[int, int], int],
    *,
    cell_px: float = CELL_PX,
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for xy0, xy1 in zip(mkpts0, mkpts1, strict=True):
        ia = index_a.get(quantize_xy(xy0, cell=cell_px))
        ib = index_b.get(quantize_xy(xy1, cell=cell_px))
        if ia is None or ib is None:
            continue
        pair = (ia, ib)
        if pair in seen:
            continue
        seen.add(pair)
        rows.append(pair)
    if not rows:
        return np.zeros((0, 2), dtype=np.uint32)
    return np.asarray(rows, dtype=np.uint32)


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return "bad_alloc" in text or "out of memory" in text or "std::bad_alloc" in text


def _npz_path(match_dir: Path, name_a: str, name_b: str) -> Path:
    return match_dir / f"{pair_id(name_a, name_b)}.npz"


def _inlier_path(match_dir: Path, name_a: str, name_b: str) -> Path:
    return match_dir / "inliers" / f"{pair_id(name_a, name_b)}.npz"


def _read_keyframes(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(row["output_name"]): row for row in rows}


def write_pairs(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pairs),
        encoding="utf-8",
    )


def load_pairs(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def match_pairs(
    pairs: Sequence[Mapping[str, Any]],
    image_root: Path,
    match_dir: Path,
    matcher: Matcher,
    *,
    resume: bool = True,
) -> int:
    match_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    total = len(pairs)
    for index, row in enumerate(pairs, start=1):
        name_a = str(row["name_a"])
        name_b = str(row["name_b"])
        path = _npz_path(match_dir, name_a, name_b)
        if resume and path.is_file():
            continue
        mkpts0, mkpts1, mconf = matcher(image_root / name_a, image_root / name_b)
        np.savez_compressed(path, mkpts0=mkpts0, mkpts1=mkpts1, mconf=mconf)
        wrote += 1
        if index % 200 == 0 or index == total:
            print(f"edm match {index}/{total} wrote={wrote}", flush=True)
    return wrote


def filter_pair_matches(
    reconstruction: Any,
    pairs: Sequence[Mapping[str, Any]],
    match_dir: Path,
    *,
    min_inliers: int = MIN_INLIERS,
    sampson_max_px: float = SAMPSON_MAX_PX,
    min_angle_deg: float = MIN_TRIANGULATION_ANGLE_DEG,
) -> dict[str, Any]:
    poses: dict[str, np.ndarray] = {}
    ks: dict[str, np.ndarray] = {}
    for image in reconstruction.images.values():
        name = str(image.name)
        poses[name] = w2c_4x4(image)
        ks[name] = k_from_camera(reconstruction.cameras[image.camera_id])

    inlier_root = match_dir / "inliers"
    if inlier_root.exists():
        for stale in inlier_root.glob("*.npz"):
            stale.unlink()
    inlier_root.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    inlier_counts: list[int] = []
    for row in pairs:
        name_a = str(row["name_a"])
        name_b = str(row["name_b"])
        path = _npz_path(match_dir, name_a, name_b)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            mkpts0 = np.asarray(payload["mkpts0"], dtype=np.float64)
            mkpts1 = np.asarray(payload["mkpts1"], dtype=np.float64)
            mconf = np.asarray(payload["mconf"], dtype=np.float64)
        mask = frozen_pose_inliers(
            mkpts0,
            mkpts1,
            poses[name_a],
            poses[name_b],
            ks[name_a],
            ks[name_b],
            sampson_max_px=sampson_max_px,
            min_angle_deg=min_angle_deg,
        )
        n_inliers = int(np.count_nonzero(mask))
        inlier_counts.append(n_inliers)
        if n_inliers < min_inliers:
            dropped += 1
            continue
        kept += 1
        np.savez_compressed(
            _inlier_path(match_dir, name_a, name_b),
            mkpts0=np.asarray(mkpts0[mask], dtype=np.float32),
            mkpts1=np.asarray(mkpts1[mask], dtype=np.float32),
            mconf=np.asarray(mconf[mask], dtype=np.float32),
        )
    counts = np.asarray(inlier_counts, dtype=np.float64)
    summary = {
        "pairs_total": len(pairs),
        "pairs_kept": kept,
        "pairs_dropped": dropped,
        "inlier_count": {
            "p10": _percentile(counts, 10),
            "median": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
        },
        "min_inliers": min_inliers,
        "sampson_max_px": sampson_max_px,
        "min_angle_deg": min_angle_deg,
    }
    write_json(match_dir / "filter_summary.json", summary)
    return summary


def _iter_inlier_pairs(
    pairs: Sequence[Mapping[str, Any]], match_dir: Path
) -> Iterable[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]]:
    for row in pairs:
        name_a = str(row["name_a"])
        name_b = str(row["name_b"])
        path = _inlier_path(match_dir, name_a, name_b)
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as payload:
            yield (
                name_a,
                name_b,
                np.asarray(payload["mkpts0"], dtype=np.float64),
                np.asarray(payload["mkpts1"], dtype=np.float64),
                np.asarray(payload["mconf"], dtype=np.float64),
            )


def build_cell_database(
    reconstruction: Any,
    pairs: Sequence[Mapping[str, Any]],
    match_dir: Path,
    keyframes: Mapping[str, Mapping[str, Any]],
    database_path: Path,
    *,
    cell_px: float = CELL_PX,
    keypoint_cap: int = KEYPOINT_CAP,
) -> dict[str, Any]:
    import pycolmap

    from .densesfm_worker import open_database_compat

    pose_only, _tracks, image_ids = pose_only_and_tracks(reconstruction, keyframes)
    keypoints, index_of, discarded_cells = accumulate_cell_keypoints(
        _iter_inlier_pairs(pairs, match_dir),
        pose_only,
        cell_px=cell_px,
        keypoint_cap=keypoint_cap,
    )
    if database_path.exists():
        database_path.unlink()
    database = open_database_compat(pycolmap.Database, database_path)
    n_match_pairs = 0
    n_match_rows = 0
    try:
        for camera_id in sorted(reconstruction.cameras):
            database.write_camera(reconstruction.cameras[camera_id], use_camera_id=True)
        for rig_id in sorted(reconstruction.rigs):
            database.write_rig(reconstruction.rigs[rig_id], use_rig_id=True)
        for frame_id in sorted(reconstruction.frames):
            database.write_frame(reconstruction.frames[frame_id], use_frame_id=True)
        for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
            image = reconstruction.images[image_id]
            database.write_image(image, use_image_id=True)
            name = str(image.name)
            points = keypoints.get(name, np.zeros((0, 2), dtype=np.float32))
            if name in pose_only:
                points = np.zeros((0, 2), dtype=np.float32)
            database.write_keypoints(image_id, points)
        for name_a, name_b, mkpts0, mkpts1, _mconf in _iter_inlier_pairs(pairs, match_dir):
            if name_a in pose_only or name_b in pose_only:
                continue
            id_a = image_ids[name_a]
            id_b = image_ids[name_b]
            values = match_rows_for_pair(
                mkpts0, mkpts1, index_of.get(name_a, {}), index_of.get(name_b, {}), cell_px=cell_px
            )
            if values.size == 0:
                continue
            left, right = id_a, id_b
            if left > right:
                left, right = right, left
                values = values[:, ::-1]
            database.write_matches(left, right, values)
            geometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = values
            database.write_two_view_geometry(left, right, geometry)
            n_match_pairs += 1
            n_match_rows += int(len(values))
    finally:
        database.close()
    n_keypoints = [len(xy) for name, xy in keypoints.items() if name not in pose_only]
    counts = np.asarray(n_keypoints or [0], dtype=np.float64)
    return {
        "database": str(database_path),
        "keypoint_cap": keypoint_cap,
        "cell_px": cell_px,
        "n_images_with_keypoints": len(n_keypoints),
        "n_pose_only": len(pose_only),
        "discarded_cells": discarded_cells,
        "keypoint_count": {
            "p10": _percentile(counts, 10),
            "median": _percentile(counts, 50),
            "p90": _percentile(counts, 90),
            "max": int(counts.max()) if counts.size else 0,
        },
        "n_match_pairs": n_match_pairs,
        "n_match_rows": n_match_rows,
    }


def retriangulate_fixed_poses(
    *,
    input_model: Path,
    database: Path,
    image_root: Path,
    output_model: Path,
    minimum_angle_deg: float = MIN_TRIANGULATION_ANGLE_DEG,
    ignore_two_view_tracks: bool = False,
) -> dict[str, Any]:
    import pycolmap
    from river_v4_optimizer.runner import analyze_model, atomic_json

    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(input_model.resolve(strict=True)))
    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.triangulation.min_angle = float(minimum_angle_deg)
    options.triangulation.ignore_two_view_tracks = bool(ignore_two_view_tracks)
    triangulation = options.triangulation
    if hasattr(triangulation, "complete_max_reproj_error"):
        triangulation.complete_max_reproj_error = 3.0
    if hasattr(triangulation, "merge_max_reproj_error"):
        triangulation.merge_max_reproj_error = 3.0
    result = pycolmap.triangulate_points(
        reconstruction,
        database.resolve(strict=True),
        image_root.resolve(strict=True),
        output,
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )
    metrics = analyze_model(output)
    receipt = {
        "schema_version": 1,
        "artifact_type": "EDM_FIXED_POSE_RETRIANGULATION",
        "input_model": str(input_model.resolve()),
        "database": str(database.resolve()),
        "image_root": str(image_root.resolve()),
        "output_model": str(output),
        "minimum_angle_deg": minimum_angle_deg,
        "ignore_two_view_tracks": ignore_two_view_tracks,
        "refine_intrinsics": False,
        "complete_max_reproj_error": getattr(triangulation, "complete_max_reproj_error", None),
        "merge_max_reproj_error": getattr(triangulation, "merge_max_reproj_error", None),
        "returned_registered_images": result.num_reg_images(),
        **metrics,
    }
    atomic_json(output.parent / "retriangulation_receipt.json", receipt)
    return receipt


def official_edm_matcher(
    *,
    edm_root: Path,
    edm_checkpoint: Path,
    model_config: Path,
    data_config: Path,
    device: str = "cuda",
) -> Matcher:
    from river_map_quality.official_edm_adapter_loo import prepare_official_megadepth_image

    from .deployment_localizer import (
        _configure_audited_runtime_site_packages,
        _match_official_prepared,
        _prepare_official_image,
        _resolve_audited_site_packages,
        _valid_matches,
    )

    _configure_audited_runtime_site_packages(_resolve_audited_site_packages(None))
    from river_map_quality.official_edm_adapter_loo import load_official_edm_runtime

    runtime = load_official_edm_runtime(
        edm_repo=Path(edm_root),
        checkpoint=Path(edm_checkpoint),
        config_path=Path(model_config),
        data_config_path=Path(data_config),
        device=device,
    )
    prepared: dict[Path, Any] = {}
    shapes: dict[Path, tuple[int, int]] = {}

    def _prepare(path: Path):
        import cv2

        resolved = path.resolve()
        cached = prepared.get(resolved)
        if cached is not None:
            return cached
        gray = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(resolved)
        image = _prepare_official_image(
            resolved, runtime, prepare_image=prepare_official_megadepth_image, image=gray
        )
        prepared[resolved] = image
        shapes[resolved] = gray.shape
        return image

    def match(path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resolved_a = path_a.resolve()
        resolved_b = path_b.resolve()
        payload = _match_official_prepared(runtime, _prepare(resolved_a), _prepare(resolved_b))
        mkpts0 = np.asarray(payload.get("mkpts0_f", payload.get("mkpts0")), dtype=np.float64)
        mkpts1 = np.asarray(payload.get("mkpts1_f", payload.get("mkpts1")), dtype=np.float64)
        mconf = np.asarray(payload["mconf"], dtype=np.float64)
        valid = _valid_matches(
            mkpts0,
            mkpts1,
            mconf,
            query_shape=shapes[resolved_a],
            reference_shape=shapes[resolved_b],
            confidence_threshold=0.0,
        )
        return mkpts0[valid], mkpts1[valid], mconf[valid]

    match.runtime = runtime  # type: ignore[attr-defined]
    return match


def run_edm_retriangulation(
    *,
    pose_model: Path,
    image_root: Path,
    keyframes_path: Path,
    work_dir: Path,
    output_model: Path,
    matcher: Matcher,
    cell_px: float = CELL_PX,
    keypoint_cap: int = KEYPOINT_CAP,
    resume: bool = True,
) -> dict[str, Any]:
    import pycolmap

    pose_model = pose_model.resolve(strict=True)
    image_root = image_root.resolve(strict=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = pycolmap.Reconstruction(str(pose_model))
    keyframes = _read_keyframes(keyframes_path)
    pose_only, tracks, _image_ids = pose_only_and_tracks(reconstruction, keyframes)
    pairs_path = work_dir / "pairs.jsonl"
    if resume and pairs_path.is_file():
        pairs = load_pairs(pairs_path)
    else:
        pairs = select_retriangulation_pairs(tracks, keyframes)
        write_pairs(pairs_path, pairs)
    pair_summary = {
        "n_pairs": len(pairs),
        "n_images": int(reconstruction.num_reg_images()),
        "n_pose_only": len(pose_only),
        "n_covisibility": sum(1 for row in pairs if row["reason"] in {"covisibility", "both"}),
        "n_temporal": sum(1 for row in pairs if row["reason"] in {"temporal", "both"}),
        "pose_only": sorted(pose_only),
    }
    write_json(work_dir / "pairs_summary.json", pair_summary)

    match_dir = work_dir / "matches"
    match_pairs(pairs, image_root, match_dir, matcher, resume=resume)
    filter_summary = filter_pair_matches(reconstruction, pairs, match_dir)
    database_path = work_dir / "database.db"

    def _build(cap: int) -> dict[str, Any]:
        return build_cell_database(
            reconstruction,
            pairs,
            match_dir,
            keyframes,
            database_path,
            cell_px=cell_px,
            keypoint_cap=cap,
        )

    try:
        database_receipt = _build(keypoint_cap)
        used_cap = keypoint_cap
    except Exception as exc:
        if not _is_oom(exc) or keypoint_cap <= KEYPOINT_CAP_OOM:
            raise
        database_receipt = _build(KEYPOINT_CAP_OOM)
        used_cap = KEYPOINT_CAP_OOM
        database_receipt["oom_fallback_keypoint_cap"] = KEYPOINT_CAP_OOM

    if output_model.exists():
        raise FileExistsError(output_model)
    triangulation = retriangulate_fixed_poses(
        input_model=pose_model,
        database=database_path,
        image_root=image_root,
        output_model=output_model,
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "ADOPTED_EDM_RETRIANGULATION",
        "cell_px": cell_px,
        "keypoint_cap": used_cap,
        "lift_distance_px": LIFT_DISTANCE_PX,
        "pairs": pair_summary,
        "filter": filter_summary,
        "database": database_receipt,
        "triangulation": triangulation,
        "output_model": str(output_model.resolve()),
    }
    write_json(work_dir / "edm_retriangulation_receipt.json", receipt)
    return receipt



__all__ = [
    "CELL_PX",
    "KEYPOINT_CAP",
    "LIFT_DISTANCE_PX",
    "accumulate_cell_keypoints",
    "filter_pair_matches",
    "frozen_pose_inliers",
    "match_rows_for_pair",
    "official_edm_matcher",
    "quantize_xy",
    "run_edm_retriangulation",
    "select_retriangulation_pairs",
]
