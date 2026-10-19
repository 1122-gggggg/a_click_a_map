from __future__ import annotations

import csv
import json
import math
import pickle
from functools import cache
from pathlib import Path
from typing import Any

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.metrics import convex_hull_area_fraction, grid_coverage


@cache
def _dependencies():
    try:
        import numpy as np
        import pycolmap
    except ImportError as exc:
        raise RuntimeError(
            "The hloc exporter requires optional dependencies. Install with: "
            "pip install 'mapdoctor-sfm[hloc]'"
        ) from exc
    return np, pycolmap


def _coverage(points: Any, width: int, height: int) -> tuple[float, int]:
    if len(points) == 0 or width <= 0 or height <= 0:
        return 0.0, 0
    hull = convex_hull_area_fraction(points, width, height)
    occupied, _ = grid_coverage(points, width, height, rows=4, cols=4)
    return hull, occupied


def _pose_matrix(ret: dict[str, Any]):
    np, _ = _dependencies()
    pose = ret.get("cam_from_world")
    if pose is None:
        raise ValueError("hloc PnP result is missing cam_from_world")
    matrix = pose.matrix() if callable(getattr(pose, "matrix", None)) else pose
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 4):
        raise ValueError(f"Expected a 3x4 cam_from_world matrix, got {matrix.shape}")
    return matrix


def _camera_center(matrix: Any):
    rotation = matrix[:, :3]
    translation = matrix[:, 3]
    return -(rotation.T @ translation)


def _rotation_distance_deg(a: Any, b: Any) -> float:
    np, _ = _dependencies()
    relative = a[:, :3] @ b[:, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _inlier_mask(ret: dict[str, Any], count: int):
    np, _ = _dependencies()
    raw = ret.get("inliers")
    if raw is None:
        raise ValueError("hloc PnP result is missing its RANSAC inlier mask")
    values = np.asarray(raw)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == count and values.dtype == np.bool_:
        return values
    if values.size == count and set(np.unique(values)).issubset({0, 1, False, True}):
        return values.astype(bool)
    if np.issubdtype(values.dtype, np.integer):
        mask = np.zeros(count, dtype=bool)
        if values.size and (values.min() < 0 or values.max() >= count):
            raise ValueError("hloc inlier indices are outside the correspondence range")
        mask[values.astype(int)] = True
        return mask
    raise ValueError("Unsupported hloc RANSAC inlier representation")


def _selected_log(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if entry.get("covisibility_clustering"):
        clusters = list(entry.get("log_clusters", []))
        best = entry.get("best_cluster")
        if best is None:
            return None, clusters
        best_index = int(best)
        if best_index < 0 or best_index >= len(clusters):
            raise ValueError("hloc best_cluster index is outside log_clusters")
        return clusters[best_index], clusters
    return entry, [entry]


def _scene_extent(reconstruction: Any) -> float:
    np, _ = _dependencies()
    xyz = [np.asarray(point.xyz, dtype=float) for point in reconstruction.points3D.values()]
    if not xyz:
        return 1.0
    points = np.stack(xyz)
    extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    return extent if math.isfinite(extent) and extent > 0 else 1.0


def _point3d(reconstruction: Any, point_id: int):
    points = reconstruction.points3D
    getter = getattr(points, "get", None)
    if callable(getter):
        return getter(point_id)
    try:
        return points[point_id]
    except (KeyError, IndexError):
        return None


def _pose_consensus(
    selected: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    *,
    max_translation: float,
    max_rotation_deg: float,
) -> float:
    np, _ = _dependencies()
    selected_ret = selected.get("PnP_ret")
    if selected_ret is None:
        return 0.0
    selected_matrix = _pose_matrix(selected_ret)
    selected_center = _camera_center(selected_matrix)
    successful = [item.get("PnP_ret") for item in hypotheses if item.get("PnP_ret") is not None]
    if not successful:
        return 0.0
    agreeing = 0
    for ret in successful:
        matrix = _pose_matrix(ret)
        center_distance = float(np.linalg.norm(_camera_center(matrix) - selected_center))
        rotation_distance = _rotation_distance_deg(matrix, selected_matrix)
        if center_distance <= max_translation and rotation_distance <= max_rotation_deg:
            agreeing += 1
    return agreeing / len(successful)


def _failed_result(query: str) -> QueryLocalizationResult:
    return QueryLocalizationResult(
        query=query,
        success=False,
        inliers=0,
        inlier_ratio=0.0,
        reproj_p90_px=None,
        hull_coverage=0.0,
        grid4_occupancy=0,
        positive_depth_ratio=0.0,
        pose_consensus=0.0,
    )


def _query_result(
    query: str,
    selected: dict[str, Any] | None,
    hypotheses: list[dict[str, Any]],
    reconstruction: Any,
    *,
    max_translation: float,
    max_rotation_deg: float,
) -> QueryLocalizationResult:
    np, _ = _dependencies()
    if selected is None or selected.get("PnP_ret") is None:
        return _failed_result(query)

    ret = selected["PnP_ret"]
    keypoints = np.asarray(selected.get("keypoints_query", []), dtype=float)
    point_ids = list(selected.get("points3D_ids", []))
    correspondence_count = len(point_ids)
    if keypoints.shape != (correspondence_count, 2):
        raise ValueError(
            f"{query}: hloc keypoints_query shape {keypoints.shape} does not match "
            f"{correspondence_count} point3D IDs"
        )
    inlier_mask = _inlier_mask(ret, correspondence_count)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / max(1, correspondence_count)

    camera = ret.get("camera")
    matrix = _pose_matrix(ret)
    rotation = matrix[:, :3]
    translation = matrix[:, 3]

    world_points = []
    valid = []
    for index, point_id in enumerate(point_ids):
        point = _point3d(reconstruction, int(point_id))
        if point is not None:
            world_points.append(np.asarray(point.xyz, dtype=float))
            valid.append(index)
    valid_indices = np.asarray(valid, dtype=int)

    reprojection_p90: float | None = None
    positive_depth_ratio = 0.0
    if valid and inlier_count:
        world = np.stack(world_points)
        camera_points = world @ rotation.T + translation
        valid_inliers = inlier_mask[valid_indices]
        if valid_inliers.any():
            inlier_camera_points = camera_points[valid_inliers]
            positive_depth_ratio = float(np.mean(inlier_camera_points[:, 2] > 0))
            if camera is not None:
                projected = np.asarray(camera.img_from_cam(inlier_camera_points), dtype=float)
                observed = keypoints[valid_indices][valid_inliers]
                if projected.shape == observed.shape:
                    errors = np.linalg.norm(projected - observed, axis=1)
                    errors = errors[np.isfinite(errors)]
                    if errors.size:
                        reprojection_p90 = float(np.percentile(errors, 90))

    inlier_keypoints = keypoints[inlier_mask]
    if camera is not None:
        hull_coverage, grid_occupancy = _coverage(
            inlier_keypoints,
            int(camera.width),
            int(camera.height),
        )
    else:
        hull_coverage, grid_occupancy = 0.0, 0

    return QueryLocalizationResult(
        query=query,
        success=True,
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        reproj_p90_px=reprojection_p90,
        hull_coverage=hull_coverage,
        grid4_occupancy=grid_occupancy,
        positive_depth_ratio=positive_depth_ratio,
        pose_consensus=_pose_consensus(
            selected,
            hypotheses,
            max_translation=max_translation,
            max_rotation_deg=max_rotation_deg,
        ),
    )


def _query_names(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    names = []
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            names.append(text.split()[0])
    return names


def export_hloc_logs(
    logs_path: str | Path,
    reference_model: str | Path,
    *,
    query_list: str | Path | None = None,
    trust_pickle: bool = False,
    consensus_translation_fraction: float = 0.01,
    consensus_max_rotation_deg: float = 5.0,
) -> list[QueryLocalizationResult]:
    if not trust_pickle:
        raise ValueError(
            "Refusing to unpickle hloc logs without explicit trust. Pass trust_pickle=True "
            "only for logs produced by a trusted hloc run."
        )
    if not 0 < consensus_translation_fraction <= 1:
        raise ValueError("consensus_translation_fraction must be in (0, 1]")
    if not 0 < consensus_max_rotation_deg <= 180:
        raise ValueError("consensus_max_rotation_deg must be in (0, 180]")

    _, pycolmap = _dependencies()
    with Path(logs_path).expanduser().open("rb") as handle:
        logs = pickle.load(handle)  # noqa: S301 - guarded by explicit trust flag
    if not isinstance(logs, dict) or not isinstance(logs.get("loc"), dict):
        raise ValueError("Not a recognized hloc localization log: missing logs['loc']")

    reconstruction = pycolmap.Reconstruction(Path(reference_model).expanduser())
    translation_threshold = _scene_extent(reconstruction) * consensus_translation_fraction
    locations = logs["loc"]
    names = _query_names(query_list) or sorted(locations)
    results = []
    for query in names:
        entry = locations.get(query)
        if entry is None:
            results.append(_failed_result(query))
            continue
        selected, hypotheses = _selected_log(entry)
        results.append(
            _query_result(
                query,
                selected,
                hypotheses,
                reconstruction,
                max_translation=translation_threshold,
                max_rotation_deg=consensus_max_rotation_deg,
            )
        )
    return results


def write_hloc_results(results: list[QueryLocalizationResult], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_dict() for result in results]
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        if not rows:
            raise ValueError("Cannot write an empty hloc export")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError("hloc export output must end in .csv or .json")
    return path
