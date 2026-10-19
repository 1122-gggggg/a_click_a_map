"""Convert MapDoctor models/rows into sfm-diagnosis types without reloading the map."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from sfm_diagnosis.models import CameraIntrinsics, MapData

if TYPE_CHECKING:
    from mapdoctor.benchmark import QueryLocalizationResult
    from mapdoctor.model import Camera, MapModel


_COLMAP_CAMERA_PARAM_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
    "RAD_TAN_THIN_PRISM_FISHEYE": 16,
}


def rotation_from_viewing_direction(direction) -> np.ndarray:
    """Build camera-to-world R_wc whose +Z column matches ``direction``."""
    forward = np.asarray(direction, dtype=float).reshape(3)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-12:
        raise ValueError("viewing direction has near-zero length")
    forward = forward / norm
    world_z = np.array([0.0, 0.0, 1.0])
    up_hint = world_z if abs(float(forward @ world_z)) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up_hint, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.column_stack((right, up, forward))


def _camera_intrinsics(camera: Camera) -> CameraIntrinsics:
    params = tuple(float(v) for v in camera.params)
    model = camera.model.upper()
    required = _COLMAP_CAMERA_PARAM_COUNTS.get(model)
    if required is not None and len(params) < required:
        raise ValueError(
            f"{model} camera {camera.id} needs {required} params, got {len(params)}"
        )
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif len(params) >= 4:
        fx, fy, cx, cy = params[:4]
    elif len(params) == 3:
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        raise ValueError(f"camera {camera.id} has fewer than 3 params")
    return CameraIntrinsics(
        camera_id=int(camera.id),
        model_name=camera.model,
        width=int(camera.width),
        height=int(camera.height),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def map_model_to_map_data(model: MapModel) -> MapData:
    """Convert a MapDoctor ``MapModel`` into sfm-diagnosis ``MapData``."""
    image_ids = []
    image_names = []
    image_camera_ids = []
    image_centers = []
    image_R_wc = []
    for image_id in sorted(model.images):
        image = model.images[image_id]
        image_ids.append(int(image.id))
        image_names.append(image.name)
        image_camera_ids.append(int(image.camera_id))
        image_centers.append(tuple(float(v) for v in image.center))
        image_R_wc.append(rotation_from_viewing_direction(image.viewing_direction))

    point_ids = []
    points_xyz = []
    point_rgb = []
    point_errors = []
    track_lengths = []
    track_image_ids = []
    for point_id in sorted(model.points3d):
        point = model.points3d[point_id]
        point_ids.append(int(point.id))
        points_xyz.append(tuple(float(v) for v in point.xyz))
        point_rgb.append(tuple(int(v) for v in point.rgb))
        point_errors.append(float(point.error))
        track_ids = np.asarray([int(el.image_id) for el in point.track], dtype=np.int64)
        track_image_ids.append(track_ids)
        track_lengths.append(int(len(track_ids)))

    cameras = {
        int(camera_id): _camera_intrinsics(model.cameras[camera_id])
        for camera_id in sorted(model.cameras)
    }
    return MapData(
        point_ids=np.asarray(point_ids, dtype=np.int64),
        points_xyz=np.asarray(points_xyz, dtype=float).reshape(-1, 3),
        point_rgb=np.asarray(point_rgb, dtype=np.uint8).reshape(-1, 3),
        point_errors=np.asarray(point_errors, dtype=float),
        track_lengths=np.asarray(track_lengths, dtype=np.int32),
        track_image_ids=track_image_ids,
        image_ids=np.asarray(image_ids, dtype=np.int64),
        image_names=image_names,
        image_camera_ids=np.asarray(image_camera_ids, dtype=np.int64),
        image_centers=np.asarray(image_centers, dtype=float).reshape(-1, 3),
        image_R_wc=np.asarray(image_R_wc, dtype=float).reshape(-1, 3, 3),
        cameras=cameras,
        metadata={"source": model.source, "format": model.format},
    )


def mapdoctor_rows_to_history_rows(results: list[QueryLocalizationResult]) -> list[dict]:
    """Map MapDoctor localization rows onto sfm-diagnosis history field names."""
    rows: list[dict] = []
    for result in results:
        if result.x is None or result.y is None or result.z is None:
            continue
        if not (math.isfinite(result.x) and math.isfinite(result.y) and math.isfinite(result.z)):
            continue
        row = {
            "query": result.query,
            "x": float(result.x),
            "y": float(result.y),
            "z": float(result.z),
            "success": result.success,
        }
        for source, target, cast in (
            (result.inliers, "pnp_inliers", int),
            (result.inlier_ratio, "inlier_ratio", float),
            (result.grid4_occupancy, "grid_occupancy", int),
            (result.hull_coverage, "hull_coverage", float),
            (result.positive_depth_ratio, "positive_depth_ratio", float),
        ):
            if source is not None:
                row[target] = cast(source)
        if result.reproj_p90_px is not None:
            row["reproj_p90"] = float(result.reproj_p90_px)
        rows.append(row)
    return rows


def nearest_mapping_rotation(map_data: MapData, center) -> np.ndarray:
    """Return ``R_wc`` of the mapping image whose center is nearest ``center``."""
    if map_data.num_images == 0:
        raise ValueError("map has no images")
    query = np.asarray(center, dtype=float).reshape(3)
    distances = np.linalg.norm(map_data.image_centers - query, axis=1)
    return np.asarray(map_data.image_R_wc[int(np.argmin(distances))], dtype=float)
