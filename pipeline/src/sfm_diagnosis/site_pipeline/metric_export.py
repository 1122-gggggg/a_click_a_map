"""Apply an admitted metres-per-map-unit scale to every point and camera of a COLMAP model."""
from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .metric_scale import model_file_hashes
from .moge_metric_worker import sha256_file

SCHEMA = "whole-map-metric-export/v1"


def export_metric_model(source_model: Path, output_dir: Path, meters_per_map_unit: float,
                        sidecar: Path) -> dict[str, Any]:
    import pycolmap

    scale = float(meters_per_map_unit)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("meters_per_map_unit must be finite and positive")
    source = source_model.resolve(strict=True)
    sidecar = sidecar.resolve(strict=True)
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    if (sidecar_data.get("status") != "ESTIMATED_UNVALIDATED"
            or sidecar_data.get("meters_per_map_unit") != meters_per_map_unit
            or sidecar_data.get("map_model_hashes") != model_file_hashes(source)):
        raise ValueError("scale sidecar does not match the admitted scale or source model")
    output = output_dir.resolve()
    native, metric = output / "model_map_units", output / "model_meters"
    if output.exists():
        raise ValueError("Refusing to overwrite an existing map export")
    native.mkdir(parents=True)
    metric.mkdir()
    for filename in model_file_hashes(source):
        shutil.copy2(source / filename, native / filename)

    rec = pycolmap.Reconstruction(str(source))
    point_ids = np.fromiter(rec.points3D.keys(), dtype=np.uint64, count=rec.num_points3D())
    original_xyz = np.empty((len(point_ids), 3), dtype=np.float64)
    for index, point_id in enumerate(point_ids):
        original_xyz[index] = rec.points3D[int(point_id)].xyz
    image_ids = sorted(rec.images)
    original_poses = np.asarray([rec.images[i].cam_from_world().matrix() for i in image_ids])
    original_camera_params = {i: rec.cameras[i].params.copy() for i in rec.cameras}
    original_errors = np.fromiter((rec.points3D[int(i)].error for i in point_ids), dtype=np.float64)
    n_images, n_points = rec.num_reg_images(), rec.num_points3D()
    # Scale translations directly: Sim3 composition renormalizes a source
    # quaternion even on observation-free frames, violating frozen rotations.
    for point in rec.points3D.values():
        point.xyz = np.asarray(point.xyz) * scale
    for frame in rec.frames.values():
        pose = frame.rig_from_world
        frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, np.asarray(pose.translation) * scale)
    for rig in rec.rigs.values():
        for sensor_id in rig.non_ref_sensors:
            pose = rig.sensor_from_rig(sensor_id)
            if pose is not None:
                rig.set_sensor_from_rig(sensor_id, pycolmap.Rigid3d(
                    pose.rotation, np.asarray(pose.translation) * scale))
    rec.write(str(metric))
    rec.export_PLY(str(output / "map_meters.ply"))
    del rec
    gc.collect()

    # Re-open the exact published binaries, not just the in-memory transform.
    result = pycolmap.Reconstruction(str(metric))
    if result.num_reg_images() != n_images or result.num_points3D() != n_points:
        raise RuntimeError("metric model image or point count changed")
    max_point_error = 0.0
    for index, point_id in enumerate(point_ids):
        point = result.points3D[int(point_id)]
        expected = original_xyz[index] * scale
        if not (np.isfinite(point.xyz).all() and np.allclose(point.xyz, expected, rtol=1e-12, atol=1e-12)):
            raise RuntimeError(f"point {int(point_id)} differs from the scaled source")
        if point.error != original_errors[index]:
            raise RuntimeError(f"point {int(point_id)} reprojection error changed")
        max_point_error = max(max_point_error, float(np.max(np.abs(point.xyz - expected))))
    max_pose_error = 0.0
    poses = []
    for index, image_id in enumerate(image_ids):
        image = result.images[image_id]
        pose = image.cam_from_world().matrix()
        expected = original_poses[index].copy()
        expected[:, 3] *= scale
        if not np.allclose(pose, expected, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"pose of image {image_id} differs from the scaled source")
        max_pose_error = max(max_pose_error, float(np.max(np.abs(pose - expected))))
        poses.append({"image_id": image_id, "image_name": image.name,
                      "camera_center_m": image.projection_center().tolist(),
                      "cam_from_world_m": pose.tolist()})
    for camera_id, params in original_camera_params.items():
        if not np.array_equal(result.cameras[camera_id].params, params):
            raise RuntimeError(f"camera {camera_id} parameters changed")
    source_bounds = np.stack((original_xyz.min(axis=0), original_xyz.max(axis=0)))
    camera_centers = np.asarray([row["camera_center_m"] for row in poses])
    receipt = {
        "schema": SCHEMA, "status": "completed",
        "scale_status": sidecar_data["status"], "absolute_accuracy": "NO_EXTERNAL_LENGTH_HOLDOUT",
        "meters_per_map_unit": scale, "registered_images": n_images, "points3D": n_points,
        "source_model": str(source), "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
        "native_model_hashes": model_file_hashes(native),
        "metric_model_hashes": model_file_hashes(metric),
        "map_meters_ply_sha256": sha256_file(output / "map_meters.ply"),
        "transform": {"scale": scale, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]},
        "all_points_verified": len(point_ids), "all_poses_verified": len(image_ids),
        "max_point_transform_error_m": max_point_error, "max_pose_transform_error_m": max_pose_error,
        "intrinsics_unchanged": True, "reprojection_errors_unchanged": True,
        "point_bounds_m": (source_bounds * scale).tolist(),
        "camera_center_bounds_m": [camera_centers.min(axis=0).tolist(), camera_centers.max(axis=0).tolist()],
        "warning": ("Estimated metres from learned depth priors; not survey/GNSS georegistration. "
                    "Native localization bundle stays in map units; apply this scale once to positions/distances."),
    }
    (output / "camera_poses_meters.json").write_text(
        json.dumps(poses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "metric_export_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt
