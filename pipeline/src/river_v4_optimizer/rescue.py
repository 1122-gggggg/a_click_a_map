from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .metrics import analyze_model
from .runner import atomic_json


def maximum_triangulation_angle_deg(camera_centers: np.ndarray, xyz: np.ndarray) -> float:
    centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    point = np.asarray(xyz, dtype=np.float64).reshape(3)
    if len(centers) < 2:
        return 0.0
    rays = point[None] - centers
    norms = np.linalg.norm(rays, axis=1)
    valid = norms > 1e-12
    rays = rays[valid] / norms[valid, None]
    maximum = 0.0
    for index, left in enumerate(rays):
        for right in rays[index + 1 :]:
            angle = math.degrees(math.acos(float(np.clip(np.dot(left, right), -1.0, 1.0))))
            maximum = max(maximum, angle)
    return maximum


def connector_quality_ok(
    *,
    component_ids: set[int],
    reprojection_errors: Iterable[float],
    triangulation_angle_deg: float,
    maximum_reprojection_px: float = 3.0,
    maximum_reprojection_p90_px: float = 2.0,
    minimum_triangulation_angle_deg: float = 1.0,
) -> bool:
    errors = np.asarray(list(reprojection_errors), dtype=np.float64)
    return bool(
        len(component_ids) >= 2
        and len(errors) >= 3
        and np.isfinite(errors).all()
        and float(errors.max()) <= maximum_reprojection_px
        and float(np.percentile(errors, 90)) <= maximum_reprojection_p90_px
        and triangulation_angle_deg >= minimum_triangulation_angle_deg
    )


def triangulate_connector(
    points2d: np.ndarray,
    cams_from_world: list[Any],
    cameras: list[Any],
) -> dict[str, Any] | None:
    import pycolmap

    options = pycolmap.EstimateTriangulationOptions()
    options.min_tri_angle = math.radians(1.0)
    options.ransac.max_error = math.radians(2.0)
    options.ransac.random_seed = 0
    return pycolmap.estimate_triangulation(points2d, cams_from_world, cameras, options)


def _components(reconstruction: Any, threshold: int) -> dict[int, int]:
    names = {
        int(image.image_id): str(image.name)
        for image in reconstruction.images.values()
        if image.has_pose
    }
    shared: Counter[tuple[int, int]] = Counter()
    for point in reconstruction.points3D.values():
        image_ids = sorted(
            {
                int(element.image_id)
                for element in point.track.elements
                if int(element.image_id) in names
            }
        )
        for offset, left in enumerate(image_ids):
            for right in image_ids[offset + 1 :]:
                shared[(left, right)] += 1
    graph = nx.Graph()
    graph.add_nodes_from(names)
    graph.add_edges_from(pair for pair, count in shared.items() if count >= threshold)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    return {
        int(image_id): index
        for index, component in enumerate(components)
        for image_id in component  # ty: ignore[not-iterable]
    }


def _camera_center(image: Any) -> np.ndarray:
    transform = image.cam_from_world()
    rotation = transform.rotation.matrix()
    return -rotation.T @ np.asarray(transform.translation, dtype=np.float64)


def rescue_connectors(
    *,
    dense_model: Path,
    base_model: Path,
    output_model: Path,
    shared_landmark_threshold: int = 15,
) -> dict[str, Any]:
    import pycolmap
    from sfm_diagnosis.site_pipeline.robust_filter import (  # ty: ignore[unresolved-import]
        RobustFilterConfig,
        filter_reconstruction,
        run_fixed_intrinsics_ba,
    )

    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    dense = pycolmap.Reconstruction(str(dense_model.resolve(strict=True)))
    reconstruction = pycolmap.Reconstruction(str(base_model.resolve(strict=True)))
    components = _components(reconstruction, shared_landmark_threshold)
    dense_images = {int(image.image_id): image for image in dense.images.values() if image.has_pose}
    base_images = {
        int(image.image_id): image for image in reconstruction.images.values() if image.has_pose
    }
    existing_ids = {int(value) for value in reconstruction.points3D}
    candidates = 0
    admitted = 0
    rejected_conflict = 0
    rejected_quality = 0
    rescued_observations = 0
    for point_id, dense_point in dense.points3D.items():
        if int(point_id) in existing_ids:
            continue
        observations = []
        points2d = []
        transforms = []
        cameras = []
        observation_components = []
        component_ids = set()
        for element in dense_point.track.elements:
            image_id = int(element.image_id)
            point2d_idx = int(element.point2D_idx)
            image = base_images.get(image_id)
            dense_image = dense_images.get(image_id)
            if image is None or dense_image is None or point2d_idx >= len(image.points2D):
                continue
            if image.points2D[point2d_idx].has_point3D():
                continue
            observed = np.asarray(dense_image.points2D[point2d_idx].xy, dtype=np.float64)
            observations.append((image_id, point2d_idx))
            points2d.append(observed)
            transforms.append(image.cam_from_world())
            cameras.append(image.camera)
            observation_components.append(int(components[image_id]))
            component_ids.add(int(components[image_id]))
        if len(component_ids) < 2 or len(observations) < 3:
            continue
        candidates += 1
        triangulated = triangulate_connector(np.asarray(points2d), transforms, cameras)
        if triangulated is None:
            rejected_quality += 1
            continue
        inliers = np.asarray(triangulated["inliers"], dtype=bool)
        xyz = np.asarray(triangulated["xyz"], dtype=np.float64)
        selected_observations = [
            row for row, keep in zip(observations, inliers, strict=True) if keep
        ]
        selected_points2d = [row for row, keep in zip(points2d, inliers, strict=True) if keep]
        selected_components = {
            row for row, keep in zip(observation_components, inliers, strict=True) if keep
        }
        centers = [_camera_center(base_images[image_id]) for image_id, _ in selected_observations]
        errors = []
        for (image_id, _), observed in zip(selected_observations, selected_points2d, strict=True):
            projected = base_images[image_id].project_point(xyz)
            if projected is None:
                errors.append(math.inf)
            else:
                errors.append(float(np.linalg.norm(np.asarray(projected) - observed)))
        angle = maximum_triangulation_angle_deg(np.asarray(centers), xyz)
        if not connector_quality_ok(
            component_ids=selected_components,
            reprojection_errors=errors,
            triangulation_angle_deg=angle,
        ):
            rejected_quality += 1
            continue
        track = pycolmap.Track()
        for image_id, point2d_idx in selected_observations:
            track.add_element(image_id, point2d_idx)
        try:
            reconstruction.add_point3D(xyz, track, dense_point.color)
        except ValueError:
            rejected_conflict += 1
            continue
        admitted += 1
        rescued_observations += len(selected_observations)
    if admitted == 0:
        raise RuntimeError("no connector tracks passed the rescue gate")
    config = RobustFilterConfig(
        max_reprojection_error_px=3.0,
        minimum_triangulation_angle_deg=1.0,
        minimum_track_length=3,
        bundle_adjustment_iterations=100,
        bundle_adjustment_threads=8,
    )
    ba_seconds = run_fixed_intrinsics_ba(reconstruction, config)
    filter_receipt = filter_reconstruction(reconstruction, config)
    output.mkdir(parents=True)
    reconstruction.write(output)
    metrics = analyze_model(output)
    receipt = {
        "schema_version": 1,
        "artifact_type": "RIVER_V4_CONNECTIVITY_RESCUE",
        "dense_model": str(dense_model.resolve()),
        "base_model": str(base_model.resolve()),
        "output_model": str(output),
        "shared_landmark_threshold": shared_landmark_threshold,
        "candidate_cross_component_tracks": candidates,
        "admitted_connector_tracks": admitted,
        "rescued_observations": rescued_observations,
        "rejected_quality": rejected_quality,
        "rejected_conflict": rejected_conflict,
        "bundle_adjustment_seconds": ba_seconds,
        "post_ba_filter": filter_receipt,
        **metrics,
    }
    atomic_json(output.parent / "rescue_receipt.json", receipt)
    return receipt
