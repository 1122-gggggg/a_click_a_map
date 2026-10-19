from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .models import CameraIntrinsics, MapData, Pose


@dataclass(frozen=True)
class VisibilityResult:
    point_indices: np.ndarray
    camera_points: np.ndarray
    distances: np.ndarray
    uv: np.ndarray
    geometric_weights: np.ndarray


class OcclusionModel(Protocol):
    def unoccluded(self, origin_w: np.ndarray, targets_w: np.ndarray) -> np.ndarray: ...


class IlluminationModel(Protocol):
    def weights(self, points_w: np.ndarray) -> np.ndarray: ...


class MeshRaycaster:
    """Optional triangle-mesh ray casting used for LIDIA-style occlusion reasoning."""

    def __init__(self, mesh_path: str | Path, epsilon: float = 1e-3):
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the mesh extra: pip install -e '.[mesh]'") from exc
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        if hasattr(mesh, "geometry") and not hasattr(mesh, "faces"):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        self.mesh = mesh
        self.epsilon = float(epsilon)

    def unoccluded(self, origin_w: np.ndarray, targets_w: np.ndarray) -> np.ndarray:
        targets = np.asarray(targets_w, dtype=float).reshape(-1, 3)
        if len(targets) == 0:
            return np.zeros(0, dtype=bool)
        origin = np.asarray(origin_w, dtype=float).reshape(3)
        delta = targets - origin
        distances = np.linalg.norm(delta, axis=1)
        valid = distances > self.epsilon
        directions = np.zeros_like(delta)
        directions[valid] = delta[valid] / distances[valid, None]
        origins = np.repeat(origin[None, :], len(targets), axis=0)
        visible = valid.copy()
        if not np.any(valid):
            return visible
        locations, ray_ids, _ = self.mesh.ray.intersects_location(
            origins[valid], directions[valid], multiple_hits=False
        )
        valid_indices = np.flatnonzero(valid)
        for hit, local_ray_id in zip(locations, ray_ids):
            global_idx = valid_indices[int(local_ray_id)]
            hit_distance = float(np.linalg.norm(hit - origin))
            if hit_distance < distances[global_idx] - self.epsilon:
                visible[global_idx] = False
        return visible

    def ray_clear(self, origins_w: np.ndarray, directions_w: np.ndarray, max_distance: float = 1e6) -> np.ndarray:
        origins = np.asarray(origins_w, dtype=float).reshape(-1, 3)
        directions = np.asarray(directions_w, dtype=float).reshape(-1, 3)
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 1e-9
        unit = np.zeros_like(directions)
        unit[valid] = directions[valid] / norms[valid, None]
        clear = valid.copy()
        if np.any(valid):
            locations, ray_ids, _ = self.mesh.ray.intersects_location(
                origins[valid] + unit[valid] * self.epsilon,
                unit[valid],
                multiple_hits=False,
            )
            valid_indices = np.flatnonzero(valid)
            for hit, local_ray_id in zip(locations, ray_ids):
                global_idx = valid_indices[int(local_ray_id)]
                dist = float(np.linalg.norm(hit - origins[global_idx]))
                if dist < max_distance:
                    clear[global_idx] = False
        return clear


@dataclass
class SunIllumination:
    """Binary direct-sun illumination mask.

    `light_travel_direction_w` points in the direction sunlight travels (sun -> scene).
    A landmark is sunlit when a ray from the landmark toward the sun, i.e. the negative
    direction, does not intersect the mesh.
    """

    light_travel_direction_w: np.ndarray
    raycaster: MeshRaycaster | None = None

    def weights(self, points_w: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, dtype=float)
        if self.raycaster is None:
            return np.ones(len(points), dtype=float)
        d = np.asarray(self.light_travel_direction_w, dtype=float).reshape(3)
        d /= max(np.linalg.norm(d), 1e-12)
        dirs = np.repeat((-d)[None, :], len(points), axis=0)
        return self.raycaster.ray_clear(points, dirs).astype(float)


@dataclass
class SpotIllumination:
    position_w: np.ndarray
    direction_w: np.ndarray
    half_angle_deg: float
    raycaster: MeshRaycaster | None = None

    def weights(self, points_w: np.ndarray) -> np.ndarray:
        points = np.asarray(points_w, dtype=float).reshape(-1, 3)
        if len(points) == 0:
            return np.zeros(0, dtype=float)
        pos = np.asarray(self.position_w, dtype=float).reshape(3)
        direction = np.asarray(self.direction_w, dtype=float).reshape(3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        rays = points - pos
        dist = np.linalg.norm(rays, axis=1)
        unit = np.zeros_like(rays)
        valid = dist > 1e-9
        unit[valid] = rays[valid] / dist[valid, None]
        cone = (unit @ direction) >= np.cos(np.radians(self.half_angle_deg))
        if self.raycaster is not None and np.any(cone):
            clear = self.raycaster.unoccluded(pos, points)
            cone &= clear
        return cone.astype(float)


def visible_points(
    map_data: MapData,
    pose: Pose,
    intrinsics: CameraIntrinsics | None = None,
    *,
    max_distance: float | None = None,
    min_depth: float = 0.05,
    occlusion: OcclusionModel | None = None,
) -> VisibilityResult:
    """Select map landmarks inside the candidate camera frustum.

    Projection uses a pinhole approximation derived from COLMAP intrinsics. Distortion
    is intentionally ignored for map-health scoring; very wide/fisheye cameras should
    use a custom visibility provider.
    """
    intr = intrinsics or map_data.median_intrinsics
    if max_distance is None:
        candidates = np.arange(map_data.num_points, dtype=int)
    else:
        candidates = np.asarray(
            sorted(
                map_data.point_tree().query_ball_point(
                    pose.center_w, float(max_distance)
                )
            ),
            dtype=int,
        )
    candidate_points = map_data.points_xyz[candidates]
    pc = pose.world_to_camera(candidate_points)
    z = pc[:, 2]
    dist = np.linalg.norm(pc, axis=1)
    mask = z > min_depth

    u = intr.fx * pc[:, 0] / np.maximum(z, 1e-12) + intr.cx
    v = intr.fy * pc[:, 1] / np.maximum(z, 1e-12) + intr.cy
    mask &= (u >= 0.0) & (u < intr.width) & (v >= 0.0) & (v < intr.height)
    local_indices = np.flatnonzero(mask)
    indices = candidates[local_indices]

    if occlusion is not None and len(indices):
        keep = occlusion.unoccluded(pose.center_w, map_data.points_xyz[indices])
        indices = indices[keep]
        local_indices = local_indices[keep]

    uv = (
        np.column_stack((u[local_indices], v[local_indices]))
        if len(indices)
        else np.empty((0, 2))
    )
    return VisibilityResult(
        point_indices=indices,
        camera_points=pc[local_indices],
        distances=dist[local_indices],
        uv=uv,
        geometric_weights=np.ones(len(indices), dtype=float),
    )


def image_grid_occupancy(uv: np.ndarray, intrinsics: CameraIntrinsics, rows: int = 4, cols: int = 4) -> int:
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if len(uv) == 0:
        return 0
    c = np.clip((uv[:, 0] / max(intrinsics.width, 1) * cols).astype(int), 0, cols - 1)
    r = np.clip((uv[:, 1] / max(intrinsics.height, 1) * rows).astype(int), 0, rows - 1)
    return len({(int(rr), int(cc)) for rr, cc in zip(r, c)})


def normalized_convex_hull_area(uv: np.ndarray, intrinsics: CameraIntrinsics) -> float:
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if len(uv) < 3:
        return 0.0
    try:
        area = float(ConvexHull(uv).volume)  # 2D hull: `volume` is area.
    except QhullError:
        return 0.0
    return float(np.clip(area / max(intrinsics.width * intrinsics.height, 1), 0.0, 1.0))


def sun_light_travel_direction_from_az_el(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Return a Z-up light-travel direction for a sun azimuth/elevation.

    Azimuth is measured counter-clockwise in the world XY plane from +X. The returned
    vector points from the sun toward the scene, which is the convention expected by
    SunIllumination. Use only after the SfM map has been aligned to a meaningful Z-up frame.
    """
    az = np.radians(float(azimuth_deg))
    el = np.radians(float(elevation_deg))
    scene_to_sun = np.array(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=float
    )
    return -scene_to_sun / max(np.linalg.norm(scene_to_sun), 1e-12)
