from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MapData, Pose


@dataclass(frozen=True)
class ViewSupportConfig:
    neighbors: int = 10
    orientation_weight: float = 5.0
    min_track_observations: int = 6
    max_observation_angle_deg: float = 45.0
    range_expansion: float = 0.40


@dataclass
class ViewSupportMetrics:
    neighbors_used: int
    neighbor_image_ids: list[int]
    candidate_points: int
    redetectable_points: int
    effective_redetectable_points: float
    paper_raw_score: float
    visible_support_fraction: float
    weighted_visible_support_fraction: float
    observation_angle_p50_deg: float | None
    observation_angle_p90_deg: float | None
    angle_extrapolated_fraction: float
    range_extrapolated_fraction: float
    point_weights: np.ndarray

    def as_dict(self) -> dict:
        return {
            "neighbors_used": self.neighbors_used,
            "neighbor_image_ids": self.neighbor_image_ids,
            "candidate_points": self.candidate_points,
            "redetectable_points": self.redetectable_points,
            "effective_redetectable_points": self.effective_redetectable_points,
            "paper_raw_score": self.paper_raw_score,
            "visible_support_fraction": self.visible_support_fraction,
            "weighted_visible_support_fraction": self.weighted_visible_support_fraction,
            "observation_angle_p50_deg": self.observation_angle_p50_deg,
            "observation_angle_p90_deg": self.observation_angle_p90_deg,
            "angle_extrapolated_fraction": self.angle_extrapolated_fraction,
            "range_extrapolated_fraction": self.range_extrapolated_fraction,
        }


def compute_view_support(
    map_data: MapData,
    pose: Pose,
    visible_point_indices: np.ndarray,
    *,
    config: ViewSupportConfig | None = None,
) -> ViewSupportMetrics:
    cfg = config or ViewSupportConfig()
    visible = np.unique(np.asarray(visible_point_indices, dtype=int).reshape(-1))
    weights = np.zeros(len(visible), dtype=float)
    if map_data.num_images == 0 or len(visible) == 0:
        return _empty(weights)

    position_distances = np.linalg.norm(map_data.image_centers - pose.center_w[None, :], axis=1)
    orientation_angles = _rotation_distances_rad(pose.R_wc, map_data.image_R_wc)
    scale = _camera_spacing_scale(map_data)
    distance = position_distances / scale + cfg.orientation_weight * orientation_angles / np.pi
    k = min(max(cfg.neighbors, 1), map_data.num_images)
    neighbors = np.argsort(distance)[:k]
    neighbor_ids = set(int(map_data.image_ids[i]) for i in neighbors)

    image_lookup = map_data.image_index()
    min_track = min(max(cfg.min_track_observations, 1), map_data.num_images)
    angles: list[float] = []
    range_failures = 0
    angle_failures = 0
    candidates = 0
    redetectable = 0
    raw_score = 0.0

    for out_i, pidx in enumerate(visible):
        observed_ids = [int(v) for v in map_data.track_image_ids[int(pidx)].tolist()]
        local_count = len(neighbor_ids.intersection(observed_ids))
        if local_count == 0:
            continue
        candidates += 1
        registered = [image_lookup[v] for v in observed_ids if v in image_lookup]
        if len(registered) < min_track:
            continue
        centers = map_data.image_centers[np.asarray(registered, dtype=int)]
        rays = map_data.points_xyz[int(pidx)] - centers
        ranges = np.linalg.norm(rays, axis=1)
        valid = ranges > 1e-9
        if not np.any(valid):
            continue
        rays = rays[valid] / ranges[valid, None]
        ranges = ranges[valid]
        query_ray = map_data.points_xyz[int(pidx)] - pose.center_w
        query_range = float(np.linalg.norm(query_ray))
        if query_range <= 1e-9:
            continue
        query_ray /= query_range
        nearest_cos = float(np.max(np.clip(rays @ query_ray, -1.0, 1.0)))
        angle = float(np.degrees(np.arccos(nearest_cos)))
        angles.append(angle)
        angle_ok = angle <= cfg.max_observation_angle_deg
        expansion = float(np.clip(cfg.range_expansion, 0.0, 0.95))
        range_ok = (
            float(np.min(ranges)) * (1.0 - expansion)
            <= query_range
            <= float(np.max(ranges)) * (1.0 + expansion)
        )
        angle_failures += int(not angle_ok)
        range_failures += int(not range_ok)
        if not (angle_ok and range_ok):
            continue
        redetectable += 1
        raw_score += local_count
        weights[out_i] = local_count / float(k)

    effective = float(np.sum(weights))
    return ViewSupportMetrics(
        neighbors_used=k,
        neighbor_image_ids=[int(map_data.image_ids[i]) for i in neighbors],
        candidate_points=candidates,
        redetectable_points=redetectable,
        effective_redetectable_points=effective,
        paper_raw_score=raw_score,
        visible_support_fraction=redetectable / max(len(visible), 1),
        weighted_visible_support_fraction=effective / max(len(visible), 1),
        observation_angle_p50_deg=float(np.percentile(angles, 50)) if angles else None,
        observation_angle_p90_deg=float(np.percentile(angles, 90)) if angles else None,
        angle_extrapolated_fraction=angle_failures / max(len(angles), 1),
        range_extrapolated_fraction=range_failures / max(len(angles), 1),
        point_weights=weights,
    )


def _rotation_distances_rad(query_R_wc: np.ndarray, image_R_wc: np.ndarray) -> np.ndarray:
    relative = np.einsum("ij,njk->nik", query_R_wc.T, image_R_wc)
    traces = np.trace(relative, axis1=1, axis2=2)
    return np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))


def _camera_spacing_scale(map_data: MapData) -> float:
    if map_data.num_images < 2:
        return 1.0
    distances, _ = map_data.image_tree().query(map_data.image_centers, k=2)
    nearest = np.asarray(distances, dtype=float)[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    return float(np.median(nearest)) if len(nearest) else 1.0


def _empty(weights: np.ndarray) -> ViewSupportMetrics:
    return ViewSupportMetrics(
        neighbors_used=0,
        neighbor_image_ids=[],
        candidate_points=0,
        redetectable_points=0,
        effective_redetectable_points=0.0,
        paper_raw_score=0.0,
        visible_support_fraction=0.0,
        weighted_visible_support_fraction=0.0,
        observation_angle_p50_deg=None,
        observation_angle_p90_deg=None,
        angle_extrapolated_fraction=0.0,
        range_extrapolated_fraction=0.0,
        point_weights=weights,
    )
