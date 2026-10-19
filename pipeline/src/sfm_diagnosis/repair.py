from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actloc import StructuralLocalizabilityProxy
from .diagnose import DiagnosticThresholds, diagnose_pose
from .heatmap import health_score, rotation_from_forward
from .models import MapData, Pose
from .visibility import visible_points


@dataclass(frozen=True)
class RepairCandidate:
    center_w: np.ndarray
    R_wc: np.ndarray
    score: float
    overlap_ratio: float
    baseline_score: float
    candidate_health: float
    shared_points: int

    def as_dict(self) -> dict:
        return {
            "center_w": self.center_w.tolist(),
            "R_wc": self.R_wc.tolist(),
            "forward_w": self.R_wc[:, 2].tolist(),
            "score": self.score,
            "overlap_ratio": self.overlap_ratio,
            "baseline_score": self.baseline_score,
            "candidate_health": self.candidate_health,
            "shared_points": self.shared_points,
        }


def suggest_capture_viewpoints(
    map_data: MapData,
    weak_pose: Pose,
    *,
    radii_m: tuple[float, ...] = (1.5, 3.0, 5.0),
    azimuth_samples: int = 12,
    elevation_deg: tuple[float, ...] = (-20.0, 0.0, 20.0),
    top_k: int = 8,
    thresholds: DiagnosticThresholds | None = None,
    world_up: np.ndarray | None = None,
) -> list[RepairCandidate]:
    """Generate *capture* candidates for weak geometry/view coverage.

    This is an observation-diversity proxy, not a claim that new unseen landmarks can
    be predicted. Candidates are rewarded for maintaining overlap with landmarks seen
    from the weak pose while adding a useful triangulation angle and having healthy
    local observability themselves.
    """
    up = -weak_pose.R_wc[:, 1] if world_up is None else np.asarray(world_up, dtype=float)
    weak_vis = visible_points(map_data, weak_pose, max_distance=30.0)
    weak_idx = weak_vis.point_indices
    if len(weak_idx) == 0:
        return []
    quality = map_data.point_quality_weights()[weak_idx]
    if np.sum(quality) > 1e-9:
        target = np.average(map_data.points_xyz[weak_idx], axis=0, weights=quality)
    else:
        target = np.mean(map_data.points_xyz[weak_idx], axis=0)
    weak_rays = map_data.points_xyz[weak_idx] - weak_pose.center_w
    weak_rays /= np.maximum(np.linalg.norm(weak_rays, axis=1, keepdims=True), 1e-12)
    weak_lookup = {int(p): i for i, p in enumerate(weak_idx)}

    basis_x = weak_pose.R_wc[:, 0]
    basis_y = weak_pose.R_wc[:, 1]
    basis_z = weak_pose.R_wc[:, 2]
    candidates: list[RepairCandidate] = []
    predictor = StructuralLocalizabilityProxy()
    for radius in radii_m:
        for elevation in elevation_deg:
            el = np.radians(elevation)
            for az_i in range(azimuth_samples):
                az = 2.0 * np.pi * az_i / azimuth_samples
                lateral = np.cos(az) * basis_x + np.sin(az) * basis_y
                center = weak_pose.center_w + radius * (np.cos(el) * lateral + np.sin(el) * basis_z)
                forward = target - center
                if np.linalg.norm(forward) < 1e-6:
                    continue
                R_wc = rotation_from_forward(forward, up)
                pose = Pose(center, R_wc)
                vis = visible_points(map_data, pose, max_distance=30.0)
                shared = np.intersect1d(vis.point_indices, weak_idx, assume_unique=False)
                if len(shared) < 4:
                    continue
                overlap = len(shared) / max(len(weak_idx), 1)
                angles = []
                for pidx in shared:
                    wi = weak_lookup[int(pidx)]
                    ray2 = map_data.points_xyz[pidx] - center
                    ray2 /= max(np.linalg.norm(ray2), 1e-12)
                    angle = np.degrees(np.arccos(np.clip(np.dot(weak_rays[wi], ray2), -1.0, 1.0)))
                    angles.append(angle)
                angles = np.asarray(angles)
                # Reward a broad but not extreme baseline, peaking around 20 degrees.
                baseline = float(np.mean(np.exp(-0.5 * ((angles - 20.0) / 15.0) ** 2)))
                diag = diagnose_pose(map_data, pose, thresholds=thresholds, predictor=predictor)
                h = health_score(diag)
                score = float(0.45 * overlap + 0.35 * baseline + 0.20 * h)
                candidates.append(
                    RepairCandidate(center, R_wc, score, float(overlap), baseline, h, len(shared))
                )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
