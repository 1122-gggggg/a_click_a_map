from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import MapData, Pose


class LocalizabilityPredictor(Protocol):
    def predict(self, map_data: MapData, pose: Pose) -> float: ...


@dataclass
class StructuralLocalizabilityProxy:
    """Training-free ActLoc-style structural prior.

    This is intentionally *not* the ActLoc network. It provides a lightweight,
    interpretable baseline using the two map modalities highlighted by ActLoc:
    landmark support and mapping-camera distribution. It can later be replaced by
    an external ActLoc checkpoint through the LocalizabilityPredictor protocol.
    """

    point_radius_m: float = 8.0
    camera_radius_m: float = 8.0
    target_points: float = 250.0
    target_cameras: float = 12.0

    def predict(self, map_data: MapData, pose: Pose) -> float:
        if map_data.num_points == 0:
            return 0.0
        pidx = map_data.point_tree().query_ball_point(pose.center_w, self.point_radius_m)
        point_score = min(len(pidx) / max(self.target_points, 1.0), 1.0)
        if pidx:
            q = map_data.point_quality_weights()[np.asarray(pidx, dtype=int)]
            point_score *= float(np.mean(q))

        if map_data.num_images == 0:
            camera_score = 0.0
            view_score = 0.0
        else:
            cidx = np.asarray(map_data.image_tree().query_ball_point(pose.center_w, self.camera_radius_m), dtype=int)
            camera_score = min(len(cidx) / max(self.target_cameras, 1.0), 1.0)
            if len(cidx):
                fw = map_data.image_R_wc[cidx, :, 2]
                dots = np.clip(fw @ pose.forward_w, -1.0, 1.0)
                # A map with at least one similar historical view is easier for feature localization.
                view_score = float(np.max((dots + 1.0) * 0.5))
            else:
                view_score = 0.0
        return float(np.clip(0.55 * point_score + 0.25 * camera_score + 0.20 * view_score, 0.0, 1.0))


@dataclass
class ExternalPredictorAdapter:
    """Wrap any user-supplied callable, including an ActLoc inference function."""

    function: object

    def predict(self, map_data: MapData, pose: Pose) -> float:
        value = self.function(map_data, pose)
        return float(np.clip(value, 0.0, 1.0))
