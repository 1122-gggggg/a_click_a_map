from __future__ import annotations

from typing import Sequence

import numpy as np

from sfm_diagnosis.models import MapData, Pose
from sfm_diagnosis.visibility import visible_points

from .edm_loo import EDMQueryResult


def edm_visibility_events(
    map_data: MapData,
    results: Sequence[EDMQueryResult],
    *,
    max_landmark_distance: float | None = None,
) -> tuple[list[dict], dict]:
    """Build Beta-matchability events from expected frustum-visible landmarks.

    This uses a query pose to define the denominator and recorded EDM PnP inlier
    point IDs for the numerator.  Without a mesh, expected visibility is explicitly
    ``frustum_only`` and never called occlusion verified.
    """

    events: list[dict] = []
    without_pose = 0
    inliers_outside_frustum = 0
    intrinsics = map_data.median_intrinsics
    for result in results:
        if result.estimated_position is None or result.estimated_R_wc is None:
            without_pose += 1
            continue
        pose = Pose(
            np.asarray(result.estimated_position, dtype=float),
            np.asarray(result.estimated_R_wc, dtype=float),
        )
        visible = visible_points(
            map_data,
            pose,
            intrinsics=intrinsics,
            max_distance=max_landmark_distance,
        )
        expected_ids = {
            int(value) for value in map_data.point_ids[visible.point_indices].tolist()
        }
        inlier_ids = {int(value) for value in result.point_ids}
        inliers_outside_frustum += len(inlier_ids - expected_ids)
        for point_id in sorted(expected_ids):
            events.append(
                {
                    "query_id": result.query_id,
                    "session_id": result.session_id,
                    "point_id": point_id,
                    "observed": True,
                    "inlier": point_id in inlier_ids,
                    "timestamp": float(result.timestamp),
                    "visibility_source": "frustum_only",
                }
            )
    return events, {
        "schema_version": 1,
        "artifact_type": "EDM_LANDMARK_MATCHABILITY_EVENTS",
        "query_count": len(results),
        "queries_without_pose": without_pose,
        "event_count": len(events),
        "inlier_event_count": sum(bool(event["inlier"]) for event in events),
        "inliers_outside_frustum": inliers_outside_frustum,
        "visibility_source": "frustum_only",
        "occlusion_verified": False,
        "denominator": "landmarks expected inside query camera frustum",
        "numerator": "recorded EDM/PnP inlier point IDs",
    }
