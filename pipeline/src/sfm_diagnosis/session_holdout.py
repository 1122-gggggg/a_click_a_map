"""Mapping-session reference-exclusion localization from a frozen reconstruction.

This is not a rebuilt strict leave-one-session-out map.  Query 2D-3D pairs are
taken from the existing reconstruction, then points that are not observed by
any other session are dropped before PnP.  Failures of pose-only / bridge
cameras are reported, but they are not calibration-eligible.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .edm_risk.edm_loo import EDMQueryResult
from .io import find_colmap_model


def session_from_image_name(name: str) -> str:
    token = str(name).strip()
    if not token:
        raise ValueError("image name must be non-empty")
    return token.split("/", 1)[0]


def cross_session_observations(
    query_name: str,
    observations: Sequence[tuple[tuple[float, float], int]],
    tracks: Mapping[int, Sequence[str]],
    *,
    min_other_session_views: int = 1,
) -> tuple[tuple[tuple[float, float], int], ...]:
    query_session = session_from_image_name(query_name)
    kept: list[tuple[tuple[float, float], int]] = []
    for xy, point_id in observations:
        other_sessions = [
            session_from_image_name(name)
            for name in tracks.get(int(point_id), ())
            if session_from_image_name(name) != query_session
        ]
        if len(other_sessions) >= int(min_other_session_views):
            kept.append((tuple(float(value) for value in xy), int(point_id)))
    return tuple(kept)


def yaw_pitch_from_R_wc(rotation: np.ndarray) -> tuple[float, float]:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    forward = matrix[:, 2]
    norm = float(np.linalg.norm(forward))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("camera rotation has no forward axis")
    forward = forward / norm
    yaw = float(np.degrees(np.arctan2(forward[1], forward[0])))
    pitch = float(np.degrees(np.arcsin(np.clip(forward[2], -1.0, 1.0))))
    return yaw, pitch


def summarize_holdout_results(
    results: Sequence[EDMQueryResult],
    *,
    min_candidates: int = 8,
) -> dict[str, Any]:
    eligible = [
        row for row in results if int(row.valid_2d3d or 0) >= int(min_candidates)
    ]
    empty = [
        row for row in results if int(row.valid_2d3d or 0) < int(min_candidates)
    ]
    by_session: dict[str, dict[str, int]] = defaultdict(
        lambda: {"queries": 0, "eligible": 0, "eligible_success": 0, "empty": 0}
    )
    for row in results:
        bucket = by_session[row.session_id]
        bucket["queries"] += 1
        if int(row.valid_2d3d or 0) >= int(min_candidates):
            bucket["eligible"] += 1
            bucket["eligible_success"] += int(bool(row.success))
        else:
            bucket["empty"] += 1
    return {
        "schema_version": 1,
        "artifact_type": "MAPPING_SESSION_REFERENCE_EXCLUSION",
        "loo_mode": "reference-exclusion",
        "queries": len(results),
        "calibration_eligible": len(eligible),
        "bridge_or_empty": len(empty),
        "eligible_successes": sum(int(bool(row.success)) for row in eligible),
        "eligible_success_rate": (
            float(np.mean([int(bool(row.success)) for row in eligible]))
            if eligible
            else None
        ),
        "min_candidates": int(min_candidates),
        "by_session": dict(by_session),
        "usable_as_failure_probability_labels": bool(eligible),
    }


def run_session_holdout(
    model_path: str | Path,
    *,
    min_other_session_views: int = 1,
    min_inliers: int = 30,
    max_reproj_error: float = 4.0,
) -> list[EDMQueryResult]:
    """PnP each registered image using only 3D points seen by another session."""

    import cv2

    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyCOLMAP is required for session holdout") from exc

    reconstruction = pycolmap.Reconstruction(str(find_colmap_model(model_path)))
    tracks: dict[int, tuple[str, ...]] = {}
    xyz_by_id: dict[int, np.ndarray] = {}
    for point_id, point in reconstruction.points3D.items():
        names = tuple(
            str(reconstruction.images[element.image_id].name)
            for element in point.track.elements
            if int(element.image_id) in reconstruction.images
        )
        tracks[int(point_id)] = names
        xyz_by_id[int(point_id)] = np.asarray(point.xyz, dtype=float).reshape(3)

    results: list[EDMQueryResult] = []
    for image_id, image in sorted(reconstruction.images.items(), key=lambda item: int(item[0])):
        if not bool(image.has_pose):
            continue
        name = str(image.name)
        observations: list[tuple[tuple[float, float], int]] = []
        for point2d in image.points2D:
            if not bool(point2d.has_point3D()):
                continue
            point3d_id = int(point2d.point3D_id)
            if point3d_id not in xyz_by_id:
                continue
            xy = np.asarray(point2d.xy, dtype=float).reshape(2)
            observations.append(((float(xy[0]), float(xy[1])), point3d_id))
        kept = cross_session_observations(
            name,
            observations,
            tracks,
            min_other_session_views=min_other_session_views,
        )
        camera = reconstruction.cameras[image.camera_id]
        K = np.array(
            [
                [float(camera.focal_length_x), 0.0, float(camera.principal_point_x)],
                [0.0, float(camera.focal_length_y), float(camera.principal_point_y)],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        cam_from_world = image.cam_from_world()
        R_cw = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
        center = np.asarray(image.projection_center(), dtype=float).reshape(3)
        yaw, pitch = yaw_pitch_from_R_wc(R_cw.T)
        success = False
        inliers = 0
        inlier_ratio = None
        reproj = None
        if kept:
            object_points = np.asarray([xyz_by_id[point_id] for _, point_id in kept], dtype=float)
            image_points = np.asarray([xy for xy, _ in kept], dtype=float)
            ok, _rvec, _tvec, inlier_idx = cv2.solvePnPRansac(
                object_points,
                image_points,
                K,
                None,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=float(max_reproj_error),
                iterationsCount=200,
                confidence=0.99,
            )
            inliers = 0 if inlier_idx is None else int(len(inlier_idx))
            inlier_ratio = float(inliers / max(len(kept), 1))
            success = bool(ok) and inliers >= int(min_inliers)
            if inlier_idx is not None and len(inlier_idx):
                projected, _ = cv2.projectPoints(
                    object_points[inlier_idx.reshape(-1)],
                    _rvec,
                    _tvec,
                    K,
                    None,
                )
                residual = image_points[inlier_idx.reshape(-1)] - projected.reshape(-1, 2)
                norms = np.linalg.norm(residual, axis=1)
                reproj = float(np.median(norms)) if len(norms) else None
        results.append(
            EDMQueryResult(
                query_id=name,
                session_id=session_from_image_name(name),
                timestamp=float(image_id),
                success=success,
                registration_success=True,
                valid_2d3d=len(kept),
                ransac_inliers=inliers,
                inlier_ratio=inlier_ratio,
                reprojection_median=reproj,
                estimated_position=tuple(float(value) for value in center),
                estimated_yaw_deg=yaw,
                estimated_pitch_deg=pitch,
                ground_truth_source="map_pose",
                loo_mode="reference-exclusion",
                point_ids=tuple(point_id for _, point_id in kept),
            )
        )
    return results
