"""Fixed-map geometry anchors and per-image MoGe optical-Z scale diagnostics.

All coordinates are the map's already-undistorted PINHOLE coordinates. A track's
length is not evidence of triangulation angle: reject low-parallax tracks before
looking at any predicted depths. Weights are heuristics, not probabilities.
"""

from __future__ import annotations

import math
import hashlib
import json
import operator
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .geometry_contract import (
    CALIBRATION_BASIS, CALIBRATION_SCHEMA,
    geometry_policy_digest, seal_depth_calibration, validate_mapping_record,
    validate_pinhole, validate_pose,
)
from .metric_dense import sample_depth_bilinear

MIN_TRIANGULATION_DEG = 1.5
MIN_SAMPLES = 30
ANCHOR_VALIDATION_VERSION = "mapping-blocked-shape-calibration/v5"


def _shape_diagnostics(entries: np.ndarray, log_scale: float) -> tuple[float, dict[str, float | None]]:
    residual = entries[:, 0] - log_scale
    mad = float(np.median(np.abs(residual)))
    trends = {}
    for label, column in (('depth', 2), ('u', 3), ('v', 4)):
        mid = np.median(entries[:, column])
        near = residual[entries[:, column] <= mid]
        far = residual[entries[:, column] > mid]
        trends[label] = float(np.median(far) - np.median(near)) if len(near) and len(far) else None
    return mad, trends


def _shape_conflict(mad: float, trends: Mapping[str, float | None]) -> bool:
    return mad > math.log(1.25) or any(
        abs(value) > math.log(1.3) for value in trends.values() if value is not None
    )


def valid_track_observations(rec: Any, point_id: int, *, mapping_image_ids: frozenset[int]
                             ) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Validate every allowlisted MAPPING track element before grouping."""
    if not isinstance(mapping_image_ids, frozenset) or not mapping_image_ids:
        raise ValueError("track validation requires the verified MAPPING image ID allowlist")
    if point_id not in rec.points3D:
        return []
    point = rec.points3D[point_id]
    xyz = np.asarray(point.xyz, dtype=np.float64)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        return []
    valid = {}
    for element in point.track.elements:
        try:
            image_id = operator.index(element.image_id)
            index = operator.index(element.point2D_idx)
        except (TypeError, ValueError, OverflowError):
            continue
        if image_id not in mapping_image_ids or image_id not in rec.images:
            continue
        image = rec.images[image_id]
        if not 0 <= index < len(image.points2D):
            continue
        obs = image.points2D[index]
        try:
            observed_point_id = operator.index(obs.point3D_id)
        except (TypeError, ValueError, OverflowError):
            continue
        if not obs.has_point3D() or observed_point_id != point_id:
            continue
        try:
            camera = rec.cameras[image.camera_id]
            if str(camera.model).split(".")[-1] != "PINHOLE":
                continue
            pose = validate_pose(image.cam_from_world().matrix())
            k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        except (KeyError, ValueError, AttributeError, TypeError):
            continue
        uv = np.asarray(obs.xy, dtype=np.float64)
        camera_xyz = pose[:, :3] @ xyz + pose[:, 3]
        if (uv.shape != (2,) or not np.isfinite(uv).all() or
                not (0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height) or
                not np.isfinite(camera_xyz).all() or camera_xyz[2] <= 0):
            continue
        projected = k[:2, :2] @ (camera_xyz[:2] / camera_xyz[2]) + k[:2, 2]
        if np.linalg.norm(projected - uv) > 2.:
            continue
        ray = xyz + pose[:, :3].T @ pose[:, 3]
        length = np.linalg.norm(ray)
        if not np.isfinite(length) or length <= 1e-8:
            continue
        valid[(image_id, index)] = (image_id, uv, ray / length)
    return [valid[key] for key in sorted(valid)]


def geometry_anchors(rec: Any, *, mapping_image_ids: frozenset[int]) -> dict[int, float]:
    """Return frozen-map weights only after exact registered MAPPING admission."""
    if not isinstance(mapping_image_ids, frozenset) or mapping_image_ids != frozenset(rec.images):
        raise ValueError("anchor inputs must exactly match the verified registered MAPPING allowlist")
    anchors: dict[int, float] = {}
    separation = math.cos(math.radians(.25))
    for point_id, point in rec.points3D.items():
        if point.track.length() < 3 or not np.isfinite(point.error) or not 0 <= point.error <= 2.0:
            continue
        observations = valid_track_observations(rec, point_id, mapping_image_ids=mapping_image_ids)
        if len(observations) < 3:
            continue
        rays = np.asarray([item[2] for item in observations])
        # Farthest-direction sampling follows validation of the entire track.
        # Sorting directions makes duplicates and track permutations immaterial.
        rays = rays[np.lexsort((rays[:, 2], rays[:, 1], rays[:, 0]))]
        representatives = [0]
        closest = rays @ rays[0]
        while len(representatives) < min(16, len(rays)):
            index = int(np.argmin(closest))
            if closest[index] >= separation:
                break
            representatives.append(index)
            closest = np.maximum(closest, rays @ rays[index])
        if len(representatives) < 3:
            continue
        distinct = rays[representatives]
        max_angle = math.degrees(math.acos(float(np.clip(np.min(distinct @ distinct.T), -1., 1.))))
        if max_angle >= MIN_TRIANGULATION_DEG:
            anchors[int(point_id)] = min(1., max_angle / 5.) / (1. + float(point.error))
    return anchors


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2.)]])


def _spread(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {key: None for key in ('p10', 'median', 'p90')}
    return {'p10': float(np.quantile(values, .1)), 'median': float(np.median(values)),
            'p90': float(np.quantile(values, .9))}


def _strata(entries: np.ndarray, residuals: np.ndarray, fit_entries: np.ndarray) -> dict[str, Any]:
    """Fixed u-v cells crossed with fit-only log-depth bins, including holes."""
    if len(fit_entries):
        edges = np.quantile(fit_entries[:, 5], [0., 1 / 3, 2 / 3, 1.])
    else:
        edges = np.array([0., 0., 0., 0.])
    def keys(a: np.ndarray) -> np.ndarray:
        if not len(a):
            return np.empty((0, 3), int)
        return np.column_stack((np.clip((a[:, 3] * 4).astype(int), 0, 3),
                                np.clip((a[:, 4] * 4).astype(int), 0, 3),
                                np.searchsorted(edges[1:-1], a[:, 5], side="right")))
    fit_keys, sample_keys = keys(fit_entries), keys(entries)
    cells = []
    for v in range(4):
        for u in range(4):
            for d in range(3):
                selected = np.all(sample_keys == (u, v, d), axis=1)
                supported = np.all(fit_keys == (u, v, d), axis=1)
                values = residuals[selected]
                weights = entries[selected, 1]
                bias = weighted_median(values, weights) if len(values) else None
                cells.append({"u_cell": u, "v_cell": v, "log_depth_bin": d,
                              "count": int(selected.sum()), "fit_support_count": int(supported.sum()),
                              "bias": bias,
                              "mad": weighted_median(np.abs(values - bias), weights) if len(values) else None,
                              "p10": float(np.quantile(values, .1)) if len(values) else None,
                              "p90": float(np.quantile(np.abs(values), .9)) if len(values) else None,
                              "supported": bool(supported.any()),
                              "extrapolated": bool(selected.any() and (not supported.any() or
                                  np.any(entries[selected, 5] < edges[0]) or
                                  np.any(entries[selected, 5] > edges[-1])))})
    return {"grid_width": 4, "log_depth_edges": edges.tolist(), "cells": cells}


def _sample_array(values: Any) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64).reshape(-1, 9)
    if len(samples) and (not np.isfinite(samples).all() or np.any(samples[:, 1:3] <= 0) or
                         np.any(samples[:, 3:5] < 0) or np.any(samples[:, 3:5] >= 1) or
                         np.any(samples[:, 6:9] < 0) or np.any(samples[:, 6:9] != np.floor(samples[:, 6:9]))):
        raise ValueError("malformed frozen calibration samples")
    return samples


def image_scale(image: Any, rec: Any, row: Mapping[str, Any], depth: np.ndarray,
                mask: np.ndarray, anchors: Mapping[int, float], source_to_work: np.ndarray,
                *, fit_enabled: bool = True) -> dict[str, Any]:
    """Pair frozen native MAPPING observations; external validation never enters."""
    validate_mapping_record(row)
    raw_pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    width, height = int(row["width"]), int(row["height"])
    k = validate_pinhole(np.asarray(row["K_source"]), width, height)
    camera = rec.cameras[image.camera_id]
    if (image.name != row["name"] or str(camera.model).split(".")[-1] != "PINHOLE" or
            int(camera.width) != width or int(camera.height) != height or
            not np.allclose(camera.calibration_matrix(), k, atol=1e-8, rtol=0)):
        raise ValueError("calibration source differs from its registered MAPPING camera")
    image_id = next((int(i) for i, candidate in rec.images.items() if candidate.name == image.name), None)
    if image_id is None or not np.array_equal(rec.images[image_id].cam_from_world().matrix(), raw_pose, equal_nan=True):
        raise ValueError("calibration image is not its registered MAPPING model observation")
    result: dict[str, Any] = {
        "name": row["name"], "image_id": image_id, "session_id": row["session_id"], "frame_index": row["frame_index"],
        "evaluation_role": "MAPPING", "image_sha256": row["image_sha256"],
        "image_path": row.get("image_path"), "width": width, "height": height,
        "K_source": k.tolist(), "cam_from_world": raw_pose.tolist(),
        "scale_role": "MAPPING_FIT" if fit_enabled else "MAPPING_DIAGNOSTIC_NOT_FIT",
        "anchor_validation_version": ANCHOR_VALIDATION_VERSION,
    }
    try:
        pose = validate_pose(raw_pose)
    except ValueError:
        return {**result, "status": "INVALID_CAMERA_POSE", "scale_role": "MAPPING_UNSUPPORTED",
                "unsupported_reason": "IMPROPER_FROZEN_WORLD_TO_CAMERA_POSE",
                "n_samples": 0, "n_validation_samples": 0,
                "calibration_fit_samples": [], "calibration_validation_samples": []}
    result["camera_center"] = (-pose[:, :3].T @ pose[:, 3]).tolist()
    if np.ndim(depth) != 2 or np.shape(mask) != np.shape(depth):
        raise ValueError("calibration depth and mask must be aligned 2D arrays")
    if not image.num_points3D:
        return {**result, "status": "POSE_ONLY", "n_samples": 0, "n_validation_samples": 0,
                "calibration_fit_samples": [], "calibration_validation_samples": []}
    fit, validation = [], []
    observations_by_point: dict[int, tuple[Any, ...]] = {}
    for observation_index, obs in enumerate(image.points2D):
        if not obs.has_point3D() or obs.point3D_id not in anchors:
            continue
        if obs.point3D_id not in rec.points3D:
            continue
        point = rec.points3D[obs.point3D_id]
        if not any(isinstance(element.image_id, (int, np.integer)) and
                   isinstance(element.point2D_idx, (int, np.integer)) and
                   element.image_id == image_id and element.point2D_idx == observation_index
                   for element in point.track.elements):
            continue
        xyz = np.asarray(point.xyz, dtype=np.float64)
        uv = np.asarray(obs.xy, dtype=np.float64)
        if xyz.shape != (3,) or uv.shape != (2,) or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
            continue
        camera = pose[:, :3] @ xyz + pose[:, 3]
        if not np.isfinite(camera).all() or camera[2] <= 0 or not (0 <= uv[0] < width and 0 <= uv[1] < height):
            continue
        error = float(np.linalg.norm(k[:2, :2] @ (camera[:2] / camera[2]) + k[:2, 2] - uv))
        if error > 2.:
            continue
        work = source_to_work @ [uv[0], uv[1], 1.]
        if not np.isfinite(work).all() or work[2] == 0:
            continue
        pixel = work[:2] / work[2]
        if not (2 <= pixel[0] < depth.shape[1] - 2 and 2 <= pixel[1] < depth.shape[0] - 2):
            continue
        point_id = int(obs.point3D_id)
        rank = (error, float(uv[0]), float(uv[1]))
        previous = observations_by_point.get(point_id)
        if previous is None or rank < previous[0]:
            observations_by_point[point_id] = (rank, (point_id, uv, pixel, float(camera[2]), float(anchors[point_id])))
    observations = [observations_by_point[point_id][1] for point_id in sorted(observations_by_point)]
    if observations:
        pixels = np.asarray([item[2] for item in observations])
        measured = sample_depth_bilinear(depth, pixels, (depth.shape[1], depth.shape[0]), mask)
        for (point_id, uv, pixel, map_z, weight), value in zip(observations, measured):
            if not np.isfinite(value) or value <= 0:
                continue
            px, py = np.rint(pixel).astype(int)
            patch = depth[py - 1:py + 2, px - 1:px + 2]
            if (not mask[py - 1:py + 2, px - 1:px + 2].all() or
                    not np.isfinite(patch).all() or np.any(patch <= 0) or
                    (float(np.max(patch)) - float(np.min(patch))) / float(value) > .2):
                continue
            block = min(3, int(4 * uv[0] / width)) + 4 * min(3, int(4 * uv[1] / height))
            entry = (math.log(float(value)) - math.log(map_z), weight, map_z,
                     float(uv[0]) / width, float(uv[1]) / height, math.log(float(value)),
                     point_id, block, int(row["frame_index"]) // 32)
            (validation if point_id % 7 == 0 or block % 7 == 0 else fit).append(entry)
    fit_array, validation_array = _sample_array(fit), _sample_array(validation)
    result.update(n_samples=len(fit), n_validation_samples=len(validation),
                  calibration_fit_samples=fit, calibration_validation_samples=validation)
    if len(fit) < MIN_SAMPLES:
        return {**result, "status": "INSUFFICIENT_TRACKS"}
    cells = {(int(item[3] * 4), int(item[4] * 4)) for item in fit_array}
    if len(cells) < 4 or np.quantile(fit_array[:, 2], .9) / np.quantile(fit_array[:, 2], .1) < 1.1:
        return {**result, "status": "INSUFFICIENT_COVERAGE", "grid_cells": len(cells)}
    # On the outer session this is an observed diagnostic ratio, not a fitted
    # parameter: calibrate_image_records never trains on these samples.
    observed = np.vstack((fit_array, validation_array)) if not fit_enabled else fit_array
    log_scale = weighted_median(observed[:, 0], observed[:, 1])
    mad, trends = _shape_diagnostics(fit_array, log_scale)
    result.update(log_scale=log_scale, log_ratio=_spread(fit_array[:, 0]),
                  shape_mad=mad, shape_trend=trends, grid_cells=len(cells),
                  fit_residual_strata=_strata(fit_array, fit_array[:, 0] - log_scale, fit_array),
                  validation_residual_strata=_strata(validation_array, validation_array[:, 0] - log_scale, fit_array),
                  validation_log_residual=_spread(validation_array[:, 0] - log_scale))
    if len(validation) < MIN_SAMPLES // 3:
        result["status"] = "INSUFFICIENT_VALIDATION"
        return result
    validation_mad, validation_trends = _shape_diagnostics(validation_array, log_scale)
    validation_cells = {(int(item[3] * 4), int(item[4] * 4)) for item in validation_array}
    result.update(validation_shape_mad=validation_mad, validation_shape_trend=validation_trends,
                  validation_grid_cells=len(validation_cells))
    result["status"] = ("INSUFFICIENT_VALIDATION" if len(validation_cells) < 4 else
                        "SHAPE_CONFLICT" if _shape_conflict(mad, trends) or
                        _shape_conflict(validation_mad, validation_trends) else "SCALE_OK")
    return result


def _basis(samples: np.ndarray, center: float) -> np.ndarray:
    return np.column_stack((samples[:, 3] - .5, samples[:, 4] - .5, samples[:, 5] - center))


def _fit_correction(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Frozen regularization/robustness; never selected on blocked/outer samples.
    normalized = weights / weights.sum()
    robust = normalized.copy()
    coefficient = np.zeros(3)
    for _ in range(3):
        coefficient = np.linalg.solve(x.T @ (robust[:, None] * x) + .01 * np.eye(3),
                                      x.T @ (robust * y))
        residual = np.abs(y - x @ coefficient)
        robust = normalized * np.minimum(1., .08 / np.maximum(residual, 1e-12))
    return coefficient


def _tail(values: np.ndarray) -> dict[str, float | int | None]:
    a = np.abs(values)
    return {"count": len(a), "median_abs": float(np.median(a)) if len(a) else None,
            "p90_abs": float(np.quantile(a, .9)) if len(a) else None}


def _sample_rays(row: Mapping[str, Any], samples: np.ndarray) -> np.ndarray:
    pixels = np.column_stack((samples[:, 3] * row["width"], samples[:, 4] * row["height"],
                              np.ones(len(samples))))
    rays = pixels @ np.linalg.inv(np.asarray(row["K_source"])).T @ np.asarray(row["cam_from_world"])[:, :3]
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _ids_digest(values: set[int]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def calibrate_image_records(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fit once on MAPPING-only sessions and globally disjoint landmarks/rays.

    The last two MAPPING sessions are blocked and outer internal checks. Neither
    session enters coefficient fitting or inner CV. Query/validation-video data
    is rejected before samples are inspected and cannot select this correction.
    """
    for row in records:
        validate_mapping_record(row)
    output = [dict(row) for row in records]
    if len({row["name"] for row in output}) != len(output):
        raise ValueError("duplicate registered MAPPING calibration image")
    supported = []
    for row in output:
        try:
            pose = validate_pose(row["cam_from_world"])
        except ValueError:
            row.update(status="INVALID_CAMERA_POSE", scale_role="MAPPING_UNSUPPORTED",
                       unsupported_reason="IMPROPER_FROZEN_WORLD_TO_CAMERA_POSE",
                       n_samples=0, n_validation_samples=0,
                       calibration_fit_samples=[], calibration_validation_samples=[])
            row.pop("log_scale", None)
            row.pop("camera_center", None)
            continue
        k = validate_pinhole(row["K_source"], int(row["width"]), int(row["height"]))
        row.update(cam_from_world=pose.tolist(), K_source=k.tolist(),
                   camera_center=(-pose[:, :3].T @ pose[:, 3]).tolist())
        supported.append(row)
    sessions = sorted({row["session_id"] for row in supported})
    blocked, outer = (sessions[-2], sessions[-1]) if len(sessions) >= 3 else (None, None)
    training_sessions = sessions[:-2] if len(sessions) >= 3 else sessions
    validation_by_name: dict[str, np.ndarray] = {}
    heldout_ids: set[int] = set()
    for row in supported:
        fit = _sample_array(row.get("calibration_fit_samples", []))
        validation = _sample_array(row.get("calibration_validation_samples", []))
        for samples in (fit, validation):
            blocks = np.floor(samples[:, 3] * 4) + 4 * np.floor(samples[:, 4] * 4)
            if (np.any(samples[:, 7] != blocks) or
                    np.any(samples[:, 8] != row["frame_index"] // 32) or
                    not np.allclose(samples[:, 0], samples[:, 5] - np.log(samples[:, 2]), atol=1e-10, rtol=0)):
                raise ValueError("calibration samples differ from their native spatial/temporal/depth provenance")
        allowed = (fit[:, 6].astype(np.int64) % 7 != 0) & (fit[:, 7].astype(np.int64) % 7 != 0)
        validation = np.vstack((validation, fit[~allowed]))
        row["calibration_fit_samples"] = fit[allowed].tolist()
        row["calibration_validation_samples"] = validation.tolist()
        validation_by_name[row["name"]] = validation
        reserved_capture = row["session_id"] in (blocked, outer) or row["frame_index"] // 32 % 7 == 6
        if reserved_capture:
            validation = np.vstack((validation, fit[allowed]))
            validation_by_name[row["name"]] = validation
            row["calibration_validation_samples"] = validation.tolist()
        heldout_ids.update(int(point_id) for point_id in validation[:, 6])
        row["scale_role"] = ("MAPPING_BLOCKED_VALIDATION" if row["session_id"] == blocked else
                             "MAPPING_OUTER_VALIDATION" if row["session_id"] == outer else
                             "MAPPING_RESERVED_CAPTURE" if row["frame_index"] // 32 % 7 == 6 else
                             "MAPPING_FIT")
    centers = np.asarray([row["camera_center"] for row in supported])
    cell_width = max(float(np.linalg.norm(np.ptp(centers, axis=0))) * .05, 1e-8) if len(centers) else 1.
    def camera_cell(row: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(np.floor(np.asarray(row["camera_center"]) / cell_width).astype(int))
    def group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return row["session_id"], row["frame_index"] // 32, camera_cell(row)
    # Reserve real world-ray neighborhoods globally, including across sessions.
    # The radius is the unchanged 0.25-degree distinct-anchor-ray boundary.
    heldout_rays: dict[tuple[int, ...], list[np.ndarray]] = {}
    for row in supported:
        validation = validation_by_name[row["name"]]
        if len(validation):
            heldout_rays.setdefault(camera_cell(row), []).append(_sample_rays(row, validation))
    ray_trees = {cell: cKDTree(np.vstack(chunks)) for cell, chunks in heldout_rays.items()}
    ray_radius = 2. * math.sin(math.radians(.25) / 2.)
    training, targets, weights, training_rows = [], [], [], []
    eligible = []
    for row in supported:
        fit = _sample_array(row["calibration_fit_samples"])
        if row["session_id"] in training_sessions and row["frame_index"] // 32 % 7 != 6:
            allowed = np.asarray([int(point_id) not in heldout_ids for point_id in fit[:, 6]], bool)
            tree = ray_trees.get(camera_cell(row))
            if tree is not None and len(fit):
                distance, _ = tree.query(_sample_rays(row, fit), k=1, distance_upper_bound=ray_radius)
                allowed &= ~np.isfinite(distance)
            row["calibration_excluded_overlap_count"] = int(np.count_nonzero(~allowed))
            fit = fit[allowed]
            row["calibration_fit_samples"] = fit.tolist()
        if len(fit) < MIN_SAMPLES or "log_scale" not in row:
            if row["scale_role"] == "MAPPING_FIT" and row.get("status") == "SCALE_OK":
                row["status"] = "INSUFFICIENT_GROUPED_FIT"
            continue
        if not math.isfinite(float(row["log_scale"])):
            raise ValueError("nonfinite MAPPING calibration scale")
        eligible.append(row)
        if row["scale_role"] != "MAPPING_FIT" or row["session_id"] not in training_sessions:
            continue
        # Global reservation happens before even the fitted per-image scale.
        row["log_scale"] = weighted_median(fit[:, 0], fit[:, 1])
        training.append(fit)
        targets.append(fit[:, 0] - float(row["log_scale"]))
        weights.append(fit[:, 1] / fit[:, 1].sum())
        training_rows.append(row)
    train = np.vstack(training) if training else np.empty((0, 9))
    group_counts: dict[tuple[Any, ...], int] = {}
    session_group_sets: dict[str, set[tuple[Any, ...]]] = {}
    for row in training_rows:
        key = group_key(row)
        group_counts[key] = group_counts.get(key, 0) + 1
        session_group_sets.setdefault(row["session_id"], set()).add(key)
    adjusted_weights = [w / group_counts[group_key(row)] / len(session_group_sets[row["session_id"]])
                        for w, row in zip(weights, training_rows)]
    flat_weights = np.concatenate(adjusted_weights) if adjusted_weights else np.empty(0)
    center = weighted_median(train[:, 5], flat_weights) if len(train) else 0.
    coeff = (_fit_correction(_basis(train, center), np.concatenate(targets), flat_weights)
             if len(train) >= MIN_SAMPLES else np.zeros(3))
    baseline_session: dict[str, list[float]] = {}
    proposed_session: dict[str, list[float]] = {}
    proposed_scales = {}
    for row in eligible:
        fit = _sample_array(row["calibration_fit_samples"])
        observed = fit if row["scale_role"] == "MAPPING_FIT" else validation_by_name[row["name"]]
        observed_before = (float(row["log_scale"]) if row["scale_role"] == "MAPPING_FIT" else
                           weighted_median(observed[:, 0], observed[:, 1]))
        observed_corrected = weighted_median(observed[:, 0] - _basis(observed, center) @ coeff, observed[:, 1])
        if row["scale_role"] == "MAPPING_FIT":
            proposed_scales[row["name"]] = observed_corrected
        else:
            row["diagnostic_log_scale_before"] = observed_before
            row["diagnostic_log_scale_candidate"] = observed_corrected
        baseline_session.setdefault(row["session_id"], []).append(observed_before)
        proposed_session.setdefault(row["session_id"], []).append(observed_corrected)
    global_baseline = float(np.median([row["log_scale"] for row in training_rows])) if training_rows else 0.
    global_corrected = float(np.median([proposed_scales[row["name"]] for row in training_rows])) if training_rows else 0.
    blocked_before, blocked_after, outer_before, outer_after = [], [], [], []
    for row in supported:
        if row["session_id"] not in (blocked, outer):
            continue
        samples = validation_by_name[row["name"]]
        before = samples[:, 0] - global_baseline
        after = samples[:, 0] - global_corrected - _basis(samples, center) @ coeff
        (outer_before if row["session_id"] == outer else blocked_before).extend(before.tolist())
        (outer_after if row["session_id"] == outer else blocked_after).extend(after.tolist())
        row["calibration_validation_before"] = _tail(before)
        row["calibration_validation_candidate"] = _tail(after)
    def improves(before: list[float], after: list[float]) -> bool:
        b, a = _tail(np.asarray(before)), _tail(np.asarray(after))
        return (b["count"] >= 10 and a["median_abs"] <= b["median_abs"] + 1e-12 and
                a["p90_abs"] < b["p90_abs"] - 1e-9)
    def spread(values: dict[str, list[float]]) -> float:
        medians = [float(np.median(a)) for a in values.values()]
        return math.exp(max(medians) - min(medians)) if medians else math.inf
    before_spread, after_spread = spread(baseline_session), spread(proposed_session)
    evaluated = (len(train) >= MIN_SAMPLES and len(blocked_before) >= 10 and len(outer_before) >= 10 and
                 blocked in baseline_session and outer in baseline_session)
    accepted = (evaluated and improves(blocked_before, blocked_after) and improves(outer_before, outer_after) and
                after_spread <= 1.20 and after_spread <= before_spread + 1e-12)
    # Inner diagnostics also hold out entire MAPPING sessions, landmark IDs,
    # spatial blocks and world-ray neighborhoods; frozen ridge is never tuned.
    inner = []
    train_sessions = sorted({row["session_id"] for row in training_rows})
    for fold in range(min(3, len(train_sessions))):
        check_sessions = {session for index, session in enumerate(train_sessions) if index % 3 == fold}
        fit_indices = [index for index, row in enumerate(training_rows) if row["session_id"] not in check_sessions]
        check_rows = [row for row in training_rows if row["session_id"] in check_sessions]
        check_samples = [np.vstack((_sample_array(row["calibration_fit_samples"]),
                                   validation_by_name[row["name"]])) for row in check_rows]
        check = np.vstack(check_samples) if check_samples else np.empty((0, 9))
        reserved_ids = {int(point_id) for point_id in check[:, 6]}
        reserved_rays: dict[tuple[int, ...], list[np.ndarray]] = {}
        for row, samples in zip(check_rows, check_samples):
            if len(samples):
                reserved_rays.setdefault(camera_cell(row), []).append(_sample_rays(row, samples))
        trees = {cell: cKDTree(np.vstack(chunks)) for cell, chunks in reserved_rays.items()}
        fold_samples, fold_targets, fold_weights, fold_scales, fold_sessions = [], [], [], [], set()
        for index in fit_indices:
            samples, row = training[index], training_rows[index]
            allowed = np.asarray([int(point_id) not in reserved_ids for point_id in samples[:, 6]], bool)
            tree = trees.get(camera_cell(row))
            if tree is not None and len(samples):
                distance, _ = tree.query(_sample_rays(row, samples), k=1, distance_upper_bound=ray_radius)
                allowed &= ~np.isfinite(distance)
            samples = samples[allowed]
            if len(samples) < MIN_SAMPLES:
                continue
            scalar = weighted_median(samples[:, 0], samples[:, 1])
            w = adjusted_weights[index][allowed]
            fold_samples.append(samples)
            fold_targets.append(samples[:, 0] - scalar)
            fold_weights.append(w / w.sum() * adjusted_weights[index].sum())
            fold_scales.append(scalar)
            fold_sessions.add(row["session_id"])
        inner_fit = np.vstack(fold_samples) if fold_samples else np.empty((0, 9))
        if len(inner_fit) < MIN_SAMPLES or len(check) < 10:
            continue
        w, y = np.concatenate(fold_weights), np.concatenate(fold_targets)
        inner_center = weighted_median(inner_fit[:, 5], w)
        c = _fit_correction(_basis(inner_fit, inner_center), y, w)
        inner_scale = float(np.median(fold_scales))
        inner.append({"fold": fold, "fit_count": len(inner_fit), "check_sessions": sorted(check_sessions),
                      "fit_sessions": sorted(fold_sessions),
                      "fit_landmark_ids_sha256": _ids_digest({int(point_id) for point_id in inner_fit[:, 6]}),
                      "heldout_landmark_ids_sha256": _ids_digest(reserved_ids),
                      "validation": _tail(check[:, 0] - inner_scale - _basis(check, inner_center) @ c)})
    fit_ids = {int(point_id) for point_id in train[:, 6]}
    if fit_ids & heldout_ids:
        raise ValueError("MAPPING calibration landmark holdout leaked into fitting")
    fit_residuals = np.concatenate([
        samples[:, 0] - proposed_scales[row["name"]] - _basis(samples, center) @ coeff if accepted else target
        for samples, row, target in zip(training, training_rows, targets)]) if training else np.empty(0)
    artifact = seal_depth_calibration({
        "schema": CALIBRATION_SCHEMA, "basis": list(CALIBRATION_BASIS),
        "policy_sha256": geometry_policy_digest(), "evaluation_role": "MAPPING",
        "selection_data": "MAPPING_INTERNAL_BLOCKED_GROUPS", "accepted": bool(accepted),
        "coefficients": coeff.tolist() if accepted else [0., 0., 0.],
        "candidate_coefficients": coeff.tolist(), "log_depth_center": center,
        "regularization": .01, "inner_selection": "FROZEN_NOT_TUNED", "inner_grouped_folds": inner,
        "fit_sessions": train_sessions, "blocked_session": blocked, "outer_session": outer,
        "fit_source_group_count": len(group_counts), "fit_landmark_ids_sha256": _ids_digest(fit_ids),
        "heldout_landmark_ids_sha256": _ids_digest(heldout_ids), "heldout_landmark_count": len(heldout_ids),
        "heldout_ray_camera_cell_count": len(ray_trees),
        "fit_image_names_sha256": hashlib.sha256(json.dumps(sorted(row["name"] for row in training_rows),
                                                              separators=(",", ":")).encode()).hexdigest(),
        "evaluation_status": "ACCEPTED" if accepted else "EVALUATED_REJECTED" if evaluated else "UNAVAILABLE_GROUPED_HOLDOUT",
        "reason": ("MAPPING_HELDOUT_TAIL_IMPROVED" if accepted else
                   "MAPPING_HELDOUT_NO_TAIL_IMPROVEMENT_OR_FROZEN_GATE_REJECTED" if evaluated else "INSUFFICIENT_GROUPED_HOLDOUT"),
        "fit_sample_count": len(train), "blocked_before": _tail(np.asarray(blocked_before)),
        "blocked_candidate": _tail(np.asarray(blocked_after)),
        "outer_before": _tail(np.asarray(outer_before)), "outer_candidate": _tail(np.asarray(outer_after)),
        "session_scale_ratio_before": before_spread if math.isfinite(before_spread) else None,
        "session_scale_ratio_candidate": after_spread if math.isfinite(after_spread) else None,
        "fit_bounds": np.vstack((train[:, [3, 4, 5]].min(axis=0), train[:, [3, 4, 5]].max(axis=0))).tolist()
                      if len(train) else None,
        "fit_support_strata": _strata(train, fit_residuals, train),
    })
    eligible_names = {row["name"] for row in eligible}
    for row in output:
        row["depth_calibration"] = artifact
        if row["scale_role"] == "MAPPING_UNSUPPORTED":
            continue
        if row["scale_role"] != "MAPPING_FIT":
            if "diagnostic_log_scale_before" not in row:
                observed = validation_by_name[row["name"]]
                row["diagnostic_log_scale_before"] = (weighted_median(observed[:, 0], observed[:, 1])
                                                       if len(observed) else row.get("log_scale"))
                row["diagnostic_log_scale_candidate"] = (
                    weighted_median(observed[:, 0] - _basis(observed, center) @ coeff, observed[:, 1])
                    if len(observed) else row.get("log_scale"))
            row["diagnostic_log_scale"] = row[
                "diagnostic_log_scale_candidate" if accepted else "diagnostic_log_scale_before"]
            row["diagnostic_log_scale_after"] = row["diagnostic_log_scale"]
            if not training_rows:
                row["status"] = "INSUFFICIENT_GROUPED_FIT"
                continue
            row["log_scale"] = global_corrected if accepted else global_baseline
        if row["name"] not in eligible_names:
            continue
        fit = _sample_array(row["calibration_fit_samples"])
        validation = _sample_array(row["calibration_validation_samples"])
        corrected_fit, corrected_validation = fit.copy(), validation.copy()
        if accepted:
            corrected_fit[:, 0] -= _basis(fit, center) @ coeff
            corrected_validation[:, 0] -= _basis(validation, center) @ coeff
            if row["name"] in proposed_scales:
                row["uncorrected_log_scale"] = row["log_scale"]
                row["log_scale"] = proposed_scales[row["name"]]
        log_scale = float(row["log_scale"])
        mad, trends = _shape_diagnostics(corrected_fit, log_scale)
        row.update(shape_mad=mad, shape_trend=trends,
                   fit_residual_strata=_strata(corrected_fit, corrected_fit[:, 0] - log_scale, train),
                   validation_residual_strata=_strata(corrected_validation, corrected_validation[:, 0] - log_scale, train))
        if len(validation) >= 10:
            vmad, vtrends = _shape_diagnostics(corrected_validation, log_scale)
            row.update(validation_shape_mad=vmad, validation_shape_trend=vtrends,
                       validation_log_residual=_spread(corrected_validation[:, 0] - log_scale))
            if row.get("status") in ("SCALE_OK", "SHAPE_CONFLICT"):
                row["status"] = "SHAPE_CONFLICT" if _shape_conflict(mad, trends) or _shape_conflict(vmad, vtrends) else "SCALE_OK"
    return output
