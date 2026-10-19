"""Frozen offline geometry semantics, calibration and evidence-array validation.

This contract is separate from neural inference identity. Updating it requires
CPU geometry revalidation, never relabeling an existing raw-depth cache.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

GEOMETRY_POLICY_VERSION = "offline-geometry-policy/v5"
LEDGER_SCHEMA = "offline-geometry-ledger/v1"
CALIBRATION_SCHEMA = "blocked-log-depth-calibration/v2"
CALIBRATION_BASIS = ("u_centered", "v_centered", "log_depth_centered")
SCORE_SEMANTICS = "uncalibrated_ranking_heuristic"
LEDGER_DTYPES = {
    "evidence_kind_mask": np.dtype("<u2"),
    "effective_same_model_support": np.dtype("<u2"),
    "reciprocal_photo_support": np.dtype("<u2"),
    "sparse_anchor_support": np.dtype("<u2"),
    "sparse_anchor_conflict": np.dtype("<u2"),
    "conflict_count": np.dtype("<u2"),
    "mean_abs_log_residual": np.dtype("<f4"),
    "p90_abs_log_residual": np.dtype("<f4"),
    "min_usable_angle_deg": np.dtype("<f4"),
    "median_usable_angle_deg": np.dtype("<f4"),
    "max_usable_angle_deg": np.dtype("<f4"),
    "independent_session_count": np.dtype("<u2"),
    "independent_source_group_count": np.dtype("<u2"),
    "edge_quality": np.dtype("<f4"),
    "local_calibration_quality": np.dtype("<f4"),
    "normal_valid": np.dtype("bool"),
    "surface_id": np.dtype("<i8"),
}
# Evidence kinds and observation flags are distinct bitsets.
SAME_MODEL = 1
RECIPROCAL_PHOTO = 2
SPARSE_ANCHOR = 4
FIXED_POSE_MVS = 8
OBS_SOURCE = 1
OBS_DEPTH_AGREE = 2
OBS_PHOTO_AGREE = 4
OBS_SPARSE_AGREE = 64
OBS_CONFLICT = 16
OBS_OCCLUDED = 32
OBS_MVS_AGREE = 8


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def geometry_policy_spec() -> dict[str, Any]:
    return {
        "input_boundary": {"evaluation_role": "MAPPING", "registered_manifest_allowlist": "exact",
                           "native_image_sha256": True, "query_validation_heldout_inputs": False,
                           "native_registered_capture_only": True,
                           "invalid_pose": "UNKNOWN_RETAIN_REGISTERED_IDENTITY"},
        "anchors": {"min_track_views": 3, "max_point_error_px": 2.0,
                    "max_observation_error_px": 2.0, "min_angle_deg": 1.5,
                    "distinct_ray_deg": 0.25, "validate_before_grouping": True,
                    "camera_model": "PINHOLE", "proper_rotation_atol": 1e-6,
                    "bounded_ray_representatives": 16, "sampling": "deterministic_farthest_valid_ray"},
        "calibration": {"basis": list(CALIBRATION_BASIS), "ridge": 0.01,
                        "robust_iterations": 3, "huber_log_depth": 0.08,
                        "grid_width": 4, "log_depth_bins": 3,
                        "heldout_landmark_modulus": 7, "heldout_block_modulus": 7,
                        "temporal_segment_frames": 32, "heldout_temporal_modulus": 7,
                        "heldout_temporal_remainder": 6, "min_fit_samples": 30,
                        "min_validation_samples": 10,
                        "blocked_session": "sorted_second_last",
                        "outer_session": "sorted_last_mapping_session",
                        "min_disjoint_sessions": 3,
                        "heldout_global_landmarks": True, "heldout_camera_cell_ray_deg": 0.25,
                        "reserve_whole_blocked_outer_and_temporal_captures": True,
                        "inner_split": "whole_mapping_sessions_reserved_landmarks_and_rays",
                        "accepted_shape_extrapolation": "unknown_outside_observed_fit_cells",
                        "acceptance": "mapping_blocked_and_outer_p90_strictly_better_median_not_worse",
                        "max_session_scale_ratio": 1.20},
        "dense": {"min_baseline_deg": 2.0, "max_abs_log_depth": 0.08,
                  "min_support": 2, "max_patch_relative_range": 0.20,
                  "max_bilinear_log_span": 0.08,
                  "temporal_segment_frames": 32, "camera_cell_extent_fraction": 0.05,
                  "one_vote_per_capture_group": True,
                  "capture_neighborhood": "same_session_and_same_temporal_segment_or_camera_cell",
                  "modalities_do_not_add_views": True, "mvs_admission": "check_only_no_gate_relaxation",
                  "voxel_default": "off", "voxel_representative": "observed_medoid",
                  "normal_invalid": "zero_with_normal_valid_false"},
        "ledger_schema": LEDGER_SCHEMA, "score_semantics": SCORE_SEMANTICS,
        "ledger_dtypes": {name: dtype.str for name, dtype in LEDGER_DTYPES.items()},
        "evidence_kinds": {"SAME_MODEL": SAME_MODEL, "RECIPROCAL_PHOTO": RECIPROCAL_PHOTO,
                           "SPARSE_ANCHOR": SPARSE_ANCHOR, "FIXED_POSE_MVS": FIXED_POSE_MVS},
        "observation_flags": {"SOURCE": OBS_SOURCE, "DEPTH_AGREE": OBS_DEPTH_AGREE,
                              "PHOTO_AGREE": OBS_PHOTO_AGREE, "MVS_AGREE": OBS_MVS_AGREE,
                              "SPARSE_AGREE": OBS_SPARSE_AGREE, "CONFLICT": OBS_CONFLICT, "OCCLUDED": OBS_OCCLUDED},
        "observation_abi": {"view_image_id": "<i4", "view_camera_center": "<f8",
                            "view_session_index": "<u4", "view_source_index": "<u4",
                            "observation_offsets": "<u8", "observation_view_index": "<u4",
                            "observation_flags": "|u1", "observation_residual": "<f4",
                            "observation_residual_view_index": "<u4",
                            "observation_family_mask": "|u1", "observation_evidence_id": "<i8",
                            "observation_group_index": "<u4",
                            "reference_surface_id": "<i8", "reference_image_id": "<i4",
                            "reference_uv": "<f4",
                            "residual_semantics": "log_candidate_optical_z_over_measured_optical_z",
                            "view_index_semantics": "actual_supporting_capture",
                            "group_index_semantics": "point_local_correlated_capture_group"},
        "authority": "OFFLINE_SUPPLEMENTAL_ONLY",
    }


def geometry_policy_digest() -> str:
    return _digest(geometry_policy_spec())


def calibration_contract() -> dict[str, Any]:
    policy = geometry_policy_spec()
    return {"schema_version": 1, "policy_version": GEOMETRY_POLICY_VERSION,
            "sha256": _digest(policy), "policy": policy}


def validate_calibration_contract(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or dict(value) != calibration_contract():
        raise ValueError("unsupported or malformed offline geometry calibration contract")


def validate_geometry_policy(value: Mapping[str, Any] | str) -> None:
    if isinstance(value, str):
        if value != geometry_policy_digest():
            raise ValueError("unsupported offline geometry policy digest")
    elif dict(value) != geometry_policy_spec():
        raise ValueError("unsupported offline geometry policy")


def validate_mapping_record(row: Mapping[str, Any]) -> None:
    """Reject missing/foreign roles before consuming offline calibration evidence."""
    if not isinstance(row, Mapping) or row.get("evaluation_role") != "MAPPING":
        raise ValueError("offline geometry requires an explicit registered MAPPING capture")
    name, session, digest = row.get("name"), row.get("session_id"), row.get("image_sha256")
    if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name or
            any(part in ("", ".", "..") for part in name.split("/")) or
            not name.lower().endswith((".jpg", ".jpeg")) or
            not isinstance(session, str) or not session or
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or
            type(row.get("frame_index")) is not int or row["frame_index"] < 0):
        raise ValueError("offline geometry requires native MAPPING identity and image SHA256")



def validate_pose(pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    if p.shape == (4, 4):
        if not np.allclose(p[3], [0., 0., 0., 1.], atol=1e-6, rtol=0):
            raise ValueError("improper homogeneous camera pose")
        p = p[:3]
    if (p.shape != (3, 4) or not np.isfinite(p).all() or
            not np.allclose(p[:, :3] @ p[:, :3].T, np.eye(3), atol=1e-6, rtol=0) or
            not math.isclose(float(np.linalg.det(p[:, :3])), 1.0, abs_tol=1e-6)):
        raise ValueError("expected a finite proper camera pose")
    return p


def validate_pinhole(k: np.ndarray, width: int, height: int) -> np.ndarray:
    a = np.asarray(k, dtype=np.float64)
    if (a.shape != (3, 3) or not np.isfinite(a).all() or width < 2 or height < 2 or
            a[0, 0] <= 0 or a[1, 1] <= 0 or
            not np.allclose(a, [[a[0, 0], 0., a[0, 2]], [0., a[1, 1], a[1, 2]],
                                [0., 0., 1.]], atol=1e-10, rtol=0)):
        raise ValueError("expected finite undistorted PINHOLE intrinsics")
    return a


def seal_depth_calibration(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("digest", None)
    result["digest"] = _digest(result)
    validate_depth_calibration(result)
    return result


def validate_depth_calibration(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("missing depth calibration artifact")
    content = dict(value)
    digest = content.pop("digest", None)
    coefficients = np.asarray(content.get("coefficients"), dtype=np.float64)
    if (digest != _digest(content) or content.get("schema") != CALIBRATION_SCHEMA or
            content.get("policy_sha256") != geometry_policy_digest() or
            content.get("basis") != list(CALIBRATION_BASIS) or coefficients.shape != (3,) or
            not np.isfinite(coefficients).all() or
            not isinstance(content.get("accepted"), bool) or
            content.get("evaluation_role") != "MAPPING" or
            content.get("selection_data") != "MAPPING_INTERNAL_BLOCKED_GROUPS" or
            content.get("regularization") != 0.01 or content.get("inner_selection") != "FROZEN_NOT_TUNED" or
            content.get("evaluation_status") not in ("ACCEPTED", "EVALUATED_REJECTED", "UNAVAILABLE_GROUPED_HOLDOUT") or
            (content.get("evaluation_status") == "ACCEPTED") != content["accepted"] or
            not isinstance(content.get("log_depth_center"), (float, int)) or
            not math.isfinite(float(content["log_depth_center"])) or
            (not content["accepted"] and np.any(coefficients != 0.))):
        raise ValueError("malformed or unsupported depth calibration artifact")
    candidate = np.asarray(content.get("candidate_coefficients"), dtype=np.float64)
    fitted_sessions = content.get("fit_sessions")
    if (candidate.shape != (3,) or not np.isfinite(candidate).all() or
            (content["accepted"] and not np.array_equal(candidate, coefficients)) or
            not isinstance(fitted_sessions, list) or
            any(not isinstance(session, str) or not session for session in fitted_sessions) or
            len(set(fitted_sessions)) != len(fitted_sessions)):
        raise ValueError("malformed depth calibration fit provenance")
    if content["evaluation_status"] != "UNAVAILABLE_GROUPED_HOLDOUT":
        blocked, outer = content.get("blocked_session"), content.get("outer_session")
        if (not fitted_sessions or not isinstance(blocked, str) or not isinstance(outer, str) or
                blocked == outer or {blocked, outer} & set(fitted_sessions) or
                type(content.get("fit_sample_count")) is not int or content["fit_sample_count"] < 30 or
                not isinstance(content.get("blocked_before"), Mapping) or
                not isinstance(content.get("outer_before"), Mapping) or
                content["blocked_before"].get("count", 0) < 10 or content["outer_before"].get("count", 0) < 10):
            raise ValueError("depth calibration requires disjoint MAPPING fit and heldout sessions")
    strata = content.get("fit_support_strata")
    bounds = content.get("fit_bounds")
    if bounds is not None:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape != (2, 3) or not np.isfinite(b).all() or np.any(b[0] > b[1]):
            raise ValueError("malformed calibration fit bounds")
    if strata is None:
        if content["accepted"] and np.any(coefficients):
            raise ValueError("accepted shape correction lacks observed fit-cell support")
        return
    if not isinstance(strata, Mapping) or strata.get("grid_width") != 4:
        raise ValueError("malformed calibration fit-cell support")
    edges = np.asarray(strata.get("log_depth_edges"), dtype=np.float64)
    cells = strata.get("cells")
    if (edges.shape != (4,) or not np.isfinite(edges).all() or np.any(edges[1:] < edges[:-1]) or
            not isinstance(cells, list) or len(cells) != 48 or
            any(not isinstance(cell, Mapping) for cell in cells)):
        raise ValueError("malformed calibration fit-cell support")
    identities = set()
    for cell in cells:
        key = tuple(cell.get(name) for name in ("u_cell", "v_cell", "log_depth_bin"))
        count = cell.get("fit_support_count")
        if (any(type(value) is not int for value in key) or
                not (0 <= key[0] < 4 and 0 <= key[1] < 4 and 0 <= key[2] < 3) or
                key in identities or type(count) is not int or count < 0 or
                cell.get("count") != count or cell.get("supported") is not (count > 0)):
            raise ValueError("malformed calibration fit-cell support")
        identities.add(key)
    if sum(cell["fit_support_count"] for cell in cells) != content.get("fit_sample_count"):
        raise ValueError("calibration support cells differ from its actual fit samples")


def apply_depth_calibration(depth: np.ndarray, u_native: np.ndarray, v_native: np.ndarray,
                            calibration: Mapping[str, Any], *, width: int,
                            height: int) -> np.ndarray:
    """Apply the same frozen correction at source, neighbor and publisher QC.

    Coordinates are native undistorted pixels, not centered work pixels. The
    correction is log-depth shape only; per-image scale is divided afterwards.
    """
    validate_depth_calibration(calibration)
    z, u, v = np.broadcast_arrays(np.asarray(depth), np.asarray(u_native), np.asarray(v_native))
    if width <= 0 or height <= 0:
        raise ValueError("calibration image dimensions must be positive")
    if not calibration["accepted"] or not np.any(calibration["coefficients"]):
        return np.asarray(z)
    c = np.asarray(calibration["coefficients"], dtype=np.float64)
    valid = np.isfinite(z) & (z > 0) & np.isfinite(u) & np.isfinite(v)
    valid &= calibration_quality(u, v, z, calibration, width=width, height=height) > 0
    result = np.full(z.shape, np.nan, dtype=np.result_type(z.dtype, np.float32))
    logz = np.log(z[valid]) - float(calibration["log_depth_center"])
    delta = c[0] * (u[valid] / width - .5) + c[1] * (v[valid] / height - .5) + c[2] * logz
    result[valid] = z[valid] * np.exp(-delta)
    return result


def calibration_quality(u: np.ndarray, v: np.ndarray, z: np.ndarray,
                        calibration: Mapping[str, Any], *, width: int, height: int) -> np.ndarray:
    """Map fit support to a diagnostic quality; never a calibrated probability."""
    validate_depth_calibration(calibration)
    u, v, z = np.broadcast_arrays(np.asarray(u), np.asarray(v), np.asarray(z))
    strata, bounds = calibration.get("fit_support_strata"), calibration.get("fit_bounds")
    if strata is None or bounds is None:
        return np.zeros(z.shape, np.float32)
    a = np.column_stack((u.ravel() / width, v.ravel() / height,
                         np.log(np.maximum(z.ravel(), 1e-30))))
    b = np.asarray(bounds, dtype=float)
    supported = np.isfinite(a).all(axis=1) & (z.ravel() > 0) & np.all((a >= b[0]) & (a <= b[1]), axis=1)
    keys = np.column_stack((np.floor(a[:, 0] * 4), np.floor(a[:, 1] * 4),
                            np.searchsorted(strata["log_depth_edges"][1:-1], a[:, 2], side="right")))
    observed = np.zeros(len(a), bool)
    for cell in strata["cells"]:
        if cell["fit_support_count"] > 0:
            observed |= np.all(keys == (cell["u_cell"], cell["v_cell"], cell["log_depth_bin"]), axis=1)
    return (supported & observed).astype(np.float32).reshape(z.shape)


def depth_patch_quality(depth: np.ndarray, mask: np.ndarray, *, radius: int = 1
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center validity, boundary flag and local coherent-patch quality.

    Boundary points are retained as candidates, not eroded or hole-filled. They
    need independent geometric support before publication and have invalid
    normals if the full tangent neighborhood crosses layers.
    """
    z = np.asarray(depth)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(z) & (z > 0)
    if z.ndim != 2 or valid.shape != z.shape or radius < 1:
        raise ValueError("depth/mask must be aligned 2D arrays")
    size = 2 * radius + 1
    low = minimum_filter(np.where(valid, z, np.inf), size=size, mode="constant", cval=np.inf)
    high = maximum_filter(np.where(valid, z, -np.inf), size=size, mode="constant", cval=-np.inf)
    complete = minimum_filter(valid, size=size, mode="constant", cval=False)
    span = np.divide(high - low, z, out=np.full(z.shape, np.inf, np.float32), where=valid)
    boundary = valid & (~complete | (span > .20))
    quality = np.where(valid & complete, np.clip(1. - span / .20, 0., 1.), 0.).astype(np.float32)
    return valid, boundary, quality


def validate_evidence_ledger(arrays: Mapping[str, Any], n_points: int,
                             metadata: Mapping[str, Any]) -> None:
    """Fail closed before offline publication or runtime derivation."""
    if metadata.get("ledger_schema") != LEDGER_SCHEMA or metadata.get("score_semantics") != SCORE_SEMANTICS:
        raise ValueError("missing or unsupported evidence ledger schema")
    validate_calibration_contract(metadata.get("calibration_contract"))
    for name, dtype in LEDGER_DTYPES.items():
        if name not in arrays:
            raise ValueError(f"evidence ledger missing {name}")
        a = np.asarray(arrays[name])
        if a.shape != (n_points,) or a.dtype != dtype:
            raise ValueError(f"evidence ledger {name} is misaligned or has wrong dtype")
        if a.dtype.kind == "f" and (not np.isfinite(a).all() or np.any(a < 0)):
            raise ValueError(f"evidence ledger {name} contains invalid values")
    for name, shape, dtype in (("xyz", (n_points, 3), np.dtype("<f4")),
                              ("confidence", (n_points,), np.dtype("<f4")),
                              ("support_count", (n_points,), np.dtype("<u2")),
                              ("source_image_id", (n_points,), np.dtype("<i4"))):
        a = np.asarray(arrays.get(name))
        if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
            raise ValueError(f"malformed evidence ledger geometry {name}")
    if np.any(arrays["confidence"] < 0) or np.any(arrays["confidence"] > 1):
        raise ValueError("evidence ranking heuristic outside [0,1]")
    if np.any(arrays["support_count"] != arrays["independent_source_group_count"]):
        raise ValueError("evidence ledger effective view support mismatch")
    for name in ("effective_same_model_support", "reciprocal_photo_support"):
        if np.any(arrays[name] > arrays["independent_source_group_count"]):
            raise ValueError(f"evidence ledger {name} double-counts capture groups")
    for name, bit in (("effective_same_model_support", SAME_MODEL), ("reciprocal_photo_support", RECIPROCAL_PHOTO),
                      ("sparse_anchor_support", SPARSE_ANCHOR)):
        if np.any((arrays[name] > 0) != ((arrays["evidence_kind_mask"] & bit) != 0)):
            raise ValueError("evidence ledger kind/count mismatch")
    if np.any(arrays["evidence_kind_mask"] & ~np.uint16(15)):
        raise ValueError("unsupported evidence ledger kind")
    if np.any(np.asarray(arrays["surface_id"]) < 0) or len(np.unique(arrays["surface_id"])) != n_points:
        raise ValueError("evidence ledger surface IDs must be unique nonnegative identifiers")
    for field in ("edge_quality", "local_calibration_quality"):
        if np.any(np.asarray(arrays[field]) > 1):
            raise ValueError(f"evidence ledger {field} outside [0,1]")
    if (np.any(arrays["min_usable_angle_deg"] > arrays["median_usable_angle_deg"]) or
            np.any(arrays["median_usable_angle_deg"] > arrays["max_usable_angle_deg"]) or
            np.any(arrays["max_usable_angle_deg"] > 180)):
        raise ValueError("evidence ledger angle summaries are inconsistent")
    normals = np.asarray(arrays.get("normals"))
    nv = np.asarray(arrays["normal_valid"])
    if (normals.shape != (n_points, 3) or normals.dtype != np.float32 or not np.isfinite(normals).all() or
            np.any(normals[~nv] != 0) or
            not np.allclose(np.linalg.norm(normals[nv], axis=1), 1., atol=1e-4)):
        raise ValueError("evidence ledger normal validity/sentinel mismatch")
    specifications = {"view_image_id": (np.dtype("<i4"), 1),
                      "view_camera_center": (np.dtype("<f8"), 2),
                      "observation_offsets": (np.dtype("<u8"), 1),
                      "observation_view_index": (np.dtype("<u4"), 1),
                      "observation_flags": (np.dtype("u1"), 1),
                      "observation_residual": (np.dtype("<f4"), 1),
                      "observation_residual_view_index": (np.dtype("<u4"), 1),
                      "observation_family_mask": (np.dtype("u1"), 1),
                      "observation_evidence_id": (np.dtype("<i8"), 1),
                      "observation_group_index": (np.dtype("<u4"), 1)}
    for name, (dtype, ndim) in specifications.items():
        if name not in arrays or np.asarray(arrays[name]).dtype != dtype or np.asarray(arrays[name]).ndim != ndim:
            raise ValueError(f"malformed observation ledger {name}")
    ids, centers = np.asarray(arrays["view_image_id"]), np.asarray(arrays["view_camera_center"])
    if centers.shape != (len(ids), 3) or not np.isfinite(centers).all() or len(np.unique(ids)) != len(ids):
        raise ValueError("malformed view identity table")
    captures, names = metadata.get("view_capture_identity"), metadata.get("view_names")
    if (not isinstance(captures, list) or not isinstance(names, list) or
            len(captures) != len(ids) or len(names) != len(ids) or len(set(names)) != len(names)):
        raise ValueError("evidence ledger lacks its exact registered MAPPING capture table")
    for image_id, name, capture in zip(ids, names, captures):
        validate_mapping_record(capture)
        if capture["name"] != name or capture.get("image_id") != int(image_id):
            raise ValueError("evidence ledger view differs from registered MAPPING identity")
    unsupported = metadata.get("unsupported_sources", [])
    registered = metadata.get("registered_capture_identity", captures)
    if not isinstance(unsupported, list) or not isinstance(registered, list):
        raise ValueError("malformed unsupported registered geometry source table")
    identity_fields = ("image_id", "name", "session_id", "frame_index", "evaluation_role", "image_sha256")
    full = {}
    for capture in registered:
        validate_mapping_record(capture)
        if type(capture.get("image_id")) is not int or capture["image_id"] in full:
            raise ValueError("duplicate or malformed registered geometry identity")
        full[capture["image_id"]] = {key: capture[key] for key in identity_fields}
    admitted = {int(capture["image_id"]): {key: capture[key] for key in identity_fields} for capture in captures}
    for capture in unsupported:
        validate_mapping_record(capture)
        if (type(capture.get("image_id")) is not int or capture["image_id"] in admitted or
                capture.get("status") != "INVALID_CAMERA_POSE" or
                capture.get("reason") != "IMPROPER_FROZEN_WORLD_TO_CAMERA_POSE"):
            raise ValueError("unsupported source lacks an exact frozen geometry exclusion")
        admitted[capture["image_id"]] = {key: capture[key] for key in identity_fields}
    if admitted != full or len({capture["name"] for capture in registered}) != len(registered):
        raise ValueError("proper views and unsupported sources do not cover the exact registered MAPPING identities")
    for name, vocabulary in (("view_session_index", "view_sessions"), ("view_source_index", "view_sources")):
        a = np.asarray(arrays.get(name))
        labels = metadata.get(vocabulary)
        if (a.shape != (len(ids),) or a.dtype != np.dtype("<u4") or
                not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels) or
                len(set(labels)) != len(labels) or np.any(a >= len(labels))):
            raise ValueError(f"malformed view identity table {name}")
    session_index = np.asarray(arrays["view_session_index"])
    if any(metadata["view_sessions"][int(index)] != capture["session_id"]
           for index, capture in zip(session_index, captures)):
        raise ValueError("view session label differs from the registered capture")
    offsets = np.asarray(arrays["observation_offsets"])
    indices = np.asarray(arrays["observation_view_index"])
    flags = np.asarray(arrays["observation_flags"])
    residual = np.asarray(arrays["observation_residual"])
    residual_view = np.asarray(arrays["observation_residual_view_index"])
    if (offsets.shape != (n_points + 1,) or offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]) or
            offsets[-1] != len(indices) or flags.shape != indices.shape or residual.shape != indices.shape or
            residual_view.shape != indices.shape or np.any(residual_view >= len(ids)) or
            np.any(indices >= len(ids)) or not np.isfinite(residual).all() or
            np.any(flags == 0) or np.any(flags & ~np.uint8(127))):
        raise ValueError("malformed point/view observation CSR")
    if n_points and np.any(np.add.reduceat((flags & OBS_SOURCE) != 0, offsets[:-1].astype(np.intp)) < 1):
        raise ValueError("every surface must retain its source observation")
    family = np.asarray(arrays["observation_family_mask"])
    evidence_id = np.asarray(arrays["observation_evidence_id"])
    group = np.asarray(arrays["observation_group_index"])
    if (family.shape != indices.shape or evidence_id.shape != indices.shape or group.shape != indices.shape or
            np.any(family == 0) or np.any(family & ~np.uint8(15)) or np.any(evidence_id < -1)):
        raise ValueError("malformed observation evidence provenance")
    agreements = (flags & (OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE | OBS_SPARSE_AGREE)) != 0
    if (np.any(agreements & (np.abs(residual) > .08)) or
            np.any(((flags & OBS_CONFLICT) != 0) & (residual >= -.08)) or
            np.any(((flags & OBS_OCCLUDED) != 0) & (residual <= .08))):
        raise ValueError("observation flags contradict their signed depth residual")
    receipts = metadata.get("independent_artifacts", [])
    if (not isinstance(receipts, list) or any(not isinstance(entry, Mapping) or
            entry.get("family") not in ("reciprocal_photo", "fixed_pose_mvs") or
            not isinstance(entry.get("manifest_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", entry["manifest_sha256"]) for entry in receipts) or
            len({entry["family"] for entry in receipts}) != len(receipts)):
        raise ValueError("malformed independent artifact identity")
    receipt_families = {entry["family"] for entry in receipts}
    for flag, kind, receipt in ((OBS_PHOTO_AGREE, RECIPROCAL_PHOTO, "reciprocal_photo"),
                               (OBS_MVS_AGREE, FIXED_POSE_MVS, "fixed_pose_mvs")):
        selected = (family & kind) != 0
        if (np.any(((flags & flag) != 0) & ~selected) or np.any(selected & (evidence_id < 0)) or
                (selected.any() and receipt not in receipt_families)):
            raise ValueError("independent observation lacks verified artifact identity")
    if (np.any(((flags & OBS_DEPTH_AGREE) != 0) & ((family & SAME_MODEL) == 0)) or
            np.any(((flags & OBS_SOURCE) != 0) & ((family & SAME_MODEL) == 0)) or
            np.any(((flags & OBS_SPARSE_AGREE) != 0) & ((family & SPARSE_ANCHOR) == 0)) or
            np.any(((family & SPARSE_ANCHOR) != 0) & (evidence_id < 0)) or
            np.any(((family & (SAME_MODEL | SPARSE_ANCHOR)) != 0) & (residual_view != indices))):
        raise ValueError("observation modality differs from its evidence family/camera")
    cell_width = metadata.get("camera_cell_width", max(float(np.linalg.norm(np.ptp(centers, axis=0))) * .05, 1e-8)
                              if len(centers) else 1.)
    if not isinstance(cell_width, (float, int)) or not math.isfinite(cell_width) or cell_width <= 0:
        raise ValueError("malformed view correlation camera cells")
    cells = np.floor(centers / cell_width).astype(np.int64)
    source_ids = np.asarray(arrays["source_image_id"])
    if not np.isin(source_ids, ids).all():
        raise ValueError("surface source identity is absent from the registered capture table")
    geometry_bits = OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE | OBS_CONFLICT | OBS_OCCLUDED
    agreement_bits = OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE
    for point in range(n_points):
        start, stop = offsets[point:point + 2].astype(np.intp)
        views, point_flags = indices[start:stop], flags[start:stop]
        point_family, point_groups = family[start:stop], group[start:stop]
        point_residual = residual[start:stop]
        source = int(np.flatnonzero(ids == source_ids[point])[0])
        if not np.any((views == source) & ((point_flags & OBS_SOURCE) != 0)):
            raise ValueError("surface source differs from its actual SOURCE observation")
        geometric = ((point_flags & geometry_bits) != 0) & (
            (point_family & np.uint8(SAME_MODEL | RECIPROCAL_PHOTO | FIXED_POSE_MVS)) != 0)
        geometry_rows = np.flatnonzero(geometric)
        directions = np.asarray(arrays["xyz"][point], float) - centers[views]
        lengths = np.linalg.norm(directions, axis=1)
        source_ray = np.asarray(arrays["xyz"][point], float) - centers[source]
        source_length = np.linalg.norm(source_ray)
        if source_length <= 0 or np.any(lengths[geometry_rows] <= 0):
            raise ValueError("observation has no usable world ray")
        directions = directions / np.maximum(lengths[:, None], 1e-12)
        ray_angles = np.degrees(np.arccos(np.clip(directions @ (source_ray / source_length), -1., 1.)))
        geometric &= ray_angles >= 2. - 1e-6
        geometry_rows = np.flatnonzero(geometric)
        for ordinal, row in enumerate(geometry_rows):
            for other in geometry_rows[:ordinal]:
                a, b = int(views[row]), int(views[other])
                correlated = (np.dot(directions[row], directions[other]) >= math.cos(math.radians(2.)) or
                    (captures[a]["session_id"] == captures[b]["session_id"] and
                     (captures[a]["frame_index"] // 32 == captures[b]["frame_index"] // 32 or
                      np.array_equal(cells[a], cells[b]))))
                if correlated and point_groups[row] != point_groups[other]:
                    raise ValueError("observation groups double-count correlated captures/modalities")
        counts = {OBS_DEPTH_AGREE: 0, OBS_PHOTO_AGREE: 0, OBS_MVS_AGREE: 0}
        supporting, conflicts, occluded, agreeing_sessions, group_residuals = 0, 0, 0, set(), []
        for identity in np.unique(point_groups[geometry_rows]):
            members = geometric & (point_groups == identity)
            values = point_residual[members]
            center_residual = float(np.median(np.unique(values)))
            group_residuals.append(abs(center_residual))
            member_flags = point_flags[members]
            conflicts += int(np.any(member_flags & OBS_CONFLICT))
            occluded += int(np.any(member_flags & OBS_OCCLUDED))
            agreement = members & ((point_flags & agreement_bits) != 0)
            if not agreement.any() or abs(center_residual) > .08:
                continue
            supporting += 1
            for bit in counts:
                counts[bit] += int(np.any(point_flags[agreement] & bit))
            representative = min(np.flatnonzero(agreement), key=lambda i: (
                abs(float(point_residual[i])), captures[int(views[i])]["session_id"],
                captures[int(views[i])]["frame_index"] // 32, tuple(cells[int(views[i])]),
                int(ids[views[i]])))
            agreeing_sessions.add(int(session_index[views[representative]]))
        sparse_agree = (point_flags & OBS_SPARSE_AGREE) != 0
        sparse_bad = ((point_family & SPARSE_ANCHOR) != 0) & ((point_flags & (OBS_CONFLICT | OBS_OCCLUDED)) != 0)
        sparse_support = len(np.unique(evidence_id[start:stop][sparse_agree]))
        sparse_conflict = len(np.unique(evidence_id[start:stop][sparse_bad]))
        expected = {"support_count": supporting, "independent_source_group_count": supporting,
                    "effective_same_model_support": counts[OBS_DEPTH_AGREE],
                    "reciprocal_photo_support": counts[OBS_PHOTO_AGREE],
                    "independent_session_count": len(agreeing_sessions),
                    "sparse_anchor_support": sparse_support, "sparse_anchor_conflict": sparse_conflict,
                    "conflict_count": conflicts + sparse_conflict}
        for name, value in (("mvs_support", counts[OBS_MVS_AGREE]), ("occluded_count", occluded)):
            if name in arrays:
                expected[name] = value
        for name, value in expected.items():
            if int(arrays[name][point]) != value:
                raise ValueError(f"evidence ledger {name} differs from its actual observation groups")
        if bool(arrays["evidence_kind_mask"][point] & FIXED_POSE_MVS) != (counts[OBS_MVS_AGREE] > 0):
            raise ValueError("evidence ledger MVS kind differs from its actual observation groups")
        mean = float(np.mean(group_residuals)) if group_residuals else 0.
        p90 = float(np.quantile(group_residuals, .9)) if group_residuals else 0.
        if (not math.isclose(float(arrays["mean_abs_log_residual"][point]), mean, abs_tol=1e-6) or
                not math.isclose(float(arrays["p90_abs_log_residual"][point]), p90, abs_tol=1e-6)):
            raise ValueError("evidence ledger residual summaries differ from real observation groups")
    for name, dtype, suffix in (("reference_surface_id", np.dtype("<i8"), ()),
                                ("reference_image_id", np.dtype("<i4"), ()),
                                ("reference_uv", np.dtype("<f4"), (2,))):
        a = np.asarray(arrays.get(name))
        count = len(np.asarray(arrays.get("reference_surface_id", [])))
        if a.dtype != dtype or a.shape != (count, *suffix):
            raise ValueError(f"malformed retained reference observation {name}")
    if (not np.isfinite(arrays["reference_uv"]).all() or
            not np.isin(arrays["reference_surface_id"], arrays["surface_id"]).all() or
            not np.isin(arrays["reference_image_id"], ids).all()):
        raise ValueError("retained reference observations contain invalid identities")
    reference_by_surface: dict[int, set[int]] = {}
    for surface, view in zip(arrays["reference_surface_id"], arrays["reference_image_id"]):
        reference_by_surface.setdefault(int(surface), set()).add(int(view))
    for point, (surface, source) in enumerate(zip(arrays["surface_id"], source_ids)):
        retained = reference_by_surface.get(int(surface), set())
        if int(source) not in retained:
            raise ValueError("surface lacks its retained native source reference observation")
        start, stop = offsets[point:point + 2].astype(np.intp)
        usable = ((flags[start:stop] & (OBS_SOURCE | agreement_bits | OBS_SPARSE_AGREE)) != 0) & (
            (flags[start:stop] & (OBS_CONFLICT | OBS_OCCLUDED)) == 0)
        actual = set(map(int, ids[indices[start:stop][usable]]))
        if not retained <= actual:
            raise ValueError("retained reference is not an actual compatible observation")
