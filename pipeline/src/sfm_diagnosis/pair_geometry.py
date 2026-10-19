from __future__ import annotations

import struct

import numpy as np
from scipy.spatial.transform import Rotation

from .models import MapData

COLMAP_DEGENERATE = 1
COLMAP_PLANAR = 4
COLMAP_PANORAMIC = 5
COLMAP_PLANAR_OR_PANORAMIC = 6
PLANAR_CONFIGS = {COLMAP_PLANAR, COLMAP_PANORAMIC, COLMAP_PLANAR_OR_PANORAMIC}


def pair_model_flags(
    row: dict,
    *,
    rot_err_deg: float | None = None,
    trans_err: float | None = None,
    min_h: float = 0.60,
    max_e: float = 0.30,
    max_rel_rot_deg: float = 15.0,
    max_rel_dir: float = 0.35,
) -> dict:
    """Classify one pair as planar, degenerate, or pose-inconsistent."""
    config = _int_or_none(row.get("two_view_config"))
    h_support = _float_or_none(row.get("homography_support"))
    e_support = _float_or_none(row.get("essential_support"))
    planar_config = config in PLANAR_CONFIGS
    planar_support = (
        h_support is not None
        and e_support is not None
        and h_support >= min_h
        and e_support <= max_e
    )
    pose_bad = False
    if rot_err_deg is not None or trans_err is not None:
        pose_bad = (rot_err_deg is not None and rot_err_deg > max_rel_rot_deg) or (
            trans_err is not None and trans_err > max_rel_dir
        )
    return {
        "planar": bool(planar_config or planar_support),
        "degenerate": config == COLMAP_DEGENERATE,
        "pose_bad": bool(pose_bad),
        "has_model": config is not None or (h_support is not None and e_support is not None),
        "has_pose": rot_err_deg is not None or trans_err is not None,
    }


def reconstructed_relative_error(map_data: MapData, row: dict) -> tuple[float, float] | None:
    """Compare reconstructed relative pose to an optional two-view pose.

    Returns ``(rot_err_deg, trans_err)`` where ``trans_err = 1 - |t_map · t_tv|``.
    """
    endpoints = _pair_indices(map_data, row)
    if endpoints is None:
        return None
    i, j = endpoints
    r_map, t_map = _map_relative(map_data, i, j)
    r_tv, t_tv = two_view_pose(row)
    if r_tv is None or t_tv is None:
        return None
    relative = r_tv.T @ r_map
    rot_err_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)))
    )
    t_tv_n = _unit(t_tv)
    if t_tv_n is None:
        return None
    trans_err = float(1.0 - abs(float(np.dot(t_map, t_tv_n))))
    return rot_err_deg, trans_err


def two_view_pose(row: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    qvec = _qvec_from_row(row)
    tvec = _tvec_from_row(row)
    if qvec is None or tvec is None:
        return None, None
    rotation = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    return rotation, np.asarray(tvec, dtype=float).reshape(3)


def parse_colmap_qvec(value) -> np.ndarray | None:
    return _parse_vec(value, 4)


def parse_colmap_tvec(value) -> np.ndarray | None:
    return _parse_vec(value, 3)


def _map_relative(map_data: MapData, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    r_i = map_data.image_R_wc[i]
    r_j = map_data.image_R_wc[j]
    r_ij = r_j.T @ r_i
    t_ij = r_j.T @ (map_data.image_centers[i] - map_data.image_centers[j])
    t_unit = _unit(t_ij)
    if t_unit is None:
        t_unit = np.zeros(3, dtype=float)
    return r_ij, t_unit


def _pair_indices(map_data: MapData, row: dict) -> tuple[int, int] | None:
    name_index = {name: k for k, name in enumerate(map_data.image_names)}
    id_index = {int(v): k for k, v in enumerate(map_data.image_ids.tolist())}
    i = _endpoint(row, "i", name_index, id_index)
    j = _endpoint(row, "j", name_index, id_index)
    if i is None or j is None:
        return None
    return i, j


def _endpoint(row: dict, suffix: str, name_index: dict, id_index: dict) -> int | None:
    name = row.get(f"image_{suffix}")
    if name in name_index:
        return name_index[name]
    image_id = row.get(f"image_id_{suffix}")
    if image_id is not None:
        return id_index.get(int(image_id))
    return None


def _qvec_from_row(row: dict) -> np.ndarray | None:
    if row.get("two_view_qvec") is not None:
        parsed = parse_colmap_qvec(row["two_view_qvec"])
        if parsed is not None:
            return parsed
    keys = ("two_view_qw", "two_view_qx", "two_view_qy", "two_view_qz")
    if all(row.get(k) is not None for k in keys):
        return np.asarray([row[k] for k in keys], dtype=float)
    return None


def _tvec_from_row(row: dict) -> np.ndarray | None:
    if row.get("two_view_tvec") is not None:
        parsed = parse_colmap_tvec(row["two_view_tvec"])
        if parsed is not None:
            return parsed
    keys = ("two_view_tx", "two_view_ty", "two_view_tz")
    if all(row.get(k) is not None for k in keys):
        return np.asarray([row[k] for k in keys], dtype=float)
    return None


def _parse_vec(value, size: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (bytes, memoryview, bytearray)):
        raw = bytes(value)
        if len(raw) < 8 * size:
            return None
        return np.asarray(struct.unpack("<" + "d" * size, raw[: 8 * size]), dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [p for p in text.replace(";", " ").replace(",", " ").split() if p]
        if len(parts) != size:
            return None
        return np.asarray([float(p) for p in parts], dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if len(arr) != size:
        return None
    return arr


def _unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        return None
    return value / norm


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
