"""Small, dependency-light reader for COLMAP's legacy ``images.bin`` poses.

Only the camera-pose portion is retained.  Point observations are consumed so
the parser can safely advance through the file without loading the (often very
large) observation table into memory.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ColmapImagePose:
    """A COLMAP image pose, with ``rotation`` mapping world to camera."""

    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    quaternion_norm: float
    valid_rigid_transform: bool
    rotation: np.ndarray | None
    center: np.ndarray | None


@dataclass(frozen=True)
class ColmapImageObservations:
    """One COLMAP image name and its ordered 2D-to-3D observation table."""

    image_id: int
    name: str
    xy: np.ndarray
    point3d_ids: np.ndarray


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated COLMAP binary record")
    return data


def _read_struct(handle, fmt: str):
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def _read_name(handle) -> str:
    chunks = bytearray()
    while True:
        byte = _read_exact(handle, 1)
        if byte == b"\0":
            try:
                return chunks.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("invalid UTF-8 image name in images.bin") from exc
        chunks.extend(byte)


def _rotation_from_qvec(qvec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(qvec))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-5:
        raise ValueError(f"invalid quaternion norm {norm!r} in images.bin")
    w, x, y, z = qvec
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_binary_image_poses(
    path: str | Path,
    *,
    strict_quaternion: bool = True,
) -> dict[int, ColmapImagePose]:
    """Read all image poses from a COLMAP little-endian ``images.bin`` file.

    The returned mapping is keyed by COLMAP image ID.  Quaternion order is
    COLMAP's ``(w, x, y, z)``; rotations are world-to-camera and centers are
    ``-R.T @ t``.  Invalid (including non-unit) quaternions are rejected rather
    than silently normalized, since that would conceal a corrupt model.
    """

    poses: dict[int, ColmapImagePose] = {}
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            (image_id,) = _read_struct(handle, "<i")
            qvec = np.asarray(_read_struct(handle, "<4d"), dtype=np.float64)
            tvec = np.asarray(_read_struct(handle, "<3d"), dtype=np.float64)
            (camera_id,) = _read_struct(handle, "<i")
            name = _read_name(handle)
            (num_points,) = _read_struct(handle, "<Q")
            # x, y, and point3D_id are not needed for pose analysis.
            _read_exact(handle, num_points * struct.calcsize("<2dq"))
            quaternion_norm = float(np.linalg.norm(qvec))
            valid_rigid_transform = bool(
                np.isfinite(quaternion_norm) and abs(quaternion_norm - 1.0) <= 1e-5
            )
            if valid_rigid_transform:
                rotation = _rotation_from_qvec(qvec)
                center = -rotation.T @ tvec
            elif strict_quaternion:
                raise ValueError(f"invalid quaternion norm {quaternion_norm!r} in images.bin")
            else:
                rotation = None
                center = None
            poses[image_id] = ColmapImagePose(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
                quaternion_norm=quaternion_norm,
                valid_rigid_transform=valid_rigid_transform,
                rotation=rotation,
                center=center,
            )
    return poses


def iter_binary_image_observations(path: str | Path):
    """Yield image names, 2D coordinates, and point IDs from ``images.bin``."""

    observation_dtype = np.dtype([("xy", "<f8", (2,)), ("point3d_id", "<i8")])
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        for _ in range(count):
            values = _read_struct(handle, "<i4d3di")
            image_id = int(values[0])
            name = _read_name(handle)
            (num_points,) = _read_struct(handle, "<Q")
            observations = np.frombuffer(
                _read_exact(handle, num_points * observation_dtype.itemsize),
                dtype=observation_dtype,
            ).copy()
            yield ColmapImageObservations(
                image_id=image_id,
                name=name,
                xy=observations["xy"],
                point3d_ids=observations["point3d_id"],
            )


def read_binary_points3d(path: str | Path) -> np.ndarray:
    """Read COLMAP's little-endian ``points3D.bin`` into a structured array.

    Track observations are consumed but not retained.  The returned fields
    are ``point_id``, ``xyz``, ``rgb``, ``error``, and ``track_length``.
    """

    dtype = np.dtype(
        [
            ("point_id", "<u8"),
            ("xyz", "<f8", (3,)),
            ("rgb", "u1", (3,)),
            ("error", "<f8"),
            ("track_length", "<u8"),
        ]
    )
    with Path(path).open("rb") as handle:
        (count,) = _read_struct(handle, "<Q")
        points = np.empty(count, dtype=dtype)
        for index in range(count):
            point_id, x, y, z, red, green, blue, error, track_length = _read_struct(
                handle, "<Q3d3BdQ"
            )
            points[index] = (point_id, (x, y, z), (red, green, blue), error, track_length)
            _read_exact(handle, track_length * struct.calcsize("<II"))
    return points
