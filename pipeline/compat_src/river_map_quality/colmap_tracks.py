"""Dependency-light COLMAP point-track index for read-only grouping diagnostics."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PointTrack:
    point_id: int
    xyz: tuple[float, float, float]
    error: float
    track_length: int
    image_ids: tuple[int, ...]


def _read_exact(stream, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("truncated COLMAP points3D.bin")
    return value


def read_point_track_index(path: str | Path) -> dict[int, PointTrack]:
    """Read point coordinates and observer image IDs without PyCOLMAP."""

    output = {}
    with Path(path).open("rb") as stream:
        (count,) = struct.unpack("<Q", _read_exact(stream, struct.calcsize("<Q")))
        for _ in range(count):
            values = struct.unpack("<Q3d3BdQ", _read_exact(stream, struct.calcsize("<Q3d3BdQ")))
            point_id = int(values[0])
            track_length = int(values[-1])
            image_ids = []
            for _ in range(track_length):
                image_id, _point2d_index = struct.unpack(
                    "<II", _read_exact(stream, struct.calcsize("<II"))
                )
                image_ids.append(int(image_id))
            output[point_id] = PointTrack(
                point_id=point_id,
                xyz=(float(values[1]), float(values[2]), float(values[3])),
                error=float(values[-2]),
                track_length=track_length,
                image_ids=tuple(image_ids),
            )
    return output
