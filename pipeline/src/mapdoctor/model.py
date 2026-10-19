from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class Observation2D:
    x: float
    y: float
    point3d_id: int | None


@dataclass(frozen=True)
class ImageRecord:
    id: int
    camera_id: int
    name: str
    center: tuple[float, float, float]
    viewing_direction: tuple[float, float, float]
    observations: tuple[Observation2D, ...] = ()


@dataclass(frozen=True)
class TrackElement:
    image_id: int
    point2d_idx: int


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: tuple[float, float, float]
    rgb: tuple[int, int, int]
    error: float
    track: tuple[TrackElement, ...] = ()


@dataclass
class MapModel:
    source: str
    format: str
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, ImageRecord] = field(default_factory=dict)
    points3d: dict[int, Point3D] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def registered_images(self) -> int:
        return len(self.images)

    @property
    def num_points3d(self) -> int:
        return len(self.points3d)
