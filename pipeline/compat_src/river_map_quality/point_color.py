"""Recover sparse-point RGB from mapping images and COLMAP observation tracks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_binary import ColmapImageObservations

ImageLoader = Callable[[Path], np.ndarray]


@dataclass(frozen=True)
class TrackColorizationResult:
    """A recolored point array and auditable coverage statistics."""

    points: np.ndarray
    colored_point_count: int
    sample_count: int
    image_count: int
    method: str = "track_mean_nearest_pixel"


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as uint8 RGB without retaining an open file handle."""

    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _validate_points(points: np.ndarray) -> None:
    names = points.dtype.names or ()
    if "point_id" not in names or "rgb" not in names:
        raise ValueError("point array must contain point_id and rgb fields")
    point_ids = np.asarray(points["point_id"], dtype=np.uint64)
    if len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("point_id values must be unique")


def colorize_points_from_tracks(
    points: np.ndarray,
    observations: Iterable[ColmapImageObservations],
    images_root: Path | str,
    *,
    image_loader: ImageLoader = load_rgb_image,
    require_all: bool = True,
) -> TrackColorizationResult:
    """Assign each point the rounded mean RGB of its valid track pixels."""

    _validate_points(points)
    point_ids = np.asarray(points["point_id"], dtype=np.uint64)
    order = np.argsort(point_ids)
    sorted_ids = point_ids[order]
    color_sums = np.zeros((len(points), 3), dtype=np.uint64)
    sample_counts = np.zeros(len(points), dtype=np.uint32)
    image_count = 0
    root = Path(images_root)
    for image_observations in observations:
        point3d_ids = np.asarray(image_observations.point3d_ids, dtype=np.int64)
        xy = np.asarray(image_observations.xy, dtype=np.float64)
        if xy.shape != (len(point3d_ids), 2):
            raise ValueError(f"invalid observation shape for image {image_observations.name}")
        linked = point3d_ids >= 0
        if not linked.any():
            continue
        linked_ids = point3d_ids[linked].astype(np.uint64)
        linked_xy = xy[linked]
        positions = np.searchsorted(sorted_ids, linked_ids)
        known = positions < len(sorted_ids)
        known_indices = np.flatnonzero(known)
        known[known_indices] &= sorted_ids[positions[known_indices]] == linked_ids[known_indices]
        if not known.any():
            continue
        point_indices = order[positions[known]]
        sample_xy = linked_xy[known]
        image_path = root / image_observations.name
        image = np.asarray(image_loader(image_path))
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"image loader must return uint8 HxWx3 RGB: {image_path}")
        image_count += 1
        pixel_xy = np.floor(sample_xy + 0.5).astype(np.int64)
        finite = np.isfinite(sample_xy).all(axis=1)
        inside = (
            finite
            & (pixel_xy[:, 0] >= 0)
            & (pixel_xy[:, 0] < image.shape[1])
            & (pixel_xy[:, 1] >= 0)
            & (pixel_xy[:, 1] < image.shape[0])
        )
        if not inside.any():
            continue
        valid_indices = point_indices[inside]
        valid_pixels = pixel_xy[inside]
        colors = image[valid_pixels[:, 1], valid_pixels[:, 0]].astype(np.uint64)
        np.add.at(color_sums, valid_indices, colors)
        np.add.at(sample_counts, valid_indices, 1)
    colored = sample_counts > 0
    uncolored_count = int((~colored).sum())
    if require_all and uncolored_count:
        raise ValueError(f"{uncolored_count} of {len(points)} points have no valid image color")
    output = points.copy()
    counts = sample_counts[colored, None].astype(np.uint64)
    output["rgb"][colored] = ((color_sums[colored] + counts // 2) // counts).astype(np.uint8)
    return TrackColorizationResult(
        points=output,
        colored_point_count=int(colored.sum()),
        sample_count=int(sample_counts.sum()),
        image_count=image_count,
    )


__all__ = [
    "TrackColorizationResult",
    "colorize_points_from_tracks",
    "load_rgb_image",
]
