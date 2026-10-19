"""Restore COLMAP point RGB by sampling the original mapping observations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from sfm_diagnosis.models import MapData


def colorize_map_from_observations(
    map_data: MapData,
    images_root: str | Path,
    *,
    reconstruction=None,
    image_loader: Callable[[Path], np.ndarray | None] | None = None,
) -> dict:
    """Replace black/missing point colors with mean observed image RGB.

    Each registered image is loaded once.  Its reconstructed 2D observations are
    sampled with nearest-pixel lookup and accumulated by Point3D ID.  Coordinates,
    tracks, poses, and all diagnosis metrics remain unchanged.
    """

    root = Path(images_root)
    if reconstruction is None:
        try:
            import pycolmap
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyCOLMAP is required for observation colorization") from exc
        model_dir = map_data.metadata.get("model_dir")
        if not model_dir:
            raise ValueError("map metadata has no model_dir for observation colorization")
        reconstruction = pycolmap.Reconstruction(str(model_dir))
    loader = image_loader or _load_rgb
    point_lookup = {
        int(point_id): index for index, point_id in enumerate(map_data.point_ids.tolist())
    }
    sums = np.zeros((map_data.num_points, 3), dtype=np.uint64)
    counts = np.zeros(map_data.num_points, dtype=np.uint32)
    missing_images: list[str] = []
    processed_images = 0
    sampled_observations = 0
    images = sorted(reconstruction.images.values(), key=lambda image: str(image.name))
    for image in images:
        relative = str(image.name)
        try:
            pixels = loader(root / relative)
        except (FileNotFoundError, OSError):
            pixels = None
        if pixels is None:
            missing_images.append(relative)
            continue
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[:, :, None], 3, axis=2)
        if pixels.ndim != 3 or pixels.shape[2] < 3:
            raise ValueError(f"mapping image has invalid pixel shape: {relative} {pixels.shape}")
        observations = [point for point in image.points2D if point.has_point3D()]
        if not observations:
            processed_images += 1
            continue
        point_indices = np.asarray(
            [point_lookup.get(int(point.point3D_id), -1) for point in observations],
            dtype=np.int64,
        )
        xy = np.asarray([point.xy for point in observations], dtype=float).reshape(-1, 2)
        valid = point_indices >= 0
        valid &= np.isfinite(xy).all(axis=1)
        if not np.any(valid):
            processed_images += 1
            continue
        point_indices = point_indices[valid]
        xy = xy[valid]
        x = np.clip(np.rint(xy[:, 0]).astype(int), 0, pixels.shape[1] - 1)
        y = np.clip(np.rint(xy[:, 1]).astype(int), 0, pixels.shape[0] - 1)
        rgb = pixels[y, x, :3].astype(np.uint64)
        np.add.at(sums, point_indices, rgb)
        np.add.at(counts, point_indices, 1)
        sampled_observations += len(point_indices)
        processed_images += 1
    colored = counts > 0
    restored = map_data.point_rgb.copy()
    restored[colored] = np.clip(
        np.rint(sums[colored] / counts[colored, None]), 0, 255
    ).astype(np.uint8)
    map_data.point_rgb = restored
    return {
        "schema_version": 1,
        "artifact_type": "TRACK_OBSERVATION_RGB_COLORIZATION",
        "images_root": str(root),
        "registered_images": len(images),
        "processed_images": processed_images,
        "missing_images": missing_images,
        "sampled_observations": sampled_observations,
        "colored_points": int(np.sum(colored)),
        "uncolored_points": int(np.sum(~colored)),
        "color_source": "mean nearest-pixel RGB over registered mapping observations",
    }


def _load_rgb(path: Path) -> np.ndarray | None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the video extra for RGB colorization") from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return None if bgr is None else bgr[:, :, ::-1]


__all__ = ["colorize_map_from_observations"]
