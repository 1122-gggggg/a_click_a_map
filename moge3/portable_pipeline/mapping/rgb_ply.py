"""Original-RGB sparse point cloud export.

Samples each reconstructed 3D point's color as the mean of its decoded
track-observation pixels from the source keyframes. No display or risk
recoloring is applied. Points whose observations all fall outside their
image bounds have no pixel to sample; only a negligible fraction is
tolerated (tracker extrapolation past the border), anything larger is a
systemic failure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_jsonl, sha256_file

MAX_UNSAMPLED_POINT_FRACTION = 1e-4


def sample_model_rgb(
    model_dir: str | Path,
    keyframes_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import cv2
    import pycolmap

    keyframes = {
        str(row["output_name"]): row for row in read_jsonl(Path(keyframes_path))
    }
    reconstruction = pycolmap.Reconstruction(str(Path(model_dir).resolve(strict=True)))
    return sample_reconstruction_rgb(
        reconstruction,
        keyframes,
        image_loader=lambda path: cv2.imread(path, cv2.IMREAD_COLOR),
    )


def sample_reconstruction_rgb(
    reconstruction: Any,
    keyframes: Mapping[str, Mapping[str, Any]],
    *,
    image_loader: Callable[[str], Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Average decoded RGB observations for every 3D point, no recoloring."""
    point_rows = sorted(reconstruction.points3D.items(), key=lambda item: int(item[0]))
    point_ids = np.asarray([int(pid) for pid, _ in point_rows], dtype=np.int64)
    xyz = np.asarray([p.xyz for _, p in point_rows], dtype=np.float64).reshape(-1, 3)
    sums = np.zeros((len(point_rows), 3), dtype=np.uint64)
    counts = np.zeros(len(point_rows), dtype=np.uint32)
    observations = 0
    for image in sorted(reconstruction.images.values(), key=lambda v: str(v.name)):
        name = str(image.name)
        keyframe = keyframes.get(name)
        if keyframe is None:
            raise RuntimeError(f"registered image is absent from keyframes: {name}")
        pixels = image_loader(str(keyframe["image_uri"]))
        if pixels is None:
            raise FileNotFoundError(str(keyframe["image_uri"]))
        height, width = pixels.shape[:2]
        material = [pt for pt in image.points2D if pt.has_point3D()]
        if not material:
            continue
        ids = np.asarray([int(pt.point3D_id) for pt in material], dtype=np.int64)
        xy = np.asarray([pt.xy for pt in material], dtype=np.float64).reshape(-1, 2)
        columns = np.rint(xy[:, 0]).astype(np.int64)
        rows = np.rint(xy[:, 1]).astype(np.int64)
        indexes = np.searchsorted(point_ids, ids)
        valid = (
            (indexes < len(point_ids))
            & (point_ids[np.minimum(indexes, len(point_ids) - 1)] == ids)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        if not np.any(valid):
            continue
        indexes = indexes[valid]
        bgr = np.asarray(pixels[rows[valid], columns[valid], :3], dtype=np.uint8)
        np.add.at(sums, indexes, bgr[:, ::-1].astype(np.uint64))
        np.add.at(counts, indexes, 1)
        observations += int(len(indexes))
    missing = counts == 0
    colors = np.zeros((len(point_rows), 3), dtype=np.uint8)
    colored = ~missing
    colors[colored] = np.rint(sums[colored] / counts[colored, None]).astype(np.uint8)
    stats: dict[str, Any] = {
        "colored_points": int(colored.sum()),
        "missing_color_points": int(missing.sum()),
        "observations": observations,
    }
    if missing.any():
        stats["missing_point_indices"] = np.flatnonzero(missing).tolist()
        stats["missing_point_ids"] = point_ids[missing].tolist()
    return xyz, colors, stats


def export_original_rgb_ply(
    model_dir: str | Path,
    keyframes_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export frame-sampled RGB PLY; return the machine-readable receipt."""
    xyz, rgb, stats = sample_model_rgb(model_dir, keyframes_path)
    missing_count = int(stats["missing_color_points"])
    dropped: dict[str, Any] = {}
    if missing_count:
        if missing_count > MAX_UNSAMPLED_POINT_FRACTION * len(xyz):
            raise RuntimeError(
                f"RGB sampling missed {missing_count} of {len(xyz)} points"
            )
        keep = np.ones(len(xyz), dtype=bool)
        keep[np.asarray(stats["missing_point_indices"], dtype=np.int64)] = False
        xyz, rgb = xyz[keep], rgb[keep]
        dropped = {
            "dropped_unsampled_point_ids": list(stats["missing_point_ids"]),
            "dropped_reason": "all observations outside image bounds",
        }
    output = write_binary_ply(
        output_path, xyz, rgb,
        comments=("mean decoded RGB over source-frame track observations; no recoloring",),
    )
    return {
        "schema_version": 1,
        "artifact_type": "ORIGINAL_RGB_PLY_RECEIPT",
        "path": str(output),
        "vertices": int(len(xyz)),
        **{k: v for k, v in stats.items() if not k.startswith("missing_point_")},
        **dropped,
        "aggregation": "MEAN_TRACK_OBSERVATION_RGB",
        "sha256": sha256_file(output),
    }


def write_binary_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    comments: tuple[str, ...] = (),
) -> Path:
    """Write a packed little-endian RGB PLY readable by CloudCompare."""
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("xyz and rgb must have the same length")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0"]
    header.extend(f"comment {comment}" for comment in comments)
    header.extend([
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ])
    vertex_dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    if vertex_dt.itemsize != 15:
        raise RuntimeError("PLY vertex record must be 15 bytes")
    records = np.empty(len(points), dtype=vertex_dt)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]
    with dest.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(records.tobytes(order="C"))
    return dest


__all__ = [
    "MAX_UNSAMPLED_POINT_FRACTION",
    "export_original_rgb_ply",
    "sample_model_rgb",
    "sample_reconstruction_rgb",
    "write_binary_ply",
]
