"""COLMAP intrinsics seed: one undistorted PINHOLE camera per resolution.

GlueMap consumes calibrated intrinsics through a COLMAP-format seed
directory (cameras.txt/images.txt/points3D.txt) referenced as
``gt_intrinsics_path``. The seed is generated from the explicit run
calibration rescaled per keyframe resolution, so no stale per-site seed
files are ever reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import Calibration, calibration_matrix_for_resolution


def write_intrinsics_seed(
    selected: Sequence[Mapping[str, Any]],
    *,
    dimensions_by_source: Mapping[str, tuple[int, int]],
    calibration: Calibration,
    output_dir: str | Path,
) -> Path:
    """Write the seed; every image name must be a safe relative path."""
    rows: list[tuple[Mapping[str, Any], tuple[int, int]]] = []
    for row in selected:
        source_id = str(row.get("video_id") or row.get("source_id") or "")
        if source_id not in dimensions_by_source:
            raise ValueError(
                f"missing calibrated dimensions for selected source: {source_id}"
            )
        rows.append((row, dimensions_by_source[source_id]))
    resolutions = sorted({resolution for _, resolution in rows})
    camera_ids = {resolution: index for index, resolution in enumerate(resolutions, 1)}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# Camera list", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for resolution in resolutions:
        width, height = resolution
        matrix = calibration_matrix_for_resolution(
            calibration, target_size=resolution
        )
        camera_lines.append(
            f"{camera_ids[resolution]} PINHOLE {width} {height} "
            f"{float(matrix[0, 0])} {float(matrix[1, 1])} "
            f"{float(matrix[0, 2])} {float(matrix[1, 2])}"
        )
    (root / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    image_lines = [
        "# Image list",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
    ]
    for image_id, (row, resolution) in enumerate(rows, 1):
        name = Path(str(row.get("output_name") or "")).as_posix()
        if not name or name == "." or Path(name).is_absolute() or ".." in Path(
            name
        ).parts:
            raise ValueError("selected keyframe has an invalid output_name")
        image_lines.extend(
            [f"{image_id} 1 0 0 0 0 0 0 {camera_ids[resolution]} {name}", ""]
        )
    (root / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (root / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")
    return root


__all__ = ["write_intrinsics_seed"]
