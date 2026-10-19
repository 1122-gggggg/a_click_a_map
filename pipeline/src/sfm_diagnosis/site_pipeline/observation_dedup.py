"""Repair COLMAP tracks that contain multiple observations from one image."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def audit_duplicate_image_observations(reconstruction: Any) -> dict[str, Any]:
    """Report raw and unique-image track support without changing the model."""

    raw_lengths: list[int] = []
    unique_lengths: list[int] = []
    duplicate_points = 0
    duplicate_observations = 0
    maximum_image_multiplicity = 1
    for point in reconstruction.points3D.values():
        image_ids = [int(element.image_id) for element in point.track.elements]
        counts: dict[int, int] = defaultdict(int)
        for image_id in image_ids:
            counts[image_id] += 1
        raw_lengths.append(len(image_ids))
        unique_lengths.append(len(counts))
        extras = len(image_ids) - len(counts)
        if extras:
            duplicate_points += 1
            duplicate_observations += extras
            maximum_image_multiplicity = max(maximum_image_multiplicity, max(counts.values()))
    raw = np.asarray(raw_lengths, dtype=np.int64)
    unique = np.asarray(unique_lengths, dtype=np.int64)
    return {
        "points3D": len(raw_lengths),
        "raw_observations": int(raw.sum()),
        "effective_unique_image_observations": int(unique.sum()),
        "duplicate_points": duplicate_points,
        "duplicate_observations": duplicate_observations,
        "duplicate_observation_fraction": (
            0.0 if not int(raw.sum()) else duplicate_observations / int(raw.sum())
        ),
        "maximum_same_image_multiplicity": maximum_image_multiplicity,
        "effective_track_length_min": None if not len(unique) else int(unique.min()),
        "effective_track_length_p50": _percentile(unique, 50),
        "effective_track_length_p90": _percentile(unique, 90),
        "effective_multi_view_track_ratio_ge5": (
            None if not len(unique) else float(np.mean(unique >= 5))
        ),
    }


def deduplicate_model(
    input_model: str | Path,
    output_model: str | Path,
    *,
    minimum_unique_images: int = 3,
) -> dict[str, Any]:
    """Keep the lowest-error observation per Point3D/image and write a copy."""

    if minimum_unique_images < 2:
        raise ValueError("minimum unique-image track length must be at least two")
    source = Path(input_model).expanduser().resolve(strict=True)
    output = Path(output_model).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)

    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(source))
    before = audit_duplicate_image_observations(reconstruction)
    cameras_before = _camera_snapshot(reconstruction)
    poses_before = _pose_snapshot(reconstruction)
    xyz_before = {
        int(point_id): tuple(float(value) for value in point.xyz)
        for point_id, point in reconstruction.points3D.items()
    }

    points_to_delete: list[int] = []
    observations_to_delete: list[tuple[int, int, int]] = []
    plan_rows: list[tuple[int, int, int, tuple[int, ...]]] = []
    for point_id, point in sorted(reconstruction.points3D.items()):
        by_image: dict[int, list[Any]] = defaultdict(list)
        for element in point.track.elements:
            by_image[int(element.image_id)].append(element)
        if len(by_image) < minimum_unique_images:
            points_to_delete.append(int(point_id))
            continue
        for image_id, elements in sorted(by_image.items()):
            if len(elements) < 2:
                continue
            image = reconstruction.images[image_id]
            ranked = sorted(
                (
                    _observation_error(image, point.xyz, int(element.point2D_idx)),
                    int(element.point2D_idx),
                )
                for element in elements
            )
            kept_index = ranked[0][1]
            removed = tuple(index for _, index in ranked[1:])
            plan_rows.append((int(point_id), image_id, kept_index, removed))
            observations_to_delete.extend(
                (int(point_id), image_id, point2d_idx) for point2d_idx in removed
            )

    for point_id in points_to_delete:
        reconstruction.delete_point3D(point_id)
    for point_id, image_id, point2d_idx in observations_to_delete:
        if reconstruction.exists_point3D(point_id):
            reconstruction.delete_observation(image_id, point2d_idx)

    after = audit_duplicate_image_observations(reconstruction)
    if after["duplicate_observations"] != 0:
        raise RuntimeError("duplicate-image observations remain after cleanup")
    if (
        after["effective_track_length_min"] is not None
        and int(after["effective_track_length_min"]) < minimum_unique_images
    ):
        raise RuntimeError("short unique-image tracks remain after cleanup")
    if _camera_snapshot(reconstruction) != cameras_before:
        raise RuntimeError("observation cleanup changed camera intrinsics")
    if _pose_snapshot(reconstruction) != poses_before:
        raise RuntimeError("observation cleanup changed registered image poses")
    surviving_xyz_unchanged = all(
        tuple(float(value) for value in point.xyz) == xyz_before[int(point_id)]
        for point_id, point in reconstruction.points3D.items()
    )
    if not surviving_xyz_unchanged:
        raise RuntimeError("observation cleanup changed surviving Point3D coordinates")

    output.mkdir(parents=True)
    reconstruction.write(str(output))
    plan_digest = hashlib.sha256(repr(plan_rows).encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "artifact_type": "UNIQUE_IMAGE_OBSERVATION_CLEANUP",
        "input_model": str(source),
        "output_model": str(output),
        "minimum_unique_images": minimum_unique_images,
        "before": before,
        "after": after,
        "removed_duplicate_observations": len(observations_to_delete),
        "removed_short_tracks": len(points_to_delete),
        "removed_short_track_point_ids": points_to_delete,
        "affected_point_image_groups": len(plan_rows),
        "plan_sha256": plan_digest,
        "fixed_intrinsics_unchanged": True,
        "registered_poses_unchanged": True,
        "surviving_xyz_unchanged": True,
    }


def _observation_error(image: Any, xyz: Any, point2d_idx: int) -> float:
    if point2d_idx < 0 or point2d_idx >= image.num_points2D():
        return float("inf")
    try:
        projected = image.project_point(np.asarray(xyz, dtype=np.float64))
    except (RuntimeError, ValueError):
        projected = None
    if projected is None:
        return float("inf")
    xy = np.asarray(image.points2D[point2d_idx].xy, dtype=np.float64)
    return float(np.linalg.norm(np.asarray(projected, dtype=np.float64) - xy))


def _camera_snapshot(reconstruction: Any) -> tuple[Any, ...]:
    return tuple(
        (
            int(camera_id),
            str(camera.model_name),
            int(camera.width),
            int(camera.height),
            tuple(float(value) for value in camera.params),
        )
        for camera_id, camera in sorted(reconstruction.cameras.items())
    )


def _pose_snapshot(reconstruction: Any) -> tuple[Any, ...]:
    rows = []
    for image_id, image in sorted(reconstruction.images.items()):
        if not image.has_pose:
            continue
        transform = image.cam_from_world()
        rows.append(
            (
                int(image_id),
                tuple(float(value) for value in transform.rotation.matrix().reshape(-1)),
                tuple(float(value) for value in transform.translation),
            )
        )
    return tuple(rows)


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return None if not len(values) else float(np.percentile(values, percentile))


__all__ = ["audit_duplicate_image_observations", "deduplicate_model"]
