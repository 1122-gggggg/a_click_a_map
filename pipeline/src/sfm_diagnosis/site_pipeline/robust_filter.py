"""Multi-resolution robust COLMAP filtering with fixed-intrinsics BA."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RobustFilterConfig:
    max_reprojection_error_px: float = 3.0
    minimum_triangulation_angle_deg: float = 1.5
    minimum_track_length: int = 3
    bundle_adjustment_iterations: int = 100
    bundle_adjustment_threads: int = 8

    def __post_init__(self) -> None:
        if self.max_reprojection_error_px <= 0:
            raise ValueError("maximum reprojection error must be positive")
        if self.minimum_triangulation_angle_deg < 0:
            raise ValueError("minimum triangulation angle must be non-negative")
        if self.minimum_track_length < 2:
            raise ValueError("minimum track length must be at least two")
        if self.bundle_adjustment_iterations <= 0 or self.bundle_adjustment_threads <= 0:
            raise ValueError("bundle-adjustment iterations and threads must be positive")


def filter_reconstruction(
    reconstruction: Any,
    config: RobustFilterConfig,
    *,
    manager_factory: Callable[[Any], Any] | None = None,
) -> dict[str, int | float]:
    """Apply the 8/24 robust depth, reprojection/angle, and track gates in order."""

    if manager_factory is None:
        import pycolmap

        manager_factory = pycolmap.ObservationManager
    manager = manager_factory(reconstruction)
    points_before = int(reconstruction.num_points3D())
    observations_before = int(reconstruction.compute_num_observations())
    negative_removed = int(manager.filter_observations_with_negative_depth())
    geometry_removed = int(
        manager.filter_all_points3D(
            float(config.max_reprojection_error_px),
            float(config.minimum_triangulation_angle_deg),
        )
    )
    short_removed = int(
        manager.filter_points3D_with_short_tracks(int(config.minimum_track_length))
    )
    return {
        "max_reprojection_error_px": float(config.max_reprojection_error_px),
        "minimum_triangulation_angle_deg": float(config.minimum_triangulation_angle_deg),
        "minimum_track_length": int(config.minimum_track_length),
        "points_before": points_before,
        "points_after": int(reconstruction.num_points3D()),
        "observations_before": observations_before,
        "observations_after": int(reconstruction.compute_num_observations()),
        "negative_depth_observations_removed": negative_removed,
        "geometry_observations_removed": geometry_removed,
        "short_track_observations_removed": short_removed,
    }


def run_fixed_intrinsics_ba(reconstruction: Any, config: RobustFilterConfig) -> float:
    """Refine poses and points while preserving every resolution-specific camera K."""

    import pycolmap

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.print_summary = True
    options.ceres.solver_options.max_num_iterations = int(config.bundle_adjustment_iterations)
    options.ceres.solver_options.num_threads = int(config.bundle_adjustment_threads)
    started = time.time()
    pycolmap.bundle_adjustment(reconstruction, options)
    return time.time() - started


def robust_filter_model(
    input_model: str | Path,
    output_model: str | Path,
    config: RobustFilterConfig,
) -> dict[str, Any]:
    """Filter a copy in memory, run fixed-K BA, re-filter, and write a new model."""

    import pycolmap

    source = Path(input_model).resolve(strict=True)
    output = Path(output_model).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(source))
    cameras_before = {
        int(camera_id): {
            "model": camera.model_name,
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    before = reconstruction_metrics(reconstruction)
    first_filter = filter_reconstruction(reconstruction, config)
    if reconstruction.num_points3D() == 0:
        raise RuntimeError("robust filtering removed every Point3D")
    ba_seconds = run_fixed_intrinsics_ba(reconstruction, config)
    second_filter = filter_reconstruction(reconstruction, config)
    cameras_after = {
        int(camera_id): {
            "model": camera.model_name,
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    if cameras_after != cameras_before:
        raise RuntimeError("fixed-intrinsics BA changed a camera calibration")
    output.mkdir(parents=True)
    reconstruction.write(str(output))
    return {
        "schema_version": 1,
        "artifact_type": "ROBUST_FIXED_INTRINSICS_MODEL",
        "input_model": str(source),
        "output_model": str(output),
        "config": {
            "max_reprojection_error_px": config.max_reprojection_error_px,
            "minimum_triangulation_angle_deg": config.minimum_triangulation_angle_deg,
            "minimum_track_length": config.minimum_track_length,
            "bundle_adjustment_iterations": config.bundle_adjustment_iterations,
            "bundle_adjustment_threads": config.bundle_adjustment_threads,
        },
        "before": before,
        "first_filter": first_filter,
        "bundle_adjustment_seconds": ba_seconds,
        "second_filter": second_filter,
        "after": reconstruction_metrics(reconstruction),
        "cameras": cameras_after,
    }


def reconstruction_metrics(reconstruction: Any) -> dict[str, Any]:
    tracks = np.asarray(
        [len(point.track.elements) for point in reconstruction.points3D.values()], dtype=int
    )
    errors = np.asarray(
        [float(point.error) for point in reconstruction.points3D.values()], dtype=float
    )
    return {
        "registered_images": int(reconstruction.num_reg_images()),
        "cameras": len(reconstruction.cameras),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "track_length_p50": _percentile(tracks, 50),
        "track_length_p90": _percentile(tracks, 90),
        "track_ge3_fraction": None if not len(tracks) else float(np.mean(tracks >= 3)),
        "point_error_p50_px": _percentile(errors, 50),
        "point_error_p90_px": _percentile(errors, 90),
        "point_error_p99_px": _percentile(errors, 99),
    }


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return None if not len(values) else float(np.percentile(values, percentile))


__all__ = [
    "RobustFilterConfig",
    "filter_reconstruction",
    "reconstruction_metrics",
    "robust_filter_model",
    "run_fixed_intrinsics_ba",
]
