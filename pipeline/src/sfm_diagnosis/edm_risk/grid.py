from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from sfm_diagnosis.heatmap import yaw_pitch_rotation
from sfm_diagnosis.models import MapData, Pose


@dataclass(frozen=True)
class SpatialGridConfig:
    voxel_size: float = 1.0
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    padding: float = 0.0
    yaw_step_deg: float = 30.0
    pitch_values_deg: tuple[float, ...] = (-30.0, 0.0, 30.0)
    max_waypoints: int = 25_000
    max_landmark_distance: float | None = None
    occupancy_radius: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0.0:
            raise ValueError("voxel_size must be finite and > 0")
        if not np.isfinite(self.yaw_step_deg) or not 0.0 < self.yaw_step_deg <= 360.0:
            raise ValueError("yaw_step_deg must be finite and in (0, 360]")
        if not self.pitch_values_deg:
            raise ValueError("pitch_values_deg must not be empty")
        if any(not np.isfinite(value) or not -90.0 <= value <= 90.0 for value in self.pitch_values_deg):
            raise ValueError("pitch values must be finite and in [-90, 90]")
        if self.max_waypoints < 1:
            raise ValueError("max_waypoints must be >= 1")
        if self.max_landmark_distance is not None and (
            not np.isfinite(self.max_landmark_distance)
            or self.max_landmark_distance <= 0.0
        ):
            raise ValueError("max_landmark_distance must be finite and > 0")
        if self.occupancy_radius is not None and (
            not np.isfinite(self.occupancy_radius) or self.occupancy_radius <= 0.0
        ):
            raise ValueError("occupancy_radius must be finite and > 0")


@dataclass(frozen=True)
class SpatialPoseSample:
    waypoint_id: int
    position: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float

    @property
    def pose(self) -> Pose:
        return Pose(
            np.asarray(self.position, dtype=float),
            yaw_pitch_rotation(self.yaw_deg, self.pitch_deg),
        )


@dataclass(frozen=True)
class SpatialPoseGrid:
    config: SpatialGridConfig
    positions: np.ndarray
    samples: tuple[SpatialPoseSample, ...]

    @property
    def num_positions(self) -> int:
        return int(len(self.positions))

    @property
    def num_pose_samples(self) -> int:
        return int(len(self.samples))

    def __iter__(self) -> Iterator[SpatialPoseSample]:
        return iter(self.samples)

    @classmethod
    def from_map(
        cls, map_data: MapData, config: SpatialGridConfig
    ) -> "SpatialPoseGrid":
        if config.bounds is None:
            if map_data.num_images:
                lo = np.min(map_data.image_centers, axis=0) - config.padding
                hi = np.max(map_data.image_centers, axis=0) + config.padding
            else:
                lo, hi = map_data.bounds
            resolved = SpatialGridConfig(
                **{
                    **config.__dict__,
                    "bounds": (tuple(lo.tolist()), tuple(hi.tolist())),
                }
            )
        else:
            resolved = config
        grid = cls.from_config(resolved)
        if resolved.occupancy_radius is None:
            return grid
        return grid.occupied(map_data, resolved.occupancy_radius)

    @classmethod
    def from_config(cls, config: SpatialGridConfig) -> "SpatialPoseGrid":
        if config.bounds is None:
            raise ValueError("bounds are required when no map is supplied")
        lo = np.asarray(config.bounds[0], dtype=float)
        hi = np.asarray(config.bounds[1], dtype=float)
        if lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi])):
            raise ValueError("bounds must contain two finite XYZ triples")
        if np.any(hi < lo):
            raise ValueError("bounds maximum must be >= minimum")
        counts = np.floor((hi - lo) / config.voxel_size + 1e-12).astype(int) + 1
        waypoint_count = int(np.prod(counts, dtype=np.int64))
        if waypoint_count > config.max_waypoints:
            raise ValueError(
                f"grid has {waypoint_count} waypoints, exceeding "
                f"max_waypoints={config.max_waypoints}"
            )
        axes = [lo[index] + np.arange(counts[index]) * config.voxel_size for index in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        positions = np.column_stack([axis.reshape(-1) for axis in mesh])
        yaws = tuple(float(value) for value in np.arange(0.0, 360.0, config.yaw_step_deg))
        samples = tuple(
            SpatialPoseSample(
                waypoint_id=waypoint_id,
                position=tuple(float(value) for value in position),
                yaw_deg=yaw,
                pitch_deg=float(pitch),
            )
            for waypoint_id, position in enumerate(positions)
            for pitch in config.pitch_values_deg
            for yaw in yaws
        )
        return cls(config=config, positions=positions, samples=samples)

    def occupied(self, map_data: MapData, radius: float) -> "SpatialPoseGrid":
        anchors = []
        if map_data.num_points:
            anchors.append(np.asarray(map_data.points_xyz, dtype=float).reshape(-1, 3))
        if map_data.num_images:
            anchors.append(np.asarray(map_data.image_centers, dtype=float).reshape(-1, 3))
        if not anchors:
            raise ValueError("occupancy filtering requires landmarks or cameras")
        from scipy.spatial import cKDTree

        tree = cKDTree(np.vstack(anchors))
        distance, _ = tree.query(self.positions, k=1)
        keep = np.isfinite(distance) & (distance <= float(radius))
        positions = self.positions[keep]
        if len(positions) == 0:
            raise ValueError("occupancy filtering removed every waypoint")
        yaws = tuple(float(value) for value in np.arange(0.0, 360.0, self.config.yaw_step_deg))
        samples = tuple(
            SpatialPoseSample(
                waypoint_id=waypoint_id,
                position=tuple(float(value) for value in position),
                yaw_deg=yaw,
                pitch_deg=float(pitch),
            )
            for waypoint_id, position in enumerate(positions)
            for pitch in self.config.pitch_values_deg
            for yaw in yaws
        )
        return SpatialPoseGrid(config=self.config, positions=positions, samples=samples)
