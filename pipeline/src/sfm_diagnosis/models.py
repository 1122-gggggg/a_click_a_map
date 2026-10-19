from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_id: int
    model_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan2(self.width * 0.5, self.fx)))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan2(self.height * 0.5, self.fy)))


@dataclass(frozen=True)
class Pose:
    """Camera pose represented by camera center in world and R_wc (camera -> world)."""

    center_w: Array
    R_wc: Array

    def __post_init__(self) -> None:
        c = np.asarray(self.center_w, dtype=float).reshape(3)
        r = np.asarray(self.R_wc, dtype=float).reshape(3, 3)
        object.__setattr__(self, "center_w", c)
        object.__setattr__(self, "R_wc", r)

    @property
    def R_cw(self) -> Array:
        return self.R_wc.T

    @property
    def forward_w(self) -> Array:
        return self.R_wc[:, 2]

    def world_to_camera(self, xyz_w: Array) -> Array:
        xyz = np.asarray(xyz_w, dtype=float)
        return (self.R_cw @ (xyz - self.center_w).T).T


@dataclass
class MapData:
    point_ids: Array
    points_xyz: Array
    point_rgb: Array
    point_errors: Array
    track_lengths: Array
    track_image_ids: list[Array]
    image_ids: Array
    image_names: list[str]
    image_camera_ids: Array
    image_centers: Array
    image_R_wc: Array
    cameras: dict[int, CameraIntrinsics]
    metadata: dict[str, object] = field(default_factory=dict)
    _default_point_quality_cache: Array | None = field(default=None, init=False, repr=False)
    _observation_diversity_cache: Array | None = field(default=None, init=False, repr=False)
    _triangulation_angle_cache: Array | None = field(default=None, init=False, repr=False)
    _point_tree_cache: object = field(default=None, init=False, repr=False)
    _image_tree_cache: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.point_ids = np.asarray(self.point_ids, dtype=np.int64)
        self.points_xyz = np.asarray(self.points_xyz, dtype=float).reshape(-1, 3)
        self.point_rgb = np.asarray(self.point_rgb, dtype=np.uint8).reshape(-1, 3)
        self.point_errors = np.asarray(self.point_errors, dtype=float).reshape(-1)
        self.track_lengths = np.asarray(self.track_lengths, dtype=np.int32).reshape(-1)
        self.image_ids = np.asarray(self.image_ids, dtype=np.int64)
        self.image_camera_ids = np.asarray(self.image_camera_ids, dtype=np.int64)
        self.image_centers = np.asarray(self.image_centers, dtype=float).reshape(-1, 3)
        self.image_R_wc = np.asarray(self.image_R_wc, dtype=float).reshape(-1, 3, 3)
        if len(self.point_ids) != len(self.points_xyz):
            raise ValueError("point arrays have inconsistent lengths")
        if len(self.image_ids) != len(self.image_centers):
            raise ValueError("image arrays have inconsistent lengths")
        if len(self.track_image_ids) != len(self.point_ids):
            raise ValueError("track_image_ids length must equal number of points")

    @property
    def num_points(self) -> int:
        return len(self.point_ids)

    @property
    def num_images(self) -> int:
        return len(self.image_ids)

    @property
    def bounds(self) -> tuple[Array, Array]:
        if self.num_points == 0:
            if self.num_images == 0:
                z = np.zeros(3)
                return z.copy(), z.copy()
            return self.image_centers.min(axis=0), self.image_centers.max(axis=0)
        return self.points_xyz.min(axis=0), self.points_xyz.max(axis=0)

    @property
    def median_intrinsics(self) -> CameraIntrinsics:
        if not self.cameras:
            raise ValueError("map has no camera intrinsics")
        cams = list(self.cameras.values())
        idx = len(cams) // 2
        cams = sorted(cams, key=lambda c: c.fx)
        return cams[idx]

    def image_index(self) -> dict[int, int]:
        return {int(image_id): i for i, image_id in enumerate(self.image_ids.tolist())}

    def point_quality_weights(
        self,
        track_midpoint: float = 4.0,
        track_slope: float = 0.8,
        error_scale_px: float = 2.0,
    ) -> Array:
        """Map-only prior for landmark repeatability in [0, 1].

        Long tracks increase the prior while large BA reprojection error decreases it.
        This is a heuristic, not a calibrated probability.
        """
        use_cache = (track_midpoint, track_slope, error_scale_px) == (4.0, 0.8, 2.0)
        if use_cache and self._default_point_quality_cache is not None:
            return self._default_point_quality_cache
        tl = self.track_lengths.astype(float)
        track_w = 1.0 / (1.0 + np.exp(-track_slope * (tl - track_midpoint)))
        errors = np.maximum(self.point_errors, 0.0)
        err_w = np.exp(-errors / max(error_scale_px, 1e-6))
        result = np.clip(track_w * err_w, 0.0, 1.0)
        if use_cache:
            self._default_point_quality_cache = result
        return result

    def observation_direction_diversity(self, point_indices: Array) -> Array:
        """Return a per-point angular spread proxy in [0, 1].

        The full-map cache is computed lazily once, avoiding repeated track traversal
        during dense heatmap queries.
        """
        if self._observation_diversity_cache is None:
            image_lookup = self.image_index()
            out = np.zeros(self.num_points, dtype=float)
            for pidx in range(self.num_points):
                obs = self.track_image_ids[pidx]
                centers = []
                for image_id in obs:
                    j = image_lookup.get(int(image_id))
                    if j is not None:
                        centers.append(self.image_centers[j])
                if len(centers) < 2:
                    continue
                rays = self.points_xyz[pidx] - np.asarray(centers)
                norms = np.linalg.norm(rays, axis=1, keepdims=True)
                valid = norms[:, 0] > 1e-9
                rays = rays[valid] / norms[valid]
                if len(rays) < 2:
                    continue
                cos = np.clip(rays @ rays.T, -1.0, 1.0)
                tri = cos[np.triu_indices(len(rays), k=1)]
                if len(tri):
                    out[pidx] = float(np.clip(np.mean(1.0 - tri) / 2.0, 0.0, 1.0))
            self._observation_diversity_cache = out
        return self._observation_diversity_cache[np.asarray(point_indices, dtype=int)]

    def triangulation_angles_deg(
        self,
        point_indices: Array | None = None,
        *,
        max_observations: int = 24,
    ) -> Array:
        """Return each landmark's maximum available triangulation angle in degrees.

        For a track, the metric uses the largest angle between any two registered
        camera-to-point rays. A large value means that at least one observation pair
        supplies a useful baseline. Extremely long tracks are deterministically
        subsampled to bound the pairwise angular cost.
        """
        if self._triangulation_angle_cache is None:
            image_lookup = self.image_index()
            out = np.zeros(self.num_points, dtype=float)
            for pidx in range(self.num_points):
                centers = [
                    self.image_centers[image_lookup[int(image_id)]]
                    for image_id in self.track_image_ids[pidx]
                    if int(image_id) in image_lookup
                ]
                if len(centers) < 2:
                    continue
                centers_arr = np.asarray(centers, dtype=float)
                if len(centers_arr) > max_observations:
                    sample = np.linspace(
                        0, len(centers_arr) - 1, max_observations, dtype=int
                    )
                    centers_arr = centers_arr[sample]
                rays = self.points_xyz[pidx] - centers_arr
                norms = np.linalg.norm(rays, axis=1, keepdims=True)
                valid = norms[:, 0] > 1e-9
                rays = rays[valid] / norms[valid]
                if len(rays) < 2:
                    continue
                cos = np.clip(rays @ rays.T, -1.0, 1.0)
                pair_cos = cos[np.triu_indices(len(rays), k=1)]
                if len(pair_cos):
                    out[pidx] = float(np.degrees(np.arccos(np.min(pair_cos))))
            self._triangulation_angle_cache = out

        if point_indices is None:
            return self._triangulation_angle_cache.copy()
        return self._triangulation_angle_cache[np.asarray(point_indices, dtype=int)]

    def point_tree(self):
        if self._point_tree_cache is None:
            from scipy.spatial import cKDTree

            self._point_tree_cache = cKDTree(self.points_xyz)
        return self._point_tree_cache

    def image_tree(self):
        if self._image_tree_cache is None:
            from scipy.spatial import cKDTree

            self._image_tree_cache = cKDTree(self.image_centers)
        return self._image_tree_cache
