from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfm_diagnosis.matchability import LandmarkMatchability
from sfm_diagnosis.models import CameraIntrinsics, MapData
from sfm_diagnosis.visibility import VisibilityResult


@dataclass(frozen=True)
class FIMConfig:
    pixel_sigma: float = 1.0
    translation_scale: float = 1.0
    regularization: float = 1e-9

    def __post_init__(self) -> None:
        if not np.isfinite(self.pixel_sigma) or self.pixel_sigma <= 0.0:
            raise ValueError("pixel_sigma must be finite and > 0")
        if not np.isfinite(self.translation_scale) or self.translation_scale <= 0.0:
            raise ValueError("translation_scale must be finite and > 0")
        if not np.isfinite(self.regularization) or self.regularization <= 0.0:
            raise ValueError("regularization must be finite and > 0")


def _skew_batch(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    output = np.zeros((len(vectors), 3, 3), dtype=float)
    output[:, 0, 1] = -vectors[:, 2]
    output[:, 0, 2] = vectors[:, 1]
    output[:, 1, 0] = vectors[:, 2]
    output[:, 1, 2] = -vectors[:, 0]
    output[:, 2, 0] = -vectors[:, 1]
    output[:, 2, 1] = vectors[:, 0]
    return output


def pixel_projection_jacobians(
    camera_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Return batched 2x6 pixel Jacobians for camera-local pose perturbations.

    The perturbation order is ``[tx, ty, tz, wx, wy, wz]`` and follows
    ``dp/dxi = [-translation_scale I, [p]_x]``.  Translation and rotation
    therefore remain explicitly parameterized rather than silently normalized.
    """

    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("translation_scale must be finite and > 0")
    if len(points) == 0:
        return np.empty((0, 2, 6), dtype=float)
    z = points[:, 2]
    if np.any(~np.isfinite(points)) or np.any(z <= 1e-12):
        raise ValueError("camera points must be finite and have positive depth")
    x, y = points[:, 0], points[:, 1]
    projection = np.zeros((len(points), 2, 3), dtype=float)
    projection[:, 0, 0] = intrinsics.fx / z
    projection[:, 0, 2] = -intrinsics.fx * x / (z**2)
    projection[:, 1, 1] = intrinsics.fy / z
    projection[:, 1, 2] = -intrinsics.fy * y / (z**2)
    perturbation = np.concatenate(
        (
            -float(translation_scale)
            * np.repeat(np.eye(3, dtype=float)[None, :, :], len(points), axis=0),
            _skew_batch(points),
        ),
        axis=2,
    )
    return projection @ perturbation


def compose_landmark_weights(
    *,
    visibility: np.ndarray,
    matchability: np.ndarray,
    static: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    factors = [
        np.asarray(value, dtype=float).reshape(-1)
        for value in (visibility, matchability, static, quality)
    ]
    lengths = {len(value) for value in factors}
    if len(lengths) != 1:
        raise ValueError("landmark weight factors must have equal length")
    clipped = [np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0) for value in factors]
    return clipped[0] * clipped[1] * clipped[2] * clipped[3]


@dataclass(frozen=True)
class PixelFIMMetrics:
    eigenvalues_descending: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    logdet: float
    trace: float
    fim_a_opt: float
    fim_d_opt: float
    fim_e_opt: float
    translation_min_eigenvalue: float
    rotation_min_eigenvalue: float
    weakest_eigenvector: np.ndarray
    rank: int
    regularization: float

    def to_dict(self) -> dict:
        return {
            "eigenvalues_descending": self.eigenvalues_descending.tolist(),
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "condition_number": self.condition_number,
            "logdet": self.logdet,
            "trace": self.trace,
            "fim_a_opt": self.fim_a_opt,
            "fim_d_opt": self.fim_d_opt,
            "fim_e_opt": self.fim_e_opt,
            "translation_min_eigenvalue": self.translation_min_eigenvalue,
            "rotation_min_eigenvalue": self.rotation_min_eigenvalue,
            "weakest_eigenvector": self.weakest_eigenvector.tolist(),
            "rank": self.rank,
            "regularization": self.regularization,
        }


@dataclass(frozen=True)
class PixelFIMResult:
    matrix: np.ndarray
    metrics: PixelFIMMetrics
    weights: np.ndarray
    matchability_source: str = "unspecified"


def pixel_projection_fim(
    camera_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    weights: np.ndarray | None = None,
    *,
    pixel_sigma: float = 1.0,
    translation_scale: float = 1.0,
    regularization: float = 1e-9,
) -> PixelFIMResult:
    points = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    if not np.isfinite(pixel_sigma) or pixel_sigma <= 0.0:
        raise ValueError("pixel_sigma must be finite and > 0")
    jacobians = pixel_projection_jacobians(
        points, intrinsics, translation_scale=translation_scale
    )
    landmark_weights = (
        np.ones(len(points), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float).reshape(-1)
    )
    if len(landmark_weights) != len(points):
        raise ValueError("weights must match camera_points")
    landmark_weights = np.clip(
        np.nan_to_num(landmark_weights, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )
    matrix = np.einsum("n,nai,naj->ij", landmark_weights, jacobians, jacobians)
    matrix /= float(pixel_sigma) ** 2
    matrix = 0.5 * (matrix + matrix.T)
    metrics = compute_pixel_fim_metrics(matrix, regularization=regularization)
    return PixelFIMResult(matrix=matrix, metrics=metrics, weights=landmark_weights)


def compute_pixel_fim_metrics(
    matrix: np.ndarray, *, regularization: float = 1e-9
) -> PixelFIMMetrics:
    information = np.asarray(matrix, dtype=float).reshape(6, 6)
    if np.any(~np.isfinite(information)):
        raise ValueError("FIM must be finite")
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    lambda_min = float(eigenvalues[0])
    lambda_max = float(eigenvalues[-1])
    damping = max(float(regularization), lambda_max * 1e-12, 1e-15)
    stabilized = eigenvalues + damping
    translation_eigenvalues = np.maximum(
        np.linalg.eigvalsh(information[:3, :3]), 0.0
    )
    rotation_eigenvalues = np.maximum(
        np.linalg.eigvalsh(information[3:, 3:]), 0.0
    )
    rank_tolerance = max(lambda_max * 1e-10, damping)
    return PixelFIMMetrics(
        eigenvalues_descending=eigenvalues[::-1],
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_number=float(lambda_max / max(lambda_min, damping)),
        logdet=float(np.sum(np.log(stabilized))),
        trace=float(np.trace(information)),
        fim_a_opt=float(np.sum(1.0 / stabilized)),
        fim_d_opt=float(np.sum(np.log(stabilized))),
        fim_e_opt=lambda_min,
        translation_min_eigenvalue=float(translation_eigenvalues[0]),
        rotation_min_eigenvalue=float(rotation_eigenvalues[0]),
        weakest_eigenvector=np.asarray(eigenvectors[:, 0], dtype=float),
        rank=int(np.sum(eigenvalues > rank_tolerance)),
        regularization=damping,
    )


def heuristic_matchability(map_data: MapData, point_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(point_indices, dtype=int).reshape(-1)
    if not len(indices):
        return np.empty(0, dtype=float)
    tracks = map_data.track_lengths[indices].astype(float)
    track_score = 1.0 - np.exp(-tracks / 4.0)
    diversity = map_data.observation_direction_diversity(indices)
    angles = map_data.triangulation_angles_deg(indices)
    angle_score = 1.0 - np.exp(-np.maximum(angles, 0.0) / 5.0)
    return np.clip(0.40 * track_score + 0.30 * diversity + 0.30 * angle_score, 0.0, 1.0)


def matchability_aware_fim(
    map_data: MapData,
    visibility: VisibilityResult,
    intrinsics: CameraIntrinsics,
    *,
    empirical: LandmarkMatchability | None = None,
    config: FIMConfig | None = None,
) -> PixelFIMResult:
    cfg = config or FIMConfig()
    indices = np.asarray(visibility.point_indices, dtype=int)
    heuristic = heuristic_matchability(map_data, indices)
    matchability = heuristic.copy()
    source = "heuristic"
    if empirical is not None and len(indices):
        lookup = empirical.p_by_point_id()
        evidenced = np.zeros(len(indices), dtype=bool)
        for offset, point_id in enumerate(map_data.point_ids[indices].tolist()):
            value = lookup.get(int(point_id))
            if value is not None:
                matchability[offset] = value
                evidenced[offset] = True
        if np.all(evidenced):
            source = "empirical"
        elif np.any(evidenced):
            source = "mixed_empirical_heuristic"
    static_metadata = map_data.metadata.get("point_static_weights")
    if static_metadata is None:
        static = np.ones(len(indices), dtype=float)
    else:
        all_static = np.asarray(static_metadata, dtype=float).reshape(-1)
        if len(all_static) != map_data.num_points:
            raise ValueError("point_static_weights metadata must match map points")
        static = all_static[indices]
    weights = compose_landmark_weights(
        visibility=visibility.geometric_weights,
        matchability=matchability,
        static=static,
        quality=map_data.point_quality_weights()[indices],
    )
    result = pixel_projection_fim(
        visibility.camera_points,
        intrinsics,
        weights,
        pixel_sigma=cfg.pixel_sigma,
        translation_scale=cfg.translation_scale,
        regularization=cfg.regularization,
    )
    return PixelFIMResult(
        matrix=result.matrix,
        metrics=result.metrics,
        weights=result.weights,
        matchability_source=source,
    )
