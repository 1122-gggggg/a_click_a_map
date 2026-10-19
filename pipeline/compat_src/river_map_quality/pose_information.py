"""Canonical scale-normalized six-degree-of-freedom pose-information diagnostics.

Every diagnostic path uses the same left-perturbation Jacobian and normalized state
coordinates.  Visual likelihood information is never silently reported as posterior
information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FIM_STATE_PARAMETERIZATION = (
    "SE3_LEFT_TANGENT_[delta_translation/median_positive_depth,delta_rotation_rad]"
)
FIM_JACOBIAN_CONVENTION = (
    "world_to_camera_left_perturbation: delta_p=delta_translation-[p_camera]_x"
    " delta_rotation"
)


@dataclass(frozen=True)
class PoseInformation:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    lambda_min: float
    lambda_max: float
    condition_number: float
    scene_scale: float
    log_determinant: float
    covariance: np.ndarray


@dataclass(frozen=True)
class PoseInformationDecomposition:
    """Visual, optional prior, and posterior information in canonical coordinates."""

    visual: PoseInformation
    prior: PoseInformation | None
    posterior: PoseInformation
    state_parameterization: str = FIM_STATE_PARAMETERIZATION
    jacobian_convention: str = FIM_JACOBIAN_CONVENTION

    @property
    def has_prior(self) -> bool:
        return self.prior is not None


def _scaled_left_tangent_jacobian(
    camera_points: np.ndarray,
    *,
    fx: float,
    fy: float,
    scene_scale: float,
) -> np.ndarray:
    """Return d(project(p))/d([delta_t / scale, delta_rotation]) for left SE(3)."""

    x, y, z = np.asarray(camera_points, dtype=float).T
    z2 = z * z
    jacobian = np.empty((len(camera_points), 2, 6), dtype=float)
    jacobian[:, 0, :3] = np.column_stack(
        (fx * scene_scale / z, np.zeros_like(z), -fx * x * scene_scale / z2)
    )
    jacobian[:, 1, :3] = np.column_stack(
        (np.zeros_like(z), fy * scene_scale / z, -fy * y * scene_scale / z2)
    )
    jacobian[:, 0, 3:] = np.column_stack(
        (-fx * x * y / z2, fx * (1.0 + x * x / z2), -fx * y / z)
    )
    jacobian[:, 1, 3:] = np.column_stack(
        (-fy * (1.0 + y * y / z2), fy * x * y / z2, fy * x / z)
    )
    return jacobian


def summarize_pose_information(
    information: np.ndarray,
    *,
    scene_scale: float,
    rank_rtol: float = 1e-10,
) -> PoseInformation:
    """Validate and summarize a positive-semidefinite FIM in canonical coordinates."""

    matrix = np.asarray(information, dtype=float)
    if matrix.shape != (6, 6) or not np.isfinite(matrix).all():
        raise ValueError("information must be a finite 6x6 matrix")
    if not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    if rank_rtol <= 0 or not np.isfinite(rank_rtol):
        raise ValueError("rank_rtol must be finite and positive")
    matrix = 0.5 * (matrix + matrix.T)
    raw_eigenvalues = np.linalg.eigvalsh(matrix)
    numerical_tolerance = max(
        float(np.max(np.abs(raw_eigenvalues))) * rank_rtol,
        np.finfo(float).eps,
    )
    if float(raw_eigenvalues[0]) < -numerical_tolerance:
        raise ValueError("information must be positive semidefinite")
    eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    lambda_max = float(eigenvalues[-1])
    rank_tolerance = max(lambda_max * rank_rtol, np.finfo(float).eps)
    rank = int(np.count_nonzero(eigenvalues > rank_tolerance))
    lambda_min = float(eigenvalues[0])
    condition = (
        float("inf")
        if rank < 6 or lambda_min <= 0
        else float(lambda_max / lambda_min)
    )
    sign, log_determinant = np.linalg.slogdet(matrix)
    if sign <= 0 or rank < 6:
        log_determinant = float("-inf")
    covariance = np.linalg.pinv(matrix, rcond=rank_rtol, hermitian=True)
    return PoseInformation(
        matrix=matrix,
        eigenvalues=eigenvalues,
        rank=rank,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        condition_number=condition,
        scene_scale=float(scene_scale),
        log_determinant=float(log_determinant),
        covariance=covariance,
    )


def scaled_pose_information(
    camera_points: np.ndarray,
    *,
    fx: float,
    fy: float,
    pixel_sigma: float,
    scene_scale: float | None = None,
    point_weights: np.ndarray | None = None,
    rank_rtol: float = 1e-10,
) -> PoseInformation:
    """Compute visual information in normalized left-tangent pose coordinates.

    The state is ``[delta_translation / s, delta_rotation_rad]`` where ``s`` is
    the supplied scene scale or median positive camera-frame depth.  ``point_weights``
    are visual inverse-variance multipliers; they never encode a temporal prior.
    """

    points = np.asarray(camera_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera_points must have shape (N, 3)")
    if fx <= 0 or fy <= 0 or pixel_sigma <= 0:
        raise ValueError("fx, fy, and pixel_sigma must be positive")
    weights = (
        np.ones(len(points), dtype=float)
        if point_weights is None
        else np.asarray(point_weights, dtype=float)
    )
    if (
        weights.shape != (len(points),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("point_weights must be finite, non-negative, and shape (N,)")

    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-9)
    selected = points[valid]
    selected_weights = weights[valid]
    if not len(selected):
        raise ValueError("at least one finite positive-depth landmark is required")
    local_scene_scale = float(np.median(selected[:, 2]))
    if scene_scale is None:
        scene_scale = local_scene_scale
    elif not np.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")

    jacobian = _scaled_left_tangent_jacobian(
        selected,
        fx=fx,
        fy=fy,
        scene_scale=float(scene_scale),
    )
    information = np.einsum(
        "nki,n,nkj->ij",
        jacobian,
        selected_weights,
        jacobian,
        optimize=True,
    ) / (pixel_sigma * pixel_sigma)
    return summarize_pose_information(
        information,
        scene_scale=float(scene_scale),
        rank_rtol=rank_rtol,
    )


def decompose_pose_information(
    camera_points: np.ndarray,
    *,
    fx: float,
    fy: float,
    pixel_sigma: float,
    scene_scale: float | None = None,
    point_weights: np.ndarray | None = None,
    prior_information: np.ndarray | None = None,
    rank_rtol: float = 1e-10,
) -> PoseInformationDecomposition:
    """Separate visual, prior, and posterior FIMs without changing their semantics.

    ``prior_information`` must already use :data:`FIM_STATE_PARAMETERIZATION`.
    With no temporal prior, ``Omega_prior = 0`` and ``posterior == visual``.
    """

    visual = scaled_pose_information(
        camera_points,
        fx=fx,
        fy=fy,
        pixel_sigma=pixel_sigma,
        scene_scale=scene_scale,
        point_weights=point_weights,
        rank_rtol=rank_rtol,
    )
    if prior_information is None:
        return PoseInformationDecomposition(
            visual=visual,
            prior=None,
            posterior=visual,
        )
    prior = summarize_pose_information(
        prior_information,
        scene_scale=visual.scene_scale,
        rank_rtol=rank_rtol,
    )
    if not np.any(prior.matrix):
        return PoseInformationDecomposition(
            visual=visual,
            prior=None,
            posterior=visual,
        )
    posterior = summarize_pose_information(
        visual.matrix + prior.matrix,
        scene_scale=visual.scene_scale,
        rank_rtol=rank_rtol,
    )
    return PoseInformationDecomposition(
        visual=visual,
        prior=prior,
        posterior=posterior,
    )


def fim_scaling_convention(*, scene_scale: float, pixel_sigma: float) -> dict[str, str | float]:
    """Return serialization-safe canonical FIM convention metadata."""

    if (
        not np.isfinite(scene_scale)
        or scene_scale <= 0
        or not np.isfinite(pixel_sigma)
        or pixel_sigma <= 0
    ):
        raise ValueError("scene_scale and pixel_sigma must be finite and positive")
    return {
        "state_parameterization": FIM_STATE_PARAMETERIZATION,
        "jacobian_convention": FIM_JACOBIAN_CONVENTION,
        "translation_scale_map_units": float(scene_scale),
        "rotation_unit": "radians",
        "pixel_sigma": float(pixel_sigma),
        "prior_coordinate_requirement": FIM_STATE_PARAMETERIZATION,
    }
