from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _skew_batch(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    out = np.zeros((len(v), 3, 3), dtype=float)
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out


def weighted_bearing_fim(
    camera_points: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    bearing_sigma_rad: float = 0.002,
    translation_scale_m: float = 1.0,
) -> np.ndarray:
    """Compute a 6x6 bearing-observation Fisher information matrix.

    The local pose perturbation is [translation / translation_scale_m, rotation(rad)].
    A larger translation_scale_m therefore states that a unit normalized translation
    perturbation corresponds to a larger metric displacement. The scaling avoids
    silently comparing meters and radians without documenting the convention.

    For each camera-frame landmark p, b=p/||p|| and
        db/dp = (I - b b^T) / ||p||
        dp/dxi = [-s I, [p]_x]
    with s=translation_scale_m. The sign convention does not affect the diagonal
    information terms and follows a camera-local perturbation convention.
    """
    p = np.asarray(camera_points, dtype=float).reshape(-1, 3)
    n = len(p)
    if n == 0:
        return np.zeros((6, 6), dtype=float)
    r = np.linalg.norm(p, axis=1)
    valid = r > 1e-9
    p = p[valid]
    r = r[valid]
    if len(p) == 0:
        return np.zeros((6, 6), dtype=float)
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    w = np.clip(w[valid], 0.0, None)

    b = p / r[:, None]
    eye = np.eye(3)[None, :, :]
    A = (eye - b[:, :, None] * b[:, None, :]) / r[:, None, None]
    dp_dxi = np.concatenate(
        (
            -float(translation_scale_m)
            * np.repeat(np.eye(3)[None, :, :], len(p), axis=0),
            _skew_batch(p),
        ),
        axis=2,
    )
    J = A @ dp_dxi
    info = np.einsum("n,nai,naj->ij", w, J, J)
    sigma2 = max(float(bearing_sigma_rad) ** 2, 1e-18)
    return info / sigma2


@dataclass(frozen=True)
class FisherMetrics:
    eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    logdet: float
    trace: float
    rank: int

    def as_dict(self) -> dict:
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "condition_number": self.condition_number,
            "logdet": self.logdet,
            "trace": self.trace,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class PoseUncertaintyMetrics:
    """Local covariance diagnostics derived from a 6DoF information matrix.

    The bearing FIM mixes normalized translation and rotation. ``translation_scale_m``
    defines the metric displacement represented by one normalized translation unit.
    Absolute covariance values inherit the assumed bearing noise and landmark weights,
    so they should be calibrated/ranked on held-out localization data rather than
    interpreted as a guaranteed posterior error bound.
    """

    covariance_normalized: np.ndarray
    translation_covariance_m2: np.ndarray
    rotation_covariance_rad2: np.ndarray
    sigma_pose_worst_normalized: float
    sigma_translation_worst_m: float
    sigma_rotation_worst_deg: float
    trace_covariance_normalized: float
    weakest_direction: np.ndarray
    weakest_translation_direction_camera: np.ndarray
    weakest_translation_fraction: float
    weakest_rotation_fraction: float
    regularization: float

    def as_dict(self) -> dict:
        return {
            "covariance_normalized": self.covariance_normalized.tolist(),
            "translation_covariance_m2": self.translation_covariance_m2.tolist(),
            "rotation_covariance_rad2": self.rotation_covariance_rad2.tolist(),
            "sigma_pose_worst_normalized": self.sigma_pose_worst_normalized,
            "sigma_translation_worst_m": self.sigma_translation_worst_m,
            "sigma_rotation_worst_deg": self.sigma_rotation_worst_deg,
            "trace_covariance_normalized": self.trace_covariance_normalized,
            "weakest_direction": self.weakest_direction.tolist(),
            "weakest_translation_direction_camera": (
                self.weakest_translation_direction_camera.tolist()
            ),
            "weakest_translation_fraction": self.weakest_translation_fraction,
            "weakest_rotation_fraction": self.weakest_rotation_fraction,
            "regularization": self.regularization,
            "calibration_note": (
                "Ranking/local uncertainty proxy only unless bearing noise and shared model "
                "correlations have been calibrated on held-out localization queries."
            ),
        }


def compute_fisher_metrics(fim: np.ndarray, regularization: float = 1e-9) -> FisherMetrics:
    F = np.asarray(fim, dtype=float).reshape(6, 6)
    F = 0.5 * (F + F.T)
    eig = np.linalg.eigvalsh(F)
    eig = np.maximum(eig, 0.0)
    lam_min = float(eig[0])
    lam_max = float(eig[-1])
    condition = float(lam_max / max(lam_min, regularization))
    logdet = float(np.sum(np.log(eig + regularization)))
    rank_tol = max(lam_max * 1e-10, regularization)
    rank = int(np.sum(eig > rank_tol))
    return FisherMetrics(
        eigenvalues=eig,
        lambda_min=lam_min,
        lambda_max=lam_max,
        condition_number=condition,
        logdet=logdet,
        trace=float(np.trace(F)),
        rank=rank,
    )


def compute_pose_uncertainty(
    fim: np.ndarray,
    *,
    translation_scale_m: float = 1.0,
    regularization: float = 1e-9,
) -> PoseUncertaintyMetrics:
    """Convert a 6DoF information matrix into interpretable uncertainty proxies.

    The full inverse is computed in the normalized perturbation coordinates used by
    :func:`weighted_bearing_fim`. Translation and rotation marginal blocks are then
    reported separately so meters and radians are not silently mixed in one score.

    ``sigma_pose_worst_normalized`` is useful for within-map ranking. The translation
    and rotation worst-direction sigmas are easier to interpret, but remain model-based
    uncertainty proxies rather than calibrated real-world error guarantees.
    """
    F = np.asarray(fim, dtype=float).reshape(6, 6)
    F = 0.5 * (F + F.T)
    eig, vec = np.linalg.eigh(F)
    eig = np.maximum(eig, 0.0)
    lam_max = float(eig[-1]) if len(eig) else 0.0
    damping = max(float(regularization), lam_max * 1e-12)
    inv_eig = 1.0 / (eig + damping)
    covariance = (vec * inv_eig[None, :]) @ vec.T
    covariance = 0.5 * (covariance + covariance.T)

    scale = max(float(translation_scale_m), 1e-12)
    translation_cov = covariance[:3, :3] * (scale**2)
    rotation_cov = covariance[3:, 3:]
    translation_sigma = _worst_sigma(translation_cov)
    rotation_sigma_rad = _worst_sigma(rotation_cov)
    pose_sigma = _worst_sigma(covariance)

    weakest = vec[:, 0].copy()
    t_norm = float(np.linalg.norm(weakest[:3]))
    r_norm = float(np.linalg.norm(weakest[3:]))
    total2 = t_norm**2 + r_norm**2
    translation_fraction = t_norm**2 / max(total2, 1e-18)
    rotation_fraction = r_norm**2 / max(total2, 1e-18)
    if t_norm > 1e-12:
        weakest_translation = weakest[:3] / t_norm
    else:
        weakest_translation = np.zeros(3, dtype=float)

    return PoseUncertaintyMetrics(
        covariance_normalized=covariance,
        translation_covariance_m2=translation_cov,
        rotation_covariance_rad2=rotation_cov,
        sigma_pose_worst_normalized=pose_sigma,
        sigma_translation_worst_m=translation_sigma,
        sigma_rotation_worst_deg=float(np.degrees(rotation_sigma_rad)),
        trace_covariance_normalized=float(np.trace(covariance)),
        weakest_direction=weakest,
        weakest_translation_direction_camera=weakest_translation,
        weakest_translation_fraction=float(translation_fraction),
        weakest_rotation_fraction=float(rotation_fraction),
        regularization=damping,
    )


def _worst_sigma(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eig = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(float(eig[-1]), 0.0))) if len(eig) else 0.0
