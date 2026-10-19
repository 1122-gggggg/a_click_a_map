from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fim import PixelFIMMetrics


@dataclass(frozen=True)
class ShapeDescriptor:
    eigenvalues_descending: np.ndarray
    linearity: float
    planarity: float
    scattering: float

    def to_dict(self) -> dict:
        return {
            "eigenvalues_descending": self.eigenvalues_descending.tolist(),
            "linearity": self.linearity,
            "planarity": self.planarity,
            "scattering": self.scattering,
        }


@dataclass(frozen=True)
class DegeneracyResult:
    landmark_shape: ShapeDescriptor
    camera_shape: ShapeDescriptor
    flags: tuple[str, ...]
    translation_observability_ratio: float
    rotation_observability_ratio: float


def shape_descriptor(points: np.ndarray) -> ShapeDescriptor:
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(xyz) < 2:
        eigenvalues = np.zeros(3, dtype=float)
    else:
        centered = xyz - np.mean(xyz, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.0)
    scale = max(float(eigenvalues[0]), 1e-12)
    return ShapeDescriptor(
        eigenvalues_descending=eigenvalues,
        linearity=float((eigenvalues[0] - eigenvalues[1]) / scale),
        planarity=float((eigenvalues[1] - eigenvalues[2]) / scale),
        scattering=float(eigenvalues[2] / scale),
    )


def diagnose_degeneracy(
    *,
    landmarks: np.ndarray,
    observer_centers: np.ndarray,
    fim: PixelFIMMetrics,
    parallax_p10_deg: float | None,
    parallax_median_deg: float | None,
    view_entropy: float | None,
) -> DegeneracyResult:
    """Combine local shape, observer geometry, parallax, and FIM block evidence."""

    landmark_shape = shape_descriptor(landmarks)
    camera_shape = shape_descriptor(observer_centers)
    flags: list[str] = []
    if landmark_shape.planarity >= 0.35 and landmark_shape.scattering <= 0.08:
        flags.append("PLANAR_DOMINANT")
    if landmark_shape.linearity >= 0.70:
        flags.append("CORRIDOR_DEGENERACY")
    if camera_shape.linearity >= 0.70 and camera_shape.scattering <= 0.10:
        flags.append("COLLINEAR_CAMERA")
    if (
        parallax_p10_deg is not None
        and parallax_median_deg is not None
        and parallax_p10_deg < 1.0
        and parallax_median_deg < 3.0
    ):
        flags.append("LOW_PARALLAX")
    if view_entropy is not None and view_entropy < 0.20:
        flags.append("ONE_SIDED_OBSERVATION")

    translation_scale = max(fim.trace / 6.0, fim.regularization)
    translation_ratio = fim.translation_min_eigenvalue / translation_scale
    rotation_ratio = fim.rotation_min_eigenvalue / translation_scale
    if translation_ratio < 1e-4:
        flags.append("LOW_TRANSLATION_OBSERVABILITY")
    if rotation_ratio < 1e-4:
        flags.append("LOW_ROTATION_OBSERVABILITY")
    return DegeneracyResult(
        landmark_shape=landmark_shape,
        camera_shape=camera_shape,
        flags=tuple(dict.fromkeys(flags)),
        translation_observability_ratio=float(translation_ratio),
        rotation_observability_ratio=float(rotation_ratio),
    )
