"""Pure contracts for reference-conditioned pose-source attribution.

The functions in this module deliberately do not run a matcher or mutate a map.  They
turn independently estimated world-to-camera pose hypotheses into auditable consensus,
geometry, and conversion-funnel evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoseHypothesis:
    label: str
    pose: np.ndarray
    inliers: int
    positive_depth_ratio: float
    heldout_reprojection_p90: float


@dataclass(frozen=True)
class PoseMode:
    labels: tuple[str, ...]
    representative_pose: np.ndarray
    support: int
    support_share: float
    rotation_dispersion_p90_deg: float
    center_dispersion_p90_normalized: float


@dataclass(frozen=True)
class ReferenceConsensus:
    status: str
    modes: tuple[PoseMode, ...]
    reliable_hypothesis_count: int
    rejected_hypothesis_count: int
    dominant_support_share: float
    m0_rotation_error_deg: float | None
    edm_rotation_error_deg: float | None


@dataclass(frozen=True)
class PlanarityDiagnostics:
    eigenvalues: np.ndarray
    rho_plane: float
    plane_rmse: float
    normalized_plane_rmse: float
    is_planar: bool


def _as_pose(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("pose must be a finite 4x4 world-to-camera matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("pose rotation must have determinant +1")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("pose must have a homogeneous final row")
    return matrix


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return the geodesic angle between two rotation matrices in degrees."""

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("rotations must have shape (3, 3)")
    cosine = np.clip((np.trace(a @ b.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def camera_center(pose: np.ndarray) -> np.ndarray:
    """Return the world-frame camera center of a world-to-camera pose."""

    matrix = _as_pose(pose)
    return -matrix[:3, :3].T @ matrix[:3, 3]


def relative_pose(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the rigid transform mapping camera-one coordinates to camera two."""

    first_matrix = _as_pose(first)
    second_matrix = _as_pose(second)
    return second_matrix @ np.linalg.inv(first_matrix)


def _pairwise_pose_distance(
    first: PoseHypothesis,
    second: PoseHypothesis,
    scene_scale: float,
) -> tuple[float, float]:
    rotation = rotation_distance_deg(first.pose[:3, :3], second.pose[:3, :3])
    center = float(np.linalg.norm(camera_center(first.pose) - camera_center(second.pose)))
    return rotation, center / scene_scale


def _components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    result: list[list[int]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        result.append(sorted(component))
    return result


def _mode(
    rows: Sequence[PoseHypothesis],
    indices: Sequence[int],
    *,
    total_support: int,
    scene_scale: float,
) -> PoseMode:
    support = sum(rows[index].inliers for index in indices)
    pair_scores = []
    for candidate in indices:
        score = 0.0
        for other in indices:
            rotation, center = _pairwise_pose_distance(rows[candidate], rows[other], scene_scale)
            score += rows[other].inliers * (rotation / 2.0 + center / 0.02)
        pair_scores.append((score, rows[candidate].label, candidate))
    representative_index = min(pair_scores)[2]
    representative = rows[representative_index]
    rotations = []
    centers = []
    for index in indices:
        rotation, center = _pairwise_pose_distance(representative, rows[index], scene_scale)
        rotations.append(rotation)
        centers.append(center)
    return PoseMode(
        labels=tuple(sorted(rows[index].label for index in indices)),
        representative_pose=np.array(representative.pose, copy=True),
        support=support,
        support_share=float(support / total_support),
        rotation_dispersion_p90_deg=float(np.percentile(rotations, 90)),
        center_dispersion_p90_normalized=float(np.percentile(centers, 90)),
    )


def _distance_to_pose(mode: PoseMode, pose: np.ndarray, scene_scale: float) -> tuple[float, float]:
    matrix = _as_pose(pose)
    rotation = rotation_distance_deg(mode.representative_pose[:3, :3], matrix[:3, :3])
    center = np.linalg.norm(camera_center(mode.representative_pose) - camera_center(matrix))
    return rotation, float(center / scene_scale)


def classify_reference_consensus(
    hypotheses: Sequence[PoseHypothesis],
    *,
    m0_pose: np.ndarray,
    edm_pose: np.ndarray,
    scene_scale: float,
    min_inliers: int = 20,
    min_positive_depth_ratio: float = 0.9,
    max_heldout_reprojection_p90: float = 5.0,
    rotation_mode_threshold_deg: float = 2.0,
    center_mode_threshold_normalized: float = 0.02,
    minimum_mode_support_share: float = 0.2,
    minimum_consensus_support_share: float = 0.8,
) -> ReferenceConsensus:
    """Classify independently estimated poses without consulting FIM conditioning."""

    if scene_scale <= 0 or not np.isfinite(scene_scale):
        raise ValueError("scene_scale must be finite and positive")
    _as_pose(m0_pose)
    _as_pose(edm_pose)
    reliable: list[PoseHypothesis] = []
    for hypothesis in hypotheses:
        _as_pose(hypothesis.pose)
        if (
            hypothesis.inliers >= min_inliers
            and hypothesis.positive_depth_ratio >= min_positive_depth_ratio
            and hypothesis.heldout_reprojection_p90 <= max_heldout_reprojection_p90
        ):
            reliable.append(hypothesis)
    rejected = len(hypotheses) - len(reliable)
    if not reliable:
        return ReferenceConsensus("INSUFFICIENT", (), 0, rejected, 0.0, None, None)

    adjacency = [set() for _ in reliable]
    for first in range(len(reliable)):
        for second in range(first + 1, len(reliable)):
            rotation, center = _pairwise_pose_distance(
                reliable[first], reliable[second], scene_scale
            )
            if (
                rotation <= rotation_mode_threshold_deg
                and center <= center_mode_threshold_normalized
            ):
                adjacency[first].add(second)
                adjacency[second].add(first)
    total_support = sum(row.inliers for row in reliable)
    modes = tuple(
        sorted(
            (
                _mode(
                    reliable,
                    component,
                    total_support=total_support,
                    scene_scale=scene_scale,
                )
                for component in _components(adjacency)
            ),
            key=lambda mode: (-mode.support, mode.labels),
        )
    )
    material_modes = tuple(
        mode for mode in modes if mode.support_share >= minimum_mode_support_share
    )
    dominant = modes[0]
    m0_rotation, m0_center = _distance_to_pose(dominant, m0_pose, scene_scale)
    edm_rotation, edm_center = _distance_to_pose(dominant, edm_pose, scene_scale)
    if len(material_modes) >= 2:
        status = "MULTIMODAL"
    elif dominant.support_share < minimum_consensus_support_share:
        status = "DIFFUSE"
    else:
        close_m0 = (
            m0_rotation <= rotation_mode_threshold_deg
            and m0_center <= center_mode_threshold_normalized
        )
        close_edm = (
            edm_rotation <= rotation_mode_threshold_deg
            and edm_center <= center_mode_threshold_normalized
        )
        if close_m0 and close_edm:
            m0_score = (
                m0_rotation / rotation_mode_threshold_deg
                + m0_center / center_mode_threshold_normalized
            )
            edm_score = (
                edm_rotation / rotation_mode_threshold_deg
                + edm_center / center_mode_threshold_normalized
            )
            status = "CONSENSUS_M0" if m0_score <= edm_score else "CONSENSUS_EDM"
        elif close_m0:
            status = "CONSENSUS_M0"
        elif close_edm:
            status = "CONSENSUS_EDM"
        else:
            status = "CONSENSUS_OTHER"
    return ReferenceConsensus(
        status=status,
        modes=modes,
        reliable_hypothesis_count=len(reliable),
        rejected_hypothesis_count=rejected,
        dominant_support_share=dominant.support_share,
        m0_rotation_error_deg=m0_rotation,
        edm_rotation_error_deg=edm_rotation,
    )


def attribute_pose_source(consensus: ReferenceConsensus, *, sequential_support: str) -> str:
    """Combine global reference modes with independent relative-motion evidence."""

    support = sequential_support.upper()
    if support not in {"M0", "EDM", "NEITHER", "CONFLICTED", "INDETERMINATE"}:
        raise ValueError("invalid sequential support")
    if consensus.status == "MULTIMODAL":
        return "REPEATED_STRUCTURE_OR_REFERENCE_AMBIGUITY"
    if consensus.status == "CONSENSUS_EDM" and support == "EDM":
        return "M0_LOCAL_POSE_SUSPECTED"
    if consensus.status == "CONSENSUS_M0" and support == "M0":
        return "EDM_REFERENCE_ANCHORING_SUSPECTED"
    return "UNRESOLVED"


def conversion_funnel(
    *, raw_matches: int, verified_2d2d: int, unique_2d3d: int, pnp_inliers: int
) -> dict[str, int | float | None]:
    """Return forward conversion rates without hiding zero denominators."""

    values = (raw_matches, verified_2d2d, unique_2d3d, pnp_inliers)
    if any(value < 0 for value in values):
        raise ValueError("conversion counts cannot be negative")
    return {
        "raw_matches": raw_matches,
        "verified_2d2d": verified_2d2d,
        "unique_2d3d": unique_2d3d,
        "pnp_inliers": pnp_inliers,
        "r_geo": float(verified_2d2d / raw_matches) if raw_matches else None,
        "r_lift": float(unique_2d3d / verified_2d2d) if verified_2d2d else None,
        "r_pnp": float(pnp_inliers / raw_matches) if raw_matches else None,
    }


def parallax_risk(angles_deg: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Describe low parallax as a structural risk, never a repair decision."""

    angles = np.asarray(angles_deg, dtype=float)
    angles = angles[np.isfinite(angles)]
    if not len(angles) or np.any(angles < 0):
        raise ValueError("at least one finite non-negative angle is required")
    p10, p25, p50 = np.percentile(angles, [10, 25, 50])
    fraction = float(np.mean(angles < 1.0))
    if p50 < 1.0 or fraction >= 0.5:
        level = "HIGH_STRUCTURAL_RISK"
    elif p10 < 1.0:
        level = "STRUCTURAL_RISK"
    else:
        level = "NONE"
    return {
        "p10_deg": float(p10),
        "p25_deg": float(p25),
        "p50_deg": float(p50),
        "fraction_lt_1deg": fraction,
        "level": level,
        "repair_trigger": False,
    }


def planarity_diagnostics(
    world_points: np.ndarray,
    *,
    camera_center_world: np.ndarray,
    rho_threshold: float = 1e-3,
    normalized_rmse_threshold: float = 0.01,
) -> PlanarityDiagnostics:
    """Measure point-set planarity using the smallest covariance eigenvalue."""

    points = np.asarray(world_points, dtype=float)
    center = np.asarray(camera_center_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("world_points must have shape (N, 3) with N >= 3")
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(points).all():
        raise ValueError("points and camera center must be finite")
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    rho = float(eigenvalues[-1] / total) if total > 0 else 0.0
    plane_rmse = float(np.sqrt(eigenvalues[-1]))
    median_depth = float(np.median(np.linalg.norm(points - center, axis=1)))
    normalized = float(plane_rmse / median_depth) if median_depth > 0 else float("inf")
    return PlanarityDiagnostics(
        eigenvalues=eigenvalues,
        rho_plane=rho,
        plane_rmse=plane_rmse,
        normalized_plane_rmse=normalized,
        is_planar=bool(rho <= rho_threshold and normalized <= normalized_rmse_threshold),
    )
