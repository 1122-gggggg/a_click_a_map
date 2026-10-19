"""Transparent ambiguity-aware localization contracts.

This module operates on immutable map metadata and already-computed pose hypotheses.  It
does not match images, triangulate points, or update an SfM reconstruction.  Fisher
information is intentionally treated as local conditioning around one hypothesis;
global uniqueness is decided from independent reference groups and SE(3) pose modes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from river_map_quality.pose_attribution import camera_center, rotation_distance_deg

FAILURE_TAXONOMY = {
    "CASE_A": "MATCHING_OR_ANCHORING_WEAKNESS",
    "CASE_B": "TRACK_FRAGMENTATION",
    "CASE_C": "LOCAL_POSE_CONDITIONING",
    "CASE_D": "LOW_PARALLAX_STRUCTURAL_RISK",
    "CASE_E": "GLOBAL_POSE_MULTIMODALITY_REFERENCE_ALIASING",
    "CASE_F": "MAP_SUPPORT_COLLAPSE",
    "CASE_V": "VIEW_GRAPH_INDEPENDENCE_WEAKNESS",
}

CURRENT_ONLY = "CURRENT_ONLY"
HISTORICAL_ONLY = "HISTORICAL_ONLY"
MIXED = "MIXED"
REFERENCE_SOURCE_ROLES = frozenset((CURRENT_ONLY, HISTORICAL_ONLY, MIXED))

ACCEPTED_DECISIONS = {
    "ACCEPT",
    "ACCEPT_TEMPORAL_DISAMBIGUATED",
    "ACCEPT_CURRENT_STRONG",
    "ACCEPT_CURRENT_REFINED",
    "ACCEPT_CURRENT_FALLBACK",
    "ACCEPT_HISTORICAL_RECOVERY",
}

# A mode is admissible only when one concrete PnP solve supplies its pose,
# support, and all quality metrics.  A future one-to-one union/refit path may
# add another provenance value, but raw member aggregation is diagnostic only.
ATOMIC_SUPPORT_PROVENANCE = "ATOMIC_PNP_SOLVE"
UNVERIFIED_MULTI_MEMBER_SUPPORT = "UNVERIFIED_MULTI_MEMBER_SUPPORT"
COMPACT_PNP_BATCH = "COMPACT_PNP_BATCH"
NON_COMPACT_TRANSITIVE_GROUP = "NON_COMPACT_TRANSITIVE_GROUP"

# Pairwise pose differences use log(T_a @ inverse(T_b)) for world-to-camera
# transforms.  This is invariant to a common world-coordinate change and makes
# the translation term coupled correctly to rotation rather than comparing
# camera centers and rotations as unrelated coordinates.
LEFT_INVARIANT_SE3_LOG_CONVENTION = "LEFT_INVARIANT_WORLD_TO_CAMERA_LOG"


@dataclass(frozen=True)
class ReferenceGroupingConfig:
    maximum_frame_gap: int = 3
    maximum_center_distance: float = 0.25
    maximum_view_angle_deg: float = 20.0
    minimum_shared_landmarks: int = 30

    def __post_init__(self) -> None:
        if self.maximum_frame_gap < 0:
            raise ValueError("maximum_frame_gap cannot be negative")
        if self.maximum_center_distance <= 0 or self.maximum_view_angle_deg <= 0:
            raise ValueError("map-pose grouping thresholds must be positive")
        if self.minimum_shared_landmarks <= 0:
            raise ValueError("minimum_shared_landmarks must be positive")


@dataclass(frozen=True)
class AmbiguityConfig:
    rotation_sigma_deg: float = 0.5
    center_sigma: float = 0.01
    normalized_mode_distance_threshold: float = 4.0
    minimum_hypothesis_inliers: int = 20
    minimum_production_inliers: int = 80
    minimum_positive_depth_ratio: float = 0.9
    maximum_reprojection_p90: float = 5.0
    minimum_hull_coverage: float = 0.10
    minimum_grid_occupancy: int = 6
    maximum_fim_condition: float = 1000.0
    rotation_residual_scale_deg: float = 2.0
    translation_residual_scale_deg: float = 20.0
    maximum_temporal_rotation_residual_deg: float = 2.0
    maximum_temporal_translation_residual_deg: float = 30.0
    absolute_weight: float = 1.0
    temporal_rotation_weight: float = 2.0
    temporal_translation_weight: float = 0.5
    minimum_temporal_margin: float = 2.0
    minimum_sequence_margin: float = 2.0
    minimum_lost_frames: int = 3
    allow_translation_antipode: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.rotation_sigma_deg,
            self.center_sigma,
            self.normalized_mode_distance_threshold,
            self.minimum_hypothesis_inliers,
            self.minimum_production_inliers,
            self.maximum_reprojection_p90,
            self.minimum_hull_coverage,
            self.minimum_grid_occupancy,
            self.maximum_fim_condition,
            self.rotation_residual_scale_deg,
            self.translation_residual_scale_deg,
            self.maximum_temporal_rotation_residual_deg,
            self.maximum_temporal_translation_residual_deg,
            self.minimum_lost_frames,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("ambiguity thresholds and scales must be positive")
        if not 0 < self.minimum_positive_depth_ratio <= 1:
            raise ValueError("minimum_positive_depth_ratio must be in (0, 1]")
        if any(
            value < 0
            for value in (
                self.absolute_weight,
                self.temporal_rotation_weight,
                self.temporal_translation_weight,
                self.minimum_temporal_margin,
                self.minimum_sequence_margin,
            )
        ):
            raise ValueError("ambiguity weights and margins cannot be negative")


@dataclass(frozen=True)
class ReferenceEdge:
    reference_a: str
    reference_b: str
    reason: str
    shared_landmarks: int
    center_distance: float
    view_angle_deg: float


@dataclass(frozen=True)
class ReferenceGroup:
    group_id: str
    reference_ids: tuple[str, ...]
    route_count: int
    center_diameter: float
    view_angle_diameter_deg: float
    admission_batch_status: str


@dataclass(frozen=True)
class ReferenceGroupingResult:
    groups: tuple[ReferenceGroup, ...]
    edges: tuple[ReferenceEdge, ...]
    config: ReferenceGroupingConfig


@dataclass(frozen=True)
class PoseHypothesisRecord:
    hypothesis_id: str
    reference_group_id: str
    pose: np.ndarray
    reference_ids: tuple[str, ...]
    raw_matches: int
    verified_matches: int
    unique_2d3d: int
    pnp_inliers: int
    pnp_inlier_ratio: float
    positive_depth_ratio: float
    reprojection_mean: float | None
    reprojection_median: float | None
    reprojection_p90: float | None
    convex_hull_coverage: float
    grid_occupancy: int
    fim_condition: float | None
    fim_lambda_min: float | None
    track_statistics: Mapping[str, float | int | None]
    parallax_statistics: Mapping[str, float | int | None]
    support_provenance: str = ATOMIC_SUPPORT_PROVENANCE
    source_role: str = CURRENT_ONLY

    def __post_init__(self) -> None:
        matrix = np.asarray(self.pose, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("hypothesis pose must be a finite 4x4 matrix")
        if not self.reference_ids:
            raise ValueError("hypothesis needs at least one reference")
        counts = (
            self.raw_matches,
            self.verified_matches,
            self.unique_2d3d,
            self.pnp_inliers,
            self.grid_occupancy,
        )
        if any(value < 0 for value in counts):
            raise ValueError("hypothesis counts cannot be negative")
        if self.source_role not in REFERENCE_SOURCE_ROLES:
            raise ValueError(f"unknown reference source role: {self.source_role}")


@dataclass(frozen=True)
class GeodesicPoseHypothesis:
    """One solved pose retained for quality-independent mode analysis."""

    hypothesis_id: str
    pose: np.ndarray

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("geodesic pose hypothesis ID cannot be empty")
        _rigid_transform(self.pose)


@dataclass(frozen=True)
class GeodesicPoseMode:
    """A compact complete-linkage cluster with its deterministic medoid."""

    mode_id: str
    hypothesis_ids: tuple[str, ...]
    representative_hypothesis_id: str
    maximum_member_geodesic_distance: float
    clustering_distance_convention: str


@dataclass(frozen=True)
class PoseModeRecord:
    mode_id: str
    hypothesis_ids: tuple[str, ...]
    reference_group_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    representative_pose: np.ndarray
    pnp_inliers: int
    reprojection_p90: float | None
    convex_hull_coverage: float
    grid_occupancy: int
    fim_condition: float | None
    absolute_score: float
    quality_pass: bool
    local_conditioning: str
    support_witness_hypothesis_id: str | None
    support_witness_source_role: str
    support_provenance: str
    raw_member_inlier_sum_diagnostic: int
    maximum_member_inliers_diagnostic: int
    maximum_member_geodesic_distance: float
    clustering_distance_convention: str
    source_roles: tuple[str, ...] = (CURRENT_ONLY,)


@dataclass(frozen=True)
class LocalizationDecision:
    status: str
    selected_mode_id: str | None
    local_conditioning: str
    global_uniqueness: str
    failure_cases: tuple[str, ...]
    reasons: tuple[str, ...]
    source_role: str | None = None
    current_first_state: str = "DISABLED"


@dataclass(frozen=True)
class RelativePoseObservation:
    rotation: np.ndarray
    translation_direction_camera2: np.ndarray
    cheirality_sign_trusted: bool

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=float)
        direction = np.asarray(self.translation_direction_camera2, dtype=float)
        if rotation.shape != (3, 3) or direction.shape != (3,):
            raise ValueError("relative rotation/direction have invalid shape")
        if not np.isfinite(rotation).all() or not np.isfinite(direction).all():
            raise ValueError("relative pose observation must be finite")
        if np.linalg.norm(direction) <= np.finfo(float).eps:
            raise ValueError("relative translation direction cannot be zero")


@dataclass(frozen=True)
class TemporalCandidate:
    mode_id: str
    temporal_rotation_residual_deg: float
    temporal_translation_direction_residual_deg: float
    translation_sign_flipped: bool
    total_score: float


@dataclass(frozen=True)
class TemporalArbitrationResult:
    status: str
    selected_mode_id: str | None
    score_margin: float | None
    candidates: tuple[TemporalCandidate, ...]


@dataclass(frozen=True)
class LostSequenceResult:
    status: str
    selected_mode_ids: tuple[str, ...]
    best_path_score: float | None
    path_score_margin: float


def _route(name: str) -> str:
    parts = PurePosixPath(name).parts
    return parts[0] if len(parts) > 1 else str(name)


def _frame(name: str) -> int | None:
    try:
        return int(PurePosixPath(name).stem)
    except ValueError:
        return None


def _view_direction(pose: np.ndarray) -> np.ndarray:
    direction = np.asarray(pose, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return direction / np.linalg.norm(direction)


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


def _components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    output = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            stack.extend(sorted(neighbors, reverse=True))
        output.append(sorted(component))
    return output


def build_reference_groups(
    references: Sequence[str],
    poses_by_name: Mapping[str, np.ndarray],
    point_ids_by_name: Mapping[str, frozenset[int] | set[int]],
    *,
    config: ReferenceGroupingConfig,
) -> ReferenceGroupingResult:
    """Build non-overlapping reference groups from map connectivity and pose metadata."""

    names = tuple(str(reference) for reference in references)
    if not names or len(set(names)) != len(names):
        raise ValueError("references must be non-empty and unique")
    missing_pose = set(names) - set(poses_by_name)
    if missing_pose:
        raise KeyError(f"reference poses are missing: {sorted(missing_pose)!r}")
    centers = [camera_center(poses_by_name[name]) for name in names]
    views = [_view_direction(poses_by_name[name]) for name in names]
    adjacency = [set() for _ in names]
    edges: list[ReferenceEdge] = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            name_a, name_b = names[first], names[second]
            center_distance = float(np.linalg.norm(centers[first] - centers[second]))
            view_angle = _angle_deg(views[first], views[second])
            shared = len(
                set(point_ids_by_name.get(name_a, ())) & set(point_ids_by_name.get(name_b, ()))
            )
            frame_a, frame_b = _frame(name_a), _frame(name_b)
            temporal = (
                _route(name_a) == _route(name_b)
                and frame_a is not None
                and frame_b is not None
                and abs(frame_a - frame_b) <= config.maximum_frame_gap
            )
            map_pose = (
                center_distance <= config.maximum_center_distance
                and view_angle <= config.maximum_view_angle_deg
            )
            shared_edge = shared >= config.minimum_shared_landmarks
            reasons = []
            if temporal:
                reasons.append("same_route_temporal")
            if map_pose:
                reasons.append("map_pose")
            if shared_edge:
                reasons.append("shared_landmarks")
            if not reasons:
                continue
            adjacency[first].add(second)
            adjacency[second].add(first)
            for reason in reasons:
                edges.append(
                    ReferenceEdge(
                        name_a,
                        name_b,
                        reason,
                        shared,
                        center_distance,
                        view_angle,
                    )
                )
    groups = []
    for index, component in enumerate(_components(adjacency)):
        component_centers = [centers[node] for node in component]
        component_views = [views[node] for node in component]
        center_diameter = max(
            (
                float(np.linalg.norm(first - second))
                for position, first in enumerate(component_centers)
                for second in component_centers[position + 1 :]
            ),
            default=0.0,
        )
        view_diameter = max(
            (
                _angle_deg(first, second)
                for position, first in enumerate(component_views)
                for second in component_views[position + 1 :]
            ),
            default=0.0,
        )
        group_names = tuple(names[node] for node in component)
        # Connectivity is deliberately allowed to be transitive for diagnostics,
        # but a transitive component cannot be used as one pooled PnP batch when
        # its endpoints violate the compactness contract.  Otherwise A--B--C
        # would silently turn two local relations into a long-baseline claim.
        admission_batch_status = (
            COMPACT_PNP_BATCH
            if (
                center_diameter <= config.maximum_center_distance
                and view_diameter <= config.maximum_view_angle_deg
            )
            else NON_COMPACT_TRANSITIVE_GROUP
        )
        groups.append(
            ReferenceGroup(
                group_id=f"reference_group_{index}",
                reference_ids=group_names,
                route_count=len({_route(name) for name in group_names}),
                center_diameter=center_diameter,
                view_angle_diameter_deg=view_diameter,
                admission_batch_status=admission_batch_status,
            )
        )
    return ReferenceGroupingResult(tuple(groups), tuple(edges), config)


def _valid_hypothesis(hypothesis: PoseHypothesisRecord, config: AmbiguityConfig) -> bool:
    return bool(
        hypothesis.pnp_inliers >= config.minimum_hypothesis_inliers
        and hypothesis.positive_depth_ratio >= config.minimum_positive_depth_ratio
        and hypothesis.reprojection_p90 is not None
        and hypothesis.reprojection_p90 <= config.maximum_reprojection_p90
    )


def _quality_pass(hypothesis: PoseHypothesisRecord, config: AmbiguityConfig) -> bool:
    return bool(
        _valid_hypothesis(hypothesis, config)
        and hypothesis.convex_hull_coverage >= config.minimum_hull_coverage
        and hypothesis.grid_occupancy >= config.minimum_grid_occupancy
        and hypothesis.fim_condition is not None
        and hypothesis.fim_condition <= config.maximum_fim_condition
    )


def _absolute_score(hypothesis: PoseHypothesisRecord, config: AmbiguityConfig) -> float:
    reprojection = (
        hypothesis.reprojection_p90 / config.maximum_reprojection_p90
        if hypothesis.reprojection_p90 is not None
        else 2.0
    )
    inlier = 1.0 - min(max(hypothesis.pnp_inlier_ratio, 0.0), 1.0)
    coverage = 1.0 - min(max(hypothesis.convex_hull_coverage, 0.0), 1.0)
    occupancy = 1.0 - min(max(hypothesis.grid_occupancy / 16.0, 0.0), 1.0)
    condition = (
        math.log1p(max(hypothesis.fim_condition, 0.0)) / math.log1p(config.maximum_fim_condition)
        if hypothesis.fim_condition is not None
        else 2.0
    )
    support = 1.0 / len(hypothesis.reference_ids)
    return float(reprojection + inlier + coverage + occupancy + condition + support)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def _rigid_transform(pose: np.ndarray) -> np.ndarray:
    transform = np.asarray(pose, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("pose must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError("pose must use homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
        raise ValueError("pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("pose rotation must have determinant one")
    return transform


def _so3_logarithm(rotation: np.ndarray) -> np.ndarray:
    """Return the principal SO(3) logarithm, including a stable pi branch."""

    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew_part = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-7:
        return 0.5 * skew_part
    if math.pi - angle >= 1e-5:
        return (angle / (2.0 * math.sin(angle))) * skew_part

    diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
    axis_index = int(np.argmax(diagonal))
    axis = np.zeros(3)
    axis[axis_index] = math.sqrt(float(diagonal[axis_index]))
    if axis[axis_index] <= np.finfo(float).eps:
        raise ValueError("could not recover a stable pi-rotation axis")
    for coordinate in range(3):
        if coordinate != axis_index:
            axis[coordinate] = (
                rotation[coordinate, axis_index] + rotation[axis_index, coordinate]
            ) / (4.0 * axis[axis_index])
    axis /= np.linalg.norm(axis)
    return angle * axis


def se3_logarithm(transform: np.ndarray) -> np.ndarray:
    """Return [translation, rotation] for the principal SE(3) logarithm."""

    rigid = _rigid_transform(transform)
    omega = _so3_logarithm(rigid[:3, :3])
    angle = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    if angle < 1e-7:
        left_jacobian_inverse = np.eye(3) - 0.5 * omega_hat + omega_hat @ omega_hat / 12.0
    else:
        coefficient = (
            1.0 / angle**2
            - (1.0 + math.cos(angle)) / (2.0 * angle * math.sin(angle))
        )
        left_jacobian_inverse = (
            np.eye(3) - 0.5 * omega_hat + coefficient * (omega_hat @ omega_hat)
        )
    return np.concatenate((left_jacobian_inverse @ rigid[:3, 3], omega))


def _inverse_rigid_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def left_invariant_se3_distance(
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    *,
    translation_sigma: float,
    rotation_sigma_deg: float,
) -> float:
    """Measure a normalized world-to-camera SE(3) log displacement."""

    if translation_sigma <= 0 or rotation_sigma_deg <= 0:
        raise ValueError("SE(3) normalization scales must be positive")
    first = _rigid_transform(first_pose)
    second = _rigid_transform(second_pose)
    tangent = se3_logarithm(first @ _inverse_rigid_transform(second))
    return float(
        np.hypot(
            np.linalg.norm(tangent[:3]) / translation_sigma,
            np.linalg.norm(tangent[3:]) / math.radians(rotation_sigma_deg),
        )
    )


def _mode_distance(
    first: PoseHypothesisRecord,
    second: PoseHypothesisRecord,
    config: AmbiguityConfig,
) -> float:
    return left_invariant_se3_distance(
        first.pose,
        second.pose,
        translation_sigma=config.center_sigma,
        rotation_sigma_deg=config.rotation_sigma_deg,
    )


def _pairwise_geodesic_distances(
    hypotheses: Sequence[GeodesicPoseHypothesis],
    *,
    translation_sigma: float,
    rotation_sigma_deg: float,
) -> np.ndarray:
    distances = np.zeros((len(hypotheses), len(hypotheses)))
    for first in range(len(hypotheses)):
        for second in range(first + 1, len(hypotheses)):
            distance = left_invariant_se3_distance(
                hypotheses[first].pose,
                hypotheses[second].pose,
                translation_sigma=translation_sigma,
                rotation_sigma_deg=rotation_sigma_deg,
            )
            distances[first, second] = distance
            distances[second, first] = distance
    return distances


def _complete_linkage_components(
    hypothesis_ids: Sequence[str],
    distances: np.ndarray,
    *,
    distance_threshold: float,
) -> tuple[tuple[int, ...], ...]:
    """Make compact pose modes without transitive adjacency chaining."""

    clusters = [(index,) for index in range(len(hypothesis_ids))]
    while True:
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        for first in range(len(clusters)):
            for second in range(first + 1, len(clusters)):
                cross_distance = max(
                    float(distances[left, right])
                    for left in clusters[first]
                    for right in clusters[second]
                )
                if cross_distance <= distance_threshold:
                    member_ids = tuple(
                        sorted(
                            hypothesis_ids[index]
                            for index in (*clusters[first], *clusters[second])
                        )
                    )
                    candidates.append((cross_distance, member_ids, first, second))
        if not candidates:
            break
        _, _, first, second = min(candidates)
        clusters[first] = tuple(sorted((*clusters[first], *clusters[second])))
        del clusters[second]
    return tuple(clusters)


def cluster_geodesic_pose_modes(
    hypotheses: Sequence[GeodesicPoseHypothesis],
    *,
    translation_sigma: float,
    rotation_sigma_deg: float,
    normalized_distance_threshold: float,
) -> tuple[GeodesicPoseMode, ...]:
    """Cluster every solved pose under compact, left-invariant SE(3) geometry."""

    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            translation_sigma,
            rotation_sigma_deg,
            normalized_distance_threshold,
        )
    ):
        raise ValueError("geodesic clustering scales and threshold must be finite and positive")
    ordered = tuple(sorted(hypotheses, key=lambda hypothesis: hypothesis.hypothesis_id))
    if len({hypothesis.hypothesis_id for hypothesis in ordered}) != len(ordered):
        raise ValueError("geodesic pose hypothesis IDs must be unique")
    distances = _pairwise_geodesic_distances(
        ordered,
        translation_sigma=translation_sigma,
        rotation_sigma_deg=rotation_sigma_deg,
    )
    modes = []
    for component in _complete_linkage_components(
        tuple(hypothesis.hypothesis_id for hypothesis in ordered),
        distances,
        distance_threshold=normalized_distance_threshold,
    ):
        member_ids = tuple(ordered[index].hypothesis_id for index in component)
        representative_index = min(
            component,
            key=lambda index: (
                float(sum(distances[index, other] for other in component)),
                ordered[index].hypothesis_id,
            ),
        )
        maximum_distance = max(
            (
                float(distances[first, second])
                for position, first in enumerate(component)
                for second in component[position + 1 :]
            ),
            default=0.0,
        )
        modes.append(
            (
                member_ids,
                ordered[representative_index].hypothesis_id,
                maximum_distance,
            )
        )
    modes.sort(key=lambda mode: mode[0])
    return tuple(
        GeodesicPoseMode(
            mode_id=f"geodesic_pose_mode_{index}",
            hypothesis_ids=member_ids,
            representative_hypothesis_id=representative_hypothesis_id,
            maximum_member_geodesic_distance=maximum_distance,
            clustering_distance_convention=LEFT_INVARIANT_SE3_LOG_CONVENTION,
        )
        for index, (member_ids, representative_hypothesis_id, maximum_distance) in enumerate(modes)
    )


def cluster_pose_modes(
    hypotheses: Sequence[PoseHypothesisRecord],
    *,
    config: AmbiguityConfig,
) -> tuple[PoseModeRecord, ...]:
    """Cluster valid hypotheses by compact left-invariant SE(3) log modes.

    Complete linkage requires every member pair in a mode to pass the same
    normalized distance gate.  It deliberately rejects the chain effect of
    connected components: A≈B and B≈C cannot silently claim A≈C.
    """

    valid = sorted(
        (hypothesis for hypothesis in hypotheses if _valid_hypothesis(hypothesis, config)),
        key=lambda hypothesis: hypothesis.hypothesis_id,
    )
    geodesic_modes = cluster_geodesic_pose_modes(
        tuple(
            GeodesicPoseHypothesis(hypothesis.hypothesis_id, hypothesis.pose)
            for hypothesis in valid
        ),
        translation_sigma=config.center_sigma,
        rotation_sigma_deg=config.rotation_sigma_deg,
        normalized_distance_threshold=config.normalized_mode_distance_threshold,
    )
    valid_by_id = {hypothesis.hypothesis_id: hypothesis for hypothesis in valid}
    provisional = []
    for geodesic_mode in geodesic_modes:
        members = [valid_by_id[hypothesis_id] for hypothesis_id in geodesic_mode.hypothesis_ids]
        medoid = valid_by_id[geodesic_mode.representative_hypothesis_id]
        references = tuple(
            sorted({reference for member in members for reference in member.reference_ids})
        )
        # Never assemble a synthetic pose from one member's inlier count and
        # another member's coverage/FIM.  A mode can be admitted only through
        # one atomic PnP solve which carries the pose and every quality metric.
        # Multi-member union support is intentionally not implemented here: it
        # needs stable query-feature/landmark identities, one-to-one conflict
        # removal, refit PnP, and recomputation of every metric on that union.
        witnesses = [
            member
            for member in members
            if (
                member.support_provenance == ATOMIC_SUPPORT_PROVENANCE
                and _quality_pass(member, config)
            )
        ]
        witness = min(
            witnesses,
            key=lambda candidate: (
                _absolute_score(candidate, config),
                -candidate.pnp_inliers,
                candidate.hypothesis_id,
            ),
            default=None,
        )
        diagnostic = witness if witness is not None else medoid
        quality = witness is not None
        support_provenance = (
            ATOMIC_SUPPORT_PROVENANCE
            if witness is not None
            else (
                members[0].support_provenance
                if len(members) == 1
                else UNVERIFIED_MULTI_MEMBER_SUPPORT
            )
        )
        local = (
            "HEALTHY"
            if (
                diagnostic.fim_condition is not None
                and diagnostic.fim_condition <= config.maximum_fim_condition
            )
            else "DEGENERATE"
        )
        maximum_member_distance = geodesic_mode.maximum_member_geodesic_distance
        provisional.append(
            {
                "members": members,
                "pose": np.array(diagnostic.pose, copy=True),
                "references": references,
                "inliers": diagnostic.pnp_inliers,
                "reprojection": diagnostic.reprojection_p90,
                "coverage": diagnostic.convex_hull_coverage,
                "occupancy": diagnostic.grid_occupancy,
                "fim_condition": diagnostic.fim_condition,
                "score": _absolute_score(diagnostic, config),
                "quality": quality,
                "local": local,
                "witness_id": None if witness is None else witness.hypothesis_id,
                "support_provenance": support_provenance,
                "witness_source_role": diagnostic.source_role,
                "raw_member_inlier_sum": sum(member.pnp_inliers for member in members),
                "maximum_member_inliers": max(member.pnp_inliers for member in members),
                "maximum_member_geodesic_distance": maximum_member_distance,
                "source_roles": tuple(sorted({member.source_role for member in members})),
            }
        )
    provisional.sort(
        key=lambda row: (
            row["score"],
            -row["inliers"],
            tuple(member.hypothesis_id for member in row["members"]),
        )
    )
    return tuple(
        PoseModeRecord(
            mode_id=f"pose_mode_{index}",
            hypothesis_ids=tuple(member.hypothesis_id for member in row["members"]),
            reference_group_ids=tuple(
                sorted({member.reference_group_id for member in row["members"]})
            ),
            reference_ids=row["references"],
            representative_pose=row["pose"],
            pnp_inliers=row["inliers"],
            reprojection_p90=row["reprojection"],
            convex_hull_coverage=row["coverage"],
            grid_occupancy=row["occupancy"],
            fim_condition=row["fim_condition"],
            absolute_score=row["score"],
            quality_pass=row["quality"],
            local_conditioning=row["local"],
            support_witness_hypothesis_id=row["witness_id"],
            support_witness_source_role=row["witness_source_role"],
            support_provenance=row["support_provenance"],
            raw_member_inlier_sum_diagnostic=row["raw_member_inlier_sum"],
            maximum_member_inliers_diagnostic=row["maximum_member_inliers"],
            maximum_member_geodesic_distance=row["maximum_member_geodesic_distance"],
            clustering_distance_convention=LEFT_INVARIANT_SE3_LOG_CONVENTION,
            source_roles=row["source_roles"],
        )
        for index, row in enumerate(provisional)
    )


def calibrate_mode_scales(
    healthy_hypotheses: Mapping[str, Sequence[PoseHypothesisRecord]],
    *,
    rotation_floor_deg: float,
    center_floor: float,
    broad_rotation_gate_deg: float,
    broad_center_gate: float,
) -> dict[str, Any]:
    """Estimate SE(3) clustering scales from within-query healthy-pose dispersion."""

    if any(
        value <= 0
        for value in (
            rotation_floor_deg,
            center_floor,
            broad_rotation_gate_deg,
            broad_center_gate,
        )
    ):
        raise ValueError("calibration floors and broad gates must be positive")
    rotations = []
    centers = []
    source_queries = []
    for query_id, hypotheses in sorted(healthy_hypotheses.items()):
        query_used = False
        for first in range(len(hypotheses)):
            for second in range(first + 1, len(hypotheses)):
                tangent = se3_logarithm(
                    _rigid_transform(hypotheses[first].pose)
                    @ _inverse_rigid_transform(_rigid_transform(hypotheses[second].pose))
                )
                rotation = math.degrees(float(np.linalg.norm(tangent[3:])))
                translation = float(np.linalg.norm(tangent[:3]))
                if (
                    rotation <= broad_rotation_gate_deg
                    and translation <= broad_center_gate
                ):
                    rotations.append(rotation)
                    centers.append(translation)
                    query_used = True
        if query_used:
            source_queries.append(query_id)
    if not rotations:
        return {
            "rotation_sigma_deg": rotation_floor_deg,
            "center_sigma": center_floor,
            "sample_pair_count": 0,
            "source_query_ids": [],
            "estimator": "p50 with configured lower floors; no eligible pair, floors used",
        }
    return {
        "rotation_sigma_deg": max(rotation_floor_deg, float(np.median(rotations))),
        "center_sigma": max(center_floor, float(np.median(centers))),
        "sample_pair_count": len(rotations),
        "source_query_ids": source_queries,
        "estimator": "p50 of broad-gated within-query healthy hypothesis pairs",
        "rotation_floor_deg": rotation_floor_deg,
        "center_floor": center_floor,
        "broad_rotation_gate_deg": broad_rotation_gate_deg,
        "broad_center_gate": broad_center_gate,
    }


def calibrate_mode_scales_from_residuals(
    healthy_residuals: Mapping[str, Mapping[str, float | None]],
    *,
    rotation_floor_deg: float,
    center_floor: float,
    broad_rotation_gate_deg: float,
    broad_center_gate: float,
) -> dict[str, Any]:
    """Fallback calibration from repeatable M0-relative healthy-control residuals.

    This is deliberately labeled as a repeatability scale.  M0 supplies a frozen
    coordinate-frame comparison, not measurement-grade ground truth.
    """

    if any(
        value <= 0
        for value in (
            rotation_floor_deg,
            center_floor,
            broad_rotation_gate_deg,
            broad_center_gate,
        )
    ):
        raise ValueError("calibration floors and broad gates must be positive")
    rotations: list[float] = []
    centers: list[float] = []
    sources: list[str] = []
    for query_id, row in sorted(healthy_residuals.items()):
        rotation = row.get("rotation_error_deg")
        center = row.get("center_error")
        if rotation is None or center is None:
            continue
        rotation_value = float(rotation)
        center_value = float(center)
        if not math.isfinite(rotation_value) or not math.isfinite(center_value):
            continue
        if (
            0 <= rotation_value <= broad_rotation_gate_deg
            and 0 <= center_value <= broad_center_gate
        ):
            rotations.append(rotation_value)
            centers.append(center_value)
            sources.append(query_id)
    return {
        "rotation_sigma_deg": max(
            rotation_floor_deg,
            float(np.median(rotations)) if rotations else rotation_floor_deg,
        ),
        "center_sigma": max(
            center_floor,
            float(np.median(centers)) if centers else center_floor,
        ),
        "sample_query_count": len(rotations),
        "sample_pair_count": 0,
        "source_query_ids": sources,
        "estimator": (
            "p50 of broad-gated M0-relative healthy-control residuals with configured lower floors"
        ),
        "rotation_floor_deg": rotation_floor_deg,
        "center_floor": center_floor,
        "broad_rotation_gate_deg": broad_rotation_gate_deg,
        "broad_center_gate": broad_center_gate,
        "ground_truth_contract": (
            "M0-relative healthy-control repeatability scale; M0 is not absolute ground truth"
        ),
    }


def _mode_cluster_key(mode: PoseModeRecord) -> tuple[str, ...]:
    routes = tuple(sorted({_route(reference) for reference in mode.reference_ids}))
    return routes or mode.reference_group_ids


def build_reference_alias_graph(
    query_modes: Mapping[str, Sequence[PoseModeRecord]],
    *,
    retrieval_scores: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """Accumulate visually confusable reference-region pairs without changing the map."""

    relationships: dict[tuple[tuple[str, ...], tuple[str, ...]], dict[str, Any]] = {}
    multimodal_query_count = 0
    for query_id, modes in sorted(query_modes.items()):
        strong = [mode for mode in modes if mode.quality_pass]
        if len(strong) < 2:
            continue
        multimodal_query_count += 1
        for first_index in range(len(strong)):
            for second_index in range(first_index + 1, len(strong)):
                first = strong[first_index]
                second = strong[second_index]
                first_key, second_key = _mode_cluster_key(first), _mode_cluster_key(second)
                key = tuple(sorted((first_key, second_key)))
                row = relationships.setdefault(
                    key,
                    {
                        "cluster_A": list(key[0]),
                        "cluster_B": list(key[1]),
                        "queries_triggered": [],
                        "rotation_separations": [],
                        "center_separations": [],
                        "visual_scores": [],
                        "cluster_A_reference_ids": set(),
                        "cluster_B_reference_ids": set(),
                    },
                )
                ordered_modes = (first, second) if first_key <= second_key else (second, first)
                row["cluster_A_reference_ids"].update(ordered_modes[0].reference_ids)
                row["cluster_B_reference_ids"].update(ordered_modes[1].reference_ids)
                row["queries_triggered"].append(query_id)
                row["rotation_separations"].append(
                    rotation_distance_deg(
                        first.representative_pose[:3, :3],
                        second.representative_pose[:3, :3],
                    )
                )
                row["center_separations"].append(
                    float(
                        np.linalg.norm(
                            camera_center(first.representative_pose)
                            - camera_center(second.representative_pose)
                        )
                    )
                )
                scores = retrieval_scores.get(query_id, {})
                first_scores = [scores[name] for name in first.reference_ids if name in scores]
                second_scores = [scores[name] for name in second.reference_ids if name in scores]
                if first_scores and second_scores:
                    row["visual_scores"].append(float(min(max(first_scores), max(second_scores))))
    output = []
    denominator = max(multimodal_query_count, 1)
    for row in relationships.values():
        output.append(
            {
                "cluster_A": row["cluster_A"],
                "cluster_B": row["cluster_B"],
                "cluster_A_reference_ids": sorted(row["cluster_A_reference_ids"]),
                "cluster_B_reference_ids": sorted(row["cluster_B_reference_ids"]),
                "visual_alias_score": (
                    float(np.median(row["visual_scores"])) if row["visual_scores"] else None
                ),
                "pose_separation": {
                    "rotation_deg_p50": float(np.median(row["rotation_separations"])),
                    "center_p50": float(np.median(row["center_separations"])),
                },
                "queries_triggered": sorted(row["queries_triggered"]),
                "mode_switch_frequency": len(set(row["queries_triggered"])) / denominator,
            }
        )
    output.sort(key=lambda row: (row["cluster_A"], row["cluster_B"]))
    return {
        "schema_version": 1,
        "contract": (
            "diagnostic reference alias metadata only; it does not merge landmarks or "
            "change map geometry"
        ),
        "multimodal_query_count": multimodal_query_count,
        "alias_relationships": output,
    }


def _pairwise_dispersions(
    hypotheses: Sequence[PoseHypothesisRecord],
) -> tuple[float | None, float | None]:
    rotations = []
    centers = []
    for first in range(len(hypotheses)):
        for second in range(first + 1, len(hypotheses)):
            rotations.append(
                rotation_distance_deg(
                    hypotheses[first].pose[:3, :3], hypotheses[second].pose[:3, :3]
                )
            )
            centers.append(
                float(
                    np.linalg.norm(
                        camera_center(hypotheses[first].pose)
                        - camera_center(hypotheses[second].pose)
                    )
                )
            )
    return (
        float(np.percentile(rotations, 90)) if rotations else None,
        float(np.percentile(centers, 90)) if centers else None,
    )


def pose_multimodality_metrics(
    hypotheses: Sequence[PoseHypothesisRecord], modes: Sequence[PoseModeRecord]
) -> dict[str, Any]:
    strong = [mode for mode in modes if mode.quality_pass]
    dominant = strong[0] if strong else (modes[0] if modes else None)
    secondary = strong[1] if len(strong) > 1 else None
    rotation_dispersion, center_dispersion = _pairwise_dispersions(hypotheses)
    rotation_separation = None
    center_separation = None
    if dominant is not None and secondary is not None:
        rotation_separation = rotation_distance_deg(
            dominant.representative_pose[:3, :3], secondary.representative_pose[:3, :3]
        )
        center_separation = float(
            np.linalg.norm(
                camera_center(dominant.representative_pose)
                - camera_center(secondary.representative_pose)
            )
        )
    return {
        "reference_group_count": len({hypothesis.reference_group_id for hypothesis in hypotheses}),
        "valid_pose_hypothesis_count": len(hypotheses),
        "raw_pose_mode_count": len(modes),
        "pose_mode_count": len(strong),
        "dominant_mode_reference_count": (
            len(dominant.reference_ids) if dominant is not None else 0
        ),
        "secondary_mode_reference_count": (
            len(secondary.reference_ids) if secondary is not None else 0
        ),
        "dominant_mode_inliers": dominant.pnp_inliers if dominant is not None else 0,
        "secondary_mode_inliers": secondary.pnp_inliers if secondary is not None else 0,
        "dominant_mode_support_witness_id": (
            dominant.support_witness_hypothesis_id if dominant is not None else None
        ),
        "secondary_mode_support_witness_id": (
            secondary.support_witness_hypothesis_id if secondary is not None else None
        ),
        "dominant_mode_support_provenance": (
            dominant.support_provenance if dominant is not None else None
        ),
        "secondary_mode_support_provenance": (
            secondary.support_provenance if secondary is not None else None
        ),
        "dominant_mode_raw_member_inlier_sum_diagnostic": (
            dominant.raw_member_inlier_sum_diagnostic if dominant is not None else 0
        ),
        "secondary_mode_raw_member_inlier_sum_diagnostic": (
            secondary.raw_member_inlier_sum_diagnostic if secondary is not None else 0
        ),
        "dominant_mode_reproj_p90": (dominant.reprojection_p90 if dominant is not None else None),
        "secondary_mode_reproj_p90": (
            secondary.reprojection_p90 if secondary is not None else None
        ),
        "mode_rotation_separation_deg": rotation_separation,
        "mode_center_separation": center_separation,
        "pose_hypothesis_rotation_dispersion": rotation_dispersion,
        "pose_hypothesis_center_dispersion": center_dispersion,
        "mode_score_margin": (
            secondary.absolute_score - dominant.absolute_score
            if dominant is not None and secondary is not None
            else None
        ),
    }


def _is_current_mode(mode: PoseModeRecord) -> bool:
    return CURRENT_ONLY in mode.source_roles or MIXED in mode.source_roles


def _is_historical_mode(mode: PoseModeRecord) -> bool:
    return HISTORICAL_ONLY in mode.source_roles or MIXED in mode.source_roles


def _current_strong_modes(
    modes: Sequence[PoseModeRecord], config: AmbiguityConfig
) -> tuple[PoseModeRecord, ...]:
    return tuple(
        mode
        for mode in modes
        if (
            mode.support_witness_source_role == CURRENT_ONLY
            and mode.quality_pass
            and mode.support_provenance == ATOMIC_SUPPORT_PROVENANCE
            and mode.pnp_inliers >= config.minimum_production_inliers
            and mode.local_conditioning == "HEALTHY"
        )
    )


def _current_first_acceptance(
    modes: Sequence[PoseModeRecord],
    *,
    dominant: PoseModeRecord,
    strong: Sequence[PoseModeRecord],
    config: AmbiguityConfig,
) -> LocalizationDecision | None:
    """Choose a strong current mode before any conflicting historical alternative."""

    current_modes = _current_strong_modes(modes, config)
    if len(current_modes) == 1:
        current = current_modes[0]
        rejected_historical = tuple(
            mode.mode_id
            for mode in strong
            if mode.mode_id != current.mode_id and _is_historical_mode(mode)
        )
        role = MIXED if _is_historical_mode(current) else CURRENT_ONLY
        state = (
            "CURRENT_STRONG_HISTORICAL_CONFLICT_REJECTED"
            if rejected_historical
            else "CURRENT_STRONG_UNCONFLICTED"
        )
        status = "ACCEPT_CURRENT_REFINED" if role == MIXED else "ACCEPT_CURRENT_STRONG"
        reasons = [
            "strong current-only support remains the selected absolute pose",
        ]
        if role == MIXED:
            reasons.append("historical support is coherent with the selected current pose mode")
        if rejected_historical:
            reasons.append(
                "conflicting historical mode(s) rejected: " + ", ".join(rejected_historical)
            )
        return LocalizationDecision(
            status,
            current.mode_id,
            current.local_conditioning,
            "CURRENT_FIRST_CONFLICT_REJECTED" if rejected_historical else "CURRENT_FIRST_UNIMODAL",
            (),
            tuple(reasons),
            role,
            state,
        )
    if len(current_modes) > 1:
        return None
    if _is_historical_mode(dominant):
        return LocalizationDecision(
            "ACCEPT_HISTORICAL_RECOVERY",
            dominant.mode_id,
            dominant.local_conditioning,
            (
                "HISTORICAL_RECOVERY_UNIMODAL"
                if len(strong) <= 1
                else "HISTORICAL_RECOVERY_MULTIMODAL"
            ),
            (),
            (
                "no strong current-only mode survived production gates; "
                "historical support is admitted as recovery",
            ),
            HISTORICAL_ONLY if not _is_current_mode(dominant) else MIXED,
            "CURRENT_WEAK_OR_LOST_HISTORICAL_RECOVERY",
        )
    return LocalizationDecision(
        "ACCEPT_CURRENT_FALLBACK",
        dominant.mode_id,
        dominant.local_conditioning,
        "CURRENT_FALLBACK_UNIMODAL" if len(strong) <= 1 else "CURRENT_FALLBACK_MULTIMODAL",
        (),
        ("no strong current-only mode survived production gates; current fallback selected",),
        CURRENT_ONLY,
        "CURRENT_WEAK_OR_LOST_CURRENT_FALLBACK",
    )


def localization_decision(
    modes: Sequence[PoseModeRecord],
    *,
    reject_multimodal: bool,
    config: AmbiguityConfig,
    current_first: bool = False,
) -> LocalizationDecision:
    if not modes:
        return LocalizationDecision(
            "REJECT_LOW_SUPPORT",
            None,
            "UNKNOWN",
            "UNKNOWN",
            ("CASE_F",),
            ("no valid grouped PnP hypothesis",),
        )
    dominant = modes[0]
    strong = [mode for mode in modes if mode.quality_pass]

    if current_first:
        resolution = _current_first_acceptance(
            modes,
            dominant=dominant,
            strong=strong,
            config=config,
        )
        if resolution is not None and resolution.current_first_state.startswith("CURRENT_STRONG"):
            return resolution

    if dominant.support_provenance != ATOMIC_SUPPORT_PROVENANCE:
        cases = ["CASE_V"]
        if dominant.support_provenance == UNVERIFIED_MULTI_MEMBER_SUPPORT:
            reason = (
                "multiple grouped hypotheses lack one atomic quality witness; "
                "raw member support is diagnostic only"
            )
        else:
            reason = (
                "reference group is not a compact PnP batch and cannot supply "
                "admissible support"
            )
        return LocalizationDecision(
            "REJECT_UNVERIFIED_SUPPORT",
            None,
            dominant.local_conditioning,
            "UNRESOLVED",
            tuple(cases),
            (reason,),
        )
    if dominant.pnp_inliers < config.minimum_production_inliers:
        cases = ["CASE_F"]
        if dominant.local_conditioning == "DEGENERATE":
            cases.append("CASE_C")
        return LocalizationDecision(
            "REJECT_LOW_SUPPORT",
            None,
            dominant.local_conditioning,
            "UNRESOLVED",
            tuple(cases),
            (f"dominant grouped inliers={dominant.pnp_inliers}",),
        )
    multimodal_candidates = (
        [
            mode
            for mode in strong
            if (
                mode.support_provenance == ATOMIC_SUPPORT_PROVENANCE
                and mode.pnp_inliers >= config.minimum_production_inliers
            )
        ]
        if current_first
        else strong
    )
    if reject_multimodal and len(multimodal_candidates) >= 2:
        return LocalizationDecision(
            "REJECT_MULTIMODAL",
            None,
            dominant.local_conditioning,
            "MULTIMODAL",
            ("CASE_E",),
            ("at least two independently supported pose modes pass quality gates",),
        )
    if dominant.local_conditioning == "DEGENERATE":
        return LocalizationDecision(
            "REJECT_LOCAL_DEGENERACY",
            None,
            "DEGENERATE",
            "UNIMODAL" if len(strong) <= 1 else "MULTIMODAL",
            ("CASE_C",),
            ("dominant pose mode has poor local FIM conditioning",),
        )
    if not dominant.quality_pass:
        return LocalizationDecision(
            "REJECT_POOR_SPATIAL_COVERAGE",
            None,
            dominant.local_conditioning,
            "UNIMODAL",
            ("CASE_F",),
            ("dominant mode fails spatial-distribution quality gates",),
        )
    if current_first:
        resolution = _current_first_acceptance(
            modes,
            dominant=dominant,
            strong=strong,
            config=config,
        )
        if resolution is not None:
            return resolution
    return LocalizationDecision(
        "ACCEPT",
        dominant.mode_id,
        dominant.local_conditioning,
        "UNIMODAL" if len(strong) <= 1 else "MULTIMODAL_NOT_REJECTED",
        (),
        ("dominant grouped pose mode passed configured quality gates",),
    )
def _temporal_candidate(
    previous_pose: np.ndarray,
    mode: PoseModeRecord,
    observation: RelativePoseObservation,
    config: AmbiguityConfig,
) -> TemporalCandidate:
    previous = np.asarray(previous_pose, dtype=float)
    current = np.asarray(mode.representative_pose, dtype=float)
    relative_rotation = current[:3, :3] @ previous[:3, :3].T
    rotation_residual = rotation_distance_deg(relative_rotation, observation.rotation)
    displacement = camera_center(current) - camera_center(previous)
    direction = np.asarray(observation.translation_direction_camera2, dtype=float)
    direction /= np.linalg.norm(direction)
    predicted_world = -current[:3, :3].T @ direction
    signed_error = _angle_deg(displacement, predicted_world)
    flipped_error = _angle_deg(displacement, -predicted_world)
    allow_flip = config.allow_translation_antipode or not observation.cheirality_sign_trusted
    flipped = bool(allow_flip and flipped_error < signed_error)
    translation_residual = flipped_error if flipped else signed_error
    total = (
        config.absolute_weight * mode.absolute_score
        + config.temporal_rotation_weight * rotation_residual / config.rotation_residual_scale_deg
        + config.temporal_translation_weight
        * translation_residual
        / config.translation_residual_scale_deg
    )
    return TemporalCandidate(
        mode.mode_id,
        rotation_residual,
        translation_residual,
        flipped,
        float(total),
    )


def temporal_arbitration(
    previous_pose: np.ndarray,
    modes: Sequence[PoseModeRecord],
    observation: RelativePoseObservation,
    *,
    config: AmbiguityConfig,
) -> TemporalArbitrationResult:
    # Temporal geometry selects among already admissible absolute hypotheses;
    # it must never upgrade an unverified group/union into an accepted pose.
    admissible = [
        mode
        for mode in modes
        if (
            mode.quality_pass
            and mode.support_provenance == ATOMIC_SUPPORT_PROVENANCE
            and mode.support_witness_hypothesis_id is not None
        )
    ]
    if not admissible:
        return TemporalArbitrationResult("RELOCALIZATION_PENDING", None, None, ())
    candidates = tuple(
        sorted(
            (
                _temporal_candidate(previous_pose, mode, observation, config)
                for mode in admissible
            ),
            key=lambda row: (row.total_score, row.mode_id),
        )
    )
    best = candidates[0]
    margin = candidates[1].total_score - best.total_score if len(candidates) > 1 else None
    valid = (
        best.temporal_rotation_residual_deg <= config.maximum_temporal_rotation_residual_deg
        and best.temporal_translation_direction_residual_deg
        <= config.maximum_temporal_translation_residual_deg
    )
    decisive = margin is None or margin >= config.minimum_temporal_margin
    if valid and decisive:
        status = "ACCEPT_TEMPORAL_DISAMBIGUATED"
        selected = best.mode_id
    else:
        status = "REJECT_MULTIMODAL" if len(admissible) >= 2 else "RELOCALIZATION_PENDING"
        selected = None
    return TemporalArbitrationResult(status, selected, margin, candidates)


def optimize_lost_sequence(
    frame_modes: Sequence[Sequence[PoseModeRecord]],
    relative_observations: Sequence[RelativePoseObservation],
    *,
    config: AmbiguityConfig,
) -> LostSequenceResult:
    if len(frame_modes) < config.minimum_lost_frames:
        return LostSequenceResult("RELOCALIZATION_PENDING", (), None, 0.0)
    if len(relative_observations) != len(frame_modes) - 1 or any(
        not modes for modes in frame_modes
    ):
        raise ValueError("lost sequence needs one relative observation per frame transition")
    paths = [
        (config.absolute_weight * mode.absolute_score, (mode.mode_id,), mode)
        for mode in frame_modes[0]
    ]
    for frame_index in range(1, len(frame_modes)):
        observation = relative_observations[frame_index - 1]
        next_paths = []
        for mode in frame_modes[frame_index]:
            for score, path, previous_mode in paths:
                transition = _temporal_candidate(
                    previous_mode.representative_pose,
                    mode,
                    observation,
                    config,
                )
                next_paths.append(
                    (
                        score
                        + config.absolute_weight * mode.absolute_score
                        + transition.total_score
                        - config.absolute_weight * mode.absolute_score,
                        (*path, mode.mode_id),
                        mode,
                    )
                )
        paths = next_paths
    paths.sort(key=lambda row: (row[0], row[1]))
    best = paths[0]
    margin = paths[1][0] - best[0] if len(paths) > 1 else float("inf")
    status = (
        "ACCEPT_TEMPORAL_DISAMBIGUATED"
        if margin >= config.minimum_sequence_margin
        else "RELOCALIZATION_PENDING"
    )
    selected = best[1] if status.startswith("ACCEPT") else ()
    return LostSequenceResult(status, selected, best[0], margin)


def high_confidence_wrong_pose_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    accepted = [row for row in rows if row.get("decision") in ACCEPTED_DECISIONS]
    m0_wrong = sum(bool(row.get("m0_disagreement")) for row in accepted)
    independent_wrong = sum(
        bool(row.get("independent_relative_geometry_confirms_wrong")) for row in accepted
    )
    return {
        "query_count": count,
        "accepted_count": len(accepted),
        "m0_relative_wrong_acceptance_count": m0_wrong,
        "independent_relative_geometry_confirmed_wrong_acceptance_count": independent_wrong,
        "m0_relative_hcwpar": float(m0_wrong / count) if count else None,
        "independent_relative_geometry_confirmed_hcwpar": (
            float(independent_wrong / count) if count else None
        ),
        "localization_coverage": float(len(accepted) / count) if count else None,
        "ground_truth_contract": (
            "M0-relative disagreement is not absolute accuracy; independent-relative "
            "confirmation is still a consistency test, not survey ground truth."
        ),
    }
