from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseHypothesis:
    hypothesis_id: str
    center_w: np.ndarray
    R_wc: np.ndarray
    support: float
    inliers: int
    reprojection_error: float | None
    reference_group: str
    spatial_coverage: float | None = None
    fim_lambda_min: float | None = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center_w, dtype=float).reshape(3)
        rotation = np.asarray(self.R_wc, dtype=float).reshape(3, 3)
        if np.any(~np.isfinite(center)) or np.any(~np.isfinite(rotation)):
            raise ValueError("pose hypothesis must be finite")
        if self.support < 0.0 or not np.isfinite(self.support):
            raise ValueError("pose hypothesis support must be finite and non-negative")
        object.__setattr__(self, "center_w", center)
        object.__setattr__(self, "R_wc", rotation)


def pose_distance(
    first: PoseHypothesis, second: PoseHypothesis
) -> tuple[float, float]:
    translation = float(np.linalg.norm(first.center_w - second.center_w))
    relative = first.R_wc @ second.R_wc.T
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation


@dataclass(frozen=True)
class PoseMode:
    mode_id: int
    hypotheses: tuple[PoseHypothesis, ...]
    support: float
    center_w: np.ndarray
    R_wc: np.ndarray


def cluster_pose_modes(
    hypotheses: Sequence[PoseHypothesis],
    *,
    translation_threshold: float,
    rotation_threshold_deg: float,
) -> tuple[PoseMode, ...]:
    """Deterministic complete-link SE(3) clustering.

    Complete linkage prevents A~B~C chains from merging when A and C are
    mutually inconsistent, which is essential for perceptual-aliasing modes.
    """

    if translation_threshold <= 0.0 or rotation_threshold_deg <= 0.0:
        raise ValueError("pose-mode thresholds must be > 0")
    clusters: list[list[PoseHypothesis]] = [
        [hypothesis]
        for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id)
    ]

    def compatible(first: list[PoseHypothesis], second: list[PoseHypothesis]) -> bool:
        return all(
            pose_distance(a, b)[0] <= translation_threshold
            and pose_distance(a, b)[1] <= rotation_threshold_deg
            for a in first
            for b in second
        )

    while True:
        candidates = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if not compatible(clusters[i], clusters[j]):
                    continue
                distances = [
                    pose_distance(a, b)[0] / translation_threshold
                    + pose_distance(a, b)[1] / rotation_threshold_deg
                    for a in clusters[i]
                    for b in clusters[j]
                ]
                candidates.append((max(distances), i, j))
        if not candidates:
            break
        _, first_index, second_index = min(candidates)
        clusters[first_index] = sorted(
            clusters[first_index] + clusters[second_index],
            key=lambda item: item.hypothesis_id,
        )
        del clusters[second_index]

    modes = [_mode(index, cluster) for index, cluster in enumerate(clusters)]
    modes.sort(key=lambda mode: (-mode.support, mode.hypotheses[0].hypothesis_id))
    return tuple(
        PoseMode(
            mode_id=index,
            hypotheses=mode.hypotheses,
            support=mode.support,
            center_w=mode.center_w,
            R_wc=mode.R_wc,
        )
        for index, mode in enumerate(modes)
    )


def _mode(mode_id: int, hypotheses: list[PoseHypothesis]) -> PoseMode:
    raw_weights = np.asarray([hypothesis.support for hypothesis in hypotheses], dtype=float)
    weights = raw_weights if float(np.sum(raw_weights)) > 0.0 else np.ones(len(hypotheses))
    weights = weights / np.sum(weights)
    center = np.sum(
        np.asarray([hypothesis.center_w for hypothesis in hypotheses]) * weights[:, None],
        axis=0,
    )
    rotation = Rotation.from_matrix(
        np.asarray([hypothesis.R_wc for hypothesis in hypotheses])
    ).mean(weights=weights).as_matrix()
    return PoseMode(
        mode_id=mode_id,
        hypotheses=tuple(hypotheses),
        support=float(np.sum(raw_weights)),
        center_w=center,
        R_wc=rotation,
    )


@dataclass(frozen=True)
class PoseModeAnalysis:
    num_pose_modes: int
    best_mode_support: float
    second_mode_support: float
    support_margin: float
    mode_translation_separation: float | None
    mode_rotation_separation_deg: float | None
    mode_entropy: float
    ambiguous: bool
    modes: tuple[PoseMode, ...]


def analyze_pose_modes(
    hypotheses: Sequence[PoseHypothesis],
    *,
    translation_threshold: float,
    rotation_threshold_deg: float,
    ambiguous_support_ratio: float = 0.5,
) -> PoseModeAnalysis:
    modes = cluster_pose_modes(
        hypotheses,
        translation_threshold=translation_threshold,
        rotation_threshold_deg=rotation_threshold_deg,
    )
    best = modes[0].support if modes else 0.0
    second = modes[1].support if len(modes) > 1 else 0.0
    total = sum(mode.support for mode in modes)
    margin = (best - second) / max(total, 1e-12)
    probability = np.asarray([mode.support for mode in modes], dtype=float)
    probability = probability[probability > 0.0]
    probability = probability / np.sum(probability) if len(probability) else probability
    entropy = (
        -float(np.sum(probability * np.log(probability))) / np.log(len(probability))
        if len(probability) > 1
        else 0.0
    )
    translation = None
    rotation = None
    if len(modes) > 1:
        translation, rotation = pose_distance(
            PoseHypothesis("best", modes[0].center_w, modes[0].R_wc, 1.0, 0, None, ""),
            PoseHypothesis("second", modes[1].center_w, modes[1].R_wc, 1.0, 0, None, ""),
        )
    ambiguous = bool(
        len(modes) > 1
        and second / max(best, 1e-12) >= ambiguous_support_ratio
        and (
            (translation is not None and translation > translation_threshold)
            or (rotation is not None and rotation > rotation_threshold_deg)
        )
    )
    return PoseModeAnalysis(
        num_pose_modes=len(modes),
        best_mode_support=best,
        second_mode_support=second,
        support_margin=float(margin),
        mode_translation_separation=translation,
        mode_rotation_separation_deg=rotation,
        mode_entropy=entropy,
        ambiguous=ambiguous,
        modes=modes,
    )


@dataclass(frozen=True)
class GroupLOOStability:
    translation_jump_max: float
    translation_jump_median: float
    rotation_jump_max_deg: float
    rotation_jump_median_deg: float
    jumps_by_group: dict[str, dict[str, float]]


def group_leave_out_stability(
    full: PoseHypothesis,
    leave_out_poses: Mapping[str, PoseHypothesis],
) -> GroupLOOStability:
    jumps = {
        group: pose_distance(full, pose)
        for group, pose in sorted(leave_out_poses.items())
    }
    translation = np.asarray([value[0] for value in jumps.values()], dtype=float)
    rotation = np.asarray([value[1] for value in jumps.values()], dtype=float)
    return GroupLOOStability(
        translation_jump_max=float(np.max(translation)) if len(translation) else 0.0,
        translation_jump_median=float(np.median(translation)) if len(translation) else 0.0,
        rotation_jump_max_deg=float(np.max(rotation)) if len(rotation) else 0.0,
        rotation_jump_median_deg=float(np.median(rotation)) if len(rotation) else 0.0,
        jumps_by_group={
            group: {"translation": value[0], "rotation_deg": value[1]}
            for group, value in jumps.items()
        },
    )
