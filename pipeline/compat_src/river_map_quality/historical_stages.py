"""E0-E5 historical experiment stages, roster policies, and promotion rollback.

Development may retune only declared roster/strategy choices.  Final replay is sealed
after code, config, thresholds, bundles, P157 split, and runtime receipts freeze.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from river_map_quality.ambiguity_localization import (
    CURRENT_ONLY,
    HISTORICAL_ONLY,
    MIXED,
    AmbiguityConfig,
    PoseHypothesisRecord,
    cluster_pose_modes,
    localization_decision,
)
from river_map_quality.historical_experiment import HistoricalExperimentError
from river_map_quality.historical_inputs import DIRECT_STRONG, STABLE
from river_map_quality.pose_solvers import project_points

STAGE_ORDER = ("E0", "E1", "E2", "E3", "E4", "E5")
FIXED_ROSTER = "fixed_roster"
DEPLOYMENT_ROSTER = "deployment_roster"
PROMOTION_FORBIDDEN_ABLATIONS = frozenset(
    {
        "pooled_current_historical",
        "no_change_mask",
        "no_bridge_graph",
        "no_utility_selection",
        "unweighted_pnp",
        "no_temporal_consistency",
    }
)


class HistoricalStageError(HistoricalExperimentError):
    """Raised when a stage, roster, or promotion gate is violated."""


@dataclass(frozen=True)
class StageBundle:
    stage: str
    roster: str
    current_references: tuple[str, ...]
    historical_references: tuple[str, ...]
    historical_associations_enabled: bool
    derivative_points_enabled: bool
    weighted_refinement: bool
    historical_on_demand: bool


def stage_policy(stage: str, roster: str) -> dict[str, object]:
    if stage not in STAGE_ORDER:
        raise HistoricalStageError(f"unknown historical stage: {stage}")
    if roster not in {FIXED_ROSTER, DEPLOYMENT_ROSTER}:
        raise HistoricalStageError(f"unknown roster family: {roster}")
    index = STAGE_ORDER.index(stage)
    historical_visible = roster == FIXED_ROSTER or index >= 1
    associations = index >= 1
    derivatives = index >= 2
    weighted = index >= 4
    on_demand = index >= 5
    if roster == DEPLOYMENT_ROSTER and stage == "E0":
        historical_visible = False
    return {
        "stage": stage,
        "roster": roster,
        "historical_images_visible": historical_visible,
        "historical_associations_enabled": associations and historical_visible,
        "derivative_points_enabled": derivatives and historical_visible,
        "weighted_refinement": weighted,
        "historical_on_demand": on_demand,
        "current_first": index >= 4 or roster == DEPLOYMENT_ROSTER,
    }


def apply_fixed_roster_identity(
    *,
    current_names: Sequence[str],
    historical_names: Sequence[str],
    stage: str,
    query_top_k: Sequence[str],
) -> dict[str, object]:
    """Keep identical reference names across stages; E0 historical rows stay inert."""

    policy = stage_policy(stage, FIXED_ROSTER)
    inert = tuple(historical_names) if policy["historical_images_visible"] else ()
    active = inert if policy["historical_associations_enabled"] else ()
    if any(name in set(current_names[: len(query_top_k)]) for name in active):
        raise HistoricalStageError(
            "historical references cannot displace current top-K on fixed roster"
        )
    return {
        "current_references": list(current_names),
        "historical_image_only": list(inert) if not active else [],
        "historical_active": list(active),
        "query_top_k": list(query_top_k),
        "current_top_k_preserved": True,
    }


def apply_deployment_roster(
    *,
    current_names: Sequence[str],
    selected_historical: Sequence[str],
    stage: str,
    current_confidence_weak: bool,
) -> tuple[str, ...]:
    policy = stage_policy(stage, DEPLOYMENT_ROSTER)
    if not policy["historical_images_visible"]:
        return tuple(current_names)
    if policy["historical_on_demand"] and not current_confidence_weak:
        return tuple(current_names)
    return tuple((*current_names, *selected_historical))


def huber_weight(residual: float, delta: float = 1.0) -> float:
    absolute = abs(float(residual))
    if absolute <= delta:
        return 1.0
    return float(delta / absolute)


def source_weight(source_type: str, thresholds: Mapping[str, Any]) -> float:
    if source_type == "current":
        return float(thresholds["current_source_weight"])
    if source_type == "historical_existing_current_point":
        return float(thresholds["historical_existing_current_point_weight"])
    if source_type == "current_confirmed_derivative":
        return float(thresholds["current_confirmed_derivative_weight"])
    raise HistoricalStageError(f"unknown correspondence source type: {source_type}")


def refine_weighted_se3(
    pose: np.ndarray,
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    source_types: Sequence[str],
    thresholds: Mapping[str, Any],
    *,
    iterations: int = 8,
    huber_delta: float = 1.0,
) -> dict[str, object]:
    """Left-SE(3) Gauss-Newton refinement with source-aware Huber weights."""

    transform = np.asarray(pose, dtype=float).copy()
    world = np.asarray(world_points, dtype=float)
    image = np.asarray(image_points, dtype=float)
    camera = np.asarray(camera_matrix, dtype=float)
    weights = np.asarray([source_weight(kind, thresholds) for kind in source_types], dtype=float)
    if transform.shape != (4, 4) or world.shape != (len(image), 3):
        raise HistoricalStageError("weighted refinement correspondences are malformed")
    last_cost = float("inf")
    for _ in range(iterations):
        projected, positive = project_points(world, transform, camera)
        residuals = image - projected
        valid = positive & np.isfinite(residuals).all(axis=1)
        if int(np.count_nonzero(valid)) < 6:
            break
        camera_points = (transform[:3, :3] @ world.T).T + transform[:3, 3]
        jacobian = np.zeros((int(np.count_nonzero(valid)) * 2, 6), dtype=float)
        rhs = np.zeros(int(np.count_nonzero(valid)) * 2, dtype=float)
        row = 0
        cost = 0.0
        for index in np.flatnonzero(valid):
            x, y, z = camera_points[index]
            fx, fy = camera[0, 0], camera[1, 1]
            residual = residuals[index]
            weight = weights[index] * huber_weight(float(np.linalg.norm(residual)), huber_delta)
            pixel_jacobian = np.array(
                [
                    [
                        fx / z,
                        0.0,
                        -fx * x / (z * z),
                        -fx * x * y / (z * z),
                        fx * (1.0 + x * x / (z * z)),
                        -fx * y / z,
                    ],
                    [
                        0.0,
                        fy / z,
                        -fy * y / (z * z),
                        -fy * (1.0 + y * y / (z * z)),
                        fy * x * y / (z * z),
                        fy * x / z,
                    ],
                ]
            )
            jacobian[row : row + 2] = weight * pixel_jacobian
            rhs[row : row + 2] = weight * residual
            cost += float(weight * residual @ residual)
            row += 2
        gram = jacobian.T @ jacobian
        try:
            delta = np.linalg.solve(gram + 1e-8 * np.eye(6), jacobian.T @ rhs)
        except np.linalg.LinAlgError:
            break
        translation = delta[:3]
        omega = delta[3:]
        angle = float(np.linalg.norm(omega))
        if angle < 1e-12:
            rotation_delta = np.eye(3)
        else:
            axis = omega / angle
            skew = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
            )
            rotation_delta = (
                np.eye(3)
                + math.sin(angle) * skew
                + (1.0 - math.cos(angle)) * (skew @ skew)
            )
        updated = np.eye(4)
        updated[:3, :3] = rotation_delta @ transform[:3, :3]
        updated[:3, 3] = rotation_delta @ transform[:3, 3] + translation
        transform = updated
        if abs(last_cost - cost) < 1e-9:
            break
        last_cost = cost
    projected, positive = project_points(world, transform, camera)
    errors = np.linalg.norm(image - projected, axis=1)
    hard_inliers = positive & (errors <= float(thresholds.get("maximum_reprojection_p90_px", 3.0)))
    return {
        "pose": transform,
        "inlier_mask": hard_inliers,
        "inlier_count": int(np.count_nonzero(hard_inliers)),
        "cost": last_cost,
    }


def arbitrate_current_first(
    hypotheses: Sequence[PoseHypothesisRecord],
    *,
    current_strong: bool,
) -> dict[str, object]:
    modes = cluster_pose_modes(hypotheses, config=AmbiguityConfig())
    decision = localization_decision(
        modes,
        reject_multimodal=True,
        config=AmbiguityConfig(),
        current_first=True,
    )
    if current_strong and decision.source_role == HISTORICAL_ONLY:
        raise HistoricalStageError("historical mode cannot override a strong current solve")
    return {
        "status": decision.status,
        "selected_mode_id": decision.selected_mode_id,
        "source_role": decision.source_role or CURRENT_ONLY,
        "current_first_state": decision.current_first_state,
        "mixed_permitted": decision.source_role == MIXED,
    }


def promotion_decision(
    *,
    predecessor: str,
    candidate: str,
    weak_region_gain: float,
    healthy_success_delta: float,
    new_high_confidence_wrong: bool,
    pose_accuracy_regression: bool,
    b0_unchanged: bool,
    receipts_complete: bool,
    latency_multiplier: float,
    gpu_peak_gib: float,
    ablation: str | None = None,
    thresholds: Mapping[str, Any],
) -> dict[str, object]:
    if candidate not in STAGE_ORDER or predecessor not in STAGE_ORDER:
        raise HistoricalStageError("promotion requires ordered E0-E5 stages")
    if STAGE_ORDER.index(candidate) != STAGE_ORDER.index(predecessor) + 1:
        raise HistoricalStageError("promotion must be predecessor-adjacent")
    if ablation in PROMOTION_FORBIDDEN_ABLATIONS:
        return {
            "decision": "ROLLBACK",
            "selected_stage": predecessor,
            "reason": "promotion_forbidden_ablation",
        }
    reasons = []
    if weak_region_gain <= 0:
        reasons.append("no_positive_weak_region_gain")
    if healthy_success_delta < -0.01:
        reasons.append("healthy_success_loss_exceeds_one_point")
    if new_high_confidence_wrong:
        reasons.append("new_high_confidence_wrong_mode")
    if pose_accuracy_regression:
        reasons.append("pose_accuracy_regression")
    if not b0_unchanged:
        reasons.append("b0_hash_changed")
    if not receipts_complete:
        reasons.append("incomplete_receipts")
    if latency_multiplier > float(thresholds["maximum_p95_latency_multiplier"]):
        reasons.append("p95_latency_exceeds_budget")
    if gpu_peak_gib >= float(thresholds["maximum_gpu_peak_gib"]):
        reasons.append("gpu_peak_exceeds_budget")
    if reasons:
        return {"decision": "ROLLBACK", "selected_stage": predecessor, "reasons": reasons}
    return {"decision": "PROMOTE", "selected_stage": candidate, "reasons": []}


def seal_final_replay(
    *,
    freeze: Mapping[str, Any],
    executed_partition: str,
) -> dict[str, object]:
    if not freeze.get("final_replay_permitted"):
        raise HistoricalStageError("final replay requires a sealed validation freeze")
    if executed_partition != "final":
        raise HistoricalStageError("sealed replay may execute only the final P157 partition")
    return {
        "status": "SEALED_FINAL_REPLAY",
        "development_retuning_permitted": False,
        "partition": "final",
        "freeze_artifact": freeze.get("artifact_type"),
    }


def admissible_historical_association(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("status")) == DIRECT_STRONG
        and str(row.get("mask_label", STABLE)) == STABLE
        and str(row.get("point_status", "PNP_ELIGIBLE")) == "PNP_ELIGIBLE"
    )


__all__ = [
    "DEPLOYMENT_ROSTER",
    "FIXED_ROSTER",
    "PROMOTION_FORBIDDEN_ABLATIONS",
    "STAGE_ORDER",
    "HistoricalStageError",
    "StageBundle",
    "admissible_historical_association",
    "apply_deployment_roster",
    "apply_fixed_roster_identity",
    "arbitrate_current_first",
    "huber_weight",
    "promotion_decision",
    "refine_weighted_se3",
    "seal_final_replay",
    "source_weight",
    "stage_policy",
]
