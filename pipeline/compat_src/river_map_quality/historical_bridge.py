"""Isolated historical bridge graphs, protected B0 local BA, and derivative gates.

Bridge reconstruction never writes B0 cameras, current poses, or original current
points.  Historical-only derivative points stay diagnostic until two current flights
confirm them.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from river_map_quality.historical_experiment import HistoricalExperimentError
from river_map_quality.historical_inputs import (
    DIRECT_PROVISIONAL_BRIDGE_ONLY,
    DIRECT_STRONG,
)
from river_map_quality.p157_sim3 import evaluate_leave_one_out_similarity
from river_map_quality.pose_attribution import camera_center

BRIDGE_GRAPH_SCHEMA = "RIVER_HISTORICAL_BRIDGE_GRAPH_V1"
BRIDGE_SUBMAP_SCHEMA = "RIVER_HISTORICAL_BRIDGE_SUBMAP_V1"
BRIDGE_SINGLE_ANCHOR = "BRIDGE_SINGLE_ANCHOR"
BRIDGE_FOLD = "BRIDGE_FOLD"
BRIDGE_AMBIGUOUS = "BRIDGE_AMBIGUOUS"
BRIDGE_UNSTABLE_SCALE = "BRIDGE_UNSTABLE_SCALE"
BRIDGE_CHANGED_REGION = "BRIDGE_CHANGED_REGION"
ACCEPTED_BRIDGE = "ACCEPTED_BRIDGE"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
PNP_ELIGIBLE = "PNP_ELIGIBLE"


class HistoricalBridgeError(HistoricalExperimentError):
    """Raised when a bridge component would mutate B0 or leak unconfirmed points."""


@dataclass(frozen=True)
class BridgeEdge:
    source: str
    target: str
    status: str
    independent_support: int
    parallax_px: float
    coverage: float


@dataclass(frozen=True)
class BridgeComponent:
    component_id: str
    image_names: tuple[str, ...]
    anchors: tuple[str, ...]
    status: str
    reason: str


def trigger_bridge_candidates(
    registrations: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Trigger only around direct-registration gaps adjacent to usable anchors."""

    by_name = {str(row["query_name"]): row for row in registrations}
    names = tuple(sorted(by_name))
    triggered: list[str] = []
    for index, name in enumerate(names):
        status = str(by_name[name]["status"])
        if status in {DIRECT_STRONG, DIRECT_PROVISIONAL_BRIDGE_ONLY}:
            continue
        neighbors = []
        if index:
            neighbors.append(by_name[names[index - 1]]["status"])
        if index + 1 < len(names):
            neighbors.append(by_name[names[index + 1]]["status"])
        if any(
            neighbor in {DIRECT_STRONG, DIRECT_PROVISIONAL_BRIDGE_ONLY} for neighbor in neighbors
        ):
            triggered.append(name)
    return tuple(triggered)


def verify_bridge_edge(
    *,
    independent_support: int,
    parallax_px: float,
    coverage: float,
    essential_inliers: int,
    changed_fraction: float,
    thresholds: Mapping[str, float],
) -> str:
    if changed_fraction >= float(thresholds.get("maximum_changed_fraction", 0.5)):
        return BRIDGE_CHANGED_REGION
    if (
        independent_support >= int(thresholds.get("minimum_independent_support", 2))
        and parallax_px >= float(thresholds.get("minimum_parallax_px", 2.0))
        and coverage >= float(thresholds.get("minimum_coverage", 0.10))
        and essential_inliers >= int(thresholds.get("minimum_essential_inliers", 20))
    ):
        return "ACCEPTED_EDGE"
    return "REJECTED_EDGE"


def connected_components(edges: Sequence[BridgeEdge]) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for edge in edges:
        if edge.status != "ACCEPTED_EDGE":
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
        nodes.update((edge.source, edge.target))
    unseen = set(nodes)
    components: list[tuple[str, ...]] = []
    while unseen:
        root = min(unseen)
        stack = [root]
        unseen.remove(root)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def classify_bridge_component(
    *,
    image_names: Sequence[str],
    anchors: Sequence[Mapping[str, Any]],
    cycle_consistent: bool,
    changed_fraction: float,
    max_relative_residual: float = 0.05,
) -> BridgeComponent:
    component_id = "bridge:" + ",".join(image_names)
    if changed_fraction >= 0.5:
        return BridgeComponent(
            component_id, tuple(image_names), (), BRIDGE_CHANGED_REGION, "changed"
        )
    accepted_anchors = [
        row
        for row in anchors
        if str(row.get("status")) in {DIRECT_STRONG, DIRECT_PROVISIONAL_BRIDGE_ONLY}
        and str(row.get("query_name")) in set(image_names)
    ]
    if len(accepted_anchors) < 3:
        return BridgeComponent(
            component_id,
            tuple(image_names),
            tuple(str(row["query_name"]) for row in accepted_anchors),
            BRIDGE_SINGLE_ANCHOR if accepted_anchors else BRIDGE_AMBIGUOUS,
            "insufficient_spatially_separated_anchors",
        )
    if not cycle_consistent:
        return BridgeComponent(
            component_id,
            tuple(image_names),
            tuple(str(row["query_name"]) for row in accepted_anchors),
            BRIDGE_AMBIGUOUS,
            "cycle_inconsistent",
        )
    source = np.asarray(
        [camera_center(np.asarray(row["historical_pose"])) for row in accepted_anchors]
    )
    target = np.asarray([camera_center(np.asarray(row["b0_pose"])) for row in accepted_anchors])
    if len(accepted_anchors) >= 4:
        loo = evaluate_leave_one_out_similarity(
            source, target, max_relative_residual=max_relative_residual
        )
        if not bool(loo["consistent"]):
            return BridgeComponent(
                component_id,
                tuple(image_names),
                tuple(str(row["query_name"]) for row in accepted_anchors),
                BRIDGE_UNSTABLE_SCALE,
                "leave_one_anchor_cluster_out_failed",
            )
    span = float(np.max(np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)))
    if span <= 1e-6:
        return BridgeComponent(
            component_id,
            tuple(image_names),
            tuple(str(row["query_name"]) for row in accepted_anchors),
            BRIDGE_FOLD,
            "anchors_are_not_spatially_separated",
        )
    return BridgeComponent(
        component_id,
        tuple(image_names),
        tuple(str(row["query_name"]) for row in accepted_anchors),
        ACCEPTED_BRIDGE,
        "three_separated_current_supported_anchors",
    )


def protected_b0_values(receipt: Mapping[str, Any]) -> dict[str, object]:
    return {
        "pose_table_sha256": receipt["pose_table_sha256"],
        "intrinsics_sha256": receipt["intrinsics_sha256"],
        "point_xyz_sha256": receipt["point_xyz_sha256"],
        "original_observation_table_sha256": receipt["original_observation_table_sha256"],
        "point_track_table_sha256": receipt["point_track_table_sha256"],
    }


def assert_protected_b0_constant(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    left = protected_b0_values(before)
    right = protected_b0_values(after)
    if left != right:
        raise HistoricalBridgeError(
            f"protected B0 values changed during local BA: {left} != {right}"
        )


def classify_derivative_point(
    *,
    view_count: int,
    positive_depth: bool,
    reprojection_p90: float,
    angle_deg: float,
    current_flight_confirmations: Sequence[str],
    thresholds: Mapping[str, float],
) -> str:
    flights = {str(name) for name in current_flight_confirmations if name}
    if (
        view_count >= 3
        and positive_depth
        and reprojection_p90 <= float(thresholds.get("maximum_reprojection_p90_px", 3.0))
        and angle_deg >= float(thresholds.get("minimum_angle_deg", 2.0))
        and len(flights) >= 2
    ):
        return PNP_ELIGIBLE
    return DIAGNOSTIC_ONLY


def select_route_utility_roster(
    candidates: Sequence[Mapping[str, Any]],
    *,
    maximum_per_cell: int,
    healthy_cell_ids: Sequence[str],
) -> dict[str, object]:
    """Greedy constrained selection that never degrades a healthy current cell."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row["cell_id"])].append(row)
    selected: list[str] = []
    rejected: list[dict[str, object]] = []
    for cell_id, rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row.get("utility", 0.0)),
                -float(row.get("stable_fraction", 0.0)),
                str(row["name"]),
            ),
        )
        kept = 0
        for row in ordered:
            if cell_id in set(healthy_cell_ids) and float(row.get("healthy_degradation", 0.0)) > 0:
                rejected.append({"name": row["name"], "reason": "healthy_cell_degradation"})
                continue
            if kept >= maximum_per_cell:
                rejected.append({"name": row["name"], "reason": "per_cell_cap"})
                continue
            selected.append(str(row["name"]))
            kept += 1
    return {
        "schema_version": 1,
        "selected": selected,
        "rejected": rejected,
        "maximum_per_cell": maximum_per_cell,
        "healthy_cells_protected": list(healthy_cell_ids),
    }


__all__ = [
    "ACCEPTED_BRIDGE",
    "BRIDGE_AMBIGUOUS",
    "BRIDGE_CHANGED_REGION",
    "BRIDGE_FOLD",
    "BRIDGE_SINGLE_ANCHOR",
    "BRIDGE_UNSTABLE_SCALE",
    "DIAGNOSTIC_ONLY",
    "PNP_ELIGIBLE",
    "BridgeComponent",
    "BridgeEdge",
    "HistoricalBridgeError",
    "assert_protected_b0_constant",
    "classify_bridge_component",
    "classify_derivative_point",
    "connected_components",
    "protected_b0_values",
    "select_route_utility_roster",
    "trigger_bridge_candidates",
    "verify_bridge_edge",
]
