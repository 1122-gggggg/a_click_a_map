"""Graph-aware diagnostic selection, role assignment and bridge reinforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .domain import (
    Contribution,
    DataStatus,
    Inclusion,
    IssueCode,
    MappingMode,
    PostSfmRole,
    PreSfmRole,
    Risk,
    RoleAssignment,
)
from .graph import connected_components


@dataclass
class Candidate:
    id: str
    coverage: set[str] = field(default_factory=set)
    connections: set[str] = field(default_factory=set)
    geometry: float = 0.0
    connectivity: float = 0.0
    loop_contribution: float = 0.0
    view_diversity: float = 0.0
    cost: float = 1.0
    parallax: float = 0.0
    registered_ratio: float | None = None
    median_track_length: float | None = None
    long_track_ratio: float | None = None
    covisibility: float | None = None
    fim_condition: float | None = None
    loo_rotation_deg: float | None = None
    loo_position_normalized: float | None = None
    verified_bridge: bool = False
    articulation: bool = False
    verified_loop: bool = False
    pnp_stable: bool = False
    geometry_redundant: bool = False
    alignment_evaluated: bool = False
    alignment_consistent: bool = True
    is_retrieval_only: bool = False
    unstable: bool = False
    rejected: bool = False
    warnings: tuple[str, ...] = ()
    hard_failures: tuple[str, ...] = ()
    pre_sfm_role: PreSfmRole = PreSfmRole.UNDECIDED


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Candidate, ...]
    warnings: tuple[str, ...] = ()
    covered: frozenset[str] = frozenset()
    cost: float = 0.0


def diagnostic_select(
    candidates: Iterable[Candidate],
    required_coverage: Iterable[str],
    budget: float,
    *,
    target_cost: float = 0.0,
    require_connected: bool = True,
) -> SelectionResult:
    """Select geometry candidates without granting authority to retrieval alone."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if target_cost < 0 or target_cost > budget:
        raise ValueError("target_cost must be between zero and budget")
    required = set(required_coverage)
    usable = [
        candidate
        for candidate in candidates
        if not candidate.is_retrieval_only
        and not candidate.rejected
        and not candidate.hard_failures
        and candidate.cost >= 0
    ]
    selected: list[Candidate] = []
    covered: set[str] = set()
    spent = 0.0

    def score(candidate: Candidate) -> tuple[float, float, float, float, float, str]:
        new_coverage = len((candidate.coverage & required) - covered)
        selected_ids = {item.id for item in selected}
        new_links = len(candidate.connections & selected_ids)
        importance = float(candidate.verified_bridge or candidate.articulation) + float(
            candidate.verified_loop
        )
        utility = (
            3.0 * new_coverage
            + 2.0 * new_links
            + candidate.connectivity
            + candidate.loop_contribution
            + candidate.view_diversity
            + candidate.geometry
            + importance
            - 0.25 * candidate.cost
        )
        return (
            float(new_coverage),
            utility,
            importance,
            candidate.connectivity,
            candidate.geometry,
            candidate.id,
        )

    while True:
        choices = [
            candidate
            for candidate in usable
            if candidate not in selected and spent + candidate.cost <= budget
        ]
        if not choices:
            break
        choices.sort(key=score, reverse=True)
        choice = choices[0]
        if score(choice)[1] <= 0 and required <= covered and spent >= target_cost:
            break
        selected.append(choice)
        covered.update(choice.coverage)
        spent += choice.cost
        if required <= covered and _selected_connected(selected, require_connected):
            mandatory_left = [
                row
                for row in choices[1:]
                if row.verified_bridge or row.articulation or row.verified_loop
            ]
            if not mandatory_left and spent >= target_cost:
                break

    warnings: list[str] = []
    if not required <= covered:
        warnings.append("REQUIRED_COVERAGE_UNMET")
    if require_connected and not _selected_connected(selected, True):
        warnings.append("CONNECTIVITY_UNMET")
    if spent < target_cost:
        warnings.append("TARGET_COST_UNMET")
    return SelectionResult(tuple(selected), tuple(warnings), frozenset(covered), spent)


def _selected_connected(selected: list[Candidate], required: bool) -> bool:
    if not required or len(selected) <= 1:
        return True
    ids = {candidate.id for candidate in selected}
    edges = {
        tuple(sorted((candidate.id, neighbor)))
        for candidate in selected
        for neighbor in candidate.connections
        if neighbor in ids and neighbor != candidate.id
    }
    return len(connected_components(ids, edges)) == 1


@dataclass(frozen=True)
class RoleDecision:
    role: PostSfmRole | None
    risk: Risk
    mapping_mode: MappingMode
    base_map: Inclusion
    localization: Inclusion
    geometry_contribution: Contribution
    connectivity_contribution: Contribution
    reasons: tuple[str, ...] = ()
    issues: tuple[IssueCode, ...] = ()

    def assignment(self, candidate: Candidate) -> RoleAssignment:
        status = (
            DataStatus.UNDECIDED
            if self.role is None
            else {
                PostSfmRole.CORE: DataStatus.ACTIVE_CORE,
                PostSfmRole.BRIDGE: DataStatus.ACTIVE_BRIDGE,
                PostSfmRole.UPDATE_ONLY: DataStatus.INACTIVE_UPDATE,
                PostSfmRole.REJECT: DataStatus.INACTIVE_REJECT,
            }[self.role]
        )
        return RoleAssignment(
            segment_id=candidate.id,
            pre_sfm_role=candidate.pre_sfm_role,
            post_sfm_role=self.role,
            data_status=status,
            mapping_mode=self.mapping_mode,
            base_map=self.base_map,
            localization=self.localization,
            risk=self.risk,
            geometry_contribution=self.geometry_contribution,
            connectivity_contribution=self.connectivity_contribution,
            reasons=self.reasons,
            issues=self.issues,
        )


Role = RoleDecision


def assign_roles(candidates: Iterable[Candidate]) -> dict[str, RoleDecision]:
    return {candidate.id: assign_role(candidate) for candidate in candidates}


def assign_role(candidate: Candidate) -> RoleDecision:
    if candidate.rejected or candidate.hard_failures:
        return RoleDecision(
            PostSfmRole.REJECT,
            Risk.CRITICAL if candidate.hard_failures else Risk.HIGH,
            MappingMode.EXCLUDE,
            Inclusion.EXCLUDE,
            Inclusion.EXCLUDE,
            Contribution.NONE,
            Contribution.NONE,
            tuple(candidate.hard_failures) or ("explicit_reject",),
        )

    connectivity_role = (
        candidate.verified_bridge or candidate.articulation or candidate.verified_loop
    )
    if connectivity_role:
        if not candidate.alignment_evaluated:
            return RoleDecision(
                None,
                Risk.UNDECIDED,
                MappingMode.EXCLUDE,
                Inclusion.BLOCKED,
                Inclusion.BLOCKED,
                Contribution.LOW,
                Contribution.HIGH,
                ("bridge_alignment_not_evaluated",),
            )
        if not candidate.alignment_consistent or candidate.unstable:
            return RoleDecision(
                PostSfmRole.REJECT,
                Risk.CRITICAL,
                MappingMode.EXCLUDE,
                Inclusion.EXCLUDE,
                Inclusion.EXCLUDE,
                Contribution.NONE,
                Contribution.HIGH,
                ("connectivity_evidence_inconsistent",),
                (IssueCode.MISSING_CONNECTIVITY, IssueCode.RESHOOT_REQUIRED),
            )
        weak_geometry = candidate.parallax < 1.0 or candidate.geometry < 0.35
        return RoleDecision(
            PostSfmRole.BRIDGE,
            Risk.HIGH if weak_geometry or candidate.articulation else Risk.MEDIUM,
            MappingMode.POSE_ONLY if weak_geometry else MappingMode.TRIANGULATE,
            Inclusion.INCLUDE,
            Inclusion.INCLUDE,
            Contribution.LOW if weak_geometry else Contribution.MEDIUM,
            Contribution.HIGH,
            ("verified_connectivity_contribution",),
        )

    geometry_ready = (
        candidate.geometry >= 0.7
        and (candidate.registered_ratio is None or candidate.registered_ratio >= 0.5)
        and (candidate.fim_condition is None or candidate.fim_condition <= 1000)
    )
    if geometry_ready and not candidate.geometry_redundant:
        return RoleDecision(
            PostSfmRole.CORE,
            Risk.LOW if not candidate.warnings else Risk.MEDIUM,
            MappingMode.TRIANGULATE,
            Inclusion.INCLUDE,
            Inclusion.INCLUDE,
            Contribution.HIGH,
            Contribution.MEDIUM,
            ("high_geometry_information",),
        )
    if candidate.pnp_stable or candidate.geometry_redundant:
        return RoleDecision(
            PostSfmRole.UPDATE_ONLY,
            Risk.LOW if candidate.pnp_stable else Risk.MEDIUM,
            MappingMode.EXCLUDE,
            Inclusion.EXCLUDE,
            Inclusion.INCLUDE,
            Contribution.LOW,
            Contribution.NONE,
            ("localizes_to_frozen_base_without_geometry_gain",),
        )
    return RoleDecision(
        None,
        Risk.UNDECIDED,
        MappingMode.EXCLUDE,
        Inclusion.BLOCKED,
        Inclusion.BLOCKED,
        Contribution.UNDECIDED,
        Contribution.UNDECIDED,
        ("insufficient_post_sfm_evidence",),
    )


@dataclass(frozen=True)
class ReinforcementResult:
    added: tuple[Candidate, ...]
    rounds: int
    disposition: str | None = None
    reasons: tuple[str, ...] = ()


def reinforce_bridges(
    bridges: Iterable[Candidate],
    pool: Iterable[Candidate],
    max_rounds: int = 2,
) -> ReinforcementResult:
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    critical = [
        bridge
        for bridge in bridges
        if bridge.verified_bridge or bridge.articulation or bridge.verified_loop
    ]
    if not critical:
        return ReinforcementResult((), 0)
    candidates = [
        row
        for row in pool
        if not row.is_retrieval_only
        and not row.rejected
        and not row.hard_failures
        and (row.connections or row.connectivity > 0 or row.verified_loop)
    ]
    candidates.sort(
        key=lambda row: (
            row.connectivity,
            row.loop_contribution,
            row.geometry,
            -row.cost,
            row.id,
        ),
        reverse=True,
    )
    if candidates and max_rounds:
        return ReinforcementResult((candidates[0],), 1, reasons=("candidate_pool_alternate_path",))
    unstable = any(row.unstable or not row.alignment_consistent for row in critical)
    disposition = IssueCode.RESHOOT_REQUIRED.value if unstable else "MISSING_REDUNDANCY"
    return ReinforcementResult((), max_rounds, disposition, ("candidate_pool_exhausted",))


def warning_flags(metrics: Mapping[str, float | int | None]) -> tuple[str, ...]:
    """Apply the agreed warnings without assigning or overriding graph roles."""

    warnings: list[str] = []
    checks = (
        ("raw_matches", lambda value: value < 2000, "RAW_MATCHES_LT_2000"),
        ("spatial_coverage", lambda value: value < 0.1, "SPATIAL_COVERAGE_LT_0_1"),
        ("grid_occupancy_4x4", lambda value: value < 6, "GRID_OCCUPANCY_LT_6"),
        ("parallax_p10_deg", lambda value: value < 1.0, "PARALLAX_P10_LT_1_DEG"),
        ("median_track_length", lambda value: value < 3, "MEDIAN_TRACK_LT_3"),
        ("long_track_ratio", lambda value: value < 0.6, "LONG_TRACK_RATIO_LT_0_6"),
        ("covisibility", lambda value: value < 10, "COVISIBILITY_LT_10"),
        ("fim_condition", lambda value: value > 1000, "FIM_KAPPA_GT_1000"),
        ("loo_rotation_deg", lambda value: value > 2, "LOO_ROTATION_GT_2_DEG"),
        (
            "loo_position_normalized",
            lambda value: value > 0.02,
            "LOO_POSITION_GT_0_02",
        ),
        ("pnp_inliers", lambda value: value < 80, "PNP_INLIERS_LT_80"),
    )
    for key, predicate, label in checks:
        value = metrics.get(key)
        if value is not None and predicate(float(value)):
            warnings.append(label)
    return tuple(warnings)


__all__ = [
    "Candidate",
    "ReinforcementResult",
    "Role",
    "RoleDecision",
    "SelectionResult",
    "assign_role",
    "assign_roles",
    "diagnostic_select",
    "reinforce_bridges",
    "warning_flags",
]
