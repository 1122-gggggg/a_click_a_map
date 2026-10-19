"""Assign leftover sessions to remainder roles. Fail-closed. Timestamp ≠ update."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from typing import Any

from .intake_tree import classify_leftover_vs_frozen_base, role_or_quarantine
from .select_core import _incident_edge_index, connecting_edges

from .types import ROLES, SessionEdgeQuality, SessionInfluence, SessionQuality, edge_is_vpr_only

_CORE_OK = frozenset({"STRONG", "USABLE"})
_BLOCKED = frozenset({"AMBIGUOUS", "REJECT", "WEAK"})


def _quality_map(rows: Iterable[SessionQuality]) -> dict[str, SessionQuality]:
    return {row.session_id: row for row in rows}


def _links(
    session_id: str,
    others: set[str],
    edges: Iterable[SessionEdgeQuality],
    index: Mapping[str, Sequence[SessionEdgeQuality]] | None = None,
) -> list[SessionEdgeQuality]:
    if index is not None:
        return connecting_edges(session_id, others, (), index)
    sequence = edges if isinstance(edges, Sequence) else list(edges)
    return connecting_edges(session_id, others, sequence)


def _loc_counts(loc_row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    loc_strong = loc_row.get("loc_strong", loc_row.get("strong"))
    loc_queries = loc_row.get("loc_queries", loc_row.get("queries"))
    if loc_strong is None and loc_row.get("registered_to_base"):
        loc_strong, loc_queries = 1, 1
    if loc_strong is None and loc_row.get("loc_quality") == "strong":
        loc_strong, loc_queries = 1, 1
    return loc_strong, loc_queries, loc_row.get("core_hit_fraction")


def classify_one(session: SessionQuality, context: Mapping[str, Any]) -> str:
    """Classify a single leftover session. Uncertain → QUARANTINE; no edge → NEW_SUBMAP."""

    extra = context.get("extra") or {}
    core: set[str] = set(context.get("core") or ())
    support: set[str] = set(context.get("support") or ())
    base = core | support
    edges = context.get("edges") or ()
    index = context.get("edge_index")

    change_score = extra.get("change_score") or {}
    loc = extra.get("loc") or {}
    influences = extra.get("influences") or {}
    appearance_shift = extra.get("appearance_shift") or {}

    influence = influences.get(session.session_id)
    high_influence = False
    if isinstance(influence, SessionInfluence):
        high_influence = bool(influence.high_influence)
    elif isinstance(influence, Mapping):
        high_influence = bool(influence.get("high_influence"))

    links = _links(session.session_id, base, edges, index) if base else []

    blocked = [edge for edge in links if edge.status in _BLOCKED]
    usable = [edge for edge in links if edge.status not in _BLOCKED and not edge_is_vpr_only(edge)]
    independent = max((edge.independent_bridge_groups for edge in usable), default=0)
    if any(edge.is_critical_bridge for edge in links) and independent < 2 and usable:
        independent = 1

    loc_row = loc.get(session.session_id) or {}
    loc_strong, loc_queries, core_hit = _loc_counts(loc_row)
    change = float(change_score.get(session.session_id) or 0.0)
    appears_different = bool(appearance_shift.get(session.session_id))
    loc_ok = bool(
        loc_row.get("registered_to_base")
        or loc_row.get("loc_quality") == "strong"
        or (loc_strong and loc_queries)
    )
    weak_base = bool(context.get("base_has_weak_region"))
    improves = bool(loc_row.get("improves_weak_region"))
    geometry_already_covered = bool(loc_ok) and not weak_base and not improves and not appears_different

    role = classify_leftover_vs_frozen_base(
        session,
        loc_strong=loc_strong,
        loc_queries=loc_queries,
        core_hit_fraction=core_hit,
        independent_bridges=independent,
        usable_edge=bool(usable),
        change_score=change,
        high_influence=high_influence,
        geometry_already_covered=geometry_already_covered,
    )
    if role == "NEW_SUBMAP" and blocked:
        role = "QUARANTINE"

    hold_out_needed = bool(context.get("need_validation_holdout"))
    already_held = bool(context.get("has_validation"))
    if hold_out_needed and not already_held and role == "APPEARANCE_REF":
        role = "VALIDATION_ONLY"

    return role_or_quarantine(role)


def classify_remainder(
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality],
    core: Iterable[str],
    support: Iterable[str],
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return session_id → role for leftovers (and echo core/support roles)."""

    del config
    quality_by_id = _quality_map(qualities)
    edge_rows = list(edges)
    edge_index = _incident_edge_index(edge_rows)

    core_ids = [sid for sid in core if sid in quality_by_id]
    support_ids = [sid for sid in support if sid in quality_by_id]
    assigned: dict[str, str] = {}
    for sid in core_ids:
        assigned[sid] = "BASE_CORE"
    for sid in support_ids:
        assigned[sid] = "BASE_SUPPORT"

    leftovers = [row for row in quality_by_id.values() if row.session_id not in assigned]
    extra = dict(extra or {})
    base_has_weak = any(
        (quality_by_id[sid].parallax_p10_deg is not None and quality_by_id[sid].parallax_p10_deg < 1.0)
        or (quality_by_id[sid].fim_condition_number is not None and quality_by_id[sid].fim_condition_number > 1000)
        for sid in core_ids
        if sid in quality_by_id
    )
    need_validation = True
    has_validation = False
    leftovers_sorted = sorted(
        leftovers,
        key=lambda row: (row.internal_status in _CORE_OK, row.internal_quality_score),
        reverse=True,
    )
    for row in leftovers_sorted:
        context = {
            "core": core_ids,
            "support": support_ids,
            "edges": edge_rows,
            "edge_index": edge_index,
            "extra": extra,
            "base_has_weak_region": base_has_weak,
            "need_validation_holdout": need_validation,
            "has_validation": has_validation,
        }

        role = classify_one(row, context)
        if role not in ROLES:
            role = "QUARANTINE"
        assigned[row.session_id] = role
        if role == "VALIDATION_ONLY":
            has_validation = True
    return assigned
