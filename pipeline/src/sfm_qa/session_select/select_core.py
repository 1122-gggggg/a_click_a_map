"""Greedy BASE_CORE / BASE_SUPPORT selection. Timestamp is never a ranking key."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .config import lookup
from .critical_bridges import session_graph_diagnostics
from .objective import compute_objective_terms, delta_utility, efficiency_coverage, efficiency_info
from .types import SessionEdgeQuality, SessionQuality, edge_is_vpr_only

_CORE_OK = frozenset({"STRONG", "USABLE"})
_GEOMETRIC_EDGE_STATUS = frozenset({"STRONG", "USABLE"})
_BLOCKED_EDGE = frozenset({"AMBIGUOUS", "REJECT", "WEAK"})


def _qualities(rows: Iterable[SessionQuality]) -> dict[str, SessionQuality]:
    return {row.session_id: row for row in rows}


def _edges(rows: Iterable[SessionEdgeQuality]) -> list[SessionEdgeQuality]:
    return list(rows)


def _incident_edge_index(
    edges: Sequence[SessionEdgeQuality],
) -> dict[str, list[SessionEdgeQuality]]:
    """Undirected endpoint → incident edges, each list in original edge order."""

    index: dict[str, list[SessionEdgeQuality]] = {}
    for edge in edges:
        index.setdefault(edge.session_a, []).append(edge)
        if edge.session_b != edge.session_a:
            index.setdefault(edge.session_b, []).append(edge)
    return index



def _has_map_evidence(row: SessionQuality) -> bool:
    """Video-only WEAK rows are proposals; mapped WEAK rows can be ranked."""

    positive_values = (
        row.registered_ratio,
        row.num_tracks,
        row.num_observations,
        row.positive_depth_ratio,
        row.convex_hull_coverage,
        row.grid_occupancy_4x4,
        row.parallax_median_deg,
        row.fim_condition_number,
    )
    for value in positive_values:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            return True
    for value in (row.reprojection_rmse, row.reprojection_p90, row.fim_logdet):
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return True
    return False


def _session_is_selectable(row: SessionQuality) -> bool:
    return row.internal_status in _CORE_OK or (
        row.internal_status == "WEAK" and _has_map_evidence(row)
    )


def connecting_edges(
    session_id: str,
    selected: set[str],
    edges: Sequence[SessionEdgeQuality],
    index: Mapping[str, Sequence[SessionEdgeQuality]] | None = None,
) -> list[SessionEdgeQuality]:
    candidates = edges if index is None else index.get(session_id, ())
    found: list[SessionEdgeQuality] = []
    for edge in candidates:
        ends = {edge.session_a, edge.session_b}
        if session_id in ends and (ends - {session_id}) & selected:
            found.append(edge)
    return found


def _link_is_geometric(edge: SessionEdgeQuality) -> bool:
    if edge.status in _BLOCKED_EDGE or edge.status not in _GEOMETRIC_EDGE_STATUS:
        return False
    support = (
        edge.num_verified_pairs,
        edge.num_cross_session_tracks,
        edge.inlier_count,
    )
    has_support = False
    for value in support:
        if value is None or isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0.0:
            has_support = True
            break
    if not has_support:
        return False
    groups = edge.independent_bridge_groups
    if groups is None or isinstance(groups, bool):
        return False
    try:
        group_count = float(groups)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(group_count) or group_count <= 0.0:
        return False
    if edge_is_vpr_only(edge):
        return False
    return True


def connection_is_admissible(
    session_id: str,
    selected: set[str],
    edges: Sequence[SessionEdgeQuality],
    index: Mapping[str, Sequence[SessionEdgeQuality]] | None = None,
) -> tuple[bool, str]:
    """Fail-closed: no REJECT / AMBIGUOUS / single-critical-bridge merge into the core."""

    if not selected:
        return True, "empty_core"
    links = connecting_edges(session_id, selected, edges, index)

    if not links:
        return False, "no_connection_to_core"
    blocked = [edge for edge in links if edge.status in _BLOCKED_EDGE]
    usable = [edge for edge in links if _link_is_geometric(edge)]
    if not usable:
        if blocked:
            return False, "only_blocked_edges"
        return False, "no_geometric_edge"
    critical_only = [edge for edge in usable if edge.is_critical_bridge]
    noncritical = [edge for edge in usable if not edge.is_critical_bridge]
    if critical_only and not noncritical and len(critical_only) == 1:
        return False, "single_critical_bridge"
    if any(edge.is_critical_bridge and edge.status != "STRONG" for edge in usable) and not noncritical:
        return False, "unverified_critical_bridge"
    return True, "admissible"


def seed_session(
    qualities: Iterable[SessionQuality],
    config: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> str | None:
    """Max internal quality among STRONG/USABLE. Timestamp is not used."""

    del config
    blocked = set(exclude)
    eligible = [
        row
        for row in qualities
        if row.internal_status in _CORE_OK and row.session_id not in blocked
    ]
    if not eligible:
        return None

    def key(row: SessionQuality) -> tuple[float, float, float, float]:
        # Timestamp is intentionally absent from this key.
        coverage = 0.0
        if row.convex_hull_coverage is not None:
            coverage += float(row.convex_hull_coverage)
        if row.grid_occupancy_4x4 is not None:
            coverage += float(row.grid_occupancy_4x4)
        info = float(row.fim_logdet) if row.fim_logdet is not None else 0.0
        consistency = 0.0 if row.rotation_cycle_error is None else -float(row.rotation_cycle_error)
        return (row.internal_quality_score, coverage, info, consistency)

    return max(eligible, key=key).session_id


def _budget_ok(
    selected: set[str],
    candidate: SessionQuality,
    qualities: Mapping[str, SessionQuality],
    config: Mapping[str, Any] | None,
) -> bool:
    cfg = dict(config or {})
    max_sessions = lookup(cfg, "selection.max_base_sessions")
    if max_sessions is not None and len(selected) + 1 > int(max_sessions):
        return False
    max_tracks = lookup(cfg, "selection.max_total_tracks")
    max_obs = lookup(cfg, "selection.max_total_observations")
    max_per = lookup(cfg, "selection.max_tracks_per_session")
    if max_per is not None and candidate.num_tracks is not None and candidate.num_tracks > int(max_per):
        return False
    tracks = sum((qualities[sid].num_tracks or 0) for sid in selected) + (candidate.num_tracks or 0)
    obs = sum((qualities[sid].num_observations or 0) for sid in selected) + (candidate.num_observations or 0)
    if max_tracks is not None and tracks > int(max_tracks):
        return False
    if max_obs is not None and obs > int(max_obs):
        return False
    return True


def split_core_vs_support(
    selected: Iterable[str],
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality],
    config: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """BASE_CORE if removing it drops coverage/connectivity/info a lot; else SUPPORT."""

    cfg = dict(config or {})
    min_info = float(lookup(cfg, "selection.min_information_gain", 0.02) or 0.02)
    min_cov = float(lookup(cfg, "selection.min_coverage_gain", 0.02) or 0.02)
    quality_by_id = _qualities(qualities)
    edge_rows = _edges(edges)
    chosen = [sid for sid in selected if sid in quality_by_id]
    if not chosen:
        return [], []
    if len(chosen) == 1:
        return list(chosen), []

    full = compute_objective_terms(quality_by_id.values(), edge_rows, chosen, cfg)
    core: list[str] = []
    support: list[str] = []
    for sid in chosen:
        remainder = [other for other in chosen if other != sid]
        reduced = compute_objective_terms(quality_by_id.values(), edge_rows, remainder, cfg)
        drop_cov = full["coverage"] - reduced["coverage"]
        drop_info = full["information"] - reduced["information"]
        drop_conn = full["connectivity"] - reduced["connectivity"]
        drop_red = full["redundancy"] - reduced["redundancy"]
        if drop_cov >= min_cov or drop_info >= min_info or drop_conn > 0.05 or drop_red > 0.15:
            core.append(sid)
        else:
            support.append(sid)
    if not core:
        best = max(chosen, key=lambda sid: quality_by_id[sid].internal_quality_score)
        core = [best]
        support = [sid for sid in chosen if sid != best]
    return core, support


def _close_cycles(
    selected: list[str],
    leftover: Sequence[SessionQuality],
    qualities: Mapping[str, SessionQuality],
    edges: Sequence[SessionEdgeQuality],
    config: Mapping[str, Any],
    index: Mapping[str, Sequence[SessionEdgeQuality]] | None = None,
) -> list[str]:
    """If the selected graph is a tree, add one USABLE closer that does not explode tracks."""

    if len(selected) < 2:
        return selected
    inner = [
        edge
        for edge in edges
        if edge.session_a in selected
        and edge.session_b in selected
        and edge.status in {"STRONG", "USABLE"}
        and _link_is_geometric(edge)
    ]
    diagnostics = session_graph_diagnostics(selected, inner)
    tree_like = diagnostics["usable_edges"] if "usable_edges" in diagnostics else len(inner)
    n = len(selected)
    if tree_like >= n:
        return selected
    closer_ids: list[str] = []
    selected_set = set(selected)
    for row in leftover:
        if not _session_is_selectable(row):
            continue
        links = connecting_edges(row.session_id, selected_set, edges, index)

        usable = [
            edge
            for edge in links
            if edge.status in {"STRONG", "USABLE"}
            and _link_is_geometric(edge)
            and not edge.is_critical_bridge
        ]
        if len(usable) >= 2 and _budget_ok(set(selected), row, qualities, config):
            closer_ids.append(row.session_id)
    if closer_ids:
        pick = max(closer_ids, key=lambda sid: qualities[sid].internal_quality_score)
        return selected + [pick]
    return selected


def greedy_select_core(
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality],
    config: Mapping[str, Any] | None = None,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Seed by internal quality, then add argmax ΔU under fail-closed connection rules."""

    cfg = dict(config or {})
    quality_by_id = _qualities(qualities)
    edge_rows = _edges(edges)
    edge_index = _incident_edge_index(edge_rows)

    weights = lookup(cfg, "selection.weights")
    blocked = set(exclude)

    seed = seed_session(quality_by_id.values(), cfg, exclude=blocked)
    relative_fallback = False
    if seed is None:
        weak_mapped = [
            row
            for row in quality_by_id.values()
            if row.session_id not in blocked
            and row.internal_status == "WEAK"
            and _has_map_evidence(row)
        ]
        if weak_mapped:
            seed = max(
                weak_mapped,
                key=lambda row: (
                    row.internal_quality_score,
                    float(row.convex_hull_coverage or 0.0),
                    float(row.fim_logdet or 0.0),
                    row.session_id,
                ),
            ).session_id
            relative_fallback = True
    if seed is None:
        return {
            "core": [],
            "support": [],
            "selected": [],
            "scores": {},
            "stop_reason": "no_strong_or_usable_seed",
            "seed": None,
            "selection_mode": "NO_MAPPED_CANDIDATE",
            "relative_fallback_used": False,
            "best_available_not_release": False,
        }

    selected = [seed]
    selected_set = {seed}
    remaining = [sid for sid in quality_by_id if sid != seed and sid not in blocked]
    stop_reason = "exhausted_candidates"

    while remaining:
        current = compute_objective_terms(quality_by_id.values(), edge_rows, selected, cfg)
        ranked: list[tuple[float, float, str, dict[str, float], str]] = []
        for sid in remaining:
            row = quality_by_id[sid]
            if not _session_is_selectable(row):
                continue
            if not _budget_ok(selected_set, row, quality_by_id, cfg):
                continue
            ok, why = connection_is_admissible(sid, selected_set, edge_rows, edge_index)

            if not ok:
                continue
            after = compute_objective_terms(quality_by_id.values(), edge_rows, selected + [sid], cfg)
            delta_u = delta_utility(current, after, weights)
            d_obs = after["observations"] - current["observations"]
            d_cov = after["coverage"] - current["coverage"]
            d_info = after["information"] - current["information"]
            eff_i = efficiency_info(d_info, d_obs) or 0.0
            eff_c = efficiency_coverage(d_cov, d_obs) or 0.0
            ranked.append((delta_u, eff_i + eff_c, sid, after, why))
        if not ranked:
            stop_reason = "no_admissible_neighbor"
            break
        ranked.sort(reverse=True)
        delta_u, _eff, sid, after, _why = ranked[0]
        d_cov = after["coverage"] - current["coverage"]
        d_info = after["information"] - current["information"]
        d_red = after["redundancy"] - current["redundancy"]
        no_measured_gain = d_cov <= 0.0 and d_info <= 0.0 and d_red <= 0.0
        if delta_u <= 0.0 and no_measured_gain:
            stop_reason = "nonpositive_delta_u"
            break
        selected.append(sid)
        selected_set.add(sid)
        remaining.remove(sid)

    leftover_rows = [
        quality_by_id[sid]
        for sid in quality_by_id
        if sid not in selected_set and sid not in blocked
    ]
    selected = _close_cycles(selected, leftover_rows, quality_by_id, edge_rows, cfg, edge_index)

    core, support = split_core_vs_support(selected, quality_by_id.values(), edge_rows, cfg)
    scores = compute_objective_terms(quality_by_id.values(), edge_rows, selected, cfg)
    scores["seed"] = seed
    return {
        "core": core,
        "support": support,
        "selected": selected,
        "scores": scores,
        "stop_reason": stop_reason,
        "seed": seed,
        "selection_mode": (
            "RELATIVE_WEAK_FALLBACK" if relative_fallback else "RELATIVE_OBJECTIVE"
        ),
        "relative_fallback_used": relative_fallback,
        "best_available_not_release": relative_fallback,
    }
