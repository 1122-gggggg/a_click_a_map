"""Base-core objective U(S) and efficiency ratios. Weights come from YAML only."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from .config import lookup
from .critical_bridges import session_graph_diagnostics
from .types import SessionEdgeQuality, SessionQuality

_CORE_OK = frozenset({"STRONG", "USABLE"})
_EDGE_OK = frozenset({"STRONG", "USABLE"})


def _weight_map(weights: Mapping[str, float] | None) -> dict[str, float]:
    defaults = {
        "coverage": 1.0,
        "quality": 1.0,
        "connectivity": 1.0,
        "redundancy": 1.0,
        "information": 1.0,
        "view_diversity": 0.5,
        "track_cost": 0.5,
        "risk": 1.5,
    }
    if not weights:
        return defaults
    out = dict(defaults)
    for key, value in weights.items():
        if value is not None:
            out[key] = float(value)
    return out


def utility(
    coverage: float,
    quality: float,
    connectivity: float,
    redundancy: float,
    information: float,
    view_diversity: float,
    track_cost: float,
    risk: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """U = α cov + β qual + γ conn + δ red + ε info + ζ view − η track_cost − θ risk."""

    w = _weight_map(weights)
    return (
        w["coverage"] * float(coverage)
        + w["quality"] * float(quality)
        + w["connectivity"] * float(connectivity)
        + w["redundancy"] * float(redundancy)
        + w["information"] * float(information)
        + w["view_diversity"] * float(view_diversity)
        - w["track_cost"] * float(track_cost)
        - w["risk"] * float(risk)
    )


def delta_utility(
    before: Mapping[str, float],
    after: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    keys = (
        "coverage",
        "quality",
        "connectivity",
        "redundancy",
        "information",
        "view_diversity",
        "track_cost",
        "risk",
    )
    return utility(*(float(after.get(k, 0.0)) for k in keys), weights=weights) - utility(
        *(float(before.get(k, 0.0)) for k in keys), weights=weights
    )


def efficiency_info(delta_info: float, delta_obs: float) -> float | None:
    if delta_obs is None or delta_obs <= 0:
        return None
    return float(delta_info) / float(delta_obs)


def efficiency_coverage(delta_cov: float, delta_obs: float) -> float | None:
    if delta_obs is None or delta_obs <= 0:
        return None
    return float(delta_cov) / float(delta_obs)


def _by_id(sessions: Iterable[SessionQuality]) -> dict[str, SessionQuality]:
    return {row.session_id: row for row in sessions}


def _edge_list(edges: Iterable[SessionEdgeQuality | Mapping[str, Any]]) -> list[SessionEdgeQuality]:
    out: list[SessionEdgeQuality] = []
    for edge in edges:
        if isinstance(edge, SessionEdgeQuality):
            out.append(edge)
        else:
            out.append(SessionEdgeQuality(**edge))  # type: ignore[arg-type]
    return out


def _incident(edges: Sequence[SessionEdgeQuality], selected: set[str]) -> list[SessionEdgeQuality]:
    return [edge for edge in edges if edge.session_a in selected and edge.session_b in selected]


def _finite(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _normalize(value: float | None, scale: float) -> float:
    if value is None or scale <= 0:
        return 0.0
    return max(0.0, min(1.0, float(value) / scale))


def _unit_grid_coverage(value: float | None) -> float:
    """Normalize either a [0,1] fraction or a 0..16 occupied-cell count."""

    if value is None:
        return 0.0
    number = max(0.0, float(value))
    if number <= 1.0:
        return number
    if number <= 16.0:
        return number / 16.0
    return 1.0


def _motion_profile(row: SessionQuality) -> tuple[float, ...] | None:
    values = (
        row.parallax_ratio,
        row.low_parallax_ratio,
        row.hover_ratio,
        row.pure_rotation_ratio,
        row.fast_motion_ratio,
        row.unproven_ratio,
    )
    if any(value is None for value in values):
        return None
    vector = [max(0.0, float(value)) for value in values]
    total = sum(vector)
    if total <= 0.0:
        return None
    return tuple(value / total for value in vector)


def _view_diversity(rows: Sequence[SessionQuality]) -> float:
    """Measured motion-profile diversity; timestamp is intentionally excluded."""

    profiles = [profile for row in rows if (profile := _motion_profile(row)) is not None]
    if len(profiles) >= 2:
        distances = [
            0.5 * sum(abs(a - b) for a, b in zip(left, right))
            for left, right in combinations(profiles, 2)
        ]
        return max(0.0, min(1.0, sum(distances) / len(distances)))
    # Legacy map-only rows may not carry motion histograms. Keep a weak
    # cardinality prior but never let capture time act as view diversity.
    return min(1.0, 0.15 * len(rows))


def compute_objective_terms(
    sessions: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality | Mapping[str, Any]],
    selected: Iterable[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Roll session/edge metrics into the eight U(S) terms for a selected set."""

    quality_by_id = _by_id(sessions)
    chosen = [sid for sid in selected if sid in quality_by_id]
    edge_rows = _edge_list(edges)
    if not chosen:
        return {
            "coverage": 0.0,
            "quality": 0.0,
            "connectivity": 0.0,
            "redundancy": 0.0,
            "information": 0.0,
            "view_diversity": 0.0,
            "track_cost": 0.0,
            "risk": 0.0,
            "utility": 0.0,
            "tracks": 0.0,
            "observations": 0.0,
        }

    selected_rows = [quality_by_id[sid] for sid in chosen]
    coverages = [
        max(
            0.0,
            min(1.0, _finite(row.convex_hull_coverage)),
            _unit_grid_coverage(row.grid_occupancy_4x4),
        )
        for row in selected_rows
    ]
    qualities = [row.internal_quality_score for row in selected_rows]
    infos = []
    for row in selected_rows:
        if row.fim_logdet is not None:
            infos.append(max(0.0, float(row.fim_logdet)))
        else:
            infos.append(_normalize(row.num_tracks, 50_000.0) * 4.0)
    tracks = sum(_finite(row.num_tracks) for row in selected_rows)
    observations = sum(_finite(row.num_observations) for row in selected_rows)
    keyframes = sum(row.num_keyframes for row in selected_rows)

    inner = _incident(edge_rows, set(chosen))
    diagnostics = session_graph_diagnostics(chosen, inner)
    usable_inner = [edge for edge in inner if edge.status in _EDGE_OK]
    n = len(chosen)
    max_edges = n * (n - 1) / 2.0
    redundancy = (len(usable_inner) / max_edges) if max_edges else 0.0
    connectivity = 0.0 if n <= 1 else min(1.0, diagnostics["fiedler_value"] / max(n, 1.0))
    cycle_bonus = 1.0 if len(usable_inner) >= n and n >= 3 else redundancy

    weak_count = sum(1 for sid in chosen if quality_by_id[sid].internal_status not in _CORE_OK)
    critical = sum(1 for edge in inner if edge.is_critical_bridge)
    ambiguous = sum(1 for edge in inner if edge.status in {"AMBIGUOUS", "REJECT"})
    risk = 0.25 * weak_count + 0.5 * critical + 1.0 * ambiguous

    terms = {
        "coverage": float(max(coverages) if coverages else _normalize(keyframes, 400.0)),
        "quality": float(sum(qualities) / len(qualities)),
        "connectivity": float(connectivity),
        "redundancy": float(cycle_bonus),
        "information": float(sum(infos) / max(len(infos), 1)),
        "view_diversity": float(_view_diversity(selected_rows)),
        "track_cost": float(
            _normalize(tracks, 400_000.0) + _normalize(observations, 2_000_000.0)
        ),
        "risk": float(risk),
        "tracks": float(tracks),
        "observations": float(observations),
        "fiedler_value": float(diagnostics["fiedler_value"]),
        "usable_edges": float(len(usable_inner)),
    }
    weights = None if config is None else lookup(dict(config), "selection.weights")
    terms["utility"] = utility(
        terms["coverage"],
        terms["quality"],
        terms["connectivity"],
        terms["redundancy"],
        terms["information"],
        terms["view_diversity"],
        terms["track_cost"],
        terms["risk"],
        weights=weights,
    )
    return terms
