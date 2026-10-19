"""Pre-build video admission planning.

This module is deliberately proposal-only. Retrieval and video QA can rank which
sessions are worth spending geometry on, but they never authorize an SfM merge.
The final BASE_CORE/BASE_SUPPORT selection still requires verified geometric
session edges in :mod:`select_core`.

The graph objective follows two ideas from the SfM literature:

* camera-triplet edge scoring: compare an edge with the strongest edge in each
  triangle, then aggregate across triangles;
* budgeted coverage: greedily prefer sessions that add new graph neighbourhood
  coverage instead of adding every redundant video.

Both are heuristic at session level here. Image-pair geometry remains the
authoritative gate later in S3/S4.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from sfm_qa.relative_quality import percentile_ranks, weighted_observed_score

from .config import lookup
from .types import SessionEdgeQuality, SessionQuality

_UNRECOVERABLE_REASONS = frozenset({"missing_video", "unreadable_video"})


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    if not math.isfinite(number):
        return low
    return max(low, min(high, number))


def _config_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = lookup(dict(config), key, default)
    return float(default if value is None else value)


def _config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = lookup(dict(config), key, default)
    return int(default if value is None else value)


def _edge_value(edge: SessionEdgeQuality | Mapping[str, Any], field: str) -> float:
    if isinstance(edge, Mapping):
        value = edge.get(field)
    else:
        value = getattr(edge, field, None)
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _edge_ends(edge: SessionEdgeQuality | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(edge, Mapping):
        left = str(edge.get("session_a") or edge.get("a") or "")
        right = str(edge.get("session_b") or edge.get("b") or "")
    else:
        left, right = str(edge.session_a), str(edge.session_b)
    return (left, right) if left <= right else (right, left)


def camera_triplet_scores(
    edges: Iterable[SessionEdgeQuality | Mapping[str, Any]],
    *,
    count_field: str = "num_candidate_pairs",
) -> dict[tuple[str, str], dict[str, float]]:
    """Return the session-edge analogue of camera-triplet support.

    For every complete triangle ``t`` and edge ``e`` in that triangle,

    ``q(e,t) = n_e / max(n_1, n_2, n_3)``.

    The final score is the mean over triangles containing the edge.  With
    ``count_field='num_candidate_pairs'`` this is proposal evidence only; with
    verified counts it becomes geometric support evidence.  A score of zero
    means "no complete triplet evidence", not "false edge".
    """

    counts: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()
    for edge in edges:
        left, right = _edge_ends(edge)
        if not left or not right or left == right:
            continue
        count = _edge_value(edge, count_field)
        if count <= 0.0:
            continue
        counts[(left, right)] = max(counts.get((left, right), 0.0), count)
        nodes.update((left, right))

    sums: dict[tuple[str, str], float] = {key: 0.0 for key in counts}
    support: dict[tuple[str, str], int] = {key: 0 for key in counts}
    ordered = sorted(nodes)
    for a, b, c in combinations(ordered, 3):
        triangle = ((a, b), (a, c), (b, c))
        if any(key not in counts for key in triangle):
            continue
        maximum = max(counts[key] for key in triangle)
        if maximum <= 0.0:
            continue
        for key in triangle:
            sums[key] += counts[key] / maximum
            support[key] += 1

    return {
        key: {
            "score": (sums[key] / support[key]) if support[key] else 0.0,
            "triplets": float(support[key]),
            "count": float(counts[key]),
        }
        for key in counts
    }


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
    array = [max(0.0, float(value)) for value in values]
    total = sum(array)
    if total <= 0.0:
        return None
    return tuple(value / total for value in array)


def motion_profile_distance(left: SessionQuality, right: SessionQuality) -> float:
    """Total-variation distance between two measured motion histograms."""

    first = _motion_profile(left)
    second = _motion_profile(right)
    if first is None or second is None:
        return 0.0
    return 0.5 * sum(abs(a - b) for a, b in zip(first, second))


def exposure_distance(left: SessionQuality, right: SessionQuality) -> float:
    """Measured exposure diversity; capture time is intentionally not a proxy."""

    if left.exposure_mean is None or right.exposure_mean is None:
        return 0.0
    return _clamp(abs(float(left.exposure_mean) - float(right.exposure_mean)) / 255.0)


def _risk_score(row: SessionQuality) -> tuple[float, float]:
    """Bounded risk and evidence completeness from observed measurements only."""

    return weighted_observed_score(
        {
            "hover": _clamp(row.hover_ratio) if row.hover_ratio is not None else None,
            "pure_rotation": (
                _clamp(row.pure_rotation_ratio)
                if row.pure_rotation_ratio is not None
                else None
            ),
            "fast_motion": (
                _clamp(row.fast_motion_ratio)
                if row.fast_motion_ratio is not None
                else None
            ),
            "low_parallax": (
                _clamp(row.low_parallax_ratio)
                if row.low_parallax_ratio is not None
                else None
            ),
            "duplicate": (
                _clamp(row.near_duplicate_ratio)
                if row.near_duplicate_ratio is not None
                else None
            ),
            "underexposed": (
                _clamp(row.underexposed_ratio)
                if row.underexposed_ratio is not None
                else None
            ),
            "overexposed": (
                _clamp(row.overexposed_ratio)
                if row.overexposed_ratio is not None
                else None
            ),
            "epipolar": (
                _clamp(row.epipolar_outlier_ratio_median)
                if row.epipolar_outlier_ratio_median is not None
                else None
            ),
            "unproven": (
                _clamp(row.unproven_ratio) if row.unproven_ratio is not None else None
            ),
        },
        {
            "hover": 0.10,
            "pure_rotation": 0.10,
            "fast_motion": 0.10,
            "low_parallax": 0.05,
            "duplicate": 0.20,
            "underexposed": 0.075,
            "overexposed": 0.075,
            "epipolar": 0.20,
            "unproven": 0.10,
        },
        empty_score=0.5,
    )


def video_risk(row: SessionQuality) -> float:
    """Bounded risk proxy without treating missing measurements as healthy."""

    return _risk_score(row)[0]


def _video_terms(
    row: SessionQuality,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float | None], dict[str, float]]:
    cfg = dict(config or {})
    sharp_reference = float(lookup(cfg, "prebuild.sharpness_reference", 100.0) or 100.0)
    terms = {
        "internal_quality": _clamp(row.internal_quality_score),
        "parallax": (
            _clamp(row.parallax_ratio) if row.parallax_ratio is not None else None
        ),
        "low_parallax_credit": (
            _clamp(row.low_parallax_ratio)
            if row.low_parallax_ratio is not None
            else None
        ),
        "sharpness": (
            _clamp(float(row.sharpness_p10) / max(sharp_reference, 1e-9))
            if row.sharpness_p10 is not None
            else None
        ),
        "motion_evidence": (
            1.0 - _clamp(row.unproven_ratio) if row.unproven_ratio is not None else None
        ),
        "non_duplicate": (
            1.0 - _clamp(row.near_duplicate_ratio)
            if row.near_duplicate_ratio is not None
            else None
        ),
        "underexposure_quality": (
            1.0 - _clamp(row.underexposed_ratio)
            if row.underexposed_ratio is not None
            else None
        ),
        "overexposure_quality": (
            1.0 - _clamp(row.overexposed_ratio)
            if row.overexposed_ratio is not None
            else None
        ),
        "epipolar_consistency": (
            1.0 - _clamp(row.epipolar_outlier_ratio_median)
            if row.epipolar_outlier_ratio_median is not None
            else None
        ),
    }
    raw_weights = lookup(cfg, "prebuild.video_weights", {}) or {}
    parallax_weight = float(raw_weights.get("parallax", 0.25))
    exposure_weight = float(raw_weights.get("exposure", 0.10))
    weights = {
        "internal_quality": float(raw_weights.get("internal_quality", 0.20)),
        "parallax": parallax_weight * 0.80,
        "low_parallax_credit": parallax_weight * 0.20,
        "sharpness": float(raw_weights.get("sharpness", 0.15)),
        "motion_evidence": float(raw_weights.get("motion_evidence", 0.10)),
        "non_duplicate": float(raw_weights.get("non_duplicate", 0.10)),
        "underexposure_quality": exposure_weight * 0.50,
        "overexposure_quality": exposure_weight * 0.50,
        "epipolar_consistency": float(raw_weights.get("epipolar_consistency", 0.10)),
    }
    return terms, weights


def _video_score(
    row: SessionQuality,
    config: Mapping[str, Any] | None = None,
) -> tuple[float, float]:
    terms, weights = _video_terms(row, config)
    return weighted_observed_score(terms, weights)


def video_admission_score(
    row: SessionQuality,
    config: Mapping[str, Any] | None = None,
) -> float:
    """Observed-metric score for geometry verification, before cohort ranking."""

    return _video_score(row, config)[0]


def _unrecoverable(row: SessionQuality) -> str | None:
    reasons = {str(reason).split(":", 1)[0] for reason in (row.reasons or ())}
    found = sorted(reasons & _UNRECOVERABLE_REASONS)
    return found[0] if found else None


def _proposal_graph(
    edges: Sequence[SessionEdgeQuality | Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, dict[str, float]],
    dict[tuple[str, str], dict[str, float]],
]:
    raw: dict[tuple[str, str], float] = {}
    for edge in edges:
        left, right = _edge_ends(edge)
        if not left or not right or left == right:
            continue
        count = _edge_value(edge, "num_candidate_pairs")
        if count > 0:
            raw[(left, right)] = max(raw.get((left, right), 0.0), count)
    maximum = max(raw.values(), default=0.0)
    strengths: dict[tuple[str, str], float] = {}
    adjacency: dict[str, dict[str, float]] = {}
    for key, count in raw.items():
        if maximum <= 1.0:
            strength = _clamp(count / max(maximum, 1.0))
        else:
            strength = math.log1p(count) / math.log1p(maximum)
        strengths[key] = _clamp(strength)
        left, right = key
        adjacency.setdefault(left, {})[right] = strengths[key]
        adjacency.setdefault(right, {})[left] = strengths[key]
    return strengths, adjacency, camera_triplet_scores(edges, count_field="num_candidate_pairs")


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def propose_prebuild_set(
    qualities: Iterable[SessionQuality],
    edges: Iterable[SessionEdgeQuality | Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Propose which videos deserve S3 geometric verification.

    This *never* returns merge authority.  ``proposed_base_sessions`` is a
    budgeted candidate set; every cross-session relation is emitted again in
    ``verification_pairs`` and must be verified geometrically before the normal
    BASE_CORE/BASE_SUPPORT selector may admit it.
    """

    cfg = dict(config or {})
    rows = {row.session_id: row for row in qualities}
    edge_rows = list(edges)
    min_video_score = float(lookup(cfg, "prebuild.min_video_score", 0.35) or 0.35)
    relative_admission = bool(lookup(cfg, "prebuild.relative_admission", True))
    marginal_keep_ratio = max(
        0.0,
        _config_float(cfg, "prebuild.relative_marginal_keep_ratio", 0.25),
    )
    min_base = int(lookup(cfg, "prebuild.min_base_sessions", 2) or 2)
    max_sessions_cfg = lookup(cfg, "prebuild.max_sessions")
    max_sessions = int(max_sessions_cfg) if max_sessions_cfg is not None else max(1, len(rows))
    max_no_graph = int(lookup(cfg, "prebuild.max_no_graph_sessions", 3) or 3)
    validation_count = max(0, _config_int(cfg, "prebuild.validation_candidates", 1))
    allow_weak = bool(lookup(cfg, "prebuild.allow_weak_video_candidates", False))

    rankable: list[str] = []
    rejected: dict[str, str] = {}
    for sid, row in rows.items():
        fatal = _unrecoverable(row)
        if fatal is not None:
            rejected[sid] = f"unrecoverable_input={fatal}"
        else:
            rankable.append(sid)
    rankable_set = set(rankable)

    component_payloads = {sid: _video_terms(row, cfg) for sid, row in rows.items()}
    measured_scores = {
        sid: weighted_observed_score(terms, weights)
        for sid, (terms, weights) in component_payloads.items()
    }
    video_scores = {sid: score for sid, (score, _) in measured_scores.items()}
    completeness = {sid: available for sid, (_, available) in measured_scores.items()}
    measured_risks = {sid: _risk_score(row) for sid, row in rows.items()}
    risks = {sid: score for sid, (score, _) in measured_risks.items()}
    risk_completeness = {
        sid: available for sid, (_, available) in measured_risks.items()
    }
    video_weights = next(
        (weights for _, weights in component_payloads.values()),
        {},
    )
    component_ranks = {
        name: percentile_ranks(
            {
                sid: (
                    component_payloads[sid][0].get(name)
                    if sid in rankable_set
                    else None
                )
                for sid in rows
            },
            higher_is_better=True,
        )
        for name in video_weights
    }
    relative_video_scores = {
        sid: weighted_observed_score(
            {name: component_ranks[name].get(sid) for name in video_weights},
            video_weights,
        )[0]
        for sid in rows
    }
    quality_ranks = percentile_ranks(
        {
            sid: relative_video_scores[sid] if sid in rankable_set else None
            for sid in rows
        },
        higher_is_better=True,
    )
    low_risk_ranks = percentile_ranks(
        {sid: risks[sid] if sid in rankable_set else None for sid in rows},
        higher_is_better=False,
    )
    absolute_weight = max(
        0.0, _config_float(cfg, "prebuild.absolute_quality_weight", 0.35)
    )
    relative_weight = max(
        0.0, _config_float(cfg, "prebuild.relative_quality_weight", 0.65)
    )
    weight_sum = max(1e-9, absolute_weight + relative_weight)

    def status_penalty(row: SessionQuality) -> float:
        return {
            "STRONG": 0.00,
            "USABLE": 0.00,
            "WEAK": 0.08,
            "REJECT": 0.20,
            "INCONSISTENT": 0.30,
        }.get(row.internal_status, 0.10)

    portfolio_scores: dict[str, float] = {}
    portfolio_risks: dict[str, float] = {}
    for sid, row in rows.items():
        relative_score = relative_video_scores[sid]
        low_risk_rank = low_risk_ranks.get(sid)
        relative_low_risk = 0.5 if low_risk_rank is None else float(low_risk_rank)
        evidence_factor = 0.70 + 0.30 * completeness[sid]
        portfolio_scores[sid] = _clamp(
            (
                absolute_weight * video_scores[sid]
                + relative_weight * relative_score
            )
            / weight_sum
            * evidence_factor
            - status_penalty(row)
        )
        portfolio_risks[sid] = _clamp(
            0.5 * risks[sid]
            + 0.5 * (1.0 - relative_low_risk)
            + 0.20 * (1.0 - risk_completeness[sid])
            + status_penalty(row)
        )

    eligible: list[str] = []
    deferred: dict[str, str] = {}
    if relative_admission:
        # Status contributes a penalty above, but never becomes a quality gate.
        # This lets a complementary WEAK session beat a redundant USABLE one.
        eligible = list(rankable)
    else:
        for sid in rankable:
            row = rows[sid]
            if row.internal_status in {"REJECT", "INCONSISTENT"}:
                rejected[sid] = f"internal_status={row.internal_status}"
                continue
            if row.internal_status == "WEAK" and not allow_weak:
                rejected[sid] = "weak_video_requires_explicit_override_or_geometry"
                continue
            if video_scores[sid] < min_video_score:
                rejected[sid] = f"video_score_below_heuristic_{min_video_score:.3f}"
                continue
            eligible.append(sid)

    absolute_reference_passed = {
        sid: (
            sid in rankable_set
            and rows[sid].internal_status in {"STRONG", "USABLE"}
            and video_scores[sid] >= min_video_score
        )
        for sid in rows
    }

    def score_payload(sid: str) -> dict[str, Any]:
        return {
            "video_admission_score": float(video_scores[sid]),
            "relative_metric_score": float(relative_video_scores[sid]),
            "relative_quality_rank": quality_ranks.get(sid),
            "risk": float(risks[sid]),
            "risk_evidence_completeness": float(risk_completeness[sid]),
            "relative_low_risk_rank": low_risk_ranks.get(sid),
            "evidence_completeness": float(completeness[sid]),
            "portfolio_score": float(portfolio_scores[sid]),
            "portfolio_risk": float(portfolio_risks[sid]),
            "absolute_reference_passed": bool(absolute_reference_passed[sid]),
            "selected_for_geometry": False,
            "validation_candidate": False,
        }

    graph_nodes = rankable_set if relative_admission else set(eligible)
    proposal_edges = [
        edge
        for edge in edge_rows
        if set(_edge_ends(edge)) <= graph_nodes
    ]
    strengths, adjacency, triplets = _proposal_graph(proposal_edges)
    graph_available = bool(strengths)
    if not eligible:
        return {
            "proposed_base_sessions": [],
            "validation_candidates": [],
            "verification_pairs": [],
            "session_scores": {sid: score_payload(sid) for sid in rows},
            "rejected": rejected,
            "deferred": deferred,
            "proposal_confidence": "NONE",
            "proposal_graph_available": graph_available,
            "selection_mode": (
                "RELATIVE_PORTFOLIO" if relative_admission else "LEGACY_ABSOLUTE_FILTER"
            ),
            "relative_fallback_used": False,
            "best_available_not_release": False,
            "requires_geometric_verification": True,
            "notes": [
                "No readable video is available for a geometry probe.",
                "Retrieval is never geometric merge authority.",
            ],
        }

    max_frames = max((rows[sid].num_frames for sid in eligible), default=1) or 1
    weights = lookup(cfg, "prebuild.weights", {}) or {}
    w_video = float(weights.get("video_quality", 0.45))
    w_bridge = float(weights.get("bridgeability", 0.25))
    w_graph_cov = float(weights.get("graph_coverage", 0.15))
    w_diversity = float(weights.get("motion_diversity", 0.10))
    w_appearance = float(weights.get("appearance_diversity", 0.05))
    w_triplet = float(weights.get("triplet_support", 0.10))
    w_multi = float(weights.get("multi_link", 0.10))
    w_risk = float(weights.get("risk", 0.20))
    w_cost = float(weights.get("frame_cost", 0.05))
    w_redundancy = float(weights.get("redundancy", 0.10))

    seed = max(
        eligible,
        key=lambda sid: (
            portfolio_scores[sid] - 0.5 * portfolio_risks[sid],
            rows[sid].num_keyframes,
            sid,
        ),
    )
    selected = [seed]
    selected_set = {seed}
    ranked_steps: list[dict[str, Any]] = [
        {
            "session_id": seed,
            "marginal_score": portfolio_scores[seed] - 0.5 * portfolio_risks[seed],
            "reason": "highest_relative_portfolio_score",
        }
    ]

    def covered_neighbourhood(chosen: set[str]) -> set[str]:
        covered = set(chosen)
        for sid in chosen:
            covered.update(adjacency.get(sid, {}))
        return covered

    remaining = [sid for sid in eligible if sid != seed]
    # Preserve an untouched validation candidate whenever the pool is large enough.
    # This is proposal-stage isolation only; the final S0 corpus lock still owns
    # the authoritative build/test split and content-hash proof.
    preferred_count = sum(
        rows[sid].internal_status in {"STRONG", "USABLE"} for sid in eligible
    )
    capacity_pool = preferred_count if preferred_count >= min_base else len(eligible)
    reserve = validation_count if capacity_pool > min_base else 0
    proposal_capacity = max(min_base, capacity_pool - max(0, reserve))
    budget = min(max_sessions, proposal_capacity, len(eligible))
    if not graph_available:
        budget = min(budget, max_no_graph)

    while remaining and len(selected) < budget:
        covered = covered_neighbourhood(selected_set)
        total_weight = sum(portfolio_scores[sid] for sid in eligible) or 1.0
        candidates: list[tuple[float, str, dict[str, float]]] = []
        for sid in remaining:
            links = [
                strengths.get(_pair_key(sid, other), 0.0)
                for other in selected
                if _pair_key(sid, other) in strengths
            ]
            bridgeability = max(links, default=0.0)
            multi_link = _clamp(len([value for value in links if value > 0.0]) / 2.0)
            triplet_support = max(
                (
                    triplets.get(_pair_key(sid, other), {}).get("score", 0.0)
                    for other in selected
                ),
                default=0.0,
            )
            profile_distances = [motion_profile_distance(rows[sid], rows[other]) for other in selected]
            diversity = (
                sum(profile_distances) / len(profile_distances) if profile_distances else 0.0
            )
            exposure_distances = [
                exposure_distance(rows[sid], rows[other]) for other in selected
            ]
            appearance_diversity = (
                sum(exposure_distances) / len(exposure_distances)
                if exposure_distances
                else 0.0
            )
            new_nodes = ({sid} | set(adjacency.get(sid, {}))) - covered
            graph_coverage = (
                sum(portfolio_scores.get(node, 0.0) for node in new_nodes) / total_weight
                if graph_available
                else 0.0
            )
            frame_cost = _clamp(rows[sid].num_frames / max_frames)
            redundancy = bridgeability * (1.0 - diversity) * (1.0 - appearance_diversity)
            terms = {
                "video_quality": portfolio_scores[sid],
                "bridgeability": bridgeability,
                "graph_coverage": graph_coverage,
                "motion_diversity": diversity,
                "appearance_diversity": appearance_diversity,
                "triplet_support": float(triplet_support),
                "multi_link": multi_link,
                "risk": portfolio_risks[sid],
                "frame_cost": frame_cost,
                "redundancy": redundancy,
            }
            marginal = (
                w_video * terms["video_quality"]
                + w_bridge * terms["bridgeability"]
                + w_graph_cov * terms["graph_coverage"]
                + w_diversity * terms["motion_diversity"]
                + w_appearance * terms["appearance_diversity"]
                + w_triplet * terms["triplet_support"]
                + w_multi * terms["multi_link"]
                - w_risk * terms["risk"]
                - w_cost * terms["frame_cost"]
                - w_redundancy * terms["redundancy"]
            )
            candidates.append((marginal, sid, terms))

        candidates.sort(
            key=lambda item: (item[0], portfolio_scores[item[1]], item[1]),
            reverse=True,
        )
        marginal, sid, terms = candidates[0]
        previous_marginals = [
            float(step["marginal_score"])
            for step in ranked_steps[1:]
            if step.get("marginal_score") is not None
        ]
        best_previous = max(previous_marginals, default=marginal)
        if len(selected) >= min_base:
            if marginal <= 0.0:
                break
            if best_previous > 0.0 and marginal < best_previous * marginal_keep_ratio:
                break
        selected.append(sid)
        selected_set.add(sid)
        remaining.remove(sid)
        ranked_steps.append(
            {
                "session_id": sid,
                "marginal_score": float(marginal),
                "terms": terms,
                "reason": "greedy_budgeted_proposal",
            }
        )

    validation_pool = rankable if relative_admission else eligible
    remaining_ranked = sorted(
        (sid for sid in validation_pool if sid not in selected_set),
        key=lambda sid: (
            max(
                (strengths.get(_pair_key(sid, other), 0.0) for other in selected),
                default=0.0,
            )
            * 0.6
            + portfolio_scores[sid] * 0.4,
            portfolio_scores[sid],
            sid,
        ),
        reverse=True,
    )
    validation = remaining_ranked[: max(0, validation_count)]
    deferred = {
        sid: "not_selected_by_relative_portfolio"
        for sid in validation_pool
        if sid not in selected_set and sid not in validation
    }

    verification_pairs: list[dict[str, Any]] = []
    for left, right in combinations(selected, 2):
        key = _pair_key(left, right)
        candidate_count = 0
        for edge in proposal_edges:
            if _edge_ends(edge) == key:
                candidate_count = max(
                    candidate_count,
                    int(_edge_value(edge, "num_candidate_pairs")),
                )
        triplet_score = float(triplets.get(key, {}).get("score", 0.0))
        strength = strengths.get(key, 0.0)
        priority = strength + 0.5 * triplet_score
        verification_pairs.append(
            {
                "session_a": left,
                "session_b": right,
                "candidate_pairs": candidate_count,
                "proposal_strength": float(strength),
                "triplet_score": triplet_score,
                "priority": float(priority),
                "forced_probe": candidate_count <= 0,
                "requires_geometric_verification": True,
                "reason": (
                    "retrieval_candidate_requires_geometry"
                    if candidate_count > 0
                    else "no_retrieval_support_force_geometry_probe"
                ),
            }
        )
    verification_pairs.sort(
        key=lambda row: (
            bool(row["forced_probe"]),
            -float(row["priority"]),
            str(row["session_a"]),
            str(row["session_b"]),
        )
    )

    if not graph_available:
        confidence = "LOW"
    else:
        selected_edges = [
            key
            for key, strength in strengths.items()
            if strength > 0.0 and key[0] in selected_set and key[1] in selected_set
        ]
        connected_nodes = {node for edge in selected_edges for node in edge}
        has_triangle = any(
            triplets.get(key, {}).get("triplets", 0.0) > 0.0 for key in selected_edges
        )
        if selected_set <= connected_nodes and has_triangle:
            confidence = "HIGH_PROPOSAL_ONLY"
        elif selected_set <= connected_nodes:
            confidence = "MEDIUM_PROPOSAL_ONLY"
        else:
            confidence = "LOW"

    session_scores = {sid: score_payload(sid) for sid in rows}
    for sid in rows:
        session_scores[sid]["selected_for_geometry"] = sid in selected_set
        session_scores[sid]["validation_candidate"] = sid in validation
    relative_fallback_used = any(
        not absolute_reference_passed[sid] for sid in selected_set
    )
    return {
        "proposed_base_sessions": selected,
        "validation_candidates": validation,
        "verification_pairs": verification_pairs,
        "session_scores": session_scores,
        "ranked_steps": ranked_steps,
        "rejected": rejected,
        "deferred": deferred,
        "proposal_confidence": confidence,
        "proposal_graph_available": graph_available,
        "selection_mode": (
            "RELATIVE_PORTFOLIO" if relative_admission else "LEGACY_ABSOLUTE_FILTER"
        ),
        "relative_fallback_used": relative_fallback_used,
        "best_available_not_release": relative_fallback_used,
        "requires_geometric_verification": True,
        "notes": [
            "This is a pre-build proposal, not merge authority.",
            "Quality thresholds are diagnostic references; relative ranking keeps a non-empty best-available probe set.",
            "VPR/candidate counts rank verification effort only; they are never geometric edges.",
            "Final BASE_CORE/BASE_SUPPORT still requires verified multi-bridge geometry.",
        ],
    }


__all__ = [
    "camera_triplet_scores",
    "motion_profile_distance",
    "propose_prebuild_set",
    "video_admission_score",
    "video_risk",
]
