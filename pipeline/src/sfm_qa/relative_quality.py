"""Small, deterministic helpers for cohort-relative quality scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping


def percentile_ranks(
    values: Mapping[str, float | int | None],
    *,
    higher_is_better: bool = True,
) -> dict[str, float | None]:
    """Return tie-aware ranks in ``[0, 1]`` without dropping missing rows.

    Ranks are meaningful only inside the supplied cohort. A singleton receives
    ``1.0``; a fully tied cohort receives ``0.5`` so ties do not become false
    evidence of either exceptional quality or exceptional risk.
    """

    output: dict[str, float | None] = {str(key): None for key in values}
    observed: list[tuple[float, str]] = []
    for key, raw in values.items():
        if raw is None or isinstance(raw, bool):
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            observed.append((number, str(key)))
    observed.sort(key=lambda item: (item[0], item[1]))
    count = len(observed)
    if count == 0:
        return output
    if count == 1:
        output[observed[0][1]] = 1.0
        return output

    index = 0
    while index < count:
        end = index + 1
        while end < count and math.isclose(
            observed[end][0],
            observed[index][0],
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            end += 1
        rank = ((index + end - 1) / 2.0) / (count - 1)
        if not higher_is_better:
            rank = 1.0 - rank
        for _, key in observed[index:end]:
            output[key] = float(rank)
        index = end
    return output


def weighted_observed_score(
    values: Mapping[str, float | None],
    weights: Mapping[str, float],
    *,
    empty_score: float = 0.0,
) -> tuple[float, float]:
    """Average observed bounded terms and return evidence completeness.

    Missing terms are omitted from the numerator and denominator instead of
    silently receiving a perfect score. Completeness is the fraction of total
    configured weight backed by a finite observation.
    """

    total_weight = sum(max(0.0, float(weight)) for weight in weights.values())
    observed_weight = 0.0
    weighted_sum = 0.0
    for name, weight_raw in weights.items():
        weight = max(0.0, float(weight_raw))
        raw = values.get(name)
        if weight == 0.0 or raw is None:
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        observed_weight += weight
        weighted_sum += weight * max(0.0, min(1.0, number))
    completeness = observed_weight / total_weight if total_weight > 0.0 else 0.0
    if observed_weight <= 0.0:
        return max(0.0, min(1.0, float(empty_score))), float(completeness)
    return float(weighted_sum / observed_weight), float(completeness)


__all__ = ["percentile_ranks", "weighted_observed_score"]
