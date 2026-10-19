from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds

from .statistics import (
    beta_posterior_mean,
    empirical_bayes_prior,
    wilson_interval,
)


@dataclass(frozen=True)
class RegionDiagnosisConfig:
    """Decision policy for confidence-aware weak-region diagnosis."""

    weak_failure_rate: float = 0.30
    healthy_failure_rate: float = 0.10
    confidence: float = 0.95
    min_samples: int = 8
    min_failures_for_weak: int = 2
    prior_strength: float = 8.0

    def __post_init__(self) -> None:
        rates = (self.healthy_failure_rate, self.weak_failure_rate)
        if any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            for rate in rates
        ):
            raise ValueError("failure-rate thresholds must be finite numbers")
        if not 0.0 <= self.healthy_failure_rate <= self.weak_failure_rate <= 1.0:
            raise ValueError(
                "require 0 <= healthy_failure_rate <= weak_failure_rate <= 1"
            )
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(self.confidence)
            or not 0.0 < self.confidence < 1.0
        ):
            raise ValueError("confidence must lie in (0, 1)")
        for name, value in (
            ("min_samples", self.min_samples),
            ("min_failures_for_weak", self.min_failures_for_weak),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (
            isinstance(self.prior_strength, bool)
            or not isinstance(self.prior_strength, (int, float))
            or not math.isfinite(self.prior_strength)
            or self.prior_strength <= 0.0
        ):
            raise ValueError("prior_strength must be finite and > 0")


@dataclass(frozen=True)
class RegionDiagnostic:
    region_id: str
    samples: int
    strict_failures: int
    raw_failure_rate: float
    posterior_failure_rate: float
    failure_rate_ci_low: float
    failure_rate_ci_high: float
    status: str
    evidence_strength: float
    failed_queries: tuple[str, ...]
    failure_reason_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["failed_queries"] = list(self.failed_queries)
        return output


@dataclass(frozen=True)
class RegionDiagnosticsReport:
    total_queries: int
    assigned_queries: int
    unassigned_queries: tuple[str, ...]
    strict_failures: int
    global_failure_rate: float
    prior_alpha: float
    prior_beta: float
    cell_size: float
    assignment_source: str
    config: RegionDiagnosisConfig
    regions: tuple[RegionDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_queries": self.total_queries,
            "assigned_queries": self.assigned_queries,
            "unassigned_queries": list(self.unassigned_queries),
            "strict_failures": self.strict_failures,
            "global_failure_rate": self.global_failure_rate,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "cell_size": self.cell_size,
            "assignment_source": self.assignment_source,
            "config": asdict(self.config),
            "regions": [region.to_dict() for region in self.regions],
        }


def _grid_region(result: QueryLocalizationResult, cell_size: float) -> str | None:
    if result.x is None or result.y is None or result.z is None:
        return None
    return "grid:{x}:{y}:{z}".format(
        x=math.floor(result.x / cell_size),
        y=math.floor(result.y / cell_size),
        z=math.floor(result.z / cell_size),
    )


def diagnose_regions(
    results: Sequence[QueryLocalizationResult],
    thresholds: LocalizationThresholds,
    *,
    assignments: Mapping[str, str] | None = None,
    cell_size: float = 5.0,
    config: RegionDiagnosisConfig | None = None,
) -> RegionDiagnosticsReport:
    """Diagnose spatial or explicit route regions without small-sample certainty.

    A query is a strict failure whenever it violates any existing MapDoctor
    localization gate, not only when the upstream localizer sets
    ``success=False``. Explicit assignments take precedence over XYZ cells.
    """

    if not results:
        raise ValueError("region diagnosis requires at least one query")
    if (
        isinstance(cell_size, bool)
        or not isinstance(cell_size, (int, float))
        or not math.isfinite(cell_size)
        or cell_size <= 0.0
    ):
        raise ValueError("cell_size must be finite and > 0")
    cfg = config or RegionDiagnosisConfig()

    result_names = [result.query for result in results]
    if len(result_names) != len(set(result_names)):
        raise ValueError("query names must be unique")

    normalized_assignments: dict[str, str] | None = None
    if assignments is not None:
        normalized_assignments = {}
        for query, region in assignments.items():
            if not isinstance(query, str) or not isinstance(region, str):
                raise ValueError("region assignments require string query and region IDs")
            query_name = query.strip()
            region_id = region.strip()
            if not query_name or not region_id:
                raise ValueError("region assignments require non-empty query and region")
            if query_name in normalized_assignments:
                raise ValueError(f"duplicate normalized region assignment: {query_name}")
            normalized_assignments[query_name] = region_id

    if normalized_assignments is not None:
        unknown_assignments = sorted(set(normalized_assignments) - set(result_names))
        if unknown_assignments:
            raise ValueError(
                "region assignments contain queries outside the benchmark: "
                + ", ".join(unknown_assignments)
            )

    strict_failure_by_query = {
        result.query: bool(result.failures(thresholds)) for result in results
    }
    total_failures = sum(strict_failure_by_query.values())
    alpha, beta = empirical_bayes_prior(
        total_failures,
        len(results),
        strength=cfg.prior_strength,
    )

    buckets: dict[str, list[QueryLocalizationResult]] = defaultdict(list)
    unassigned: list[str] = []
    for result in results:
        region_id = None
        if normalized_assignments is not None:
            region_id = normalized_assignments.get(result.query)
        if region_id is None:
            region_id = _grid_region(result, cell_size)
        if region_id is None:
            unassigned.append(result.query)
        else:
            buckets[region_id].append(result)

    rows: list[RegionDiagnostic] = []
    for region_id, members in buckets.items():
        failures = [
            member for member in members if strict_failure_by_query[member.query]
        ]
        count = len(members)
        failure_count = len(failures)
        interval = wilson_interval(failure_count, count, cfg.confidence)
        posterior = beta_posterior_mean(failure_count, count, alpha, beta)

        if count < cfg.min_samples:
            status = "INSUFFICIENT_EVIDENCE"
        elif (
            failure_count >= cfg.min_failures_for_weak
            and interval.low >= cfg.weak_failure_rate
        ):
            status = "WEAK"
        elif interval.high <= cfg.healthy_failure_rate:
            status = "HEALTHY"
        else:
            status = "UNCERTAIN"

        reason_counts: Counter[str] = Counter()
        for member in failures:
            reason_counts.update(member.failures(thresholds))

        rows.append(
            RegionDiagnostic(
                region_id=region_id,
                samples=count,
                strict_failures=failure_count,
                raw_failure_rate=failure_count / count,
                posterior_failure_rate=posterior,
                failure_rate_ci_low=interval.low,
                failure_rate_ci_high=interval.high,
                status=status,
                evidence_strength=max(0.0, 1.0 - (interval.high - interval.low)),
                failed_queries=tuple(sorted(member.query for member in failures)),
                failure_reason_counts=dict(sorted(reason_counts.items())),
            )
        )

    status_priority = {
        "WEAK": 0,
        "UNCERTAIN": 1,
        "INSUFFICIENT_EVIDENCE": 2,
        "HEALTHY": 3,
    }
    rows.sort(
        key=lambda row: (
            status_priority[row.status],
            -row.posterior_failure_rate,
            row.region_id,
        )
    )

    return RegionDiagnosticsReport(
        total_queries=len(results),
        assigned_queries=sum(len(members) for members in buckets.values()),
        unassigned_queries=tuple(sorted(unassigned)),
        strict_failures=total_failures,
        global_failure_rate=total_failures / len(results),
        prior_alpha=alpha,
        prior_beta=beta,
        cell_size=float(cell_size),
        assignment_source="manifest+xyz" if assignments is not None else "xyz",
        config=cfg,
        regions=tuple(rows),
    )
