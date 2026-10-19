from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from statistics import median
from typing import Any, Sequence

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import ComparisonThresholds, LocalizationThresholds


@dataclass
class ComparisonResult:
    status: str
    compared_queries: int
    base_strict_success_rate: float
    candidate_strict_success_rate: float
    newly_failed: list[str]
    newly_recovered: list[str]
    common_success_inlier_relative_change_median: float | None
    common_success_reprojection_change_median_px: float | None
    gate_failures: list[str]
    query_deltas: list[dict[str, Any]]
    query_universe_source: str = "union"
    missing_from_base: list[str] = field(default_factory=list)
    missing_from_candidate: list[str] = field(default_factory=list)
    extra_in_base: list[str] = field(default_factory=list)
    extra_in_candidate: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_query_manifest(required_queries: Sequence[str]) -> list[str]:
    if isinstance(required_queries, (str, bytes)):
        raise ValueError("query manifest must be a sequence of strings, not one string")
    names: list[str] = []
    seen: set[str] = set()
    for value in required_queries:
        if not isinstance(value, str):
            raise ValueError("query manifest values must be strings")
        name = value.strip()
        if not name:
            raise ValueError("query manifest contains an empty query name")
        if name in seen:
            raise ValueError(f"query manifest contains duplicate query: {name}")
        seen.add(name)
        names.append(name)
    if not names:
        raise ValueError("query manifest cannot be empty")
    return names


def _index_results(
    results: Sequence[QueryLocalizationResult],
    label: str,
) -> dict[str, QueryLocalizationResult]:
    indexed: dict[str, QueryLocalizationResult] = {}
    for result in results:
        if result.query in indexed:
            raise ValueError(f"{label} benchmark contains duplicate query: {result.query}")
        indexed[result.query] = result
    return indexed


def _rate_exceeds(count: int, total: int, threshold: float) -> bool:
    bound = Fraction(threshold).limit_denominator()
    return count * bound.denominator > bound.numerator * total


def compare_results(
    base: list[QueryLocalizationResult],
    candidate: list[QueryLocalizationResult],
    localization: LocalizationThresholds,
    comparison: ComparisonThresholds,
    required_queries: Sequence[str] | None = None,
) -> ComparisonResult:
    """Compare a candidate on a fail-closed query universe.

    When no immutable manifest is supplied, the union is used. Candidate
    omissions therefore become explicit failures instead of disappearing from
    an intersection-only comparison.
    """

    before = _index_results(base, "base")
    after = _index_results(candidate, "candidate")
    if required_queries is None:
        names = sorted(before.keys() | after.keys())
        source = "union"
    else:
        names = _normalize_query_manifest(required_queries)
        source = "manifest"
    if not names:
        raise ValueError("Base and candidate contain no queries")

    required_set = set(names)
    missing_from_base = sorted(required_set - set(before))
    missing_from_candidate = sorted(required_set - set(after))
    extra_in_base = sorted(set(before) - required_set)
    extra_in_candidate = sorted(set(after) - required_set)

    base_pass = {
        name: name in before and before[name].passes(localization)
        for name in names
    }
    candidate_pass = {
        name: name in after and after[name].passes(localization)
        for name in names
    }
    base_rate = sum(base_pass.values()) / len(names)
    candidate_rate = sum(candidate_pass.values()) / len(names)
    newly_failed = [
        name for name in names if base_pass[name] and not candidate_pass[name]
    ]
    newly_recovered = [
        name
        for name in names
        if name in before
        and name in after
        and not base_pass[name]
        and candidate_pass[name]
    ]
    common_success = [
        name
        for name in names
        if name in before
        and name in after
        and base_pass[name]
        and candidate_pass[name]
    ]

    inlier_changes = [
        (after[name].inliers - before[name].inliers) / before[name].inliers
        for name in common_success
        if before[name].inliers is not None
        and after[name].inliers is not None
        and before[name].inliers > 0
    ]
    reproj_changes = [
        after[name].reproj_p90_px - before[name].reproj_p90_px
        for name in common_success
        if before[name].reproj_p90_px is not None
        and after[name].reproj_p90_px is not None
    ]
    inlier_median = median(inlier_changes) if inlier_changes else None
    reproj_median = median(reproj_changes) if reproj_changes else None

    failures: list[str] = []
    if missing_from_base:
        failures.append(
            "base benchmark is missing required queries: "
            + ", ".join(missing_from_base)
        )
    if missing_from_candidate:
        failures.append(
            "candidate benchmark is missing required queries: "
            + ", ".join(missing_from_candidate)
        )
    if _rate_exceeds(
        sum(base_pass.values()) - sum(candidate_pass.values()),
        len(names),
        comparison.max_success_rate_drop,
    ):
        failures.append("strict success-rate regression exceeds gate")
    if _rate_exceeds(
        len(newly_failed),
        len(names),
        comparison.max_new_failure_rate,
    ):
        failures.append("new-failure rate exceeds gate")
    if (
        inlier_median is not None
        and inlier_median < -comparison.max_common_success_inlier_drop
    ):
        failures.append("common-success median inlier regression exceeds gate")
    if (
        reproj_median is not None
        and reproj_median > comparison.max_common_success_reprojection_increase_px
    ):
        failures.append("common-success reprojection regression exceeds gate")

    deltas: list[dict[str, Any]] = []
    for name in names:
        base_result = before.get(name)
        candidate_result = after.get(name)
        deltas.append(
            {
                "query": name,
                "base_present": base_result is not None,
                "candidate_present": candidate_result is not None,
                "base_pass": base_pass[name],
                "candidate_pass": candidate_pass[name],
                "inliers_base": base_result.inliers if base_result is not None else None,
                "inliers_candidate": (
                    candidate_result.inliers if candidate_result is not None else None
                ),
                "reproj_p90_base_px": (
                    base_result.reproj_p90_px if base_result is not None else None
                ),
                "reproj_p90_candidate_px": (
                    candidate_result.reproj_p90_px
                    if candidate_result is not None
                    else None
                ),
            }
        )

    return ComparisonResult(
        status="PASS" if not failures else "FAIL",
        compared_queries=len(names),
        base_strict_success_rate=base_rate,
        candidate_strict_success_rate=candidate_rate,
        newly_failed=newly_failed,
        newly_recovered=newly_recovered,
        common_success_inlier_relative_change_median=inlier_median,
        common_success_reprojection_change_median_px=reproj_median,
        gate_failures=failures,
        query_deltas=deltas,
        query_universe_source=source,
        missing_from_base=missing_from_base,
        missing_from_candidate=missing_from_candidate,
        extra_in_base=extra_in_base,
        extra_in_candidate=extra_in_candidate,
    )
