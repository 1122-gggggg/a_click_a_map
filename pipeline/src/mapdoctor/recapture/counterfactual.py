from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mapdoctor.comparison import ComparisonResult

from .types import Availability, MetricValue


@dataclass(frozen=True)
class ExistingDataRepairTrial:
    """One non-capture intervention evaluated on frozen weak and stable holdouts."""

    stage: str
    weak_comparison: ComparisonResult
    stable_comparison: ComparisonResult


@dataclass(frozen=True)
class ExistingDataCounterfactualSummary:
    required_stages: tuple[str, ...]
    attempted_stages: tuple[str, ...]
    complete: bool
    best_stage: str | None
    repairability: float
    trial_scores: tuple[tuple[str, float], ...]

    def metric_values(self) -> dict[str, MetricValue]:
        return {
            "existing_data_repairability": MetricValue(
                self.repairability,
                Availability.DERIVED,
                source="recapture.counterfactual",
                formula_version="deficit-closure-v1",
            ),
            "existing_data_counterfactual_complete": MetricValue(
                self.complete,
                Availability.DERIVED,
                source="recapture.counterfactual",
                formula_version="declared-stage-coverage-v1",
            ),
        }


def _repair_score(
    weak: ComparisonResult,
    stable: ComparisonResult,
    *,
    healthy_success_rate: float,
) -> float:
    """Measure how much weak-region success deficit a safe repair closes.

    A candidate that violates either the weak-region regression gates or the
    stable-region regression gates receives zero repairability. This prevents a
    repair that merely trades one failure for another from being treated as a
    viable alternative to recapture.
    """
    if not 0.0 < healthy_success_rate <= 1.0:
        raise ValueError("healthy_success_rate must be in (0, 1]")
    if weak.status != "PASS" or stable.status != "PASS":
        return 0.0
    base = weak.base_strict_success_rate
    candidate = weak.candidate_strict_success_rate
    if base >= healthy_success_rate:
        return 1.0
    deficit = healthy_success_rate - base
    gain = max(0.0, candidate - base)
    return max(0.0, min(1.0, gain / max(deficit, 1e-12)))


def summarize_existing_data_counterfactual(
    trials: Iterable[ExistingDataRepairTrial],
    *,
    required_stages: Iterable[str],
    healthy_success_rate: float = 0.95,
) -> ExistingDataCounterfactualSummary:
    """Aggregate predeclared non-capture repair trials into hard-gate evidence.

    `required_stages` must be chosen before looking at outcomes, based on the
    diagnosed failure funnel. The summary is complete only when every declared
    stage has a frozen weak+stable holdout comparison. The planner may use a
    high score to prefer existing-data repair even when the set is incomplete,
    but a low score cannot authorize recapture until `complete` is true.
    """
    required = tuple(dict.fromkeys(str(stage).strip() for stage in required_stages if str(stage).strip()))
    if not required:
        raise ValueError("required_stages must contain at least one predeclared repair stage")

    trial_list = tuple(trials)
    attempted = tuple(dict.fromkeys(trial.stage.strip() for trial in trial_list if trial.stage.strip()))
    scores: list[tuple[str, float]] = []
    for trial in trial_list:
        stage = trial.stage.strip()
        if not stage:
            raise ValueError("repair trial stage must not be empty")
        scores.append(
            (
                stage,
                _repair_score(
                    trial.weak_comparison,
                    trial.stable_comparison,
                    healthy_success_rate=healthy_success_rate,
                ),
            )
        )

    required_set = set(required)
    attempted_set = set(attempted)
    complete = required_set.issubset(attempted_set)
    if scores:
        best_stage, best_score = max(scores, key=lambda item: item[1])
    else:
        best_stage, best_score = None, 0.0

    return ExistingDataCounterfactualSummary(
        required_stages=required,
        attempted_stages=attempted,
        complete=complete,
        best_stage=best_stage,
        repairability=float(best_score),
        trial_scores=tuple(scores),
    )
