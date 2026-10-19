from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from mapdoctor.benchmark import QueryLocalizationResult
from mapdoctor.config import LocalizationThresholds

from ._calibration_fit import _fit_method, _validate_score_labels


@dataclass(frozen=True)
class CalibrationCandidateResult:
    method: str
    available: bool
    out_of_fold_brier: float | None
    out_of_fold_log_loss: float | None
    out_of_fold_adaptive_ece: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CrossFittedCalibrationReport:
    selected_method: str
    num_samples: int
    num_groups: int
    folds: int
    grouping_mode: str
    min_training_samples: int
    raw_brier: float
    raw_log_loss: float
    raw_adaptive_ece: float
    calibrated_oof_brier: float
    calibrated_oof_log_loss: float
    calibrated_oof_adaptive_ece: float
    candidates: tuple[CalibrationCandidateResult, ...]
    final_calibrator: dict[str, object]
    calibrated_risks: dict[str, float]
    out_of_fold_risks: dict[str, float]
    fold_by_query: dict[str, int]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_method": self.selected_method,
            "num_samples": self.num_samples,
            "num_groups": self.num_groups,
            "folds": self.folds,
            "grouping_mode": self.grouping_mode,
            "min_training_samples": self.min_training_samples,
            "raw_metrics": {
                "brier": self.raw_brier,
                "log_loss": self.raw_log_loss,
                "adaptive_ece": self.raw_adaptive_ece,
            },
            "calibrated_out_of_fold_metrics": {
                "brier": self.calibrated_oof_brier,
                "log_loss": self.calibrated_oof_log_loss,
                "adaptive_ece": self.calibrated_oof_adaptive_ece,
            },
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "final_calibrator": self.final_calibrator,
            "calibrated_risks": self.calibrated_risks,
            "out_of_fold_risks": self.out_of_fold_risks,
            "fold_by_query": self.fold_by_query,
            "warnings": list(self.warnings),
            "evaluation_note": (
                "Use out_of_fold_risks for leakage-resistant model comparison on this "
                "dataset. Because the same folds select the winning method, final claims "
                "and operating-point certification still require a separate untouched set. "
                "Use final_calibrator only for future, untouched queries."
            ),
        }


def cross_fit_failure_calibration(
    results: Sequence[QueryLocalizationResult],
    risks: Mapping[str, float],
    thresholds: LocalizationThresholds,
    *,
    groups: Mapping[str, str] | None = None,
    folds: int = 5,
    seed: int = 0,
    method: str = "auto",
    min_samples: int = 20,
    ece_bins: int = 10,
) -> CrossFittedCalibrationReport:
    pairs = _validate_results_and_risks(results, risks)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples < 2
    ):
        raise ValueError("min_samples must be an integer >= 2")
    allowed_methods = {"auto", "identity", "isotonic", "beta"}
    if method not in allowed_methods:
        raise ValueError(f"method must be one of {sorted(allowed_methods)}")

    names = [result.query for result, _ in pairs]
    score_values = np.asarray([risk for _, risk in pairs], dtype=float)
    labels = np.asarray(
        [int(bool(result.failures(thresholds))) for result, _ in pairs],
        dtype=float,
    )
    group_by_query, grouping_mode, warnings = _normalize_groups(names, groups)
    group_values = [group_by_query[name] for name in names]
    unique_groups = sorted(set(group_values))
    if folds > len(unique_groups):
        raise ValueError(
            f"folds={folds} exceeds the number of independent groups "
            f"({len(unique_groups)})"
        )

    fold_by_group = _balanced_group_folds(
        group_values, labels, folds=folds, seed=seed
    )
    fold_indices = np.asarray(
        [fold_by_group[group] for group in group_values], dtype=int
    )
    smallest_train = min(
        int(np.sum(fold_indices != fold)) for fold in range(folds)
    )
    if smallest_train < min_samples:
        raise ValueError(
            "each calibration training split must contain at least "
            f"{min_samples} rows; the smallest contains {smallest_train}. "
            "Reduce folds/min_samples or collect more independent groups."
        )

    methods = ("identity", "isotonic", "beta") if method == "auto" else (method,)
    candidate_rows: list[CalibrationCandidateResult] = []
    predictions_by_method: dict[str, np.ndarray] = {}
    for candidate_method in methods:
        try:
            predictions = np.empty(len(pairs), dtype=float)
            for fold in range(folds):
                train = fold_indices != fold
                test = fold_indices == fold
                calibrator = _fit_method(
                    candidate_method,
                    score_values[train],
                    labels[train],
                    min_samples=min_samples,
                )
                predictions[test] = np.asarray(
                    calibrator.predict(score_values[test]), dtype=float
                )
            metrics = calibration_metrics(predictions, labels, bins=ece_bins)
            predictions_by_method[candidate_method] = predictions
            candidate_rows.append(
                CalibrationCandidateResult(
                    method=candidate_method,
                    available=True,
                    out_of_fold_brier=metrics["brier"],
                    out_of_fold_log_loss=metrics["log_loss"],
                    out_of_fold_adaptive_ece=metrics["adaptive_ece"],
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            candidate_rows.append(
                CalibrationCandidateResult(
                    method=candidate_method,
                    available=False,
                    out_of_fold_brier=None,
                    out_of_fold_log_loss=None,
                    out_of_fold_adaptive_ece=None,
                    error=str(exc),
                )
            )

    available = [row for row in candidate_rows if row.available]
    if not available:
        raise RuntimeError("all requested calibration methods failed")
    complexity_order = {"identity": 0, "beta": 1, "isotonic": 2}
    selected = min(
        available,
        key=lambda row: (
            round(float(row.out_of_fold_brier), 8),
            complexity_order[row.method],
        ),
    )
    selected_predictions = predictions_by_method[selected.method]
    final_calibrator = _fit_method(
        selected.method,
        score_values,
        labels,
        min_samples=min_samples,
    )
    final_predictions = np.asarray(
        final_calibrator.predict(score_values), dtype=float
    )
    raw_metrics = calibration_metrics(score_values, labels, bins=ece_bins)
    oof_metrics = calibration_metrics(selected_predictions, labels, bins=ece_bins)

    if groups is None:
        warnings.append(
            "No session/spatial group assignments were supplied. Query-level folds can "
            "still leak adjacent video frames and overestimate generalization."
        )
    if len(set(labels.tolist())) < 2:
        warnings.append(
            "The calibration set contains only one outcome class; probability estimates "
            "cannot be validated across both successes and failures."
        )

    return CrossFittedCalibrationReport(
        selected_method=selected.method,
        num_samples=len(pairs),
        num_groups=len(unique_groups),
        folds=folds,
        grouping_mode=grouping_mode,
        min_training_samples=smallest_train,
        raw_brier=raw_metrics["brier"],
        raw_log_loss=raw_metrics["log_loss"],
        raw_adaptive_ece=raw_metrics["adaptive_ece"],
        calibrated_oof_brier=oof_metrics["brier"],
        calibrated_oof_log_loss=oof_metrics["log_loss"],
        calibrated_oof_adaptive_ece=oof_metrics["adaptive_ece"],
        candidates=tuple(candidate_rows),
        final_calibrator=final_calibrator.to_dict(),
        calibrated_risks={
            name: float(value)
            for name, value in zip(names, final_predictions, strict=True)
        },
        out_of_fold_risks={
            name: float(value)
            for name, value in zip(names, selected_predictions, strict=True)
        },
        fold_by_query={
            name: int(fold)
            for name, fold in zip(names, fold_indices, strict=True)
        },
        warnings=tuple(warnings),
    )


def spatial_block_groups(
    results: Sequence[QueryLocalizationResult],
    block_size: float,
) -> dict[str, str]:
    if not math.isfinite(block_size) or block_size <= 0:
        raise ValueError("block_size must be finite and > 0")
    groups: dict[str, str] = {}
    for result in results:
        if result.x is None or result.y is None or result.z is None:
            raise ValueError(
                f"{result.query}: x, y and z are required for "
                "spatial-block calibration"
            )
        cell = (
            math.floor(result.x / block_size),
            math.floor(result.y / block_size),
            math.floor(result.z / block_size),
        )
        groups[result.query] = f"spatial:{cell[0]}:{cell[1]}:{cell[2]}"
    return groups


def calibration_metrics(
    probabilities: Sequence[float] | np.ndarray,
    failures: Sequence[int | bool | float] | np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float]:
    p, y = _validate_score_labels(probabilities, failures, min_samples=1)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be an integer >= 1")
    eps = 1e-12
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(
        -np.mean(
            y * np.log(np.clip(p, eps, 1.0))
            + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0))
        )
    )
    chunks = _adaptive_tie_bins(p, bins)
    adaptive_ece = 0.0
    for chunk in chunks:
        adaptive_ece += len(chunk) / len(p) * abs(
            float(np.mean(p[chunk]) - np.mean(y[chunk]))
        )
    return {
        "brier": brier,
        "log_loss": log_loss,
        "adaptive_ece": float(adaptive_ece),
    }


def _adaptive_tie_bins(
    probabilities: np.ndarray,
    bins: int,
) -> list[np.ndarray]:
    """Build approximately equal-mass bins without splitting equal predictions."""
    tied: list[list[int]] = []
    for index in np.argsort(probabilities, kind="mergesort"):
        index = int(index)
        if tied and probabilities[tied[-1][0]] == probabilities[index]:
            tied[-1].append(index)
        else:
            tied.append([index])
    target = max(
        1,
        math.ceil(len(probabilities) / min(bins, len(probabilities))),
    )
    output: list[np.ndarray] = []
    current: list[int] = []
    for tie_group in tied:
        if current and len(current) >= target and len(output) < bins - 1:
            output.append(np.asarray(current, dtype=int))
            current = []
        current.extend(tie_group)
    if current:
        output.append(np.asarray(current, dtype=int))
    return output


def _validate_results_and_risks(
    results: Sequence[QueryLocalizationResult],
    risks: Mapping[str, float],
) -> list[tuple[QueryLocalizationResult, float]]:
    names = [result.query for result in results]
    if not names:
        raise ValueError("calibration requires at least one localization result")
    if len(names) != len(set(names)):
        raise ValueError("localization query names must be unique")
    if any(not isinstance(key, str) for key in risks):
        raise ValueError("risk-score query IDs must be strings")
    missing = sorted(set(names) - set(risks))
    extra = sorted(set(risks) - set(names))
    if missing:
        raise ValueError(
            "risk scores are missing required queries: " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "risk scores contain queries outside the benchmark: "
            + ", ".join(extra)
        )
    output: list[tuple[QueryLocalizationResult, float]] = []
    for result in results:
        raw = risks[result.query]
        if isinstance(raw, bool):
            raise ValueError(f"{result.query}: risk must be numeric")
        try:
            risk = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{result.query}: risk must be numeric") from exc
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(
                f"{result.query}: risk must be finite and in [0, 1]"
            )
        output.append((result, risk))
    return output


def _normalize_groups(
    names: Sequence[str],
    groups: Mapping[str, str] | None,
) -> tuple[dict[str, str], str, list[str]]:
    if groups is None:
        return {name: name for name in names}, "query", []
    if any(not isinstance(key, str) for key in groups):
        raise ValueError("group assignment query IDs must be strings")
    missing = sorted(set(names) - set(groups))
    extra = sorted(set(groups) - set(names))
    if missing:
        raise ValueError(
            "group assignments are missing required queries: " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "group assignments contain queries outside the benchmark: "
            + ", ".join(extra)
        )
    output: dict[str, str] = {}
    for name in names:
        group = groups[name]
        if not isinstance(group, str) or not group.strip():
            raise ValueError(
                f"{name}: calibration group must be a non-empty string"
            )
        output[name] = group.strip()
    return output, "explicit_group", []


def _balanced_group_folds(
    groups: Sequence[str],
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> dict[str, int]:
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    rng = random.Random(seed)
    keys = list(members)
    rng.shuffle(keys)
    global_rate = float(np.mean(labels))
    keys.sort(
        key=lambda group: (
            -len(members[group]),
            -abs(float(np.mean(labels[members[group]])) - global_rate),
        )
    )

    target_n = len(groups) / folds
    target_failures = float(np.sum(labels)) / folds
    fold_n = [0] * folds
    fold_failures = [0.0] * folds
    output: dict[str, int] = {}
    for index, group in enumerate(keys):
        group_n = len(members[group])
        group_failures = float(np.sum(labels[members[group]]))
        if index < folds:
            chosen = index
        else:
            candidates = []
            for fold in range(folds):
                new_n = fold_n[fold] + group_n
                new_failures = fold_failures[fold] + group_failures
                size_cost = ((new_n - target_n) / max(target_n, 1.0)) ** 2
                failure_cost = (
                    (new_failures - target_failures)
                    / max(target_failures, 1.0)
                ) ** 2
                candidates.append(
                    (
                        size_cost + failure_cost,
                        fold_n[fold],
                        fold_failures[fold],
                        fold,
                    )
                )
            chosen = min(candidates)[-1]
        output[group] = chosen
        fold_n[chosen] += group_n
        fold_failures[chosen] += group_failures
    return output
