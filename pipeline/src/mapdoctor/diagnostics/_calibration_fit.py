from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from ._calibration_models import (
    BetaFailureCalibrator,
    FailureCalibrator,
    IdentityFailureCalibrator,
    IsotonicFailureCalibrator,
    _sigmoid,
)


def fit_isotonic_failure_calibrator(
    scores: Sequence[float] | np.ndarray,
    failures: Sequence[int | bool | float] | np.ndarray,
    *,
    min_samples: int = 20,
) -> IsotonicFailureCalibrator:
    values, labels = _validate_score_labels(scores, failures, min_samples=min_samples)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    labels = labels[order]

    blocks: list[dict[str, float]] = []
    for score, failure in zip(values, labels, strict=True):
        if blocks and abs(blocks[-1]["upper"] - float(score)) <= 1e-15:
            blocks[-1]["count"] += 1.0
            blocks[-1]["failures"] += float(failure)
        else:
            blocks.append(
                {
                    "lower": float(score),
                    "upper": float(score),
                    "count": 1.0,
                    "failures": float(failure),
                }
            )
        while (
            len(blocks) >= 2
            and _block_rate(blocks[-2]) > _block_rate(blocks[-1]) + 1e-15
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "count": left["count"] + right["count"],
                    "failures": left["failures"] + right["failures"],
                }
            )

    output_epsilon = 1.0 / (2.0 * (len(values) + 1.0))
    raw_probabilities = np.asarray(
        [_block_rate(block) for block in blocks], dtype=float
    )
    probabilities = output_epsilon + (
        1.0 - 2.0 * output_epsilon
    ) * raw_probabilities
    thresholds = tuple(
        float((blocks[index]["upper"] + blocks[index + 1]["lower"]) / 2.0)
        for index in range(len(blocks) - 1)
    )
    return IsotonicFailureCalibrator(
        block_lower_score=tuple(float(block["lower"]) for block in blocks),
        block_upper_score=tuple(float(block["upper"]) for block in blocks),
        block_failure_probability=tuple(float(value) for value in probabilities),
        block_counts=tuple(int(block["count"]) for block in blocks),
        block_failures=tuple(int(block["failures"]) for block in blocks),
        decision_thresholds=thresholds,
        output_epsilon=float(output_epsilon),
    )


def fit_beta_failure_calibrator(
    scores: Sequence[float] | np.ndarray,
    failures: Sequence[int | bool | float] | np.ndarray,
    *,
    min_samples: int = 20,
    l2_regularization: float = 1e-3,
    max_iterations: int = 500,
) -> BetaFailureCalibrator:
    values, labels = _validate_score_labels(scores, failures, min_samples=min_samples)
    if not math.isfinite(l2_regularization) or l2_regularization < 0:
        raise ValueError("l2_regularization must be finite and >= 0")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be an integer >= 1")

    input_epsilon = 1e-6
    p = np.clip(values, input_epsilon, 1.0 - input_epsilon)
    log_p = np.log(p)
    neg_log_one_minus_p = -np.log1p(-p)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b, c = theta
        z = a * log_p + b * neg_log_one_minus_p + c
        probability = _sigmoid(z)
        eps = 1e-12
        loss = -np.mean(
            labels * np.log(np.clip(probability, eps, 1.0))
            + (1.0 - labels)
            * np.log(np.clip(1.0 - probability, eps, 1.0))
        )
        loss += 0.5 * l2_regularization * (
            (a - 1.0) ** 2 + (b - 1.0) ** 2 + c**2
        )
        error = probability - labels
        gradient = np.asarray(
            [
                np.mean(error * log_p) + l2_regularization * (a - 1.0),
                np.mean(error * neg_log_one_minus_p)
                + l2_regularization * (b - 1.0),
                np.mean(error) + l2_regularization * c,
            ],
            dtype=float,
        )
        return float(loss), gradient

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=np.asarray([1.0, 1.0, 0.0], dtype=float),
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        bounds=((0.0, 100.0), (0.0, 100.0), (-100.0, 100.0)),
        options={"maxiter": max_iterations, "ftol": 1e-12},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"beta calibration optimizer failed: {result.message}")
    output_epsilon = 1.0 / (2.0 * (len(values) + 1.0))
    return BetaFailureCalibrator(
        coefficient_log_score=float(result.x[0]),
        coefficient_log_one_minus_score=float(result.x[1]),
        intercept=float(result.x[2]),
        input_epsilon=input_epsilon,
        output_epsilon=float(output_epsilon),
        l2_regularization=float(l2_regularization),
        optimizer_iterations=int(result.nit),
    )


def _fit_method(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int,
) -> FailureCalibrator:
    if method == "identity":
        return IdentityFailureCalibrator()
    if method == "isotonic":
        return fit_isotonic_failure_calibrator(
            scores, labels, min_samples=min_samples
        )
    if method == "beta":
        return fit_beta_failure_calibrator(scores, labels, min_samples=min_samples)
    raise ValueError(f"unsupported calibration method: {method}")


def _validate_score_labels(
    scores: Sequence[float] | np.ndarray,
    failures: Sequence[int | bool | float] | np.ndarray,
    *,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=float).reshape(-1)
    raw_labels = np.asarray(failures).reshape(-1)
    if len(values) != len(raw_labels):
        raise ValueError("scores and failures must have the same length")
    if len(values) < min_samples:
        raise ValueError(
            f"calibration requires at least {min_samples} samples; "
            f"received {len(values)}"
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("scores must be finite and in [0, 1]")
    labels = np.empty(len(raw_labels), dtype=float)
    for index, value in enumerate(raw_labels):
        if isinstance(value, (bool, np.bool_)):
            labels[index] = float(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("failures must contain only binary values") from exc
        if not math.isfinite(number) or number not in {0.0, 1.0}:
            raise ValueError("failures must contain only binary values")
        labels[index] = number
    return values, labels


def _block_rate(block: Mapping[str, float]) -> float:
    return float(block["failures"] / max(block["count"], 1.0))
