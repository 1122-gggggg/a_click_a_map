from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np


class FailureCalibrator(Protocol):
    method: str

    def predict(self, scores: float | Sequence[float] | np.ndarray) -> float | np.ndarray: ...

    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class IdentityFailureCalibrator:
    method: str = "identity"

    def predict(self, scores: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        values = _validate_prediction_scores(scores)
        output = values.copy()
        if np.ndim(scores) == 0:
            return float(output)
        return output

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method}


@dataclass(frozen=True)
class IsotonicFailureCalibrator:
    block_lower_score: tuple[float, ...]
    block_upper_score: tuple[float, ...]
    block_failure_probability: tuple[float, ...]
    block_counts: tuple[int, ...]
    block_failures: tuple[int, ...]
    decision_thresholds: tuple[float, ...]
    output_epsilon: float
    method: str = "isotonic"

    def predict(self, scores: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        values = _validate_prediction_scores(scores)
        indices = np.searchsorted(
            np.asarray(self.decision_thresholds, dtype=float),
            values,
            side="right",
        )
        probabilities = np.asarray(self.block_failure_probability, dtype=float)[indices]
        if np.ndim(scores) == 0:
            return float(probabilities)
        return probabilities

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "block_lower_score": list(self.block_lower_score),
            "block_upper_score": list(self.block_upper_score),
            "block_failure_probability": list(self.block_failure_probability),
            "block_counts": list(self.block_counts),
            "block_failures": list(self.block_failures),
            "decision_thresholds": list(self.decision_thresholds),
            "output_epsilon": self.output_epsilon,
            "note": (
                "Piecewise-constant binomial isotonic regression. Higher input score is "
                "assumed to mean higher failure risk."
            ),
        }


@dataclass(frozen=True)
class BetaFailureCalibrator:
    coefficient_log_score: float
    coefficient_log_one_minus_score: float
    intercept: float
    input_epsilon: float
    output_epsilon: float
    l2_regularization: float
    optimizer_iterations: int
    method: str = "beta"

    def predict(self, scores: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
        values = _validate_prediction_scores(scores)
        p = np.clip(values, self.input_epsilon, 1.0 - self.input_epsilon)
        z = (
            self.coefficient_log_score * np.log(p)
            - self.coefficient_log_one_minus_score * np.log1p(-p)
            + self.intercept
        )
        probabilities = _sigmoid(z)
        probabilities = self.output_epsilon + (
            1.0 - 2.0 * self.output_epsilon
        ) * probabilities
        if np.ndim(scores) == 0:
            return float(probabilities)
        return probabilities

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def failure_calibrator_from_dict(
    payload: Mapping[str, object],
) -> FailureCalibrator:
    """Restore a serialized calibrator or a full cross-fitting report."""
    if not isinstance(payload, Mapping):
        raise ValueError("calibrator payload must be an object")
    if "final_calibrator" in payload:
        nested = payload["final_calibrator"]
        if not isinstance(nested, Mapping):
            raise ValueError("final_calibrator must be an object")
        payload = nested
    method = payload.get("method")
    if method == "identity":
        return IdentityFailureCalibrator()
    if method == "isotonic":
        calibrator = IsotonicFailureCalibrator(
            block_lower_score=_float_tuple(payload, "block_lower_score"),
            block_upper_score=_float_tuple(payload, "block_upper_score"),
            block_failure_probability=_float_tuple(
                payload, "block_failure_probability"
            ),
            block_counts=_int_tuple(payload, "block_counts"),
            block_failures=_int_tuple(payload, "block_failures"),
            decision_thresholds=_float_tuple(payload, "decision_thresholds"),
            output_epsilon=_finite_float(payload, "output_epsilon"),
        )
        _validate_isotonic_calibrator(calibrator)
        return calibrator
    if method == "beta":
        calibrator = BetaFailureCalibrator(
            coefficient_log_score=_finite_float(
                payload, "coefficient_log_score"
            ),
            coefficient_log_one_minus_score=_finite_float(
                payload, "coefficient_log_one_minus_score"
            ),
            intercept=_finite_float(payload, "intercept"),
            input_epsilon=_finite_float(payload, "input_epsilon"),
            output_epsilon=_finite_float(payload, "output_epsilon"),
            l2_regularization=_finite_float(payload, "l2_regularization"),
            optimizer_iterations=_finite_int(payload, "optimizer_iterations"),
        )
        if (
            calibrator.coefficient_log_score < 0.0
            or calibrator.coefficient_log_one_minus_score < 0.0
            or not 0.0 < calibrator.input_epsilon < 0.5
            or not 0.0 <= calibrator.output_epsilon < 0.5
            or calibrator.l2_regularization < 0.0
            or calibrator.optimizer_iterations < 0
        ):
            raise ValueError("serialized beta calibrator has invalid parameters")
        return calibrator
    raise ValueError("calibrator method must be 'identity', 'isotonic', or 'beta'")


def apply_failure_calibrator(
    calibrator: FailureCalibrator,
    risks: Mapping[str, float],
) -> dict[str, float]:
    """Apply a fitted calibrator to arbitrary future risk-score rows."""
    if not risks:
        raise ValueError("risk scores cannot be empty")
    names: list[str] = []
    values: list[float] = []
    for name, raw_risk in risks.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("risk-score query IDs must be non-empty strings")
        if isinstance(raw_risk, bool):
            raise ValueError(f"{name}: risk must be numeric")
        try:
            risk = float(raw_risk)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: risk must be numeric") from exc
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"{name}: risk must be finite and in [0, 1]")
        names.append(name)
        values.append(risk)
    calibrated = np.asarray(calibrator.predict(values), dtype=float)
    if calibrated.shape != (len(values),) or not np.all(np.isfinite(calibrated)):
        raise RuntimeError("calibrator returned invalid predictions")
    return {
        name: float(value)
        for name, value in zip(names, calibrated, strict=True)
    }


def _validate_prediction_scores(
    scores: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("scores must be finite and in [0, 1]")
    return values


def _finite_float(payload: Mapping[str, object], key: str) -> float:
    if key not in payload or isinstance(payload[key], bool):
        raise ValueError(f"serialized calibrator requires numeric {key}")
    try:
        value = float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"serialized calibrator requires numeric {key}") from exc
    if not math.isfinite(value):
        raise ValueError(f"serialized calibrator requires finite {key}")
    return value


def _finite_int(payload: Mapping[str, object], key: str) -> int:
    value = _finite_float(payload, key)
    if not value.is_integer():
        raise ValueError(f"serialized calibrator requires integer {key}")
    return int(value)


def _float_tuple(payload: Mapping[str, object], key: str) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"serialized calibrator requires an array for {key}")
    values = []
    for item in raw:
        if isinstance(item, bool):
            raise ValueError(f"serialized calibrator requires numeric {key}")
        try:
            value = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"serialized calibrator requires numeric {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"serialized calibrator requires finite {key}")
        values.append(value)
    return tuple(values)


def _int_tuple(payload: Mapping[str, object], key: str) -> tuple[int, ...]:
    values = _float_tuple(payload, key)
    if any(not value.is_integer() for value in values):
        raise ValueError(f"serialized calibrator requires integer {key}")
    return tuple(int(value) for value in values)


def _validate_isotonic_calibrator(
    calibrator: IsotonicFailureCalibrator,
) -> None:
    lengths = {
        len(calibrator.block_lower_score),
        len(calibrator.block_upper_score),
        len(calibrator.block_failure_probability),
        len(calibrator.block_counts),
        len(calibrator.block_failures),
    }
    if len(lengths) != 1 or not calibrator.block_counts:
        raise ValueError("serialized isotonic calibrator has inconsistent blocks")
    if len(calibrator.decision_thresholds) != len(calibrator.block_counts) - 1:
        raise ValueError("serialized isotonic calibrator has invalid thresholds")
    if any(count <= 0 for count in calibrator.block_counts):
        raise ValueError("serialized isotonic calibrator has invalid counts")
    if any(
        failures < 0 or failures > count
        for failures, count in zip(
            calibrator.block_failures, calibrator.block_counts, strict=True
        )
    ):
        raise ValueError("serialized isotonic calibrator has invalid failures")
    if any(
        not 0.0 < probability < 1.0
        for probability in calibrator.block_failure_probability
    ):
        raise ValueError("serialized isotonic probabilities must lie in (0, 1)")
    if any(
        left > right
        for left, right in zip(
            calibrator.block_failure_probability,
            calibrator.block_failure_probability[1:],
        )
    ):
        raise ValueError("serialized isotonic probabilities must be monotone")
    if any(
        lower > upper
        for lower, upper in zip(
            calibrator.block_lower_score, calibrator.block_upper_score, strict=True
        )
    ):
        raise ValueError("serialized isotonic score ranges are invalid")
    if any(
        left >= right
        for left, right in zip(
            calibrator.decision_thresholds, calibrator.decision_thresholds[1:]
        )
    ):
        raise ValueError("serialized isotonic thresholds must be increasing")
    if not 0.0 <= calibrator.output_epsilon < 0.5:
        raise ValueError("serialized isotonic output_epsilon is invalid")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output
