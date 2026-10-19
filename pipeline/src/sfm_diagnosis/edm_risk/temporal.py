from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .edm_loo import EDMQueryResult


@dataclass(frozen=True)
class CausalTemporalState:
    query_id: str
    timestamp: float
    single_frame_success: bool
    window_success: bool
    consecutive_failure_length: int
    history_query_ids: tuple[str, ...]
    recovery_time: float | None


class CausalTemporalEvaluator:
    def __init__(self, *, window_size: int = 3):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)
        self._history: deque[EDMQueryResult] = deque(maxlen=self.window_size)
        self._failure_length = 0
        self._failure_start_time: float | None = None
        self._last_timestamp: float | None = None

    def update(self, result: EDMQueryResult) -> CausalTemporalState:
        if self._last_timestamp is not None and result.timestamp < self._last_timestamp:
            raise ValueError("temporal results must be supplied in causal timestamp order")
        self._last_timestamp = float(result.timestamp)
        recovery = None
        if result.success:
            if self._failure_start_time is not None:
                recovery = float(result.timestamp - self._failure_start_time)
            self._failure_length = 0
            self._failure_start_time = None
        else:
            if self._failure_length == 0:
                self._failure_start_time = float(result.timestamp)
            self._failure_length += 1
        self._history.append(result)
        return CausalTemporalState(
            query_id=result.query_id,
            timestamp=float(result.timestamp),
            single_frame_success=bool(result.success),
            window_success=any(item.success for item in self._history),
            consecutive_failure_length=self._failure_length,
            history_query_ids=tuple(item.query_id for item in self._history),
            recovery_time=recovery,
        )


@dataclass(frozen=True)
class TemporalSummary:
    query_count: int
    single_frame_success_rate: float
    window_success_rate: float
    failure_burst_count: int
    max_consecutive_failures: int
    recovery_time_median: float | None
    consecutive_failure_probability: dict[int, float]
    states: tuple[CausalTemporalState, ...]

    def to_dict(self) -> dict:
        return {
            "query_count": self.query_count,
            "single_frame_success_rate": self.single_frame_success_rate,
            "window_success_rate": self.window_success_rate,
            "failure_burst_count": self.failure_burst_count,
            "max_consecutive_failures": self.max_consecutive_failures,
            "recovery_time_median": self.recovery_time_median,
            "consecutive_failure_probability": {
                str(length): probability
                for length, probability in self.consecutive_failure_probability.items()
            },
            "states": [state.__dict__ for state in self.states],
            "causal_contract": "state at t uses only frames <= t",
        }


def summarize_temporal(
    results: Sequence[EDMQueryResult],
    *,
    window_size: int = 3,
    consecutive_lengths: tuple[int, ...] = (2, 3, 5),
) -> TemporalSummary:
    ordered = sorted(results, key=lambda result: result.timestamp)
    evaluator = CausalTemporalEvaluator(window_size=window_size)
    states = tuple(evaluator.update(result) for result in ordered)
    failures = np.asarray([not result.success for result in ordered], dtype=bool)
    bursts = 0
    previous_failure = False
    for failure in failures:
        if failure and not previous_failure:
            bursts += 1
        previous_failure = bool(failure)
    recoveries = [state.recovery_time for state in states if state.recovery_time is not None]
    probabilities: dict[int, float] = {}
    for length in consecutive_lengths:
        if length < 1:
            raise ValueError("consecutive failure lengths must be >= 1")
        windows = max(len(failures) - length + 1, 0)
        probabilities[length] = (
            float(
                np.mean(
                    [np.all(failures[start : start + length]) for start in range(windows)]
                )
            )
            if windows
            else 0.0
        )
    return TemporalSummary(
        query_count=len(ordered),
        single_frame_success_rate=float(np.mean(~failures)) if len(failures) else 0.0,
        window_success_rate=(
            float(np.mean([state.window_success for state in states])) if states else 0.0
        ),
        failure_burst_count=bursts,
        max_consecutive_failures=max(
            (state.consecutive_failure_length for state in states), default=0
        ),
        recovery_time_median=(
            float(np.median(recoveries)) if recoveries else None
        ),
        consecutive_failure_probability=probabilities,
        states=states,
    )
