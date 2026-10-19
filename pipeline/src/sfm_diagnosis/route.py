from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class RouteAuditConfig:
    """Configuration for route-conditioned localizability analysis.

    ``route_risk`` is a ranking score unless a held-out calibration table is supplied.
    With calibration, ``base_risk`` is an isotonic estimate of failure probability;
    directional and motion penalties remain planning costs rather than calibrated
    probabilities.
    """

    sample_spacing_m: float = 0.5
    max_heatmap_distance_m: float = 2.0
    smoothness_weight: float = 0.25
    task_forward_weight: float = 0.05
    weak_direction_weight: float = 0.10
    sigma_reference_m: float = 1.0
    max_turn_deg_per_m: float | None = 90.0
    turn_violation_weight: float = 2.0
    enter_risk: float = 0.45
    exit_risk: float = 0.35
    min_segment_length_m: float = 1.0
    calibration_min_samples: int = 20
    no_support_risk: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_spacing_m <= 0:
            raise ValueError("sample_spacing_m must be positive")
        if self.max_heatmap_distance_m <= 0:
            raise ValueError("max_heatmap_distance_m must be positive")
        if self.sigma_reference_m <= 0:
            raise ValueError("sigma_reference_m must be positive")
        if not 0.0 <= self.exit_risk <= self.enter_risk <= 1.0:
            raise ValueError("risk thresholds must satisfy 0 <= exit <= enter <= 1")
        if self.min_segment_length_m < 0:
            raise ValueError("min_segment_length_m must be non-negative")
        if self.calibration_min_samples < 2:
            raise ValueError("calibration_min_samples must be at least 2")
        if not 0.0 <= self.no_support_risk <= 1.0:
            raise ValueError("no_support_risk must lie in [0, 1]")
        for name in (
            "smoothness_weight",
            "task_forward_weight",
            "weak_direction_weight",
            "turn_violation_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class MonotonicRiskCalibrator:
    """Piecewise-constant isotonic calibration of failure risk versus health.

    The fitted success probability is constrained to be non-decreasing in health
    using the pool-adjacent-violators algorithm. Failure probability is one minus
    that monotone success estimate.
    """

    block_lower_health: np.ndarray
    block_upper_health: np.ndarray
    block_success_probability: np.ndarray
    block_counts: np.ndarray
    num_samples: int
    brier_score: float
    empirical_failure_rate: float
    reliability: tuple[dict, ...]
    risk_coverage: tuple[dict, ...]

    def predict_failure(self, health: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(health, dtype=float)
        values = np.clip(values, 0.0, 1.0)
        idx = np.searchsorted(self.block_upper_health, values, side="left")
        idx = np.clip(idx, 0, len(self.block_upper_health) - 1)
        risk = 1.0 - self.block_success_probability[idx]
        if np.ndim(health) == 0:
            return float(risk)
        return risk

    def as_dict(self) -> dict:
        return {
            "method": "isotonic_pool_adjacent_violators",
            "num_samples": self.num_samples,
            "num_blocks": int(len(self.block_upper_health)),
            "brier_score": self.brier_score,
            "empirical_failure_rate": self.empirical_failure_rate,
            "reliability": list(self.reliability),
            "risk_coverage": list(self.risk_coverage),
            "note": (
                "Calibrated only for the supplied held-out distribution. Refit after changing "
                "the localizer, camera, map, environment, or decision thresholds."
            ),
        }


@dataclass(frozen=True)
class RouteAuditResult:
    samples: list[dict]
    weak_segments: list[dict]
    summary: dict
    calibration: dict | None = None

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "weak_segments": self.weak_segments,
            "calibration": self.calibration,
        }


@dataclass(frozen=True)
class _RouteSamples:
    positions: np.ndarray
    distance_m: np.ndarray
    tangent_w: np.ndarray
    source_segment: np.ndarray
    input_waypoints: int


@dataclass(frozen=True)
class _Candidate:
    row: dict
    forward_w: np.ndarray
    health: float
    base_risk: float
    weak_alignment: float
    directional_penalty: float
    route_risk: float
    local_cost: float


@dataclass(frozen=True)
class _HeatmapIndex:
    positions: np.ndarray
    groups: tuple[tuple[dict, ...], ...]
    tree: cKDTree

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> _HeatmapIndex:
        if not rows:
            raise ValueError("pose-health table is empty")
        grouped: dict[tuple[float, float, float], list[dict]] = {}
        for line_no, row in enumerate(rows, start=2):
            try:
                position = tuple(round(float(row[k]), 9) for k in ("x", "y", "z"))
                _ = float(row["health_score"])
                forward = np.asarray(
                    [float(row[k]) for k in ("forward_x", "forward_y", "forward_z")],
                    dtype=float,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "pose-health rows require numeric x,y,z,health_score and "
                    f"forward_x,forward_y,forward_z (invalid row near line {line_no})"
                ) from exc
            if np.linalg.norm(forward) < 1e-9:
                raise ValueError(f"pose-health row near line {line_no} has a zero forward vector")
            grouped.setdefault(position, []).append(dict(row))
        keys = sorted(grouped)
        positions = np.asarray(keys, dtype=float).reshape(-1, 3)
        groups = tuple(tuple(grouped[key]) for key in keys)
        return cls(positions=positions, groups=groups, tree=cKDTree(positions))

    def query(self, position: np.ndarray, max_distance_m: float) -> tuple[float, tuple[dict, ...]]:
        distance, index = self.tree.query(np.asarray(position, dtype=float), k=1)
        distance = float(distance)
        if not np.isfinite(distance) or distance > max_distance_m:
            return distance, ()
        return distance, self.groups[int(index)]


def audit_route(
    pose_health: str | Path | Sequence[dict],
    route: str | Path | Sequence[dict] | np.ndarray,
    *,
    config: RouteAuditConfig | None = None,
    calibration: str | Path | Sequence[dict] | None = None,
) -> RouteAuditResult:
    """Audit localizability along a deployment route and select smooth viewpoints.

    Parameters
    ----------
    pose_health:
        Detailed ``pose_health.csv`` rows produced by ``sfm-diagnosis heatmap``.
    route:
        Polyline waypoints containing x, y and z.
    calibration:
        Optional held-out rows containing ``health_score`` and binary ``success``.
    """

    cfg = config or RouteAuditConfig()
    heat_rows = _coerce_rows(pose_health)
    route_rows = _coerce_route_rows(route)
    samples = resample_route(route_rows, spacing_m=cfg.sample_spacing_m)
    index = _HeatmapIndex.from_rows(heat_rows)

    calibrator = None
    if calibration is not None:
        calibrator = fit_monotonic_risk_calibrator(
            _coerce_rows(calibration),
            min_samples=cfg.calibration_min_samples,
        )

    candidate_sets: list[list[_Candidate]] = []
    support_distance: list[float] = []
    support_position: list[np.ndarray | None] = []
    for position, tangent in zip(samples.positions, samples.tangent_w, strict=True):
        distance, rows = index.query(position, cfg.max_heatmap_distance_m)
        support_distance.append(distance)
        if not rows:
            candidate_sets.append([])
            support_position.append(None)
            continue
        nearest_position = index.positions[int(index.tree.query(position, k=1)[1])]
        support_position.append(nearest_position)
        candidates = [
            _make_candidate(row, tangent, cfg=cfg, calibrator=calibrator) for row in rows
        ]
        candidate_sets.append(candidates)

    selected = _select_orientation_sequence(
        candidate_sets,
        samples.distance_m,
        cfg=cfg,
    )
    rows = _build_sample_rows(
        samples,
        candidate_sets,
        selected,
        support_distance,
        support_position,
        cfg=cfg,
        calibrated=calibrator is not None,
    )
    sample_weights = _sample_path_weights(samples.distance_m)
    segments = _weak_segments(rows, sample_weights, cfg=cfg)
    summary = _summarize_route(
        rows,
        segments,
        sample_weights,
        samples,
        calibrated=calibrator is not None,
        cfg=cfg,
    )
    calibration_payload = calibrator.as_dict() if calibrator is not None else None
    return RouteAuditResult(
        samples=rows,
        weak_segments=segments,
        summary=summary,
        calibration=calibration_payload,
    )


def fit_monotonic_risk_calibrator(
    rows: Sequence[dict],
    *,
    min_samples: int = 20,
    beta_prior_success: float = 1.0,
    beta_prior_failure: float = 1.0,
) -> MonotonicRiskCalibrator:
    """Fit a monotone health-to-failure mapping with PAV isotonic regression."""

    if beta_prior_success < 0 or beta_prior_failure < 0:
        raise ValueError("beta prior pseudo-counts must be non-negative")
    parsed: list[tuple[float, float]] = []
    for line_no, row in enumerate(rows, start=2):
        try:
            health = float(row["health_score"])
            success = _parse_success(row["success"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "calibration rows require numeric health_score and binary success "
                f"(invalid row near line {line_no})"
            ) from exc
        if not np.isfinite(health):
            raise ValueError(f"calibration health_score near line {line_no} is not finite")
        parsed.append((float(np.clip(health, 0.0, 1.0)), success))
    if len(parsed) < min_samples:
        raise ValueError(
            f"calibration requires at least {min_samples} rows; received {len(parsed)}"
        )

    parsed.sort(key=lambda item: item[0])
    grouped: list[dict] = []
    for health, success in parsed:
        if grouped and abs(grouped[-1]["health"] - health) <= 1e-12:
            grouped[-1]["raw_count"] += 1.0
            grouped[-1]["raw_success"] += success
            grouped[-1]["weight"] += 1.0
            grouped[-1]["success_sum"] += success
        else:
            grouped.append(
                {
                    "health": health,
                    "lower": health,
                    "upper": health,
                    "raw_count": 1.0,
                    "raw_success": success,
                    "weight": 1.0,
                    "success_sum": success,
                }
            )

    # Add a symmetric Beta prior to every unique-health group before PAV. This
    # regularizes tiny groups while preserving the monotonic optimization.
    blocks: list[dict] = []
    for group in grouped:
        block = dict(group)
        block["weight"] = group["weight"] + beta_prior_success + beta_prior_failure
        block["success_sum"] = group["success_sum"] + beta_prior_success
        blocks.append(block)
        while len(blocks) >= 2 and _block_mean(blocks[-2]) > _block_mean(blocks[-1]) + 1e-15:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                {
                    "health": left["health"],
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "raw_count": left["raw_count"] + right["raw_count"],
                    "raw_success": left["raw_success"] + right["raw_success"],
                    "weight": left["weight"] + right["weight"],
                    "success_sum": left["success_sum"] + right["success_sum"],
                }
            )

    lower = np.asarray([block["lower"] for block in blocks], dtype=float)
    upper = np.asarray([block["upper"] for block in blocks], dtype=float)
    success_probability = np.asarray([_block_mean(block) for block in blocks], dtype=float)
    counts = np.asarray([block["raw_count"] for block in blocks], dtype=int)

    provisional = MonotonicRiskCalibrator(
        block_lower_health=lower,
        block_upper_health=upper,
        block_success_probability=success_probability,
        block_counts=counts,
        num_samples=len(parsed),
        brier_score=0.0,
        empirical_failure_rate=0.0,
        reliability=(),
        risk_coverage=(),
    )
    health_values = np.asarray([item[0] for item in parsed], dtype=float)
    success_values = np.asarray([item[1] for item in parsed], dtype=float)
    failure_values = 1.0 - success_values
    predicted = np.asarray(provisional.predict_failure(health_values), dtype=float)
    brier = float(np.mean((predicted - failure_values) ** 2))

    reliability = []
    for block, predicted_success in zip(blocks, success_probability, strict=True):
        raw_count = int(block["raw_count"])
        raw_failure = 1.0 - block["raw_success"] / max(block["raw_count"], 1.0)
        reliability.append(
            {
                "health_min": float(block["lower"]),
                "health_max": float(block["upper"]),
                "count": raw_count,
                "empirical_failure_rate": float(raw_failure),
                "predicted_failure_probability": float(1.0 - predicted_success),
            }
        )

    risk_coverage = _risk_coverage_curve(predicted, failure_values)
    return MonotonicRiskCalibrator(
        block_lower_health=lower,
        block_upper_health=upper,
        block_success_probability=success_probability,
        block_counts=counts,
        num_samples=len(parsed),
        brier_score=brier,
        empirical_failure_rate=float(np.mean(failure_values)),
        reliability=tuple(reliability),
        risk_coverage=tuple(risk_coverage),
    )


def resample_route(route_rows: Sequence[dict], *, spacing_m: float) -> _RouteSamples:
    """Resample a 3D waypoint polyline at approximately uniform arc length."""

    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    points = []
    for line_no, row in enumerate(route_rows, start=2):
        try:
            point = np.asarray([float(row[k]) for k in ("x", "y", "z")], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"route rows require numeric x,y,z (invalid row near line {line_no})"
            ) from exc
        if not np.all(np.isfinite(point)):
            raise ValueError(f"route point near line {line_no} is not finite")
        if not points or np.linalg.norm(point - points[-1]) > 1e-9:
            points.append(point)
    if len(points) < 2:
        raise ValueError("route must contain at least two distinct waypoints")

    p = np.asarray(points, dtype=float)
    segment_vectors = np.diff(p, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    distances = np.arange(0.0, total, spacing_m, dtype=float)
    if len(distances) == 0 or not np.isclose(distances[-1], total):
        distances = np.concatenate((distances, [total]))

    positions = np.zeros((len(distances), 3), dtype=float)
    source_segment = np.zeros(len(distances), dtype=int)
    for i, distance in enumerate(distances):
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(p) - 2)
        source_segment[i] = segment
        local = distance - cumulative[segment]
        fraction = local / max(segment_lengths[segment], 1e-12)
        positions[i] = p[segment] + fraction * segment_vectors[segment]

    tangent = np.zeros_like(positions)
    tangent[0] = positions[1] - positions[0]
    tangent[-1] = positions[-1] - positions[-2]
    if len(positions) > 2:
        tangent[1:-1] = positions[2:] - positions[:-2]
    norm = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(norm[:, None], 1e-12)
    return _RouteSamples(
        positions=positions,
        distance_m=distances,
        tangent_w=tangent,
        source_segment=source_segment,
        input_waypoints=len(points),
    )


def save_route_audit(output_dir: str | Path, result: RouteAuditResult) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "route_samples.csv", result.samples)
    _write_csv(out / "weak_segments.csv", result.weak_segments)
    _write_json(out / "route_summary.json", result.as_dict())
    _maybe_plot(out / "route_plan.html", result.samples)


def _make_candidate(
    row: dict,
    tangent_w: np.ndarray,
    *,
    cfg: RouteAuditConfig,
    calibrator: MonotonicRiskCalibrator | None,
) -> _Candidate:
    forward = _unit(
        np.asarray([float(row[k]) for k in ("forward_x", "forward_y", "forward_z")])
    )
    health = float(np.clip(float(row["health_score"]), 0.0, 1.0))
    base_risk = (
        float(calibrator.predict_failure(health)) if calibrator is not None else 1.0 - health
    )

    weak_direction = _optional_vector(
        row,
        (
            "fim_weakest_translation_world_x",
            "fim_weakest_translation_world_y",
            "fim_weakest_translation_world_z",
        ),
    )
    weak_alignment = 0.0
    directional_penalty = 0.0
    if weak_direction is not None:
        weak_alignment = abs(float(np.dot(_unit(tangent_w), _unit(weak_direction))))
        weak_fraction = float(
            np.clip(_optional_float(row, "fim_weakest_translation_fraction", 0.0), 0.0, 1.0)
        )
        sigma = _optional_float(row, "fim_sigma_translation_worst_m", cfg.sigma_reference_m)
        sigma_factor = float(np.clip(sigma / cfg.sigma_reference_m, 0.0, 1.0))
        directional_penalty = (
            cfg.weak_direction_weight * weak_alignment * weak_fraction * sigma_factor
        )

    route_risk = float(np.clip(base_risk + directional_penalty, 0.0, 1.0))
    task_angle = _angle_rad(forward, tangent_w)
    local_cost = route_risk + cfg.task_forward_weight * task_angle / np.pi
    return _Candidate(
        row=row,
        forward_w=forward,
        health=health,
        base_risk=base_risk,
        weak_alignment=weak_alignment,
        directional_penalty=directional_penalty,
        route_risk=route_risk,
        local_cost=float(local_cost),
    )


def _select_orientation_sequence(
    candidate_sets: Sequence[Sequence[_Candidate]],
    distance_m: np.ndarray,
    *,
    cfg: RouteAuditConfig,
) -> list[int | None]:
    selected: list[int | None] = [None] * len(candidate_sets)
    start = 0
    while start < len(candidate_sets):
        while start < len(candidate_sets) and not candidate_sets[start]:
            start += 1
        if start >= len(candidate_sets):
            break
        end = start
        while end + 1 < len(candidate_sets) and candidate_sets[end + 1]:
            end += 1
        block = candidate_sets[start : end + 1]
        chosen = _dynamic_program(block, distance_m[start : end + 1], cfg=cfg)
        for offset, value in enumerate(chosen):
            selected[start + offset] = value
        start = end + 1
    return selected


def _dynamic_program(
    candidate_sets: Sequence[Sequence[_Candidate]],
    distance_m: np.ndarray,
    *,
    cfg: RouteAuditConfig,
) -> list[int]:
    costs = np.asarray([candidate.local_cost for candidate in candidate_sets[0]], dtype=float)
    back: list[np.ndarray] = []
    for i in range(1, len(candidate_sets)):
        previous = candidate_sets[i - 1]
        current = candidate_sets[i]
        next_cost = np.full(len(current), np.inf, dtype=float)
        next_back = np.full(len(current), -1, dtype=int)
        ds = max(float(distance_m[i] - distance_m[i - 1]), 1e-9)
        max_turn = None
        if cfg.max_turn_deg_per_m is not None:
            max_turn = np.radians(max(cfg.max_turn_deg_per_m, 0.0) * ds)
        for k, candidate in enumerate(current):
            for j, prior in enumerate(previous):
                turn = _angle_rad(prior.forward_w, candidate.forward_w)
                transition = cfg.smoothness_weight * (turn / np.pi) ** 2
                if max_turn is not None and turn > max_turn:
                    excess = (turn - max_turn) / np.pi
                    transition += cfg.turn_violation_weight * excess**2
                value = costs[j] + transition + candidate.local_cost
                if value < next_cost[k]:
                    next_cost[k] = value
                    next_back[k] = j
        costs = next_cost
        back.append(next_back)

    chosen = [0] * len(candidate_sets)
    chosen[-1] = int(np.argmin(costs))
    for i in range(len(candidate_sets) - 1, 0, -1):
        chosen[i - 1] = int(back[i - 1][chosen[i]])
    return chosen


def _build_sample_rows(
    samples: _RouteSamples,
    candidate_sets: Sequence[Sequence[_Candidate]],
    selected: Sequence[int | None],
    support_distance: Sequence[float],
    support_position: Sequence[np.ndarray | None],
    *,
    cfg: RouteAuditConfig,
    calibrated: bool,
) -> list[dict]:
    rows: list[dict] = []
    previous_forward = None
    previous_distance = None
    for i, (position, tangent, candidates, selected_index) in enumerate(
        zip(samples.positions, samples.tangent_w, candidate_sets, selected, strict=True)
    ):
        base = {
            "route_index": i,
            "route_distance_m": float(samples.distance_m[i]),
            "source_segment": int(samples.source_segment[i]),
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
            "tangent_x": float(tangent[0]),
            "tangent_y": float(tangent[1]),
            "tangent_z": float(tangent[2]),
            "heatmap_support_distance_m": (
                float(support_distance[i]) if np.isfinite(support_distance[i]) else None
            ),
            "risk_source": (
                "isotonic_calibration_plus_planning_penalties"
                if calibrated
                else "one_minus_health_plus_planning_penalties"
            ),
        }
        support = support_position[i]
        if support is not None:
            base.update(
                {
                    "heatmap_x": float(support[0]),
                    "heatmap_y": float(support[1]),
                    "heatmap_z": float(support[2]),
                }
            )
        else:
            base.update({"heatmap_x": None, "heatmap_y": None, "heatmap_z": None})

        if selected_index is None or not candidates:
            row = {
                **base,
                "supported": False,
                "selected_health": None,
                "base_failure_risk": cfg.no_support_risk,
                "predicted_failure_probability": None,
                "weak_direction_alignment": None,
                "directional_risk_penalty": 0.0,
                "route_risk": cfg.no_support_risk,
                "best_available_risk": cfg.no_support_risk,
                "orientation_regret": 0.0,
                "forward_x": None,
                "forward_y": None,
                "forward_z": None,
                "turn_deg": None,
                "turn_deg_per_m": None,
                "turn_limit_violation": False,
                "primary": "NO_HEATMAP_SUPPORT",
                "codes": "NO_HEATMAP_SUPPORT",
                "limitation": "NO_HEATMAP_SUPPORT",
            }
            previous_forward = None
            previous_distance = None
            rows.append(row)
            continue

        candidate = candidates[int(selected_index)]
        best = min(candidates, key=lambda item: item.route_risk)
        turn_deg = None
        turn_rate = None
        turn_violation = False
        if previous_forward is not None and previous_distance is not None:
            turn_deg = float(np.degrees(_angle_rad(previous_forward, candidate.forward_w)))
            ds = max(float(samples.distance_m[i] - previous_distance), 1e-9)
            turn_rate = turn_deg / ds
            if cfg.max_turn_deg_per_m is not None:
                turn_violation = turn_rate > cfg.max_turn_deg_per_m + 1e-9
        previous_forward = candidate.forward_w
        previous_distance = float(samples.distance_m[i])

        limitation = _limitation(
            candidate.route_risk,
            best.route_risk,
            cfg=cfg,
        )
        row = {
            **base,
            "supported": True,
            "selected_health": candidate.health,
            "base_failure_risk": candidate.base_risk,
            "predicted_failure_probability": candidate.base_risk if calibrated else None,
            "weak_direction_alignment": candidate.weak_alignment,
            "directional_risk_penalty": candidate.directional_penalty,
            "route_risk": candidate.route_risk,
            "best_available_risk": best.route_risk,
            "orientation_regret": max(candidate.route_risk - best.route_risk, 0.0),
            "forward_x": float(candidate.forward_w[0]),
            "forward_y": float(candidate.forward_w[1]),
            "forward_z": float(candidate.forward_w[2]),
            "turn_deg": turn_deg,
            "turn_deg_per_m": turn_rate,
            "turn_limit_violation": turn_violation,
            "primary": str(candidate.row.get("primary", "UNKNOWN") or "UNKNOWN"),
            "codes": str(candidate.row.get("codes", "") or ""),
            "limitation": limitation,
        }
        rows.append(row)
    return rows


def _weak_segments(
    rows: Sequence[dict],
    sample_weights: np.ndarray,
    *,
    cfg: RouteAuditConfig,
) -> list[dict]:
    ranges: list[tuple[int, int]] = []
    active_start = None
    for i, row in enumerate(rows):
        risk = float(row["route_risk"])
        if active_start is None and risk >= cfg.enter_risk:
            active_start = i
        elif active_start is not None and risk <= cfg.exit_risk:
            end = max(i - 1, active_start)
            ranges.append((active_start, end))
            active_start = None
    if active_start is not None:
        ranges.append((active_start, len(rows) - 1))

    segments: list[dict] = []
    for start, end in ranges:
        weights = sample_weights[start : end + 1]
        length = float(np.sum(weights))
        if length + 1e-9 < cfg.min_segment_length_m:
            continue
        subset = rows[start : end + 1]
        risks = np.asarray([float(row["route_risk"]) for row in subset], dtype=float)
        weighted_mean = float(np.sum(weights * risks) / max(np.sum(weights), 1e-12))
        worst_local = int(np.argmax(risks))
        worst = subset[worst_local]
        limitation_counts = Counter(str(row["limitation"]) for row in subset)
        code_counts = Counter()
        primary_counts = Counter()
        for row in subset:
            primary_counts[str(row["primary"])] += 1
            for code in str(row.get("codes", "")).split(";"):
                code = code.strip()
                if code:
                    code_counts[code] += 1
        dominant_codes = [code for code, _ in code_counts.most_common(4)]
        dominant_primary = primary_counts.most_common(1)[0][0] if primary_counts else "UNKNOWN"
        excess = np.maximum(risks - cfg.enter_risk, 0.0)
        integrated_excess = float(np.sum(weights * excess))
        unsupported_fraction = limitation_counts["NO_HEATMAP_SUPPORT"] / len(subset)
        map_limited_fraction = limitation_counts["MAP_LIMITED"] / len(subset)
        orientation_limited_fraction = limitation_counts["ORIENTATION_LIMITED"] / len(subset)
        priority = (
            integrated_excess
            + 0.50 * length * unsupported_fraction
            + 0.25 * length * map_limited_fraction
            + 0.25 * float(np.max(risks))
        )
        repair_class, action = _repair_action(
            dominant_codes,
            dominant_primary,
            unsupported_fraction=unsupported_fraction,
            map_limited_fraction=map_limited_fraction,
            orientation_limited_fraction=orientation_limited_fraction,
        )
        segments.append(
            {
                "segment_id": len(segments),
                "start_index": start,
                "end_index": end,
                "start_distance_m": float(rows[start]["route_distance_m"]),
                "end_distance_m": float(rows[end]["route_distance_m"]),
                "length_m": length,
                "mean_risk": weighted_mean,
                "max_risk": float(np.max(risks)),
                "integrated_excess_risk_m": integrated_excess,
                "priority_score": float(priority),
                "unsupported_fraction": float(unsupported_fraction),
                "map_limited_fraction": float(map_limited_fraction),
                "orientation_limited_fraction": float(orientation_limited_fraction),
                "dominant_primary": dominant_primary,
                "dominant_codes": ";".join(dominant_codes),
                "worst_route_index": int(worst["route_index"]),
                "worst_x": float(worst["x"]),
                "worst_y": float(worst["y"]),
                "worst_z": float(worst["z"]),
                "repair_class": repair_class,
                "recommended_action": action,
            }
        )
    segments.sort(key=lambda segment: segment["priority_score"], reverse=True)
    for rank, segment in enumerate(segments, start=1):
        segment["priority_rank"] = rank
    return segments


def _summarize_route(
    rows: Sequence[dict],
    segments: Sequence[dict],
    sample_weights: np.ndarray,
    samples: _RouteSamples,
    *,
    calibrated: bool,
    cfg: RouteAuditConfig,
) -> dict:
    risk = np.asarray([float(row["route_risk"]) for row in rows], dtype=float)
    total_length = float(samples.distance_m[-1])
    normalization = max(float(np.sum(sample_weights)), 1e-12)

    def weighted_fraction(predicate) -> float:
        mask = np.asarray([bool(predicate(row)) for row in rows], dtype=float)
        return float(np.sum(sample_weights * mask) / normalization)

    supported_fraction = weighted_fraction(lambda row: bool(row["supported"]))
    robust_fraction = weighted_fraction(lambda row: float(row["route_risk"]) < cfg.enter_risk)
    weak_length = float(sum(float(segment["length_m"]) for segment in segments))
    turns = np.asarray(
        [float(row["turn_deg"]) for row in rows if row["turn_deg"] is not None],
        dtype=float,
    )
    turn_rates = np.asarray(
        [float(row["turn_deg_per_m"]) for row in rows if row["turn_deg_per_m"] is not None],
        dtype=float,
    )
    summary = {
        "schema_version": 1,
        "input_waypoints": samples.input_waypoints,
        "route_samples": len(rows),
        "path_length_m": total_length,
        "sample_spacing_m": cfg.sample_spacing_m,
        "supported_fraction_by_length": supported_fraction,
        "robust_fraction_by_length": robust_fraction,
        "weak_fraction_by_length": min(weak_length / max(total_length, 1e-12), 1.0),
        "mean_route_risk": float(np.sum(sample_weights * risk) / normalization),
        "p90_route_risk": float(np.quantile(risk, 0.90)),
        "max_route_risk": float(np.max(risk)),
        "num_weak_segments": len(segments),
        "weak_length_m": weak_length,
        "unsupported_length_m": float(
            np.sum(
                sample_weights
                * np.asarray(
                    [row["limitation"] == "NO_HEATMAP_SUPPORT" for row in rows], dtype=float
                )
            )
        ),
        "map_limited_length_m": float(
            np.sum(
                sample_weights
                * np.asarray([row["limitation"] == "MAP_LIMITED" for row in rows], dtype=float)
            )
        ),
        "orientation_limited_length_m": float(
            np.sum(
                sample_weights
                * np.asarray(
                    [row["limitation"] == "ORIENTATION_LIMITED" for row in rows], dtype=float
                )
            )
        ),
        "turn_limit_violations": int(sum(bool(row["turn_limit_violation"]) for row in rows)),
        "mean_turn_deg": float(np.mean(turns)) if len(turns) else 0.0,
        "p95_turn_deg_per_m": float(np.quantile(turn_rates, 0.95)) if len(turn_rates) else 0.0,
        "risk_interpretation": (
            "base_failure_risk is calibrated probability; route_risk adds uncalibrated "
            "FIM-direction and planning penalties"
            if calibrated
            else "route_risk is an interpretable ranking score, not a failure probability"
        ),
        "top_weak_segments": list(segments[:10]),
        "limitations": [
            "Nearest heatmap-position lookup does not render the exact route pose.",
            "FIM direction penalties are geometric proxies and require held-out validation.",
            "No collision, geofence, dynamics, or gimbal feasibility is modeled.",
        ],
    }
    if calibrated:
        probability = np.asarray(
            [
                float(row["predicted_failure_probability"])
                for row in rows
                if row["predicted_failure_probability"] is not None
            ],
            dtype=float,
        )
        summary["mean_predicted_failure_probability_supported"] = (
            float(np.mean(probability)) if len(probability) else None
        )
    return summary


def _limitation(selected_risk: float, best_risk: float, *, cfg: RouteAuditConfig) -> str:
    if best_risk >= cfg.enter_risk:
        return "MAP_LIMITED"
    if selected_risk >= cfg.enter_risk and selected_risk - best_risk >= 0.05:
        return "ORIENTATION_LIMITED"
    if selected_risk >= cfg.enter_risk:
        return "MARGINAL"
    return "HEALTHY"


def _repair_action(
    dominant_codes: Sequence[str],
    dominant_primary: str,
    *,
    unsupported_fraction: float,
    map_limited_fraction: float,
    orientation_limited_fraction: float,
) -> tuple[str, str]:
    if unsupported_fraction > 0.25:
        return (
            "EVIDENCE_GAP",
            "Expand the diagnostic heatmap/model bounds first; do not infer map weakness from "
            "unsupported route samples.",
        )
    if orientation_limited_fraction > map_limited_fraction and orientation_limited_fraction > 0.25:
        return (
            "ORIENTATION_REPLAN",
            "Use the selected smooth camera headings, relax the task-facing constraint, or "
            "change the path so the camera can observe stronger mapped directions.",
        )
    codes = set(dominant_codes) | {dominant_primary}
    if "PERCEPTUAL_ALIASING_SUSPECTED" in codes or "REFERENCE_DISAGREEMENT" in codes:
        return (
            "DISAMBIGUATION",
            "Add discriminative references or sequence anchors and retain per-reference pose "
            "hypotheses; do not average incompatible modes.",
        )
    if "RETRIEVAL_WEAK" in codes:
        return (
            "LOCALIZER_RETRIEVAL",
            "Improve retrieval/index coverage or add appearance-diverse references before "
            "changing geometry.",
        )
    if "MATCHING_WEAK" in codes or "PNP_DEGENERATE" in codes or "LANDMARK_MATCHABILITY_WEAK" in codes:
        return (
            "LOCALIZER_MATCHING_PNP",
            "Run targeted stronger matching and inspect calibration, 2D-3D spread, positive "
            "depth, and multi-reference pose consistency.",
        )
    if "OBSERVATION_SCALE_WEAK" in codes:
        return (
            "STANDOFF_OR_ZOOM_CHANGE",
            "Change camera standoff, zoom, or altitude so the query range matches mapped "
            "observation scale; do not rebuild geometry first.",
        )
    if "VIEW_COVERAGE_WEAK" in codes:
        return (
            "APPEARANCE_REFERENCE_CAPTURE",
            "Capture overlapping references from the missing approach direction; preserve "
            "existing geometry when FIM is already healthy.",
        )
    if (
        "GEOMETRY_WEAK" in codes
        or "QUERY_PARALLAX_WEAK" in codes
        or "DATA_SPARSE" in codes
        or map_limited_fraction > 0.25
    ):
        return (
            "GEOMETRY_RECAPTURE",
            "Capture lateral/oblique bridge views with useful baseline and broad image support, "
            "then re-triangulate and run anchor-constrained local bundle adjustment.",
        )
    if "ILLUMINATION_WEAK" in codes:
        return (
            "APPEARANCE_CONDITION_CAPTURE",
            "Add references under the deployment illumination or schedule the route for a "
            "better-supported lighting condition.",
        )
    return (
        "REVALIDATE",
        "Inspect the exported component metrics and run unchanged held-out localization queries "
        "before choosing recapture or localizer changes.",
    )


def _sample_path_weights(distance_m: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance_m, dtype=float)
    if len(distance) == 1:
        return np.ones(1, dtype=float)
    delta = np.diff(distance)
    weights = np.zeros(len(distance), dtype=float)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    if len(distance) > 2:
        weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return weights


def _risk_coverage_curve(predicted_risk: np.ndarray, failures: np.ndarray) -> list[dict]:
    order = np.argsort(predicted_risk)
    predicted = predicted_risk[order]
    observed = failures[order]
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        count = max(1, int(np.ceil(len(observed) * coverage)))
        rows.append(
            {
                "coverage": float(count / len(observed)),
                "selective_failure_rate": float(np.mean(observed[:count])),
                "max_accepted_predicted_risk": float(predicted[count - 1]),
                "accepted_samples": count,
            }
        )
    return rows


def _coerce_rows(source: str | Path | Sequence[dict]) -> list[dict]:
    if isinstance(source, (str, Path)):
        return _read_rows(source)
    return [dict(row) for row in source]


def _coerce_route_rows(source: str | Path | Sequence[dict] | np.ndarray) -> list[dict]:
    if isinstance(source, np.ndarray):
        array = np.asarray(source, dtype=float).reshape(-1, 3)
        return [{"x": row[0], "y": row[1], "z": row[2]} for row in array]
    return _coerce_rows(source)


def _read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("rows", "route", "samples", "queries"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise ValueError(f"JSON table {p} must be a list or contain rows/route/samples")
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _maybe_plot(path: Path, rows: Sequence[dict]) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return
    if not rows:
        return
    hover = [
        (
            f"s={row['route_distance_m']:.2f} m<br>risk={row['route_risk']:.3f}"
            f"<br>{row['limitation']}<br>{row['codes']}"
        )
        for row in rows
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[row["x"] for row in rows],
                y=[row["y"] for row in rows],
                z=[row["z"] for row in rows],
                mode="lines+markers",
                marker={
                    "size": 4,
                    "color": [row["route_risk"] for row in rows],
                    "colorscale": "RdYlGn_r",
                    "cmin": 0.0,
                    "cmax": 1.0,
                    "colorbar": {"title": "route risk"},
                },
                line={"width": 3},
                text=hover,
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(title="Route-conditioned localization risk")
    fig.write_html(str(path), include_plotlyjs="cdn")


def _optional_vector(row: dict, keys: tuple[str, str, str]) -> np.ndarray | None:
    values = []
    for key in keys:
        raw = row.get(key)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            return None
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)) or np.linalg.norm(vector) < 1e-9:
        return None
    return vector


def _optional_float(row: dict, key: str, default: float) -> float:
    raw = row.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _parse_success(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "success", "ok", "pass", "1"}:
        return 1.0
    if text in {"false", "no", "n", "failure", "fail", "lost", "0"}:
        return 0.0
    number = float(value)
    if number not in {0.0, 1.0}:
        raise ValueError("success must be binary")
    return number


def _block_mean(block: dict) -> float:
    return float(block["success_sum"] / max(block["weight"], 1e-12))


def _angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    aa = _unit(a)
    bb = _unit(b)
    return float(np.arccos(np.clip(np.dot(aa, bb), -1.0, 1.0)))


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def _json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)
