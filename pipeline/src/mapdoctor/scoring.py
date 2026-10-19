from __future__ import annotations

from dataclasses import asdict, dataclass

from mapdoctor.config import HealthThresholds
from mapdoctor.metrics import HealthMetrics


@dataclass
class ReadinessResult:
    score: float
    grade: str
    checks: dict[str, dict]
    disclaimer: str = (
        "The readiness score is an engineering screening heuristic, not a guarantee of localization accuracy. "
        "Use held-out query benchmarks as the acceptance test."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def _check(value, target, higher_is_better: bool, weight: float) -> tuple[float, dict]:
    if value is None:
        return 0.0, {"value": None, "target": target, "pass": False, "weight": weight}
    passed = value >= target if higher_is_better else value <= target
    if higher_is_better:
        ratio = min(1.0, max(0.0, value / target)) if target else 1.0
    else:
        ratio = 1.0 if value <= target else max(0.0, target / value)
    return weight * ratio, {"value": value, "target": target, "pass": passed, "weight": weight}


def score(metrics: HealthMetrics, thresholds: HealthThresholds | None = None) -> ReadinessResult:
    thresholds = thresholds or HealthThresholds()
    specs = {
        "observations_per_image_p10": (metrics.observations_per_image_p10, float(thresholds.min_observations_per_image), True, 0.20),
        "track_length_median": (metrics.track_length_median, thresholds.min_track_length_median, True, 0.15),
        "reprojection_error_p90_px": (metrics.reprojection_error_p90_px, thresholds.max_reprojection_p90_px, False, 0.20),
        "hull_coverage_p10": (metrics.hull_coverage_p10, thresholds.min_hull_coverage, True, 0.15),
        "grid4_coverage_p10": (metrics.grid4_coverage_p10, thresholds.min_grid4_occupancy / 16.0, True, 0.15),
        "largest_covisibility_component_ratio": (metrics.largest_covisibility_component_ratio, thresholds.min_covisibility_component_ratio, True, 0.15),
    }
    total = 0.0
    checks = {}
    for name, (value, target, higher, weight) in specs.items():
        subtotal, detail = _check(value, target, higher, weight)
        total += subtotal
        checks[name] = detail
    numeric_score = round(total * 100.0, 1)
    grade = "A" if numeric_score >= 90 else "B" if numeric_score >= 75 else "C" if numeric_score >= 60 else "D"
    return ReadinessResult(score=numeric_score, grade=grade, checks=checks)
