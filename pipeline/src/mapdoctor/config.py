from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class HealthThresholds:
    min_observations_per_image: int = 50
    min_hull_coverage: float = 0.15
    min_grid4_occupancy: int = 6
    max_reprojection_p90_px: float = 3.0
    min_track_length_median: float = 3.0
    min_covisibility_component_ratio: float = 0.95


@dataclass(frozen=True)
class LocalizationThresholds:
    min_inliers: int = 25
    min_inlier_ratio: float = 0.25
    max_reprojection_p90_px: float = 3.0
    min_hull_coverage: float = 0.15
    min_grid4_occupancy: int = 6
    min_positive_depth_ratio: float = 1.0
    min_pose_consensus: float = 0.67
    min_strict_success_rate: float = 0.95
    required_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.min_strict_success_rate, bool):
            raise ValueError("min_strict_success_rate must be finite and in [0, 1]")
        try:
            rate = float(self.min_strict_success_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "min_strict_success_rate must be finite and in [0, 1]"
            ) from exc
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("min_strict_success_rate must be finite and in [0, 1]")
        supported = {
            "inliers",
            "inlier_ratio",
            "reproj_p90_px",
            "hull_coverage",
            "grid4_occupancy",
            "positive_depth_ratio",
            "pose_consensus",
        }
        required = tuple(str(name) for name in self.required_metrics)
        unknown = set(required) - supported
        if unknown:
            raise ValueError(
                "Unknown required localization metrics: "
                + ", ".join(sorted(unknown))
            )
        object.__setattr__(self, "required_metrics", required)


@dataclass(frozen=True)
class ComparisonThresholds:
    max_success_rate_drop: float = 0.02
    max_common_success_inlier_drop: float = 0.05
    max_common_success_reprojection_increase_px: float = 0.5
    max_new_failure_rate: float = 0.02


@dataclass(frozen=True)
class Settings:
    health: HealthThresholds = field(default_factory=HealthThresholds)
    localization: LocalizationThresholds = field(default_factory=LocalizationThresholds)
    comparison: ComparisonThresholds = field(default_factory=ComparisonThresholds)
    region_cell_size: float = 5.0


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        return Settings()
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("MapDoctor config must be a JSON object")
    known = {"health", "localization", "comparison", "region_cell_size"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level settings: {', '.join(sorted(unknown))}")

    def make(cls, section: str):
        values = raw.get(section, {})
        allowed = set(cls.__dataclass_fields__)
        extra = set(values) - allowed
        if extra:
            raise ValueError(f"Unknown {section} settings: {', '.join(sorted(extra))}")
        return cls(**values)

    return Settings(
        health=make(HealthThresholds, "health"),
        localization=make(LocalizationThresholds, "localization"),
        comparison=make(ComparisonThresholds, "comparison"),
        region_cell_size=float(raw.get("region_cell_size", 5.0)),
    )
