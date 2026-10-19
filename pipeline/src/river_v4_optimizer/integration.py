from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import rank_variants
from .runner import DEFAULT_FILTER_VARIANTS


@dataclass(frozen=True)
class OptimizationRecipe:
    """Serializable contract used by the SitePipeline optimizer adapter."""

    enabled: bool = False
    filter_variants: tuple[Mapping[str, Any], ...] = DEFAULT_FILTER_VARIANTS
    connectivity_rescue: bool = True
    selection_rewires: tuple[Mapping[str, Any], ...] = ()
    retriangulation_variants: tuple[Mapping[str, Any], ...] = ()
    cleanup_losers: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "OptimizationRecipe":
        values = dict(payload or {})
        filters = tuple(
            dict(row) for row in values.get("filter_variants") or DEFAULT_FILTER_VARIANTS
        )
        rewires = tuple(dict(row) for row in values.get("selection_rewires") or ())
        retriangulations = tuple(dict(row) for row in values.get("retriangulation_variants") or ())
        for row in filters:
            if not str(row.get("name") or "").strip():
                raise ValueError("each filter variant requires a name")
            if "max_reprojection_error_px" not in row:
                raise ValueError("each filter variant requires max_reprojection_error_px")
            if not ({"minimum_angle_deg", "minimum_triangulation_angle_deg"} & row.keys()):
                raise ValueError("each filter variant requires minimum_angle_deg")
        for row in rewires:
            if not str(row.get("name") or "").strip():
                raise ValueError("each selection rewire requires a name")
            if not row.get("remove") and not row.get("add"):
                raise ValueError("each selection rewire requires a non-empty remove or add list")
        for row in retriangulations:
            if not str(row.get("name") or "").strip():
                raise ValueError("each retriangulation variant requires a name")
            if "minimum_angle_deg" not in row:
                raise ValueError("each retriangulation variant requires minimum_angle_deg")
        return cls(
            enabled=bool(values.get("enabled", False)),
            filter_variants=filters,
            connectivity_rescue=bool(values.get("connectivity_rescue", True)),
            selection_rewires=rewires,
            retriangulation_variants=retriangulations,
            cleanup_losers=bool(values.get("cleanup_losers", False)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "filter_variants": [dict(row) for row in self.filter_variants],
            "connectivity_rescue": self.connectivity_rescue,
            "selection_rewires": [dict(row) for row in self.selection_rewires],
            "retriangulation_variants": [dict(row) for row in self.retriangulation_variants],
            "cleanup_losers": self.cleanup_losers,
        }


def build_candidate_plan(recipe: OptimizationRecipe) -> list[dict[str, Any]]:
    if not recipe.enabled:
        return []
    plan = [
        {"method": "robust_filter_sweep", "variant": str(row["name"]), "config": dict(row)}
        for row in recipe.filter_variants
    ]
    if recipe.connectivity_rescue:
        plan.append({"method": "connectivity_aware_rescue", "variant": "quality_gated"})
    plan.extend(
        {
            "method": "connector_frame_rewire",
            "variant": str(row["name"]),
            "config": dict(row),
        }
        for row in recipe.selection_rewires
    )
    plan.extend(
        {
            "method": "fixed_pose_retriangulation",
            "variant": str(row["name"]),
            "config": dict(row),
        }
        for row in recipe.retriangulation_variants
    )
    return plan


def select_winner(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ranked = rank_variants(candidates)
    passing = [row for row in ranked if row["geometry_gate_pass"] and row.get("model")]
    if not passing:
        raise RuntimeError("no optimization candidate passed the geometry gate")
    return passing[0]


def prunable_loser_models(
    candidate_root: Path,
    winner_model: Path,
    candidates: Sequence[Mapping[str, Any]],
) -> list[Path]:
    root = candidate_root.resolve(strict=True)
    winner = winner_model.resolve(strict=True)
    removable: set[Path] = set()
    for row in candidates:
        value = row.get("model")
        if not value:
            continue
        model = Path(str(value)).resolve(strict=True)
        if model == winner or not model.is_relative_to(root):
            continue
        removable.add(model)
    return sorted(removable)


__all__ = [
    "OptimizationRecipe",
    "build_candidate_plan",
    "prunable_loser_models",
    "select_winner",
]
