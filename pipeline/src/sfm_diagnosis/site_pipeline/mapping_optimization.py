from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


OPTIMIZATION_METHODS = (
    "robust_filter_sweep",
    "connectivity_aware_rescue",
    "connector_frame_rewire",
    "fixed_pose_retriangulation",
)


@dataclass(frozen=True)
class MappingOptimizationContract:
    """Fail-closed contract between Stage 12 and the heavy optimizer worker."""

    enabled: bool = False
    required_methods: tuple[str, ...] = OPTIMIZATION_METHODS
    cleanup_losers: bool = False
    require_geometry_gate: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "MappingOptimizationContract":
        values = dict(payload or {})
        methods = tuple(
            str(value) for value in values.get("required_methods") or OPTIMIZATION_METHODS
        )
        unknown = sorted(set(methods) - set(OPTIMIZATION_METHODS))
        if unknown:
            raise ValueError(f"unknown mapping optimization methods: {unknown}")
        if len(methods) != len(set(methods)):
            raise ValueError("mapping optimization required_methods must not contain duplicates")
        return cls(
            enabled=bool(values.get("enabled", False)),
            required_methods=methods,
            cleanup_losers=bool(values.get("cleanup_losers", False)),
            require_geometry_gate=bool(values.get("require_geometry_gate", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required_methods": list(self.required_methods),
            "cleanup_losers": self.cleanup_losers,
            "require_geometry_gate": self.require_geometry_gate,
        }


def optimization_adapter_payload(
    run_dir: Path,
    input_model: Path,
    output_model: Path,
    contract: MappingOptimizationContract,
) -> dict[str, Any]:
    root = run_dir.resolve()
    optimization = root / "artifacts/mapping/optimization"
    return {
        "run_root": str(root),
        "input_model": str(input_model),
        "output_model": str(output_model),
        "selection": str(root / "decisions/final_selection.json"),
        "roles": str(root / "artifacts/selection/roles.jsonl"),
        "keyframes": str(root / "artifacts/keyframes/keyframes.jsonl"),
        "pair_geometry": str(root / "artifacts/pairs/geometry.jsonl"),
        "corpus_manifest": str(root / "inputs/corpus_manifest.json"),
        "comparison": str(optimization / "comparison.json"),
        "summary": str(optimization / "summary.json"),
        "optimization_root": str(optimization / "candidates"),
        "contract": contract.as_dict(),
    }


def validate_optimizer_result(
    payload: Mapping[str, Any], contract: MappingOptimizationContract
) -> dict[str, Any]:
    if str(payload.get("status") or "").lower() not in {"ok", "success", "completed"}:
        raise RuntimeError("mapping optimizer did not report successful completion")
    completed = {str(value) for value in payload.get("methods_completed") or ()}
    missing = sorted(set(contract.required_methods) - completed)
    if missing:
        raise RuntimeError(f"mapping optimizer missing required methods: {missing}")
    winner = dict(payload.get("winner") or {})
    if contract.require_geometry_gate and not winner.get("geometry_gate_pass"):
        raise RuntimeError("mapping optimizer winner did not pass the geometry gate")
    if not str(winner.get("model") or "").strip():
        raise RuntimeError("mapping optimizer winner is missing its model path")
    return winner


__all__ = [
    "MappingOptimizationContract",
    "OPTIMIZATION_METHODS",
    "optimization_adapter_payload",
    "validate_optimizer_result",
]
