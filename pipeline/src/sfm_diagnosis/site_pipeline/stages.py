from __future__ import annotations

from dataclasses import dataclass

from .config import CANONICAL_STAGES_V2


@dataclass(frozen=True)
class StageSpec:
    name: str
    outputs: tuple[str, ...]
    requires_metadata: bool = False
    approval_gate: bool = False


STAGE_SPECS = (
    StageSpec("stage00_inventory", ("inputs/corpus_manifest.json", "inputs/metadata.csv")),
    StageSpec("stage01_sanitization", ("artifacts/sanitization/frames.jsonl",)),
    StageSpec(
        "stage02_segment_keyframes",
        ("artifacts/keyframes/segments.jsonl", "artifacts/keyframes/keyframes.jsonl"),
    ),
    StageSpec("stage03_retrieval", ("artifacts/retrieval/candidates.jsonl",), True),
    StageSpec("stage04_pair_geometry", ("artifacts/pairs/geometry.jsonl",), True),
    StageSpec("stage05_graph_build", ("artifacts/graphs/graph_bundle.json",), True),
    StageSpec("stage06_pre_sfm_diagnosis", ("artifacts/diagnosis/pre_sfm.json",), True),
    StageSpec(
        "stage07_diagnostic_selection",
        ("artifacts/selection/diagnostic_selection.json",),
        True,
    ),
    StageSpec("stage08_diagnostic_mapping", ("artifacts/mapping/diagnostic/model",), True),
    StageSpec("stage09_post_sfm_diagnosis", ("artifacts/diagnosis/post_sfm.json",), True),
    StageSpec("stage10_role_assignment", ("artifacts/selection/roles.jsonl",), True),
    StageSpec(
        "stage11_reinforcement",
        (
            "decisions/final_build_decision.json",
            "decisions/final_selection.json",
            "products/candidate_pool/manifest.jsonl",
            "products/weak_region_reshoot_plan.json",
        ),
        True,
    ),
    StageSpec("stage12_final_mapping", ("artifacts/mapping/final/model",), True, True),
    StageSpec("stage13_final_diagnosis", ("artifacts/diagnosis/final_map.json",), True),
    StageSpec(
        "stage14_localization_validation",
        (
            "artifacts/localization/validation.json",
            "products/localization_reference/manifest.jsonl",
        ),
        True,
    ),
    StageSpec(
        "stage15_publish",
        (
            "products/base_geometry/model",
            "products/rejection_manifest.jsonl",
            "products/weak_region_reshoot_plan.md",
            "products/selection_manifest.csv",
            "products/FINAL_RECEIPT.json",
        ),
        True,
    ),
)

if tuple(spec.name for spec in STAGE_SPECS) != CANONICAL_STAGES_V2:  # pragma: no cover
    raise RuntimeError("stage specification order diverged from CANONICAL_STAGES_V2")

STAGE_BY_NAME = {spec.name: spec for spec in STAGE_SPECS}


__all__ = ["STAGE_BY_NAME", "STAGE_SPECS", "StageSpec"]
