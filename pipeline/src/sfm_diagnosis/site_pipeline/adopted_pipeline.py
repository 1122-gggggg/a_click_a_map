"""Adopted mapping: all-sequence GlueMap poses, then official EDM retriangulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sfm_diagnosis.io import write_json

from .direct_mapping import (
    DirectMappingRequest,
    DirectMappingRuntime,
    _complete_model,
    run_direct_mapping,
    run_localization_packaging_isolated,
)
from .edm_retriangulate import (
    CELL_PX,
    KEYPOINT_CAP,
    LIFT_DISTANCE_PX,
    official_edm_matcher,
    run_edm_retriangulation,
)

PACKAGE_EDM_CONFIGS = Path(__file__).resolve().parent / "edm_configs"
DEFAULT_EDM_MODEL_CONFIG = PACKAGE_EDM_CONFIGS / "rt5060_edm_base.py"
DEFAULT_EDM_DATA_CONFIG = PACKAGE_EDM_CONFIGS / "rt5060_megadepth.py"


def run_adopted_mapping(
    request: DirectMappingRequest,
    runtime: DirectMappingRuntime,
    *,
    resume: bool = False,
    preprocess_only: bool = False,
    edm_model_config: Path | None = None,
    edm_data_config: Path | None = None,
    cell_px: float = CELL_PX,
) -> dict[str, Any]:
    """Build the adopted map: GlueMap poses + EDM observations + MegaLoc bank."""

    glue = run_direct_mapping(
        request,
        runtime,
        resume=resume,
        preprocess_only=preprocess_only,
    )
    if preprocess_only:
        glue["pipeline"] = "adopted_preprocess_only"
        return glue
    if runtime.edm_root is None or runtime.edm_checkpoint is None:
        raise ValueError("adopted mapping requires edm_root and edm_checkpoint")

    pose_model = request.run_dir / "artifacts/mapping/direct/model"
    work_dir = request.run_dir / "artifacts/mapping/edm"
    output_model = work_dir / "model"
    keyframes_path = request.run_dir / "artifacts/keyframes/keyframes.jsonl"
    image_root = request.run_dir / "artifacts/keyframes/images"
    model_config = Path(edm_model_config or DEFAULT_EDM_MODEL_CONFIG)
    data_config = Path(edm_data_config or DEFAULT_EDM_DATA_CONFIG)
    if resume and _complete_model(output_model):
        retriangulation = {
            "status": "resumed",
            "output_model": str(output_model),
        }
    else:
        matcher = official_edm_matcher(
            edm_root=runtime.edm_root,
            edm_checkpoint=runtime.edm_checkpoint,
            model_config=model_config,
            data_config=data_config,
        )
        retriangulation = run_edm_retriangulation(
            pose_model=pose_model,
            image_root=image_root,
            keyframes_path=keyframes_path,
            work_dir=work_dir,
            output_model=output_model,
            matcher=matcher,
            cell_px=cell_px,
            keypoint_cap=KEYPOINT_CAP,
            resume=resume,
        )

    calibration = json.loads(request.intrinsics_path.read_text(encoding="utf-8"))
    localization = run_localization_packaging_isolated(
        {
            "model_dir": str(output_model),
            "keyframes_path": str(keyframes_path),
            "output_dir": str(request.run_dir / "products/localization"),
            "runtime": {
                "gluemap_root": str(runtime.gluemap_root),
                "base_config_path": str(runtime.base_config_path),
                "workspace_root": str(runtime.workspace_root),
                "megaloc_source": str(runtime.megaloc_source),
                "megaloc_checkpoint": str(runtime.megaloc_checkpoint),
                "edm_root": str(runtime.edm_root),
                "edm_checkpoint": str(runtime.edm_checkpoint),
            },
            "intrinsics": calibration,
        }
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "ADOPTED_MAPPING_FINAL_RECEIPT",
        "pipeline": "gluemap_poses_edm_observations",
        "status": "MAP_BUILT_UNVALIDATED_ALL_INPUTS",
        "validation": "NONE",
        "cell_px": cell_px,
        "lift_distance_px": LIFT_DISTANCE_PX,
        "gluemap": glue,
        "edm_retriangulation": retriangulation,
        "localization": localization,
        "products": {
            "pose_model": str(pose_model),
            "edm_model": str(output_model),
            "localization": str(request.run_dir / "products/localization"),
        },
    }
    write_json(request.run_dir / "products/FINAL_RECEIPT.json", receipt)
    return receipt


__all__ = [
    "DEFAULT_EDM_DATA_CONFIG",
    "DEFAULT_EDM_MODEL_CONFIG",
    "run_adopted_mapping",
]
