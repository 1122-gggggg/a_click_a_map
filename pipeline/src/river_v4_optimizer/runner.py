from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .metrics import analyze_model, rank_variants


def atomic_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


DEFAULT_FILTER_VARIANTS = (
    {
        "name": "f_3px_1p5deg",
        "max_reprojection_error_px": 3.0,
        "minimum_triangulation_angle_deg": 1.5,
    },
    {
        "name": "f_4px_1p5deg",
        "max_reprojection_error_px": 4.0,
        "minimum_triangulation_angle_deg": 1.5,
    },
    {
        "name": "f_3px_1deg",
        "max_reprojection_error_px": 3.0,
        "minimum_triangulation_angle_deg": 1.0,
    },
    {
        "name": "f_3p5px_1deg",
        "max_reprojection_error_px": 3.5,
        "minimum_triangulation_angle_deg": 1.0,
    },
    {
        "name": "f_4px_1deg",
        "max_reprojection_error_px": 4.0,
        "minimum_triangulation_angle_deg": 1.0,
    },
    {
        "name": "f_4px_0p75deg",
        "max_reprojection_error_px": 4.0,
        "minimum_triangulation_angle_deg": 0.75,
    },
)


def run_filter_sweep(
    input_model: Path,
    output_root: Path,
    variants: Sequence[Mapping[str, Any]] = DEFAULT_FILTER_VARIANTS,
) -> list[dict[str, Any]]:
    from sfm_diagnosis.site_pipeline.robust_filter import (  # ty: ignore[unresolved-import]
        RobustFilterConfig,
        robust_filter_model,
    )

    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants:
        name = str(variant["name"])
        model = root / name / "model"
        if model.exists() or model.is_symlink():
            raise FileExistsError(model)
        config = RobustFilterConfig(
            max_reprojection_error_px=float(variant["max_reprojection_error_px"]),
            minimum_triangulation_angle_deg=float(variant["minimum_triangulation_angle_deg"]),
            minimum_track_length=3,
            bundle_adjustment_iterations=100,
            bundle_adjustment_threads=8,
        )
        receipt = robust_filter_model(input_model, model, config)
        metrics = analyze_model(model)
        row = {"name": name, "filter_receipt": receipt, **metrics}
        atomic_json(root / name / "receipt.json", row)
        rows.append(row)
    ranked = rank_variants(rows)
    atomic_json(root / "comparison.json", {"schema_version": 1, "variants": ranked})
    return ranked


def retriangulate_fixed_poses(
    *,
    input_model: Path,
    database: Path,
    image_root: Path,
    output_model: Path,
    minimum_angle_deg: float,
    ignore_two_view_tracks: bool,
) -> dict[str, Any]:
    """Use COLMAP's known-pose point triangulator without refining intrinsics.

    Source: https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.triangulate_points
    """
    import pycolmap

    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(input_model.resolve(strict=True)))
    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = False
    options.ba_refine_focal_length = False
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.triangulation.min_angle = float(minimum_angle_deg)
    options.triangulation.ignore_two_view_tracks = bool(ignore_two_view_tracks)
    result = pycolmap.triangulate_points(
        reconstruction,
        database.resolve(strict=True),
        image_root.resolve(strict=True),
        output,
        clear_points=True,
        options=options,
        refine_intrinsics=False,
    )
    metrics = analyze_model(output)
    receipt = {
        "schema_version": 1,
        "artifact_type": "RIVER_V4_FIXED_POSE_RETRIANGULATION",
        "input_model": str(input_model.resolve()),
        "database": str(database.resolve()),
        "image_root": str(image_root.resolve()),
        "output_model": str(output),
        "minimum_angle_deg": minimum_angle_deg,
        "ignore_two_view_tracks": ignore_two_view_tracks,
        "refine_intrinsics": False,
        "returned_registered_images": result.num_reg_images(),
        **metrics,
    }
    atomic_json(output.parent / "retriangulation_receipt.json", receipt)
    return receipt
