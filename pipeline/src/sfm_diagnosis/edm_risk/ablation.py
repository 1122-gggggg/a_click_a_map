"""Session-safe A--F component ablation for EDM failure prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sfm_diagnosis.io import write_json

from .calibration import RiskCalibrationConfig, RiskCalibrator, RiskTrainingSample


_MAP_FEATURES = {
    "visible_landmarks",
    "effective_landmarks",
    "median_track_length",
    "track_ge_3_ratio",
    "track_ge_5_ratio",
    "covisibility",
    "independent_observers",
    "parallax_p10_deg",
    "parallax_p25_deg",
    "parallax_median_deg",
    "viewpoint_entropy",
    "view_entropy",
    "convex_hull_coverage",
    "grid_occupancy",
    "landmark_spatial_entropy",
    "median_reprojection_error",
    "reprojection_error_p90",
    "positive_depth_ratio",
    "landmark_linearity",
    "landmark_planarity",
    "landmark_scattering",
    "mapping_camera_linearity",
    "mapping_camera_planarity",
    "mapping_camera_scattering",
}
_FIM_FEATURES = {
    "fim_lambda_min",
    "fim_lambda_max",
    "fim_condition",
    "fim_logdet",
    "fim_trace",
    "fim_a_opt",
    "fim_d_opt",
    "fim_e_opt",
    "fim_translation_min_eigenvalue",
    "fim_rotation_min_eigenvalue",
}
_ACTLOC_FEATURES = {"actloc_score"}
_EDM_FEATURES = {
    "empirical_edm_success_rate",
    "empirical_pnp_inliers",
    "empirical_inlier_ratio",
    "empirical_reprojection_error",
    "consecutive_failure_probability",
}
_AMBIGUITY_FEATURES = {
    "num_pose_modes",
    "mode_support_margin",
    "mode_translation_separation",
    "mode_rotation_separation_deg",
    "mode_entropy",
    "loo_translation_jump",
    "loo_rotation_jump_deg",
}

_VARIANTS = {
    "A": _MAP_FEATURES,
    "B": _MAP_FEATURES | _FIM_FEATURES,
    "C": _MAP_FEATURES | _FIM_FEATURES | _ACTLOC_FEATURES,
    "D": _MAP_FEATURES | _FIM_FEATURES | _EDM_FEATURES,
    "E": _MAP_FEATURES | _FIM_FEATURES | _EDM_FEATURES | _AMBIGUITY_FEATURES,
    "F": (
        _MAP_FEATURES
        | _FIM_FEATURES
        | _ACTLOC_FEATURES
        | _EDM_FEATURES
        | _AMBIGUITY_FEATURES
    ),
}


def run_ablation(
    samples: Sequence[RiskTrainingSample],
    *,
    config: RiskCalibrationConfig | None = None,
) -> dict:
    variants = {}
    for name, allowed in _VARIANTS.items():
        selected = select_variant_samples(samples, name)
        result = RiskCalibrator(config).fit(selected)
        variants[name] = {
            "components": _components(name),
            "feature_names": list(result.feature_names),
            "feature_count": len(result.feature_names),
            "metrics": result.metrics,
            "threshold": result.threshold,
            "group_leakage_detected": result.group_leakage_detected,
        }
    return {
        "schema_version": 1,
        "artifact_type": "EDM_RISK_ABLATION",
        "variants": variants,
        "evaluation_contract": "All metrics are out-of-fold and session/video-disjoint.",
    }


def select_variant_samples(
    samples: Sequence[RiskTrainingSample], variant: str
) -> list[RiskTrainingSample]:
    if variant not in _VARIANTS:
        raise ValueError(f"unknown ablation variant {variant!r}")
    allowed = _VARIANTS[variant]
    return [
        RiskTrainingSample(
            query_id=sample.query_id,
            group=sample.group,
            failure=sample.failure,
            features={
                key: value for key, value in sample.features.items() if key in allowed
            },
        )
        for sample in samples
    ]


def _components(name: str) -> list[str]:
    return {
        "A": ["handcrafted_map_diagnostics"],
        "B": ["handcrafted_map_diagnostics", "FIM"],
        "C": ["handcrafted_map_diagnostics", "FIM", "ActLoc"],
        "D": ["handcrafted_map_diagnostics", "FIM", "EDM_LOO"],
        "E": ["handcrafted_map_diagnostics", "FIM", "EDM_LOO", "ambiguity"],
        "F": ["handcrafted_map_diagnostics", "FIM", "ActLoc", "EDM_LOO", "ambiguity", "calibration"],
    }[name]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run EDM risk A-F ablation")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("logistic", "hist_gradient_boosting"), default="logistic")
    parser.add_argument("--splitter", choices=("leave_one_group_out", "group_kfold"), default="leave_one_group_out")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    arguments = parser.parse_args(argv)
    payload = json.loads(arguments.samples.read_text(encoding="utf-8"))
    rows = payload.get("samples") if isinstance(payload, dict) else payload
    samples = [RiskTrainingSample(**row) for row in rows]
    report = run_ablation(
        samples,
        config=RiskCalibrationConfig(
            model_type=arguments.model,
            splitter=arguments.splitter,
            folds=arguments.folds,
            target_recall=arguments.target_recall,
        ),
    )
    write_json(arguments.output, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["run_ablation", "select_variant_samples"]
