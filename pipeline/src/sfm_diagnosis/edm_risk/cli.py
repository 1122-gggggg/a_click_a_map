from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from sfm_diagnosis.io import load_gluemap
from sfm_diagnosis.matchability import load_matchability_source
from sfm_diagnosis.io import write_json

from .calibration import RiskCalibrationConfig
from .colorize import colorize_map_from_observations
from .ablation import run_ablation
from .edm_artifacts import filter_edm_results, load_edm_results
from .fusion import build_training_samples
from .fim import FIMConfig
from .actloc_provider import build_actloc_provider
from .grid import SpatialGridConfig
from .pipeline import EDMRiskDiagnosis
from .occlusion import PointCloudOcclusionProxy, load_point_cloud


def parse_pitch_values(value: str) -> tuple[float, ...]:
    try:
        pitches = tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pitch values must be comma-separated numbers") from exc
    if not pitches:
        raise argparse.ArgumentTypeError("pitch values must not be empty")
    return pitches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edm-risk-diagnosis",
        description=("Predict post-map EDM localization failure risk over an XYZ/yaw/pitch grid."),
    )
    parser.add_argument("--map", dest="map_path", type=Path, required=True)
    parser.add_argument("--map-adapter", choices=("colmap", "gluemap"), default="gluemap")
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--edm-results", type=Path)
    parser.add_argument("--edm-format", choices=("auto", "canonical", "river"), default="auto")
    parser.add_argument("--ambiguity-results", type=Path)
    parser.add_argument("--loo-mode", choices=("strict", "reference-exclusion"), default="strict")
    parser.add_argument(
        "--calibration-model",
        choices=("logistic", "hist_gradient_boosting"),
        default="logistic",
    )
    parser.add_argument(
        "--splitter",
        choices=("leave_one_group_out", "group_kfold"),
        default="leave_one_group_out",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-fnr", type=float)
    parser.add_argument(
        "--risk-feature-set",
        choices=("A", "B", "C", "D", "E", "F"),
        default="F",
        help="A-F ablation feature set used by the final calibrator",
    )
    parser.add_argument(
        "--edm-include-sessions",
        default="",
        help="comma-separated session IDs kept as calibration labels",
    )
    parser.add_argument(
        "--edm-exclude-sessions",
        default="",
        help="comma-separated session IDs dropped from calibration labels",
    )
    parser.add_argument(
        "--min-roc-auc",
        type=float,
        default=0.6,
        help="below this out-of-fold ROC-AUC the run is UNCALIBRATED_EDM_LOO",
    )
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument(
        "--bounds", type=float, nargs=6, metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1")
    )
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--yaw-step", type=float, default=30.0)
    parser.add_argument("--pitch-values", type=parse_pitch_values, default=(-30.0, 0.0, 30.0))
    parser.add_argument("--max-waypoints", type=int, default=25_000)
    parser.add_argument("--max-landmark-distance", type=float)
    parser.add_argument(
        "--occupancy-radius",
        type=float,
        help="keep only waypoints within this distance of a landmark or camera",
    )
    parser.add_argument("--matchability", type=Path)
    parser.add_argument(
        "--color-images",
        type=Path,
        help="restore Point3D RGB from registered mapping-image observations",
    )
    parser.add_argument("--pixel-sigma", type=float, default=1.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--fim-regularization", type=float, default=1e-9)
    parser.add_argument(
        "--actloc-mode", choices=("disabled", "fallback", "official"), default="disabled"
    )
    parser.add_argument("--actloc-source", type=Path)
    parser.add_argument("--actloc-checkpoint", type=Path)
    parser.add_argument("--actloc-cache", type=Path)
    parser.add_argument("--actloc-fail-closed", action="store_true")
    parser.add_argument("--occlusion-point-cloud", type=Path)
    parser.add_argument("--occlusion-splat-radius-px", type=int, default=1)
    parser.add_argument("--occlusion-depth-tolerance", type=float, default=0.05)
    parser.add_argument("--occlusion-angle-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--occlusion-min-support-count", type=int, default=2)
    parser.add_argument("--occlusion-max-depth-spread", type=float, default=0.25)
    parser.add_argument("--occlusion-max-search-radius-px", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "full" and args.edm_results is None:
        parser.error("--mode full requires --edm-results from held-out EDM localization")
    bounds = None
    if args.bounds is not None:
        bounds = (tuple(args.bounds[:3]), tuple(args.bounds[3:]))
    config = SpatialGridConfig(
        voxel_size=args.voxel_size,
        bounds=bounds,
        padding=args.padding,
        yaw_step_deg=args.yaw_step,
        pitch_values_deg=tuple(args.pitch_values),
        max_waypoints=args.max_waypoints,
        max_landmark_distance=args.max_landmark_distance,
        occupancy_radius=args.occupancy_radius,
    )
    map_data = load_gluemap(args.map_path)
    colorization = (
        None
        if args.color_images is None
        else colorize_map_from_observations(map_data, args.color_images)
    )
    matchability = (
        None if args.matchability is None else load_matchability_source(args.matchability, map_data)
    )
    fim_config = FIMConfig(
        pixel_sigma=args.pixel_sigma,
        translation_scale=args.translation_scale,
        regularization=args.fim_regularization,
    )
    actloc = build_actloc_provider(
        mode=args.actloc_mode,
        map_data=map_data,
        source_root=args.actloc_source,
        checkpoint=args.actloc_checkpoint,
        cache_dir=args.actloc_cache,
        fail_open=not args.actloc_fail_closed,
    )
    occlusion = None
    if args.occlusion_point_cloud is not None:
        occlusion = PointCloudOcclusionProxy(
            load_point_cloud(args.occlusion_point_cloud),
            map_data.median_intrinsics,
            splat_radius_px=args.occlusion_splat_radius_px,
            depth_tolerance=args.occlusion_depth_tolerance,
            angle_tolerance_deg=args.occlusion_angle_tolerance_deg,
            min_support_count=args.occlusion_min_support_count,
            max_depth_spread=args.occlusion_max_depth_spread,
            max_search_radius_px=args.occlusion_max_search_radius_px,
            source_path=args.occlusion_point_cloud,
        )
    edm_results = []
    if args.mode == "full":
        edm_results = filter_edm_results(
            load_edm_results(
                args.edm_results,
                evidence_mode=args.loo_mode,
                artifact_format=args.edm_format,
            ),
            include_sessions=_csv_tokens(args.edm_include_sessions) or None,
            exclude_sessions=_csv_tokens(args.edm_exclude_sessions),
        )
        if not edm_results:
            parser.error("no EDM results remain after session include/exclude filters")
        ambiguity = _load_ambiguity(args.ambiguity_results)
        calibration_config = RiskCalibrationConfig(
            model_type=args.calibration_model,
            splitter=args.splitter,
            folds=args.folds,
            target_recall=args.target_recall,
            max_fnr=args.max_fnr,
            min_roc_auc=args.min_roc_auc,
        )
        result = EDMRiskDiagnosis(map_data).run_full(
            config,
            edm_results=edm_results,
            matchability=matchability,
            fim_config=fim_config,
            actloc_provider=actloc,
            calibration_config=calibration_config,
            ambiguity_by_query=ambiguity,
            risk_feature_set=args.risk_feature_set,
            occlusion_proxy=occlusion,
        )
    else:
        result = EDMRiskDiagnosis(map_data).run_fast(
            config,
            matchability=matchability,
            fim_config=fim_config,
            actloc_provider=actloc,
            occlusion_proxy=occlusion,
        )
    if colorization is not None:
        result.metadata["rgb_colorization"] = colorization
    outputs = result.save(args.output)
    if colorization is not None:
        colorization_path = args.output / "rgb_colorization.json"
        write_json(colorization_path, colorization)
        outputs["rgb_colorization_json"] = colorization_path
    if args.mode == "full":
        calibration_path = args.output / "calibration.json"
        edm_path = args.output / "edm_loo_results.json"
        write_json(calibration_path, result.metadata["calibration"])
        write_json(
            edm_path,
            {
                "schema_version": 1,
                "artifact_type": "EDM_HELD_OUT_RESULTS",
                "loo_mode": args.loo_mode,
                "strict_loo": args.loo_mode == "strict",
                "pseudo_loo": args.loo_mode != "strict",
                "results": [row.to_dict() for row in edm_results],
            },
        )
        outputs.update({"calibration_json": calibration_path, "edm_loo_json": edm_path})
        samples, _ = build_training_samples(
            result.rows,
            edm_results,
            ambiguity_by_query=ambiguity,
            max_position_distance=max(config.voxel_size * 2.0, 1e-6),
            max_orientation_distance_deg=max(config.yaw_step_deg, 30.0),
        )
        samples_path = args.output / "calibration_samples.json"
        ablation_path = args.output / "ablation.json"
        write_json(
            samples_path,
            {
                "schema_version": 1,
                "artifact_type": "EDM_RISK_TRAINING_SAMPLES",
                "samples": [
                    {
                        "query_id": sample.query_id,
                        "group": sample.group,
                        "failure": sample.failure,
                        "features": dict(sample.features),
                    }
                    for sample in samples
                ],
            },
        )
        write_json(ablation_path, run_ablation(samples, config=calibration_config))
        outputs.update({"calibration_samples_json": samples_path, "ablation_json": ablation_path})
    print(
        json.dumps(
            {
                "mode": result.mode,
                "pose_samples": len(result.rows),
                "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
                "probability_status": result.metadata["probability_status"],
            },
            indent=2,
        )
    )
    return 0


def _csv_tokens(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(token.strip() for token in value.split(",") if token.strip())


def _load_ambiguity(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("analyses"), list):
        rows = payload["analyses"]
    elif isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        return {str(key): dict(value) for key, value in payload.items()}
    else:
        raise ValueError("ambiguity artifact must be a mapping or list")
    return {
        str(row["query_id"]): {key: value for key, value in row.items() if key != "query_id"}
        for row in rows
    }
