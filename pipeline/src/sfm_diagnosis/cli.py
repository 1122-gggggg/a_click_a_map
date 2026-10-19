from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from mapdoctor.adapters import get_adapter
from scipy.spatial.transform import Rotation
from sfm_qa.bridge import map_model_to_map_data

from .diagnose import diagnose_pose, thresholds_from_dict
from .evidence import load_build_evidence
from .heatmap import HeatmapConfig, build_heatmap, save_heatmap
from .io import write_json
from .risk_ply import write_risk_ply
from .logs import LocalizationHistory
from .matchability import (
    matchability_config_from_dict,
    load_matchability_source,
    save_landmark_matchability,
)
from .models import MapData, Pose
from .reference_consensus import analyze_reference_hypotheses
from .repair import suggest_capture_viewpoints
from .report import map_health_summary
from .route import RouteAuditConfig, audit_route, save_route_audit
from .visibility import (
    MeshRaycaster,
    SunIllumination,
    sun_light_travel_direction_from_az_el,
)
from .weak_regions import (
    analyze_weak_regions,
    save_weak_region_analysis,
    weak_region_config_from_dict,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    thresholds = thresholds_from_dict(cfg.get("thresholds", {}))

    if args.command == "inspect":
        m = _load_map(args)
        result = map_health_summary(m)
        _emit(result, args.output)
        return 0

    if args.command == "analyze":
        m = _load_map(args)
        region_cfg = dict(cfg.get("weak_regions", {}))
        for key, value in {
            "cluster_radius": args.cluster_radius,
            "anchor_radius": args.anchor_radius,
            "max_report_regions": args.max_regions,
        }.items():
            if value is not None:
                region_cfg[key] = value
        evidence = load_build_evidence(
            m,
            database=args.database,
            pairs=args.pairs,
            images_manifest=args.images_manifest,
            images_dir=args.images_dir,
        )
        analysis = analyze_weak_regions(
            m,
            evidence=evidence,
            config=weak_region_config_from_dict(region_cfg),
        )
        save_weak_region_analysis(args.output, analysis)
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).resolve()),
                    "diagnostic_mode": analysis.summary["diagnostic_mode"],
                    "weak_images": analysis.summary["num_weak_images"],
                    "weak_regions": analysis.summary["num_weak_regions"],
                    "cause_counts": analysis.summary["cause_counts"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "consensus":
        results = analyze_reference_hypotheses(
            args.hypotheses,
            student_t_nu=args.student_t_nu,
            covariance_floor_m=args.covariance_floor,
            sigma_inflation=args.sigma_inflation,
            max_dispersion_m=args.max_dispersion,
            max_consensus_sigma_m=args.max_consensus_sigma,
            max_rotation_dispersion_deg=args.max_rotation_dispersion,
            min_covariance_eligible_ratio=args.min_covariance_eligible_ratio,
            held_out_path=args.held_out_hypotheses,
        )
        payload = {
            "queries": results,
            "num_queries": len(results),
            "source": str(Path(args.hypotheses)),
            "interpretation": (
                "sigma_disp measures cross-reference disagreement; sigma_cons measures "
                "information/covariance weakness. sigma_joint is RIC-Loc's held-out "
                "median-normalized max fusion and is only set when --held-out-hypotheses "
                "is provided. Retrieval-score gaps are not a pose-failure signal."
            ),
        }
        _emit(payload, args.output)
        return 0

    if args.command == "route":
        route_cfg = dict(cfg.get("route", {}))
        for key, value in {
            "sample_spacing_m": args.sample_spacing,
            "max_heatmap_distance_m": args.max_heatmap_distance,
            "smoothness_weight": args.smoothness_weight,
            "task_forward_weight": args.task_forward_weight,
            "weak_direction_weight": args.weak_direction_weight,
            "max_turn_deg_per_m": args.max_turn_deg_per_m,
            "enter_risk": args.enter_risk,
            "exit_risk": args.exit_risk,
            "min_segment_length_m": args.min_segment_length,
            "calibration_min_samples": args.calibration_min_samples,
        }.items():
            if value is not None:
                route_cfg[key] = value
        if args.no_turn_limit:
            route_cfg["max_turn_deg_per_m"] = None
        result = audit_route(
            args.heatmap,
            args.route,
            config=RouteAuditConfig(**route_cfg),
            calibration=args.calibration,
        )
        save_route_audit(args.output, result)
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).resolve()),
                    "route_samples": result.summary["route_samples"],
                    "weak_segments": result.summary["num_weak_segments"],
                    "weak_length_m": result.summary["weak_length_m"],
                    "supported_fraction_by_length": result.summary[
                        "supported_fraction_by_length"
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "matchability":
        m = _load_map(args)
        match_cfg = matchability_config_from_dict(cfg.get("matchability", {}))
        table = load_matchability_source(args.events, m, config=match_cfg)
        path = save_landmark_matchability(args.output, table)
        print(
            json.dumps(
                {
                    "output": str(path.resolve()),
                    "landmarks": len(table.point_ids),
                    "reweight_fim": match_cfg.reweight_fim,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "risk-ply":
        m = _load_map(args)
        receipt = write_risk_ply(
            m,
            args.output,
            heatmap=args.heatmap,
            weak_regions=args.weak_regions,
            localization=args.logs,
            image_roles=getattr(args, "image_roles", None),
            include_actloc_shadow=bool(args.include_actloc_shadow),
            filename=getattr(args, "filename", "localization_risk_spheres.ply"),
            sphere_radius=getattr(args, "sphere_radius", None),
            sphere_samples=getattr(args, "sphere_samples", None) or 96,
        )
        print(
            json.dumps(
                {
                    "output": receipt["ply"],
                    "ply_base": receipt.get("ply_base"),
                    "ply_markers": receipt.get("ply_markers"),
                    "ply_markers_raw": receipt.get("ply_markers_raw"),
                    "ply_classes": receipt.get("ply_classes"),
                    "ply_full": receipt.get("ply_full"),
                    "ply_mesh": receipt.get("ply_mesh"),
                    "format": receipt.get("format"),
                    "map_vertices": receipt["map_vertices"],
                    "map_vertices_retained": receipt.get("map_vertices_retained"),
                    "map_vertices_excluded": receipt.get("map_vertices_excluded"),
                    "marker_spheres": receipt["marker_spheres"],
                    "display_spheres": receipt.get("display_spheres"),
                    "sphere_radius": receipt.get("sphere_radius"),
                    "sphere_radius_source": receipt.get("sphere_radius_source"),
                    "vertex_count": receipt.get("vertex_count"),
                    "counts": receipt["counts"],
                    "aggregation": {
                        "radius": (receipt.get("aggregation") or {}).get("radius"),
                        "cluster_count": (receipt.get("aggregation") or {}).get("cluster_count"),
                        "raw_marker_count": (receipt.get("aggregation") or {}).get("raw_marker_count"),
                    },
                    "visual_balance": receipt.get("visual_balance"),
                    "clip": {
                        "robust_diagonal": (receipt.get("clip") or {}).get("robust_diagonal"),
                        "full_diagonal": (receipt.get("clip") or {}).get("full_diagonal"),
                        "camera_diagonal": (receipt.get("clip") or {}).get("camera_diagonal"),
                        "camera_nn": (receipt.get("clip") or {}).get("camera_nn"),
                        "retained_count": (receipt.get("clip") or {}).get("retained_count"),
                        "excluded_count": (receipt.get("clip") or {}).get("excluded_count"),
                    },
                    "visible_rgb": (receipt.get("cloudcompare") or {}).get("visible_rgb"),
                    "fim_recomputed": receipt["fim_recomputed"],
                    "actloc": receipt["actloc"],
                    "caveats": receipt["caveats"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    m = _load_map(args)
    history = LocalizationHistory.load(args.logs) if getattr(args, "logs", None) else None
    occlusion, illumination = _environment(args)
    match_cfg = matchability_config_from_dict(cfg.get("matchability", {}))
    match_table = None
    if getattr(args, "matchability", None):
        match_table = load_matchability_source(args.matchability, m, config=match_cfg)

    if args.command == "pose":
        pose, intr = _resolve_pose(m, args)
        result = diagnose_pose(
            m,
            pose,
            intrinsics=intr,
            thresholds=thresholds,
            history=history,
            illumination=illumination,
            occlusion=occlusion,
            matchability=match_table,
            matchability_config=match_cfg,
        )
        payload = {
            "pose": {"center_w": pose.center_w.tolist(), "R_wc": pose.R_wc.tolist()},
            "diagnosis": result.as_dict(include_point_indices=args.include_points),
        }
        _emit(payload, args.output)
        return 0

    if args.command == "heatmap":
        hcfg_dict = dict(cfg.get("heatmap", {}))
        for key, value in {
            "spacing_m": args.spacing,
            "orientation_mode": args.orientation_mode,
            "sample_mode": getattr(args, "sample_mode", None),
            "orientations_per_position": args.orientations_per_position,
            "yaw_step_deg": args.yaw_step,
        }.items():
            if value is not None:
                hcfg_dict[key] = value
        if args.pitches:
            hcfg_dict["pitches_deg"] = tuple(args.pitches)
        hcfg = HeatmapConfig(**hcfg_dict)
        bounds = None
        if args.bounds:
            b = np.asarray(args.bounds, dtype=float)
            bounds = (b[:3], b[3:])
        detailed, aggregate = build_heatmap(
            m,
            config=hcfg,
            thresholds=thresholds,
            history=history,
            illumination=illumination,
            occlusion=occlusion,
            matchability=match_table,
            bounds=bounds,
        )
        save_heatmap(args.output, detailed, aggregate)
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).resolve()),
                    "pose_samples": len(detailed),
                    "position_samples": len(aggregate),
                },
                indent=2,
            )
        )
        return 0


    if args.command == "repair":
        pose, _ = _resolve_pose(m, args)
        diagnosis = diagnose_pose(
            m,
            pose,
            thresholds=thresholds,
            history=history,
            illumination=illumination,
            occlusion=occlusion,
        )
        candidates = suggest_capture_viewpoints(
            m,
            pose,
            top_k=args.top_k,
            thresholds=thresholds,
        )
        payload = {
            "weak_pose_diagnosis": diagnosis.as_dict(),
            "capture_candidates": [c.as_dict() for c in candidates],
            "caveat": (
                "Candidates optimize overlap/viewpoint-diversity proxies over existing landmarks; "
                "they do not predict unseen geometry or flight feasibility. Apply "
                "geofence/collision checks separately."
            ),
        }
        _emit(payload, args.output)
        return 0

    parser.error("unknown command")
    return 2


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sfm-diagnosis",
        description="Adapter-based diagnostics for map localizability and repair.",
    )
    p.add_argument("--config", help="YAML config; defaults are built in")
    sub = p.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser(
        "inspect",
        help="Audit global reconstruction health through a map adapter",
    )
    inspect.add_argument("model")
    _add_map_adapter_arg(inspect)
    inspect.add_argument("--output", help="write JSON instead of stdout")

    analyze = sub.add_parser(
        "analyze",
        help="Locate weak SfM regions, explain root causes, and generate a repair plan",
    )
    analyze.add_argument("model")
    _add_map_adapter_arg(analyze)
    analyze.add_argument("--output", required=True, help="output directory")
    analyze.add_argument(
        "--database",
        help="optional COLMAP/GlueMap database.db for pair match/inlier evidence",
    )
    analyze.add_argument(
        "--pairs",
        help=(
            "optional CSV/JSON/JSONL retrieval/pair table; preserve unselected candidates "
            "to diagnose retrieval gaps"
        ),
    )
    analyze.add_argument(
        "--images-manifest",
        help="optional CSV/JSON/JSONL image metadata/quality/route table",
    )
    analyze.add_argument(
        "--images-dir",
        help="optional source image root; computes blur/texture/exposure diagnostics",
    )
    analyze.add_argument(
        "--cluster-radius",
        type=float,
        help="weak-image clustering radius in map units; default is map-adaptive",
    )
    analyze.add_argument(
        "--anchor-radius",
        type=float,
        help="healthy-anchor search radius in map units; default derives from cluster radius",
    )
    analyze.add_argument("--max-regions", type=int, help="maximum regions to report")

    consensus = sub.add_parser(
        "consensus",
        help=(
            "Diagnose per-reference pose hypotheses using RIC-Loc-style robust consensus "
            "and covariance fusion"
        ),
    )
    consensus.add_argument("hypotheses", help="CSV/JSON/JSONL per-reference hypothesis table")
    consensus.add_argument("--output", help="write JSON instead of stdout")
    consensus.add_argument("--student-t-nu", type=float, default=5.0)
    consensus.add_argument("--covariance-floor", type=float, default=1e-3)
    consensus.add_argument(
        "--sigma-inflation",
        type=float,
        default=1.0,
        help="held-out calibration factor for correlated/shared-model covariance",
    )
    consensus.add_argument("--max-dispersion", type=float, default=0.5)
    consensus.add_argument("--max-consensus-sigma", type=float, default=0.5)
    consensus.add_argument("--max-rotation-dispersion", type=float, default=5.0)
    consensus.add_argument("--min-covariance-eligible-ratio", type=float, default=0.5)
    consensus.add_argument(
        "--held-out-hypotheses",
        help=(
            "disjoint hypothesis table used only to fit RIC-Loc σ_joint medians; "
            "do not pass the query set being ranked"
        ),
    )

    route = sub.add_parser(
        "route",
        help="Audit a deployment route and choose a smooth sequence of localizable viewpoints",
    )
    route.add_argument("route", help="CSV/JSON/JSONL waypoint table with x,y,z")
    route.add_argument(
        "--heatmap",
        required=True,
        help="detailed pose_health.csv produced by `sfm-diagnosis heatmap`",
    )
    route.add_argument("--output", required=True, help="output directory")
    route.add_argument(
        "--calibration",
        help="optional held-out table with health_score and binary success",
    )
    route.add_argument("--sample-spacing", type=float)
    route.add_argument("--max-heatmap-distance", type=float)
    route.add_argument("--smoothness-weight", type=float)
    route.add_argument("--task-forward-weight", type=float)
    route.add_argument("--weak-direction-weight", type=float)
    route.add_argument("--max-turn-deg-per-m", type=float)
    route.add_argument(
        "--no-turn-limit",
        action="store_true",
        help="disable the soft camera turn-rate limit",
    )
    route.add_argument("--enter-risk", type=float)
    route.add_argument("--exit-risk", type=float)
    route.add_argument("--min-segment-length", type=float)
    route.add_argument("--calibration-min-samples", type=int)
    matchability = sub.add_parser(
        "matchability",
        help="Build a per-landmark historical matchability table from correspondence events",
    )
    matchability.add_argument("events", help="CSV/JSON/JSONL query-landmark events")
    matchability.add_argument("--model", required=True, help="map input")
    _add_map_adapter_arg(matchability)
    matchability.add_argument("--output", required=True, help="output directory")

    pose = sub.add_parser("pose", help="Diagnose one camera pose")
    _add_model_pose_args(pose)
    _add_history_environment_args(pose)
    pose.add_argument("--include-points", action="store_true")
    pose.add_argument(
        "--matchability",
        help="correspondence events or landmark_matchability.csv",
    )
    pose.add_argument("--output")

    heat = sub.add_parser("heatmap", help="Build position/view localizability heatmaps")
    heat.add_argument("model")
    _add_map_adapter_arg(heat)
    heat.add_argument("--output", required=True, help="output directory")
    heat.add_argument("--logs")
    heat.add_argument("--spacing", type=float)
    heat.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
    )
    heat.add_argument("--orientation-mode", choices=["map", "yaw_pitch"])
    heat.add_argument(
        "--sample-mode",
        choices=["grid", "cameras"],
        help="grid over bounds, or one sample at each reconstructed camera center",
    )
    heat.add_argument("--orientations-per-position", type=int)
    heat.add_argument("--yaw-step", type=float)
    heat.add_argument("--pitches", nargs="+", type=float)
    heat.add_argument(
        "--matchability",
        help="correspondence events or landmark_matchability.csv",
    )
    _add_environment_args(heat)

    risk = sub.add_parser(
        "risk-ply",
        help="Write layered CloudCompare base/marker PLYs plus archival risk spheres",
    )
    risk.add_argument("model")
    _add_map_adapter_arg(risk)
    risk.add_argument("--output", required=True, help="output directory")
    risk.add_argument("--heatmap", help="pose_health.csv or heatmap directory")
    risk.add_argument("--weak-regions", help="weak-region JSON/CSV or analyze directory")
    risk.add_argument("--logs", help="optional localization CSV/JSON/JSONL")
    risk.add_argument(
        "--image-roles",
        help="optional image-name to role JSON/JSONL/CSV or frame manifest",
    )
    risk.add_argument(
        "--include-actloc-shadow",
        action="store_true",
        help="include explicit ActLoc-proxy shadow markers; still not authorized evidence",
    )
    risk.add_argument(
        "--filename",
        default="localization_risk_spheres.ply",
        help=(
            "base PLY name; writes *_cloudcompare_base.ply, per-class overlays, "
            "aggregated *_cloudcompare.ply, *_full.ply, and *_markers_mesh.ply"
        ),
    )
    risk.add_argument("--sphere-radius", type=float, help="override marker radius; default is camera-NN capped by camera path diagonal")
    risk.add_argument("--sphere-samples", type=int, help="override marker shell samples")

    repair = sub.add_parser(
        "repair",
        help="Suggest capture viewpoints for a diagnosed weak pose",
    )
    _add_model_pose_args(repair)
    _add_history_environment_args(repair)
    repair.add_argument("--top-k", type=int, default=8)
    repair.add_argument("--output")
    return p


def _add_model_pose_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model")
    _add_map_adapter_arg(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image-id", type=int, help="use a registered map image pose")
    g.add_argument(
        "--pose",
        nargs=7,
        type=float,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="camera center + camera-to-world quaternion (xyzw)",
    )


def _add_map_adapter_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map-adapter",
        "--backend",
        dest="map_adapter",
        required=True,
        help="Map input adapter: a built-in name or package.module:AdapterClass",
    )


def _load_map(args: argparse.Namespace) -> MapData:
    model = get_adapter(args.map_adapter).load(args.model)
    return map_model_to_map_data(model)


def _add_history_environment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--logs", help="CSV/JSONL actual localization logs")
    _add_environment_args(p)


def _add_environment_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mesh", help="triangle mesh for camera/sun ray-casting")
    p.add_argument("--sun-azimuth", type=float, help="Z-up map azimuth from +X, degrees")
    p.add_argument("--sun-elevation", type=float, help="Z-up map elevation, degrees")


def _resolve_pose(map_data: MapData, args) -> tuple[Pose, object | None]:
    if args.image_id is not None:
        matches = np.flatnonzero(map_data.image_ids == int(args.image_id))
        if not len(matches):
            raise ValueError(f"image id {args.image_id} is not a registered map image")
        i = int(matches[0])
        pose = Pose(map_data.image_centers[i], map_data.image_R_wc[i])
        intr = map_data.cameras.get(int(map_data.image_camera_ids[i]))
        return pose, intr
    values = np.asarray(args.pose, dtype=float)
    center = values[:3]
    R_wc = Rotation.from_quat(values[3:]).as_matrix()
    return Pose(center, R_wc), None


def _environment(args):
    mesh = getattr(args, "mesh", None)
    raycaster = MeshRaycaster(mesh) if mesh else None
    az = getattr(args, "sun_azimuth", None)
    el = getattr(args, "sun_elevation", None)
    if (az is None) ^ (el is None):
        raise ValueError("--sun-azimuth and --sun-elevation must be provided together")
    illumination = None
    if az is not None and el is not None:
        direction = sun_light_travel_direction_from_az_el(az, el)
        illumination = SunIllumination(direction, raycaster)
    return raycaster, illumination


def _load_config(path: str | None) -> dict:
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError("config root must be a mapping")
    return data


def _emit(payload: dict, output: str | None) -> None:
    if output:
        write_json(output, payload)
        print(str(Path(output).resolve()))
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
