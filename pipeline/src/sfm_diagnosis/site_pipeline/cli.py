from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..io import _json_default
from .config import PipelineConfig
from .pipeline import ApprovalRequired, SitePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="site-sfm-pipeline",
        description="Adopted all-sequence GlueMap + EDM retriangulation mapping and two-rate localization.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    mapped = commands.add_parser(
        "map",
        help="Adopted mapping: all-sequence GlueMap poses, then official EDM retriangulation",
    )
    mapped.add_argument("--site-name", required=True)
    mapped.add_argument("--corpus", type=Path, required=True)
    mapped.add_argument("--run", type=Path, required=True)
    mapped.add_argument("--intrinsics", type=Path, required=True)
    mapped.add_argument("--gluemap-root", type=Path, required=True)
    mapped.add_argument("--gluemap-config", type=Path, required=True)
    mapped.add_argument("--workspace-root", type=Path, required=True)
    mapped.add_argument("--megaloc-source", type=Path, required=True)
    mapped.add_argument("--megaloc-checkpoint", type=Path, required=True)
    mapped.add_argument("--edm-root", type=Path, required=True)
    mapped.add_argument("--edm-checkpoint", type=Path, required=True)
    mapped.add_argument("--edm-model-config", type=Path)
    mapped.add_argument("--edm-data-config", type=Path)
    mapped.add_argument("--probe-fps", type=float, default=4.0)
    mapped.add_argument("--baseline-fps", type=float, default=1.0)
    mapped.add_argument("--fast-fps", type=float, default=2.0)
    mapped.add_argument("--hover-fps", type=float, default=0.2)
    mapped.add_argument("--rotation-fps", type=float, default=2.0)
    mapped.add_argument("--min-gap-seconds", type=float, default=0.25)
    mapped.add_argument("--preprocess-only", action="store_true")
    mapped.add_argument("--resume", action="store_true")
    mapped.add_argument("--cell-px", type=float, default=2.0)

    localize = commands.add_parser(
        "localize",
        help="Adopted two-rate localization: KLT+PnP fast loop and MegaLoc+EDM reloc",
    )
    localize.add_argument("--map-model", type=Path, required=True)
    localize.add_argument("--keyframes", type=Path, required=True)
    localize.add_argument("--frames", type=Path, required=True)
    localize.add_argument("--output", type=Path, required=True)
    localize.add_argument("--localizer-config", type=Path, required=True)
    localize.add_argument("--localization-dir", type=Path, required=True)
    localize.add_argument("--edm-root", type=Path, required=True)
    localize.add_argument("--edm-checkpoint", type=Path, required=True)
    localize.add_argument("--edm-model-config", type=Path)
    localize.add_argument("--edm-data-config", type=Path)
    localize.add_argument("--reference-depth-dir", type=Path)
    localize.add_argument("--metric-scale", type=Path)
    localize.add_argument("--excluded-session")
    localize.add_argument("--gt-video-id")
    localize.add_argument("--route", default="stream")
    localize.add_argument("--reloc-matcher", default="edm",
                          choices=["edm", "xfeat_lg", "aliked_lg", "xfeat_mnn"])
    localize.add_argument("--sparse-max-keypoints", type=int, default=4096)
    localize.add_argument("--sparse-ref-cache", type=int, default=128)
    localize.add_argument("--resume", action="store_true")

    metric = commands.add_parser("metric-scale", help="Infer pinned offline MoGe-3 metric depth and calibrate the adopted map")
    metric.add_argument("--map-model", type=Path, required=True)
    metric.add_argument("--keyframes", type=Path, required=True)
    metric.add_argument("--output", type=Path, required=True)
    metric.add_argument("--moge-root", type=Path, required=True)
    metric.add_argument("--moge-python", type=Path, required=True)
    metric.add_argument("--moge-checkpoint", type=Path, required=True)
    metric.add_argument("--moge-model", required=True, choices=("Ruicheng/moge-3-vitl", "Ruicheng/moge-3-vitg"),
                        help="Explicit official Large or Giant checkpoint identity; never inferred from a filename")
    metric.add_argument("--dense", action="store_true", help="Verify and fuse supported surfaces offline")
    metric.add_argument("--resume", action="store_true")
    metric.add_argument("--photo-evidence", type=Path)
    metric.add_argument("--mvs-evidence", type=Path)

    revalidate = commands.add_parser(
        "metric-revalidate", help="Revalidate immutable offline raw depth without loading a neural model"
    )
    revalidate.add_argument("--map-model", type=Path, required=True)
    revalidate.add_argument("--keyframes", type=Path, required=True)
    revalidate.add_argument("--output", type=Path, required=True)
    revalidate.add_argument("--baseline-dir", type=Path, required=True)
    revalidate.add_argument("--baseline-inventory", type=Path, required=True,
                            help="Trusted frozen-raw-cache-inventory/v1 request and raw-byte digests")
    revalidate.add_argument("--dense", action="store_true")
    revalidate.add_argument("--photo-evidence", type=Path)
    revalidate.add_argument("--mvs-evidence", type=Path)

    amb3r = commands.add_parser(
        "amb3r-scale", help="Estimate metres per map unit from AMB3R star baselines; optionally export a metric model"
    )
    amb3r.add_argument("--star-cache", type=Path, required=True)
    amb3r.add_argument("--map-model", type=Path, required=True)
    amb3r.add_argument("--keyframes", type=Path, required=True)
    amb3r.add_argument("--output", type=Path, required=True)
    amb3r.add_argument("--moge-sidecar", type=Path)
    amb3r.add_argument("--export-dir", type=Path)
    initialize = commands.add_parser("init", help="Create immutable corpus inventory")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--corpus", type=Path, required=True)
    initialize.add_argument("--metadata", type=Path)
    initialize.add_argument("--output", type=Path, required=True)
    initialize.add_argument("--force", action="store_true")

    run = commands.add_parser("run", help="Run or resume pipeline stages")
    run.add_argument("--run", type=Path, required=True)
    run.add_argument("--from-stage")
    run.add_argument("--to-stage")
    run.add_argument("--force", action="store_true")

    status = commands.add_parser("status", help="Show stage and approval status")
    status.add_argument("--run", type=Path, required=True)

    approve = commands.add_parser("approve-final", help="Approve exact Final input")
    approve.add_argument("--run", type=Path, required=True)
    approve.add_argument("--decision-sha", required=True)
    approve.add_argument("--approver", required=True)

    export = commands.add_parser("export", help="Regenerate final manifests")
    export.add_argument("--run", type=Path, required=True)

    backfill = commands.add_parser("backfill", help="Import legacy manifests as candidate evidence")
    backfill.add_argument("--run", type=Path, required=True)
    backfill.add_argument("--legacy-run", type=Path, required=True)

    normalize = commands.add_parser(
        "normalize-selection",
        help="Remove zero-degree keyframes before final approval",
    )
    normalize.add_argument("--run", type=Path, required=True)

    enrich = commands.add_parser(
        "enrich-bridge",
        help="Add evidence-selected bridge keyframes and invalidate prior products",
    )
    enrich.add_argument("--run", type=Path, required=True)
    enrich.add_argument("--segment", action="append", default=[])
    enrich.add_argument("--keyframe", action="append", default=[])
    enrich.add_argument("--attempt-name", required=True)

    layers = commands.add_parser(
        "materialize-layers",
        help="Create robust base geometry and retain the dense localization layer",
    )
    layers.add_argument("--run", type=Path, required=True)
    layers.add_argument("--export-ply", action="store_true")

    validate = commands.add_parser(
        "validate-layers",
        help="Run strict robust/dense localization and fuse the results",
    )
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--maximum-position-normalized", type=float, default=0.02)
    validate.add_argument("--maximum-rotation-deg", type=float, default=2.0)
    validate.add_argument("--target-rate", type=float, default=0.95)
    validate.add_argument("--outer-holdout-frozen", action="store_true")

    fuse = commands.add_parser(
        "fuse-localization",
        help="Fuse strict robust/dense localization validations",
    )
    fuse.add_argument("--run", type=Path, required=True)
    fuse.add_argument("--robust-validation", type=Path, required=True)
    fuse.add_argument("--dense-validation", type=Path, required=True)
    fuse.add_argument("--scene-scale", type=float)
    fuse.add_argument("--maximum-position-normalized", type=float, default=0.02)
    fuse.add_argument("--maximum-rotation-deg", type=float, default=2.0)
    fuse.add_argument("--target-rate", type=float, default=0.95)
    fuse.add_argument("--outer-holdout-frozen", action="store_true")

    refinement = commands.add_parser(
        "refine-map", help="Run an immutable PixSfM or Dense-SfM candidate refinement"
    )
    refinement.add_argument(
        "--backend", choices=("pixsfm", "densesfm-refine", "densesfm-full"), required=True
    )
    refinement.add_argument("--input-model", type=Path, required=True)
    refinement.add_argument("--images", type=Path, required=True)
    refinement.add_argument("--pairs", type=Path)
    refinement.add_argument("--intrinsics", type=Path, required=True)
    refinement.add_argument("--run", type=Path, required=True)
    refinement.add_argument("--cache-dir", type=Path)
    refinement.add_argument("--runtime-python", type=Path, required=True)
    refinement.add_argument("--runtime-root", type=Path)
    refinement.add_argument("--continuation-receipt", type=Path)
    refinement.add_argument("--timeout-seconds", type=int, default=86_400)
    refinement.add_argument("--max-cache-gb", type=int, default=500)
    refinement.add_argument("--pixsfm-patch-size", type=int, default=4)
    refinement.add_argument("--max-reprojection-error-px", type=float, default=3.0)
    refinement.add_argument("--minimum-triangulation-angle-deg", type=float, default=1.5)
    refinement.add_argument("--minimum-track-length", type=int, default=3)
    refinement.add_argument("--bundle-adjustment-iterations", type=int, default=100)
    refinement.add_argument("--bundle-adjustment-threads", type=int, default=8)
    refinement.add_argument("--dry-run", action="store_true")
    refinement.add_argument("--resume", action="store_true")

    mvroma = commands.add_parser(
        "mvroma-augment", help="Add strictly gated MV-RoMa tracks to an immutable map copy"
    )
    mvroma.add_argument("--scope", choices=("targeted", "full"), required=True)
    mvroma.add_argument("--input-model", type=Path, required=True)
    mvroma.add_argument("--images", type=Path, required=True)
    mvroma.add_argument("--selection", type=Path, required=True)
    mvroma.add_argument("--intrinsics", type=Path, required=True)
    mvroma.add_argument("--run", type=Path, required=True)
    mvroma.add_argument("--runtime-python", type=Path, required=True)
    mvroma.add_argument("--runtime-root", type=Path, required=True)
    mvroma.add_argument("--weight", type=Path, required=True)
    mvroma.add_argument("--continuation-receipt", type=Path)
    mvroma.add_argument("--coarse-height", type=int, default=378)
    mvroma.add_argument("--coarse-width", type=int, default=672)
    mvroma.add_argument("--target-height", type=int, default=756)
    mvroma.add_argument("--target-width", type=int, default=1344)
    mvroma.add_argument("--certainty-threshold", type=float, default=0.5)
    mvroma.add_argument("--max-samples-per-group", type=int, default=1024)
    mvroma.add_argument("--timeout-seconds", type=int, default=86_400)
    mvroma.add_argument("--sparse-video", action="append", default=[])
    mvroma.add_argument("--weak-keyframe", action="append", default=[])
    mvroma.add_argument("--dry-run", action="store_true")
    mvroma.add_argument("--resume", action="store_true")

    direct = commands.add_parser(
        "direct-map",
        help="Map every adaptively sampled sequence in one native GLUEMAP job",
    )
    direct.add_argument("--site-name", required=True)
    direct.add_argument("--corpus", type=Path, required=True)
    direct.add_argument("--run", type=Path, required=True)
    direct.add_argument("--intrinsics", type=Path, required=True)
    direct.add_argument("--gluemap-root", type=Path, required=True)
    direct.add_argument("--gluemap-config", type=Path, required=True)
    direct.add_argument("--workspace-root", type=Path, required=True)
    direct.add_argument("--megaloc-source", type=Path, required=True)
    direct.add_argument("--megaloc-checkpoint", type=Path, required=True)
    direct.add_argument("--edm-root", type=Path)
    direct.add_argument("--edm-checkpoint", type=Path)
    direct.add_argument("--probe-fps", type=float, default=4.0)
    direct.add_argument("--baseline-fps", type=float, default=1.0)
    direct.add_argument("--fast-fps", type=float, default=2.0)
    direct.add_argument("--hover-fps", type=float, default=0.2)
    direct.add_argument("--rotation-fps", type=float, default=2.0)
    direct.add_argument("--min-gap-seconds", type=float, default=0.25)
    direct.add_argument("--preprocess-only", action="store_true")
    direct.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "map":
            from .adopted_pipeline import run_adopted_mapping
            from .direct_mapping import DirectMappingRequest, DirectMappingRuntime
            from .preprocessing import DirectSamplingPolicy

            request = DirectMappingRequest(
                site_name=args.site_name,
                corpus_root=args.corpus,
                run_dir=args.run,
                intrinsics_path=args.intrinsics,
                policy=DirectSamplingPolicy(
                    probe_fps=args.probe_fps,
                    baseline_fps=args.baseline_fps,
                    fast_fps=args.fast_fps,
                    hover_fps=args.hover_fps,
                    rotation_fps=args.rotation_fps,
                    min_gap_seconds=args.min_gap_seconds,
                ),
            )
            runtime = DirectMappingRuntime(
                gluemap_root=args.gluemap_root,
                base_config_path=args.gluemap_config,
                workspace_root=args.workspace_root,
                megaloc_source=args.megaloc_source,
                megaloc_checkpoint=args.megaloc_checkpoint,
                edm_root=args.edm_root,
                edm_checkpoint=args.edm_checkpoint,
            )
            payload = run_adopted_mapping(
                request,
                runtime,
                resume=args.resume,
                preprocess_only=args.preprocess_only,
                edm_model_config=args.edm_model_config,
                edm_data_config=args.edm_data_config,
                cell_px=args.cell_px,
            )
        elif args.command == "metric-scale":
            from .metric_scale import run_metric_scale

            payload = run_metric_scale(
                map_model=args.map_model, keyframes=args.keyframes, output_dir=args.output,
                moge_root=args.moge_root, moge_python=args.moge_python,
                checkpoint=args.moge_checkpoint, model_id=args.moge_model, resume=args.resume, dense=args.dense,
                photo_evidence_path=args.photo_evidence,
                mvs_evidence_path=args.mvs_evidence,
            )
        elif args.command == "amb3r-scale":
            import sys

            from .amb3r_scale import STATUS_OK, run_amb3r_scale
            from .metric_export import export_metric_model

            payload = run_amb3r_scale(
                star_cache=args.star_cache, map_model=args.map_model, keyframes=args.keyframes,
                output=args.output, moge_sidecar=args.moge_sidecar,
            )
            if payload["status"] != STATUS_OK:
                print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
                print(f"{payload['status']}: scale sidecar written to {args.output}; metric model not exported",
                      file=sys.stderr)
                return 2
            if args.export_dir is not None:
                payload["export"] = export_metric_model(
                    args.map_model, args.export_dir, payload["meters_per_map_unit"], args.output)
            payload["note"] = "model-estimated scale, no external ground truth"
        elif args.command == "metric-revalidate":
            from .metric_scale import revalidate_metric_geometry

            inventory = json.loads(args.baseline_inventory.read_text(encoding="utf-8"))
            if inventory.get("schema") != "frozen-raw-cache-inventory/v1":
                raise ValueError("Unsupported frozen raw-depth inventory")
            payload = revalidate_metric_geometry(
                map_model=args.map_model, keyframes=args.keyframes, output_dir=args.output,
                baseline_request=args.baseline_dir / "request.json",
                baseline_request_sha256=inventory["baseline_request_sha256"],
                baseline_request_file_sha256=inventory["baseline_request_file_sha256"],
                baseline_raw_depth_dir=args.baseline_dir / "raw_depth",
                baseline_raw_hashes=inventory["baseline_raw_hashes"],
                dense=args.dense,
                photo_evidence_path=args.photo_evidence,
                mvs_evidence_path=args.mvs_evidence,
            )
        elif args.command == "localize":
            from .adopted_pipeline import DEFAULT_EDM_DATA_CONFIG, DEFAULT_EDM_MODEL_CONFIG
            from .deploy_loop import DeployConfig, run_deploy_loop

            payload = run_deploy_loop(
                DeployConfig(
                    map_model=args.map_model,
                    keyframes=args.keyframes,
                    frames_dir=args.frames,
                    output_dir=args.output,
                    localizer_config=args.localizer_config,
                    localization_dir=args.localization_dir,
                    edm_repo=args.edm_root,
                    edm_checkpoint=args.edm_checkpoint,
                    edm_model_config=args.edm_model_config or DEFAULT_EDM_MODEL_CONFIG,
                    edm_data_config=args.edm_data_config or DEFAULT_EDM_DATA_CONFIG,
                    reference_depth_dir=args.reference_depth_dir,
                    metric_scale=args.metric_scale,
                    excluded_session=args.excluded_session,
                    gt_video_id=args.gt_video_id,
                    route=args.route,
                    reloc_matcher=args.reloc_matcher,
                    sparse_max_keypoints=args.sparse_max_keypoints,
                    sparse_ref_cache=args.sparse_ref_cache,
                )
            )
        elif args.command == "direct-map":
            from .direct_mapping import (
                DirectMappingRequest,
                DirectMappingRuntime,
                run_direct_mapping,
            )
            from .preprocessing import DirectSamplingPolicy

            request = DirectMappingRequest(
                site_name=args.site_name,
                corpus_root=args.corpus,
                run_dir=args.run,
                intrinsics_path=args.intrinsics,
                policy=DirectSamplingPolicy(
                    probe_fps=args.probe_fps,
                    baseline_fps=args.baseline_fps,
                    fast_fps=args.fast_fps,
                    hover_fps=args.hover_fps,
                    rotation_fps=args.rotation_fps,
                    min_gap_seconds=args.min_gap_seconds,
                ),
            )
            runtime = DirectMappingRuntime(
                gluemap_root=args.gluemap_root,
                base_config_path=args.gluemap_config,
                workspace_root=args.workspace_root,
                megaloc_source=args.megaloc_source,
                megaloc_checkpoint=args.megaloc_checkpoint,
                edm_root=args.edm_root,
                edm_checkpoint=args.edm_checkpoint,
            )
            payload = run_direct_mapping(
                request,
                runtime,
                resume=args.resume,
                preprocess_only=args.preprocess_only,
            )
        elif args.command == "init":
            pipeline = SitePipeline(PipelineConfig.from_toml(args.config))
            result = pipeline.initialize(
                args.corpus,
                args.output,
                metadata=args.metadata,
                force=args.force,
            )
            payload = _result_payload(result)
        elif args.command == "backfill":
            from .migration import backfill_legacy_run

            receipt = backfill_legacy_run(args.legacy_run, args.run)
            payload = {"backfill_receipt": str(receipt)}
        elif args.command == "normalize-selection":
            from .selection_tools import normalize_run_selection

            payload = {"normalization_receipt": str(normalize_run_selection(args.run))}
        elif args.command == "enrich-bridge":
            from .selection_tools import enrich_run_selection

            payload = {
                "bridge_enrichment_receipt": str(
                    enrich_run_selection(
                        args.run,
                        target_segments=set(args.segment),
                        target_keyframes=set(args.keyframe),
                        attempt_name=args.attempt_name,
                    )
                )
            }
        elif args.command == "materialize-layers":
            from .release import materialize_layers

            payload = {
                "robust_filter_receipt": str(
                    materialize_layers(args.run, export_ply=args.export_ply)
                )
            }
        elif args.command == "validate-layers":
            from .release import validate_localization_layers

            payload = {
                "ensemble_validation_receipt": str(
                    validate_localization_layers(
                        args.run,
                        maximum_position_normalized=args.maximum_position_normalized,
                        maximum_rotation_deg=args.maximum_rotation_deg,
                        target_rate=args.target_rate,
                        outer_holdout_frozen=args.outer_holdout_frozen,
                    )
                )
            }
        elif args.command == "fuse-localization":
            from .release import fuse_localization_results

            payload = {
                "ensemble_validation_receipt": str(
                    fuse_localization_results(
                        args.run,
                        args.robust_validation,
                        args.dense_validation,
                        scene_scale=args.scene_scale,
                        maximum_position_normalized=args.maximum_position_normalized,
                        maximum_rotation_deg=args.maximum_rotation_deg,
                        target_rate=args.target_rate,
                        outer_holdout_frozen=args.outer_holdout_frozen,
                    )
                )
            }
        elif args.command == "refine-map":
            from .map_refinement import RefinementRequest, run_refinement
            from .robust_filter import RobustFilterConfig

            cache_dir = args.cache_dir or (args.run.parent / "refinement_cache" / args.backend)
            request = RefinementRequest(
                backend=args.backend,
                input_model=args.input_model,
                images=args.images,
                intrinsics=args.intrinsics,
                pairs=args.pairs,
                run_dir=args.run,
                cache_dir=cache_dir,
                runtime_python=args.runtime_python,
                runtime_root=args.runtime_root,
                continuation_receipt=args.continuation_receipt,
                timeout_seconds=args.timeout_seconds,
                max_cache_gb=args.max_cache_gb,
                pixsfm_patch_size=args.pixsfm_patch_size,
                robust_filter=RobustFilterConfig(
                    max_reprojection_error_px=args.max_reprojection_error_px,
                    minimum_triangulation_angle_deg=args.minimum_triangulation_angle_deg,
                    minimum_track_length=args.minimum_track_length,
                    bundle_adjustment_iterations=args.bundle_adjustment_iterations,
                    bundle_adjustment_threads=args.bundle_adjustment_threads,
                ),
            )
            payload = run_refinement(request, dry_run=args.dry_run, resume=args.resume)
        elif args.command == "mvroma-augment":
            from river_mvroma.runner import (
                RIVER_V3_SPARSE_VIDEOS,
                RIVER_V3_WEAK_KEYFRAMES,
                MVRoMaRequest,
                run_mvroma,
            )

            request = MVRoMaRequest(
                scope=args.scope,
                input_model=args.input_model,
                images=args.images,
                selection=args.selection,
                intrinsics=args.intrinsics,
                run_dir=args.run,
                runtime_python=args.runtime_python,
                runtime_root=args.runtime_root,
                weight=args.weight,
                continuation_receipt=args.continuation_receipt,
                coarse_size=(args.coarse_height, args.coarse_width),
                target_size=(args.target_height, args.target_width),
                certainty_threshold=args.certainty_threshold,
                max_samples_per_group=args.max_samples_per_group,
                sparse_video_ids=tuple(args.sparse_video) or RIVER_V3_SPARSE_VIDEOS,
                weak_keyframe_ids=tuple(args.weak_keyframe) or RIVER_V3_WEAK_KEYFRAMES,
                timeout_seconds=args.timeout_seconds,
            )
            payload = run_mvroma(request, dry_run=args.dry_run, resume=args.resume)
        else:
            pipeline = SitePipeline.open(args.run)
            if args.command == "run":
                payload = _result_payload(
                    pipeline.run(
                        args.run,
                        from_stage=args.from_stage,
                        to_stage=args.to_stage,
                        force=args.force,
                    )
                )
            elif args.command == "status":
                payload = pipeline.status(args.run)
            elif args.command == "approve-final":
                approval = pipeline.approve_final(
                    args.run, args.decision_sha, approver=args.approver
                )
                payload = {"approval": str(approval)}
            else:
                payload = _result_payload(pipeline.export(args.run))
    except ApprovalRequired as error:
        print(json.dumps({"status": "APPROVAL_REQUIRED", "reason": str(error)}, indent=2))
        return 3
    except (OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAILED", "error_type": type(error).__name__, "reason": str(error)},
                indent=2,
            )
        )
        return 2
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))
    return 0


def _result_payload(result) -> dict[str, object]:
    return {
        "run_dir": str(result.run_dir),
        "status": result.status,
        "receipt": str(result.receipt),
        "stages": [{"name": stage.name, "status": stage.status} for stage in result.stages],
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
