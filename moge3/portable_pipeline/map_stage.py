#!/usr/bin/env python3
"""Portable multi-video mapping stage: keyframes + MoGe3/MegaLoc GlueMap.

Reads arbitrary mapping videos (or image-sequence directories), extracts
keyframes with the proven adaptive sampler, reconstructs them with the
pinned GlueMap checkout (MegaLoc retrieval, Pi3 backbone, MoGe-3 depth,
SIFT refinement), and exports the shared data contract::

    <output>/keyframes/images/<source>/frame_<index>.jpg
    <output>/keyframes/keyframes.jsonl
    <output>/model/                      (COLMAP binary model)
    <output>/sparse_rgb.ply              (original-RGB sparse cloud)
    <output>/intrinsics.json             (run calibration + query contract)
    <output>/map_receipt.json            (machine-readable run receipt)

Calibration is explicit: pass --intrinsics (recommended), explicitly opt
into --auto_focal/--auto_resolution, or pass --auto_calibrate for the
documented video-only heuristic (focal 1.2*max(W,H) at the first source
resolution, refined by bundle adjustment; recorded APPROXIMATE and never
presented as metrically calibrated). Arbitrary cameras never silently
assume any focal length. Input frames must already be undistorted PINHOLE;
distorted input is rejected, not silently misused. Full runs write
output/intrinsics.json from the REFINED reconstruction cameras so
localization uses actual map intrinsics, never the initial seed.

All executables, repos and checkpoints are configurable via CLI flags or
MAP_* environment variables; nothing references the source machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapping.calibration import (  # noqa: E402
    CALIBRATION_SCHEMA,
    Calibration,
    auto_focal_for_resolution,
    calibration_from_auto_focal,
    calibration_matrix_for_resolution,
    load_calibration,
    query_calibration_contract,
    refined_query_contract,
)
from mapping.frames import analyze_source, extract_frames  # noqa: E402
from mapping.gluemap_runner import build_job_payload, run_gluemap_job  # noqa: E402
from mapping.intrinsics_seed import write_intrinsics_seed  # noqa: E402
from mapping.inventory import SourceSpec, discover_sources  # noqa: E402
from mapping.io_utils import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from mapping.keyframes import SamplingPolicy, plan_keyframes, sanitize_frames  # noqa: E402
from mapping.pose_only import assert_pose_only_observation_free  # noqa: E402
from mapping.rgb_ply import export_original_rgb_ply  # noqa: E402

SCHEMA_VERSION = 1


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="mapping videos, image files, or directories holding them",
    )
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument(
        "--intrinsics", default=_env("MAP_INTRINSICS") or None,
        help="explicit calibration JSON (recommended). " + CALIBRATION_SCHEMA,
    )
    parser.add_argument(
        "--auto_focal", type=float, default=None,
        help="EXPLICIT APPROXIMATE fallback: focal length in pixels at "
        "--auto_resolution. Recorded as APPROXIMATE_AUTO_FOCAL; the map must "
        "not be presented as metrically calibrated.",
    )
    parser.add_argument(
        "--auto_resolution", default=None,
        help="WxH resolution the --auto_focal value is defined at",
    )
    parser.add_argument(
        "--auto_calibrate", action="store_true",
        help="video-only mode: derive the initial focal as "
        "1.2*max(width,height) at the first source resolution, refine it "
        "with bundle adjustment, and record APPROXIMATE_AUTO_CALIBRATE. "
        "Explicit --intrinsics always overrides this flag.",
    )
    parser.add_argument(
        "--gluemap_root", default=_env("MAP_GLUEMAP_ROOT") or None,
        help="GlueMap checkout root (contains gluemap/; default: repo parent)",
    )
    parser.add_argument(
        "--gluemap_python", default=_env("MAP_GLUEMAP_PYTHON") or None,
        help="interpreter with the GlueMap stack (default: this interpreter)",
    )
    parser.add_argument(
        "--path_feedforward", default=_env("MAP_PATH_FEEDFORWARD") or "",
        help="Pi3 checkpoint (.safetensors)",
    )
    parser.add_argument(
        "--path_retrieval", default=_env("MAP_PATH_RETRIEVAL") or "",
        help="MegaLoc checkpoint",
    )
    parser.add_argument(
        "--path_tracker", default=_env("MAP_PATH_TRACKER") or "",
        help="VGGSfM tracker checkpoint",
    )
    parser.add_argument(
        "--path_dg", default=_env("MAP_PATH_DG") or "",
        help="Doppelgangers++ checkpoint",
    )
    parser.add_argument(
        "--path_depth", default=_env("MAP_PATH_DEPTH") or "",
        help="MoGe-3 checkpoint (local model.pt or HF repo id)",
    )
    parser.add_argument(
        "--moge_python", default=_env("MAP_MOGE_PYTHON") or "",
        help="interpreter for the MoGe-3 worker (empty = GlueMap interpreter)",
    )
    parser.add_argument(
        "--ffprobe", default=_env("MAP_FFPROBE", "ffprobe"),
        help="ffprobe executable for video probing",
    )
    parser.add_argument("--probe_fps", type=float, default=4.0)
    parser.add_argument("--baseline_fps", type=float, default=1.0)
    parser.add_argument("--fast_fps", type=float, default=2.0)
    parser.add_argument("--hover_fps", type=float, default=0.2)
    parser.add_argument("--rotation_fps", type=float, default=2.0)
    parser.add_argument("--min_gap_seconds", type=float, default=0.25)
    parser.add_argument("--num_track_per_img", type=int, default=512)
    parser.add_argument("--num_neighbors", type=int, default=20)
    parser.add_argument("--num_neighbors_sequential", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--retrieval_batch_size", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument(
        "--skip_doppelgangers", action="store_true",
        help="treat all retrieval pairs as valid (scenes without repetition)",
    )
    parser.add_argument(
        "--coarse_only", action="store_true",
        help="stop after global mapping; skip SIFT refinement",
    )
    parser.add_argument(
        "--refine_intrinsics", action=argparse.BooleanOptionalAction,
        default=None,
        help="let BA refine intrinsics (default: False with --intrinsics, "
        "True with --auto_focal)",
    )
    parser.add_argument(
        "--sift_device", default=_env("MAP_SIFT_DEVICE", "cuda"),
        help="device for SIFT extraction/matching (cuda, cuda:N, cpu)",
    )
    parser.add_argument(
        "--exclude_pose_only", action="store_true",
        help="exclude pure-rotation POSE_ONLY frames from reconstruction "
        "instead of verifying them observation-free (drops graph edges)",
    )
    parser.add_argument(
        "--keyframes_only", action="store_true",
        help="extract keyframes and stop before GlueMap reconstruction",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed preprocessing and GlueMap caches in --output",
    )
    return parser


def parse_auto_resolution(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        width_text, height_text = value.lower().split("x", 1)
        return int(width_text), int(height_text)
    except ValueError as error:
        raise ValueError("--auto_resolution must look like WIDTHxHEIGHT") from error


def default_gluemap_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    if args.intrinsics and (args.auto_focal is not None or args.auto_calibrate):
        raise ValueError(
            "explicit --intrinsics overrides auto modes; pass only one "
            "calibration route"
        )
    if args.auto_focal is not None and args.auto_calibrate:
        raise ValueError("pass --auto_focal or --auto_calibrate, not both")
    policy = SamplingPolicy(
        probe_fps=args.probe_fps,
        baseline_fps=args.baseline_fps,
        fast_fps=args.fast_fps,
        hover_fps=args.hover_fps,
        rotation_fps=args.rotation_fps,
        min_gap_seconds=args.min_gap_seconds,
    )
    policy.validate()

    keyframes_dir = output / "keyframes"
    images_root = keyframes_dir / "images"
    keyframes_path = keyframes_dir / "keyframes.jsonl"
    work_dir = output / "work"
    preprocessing_receipt = work_dir / "preprocessing.json"

    sources = discover_sources(list(args.inputs), ffprobe=args.ffprobe)
    if args.auto_calibrate:
        first = sources[0]
        focal = auto_focal_for_resolution(first.width, first.height)
        calibration = calibration_from_auto_focal(
            focal, (first.width, first.height),
            source="APPROXIMATE_AUTO_CALIBRATE",
        )
    else:
        calibration = load_calibration(
            args.intrinsics,
            auto_focal=args.auto_focal,
            auto_resolution=parse_auto_resolution(args.auto_resolution),
        )
    for source in sources:
        calibration_matrix_for_resolution(
            calibration, target_size=(source.width, source.height)
        )

    keyframe_rows: list[dict[str, Any]]
    per_source: list[dict[str, Any]]
    forced_ids: list[tuple[str, str]]
    if args.resume and preprocessing_receipt.is_file() and keyframes_path.is_file():
        saved = json.loads(preprocessing_receipt.read_text(encoding="utf-8"))
        keyframe_rows = read_jsonl(keyframes_path)
        per_source = saved.get("per_source", [])
        forced_ids = [tuple(pair) for pair in saved.get("forced_pairs", [])]
        _verify_resume_images(keyframe_rows, images_root)
    else:
        if output.exists() and not args.resume and any(output.iterdir()):
            raise ValueError(
                f"output {output} is not empty; pass --resume or use an empty directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        keyframe_rows, per_source, forced_ids = extract_keyframes(
            sources, calibration, policy, images_root, keyframes_path, work_dir
        )
        write_json(preprocessing_receipt, {
            "schema_version": SCHEMA_VERSION,
            "sources": [spec.manifest_row() for spec in sources],
            "per_source": per_source,
            "keyframes": len(keyframe_rows),
            "forced_pairs": [list(pair) for pair in forced_ids],
        })

    write_json(output / "intrinsics.json", {
        **calibration.as_dict(),
        "calibration_path": calibration.path,
        "query_contract": query_calibration_contract(
            calibration, (sources[0].width, sources[0].height)
        ),
    })
    if args.keyframes_only:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "KEYFRAMES_ONLY",
            "calibration_source": calibration.source,
            "keyframes": len(keyframe_rows),
            "per_source": per_source,
        }
        write_json(output / "map_receipt.json", receipt)
        return receipt

    gluemap_root = Path(args.gluemap_root or default_gluemap_root()).expanduser().resolve()
    if not (gluemap_root / "gluemap").is_dir():
        raise ValueError(f"gluemap_root has no gluemap/ package: {gluemap_root}")
    for flag in (
        "path_feedforward", "path_retrieval", "path_tracker", "path_dg", "path_depth",
    ):
        value = getattr(args, flag)
        if not value:
            raise ValueError(
                f"--{flag} is required (or MAP_{flag.upper()}); "
                "checkpoints are never defaulted to source-machine paths"
            )

    model_dir = output / "model"
    name_by_id = {str(row["keyframe_id"]): str(row["output_name"]) for row in keyframe_rows}
    selected_rows = list(keyframe_rows)
    if args.exclude_pose_only:
        selected_rows = [
            row for row in keyframe_rows if row.get("mapping_mode") != "POSE_ONLY"
        ]
        if not selected_rows:
            raise ValueError("--exclude_pose_only removed every keyframe")
    selected_ids = {str(row["keyframe_id"]) for row in selected_rows}
    forced_names = [
        (name_by_id[left], name_by_id[right])
        for left, right in forced_ids
        if left in name_by_id and right in name_by_id
        and left in selected_ids and right in selected_ids
    ]
    forced_pairs_path = work_dir / "forced_pairs.txt"
    forced_pairs_path.write_text(
        "".join(f"{left} {right}\n" for left, right in forced_names), encoding="utf-8"
    )

    dimensions = {spec.source_id: (spec.width, spec.height) for spec in sources}
    seed_dir = write_intrinsics_seed(
        selected_rows,
        dimensions_by_source=dimensions,
        calibration=calibration,
        output_dir=work_dir / "intrinsics_seed",
    )
    job_workspace = work_dir / "gluemap_job"
    job_images = job_workspace / "images"
    _materialize_images(selected_rows, job_images)
    gluemap_config = {
        "chosen_model": "pi3",
        "path_feedforward": args.path_feedforward,
        "path_retrieval": args.path_retrieval,
        "path_tracker": args.path_tracker,
        "path_dg": args.path_dg,
        "path_depth": args.path_depth,
        "moge_python": args.moge_python,
        "images_path": str(job_images),
        "write_path": str(job_workspace / "gluemap"),
        "temp_path": str(job_workspace / "tmp"),
        "camera_model": "PINHOLE",
        "intrinsics_mode": "SHARED",
        "gt_intrinsics_path": str(seed_dir),
        "refine_intrinsics": (
            args.refine_intrinsics
            if args.refine_intrinsics is not None
            else calibration.source != "EXPLICIT_CALIBRATION"
        ),
        "num_track_per_img": args.num_track_per_img,
        "num_neighbors": args.num_neighbors,
        "num_neighbors_sequential": args.num_neighbors_sequential,
        "batch_size": args.batch_size,
        "retrieval_batch_size": args.retrieval_batch_size,
        "num_workers": args.num_workers,
        "checkpoint_every": args.checkpoint_every,
        "valid_pose_threshold": 0.05,
        "valid_dg_threshold": 0.8,
        "force_load": bool(args.resume),
        "rerun_from": None,
        "coarse_only": bool(args.coarse_only),
        "use_dummy_tracks": False,
        "skip_doppelgangers": bool(args.skip_doppelgangers),
        "max_num_tracks": None,
        "is_multi_sequence": True,
        "is_sequential": True,
        "subfolder_regex": ".*",
        "extra_pairs_path": str(forced_pairs_path),
        "sift_device": args.sift_device,
        "save_result": True,
        "cpu": False,
    }
    job_path = job_workspace / "job.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    pose_only_names = [
        str(row["output_name"]) for row in selected_rows
        if row.get("mapping_mode") == "POSE_ONLY"
    ]
    write_json(job_path, build_job_payload(
        gluemap_root=gluemap_root,
        workspace=job_workspace,
        config=gluemap_config,
        forced_pairs_path=forced_pairs_path if forced_names else None,
        pose_only_names=pose_only_names,
        mapping_package_parent=Path(__file__).resolve().parent,
    ))
    job_result = run_gluemap_job(
        gluemap_root=gluemap_root,
        job_path=job_path,
        python=args.gluemap_python or None,
        log_path=work_dir / "gluemap.log",
    )
    prediction_dir = job_result.get("prediction_dir")
    if not prediction_dir:
        raise RuntimeError("GlueMap produced no reconstruction directory")
    built_model = job_workspace / "gluemap" / prediction_dir
    if not _complete_model(built_model):
        raise RuntimeError(f"GlueMap model is incomplete: {built_model}")
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(built_model, model_dir)

    verification = verify_model(model_dir, selected_rows, exclude_pose_only=args.exclude_pose_only)
    # Full runs MUST publish the refined reconstruction cameras as the query
    # contract: localization uses actual map intrinsics, never the seed.
    refined_cameras = read_model_cameras(model_dir)
    refined_intrinsics_flag = bool(gluemap_config["refine_intrinsics"])
    write_json(output / "intrinsics.json", {
        "schema_version": SCHEMA_VERSION,
        "seed_calibration": {
            **calibration.as_dict(),
            "calibration_path": calibration.path,
        },
        "refined_cameras": refined_cameras,
        "query_contract": refined_query_contract(
            refined_cameras,
            calibration_source=calibration.source,
            refined_intrinsics=refined_intrinsics_flag,
        ),
    })
    rgb_receipt = export_original_rgb_ply(model_dir, keyframes_path, output / "sparse_rgb.ply")
    write_json(output / "sparse_rgb_receipt.json", rgb_receipt)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "MAP_BUILT",
        "calibration_source": calibration.source,
        "refined_intrinsics": refined_intrinsics_flag,
        "keyframes": len(keyframe_rows),
        "reconstructed_images": len(selected_rows),
        "per_source": per_source,
        "forced_pairs": len(forced_names),
        "pose_only_names": pose_only_names,
        "model": str(model_dir),
        "sparse_rgb_ply": rgb_receipt,
        "verification": verification,
        "gluemap_prediction_dir": prediction_dir,
        "gluemap_timing": job_result.get("timing", {}),
        "pose_only_mask": job_result.get("pose_only_mask"),
    }
    write_json(output / "map_receipt.json", receipt)
    return receipt


def extract_keyframes(
    sources: list[SourceSpec],
    calibration: Calibration,
    policy: SamplingPolicy,
    images_root: Path,
    keyframes_path: Path,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]]]:
    frame_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    forced_ids: list[tuple[str, str]] = []
    per_source: list[dict[str, Any]] = []
    for source in sources:
        matrix = calibration_matrix_for_resolution(
            calibration, target_size=(source.width, source.height)
        )
        analyzed = analyze_source(source, fps=policy.probe_fps, intrinsics=matrix)
        for row in analyzed:
            row["duplicate_previous"] = bool(row.get("near_duplicate"))
            row["evaluation_role"] = "MAPPING"
        sanitized = sanitize_frames(analyzed)
        plan = plan_keyframes(sanitized.kept, policy=policy)
        requests: list[dict[str, Any]] = []
        source_keyframes: list[dict[str, Any]] = []
        for item in plan.keyframes:
            frame = item.frame
            relative = Path(source.source_id) / f"frame_{frame.frame_index:08d}.jpg"
            image_path = images_root / relative
            requests.append({
                "source_frame_index": frame.frame_index,
                "source_image_path": frame.source.get("source_image_path"),
                "output": image_path,
            })
            source_keyframes.append({
                **dict(frame.source),
                "keyframe_id": frame.frame_id,
                "frame_id": frame.frame_id,
                "video_id": source.source_id,
                "session_id": source.path.stem,
                "source_uri": str(source.path),
                "source_frame_index": frame.frame_index,
                "source_pts_seconds": frame.timestamp,
                "image_uri": str(image_path),
                "output_name": relative.as_posix(),
                "mapping_mode": item.mapping_mode,
                "status": "CANDIDATE",
                "warnings": list(frame.warnings),
            })
        extract_frames(source, requests)
        for row in source_keyframes:
            row["image_sha256"] = sha256_file(Path(str(row["image_uri"])))
        keyframe_rows.extend(source_keyframes)
        forced_ids.extend(plan.forced_pairs)
        frame_rows.extend(_sanitization_rows(sanitized))
        per_source.append({
            "video_id": source.source_id,
            "analyzed_frames": len(analyzed),
            "candidate_frames": len(sanitized.kept),
            "removed_frames": len(sanitized.removed),
            "keyframes": len(source_keyframes),
            "pose_only": sum(r["mapping_mode"] == "POSE_ONLY" for r in source_keyframes),
        })
    if not keyframe_rows:
        raise ValueError("keyframe extraction produced zero keyframes")
    write_jsonl(work_dir / "frames.jsonl", frame_rows)
    write_jsonl(keyframes_path, keyframe_rows)
    return keyframe_rows, per_source, forced_ids


def _sanitization_rows(sanitized: Any) -> list[dict[str, Any]]:
    return [
        {**dict(record.source), "frame_id": record.frame_id,
         "status": record.status, "warnings": list(record.warnings)}
        for record in sanitized.kept
    ] + [
        {**dict(record.source), "frame_id": record.frame_id,
         "status": "INACTIVE_REJECT", "rejection_reason": record.reason}
        for record in sanitized.removed
    ]


def _verify_resume_images(rows: list[dict[str, Any]], images_root: Path) -> None:
    missing = [
        str(row.get("output_name"))
        for row in rows
        if not (images_root / str(row.get("output_name") or "")).is_file()
    ]
    if missing:
        raise ValueError(
            f"--resume found {len(missing)} keyframe images missing, "
            f"e.g. {missing[:3]}; re-run without --resume"
        )


def _materialize_images(rows: list[dict[str, Any]], root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = Path(str(row["output_name"])).as_posix()
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"keyframe has an invalid output_name: {name!r}")
        source = Path(str(row["image_uri"])).resolve()
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copyfile(source, target)


def _complete_model(path: Path) -> bool:
    return all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    )
def read_model_cameras(model_dir: Path) -> list[dict[str, Any]]:
    """Read refined PINHOLE cameras from the finished COLMAP model."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir.resolve(strict=True)))
    entries: list[dict[str, Any]] = []
    for camera_id in sorted(reconstruction.cameras):
        camera = reconstruction.cameras[camera_id]
        model_name = str(camera.model).split(".")[-1].upper()
        if model_name != "PINHOLE":
            raise RuntimeError(
                f"map camera {camera_id} is {model_name}, not PINHOLE; "
                "query contract requires undistorted PINHOLE cameras"
            )
        fx, fy, cx, cy = (float(v) for v in camera.params)
        entries.append({
            "camera_id": int(camera_id),
            "model": "PINHOLE",
            "width": int(camera.width),
            "height": int(camera.height),
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        })
    if not entries:
        raise RuntimeError("reconstruction has no cameras")
    return entries


def verify_model(
    model_dir: Path, selected_rows: list[dict[str, Any]], *, exclude_pose_only: bool
) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir.resolve(strict=True)))
    registered = {str(image.name) for image in reconstruction.images.values()}
    expected = {str(row["output_name"]) for row in selected_rows}
    missing = sorted(expected - registered)
    if missing:
        raise RuntimeError(
            f"GlueMap registered {len(registered)}/{len(expected)} images; "
            f"missing e.g. {missing[:5]}"
        )
    points = len(reconstruction.points3D)
    if points == 0:
        raise RuntimeError("GlueMap produced zero 3D points")
    pose_names = [
        str(row["output_name"]) for row in selected_rows
        if row.get("mapping_mode") == "POSE_ONLY"
    ]
    pose_check: dict[str, Any] | None = None
    if pose_names and not exclude_pose_only:
        pose_check = assert_pose_only_observation_free(reconstruction, pose_names)
    return {
        "registered_images": len(registered),
        "points3D": points,
        "observations": int(reconstruction.compute_num_observations()),
        "mean_reprojection_error": float(reconstruction.compute_mean_reprojection_error()),
        "pose_only_check": pose_check,
        "model_hashes": {
            name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
            for name in ("cameras.bin", "images.bin", "points3D.bin")
            if (model_dir / name).is_file()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
