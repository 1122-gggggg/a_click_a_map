"""Portable multi-video localization stage: EDM + MegaLoc bundle builder.

Reads an arbitrary mapping output (``model/``, ``keyframes/keyframes.jsonl``,
``keyframes/images/``, ``intrinsics.json``), builds the frozen MegaLoc
reference bank with the pinned offline runtime, and stages a relocatable
localization bundle that another machine installs with ``install.py`` and
queries with ``run_localize.py`` (query video or frame folder).

The query calibration comes from the mapping stage's REFINED cameras
(``query_contract.cameras``); the seed calibration is never substituted.
Multi-camera maps require ``--camera_id``. A keyframes-only contract without
``cameras[]`` is rejected unless ``--query_intrinsics`` supplies the query
camera's own undistorted PINHOLE calibration.

All repos, checkpoints and executables are explicit CLI flags or LOC_*
environment variables; nothing references the build machine. Coordinates are
preserved: the bundle ships the frozen model bytes unchanged.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from localization.bundle import (  # noqa: E402
    MANIFEST_SCHEMA,
    build_bundle,
    check_no_source_paths,
    copy_hash,
    render_readme,
    stage_code,
    stage_keyframes,
    stage_model,
    stage_reference_bank,
    write_config_template,
    write_entrypoints,
    write_manifest,
)
from localization.io_utils import read_json, sha256_file, write_json  # noqa: E402
from localization.query_calibration import (  # noqa: E402
    SUPPORTED_SOURCES,
    QueryCalibration,
    resolve_query_calibration,
)
from localization.reference_bank import (  # noqa: E402
    MEGALOC_DESCRIPTOR_DIM,
    build_reference_bank,
    validate_reference_bank,
)

SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Validated Python runtime for install + run_localize.py. Versions are the
# validated Wulai-run pins where known; unpinned entries carry the minimum
# the vendored code imports. Parent orchestration owns the environment file;
# this list is the stage's precise requirement declaration.
RUNTIME_REQUIREMENTS: dict[str, Any] = {
    "python": ">=3.11",
    "cuda": True,
    "system_packages": ["ffmpeg", "ffprobe"],
    "python_packages": [
        ("torch", "2.7.1+cu128"),
        ("torchvision", "0.22.1+cu128"),
        ("pycolmap-cuda12", "4.0.4"),
        ("numpy", "2.4.6"),
        ("opencv-python", "4.12.0"),
        ("scipy", "1.16.2"),
        ("safetensors", "0.6.2"),
        ("yacs", "0.1.8"),
        ("einops", "0.8.1"),
        ("joblib", "1.5.2"),
        ("loguru", "0.7.3"),
        ("pyyaml", "6.0.2"),
    ],
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="mapping output model/ (COLMAP binary model)")
    parser.add_argument("--keyframes", required=True, help="mapping keyframes/keyframes.jsonl")
    parser.add_argument("--intrinsics", required=True, help="mapping output/intrinsics.json (query contract)")
    parser.add_argument("--output", required=True, help="bundle output directory")
    parser.add_argument("--map-name", default=None, help="bundle/map name (default: output directory name)")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="query camera id for multi-camera maps (required when >1 camera)")
    parser.add_argument("--query-intrinsics", default=None,
                        help="explicit query-camera calibration JSON (only for another camera, "
                        "or when the mapping contract is keyframes-only)")
    parser.add_argument("--sparse-ply", default=None, help="sparse RGB PLY to ship under products/")
    parser.add_argument("--dense-ply", default=None, help="dense PLY to ship under products/")
    parser.add_argument("--megaloc-source", default=_env("LOC_MEGALOC_SOURCE") or None,
                        help="MegaLoc source directory (megaloc_model.py + hubconf.py)")
    parser.add_argument("--megaloc-checkpoint", default=_env("LOC_MEGALOC_CHECKPOINT") or None,
                        help="MegaLoc model.safetensors checkpoint")
    parser.add_argument("--edm-root", default=_env("LOC_EDM_ROOT") or None,
                        help="EDM repo root (contains src/)")
    parser.add_argument("--edm-checkpoint", default=_env("LOC_EDM_CHECKPOINT") or None,
                        help="EDM outdoor checkpoint (.ckpt)")
    parser.add_argument("--sfm-diagnosis", default=_env("LOC_SFM_DIAGNOSIS") or None,
                        help="sfm_diagnosis source tree to vendor")
    parser.add_argument("--river-map-quality", default=_env("LOC_RIVER_MAP_QUALITY") or None,
                        help="river_map_quality source tree to vendor")
    parser.add_argument("--localizer-python", default=_env("LOC_PYTHON") or sys.executable,
                        help="interpreter with torch+CUDA for the reference-bank worker")
    parser.add_argument("--descriptor-batch-size", type=int, default=8)
    parser.add_argument("--bundle-weights", action="store_true",
                        help="copy the two checkpoints into <bundle>/weights/ (default: external only)")
    parser.add_argument("--resume-bank", action="store_true",
                        help="reuse a previously built reference bank found in --output/work/reference_bank")
    return parser


def _need_dir(value: str | None, flag: str) -> Path:
    if not value:
        raise ValueError(f"--{flag} is required (or LOC_{flag.upper().replace('-', '_')}); "
                         "repos are never defaulted to build-machine paths")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"--{flag} is not a directory: {path}")
    return path


def _need_file(value: str | None, flag: str) -> Path:
    if not value:
        raise ValueError(f"--{flag} is required (or LOC_{flag.upper().replace('-', '_')}); "
                         "checkpoints are never defaulted to build-machine paths")
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"--{flag} missing or empty: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"bundle: {receipt['bundle_dir']}")
    print(f"references: {receipt['references']}  "
          f"registered: {receipt['registered_images']}  points3D: {receipt['points3D']}")
    return 0


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    map_name = args.map_name or output.name
    if not SAFE_NAME.fullmatch(map_name):
        raise ValueError(f"map name {map_name!r} is unsafe (use [A-Za-z0-9._-])")
    if output.exists() and not args.resume_bank and any(output.iterdir()):
        raise ValueError(f"output {output} is not empty; use an empty directory or --resume-bank")

    model_dir = Path(args.model).expanduser().resolve()
    keyframes_path = Path(args.keyframes).expanduser().resolve()
    intrinsics_path = Path(args.intrinsics).expanduser().resolve()
    for path in (model_dir / "cameras.bin", model_dir / "images.bin", model_dir / "points3D.bin"):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"map model is incomplete, missing {path.name}: {model_dir}")
    if not keyframes_path.is_file():
        raise FileNotFoundError(keyframes_path)
    if not intrinsics_path.is_file():
        raise FileNotFoundError(intrinsics_path)
    images_root = keyframes_path.parent / "images"
    if not images_root.is_dir():
        raise ValueError(f"keyframes images root not found: {images_root}")

    megaloc_source = _need_dir(args.megaloc_source, "megaloc-source")
    megaloc_checkpoint = _need_file(args.megaloc_checkpoint, "megaloc-checkpoint")
    edm_root = _need_dir(args.edm_root, "edm-root")
    edm_checkpoint = _need_file(args.edm_checkpoint, "edm-checkpoint")
    sfm_diagnosis = _need_dir(args.sfm_diagnosis, "sfm-diagnosis")
    river_map_quality = _need_dir(args.river_map_quality, "river-map-quality")
    if not (megaloc_source / "megaloc_model.py").is_file():
        raise ValueError(f"--megaloc-source lacks megaloc_model.py: {megaloc_source}")
    if not (edm_root / "src").is_dir():
        raise ValueError(f"--edm-root lacks src/: {edm_root}")
    if not (sfm_diagnosis / "site_pipeline" / "deploy_loop.py").is_file():
        raise ValueError(f"--sfm-diagnosis lacks site_pipeline/deploy_loop.py: {sfm_diagnosis}")
    if not (river_map_quality / "megaloc_edm_catalog.py").is_file():
        raise ValueError(f"--river-map-quality lacks megaloc_edm_catalog.py: {river_map_quality}")

    query: QueryCalibration = resolve_query_calibration(
        read_json(intrinsics_path),
        camera_id=args.camera_id,
        query_intrinsics=args.query_intrinsics,
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "map_name": map_name,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "model": str(model_dir),
            "keyframes": str(keyframes_path),
            "intrinsics": str(intrinsics_path),
            "camera_id": query.camera_id,
            "calibration_source": query.calibration_source,
            "refined_intrinsics": query.refined_intrinsics,
        },
    }

    # 1. Reference bank (fresh CUDA process), resumable.
    work_bank = output / "work" / "reference_bank"
    bank_receipt: dict[str, Any]
    if args.resume_bank and (work_bank / "megaloc_references.npy").is_file():
        bank_receipt = {"status": "resumed", **validate_reference_bank(
            work_bank / "megaloc_references.npy",
            work_bank / "megaloc_references.names.json")}
        report["reference_bank"] = bank_receipt
    else:
        code_roots = [str(sfm_diagnosis.parent.resolve()), str(river_map_quality.parent.resolve())]
        bank_receipt = build_reference_bank(
            model_dir=model_dir,
            keyframes_path=keyframes_path,
            output_dir=work_bank,
            python=args.localizer_python,
            code_roots=code_roots,
            megaloc_source=megaloc_source,
            megaloc_checkpoint=megaloc_checkpoint,
            descriptor_batch_size=args.descriptor_batch_size,
        )
        validate_reference_bank(work_bank / "megaloc_references.npy",
                                work_bank / "megaloc_references.names.json")
        report["reference_bank"] = bank_receipt

    # 2. Stage the relocatable bundle.
    stage = output / "bundle"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)
    model_rows = stage_model(model_dir=model_dir, stage=stage, report=report)
    stage_keyframes(keyframes_path=keyframes_path, images_root=images_root,
                    stage=stage, report=report)
    bank_staged = stage_reference_bank(reference_dir=work_bank, stage=stage, report=report)

    sparse_ply_name = _stage_product(args.sparse_ply, stage, "sparse")
    dense_ply_name = _stage_product(args.dense_ply, stage, "dense")
    report["products"] = {"sparse_ply": sparse_ply_name, "dense_ply": dense_ply_name}
    write_config_template(stage=stage, map_name=map_name, query=query,
                          sparse_ply_name=sparse_ply_name, dense_ply_name=dense_ply_name)
    build_bundle(stage=stage, map_name=map_name, query=query,
                 sparse_ply_name=sparse_ply_name, dense_ply_name=dense_ply_name, report=report)
    stage_code(stage=stage, sfm_diagnosis=sfm_diagnosis, river_map_quality=river_map_quality,
               megaloc_source=megaloc_source, edm_root=edm_root, report=report)
    if report["code"]["unresolved_static_imports"]:
        raise RuntimeError(
            "vendored code has unresolved imports: "
            f"{report['code']['unresolved_static_imports'][:5]}")
    weights_receipt = _stage_weights(
        stage=stage, bundle_weights=bool(args.bundle_weights),
        edm_checkpoint=edm_checkpoint, megaloc_checkpoint=megaloc_checkpoint)
    report["weights"] = weights_receipt
    write_entrypoints(stage=stage)
    provenance = {
        "package": map_name,
        "created_utc": report["built_utc"],
        "schema": "portable-localization-provenance/v1",
        "source_model": str(model_dir),
        "source_keyframes": str(keyframes_path),
        "source_intrinsics": str(intrinsics_path),
        "calibration_source": query.calibration_source,
        "supported_calibration_sources": list(SUPPORTED_SOURCES),
        "code_provenance": {
            "sfm_diagnosis": str(sfm_diagnosis),
            "river_map_quality": str(river_map_quality),
            "megaloc": str(megaloc_source),
            "edm": str(edm_root),
        },
        "model_sha256": {name: row["sha256"] for name, row in model_rows.items()},
        "reference_bank": {
            "references": bank_staged["megaloc_references"]["shape"][0],
            "descriptor_dim": MEGALOC_DESCRIPTOR_DIM,
            "descriptors_sha256": bank_staged["megaloc_references"]["sha256"],
            "names_sha256": bank_staged["megaloc_names"]["sha256"],
        },
        "weights": weights_receipt,
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "map_units": "native, no metric scale; validation NONE",
        "portability": {
            "audited_torch_override": "deploy_loop calls deployment_localizer._configure_audited_runtime_site_packages "
            "with sys.prefix site-packages before any torch import, overriding the vendored "
            "river_map_quality historical_experiment.AUDITED_TORCH_SITE_PACKAGES build-machine default",
            "audited_edm_override": "same override sets official_edm_adapter_loo.AUDITED_EDM_SITE_PACKAGES; "
            "run run_localize.py from the interpreter whose site-packages hold the torch stack",
            "vendored_patches": report["code"]["portability_patches"],
        },
        "report": report,
    }
    write_json(stage / "provenance.json", provenance)
    (stage / "README.md").write_text(
        render_readme(
            map_name=map_name,
            registered_images=report["model"]["registered_images"],
            points3d=report["model"]["points3D"],
            references=bank_staged["megaloc_references"]["shape"][0],
            descriptor_shape=bank_staged["megaloc_references"]["shape"],
            query=query,
            sparse_ply_name=sparse_ply_name,
            dense_ply_name=dense_ply_name,
            bundle_weights=bool(args.bundle_weights),
            requirements=RUNTIME_REQUIREMENTS,
        ),
        encoding="utf-8",
    )
    check_no_source_paths(stage=stage, report=report)
    write_json(stage / "provenance.json", provenance)
    manifest_receipt = write_manifest(stage=stage, map_name=map_name,
                                      bundle_weights=bool(args.bundle_weights))
    report["manifest"] = {"schema": MANIFEST_SCHEMA, **manifest_receipt}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCALIZATION_BUNDLE_BUILT",
        "map_name": map_name,
        "bundle_dir": str(stage),
        "registered_images": report["model"]["registered_images"],
        "points3D": report["model"]["points3D"],
        "references": bank_staged["megaloc_references"]["shape"][0],
        "coordinate_frame_id": report["direct_bundle"]["coordinate_frame_id"],
        "weights_bundled": bool(args.bundle_weights),
        "report": report,
    }
    write_json(output / "localization_receipt.json", receipt)
    return receipt


def _stage_product(value: str | None, stage: Path, kind: str) -> str | None:
    if not value:
        return None
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"--{kind}-ply not found: {source}")
    digest, size = copy_hash(source, stage / "products" / source.name)
    _ = (digest, size)
    return source.name


def _stage_weights(
    *,
    stage: Path,
    bundle_weights: bool,
    edm_checkpoint: Path,
    megaloc_checkpoint: Path,
) -> dict[str, Any]:
    if not bundle_weights:
        return {
            "bundled": False,
            "edm_checkpoint": {"path": str(edm_checkpoint),
                               "size_bytes": edm_checkpoint.stat().st_size,
                               "sha256": sha256_file(edm_checkpoint)},
            "megaloc_checkpoint": {"path": str(megaloc_checkpoint),
                                   "size_bytes": megaloc_checkpoint.stat().st_size,
                                   "sha256": sha256_file(megaloc_checkpoint)},
            "note": "weights stay external; pass them to the bundle install.py",
        }
    edm_digest, edm_size = copy_hash(edm_checkpoint, stage / "weights" / "edm.ckpt")
    megaloc_digest, megaloc_size = copy_hash(
        megaloc_checkpoint, stage / "weights" / "megaloc.safetensors")
    return {
        "bundled": True,
        "edm_checkpoint": {"path": "weights/edm.ckpt", "size_bytes": edm_size, "sha256": edm_digest},
        "megaloc_checkpoint": {"path": "weights/megaloc.safetensors",
                               "size_bytes": megaloc_size, "sha256": megaloc_digest},
    }


if __name__ == "__main__":
    raise SystemExit(main())
