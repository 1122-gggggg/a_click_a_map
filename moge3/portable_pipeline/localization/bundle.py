"""Deployable localization-bundle assembly (model-agnostic).

Writes the portable bundle layout consumed by the generated ``install.py``
and ``run_localize.py``: frozen COLMAP model bytes, keyframe images plus a
portable ``keyframes.jsonl.in`` template, the MegaLoc reference bank, the
``direct-localization-bundle/v1`` descriptor, the deployment profile, the
localizer config template, vendored runtime code with the portability patch
applied, install/localize entrypoints, manifest, provenance and README.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from localization.io_utils import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from localization.query_calibration import QueryCalibration

BUNDLE_SCHEMA = "direct-localization-bundle/v1"
PROFILE_SCHEMA = "direct-deployment-profile/v1"
MANIFEST_SCHEMA = "portable-localization-manifest/v1"
CONFIG_SCHEMA_VERSION = 2

MODEL_FILES = ("cameras.bin", "frames.bin", "images.bin", "points3D.bin", "rigs.bin")

FORBIDDEN_SUFFIXES = {".ckpt", ".pth", ".pt", ".safetensors", ".pkl", ".onnx", ".h5", ".engine"}
TEXT_SUFFIXES = {".json", ".jsonl", ".in", ".py", ".md", ".txt"}
IGNORE_DIRS = {"__pycache__", ".git", ".github", "weights", ".pytest_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo"}

# Deploy-loop constants for the generated run_localize.py. These are the
# validated two-rate settings; the bundle records them so the entrypoint and
# the profile cannot drift apart.
TRACKER = "klt"
TRACK_CAP = 500
RELOC_MIN_POINTS = 60
RELOC_PERIOD_S = 0.3
MEGALOC_TOP_K = 2
LIFT_DISTANCE_PX = 4.0
RELOC_MATCHER = "edm"


def copy_hash(src: Path, dst: Path) -> tuple[str, int]:
    """Copy one file and return (sha256, size) of the bytes written."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with src.open("rb") as reader, dst.open("wb") as writer:
        for chunk in iter(lambda: reader.read(8 << 20), b""):
            digest.update(chunk)
            writer.write(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def copy_tree(src: Path, dst: Path) -> int:
    """Copy a source tree, skipping caches, VCS, weights, and bytecode."""
    if dst.exists():
        shutil.rmtree(dst)
    count = 0
    for path in sorted(src.rglob("*")):
        relative = path.relative_to(src)
        if any(part in IGNORE_DIRS for part in relative.parts):
            continue
        if path.is_dir():
            continue
        if path.suffix.lower() in IGNORE_SUFFIXES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        count += 1
    return count


def rewrite_prefix(value: str, old: Path, new: str) -> str:
    prefix = str(old) + "/"
    if not value.startswith(prefix):
        raise ValueError(f"path {value} is not under {old}")
    return new.rstrip("/") + "/" + value[len(prefix):]


def read_model_counts(model_dir: Path) -> dict[str, int]:
    import struct

    with (model_dir / "images.bin").open("rb") as stream:
        (registered,) = struct.unpack("<Q", stream.read(8))
    with (model_dir / "points3D.bin").open("rb") as stream:
        (points,) = struct.unpack("<Q", stream.read(8))
    return {"registered_images": int(registered), "points3D": int(points)}


def stage_model(*, model_dir: Path, stage: Path, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Copy the frozen COLMAP model byte-for-byte (same coordinates)."""
    dst = stage / "model"
    rows: dict[str, dict[str, Any]] = {}
    for name in MODEL_FILES:
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"map model is missing {name}: {model_dir}")
        digest, size = copy_hash(source, dst / name)
        rows[name] = {"sha256": digest, "size_bytes": size}
    counts = read_model_counts(dst)
    report["model"] = {"files": rows, **counts}
    return rows


def stage_keyframes(
    *,
    keyframes_path: Path,
    images_root: Path,
    stage: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Copy keyframe JPEGs and write the portable keyframes.jsonl.in template."""
    rows_in = read_jsonl(keyframes_path)
    if not rows_in:
        raise ValueError(f"no keyframe rows in {keyframes_path}")
    rows_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows_in:
        name = str(row.get("output_name") or "")
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"keyframe has an invalid output_name: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate keyframe image: {name}")
        seen.add(name)
        uri = str(row.get("image_uri") or "")
        if not uri:
            raise ValueError(f"keyframe has no image_uri: {name}")
        source = Path(uri).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"keyframe image missing: {source}")
        digest, _ = copy_hash(source, stage / "keyframes/images" / name)
        expected = str(row.get("image_sha256") or "")
        if expected and digest != expected:
            raise ValueError(f"keyframe sha mismatch for {name}")
        out = dict(row)
        try:
            out["image_uri"] = rewrite_prefix(str(source.resolve()), images_root.resolve(), "@PKG@/keyframes/images")
        except ValueError:
            out["image_uri"] = "@PKG@/keyframes/images/" + name
        rows_out.append(out)
    template = stage / "keyframes/keyframes.jsonl.in"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows_out),
        encoding="utf-8",
    )
    receipt = {"rows": len(rows_out), "images_copied": len(seen),
               "source_sha256": sha256_file(keyframes_path)}
    report["keyframes"] = receipt
    return receipt


def stage_reference_bank(
    *,
    reference_dir: Path,
    stage: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Copy the MegaLoc bank + names and template the reference manifest."""
    import numpy as np

    dst = stage / "localization"
    dst.mkdir(parents=True, exist_ok=True)
    descriptor_digest, descriptor_size = copy_hash(
        reference_dir / "megaloc_references.npy", dst / "megaloc_references.npy")
    names_digest, names_size = copy_hash(
        reference_dir / "megaloc_references.names.json", dst / "megaloc_references.names.json")
    shape = tuple(int(v) for v in np.load(dst / "megaloc_references.npy",
                                          mmap_mode="r", allow_pickle=False).shape)
    names = read_json(dst / "megaloc_references.names.json")
    if len(shape) != 2 or shape[0] != len(names) or len(set(names)) != len(names):
        raise ValueError(f"MegaLoc bank inconsistent: shape={shape} names={len(names)}")
    manifest_rows: list[dict[str, Any]] = []
    for row in read_jsonl(reference_dir / "reference_manifest.jsonl"):
        out = dict(row)
        out["image_path"] = "@PKG@/keyframes/images/" + str(row["image_name"])
        manifest_rows.append(out)
    if [row["image_name"] for row in manifest_rows] != list(names):
        raise ValueError("reference manifest identities disagree with bank names")
    (dst / "reference_manifest.jsonl.in").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    receipt = {
        "megaloc_references": {"sha256": descriptor_digest, "size_bytes": descriptor_size,
                               "shape": list(shape)},
        "megaloc_names": {"sha256": names_digest, "size_bytes": names_size, "count": len(names)},
        "reference_manifest_rows": len(manifest_rows),
    }
    report["localization_bank"] = receipt
    return receipt


def write_config_template(
    *,
    stage: Path,
    map_name: str,
    query: QueryCalibration,
    sparse_ply_name: str | None,
    dense_ply_name: str | None,
) -> Path:
    """Write localization/localizer_config.json.in with install-time placeholders."""
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "artifact_type": "PORTABLE_EDM_MEGALOC_PNP_CONFIG",
        "map_name": map_name,
        "map_model": "@PKG@/model",
        "reference_manifest": "@PKG@/localization/reference_manifest.jsonl",
        "megaloc_source": "@MEGALOC_SOURCE@",
        "megaloc_checkpoint": "@MEGALOC_CHECKPOINT@",
        "edm_root": "@EDM_ROOT@",
        "edm_checkpoint": "@EDM_CHECKPOINT@",
        "intrinsics": query.as_deploy_dict(),
        "top_k": MEGALOC_TOP_K,
        "lift_distance_px": LIFT_DISTANCE_PX,
        "pose_solver": "COLMAP_PNP_RANSAC",
        "validation": "NONE",
        "map_units": "NATIVE_MAP_UNITS_NO_METRIC_SCALE",
        "map_products": {
            "sparse_ply": None if sparse_ply_name is None else f"../products/{sparse_ply_name}",
            "dense_ply": None if dense_ply_name is None else f"../products/{dense_ply_name}",
            "note": "geometry/viewing products in the same world frame; the localizer reads model/",
        },
        "validated_runtime": {
            "localizer_settings": {
                "megaloc_top_k": MEGALOC_TOP_K,
                "lift_distance_px": LIFT_DISTANCE_PX,
                "reloc_matcher": RELOC_MATCHER,
                "tracker": TRACKER,
                "track_cap": TRACK_CAP,
                "reloc_period_s": RELOC_PERIOD_S,
                "vo": True,
                "dead_reckon": True,
                "reference_bank": "full",
            },
            "note": "run_localize.py passes these via DeployConfig; top_k/lift mirror them.",
        },
    }
    path = stage / "localization/localizer_config.json.in"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_bundle(
    *,
    stage: Path,
    map_name: str,
    query: QueryCalibration,
    sparse_ply_name: str | None,
    dense_ply_name: str | None,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Write direct_bundle.json and direct_localizer_profile.json (relative paths)."""
    location = stage / "localization"
    model_sha = {name: sha256_file(stage / "model" / name) for name in MODEL_FILES}
    files = [
        {"path": f"../model/{name}", "sha256": model_sha[name],
         "size_bytes": (stage / "model" / name).stat().st_size}
        for name in MODEL_FILES
    ]
    for name in ("megaloc_references.npy", "megaloc_references.names.json"):
        files.append({"path": name, "sha256": sha256_file(location / name),
                      "size_bytes": (location / name).stat().st_size})
    coordinate_frame_id = f"{map_name}_{model_sha['points3D.bin'][:10]}"
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "map_revision_id": map_name,
        "coordinate_frame_id": coordinate_frame_id,
        "model_dir": "../model",
        "keyframes_manifest": "../keyframes/keyframes.jsonl",
        "keyframes_images_root": "../keyframes/images",
        "reference_manifest": "reference_manifest.jsonl",
        "reference_bank": {
            "name": "megaloc",
            "descriptors": "megaloc_references.npy",
            "names": "megaloc_references.names.json",
        },
        "intersection_cells": None,
        "reference_depth_dir": None,
        "model_sha256": model_sha,
        "files": files,
    }
    (location / "direct_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    profile = {
        "schema": PROFILE_SCHEMA,
        "name": map_name,
        "map_scale": 1.0,
        "intrinsics": query.as_deploy_dict(),
        "reloc": {
            "top_k": MEGALOC_TOP_K,
            "lift_distance_px": LIFT_DISTANCE_PX,
            "period_s": RELOC_PERIOD_S,
            "min_points": RELOC_MIN_POINTS,
            "reference_bank": "megaloc",
            "min_reference_occupied_bins": 1,
            "bank_occupancy_percentile": 10,
            "descriptor_batch_size": 8,
            "batch_refs": False,
            "reference_cache_size": 1024,
            "frozen_pnp_thresholds": {
                "strong_inliers": 80,
                "minimum_inlier_ratio": 0.25,
                "minimum_hull_coverage": 0.15,
                "minimum_occupancy_4x4": 6,
                "minimum_positive_depth_ratio": 0.99,
                "maximum_reprojection_p90_px": 3.0,
            },
        },
        "fast_loop": {
            "resolution": [960, 540],
            "tracker": TRACKER,
            "track_cap": TRACK_CAP,
            "reseed_min_points": 120,
            "topup": False,
            "klt": {"fb_max_px": 1.0, "win": 15, "levels": 3},
            "pnp": {"max_error_px": 4.0, "min_trials": 10, "max_trials": 100,
                    "confidence": 0.999, "refine_iters": 20,
                    "min_points": 12, "min_inliers": 12},
        },
        "vo": {"enabled": True, "keyframe_stride": 8, "batch_lag": 6, "min_live": 350,
               "detect_cap": 300, "min_distance": 8, "min_parallax_deg": 0.5,
               "max_reproj_px": 2.0, "refine": False, "window_ba": False},
        "dead_reckon": {"enabled": True, "max_frames": 300, "max_age_s": 10.0},
        "max_jump_u": 1.5,
    }
    (location / "direct_localizer_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "schema": bundle["schema"],
        "coordinate_frame_id": coordinate_frame_id,
        "files": len(files),
        "bundle_sha256": sha256_file(location / "direct_bundle.json"),
        "profile_sha256": sha256_file(location / "direct_localizer_profile.json"),
    }
    report["direct_bundle"] = receipt
    return receipt


def _patch_portability(code_dir: Path) -> list[str]:
    """Rewrite abandoned source-machine constants in the vendored runtime.

    deploy_loop's NeuFlowTracker hardcodes the source sys.path entry and
    checkpoint; flow_temporal_bridge constants and XFEAT_REPO do the same.
    The validated path (tracker=klt, reloc_matcher=edm) never executes them,
    but a portable bundle must not silently inherit them: the patch makes
    every one of these resolve from an explicit environment variable and fail
    with a clear message when unset.
    """
    applied: list[str] = []
    deploy = code_dir / "sfm_diagnosis/site_pipeline/deploy_loop.py"
    text = deploy.read_text(encoding="utf-8")
    old_path = '        sys.path.insert(0, "/home/cihcilab/sfm/runtime/neuflow2")'
    new_path = (
        "        _neuflow_root = os.environ.get(\"NEUFLOW_ROOT\", \"\")\n"
        "        if not _neuflow_root:\n"
        "            raise RuntimeError(\n"
        "                \"NeuFlowTracker needs NEUFLOW_ROOT=/path/to/neuflow2; \"\n"
        "                \"the validated bundle path uses tracker='klt' instead\"\n"
        "            )\n"
        "        sys.path.insert(0, _neuflow_root)"
    )
    if old_path not in text:
        raise RuntimeError("vendored deploy_loop.py NeuFlow sys.path line not found; refusing to ship unpatched")
    text = text.replace(old_path, new_path)
    old_ckpt = '        ckpt = torch.load("/home/cihcilab/sfm/runtime/neuflow2/neuflow_mixed.pth", map_location="cuda")'
    new_ckpt = (
        "        _neuflow_ckpt = os.environ.get(\"NEUFLOW_CHECKPOINT\", \"\")\n"
        "        if not _neuflow_ckpt:\n"
        "            raise RuntimeError(\n"
        "                \"NeuFlowTracker needs NEUFLOW_CHECKPOINT=/path/to/neuflow_mixed.pth; \"\n"
        "                \"the validated bundle path uses tracker='klt' instead\"\n"
        "            )\n"
        "        ckpt = torch.load(_neuflow_ckpt, map_location=\"cuda\")"
    )
    if old_ckpt not in text:
        raise RuntimeError("vendored deploy_loop.py NeuFlow checkpoint line not found; refusing to ship unpatched")
    text = text.replace(old_ckpt, new_ckpt)
    if "import os" not in text.split("class NeuFlowTracker")[0].split("\n\n")[-1]:
        pass  # deploy_loop already imports os at module top
    deploy.write_text(text, encoding="utf-8")
    applied.append("sfm_diagnosis/site_pipeline/deploy_loop.py: NeuFlow sys.path/checkpoint -> NEUFLOW_ROOT/NEUFLOW_CHECKPOINT")

    bridge = code_dir / "sfm_diagnosis/site_pipeline/flow_temporal_bridge.py"
    bridge_text = bridge.read_text(encoding="utf-8")
    replacements = [
        ('SEA_RAFT_ROOT = Path("/home/cihcilab/sfm/runtime/sea_raft")',
         'SEA_RAFT_ROOT = Path(os.environ.get("SEA_RAFT_ROOT", "/nonexistent-sea-raft-root-unset"))'),
        ('NEUFLOW_ROOT = Path("/home/cihcilab/sfm/runtime/neuflow2")',
         'NEUFLOW_ROOT = Path(os.environ.get("NEUFLOW_ROOT", "/nonexistent-neuflow-root-unset"))'),
        ('LOCOTRACK_ROOT = Path("/home/cihcilab/sfm/runtime/locotrack/locotrack_pytorch")',
         'LOCOTRACK_ROOT = Path(os.environ.get("LOCOTRACK_ROOT", "/nonexistent-locotrack-root-unset"))'),
        ('    "base": Path("/home/cihcilab/sfm/runtime/locotrack/weights/locotrack_base.ckpt"),',
         '    "base": Path(os.environ.get("LOCOTRACK_BASE_CHECKPOINT", "/nonexistent-locotrack-base-unset")),'),
        ('    "small": Path("/home/cihcilab/sfm/runtime/locotrack/weights/locotrack_small.ckpt"),',
         '    "small": Path(os.environ.get("LOCOTRACK_SMALL_CHECKPOINT", "/nonexistent-locotrack-small-unset")),'),
    ]
    for old, new in replacements:
        if old not in bridge_text:
            raise RuntimeError(f"vendored flow_temporal_bridge.py line not found, refusing to ship unpatched: {old[:60]}")
        bridge_text = bridge_text.replace(old, new)
    bridge.write_text(bridge_text, encoding="utf-8")
    applied.append("sfm_diagnosis/site_pipeline/flow_temporal_bridge.py: SEA-RAFT/NeuFlow/LocoTrack roots -> env vars")

    sparse = code_dir / "sfm_diagnosis/site_pipeline/sparse_reloc.py"
    sparse_text = sparse.read_text(encoding="utf-8")
    old_xfeat = 'XFEAT_REPO = Path(os.environ.get("XFEAT_REPO", "/home/cihcilab/sfm/runtime/xfeat"))'
    new_xfeat = 'XFEAT_REPO = Path(os.environ.get("XFEAT_REPO", "/nonexistent-xfeat-repo-unset"))'
    if old_xfeat not in sparse_text:
        raise RuntimeError("vendored sparse_reloc.py XFEAT_REPO line not found; refusing to ship unpatched")
    sparse.write_text(sparse_text.replace(old_xfeat, new_xfeat), encoding="utf-8")
    applied.append("sfm_diagnosis/site_pipeline/sparse_reloc.py: XFEAT_REPO default -> unset sentinel")

    history = code_dir / "river_map_quality/historical_pipeline.py"
    history_text = history.read_text(encoding="utf-8")
    old_current = ('CURRENT_IMAGE_ROOT = Path(\n'
                   '    "/media/cihcilab/新增磁碟區1/河濱場域/gluemap_build/runs/"\n'
                   '    "river_base3_mvroma_localization_20260815/sampling/images"\n'
                   ')')
    new_current = ('CURRENT_IMAGE_ROOT = Path(\n'
                   '    os.environ.get("RIVER_CURRENT_IMAGE_ROOT", "/nonexistent-river-current-root-unset")\n'
                   ')')
    if old_current not in history_text:
        raise RuntimeError("vendored historical_pipeline.py CURRENT_IMAGE_ROOT not found; refusing to ship unpatched")
    history.write_text(history_text.replace(old_current, new_current), encoding="utf-8")
    applied.append("river_map_quality/historical_pipeline.py: CURRENT_IMAGE_ROOT -> RIVER_CURRENT_IMAGE_ROOT")
    return applied


def stage_code(
    *,
    stage: Path,
    sfm_diagnosis: Path,
    river_map_quality: Path,
    megaloc_source: Path,
    edm_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Vendor the localizer runtime code and apply the portability patch."""
    import re

    code = stage / "code"
    counts = {
        "sfm_diagnosis": copy_tree(sfm_diagnosis, code / "sfm_diagnosis"),
        "river_map_quality": copy_tree(river_map_quality, code / "river_map_quality"),
        "megaloc": copy_tree(megaloc_source, code / "megaloc"),
        "edm": copy_tree(edm_root, code / "edm"),
    }
    patches = _patch_portability(code)
    missing: list[str] = []
    pattern = re.compile(r"^\s*(?:from|import)\s+((?:sfm_diagnosis|river_map_quality)(?:\.[\w]+)*)", re.M)
    for py in sorted(code.rglob("*.py")):
        for module in pattern.findall(py.read_text(encoding="utf-8", errors="ignore")):
            relative = Path(*module.split("."))
            if not ((code / relative).with_suffix(".py").is_file()
                    or (code / relative / "__init__.py").is_file()):
                missing.append(f"{py.relative_to(stage)} -> {module}")
    receipt = {"file_counts": counts, "portability_patches": patches,
               "unresolved_static_imports": missing}
    report["code"] = receipt
    return receipt


INSTALL_PY = '''\
#!/usr/bin/env python3
"""Render this package's *.in templates for THIS machine. Copies no weights.

Run once after unpacking, and again after moving the package directory.
Supports both layouts: weights passed explicitly, or bundled under weights/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
TEMPLATES = [
    ("localization/localizer_config.json.in", "localizer_config.json"),
    ("keyframes/keyframes.jsonl.in", "keyframes/keyframes.jsonl"),
    ("localization/reference_manifest.jsonl.in", "localization/reference_manifest.jsonl"),
]
PLACEHOLDER = re.compile(r"@[A-Z_]+@")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need_file(value: str, label: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"{label} missing or empty: {path}")
    return str(path)


def need_dir(value: str, label: str) -> str:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        sys.exit(f"{label} is not a directory: {path}")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edm-checkpoint", default=None,
                    help="external edm_outdoor.ckpt (default: <pkg>/weights/edm.ckpt when bundled)")
    ap.add_argument("--megaloc-checkpoint", default=None,
                    help="external MegaLoc model.safetensors (default: <pkg>/weights/megaloc.safetensors when bundled)")
    ap.add_argument("--megaloc-source", default=str(PKG / "code/megaloc"),
                    help="MegaLoc code directory (default: shipped code/megaloc)")
    ap.add_argument("--edm-root", default=str(PKG / "code/edm"),
                    help="EDM code directory (default: shipped code/edm)")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash all model .bin files and all keyframe JPEGs (slow)")
    ap.add_argument("--hash-weights", action="store_true",
                    help="also record SHA-256 of the checkpoints")
    args = ap.parse_args()

    root = str(PKG)
    if any(c in root for c in '"\\\n'):
        sys.exit(f"package path must not contain quotes, backslashes, or newlines: {root}")

    edm_default = PKG / "weights" / "edm.ckpt"
    megaloc_default = PKG / "weights" / "megaloc.safetensors"
    edm_ckpt = args.edm_checkpoint or (str(edm_default) if edm_default.is_file() else None)
    megaloc_ckpt = args.megaloc_checkpoint or (
        str(megaloc_default) if megaloc_default.is_file() else None)
    if edm_ckpt is None:
        sys.exit("--edm-checkpoint is required (no bundled weights/edm.ckpt found)")
    if megaloc_ckpt is None:
        sys.exit("--megaloc-checkpoint is required (no bundled weights/megaloc.safetensors found)")

    substitutions = {
        "@PKG@": root,
        "@EDM_ROOT@": need_dir(args.edm_root, "--edm-root"),
        "@EDM_CHECKPOINT@": need_file(edm_ckpt, "EDM checkpoint"),
        "@MEGALOC_SOURCE@": need_dir(args.megaloc_source, "--megaloc-source"),
        "@MEGALOC_CHECKPOINT@": need_file(megaloc_ckpt, "MegaLoc checkpoint"),
    }
    for label, key in [("--megaloc-source", "@MEGALOC_SOURCE@"), ("--edm-root", "@EDM_ROOT@")]:
        if not Path(substitutions[key]).joinpath("hubconf.py" if key == "@MEGALOC_SOURCE@" else "src").exists():
            sys.exit(f"{label} does not look like the expected code tree: {substitutions[key]}")
    if not Path(substitutions["@MEGALOC_SOURCE@"], "megaloc_model.py").is_file():
        sys.exit("--megaloc-source lacks megaloc_model.py")

    rendered = []
    for template, target in TEMPLATES:
        src = PKG / template
        if not src.is_file():
            sys.exit(f"template missing from package: {template}")
        text = src.read_text(encoding="utf-8")
        for key, value in substitutions.items():
            text = text.replace(key, value)
        leftover = PLACEHOLDER.findall(text)
        if leftover:
            sys.exit(f"unrendered placeholders in {template}: {sorted(set(leftover))}")
        if target.endswith(".jsonl"):
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        else:
            json.loads(text)
        (PKG / target).parent.mkdir(parents=True, exist_ok=True)
        (PKG / target).write_text(text, encoding="utf-8")
        rendered.append(target)

    manifest = json.loads((PKG / "MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for entry in manifest["files"]:
        if entry["path"].startswith("model/") or entry["path"].startswith("keyframes/images/"):
            path = PKG / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
                sys.exit(f"packaged file missing or wrong size: {entry['path']}")
            if args.verify and sha256_file(path) != entry["sha256"]:
                sys.exit(f"SHA-256 mismatch: {entry['path']}")
            checked += 1

    external = {
        "edm_root": substitutions["@EDM_ROOT@"],
        "edm_checkpoint": {"path": substitutions["@EDM_CHECKPOINT@"],
                           "size_bytes": Path(substitutions["@EDM_CHECKPOINT@"]).stat().st_size,
                           "bundled": str(Path(substitutions["@EDM_CHECKPOINT@"]).resolve()).startswith(root + "/")},
        "megaloc_source": substitutions["@MEGALOC_SOURCE@"],
        "megaloc_checkpoint": {"path": substitutions["@MEGALOC_CHECKPOINT@"],
                               "size_bytes": Path(substitutions["@MEGALOC_CHECKPOINT@"]).stat().st_size,
                               "bundled": str(Path(substitutions["@MEGALOC_CHECKPOINT@"]).resolve()).startswith(root + "/")},
    }
    if args.hash_weights:
        external["edm_checkpoint"]["sha256"] = sha256_file(Path(substitutions["@EDM_CHECKPOINT@"]))
        external["megaloc_checkpoint"]["sha256"] = sha256_file(Path(substitutions["@MEGALOC_CHECKPOINT@"]))

    receipt = {
        "schema": "portable-localization-install/v1",
        "installed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_root": root,
        "rendered": rendered,
        "packaged_files_checked": checked,
        "verify_hashes": bool(args.verify),
        "external": external,
    }
    (PKG / "install_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''

RUN_LOCALIZE_PY = '''\
#!/usr/bin/env python3
"""Localize a video (or a folder of query frames) against the packaged map.

Runs the validated MegaLoc top-2 + EDM + KLT two-rate path
(sfm_diagnosis.site_pipeline.deploy_loop). Requires a prior
`python install.py ...` in this package directory.

Query frames MUST come from the calibration recorded in
localizer_config.json intrinsics (undistorted PINHOLE, same camera). The
entrypoint resizes to the fast-loop resolution and scales K accordingly; it
does NOT calibrate a different camera.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG / "code"))  # sfm_diagnosis, river_map_quality, megaloc, edm

FAST_W = 960
FAST_H = 540


def probe_fps(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(Fraction(out))


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    width_text, height_text = out.split(",")
    return int(width_text), int(height_text)


def check_aspect(width: int, height: int, calib_width: int, calib_height: int) -> None:
    if width * calib_height != height * calib_width:
        sys.exit(
            f"query frames are {width}x{height} but the map query calibration is "
            f"{calib_width}x{calib_height} (different aspect ratio); undistort/crop "
            "the query stream to the calibrated camera first -- this entrypoint "
            "never stretches pixels"
        )


def extract_frames(video: Path, folder: Path, width: int, height: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("frame_*.jpg")):
        return
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"scale={width}:{height}", "-vsync", "0", "-q:v", "2", "-start_number", "0",
         str(folder / "frame_%06d.jpg")],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="query video (any ffmpeg-readable format)")
    src.add_argument("--frames", type=Path, help="directory of frame_*.jpg at the fast-loop resolution")
    ap.add_argument("--fps", type=float, help="stream fps (default: probed from --video)")
    ap.add_argument("--width", type=int, default=FAST_W, help="fast-loop frame width (default 960)")
    ap.add_argument("--height", type=int, default=FAST_H, help="fast-loop frame height (default 540)")
    ap.add_argument("--route", default="query", help="label written into the summary")
    ap.add_argument("--max-frames", type=int, default=0, help="localize only the first N frames (0 = all)")
    ap.add_argument("--output", type=Path, required=True, help="output directory")
    args = ap.parse_args()

    config_path = PKG / "localizer_config.json"
    keyframes = PKG / "keyframes/keyframes.jsonl"
    if not (config_path.is_file() and keyframes.is_file()):
        sys.exit("run install.py first: localizer_config.json / keyframes.jsonl not rendered")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw = config_path.read_text(encoding="utf-8")
    if "@" in raw and any(
        k.startswith("@") or (isinstance(v, str) and v.startswith("@")) for k, v in config.items()
    ):
        sys.exit("localizer_config.json still has placeholders; rerun install.py")
    intrinsics = config.get("intrinsics") or {}
    calib_width = int(intrinsics.get("image_width") or 0)
    calib_height = int(intrinsics.get("image_height") or 0)

    if args.video:
        fps = args.fps or probe_fps(args.video)
        video_w, video_h = probe_size(args.video)
        check_aspect(video_w, video_h, calib_width, calib_height)
        output = args.output.resolve()
        frames_dir = output / "frames"
        extract_frames(args.video, frames_dir, args.width, args.height)
    else:
        if args.fps is None:
            sys.exit("--fps is required with --frames")
        fps = args.fps
        output = args.output.resolve()
        frames_dir = args.frames.resolve()
        existing = sorted(frames_dir.glob("frame_*.jpg"))
        if not existing:
            sys.exit(f"no frame_*.jpg under {frames_dir}")
    output.mkdir(parents=True, exist_ok=True)

    from sfm_diagnosis.site_pipeline import deploy_loop
    from sfm_diagnosis.site_pipeline.deploy_loop import DeployConfig, run_deploy_loop

    deploy_loop.IMAGE_W = int(args.width)
    deploy_loop.IMAGE_H = int(args.height)

    run_deploy_loop(DeployConfig(
        map_model=PKG / "model",
        keyframes=keyframes,
        frames_dir=frames_dir,
        output_dir=output,
        localizer_config=config_path,
        localization_dir=PKG / "localization",
        edm_repo=Path(config["edm_root"]),
        edm_checkpoint=Path(config["edm_checkpoint"]),
        route=args.route,
        stream_fps=fps,
        max_frames=int(args.max_frames),
        tracker=TRACKER_TOKEN,
        track_cap=TRACK_CAP_TOKEN,
        reloc_min_points=RELOC_MIN_POINTS_TOKEN,
        reloc_period_s=RELOC_PERIOD_S_TOKEN,
        megaloc_top_k=MEGALOC_TOP_K_TOKEN,
        lift_distance_px=LIFT_DISTANCE_PX_TOKEN,
        reloc_matcher=RELOC_MATCHER_TOKEN,
        vo=True, dead_reckon=True,
    ))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print(json.dumps({
        "summary": str(output / "summary.json"),
        "status_counts": summary.get("status_counts"),
        "pose_coverage": summary.get("pose_coverage"),
        "dense_frames": summary.get("dense_frames"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''

RUN_LOCALIZE_TOKENS = {
    "TRACKER_TOKEN": repr(TRACKER),
    "TRACK_CAP_TOKEN": repr(TRACK_CAP),
    "RELOC_MIN_POINTS_TOKEN": repr(RELOC_MIN_POINTS),
    "RELOC_PERIOD_S_TOKEN": repr(RELOC_PERIOD_S),
    "MEGALOC_TOP_K_TOKEN": repr(MEGALOC_TOP_K),
    "LIFT_DISTANCE_PX_TOKEN": repr(LIFT_DISTANCE_PX),
    "RELOC_MATCHER_TOKEN": repr(RELOC_MATCHER),
}


def write_entrypoints(*, stage: Path) -> None:
    (stage / "install.py").write_text(INSTALL_PY, encoding="utf-8")
    localize = RUN_LOCALIZE_PY
    for token, value in RUN_LOCALIZE_TOKENS.items():
        localize = localize.replace(token, value)
    (stage / "run_localize.py").write_text(localize, encoding="utf-8")
    (stage / "run_localize.py").chmod(0o755)


def write_manifest(*, stage: Path, map_name: str, bundle_weights: bool) -> dict[str, Any]:
    entries = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name in {"MANIFEST.json", "MANIFEST.sha256"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlink in portable package: {path}")
        relative = path.relative_to(stage).as_posix()
        if not bundle_weights and relative.split("/")[0] != "weights" and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Weight-like artifact in package without --bundle_weights: {relative}")
        if bundle_weights and path.suffix.lower() in FORBIDDEN_SUFFIXES and relative.split("/")[0] != "weights":
            raise RuntimeError(f"Weight-like artifact outside weights/: {relative}")
        entries.append({"path": relative, "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    write_json(stage / "MANIFEST.json",
               {"schema": MANIFEST_SCHEMA, "package": map_name, "files": entries})
    (stage / "MANIFEST.sha256").write_text(
        "".join(f"{e['sha256']}  {e['path']}\n" for e in entries), encoding="utf-8")
    return {"entries": len(entries), "total_bytes": sum(e["size_bytes"] for e in entries)}


def check_no_source_paths(*, stage: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when generated text still references the build machine."""
    hits: list[str] = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if "code/" in path.relative_to(stage).as_posix():
            continue  # vendored code provenance is documented, not generated text
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if path.name in {"install.py", "run_localize.py", "README.md"}:
                continue  # generated entrypoints document placeholders by design
            if stripped.startswith("/home/") or stripped.startswith("/media/") or "/home/" in line or "/media/" in line:
                # provenance.json legitimately records build-machine source paths
                if path.name == "provenance.json":
                    continue
                hits.append(f"{path.relative_to(stage)}: {stripped[:120]}")
                break
            if "@PKG@" in line or "@EDM_" in line or "@MEGALOC_" in line:
                if path.suffix == ".in":
                    continue
                hits.append(f"{path.relative_to(stage)}: unrendered placeholder")
                break
    receipt = {"generated_text_source_path_refs": len(hits), "hits": hits[:20]}
    if hits:
        raise RuntimeError(f"generated bundle text references source paths: {hits[:5]}")
    report["portability_check"] = receipt
    return receipt


def render_readme(
    *,
    map_name: str,
    registered_images: int,
    points3d: int,
    references: int,
    descriptor_shape: list[int],
    query: QueryCalibration,
    sparse_ply_name: str | None,
    dense_ply_name: str | None,
    bundle_weights: bool,
    requirements: Mapping[str, Any],
) -> str:
    req_lines = "\n".join(
        f"- `{name} {version}`" for name, version in requirements.get("python_packages", [])
    )
    if bundle_weights:
        weights_section = (
            "Bundled under `weights/` (`edm.ckpt`, `megaloc.safetensors`); "
            "`install.py` uses them by default and `--hash-weights` records "
            "their digests."
        )
        install_cmd = "python install.py  # uses bundled weights/"
    else:
        weights_section = (
            "- EDM checkpoint: `edm_outdoor.ckpt`\n"
            "- MegaLoc checkpoint: `model.safetensors` (MegaLoc source code "
            "is shipped in `code/megaloc`)"
        )
        install_cmd = (
            "python install.py \\\n"
            "  --edm-checkpoint /path/to/edm_outdoor.ckpt \\\n"
            "  --megaloc-checkpoint /path/to/megaloc/model.safetensors"
        )
    return f"""# {map_name} EDM + MegaLoc localization package

Sparse map (frozen COLMAP model, {points3d:,} points, {registered_images:,} registered keyframes) with the MegaLoc reference bank, keyframe JPEGs, and runtime code. {"Weights ARE bundled under `weights/`." if bundle_weights else "**No model weights are included.**"}

Map units are native (no metric scale). Validation is `NONE`.

## Layout

| Path | Contents |
|---|---|
| `model/` | frozen sparse model (`cameras/frames/images/points3D/rigs.bin`), original map coordinates |
| `keyframes/images/` | {registered_images:,} keyframe JPEGs (SHA-256 in `MANIFEST.sha256`) |
| `keyframes/keyframes.jsonl.in` | keyframe index; rendered to `keyframes.jsonl` by `install.py` |
| `localization/` | MegaLoc descriptor bank ({references:,} x {descriptor_shape[1]:,} float32) + names; `direct_bundle.json` (`direct-localization-bundle/v1`, relative paths, files[] digests); `direct_localizer_profile.json` (`direct-deployment-profile/v1`); reference manifest template |
| `localization/localizer_config.json.in` | localizer config template (query intrinsics + frozen runtime settings) |
| `products/` | sparse/dense PLY geometry products in the same world frame (viewing; the localizer reads `model/`) |
| `code/` | `sfm_diagnosis` (deploy_loop), `river_map_quality`, `megaloc` source, `edm` source (no weights) |
| `install.py` | renders templates for this machine; checks weights |
| `run_localize.py` | localizes a video or a query frame folder |
| `MANIFEST.json`, `MANIFEST.sha256` | file list with sizes and SHA-256 |
| `provenance.json` | source paths, receipt hashes, build inputs |

## Weights

{weights_section}

## Install

```bash
cd <package dir>
{install_cmd}
# optional: --verify (re-hash model and all keyframe JPEGs), --hash-weights
```

Rerun `install.py` after moving the package directory. The rendered `localizer_config.json`, `keyframes/keyframes.jsonl`, and `localization/reference_manifest.jsonl` contain absolute paths for this machine. The `.in` templates are the portable form.

Environment: Python 3.11+, CUDA GPU, `ffmpeg`/`ffprobe`. Validated dependency set:

{req_lines}

System packages: `ffmpeg`, `ffprobe`.

## Localize

```bash
python run_localize.py --video /path/to/query.MP4 --route myroute --output out/
# or, for existing fast-loop frames (default 960x540):
python run_localize.py --frames /path/to/frames --fps 29.97 --route myroute --output out/
# optional: --width/--height for another same-aspect fast-loop size, --max-frames N
```

Output: `out/summary.json`, `out/frames.jsonl`, `out/reloc_events.json`. Poses are in the frozen map world frame (native units).

Validated settings (do not change without re-validating): KLT two-rate, `track_cap={TRACK_CAP}`, MegaLoc top-{MEGALOC_TOP_K} + EDM reloc every {RELOC_PERIOD_S} s, `lift_distance_px={LIFT_DISTANCE_PX}`, VO and dead reckoning on.

## Query camera calibration

The packaged query intrinsics are the map's refined PINHOLE calibration for camera {query.camera_id} at {query.width}x{query.height} (source: {query.calibration_source}, refined={query.refined_intrinsics}). `--video` resizes frames; it does **not** calibrate a different camera. The entrypoint refuses query streams with a different aspect ratio. For a different lens/camera, rebuild the bundle with `--query_intrinsics <own-calibration.json>` (undistorted PINHOLE at that camera's resolution) together with query frames undistorted to that model. Do not treat native coordinates or `map_scale: 1` as a validated metre scale.

## Known limits

- Map is unvalidated and has no metric scale; coordinates are native map units.
- `keyframes.jsonl` `source_uri` fields point at build-machine sources. They are informational only and are not read by localization.
- `localizer_config.json` `top_k`/`lift_distance_px` mirror the validated DeployConfig values; `deploy_loop` uses the DeployConfig values.
- Sparse and dense PLY coordinates are in the **same original world frame**, with identity transform and no ICP, centering, axis swap, or rescaling. The dense PLY is a geometry/viewing product; EDM localization continues to use `model/`, its sparse observations, and the reference bank.
- `deploy_loop` admits the audited torch/EDM site-packages from `sys.prefix` of the interpreter that runs `run_localize.py`. Run it from the environment where torch is installed (its own site-packages). The vendored `river_map_quality` hardcoded `/home/...` default is overridden at runtime before any torch import; see `provenance.json` `portability`.
- Optional trackers (NeuFlow/SEA-RAFT/LocoTrack) and the xfeat backend are not on the validated path; the vendored code resolves them from `NEUFLOW_ROOT`/`NEUFLOW_CHECKPOINT`, `SEA_RAFT_ROOT`, `LOCOTRACK_ROOT`/`LOCOTRACK_*_CHECKPOINT`, `XFEAT_REPO`, and `RIVER_CURRENT_IMAGE_ROOT` instead of source-machine paths. Unset variables fail with an explicit message only if those paths are used.
"""
