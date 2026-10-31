#!/usr/bin/env python3
"""Videos -> localizable sparse map + MVSFormer++ dense map + EDM/MegaLoc localization bundle.

    python run_pipeline.py VIDEO [VIDEO ...] --output OUT [--intrinsics K.json | --auto_calibrate]

Stages (each resumable; a finished stage with its receipt present is skipped):
    1. OUT/map/            map_stage.py     keyframes + MoGe-3/MegaLoc GlueMap -> model/, sparse_rgb.ply
    2. OUT/dense/          dense_stage.py   MVSFormer++ densification           -> dense.ply (same world frame)
    3. OUT/localization/   localization_stage.py  portable bundle (map + MegaLoc bank + EDM + run_localize.py)

Final deliverables are listed in OUT/pipeline_receipt.json. Copy OUT/localization/bundle to the target machine,
run `python install.py` there, then `python run_localize.py --video q.mp4 --route r --output out/`.

Layout expected (created by setup_envs.sh): <toolkit>/{repo, resources, weights, envs}. Every path can be
overridden with flags. Input videos must be undistorted pinhole footage (or pass --intrinsics for the exact
undistorted camera); the map is NOT metric (monocular, up to Sim3).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def default_toolkit() -> Path | None:
    for candidate in (os.environ.get("TOOLKIT_ROOT"), HERE.parent.parent):
        if candidate and (Path(candidate) / "weights").is_dir():
            return Path(candidate).resolve()
    return None


def step(name: str, cmd: list[str], env: dict[str, str]) -> None:
    print(f"\n=== {name} ===\n+ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    code = subprocess.run(cmd, env=env, check=False).returncode
    if code != 0:
        sys.exit(f"error: stage '{name}' failed with exit code {code}")
    print(f"=== {name} done in {time.time() - t0:.0f}s ===", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="mapping videos / image dirs")
    ap.add_argument("--output", required=True)
    ap.add_argument("--toolkit", default=None, help="toolkit root containing repo/ resources/ weights/ envs/")
    cal = ap.add_mutually_exclusive_group()
    cal.add_argument("--intrinsics", default=None, help="undistorted PINHOLE calibration JSON (recommended)")
    cal.add_argument("--auto_calibrate", action="store_true", help="video-only approximate calibration (default)")
    ap.add_argument("--gluemap-python", default=None)
    ap.add_argument("--moge-python", default=None)
    ap.add_argument("--mvs-python", default=None)
    ap.add_argument("--loc-python", default=os.environ.get("LOC_PYTHON"), help="interpreter for localization stage (default: gluemap python, or $LOC_PYTHON); its sys.prefix must own torch")
    ap.add_argument("--camera-id", default=None, help="query camera id when the map has several cameras")
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--no-bundle-weights", action="store_true", help="do not copy EDM/MegaLoc weights into the bundle")
    ap.add_argument("--map-arg", action="append", default=[], help="extra argument for map_stage.py (repeatable)")
    args = ap.parse_args()

    toolkit = Path(args.toolkit).resolve() if args.toolkit else default_toolkit()
    if toolkit is None:
        sys.exit("error: toolkit root not found; pass --toolkit (must contain weights/ resources/ envs/)")
    weights, res, envs = toolkit / "weights", toolkit / "resources", toolkit / "envs"
    gluemap_py = args.gluemap_python or str(envs / "gluemap/bin/python")
    moge_py = args.moge_python or str(envs / "moge/bin/python")
    mvs_py = args.mvs_python or str(envs / "mvs/bin/python")
    required = {
        "pi3": weights / "pi3.safetensors", "tracker": weights / "tracker.bin", "dg": weights / "dg.pth",
        "moge3": weights / "moge3.pt", "megaloc": weights / "megaloc.safetensors", "edm": weights / "edm.ckpt",
        "gluemap python": Path(gluemap_py), "moge python": Path(moge_py),
    }
    if not args.skip_dense:
        required.update({"mvsformerpp": weights / "mvsformerpp.pth", "mvs config": weights / "mvsformerpp_config.json",
                         "mvs python": Path(mvs_py)})
    missing = [f"{k}: {v}" for k, v in required.items() if not v.exists()]
    if missing:
        sys.exit("error: missing toolkit assets:\n  " + "\n  ".join(missing))

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update({"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "4"), "PYTHONUNBUFFERED": "1"})
    py = sys.executable
    map_dir, dense_dir, loc_dir = out / "map", out / "dense", out / "localization"

    if not (map_dir / "map_receipt.json").exists():
        cmd = [gluemap_py, str(HERE / "map_stage.py"), *args.videos, "--output", str(map_dir),
               "--gluemap_root", str(toolkit / "repo"), "--gluemap_python", gluemap_py,
               "--path_feedforward", str(weights / "pi3.safetensors"), "--path_retrieval", str(weights / "megaloc.safetensors"),
               "--path_tracker", str(weights / "tracker.bin"), "--path_dg", str(weights / "dg.pth"),
               "--path_depth", str(weights / "moge3.pt"), "--moge_python", moge_py]
        cmd += ["--intrinsics", args.intrinsics] if args.intrinsics else ["--auto_calibrate"]
        cmd += args.map_arg
        step("1/3 mapping", cmd, env)
    else:
        print("1/3 mapping: receipt present, skipped")

    if not args.skip_dense:
        if not (dense_dir / "dense_receipt.json").exists():
            step("2/3 densification", [mvs_py, str(HERE / "dense_stage.py"), "--model", str(map_dir / "model"),
                 "--images", str(map_dir / "keyframes/images"), "--output", str(dense_dir),
                 "--checkpoint", str(weights / "mvsformerpp.pth"), "--config", str(weights / "mvsformerpp_config.json"),
                 "--mvsformerpp-root", str(res / "mvsformerpp"), "--mvs-python", mvs_py, "--scene-python", gluemap_py], env)
        else:
            print("2/3 densification: receipt present, skipped")

    if not (loc_dir / "localization_receipt.json").exists():
        cmd = [gluemap_py, str(HERE / "localization_stage.py"), "--model", str(map_dir / "model"),
               "--keyframes", str(map_dir / "keyframes/keyframes.jsonl"), "--intrinsics", str(map_dir / "intrinsics.json"),
               "--output", str(loc_dir), "--sparse-ply", str(map_dir / "sparse_rgb.ply"),
               "--megaloc-source", str(res / "megaloc"), "--megaloc-checkpoint", str(weights / "megaloc.safetensors"),
               "--edm-root", str(res / "edm"), "--edm-checkpoint", str(weights / "edm.ckpt"),
               "--sfm-diagnosis", str(res / "normal_map_pipeline/sfm_diagnosis"),
               "--river-map-quality", str(res / "river_map_quality/river_map_quality"), "--localizer-python", args.loc_python or gluemap_py]
        if not args.skip_dense:
            cmd += ["--dense-ply", str(dense_dir / "dense.ply")]
        if not args.no_bundle_weights:
            cmd.append("--bundle-weights")
        if args.camera_id:
            cmd += ["--camera-id", args.camera_id]
        step("3/3 localization bundle", cmd, env)
    else:
        print("3/3 localization bundle: receipt present, skipped")

    receipt = {
        "schema": "portable-pipeline/v1",
        "sparse_rgb_ply": str(map_dir / "sparse_rgb.ply"),
        "colmap_model": str(map_dir / "model"),
        "dense_ply": None if args.skip_dense else str(dense_dir / "dense.ply"),
        "localization_bundle": str(loc_dir / "bundle"),
        "coordinate_frame": "all three outputs share the map_stage COLMAP world frame (identity transform)",
        "scale": "monocular map: arbitrary unit (up to Sim3), NOT metric",
        "calibration_source": json.loads((map_dir / "map_receipt.json").read_text())["calibration_source"],
        "python": py,
    }
    (out / "pipeline_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("\n" + json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
