#!/usr/bin/env python3
"""Portable MVSFormer++ densification stage.

Densifies a calibrated COLMAP model (PINHOLE cameras, e.g. the map_stage output) with the official
MVSFormer++ checkpoint and writes a dense RGB PLY in the *unchanged* map coordinate frame:

    <output>/dense.ply            float64 xyz + uint8 rgb, same world frame/scale as the input model
    <output>/dense_receipt.json   counts, hashes, settings, checkpoint identity
    <output>/work/                scene conversion + (unless removed) depth maps and fusion shards

Pipeline (each step is a subprocess so the COLMAP and the torch environments may differ):
    1. prepare_scene.py   COLMAP model -> MVSNet scene (exact poses/intrinsics, co-visibility view pairs)  [scene python: pycolmap, numpy, Pillow]
    2. infer_depth.py     MVSFormer++ depth + confidence per view                                        [mvs python: torch+CUDA, cv2, omegaconf, tensorboardX, timm, plyfile]
    3. prepare_exact_geometry.py  float64 cameras for output geometry
    4. fuse_depth.py      dynamic geometric-consistency fusion + voxel averaging                         [mvs python]

No pose or scale is modified; input poses are copied bit-exact. Output units are the input map's units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "dense"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(path: str | None, label: str, *, directory: bool = False) -> Path:
    if not path:
        sys.exit(f"error: {label} is required (flag or environment variable)")
    p = Path(path).expanduser().resolve()
    if not (p.is_dir() if directory else p.is_file()):
        sys.exit(f"error: {label} not found: {p}")
    return p


def run(cmd: list[str], env: dict[str, str], log: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    with log.open("ab") as fh:
        fh.write(("\n$ " + " ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log.read_text(errors="replace").splitlines()[-25:]
        sys.exit(f"error: step failed ({proc.returncode}): {cmd[1]}\n" + "\n".join(tail))


def main() -> None:
    env_default = os.environ.get
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="COLMAP binary model directory (PINHOLE cameras)")
    ap.add_argument("--images", required=True, help="image root; model image names are relative to it")
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default=env_default("DENSE_CHECKPOINT"), help="MVSFormer++ model_best.pth")
    ap.add_argument("--config", default=env_default("DENSE_CONFIG"), help="MVSFormer++ checkpoint config.json")
    ap.add_argument("--mvsformerpp-root", default=env_default("MVSFORMERPP_ROOT"), help="MVSFormer++ source tree (SDPA-patched attention)")
    ap.add_argument("--mvs-python", default=env_default("DENSE_MVS_PYTHON", sys.executable), help="interpreter with torch+CUDA etc.")
    ap.add_argument("--scene-python", default=env_default("DENSE_SCENE_PYTHON", sys.executable), help="interpreter with pycolmap, numpy, Pillow")
    ap.add_argument("--width", type=int, default=1536, help="inference width (multiple of 64)")
    ap.add_argument("--height", type=int, default=832, help="inference height (multiple of 64)")
    ap.add_argument("--views", type=int, default=5, help="views per inference (reference + sources)")
    ap.add_argument("--num-src", type=int, default=10, help="source views per reference for fusion")
    ap.add_argument("--voxel", type=float, default=None, help="fusion voxel size in map units (default 2.2e-4 x scene extent)")
    ap.add_argument("--keep-work", action="store_true", help="keep depth maps and fusion shards (large)")
    ap.add_argument("--resume", action="store_true", help="reuse existing scene/depths in --output/work")
    args = ap.parse_args()

    model = need(args.model, "--model", directory=True)
    images = need(args.images, "--images", directory=True)
    checkpoint = need(args.checkpoint, "--checkpoint")
    config = need(args.config, "--config")
    repo = need(args.mvsformerpp_root, "--mvsformerpp-root", directory=True)
    mvs_python, scene_python = args.mvs_python, args.scene_python
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (model / name).is_file():
            sys.exit(f"error: not a COLMAP binary model, missing {name}: {model}")

    out = Path(args.output).expanduser().resolve()
    work = out / "work"
    if work.exists() and not args.resume:
        sys.exit(f"error: {work} exists; use --resume to continue or remove it")
    (work / "logs").mkdir(parents=True, exist_ok=True)
    log = work / "logs/dense.log"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["MVSFORMERPP_ROOT"] = str(repo)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    scene_name = "scene"

    if not (work / "conversion_receipt.json").exists():
        run([scene_python, str(HERE / "prepare_scene.py"), "--model", str(model), "--image_root", str(images),
             "--work_root", str(work), "--scene", scene_name, "--num_src", str(args.num_src)], env, log)
    run([mvs_python, str(HERE / "infer_depth.py"), "--run", str(work), "--checkpoint", str(checkpoint),
         "--config", str(config), "--width", str(args.width), "--height", str(args.height),
         "--views", str(args.views)], env, log)
    run([mvs_python, str(HERE / "prepare_exact_geometry.py"), "--run", str(work)], env, log)
    fuse = [mvs_python, str(HERE / "fuse_depth.py"), "--run", str(work)]
    if args.voxel is not None:
        fuse += ["--voxel", str(args.voxel)]
    run(fuse, env, log)

    produced = work / "products" / f"{scene_name}_mvsformerpp.ply"
    if not produced.is_file():
        sys.exit(f"error: fusion did not produce {produced}")
    dense = out / "dense.ply"
    shutil.move(str(produced), dense)
    fusion = json.loads((work / "fusion/fusion_receipt.json").read_text())
    conversion = json.loads((work / "conversion_receipt.json").read_text())
    receipt = {
        "schema": "portable-dense-stage/v1",
        "dense_ply": str(dense),
        "dense_ply_sha256": sha256(dense),
        "points": fusion["points_written"],
        "raw_geometrically_fused_points": fusion["raw_geometrically_fused_points"],
        "voxel_size_map_units": fusion["voxel_size_original_map_units"],
        "views": conversion["conversion"]["registered_views"],
        "scene_extent_map_units": conversion["conversion"]["scene_extent_m"],
        "world_transform": "identity (input model coordinate frame and scale unchanged; units are the input map's units)",
        "model": {name: sha256(model / name) for name in ("cameras.bin", "images.bin", "points3D.bin")},
        "checkpoint": {"path_name": checkpoint.name, "sha256": sha256(checkpoint)},
        "inference": {"width": args.width, "height": args.height, "views": args.views, "num_src": args.num_src},
        "seconds": time.time() - t0,
    }
    (out / "dense_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not args.keep_work:
        for name in ("depths", "fusion"):
            shutil.rmtree(work / name, ignore_errors=True)
    print(json.dumps({k: receipt[k] for k in ("dense_ply", "points", "views", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
