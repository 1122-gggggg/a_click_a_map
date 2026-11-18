"""Export a finished original-entry run as a localization-side site folder.

The other machine runs wulai-sfm-localization, whose importer only accepts the
layout described in that repo's `site_profiles/建圖端輸出規格.md`::

    <site>/
      site_profile.json                  (schema_version 2)
      map/map.ply                        (RGB cloud, x y z + red green blue)
    map/map_dense.ply                  (optional dense display cloud, if the run has one)
      model/{cameras,images,points3D,rigs,frames}.bin   (exactly these; no extra files)
      keyframes/keyframes.jsonl + keyframes/images/<video>/*.jpg
      localization/direct_bundle.json    (direct-localization-bundle/v1)
      localization/direct_localizer_profile.json (direct-deployment-profile/v1)
      localization/reference_poses.json  (reference-poses/v2)
      localization/T_align_gravity.json  (sfm-align/v2)
      localization/megaloc_references_sideview.{npy,names.json}
      localization/reference_manifest.jsonl

Everything the importer verifies is produced here: file bytes, SHA-256 lists,
frame digest, gravity alignment, poses. The frozen localizer profile values are
copied from wulai's builder (`tools/build_direct_site_release.py`, FROZEN_*
constants); the only intentional deviation is `reloc.reference_bank:
"sideview"`, because this pipeline ships the 8448-d MegaLoc side-view bank,
not the newer 16384-d BoQ bank.

Query camera assumption: the deployment streams 1280x720. The model K is
rescaled to that resolution (same aspect required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

QUERY_WIDTH = 1280
QUERY_HEIGHT = 720
MAP_PLY_TARGET_POINTS = 150_000

MODEL_BINS_REQUIRED = ("cameras.bin", "images.bin", "points3D.bin")
MODEL_BINS_OPTIONAL = ("rigs.bin", "frames.bin")

# Frozen localizer profile, copied from wulai-sfm-localization
# tools/build_direct_site_release.py (FROZEN_RELOC / FROZEN_FAST_LOOP / FROZEN_VO /
# FROZEN_DEAD_RECKON), except reloc.reference_bank which must name the bank this
# pipeline actually ships ("sideview" = MegaLoc 8448-d, not "sideview-boq").
FROZEN_RELOC = {
    "top_k": 2,
    "lift_distance_px": 2.0,
    "period_s": 1.0,
    "min_points": 60,
    "reference_bank": "sideview",
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
}
FROZEN_FAST_LOOP = {
    "resolution": [960, 540],
    "tracker": "klt",
    "track_cap": 500,
    "reseed_min_points": 120,
    "topup": False,
    "klt": {"fb_max_px": 1.0, "win": 15, "levels": 3},
    "pnp": {
        "max_error_px": 4.0,
        "min_trials": 10,
        "max_trials": 100,
        "confidence": 0.999,
        "refine_iters": 2,
        "min_points": 12,
        "min_inliers": 12,
    },
}
FROZEN_VO = {
    "enabled": True,
    "keyframe_stride": 8,
    "batch_lag": 6,
    "min_live": 350,
    "detect_cap": 300,
    "min_distance": 8,
    "min_parallax_deg": 0.5,
    "max_reproj_px": 2.0,
    "refine": False,
    "window_ba": False,
}
FROZEN_DEAD_RECKON = {"enabled": True, "max_frames": 300, "max_age_s": 10.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(model_dir: Path) -> str:
    """SHA-256 over cameras.bin + images.bin + points3D.bin contents, in order."""
    digest = hashlib.sha256()
    for name in MODEL_BINS_REQUIRED:
        with (model_dir / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return sha256_file(path)


def query_camera_from_model_K(K: np.ndarray, width: int, height: int) -> dict:
    """Rescale the map calibration to the 1280x720 deployment stream."""
    if width <= 0 or height <= 0:
        raise ValueError(f"bad model resolution {width}x{height}")
    scale = QUERY_WIDTH / width
    if abs(height * scale - QUERY_HEIGHT) > 1e-6:
        raise ValueError(
            f"model aspect {width}x{height} does not match the "
            f"{QUERY_WIDTH}x{QUERY_HEIGHT} deployment stream; recalibrate first"
        )
    fx, _, cx = float(K[0, 0]) * scale, 0.0, float(K[0, 2]) * scale
    _, fy, cy = 0.0, float(K[1, 1]) * scale, float(K[1, 2]) * scale
    if not (fx > 0 and fy > 0):
        raise ValueError("model focal lengths must be positive")
    return {"model": "PINHOLE", "width": QUERY_WIDTH, "height": QUERY_HEIGHT,
            "params": [fx, fy, cx, cy]}


def write_map_ply(points_xyz: np.ndarray, points_rgb: np.ndarray, path: Path,
                  target_points: int = MAP_PLY_TARGET_POINTS) -> None:
    """Deterministic stride sample of the sparse cloud, binary little-endian PLY."""
    order = np.lexsort((points_xyz[:, 2], points_xyz[:, 1], points_xyz[:, 0]))
    xyz, rgb = points_xyz[order], points_rgb[order]
    stride = max(1, int(math.ceil(len(xyz) / max(1, target_points))))
    xyz, rgb = xyz[::stride], rgb[::stride]
    record = struct.Struct("<fffBBB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(
            ("ply\nformat binary_little_endian 1.0\n"
             f"comment deterministic stride sample: every {stride} of {len(order)} points; display only\n"
             f"element vertex {len(xyz)}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode("ascii"))
        chunk = bytearray()
        for (x, y, z), (r, g, b) in zip(xyz.tolist(), rgb.tolist()):
            chunk += record.pack(float(x), float(y), float(z), int(r), int(g), int(b))
            if len(chunk) >= 1 << 20:
                stream.write(chunk)
                chunk = bytearray()
        stream.write(chunk)


def derive_gravity_alignment(rotations: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gravity-from-poses measurement (same math as wulai's derive_map_gravity)."""
    if len(rotations) < 100:
        raise ValueError(f"need at least 100 registered images to measure gravity alignment, got {len(rotations)} "
                         "(the localization side requires map_align; map more footage)")
    right = rotations[:, 0]
    down = rotations[:, 1]
    scatter = np.einsum("ni,nj->ij", right, right)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    gravity = eigenvectors[:, 0] / np.linalg.norm(eigenvectors[:, 0])
    agreement = float(np.mean(down @ gravity))
    if agreement < 0.0:
        gravity, agreement = -gravity, -agreement
    if agreement < 0.30:
        raise ValueError(f"cannot determine DOWN end of gravity axis (agreement {agreement:.3f})")
    residual_p95 = float(np.percentile(np.abs(right @ gravity), 95))
    residual_deg = math.degrees(math.asin(min(1.0, residual_p95)))
    if residual_deg > 5.0:
        raise ValueError(f"camera right axes not horizontal enough (p95 {residual_deg:.2f} deg)")
    up = -gravity / np.linalg.norm(gravity)
    east = np.array([1.0, 0.0, 0.0]) - float(np.dot([1.0, 0.0, 0.0], up)) * up
    if float(np.linalg.norm(east)) < 1e-6:
        raise ValueError("gravity is parallel to GLOMAP +X; azimuth reference degenerate")
    east = east / np.linalg.norm(east)
    rotation = np.vstack([east, np.cross(up, east), up])
    deviation = math.degrees(math.acos(float(np.clip(np.dot(rotation[2], [0.0, -1.0, 0.0]), -1.0, 1.0))))
    return rotation, {
        "n_poses": int(len(rotations)),
        "residual_p95": residual_p95,
        "residual_deg": residual_deg,
        "sign_agreement": agreement,
        "deviation_from_old_Yup_deg": deviation,
    }


def gravity_document(frame_name: str, rotations: np.ndarray, centres: np.ndarray) -> dict:
    rotation, evidence = derive_gravity_alignment(rotations)
    aligned = centres @ rotation.T
    extent = (aligned.max(axis=0) - aligned.min(axis=0)).tolist()
    return {
        "schema": "sfm-align/v2",
        "frame_from": f"{frame_name}_glomap",
        "frame_to": f"{frame_name}_gravity_aligned_Zup",
        "R": rotation.tolist(),
        "t": [0.0, 0.0, 0.0],
        "note": "p_aligned = R @ p_glomap. Z is up (anti-gravity), X keeps the GLOMAP +X azimuth.",
        "gravity_glomap": (-rotation[2]).tolist(),
        "derivation": {
            "method": "smallest eigenvector of sum(right_i right_i^T) over camera X axes",
            "assumption": "gimbal holds roll ~0, so every camera right axis is horizontal",
            "sign_convention": "gravity points DOWN, resolved against the mean camera down axis (never against a world axis)",
            "n_poses": evidence["n_poses"],
            "residual_right_dot_g_p95": evidence["residual_p95"],
            "residual_deg_p95": evidence["residual_deg"],
            "sign_agreement": evidence["sign_agreement"],
            "deviation_from_old_Yup_deg": evidence["deviation_from_old_Yup_deg"],
            "source_poses": "reference_poses.json",
            "tool": "acam/export_site.py (same math as wulai derive_map_gravity.py)",
        },
        "aligned_extent": {"camera_bbox": extent, "height_range": extent[2]},
    }


def find_run_model(run: Path) -> Path:
    model = run / "artifacts/mapping/edm/model"
    for name in MODEL_BINS_REQUIRED:
        if not (model / name).is_file():
            raise ValueError(f"run model is missing {name}: {model}")
    return model


def export_site(run: Path, out: Path, *, site_id: str, display_name: str, release_id: str) -> Path:
    """Build the localization-side site folder from a finished original-entry run."""
    import pycolmap

    if not site_id or not site_id.replace(".", "").replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"bad site_id {site_id!r}: letters, digits, dot, underscore, hyphen")
    run = run.resolve(strict=True)
    model_dir = find_run_model(run)
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    if reconstruction.num_images() == 0 or reconstruction.num_points3D() == 0:
        raise ValueError("model has no images or points")

    keyframes_manifest_src = run / "artifacts/keyframes/keyframes.jsonl"
    if not keyframes_manifest_src.is_file():
        raise ValueError(f"run is missing keyframes manifest: {keyframes_manifest_src}")
    bank_dir = run / "products/localization"
    bank_names = json.loads((bank_dir / "megaloc_references.names.json").read_text(encoding="utf-8"))
    ref_manifest_rows = [json.loads(line) for line in (bank_dir / "reference_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ref_names = [str(row["image_name"]) for row in ref_manifest_rows]
    if len(set(ref_names)) != len(ref_names):
        raise ValueError("reference identity list is not unique")
    if list(bank_names) != ref_names:
        raise ValueError("bank names and reference manifest disagree; refusing to export a mixed release")

    # --- camera: model K rescaled to the deployment stream -------------------
    cameras = list(reconstruction.cameras.values())
    if len({(c.width, c.height) for c in cameras}) != 1:
        raise ValueError("model mixes resolutions; export one calibration group at a time")
    camera = cameras[0]
    if camera.model_name.upper() != "PINHOLE":
        raise ValueError(f"model camera must be PINHOLE, got {camera.model_name}")
    fx, fy, cx, cy = (float(v) for v in camera.params)
    query_camera = query_camera_from_model_K(
        np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]), camera.width, camera.height)

    # --- poses, scale ---------------------------------------------------------
    poses: dict[str, dict] = {}
    centres = np.zeros((reconstruction.num_images(), 3))
    rotations = np.zeros((reconstruction.num_images(), 3, 3))
    for index, image in enumerate(sorted(reconstruction.images.values(), key=lambda item: item.name)):
        cam_from_world = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
        matrix = cam_from_world.matrix()
        rotation = np.asarray(matrix[:3, :3], dtype=np.float64)
        translation = np.asarray(matrix[:3, 3], dtype=np.float64)
        centres[index] = -rotation.T @ translation
        rotations[index] = rotation
        poses[image.name] = {"R": rotation.tolist(), "t": translation.tolist(), "camera_id": "query_camera"}
    missing = [name for name in ref_names if name not in poses]
    if missing:
        raise ValueError(f"{len(missing)} reference images are not registered, first: {missing[0]}")
    scale_center = np.median(centres, axis=0)
    map_scale = float(2.0 * np.percentile(np.linalg.norm(centres - scale_center, axis=1), 95))

    # --- identity -------------------------------------------------------------
    digest = frame_digest(model_dir)
    coordinate_frame_id = f"{release_id}_{digest[:12]}"

    site = out / site_id
    model_out = site / "model"
    keyframes_out = site / "keyframes"
    localization = site / "localization"

    # --- bulk payload ----------------------------------------------------------
    # Layout matches the localization side's import contract exactly: model bins
    # at top-level model/ (occupancy.json is NOT part of it).
    present_bins = [n for n in MODEL_BINS_REQUIRED + MODEL_BINS_OPTIONAL if (model_dir / n).is_file()]
    model_out.mkdir(parents=True)
    for name in present_bins:
        shutil.copy2(model_dir / name, model_out / name)

    keyframes_out.mkdir(parents=True)
    shutil.copy2(keyframes_manifest_src, keyframes_out / "keyframes.jsonl")
    shutil.copytree(run / "artifacts/keyframes/images", keyframes_out / "images",
                    ignore_dangling_symlinks=True)

    localization.mkdir(parents=True)
    bank_descriptors = localization / "megaloc_references_sideview.npy"
    bank_names_path = localization / "megaloc_references_sideview.names.json"
    shutil.copy2(bank_dir / "megaloc_references.npy", bank_descriptors)
    shutil.copy2(bank_dir / "megaloc_references.names.json", bank_names_path)
    ref_manifest_path = localization / "reference_manifest.jsonl"
    with ref_manifest_path.open("w", encoding="utf-8") as stream:
        for row in ref_manifest_rows:
            stream.write(json.dumps({"image_name": row["image_name"], "observations": row.get("observations", 0)}) + "\n")

    # --- map ply (stride sample of the sparse model) ---------------------------
    points_xyz = np.array([p.xyz for p in reconstruction.points3D.values()])
    points_rgb = np.array([p.color for p in reconstruction.points3D.values()], dtype=np.uint8)
    map_ply = site / "map/map.ply"
    write_map_ply(points_xyz, points_rgb, map_ply)
    # Optional dense display cloud (not consumed by reloc, for the UI only).
    dense_src = run / "products/map/dense/dense.ply"
    dense_dst = site / "map/map_dense.ply"
    has_dense_display = dense_src.is_file()
    if has_dense_display:
        shutil.copy2(dense_src, dense_dst)

    # --- reference poses + gravity ---------------------------------------------
    reference_poses = {
        "schema": "reference-poses/v2",
        "schema_version": 2,
        "coordinate_frame_id": coordinate_frame_id,
        "source_model_sha256": digest,
        "camera": query_camera,
        "cameras": {"query_camera": query_camera},
        "ref_names": ref_names,
        "poses": poses,
    }
    poses_path = localization / "reference_poses.json"
    write_json(poses_path, reference_poses)
    gravity = gravity_document(release_id, rotations, centres)
    gravity_path = localization / "T_align_gravity.json"
    write_json(gravity_path, gravity)

    # --- direct bundle manifest --------------------------------------------------
    def entry(relative: str, absolute: Path) -> dict:
        return {"path": relative, "sha256": sha256_file(absolute), "size_bytes": absolute.stat().st_size}

    model_sha256 = {name: sha256_file(model_out / name) for name in present_bins}
    files = [entry(f"../model/{name}", model_out / name) for name in present_bins]
    files.append(entry("reference_manifest.jsonl", localization / "reference_manifest.jsonl"))
    files.append(entry("megaloc_references_sideview.npy", bank_descriptors))
    files.append(entry("megaloc_references_sideview.names.json", bank_names_path))
    files.append(entry("../keyframes/keyframes.jsonl", keyframes_out / "keyframes.jsonl"))
    if has_dense_display:
        files.append(entry("../map/map_dense.ply", dense_dst))
    bundle = {
        "schema": "direct-localization-bundle/v1",
        "map_revision_id": release_id,
        "coordinate_frame_id": coordinate_frame_id,
        "model_dir": "../model",
        "keyframes_manifest": "../keyframes/keyframes.jsonl",
        "keyframes_images_root": "../keyframes/images",
        "reference_manifest": "reference_manifest.jsonl",
        "reference_bank": {"name": "sideview", "descriptors": "megaloc_references_sideview.npy",
                           "names": "megaloc_references_sideview.names.json"},
        "intersection_cells": None,
        "reference_depth_dir": None,
        "model_sha256": model_sha256,
        "files": files,
    }
    bundle_path = localization / "direct_bundle.json"
    bundle_sha = write_json(bundle_path, bundle)

    # --- frozen localizer profile --------------------------------------------------
    profile = {
        "schema": "direct-deployment-profile/v1",
        "name": f"{site_id}_direct_frozen",
        "map_scale": map_scale,
        "intrinsics": {"schema_version": 1, "image_width": QUERY_WIDTH, "image_height": QUERY_HEIGHT,
                       "K": [[query_camera["params"][0], 0.0, query_camera["params"][2]],
                             [0.0, query_camera["params"][1], query_camera["params"][3]],
                             [0.0, 0.0, 1.0]],
                       "images_are_undistorted": True},
        "reloc": FROZEN_RELOC,
        "fast_loop": FROZEN_FAST_LOOP,
        "vo": FROZEN_VO,
        "dead_reckon": FROZEN_DEAD_RECKON,
        "max_jump_u": 1.5,
    }
    profile_path = localization / "direct_localizer_profile.json"
    profile_sha = write_json(profile_path, profile)

    # --- site profile --------------------------------------------------------------
    map_ply_sha = sha256_file(map_ply)
    poses_sha = sha256_file(poses_path)
    gravity_sha = sha256_file(gravity_path)
    site_profile = {
        "schema_version": 2,
        "site_id": site_id,
        "display_name": display_name,
        "localizer": "direct",
        # Installed sites live at <workspace>/地圖檔/場域/<site_id>/, i.e. three
        # levels below the repo root, so the deploy dir is three ups. (The
        # 建圖端輸出規格 example shows five ups, which matches the release
        # builder's staging layout, not the importer's managed layout.)
        "localizer_deploy_dir": "../../../定位演算法/deploy_code/sfm_direct_deploy",
        "localizer_profile": "localization/direct_localizer_profile.json",
        "map_reference_poses": "localization/reference_poses.json",
        "map_align": "localization/T_align_gravity.json",
        "query_camera": query_camera,
        "coordinate_frame": {"id": coordinate_frame_id, "convention": "glomap",
                             "horizontal_axes": ["x", "z"], "up_axis": "-y",
                             "handedness": "right", "units": "map"},
        "asset_sha256": {"map_ply": map_ply_sha, "localization_bundle": bundle_sha,
                         "localizer_profile": profile_sha, "map_reference_poses": poses_sha,
                         "map_align": gravity_sha},
        "assets": {"map_ply": "map/map.ply", "route_json": None,
                   "localization_bundle": "localization/direct_bundle.json",
                   "megaloc_cache": None, "track_landmarks": None, "poles_json": None},
    }
    write_json(site / "site_profile.json", site_profile)
    return site


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog="acam-export-site")
    parser.add_argument("run", type=Path, help="finished run dir (contains artifacts/, products/)")
    parser.add_argument("--out", type=Path, required=True, help="output dir holding <site_id>/")
    parser.add_argument("--site-id", required=True, help="path-safe slug, e.g. riverside02")
    parser.add_argument("--display-name", default="", help="human name (defaults to site-id)")
    parser.add_argument("--release-id", default="", help="map revision id (defaults to <site_id>_<date>)")
    args = parser.parse_args(argv)
    import datetime
    release = args.release_id or f"{args.site_id}_{datetime.date.today():%Y%m%d}"
    try:
        site = export_site(args.run, args.out, site_id=args.site_id,
                           display_name=args.display_name or args.site_id, release_id=release)
    except (ValueError, OSError) as error:
        print(f"export-site failed: {error}", file=sys.stderr)
        return 1
    print(site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
