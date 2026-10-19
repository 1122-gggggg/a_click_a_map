"""Compatibility worker for the public Dense-SfM refinement implementation.

Dense-SfM loads images by basename and expects an adjacent COLMAP database.
River models use hierarchical names and retain no database, so this worker
creates an isolated, reversible staging model/database before invoking the
official post-optimization function.  It uses PyCOLMAP 3.11's exposed bundle
adjustment path instead of the repository's custom COLMAP executable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .map_refinement import flatten_image_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="densesfm-worker")
    commands = parser.add_subparsers(dest="command", required=True)
    refine = commands.add_parser("refine")
    refine.add_argument("--img_folder", type=Path, required=True)
    refine.add_argument("--colmap_coarse_dir", type=Path, required=True)
    refine.add_argument("--staging-dir", type=Path, required=True)
    refine.add_argument("--refined_colmap_dir", type=Path, required=True)
    refine.add_argument("--config", type=Path, required=True)
    refine.add_argument("--database-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_refine(
            args.img_folder,
            args.colmap_coarse_dir,
            args.staging_dir,
            args.refined_colmap_dir,
            args.config,
            args.database_path,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "reason": str(error)}))
        return 2
    print(json.dumps({"status": "COMPLETED", **result}))
    return 0


def run_refine(
    image_root: Path,
    input_model: Path,
    staging_dir: Path,
    output_model: Path,
    config_path: Path,
    database_path: Path,
) -> dict[str, object]:
    import pycolmap
    import yaml

    from src.post_optimization.post_optimization import post_optimization  # ty: ignore[unresolved-import]

    if staging_dir.exists():
        raise FileExistsError(staging_dir)
    staged_images = staging_dir / "images"
    staged_model = staging_dir / "model"
    staged_images.mkdir(parents=True)
    staged_model.mkdir()

    reconstruction = pycolmap.Reconstruction(str(input_model.resolve(strict=True)))
    name_map: dict[str, str] = {}
    seen: set[str] = set()
    registered = set(int(value) for value in reconstruction.reg_image_ids())
    for image_id in sorted(registered):
        image = reconstruction.images[image_id]
        original = str(image.name)
        flattened = flatten_image_name(original)
        if flattened in seen:
            raise ValueError(f"Dense-SfM staging name collision: {flattened}")
        seen.add(flattened)
        source = image_root.resolve(strict=True) / original
        if not source.is_file():
            raise FileNotFoundError(source)
        target = staged_images / flattened
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        image.name = flattened
        name_map[flattened] = original
    reconstruction.write(str(staged_model))
    (staging_dir / "image_name_map.json").write_text(
        json.dumps(name_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    database = open_database_compat(pycolmap.Database, database_path)
    try:
        for camera_id in sorted(reconstruction.cameras):
            database.write_camera(reconstruction.cameras[camera_id], use_camera_id=True)
        for image_id in sorted(registered):
            image = reconstruction.images[image_id]
            database.write_image(image, use_image_id=True)
            points = np.asarray([point.xy for point in image.points2D], dtype=np.float32)
            database.write_keypoints(image_id, points)
        tracks = (
            [(int(element.image_id), int(element.point2D_idx)) for element in point.track.elements]
            for point in reconstruction.points3D.values()
        )
        pair_matches = aggregate_track_matches(tracks)
        for (image_id1, image_id2), rows in sorted(pair_matches.items()):
            values = np.asarray(rows, dtype=np.uint32)
            database.write_matches(image_id1, image_id2, values)
            geometry = pycolmap.TwoViewGeometry()
            geometry.config = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            geometry.inlier_matches = values
            database.write_two_view_geometry(image_id1, image_id2, geometry)
    finally:
        database.close()

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    neural = dict(payload.get("neuralsfm") or {})
    colmap = dict(payload.get("colmap_cfg") or {})
    colmap["use_pycolmap"] = True
    images = sorted(staged_images.iterdir())
    state = post_optimization(
        [str(path) for path in images],
        None,
        match_out_pth=None,
        chunk_size=int(neural.get("NEUSFM_refinement_chunk_size", 500)),
        matcher_model_path=str(neural["NEUSFM_fine_match_model_path"]),
        matcher_cfg_path=str(neural["NEUSFM_fine_match_cfg_path"]),
        img_resize=int(neural.get("img_resize", 1200)),
        img_preload=bool(neural.get("img_preload", False)),
        colmap_coarse_dir=str(staged_model),
        refined_model_save_dir=str(output_model),
        only_basename_in_colmap=True,
        fine_match_use_ray=False,
        ray_cfg=None,
        colmap_configs=colmap,
        refine_iter_n_times=int(neural.get("refine_iter_n_times", 2)),
        refine_3D_pts_only=False,
        database_path=str(database_path),
        use_pycolmap=True,
    )
    if not state:
        raise RuntimeError("Dense-SfM post-optimization returned an empty dataset")
    for filename in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (output_model / filename).is_file():
            raise RuntimeError(f"Dense-SfM output is missing {filename}")

    output = pycolmap.Reconstruction(str(output_model))
    for image in output.images.values():
        if image.name not in name_map:
            raise RuntimeError(f"Dense-SfM output contains unknown image {image.name}")
        image.name = name_map[image.name]
    output.write(str(output_model))
    return {
        "staged_images": len(images),
        "staged_model": str(staged_model),
        "database": str(database_path),
        "output_model": str(output_model),
        "restored_hierarchical_names": True,
        "pycolmap": pycolmap.__version__,
    }


def open_database_compat(database_type, path: Path):
    """Open both PyCOLMAP 3.11's instance API and 4.x's class API."""

    try:
        return database_type.open(str(path))
    except TypeError:
        database = database_type()
        database.open(str(path))
        return database


def aggregate_track_matches(
    tracks: Iterable[Sequence[tuple[int, int]]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Convert multi-view tracks into orientation-stable pairwise correspondences."""

    matches: defaultdict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for track in tracks:
        # GlueMap may attach multiple Point2D observations from one image to
        # the same Point3D.  COLMAP's database correspondence graph requires
        # a one-to-one feature relation per image pair, so keep one stable
        # representative observation per image for database connectivity.
        by_image: dict[int, int] = {}
        for image_id, point2d_idx in track:
            by_image[image_id] = min(point2d_idx, by_image.get(image_id, point2d_idx))
        for left, right in combinations(sorted(by_image.items()), 2):
            if left[0] < right[0]:
                pair = (left[0], right[0])
                indexes = (left[1], right[1])
            else:
                pair = (right[0], left[0])
                indexes = (right[1], left[1])
            matches[pair].append(indexes)
    return dict(matches)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "aggregate_track_matches",
    "build_parser",
    "main",
    "open_database_compat",
    "run_refine",
]
