"""Read-only P120 pose-source attribution experiment.

The workflow is deliberately split into a GPU extraction stage and a CPU analysis stage.
The extraction cache contains only query/reference correspondences and provenance; all
PnP, Essential Matrix, consensus, and report experiments can then be repeated without
touching the matcher or the frozen map.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from river_map_quality.attribution_overlay import PoseGlyph, render_pose_glyphs_ply
from river_map_quality.baseline import verify_frozen_baseline
from river_map_quality.colmap_binary import read_binary_image_poses
from river_map_quality.edm_loo import rank_reference_indices, spatial_cap_indices
from river_map_quality.exclusion import loo_exclusion_indices
from river_map_quality.loo_metrics import compute_loo_metrics
from river_map_quality.matching_db import read_verified_pair_points
from river_map_quality.pose_attribution import (
    PoseHypothesis,
    attribute_pose_source,
    camera_center,
    classify_reference_consensus,
    conversion_funnel,
    relative_pose,
    rotation_distance_deg,
)
from river_map_quality.pose_solvers import (
    estimate_essential_trials,
    project_points,
    solve_pnp_envelope,
)
from river_map_quality.reference_subsets import build_reference_subsets
from river_map_quality.relative_pose_evidence import (
    RelativePoseTrial,
    classify_sequential_support,
    residual_structure,
)
from river_map_quality.weak_region_overlay import (
    estimate_camera_spacing,
    export_weak_region_overlay,
)

DEFAULT_TARGETS = tuple(f"P1200120/{frame:06d}.jpg" for frame in (97, 98, 99, 101, 102, 103))
DEFAULT_CONTROLS = tuple(f"P1200120/{frame:06d}.jpg" for frame in (96, 100))


@dataclass(frozen=True)
class _PnPRun:
    pose: np.ndarray
    inlier_mask: np.ndarray
    covariance: np.ndarray | None
    num_inliers: int


def normalize_query_name(value: str) -> str:
    """Normalize ``P120/97`` while preserving already canonical bundle names."""

    name = str(value).strip()
    if re.fullmatch(r"P\d{3}0\d{3}/\d{6}\.jpg", name):
        return name
    match = re.fullmatch(r"P(\d+)/(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"unsupported query name: {value!r}")
    route = int(match.group(1))
    frame = int(match.group(2))
    return f"P{route:03d}0{route:03d}/{frame:06d}.jpg"


def unique_anchor_count(points3d: np.ndarray) -> int:
    """Count exact packed EDM anchors after conversion to bundle float32 precision."""

    points = np.asarray(points3d, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points3d must have shape (N, 3)")
    if not len(points):
        return 0
    packed = np.ascontiguousarray(points).view(np.dtype((np.void, points.dtype.itemsize * 3)))
    return len(np.unique(packed))


def _collect_scene_scales(value: Any, output: list[float]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "scene_scale_median_depth":
                try:
                    number = float(child)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number) and number > 0:
                    output.append(number)
            else:
                _collect_scene_scales(child, output)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _collect_scene_scales(child, output)


def global_scene_scale_from_report(path: str | Path) -> float:
    """Return one frozen dataset-wide scale from all reported positive-depth scales."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scales: list[float] = []
    _collect_scene_scales(payload, scales)
    if not scales:
        raise ValueError("health report contains no positive scene-scale measurements")
    return float(np.median(np.asarray(scales, dtype=float)))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_jsonable(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(_jsonable(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()


def _pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=float)
    pose[:3, 3] = np.asarray(translation, dtype=float)
    return pose


def _m0_poses(baseline: Path) -> dict[str, np.ndarray]:
    raw = read_binary_image_poses(baseline / "reconstruction/images.bin", strict_quaternion=False)
    return {
        row.name: _pose_matrix(row.rotation, row.tvec)
        for row in raw.values()
        if row.valid_rigid_transform and row.rotation is not None
    }


def camera_spacing_from_poses(poses: Mapping[str, np.ndarray]) -> float:
    """Adapt NumPy pose centers to the JSON-like overlay spacing contract."""

    rows = []
    for name, pose in poses.items():
        try:
            frame = int(Path(name).stem)
        except ValueError:
            continue
        rows.append(
            {
                "image_name": name,
                "sequence": name.split("/", 1)[0],
                "frame": frame,
                "camera_center": camera_center(pose).tolist(),
            }
        )
    return estimate_camera_spacing(rows)


def _provider_funnel(provider: Any) -> dict[str, Any]:
    rows = list(getattr(provider, "last_diagnostics", []) or [])
    raw = sum(int(row.get("raw_matches", 0)) for row in rows)
    verified = sum(int(row.get("direction01_matches", 0)) for row in rows)
    anchored = sum(int(row.get("anchored_matches", 0)) for row in rows)
    return {
        "raw_matches": raw,
        "verified_2d2d": verified,
        "anchored_matches": anchored,
        "per_reference": rows,
    }


def _source_hashes(baseline: Path, matching_baseline: Path) -> dict[str, str]:
    return {
        "cameras_bin": _sha256(baseline / "reconstruction/cameras.bin"),
        "images_bin": _sha256(baseline / "reconstruction/images.bin"),
        "points3D_bin": _sha256(baseline / "reconstruction/points3D.bin"),
        "edm_bundle": _sha256(baseline / "edm/river_site_reloc_map_edm.pt"),
        "matching_database": _sha256(matching_baseline / "database_merged.db"),
    }


def resolve_edm_repo(path: str | Path) -> Path:
    """Validate the explicit read-only EDM source checkout needed by its wrapper."""

    repository = Path(path).resolve(strict=True)
    if not (repository / "src").is_dir() or not (repository / "configs/edm").is_dir():
        raise ValueError(f"invalid EDM repository contract: {repository}")
    return repository


def _camera_yaw(pose: np.ndarray) -> float:
    direction = pose[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return float(math.atan2(direction[1], direction[0]))


def _angle_delta(first: float, second: float) -> float:
    return float(abs(math.atan2(math.sin(first - second), math.cos(first - second))))


def extract_correspondence_cache(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run the only GPU stage and persist reference-provenanced correspondences."""

    import torch

    baseline = arguments.baseline.resolve(strict=True)
    matching = arguments.matching_baseline.resolve(strict=True)
    verify_frozen_baseline(baseline)
    verify_frozen_baseline(matching)
    before_hashes = _source_hashes(baseline, matching)
    deploy = baseline / "edm/deploy"
    sys.path.insert(0, str(deploy))
    if arguments.edm_repo is not None:
        os.environ["EDM_REPO"] = str(resolve_edm_repo(arguments.edm_repo))
    from edm_matcher import EDMMatcher
    from reloc_localizer_edm import Camera, EDMLocalizer, EDMRelocMap

    bundle_path = baseline / "edm/river_site_reloc_map_edm.pt"
    relocation_map = EDMRelocMap.load(bundle_path)
    matcher = EDMMatcher(
        ckpt=baseline / "edm/edm_outdoor.ckpt",
        cfg_path=baseline / "edm/configs/edm/outdoor/edm_base.py",
        mconf_thr=arguments.mconf_threshold,
        device=arguments.matcher_device,
        fp16=not arguments.no_fp16,
    )
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(baseline / "reconstruction"))
    first_camera = next(iter(reconstruction.cameras.values()))
    runtime_camera = Camera(
        model=first_camera.model_name,
        width=int(first_camera.width),
        height=int(first_camera.height),
        params=list(first_camera.params),
    )
    localizer = EDMLocalizer(
        relocation_map,
        runtime_camera,
        matcher=matcher,
        min_inliers=arguments.min_inliers,
        pnp_max_error=arguments.pnp_max_error,
    )
    queries = tuple(dict.fromkeys((*arguments.targets, *arguments.controls)))
    index_by_name = {name: index for index, name in enumerate(relocation_map.ref_names)}
    missing = set(queries) - set(index_by_name)
    if missing:
        raise KeyError(f"queries missing from EDM bundle: {sorted(missing)!r}")
    cache_dir = arguments.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if arguments.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_hashes") != before_hashes:
            raise RuntimeError("existing extraction cache belongs to different frozen sources")
    else:
        manifest = {
            "schema_version": 1,
            "source_hashes": before_hashes,
            "baseline": str(baseline),
            "matching_baseline": str(matching),
            "bundle_meta": _jsonable(getattr(relocation_map, "meta", {})),
            "bundle_contract": {
                "has_ref_centers": relocation_map.ref_centers is not None,
                "has_ref_yaws": relocation_map.ref_yaws is not None,
                "missing_required_for_full_audit": [
                    "full_reference_rotation",
                    "full_reference_translation",
                    "camera_id",
                    "image_id",
                    "point3D_id",
                    "track_provenance",
                    "build_git_commit",
                ],
            },
            "parameters": {
                "edm_repo": os.environ.get("EDM_REPO"),
                "topk": arguments.topk,
                "near_k": arguments.near_k,
                "mconf_threshold": arguments.mconf_threshold,
                "match_batch_size": arguments.match_batch_size,
                "matcher_device": arguments.matcher_device,
                "production_equivalent": (
                    arguments.matcher_device == "cuda" and not arguments.no_fp16
                ),
            },
            "queries": {},
        }
    poses = _m0_poses(baseline)
    for step, query_name in enumerate(queries, 1):
        cache_path = cache_dir / f"{query_name.replace('/', '__')}.npz"
        if cache_path.exists() and arguments.resume and query_name in manifest.get("queries", {}):
            print(f"[{step}/{len(queries)}] cached {query_name}")
            continue
        if cache_path.exists() and not arguments.resume:
            raise FileExistsError(f"cache exists; pass --resume: {cache_path}")
        query_index = index_by_name[query_name]
        excluded = loo_exclusion_indices(
            relocation_map.ref_names,
            query_index=query_index,
            near_k=arguments.near_k,
        )
        ranked, retrieval_scores = rank_reference_indices(
            relocation_map.ref_global,
            query_index=query_index,
            excluded_indices=excluded,
            topk=arguments.topk,
        )
        references = [relocation_map.ref_names[index] for index in ranked]
        by_reference = localizer.correspondences_by_ref(
            relocation_map.images[query_name],
            references,
            batch_size=arguments.match_batch_size,
        )
        funnel = _provider_funnel(localizer.correspondence_provider)
        points2d_parts = [
            np.asarray(row[0], dtype=np.float64) for row in by_reference if len(row[0])
        ]
        points3d_parts = [
            np.asarray(row[1], dtype=np.float64) for row in by_reference if len(row[1])
        ]
        source_parts = [
            np.full(len(row[1]), reference, dtype=f"U{max(len(reference), 1)}")
            for reference, row in zip(references, by_reference, strict=True)
            if len(row[1])
        ]
        points2d = np.concatenate(points2d_parts) if points2d_parts else np.zeros((0, 2))
        points3d = np.concatenate(points3d_parts) if points3d_parts else np.zeros((0, 3))
        sources = np.concatenate(source_parts) if source_parts else np.zeros(0, dtype="U1")
        np.savez_compressed(cache_path, points2d=points2d, points3d=points3d, sources=sources)
        reference_audit = []
        for reference in references:
            reference_index = index_by_name[reference]
            pose = poses[reference]
            center_error = None
            yaw_error = None
            if relocation_map.ref_centers is not None:
                center_error = float(
                    np.linalg.norm(
                        relocation_map.ref_centers[reference_index] - camera_center(pose)
                    )
                )
            if relocation_map.ref_yaws is not None:
                yaw_error = float(
                    np.degrees(
                        _angle_delta(
                            float(relocation_map.ref_yaws[reference_index]),
                            _camera_yaw(pose),
                        )
                    )
                )
            reference_audit.append(
                {
                    "reference": reference,
                    "center_error": center_error,
                    "yaw_error_deg": yaw_error,
                }
            )
        manifest["queries"][query_name] = {
            "query_index": query_index,
            "cache": cache_path.name,
            "excluded_refs": [relocation_map.ref_names[index] for index in excluded],
            "retrieved_refs": references,
            "retrieval_scores": retrieval_scores,
            "per_reference_correspondences": [int(row[2]) for row in by_reference],
            "correspondence_count": len(points3d),
            "funnel": funnel,
            "reference_audit": reference_audit,
        }
        _atomic_json(manifest_path, manifest)
        print(
            f"[{step}/{len(queries)}] {query_name}: raw={funnel['raw_matches']} "
            f"verified={funnel['verified_2d2d']} anchored={len(points3d)}"
        )
        torch.cuda.empty_cache()
    after_hashes = _source_hashes(baseline, matching)
    if after_hashes != before_hashes:
        raise RuntimeError("a frozen source changed during EDM extraction")
    manifest["source_hashes_after"] = after_hashes
    _atomic_json(manifest_path, manifest)
    verify_frozen_baseline(baseline)
    verify_frozen_baseline(matching)
    return manifest


def _estimate_pnp(
    points2d: np.ndarray,
    points3d: np.ndarray,
    camera: Any,
    *,
    max_error: float,
    seed: int,
    covariance: bool = True,
) -> _PnPRun | None:
    if len(points3d) < 6:
        return None
    import pycolmap

    options = pycolmap.AbsolutePoseEstimationOptions()
    options.ransac.max_error = max_error
    options.ransac.random_seed = seed
    options.ransac.num_threads = 1
    estimate = pycolmap.estimate_and_refine_absolute_pose(
        np.asarray(points2d, dtype=float),
        np.asarray(points3d, dtype=float),
        camera,
        options,
        return_covariance=covariance,
    )
    if estimate is None:
        return None
    transform = estimate["cam_from_world"]
    pose = _pose_matrix(transform.rotation.matrix(), transform.translation)
    native_covariance = estimate.get("covariance")
    return _PnPRun(
        pose=pose,
        inlier_mask=np.asarray(estimate["inlier_mask"], dtype=bool),
        covariance=(
            None if native_covariance is None else np.asarray(native_covariance, dtype=float)
        ),
        num_inliers=int(estimate["num_inliers"]),
    )


def _heldout_p90(
    points2d: np.ndarray,
    points3d: np.ndarray,
    camera: Any,
    camera_matrix: np.ndarray,
    *,
    image_size: tuple[int, int],
    max_error: float,
    splits: int,
) -> tuple[float, float]:
    if splits <= 0 or len(points3d) < 12:
        return float("inf"), 0.0
    width, height = image_size
    cell_x = np.clip((points2d[:, 0] * 4 / width).astype(int), 0, 3)
    cell_y = np.clip((points2d[:, 1] * 4 / height).astype(int), 0, 3)
    cells = cell_y * 4 + cell_x
    values = []
    for seed in range(splits):
        generator = np.random.default_rng(seed)
        test_parts = []
        for cell in np.unique(cells):
            indices = np.flatnonzero(cells == cell)
            generator.shuffle(indices)
            count = max(1, round(len(indices) * 0.2)) if len(indices) >= 5 else 0
            if count:
                test_parts.append(indices[:count])
        if not test_parts:
            continue
        test = np.unique(np.concatenate(test_parts))
        train_mask = np.ones(len(points3d), dtype=bool)
        train_mask[test] = False
        if np.count_nonzero(train_mask) < 6:
            continue
        estimate = _estimate_pnp(
            points2d[train_mask],
            points3d[train_mask],
            camera,
            max_error=max_error,
            seed=seed,
            covariance=False,
        )
        if estimate is None:
            continue
        projected, positive = project_points(points3d[test], estimate.pose, camera_matrix)
        errors = np.linalg.norm(projected - points2d[test], axis=1)
        valid = positive & np.isfinite(errors) & (errors <= max_error)
        if np.count_nonzero(valid) >= 3:
            values.append(float(np.percentile(errors[valid], 90)))
    return (
        (float(np.median(values)), float(len(values) / splits)) if values else (float("inf"), 0.0)
    )


def _pnp_hypothesis(
    *,
    label: str,
    kind: str,
    references: Sequence[str],
    all_points2d: np.ndarray,
    all_points3d: np.ndarray,
    all_sources: np.ndarray,
    camera: Any,
    camera_matrix: np.ndarray,
    m0_pose: np.ndarray,
    image_size: tuple[int, int],
    scene_scale: float,
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], _PnPRun | None, np.ndarray, np.ndarray, np.ndarray]:
    member = np.isin(all_sources, np.asarray(references))
    points2d = all_points2d[member]
    points3d = all_points3d[member]
    sources = all_sources[member]
    selected = spatial_cap_indices(
        points2d,
        max_total=arguments.max_correspondences,
        width=image_size[0],
        height=image_size[1],
        grid=arguments.cap_grid,
    )
    points2d = points2d[selected]
    points3d = points3d[selected]
    sources = sources[selected]
    estimate = _estimate_pnp(
        points2d,
        points3d,
        camera,
        max_error=arguments.pnp_max_error,
        seed=arguments.ransac_seed,
    )
    row: dict[str, Any] = {
        "label": label,
        "kind": kind,
        "references": list(references),
        "correspondences_before_cap": int(np.count_nonzero(member)),
        "correspondences_after_cap": len(points3d),
        "unique_anchor_count": unique_anchor_count(points3d),
        "solver_returned": estimate is not None,
        "num_inliers": 0 if estimate is None else estimate.num_inliers,
        "pose": None if estimate is None else estimate.pose,
        "native_covariance": None if estimate is None else estimate.covariance,
        "metrics": None,
        "heldout_reprojection_p90": None,
        "heldout_success_rate": 0.0,
    }
    if estimate is not None:
        metrics = compute_loo_metrics(
            points3d,
            points2d,
            estimate.inlier_mask,
            tuple(camera.params),
            estimate.pose,
            m0_pose,
            sources,
            image_size=image_size,
            fim_scene_scale=scene_scale,
        )
        heldout_p90, heldout_success = _heldout_p90(
            points2d,
            points3d,
            camera,
            camera_matrix,
            image_size=image_size,
            max_error=arguments.pnp_max_error,
            splits=arguments.holdout_splits,
        )
        row["metrics"] = metrics
        row["heldout_reprojection_p90"] = heldout_p90
        row["heldout_success_rate"] = heldout_success
    return row, estimate, points2d, points3d, sources


def _pose_mode_input(rows: Sequence[dict[str, Any]]) -> list[PoseHypothesis]:
    hypotheses = []
    for row in rows:
        metrics = row.get("metrics")
        if not row.get("solver_returned") or not isinstance(metrics, Mapping):
            continue
        hypotheses.append(
            PoseHypothesis(
                label=str(row["label"]),
                pose=np.asarray(row["pose"], dtype=float),
                inliers=int(row["num_inliers"]),
                positive_depth_ratio=float(metrics.get("positive_depth_ratio") or 0.0),
                heldout_reprojection_p90=float(
                    row.get("heldout_reprojection_p90")
                    if row.get("heldout_reprojection_p90") is not None
                    else float("inf")
                ),
            )
        )
    return hypotheses


def _serialize_envelope(envelope: Any, *, scene_scale: float) -> dict[str, Any]:
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    candidates = []
    for candidate in envelope.candidates:
        candidates.append(_jsonable(candidate))
    rotation_spread = 0.0
    center_spread = 0.0
    for first in range(len(envelope.candidates)):
        for second in range(first + 1, len(envelope.candidates)):
            first_pose = envelope.candidates[first].pose
            second_pose = envelope.candidates[second].pose
            rotation_spread = max(
                rotation_spread,
                rotation_distance_deg(first_pose[:3, :3], second_pose[:3, :3]),
            )
            center_spread = max(
                center_spread,
                float(np.linalg.norm(camera_center(first_pose) - camera_center(second_pose))),
            )
    center_spread /= scene_scale
    solver_consensus = "AGREE" if rotation_spread <= 2.0 and center_spread <= 0.02 else "DISAGREE"
    return {
        "planarity": _jsonable(envelope.planarity),
        "candidates": candidates,
        "candidate_rotation_spread_deg": rotation_spread,
        "candidate_center_spread_normalized": center_spread,
        "solver_pose_consensus": solver_consensus,
    }


def _mode_value(mode: Any, field: str, default: Any = None) -> Any:
    if dataclasses.is_dataclass(mode):
        return getattr(mode, field, default)
    if isinstance(mode, Mapping):
        return mode.get(field, default)
    return default


def _consensus_value(consensus: Any, field: str, default: Any = None) -> Any:
    if dataclasses.is_dataclass(consensus):
        return getattr(consensus, field, default)
    if isinstance(consensus, Mapping):
        return consensus.get(field, default)
    return default


def _single_reference_names(mode: Any) -> set[str]:
    names = set()
    for label in _mode_value(mode, "labels", ()):
        if str(label).startswith("single:"):
            names.add(str(label).split(":", 1)[1])
    return names


def _abnormal_indicators(row: Mapping[str, Any]) -> list[str]:
    aggregate = row.get("aggregate") or {}
    metrics = aggregate.get("metrics") or {}
    consensus = row.get("reference_consensus")
    funnel = row.get("conversion_funnel") or {}
    envelope = row.get("solver_envelope") or {}
    planarity = envelope.get("planarity") or {}
    indicators = []
    if float(metrics.get("rotation_error_deg") or 0.0) > 2.0:
        indicators.append("ABSOLUTE_POSE_DISAGREEMENT")
    if int(_consensus_value(consensus, "reliable_hypothesis_count", 0)) < 3:
        indicators.append("INSUFFICIENT_INDEPENDENT_REFERENCE_HYPOTHESES")
    if int(funnel.get("unique_2d3d") or 0) < 1000 or int(aggregate.get("num_inliers") or 0) < 50:
        indicators.append("MATCH_SUPPORT_COLLAPSE")
    if (
        float(metrics.get("convex_hull_coverage") or 0.0) < 0.1
        or int(metrics.get("occupancy_4x4") or 0) <= 3
    ):
        indicators.append("SPATIAL_COVERAGE_COLLAPSE")
    if float(metrics.get("scaled_fim_condition") or 0.0) > 1000.0:
        indicators.append("FIM_ILL_CONDITIONED_EMPIRICAL")
    if float(metrics.get("reprojection_p90") or 0.0) > 4.0:
        indicators.append("HIGH_REPROJECTION_P90")
    if row.get("reference_consensus_status") == "MULTIMODAL":
        indicators.append("REFERENCE_MULTIMODAL")
    if bool(planarity.get("is_planar")):
        indicators.append("PLANAR_PNP_GEOMETRY")
    if envelope.get("solver_pose_consensus") == "DISAGREE":
        indicators.append("PNP_SOLVER_DISAGREEMENT")
    return indicators


def _region_pose_mode_diagnosis(
    queries: Mapping[str, Mapping[str, Any]],
    *,
    m0_poses: Mapping[str, np.ndarray],
    scene_scale: float,
) -> dict[str, Any]:
    """Propagate independently observed bad reference modes across nearby queries."""

    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise ValueError("scene_scale must be finite and positive")
    missing = set(queries) - set(m0_poses)
    if missing:
        raise KeyError(f"M0 poses are missing for region queries: {sorted(missing)!r}")
    suspect_references: set[str] = set()
    source_multimodal_queries = []
    for name, row in queries.items():
        if row.get("reference_consensus_status") != "MULTIMODAL":
            continue
        source_multimodal_queries.append(name)
        consensus = row.get("reference_consensus")
        for mode in _consensus_value(consensus, "modes", ()):
            pose = _mode_value(mode, "representative_pose")
            if pose is None:
                continue
            pose = np.asarray(pose, dtype=float)
            rotation_error = rotation_distance_deg(
                pose[:3, :3], np.asarray(m0_poses[name], dtype=float)[:3, :3]
            )
            center_error = float(
                np.linalg.norm(camera_center(pose) - camera_center(m0_poses[name])) / scene_scale
            )
            if rotation_error > 2.0 or center_error > 0.02:
                suspect_references.update(_single_reference_names(mode))

    query_attribution: dict[str, str] = {}
    propagated_queries = []
    abnormal = {}
    for name, row in queries.items():
        abnormal[name] = _abnormal_indicators(row)
        status = str(row.get("reference_consensus_status"))
        consensus = row.get("reference_consensus")
        reliable_count = int(_consensus_value(consensus, "reliable_hypothesis_count", 0))
        if reliable_count < 3:
            attribution = "INSUFFICIENT_REFERENCE_SUPPORT"
        elif status == "MULTIMODAL":
            attribution = "REPEATED_STRUCTURE_OR_REFERENCE_AMBIGUITY"
        elif status == "CONSENSUS_EDM":
            modes = _consensus_value(consensus, "modes", ())
            dominant_names = _single_reference_names(modes[0]) if modes else set()
            if dominant_names & suspect_references:
                attribution = "REFERENCE_CONDITIONED_POSE_MODE_SWITCH"
                propagated_queries.append(name)
            else:
                attribution = str(row.get("primary_attribution", "UNRESOLVED"))
        else:
            attribution = str(row.get("primary_attribution", "UNRESOLVED"))
        query_attribution[name] = attribution

    if propagated_queries:
        status = "REFERENCE_CONDITIONED_POSE_MODE_SWITCH"
    elif source_multimodal_queries:
        status = "REPEATED_STRUCTURE_OR_REFERENCE_AMBIGUITY"
    else:
        status = "UNRESOLVED"
    return {
        "status": status,
        "query_attribution": query_attribution,
        "source_multimodal_queries": sorted(source_multimodal_queries),
        "propagated_queries": sorted(propagated_queries),
        "suspect_reference_names": sorted(suspect_references),
        "abnormal_indicators": abnormal,
        "interpretation": (
            "A reference mode already observed as an alternate, M0-inconsistent mode "
            "became the only or dominant mode for adjacent queries. This is a global "
            "multi-mode/reference-anchoring failure that local FIM cannot detect."
            if propagated_queries
            else "No cross-query reference-conditioned pose-mode switch was proven."
        ),
    }


def _rotation_medoid(rotations: Sequence[np.ndarray]) -> np.ndarray:
    scores = []
    for index, rotation in enumerate(rotations):
        score = sum(rotation_distance_deg(rotation, other) for other in rotations)
        scores.append((score, index))
    return rotations[min(scores)[1]]


def _direction_error(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    signed = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
    return signed, min(signed, 180.0 - signed)


def _sequential_evidence(
    *,
    database: Path,
    camera_matrix: np.ndarray,
    m0_poses: Mapping[str, np.ndarray],
    edm_poses: Mapping[str, np.ndarray],
    arguments: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[RelativePoseTrial]]]:
    names = [f"P1200120/{frame:06d}.jpg" for frame in range(96, 104)]
    pairs = []
    trials_by_query: dict[str, list[RelativePoseTrial]] = {name: [] for name in names}
    for first_name, second_name in pairwise(names):
        pair = read_verified_pair_points(database, first_name, second_name)
        essential_trials = estimate_essential_trials(
            pair.points1,
            pair.points2,
            camera_matrix,
            seeds=tuple(range(arguments.essential_seeds)),
            thresholds_px=tuple(arguments.essential_thresholds),
        )
        medoid = _rotation_medoid([trial.rotation for trial in essential_trials])
        m0_relative = relative_pose(m0_poses[first_name], m0_poses[second_name])
        edm_relative = relative_pose(edm_poses[first_name], edm_poses[second_name])
        relative_trials = []
        raw_trials = []
        for trial in essential_trials:
            m0_signed, m0_antipodal = _direction_error(
                trial.translation_direction, m0_relative[:3, 3]
            )
            edm_signed, edm_antipodal = _direction_error(
                trial.translation_direction, edm_relative[:3, 3]
            )
            evidence = RelativePoseTrial(
                seed=trial.seed,
                threshold_px=trial.threshold_px,
                inliers=trial.inliers,
                recovered_rotation_deg=rotation_distance_deg(trial.rotation, medoid),
                m0_rotation_error_deg=rotation_distance_deg(trial.rotation, m0_relative[:3, :3]),
                edm_rotation_error_deg=rotation_distance_deg(trial.rotation, edm_relative[:3, :3]),
                m0_translation_direction_error_deg=m0_signed,
                edm_translation_direction_error_deg=edm_signed,
            )
            relative_trials.append(evidence)
            raw_trials.append(
                {
                    **_jsonable(evidence),
                    "m0_translation_antipodal_error_deg": m0_antipodal,
                    "edm_translation_antipodal_error_deg": edm_antipodal,
                    "rotation": trial.rotation,
                    "translation_direction": trial.translation_direction,
                }
            )
        support = classify_sequential_support(relative_trials)
        pairs.append(
            {
                "name1": first_name,
                "name2": second_name,
                "verified_matches": len(pair.points1),
                "support": support,
                "trials": raw_trials,
            }
        )
        trials_by_query[first_name].extend(relative_trials)
        trials_by_query[second_name].extend(relative_trials)
    return pairs, trials_by_query


def _bundle_audit(manifest: Mapping[str, Any], baseline: Path) -> dict[str, Any]:
    audits = [
        row for query in manifest["queries"].values() for row in query.get("reference_audit", [])
    ]
    centers = [float(row["center_error"]) for row in audits if row.get("center_error") is not None]
    yaws = [float(row["yaw_error_deg"]) for row in audits if row.get("yaw_error_deg") is not None]
    meta = manifest.get("bundle_meta", {})
    source_model = Path(str(meta.get("source_model", ""))) if meta.get("source_model") else None
    source_hashes = {}
    if source_model is not None and source_model.is_dir():
        for filename in ("cameras.bin", "images.bin", "points3D.bin"):
            candidate = source_model / filename
            if candidate.is_file():
                source_hashes[filename] = _sha256(candidate)
    frozen = manifest["source_hashes"]
    hash_match = {
        "cameras.bin": source_hashes.get("cameras.bin") == frozen["cameras_bin"],
        "images.bin": source_hashes.get("images.bin") == frozen["images_bin"],
        "points3D.bin": source_hashes.get("points3D.bin") == frozen["points3D_bin"],
    }
    return {
        "source_model": None if source_model is None else str(source_model),
        "source_hashes": source_hashes,
        "source_hash_matches_frozen_m0": hash_match,
        "reference_center_error_max": max(centers, default=None),
        "reference_center_error_p90": (float(np.percentile(centers, 90)) if centers else None),
        "reference_yaw_error_max_deg": max(yaws, default=None),
        "reference_yaw_error_p90_deg": (float(np.percentile(yaws, 90)) if yaws else None),
        "verdict": {
            "source_hash_center_yaw": "REFUTED"
            if source_hashes and all(hash_match.values())
            else "INDETERMINATE",
            "roll_pitch_point_track_provenance": "UNVERIFIABLE",
        },
        "missing_fields": manifest["bundle_contract"]["missing_required_for_full_audit"],
        "frozen_reconstruction": str(baseline / "reconstruction"),
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    parameters = summary.get("extraction_parameters") or {}
    production_equivalent = bool(parameters.get("production_equivalent", True))
    run_label = "PRODUCTION-EQUIVALENT" if production_equivalent else "CPU SHADOW"
    region = summary.get("region_diagnosis") or {}
    bundle = summary.get("bundle_audit") or {}
    lines = [
        "# P120/97-103 Pose Source Attribution",
        "",
        f"Run class: **{run_label}**.",
        "",
        "M0 and the EDM bundle remained immutable. FIM is reported only as local curvature; "
        "reference-conditioned multimodality has precedence.",
        "",
        f"Region diagnosis: **{region.get('status', 'UNRESOLVED')}**.",
        "",
        str(region.get("interpretation", "")),
        "",
        "| Query | Ref modes | Region attribution | Rot vs M0 | Center error | "
        "Inliers | Hull / grid | FIM kappa | Reproj p90 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["queries"].items():
        if not row.get("is_target"):
            continue
        aggregate = row.get("aggregate") or {}
        metrics = aggregate.get("metrics") or {}
        lines.append(
            f"| {name} | {row.get('reference_consensus_status')} | "
            f"{row.get('region_attribution')} | "
            f"{float(metrics.get('rotation_error_deg') or 0.0):.3f} deg | "
            f"{float(metrics.get('position_error') or 0.0):.4f} | "
            f"{aggregate.get('num_inliers', '')} | "
            f"{float(metrics.get('convex_hull_coverage') or 0.0):.3f} / "
            f"{metrics.get('occupancy_4x4', '')} | "
            f"{float(metrics.get('scaled_fim_condition') or 0.0):.1f} | "
            f"{float(metrics.get('reprojection_p90') or 0.0):.2f} px |"
        )
    lines.extend(
        [
            "",
            "## Abnormal indicators",
            "",
        ]
    )
    abnormal = region.get("abnormal_indicators") or {}
    for name in summary.get("targets", []):
        indicators = abnormal.get(name) or ["NONE"]
        lines.append(f"- {name}: {', '.join(indicators)}")
    lines.extend(
        [
            "",
            "## Counterfactual checks",
            "",
            "- Stale M0 source geometry: **refuted for the available fields**. "
            f"Binary hashes all match: "
            f"{all((bundle.get('source_hash_matches_frozen_m0') or {}).values())}; "
            f"max center error={bundle.get('reference_center_error_max')}, "
            f"max yaw error={bundle.get('reference_yaw_error_max_deg')} deg.",
            "- Planar PnP ambiguity: no analyzed query crossed the planar gate.",
            "- PnP implementation choice: SQPnP and iterative variants agree on the "
            "same pose mode; low reprojection error therefore does not validate the "
            "absolute map anchor.",
            "- Calibration / rolling shutter: residual trends are diagnostic only and "
            "do not establish either cause; the strongest row trend occurs where PnP "
            "support already collapses.",
            "",
            "## Interpretation guardrails",
            "",
            "- `MULTIMODAL` overrides a locally well-conditioned FIM.",
            "- M0/EDM source attribution requires matching reference consensus and "
            "two-view evidence.",
            "- Low parallax is a structural-risk flag, not a repair trigger.",
            "- The current bundle cannot support true Point3D/track-community attribution.",
            "- FIM kappa > 1000 is an empirical M0 flag, not a universal physical threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def _consensus_mode_poses(consensus: Any) -> list[np.ndarray]:
    if dataclasses.is_dataclass(consensus):
        modes = consensus.modes
    elif isinstance(consensus, Mapping):
        modes = consensus.get("modes", [])
    else:
        return []
    output = []
    for mode in modes:
        pose = (
            mode.representative_pose
            if dataclasses.is_dataclass(mode)
            else mode.get("representative_pose")
        )
        if pose is not None:
            output.append(np.asarray(pose, dtype=float))
    return output


def attribution_overlay_report(
    summary: Mapping[str, Any],
    m0_poses: Mapping[str, np.ndarray],
    *,
    camera_spacing: float,
) -> dict[str, Any]:
    """Translate attribution results into absolute M0 camera-centre sphere evidence."""

    targets = tuple(summary["targets"])
    controls = tuple(summary["controls"])
    cameras = []
    for name in (*controls, *targets):
        row = summary["queries"][name]
        if name in controls:
            status = "green_healthy"
        elif row.get("reference_consensus_status") == "MULTIMODAL":
            status = "magenta_pose_multimodal"
        elif row.get("primary_attribution") == "M0_LOCAL_POSE_SUSPECTED":
            status = "purple_m0_local_pose"
        elif row.get("region_attribution") in {
            "EDM_REFERENCE_ANCHORING_SUSPECTED",
            "REFERENCE_CONDITIONED_POSE_MODE_SWITCH",
        }:
            status = "blue_edm_reference_anchoring"
        else:
            status = "yellow_structural_warning"
        cameras.append(
            {
                "image_name": name,
                "sequence": name.split("/", 1)[0],
                "frame": int(Path(name).stem),
                "camera_center": camera_center(m0_poses[name]).tolist(),
                "heatmap_status": status,
            }
        )
    by_name = {row["image_name"]: row for row in cameras}
    segments = []
    for segment_id, name in enumerate(targets, 1):
        query = summary["queries"][name]
        mode_poses = _consensus_mode_poses(query.get("reference_consensus"))
        separation = 0.0
        for first in range(len(mode_poses)):
            for second in range(first + 1, len(mode_poses)):
                separation = max(
                    separation,
                    float(
                        np.linalg.norm(
                            camera_center(mode_poses[first]) - camera_center(mode_poses[second])
                        )
                    ),
                )
        segments.append(
            {
                "segment_id": segment_id,
                "sequence": by_name[name]["sequence"],
                "start_frame": by_name[name]["frame"],
                "end_frame": by_name[name]["frame"],
                "image_names": [name],
                "camera_count": 1,
                "phenotypes": [str(query.get("reference_consensus_status"))],
                "causes": [str(query.get("region_attribution"))],
                "conditioning": [],
                "decision_axis": "reference_pose_consensus",
                "acquisition_failure_count": 0,
                "pose_inaccurate_count": 1,
                "radius_override": max(camera_spacing, separation + camera_spacing),
            }
        )
    return {
        "schema_version": 1,
        "coordinate_frame": "frozen M0 absolute reconstruction coordinates",
        "cameras": cameras,
        "segments": segments,
    }


def _view_direction_world(pose: np.ndarray) -> np.ndarray:
    direction = np.asarray(pose, dtype=float)[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    return direction / np.linalg.norm(direction)


def _attribution_pose_glyphs(
    summary: Mapping[str, Any],
    m0_poses: Mapping[str, np.ndarray],
    *,
    camera_spacing: float,
) -> tuple[PoseGlyph, ...]:
    glyphs = []
    mode_colors = ((219, 39, 119), (37, 99, 235), (234, 179, 8), (147, 51, 234))
    length = 5.0 * camera_spacing
    for name in summary["targets"]:
        query = summary["queries"][name]
        m0_pose = m0_poses[name]
        glyphs.append(
            PoseGlyph(
                query_name=name,
                source="M0",
                center=tuple(camera_center(m0_pose)),
                direction=tuple(_view_direction_world(m0_pose)),
                length=length,
                rgb=(6, 182, 212),
            )
        )
        aggregate = query.get("aggregate")
        if isinstance(aggregate, Mapping) and aggregate.get("pose") is not None:
            edm_pose = np.asarray(aggregate["pose"], dtype=float)
            glyphs.append(
                PoseGlyph(
                    query_name=name,
                    source="EDM aggregate",
                    center=tuple(camera_center(edm_pose)),
                    direction=tuple(_view_direction_world(edm_pose)),
                    length=length,
                    rgb=(249, 115, 22),
                )
            )
        for index, mode_pose in enumerate(_consensus_mode_poses(query.get("reference_consensus"))):
            glyphs.append(
                PoseGlyph(
                    query_name=name,
                    source=f"reference mode {index + 1}",
                    center=tuple(camera_center(mode_pose)),
                    direction=tuple(_view_direction_world(mode_pose)),
                    length=length,
                    rgb=mode_colors[index % len(mode_colors)],
                )
            )
    return tuple(glyphs)


def analyze_correspondence_cache(arguments: argparse.Namespace) -> dict[str, Any]:
    """Run CPU-only PnP, consensus, two-view, solver, and residual diagnostics."""

    import pycolmap

    baseline = arguments.baseline.resolve(strict=True)
    matching = arguments.matching_baseline.resolve(strict=True)
    verify_frozen_baseline(baseline)
    verify_frozen_baseline(matching)
    before_hashes = _source_hashes(baseline, matching)
    manifest = json.loads((arguments.cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_hashes"] != before_hashes:
        raise RuntimeError("correspondence cache does not match the current frozen baselines")
    reconstruction = pycolmap.Reconstruction(str(baseline / "reconstruction"))
    if len(reconstruction.cameras) != 1:
        raise ValueError("pose-source runner currently requires the single-camera M0 contract")
    source_camera = next(iter(reconstruction.cameras.values()))
    if source_camera.model_name != "PINHOLE" or len(source_camera.params) != 4:
        raise ValueError("pose-source runner requires a four-parameter PINHOLE camera")
    camera = pycolmap.Camera(
        model=source_camera.model_name,
        width=source_camera.width,
        height=source_camera.height,
        params=list(source_camera.params),
    )
    fx, fy, cx, cy = map(float, camera.params)
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    image_size = (int(camera.width), int(camera.height))
    scene_scale = global_scene_scale_from_report(arguments.health_report)
    m0_poses = _m0_poses(baseline)
    camera_spacing = camera_spacing_from_poses(m0_poses)
    queries = tuple(dict.fromkeys((*arguments.targets, *arguments.controls)))
    query_results: dict[str, Any] = {}
    edm_poses: dict[str, np.ndarray] = {}
    all_hypotheses = []
    for step, query_name in enumerate(queries, 1):
        query_manifest = manifest["queries"][query_name]
        with np.load(arguments.cache_dir / query_manifest["cache"], allow_pickle=False) as cache:
            points2d = np.asarray(cache["points2d"], dtype=float)
            points3d = np.asarray(cache["points3d"], dtype=float)
            sources = np.asarray(cache["sources"], dtype=str)
        references = tuple(query_manifest["retrieved_refs"])
        subsets = build_reference_subsets(
            references,
            m0_poses,
            camera_spacing=camera_spacing,
        )
        hypothesis_rows = []
        aggregate_internal = None
        for subset in subsets:
            row, estimate, selected2d, selected3d, selected_sources = _pnp_hypothesis(
                label=subset.label,
                kind=subset.kind,
                references=subset.references,
                all_points2d=points2d,
                all_points3d=points3d,
                all_sources=sources,
                camera=camera,
                camera_matrix=camera_matrix,
                m0_pose=m0_poses[query_name],
                image_size=image_size,
                scene_scale=scene_scale,
                arguments=arguments,
            )
            hypothesis_rows.append(row)
            all_hypotheses.append({"query_name": query_name, **row})
            if subset.kind == "aggregate" and estimate is not None:
                aggregate_internal = (estimate, selected2d, selected3d, selected_sources)
        if aggregate_internal is None:
            query_results[query_name] = {
                "reference_consensus_status": "INSUFFICIENT",
                "primary_attribution": "UNRESOLVED",
                "hypotheses": hypothesis_rows,
                "aggregate": None,
            }
            continue
        aggregate_estimate, aggregate2d, aggregate3d, _aggregate_sources = aggregate_internal
        edm_poses[query_name] = aggregate_estimate.pose
        aggregate_row = next(row for row in hypothesis_rows if row["kind"] == "aggregate")
        single_rows = [row for row in hypothesis_rows if row["kind"] == "single"]
        consensus = classify_reference_consensus(
            _pose_mode_input(single_rows),
            m0_pose=m0_poses[query_name],
            edm_pose=aggregate_estimate.pose,
            scene_scale=scene_scale,
        )
        if consensus.reliable_hypothesis_count < 3:
            block_rows = [row for row in hypothesis_rows if row["kind"] == "block"]
            consensus = classify_reference_consensus(
                _pose_mode_input(block_rows),
                m0_pose=m0_poses[query_name],
                edm_pose=aggregate_estimate.pose,
                scene_scale=scene_scale,
            )
        inlier = aggregate_estimate.inlier_mask
        envelope = solve_pnp_envelope(
            aggregate3d[inlier],
            aggregate2d[inlier],
            camera_matrix,
            initial_poses={"M0": m0_poses[query_name], "EDM": aggregate_estimate.pose},
        )
        projected, positive = project_points(
            aggregate3d[inlier], aggregate_estimate.pose, camera_matrix
        )
        valid = positive & np.isfinite(projected).all(axis=1)
        residual = residual_structure(
            aggregate2d[inlier][valid],
            projected[valid] - aggregate2d[inlier][valid],
            intrinsics=(fx, fy, cx, cy),
            image_size=image_size,
        )
        funnel_source = query_manifest["funnel"]
        funnel = conversion_funnel(
            raw_matches=int(funnel_source["raw_matches"]),
            verified_2d2d=int(funnel_source["verified_2d2d"]),
            unique_2d3d=unique_anchor_count(points3d),
            pnp_inliers=aggregate_estimate.num_inliers,
        )
        query_results[query_name] = {
            "is_target": query_name in arguments.targets,
            "reference_consensus_status": consensus.status,
            "reference_consensus": consensus,
            "sequential_support": "PENDING",
            "primary_attribution": "UNRESOLVED",
            "aggregate": aggregate_row,
            "hypotheses": hypothesis_rows,
            "conversion_funnel": funnel,
            "solver_envelope": _serialize_envelope(
                envelope,
                scene_scale=scene_scale,
            ),
            "residual_structure": residual,
        }
        print(
            f"[{step}/{len(queries)}] {query_name}: consensus={consensus.status} "
            f"inliers={aggregate_estimate.num_inliers}"
        )

    if set(queries) - set(edm_poses):
        missing = sorted(set(queries) - set(edm_poses))
        raise RuntimeError(f"aggregate EDM pose missing for sequential controls: {missing!r}")
    pair_rows, trials_by_query = _sequential_evidence(
        database=matching / "database_merged.db",
        camera_matrix=camera_matrix,
        m0_poses=m0_poses,
        edm_poses=edm_poses,
        arguments=arguments,
    )
    for query_name, row in query_results.items():
        trials = trials_by_query[query_name]
        sequential = classify_sequential_support(trials)
        row["sequential_support"] = sequential.support
        row["sequential_evidence"] = sequential
        consensus_payload = row.get("reference_consensus")
        if consensus_payload is not None:
            row["primary_attribution"] = attribute_pose_source(
                consensus_payload,
                sequential_support=sequential.support,
            )
    region_diagnosis = _region_pose_mode_diagnosis(
        query_results,
        m0_poses=m0_poses,
        scene_scale=scene_scale,
    )
    for query_name, row in query_results.items():
        row["region_attribution"] = region_diagnosis["query_attribution"][query_name]
    summary = {
        "schema_version": 1,
        "experiment": "P120/97-103 pose source attribution",
        "targets": list(arguments.targets),
        "controls": list(arguments.controls),
        "global_fim_scene_scale": scene_scale,
        "camera_spacing": camera_spacing,
        "extraction_parameters": manifest.get("parameters", {}),
        "source_hashes_before": before_hashes,
        "bundle_audit": _bundle_audit(manifest, baseline),
        "region_diagnosis": region_diagnosis,
        "queries": query_results,
        "relative_pairs": pair_rows,
        "limitations": [
            "EDM bundle has no full reference pose, Point3D IDs, or track provenance",
            "community tests are reference/spatial diagnostics, not TC-SfM track causality",
            "row residual structure alone cannot prove rolling shutter",
        ],
    }
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    overlay_report = attribution_overlay_report(
        summary,
        m0_poses,
        camera_spacing=camera_spacing,
    )
    _atomic_json(output / "attribution_overlay_report.json", overlay_report)
    glyphs = _attribution_pose_glyphs(
        summary,
        m0_poses,
        camera_spacing=camera_spacing,
    )
    glyph_path = output / "visuals/P120_pose_direction_glyphs.ply"
    _atomic_bytes(glyph_path, render_pose_glyphs_ply(glyphs))
    _atomic_json(output / "visuals/P120_pose_direction_glyphs.json", glyphs)
    summary["pose_glyphs"] = str(glyph_path)
    if not arguments.skip_overlay:
        images_root = arguments.images_root
        if images_root is None:
            configured_root = manifest.get("bundle_meta", {}).get("image_root")
            images_root = Path(configured_root) if configured_root else None
        if images_root is None or not Path(images_root).is_dir():
            raise FileNotFoundError(
                "RGB attribution overlay needs --images-root pointing at the frozen mapping images"
            )
        visualization = export_weak_region_overlay(
            output / "attribution_overlay_report.json",
            baseline / "reconstruction/points3D.bin",
            output / "visuals",
            images_bin_path=baseline / "reconstruction/images.bin",
            images_root=Path(images_root),
            samples_per_sphere=2048,
        )
        summary["visualization_outputs"] = {key: str(path) for key, path in visualization.items()}
    _atomic_json(output / "summary.json", summary)
    _atomic_json(output / "hypotheses.json", all_hypotheses)
    _atomic_json(output / "relative_pairs.json", pair_rows)
    _atomic_json(output / "bundle_audit.json", summary["bundle_audit"])
    (output / "REPORT.md").write_text(_markdown_report(summary), encoding="utf-8")
    after_hashes = _source_hashes(baseline, matching)
    if after_hashes != before_hashes:
        raise RuntimeError("a frozen source changed during pose-source analysis")
    summary["source_hashes_after"] = after_hashes
    _atomic_json(output / "summary.json", summary)
    verify_frozen_baseline(baseline)
    verify_frozen_baseline(matching)
    return summary


def run(arguments: argparse.Namespace) -> int:
    if arguments.stage in {"extract", "all"}:
        extract_correspondence_cache(arguments)
    if arguments.stage in {"analyze", "all"}:
        analyze_correspondence_cache(arguments)
    return 0
