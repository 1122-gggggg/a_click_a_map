"""Optional, immutable map-refinement backends for COLMAP reconstructions.

PixSfM invocation follows the official existing-model bundle-adjuster interface:
https://github.com/cvg/pixel-perfect-sfm#bundle-adjustment

Dense-SfM invocation follows its public refinement-only/full-pipeline entrypoints:
https://github.com/IceTea-CV/DenseSfM-Refine#usage
The public Dense-SfM release does not include the paper's Gaussian-Splatting
track-extension stage, so receipts deliberately identify it as a partial public
implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import ExclusiveResourceLease, probe_python_runtime, require_compute_capability
from .robust_filter import RobustFilterConfig, reconstruction_metrics, robust_filter_model

REFINEMENT_BACKENDS = ("pixsfm", "densesfm-refine", "densesfm-full")
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class RefinementRequest:
    backend: str
    input_model: Path
    images: Path
    intrinsics: Path
    pairs: Path | None
    run_dir: Path
    cache_dir: Path
    runtime_python: Path
    runtime_root: Path | None = None
    continuation_receipt: Path | None = None
    timeout_seconds: int = 86_400
    max_cache_gb: int = 500
    pixsfm_patch_size: int = 4
    robust_filter: RobustFilterConfig = RobustFilterConfig()

    def with_runtime_root(self, value: Path) -> "RefinementRequest":
        return replace(self, runtime_root=value)

    def with_run_dir(self, value: Path) -> "RefinementRequest":
        return replace(self, run_dir=value)

    def with_pairs(self, value: Path) -> "RefinementRequest":
        return replace(self, pairs=value)

    def with_pixsfm_patch_size(self, value: int) -> "RefinementRequest":
        return replace(self, pixsfm_patch_size=value)

    @property
    def raw_model(self) -> Path:
        return self.run_dir / "candidates" / self.backend / "raw_model"

    @property
    def robust_model(self) -> Path:
        return self.run_dir / "candidates" / self.backend / "robust_model"

    @property
    def dense_config(self) -> Path:
        return self.run_dir / "bootstrap" / "densesfm.yaml"

    @property
    def dense_staging(self) -> Path:
        return self.run_dir / "staging" / "densesfm"

    @property
    def dense_database(self) -> Path:
        return self.dense_staging / "database.db"

    @property
    def pixsfm_cache(self) -> Path:
        return self.cache_dir / "s2dnet_featuremaps_sparse.h5"

    def validate(self) -> None:
        if self.backend not in REFINEMENT_BACKENDS:
            raise ValueError(f"unknown refinement backend: {self.backend}")
        model = self.input_model.expanduser().resolve(strict=True)
        missing = [name for name in MODEL_FILES if not (model / name).is_file()]
        if missing:
            raise ValueError(f"input model is missing files: {missing}")
        if not self.images.expanduser().resolve(strict=True).is_dir():
            raise ValueError("images must be a directory")
        if not self.intrinsics.expanduser().resolve(strict=True).is_file():
            raise ValueError("intrinsics must be a file")
        runtime = self.runtime_python.expanduser().resolve(strict=True)
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise ValueError("runtime Python must be executable")
        if self.backend == "densesfm-full" and self.pairs is None:
            raise ValueError("Dense-SfM full refinement requires a frozen pair manifest")
        if self.pairs is not None and not self.pairs.expanduser().resolve(strict=True).is_file():
            raise ValueError("pair manifest must be a file")
        if self.backend.startswith("densesfm"):
            if self.runtime_root is None:
                raise ValueError("Dense-SfM requires an external runtime root")
            root = self.runtime_root.expanduser().resolve(strict=True)
            script = "run_refinement.py" if self.backend == "densesfm-refine" else "run_full.py"
            if not (root / script).is_file():
                raise ValueError(f"Dense-SfM runtime is missing {script}")
        if self.backend == "densesfm-full":
            if self.continuation_receipt is None:
                raise ValueError(
                    "Dense-SfM full requires a passing refinement continuation receipt"
                )
            continuation = self.continuation_receipt.expanduser().resolve(strict=True)
            payload = json.loads(continuation.read_text(encoding="utf-8"))
            if not bool((payload.get("geometry_gate") or {}).get("passes")):
                raise ValueError("Dense-SfM refinement continuation receipt did not pass its gate")
        run = self.run_dir.expanduser().absolute()
        try:
            run.resolve().relative_to(model)
        except ValueError:
            pass
        else:
            raise ValueError("refinement run must be outside the input model")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        if self.max_cache_gb <= 0:
            raise ValueError("cache budget must be positive")
        if not 2 <= self.pixsfm_patch_size <= 8:
            raise ValueError("PixSfM patch size must be between 2 and 8")

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "backend": self.backend,
            "input_model": _model_hashes(self.input_model),
            "images": str(self.images.expanduser().resolve()),
            "intrinsics_sha256": _sha256(self.intrinsics),
            "pairs_sha256": None if self.pairs is None else _sha256(self.pairs),
            "runtime_python": str(self.runtime_python.expanduser().resolve()),
            "runtime_root": (
                None if self.runtime_root is None else str(self.runtime_root.expanduser().resolve())
            ),
            "continuation_receipt_sha256": (
                None if self.continuation_receipt is None else _sha256(self.continuation_receipt)
            ),
            "timeout_seconds": self.timeout_seconds,
            "max_cache_gb": self.max_cache_gb,
            "pixsfm_patch_size": self.pixsfm_patch_size,
            "robust_filter": asdict(self.robust_filter),
        }
        return _json_hash(payload)


def build_backend_command(request: RefinementRequest) -> tuple[str, ...]:
    """Build a shell-free, receipt-friendly command for the selected backend."""

    request.validate()
    python = str(request.runtime_python.expanduser().resolve())
    if request.backend == "pixsfm":
        return (
            python,
            "-m",
            "pixsfm.refine_colmap",
            "bundle_adjuster",
            "--input_path",
            str(request.input_model.expanduser().resolve()),
            "--output_path",
            str(request.raw_model),
            "--image_dir",
            str(request.images.expanduser().resolve()),
            "--cache_path",
            str(request.pixsfm_cache),
            "--config",
            "low_memory",
            "mapping.dense_features.overwrite_cache=false",
            # The retained PixSfM extension is linked to HDF5 2.1 and its
            # chunked FeatureSet cache rejects datatype handles at runtime.
            # Four-pixel in-memory sparse patches retain the official
            # low-memory strategy while keeping River V3 within host RAM.
            "mapping.dense_features.use_cache=false",
            f"mapping.dense_features.patch_size={request.pixsfm_patch_size}",
            "mapping.BA.optimizer.refine_focal_length=false",
            "mapping.BA.optimizer.refine_principal_point=false",
            "mapping.BA.optimizer.refine_extra_params=false",
            "mapping.BA.optimizer.refine_extrinsics=true",
            # HDF5 feature caches are not thread-safe in the retained PixSfM
            # runtime.  Serial extraction avoids cross-thread datatype handles.
            "mapping.BA.references.num_threads=1",
            "mapping.BA.costmaps.num_threads=1",
        )
    if request.runtime_root is None:  # guarded by request.validate(), retained for type safety
        raise ValueError("Dense-SfM requires an external runtime root")
    root = request.runtime_root.expanduser().resolve()
    if request.backend == "densesfm-refine":
        return (
            python,
            "-m",
            "sfm_diagnosis.site_pipeline.densesfm_worker",
            "refine",
            "--img_folder",
            str(request.images.expanduser().resolve()),
            "--colmap_coarse_dir",
            str(request.input_model.expanduser().resolve()),
            "--staging-dir",
            str(request.dense_staging),
            "--refined_colmap_dir",
            str(request.raw_model),
            "--config",
            str(request.dense_config),
            "--database-path",
            str(request.dense_database),
        )
    return (
        python,
        str(root / "run_full.py"),
        f"dataset_base_dir={request.run_dir / 'densesfm_dataset'}",
        "dataset_name=river",
        "scene_list=[scene]",
        "method=DenseSfM",
        "visualize=false",
        "ray.enable=false",
        "sub_use_ray=false",
        "use_prior_intrin=true",
        "neuralsfm.triangulation_mode=false",
        "neuralsfm.NEUSFM_coarse_matcher=RoMa",
        "neuralsfm.img_pair_strategy=retrieval",
        "colmap_cfg.no_refine_intrinsics=true",
    )


def render_densesfm_config(request: RefinementRequest) -> str:
    if request.runtime_root is None:
        raise ValueError("Dense-SfM requires an external runtime root")
    root = request.runtime_root.expanduser().resolve()
    payload = {
        "neuralsfm": {
            "refine_iter_n_times": 2,
            "NEUSFM_refinement_chunk_size": 500,
            "img_resize": 1200,
            "img_preload": False,
            "triangulation_mode": False,
            "NEUSFM_fine_match_model_path": str(root / "weight/mv_refinement.ckpt"),
            "NEUSFM_fine_match_cfg_path": str(
                root / "hydra_training_configs/experiment/multiview_refinement_matching.yaml"
            ),
        },
        "colmap_cfg": {
            "ImageReader_camera_mode": "auto",
            "ImageReader_single_camera": True,
            "min_model_size": 6,
            "no_refine_intrinsics": True,
            "n_threads": 8,
            "use_pba": False,
            "geometry_verify_thr": 10.0,
            "reregistration": {
                "abs_pose_max_error": 12,
                "abs_pose_min_num_inliers": 30,
                "abs_pose_min_inlier_ratio": 0.25,
                "filter_max_reproj_error": 5,
            },
            "colmap_mapper_cfgs": {
                "init_max_error": 10,
                "abs_pose_max_error": 12,
                "filter_max_reproj_error": 10,
                "tri_merge_max_reproj_error": 10,
                "tri_complete_max_reproj_error": 10,
                "tri_ignore_two_view_tracks": 1,
            },
        },
    }
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise RuntimeError("PyYAML is required to render Dense-SfM config") from error
    return yaml.safe_dump(payload, sort_keys=False)


def run_refinement(
    request: RefinementRequest, *, dry_run: bool = False, resume: bool = False
) -> dict[str, Any]:
    """Run one backend, apply the canonical robust filter, and emit an immutable receipt."""

    fingerprint = request.fingerprint()
    command = build_backend_command(request)
    receipt_path = request.run_dir / "receipts" / f"refinement_{request.backend}.json"
    if dry_run:
        return {
            "status": "DRY_RUN_READY",
            "backend": request.backend,
            "request_fingerprint": fingerprint,
            "command": list(command),
            "raw_model": str(request.raw_model),
            "robust_model": str(request.robust_model),
        }
    if request.run_dir.exists():
        if resume and receipt_path.is_file():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                previous.get("status") == "COMPLETED"
                and previous.get("request_fingerprint") == fingerprint
                and request.robust_model.is_dir()
            ):
                return previous
        raise FileExistsError(request.run_dir)

    (request.run_dir / "bootstrap").mkdir(parents=True)
    (request.run_dir / "logs").mkdir()
    receipt_path.parent.mkdir()
    request.cache_dir.mkdir(parents=True, exist_ok=True)
    if request.backend.startswith("densesfm"):
        request.dense_config.write_text(render_densesfm_config(request), encoding="utf-8")
    request_payload = {
        "schema_version": 1,
        "backend": request.backend,
        "request_fingerprint": fingerprint,
        "input_model": str(request.input_model.expanduser().resolve()),
        "input_model_hashes": _model_hashes(request.input_model),
        "images": str(request.images.expanduser().resolve()),
        "intrinsics": str(request.intrinsics.expanduser().resolve()),
        "intrinsics_sha256": _sha256(request.intrinsics),
        "pairs": None if request.pairs is None else str(request.pairs.expanduser().resolve()),
        "pairs_sha256": None if request.pairs is None else _sha256(request.pairs),
        "command": list(command),
        "fixed_intrinsics": True,
        "refine_extrinsics": True,
        "pixsfm_patch_size": request.pixsfm_patch_size,
        "pixsfm_hdf5_fallback": (
            "IN_MEMORY_SPARSE_PATCHES_SIZE_4" if request.backend == "pixsfm" else None
        ),
    }
    (request.run_dir / "request.json").write_text(
        json.dumps(request_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    runtime = probe_python_runtime(request.runtime_python)
    require_compute_capability(runtime, "12.0")
    (request.run_dir / "runtime_preflight.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    log_path = request.run_dir / "logs" / f"{request.backend}.log"
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if request.runtime_root is not None:
        root = str(request.runtime_root.expanduser().resolve())
        environment["PYTHONPATH"] = root + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
    request.raw_model.parent.mkdir(parents=True)
    with ExclusiveResourceLease(request.run_dir / "locks/heavy.lock", "gpu_heavy"):
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                command,
                cwd=None if request.runtime_root is None else request.runtime_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
    if completed.returncode:
        raise RuntimeError(f"{request.backend} failed ({completed.returncode}); see {log_path}")
    missing = [name for name in MODEL_FILES if not (request.raw_model / name).is_file()]
    if missing:
        raise RuntimeError(f"refinement output is missing model files: {missing}")

    filter_receipt = robust_filter_model(
        request.raw_model, request.robust_model, request.robust_filter
    )
    input_metrics = _load_metrics(request.input_model)
    candidate_metrics = _load_metrics(request.robust_model)
    qa = evaluate_geometry_gate(input_metrics, candidate_metrics)
    from .loo import aligned_camera_stability

    pose_stability = aligned_camera_stability(
        _camera_pose_map(request.input_model), _camera_pose_map(request.robust_model)
    )
    qa = apply_pose_stability_gate(qa, pose_stability)
    from river_v4_optimizer.metrics import analyze_model

    baseline_topology = _topology_summary(analyze_model(request.input_model))
    candidate_topology = _topology_summary(analyze_model(request.robust_model))
    qa = apply_topology_gate(qa, baseline_topology, candidate_topology)
    qa_path = request.run_dir / "qa.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    filter_path = request.run_dir / "robust_filter_receipt.json"
    filter_path.write_text(json.dumps(filter_receipt, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": 1,
        "status": "COMPLETED",
        "backend": request.backend,
        "implementation_scope": (
            "PIXEL_PERFECT_SFM_FEATUREMETRIC_BA"
            if request.backend == "pixsfm"
            else "PUBLIC_DENSESFM_WITHOUT_GS_TRACK_EXTENSION"
        ),
        "request_fingerprint": fingerprint,
        "runtime": runtime,
        "command": list(command),
        "raw_model": str(request.raw_model),
        "robust_model": str(request.robust_model),
        "raw_model_hashes": _model_hashes(request.raw_model),
        "robust_model_hashes": _model_hashes(request.robust_model),
        "geometry_gate": qa,
        "filter_receipt": str(filter_path),
        "log": str(log_path),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return receipt


def evaluate_geometry_gate(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *, tolerance: float = 0.05
) -> dict[str, Any]:
    baseline_registered = int(baseline.get("registered_images") or 0)
    candidate_registered = int(candidate.get("registered_images") or 0)
    checks = {
        "fixed_intrinsics_unchanged": baseline.get("cameras") == candidate.get("cameras"),
        "registered_images_non_regression": candidate_registered >= baseline_registered,
        "reprojection_p90_within_five_percent": _within_upper_tolerance(
            baseline.get("point_error_p90_px"), candidate.get("point_error_p90_px"), tolerance
        ),
        "reprojection_p99_within_five_percent": _within_upper_tolerance(
            baseline.get("point_error_p99_px"), candidate.get("point_error_p99_px"), tolerance
        ),
        "track_p50_non_regression": _not_lower(
            baseline.get("track_length_p50"), candidate.get("track_length_p50")
        ),
        "observations_within_five_percent": _within_lower_tolerance(
            baseline.get("observations"), candidate.get("observations"), tolerance
        ),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "baseline": dict(baseline),
        "candidate": dict(candidate),
    }


def flatten_image_name(name: str) -> str:
    """Make hierarchical COLMAP names safe for Dense-SfM's basename-only loader."""

    normalized = name.replace("\\", "/").strip("/")
    return normalized.replace("/", "__")


def evaluate_localization_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_successes: int = 8,
) -> dict[str, Any]:
    baseline_rows = {str(row["query_id"]): row for row in baseline.get("results") or ()}
    candidate_rows = {str(row["query_id"]): row for row in candidate.get("results") or ()}
    missing = sorted(set(baseline_rows) - set(candidate_rows))
    baseline_successes = {
        query_id for query_id, row in baseline_rows.items() if bool(row.get("success"))
    }
    candidate_successes = {
        query_id for query_id, row in candidate_rows.items() if bool(row.get("success"))
    }
    lost = sorted(baseline_successes - candidate_successes)
    baseline_inliers = _median_field(baseline_rows.values(), "ransac_inliers")
    candidate_inliers = _median_field(candidate_rows.values(), "ransac_inliers")
    checks = {
        "query_set_complete": not missing,
        "minimum_eight_successes": len(candidate_successes) >= minimum_successes,
        "baseline_successes_preserved": not lost,
        "median_inliers_non_regression": candidate_inliers >= baseline_inliers,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "baseline_successes": len(baseline_successes),
        "candidate_successes": len(candidate_successes),
        "lost_baseline_successes": lost,
        "missing_queries": missing,
        "baseline_median_inliers": baseline_inliers,
        "candidate_median_inliers": candidate_inliers,
    }


def apply_pose_stability_gate(
    geometry: Mapping[str, Any], stability: Mapping[str, Any]
) -> dict[str, Any]:
    result = dict(geometry)
    checks = dict(result.get("checks") or {})
    checks.update(
        {
            "sim3_alignment_available": stability.get("status") == "OK",
            "sim3_position_p90_within_0_02": (
                stability.get("status") == "OK"
                and stability.get("position_p90_normalized") is not None
                and float(stability["position_p90_normalized"]) <= 0.02
            ),
            "sim3_rotation_p90_within_2deg": (
                stability.get("status") == "OK"
                and stability.get("rotation_p90_deg") is not None
                and float(stability["rotation_p90_deg"]) <= 2.0
            ),
        }
    )
    result["checks"] = checks
    result["pose_stability"] = dict(stability)
    result["passes"] = all(checks.values())
    return result


def apply_topology_gate(
    geometry: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(geometry)
    checks = dict(result.get("checks") or {})
    checks.update(
        {
            "largest_component_non_regression": float(
                candidate.get("largest_component_ratio") or 0.0
            )
            >= float(baseline.get("largest_component_ratio") or 0.0),
            "articulation_count_non_regression": int(candidate.get("articulation_count") or 0)
            <= int(baseline.get("articulation_count") or 0),
            "bridge_count_non_regression": int(candidate.get("bridge_count") or 0)
            <= int(baseline.get("bridge_count") or 0),
            "multi_view_track_non_regression": float(
                candidate.get("multi_view_track_ratio_ge5") or 0.0
            )
            >= float(baseline.get("multi_view_track_ratio_ge5") or 0.0),
            "low_observation_images_non_regression": len(
                candidate.get("low_observation_images") or ()
            )
            <= len(baseline.get("low_observation_images") or ()),
        }
    )
    result["checks"] = checks
    result["topology"] = {"baseline": dict(baseline), "candidate": dict(candidate)}
    result["passes"] = all(checks.values())
    return result


def _load_metrics(model: Path) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    metrics = reconstruction_metrics(reconstruction)
    metrics["cameras"] = {
        str(camera_id): {
            "model": camera.model_name,
            "width": int(camera.width),
            "height": int(camera.height),
            "params": [float(value) for value in camera.params],
        }
        for camera_id, camera in sorted(reconstruction.cameras.items())
    }
    return metrics


def _camera_pose_map(model: Path) -> dict[str, dict[str, Any]]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    poses: dict[str, dict[str, Any]] = {}
    for image in reconstruction.images.values():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world()
        poses[image.name] = {
            "center": cam_from_world.inverse().translation.tolist(),
            "rotation": cam_from_world.rotation.matrix().tolist(),
        }
    return poses


def _topology_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "largest_component_ratio",
            "component_sizes",
            "articulation_count",
            "articulation_images",
            "bridge_count",
            "bridge_edges",
            "multi_view_track_ratio_ge5",
            "low_observation_images",
        )
    }


def _within_upper_tolerance(baseline: Any, candidate: Any, tolerance: float) -> bool:
    if baseline is None or candidate is None:
        return False
    return float(candidate) <= float(baseline) * (1.0 + tolerance)


def _within_lower_tolerance(baseline: Any, candidate: Any, tolerance: float) -> bool:
    if baseline is None or candidate is None:
        return False
    return float(candidate) >= float(baseline) * (1.0 - tolerance)


def _not_lower(baseline: Any, candidate: Any) -> bool:
    if baseline is None or candidate is None:
        return False
    return float(candidate) >= float(baseline)


def _median_field(rows: Sequence[Mapping[str, Any]] | Any, field: str) -> float:
    values = [float(row.get(field) or 0) for row in rows]
    return 0.0 if not values else float(statistics.median(values))


def _model_hashes(path: Path) -> dict[str, str]:
    model = path.expanduser().resolve(strict=True)
    return {name: _sha256(model / name) for name in MODEL_FILES if (model / name).is_file()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode()).hexdigest()


__all__ = [
    "REFINEMENT_BACKENDS",
    "RefinementRequest",
    "apply_pose_stability_gate",
    "apply_topology_gate",
    "build_backend_command",
    "evaluate_geometry_gate",
    "evaluate_localization_gate",
    "flatten_image_name",
    "render_densesfm_config",
    "run_refinement",
]
