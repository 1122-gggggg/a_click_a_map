"""Immutable all-sequence GLUEMAP preparation without graph-aware selection."""

from __future__ import annotations

import gc
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from sfm_diagnosis.io import write_json
from sfm_diagnosis.risk_ply import write_binary_ply

from .config import PipelineConfig
from .frame_analyzer import analyze_frames, extract_frames
from .intrinsics import calibration_matrix_for_resolution
from .inventory import discover_corpus, sha256_file
from .preprocessing import DirectSamplingPolicy, plan_direct_keyframes, sanitize_frames
from .gluemap_worker import run_adapter_request as run_gluemap_adapter

SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Points with no in-image observation tolerated in the RGB preview export.
MAX_UNSAMPLED_POINT_FRACTION = 1e-4


@dataclass(frozen=True)
class DirectMappingRequest:
    site_name: str
    corpus_root: Path
    run_dir: Path
    intrinsics_path: Path
    policy: DirectSamplingPolicy = field(default_factory=DirectSamplingPolicy)

    def __post_init__(self) -> None:
        if not SAFE_NAME.fullmatch(self.site_name):
            raise ValueError("direct mapping site name is unsafe")
        for field_name in ("corpus_root", "run_dir", "intrinsics_path"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)).expanduser().resolve())
        if not self.corpus_root.is_dir():
            raise FileNotFoundError(self.corpus_root)
        if not self.intrinsics_path.is_file():
            raise FileNotFoundError(self.intrinsics_path)
        if self.run_dir == self.corpus_root or self.corpus_root in self.run_dir.parents:
            raise ValueError("direct mapping run must not be created inside the source corpus")


@dataclass(frozen=True)
class DirectMappingRuntime:
    gluemap_root: Path
    base_config_path: Path
    workspace_root: Path
    megaloc_source: Path
    megaloc_checkpoint: Path
    edm_root: Path | None = None
    edm_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "gluemap_root",
            "base_config_path",
            "workspace_root",
            "megaloc_source",
            "megaloc_checkpoint",
            "edm_root",
            "edm_checkpoint",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value).expanduser().resolve())
        required = (
            self.gluemap_root / "gluemap",
            self.base_config_path,
            self.megaloc_source,
            self.megaloc_checkpoint,
        )
        if any(not path.exists() for path in required):
            raise FileNotFoundError(next(path for path in required if not path.exists()))


def prepare_direct_run(request: DirectMappingRequest, *, resume: bool = False) -> dict[str, Any]:
    """Hash every source, adaptively extract frames, and write direct inputs."""

    receipt_path = request.run_dir / "receipts/direct_preprocessing.json"
    if resume and receipt_path.is_file():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    request.run_dir.mkdir(parents=True, exist_ok=False)
    calibration = json.loads(request.intrinsics_path.read_text(encoding="utf-8"))
    if calibration.get("images_are_undistorted") is not True:
        raise ValueError("direct mapping requires already-undistorted input calibration")
    config = PipelineConfig(
        site_name=request.site_name,
        heldout_patterns=(),
        required_metadata=(),
    )
    manifest = discover_corpus(request.corpus_root, config)
    sources = [
        row
        for row in manifest.get("sources", ())
        if row.get("source_kind") != "archive"
    ]
    if not sources or any(row.get("evaluation_role") != "MAPPING" for row in sources):
        raise ValueError("direct mapping accepts mapping sources only")

    inputs = request.run_dir / "inputs"
    frames_root = request.run_dir / "artifacts/preprocessing"
    image_root = request.run_dir / "artifacts/keyframes/images"
    inputs.mkdir(parents=True)
    frames_root.mkdir(parents=True)
    image_root.mkdir(parents=True)
    write_json(inputs / "corpus_manifest.json", manifest)
    (inputs / "intrinsics.json").write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_json(
        inputs / "direct_config.json",
        {
            "schema_version": 1,
            "artifact_type": "DIRECT_MAPPING_CONFIG",
            "site_name": request.site_name,
            "validation": "NONE",
            "graph_checks": "SKIPPED_BY_REQUEST",
            "sampling": asdict(request.policy),
            "images_are_undistorted": True,
        },
    )

    frame_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    forced_ids: list[tuple[str, str]] = []
    per_source: list[dict[str, Any]] = []
    for source in sources:
        width, height = int(source.get("width") or 0), int(source.get("height") or 0)
        if min(width, height) <= 0:
            raise ValueError(f"source resolution is unavailable: {source.get('relative_path')}")
        matrix = calibration_matrix_for_resolution(
            calibration,
            target_size=(width, height),
        )
        video_id = str(source.get("video_id") or source["source_id"])
        source_path = Path(str(source["path"])).resolve(strict=True)
        analyzed = analyze_frames(
            source_path,
            video_id=video_id,
            session_id=source_path.stem,
            fps=request.policy.probe_fps,
            intrinsics=matrix,
        )
        for row in analyzed:
            row["duplicate_previous"] = bool(row.get("near_duplicate"))
            row["evaluation_role"] = "MAPPING"
        sanitized = sanitize_frames(analyzed)
        plan = plan_direct_keyframes(sanitized.kept, policy=request.policy)
        extraction_requests = []
        source_keyframes: list[dict[str, Any]] = []
        for item in plan.keyframes:
            frame = item.frame
            relative = Path(video_id) / f"frame_{frame.frame_index:08d}.jpg"
            image_path = image_root / relative
            extraction_requests.append(
                {
                    "source_frame_index": frame.frame_index,
                    "source_image_path": frame.source.get("source_image_path"),
                    "output": image_path,
                }
            )
            source_keyframes.append(
                {
                    **dict(frame.source),
                    "keyframe_id": frame.frame_id,
                    "frame_id": frame.frame_id,
                    "video_id": video_id,
                    "session_id": source_path.stem,
                    "source_uri": str(source_path),
                    "source_frame_index": frame.frame_index,
                    "source_pts_seconds": frame.timestamp,
                    "image_uri": str(image_path),
                    "output_name": relative.as_posix(),
                    "mapping_mode": item.mapping_mode,
                    "status": "CANDIDATE",
                    "warnings": list(frame.warnings),
                }
            )
        extract_frames(source_path, extraction_requests)
        for row in source_keyframes:
            image_path = Path(str(row["image_uri"]))
            row["image_sha256"] = sha256_file(image_path)
        keyframe_rows.extend(source_keyframes)
        forced_ids.extend(plan.forced_pairs)
        frame_rows.extend(_sanitization_rows(sanitized))
        per_source.append(
            {
                "video_id": video_id,
                "analyzed_frames": len(analyzed),
                "candidate_frames": len(sanitized.kept),
                "removed_frames": len(sanitized.removed),
                "keyframes": len(source_keyframes),
                "pose_only": sum(row["mapping_mode"] == "POSE_ONLY" for row in source_keyframes),
            }
        )

    _write_jsonl(frames_root / "frames.jsonl", frame_rows)
    keyframes_path = request.run_dir / "artifacts/keyframes/keyframes.jsonl"
    _write_jsonl(keyframes_path, keyframe_rows)
    name_by_id = {str(row["keyframe_id"]): str(row["output_name"]) for row in keyframe_rows}
    forced_names = [
        (name_by_id[left], name_by_id[right])
        for left, right in forced_ids
        if left in name_by_id and right in name_by_id
    ]
    (inputs / "forced_pairs.txt").write_text(
        "".join(f"{left} {right}\n" for left, right in forced_names),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "DIRECT_MAPPING_PREPROCESSING_RECEIPT",
        "status": "PREPARED",
        "validation": "NONE",
        "graph_checks": "SKIPPED_BY_REQUEST",
        "source_count": len(sources),
        "keyframes": len(keyframe_rows),
        "pose_only_keyframes": sum(
            row["mapping_mode"] == "POSE_ONLY" for row in keyframe_rows
        ),
        "forced_pairs": len(forced_names),
        "per_source": per_source,
        "inputs": {
            "corpus_manifest": str(inputs / "corpus_manifest.json"),
            "keyframes": str(keyframes_path),
            "forced_pairs": str(inputs / "forced_pairs.txt"),
        },
    }
    write_json(receipt_path, receipt)
    return receipt


def sample_reconstruction_rgb(
    reconstruction: Any,
    keyframes: Mapping[str, Mapping[str, Any]],
    *,
    image_loader: Callable[[str], Any],
) -> tuple[Any, Any, dict[str, int]]:
    """Average decoded RGB observations for every 3D point without recoloring."""

    import numpy as np

    point_rows = sorted(reconstruction.points3D.items(), key=lambda item: int(item[0]))
    point_ids = np.asarray([int(point_id) for point_id, _ in point_rows], dtype=np.int64)
    xyz = np.asarray([point.xyz for _, point in point_rows], dtype=np.float64).reshape(-1, 3)
    sums = np.zeros((len(point_rows), 3), dtype=np.uint64)
    counts = np.zeros(len(point_rows), dtype=np.uint32)
    observations = 0
    for image in sorted(reconstruction.images.values(), key=lambda value: str(value.name)):
        name = str(image.name)
        keyframe = keyframes.get(name)
        if keyframe is None:
            raise RuntimeError(f"registered image is absent from keyframes: {name}")
        pixels = image_loader(str(keyframe["image_uri"]))
        if pixels is None:
            raise FileNotFoundError(str(keyframe["image_uri"]))
        height, width = pixels.shape[:2]
        material = [point for point in image.points2D if point.has_point3D()]
        if not material:
            continue
        ids = np.asarray([int(point.point3D_id) for point in material], dtype=np.int64)
        xy = np.asarray([point.xy for point in material], dtype=np.float64).reshape(-1, 2)
        columns = np.rint(xy[:, 0]).astype(np.int64)
        rows = np.rint(xy[:, 1]).astype(np.int64)
        indexes = np.searchsorted(point_ids, ids)
        valid = (
            (indexes < len(point_ids))
            & (point_ids[np.minimum(indexes, len(point_ids) - 1)] == ids)
            & (columns >= 0)
            & (columns < width)
            & (rows >= 0)
            & (rows < height)
        )
        if not np.any(valid):
            continue
        indexes = indexes[valid]
        bgr = np.asarray(pixels[rows[valid], columns[valid], :3], dtype=np.uint8)
        rgb = bgr[:, ::-1]
        np.add.at(sums, indexes, rgb.astype(np.uint64))
        np.add.at(counts, indexes, 1)
        observations += int(len(indexes))
    missing = counts == 0
    colors = np.zeros((len(point_rows), 3), dtype=np.uint8)
    colors[~missing] = np.rint(
        sums[~missing] / counts[~missing, None]
    ).astype(np.uint8)
    stats = {
        "colored_points": int((~missing).sum()),
        "missing_color_points": int(missing.sum()),
        "observations": observations,
    }
    if missing.any():
        stats["missing_point_indices"] = np.flatnonzero(missing).tolist()
        stats["missing_point_ids"] = point_ids[missing].tolist()
    return xyz, colors, stats


def sample_model_rgb(
    model_dir: str | Path,
    keyframes_path: str | Path,
) -> tuple[Any, Any, dict[str, int]]:
    import cv2
    import pycolmap

    keyframes = {
        str(row["output_name"]): row for row in _read_jsonl(Path(keyframes_path))
    }
    reconstruction = pycolmap.Reconstruction(str(Path(model_dir).resolve(strict=True)))
    return sample_reconstruction_rgb(
        reconstruction,
        keyframes,
        image_loader=lambda path: cv2.imread(path, cv2.IMREAD_COLOR),
    )


def export_original_rgb_ply(
    model_dir: str | Path,
    keyframes_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Export frame-sampled RGB without display or risk recoloring."""

    import numpy as np

    xyz, rgb, stats = sample_model_rgb(model_dir, keyframes_path)
    missing_count = int(stats["missing_color_points"])
    dropped: dict[str, Any] = {}
    if missing_count:
        # A point whose every observation lies outside its image (tracker
        # extrapolation past the border) has no pixel to sample. Tolerate only a
        # negligible fraction; anything larger is a systemic sampling failure.
        if missing_count > MAX_UNSAMPLED_POINT_FRACTION * len(xyz):
            raise RuntimeError(
                f"RGB sampling missed {missing_count} of {len(xyz)} reconstructed points"
            )
        keep = np.ones(len(xyz), dtype=bool)
        keep[np.asarray(stats["missing_point_indices"], dtype=np.int64)] = False
        xyz, rgb = xyz[keep], rgb[keep]
        dropped = {
            "dropped_unsampled_point_ids": list(stats["missing_point_ids"]),
            "dropped_reason": "all observations outside image bounds",
        }
    output = write_binary_ply(
        output_path,
        xyz,
        rgb,
        comments=("mean decoded RGB over source-frame track observations; no recoloring",),
    )
    return {
        "schema_version": 1,
        "artifact_type": "ORIGINAL_RGB_PLY_RECEIPT",
        "path": str(output),
        "vertices": int(len(xyz)),
        **{key: value for key, value in stats.items() if not key.startswith("missing_point_")},
        **dropped,
        "aggregation": "MEAN_TRACK_OBSERVATION_RGB",
        "sha256": sha256_file(output),
    }


def localization_reference_rows(
    reconstruction: Any,
    keyframes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return registered triangulating images that can lift EDM matches to 3D."""

    rows: list[dict[str, Any]] = []
    for image in sorted(reconstruction.images.values(), key=lambda value: str(value.name)):
        name = str(image.name)
        keyframe = keyframes.get(name)
        if keyframe is None:
            raise RuntimeError(f"registered image is absent from keyframes: {name}")
        observations = sum(bool(point.has_point3D()) for point in image.points2D)
        if keyframe.get("mapping_mode") == "POSE_ONLY" or observations == 0:
            continue
        rows.append(
            {
                "image_name": name,
                "image_path": str(keyframe["image_uri"]),
                "video_id": str(keyframe["video_id"]),
                "observations": observations,
            }
        )
    return rows


def build_localization_package(
    *,
    model_dir: str | Path,
    keyframes_path: str | Path,
    output_dir: str | Path,
    runtime: DirectMappingRuntime,
    intrinsics: Mapping[str, Any],
    descriptor_batch_size: int = 8,
) -> dict[str, Any]:
    """Build the frozen MegaLoc reference bank consumed by EDM+PnP queries."""

    import pycolmap
    from river_map_quality.megaloc_edm_catalog import (
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )

    keyframes = {
        str(row["output_name"]): row
        for row in _read_jsonl(Path(keyframes_path))
    }
    reconstruction = pycolmap.Reconstruction(str(Path(model_dir).resolve(strict=True)))
    references = localization_reference_rows(reconstruction, keyframes)
    if not references:
        raise RuntimeError("direct map contains no references with 3D observations")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "reference_manifest.jsonl"
    _write_jsonl(manifest_path, references)
    image_paths = [Path(str(row["image_path"])).resolve(strict=True) for row in references]
    megaloc_runtime = load_offline_megaloc_runtime(
        source=runtime.megaloc_source,
        checkpoint=runtime.megaloc_checkpoint,
        device="cuda",
    )
    descriptors = extract_megaloc_descriptors(
        megaloc_runtime,
        image_paths,
        batch_size=descriptor_batch_size,
    )
    del megaloc_runtime
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - runtime dependent
        pass
    import numpy as np

    descriptor_array = np.ascontiguousarray(descriptors, dtype=np.float32)
    if descriptor_array.ndim != 2 or descriptor_array.shape[0] != len(references):
        raise RuntimeError("MegaLoc descriptor rows disagree with reference identities")
    descriptor_path = output / "megaloc_references.npy"
    np.save(descriptor_path, descriptor_array, allow_pickle=False)
    names_path = output / "megaloc_references.names.json"
    names_path.write_text(
        json.dumps([row["image_name"] for row in references], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    config_path = output / "localizer_config.json"
    write_json(
        config_path,
        {
            "schema_version": 1,
            "artifact_type": "DIRECT_MEGALOC_EDM_PNP_CONFIG",
            "map_model": str(Path(model_dir).resolve()),
            "reference_manifest": str(manifest_path),
            "megaloc_source": str(runtime.megaloc_source),
            "megaloc_checkpoint": str(runtime.megaloc_checkpoint),
            "edm_root": None if runtime.edm_root is None else str(runtime.edm_root),
            "edm_checkpoint": (
                None if runtime.edm_checkpoint is None else str(runtime.edm_checkpoint)
            ),
            "intrinsics": dict(intrinsics),
            "top_k": 5,
            "lift_distance_px": 2.0,
            "pose_solver": "COLMAP_PNP_RANSAC",
            "validation": "NONE",
        },
    )
    return {
        "references": len(references),
        "descriptor_shape": list(descriptor_array.shape),
        "reference_manifest": str(manifest_path),
        "descriptors": str(descriptor_path),
        "descriptor_sha256": sha256_file(descriptor_path),
        "config": str(config_path),
    }


def run_direct_mapping(
    request: DirectMappingRequest,
    runtime: DirectMappingRuntime,
    *,
    resume: bool = False,
    preprocess_only: bool = False,
) -> dict[str, Any]:
    """Run preprocessing, one native multi-sequence GLUEMAP job, and packaging."""

    preprocessing = prepare_direct_run(request, resume=resume)
    if preprocess_only:
        return preprocessing
    calibration = json.loads(request.intrinsics_path.read_text(encoding="utf-8"))
    config = json.loads(runtime.base_config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "coarse_only": False,
            "extra_pairs_path": str(request.run_dir / "inputs/forced_pairs.txt"),
            "is_multi_sequence": True,
            "is_sequential": True,
            "sample_frequency": 1,
            "skip_doppelgangers": False,
            "subfolder_regex": ".*",
        }
    )
    gluemap_config = request.run_dir / "inputs/gluemap_config.json"
    gluemap_config.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_model = request.run_dir / "artifacts/mapping/direct/model"
    adapter_payload = {
            "config": {
                "gluemap_root": str(runtime.gluemap_root),
                "config_file": str(gluemap_config),
                "mode": "final",
                "evidence_level": "refined_geometry",
                "pair_source": "native",
                "reuse_inference_cache": bool(resume),
                "workspace_root": str(runtime.workspace_root),
                "skip_doppelgangers": False,
                "sift_device": "cpu",
                "intrinsics_calibration": calibration,
            },
            "payload": {
                "mode": "final",
                "keyframes": str(
                    request.run_dir / "artifacts/keyframes/keyframes.jsonl"
                ),
                "output_model": str(output_model),
                "run_root": str(request.run_dir),
                "corpus_manifest": str(request.run_dir / "inputs/corpus_manifest.json"),
            },
        }
    if resume and _complete_model(output_model):
        mapping = _recover_completed_mapping(
            output_model,
            request.run_dir / "artifacts/keyframes/keyframes.jsonl",
        )
    else:
        mapping = run_gluemap_adapter(adapter_payload)
    products = request.run_dir / "products"
    rgb_ply = export_original_rgb_ply(
        output_model,
        request.run_dir / "artifacts/keyframes/keyframes.jsonl",
        products / "map/original_rgb.ply",
    )
    write_json(products / "map/original_rgb_receipt.json", rgb_ply)
    localization = run_localization_packaging_isolated(
        {
            "model_dir": str(output_model),
            "keyframes_path": str(
                request.run_dir / "artifacts/keyframes/keyframes.jsonl"
            ),
            "output_dir": str(products / "localization"),
            "runtime": {
                "gluemap_root": str(runtime.gluemap_root),
                "base_config_path": str(runtime.base_config_path),
                "workspace_root": str(runtime.workspace_root),
                "megaloc_source": str(runtime.megaloc_source),
                "megaloc_checkpoint": str(runtime.megaloc_checkpoint),
                "edm_root": None if runtime.edm_root is None else str(runtime.edm_root),
                "edm_checkpoint": (
                    None if runtime.edm_checkpoint is None else str(runtime.edm_checkpoint)
                ),
            },
            "intrinsics": calibration,
        }
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "DIRECT_MAPPING_FINAL_RECEIPT",
        "status": "MAP_BUILT_UNVALIDATED_ALL_INPUTS",
        "validation": "NONE",
        "graph_checks": "SKIPPED_BY_REQUEST",
        "preprocessing": preprocessing,
        "mapping": mapping,
        "rgb_ply": rgb_ply,
        "localization": localization,
    }
    write_json(request.run_dir / "products/FINAL_RECEIPT.json", receipt)
    return receipt


def run_localization_packaging_isolated(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build MegaLoc assets in a fresh process before any GLUEMAP torch import."""

    completed = subprocess.run(
        [sys.executable, "-m", "sfm_diagnosis.site_pipeline.localization_package_worker"],
        input=json.dumps(dict(payload)),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated localization packaging failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated localization packaging returned invalid JSON") from error
    if result.get("status") != "completed":
        raise RuntimeError(f"isolated localization packaging failed: {result}")
    return dict(result)


def _complete_model(path: Path) -> bool:
    return all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    )


def _recover_completed_mapping(model: Path, keyframes_path: Path) -> dict[str, Any]:
    import pycolmap

    from .pose_only import assert_pose_only_observation_free

    reconstruction = pycolmap.Reconstruction(str(model.resolve(strict=True)))
    pose_names = {
        str(row["output_name"])
        for row in _read_jsonl(keyframes_path)
        if row.get("mapping_mode") == "POSE_ONLY"
    }
    return {
        "status": "completed",
        "outputs": [str(model)],
        "pair_source": "native",
        "cache_reuse": True,
        "recovered_completed_model": True,
        "registered_images": reconstruction.num_reg_images(),
        "points3D": len(reconstruction.points3D),
        "pose_only_check": assert_pose_only_observation_free(reconstruction, pose_names),
    }


def _sanitization_rows(result) -> list[dict[str, Any]]:
    return [
        {
            **dict(record.source),
            "frame_id": record.frame_id,
            "status": record.status,
            "warnings": list(record.warnings),
        }
        for record in result.kept
    ] + [
        {
            **dict(record.source),
            "frame_id": record.frame_id,
            "status": "INACTIVE_REJECT",
            "rejection_reason": record.reason,
        }
        for record in result.removed
    ]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


__all__ = [
    "DirectMappingRequest",
    "DirectMappingRuntime",
    "build_localization_package",
    "export_original_rgb_ply",
    "localization_reference_rows",
    "prepare_direct_run",
    "run_localization_packaging_isolated",
    "run_direct_mapping",
    "sample_model_rgb",
    "sample_reconstruction_rgb",
]
