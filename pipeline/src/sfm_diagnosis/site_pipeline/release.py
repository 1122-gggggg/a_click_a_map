"""Materialize robust/dense products and fuse strict dual-layer localization."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sfm_diagnosis.io import load_gluemap

from .adapters import (
    AdapterRequest,
    CommandAdapter,
    ExclusiveResourceLease,
    probe_python_runtime,
    require_compute_capability,
)
from .localization_ensemble import combine_result_sets
from .robust_filter import RobustFilterConfig, robust_filter_model

_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_FUSION_RULE = (
    "accept a single strict layer; if both are strict require cross-layer agreement; "
    "never use ground truth"
)


def materialize_layers(
    run_dir: str | Path,
    *,
    export_ply: bool = False,
) -> Path:
    """Create the robust base geometry while retaining dense localization support."""

    run = Path(run_dir).expanduser().resolve(strict=True)
    dense_model = (run / "artifacts/mapping/final/model").resolve(strict=True)
    robust_model = run / "artifacts/mapping/robust/model"
    config_path = run / "inputs/pipeline_config.json"
    pipeline_config = _read_json(config_path)
    raw_filter = dict((pipeline_config.get("resources") or {}).get("robust_filter") or {})
    if not raw_filter:
        raise RuntimeError("resources.robust_filter is required for robust release materialization")
    allowed = set(RobustFilterConfig.__dataclass_fields__)
    unknown = set(raw_filter) - allowed
    if unknown:
        raise ValueError(f"unknown robust-filter settings: {sorted(unknown)}")
    filter_config = RobustFilterConfig(**raw_filter)
    payload = robust_filter_model(dense_model, robust_model, filter_config)
    payload.update(
        dense_model_hashes=model_hashes(dense_model),
        robust_model_hashes=model_hashes(robust_model),
        config_sha256=_sha256(config_path),
    )
    products = run / "products"
    _link_directory(products / "base_geometry/model", robust_model)
    _link_directory(products / "localization_dense/model", dense_model)
    if export_ply:
        payload["products"] = _export_plys(run, dense_model, robust_model)
    receipt = run / "receipts/robust_filter.json"
    _write_json_atomic(receipt, payload)
    _write_json_atomic(
        products / "localization_ensemble/MANIFEST.json",
        _layer_manifest(run, payload),
    )
    return receipt

def validate_localization_layers(
    run_dir: str | Path,
    *,
    outer_holdout_frozen: bool = False,
    maximum_position_normalized: float = 0.02,
    maximum_rotation_deg: float = 2.0,
    target_rate: float = 0.95,
) -> Path:
    """Run the configured strict localizer on both layers, then fuse its results."""

    run = Path(run_dir).expanduser().resolve(strict=True)
    pipeline_config = _read_json(run / "inputs/pipeline_config.json")
    localizer = copy.deepcopy((pipeline_config.get("adapters") or {}).get("localizer") or {})
    if not localizer:
        raise RuntimeError("adapters.localizer is required for dual-layer validation")
    command = [
        str(value).replace("{run_dir}", str(run))
        for value in localizer.get("command") or ()
    ]
    if not command:
        raise RuntimeError("adapters.localizer.command is required")
    if str(localizer.get("loo_mode") or "") != "strict":
        raise RuntimeError("dual-layer release validation requires loo_mode='strict'")
    resource_class = str(localizer.get("resource_class") or "matcher")
    preflight = probe_python_runtime(command[0])
    required_capability = (pipeline_config.get("resources") or {}).get(
        "require_compute_capability"
    )
    if required_capability:
        require_compute_capability(preflight, str(required_capability))
    validations: dict[str, Path] = {}
    references: dict[str, Path] = {}
    for layer, model in (
        ("robust", run / "artifacts/mapping/robust/model"),
        ("dense", run / "artifacts/mapping/final/model"),
    ):
        model.resolve(strict=True)
        layer_config = _expand_run_paths(copy.deepcopy(localizer), run)
        provider_kwargs = dict(layer_config.get("provider_kwargs") or {})
        provider_kwargs["map_model"] = str(model)
        provider_kwargs.setdefault(
            "keyframes", str(run / "artifacts/keyframes/keyframes.jsonl")
        )
        provider_kwargs["cache_dir"] = str(run / "artifacts/localization" / layer / "cache")
        layer_config["provider_kwargs"] = provider_kwargs
        validation = run / "artifacts/localization" / layer / "validation.json"
        reference = run / "artifacts/localization" / layer / "references.jsonl"
        request = AdapterRequest(
            f"localization_{layer}",
            payload={
                "model": str(model),
                "corpus": str(run / "inputs/corpus_manifest.json"),
                "metadata": str(run / "inputs/metadata.csv"),
                "roles": str(run / "artifacts/selection/roles.jsonl"),
                "output_validation": str(validation),
                "output_references": str(reference),
            },
            config=layer_config,
            input_paths=(
                str(model),
                str(run / "inputs/corpus_manifest.json"),
                str(run / "inputs/metadata.csv"),
            ),
            output_dir=str(validation.parent),
            resource_class=resource_class,
        )
        adapter = CommandAdapter(
            command,
            cwd=(
                str(layer_config["cwd"])
                if layer_config.get("cwd")
                else None
            ),
            env={
                str(key): str(value)
                for key, value in dict(layer_config.get("env") or {}).items()
            },
            timeout_seconds=(
                None
                if layer_config.get("timeout_seconds") is None
                else float(layer_config["timeout_seconds"])
            ),
        )
        with ExclusiveResourceLease(run / "locks/heavy.lock", resource_class):
            adapter.run(request)
        if not validation.is_file() or not reference.is_file():
            raise RuntimeError(f"{layer} localizer did not produce contracted outputs")
        validations[layer] = validation
        references[layer] = reference
    if _sha256(references["robust"]) != _sha256(references["dense"]):
        raise RuntimeError("robust and dense localization reference manifests disagree")
    product_references = run / "products/localization_reference/manifest.jsonl"
    product_references.parent.mkdir(parents=True, exist_ok=True)
    product_references.write_bytes(references["robust"].read_bytes())
    return fuse_localization_results(
        run,
        validations["robust"],
        validations["dense"],
        maximum_position_normalized=maximum_position_normalized,
        maximum_rotation_deg=maximum_rotation_deg,
        target_rate=target_rate,
        outer_holdout_frozen=outer_holdout_frozen,
    )



def fuse_localization_results(
    run_dir: str | Path,
    robust_validation: str | Path,
    dense_validation: str | Path,
    *,
    scene_scale: float | None = None,
    maximum_position_normalized: float = 0.02,
    maximum_rotation_deg: float = 2.0,
    target_rate: float = 0.95,
    outer_holdout_frozen: bool = False,
) -> Path:
    """Fuse identical strict query sets without using ground truth to select a layer."""

    run = Path(run_dir).expanduser().resolve(strict=True)
    robust_path = Path(robust_validation).expanduser().resolve(strict=True)
    dense_path = Path(dense_validation).expanduser().resolve(strict=True)
    robust = _read_json(robust_path)
    dense = _read_json(dense_path)
    _require_strict_validation(robust, "robust")
    _require_strict_validation(dense, "dense")
    robust_results = list(robust.get("results") or ())
    dense_results = list(dense.get("results") or ())
    if scene_scale is None:
        scene_scale = map_scene_scale(run / "artifacts/mapping/robust/model")
    fused = combine_result_sets(
        robust_results,
        dense_results,
        scene_scale=scene_scale,
        maximum_position_normalized=maximum_position_normalized,
        maximum_rotation_deg=maximum_rotation_deg,
    )
    successes = sum(bool(row["success"]) for row in fused)
    rate = successes / len(fused)
    decision_counts = Counter(str(row["pose_consistency"]) for row in fused)
    selected_counts = Counter(
        str(row["selected_layer"]) for row in fused if row.get("selected_layer") is not None
    )
    sessions: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in fused:
        grouped[str(row.get("session_id") or "unknown")].append(row)
    for session_id, rows in sorted(grouped.items()):
        count = len(rows)
        accepted = sum(bool(row["success"]) for row in rows)
        sessions[session_id] = {
            "queries": count,
            "successes": accepted,
            "success_rate": accepted / count,
        }
    status = (
        "READY"
        if outer_holdout_frozen and rate >= target_rate
        else "LOCALIZATION_BELOW_TARGET"
        if outer_holdout_frozen
        else "VALIDATED_DEVELOPMENT_ONLY"
    )
    output = run / "artifacts/localization/ensemble/validation.json"
    validation = {
        "schema_version": 1,
        "artifact_type": "STRICT_DUAL_LAYER_LOCALIZATION_ENSEMBLE",
        "status": status,
        "deployment_authorized": status == "READY",
        "strict_loo": True,
        "pseudo_loo": False,
        "map_query_identity_overlap": False,
        "outer_holdout_frozen_before_evaluation": bool(outer_holdout_frozen),
        "query_count": len(fused),
        "strict_successes": successes,
        "strict_success_rate": rate,
        "target_rate": float(target_rate),
        "scene_scale": float(scene_scale),
        "cross_layer_thresholds": {
            "maximum_position_normalized": float(maximum_position_normalized),
            "maximum_rotation_deg": float(maximum_rotation_deg),
        },
        "selection_rule": _FUSION_RULE,
        "sessions": sessions,
        "selected_layer_counts": dict(sorted(selected_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "inputs": {
            "robust_validation": str(robust_path),
            "robust_validation_sha256": _sha256(robust_path),
            "dense_validation": str(dense_path),
            "dense_validation_sha256": _sha256(dense_path),
        },
        "results": fused,
    }
    _write_json_atomic(output, validation)
    receipt = run / "receipts/ensemble_validation.json"
    _write_json_atomic(
        receipt,
        {
            "schema_version": 1,
            "artifact_type": "DUAL_LAYER_LOCALIZATION_VALIDATION_RECEIPT",
            "status": status,
            "query_count": len(fused),
            "strict_successes": successes,
            "strict_success_rate": rate,
            "scene_scale": float(scene_scale),
            "validation": str(output),
            "validation_sha256": _sha256(output),
        },
    )
    manifest_path = run / "products/localization_ensemble/MANIFEST.json"
    manifest = (
        _read_json(manifest_path)
        if manifest_path.is_file()
        else _layer_manifest_from_models(run)
    )
    manifest.update(
        status=status,
        deployment_authorized=status == "READY",
        validation=str(output),
        validation_sha256=_sha256(output),
        strict_success_rate=rate,
        fusion={
            "maximum_position_normalized": float(maximum_position_normalized),
            "maximum_rotation_deg": float(maximum_rotation_deg),
            "rule": _FUSION_RULE,
            "scene_scale": float(scene_scale),
        },
    )
    _write_json_atomic(manifest_path, manifest)
    return receipt


def map_scene_scale(model: str | Path) -> float:
    """Median positive pairwise camera-center distance used by the validated fusion gate."""

    centers = np.asarray(load_gluemap(Path(model)).image_centers, dtype=float)
    if len(centers) < 2:
        raise RuntimeError("at least two registered cameras are required for scene scale")
    from scipy.spatial.distance import pdist

    distances = pdist(centers, metric="euclidean")
    positive = distances[distances > 0]
    if not len(positive):
        raise RuntimeError("registered camera centers have zero scene extent")
    return float(np.median(positive))


def model_hashes(model: str | Path) -> dict[str, str]:
    root = Path(model).resolve(strict=True)
    missing = [name for name in _MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"COLMAP model is incomplete: {missing}")
    return {name: _sha256(root / name) for name in _MODEL_FILES}


def _require_strict_validation(payload: Mapping[str, Any], layer: str) -> None:
    if payload.get("strict_loo") is not True or payload.get("pseudo_loo") is True:
        raise ValueError(f"{layer} validation is not strict mapping-disjoint LOO evidence")
    if payload.get("map_query_identity_overlap") is True:
        raise ValueError(f"{layer} validation contains map/query identity overlap")
    if not payload.get("results"):
        raise ValueError(f"{layer} validation contains no query results")


def _layer_manifest(run: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    dense = receipt["before"]
    robust = receipt["after"]
    dense_p90 = dense.get("point_error_p90_px")
    threshold = receipt["config"]["max_reprojection_error_px"]
    return {
        "schema_version": 1,
        "artifact_type": "DENSE_ROBUST_LOCALIZATION_ENSEMBLE",
        "status": "PENDING_NEW_OUTER_HOLDOUT",
        "deployment_authorized": False,
        "base_geometry": {
            "role": "ROBUST_BASE_GEOMETRY",
            "model": str(run / "artifacts/mapping/robust/model"),
            "registered_images": robust.get("registered_images"),
            "points": robust.get("points3D"),
            "observations": robust.get("observations"),
            "model_hashes": receipt["robust_model_hashes"],
        },
        "dense_layer": {
            "role": "LOCALIZATION_HYPOTHESIS_ONLY",
            "model": str(run / "artifacts/mapping/final/model"),
            "registered_images": dense.get("registered_images"),
            "points": dense.get("points3D"),
            "observations": dense.get("observations"),
            "geometry_screen": (
                "FAILED_REPROJECTION"
                if dense_p90 is not None and float(dense_p90) > float(threshold)
                else "PASSED"
            ),
            "model_hashes": receipt["dense_model_hashes"],
        },
        "candidate_runtime_policies": {
            "full_dual": _FUSION_RULE,
            "robust_first_fallback": (
                "run robust first; invoke dense only after strict failure; validate latency "
                "and yield on the frozen outer holdout"
            ),
        },
        "agreement_gate_to_validate": {
            "maximum_center_difference_scene_scale_fraction": 0.02,
            "maximum_rotation_difference_deg": 2.0,
        },
        "required_next_evidence": (
            "a completely new video frozen before localization inspection and excluded from "
            "mapping, selection, matching, bridge repair, and threshold tuning"
        ),
    }


def _layer_manifest_from_models(run: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "DENSE_ROBUST_LOCALIZATION_ENSEMBLE",
        "base_geometry": {
            "role": "ROBUST_BASE_GEOMETRY",
            "model": str(run / "artifacts/mapping/robust/model"),
            "model_hashes": model_hashes(run / "artifacts/mapping/robust/model"),
        },
        "dense_layer": {
            "role": "LOCALIZATION_HYPOTHESIS_ONLY",
            "model": str(run / "artifacts/mapping/final/model"),
            "model_hashes": model_hashes(run / "artifacts/mapping/final/model"),
        },
    }


def _link_directory(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        raise RuntimeError(f"product link points to a different model: {link}")
    if link.exists():
        raise RuntimeError(f"product path already exists and is not a symlink: {link}")
    link.symlink_to(target.resolve(), target_is_directory=True)


def _export_plys(run: Path, dense_model: Path, robust_model: Path) -> dict[str, str]:
    import pycolmap

    products = run / "products"
    dense_ply = products / "dense_localization_layer.ply"
    robust_ply = products / "robust_base_geometry.ply"
    for path in (dense_ply, robust_ply):
        if path.exists():
            raise FileExistsError(path)
    pycolmap.Reconstruction(str(dense_model)).export_PLY(str(dense_ply))
    pycolmap.Reconstruction(str(robust_model)).export_PLY(str(robust_ply))
    return {
        "dense_ply": str(dense_ply),
        "dense_ply_sha256": _sha256(dense_ply),
        "robust_ply": str(robust_ply),
        "robust_ply_sha256": _sha256(robust_ply),
    }


def _expand_run_paths(value: Any, run: Path) -> Any:
    if isinstance(value, str):
        return value.replace("{run_dir}", str(run))
    if isinstance(value, Mapping):
        return {key: _expand_run_paths(item, run) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_run_paths(item, run) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_run_paths(item, run) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "fuse_localization_results",
    "map_scene_scale",
    "materialize_layers",
    "model_hashes",
    "validate_localization_layers",
]
