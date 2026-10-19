"""Offline MoGe metric calibration and lightweight map-unit to metre output helpers."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from sfm_diagnosis.io import write_json

from .adapters import AdapterRequest, CommandAdapter
from .inventory import sha256_file
from .geometry_contract import calibration_contract, validate_calibration_contract
from .metric_anchors import calibrate_image_records, geometry_anchors, image_scale
from .moge_metric_worker import INFERENCE_SETTINGS, MOGE_MODELS, moge_source_hashes

REQUIRED_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
OPTIONAL_MODEL_FILES = ("frames.bin", "rigs.bin")
RAW_KEYS = {"depth_m", "mask", "K_work", "source_to_work", "request_sha256", "image_sha256"}
_GEOMETRY_PRODUCT_NAMES = ("metric_scale.json", "dense_map.npz", "dense_map.ply", "dense_summary.json",
                           "geometry_receipt.json", "geometry_run.json", "image_scales.jsonl")
NEURAL_SETTINGS = {key: INFERENCE_SETTINGS[key] for key in (
    "resolution_level", "use_fp16", "refine_steps", "return_per_step", "apply_mask", "force_projection")}


def model_file_hashes(model_dir: Path) -> dict[str, str]:
    for filename in REQUIRED_MODEL_FILES:
        if not (model_dir / filename).is_file():
            raise ValueError(f"missing map model file: {model_dir / filename}")
    return {name: sha256_file(model_dir / name) for name in (*REQUIRED_MODEL_FILES, *OPTIONAL_MODEL_FILES)
            if (model_dir / name).is_file()}


def request_digest(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    material = json.dumps({"payload": payload, "config": config}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def load_metric_scale(sidecar: Path, model_dir: Path) -> dict[str, Any]:
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    model_id = data.get("checkpoint_id") if isinstance(data, dict) else None
    spec = MOGE_MODELS.get(model_id) if isinstance(model_id, str) else None
    if (
        spec is None
        or data.get("schema_version") != 2
        or data.get("status") != "ESTIMATED_UNVALIDATED"
        or data.get("method") != spec["method"]
        or data.get("checkpoint_revision") != spec["checkpoint_revision"]
        or data.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or any(data.get(key) != value or type(data.get(key)) is not type(value)
               for key, value in NEURAL_SETTINGS.items())
    ):
        raise ValueError("invalid metric scale sidecar contract")
    geometry = data.get("geometry_validation")
    if not isinstance(geometry, dict):
        raise ValueError("metric scale geometry validation is missing; offline revalidation is required")
    validate_calibration_contract(geometry.get("calibration_contract"))
    scale = data.get("meters_per_map_unit")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("metric scale must be finite and positive")
    hashes = data.get("model_hashes")
    if not isinstance(hashes, dict) or hashes != model_file_hashes(model_dir):
        raise ValueError("metric scale sidecar model hash mismatch")
    if any(not isinstance(data.get(k), str) or len(data[k]) != 64
           or any(c not in "0123456789abcdef" for c in data[k]) for k in
           ("keyframes_sha256", "checkpoint_sha256", "request_sha256", "worker_sha256",
            "request_file_sha256", "raw_depth_manifest_sha256", "geometry_run_sha256")):
        raise ValueError("metric scale source hashes missing")
    source_hashes = data.get("moge_source_hashes")
    if (
        not isinstance(source_hashes, dict)
        or "moge/model/v3.py" not in source_hashes
        or any(not isinstance(value, str) or len(value) != 64
               or any(c not in "0123456789abcdef" for c in value) for value in source_hashes.values())
    ):
        raise ValueError("metric scale MoGe-3 source hashes missing")
    geometry_run = geometry.get("geometry_run")
    if (
        not isinstance(geometry_run, dict)
        or geometry_run.get("schema") != "moge-offline-geometry-run/v1"
        or geometry_run.get("model_hashes") != hashes
        or geometry_run.get("keyframes_sha256") != data["keyframes_sha256"]
        or geometry_run.get("raw_depth_manifest_sha256") != data["raw_depth_manifest_sha256"]
        or geometry_run.get("calibration_contract") != geometry["calibration_contract"]
        or hashlib.sha256(_json_bytes(geometry_run)).hexdigest() != data["geometry_run_sha256"]
    ):
        raise ValueError("metric scale geometry publication binding mismatch")
    provenance = data.get("inference_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema") != "moge-original-inference/v1":
        raise ValueError("metric scale original inference provenance missing")
    origins = provenance.get("requests")
    if not isinstance(origins, dict) or not origins:
        raise ValueError("metric scale original inference requests missing")
    for origin, identity in origins.items():
        if (
            not isinstance(origin, str) or len(origin) != 64 or any(c not in "0123456789abcdef" for c in origin)
            or not isinstance(identity, dict) or identity.get("request_sha256") != origin
            or not isinstance(identity.get("request_path"), str) or not identity["request_path"]
            or not isinstance(identity.get("request_file_sha256"), str)
            or len(identity["request_file_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in identity["request_file_sha256"])
        ):
            raise ValueError("metric scale original request identity mismatch")
        original_config = identity.get("config") or {}
        _validate_pinned_config(original_config)
        if (
            original_config["model_id"] != data["checkpoint_id"]
            or original_config["moge_source_hashes"] != data["moge_source_hashes"]
            or any(original_config.get(key) != data[key] for key in NEURAL_SETTINGS)
        ):
            raise ValueError("metric scale original neural identity mismatch")
    n = data.get("num_registered_images")
    if type(n) is not int or n <= 0 or type(data.get("num_depth_images")) is not int or data["num_depth_images"] != n:
        raise ValueError("metric scale incomplete depth coverage")
    if type(data.get("num_scale_images")) is not int or not 0 < data["num_scale_images"] <= n:
        raise ValueError("metric scale incomplete calibration")
    if not isinstance(data.get("diagnostics"), dict):
        raise ValueError("metric scale diagnostics missing")
    diagnostics = data["diagnostics"]
    ratio = diagnostics.get("session_scale_max_over_min")
    if (
        isinstance(ratio, bool) or not isinstance(ratio, (int, float))
        or not math.isfinite(ratio) or not 1. <= ratio <= 1.20
        or diagnostics.get("session_disagreement") is not False
        or diagnostics.get("absolute_accuracy") != "NO_EXTERNAL_LENGTH_HOLDOUT"
    ):
        raise ValueError("metric scale session consistency or external-truth contract is invalid")
    return data


def add_metric_position_fields(row: dict[str, Any], meters_per_map_unit: float) -> None:
    row["position_m"] = None if row.get("position") is None else (np.asarray(row["position"], dtype=float) * meters_per_map_unit).tolist()
    if "position_raw" in row:
        row["position_raw_m"] = None if row["position_raw"] is None else (np.asarray(row["position_raw"], dtype=float) * meters_per_map_unit).tolist()


def metric_landmarks(cam_from_world: np.ndarray, point_ids: np.ndarray, query_xy: np.ndarray,
                     world_xyz: np.ndarray, *, meters_per_map_unit: float) -> dict[str, list]:
    ids = np.asarray(point_ids)
    xy = np.asarray(query_xy)
    xyz = np.asarray(world_xyz)
    pose = np.asarray(cam_from_world)
    if ids.ndim != 1 or xy.shape != (len(ids), 2) or xyz.shape != (len(ids), 3) or pose.shape != (3, 4):
        raise ValueError("matched landmark arrays must be aligned with a 3x4 camera pose")
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    keep = (ids > 0) & np.isfinite(xy).all(axis=1) & np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    selected = camera[keep]
    return {"point_ids": ids[keep].astype(np.int64).tolist(), "query_xy": xy[keep].tolist(),
            "optical_z_m": (selected[:, 2] * meters_per_map_unit).tolist(),
            "range_m": (np.linalg.norm(selected, axis=1) * meters_per_map_unit).tolist()}


def work_intrinsics(source: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(source, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or width <= 1 or height <= 1 or k[0, 0] <= 0 or k[1, 1] <= 0 or not 0 < k[0, 2] < width - 1 or not 0 < k[1, 2] < height - 1 or not np.allclose(k, [[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]]):
        raise ValueError("expected positive undistorted PINHOLE intrinsics with interior principal point")
    cxw, cyw = (width - 1) / 2, (height - 1) / 2
    focal = max(k[0, 0] * cxw / min(k[0, 2], width - 1 - k[0, 2]),
                k[1, 1] * cyw / min(k[1, 2], height - 1 - k[1, 2]))
    target = np.array([[focal, 0, cxw], [0, focal, cyw], [0, 0, 1]], dtype=np.float64)
    return target, target @ np.linalg.inv(k)


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("p10", "p25", "median", "p75", "p90")}
    return {key: float(np.quantile(values, q)) for key, q in
            (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))}


def estimate_metric_scale(image_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    roles = {"MAPPING_FIT", "MAPPING_BLOCKED_VALIDATION", "MAPPING_OUTER_VALIDATION",
             "MAPPING_RESERVED_CAPTURE", "MAPPING_UNSUPPORTED"}
    for row in image_records:
        if row.get("evaluation_role") != "MAPPING":
            raise ValueError("Metric calibration requires explicitly declared MAPPING sources")
        if row.get("scale_role") not in roles:
            raise ValueError("Metric calibration requires an explicit fitting/diagnostic scale role")
        sessions[str(row["session_id"])].append(row)
    if not sessions:
        raise ValueError("no registered images for metric calibration")

    def observed_log(row: Mapping[str, Any]) -> Any:
        return row["log_scale"] if row["scale_role"] == "MAPPING_FIT" else row.get("diagnostic_log_scale")

    accepted = {
        session: sorted((r for r in records if r.get("status") == "SCALE_OK" and
                         r["scale_role"] != "MAPPING_UNSUPPORTED" and
                         isinstance(r.get("log_scale"), (float, int)) and math.isfinite(r["log_scale"]) and
                         isinstance(observed_log(r), (float, int)) and math.isfinite(observed_log(r))),
                        key=lambda r: (r["frame_index"], r["name"]))
        for session, records in sessions.items()
    }
    unsupported_sessions = sorted(session for session, records in sessions.items()
                                  if all(r.get("status") == "INVALID_CAMERA_POSE" for r in records))
    supported_sessions = sorted(set(sessions) - set(unsupported_sessions))
    missing = [session for session in supported_sessions if not accepted[session]]
    if missing:
        raise ValueError(f"no supported metric scale images in sessions: {missing}")
    if not supported_sessions:
        raise ValueError("no supported metric scale images")
    # Diagnostic observed scales retain real held-out disagreement; deployment
    # scales for those captures are training-only and cannot replace this evidence.
    centers = np.asarray([r["camera_center"] for good in accepted.values() for r in good
                          if r.get("camera_center") is not None], dtype=float)
    cell_width = max(float(np.linalg.norm(np.ptp(centers, axis=0))) * .05, 1e-8) if len(centers) else None
    session_stats: dict[str, Any] = {}
    session_logs: dict[str, float] = {}
    fitted_logs, fit_sessions, even_logs, odd_logs = [], [], [], []
    outer = sorted({session for session, records in sessions.items()
                    if any(r["scale_role"] == "MAPPING_OUTER_VALIDATION" for r in records)})
    if len(outer) > 1:
        raise ValueError("multiple outer metric validation sessions")
    heldout_session = outer[0] if outer else None
    for session in supported_sessions:
        records, good = sessions[session], accepted[session]
        logs = [float(observed_log(r)) for r in good]
        cells: dict[tuple[int, int, int] | tuple[str], list[float]] = defaultdict(list)
        fit_cells: dict[tuple[int, int, int] | tuple[str], list[float]] = defaultdict(list)
        for row in good:
            center = row.get("camera_center")
            key = tuple(np.floor(np.asarray(center, dtype=float) / cell_width).astype(int)) if (
                cell_width and center is not None and np.isfinite(center).all()) else ("unknown",)
            cells[key].append(float(observed_log(row)))
            if row["scale_role"] == "MAPPING_FIT":
                fit_cells[key].append(float(row["log_scale"]))
        session_logs[session] = float(np.median([np.median(values) for values in cells.values()]))
        if fit_cells:
            fitted_logs.append(float(np.median([np.median(values) for values in fit_cells.values()])))
            fit_sessions.append(session)
        ordinals = {r["name"]: i for i, r in enumerate(
            sorted(records, key=lambda r: (r["frame_index"], r["name"])))}
        halves = [[float(observed_log(r)) for r in good if ordinals[r["name"]] % 2 == parity]
                  for parity in (0, 1)]
        if halves[0]:
            even_logs.append(float(np.median(halves[0])))
        if halves[1]:
            odd_logs.append(float(np.median(halves[1])))
        even_s, odd_s = (math.exp(float(np.median(h))) if h else None for h in halves)
        session_stats[session] = {"num_images": len(records), "num_scale_images": len(good),
                                  "num_spatial_cells": len(cells), "log_scale": _distribution(logs),
                                  "scale_m_per_unit": _distribution([math.exp(v) for v in logs]),
                                  "even": {"count": len(halves[0]), "scale": even_s},
                                  "odd": {"count": len(halves[1]), "scale": odd_s},
                                  "odd_over_even": odd_s / even_s if even_s and odd_s else None}
    for session in unsupported_sessions:
        session_stats[session] = {"num_images": len(sessions[session]), "num_scale_images": 0,
                                  "status": "INVALID_CAMERA_POSE"}
    if not fitted_logs:
        raise ValueError("no supported training-only metric scale images")
    all_scales = [math.exp(float(observed_log(r))) for good in accepted.values() for r in good]
    ratio = math.exp(max(session_logs.values()) - min(session_logs.values()))
    diagnostics = {
        "counts": dict(sorted(Counter(r.get("status", "UNKNOWN") for r in image_records).items())),
        "num_images": len(image_records), "num_scale_images": len(all_scales),
        "sessions": session_stats, "image_scale_m_per_unit": _distribution(all_scales),
        "session_scale_max_over_min": ratio, "session_disagreement": ratio > 1.20,
        "heldout_session": heldout_session, "fit_sessions": fit_sessions,
        "unsupported_sessions": unsupported_sessions,
        "absolute_accuracy": "NO_EXTERNAL_LENGTH_HOLDOUT",
        "alternate": {
            "even_supported_sessions": len(even_logs), "odd_supported_sessions": len(odd_logs),
            "even_scale": math.exp(float(np.median(even_logs))) if len(even_logs) == len(supported_sessions) else None,
            "odd_scale": math.exp(float(np.median(odd_logs))) if len(odd_logs) == len(supported_sessions) else None,
        },
    }
    alternate = diagnostics["alternate"]
    alternate["odd_over_even"] = (alternate["odd_scale"] / alternate["even_scale"]
                                   if alternate["odd_scale"] and alternate["even_scale"] else None)
    scale = math.exp(float(np.median(fitted_logs)))
    diagnostics["heldout_over_fitted"] = (
        math.exp(session_logs[heldout_session]) / scale if heldout_session in session_logs else None)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("no finite positive metric scale")
    return {"meters_per_map_unit": scale, "diagnostics": diagnostics,
            "num_scale_images": len(all_scales)}


def _safe_name(name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError(f"unsafe map image name: {name!r}")
    path = Path(name)
    if not name or path.is_absolute() or any(part in (".", "..") for part in path.parts) or str(path) != name:
        raise ValueError(f"unsafe map image name: {name!r}")
    return path


def _load_raw(path: Path, row: Mapping[str, Any], digest: str) -> tuple[np.ndarray, np.ndarray]:
    encoded = path.read_bytes()
    if row.get("raw_depth_sha256") is not None and hashlib.sha256(encoded).hexdigest() != row["raw_depth_sha256"]:
        raise ValueError(f"metric depth content hash mismatch: {path}")
    with np.load(io.BytesIO(encoded), allow_pickle=False) as data:
        if set(data.files) != RAW_KEYS or len(data.files) != len(RAW_KEYS):
            raise ValueError(f"incomplete metric depth cache: {path}")
        depth, mask = data["depth_m"], data["mask"]
        expected_k, expected_h = work_intrinsics(np.asarray(row["K_source"]), row["width"], row["height"])
        if (
            depth.shape != (row["height"], row["width"]) or depth.dtype != np.float32
            or mask.shape != depth.shape or mask.dtype != bool
            or data["K_work"].shape != (3, 3) or data["K_work"].dtype != np.float64
            or data["source_to_work"].shape != (3, 3) or data["source_to_work"].dtype != np.float64
            or not np.allclose(data["K_work"], expected_k, atol=1e-6, rtol=0)
            or not np.allclose(data["source_to_work"], expected_h, atol=1e-8, rtol=0)
            or data["request_sha256"].shape != () or str(data["request_sha256"]) != digest
            or data["image_sha256"].shape != () or str(data["image_sha256"]) != row["image_sha256"]
            or (row.get("raw_request_sha256") is not None and row["raw_request_sha256"] != digest)
            or np.any(mask & (~np.isfinite(depth) | (depth <= 0)))
            or np.any(~mask & ~np.isnan(depth))
        ):
            raise ValueError(f"incompatible metric depth cache: {path}")
        return depth, mask


def _json_bytes(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, encoded: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def _validate_pinned_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_id = config.get("model_id")
    if not isinstance(model_id, str) or model_id not in MOGE_MODELS:
        raise ValueError("Baseline request has no supported explicit MoGe-3 model identity")
    spec = MOGE_MODELS[model_id]
    for key in ("checkpoint_revision", "checkpoint_sha256"):
        if config.get(key) != spec[key]:
            raise ValueError(f"Baseline request pinned {key} mismatch")
    if any(config.get(key) != value or type(config.get(key)) is not type(value)
           for key, value in NEURAL_SETTINGS.items()):
        raise ValueError("Baseline neural settings are incompatible with full native MoGe-3 inference")
    original_map = config.get("model_hashes")
    if not isinstance(original_map, dict) or not set(REQUIRED_MODEL_FILES) <= set(original_map):
        raise ValueError("Baseline original map identity is incomplete")
    hashes = config.get("moge_source_hashes")
    if not isinstance(hashes, dict) or "moge/model/v3.py" not in hashes:
        raise ValueError("Baseline MoGe source identities are missing")
    for value in [config.get("keyframes_sha256"), config.get("worker_sha256"),
                  *original_map.values(), *hashes.values()]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Baseline source hashes are invalid")
    return spec


def _read_request(path: Path, *, semantic_sha256: str | None = None,
                  file_sha256: str | None = None) -> tuple[dict[str, Any], bytes]:
    encoded = path.read_bytes()
    if file_sha256 is not None and hashlib.sha256(encoded).hexdigest() != file_sha256:
        raise ValueError("Baseline request file SHA256 mismatch")
    request = json.loads(encoded)
    if not isinstance(request, dict) or request.get("stage") != "moge_metric_depth":
        raise ValueError("Invalid offline depth request")
    payload, config = request.get("payload"), request.get("config")
    if not isinstance(payload, dict) or not isinstance(config, dict):
        raise ValueError("Incomplete offline depth request")
    digest = request_digest(payload, config)
    if request.get("request_sha256") != digest or (semantic_sha256 is not None and semantic_sha256 != digest):
        raise ValueError("Baseline semantic request SHA256 mismatch")
    _validate_pinned_config(config)
    return request, encoded


def _mapping_keyframe_entries(keyframes: Path) -> tuple[dict[str, Any], str]:
    """Only declared MAPPING sources may contribute depth or calibration evidence."""
    encoded = keyframes.read_bytes()
    entries: dict[str, Any] = {}
    for line in encoded.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or item.get("evaluation_role") != "MAPPING":
            raise ValueError("Depth calibration requires an explicit MAPPING-only registered manifest")
        name = item.get("output_name")
        _safe_name(name)
        if name in entries:
            raise ValueError(f"Duplicate keyframe: {name}")
        entries[name] = item
    if not entries:
        raise ValueError("Empty MAPPING keyframe manifest")
    return entries, hashlib.sha256(encoded).hexdigest()


def _collect_geometry_inputs(map_model: Path, keyframes: Path) -> tuple[Any, list[dict[str, Any]], dict[str, str], str]:
    # Role validation precedes even reconstruction loading, not just model launch.
    entries, keyframes_hash = _mapping_keyframe_entries(keyframes)
    import pycolmap

    rec = pycolmap.Reconstruction(str(map_model))
    if not rec.images or rec.num_reg_images() != len(rec.images):
        raise ValueError("Empty or non-registered map image set")
    if set(entries) != {im.name for im in rec.images.values()}:
        raise ValueError("Registered map image identities differ from keyframes manifest")
    images_root = (keyframes.parent / "images").resolve(strict=True)
    hashes = model_file_hashes(map_model)
    images = []
    for im in sorted(rec.images.values(), key=lambda image: image.name):
        item, cam = entries[im.name], rec.cameras[im.camera_id]
        if str(cam.model).split(".")[-1] != "PINHOLE":
            raise ValueError(f"Not an undistorted PINHOLE camera: {im.name}")
        path = Path(item["image_uri"]).resolve(strict=True)
        if not path.is_relative_to(images_root) or path != (images_root / _safe_name(im.name)).resolve(strict=True):
            raise ValueError(f"Keyframe outside expected image root: {im.name}")
        k = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        work_intrinsics(k, cam.width, cam.height)
        probe = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if probe is None or probe.shape[:2] != (cam.height, cam.width):
            raise ValueError(f"Keyframe unreadable or dimensions differ from camera: {path}")
        digest = sha256_file(path)
        if digest != item["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        if not item.get("session_id"):
            raise ValueError(f"Missing session_id: {im.name}")
        images.append({
            "name": im.name, "image_path": str(path), "image_sha256": digest,
            "session_id": item["session_id"], "frame_index": int(item["frame_index"]),
            "width": cam.width, "height": cam.height, "K_source": k.tolist(),
        })
    return rec, images, hashes, keyframes_hash


def _clear_geometry_products(output_dir: Path) -> None:
    for name in _GEOMETRY_PRODUCT_NAMES:
        (output_dir / name).unlink(missing_ok=True)


def _publish_geometry(
    *, reconstruction: Any, images: list[dict[str, Any]], anchors: Any,
    map_model: Path, keyframes: Path, hashes: Mapping[str, str], keyframes_hash: str,
    output_dir: Path, raw_depth_dir: Path, request: Mapping[str, Any], dense: bool,
    assert_neural_inputs: Any = None, photo_evidence_path: Path | None = None,
    mvs_evidence_path: Path | None = None, zero_forward: bool = False,
    original_request_path: Path | None = None,
) -> dict[str, Any]:
    digest, config = request["request_sha256"], request["config"]
    spec = _validate_pinned_config(config)
    request_path = output_dir / "request.json"
    _, request_bytes = _read_request(request_path, semantic_sha256=digest)
    request_file_sha = hashlib.sha256(request_bytes).hexdigest()
    origin_request_path = request_path if original_request_path is None else original_request_path
    _read_request(origin_request_path, semantic_sha256=digest, file_sha256=request_file_sha)
    contract = calibration_contract()
    evidence_hashes = {
        "photo": None if photo_evidence_path is None else sha256_file(photo_evidence_path),
        "mvs": None if mvs_evidence_path is None else sha256_file(mvs_evidence_path),
    }
    mapping_entries, mapping_sha = _mapping_keyframe_entries(keyframes)
    if mapping_sha != keyframes_hash or set(mapping_entries) != {image["name"] for image in images}:
        raise ValueError("MAPPING source identities changed before geometry calibration")
    images = [dict(image, evaluation_role=mapping_entries[image["name"]]["evaluation_role"]) for image in images]
    image_by_name = {image.name: image for image in reconstruction.images.values()}
    sessions = sorted({str(row["session_id"]) for row in images})
    correction_validation_sessions = frozenset(sessions[-2:])
    rows = []
    for image in images:
        path = Path(image.get("raw_depth_path") or raw_depth_dir / (image["name"] + ".npz"))
        expected_sha = image.get("raw_depth_sha256") or sha256_file(path)
        image.update(raw_depth_path=str(path), raw_depth_sha256=expected_sha, raw_request_sha256=digest)
        depth, mask = _load_raw(path, image, digest)
        _, transform = work_intrinsics(np.asarray(image["K_source"]), image["width"], image["height"])
        rows.append(image_scale(
            image_by_name[image["name"]], reconstruction, image, depth, mask, anchors, transform,
            fit_enabled=str(image["session_id"]) not in correction_validation_sessions))
    rows = calibrate_image_records(rows)
    if {row["name"] for row in rows} != set(image_by_name):
        raise ValueError("Incomplete registered image depth identity set")
    _atomic_bytes(output_dir / "image_scales.jsonl",
                  ("\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n").encode("utf-8"))
    estimate = estimate_metric_scale(rows)

    def check_integrity() -> None:
        if model_file_hashes(map_model) != hashes or sha256_file(keyframes) != keyframes_hash:
            raise ValueError("Map or keyframes changed during offline geometry validation")
        for image in images:
            if sha256_file(Path(image["image_path"])) != image["image_sha256"]:
                raise ValueError(f"Mapping image changed during geometry validation: {image['name']}")
            if sha256_file(Path(image["raw_depth_path"])) != image["raw_depth_sha256"]:
                raise ValueError(f"Original raw depth changed during geometry validation: {image['name']}")
        for kind, path in (("photo", photo_evidence_path), ("mvs", mvs_evidence_path)):
            if path is not None and sha256_file(path) != evidence_hashes[kind]:
                raise ValueError("Independent geometry evidence changed during calibration")
        validate_calibration_contract(contract)
        if sha256_file(request_path) != request_file_sha:
            raise ValueError("Original inference request changed during geometry validation")
        if sha256_file(origin_request_path) != request_file_sha:
            raise ValueError("Original producer request changed during CPU geometry revalidation")
        if assert_neural_inputs is not None:
            assert_neural_inputs()

    check_integrity()
    origins = {digest: {
        "request_sha256": digest, "request_path": str(origin_request_path),
        "request_file_sha256": request_file_sha, "config": config,
        "geometry_request_copy_path": str(request_path),
    }}
    raw_manifest = {
        "schema": "moge-immutable-raw-depth/v1", "request_sha256": digest,
        "images": images, "inference_provenance": origins,
    }
    manifest_bytes = _json_bytes(raw_manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    _atomic_bytes(output_dir / "raw_depth_manifest.json", manifest_bytes)
    geometry_run = {
        "schema": "moge-offline-geometry-run/v1", "model_hashes": dict(hashes),
        "keyframes_sha256": keyframes_hash, "raw_depth_manifest_sha256": manifest_sha,
        "calibration_contract": contract, "photo_evidence_sha256": evidence_hashes["photo"],
        "mvs_evidence_sha256": evidence_hashes["mvs"],
    }
    geometry_bytes = _json_bytes(geometry_run)
    geometry_digest = hashlib.sha256(geometry_bytes).hexdigest()
    _atomic_bytes(output_dir / "geometry_run.json", geometry_bytes)
    sidecar = {
        "schema_version": 2,
        "geometry_validation": {
            "calibration_contract": contract,
            "anchor_validator_sha256": sha256_file(Path(__file__).with_name("metric_anchors.py")),
            "scale_validator_sha256": sha256_file(Path(__file__)), "geometry_run": geometry_run,
        },
        "status": ("SCALE_DISAGREEMENT_UNVALIDATED" if estimate["diagnostics"]["session_disagreement"]
                   else "ESTIMATED_UNVALIDATED"),
        "method": spec["method"], "meters_per_map_unit": estimate["meters_per_map_unit"],
        "model_hashes": dict(hashes), "keyframes_sha256": keyframes_hash,
        "checkpoint_sha256": config["checkpoint_sha256"], "checkpoint_id": config["model_id"],
        "checkpoint_revision": config["checkpoint_revision"], "moge_source_hashes": config["moge_source_hashes"],
        "worker_sha256": config["worker_sha256"], **{key: config[key] for key in INFERENCE_SETTINGS if key in config},
        "request_sha256": digest, "request_file_sha256": request_file_sha,
        "raw_depth_manifest_sha256": manifest_sha, "geometry_run_sha256": geometry_digest,
        "inference_provenance": {"schema": "moge-original-inference/v1", "requests": origins},
        "num_registered_images": len(images), "num_depth_images": len(rows),
        "num_scale_images": estimate["num_scale_images"], "diagnostics": estimate["diagnostics"],
    }
    sidecar_path = output_dir / "metric_scale.json"
    _atomic_json(sidecar_path, sidecar)
    dense_result = None
    try:
        if dense:
            if sidecar["status"] != "ESTIMATED_UNVALIDATED":
                raise ValueError("Session scale groups disagree; refusing metre-valued dense export")
            from .metric_dense import build_dense_map

            dense_result = build_dense_map(
                reconstruction=reconstruction, image_rows=images, image_scales=rows,
                raw_depth_dir=raw_depth_dir, output_dir=output_dir,
                request_sha256=digest, meters_per_map_unit=estimate["meters_per_map_unit"],
                model_hashes=hashes, metric_scale_sha256=sha256_file(sidecar_path),
                scale_status=sidecar["status"],
                session_disagreement=estimate["diagnostics"]["session_scale_max_over_min"] - 1.,
                geometry_anchor_ids=anchors, calibration_contract_data=contract,
                geometry_run_sha256=geometry_digest, raw_depth_manifest_sha256=manifest_sha,
                request_file_sha256=request_file_sha, inference_provenance=sidecar["inference_provenance"],
                keyframes_sha256=keyframes_hash, photo_evidence_path=photo_evidence_path,
                mvs_evidence_path=mvs_evidence_path, photo_evidence_sha256=evidence_hashes["photo"],
                mvs_evidence_sha256=evidence_hashes["mvs"],
            )
        check_integrity()
    except Exception:
        if sidecar["status"] == "ESTIMATED_UNVALIDATED":
            _clear_geometry_products(output_dir)
        raise
    _atomic_json(output_dir / "geometry_receipt.json", {
        "schema": "offline-depth-geometry-revalidation/v1", "status": "completed",
        "geometry_run_sha256": geometry_digest, "request_sha256": digest,
        "mode": "CPU_REVALIDATION" if zero_forward else "OFFLINE_INFERENCE_CALIBRATION",
        "model_forward_count": 0 if zero_forward else None,
        "registered_mapping_images": len(images), "inference_provenance_preserved": True,
    })
    return sidecar if not dense else {"metric_scale": sidecar, "dense_map": dense_result}


def run_metric_scale(
    *, map_model: Path, keyframes: Path, output_dir: Path, moge_root: Path,
    moge_python: Path, checkpoint: Path, model_id: str, resume: bool = False, dense: bool = False,
    photo_evidence_path: Path | None = None, mvs_evidence_path: Path | None = None,
) -> dict[str, Any]:
    map_model, keyframes, output_dir = map_model.resolve(), keyframes.resolve(), output_dir.resolve()
    if any((output_dir / name).exists() for name in _GEOMETRY_PRODUCT_NAMES):
        raise ValueError("Existing geometry products are immutable; use CPU revalidation in an independent directory")
    rec, images, hashes, keyframes_hash = _collect_geometry_inputs(map_model, keyframes)
    checkpoint, moge_root = checkpoint.resolve(strict=True), moge_root.resolve(strict=True)
    moge_python = moge_python.absolute()
    if not moge_python.is_file():
        raise ValueError(f"Missing MoGe interpreter: {moge_python}")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ValueError(f"Metric output directory is not empty: {output_dir}")
    if not isinstance(model_id, str) or model_id not in MOGE_MODELS:
        raise ValueError("Unsupported explicit MoGe-3 model identity")
    spec = MOGE_MODELS[model_id]
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != spec["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint does not match pinned {model_id}")
    source_hashes = moge_source_hashes(moge_root)
    worker = Path(__file__).with_name("moge_metric_worker.py").resolve(strict=True)
    worker_hash = sha256_file(worker)
    config = {
        "moge_root": str(moge_root), "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
        "model_id": model_id, "checkpoint_revision": spec["checkpoint_revision"], "model_hashes": hashes,
        "moge_source_hashes": source_hashes, "worker_sha256": worker_hash,
        "keyframes_sha256": keyframes_hash, **INFERENCE_SETTINGS,
    }
    anchors = geometry_anchors(rec, mapping_image_ids=frozenset(rec.images))
    if not anchors:
        raise ValueError("No reliable parallax-supported GlueMap anchors")
    payload = {"images": images}
    digest = request_digest(payload, config)
    request = {"stage": "moge_metric_depth", "resource_class": "gpu", "payload": payload,
               "config": config, "request_sha256": digest}
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    if resume and request_path.exists():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous != request:
            raise ValueError("Metric depth request/source changed; cannot resume")
    elif resume and any(output_dir.iterdir()):
        raise ValueError("Cannot resume metric output without matching request.json")
    else:
        _atomic_json(request_path, request)
    _clear_geometry_products(output_dir)
    raw_depth_dir = output_dir / "raw_depth"
    for image in images:
        existing = raw_depth_dir / (image["name"] + ".npz")
        if existing.is_file():
            _load_raw(existing, image, digest)
    if not all((raw_depth_dir / (image["name"] + ".npz")).is_file() for image in images):
        adapter = CommandAdapter([str(moge_python), str(worker)], timeout_seconds=None)
        receipt = adapter.run(AdapterRequest(
            stage="moge_metric_depth", resource_class="gpu", payload=payload,
            config=config, output_dir=str(output_dir)))
        _atomic_json(output_dir / "worker_receipt.json", receipt.output)

    def check_neural_inputs() -> None:
        if (sha256_file(checkpoint) != checkpoint_hash or moge_source_hashes(moge_root) != source_hashes
                or sha256_file(worker) != worker_hash):
            raise ValueError("MoGe checkpoint or implementation changed during depth inference")

    return _publish_geometry(
        reconstruction=rec, images=images, anchors=anchors, map_model=map_model,
        keyframes=keyframes, hashes=hashes, keyframes_hash=keyframes_hash, raw_depth_dir=raw_depth_dir,
        output_dir=output_dir, request=request, dense=dense, assert_neural_inputs=check_neural_inputs,
        photo_evidence_path=photo_evidence_path, mvs_evidence_path=mvs_evidence_path)


def revalidate_metric_geometry(
    *, map_model: Path, keyframes: Path, output_dir: Path, baseline_request: Path,
    baseline_request_sha256: str, baseline_request_file_sha256: str,
    baseline_raw_depth_dir: Path, baseline_raw_hashes: Mapping[str, str],
    dense: bool = False, photo_evidence_path: Path | None = None,
    mvs_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """CPU-only accuracy revalidation of frozen original six-key depth archives."""
    for digest in (baseline_request_sha256, baseline_request_file_sha256):
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Both semantic and file-byte baseline request SHA256 bindings are required")
    if not isinstance(baseline_raw_hashes, Mapping):
        raise ValueError("A complete name-to-SHA256 baseline raw hash mapping is required")
    map_model, keyframes, output_dir = map_model.resolve(), keyframes.resolve(), output_dir.resolve()
    # Manifest roles are enforced before raw import or calibration can begin.
    rec, images, hashes, keyframes_hash = _collect_geometry_inputs(map_model, keyframes)
    baseline_request = baseline_request.resolve(strict=True)
    baseline_raw_depth_dir = baseline_raw_depth_dir.resolve(strict=True)
    if output_dir.is_relative_to(baseline_request.parent) or output_dir.is_relative_to(baseline_raw_depth_dir):
        raise ValueError("Revalidated geometry must use an independent output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Revalidation output directory must be empty")
    request, original_bytes = _read_request(
        baseline_request, semantic_sha256=baseline_request_sha256, file_sha256=baseline_request_file_sha256)
    config = request["config"]
    if config["model_hashes"] != hashes or config["keyframes_sha256"] != keyframes_hash:
        raise ValueError("Revalidation requires the exact frozen original map and MAPPING manifest")
    original_rows = request["payload"].get("images")
    if not isinstance(original_rows, list) or not original_rows:
        raise ValueError("Baseline depth request is partial")
    originals = {}
    for original in original_rows:
        name = original.get("name")
        _safe_name(name)
        if name in originals:
            raise ValueError("Duplicate baseline raw image identity")
        originals[name] = original
    names = {image["name"] for image in images}
    if set(originals) != names or set(baseline_raw_hashes) != names:
        raise ValueError("Baseline request/raw hash coverage is partial or stale")
    for image in images:
        original = originals[image["name"]]
        if (
            any(original.get(key) != image[key] for key in ("image_sha256", "width", "height", "session_id", "frame_index"))
            or not np.array_equal(np.asarray(original.get("K_source"), dtype=np.float64),
                                  np.asarray(image["K_source"], dtype=np.float64))
        ):
            raise ValueError(f"Baseline has stale pixels/intrinsics/source grouping: {image['name']}")
        expected_sha = baseline_raw_hashes[image["name"]]
        if not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError("Invalid original raw depth content hash")
        path = (baseline_raw_depth_dir / (_safe_name(image["name"]).as_posix() + ".npz")).resolve(strict=True)
        if not path.is_relative_to(baseline_raw_depth_dir):
            raise ValueError("Baseline raw reference escapes its original depth root")
        image.update(raw_depth_path=str(path), raw_depth_sha256=expected_sha,
                     raw_request_sha256=baseline_request_sha256)
    anchors = geometry_anchors(rec, mapping_image_ids=frozenset(rec.images))
    if not anchors:
        raise ValueError("No reliable parallax-supported GlueMap anchors")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output_dir / "request.json", original_bytes)
    return _publish_geometry(
        reconstruction=rec, images=images, anchors=anchors, map_model=map_model,
        keyframes=keyframes, hashes=hashes, keyframes_hash=keyframes_hash,
        raw_depth_dir=baseline_raw_depth_dir, output_dir=output_dir, request=request,
        dense=dense, zero_forward=True, photo_evidence_path=photo_evidence_path,
        mvs_evidence_path=mvs_evidence_path, original_request_path=baseline_request)
