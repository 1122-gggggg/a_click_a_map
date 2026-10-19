"""Batch Stage-4 worker for the installed EDM matcher."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .intrinsics import calibration_matrix_for_resolution
from .pair_geometry import PairGeometryResult, verify_pair


@dataclass(frozen=True)
class EDMWorkerConfig:
    repo: str
    checkpoint: str
    model_config: str
    data_config: str
    checkpoint_sha256: str | None = None
    input_size: tuple[int, int] = (640, 480)
    confidence_threshold: float = 0.0
    min_inliers_for_verified: int = 30
    min_inlier_ratio_for_verified: float = 0.1
    min_cheirality_for_verified: float = 0.5
    intrinsics_calibration: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class EDMPairRequest:
    image_i: str
    image_j: str
    output_dir: str
    config: EDMWorkerConfig
    pair_id: str | None = None
    keyframe_i: str | None = None
    keyframe_j: str | None = None
    K_i: tuple[tuple[float, ...], ...] | None = None
    K_j: tuple[tuple[float, ...], ...] | None = None


def parse_config(payload: Mapping[str, Any]) -> EDMWorkerConfig:
    accepted = {item.name for item in fields(EDMWorkerConfig)}
    values = {key: value for key, value in payload.items() if key in accepted}
    missing = [
        key for key in ("repo", "checkpoint", "model_config", "data_config") if not values.get(key)
    ]
    if missing:
        raise ValueError(f"EDM config missing: {missing}")
    size = values.get("input_size", (640, 480))
    if len(size) != 2 or any(int(value) <= 0 for value in size):
        raise ValueError("input_size must contain two positive integers")
    values["input_size"] = (int(size[0]), int(size[1]))
    if values.get("intrinsics_calibration") is not None:
        values["intrinsics_calibration"] = dict(values["intrinsics_calibration"])
    return EDMWorkerConfig(**values)


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    width, height = size
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width), dtype=np.uint8)
    pad_x = (width - resized_width) // 2
    pad_y = (height - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": source_width,
        "original_height": source_height,
    }


def restore_points(points: Any, transform: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    scale = float(transform["scale"])
    if scale <= 0:
        raise ValueError("letterbox scale must be positive")
    values[:, 0] = (values[:, 0] - float(transform["pad_x"])) / scale
    values[:, 1] = (values[:, 1] - float(transform["pad_y"])) / scale
    return values


class _TorchEDMMatcher:
    def __init__(self, model: Any, torch_module: Any) -> None:
        self.model = model
        self.torch = torch_module
        self.version = f"EDM:{type(model).__name__}"

    def match(self, image_i: np.ndarray, image_j: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        first = torch.from_numpy(image_i)[None, None].cuda().float() / 255.0
        second = torch.from_numpy(image_j)[None, None].cuda().float() / 255.0
        batch = {"image0": first, "image1": second}
        with torch.no_grad():
            self.model(batch)
        required = ("mkpts0_f", "mkpts1_f", "mconf")
        if any(key not in batch for key in required):
            raise RuntimeError("EDM runtime did not populate match outputs")
        return {key: batch[key].detach().cpu().numpy() for key in required}


def load_matcher(config: EDMWorkerConfig) -> Any:
    repo = Path(config.repo).expanduser().resolve()
    if not (repo / "src/edm/edm.py").is_file():
        raise FileNotFoundError(repo / "src/edm/edm.py")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import torch
        from src.config.default import get_cfg_defaults
        from src.edm.edm import EDM
        from src.utils.misc import lower_config
    except Exception as error:
        raise RuntimeError(f"unable to import EDM runtime from {repo}") from error
    configuration = get_cfg_defaults()
    configuration.merge_from_file(str(Path(config.model_config).expanduser().resolve()))
    configuration.merge_from_file(str(Path(config.data_config).expanduser().resolve()))
    model = EDM(config=lower_config(configuration)["edm"]).cuda()
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    checkpoint_sha = _sha256(checkpoint_path)
    if config.checkpoint_sha256 and checkpoint_sha != config.checkpoint_sha256:
        raise RuntimeError("EDM checkpoint SHA256 does not match deployment config")
    checkpoint = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        str(checkpoint_path), map_location="cpu", weights_only=True
    )
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return _TorchEDMMatcher(model, torch)


def run_pair(
    request: EDMPairRequest,
    *,
    matcher: Any,
    verifier: Callable[..., PairGeometryResult] = verify_pair,
) -> dict[str, Any]:
    import cv2

    output = Path(request.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(request, str(getattr(matcher, "version", type(matcher).__name__)))
    arrays_path = output / f"{cache_key}.npz"
    metadata_path = output / f"{cache_key}.json"
    if arrays_path.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    original_i = cv2.imread(request.image_i, cv2.IMREAD_GRAYSCALE)
    original_j = cv2.imread(request.image_j, cv2.IMREAD_GRAYSCALE)
    if original_i is None or original_j is None:
        raise ValueError("EDM could not decode one or both pair images")
    image_i, transform_i = letterbox(original_i, request.config.input_size)
    image_j, transform_j = letterbox(original_j, request.config.input_size)
    result = (
        matcher.match(image_i, image_j) if hasattr(matcher, "match") else matcher(image_i, image_j)
    )
    required = ("mkpts0_f", "mkpts1_f", "mconf")
    if any(key not in result for key in required):
        raise RuntimeError("EDM output missing mkpts0_f/mkpts1_f/mconf")
    points_i = restore_points(result["mkpts0_f"], transform_i)
    points_j = restore_points(result["mkpts1_f"], transform_j)
    confidence = np.asarray(result["mconf"], dtype=float).reshape(-1)
    if len(points_i) != len(points_j) or len(points_i) != len(confidence):
        raise RuntimeError("EDM point/confidence lengths differ")
    finite = (
        np.isfinite(points_i).all(axis=1)
        & np.isfinite(points_j).all(axis=1)
        & np.isfinite(confidence)
    )
    in_bounds = (
        (points_i[:, 0] >= 0)
        & (points_i[:, 0] < original_i.shape[1])
        & (points_i[:, 1] >= 0)
        & (points_i[:, 1] < original_i.shape[0])
        & (points_j[:, 0] >= 0)
        & (points_j[:, 0] < original_j.shape[1])
        & (points_j[:, 1] >= 0)
        & (points_j[:, 1] < original_j.shape[0])
    )
    keep = finite & in_bounds & (confidence >= request.config.confidence_threshold)
    points_i, points_j, confidence = points_i[keep], points_j[keep], confidence[keep]
    if len(points_i) < 8:
        geometry = None
        admission = "REJECTED"
        metrics: dict[str, Any] = {
            "raw_matches": int(len(points_i)),
            "degeneracy_flags": ["insufficient_correspondences"],
        }
    else:
        geometry = verifier(
            points_i,
            points_j,
            image_shape_i=original_i.shape[:2],
            image_shape_j=original_j.shape[:2],
            K_i=None if request.K_i is None else np.asarray(request.K_i, dtype=float),
            K_j=None if request.K_j is None else np.asarray(request.K_j, dtype=float),
        )
        metrics = geometry.metadata()
        admission = _admission(geometry, request.config)
    arrays = {"mkpts0_f": points_i, "mkpts1_f": points_j, "mconf": confidence}
    if geometry is not None:
        arrays.update(geometry.arrays())
    np.savez_compressed(arrays_path, **arrays)
    row = {
        "pair_id": request.pair_id or cache_key,
        "image_i": request.keyframe_i or request.image_i,
        "image_j": request.keyframe_j or request.image_j,
        "image_path_i": request.image_i,
        "image_path_j": request.image_j,
        "admission": admission,
        "match_artifact": str(arrays_path),
        "match_artifact_sha256": _sha256(arrays_path),
        "cache_key": cache_key,
        "matcher": str(getattr(matcher, "version", type(matcher).__name__)),
        "checkpoint": request.config.checkpoint,
        "checkpoint_sha256": _checkpoint_identity(request.config),
        "transform_i": transform_i,
        "transform_j": transform_j,
        **metrics,
    }
    metadata_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def run_adapter_request(
    payload: Mapping[str, Any],
    *,
    matcher: Any | None = None,
    verifier: Callable[..., PairGeometryResult] = verify_pair,
) -> dict[str, Any]:
    config = parse_config(dict(payload.get("config") or {}))
    request = dict(payload.get("payload") or {})
    candidate_path = Path(str(request.get("candidate_pairs") or ""))
    keyframe_path = Path(str(request.get("keyframes") or ""))
    output_path = Path(str(request.get("output_geometry") or ""))
    artifact_dir = Path(str(request.get("match_artifact_dir") or output_path.parent / "matches"))
    if not candidate_path.is_file() or not keyframe_path.is_file() or not str(output_path):
        raise ValueError("EDM adapter requires candidate_pairs, keyframes and output_geometry")
    keyframes = {str(row["keyframe_id"]): row for row in _jsonl(keyframe_path)}
    dimensions = _metadata_dimensions(Path(str(request.get("metadata") or "")))
    pair_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _jsonl(candidate_path):
        left, right = sorted((str(row["image_i"]), str(row["image_j"])))
        if left not in keyframes or right not in keyframes:
            raise ValueError("retrieval pair references an unknown keyframe")
        merged = pair_rows.setdefault((left, right), {"categories": [], "retrieval_score": -1.0})
        category = row.get("category")
        if category and category not in merged["categories"]:
            merged["categories"].append(category)
        merged["retrieval_score"] = max(float(row.get("score") or 0.0), merged["retrieval_score"])
    runtime = matcher or load_matcher(config)
    results = []
    for (left, right), retrieval in sorted(pair_rows.items()):
        first, second = keyframes[left], keyframes[right]
        K_i = _keyframe_intrinsics(first, config, dimensions)
        K_j = _keyframe_intrinsics(second, config, dimensions)
        result = run_pair(
            EDMPairRequest(
                image_i=str(first["image_uri"]),
                image_j=str(second["image_uri"]),
                output_dir=str(artifact_dir),
                config=config,
                pair_id=f"{left}|{right}",
                keyframe_i=left,
                keyframe_j=right,
                K_i=K_i,
                K_j=K_j,
            ),
            matcher=runtime,
            verifier=verifier,
        )
        result.update(retrieval)
        results.append(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    return {
        "status": "completed",
        "outputs": [str(output_path)],
        "pair_count": len(results),
        "verified_pairs": sum(row["admission"] == "VERIFIED" for row in results),
    }


def _admission(result: PairGeometryResult, config: EDMWorkerConfig) -> str:
    if result.F is None or result.inliers_F < config.min_inliers_for_verified:
        return "REJECTED"
    if result.inlier_ratio < config.min_inlier_ratio_for_verified:
        return "REJECTED"
    if "homography_dominant" in result.degeneracy_flags:
        return "AMBIGUOUS"
    if (
        result.cheirality_ratio is not None
        and result.cheirality_ratio < config.min_cheirality_for_verified
    ):
        return "AMBIGUOUS"
    return "VERIFIED"


def _cache_key(request: EDMPairRequest, matcher_id: str) -> str:
    material = {
        "image_i": _sha256(Path(request.image_i)),
        "image_j": _sha256(Path(request.image_j)),
        "matcher": matcher_id,
        "config": asdict(request.config),
        "checkpoint_sha256": _checkpoint_identity(request.config),
        "K_i": request.K_i,
        "K_j": request.K_j,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def _matrix_tuple(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("keyframe K must be 3x3")
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _metadata_dimensions(path: Path) -> dict[str, tuple[int, int]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, tuple[int, int]] = {}
    for row in rows:
        width, height = int(row.get("width") or 0), int(row.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        for key in (row.get("video_id"), row.get("source_id")):
            if key:
                result[str(key)] = (width, height)
    return result


def _keyframe_intrinsics(
    keyframe: Mapping[str, Any],
    config: EDMWorkerConfig,
    dimensions: Mapping[str, tuple[int, int]],
) -> tuple[tuple[float, ...], ...] | None:
    embedded = _matrix_tuple(keyframe.get("K"))
    if embedded is not None or config.intrinsics_calibration is None:
        return embedded
    identity = str(keyframe.get("video_id") or keyframe.get("source_id") or "")
    if identity not in dimensions:
        raise ValueError(f"missing image dimensions for calibrated keyframe source: {identity}")
    matrix = calibration_matrix_for_resolution(
        config.intrinsics_calibration,
        target_size=dimensions[identity],
    )
    return _matrix_tuple(matrix)


def _checkpoint_identity(config: EDMWorkerConfig) -> str:
    path = Path(config.checkpoint).expanduser()
    if path.is_file():
        return _sha256(path)
    if config.checkpoint_sha256:
        return config.checkpoint_sha256
    return f"UNAVAILABLE:{path}"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            result = run_adapter_request(json.load(sys.stdin))
    except Exception as error:
        output.write(json.dumps({"status": "error", "error": str(error)}) + "\n")
        return 1
    output.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EDMPairRequest",
    "EDMWorkerConfig",
    "letterbox",
    "load_matcher",
    "parse_config",
    "restore_points",
    "run_adapter_request",
    "run_pair",
]
