#!/usr/bin/env python3
"""Standalone GPU worker for pinned MoGe-3 Large/Giant metric depth estimation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
from pathlib import Path
import sys
import time
from typing import Any, Mapping
import cv2
import numpy as np

RAW_KEYS = {"depth_m", "mask", "K_work", "source_to_work", "request_sha256", "image_sha256"}
MOGE_MODELS = {
    "Ruicheng/moge-3-vitl": {
        "checkpoint_revision": "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be",
        "checkpoint_sha256": "9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925",
        "backbone": "dinov2_vitl14",
        "embed_dim": 1024,
        "intermediate_layers": [5, 11, 17, 23],
        "depth": 24,
        "num_heads": 16,
        "method": "MOGE3_VITL_GEOMETRY_TRACK_LOG_RATIO",
    },
    "Ruicheng/moge-3-vitg": {
        "checkpoint_revision": "6ef26c5a4b4148dab5ccaacdb08b72dc66380475",
        "checkpoint_sha256": "ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c",
        "backbone": "dinov2_vitg14",
        "embed_dim": 1536,
        "intermediate_layers": [9, 19, 29, 39],
        "depth": 40,
        "num_heads": 24,
        "method": "MOGE3_VITG_GEOMETRY_TRACK_LOG_RATIO",
    },
}
INFERENCE_SETTINGS = {
    "resolution_level": 9,
    "use_fp16": True,
    "refine_steps": 3,
    "return_per_step": False,
    "apply_mask": True,
    "force_projection": True,
    "torch_num_threads": 8,
    "torch_num_interop_threads": 1,
    "cuda_memory_fraction": 0.93,
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def moge_source_hashes(moge_root: Path) -> dict[str, str]:
    if not (moge_root / "moge" / "model" / "v3.py").is_file():
        raise ValueError(f"Missing official MoGe-3 implementation under {moge_root}")
    return {
        path.relative_to(moge_root).as_posix(): sha256_file(path)
        for path in sorted((moge_root / "moge").rglob("*.py"))
        if path.is_file()
    }


def request_digest(payload: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    material = json.dumps(
        {"payload": payload, "config": config},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def safe_image_name(name: str, raw_depth_dir: Path) -> Path:
    if not name:
        raise ValueError("Image name cannot be empty")
    rel_path = Path(name)
    if rel_path.is_absolute() or any(part in (".", "..") for part in rel_path.parts) or str(rel_path) != name:
        raise ValueError(f"Unsafe map image name: {name!r}")
    target_path = (raw_depth_dir / (str(rel_path) + ".npz")).resolve()
    if not target_path.is_relative_to(raw_depth_dir) or target_path == raw_depth_dir:
        raise ValueError(f"Path traversal detected in image name: {name!r}")
    return target_path


def work_intrinsics(source: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, float]:
    k = np.asarray(source, dtype=np.float64)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or width <= 1
        or height <= 1
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not (0.0 < k[0, 2] < width - 1)
        or not (0.0 < k[1, 2] < height - 1)
        or not np.allclose(k, [[k[0, 0], 0, k[0, 2]], [0, k[1, 1], k[1, 2]], [0, 0, 1]])
    ):
        raise ValueError(f"Expected positive undistorted PINHOLE intrinsics with interior principal point, got:\n{k}")
    cxw = (width - 1) / 2.0
    cyw = (height - 1) / 2.0
    focal = max(
        k[0, 0] * cxw / min(k[0, 2], width - 1 - k[0, 2]),
        k[1, 1] * cyw / min(k[1, 2], height - 1 - k[1, 2]),
    )
    target = np.array([[focal, 0.0, cxw], [0.0, focal, cyw], [0.0, 0.0, 1.0]], dtype=np.float64)
    h_matrix = target @ np.linalg.inv(k)
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * focal)))
    return target, h_matrix, fov_x


def validate_cached_npz(
    path: Path,
    expected_row: Mapping[str, Any],
    expected_digest: str,
    expected_k: np.ndarray,
    expected_h: np.ndarray,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != RAW_KEYS:
                raise ValueError(f"Incomplete metric depth cache: {path}")
            depth = data["depth_m"]
            mask = data["mask"]
            k_work = data["K_work"]
            source_to_work = data["source_to_work"]
            req_sha = str(data["request_sha256"])
            img_sha = str(data["image_sha256"])

            width = int(expected_row["width"])
            height = int(expected_row["height"])
            if (
                depth.shape != (height, width)
                or depth.dtype != np.float32
                or mask.shape != (height, width)
                or mask.dtype != bool
                or k_work.dtype != np.float64
                or source_to_work.dtype != np.float64
                or not np.allclose(k_work, expected_k, atol=1e-6)
                or not np.allclose(source_to_work, expected_h, atol=1e-8)
                or req_sha != expected_digest
                or img_sha != expected_row["image_sha256"]
                or np.any(mask & (~np.isfinite(depth) | (depth <= 0)))
                or np.any(~mask & ~np.isnan(depth))
            ):
                raise ValueError(f"Incompatible metric depth cache: {path}")
            return True
    except Exception as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError(f"Failed to read cached NPZ {path}: {error}") from error


def load_moge_model(moge_root: Path, checkpoint_path: Path, device: Any, model_id: str) -> Any:
    import torch

    spec = MOGE_MODELS[model_id]
    moge_root_str = str(moge_root.resolve())
    if moge_root_str not in sys.path:
        sys.path.insert(0, moge_root_str)
    try:
        from moge.model.v3 import MoGeModel
    except ImportError as error:
        raise RuntimeError(
            "MoGe-3 requires its real Sparse3DUNet dependencies: "
            "flex-gemm@b2fadb29d41846c7981ade6801ffc689fae119cf and "
            "utils3d_moge@62f09d58509485564e24d5d9f6aac9ee9ebc0c37; "
            f"cannot import the complete model: {error}"
        ) from error
    if Path(sys.modules[MoGeModel.__module__].__file__).resolve() != (
        moge_root / "moge" / "model" / "v3.py"
    ).resolve():
        raise RuntimeError("Imported MoGe-3 implementation does not match the pinned source root")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(ckpt, Mapping) or "model_config" not in ckpt or "model" not in ckpt:
        raise ValueError(f"Invalid checkpoint structure in {checkpoint_path}")
    model_config = ckpt["model_config"]
    weights = ckpt["model"]
    encoder_cfg = model_config.get("encoder", {})
    if (
        encoder_cfg.get("backbone") != spec["backbone"]
        or encoder_cfg.get("dim_out") != spec["embed_dim"]
        or encoder_cfg.get("intermediate_layers") != spec["intermediate_layers"]
    ):
        raise ValueError(f"Checkpoint architecture does not match explicit model_id {model_id!r}")
    if model_config.get("scale_head", {}).get("dims") != [
        spec["embed_dim"], spec["embed_dim"], spec["embed_dim"], 1
    ] or not any(name.startswith("scale_head.") for name in weights):
        raise ValueError(f"MoGe-3 checkpoint has no matching trained metric scale head: {checkpoint_path}")
    refiner_cfg = {
        "encoder_channels": spec["embed_dim"] + 2,
        "model_channels": [32, 64, 128, 256, 512],
        "encoder_blocks_per_level": 1,
        "decoder_blocks_per_level": 1,
        "bottleneck_blocks": 1,
        "downsample_factors": [2, 2, 2, 2],
        "encoder_downsample": 16,
        "in_channels": 3,
        "out_channels": 1,
    }
    if (
        model_config.get("refiner") != refiner_cfg
        or model_config.get("refiner_depth_resolution") != 256
        or not any(name.startswith("refiner.") for name in weights)
    ):
        raise ValueError(f"MoGe-3 checkpoint has no matching trained sparse refiner: {checkpoint_path}")
    if model_config.get("num_tokens_range") != [1200, 3600]:
        raise ValueError(f"Unexpected MoGe-3 inference token range: {checkpoint_path}")

    # Allocate parameters directly on the capped GPU, not a duplicate CPU model.
    with torch.device(device):
        model = MoGeModel(**model_config)
    backbone = model.encoder.backbone
    if (
        backbone.embed_dim != spec["embed_dim"]
        or backbone.n_blocks != spec["depth"]
        or backbone.num_heads != spec["num_heads"]
        or backbone.patch_size != 14
        or not hasattr(model, "scale_head")
        or not hasattr(model, "refiner")
    ):
        raise ValueError(f"Instantiated MoGe-3 architecture does not match {model_id!r}")
    model.load_state_dict(weights, strict=True)
    del weights, ckpt
    return model.eval()


def main() -> None:
    import torch

    torch.set_num_threads(INFERENCE_SETTINGS["torch_num_threads"])
    torch.set_num_interop_threads(INFERENCE_SETTINGS["torch_num_interop_threads"])
    cv2.setNumThreads(INFERENCE_SETTINGS["torch_num_threads"])
    started_at = time.perf_counter()
    parser = argparse.ArgumentParser(description="Pinned MoGe-3 metric depth worker")
    parser.add_argument("--request", type=str, default=None, help="Path to request JSON file")
    args = parser.parse_args()

    if args.request:
        with open(args.request, "r", encoding="utf-8") as f:
            request_data = json.load(f)
    else:
        request_data = json.load(sys.stdin)

    stage = request_data.get("stage", "moge_metric_depth")
    if stage != "moge_metric_depth":
        raise ValueError(f"Unexpected stage {stage!r}, expected 'moge_metric_depth'")

    output_dir_str = request_data.get("output_dir")
    if not output_dir_str:
        raise ValueError("output_dir must be specified in request")
    output_dir = Path(output_dir_str).resolve()
    raw_depth_dir = (output_dir / "raw_depth").resolve()
    raw_depth_dir.mkdir(parents=True, exist_ok=True)

    payload = request_data.get("payload") or {}
    images = payload.get("images") or []
    config = request_data.get("config") or {}
    if not images:
        raise ValueError("Payload contains no images")

    digest = request_digest(payload, config)
    if request_data.get("request_sha256") and request_data["request_sha256"] != digest:
        raise ValueError(
            f"Request SHA256 mismatch: provided {request_data['request_sha256']!r} != computed {digest!r}"
        )
    model_id = config.get("model_id")
    if model_id not in MOGE_MODELS:
        raise ValueError(f"Expected an explicit pinned MoGe-3 model_id, got {model_id!r}")
    spec = MOGE_MODELS[model_id]
    for key in ("checkpoint_revision", "checkpoint_sha256"):
        if config.get(key) != spec[key]:
            raise ValueError(f"MoGe-3 {key} does not match official pinned {model_id}")
    for key, value in INFERENCE_SETTINGS.items():
        if config.get(key) != value:
            raise ValueError(f"MoGe-3 requires {key}={value!r}, got {config.get(key)!r}")

    moge_root = Path(config["moge_root"]).resolve(strict=True)
    checkpoint = Path(config["checkpoint"]).resolve(strict=True)
    actual_ckpt_sha = sha256_file(checkpoint)
    if actual_ckpt_sha != spec["checkpoint_sha256"]:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch for {checkpoint}: "
            f"expected {spec['checkpoint_sha256']}, got {actual_ckpt_sha}"
        )
    if config.get("moge_source_hashes") != moge_source_hashes(moge_root):
        raise RuntimeError("MoGe-3 source hashes differ from the pinned request")
    if config.get("worker_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("MoGe-3 worker hash differs from the pinned request")

    seen_names: set[str] = set()
    preprocessed_images: list[tuple[Mapping[str, Any], Path, np.ndarray, np.ndarray, float, bool]] = []
    pending_indices: list[int] = []

    for index, row in enumerate(images):
        name = row.get("name")
        if not name:
            raise ValueError(f"Image row at index {index} missing 'name'")
        if name in seen_names:
            raise ValueError(f"Duplicate image name in request: {name!r}")
        seen_names.add(name)

        target_path = safe_image_name(name, raw_depth_dir)
        width = int(row["width"])
        height = int(row["height"])
        k_src = np.asarray(row["K_source"], dtype=np.float64)
        k_work, h_matrix, fov_x = work_intrinsics(k_src, width, height)

        is_cached = False
        if target_path.is_file():
            validate_cached_npz(target_path, row, digest, k_work, h_matrix)
            is_cached = True
        else:
            pending_indices.append(index)

        preprocessed_images.append((row, target_path, k_work, h_matrix, fov_x, is_cached))

    reused_count = len(images) - len(pending_indices)
    forward_count = 0
    device = None

    if pending_indices:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for moge_metric_worker but not available")
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(INFERENCE_SETTINGS["cuda_memory_fraction"], device)
        torch.cuda.reset_peak_memory_stats(device)
        model = load_moge_model(moge_root, checkpoint, device, model_id)

        for step, idx in enumerate(pending_indices, start=1):
            row, target_path, k_work, h_matrix, fov_x, _ = preprocessed_images[idx]
            name = row["name"]
            width = int(row["width"])
            height = int(row["height"])

            image_path = Path(row["image_path"]).resolve(strict=True)
            expected_img_sha = str(row["image_sha256"])
            actual_img_sha = sha256_file(image_path)
            if actual_img_sha.lower() != expected_img_sha.lower():
                raise RuntimeError(
                    f"Image SHA256 mismatch for {image_path}: expected {expected_img_sha}, got {actual_img_sha}"
                )

            img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError(f"Failed to read image from {image_path}")
            if img_bgr.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Image dimension mismatch for {image_path}: read {img_bgr.shape[:2]} != expected ({height}, {width})"
                )

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_warped_rgb = cv2.warpPerspective(img_rgb, h_matrix, (width, height), flags=cv2.INTER_LINEAR)

            src_mask = np.ones((height, width), dtype=np.float32)
            warped_mask = cv2.warpPerspective(src_mask, h_matrix, (width, height), flags=cv2.INTER_LINEAR)
            valid_warp_mask = warped_mask >= (1.0 - 1e-6)

            img_tensor = torch.from_numpy(img_warped_rgb).permute(2, 0, 1).float().div_(255.0).to(device)

            with torch.inference_mode():
                output = model.infer(
                    img_tensor,
                    resolution_level=INFERENCE_SETTINGS["resolution_level"],
                    use_fp16=INFERENCE_SETTINGS["use_fp16"],
                    refine_steps=INFERENCE_SETTINGS["refine_steps"],
                    return_per_step=INFERENCE_SETTINGS["return_per_step"],
                    apply_mask=INFERENCE_SETTINGS["apply_mask"],
                    force_projection=INFERENCE_SETTINGS["force_projection"],
                    fov_x=fov_x,
                )

            k_norm = output["intrinsics"].detach().cpu().numpy().astype(np.float64)
            s_mat = np.array([[width, 0.0, -0.5], [0.0, height, -0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
            k_recovered = s_mat @ k_norm
            if not np.allclose(k_recovered, k_work, atol=1e-3, rtol=1e-4):
                raise RuntimeError(
                    f"Recovered intrinsics for {name} does not match expected K_work:\nRecovered:\n{k_recovered}\nExpected:\n{k_work}"
                )

            moge_depth = output["depth"].detach().cpu().numpy().astype(np.float32)
            moge_mask = output["mask"].detach().cpu().numpy().astype(bool)
            del output, img_tensor

            valid_final = moge_mask & valid_warp_mask & np.isfinite(moge_depth) & (moge_depth > 0)
            depth_m = np.where(valid_final, moge_depth, np.nan).astype(np.float32)
            mask = valid_final.astype(bool)

            target_path.parent.mkdir(parents=True, exist_ok=True)
            for stale in target_path.parent.glob(f"{target_path.name}.tmp.*"):
                try:
                    stale.unlink()
                except OSError:
                    pass
            temp_path = target_path.with_name(f"{target_path.name}.tmp.{os.getpid()}.npz")
            try:
                np.savez_compressed(
                    temp_path,
                    depth_m=depth_m,
                    mask=mask,
                    K_work=k_work,
                    source_to_work=h_matrix,
                    request_sha256=digest,
                    image_sha256=expected_img_sha,
                )
                os.replace(temp_path, target_path)
            finally:
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

            forward_count += 1
            if step % 25 == 0 or step == len(pending_indices):
                elapsed_so_far = time.perf_counter() - started_at
                fps = forward_count / max(elapsed_so_far, 1e-6)
                print(
                    f"[moge_worker] progress: {step}/{len(pending_indices)} forward ({reused_count} cached), {fps:.2f} fps",
                    file=sys.stderr,
                    flush=True,
                )

    elapsed_total = time.perf_counter() - started_at
    receipt = {
        "status": "completed",
        "stage": stage,
        "request_sha256": digest,
        "model_id": model_id,
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "refine_steps": INFERENCE_SETTINGS["refine_steps"],
        "total_images": len(images),
        "completed_count": len(images),
        "reused_count": reused_count,
        "forward_count": forward_count,
        "elapsed_seconds": round(elapsed_total, 3),
        "resources": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "cuda_memory_fraction": (
                None if device is None else torch.cuda.get_per_process_memory_fraction(device)
            ),
            "cuda_peak_allocated_bytes": (
                None if device is None else torch.cuda.max_memory_allocated(device)
            ),
            "cuda_peak_reserved_bytes": (
                None if device is None else torch.cuda.max_memory_reserved(device)
            ),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
