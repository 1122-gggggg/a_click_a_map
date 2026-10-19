"""Isolated official MV-RoMa inference worker.

Official API contract:
https://github.com/IceTea-CV/MV-RoMa/blob/acb09efb0212129ac191031f6e56f150524b304f/demo.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvroma-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8"))
        receipt = run_worker(payload)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "FAILED", "reason": str(error)}))
        return 2
    print(json.dumps({"status": "COMPLETED", "receipt": str(args.receipt)}))
    return 0


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch.version

    from demo import build_model_matcher  # ty: ignore[unresolved-import]
    from src.run_model import run_model_test  # ty: ignore[unresolved-import]

    groups = list(payload.get("groups") or ())
    if not groups:
        raise ValueError("MV-RoMa worker requires at least one group")
    if any(
        "P168" in json.dumps(group).upper() or "vid_540e7" in json.dumps(group) for group in groups
    ):
        raise ValueError("P168 is forbidden from MV-RoMa mapping inference")
    output_dir = Path(payload["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    weight = Path(payload["weight"]).resolve(strict=True)
    coarse_size = tuple(int(value) for value in payload["coarse_size"])
    target_size = tuple(int(value) for value in payload["target_size"])
    device = str(payload.get("device") or "cuda:0")
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("MV-RoMa CUDA runtime is unavailable")
    torch.cuda.reset_peak_memory_stats()
    prematch_name, prematch, model = build_model_matcher(device=device, weight_path=str(weight))
    shards = []
    with torch.inference_mode():
        for index, group in enumerate(groups):
            source = Path(group["source_path"]).resolve(strict=True)
            targets = [Path(value).resolve(strict=True) for value in group["target_paths"]]
            correspondences = run_model_test(
                model,
                {"query_img_path": str(source), "ref_img_paths": [str(path) for path in targets]},
                coarse_res_hw=coarse_size,
                target_res_hw=target_size,
                prematch_model=prematch,
                prematch_model_name=prematch_name,
                upsample_preds=True,
                num_cluster=512,
                device=device,
            )
            finest = min(correspondences)
            flow = correspondences[finest]["flow"][0].detach().cpu().numpy().astype(np.float16)
            certainty = (
                correspondences[finest]["certainty"][0, :, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            shard = output_dir / f"{index:04d}_{group['group_id']}.npz"
            np.savez(
                shard,
                flow=flow,
                certainty_logits=certainty,
                source_name=np.asarray(group["source_name"]),
                target_names=np.asarray(group["target_names"]),
            )
            shards.append(
                {
                    "group_id": group["group_id"],
                    "path": str(shard),
                    "sha256": _sha256(shard),
                    "flow_shape": list(flow.shape),
                    "certainty_shape": list(certainty.shape),
                }
            )
    return {
        "schema_version": 1,
        "artifact_type": "MVROMA_INFERENCE",
        "status": "COMPLETED",
        "groups": len(groups),
        "shards": shards,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "arch_list": torch.cuda.get_arch_list(),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "weight": str(weight),
        "weight_sha256": _sha256(weight),
        "coarse_size": list(coarse_size),
        "target_size": list(target_size),
        "pid": os.getpid(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_worker"]
