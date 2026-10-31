"""MegaLoc reference-bank construction for an arbitrary COLMAP map.

Mirrors the validated ``build_localization_package`` contract: every
registered map image with at least one 3D observation and a non-POSE_ONLY
keyframe row becomes a reference row; its undistorted keyframe JPEG is
embedded into a normalized 8,448-D MegaLoc descriptor with the pinned
offline runtime (local source + explicit checkpoint, no hub downloads).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

MEGALOC_DESCRIPTOR_DIM = 8448

_REFERENCE_WORKER = r'''
import json, sys
from pathlib import Path

def fail(message):
    sys.stdout.write(json.dumps({"status": "error", "error": message}) + "\n")
    raise SystemExit(1)

try:
    payload = json.load(sys.stdin)
    model_dir = Path(payload["model_dir"]).expanduser().resolve(strict=True)
    keyframes_path = Path(payload["keyframes_path"]).expanduser().resolve(strict=True)
    output_dir = Path(payload["output_dir"]).expanduser()
    code_roots = [Path(p).expanduser().resolve() for p in payload["code_roots"]]
    megaloc_source = Path(payload["megaloc_source"]).expanduser().resolve(strict=True)
    megaloc_checkpoint = Path(payload["megaloc_checkpoint"]).expanduser().resolve(strict=True)
    batch_size = int(payload.get("descriptor_batch_size", 8))
    if batch_size <= 0:
        fail("descriptor_batch_size must be positive")
    for root in code_roots:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    import pycolmap
    from river_map_quality.megaloc_edm_catalog import (
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )
except Exception as error:
    fail(f"{type(error).__name__}: {error}")

try:
    rows = [json.loads(line) for line in keyframes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keyframes = {str(row["output_name"]): row for row in rows}
    if len(keyframes) != len(rows):
        fail("keyframe output names must be unique")
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    references = []
    for image in sorted(reconstruction.images.values(), key=lambda value: str(value.name)):
        name = str(image.name)
        keyframe = keyframes.get(name)
        if keyframe is None:
            fail(f"registered image is absent from keyframes: {name}")
        observations = sum(bool(point.has_point3D()) for point in image.points2D)
        if keyframe.get("mapping_mode") == "POSE_ONLY" or observations == 0:
            continue
        image_uri = keyframe.get("image_uri")
        if not image_uri:
            fail(f"keyframe has no image_uri: {name}")
        references.append({
            "image_name": name,
            "image_path": str(Path(str(image_uri)).expanduser().resolve(strict=True)),
            "video_id": str(keyframe.get("video_id", "")),
            "observations": int(observations),
        })
    if not references:
        fail("map contains no references with 3D observations")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reference_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in references),
        encoding="utf-8",
    )
    runtime = load_offline_megaloc_runtime(
        source=megaloc_source, checkpoint=megaloc_checkpoint, device="cuda",
    )
    descriptors = extract_megaloc_descriptors(
        runtime, [Path(row["image_path"]) for row in references], batch_size=batch_size,
    )
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    import numpy as np
    array = np.ascontiguousarray(descriptors, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != len(references):
        fail("MegaLoc descriptor rows disagree with reference identities")
    descriptor_path = output_dir / "megaloc_references.npy"
    np.save(descriptor_path, array, allow_pickle=False)
    names_path = output_dir / "megaloc_references.names.json"
    names_path.write_text(
        json.dumps([row["image_name"] for row in references], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(json.dumps({
        "status": "completed",
        "references": len(references),
        "descriptor_shape": list(array.shape),
        "reference_manifest": str(manifest_path),
        "descriptors": str(descriptor_path),
        "names": str(names_path),
    }) + "\n")
except SystemExit:
    raise
except Exception as error:
    fail(f"{type(error).__name__}: {error}")
'''


def build_reference_bank(
    *,
    model_dir: str | Path,
    keyframes_path: str | Path,
    output_dir: str | Path,
    python: str | Path,
    code_roots: list[str | Path],
    megaloc_source: str | Path,
    megaloc_checkpoint: str | Path,
    descriptor_batch_size: int = 8,
) -> dict[str, Any]:
    """Build MegaLoc reference assets in a fresh CUDA process.

    A fresh interpreter is required: the MegaLoc runtime must load before any
    other torch consumer pins a conflicting import order, and descriptor
    extraction needs CUDA without the parent's address space.
    """
    payload = {
        "model_dir": str(model_dir),
        "keyframes_path": str(keyframes_path),
        "output_dir": str(output_dir),
        "code_roots": [str(root) for root in code_roots],
        "megaloc_source": str(megaloc_source),
        "megaloc_checkpoint": str(megaloc_checkpoint),
        "descriptor_batch_size": int(descriptor_batch_size),
    }
    completed = subprocess.run(
        [str(python), "-c", _REFERENCE_WORKER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr.strip() or completed.stdout.strip() or "unknown error")
        raise RuntimeError(f"reference-bank worker failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "reference-bank worker returned invalid JSON: "
            + (completed.stderr.strip() or completed.stdout.strip())
        ) from error
    if result.get("status") != "completed":
        raise RuntimeError(f"reference-bank worker failed: {result}")
    shape = list(result.get("descriptor_shape", []))
    if len(shape) != 2 or shape[0] != result.get("references") or shape[1] != MEGALOC_DESCRIPTOR_DIM:
        raise RuntimeError(f"reference-bank descriptor shape violates contract: {shape}")
    return dict(result)


def validate_reference_bank(descriptors_path: str | Path, names_path: str | Path) -> dict[str, Any]:
    """Validate a staged bank: 2-D float32, 8448-D rows, unique aligned names."""
    import numpy as np

    array = np.load(str(descriptors_path), mmap_mode="r", allow_pickle=False)
    names = json.loads(Path(names_path).read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError("reference names sidecar must be a JSON list of strings")
    if array.ndim != 2 or array.shape[1] != MEGALOC_DESCRIPTOR_DIM:
        raise ValueError(f"reference bank shape {array.shape} violates 8448-D contract")
    if array.shape[0] != len(names):
        raise ValueError("reference bank rows disagree with reference names")
    if len(set(names)) != len(names):
        raise ValueError("reference names must be unique")
    return {"references": len(names), "descriptor_shape": [int(v) for v in array.shape]}


__all__ = ["MEGALOC_DESCRIPTOR_DIM", "build_reference_bank", "validate_reference_bank"]
