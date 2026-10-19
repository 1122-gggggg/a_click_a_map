"""Stage-3 GLUEMAP SALAD descriptor and retrieval candidate worker."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .domain import KeyframeRecord
from .retrieval import RetrievalCategory, generate_candidates


def materialize_images(
    rows: Sequence[Mapping[str, Any]], output: Path, *, approved_root: Path
) -> list[tuple[str, str]]:
    """Create a deterministic flat image view and return (keyframe, filename)."""

    output.mkdir(parents=True, exist_ok=True)
    mapping = []
    approved = approved_root.resolve()
    for index, row in enumerate(sorted(rows, key=lambda item: str(item["keyframe_id"]))):
        source = Path(str(row.get("image_uri") or "")).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if approved not in source.parents:
            raise ValueError(f"SALAD image escapes approved keyframe root: {source}")
        expected_sha = str(row.get("image_sha256") or "")
        if not expected_sha:
            raise ValueError("SALAD keyframe requires image_sha256")
        if _sha256(source) != expected_sha:
            raise ValueError(f"SALAD keyframe hash changed: {source}")
        name = f"{index:08d}_{source.name}"
        target = output / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"SALAD materialization collision: {target}")
        else:
            target.symlink_to(source)
        mapping.append((str(row["keyframe_id"]), name))
    return mapping


def extract_salad_descriptors(
    images_root: Path,
    workspace: Path,
    config: Mapping[str, Any],
) -> np.ndarray:
    root = Path(str(config.get("gluemap_root") or "")).expanduser().resolve()
    config_file = Path(str(config.get("config_file") or "")).expanduser().resolve()
    if not (root / "gluemap").is_dir() or not config_file.is_file():
        raise ValueError("SALAD worker requires gluemap_root and config_file")
    configured = json.loads(config_file.read_text(encoding="utf-8"))
    retrieval_checkpoint = Path(str(configured.get("path_retrieval") or "")).resolve()
    if not retrieval_checkpoint.is_file():
        raise FileNotFoundError(retrieval_checkpoint)
    checkpoint_sha = _sha256(retrieval_checkpoint)
    if config.get("checkpoint_sha256") and checkpoint_sha != config["checkpoint_sha256"]:
        raise RuntimeError("SALAD checkpoint SHA256 does not match deployment config")
    sys.path.insert(0, str(root))
    from gluemap.utils.cli import get_args_parser

    values = vars(get_args_parser().parse_args([]))
    values.update(configured)
    values.update(
        images_path=str(images_root),
        write_path=str(workspace),
        curr_processed=str(workspace),
        curr_path=str(workspace),
        rerun_from="retrieval",
        force_load=False,
    )
    args = Namespace(**values)
    previous = Path.cwd()
    try:
        os.chdir(root)
        from gluemap.controllers.image_retrieval import run_preprocessing_pipeline
        from gluemap.utils.gpu import init_distributed

        rank, world_size, _, _ = init_distributed(args)
        run_preprocessing_pipeline(args, world_size, rank)
        import torch

        # This file was generated in this workspace and weights_only forbids arbitrary objects.
        # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        descriptors = torch.load(
            workspace / "salad_descriptors.pt", map_location="cpu", weights_only=True
        )
    finally:
        os.chdir(previous)
    if hasattr(descriptors, "detach"):
        descriptors = descriptors.detach().cpu().numpy()
    return np.asarray(descriptors, dtype=np.float32)


def run_adapter_request(
    payload: Mapping[str, Any],
    *,
    descriptor_extractor: Callable[
        [Path, Path, Mapping[str, Any]], np.ndarray
    ] = extract_salad_descriptors,
) -> dict[str, Any]:
    config = dict(payload.get("config") or {})
    request = dict(payload.get("payload") or {})
    keyframe_path = Path(str(request.get("keyframes") or ""))
    output_path = Path(str(request.get("output_candidates") or ""))
    if not keyframe_path.is_file() or not str(output_path):
        raise ValueError("SALAD adapter requires keyframes and output_candidates")
    run_root_value = str(request.get("run_root") or "").strip()
    if not run_root_value:
        raise ValueError("SALAD adapter requires trusted run_root")
    run_root = Path(run_root_value).expanduser().resolve()
    if (
        run_root not in keyframe_path.resolve().parents
        or run_root not in output_path.resolve().parents
    ):
        raise ValueError("SALAD request paths escape run_root")
    rows = _jsonl(keyframe_path)
    if any(str(row.get("evaluation_role") or "MAPPING") == "HOLDOUT" for row in rows):
        raise RuntimeError("hold-out keyframe leaked into SALAD descriptor request")
    workspace = output_path.parent / "salad"
    mapping = materialize_images(
        rows,
        workspace / "images",
        approved_root=run_root / "artifacts/keyframes/images",
    )
    descriptors = descriptor_extractor(workspace / "images", workspace, config)
    if len(descriptors) != len(mapping):
        raise RuntimeError("SALAD descriptor count differs from keyframe count")
    by_id = {str(row["keyframe_id"]): row for row in rows}
    records = []
    for keyframe_id, _ in mapping:
        row = by_id[keyframe_id]
        records.append(
            KeyframeRecord(
                keyframe_id=keyframe_id,
                segment_id=str(row["segment_id"]),
                video_id=str(row["video_id"]),
                source_frame_index=int(row.get("source_frame_index") or 0),
                source_pts_seconds=float(row.get("source_pts_seconds") or 0.0),
                image_uri=str(row.get("image_uri") or ""),
                metadata={"session_id": row.get("session_id")},
            )
        )
    caps = {
        RetrievalCategory(key): int(value)
        for key, value in dict(config.get("category_caps") or {}).items()
    }
    result = generate_candidates(
        descriptors,
        records,
        temporal_radius=int(config.get("temporal_radius", 4)),
        top_k=int(config.get("top_k", 20)),
        category_caps=caps,
        loop_min_time_delta=float(config.get("loop_min_time_delta", 30.0)),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(
                {
                    "image_i": row.image_i,
                    "image_j": row.image_j,
                    "category": row.category.value,
                    "score": row.score,
                    "admission": row.admission,
                    "is_geometry_edge": row.is_geometry_edge,
                },
                sort_keys=True,
            )
            + "\n"
            for row in result.candidates
        ),
        encoding="utf-8",
    )
    np.save(workspace / "descriptors.npy", descriptors)
    return {
        "status": "completed",
        "outputs": [str(output_path)],
        "candidate_count": len(result.candidates),
        "descriptor_artifact": str(workspace / "descriptors.npy"),
    }


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


__all__ = ["extract_salad_descriptors", "materialize_images", "run_adapter_request"]
