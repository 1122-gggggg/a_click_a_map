"""Launch the pinned GlueMap checkout for one multi-sequence mapping job.

This module drives the real pipeline stages (retrieval -> two-view ->
star -> global mapping -> refinement) through the GlueMap Python
orchestration in a subprocess, so the mapping stage never imports torch
or GlueMap itself. Extra forced pairs (pose-only bridges) are unioned
into the retrieval pair graph by a small sitecustomize-style patch that
is injected through PYTHONPATH; the patch file is written next to the
job workspace so the mechanism is auditable.

Only this checkout's GlueMap API is supported (run_preprocessing_pipeline_multi,
MultiSequencePairs, GluemapPipeline.run). Anything else fails loudly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

_RUNNER = r'''
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(job["gluemap_root"]).expanduser().resolve()
sys.path.insert(0, str(root))
os.chdir(root)
from contextlib import nullcontext as _nullcontext
from gluemap.controllers import gluemap_impl as _impl
from gluemap.controllers.gluemap_impl import GluemapPipeline
from gluemap.controllers.image_retrieval import run_preprocessing_pipeline_multi
from gluemap.datasets.multi_sequence_twoview import MultiSequencePairs
from gluemap.utils.gpu import init_distributed

_NATIVE_PREPARE_SIFT = _impl.prepare_sift_database


def _prepare_sift_with_device(*args, **kwargs):
    kwargs.setdefault("device", _SIFT_DEVICE)
    return _NATIVE_PREPARE_SIFT(*args, **kwargs)


_SIFT_DEVICE = str(job["config"].get("sift_device") or "cuda")
_impl.prepare_sift_database = _prepare_sift_with_device

config = dict(job["config"])
args = Namespace(**config)
args.curr_processed = args.write_path
args.curr_path = args.write_path
args.is_multi_sequence = True
args.is_sequential = True

rank, world_size, device, dtype = init_distributed(args)
datasets = [
    child.name for child in sorted(Path(args.images_path).iterdir())
    if child.is_dir()
]
if not datasets:
    raise RuntimeError("no GlueMap image sequence folders were materialized")
run_preprocessing_pipeline_multi(args, world_size, rank, datasets)
dataset = MultiSequencePairs(args, datasets)
pipeline = GluemapPipeline(args, world_size, rank, device, dtype)
_pose_names = list(job.get("pose_only_names") or [])
if _pose_names:
    sys.path.insert(0, job["mapping_package_parent"])
    from mapping.pose_only import gluemap_pose_only_mask as _pose_mask
    _mask_context = _pose_mask(_pose_names, workspace_root=job["workspace"])
else:
    _mask_context = _nullcontext(None)
with _mask_context as _pose_state:
    prediction_dir, timing = pipeline.run(dataset, pairs=dataset.pairs)
print(json.dumps({"prediction_dir": prediction_dir, "timing": timing, "pose_only_mask": _pose_state}))
'''

_PAIR_PATCH = r'''
"""Union forced pairs into MultiSequencePairs (injected via PYTHONPATH)."""
import numpy as _np
from pathlib import Path as _Path

_FORCED = _Path(__file__).with_name("forced_pairs.txt")

try:
    from gluemap.datasets import multi_sequence_twoview as _mst
    _ORIGINAL = _mst.MultiSequencePairs._construct_pairs

    def _patched(self, args, img_list):
        _ORIGINAL(self, args, img_list)
        path = getattr(args, "extra_pairs_path", None) or (
            str(_FORCED) if _FORCED.is_file() else None
        )
        if not path:
            return
        index_of = {_Path(n).as_posix(): i for i, n in enumerate(self.images_list)}
        forced, missing = [], []
        with open(path) as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"{path}:{lineno}: expected 'nameA nameB'")
                a, b = (_Path(n).as_posix() for n in parts)
                if a not in index_of or b not in index_of:
                    missing.append((lineno, a if a not in index_of else b))
                    continue
                i, j = index_of[a], index_of[b]
                if i != j:
                    forced.append((min(i, j), max(i, j)))
        if missing:
            raise ValueError(f"forced pairs reference unknown images: {missing[:5]}")
        if forced:
            pairs = _np.asarray(self.pairs, dtype=_np.int64).reshape(-1, 2)
            extra = _np.asarray(sorted(set(forced)), dtype=_np.int64)
            merged = _np.unique(_np.concatenate([pairs, extra], axis=0), axis=0)
            merged = merged[merged[:, 0] < merged[:, 1]]
            self.pairs = merged

    _mst.MultiSequencePairs._construct_pairs = _patched
except ImportError:
    pass
'''


def run_gluemap_job(
    *,
    gluemap_root: str | Path,
    job_path: str | Path,
    python: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run one GlueMap job file; return the runner's JSON result."""
    job_file = Path(job_path)
    payload = json.loads(job_file.read_text(encoding="utf-8"))
    workspace = Path(payload["workspace"])
    runner_path = workspace / "gluemap_runner.py"
    runner_path.write_text(_RUNNER, encoding="utf-8")
    patch_dir = workspace / "pair_patch"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "gluemap_pair_patch.py").write_text(_PAIR_PATCH, encoding="utf-8")
    forced = payload.get("forced_pairs_path")
    if forced and Path(forced).is_file():
        (patch_dir / "forced_pairs.txt").write_text(
            Path(forced).read_text(encoding="utf-8"), encoding="utf-8"
        )
    env = dict(os.environ)
    if extra_env:
        env.update(dict(extra_env))
    env["PYTHONPATH"] = str(patch_dir) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # sitecustomize is auto-imported at interpreter startup when found on
    # sys.path (via PYTHONPATH), so the pair patch installs before GlueMap.
    sitecustomize = patch_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "import gluemap_pair_patch  # noqa: F401\n", encoding="utf-8"
    )
    command = [python or sys.executable, str(runner_path), str(job_file)]
    log_file = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        completed = subprocess.run(
            command,
            capture_output=log_file is None,
            text=True,
            check=False,
            timeout=timeout,
            stdout=log_file if log_file else None,
            stderr=subprocess.STDOUT if log_file else None,
            env=env,
        )
    finally:
        if log_file is not None:
            log_file.close()
    if completed.returncode != 0:
        detail = ""
        if log_path:
            detail = f"see {log_path}"
        elif completed.stderr:
            detail = completed.stderr.strip()[-4000:]
        raise RuntimeError(f"GlueMap job failed (exit {completed.returncode}): {detail}")
    output = completed.stdout or ""
    if log_path:
        # Runner JSON is the last JSON line of the log.
        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
        output = next(
            (line for line in reversed(lines) if line.strip().startswith("{")),
            "",
        )
    try:
        return json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise RuntimeError(
            f"GlueMap runner returned no JSON result: {output[-2000:]}"
        ) from error


def build_job_payload(
    *,
    gluemap_root: str | Path,
    workspace: str | Path,
    config: Mapping[str, Any],
    forced_pairs_path: str | Path | None,
    pose_only_names: list[str] | None = None,
    mapping_package_parent: str | Path | None = None,
) -> dict[str, Any]:
    return {
        "gluemap_root": str(gluemap_root),
        "workspace": str(workspace),
        "config": dict(config),
        "forced_pairs_path": str(forced_pairs_path) if forced_pairs_path else None,
        "pose_only_names": list(pose_only_names or []),
        "mapping_package_parent": (
            str(mapping_package_parent) if mapping_package_parent else None
        ),
    }


__all__ = ["build_job_payload", "run_gluemap_job"]
