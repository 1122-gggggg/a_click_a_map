from __future__ import annotations

import hashlib
import fnmatch
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sfm_diagnosis.io import write_json

from .config import StageConfig, WorkflowConfig


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    receipt: Path
    outputs: tuple[Path, ...]


@dataclass(frozen=True)
class WorkflowResult:
    run_dir: Path
    stages: tuple[StageResult, ...]
    receipt: Path


class SiteWorkflow:
    """Deep module for a reproducible multi-video site workflow.

    Stage implementations are command adapters at the seam.  The module owns
    ordering, immutable video inventory, placeholder expansion, input/output
    contracts, caching, logs, and receipts.  Commands are argument arrays and
    never pass through a shell.
    """

    def __init__(self, config: WorkflowConfig):
        self.config = config

    def run(
        self,
        videos: Sequence[str | Path],
        run_dir: str | Path,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
        force: bool = False,
    ) -> WorkflowResult:
        output = Path(run_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        selected = _stage_slice(self.config.stages, from_stage, to_stage)
        inventory = self._inventory(videos, output, force=force)
        results = [inventory]
        context = self._context(output)
        context["video_manifest"] = str(output / "inputs" / "videos.json")
        for stage in selected:
            results.append(self._run_stage(stage, context, output, force=force))
        receipt = output / "workflow_receipt.json"
        write_json(
            receipt,
            {
                "schema_version": 1,
                "artifact_type": "MULTI_VIDEO_SITE_WORKFLOW",
                "site_name": self.config.site_name,
                "run_dir": str(output),
                "stages": [
                    {
                        "name": stage.name,
                        "status": stage.status,
                        "receipt": str(stage.receipt),
                        "outputs": [str(path) for path in stage.outputs],
                    }
                    for stage in results
                ],
            },
        )
        return WorkflowResult(output, tuple(results), receipt)

    def _context(self, run_dir: Path) -> dict[str, str]:
        context = {"run_dir": str(run_dir), "site_name": self.config.site_name}
        for key, value in self.config.paths.items():
            path = Path(value)
            context[key] = str(path if path.is_absolute() else run_dir / path)
        return context

    def _inventory(
        self, videos: Sequence[str | Path], run_dir: Path, *, force: bool
    ) -> StageResult:
        paths = sorted({Path(value).expanduser().resolve() for value in videos})
        if not paths:
            raise ValueError("site workflow requires at least one video")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"video inputs do not exist: {missing}")
        rows = [
            _video_row(
                path,
                hash_contents=self.config.hash_video_contents,
                role=(
                    "heldout"
                    if any(
                        fnmatch.fnmatch(path.name, pattern)
                        for pattern in self.config.heldout_patterns
                    )
                    else "mapping"
                ),
            )
            for path in paths
        ]
        if not any(row["role"] == "mapping" for row in rows):
            raise ValueError("site workflow requires at least one mapping video")
        fingerprint = _json_hash(rows)
        manifest = run_dir / "inputs" / "videos.json"
        receipt = run_dir / "receipts" / "inventory.json"
        cached = _receipt_matches(receipt, fingerprint) and manifest.is_file() and not force
        if not cached:
            write_json(
                manifest,
                {
                    "schema_version": 1,
                    "artifact_type": "SITE_VIDEO_INVENTORY",
                    "site_name": self.config.site_name,
                    "videos": rows,
                },
            )
            write_json(receipt, {"stage": "inventory", "fingerprint": fingerprint})
        return StageResult(
            "inventory", "cached" if cached else "completed", receipt, (manifest,)
        )

    def _run_stage(
        self,
        stage: StageConfig,
        context: dict[str, str],
        run_dir: Path,
        *,
        force: bool,
    ) -> StageResult:
        command = tuple(_expand(value, context) for value in stage.command)
        requires = tuple(Path(_expand(value, context)) for value in stage.requires)
        produces = tuple(Path(_expand(value, context)) for value in stage.produces)
        missing = [str(path) for path in requires if not path.exists()]
        if missing:
            raise RuntimeError(f"stage {stage.name} missing required inputs: {missing}")
        fingerprint = _json_hash(
            {
                "stage": asdict(stage),
                "command": command,
                "inputs": [_path_fingerprint(path) for path in requires],
                "video_manifest": _path_fingerprint(Path(context["video_manifest"])),
            }
        )
        receipt = run_dir / "receipts" / f"{stage.name}.json"
        if (
            not force
            and _receipt_matches(receipt, fingerprint)
            and all(path.exists() for path in produces)
        ):
            return StageResult(stage.name, "cached", receipt, produces)
        log = run_dir / "logs" / f"{stage.name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cwd = run_dir if stage.cwd is None else Path(_expand(stage.cwd, context))
        env = os.environ.copy()
        env.update({key: _expand(value, context) for key, value in stage.env.items()})
        started = time.time()
        with log.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stage {stage.name} failed with exit code {completed.returncode}; log={log}"
            )
        absent = [str(path) for path in produces if not path.exists()]
        if absent:
            raise RuntimeError(f"stage {stage.name} did not produce contracted outputs: {absent}")
        write_json(
            receipt,
            {
                "schema_version": 1,
                "artifact_type": "SITE_WORKFLOW_STAGE_RECEIPT",
                "stage": stage.name,
                "fingerprint": fingerprint,
                "command": list(command),
                "requires": [str(path) for path in requires],
                "produces": [str(path) for path in produces],
                "log": str(log),
                "returncode": completed.returncode,
                "runtime_seconds": time.time() - started,
            },
        )
        return StageResult(stage.name, "completed", receipt, produces)


def _stage_slice(
    stages: Sequence[StageConfig], from_stage: str | None, to_stage: str | None
) -> tuple[StageConfig, ...]:
    names = [stage.name for stage in stages]
    if from_stage is not None and from_stage not in names:
        raise ValueError(f"unknown from_stage {from_stage!r}")
    if to_stage is not None and to_stage not in names:
        raise ValueError(f"unknown to_stage {to_stage!r}")
    start = names.index(from_stage) if from_stage is not None else 0
    stop = names.index(to_stage) + 1 if to_stage is not None else len(stages)
    if start >= stop:
        raise ValueError("from_stage must not come after to_stage")
    return tuple(stages[start:stop])


def _expand(value: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in context:
            raise ValueError(f"unknown workflow placeholder {name!r}")
        return context[name]

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, str(value))


def _video_row(path: Path, *, hash_contents: bool, role: str) -> dict:
    stat = path.stat()
    row = {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "role": role,
    }
    if hash_contents:
        row["sha256"] = _file_hash(path)
    return row


def _path_fingerprint(path: Path) -> dict:
    if path.is_file():
        stat = path.stat()
        return {"path": str(path), "type": "file", "size": stat.st_size, "sha256": _file_hash(path)}
    if path.is_dir():
        rows = [
            (str(child.relative_to(path)), child.stat().st_size, child.stat().st_mtime_ns)
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ]
        return {"path": str(path), "type": "directory", "files": rows}
    return {"path": str(path), "type": "missing"}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _receipt_matches(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint") == fingerprint
    except (OSError, json.JSONDecodeError):
        return False


__all__ = ["SiteWorkflow", "StageResult", "WorkflowResult"]
