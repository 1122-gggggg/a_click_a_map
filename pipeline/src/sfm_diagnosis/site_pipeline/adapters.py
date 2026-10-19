"""External-tool seams for SALAD, EDM, GLUEMAP and localization workers.

The official GLUEMAP stage/config assumptions are documented at:
https://github.com/colmap/gluemap/blob/main/README.md
https://github.com/colmap/gluemap/blob/main/gluemap/controllers/gluemap_impl.py
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class AdapterRequest:
    stage: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    input_paths: tuple[str, ...] = ()
    output_dir: str | None = None
    resource_class: str = "cpu"

    def fingerprint(self) -> str:
        material = asdict(self)
        material["inputs"] = [_path_identity(Path(path)) for path in self.input_paths]
        return _json_hash(material)


@dataclass(frozen=True)
class AdapterReceipt:
    stage: str
    request_fingerprint: str
    status: str
    output: Mapping[str, Any] = field(default_factory=dict)
    command: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)


class Adapter(Protocol):
    def run(self, request: AdapterRequest) -> AdapterReceipt: ...


class CommandAdapter:
    """Execute a JSON-in/JSON-out command without a shell."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = tuple(str(value) for value in command)
        executable = Path(self.command[0]).expanduser()
        if not executable.is_absolute() or not executable.is_file():
            raise ValueError("adapter executable must be an existing absolute path")
        self.cwd = None if cwd is None else str(cwd)
        self.env = dict(env or {})
        self.timeout_seconds = timeout_seconds

    def run(self, request: AdapterRequest) -> AdapterReceipt:
        environment = os.environ.copy()
        environment.update(self.env)
        try:
            completed = subprocess.run(  # nosemgrep: python.django.security.injection.command.subprocess-injection.subprocess-injection
                self.command,
                input=json.dumps(asdict(request), ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=self.cwd,
                env=environment,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"adapter timed out: {self.command[0]}") from error
        if completed.returncode:
            tail = (completed.stderr or completed.stdout)[-2000:]
            raise RuntimeError(f"adapter failed ({completed.returncode}): {tail}")
        try:
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            output = json.loads(lines[-1] if lines else "")
        except json.JSONDecodeError as error:
            raise ValueError("adapter stdout must end with one JSON object") from error
        if not isinstance(output, dict):
            raise ValueError("adapter stdout must be a JSON object")
        status = str(output.get("status") or "ok")
        if status.lower() not in {"ok", "completed", "success"}:
            raise RuntimeError(f"adapter returned non-success status {status!r}")
        return AdapterReceipt(
            request.stage,
            request.fingerprint(),
            status,
            output,
            self.command,
            output.get("runtime") or {},
        )


class FakeAdapter:
    def __init__(self, output: Mapping[str, Any] | None = None) -> None:
        self.output = dict(output or {})
        self.requests: list[AdapterRequest] = []

    def run(self, request: AdapterRequest) -> AdapterReceipt:
        self.requests.append(request)
        return AdapterReceipt(request.stage, request.fingerprint(), "ok", self.output)


def runtime_preflight() -> dict[str, Any]:
    """Return facts for the current interpreter; importing Torch is optional."""

    result: dict[str, Any] = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch  # type: ignore
    except Exception as error:  # pragma: no cover - optional dependency
        result["torch_error"] = type(error).__name__
        return result
    result.update(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        result.update(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "arch_list": list(torch.cuda.get_arch_list()),
                "sm_120": "sm_120" in torch.cuda.get_arch_list(),
            }
        )
    return result


def probe_python_runtime(python_executable: str | Path) -> dict[str, Any]:
    """Probe a worker environment without importing its packages into the core."""

    source = (
        "import json,sys; out={'python':sys.version}; "
        "import torch; out.update(torch=torch.__version__,cuda=torch.version.cuda,"
        "cuda_available=torch.cuda.is_available()); "
        "out.update(gpu=torch.cuda.get_device_name(0),"
        "compute_capability='.'.join(map(str,torch.cuda.get_device_capability(0))),"
        "arch_list=torch.cuda.get_arch_list(),sm_120='sm_120' in torch.cuda.get_arch_list()) "
        "if torch.cuda.is_available() else None; print(json.dumps(out))"
    )
    completed = subprocess.run(
        [str(python_executable), "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"runtime preflight failed: {completed.stderr[-1000:]}")
    return json.loads(completed.stdout)


def require_compute_capability(runtime: Mapping[str, Any], capability: str = "12.0") -> None:
    if not runtime.get("cuda_available"):
        raise RuntimeError("CUDA is unavailable in the configured worker environment")
    if str(runtime.get("compute_capability")) != capability:
        raise RuntimeError(
            f"expected compute capability {capability}, got {runtime.get('compute_capability')}"
        )
    arch = f"sm_{capability.replace('.', '')}"
    if arch not in set(runtime.get("arch_list") or ()):
        raise RuntimeError(f"worker Torch build does not contain {arch}")


def validate_admitted_pairs(
    pairs: Sequence[Mapping[str, Any]], *, allowed_ids: set[str]
) -> tuple[tuple[str, str], ...]:
    normalized: set[tuple[str, str]] = set()
    for pair in pairs:
        missing = {key for key in ("image_i", "image_j") if key not in pair}
        if missing:
            raise ValueError(f"pair missing fields: {sorted(missing)}")
        left, right = str(pair["image_i"]), str(pair["image_j"])
        if left not in allowed_ids or right not in allowed_ids:
            raise ValueError("pair references image outside admitted set")
        if left == right:
            raise ValueError("self-pair is not admitted")
        if str(pair.get("admission", "VERIFIED")) != "VERIFIED":
            raise ValueError("only VERIFIED pairs may enter a mapping manifest")
        normalized.add((left, right) if left < right else (right, left))
    return tuple(sorted(normalized))


def materialize_admitted_pairs(
    pairs: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    allowed_ids: set[str],
) -> Path:
    normalized = validate_admitted_pairs(pairs, allowed_ids=allowed_ids)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_pair = {
        tuple(sorted((str(row["image_i"]), str(row["image_j"])))): dict(row) for row in pairs
    }
    path.write_text(
        "".join(json.dumps(by_pair[pair], sort_keys=True) + "\n" for pair in normalized),
        encoding="utf-8",
    )
    return path


def materialize_gluemap_pairs(
    pairs: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    allowed_ids: set[str],
) -> Path:
    normalized = validate_admitted_pairs(pairs, allowed_ids=allowed_ids)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{left} {right}\n" for left, right in normalized), encoding="utf-8")
    return path


HEAVY_RESOURCE_CLASSES = frozenset({"gpu_heavy", "ba", "gluemap", "matcher"})


def assert_resource_available(resource_class: str, active_classes: Sequence[str]) -> None:
    if resource_class in HEAVY_RESOURCE_CLASSES and any(
        item in HEAVY_RESOURCE_CLASSES for item in active_classes
    ):
        raise RuntimeError(f"exclusive resource busy: {resource_class}")


class ExclusiveResourceLease:
    """Cross-process advisory lock for memory-heavy stages."""

    def __init__(self, path: str | Path, resource_class: str) -> None:
        self.path = Path(path)
        self.resource_class = resource_class
        self._stream = None

    def __enter__(self) -> "ExclusiveResourceLease":
        if self.resource_class not in HEAVY_RESOURCE_CLASSES:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise RuntimeError(f"exclusive resource busy: {self.resource_class}") from error
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(self.resource_class)
        self._stream.flush()
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


def _path_identity(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}
    if path.is_dir():
        return {
            "path": str(path),
            "files": [
                (str(child.relative_to(path)), child.stat().st_size, _sha256(child))
                for child in sorted(path.rglob("*"))
                if child.is_file()
            ],
        }
    return {"path": str(path), "missing": True}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


__all__ = [
    "Adapter",
    "AdapterReceipt",
    "AdapterRequest",
    "CommandAdapter",
    "ExclusiveResourceLease",
    "FakeAdapter",
    "assert_resource_available",
    "materialize_admitted_pairs",
    "materialize_gluemap_pairs",
    "probe_python_runtime",
    "require_compute_capability",
    "runtime_preflight",
    "validate_admitted_pairs",
]
