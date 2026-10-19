from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


CANONICAL_STAGES = (
    "selection",
    "mapping",
    "map_diagnosis",
    "localization_loo",
    "risk_diagnosis",
)


@dataclass(frozen=True)
class StageConfig:
    name: str
    command: tuple[str, ...]
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageConfig":
        name = str(payload.get("name") or "").strip()
        command = tuple(str(value) for value in payload.get("command") or ())
        if not name or not command:
            raise ValueError("each workflow stage requires name and command")
        return cls(
            name=name,
            command=command,
            requires=tuple(str(value) for value in payload.get("requires") or ()),
            produces=tuple(str(value) for value in payload.get("produces") or ()),
            cwd=None if payload.get("cwd") is None else str(payload["cwd"]),
            env={str(key): str(value) for key, value in dict(payload.get("env") or {}).items()},
        )


@dataclass(frozen=True)
class WorkflowConfig:
    site_name: str
    stages: tuple[StageConfig, ...]
    paths: Mapping[str, str] = field(default_factory=dict)
    hash_video_contents: bool = True
    heldout_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.site_name.strip():
            raise ValueError("workflow site_name must not be empty")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("workflow stage names must be unique")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowConfig":
        return cls(
            site_name=str(payload.get("site_name") or ""),
            stages=tuple(StageConfig.from_dict(row) for row in payload.get("stages") or ()),
            paths={str(key): str(value) for key, value in dict(payload.get("paths") or {}).items()},
            hash_video_contents=bool(payload.get("hash_video_contents", True)),
            heldout_patterns=tuple(
                str(value) for value in payload.get("heldout_patterns") or ()
            ),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "WorkflowConfig":
        with Path(path).open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))


__all__ = ["CANONICAL_STAGES", "StageConfig", "WorkflowConfig"]
