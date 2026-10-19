from __future__ import annotations

import json
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


CANONICAL_STAGES_V2 = (
    "stage00_inventory",
    "stage01_sanitization",
    "stage02_segment_keyframes",
    "stage03_retrieval",
    "stage04_pair_geometry",
    "stage05_graph_build",
    "stage06_pre_sfm_diagnosis",
    "stage07_diagnostic_selection",
    "stage08_diagnostic_mapping",
    "stage09_post_sfm_diagnosis",
    "stage10_role_assignment",
    "stage11_reinforcement",
    "stage12_final_mapping",
    "stage13_final_diagnosis",
    "stage14_localization_validation",
    "stage15_publish",
)

DEFAULT_REQUIRED_METADATA = (
    "session_id",
    "route",
    "direction",
    "camera_mode",
    "crop_state",
    "stabilization_state",
    "intrinsics_group",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration owned by the v2 deep module.

    External commands are adapter configuration, not per-stage orchestration.
    The module owns stage order and artifact contracts.
    """

    site_name: str
    schema_version: int = 2
    hash_contents: bool = True
    heldout_patterns: tuple[str, ...] = ()
    holdout_provenance: str = "MAPPING_DISJOINT_UNVERIFIED_OUTER"
    required_metadata: tuple[str, ...] = DEFAULT_REQUIRED_METADATA
    allow_unknown_metadata: bool = False
    max_reinforcement_rounds: int = 2
    mapping_optimization: Mapping[str, Any] = field(default_factory=dict)
    adapters: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("site pipeline requires schema_version=2")
        if not self.site_name.strip():
            raise ValueError("site_name must not be empty")
        if not self.hash_contents:
            raise ValueError("v2 immutable corpus provenance requires content hashes")
        if self.max_reinforcement_rounds < 0:
            raise ValueError("max_reinforcement_rounds must be non-negative")
        if len(self.required_metadata) != len(set(self.required_metadata)):
            raise ValueError("required_metadata must not contain duplicates")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PipelineConfig":
        return cls(
            site_name=str(payload.get("site_name") or ""),
            schema_version=int(payload.get("schema_version", 2)),
            hash_contents=bool(payload.get("hash_contents", True)),
            heldout_patterns=tuple(str(value) for value in payload.get("heldout_patterns") or ()),
            holdout_provenance=str(
                payload.get("holdout_provenance") or "MAPPING_DISJOINT_UNVERIFIED_OUTER"
            ),
            required_metadata=tuple(
                str(value) for value in payload.get("required_metadata", DEFAULT_REQUIRED_METADATA)
            ),
            allow_unknown_metadata=bool(payload.get("allow_unknown_metadata", False)),
            max_reinforcement_rounds=int(payload.get("max_reinforcement_rounds", 2)),
            mapping_optimization=dict(payload.get("mapping_optimization") or {}),
            adapters={
                str(key): dict(value) for key, value in dict(payload.get("adapters") or {}).items()
            },
            thresholds=dict(payload.get("thresholds") or {}),
            resources=dict(payload.get("resources") or {}),
        )

    @classmethod
    def from_toml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("rb") as stream:
            return cls.from_dict(tomllib.load(stream))

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["heldout_patterns"] = list(self.heldout_patterns)
        payload["required_metadata"] = list(self.required_metadata)
        return payload


__all__ = ["CANONICAL_STAGES_V2", "DEFAULT_REQUIRED_METADATA", "PipelineConfig"]
