"""Immutable records and ubiquitous language for the graph-aware site pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class DataStatus(StringEnum):
    RAW = "RAW"
    SANITIZED = "SANITIZED"
    CANDIDATE = "CANDIDATE"
    ACTIVE_CORE = "ACTIVE_CORE"
    ACTIVE_BRIDGE = "ACTIVE_BRIDGE"
    INACTIVE_UPDATE = "INACTIVE_UPDATE"
    INACTIVE_REJECT = "INACTIVE_REJECT"
    UNDECIDED = "UNDECIDED"


class PreSfmRole(StringEnum):
    CORE_CANDIDATE = "CORE_CANDIDATE"
    BRIDGE_CANDIDATE = "BRIDGE_CANDIDATE"
    UPDATE_CANDIDATE = "UPDATE_CANDIDATE"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    UNDECIDED = "UNDECIDED"


class PostSfmRole(StringEnum):
    CORE = "CORE"
    BRIDGE = "BRIDGE"
    UPDATE_ONLY = "UPDATE_ONLY"
    REJECT = "REJECT"


class MappingMode(StringEnum):
    TRIANGULATE = "TRIANGULATE"
    POSE_ONLY = "POSE_ONLY"
    EXCLUDE = "EXCLUDE"


class Inclusion(StringEnum):
    INCLUDE = "INCLUDE"
    EXCLUDE = "EXCLUDE"
    HOLDOUT = "HOLDOUT"
    BLOCKED = "BLOCKED"


class EvaluationRole(StringEnum):
    MAPPING = "MAPPING"
    HOLDOUT = "HOLDOUT"
    ARCHIVE_ONLY = "ARCHIVE_ONLY"


class Risk(StringEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    UNDECIDED = "UNDECIDED"


class Contribution(StringEnum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNDECIDED = "UNDECIDED"


class CandidateFlag(StringEnum):
    LOOP_CLOSURE = "LOOP_CLOSURE"
    COVERAGE_REPRESENTATIVE = "COVERAGE_REPRESENTATIVE"
    WEAK_REGION_ALTERNATE = "WEAK_REGION_ALTERNATE"
    CROSS_VIDEO = "CROSS_VIDEO"
    CROSS_SESSION = "CROSS_SESSION"


class IssueCode(StringEnum):
    MISSING_CONNECTIVITY = "MISSING_CONNECTIVITY"
    RESHOOT_REQUIRED = "RESHOOT_REQUIRED"
    MULTIMODAL_ALIGNMENT = "MULTIMODAL_ALIGNMENT"
    INCONSISTENT_GEOMETRY = "INCONSISTENT_GEOMETRY"
    HOLDOUT_LEAKAGE = "HOLDOUT_LEAKAGE"
    METADATA_INCOMPLETE = "METADATA_INCOMPLETE"


class EdgeAdmission(StringEnum):
    CANDIDATE = "CANDIDATE"
    VERIFIED = "VERIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED = "REJECTED"


def content_id(value: bytes | str | Mapping[str, Any] | Iterable[Any], prefix: str = "") -> str:
    """Return a stable content-derived identifier."""

    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return f"{prefix}_{digest}" if prefix else digest


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    media_type: str | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("artifact uri must not be empty")
        if not self.sha256:
            raise ValueError("artifact sha256 must not be empty")


@dataclass(frozen=True)
class VideoRecord:
    source_id: str
    video_id: str
    source_uri: str
    sha256: str
    source_kind: str = "video"
    evaluation_role: EvaluationRole = EvaluationRole.MAPPING
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SegmentRecord:
    video_id: str
    session_id: str
    segment_id: str
    source_uri: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    status: DataStatus = DataStatus.CANDIDATE
    pre_sfm_role: PreSfmRole = PreSfmRole.UNDECIDED
    candidate_flags: tuple[CandidateFlag, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KeyframeRecord:
    keyframe_id: str
    segment_id: str
    video_id: str
    source_frame_index: int
    source_pts_seconds: float
    image_uri: str | None = None
    status: DataStatus = DataStatus.CANDIDATE
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairEdgeRecord:
    pair_id: str
    image_i: str
    image_j: str
    segment_i: str
    segment_j: str
    admission: EdgeAdmission = EdgeAdmission.CANDIDATE
    retrieval_score: float | None = None
    confidence: float | None = None
    warnings: tuple[str, ...] = ()
    artifact: ArtifactRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleAssignment:
    segment_id: str
    pre_sfm_role: PreSfmRole
    post_sfm_role: PostSfmRole | None
    data_status: DataStatus
    mapping_mode: MappingMode
    base_map: Inclusion
    localization: Inclusion
    risk: Risk
    geometry_contribution: Contribution
    connectivity_contribution: Contribution
    reasons: tuple[str, ...] = ()
    issues: tuple[IssueCode, ...] = ()


def derive_status(role: PostSfmRole | str | None, *, sanitized: bool = True) -> DataStatus:
    if role is None:
        return DataStatus.SANITIZED if sanitized else DataStatus.RAW
    value = PostSfmRole(role)
    return {
        PostSfmRole.CORE: DataStatus.ACTIVE_CORE,
        PostSfmRole.BRIDGE: DataStatus.ACTIVE_BRIDGE,
        PostSfmRole.UPDATE_ONLY: DataStatus.INACTIVE_UPDATE,
        PostSfmRole.REJECT: DataStatus.INACTIVE_REJECT,
    }[value]


def legacy_session_role(
    role: PostSfmRole | str,
    *,
    scene_change: bool = False,
    internally_usable_without_base_edge: bool = False,
) -> str:
    value = PostSfmRole(role)
    if value is PostSfmRole.CORE:
        return "BASE_CORE"
    if value is PostSfmRole.BRIDGE:
        return "BASE_SUPPORT"
    if value is PostSfmRole.UPDATE_ONLY:
        return "UPDATE_CANDIDATE" if scene_change else "APPEARANCE_REF"
    if internally_usable_without_base_edge:
        return "NEW_SUBMAP"
    return "REJECT"


def record_dict(record: Any) -> dict[str, Any]:
    return _json_value(asdict(record))


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "ArtifactRef",
    "CandidateFlag",
    "Contribution",
    "DataStatus",
    "EdgeAdmission",
    "EvaluationRole",
    "Inclusion",
    "IssueCode",
    "KeyframeRecord",
    "MappingMode",
    "PairEdgeRecord",
    "PostSfmRole",
    "PreSfmRole",
    "Risk",
    "RoleAssignment",
    "SegmentRecord",
    "VideoRecord",
    "content_id",
    "derive_status",
    "legacy_session_role",
    "record_dict",
]
