"""Conservative sanitation and adaptive keyframe planning.

Ports the proven mechanism: hard-reject only near-certain garbage
(corrupt/blur/exposure/duplicates), keep everything else as candidates,
then greedily select keyframes with motion-regime-aware sampling rates.
Pure-rotation frames are marked POSE_ONLY (pose-but-no-triangulation) and
bridge-forced to their neighbors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    video_id: str
    session_id: str
    frame_index: int
    timestamp: float
    status: str
    warnings: tuple[str, ...]
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RemovedFrame:
    frame_id: str
    frame_index: int
    reason: str
    source: Mapping[str, Any]


@dataclass(frozen=True)
class SanitizationResult:
    kept: tuple[FrameRecord, ...]
    removed: tuple[RemovedFrame, ...]
    candidate_pool: tuple[str, ...]


@dataclass(frozen=True)
class SamplingPolicy:
    probe_fps: float = 4.0
    baseline_fps: float = 1.0
    fast_fps: float = 2.0
    hover_fps: float = 0.2
    rotation_fps: float = 2.0
    min_gap_seconds: float = 0.25

    def validate(self) -> None:
        for name in (
            "probe_fps", "baseline_fps", "fast_fps",
            "hover_fps", "rotation_fps",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if float(self.min_gap_seconds) < 0:
            raise ValueError("min_gap_seconds must be non-negative")


@dataclass(frozen=True)
class PlannedKeyframe:
    frame: FrameRecord
    mapping_mode: str


@dataclass(frozen=True)
class KeyframePlan:
    keyframes: tuple[PlannedKeyframe, ...]
    forced_pairs: tuple[tuple[str, str], ...]


def frame_id(frame: Mapping[str, Any]) -> str:
    return f"{frame['video_id']}:{int(frame['frame_index']):08d}"


def sanitize_frames(frames: Iterable[Mapping[str, Any]]) -> SanitizationResult:
    """Remove only near-certain garbage; preserve the rest as candidates."""
    kept: list[FrameRecord] = []
    removed: list[RemovedFrame] = []
    for raw in frames:
        identifier = frame_id(raw)
        reason = _hard_reject_reason(raw)
        if reason is not None:
            removed.append(RemovedFrame(
                identifier, int(raw["frame_index"]), reason, raw))
            continue
        warnings: list[str] = []
        if _number(raw.get("parallax"), default=999.0) < 1.0:
            warnings.append("low_parallax")
        blur = _number(raw.get("blur_variance", raw.get("blur_score")),
                       default=100.0)
        if 2.0 <= blur < 25.0:
            warnings.append("ordinary_blur")
        if raw.get("motion_class") == "unproven":
            warnings.append("unproven_geometry")
        if raw.get("motion_class") == "pure_rotation":
            warnings.append("rotation_dominated")
        kept.append(FrameRecord(
            identifier, str(raw["video_id"]), str(raw.get("session_id") or ""),
            int(raw["frame_index"]),
            float(raw.get("timestamp", raw["frame_index"])),
            "CANDIDATE", tuple(warnings), dict(raw),
        ))
    return SanitizationResult(
        tuple(kept), tuple(removed),
        tuple(record.frame_id for record in kept),
    )


def _hard_reject_reason(frame: Mapping[str, Any]) -> str | None:
    for key, reason in (
        ("corrupt", "corrupt"),
        ("severe_blur", "severe_blur"),
        ("severe_motion_blur", "severe_motion_blur"),
        ("exposure_failed", "exposure_failed"),
        ("duplicate_previous", "duplicate"),
        ("invalid_segment", "invalid_segment"),
    ):
        if frame.get(key) is True:
            return reason
    for key, threshold, reason in (
        ("black_fraction", 0.98, "mostly_black"),
        ("white_fraction", 0.98, "mostly_white"),
        ("dynamic_fraction", 0.95, "dynamic_occlusion"),
    ):
        value = frame.get(key)
        if value is not None and _number(value) >= threshold:
            return reason
    return None


def adaptive_keyframes(
    frames: Iterable[FrameRecord | Mapping[str, Any]],
    *,
    baseline_fps: float = 1.0,
    min_gap: float = 0.5,
    fast_fps: float = 2.0,
    static_fps: float = 0.5,
    rotation_fps: float | None = None,
    motion_threshold: float = 2.0,
    novelty_threshold: float = 0.8,
) -> tuple[FrameRecord, ...]:
    """Greedily retain boundaries; sample turns/novelty more densely."""
    if min_gap < 0:
        raise ValueError("min_gap must be non-negative")
    baseline = min(2.0, max(0.5, baseline_fps))
    fast = min(2.0, max(baseline, fast_fps))
    static = min(baseline, max(0.05, static_fps))
    rotation = fast if rotation_fps is None else min(fast, max(0.05, rotation_fps))
    records = _records(frames)
    if not records:
        return ()
    records.sort(key=lambda row: (row.timestamp, row.frame_index))
    chosen = [records[0]]
    for index, current in enumerate(records[1:], 1):
        source = current.source
        last = chosen[-1]
        novelty = _number(source.get("appearance_novelty"), default=0.0)
        motion = _number(source.get("motion"), default=0.0)
        motion_class = source.get("motion_class")
        is_last = index == len(records) - 1
        mandatory = bool(
            is_last
            or novelty >= novelty_threshold
            or (source.get("turn_event") is True
                and motion_class != "pure_rotation")
            or source.get("scene_cut") is True
            or source.get("segment_boundary") is True
        )
        if motion_class == "pure_rotation":
            rate = rotation
        elif (motion_class == "fast_motion"
                or (motion_class is None and motion >= motion_threshold)
                or novelty >= novelty_threshold):
            rate = fast
        elif (motion_class == "hover"
                or source.get("near_duplicate") is True
                or source.get("static") is True):
            rate = static
        else:
            rate = baseline
        interval = max(min_gap, 1.0 / rate)
        elapsed = current.timestamp - last.timestamp
        if mandatory:
            chosen.append(current)
        elif elapsed + 1e-9 >= interval:
            chosen.append(current)
    if chosen[-1].frame_id != records[-1].frame_id:
        chosen.append(records[-1])
    return tuple(chosen)


def plan_keyframes(
    frames: Iterable[FrameRecord | Mapping[str, Any]],
    *,
    policy: SamplingPolicy | None = None,
) -> KeyframePlan:
    """Plan all-sequence keyframes for one video (no site selection)."""
    settings = policy or SamplingPolicy()
    selected = adaptive_keyframes(
        frames,
        baseline_fps=settings.baseline_fps,
        fast_fps=settings.fast_fps,
        static_fps=settings.hover_fps,
        rotation_fps=settings.rotation_fps,
        min_gap=settings.min_gap_seconds,
    )
    planned = tuple(
        PlannedKeyframe(
            frame=row,
            mapping_mode=(
                "POSE_ONLY"
                if row.source.get("motion_class") == "pure_rotation"
                else "TRIANGULATE"
            ),
        )
        for row in selected
    )
    forced = tuple(
        (left.frame.frame_id, right.frame.frame_id)
        for left, right in zip(planned[:-1], planned[1:], strict=True)
        if "POSE_ONLY" in {left.mapping_mode, right.mapping_mode}
    )
    return KeyframePlan(planned, forced)


def _records(frames: Iterable[FrameRecord | Mapping[str, Any]]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for item in frames:
        if isinstance(item, FrameRecord):
            records.append(item)
        else:
            result = sanitize_frames([item])
            if result.kept:
                records.append(result.kept[0])
    return records


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "FrameRecord",
    "KeyframePlan",
    "PlannedKeyframe",
    "RemovedFrame",
    "SamplingPolicy",
    "SanitizationResult",
    "adaptive_keyframes",
    "frame_id",
    "plan_keyframes",
    "sanitize_frames",
]
