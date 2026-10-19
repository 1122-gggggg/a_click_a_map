"""Conservative sanitation, persistent segmentation and adaptive keyframes."""

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
    warnings: tuple[str, ...] = ()
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
class Segment:
    segment_id: str
    video_id: str
    session_id: str
    frames: tuple[FrameRecord, ...]
    start_seconds: float
    end_seconds: float
    boundary_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectSamplingPolicy:
    probe_fps: float = 4.0
    baseline_fps: float = 1.0
    fast_fps: float = 2.0
    hover_fps: float = 0.2
    rotation_fps: float = 2.0
    min_gap_seconds: float = 0.25


@dataclass(frozen=True)
class PlannedKeyframe:
    frame: FrameRecord
    mapping_mode: str


@dataclass(frozen=True)
class DirectKeyframePlan:
    keyframes: tuple[PlannedKeyframe, ...]
    forced_pairs: tuple[tuple[str, str], ...]


def frame_id(frame: Mapping[str, Any]) -> str:
    return f"{frame['video_id']}:{int(frame['frame_index']):08d}"


def sanitize_frames(frames: Iterable[Mapping[str, Any]]) -> SanitizationResult:
    """Remove only near-certain garbage; preserve all other frames as candidates."""

    kept: list[FrameRecord] = []
    removed: list[RemovedFrame] = []
    for raw in frames:
        identifier = frame_id(raw)
        reason = _hard_reject_reason(raw)
        if reason is not None:
            removed.append(RemovedFrame(identifier, int(raw["frame_index"]), reason, raw))
            continue
        warnings: list[str] = []
        if _number(raw.get("parallax"), default=999.0) < 1.0:
            warnings.append("low_parallax")
        if _number(raw.get("retrieval_score"), default=1.0) < 0.1:
            warnings.append("low_retrieval")
        blur = _number(raw.get("blur_variance", raw.get("blur_score")), default=100.0)
        if 2.0 <= blur < 25.0:
            warnings.append("ordinary_blur")
        if raw.get("motion_class") == "unproven":
            warnings.append("unproven_geometry")
        if raw.get("motion_class") == "pure_rotation":
            warnings.append("rotation_dominated")
        kept.append(
            FrameRecord(
                identifier,
                str(raw["video_id"]),
                str(raw.get("session_id") or ""),
                int(raw["frame_index"]),
                float(raw.get("timestamp", raw["frame_index"])),
                "CANDIDATE",
                tuple(warnings),
                dict(raw),
            )
        )
    return SanitizationResult(
        tuple(kept),
        tuple(removed),
        tuple(record.frame_id for record in kept),
    )


def _hard_reject_reason(frame: Mapping[str, Any]) -> str | None:
    boolean_checks = (
        ("corrupt", "corrupt"),
        ("severe_blur", "severe_blur"),
        ("severe_motion_blur", "severe_motion_blur"),
        ("exposure_failed", "exposure_failed"),
        ("duplicate_previous", "duplicate"),
        ("invalid_segment", "invalid_segment"),
    )
    for key, reason in boolean_checks:
        if frame.get(key) is True:
            return reason
    fraction_checks = (
        ("black_fraction", 0.98, "mostly_black"),
        ("white_fraction", 0.98, "mostly_white"),
        ("dynamic_fraction", 0.95, "dynamic_occlusion"),
    )
    for key, threshold, reason in fraction_checks:
        value = frame.get(key)
        if value is not None and _number(value) >= threshold:
            return reason
    return None


def split_segments(
    frames: Iterable[FrameRecord | Mapping[str, Any]],
    *,
    max_gap: float = 1.0,
    min_segment_frames: int = 2,
) -> tuple[Segment, ...]:
    """Split one ordered video and smooth isolated one-frame change spikes."""

    if max_gap <= 0 or min_segment_frames <= 0:
        raise ValueError("max_gap and min_segment_frames must be positive")
    records = _records(frames)
    if not records:
        return ()
    records.sort(key=lambda row: (row.timestamp, row.frame_index))
    video_ids = {row.video_id for row in records}
    if len(video_ids) != 1:
        raise ValueError("split_segments accepts one video at a time")
    chunks: list[list[FrameRecord]] = [[records[0]]]
    reasons: list[list[str]] = [[]]
    for previous, current in zip(records[:-1], records[1:], strict=True):
        boundary_reasons = _boundary_reasons(previous, current, max_gap)
        if boundary_reasons:
            chunks.append([])
            reasons.append(boundary_reasons)
        chunks[-1].append(current)
    chunks, reasons = _smooth_short_segments(chunks, reasons, min_segment_frames)
    return tuple(
        Segment(
            f"{records[0].video_id}:S{index:02d}",
            records[0].video_id,
            records[0].session_id,
            tuple(chunk),
            chunk[0].timestamp,
            chunk[-1].timestamp,
            tuple(reasons[index - 1]),
        )
        for index, chunk in enumerate(chunks, 1)
    )


def _boundary_reasons(
    previous: FrameRecord,
    current: FrameRecord,
    max_gap: float,
) -> list[str]:
    found: list[str] = []
    if current.timestamp - previous.timestamp > max_gap:
        found.append("temporal_gap")
    previous_scene, current_scene = previous.source.get("scene_id"), current.source.get("scene_id")
    if previous_scene is not None and current_scene is not None and previous_scene != current_scene:
        found.append("scene_change")
    previous_class = previous.source.get("motion_class")
    current_class = current.source.get("motion_class")
    if (
        previous_class
        and current_class
        and _motion_regime(previous_class) != _motion_regime(current_class)
    ):
        found.append("motion_regime_change")
    if current.source.get("scene_cut") is True or current.source.get("turn_event") is True:
        found.append("event_boundary")
    return found


def _motion_regime(motion_class: Any) -> str:
    """Map sampling-level motion classes onto persistent segment regimes."""

    value = str(motion_class)
    return {
        "parallax": "geometry",
        "fast_motion": "geometry",
        "low_parallax": "weak_geometry",
        "pure_rotation": "rotation",
        "hover": "static",
        "unproven": "unproven",
    }.get(value, value)


def _smooth_short_segments(
    chunks: list[list[FrameRecord]],
    reasons: list[list[str]],
    minimum: int,
) -> tuple[list[list[FrameRecord]], list[list[str]]]:
    material = [(list(chunk), list(reason)) for chunk, reason in zip(chunks, reasons, strict=True)]
    index = 0
    while len(material) > 1 and index < len(material):
        chunk, _ = material[index]
        if len(chunk) >= minimum:
            index += 1
            continue
        if index + 1 < len(material) and len(material[index + 1][0]) < minimum:
            material[index][0].extend(material[index + 1][0])
            material.pop(index + 1)
        elif index == 0:
            material[1][0][:0] = chunk
            material[1][1][:] = material[0][1]
            material.pop(0)
        else:
            material[index - 1][0].extend(chunk)
            material.pop(index)
            index -= 1
    return [row[0] for row in material], [row[1] for row in material]


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
    """Greedily retain boundaries and increase sampling for turns/novelty."""

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
            or (source.get("turn_event") is True and motion_class != "pure_rotation")
            or source.get("scene_cut") is True
            or source.get("segment_boundary") is True
        )
        if motion_class == "pure_rotation":
            rate = rotation
        elif (
            motion_class == "fast_motion"
            or (motion_class is None and motion >= motion_threshold)
            or novelty >= novelty_threshold
        ):
            rate = fast
        elif (
            motion_class == "hover"
            or source.get("near_duplicate") is True
            or source.get("static") is True
        ):
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


def plan_direct_keyframes(
    frames: Iterable[FrameRecord | Mapping[str, Any]],
    *,
    policy: DirectSamplingPolicy | None = None,
) -> DirectKeyframePlan:
    """Plan all-in-one inputs for one video without graph/session selection."""

    settings = policy or DirectSamplingPolicy()
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
                "POSE_ONLY" if row.source.get("motion_class") == "pure_rotation" else "TRIANGULATE"
            ),
        )
        for row in selected
    )
    forced = tuple(
        (left.frame.frame_id, right.frame.frame_id)
        for left, right in zip(planned[:-1], planned[1:], strict=True)
        if "POSE_ONLY" in {left.mapping_mode, right.mapping_mode}
    )
    return DirectKeyframePlan(planned, forced)


def _records(frames: Iterable[FrameRecord | Mapping[str, Any]]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for item in frames:
        if isinstance(item, FrameRecord):
            records.append(item)
            continue
        result = sanitize_frames([item])
        if result.kept:
            records.append(result.kept[0])
    return records


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "DirectKeyframePlan",
    "DirectSamplingPolicy",
    "FrameRecord",
    "PlannedKeyframe",
    "RemovedFrame",
    "SanitizationResult",
    "Segment",
    "adaptive_keyframes",
    "frame_id",
    "plan_direct_keyframes",
    "sanitize_frames",
    "split_segments",
]
