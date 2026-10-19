"""PTS-driven, bridge-aware native-frame sampling for the base-only B0 corpus."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    BASE_VIDEO_NAMES,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    write_native_frame,
)
from river_map_quality.rotation_motion import (
    aggregate_temporal_rotation_diagnostics,
    inlier_correspondences,
    rotation_only_diagnostics,
)

MOTION_PROFILE: dict[str, dict[str, object]] = {
    "parallax": {"hz": 1.0, "role": "triangulation"},
    "low_parallax": {"hz": 1.0, "role": "triangulation", "explicitly_weak": True},
    "hover": {"hz": 0.2, "role": "triangulation", "maximum_fraction": 0.02},
    "pure_rotation": {"hz": 0.5, "role": "bridge_only"},
    "unproven": {"hz": 0.5, "role": "bridge_only"},
    "fast_motion": {"hz": 0.0, "role": "dropped"},
}

MOTION_THRESHOLDS = {
    "analysis_interval_seconds": 0.5,
    "minimum_tracked_features": 12,
    "hover_flow_median_px": 0.8,
    "fast_motion_flow_median_px": 35.0,
    "fast_motion_blur_variance": 25.0,
    "pure_rotation_min_degrees": 0.35,
    "pure_rotation_max_parallax_px": 0.75,
    "pure_rotation_homography_to_essential_ratio": 0.9,
    "pure_rotation_beta_p50_deg": 0.5,
    "pure_rotation_beta_p90_deg": 1.0,
    "pure_rotation_max_high_beta_grid_fraction": 0.1,
    "pure_rotation_min_temporal_baseline_seconds": 0.75,
    "pure_rotation_min_rate_deg_s": 0.5,
    "pure_rotation_min_duration_seconds": 0.75,
    "pure_rotation_segment_min_duration_seconds": 1.0,
    "adaptive_minimum_samples": 8,
    "minimum_essential_inliers": 20,
    "low_parallax_max_px": 2.0,
    "feature_max_corners": 1500,
    "feature_quality_level": 0.01,
    "feature_min_distance_px": 8.0,
    "ransac_threshold_px": 2.0,
}


class SamplingInvariantError(RuntimeError):
    """Raised when immutable sampling conditions cannot be satisfied safely."""


def derive_motion_thresholds(
    rows: Iterable[Mapping[str, object]],
    base: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Robust cohort overlay for soft motion targets; structural gates remain fixed."""

    thresholds = dict(MOTION_THRESHOLDS if base is None else base)
    material = list(rows)
    minimum = int(thresholds.get("adaptive_minimum_samples", 8))

    def values(*keys: str) -> np.ndarray:
        found: list[float] = []
        for row in material:
            value = next(
                (row.get(key) for key in keys if row.get(key) is not None),
                None,
            )
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                found.append(number)
        return np.asarray(found, dtype=float)

    beta50 = values("derotated_beta_p50_deg")
    beta90 = values("derotated_beta_p90_deg")
    rates = values("rotation_rate_p50_deg_s", "rotation_rate_deg_s")
    if len(beta50) >= minimum:
        thresholds["pure_rotation_beta_p50_deg"] = float(
            np.clip(np.percentile(beta50, 25), 0.1, 1.0)
        )
    if len(beta90) >= minimum:
        thresholds["pure_rotation_beta_p90_deg"] = float(
            np.clip(np.percentile(beta90, 25), 0.2, 2.0)
        )
    if len(rates) >= minimum:
        thresholds["pure_rotation_min_rate_deg_s"] = float(
            np.clip(np.percentile(rates, 25), 0.1, 10.0)
        )
    return thresholds


def smooth_short_motion_segments(
    rows: Iterable[Mapping[str, object]],
    *,
    minimum_duration_seconds: float,
) -> list[dict[str, object]]:
    """Suppress isolated pure-rotation spikes without erasing sustained turns."""

    result = [dict(row) for row in rows]
    minimum = float(minimum_duration_seconds)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_duration_seconds must be finite and non-negative")
    index = 0
    while index < len(result):
        if result[index].get("motion_class") != "pure_rotation":
            index += 1
            continue
        start = index
        while index + 1 < len(result) and result[index + 1].get(
            "motion_class"
        ) == "pure_rotation":
            index += 1
        end = index
        interval_start = result[start].get("interval_start_pts_seconds")
        start_time = float(
            interval_start
            if interval_start is not None
            else result[start].get("source_pts_seconds", start)
        )
        interval_end = result[end].get("interval_end_pts_seconds")
        if interval_end is not None:
            end_time = float(interval_end)
        elif end + 1 < len(result):
            end_time = float(result[end + 1].get("source_pts_seconds", end + 1))
        else:
            end_time = float(result[end].get("source_pts_seconds", end))
        duration = max(0.0, end_time - start_time)
        if duration < minimum:
            # A short rotation-only interval has not demonstrated translation.
            # Smoothing may suppress a spike, but must not promote bridge-only
            # evidence into a triangulation role merely because its neighbours
            # are stronger.
            replacement = "unproven"
            for offset in range(start, end + 1):
                result[offset]["raw_motion_class"] = "pure_rotation"
                result[offset]["motion_class"] = replacement
                result[offset]["role"] = MOTION_PROFILE[replacement]["role"]
                result[offset]["decision_reason"] = (
                    "short_pure_rotation_segment_smoothed"
                )
        index += 1
    return result


def classify_motion(
    evidence: Mapping[str, object],
    *,
    thresholds: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify one measured interval. Ambiguous geometry is bridge-only by default."""

    limits = dict(MOTION_THRESHOLDS)
    if thresholds is not None:
        limits.update(thresholds)
    tracked = int(evidence.get("tracked_count") or 0)
    flow = float(evidence.get("flow_median_px") or 0.0)
    blur = float(evidence.get("blur_variance") or 0.0)
    homography = int(evidence.get("homography_inliers") or 0)
    essential = int(evidence.get("essential_inliers") or 0)
    rotation = float(evidence.get("rotation_degrees") or 0.0)
    parallax = float(evidence.get("parallax_px") or 0.0)
    temporal_rotation = str(
        evidence.get("temporal_rotation_classification") or "UNAVAILABLE"
    )
    beta_p50 = evidence.get("derotated_beta_p50_deg")
    beta_p90 = evidence.get("derotated_beta_p90_deg")
    high_beta_grid_fraction = evidence.get("high_beta_grid_fraction")
    beta_complete = all(
        value is not None and np.isfinite(float(value))
        for value in (beta_p50, beta_p90, high_beta_grid_fraction)
    )
    rotation_rate = evidence.get("rotation_rate_p50_deg_s")
    rotation_duration = evidence.get("rotation_dominated_duration_seconds")
    verified_rotation = (
        evidence.get("rotation_correspondence_source") == "essential_pose_inliers"
    )
    rotation_persistent = all(
        value is not None and np.isfinite(float(value))
        for value in (rotation_rate, rotation_duration)
    ) and (
        float(rotation_rate) >= float(limits["pure_rotation_min_rate_deg_s"])
        and float(rotation_duration)
        >= float(limits["pure_rotation_min_duration_seconds"])
    )
    if blur < float(limits["fast_motion_blur_variance"]) and flow >= float(
        limits["fast_motion_flow_median_px"]
    ):
        motion_class = "fast_motion"
        reason = "high_flow_with_low_blur_variance"
    elif tracked >= int(limits["minimum_tracked_features"]) and flow <= float(
        limits["hover_flow_median_px"]
    ):
        motion_class = "hover"
        reason = "stable_feature_flow"
    elif tracked < int(limits["minimum_tracked_features"]):
        motion_class = "unproven"
        reason = "insufficient_tracked_features"
    elif temporal_rotation == "TRANSLATION_EVIDENCE":
        motion_class = "parallax"
        reason = "long_baseline_derotated_translation_evidence"
    elif beta_complete and temporal_rotation == "ROTATION_DOMINATED" and (
        essential >= int(limits["minimum_essential_inliers"])
        and verified_rotation
        and rotation >= float(limits["pure_rotation_min_degrees"])
        and rotation_persistent
        and float(beta_p50) <= float(limits["pure_rotation_beta_p50_deg"])
        and float(beta_p90) <= float(limits["pure_rotation_beta_p90_deg"])
        and float(high_beta_grid_fraction)
        <= float(limits["pure_rotation_max_high_beta_grid_fraction"])
        and homography
        >= float(limits["pure_rotation_homography_to_essential_ratio"])
        * max(essential, 1)
    ):
        motion_class = "pure_rotation"
        reason = "long_baseline_rotation_fit_with_negligible_derotated_parallax"
    elif essential < int(limits["minimum_essential_inliers"]):
        motion_class = "unproven"
        reason = "insufficient_essential_inliers"
    elif parallax <= float(limits["low_parallax_max_px"]):
        motion_class = "low_parallax"
        reason = "essential_geometry_with_low_parallax"
    else:
        motion_class = "parallax"
        reason = "essential_geometry_with_usable_parallax"
    profile = MOTION_PROFILE[motion_class]
    return {
        "motion_class": motion_class,
        "role": profile["role"],
        "decision_thresholds": limits,
        "decision_reason": reason,
    }


def _motion_evidence(
    previous: np.ndarray, current: np.ndarray, camera_matrix: np.ndarray
) -> dict[str, object]:
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(MOTION_THRESHOLDS["feature_max_corners"]),
        qualityLevel=float(MOTION_THRESHOLDS["feature_quality_level"]),
        minDistance=float(MOTION_THRESHOLDS["feature_min_distance_px"]),
        blockSize=7,
    )
    blur = float(cv2.Laplacian(current, cv2.CV_64F).var())
    if corners is None or len(corners) == 0:
        return {
            "tracked_count": 0,
            "flow_median_px": 0.0,
            "blur_variance": blur,
            "homography_inliers": 0,
            "essential_inliers": 0,
            "rotation_degrees": 0.0,
            "parallax_px": 0.0,
        }
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        corners,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return {
            "tracked_count": 0,
            "flow_median_px": 0.0,
            "blur_variance": blur,
            "homography_inliers": 0,
            "essential_inliers": 0,
            "rotation_degrees": 0.0,
            "parallax_px": 0.0,
        }
    valid = status.reshape(-1).astype(bool)
    previous_xy = corners.reshape(-1, 2)[valid].astype(np.float64)
    current_xy = tracked.reshape(-1, 2)[valid].astype(np.float64)
    count = len(previous_xy)
    if not count:
        return {
            "tracked_count": 0,
            "flow_median_px": 0.0,
            "blur_variance": blur,
            "homography_inliers": 0,
            "essential_inliers": 0,
            "rotation_degrees": 0.0,
            "parallax_px": 0.0,
        }
    flow = np.linalg.norm(current_xy - previous_xy, axis=1)
    homography = None
    homography_mask = None
    if count >= 4:
        homography, homography_mask = cv2.findHomography(
            previous_xy,
            current_xy,
            cv2.RANSAC,
            float(MOTION_THRESHOLDS["ransac_threshold_px"]),
        )
    homography_inliers = int(homography_mask.sum()) if homography_mask is not None else 0
    parallax = float(np.median(flow))
    if homography is not None and homography_mask is not None and homography_inliers:
        predicted = cv2.perspectiveTransform(
            previous_xy.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        residual = np.linalg.norm(current_xy - predicted, axis=1)
        parallax = float(np.median(residual[homography_mask.reshape(-1).astype(bool)]))
    essential = None
    essential_mask = None
    rotation_degrees = 0.0
    if count >= 5:
        try:
            essential, essential_mask = cv2.findEssentialMat(
                previous_xy,
                current_xy,
                camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=float(MOTION_THRESHOLDS["ransac_threshold_px"]),
            )
            if essential is not None:
                _, rotation, _, pose_mask = cv2.recoverPose(
                    essential[:3, :3], previous_xy, current_xy, camera_matrix, mask=essential_mask
                )
                essential_mask = pose_mask
                cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
                rotation_degrees = float(np.degrees(np.arccos(cosine)))
        except cv2.error:
            essential_mask = None
    essential_inliers = int(essential_mask.sum()) if essential_mask is not None else 0
    rotation_diagnostics: dict[str, object] = {}
    if count >= 3:
        rotation_previous, rotation_current = previous_xy, current_xy
        rotation_source = "tracked_features_fallback"
        if essential_mask is not None:
            try:
                rotation_previous, rotation_current = inlier_correspondences(
                    previous_xy,
                    current_xy,
                    essential_mask,
                    minimum_count=3,
                )
                rotation_source = "essential_pose_inliers"
            except ValueError:
                pass
        try:
            rotation_diagnostics = rotation_only_diagnostics(
                rotation_previous,
                rotation_current,
                camera_matrix,
                (previous.shape[1], previous.shape[0]),
                translation_threshold_deg=float(
                    MOTION_THRESHOLDS["pure_rotation_beta_p90_deg"]
                ),
            )
            rotation_diagnostics["rotation_correspondence_source"] = rotation_source
        except (FloatingPointError, ValueError):
            rotation_diagnostics = {"classification": "UNAVAILABLE"}
    homography_dominant = bool(
        homography_inliers
        >= float(MOTION_THRESHOLDS["pure_rotation_homography_to_essential_ratio"])
        * max(essential_inliers, 1)
    )
    return {
        "tracked_count": count,
        "flow_median_px": float(np.median(flow)),
        "blur_variance": blur,
        "homography_inliers": homography_inliers,
        "essential_inliers": essential_inliers,
        "rotation_degrees": rotation_degrees,
        "parallax_px": parallax,
        "homography_dominant": homography_dominant,
        **rotation_diagnostics,
    }


def scan_motion(
    video: Path, *, camera_matrix: np.ndarray, analysis_interval_seconds: float
) -> list[dict[str, object]]:
    """Measure native-resolution interval geometry using decoded presentation timestamps."""

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    previous_gray: np.ndarray | None = None
    previous_pts: float | None = None
    next_sample_pts = 0.0
    frame_index = 0
    rows: list[dict[str, object]] = []
    history: list[tuple[np.ndarray, float, int]] = []
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(decoded_fps) or decoded_fps <= 0.0:
        decoded_fps = 24.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            pts_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if pts_seconds + 1e-9 < next_sample_pts:
                frame_index += 1
                continue
            if frame.shape[:2] != (NATIVE_HEIGHT, NATIVE_WIDTH):
                raise SamplingInvariantError(
                    f"decoded non-native frame {frame.shape[1]}x{frame.shape[0]} from {video}"
                )
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            evidence: dict[str, object] = (
                _motion_evidence(previous_gray, current_gray, camera_matrix)
                if previous_gray is not None
                else {
                    "tracked_count": 0,
                    "flow_median_px": 0.0,
                    "blur_variance": float(cv2.Laplacian(current_gray, cv2.CV_64F).var()),
                    "homography_inliers": 0,
                    "essential_inliers": 0,
                    "rotation_degrees": 0.0,
                    "parallax_px": 0.0,
                }
            )
            temporal_rows: list[dict[str, object]] = []
            if history:
                temporal_rows.append(
                    {
                        **evidence,
                        "gap": frame_index - history[-1][2],
                        "fps": decoded_fps,
                        "baseline_seconds": pts_seconds - history[-1][1],
                    }
                )
            if len(history) >= 2:
                longer = _motion_evidence(
                    history[-2][0], current_gray, camera_matrix
                )
                temporal_rows.append(
                    {
                        **longer,
                        "gap": frame_index - history[-2][2],
                        "fps": decoded_fps,
                        "baseline_seconds": pts_seconds - history[-2][1],
                    }
                )
            temporal = aggregate_temporal_rotation_diagnostics(
                temporal_rows,
                fps=decoded_fps,
                min_baseline_seconds=float(
                    MOTION_THRESHOLDS["pure_rotation_min_temporal_baseline_seconds"]
                ),
                translation_threshold_deg=float(
                    MOTION_THRESHOLDS["pure_rotation_beta_p90_deg"]
                ),
            )
            evidence.update(
                {
                    "temporal_rotation_classification": temporal["classification"],
                    "temporal_long_baseline_count": temporal["long_baseline_count"],
                    "derotated_beta_p50_deg": temporal["global_beta_p50_deg"],
                    "derotated_beta_p90_deg": temporal["global_beta_p90_deg"],
                    "high_beta_grid_fraction": temporal["high_beta_grid_fraction"],
                    "rotation_rate_p50_deg_s": temporal[
                        "rotation_rate_p50_deg_s"
                    ],
                    "rotation_dominated_duration_seconds": temporal[
                        "rotation_dominated_duration_seconds"
                    ],
                }
            )
            rows.append(
                {
                    "source_pts": round(pts_seconds * 1_000_000),
                    "source_pts_seconds": pts_seconds,
                    "source_frame_index": frame_index,
                    "interval_start_pts_seconds": previous_pts,
                    "interval_end_pts_seconds": pts_seconds,
                    **evidence,
                }
            )
            previous_gray = current_gray
            previous_pts = pts_seconds
            history.append((current_gray, pts_seconds, frame_index))
            history = history[-3:]
            next_sample_pts = pts_seconds + analysis_interval_seconds
            frame_index += 1
    finally:
        capture.release()
    if len(rows) < 2:
        raise SamplingInvariantError(f"insufficient decoded PTS intervals in {video}")
    adaptive_thresholds = derive_motion_thresholds(rows)
    rows = [
        {**row, **classify_motion(row, thresholds=adaptive_thresholds)}
        for row in rows
    ]
    rows = smooth_short_motion_segments(
        rows,
        minimum_duration_seconds=float(
            adaptive_thresholds["pure_rotation_segment_min_duration_seconds"]
        ),
    )
    segment_id = 0
    previous_class: str | None = None
    for row in rows:
        motion_class = str(row["motion_class"])
        if previous_class is not None and motion_class != previous_class:
            segment_id += 1
        row["segment_id"] = segment_id
        previous_class = motion_class
    return rows


def _segment_rows(intervals: Iterable[Mapping[str, object]]) -> list[list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for raw in intervals:
        row = dict(raw)
        row.setdefault("source_pts", round(float(row["source_pts_seconds"]) * 1_000_000))
        grouped[(str(row["video"]), int(row["segment_id"]))].append(row)
    result: list[list[dict[str, object]]] = []
    for key in sorted(grouped):
        rows = sorted(
            grouped[key],
            key=lambda row: (float(row["source_pts_seconds"]), int(row["source_frame_index"])),
        )
        motion_classes = {str(row["motion_class"]) for row in rows}
        if len(motion_classes) != 1:
            raise ValueError(f"motion segment {key} contains multiple classes")
        result.append(rows)
    return result


def select_sample_frames(
    intervals: Iterable[Mapping[str, object]],
    *,
    profile: Mapping[str, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Select source frames from PTS grids while preserving non-fast segment boundaries."""

    motion_profile = profile or MOTION_PROFILE
    direct: list[dict[str, object]] = []
    hover_candidates: list[dict[str, object]] = []
    mandatory_hover_keys: set[tuple[str, int, int]] = set()
    for segment in _segment_rows(intervals):
        motion_class = str(segment[0]["motion_class"])
        class_profile = motion_profile[motion_class]
        rate = float(class_profile["hz"])
        if rate == 0.0:
            continue
        period = 1.0 / rate
        selected_indices = {0, len(segment) - 1}
        next_pts = float(segment[0]["source_pts_seconds"])
        for index, row in enumerate(segment):
            current_pts = float(row["source_pts_seconds"])
            if current_pts + 1e-9 >= next_pts:
                selected_indices.add(index)
                while next_pts <= current_pts + 1e-9:
                    next_pts += period
        selected = []
        for index in sorted(selected_indices):
            row = dict(segment[index])
            row["role"] = class_profile["role"]
            row["boundary"] = index in {0, len(segment) - 1}
            selected.append(row)
        if motion_class == "hover":
            hover_candidates.extend(selected)
            for row in selected:
                if row["boundary"]:
                    mandatory_hover_keys.add(
                        (str(row["video"]), int(row["source_pts"]), int(row["source_frame_index"]))
                    )
        else:
            direct.extend(selected)
    non_hover_count = len(direct)
    hover_fraction = float(motion_profile["hover"]["maximum_fraction"])
    maximum_hover = int(hover_fraction * non_hover_count / (1.0 - hover_fraction))
    mandatory_hover = [
        row
        for row in hover_candidates
        if (str(row["video"]), int(row["source_pts"]), int(row["source_frame_index"]))
        in mandatory_hover_keys
    ]
    if len(mandatory_hover) > maximum_hover:
        raise SamplingInvariantError(
            "hover boundary retention conflicts with the locked hover cap; "
            "capture more usable parallax"
        )
    selected_hover = sorted(
        hover_candidates,
        key=lambda row: (
            not bool(row["boundary"]),
            float(row["source_pts_seconds"]),
            int(row["source_frame_index"]),
        ),
    )[:maximum_hover]
    selected = direct + selected_hover
    selected.sort(
        key=lambda row: (
            str(row["video"]),
            float(row["source_pts_seconds"]),
            int(row["source_frame_index"]),
        )
    )
    return selected


def validate_frame_records(
    records: Iterable[Mapping[str, object]],
    *,
    profile: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Validate image identity, role, and hover constraints before committing extraction."""

    motion_profile = profile or MOTION_PROFILE
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("frame manifest cannot be empty")
    identities: set[tuple[str, int, int]] = set()
    names: set[str] = set()
    hover_count = 0
    for row in rows:
        identity = (
            str(row["video_sha256"]),
            int(row["source_pts"]),
            int(row["source_frame_index"]),
        )
        if identity in identities:
            raise ValueError(f"duplicate frame identity: {identity}")
        identities.add(identity)
        output_name = str(row["output_name"])
        if output_name in names:
            raise ValueError(f"duplicate output name: {output_name}")
        names.add(output_name)
        motion_class = str(row["motion_class"])
        role = str(row["role"])
        expected_role = str(motion_profile[motion_class]["role"])
        if role != expected_role:
            raise ValueError(f"motion class {motion_class} must be {expected_role}, not {role}")
        if role == "dropped":
            raise ValueError("dropped fast-motion frames cannot be extracted")
        if motion_class == "hover":
            hover_count += 1
        if not row.get("image_sha256"):
            raise ValueError(f"missing image hash for {output_name}")
    if hover_count / len(rows) > float(motion_profile["hover"]["maximum_fraction"]):
        raise ValueError("hover frame fraction exceeds locked cap")



def verify_extracted_frames(
    *, image_root: Path, records: Iterable[Mapping[str, object]]
) -> int:
    """Re-hash and decode every committed image before mapping can consume it."""

    count = 0
    for record in records:
        relative = Path(str(record["output_name"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SamplingInvariantError(f"unsafe extracted image path: {relative}")
        path = image_root / relative
        if path.is_symlink() or not path.is_file():
            raise SamplingInvariantError(f"missing extracted image: {path}")
        fingerprint = fingerprint_file(path, sha256=True)
        if fingerprint.sha256 != record["image_sha256"]:
            raise SamplingInvariantError(f"image hash mismatch: {relative}")
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if pixels is None or pixels.shape[:2] != (int(record["height"]), int(record["width"])):
            raise SamplingInvariantError(f"native dimensions mismatch: {relative}")
        count += 1
    return count


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_locked_base_sources(run_root: Path, raw_root: Path) -> dict[str, Mapping[str, object]]:
    source_lock_path = run_root / "input_lock/source_videos.json"
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    base = source_lock["base"]
    if tuple(base) != BASE_VIDEO_NAMES:
        raise SamplingInvariantError("locked base source order or membership changed")
    for name in BASE_VIDEO_NAMES:
        path = raw_root / "base" / name
        current = fingerprint_file(path, sha256=True)
        expected = base[name]
        if current.sha256 != expected["sha256"] or current.size != expected["size"]:
            raise SamplingInvariantError(f"locked base source changed: {path}")
        if str(path) != expected["path"]:
            raise SamplingInvariantError(f"base source path differs from lock: {path}")
    return base


def _extract_selected_frames(
    *,
    selections: list[dict[str, object]],
    raw_root: Path,
    image_root: Path,
    source_lock: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    selected_by_video: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for selection in selections:
        name = str(selection["video"])
        index = int(selection["source_frame_index"])
        if index in selected_by_video[name]:
            raise SamplingInvariantError(f"duplicate selected source frame: {name}:{index}")
        selected_by_video[name][index] = selection
    output: list[dict[str, object]] = []
    for name in BASE_VIDEO_NAMES:
        selected = selected_by_video.get(name, {})
        if not selected:
            continue
        video = raw_root / "base" / name
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise FileNotFoundError(video)
        frame_index = 0
        try:
            while selected:
                ok, pixels = capture.read()
                if not ok:
                    break
                selection = selected.pop(frame_index, None)
                if selection is not None:
                    pts_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    expected_pts = float(selection["source_pts_seconds"])
                    if abs(pts_seconds - expected_pts) > 1e-4:
                        raise SamplingInvariantError(
                            f"non-deterministic decoded PTS for {name}:{frame_index}: "
                            f"{pts_seconds} != {expected_pts}"
                        )
                    source_pts = int(selection["source_pts"])
                    filename = f"pts_{source_pts:012d}_frame_{frame_index:06d}.jpg"
                    relative = Path(Path(name).stem) / filename
                    record = write_native_frame(
                        pixels,
                        image_root / relative,
                        width=NATIVE_WIDTH,
                        height=NATIVE_HEIGHT,
                        writer=lambda path, image: cv2.imwrite(
                            path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        ),
                    )
                    output.append(
                        {
                            "video_sha256": source_lock[name]["sha256"],
                            "source_pts": source_pts,
                            "source_pts_seconds": pts_seconds,
                            "source_frame_index": frame_index,
                            "output_name": relative.as_posix(),
                            "image_sha256": record["image_sha256"],
                            "motion_class": selection["motion_class"],
                            "role": selection["role"],
                            "width": NATIVE_WIDTH,
                            "height": NATIVE_HEIGHT,
                        }
                    )
                frame_index += 1
        finally:
            capture.release()
        if selected:
            raise SamplingInvariantError(
                f"source frame indexes were not decoded from {video}: {sorted(selected)}"
            )
    output.sort(key=lambda row: (str(row["output_name"]), int(row["source_frame_index"])))
    return output


def run_adaptive_sampling(*, run_root: Path, raw_root: Path) -> dict[str, object]:
    """Scan all locked base videos and atomically publish sampled native image identities."""

    run_root = run_root.resolve(strict=True)
    raw_root = raw_root.resolve(strict=True)
    sampling = run_root / "sampling"
    images = sampling / "images"
    if (sampling / "motion_manifest.json").exists() or (sampling / "frame_manifest.json").exists():
        raise FileExistsError("adaptive sampling artifacts already exist")
    if not images.is_dir() or any(images.iterdir()):
        raise SamplingInvariantError("sampling images must be the empty initialized directory")
    source_lock = _verify_locked_base_sources(run_root, raw_root)
    intrinsics = json.loads((run_root / "input_lock/intrinsics.json").read_text(encoding="utf-8"))
    if intrinsics.get("distortion") != [0.0, 0.0, 0.0, 0.0, 0.0]:
        raise SamplingInvariantError("sampling requires fixed zero-distortion native intrinsics")
    camera_matrix = np.asarray(intrinsics["K"], dtype=float)
    if camera_matrix.shape != (3, 3) or not np.isfinite(camera_matrix).all():
        raise SamplingInvariantError("invalid locked camera matrix")
    intervals_by_video: dict[str, list[dict[str, object]]] = {}
    selected_by_video: dict[str, list[dict[str, object]]] = {}
    for name in BASE_VIDEO_NAMES:
        intervals = scan_motion(
            raw_root / "base" / name,
            camera_matrix=camera_matrix,
            analysis_interval_seconds=float(MOTION_THRESHOLDS["analysis_interval_seconds"]),
        )
        for row in intervals:
            row["video"] = name
        intervals_by_video[name] = intervals
        selected_by_video[name] = select_sample_frames(intervals)
    selections = [row for rows in selected_by_video.values() for row in rows]
    stage_parent = Path(tempfile.mkdtemp(prefix=".sampling.", dir=run_root))
    stage_sampling = stage_parent / "sampling"
    stage_images = stage_sampling / "images"
    stage_images.mkdir(parents=True)
    try:
        frame_records = _extract_selected_frames(
            selections=selections,
            raw_root=raw_root,
            image_root=stage_images,
            source_lock=source_lock,
        )
        validate_frame_records(frame_records)
        motion_manifest = {
            "schema_version": 1,
            "artifact_type": "MOTION_MANIFEST",
            "analysis_uses_decoded_pts": True,
            "thresholds": MOTION_THRESHOLDS,
            "profile": MOTION_PROFILE,
            "videos": intervals_by_video,
        }
        frame_manifest = {
            "schema_version": 1,
            "artifact_type": "FRAME_MANIFEST",
            "image_identity_authority": [
                "video_sha256",
                "source_pts",
                "source_frame_index",
                "output_name",
                "image_sha256",
                "motion_class",
                "role",
            ],
            "native_pixels_only": True,
            "frames": frame_records,
        }
        _atomic_json(stage_sampling / "motion_manifest.json", motion_manifest)
        _atomic_json(stage_sampling / "frame_manifest.json", frame_manifest)
        images.rmdir()
        os.replace(stage_sampling, sampling)
        stage_parent.rmdir()
    except Exception:
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    return {"motion_manifest": motion_manifest, "frame_manifest": frame_manifest}
