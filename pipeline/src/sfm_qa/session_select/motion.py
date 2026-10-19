"""Two-view motion classes. Ambiguous geometry is recorded as unproven."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime
    cv2 = None


MOTION_CLASSES = (
    "parallax",
    "low_parallax",
    "hover",
    "pure_rotation",
    "unproven",
    "fast_motion",
)

# Heuristic two-view gates. Missing / ambiguous evidence → unproven.
MOTION_THRESHOLDS = {
    "analysis_interval_seconds": 0.5,
    "minimum_tracked_features": 12,
    "hover_flow_median_px": 0.8,
    "fast_motion_flow_median_px": 35.0,
    "fast_motion_blur_variance": 25.0,
    "pure_rotation_min_degrees": 0.35,
    "pure_rotation_max_parallax_px": 0.75,
    "pure_rotation_homography_to_essential_ratio": 0.9,
    "minimum_essential_inliers": 20,
    "low_parallax_max_px": 2.0,
    "feature_max_corners": 1500,
    "feature_quality_level": 0.01,
    "feature_min_distance_px": 8.0,
    "ransac_threshold_px": 2.0,
}


def _empty_histogram() -> dict[str, int]:
    return {name: 0 for name in MOTION_CLASSES}


def _finite_median(rows: list[dict[str, Any]], key: str, *, require_tracks: bool = False) -> float | None:
    values: list[float] = []
    for row in rows:
        if require_tracks and int(row.get("tracked_count") or 0) < int(
            MOTION_THRESHOLDS["minimum_tracked_features"]
        ):
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _result(
    histogram: Mapping[str, int],
    *,
    reasons: tuple[str, ...] = (),
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    hist = _empty_histogram()
    for key, value in histogram.items():
        name = "unproven" if key in {"ambiguous", "unknown"} else str(key)
        if name not in hist:
            name = "unproven"
        hist[name] += int(value)
    total = sum(hist.values())
    dominant = "unproven"
    if total:
        dominant = max(hist.items(), key=lambda item: (item[1], item[0] == "unproven"))[0]
    measured_rows = list(rows or ())
    payload: dict[str, Any] = {
        "histogram": hist,
        "classes": hist,
        "intervals": total,
        "dominant": dominant,
        "parallax_ratio": (hist["parallax"] / total) if total else 0.0,
        "low_parallax_ratio": (hist["low_parallax"] / total) if total else 0.0,
        "hover_ratio": (hist["hover"] / total) if total else 0.0,
        "pure_rotation_ratio": (hist["pure_rotation"] / total) if total else 0.0,
        "fast_motion_ratio": (hist["fast_motion"] / total) if total else 0.0,
        "unproven_ratio": (hist["unproven"] / total) if total else 1.0,
        "epipolar_outlier_ratio_median": _finite_median(
            measured_rows, "epipolar_outlier_ratio", require_tracks=True
        ),
        "essential_inlier_ratio_median": _finite_median(
            measured_rows, "essential_inlier_ratio", require_tracks=True
        ),
        "homography_inlier_ratio_median": _finite_median(
            measured_rows, "homography_inlier_ratio", require_tracks=True
        ),
        "flow_median_px": _finite_median(measured_rows, "flow_median_px", require_tracks=True),
        "flow_mad_px": _finite_median(measured_rows, "flow_mad_px", require_tracks=True),
        "motion_parallax_median_px": _finite_median(
            measured_rows, "parallax_px", require_tracks=True
        ),
        "reasons": reasons,
        "rows": measured_rows,
    }
    payload.update(hist)
    return payload


def classify_interval(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one measured interval. Ambiguous geometry → ``unproven``."""

    if evidence.get("motion_class") in MOTION_CLASSES:
        motion_class = str(evidence["motion_class"])
        reason = str(evidence.get("decision_reason") or "provided_motion_class")
    elif str(evidence.get("motion_class") or "") in {"ambiguous", "unknown"}:
        motion_class = "unproven"
        reason = "ambiguous_mapped_to_unproven"
    else:
        tracked = int(evidence.get("tracked_count") or 0)
        flow = float(evidence.get("flow_median_px") or 0.0)
        blur = float(evidence.get("blur_variance") or 0.0)
        homography = int(evidence.get("homography_inliers") or 0)
        essential = int(evidence.get("essential_inliers") or 0)
        rotation = float(evidence.get("rotation_degrees") or 0.0)
        parallax = float(evidence.get("parallax_px") or 0.0)
        if blur < float(MOTION_THRESHOLDS["fast_motion_blur_variance"]) and flow >= float(
            MOTION_THRESHOLDS["fast_motion_flow_median_px"]
        ):
            motion_class = "fast_motion"
            reason = "high_flow_with_low_blur_variance"
        elif tracked >= int(MOTION_THRESHOLDS["minimum_tracked_features"]) and flow <= float(
            MOTION_THRESHOLDS["hover_flow_median_px"]
        ):
            motion_class = "hover"
            reason = "stable_feature_flow"
        elif tracked < int(MOTION_THRESHOLDS["minimum_tracked_features"]):
            motion_class = "unproven"
            reason = "insufficient_tracked_features"
        elif essential >= int(MOTION_THRESHOLDS["minimum_essential_inliers"]) and (
            rotation >= float(MOTION_THRESHOLDS["pure_rotation_min_degrees"])
            and parallax <= float(MOTION_THRESHOLDS["pure_rotation_max_parallax_px"])
            and homography
            >= float(MOTION_THRESHOLDS["pure_rotation_homography_to_essential_ratio"]) * essential
        ):
            motion_class = "pure_rotation"
            reason = "homography_dominant_rotation_with_negligible_parallax"
        elif essential < int(MOTION_THRESHOLDS["minimum_essential_inliers"]):
            motion_class = "unproven"
            reason = "insufficient_essential_inliers"
        elif parallax <= float(MOTION_THRESHOLDS["low_parallax_max_px"]):
            motion_class = "low_parallax"
            reason = "essential_geometry_with_low_parallax"
        else:
            motion_class = "parallax"
            reason = "essential_geometry_with_usable_parallax"
    return {
        "motion_class": motion_class,
        "decision_reason": reason,
        "decision_thresholds": dict(MOTION_THRESHOLDS),
    }


def _default_camera_matrix(width: int, height: int) -> np.ndarray:
    focal = float(max(width, height))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _base_evidence(*, blur: float = 0.0) -> dict[str, Any]:
    return {
        "tracked_count": 0,
        "flow_median_px": 0.0,
        "flow_mad_px": 0.0,
        "blur_variance": float(blur),
        "homography_inliers": 0,
        "homography_inlier_ratio": None,
        "essential_inliers": 0,
        "essential_inlier_ratio": None,
        "epipolar_outlier_ratio": None,
        "rotation_degrees": 0.0,
        "parallax_px": 0.0,
    }


def _motion_evidence(
    previous: np.ndarray,
    current: np.ndarray,
    camera_matrix: np.ndarray,
) -> dict[str, Any]:
    if cv2 is None:
        return _base_evidence()
    corners = cv2.goodFeaturesToTrack(
        previous,
        maxCorners=int(MOTION_THRESHOLDS["feature_max_corners"]),
        qualityLevel=float(MOTION_THRESHOLDS["feature_quality_level"]),
        minDistance=float(MOTION_THRESHOLDS["feature_min_distance_px"]),
        blockSize=7,
    )
    blur = float(cv2.Laplacian(current, cv2.CV_64F).var())
    empty = _base_evidence(blur=blur)
    if corners is None or len(corners) == 0:
        return empty
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
        return empty
    valid = status.reshape(-1).astype(bool)
    previous_xy = corners.reshape(-1, 2)[valid].astype(np.float64)
    current_xy = tracked.reshape(-1, 2)[valid].astype(np.float64)
    count = len(previous_xy)
    if not count:
        return empty

    flow = np.linalg.norm(current_xy - previous_xy, axis=1)
    flow_median = float(np.median(flow))
    flow_mad = float(np.median(np.abs(flow - flow_median)))

    homography = None
    homography_mask = None
    if count >= 4:
        homography, homography_mask = cv2.findHomography(
            previous_xy,
            current_xy,
            cv2.RANSAC,
            float(MOTION_THRESHOLDS["ransac_threshold_px"]),
        )
    # OpenCV masks can be 0/1 or 0/255. count_nonzero is the only safe count.
    homography_inliers = int(np.count_nonzero(homography_mask)) if homography_mask is not None else 0
    homography_ratio = (homography_inliers / count) if count else None

    parallax = flow_median
    if homography is not None and homography_mask is not None and homography_inliers:
        predicted = cv2.perspectiveTransform(
            previous_xy.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        residual = np.linalg.norm(current_xy - predicted, axis=1)
        inlier = homography_mask.reshape(-1).astype(bool)
        if inlier.any():
            parallax = float(np.median(residual[inlier]))

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
                    essential[:3, :3],
                    previous_xy,
                    current_xy,
                    camera_matrix,
                    mask=essential_mask,
                )
                essential_mask = pose_mask
                cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
                rotation_degrees = float(np.degrees(np.arccos(cosine)))
        except cv2.error:
            essential_mask = None

    essential_inliers = int(np.count_nonzero(essential_mask)) if essential_mask is not None else 0
    essential_ratio = (essential_inliers / count) if count else None
    epipolar_outlier = (
        max(0.0, min(1.0, 1.0 - essential_ratio))
        if essential_ratio is not None and essential_inliers > 0
        else None
    )
    return {
        "tracked_count": count,
        "flow_median_px": flow_median,
        "flow_mad_px": flow_mad,
        "blur_variance": blur,
        "homography_inliers": homography_inliers,
        "homography_inlier_ratio": homography_ratio,
        "essential_inliers": essential_inliers,
        "essential_inlier_ratio": essential_ratio,
        "epipolar_outlier_ratio": epipolar_outlier,
        "rotation_degrees": rotation_degrees,
        "parallax_px": parallax,
    }


def scan_video(path: str | Path, interval_seconds: float = 0.5) -> dict[str, Any]:
    """Class histogram over sampled intervals. Ambiguous → unproven."""

    video = Path(path)
    if cv2 is None:
        return _result({"unproven": 1}, reasons=("missing_opencv",))
    if not video.is_file():
        return _result({"unproven": 1}, reasons=("missing_video",))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return _result({"unproven": 1}, reasons=("unreadable_video",))

    interval = float(interval_seconds) if interval_seconds and interval_seconds > 0 else 0.5
    previous_gray: np.ndarray | None = None
    next_sample = 0.0
    frame_index = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    rows: list[dict[str, Any]] = []
    histogram = _empty_histogram()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            msec = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if msec > 0:
                pts = msec / 1000.0
            elif fps > 0:
                pts = frame_index / fps
            else:
                pts = float(frame_index)
            if pts + 1e-9 < next_sample:
                frame_index += 1
                continue
            if frame is None:
                frame_index += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            height, width = gray.shape[:2]
            camera = _default_camera_matrix(int(width), int(height))
            if previous_gray is None:
                evidence = _base_evidence(
                    blur=float(cv2.Laplacian(gray, cv2.CV_64F).var())
                )
            else:
                evidence = _motion_evidence(previous_gray, gray, camera)
            classified = classify_interval(evidence)
            motion_class = str(classified["motion_class"])
            histogram[motion_class] = histogram.get(motion_class, 0) + 1
            rows.append(
                {
                    "source_pts_seconds": pts,
                    "source_frame_index": frame_index,
                    **evidence,
                    **classified,
                }
            )
            previous_gray = gray
            next_sample = pts + interval
            frame_index += 1
    finally:
        capture.release()

    if not rows:
        return _result({"unproven": 1}, reasons=("unreadable_video",))
    if len(rows) < 2:
        histogram = _empty_histogram()
        histogram["unproven"] = max(1, len(rows))
        return _result(histogram, reasons=("insufficient_intervals",), rows=rows)
    return _result(histogram, rows=rows)


__all__ = [
    "MOTION_CLASSES",
    "MOTION_THRESHOLDS",
    "classify_interval",
    "scan_video",
]
