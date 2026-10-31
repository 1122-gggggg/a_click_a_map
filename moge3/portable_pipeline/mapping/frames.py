"""Frame evidence analysis for keyframe selection.

Ports the proven mechanism: decode each source at a probe rate, compute
per-frame quality evidence (blur/exposure/black fraction), and per-pair
geometry evidence (KLT tracks, F/E/H inliers, optical flow, calibrated
rotation) used to classify motion regimes. OpenCV is required.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .inventory import SourceSpec

try:
    import cv2
except ImportError:  # pragma: no cover - import-time environment check
    cv2 = None  # type: ignore[assignment]


def require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("opencv (cv2) is required for frame analysis")
    return cv2


def gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
    return image.astype(np.uint8, copy=False)


def histogram(image: np.ndarray) -> np.ndarray:
    return np.histogram(gray(image), bins=32, range=(0, 256), density=True)[0]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def basic_evidence(image: np.ndarray) -> dict[str, Any]:
    require_cv2()
    frame = gray(image)
    blur = float(cv2.Laplacian(frame, cv2.CV_64F).var())
    mean = float(frame.mean())
    black = float((frame < 8).mean())
    white = float((frame > 247).mean())
    return {
        "blur_score": blur,
        "blur_variance": blur,
        "severe_blur": bool(blur < 2.0),
        "exposure_failed": bool(mean < 4.0 or mean > 251.0),
        "black_fraction": black,
        "white_fraction": white,
        "corrupt": False,
    }


def calibrated_rotation_evidence(
    homography: np.ndarray,
    points_i: np.ndarray,
    points_j: np.ndarray,
    K: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Recover yaw/pitch/roll from K R K^-1 plus residual parallax.

    A frame pair is pure-rotation geometry ONLY when a real rotation is
    measured (angle >= 0.35 deg) AND the derotated residual is sub-pixel
    (<= 0.75 px at 720p, scaled with resolution). A near-zero rotation
    with a multi-pixel residual is translation, not rotation: the
    homography fit a translating scene and the derotation explains
    nothing. Misclassifying those as pure_rotation would wrongly condemn
    triangulable frames to POSE_ONLY.
    """
    h = np.asarray(homography, dtype=np.float64)
    k = np.asarray(K, dtype=np.float64)
    first = np.asarray(points_i, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(points_j, dtype=np.float64).reshape(-1, 2)
    empty: dict[str, Any] = {
        "rotation_degrees": None,
        "rotation_residual_p50_px": None,
        "pure_rotation_geometry": False,
    }
    if h.shape != (3, 3) or k.shape != (3, 3):
        return empty
    if len(first) < 4 or len(first) != len(second):
        return empty
    if not np.isfinite(h).all() or not np.isfinite(k).all():
        return empty
    try:
        inverse_k = np.linalg.inv(k)
        u, _, vt = np.linalg.svd(inverse_k @ h @ k)
    except np.linalg.LinAlgError:
        return empty
    handedness = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ np.diag([1.0, 1.0, handedness]) @ vt
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cosine)))
    rotation_h = k @ rotation @ inverse_k
    homogeneous = np.column_stack((first, np.ones(len(first), dtype=np.float64)))
    projected = (rotation_h @ homogeneous.T).T
    valid = np.abs(projected[:, 2]) > 1e-12
    if not np.any(valid):
        return empty
    predicted = projected[valid, :2] / projected[valid, 2:]
    residual_p50 = float(
        np.median(np.linalg.norm(predicted - second[valid], axis=1))
    )
    height, width = image_shape
    pixel_scale = max(float(width) / 1280.0, float(height) / 720.0, 1.0)
    return {
        "rotation_degrees": angle,
        "rotation_residual_p50_px": residual_p50,
        "pure_rotation_geometry": bool(
            angle >= 0.35 and residual_p50 <= 0.75 * pixel_scale
        ),
        "relative_rotation_calibrated": rotation.tolist(),
    }


def analyze_frame_pair(
    image_i: np.ndarray,
    image_j: np.ndarray,
    *,
    intrinsics: np.ndarray | None = None,
    dynamic_mask_callback: Callable[..., np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Compute canonical, JSON-friendly evidence for two decoded images."""
    require_cv2()
    first, second = gray(image_i), gray(image_j)
    result: dict[str, Any] = {
        "raw_matches": 0,
        "inliers_F": 0,
        "inliers_E": 0,
        "inliers_H": 0,
        "inlier_ratio": 0.0,
        "spatial_coverage": 0.0,
        "grid_occupancy": 0,
        "relative_rotation": None,
        "translation_direction": None,
        "cheirality_ratio": None,
        "parallax_p10": None,
        "parallax_p50": None,
        "flow_median_px": 0.0,
        "motion": 0.0,
        "motion_class": "unproven",
        "near_duplicate": False,
        "appearance_novelty": 0.0,
        "scene_cut": False,
        "turn_event": False,
        "homography_dominant": False,
        "relative_rotation_2d_deg": None,
        "rotation_degrees": None,
        "rotation_residual_p50_px": None,
        "pure_rotation_geometry": False,
        "dynamic_fraction": None,
        "warnings": [],
        "status": "CANDIDATE",
    }
    corr = correlation(histogram(first), histogram(second))
    result["near_duplicate"] = bool(corr >= 0.999 and np.array_equal(first, second))
    result["appearance_novelty"] = float(max(0.0, 1.0 - corr))
    result["scene_cut"] = bool(result["appearance_novelty"] >= 0.25)
    if dynamic_mask_callback is not None:
        mask = dynamic_mask_callback(image_i, image_j)
        if mask is not None:
            result["dynamic_fraction"] = float(np.mean(np.asarray(mask, dtype=bool)))
    points = cv2.goodFeaturesToTrack(
        first, maxCorners=1200, qualityLevel=0.01, minDistance=5
    )
    if points is None or len(points) < 4:
        return result
    tracked, valid, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    valid = valid.reshape(-1).astype(bool) if valid is not None else np.zeros(
        len(points), bool
    )
    p, q = points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]
    result["raw_matches"] = int(len(p))
    if len(p) < 4:
        return result
    flow = np.linalg.norm(q - p, axis=1)
    result["parallax_p10"], result["parallax_p50"] = map(
        float, np.percentile(flow, [10, 50])
    )
    result["flow_median_px"] = result["parallax_p50"]
    result["motion"] = result["flow_median_px"]
    result["parallax"] = result["parallax_p10"]
    result["spatial_coverage"] = float(
        np.ptp(p[:, 0]) * np.ptp(p[:, 1]) / max(1, first.shape[0] * first.shape[1])
    )
    cells = np.unique(
        np.floor(p / np.array([first.shape[1], first.shape[0]]) * 4).astype(int),
        axis=0,
    )
    result["grid_occupancy"] = int(len(cells))
    f, mask_f = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, 1.5, 0.99)
    if mask_f is not None:
        result["inliers_F"] = int(mask_f.sum())
        result["inlier_ratio"] = float(mask_f.mean())
    h, mask_h = cv2.findHomography(p, q, cv2.RANSAC, 3.0)
    if mask_h is not None:
        result["inliers_H"] = int(mask_h.sum())
    if h is not None and np.isfinite(h).all():
        rotation_2d = float(np.degrees(np.arctan2(h[1, 0], h[0, 0])))
        result["relative_rotation_2d_deg"] = rotation_2d
        if intrinsics is not None:
            keep_h = (
                mask_h.reshape(-1).astype(bool)
                if mask_h is not None
                else np.ones(len(p), dtype=bool)
            )
            result.update(
                calibrated_rotation_evidence(
                    h, p[keep_h], q[keep_h],
                    np.asarray(intrinsics, dtype=np.float64),
                    image_shape=first.shape,
                )
            )
        calibrated_turn = bool(
            result.get("pure_rotation_geometry") is True
            and result.get("rotation_degrees") is not None
            and float(result["rotation_degrees"]) >= 5.0
        )
        result["turn_event"] = bool(
            (calibrated_turn or abs(rotation_2d) >= 5.0)
            and result["inliers_H"] >= max(12, 0.6 * len(p))
        )
    result["homography_dominant"] = bool(
        result["inliers_H"] >= 0.9 * max(result["inliers_F"], 1)
    )
    if intrinsics is not None:
        e, mask_e = cv2.findEssentialMat(
            p, q, np.asarray(intrinsics, dtype=np.float64), cv2.RANSAC, 0.999, 1.0
        )
        if e is not None and mask_e is not None:
            result["inliers_E"] = int(mask_e.sum())
            _, rotation, translation, pose_mask = cv2.recoverPose(
                e, p, q, np.asarray(intrinsics, dtype=np.float64), mask=mask_e
            )
            result["relative_rotation"] = rotation.tolist()
            result["translation_direction"] = translation.reshape(-1).tolist()
            result["cheirality_ratio"] = (
                float(pose_mask.mean()) if pose_mask is not None else None
            )
    if len(p) < 12:
        result["motion_class"] = "unproven"
    elif result["near_duplicate"] or result["flow_median_px"] < 0.5:
        result["motion_class"] = "hover"
    elif result["homography_dominant"] and (
        result.get("pure_rotation_geometry") is True or result["turn_event"]
    ):
        result["motion_class"] = "pure_rotation"
    elif result["flow_median_px"] > 35.0:
        result["motion_class"] = "fast_motion"
    elif (result["parallax_p10"] or 0.0) < 1.0:
        result["motion_class"] = "low_parallax"
    else:
        result["motion_class"] = "parallax"
    return result


def decode_source(
    source: SourceSpec, fps: float
) -> Iterator[tuple[int, float, np.ndarray, Path | None]]:
    """Decode a video or image directory at the probe rate."""
    require_cv2()
    if source.kind == "image_dir":
        if source.path.is_file():
            image = cv2.imread(str(source.path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                yield 0, 0.0, image, source.path
            return
        for index, item in enumerate(sorted(source.path.iterdir())):
            if not item.is_file():
                continue
            image = cv2.imread(str(item), cv2.IMREAD_UNCHANGED)
            if image is not None:
                yield index, (index / fps) if fps > 0 else float(index), image, item
        return
    capture = cv2.VideoCapture(str(source.path))
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
        step = max(1, round(source_fps / fps))
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if index % step == 0:
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield index, float(timestamp), image, None
            index += 1
    finally:
        capture.release()


def analyze_source(
    source: SourceSpec,
    *,
    fps: float,
    intrinsics: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Probe one source and return preprocessing-compatible frame rows."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    rows: list[dict[str, Any]] = []
    previous = None
    for source_index, timestamp, image, image_path in decode_source(source, fps):
        base = basic_evidence(image)
        pair = (
            analyze_frame_pair(previous, image, intrinsics=intrinsics)
            if previous is not None
            else {}
        )
        warnings: list[str] = []
        if base["severe_blur"]:
            warnings.append("severe_blur")
        if base["exposure_failed"]:
            warnings.append("exposure_failed")
        if pair.get("near_duplicate"):
            warnings.append("duplicate_previous")
        row = {
            **base,
            **pair,
            "video_id": source.source_id,
            "session_id": source.path.stem,
            "segment_id": f"{source.source_id}:S01",
            "frame_index": source_index,
            "source_frame_index": source_index,
            "frame_id": f"{source.source_id}:{source_index:08d}",
            "timestamp": timestamp,
            "source_pts_seconds": timestamp,
            "source_uri": str(source.path),
            "source_image_path": None if image_path is None else str(image_path),
            "status": "SANITIZED" if not warnings else "CANDIDATE",
            "warnings": warnings,
        }
        row.setdefault("motion", 0.0)
        row.setdefault("flow_median_px", 0.0)
        row.setdefault("motion_class", "unproven")
        row.setdefault("scene_cut", False)
        row.setdefault("turn_event", False)
        row["severe_motion_blur"] = bool(
            row["motion_class"] in {"fast_motion", "pure_rotation"}
            and row.get("blur_variance") is not None
            and float(row["blur_variance"]) < 25.0
        )
        rows.append(row)
        previous = image
    return rows


def extract_frames(
    source: SourceSpec,
    requests: Sequence[Mapping[str, Any]],
) -> dict[int, Path]:
    """Extract many frames with one sequential video decode."""
    require_cv2()
    normalized: dict[int, tuple[Path, str | Path | None]] = {}
    for request in requests:
        index = int(request["source_frame_index"])
        if index < 0 or index in normalized:
            raise ValueError("source frame indexes must be unique and non-negative")
        normalized[index] = (Path(request["output"]), request.get("source_image_path"))
    if not normalized:
        return {}
    extracted: dict[int, Path] = {}
    video_indexes: list[int] = []
    for index, (output, source_image) in sorted(normalized.items()):
        if source_image:
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode image-sequence frame: {source_image}")
            _write_image(output, image)
            extracted[index] = output
        else:
            video_indexes.append(index)
    if video_indexes:
        capture = cv2.VideoCapture(str(source.path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open video for extraction: {source.path}")
        wanted = set(video_indexes)
        maximum = max(wanted)
        index = 0
        try:
            while index <= maximum:
                ok, image = capture.read()
                if not ok:
                    break
                if index in wanted:
                    output = normalized[index][0]
                    _write_image(output, image)
                    extracted[index] = output
                index += 1
        finally:
            capture.release()
    missing = sorted(set(normalized) - set(extracted))
    if missing:
        raise ValueError(f"source frames could not be extracted: {missing}")
    return extracted


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write keyframe {path}")


__all__ = [
    "analyze_frame_pair",
    "analyze_source",
    "basic_evidence",
    "calibrated_rotation_evidence",
    "decode_source",
    "extract_frames",
    "require_cv2",
]
