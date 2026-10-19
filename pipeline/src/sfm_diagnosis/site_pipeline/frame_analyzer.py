"""Optional-OpenCV frame evidence for conservative Stage 1 sanitization.

The functions in this module are observational: they never modify the source
media and low geometry scores are warnings, not rejection decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:  # OpenCV is deliberately optional for metadata-only deployments.
    import cv2
except ImportError:  # pragma: no cover - exercised in minimal installations
    cv2 = None


def _gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3:
        if cv2 is not None:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image[..., :3].mean(axis=2).astype(np.uint8)
    return image.astype(np.uint8, copy=False)


def _histogram(image: np.ndarray) -> np.ndarray:
    return np.histogram(_gray(image), bins=32, range=(0, 256), density=True)[0]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _basic_evidence(image: np.ndarray) -> dict[str, Any]:
    gray = _gray(image)
    pixels = gray.astype(np.float32) / 255.0
    black = float(np.mean(pixels <= 0.01))
    white = float(np.mean(pixels >= 0.99))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if cv2 is not None else None
    return {
        "black_fraction": black,
        "white_fraction": white,
        "blur_score": blur,
        "blur_variance": blur,
        "severe_blur": bool(blur is not None and blur < 2.0),
        "corrupt": False,
        "exposure_failed": bool(black >= 0.98 or white >= 0.98),
    }


def calibrated_rotation_evidence(
    homography: np.ndarray,
    points_i: np.ndarray,
    points_j: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Recover yaw/pitch/roll from ``K R K^-1`` and residual parallax."""

    h = np.asarray(homography, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    first = np.asarray(points_i, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(points_j, dtype=np.float64).reshape(-1, 2)
    empty = {
        "rotation_degrees": None,
        "rotation_residual_p50_px": None,
        "pure_rotation_geometry": False,
    }
    if h.shape != (3, 3) or k.shape != (3, 3) or len(first) < 4 or len(first) != len(second):
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
    residual_p50 = float(np.median(np.linalg.norm(predicted - second[valid], axis=1)))
    height, width = image_shape
    pixel_scale = max(float(width) / 1280.0, float(height) / 720.0, 1.0)
    return {
        "rotation_degrees": angle,
        "rotation_residual_p50_px": residual_p50,
        "pure_rotation_geometry": bool(angle >= 0.35 and residual_p50 <= 0.75 * pixel_scale),
        "relative_rotation_calibrated": rotation.tolist(),
    }


def analyze_frame_pair(
    image_i: np.ndarray,
    image_j: np.ndarray,
    *,
    intrinsics: np.ndarray | None = None,
    dynamic_mask_callback: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
) -> dict[str, Any]:
    """Compute canonical, JSON-friendly evidence for two decoded images."""
    first, second = _gray(image_i), _gray(image_j)
    result: dict[str, Any] = {
        "raw_matches": 0,
        "inliers_F": 0,
        "inliers_E": 0,
        "inliers_H": 0,
        "inlier_ratio": 0.0,
        "median_sampson_error": None,
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
    correlation = _correlation(_histogram(first), _histogram(second))
    result["near_duplicate"] = bool(correlation >= 0.999 and np.array_equal(first, second))
    result["appearance_novelty"] = float(max(0.0, 1.0 - correlation))
    result["scene_cut"] = result["appearance_novelty"] >= 0.25
    if dynamic_mask_callback is not None:
        mask = dynamic_mask_callback(image_i, image_j)
        if mask is not None:
            result["dynamic_fraction"] = float(np.mean(np.asarray(mask, dtype=bool)))
    if cv2 is None:
        return result
    points = cv2.goodFeaturesToTrack(first, maxCorners=1200, qualityLevel=0.01, minDistance=5)
    if points is None or len(points) < 4:
        return result
    tracked, valid, _ = cv2.calcOpticalFlowPyrLK(first, second, points, None)
    valid = valid.reshape(-1).astype(bool) if valid is not None else np.zeros(len(points), bool)
    p, q = points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid]
    result["raw_matches"] = int(len(p))
    if len(p) < 4:
        return result
    flow = np.linalg.norm(q - p, axis=1)
    result["parallax_p10"], result["parallax_p50"] = map(float, np.percentile(flow, [10, 50]))
    result["flow_median_px"] = result["parallax_p50"]
    result["motion"] = result["flow_median_px"]
    result["parallax"] = result["parallax_p10"]
    result["spatial_coverage"] = float(
        np.ptp(p[:, 0]) * np.ptp(p[:, 1]) / max(1, first.shape[0] * first.shape[1])
    )
    cells = np.unique(
        np.floor(p / np.array([first.shape[1], first.shape[0]]) * 4).astype(int), axis=0
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
                    h,
                    p[keep_h],
                    q[keep_h],
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
    result["homography_dominant"] = bool(result["inliers_H"] >= 0.9 * max(result["inliers_F"], 1))
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
            result["cheirality_ratio"] = float(pose_mask.mean()) if pose_mask is not None else None
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


def analyze_frames(
    source: str | Path | Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    video_id: str,
    session_id: str = "",
    fps: float = 2.0,
    intrinsics: np.ndarray | None = None,
    segment_id: str | None = None,
    dynamic_mask_callback: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
) -> list[dict[str, Any]]:
    """Probe a video or image sequence and return preprocessing-compatible rows."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    samples = _decode(source, fps)
    rows: list[dict[str, Any]] = []
    previous = None
    for source_index, timestamp, image, image_path in samples:
        base = _basic_evidence(image)
        pair = (
            analyze_frame_pair(
                previous, image, intrinsics=intrinsics, dynamic_mask_callback=dynamic_mask_callback
            )
            if previous is not None
            else {}
        )
        warnings = []
        if base["severe_blur"]:
            warnings.append("severe_blur")
        if base["exposure_failed"]:
            warnings.append("exposure_failed")
        if pair.get("near_duplicate"):
            warnings.append("duplicate_previous")
        row = {
            **base,
            **pair,
            "video_id": video_id,
            "session_id": session_id,
            "segment_id": segment_id or f"{video_id}:S01",
            "frame_index": source_index,
            "source_frame_index": source_index,
            "frame_id": f"{video_id}:{source_index:08d}",
            "timestamp": timestamp,
            "source_pts_seconds": timestamp,
            "source_uri": str(source) if isinstance(source, (str, Path)) else None,
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


def _decode(
    source: str | Path | Sequence[np.ndarray] | Iterable[np.ndarray], fps: float
) -> Iterator[tuple[int, float, np.ndarray, Path | None]]:
    if not isinstance(source, (str, Path)):
        for index, image in enumerate(source):
            yield index, index / fps, image, None
        return
    if cv2 is None:
        raise RuntimeError("OpenCV is required to decode video/image paths")
    path = Path(source)
    if path.is_dir():
        for index, item in enumerate(sorted(path.iterdir())):
            if not item.is_file():
                continue
            image = cv2.imread(str(item), cv2.IMREAD_UNCHANGED)
            if image is not None:
                yield index, index / fps, image, item
        return
    capture = cv2.VideoCapture(str(path))
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
    capture.release()


def extract_frame(
    source: str | Path,
    source_frame_index: int,
    output: str | Path,
    *,
    source_image_path: str | Path | None = None,
) -> Path:
    """Write one selected frame without modifying the source media."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required to extract keyframes")
    if source_image_path is not None:
        image = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    else:
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_frame_index))
        ok, image = capture.read()
        capture.release()
        if not ok:
            image = None
    if image is None:
        raise ValueError(f"cannot decode source frame {source_frame_index}: {source}")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write keyframe {path}")
    return path


def extract_frames(
    source: str | Path,
    requests: Sequence[Mapping[str, Any]],
) -> dict[int, Path]:
    """Extract many frames with one sequential video decode.

    Image-sequence requests may provide ``source_image_path`` and are decoded
    directly. Video requests share one ``VideoCapture`` and stop after the last
    requested source index.
    """

    if cv2 is None:
        raise RuntimeError("OpenCV is required to extract keyframes")
    normalized: dict[int, tuple[Path, str | Path | None]] = {}
    for request in requests:
        index = int(request["source_frame_index"])
        if index < 0 or index in normalized:
            raise ValueError("source frame indexes must be unique and non-negative")
        normalized[index] = (
            Path(request["output"]),
            request.get("source_image_path"),
        )
    if not normalized:
        return {}
    extracted: dict[int, Path] = {}
    video_indexes = []
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
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open video for extraction: {source}")
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
