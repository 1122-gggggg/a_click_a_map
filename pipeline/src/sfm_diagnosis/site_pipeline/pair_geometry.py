"""EDM correspondence geometry verification; OpenCV is optional at import time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageTransform:
    scale_x: float
    scale_y: float
    pad_x: float = 0.0
    pad_y: float = 0.0

    def to_source(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("image transform scales must be positive")
        return np.column_stack(
            (
                (values[:, 0] - self.pad_x) / self.scale_x,
                (values[:, 1] - self.pad_y) / self.scale_y,
            )
        )


@dataclass(frozen=True)
class PairGeometryResult:
    raw_matches: int
    inliers_F: int
    inliers_E: int | None
    inliers_H: int
    inlier_ratio: float
    median_sampson_error: float | None
    spatial_coverage: float
    coverage_i: float
    coverage_j: float
    grid_occupancy_i: int
    grid_occupancy_j: int
    relative_rotation: np.ndarray | None
    translation_direction: np.ndarray | None
    cheirality_ratio: float | None
    parallax_p10_deg: float | None
    parallax_p50_deg: float | None
    degeneracy_flags: tuple[str, ...]
    F: np.ndarray | None
    E: np.ndarray | None
    H: np.ndarray | None
    fundamental_mask: np.ndarray
    essential_mask: np.ndarray | None
    homography_mask: np.ndarray
    raw_points_i: np.ndarray
    raw_points_j: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "raw_matches": self.raw_matches,
            "inliers_F": self.inliers_F,
            "inliers_E": self.inliers_E,
            "inliers_H": self.inliers_H,
            "inlier_ratio": self.inlier_ratio,
            "median_sampson_error": self.median_sampson_error,
            "spatial_coverage": self.spatial_coverage,
            "coverage_i": self.coverage_i,
            "coverage_j": self.coverage_j,
            "grid_occupancy_i": self.grid_occupancy_i,
            "grid_occupancy_j": self.grid_occupancy_j,
            "relative_rotation": (
                None if self.relative_rotation is None else self.relative_rotation.tolist()
            ),
            "translation_direction": (
                None if self.translation_direction is None else self.translation_direction.tolist()
            ),
            "cheirality_ratio": self.cheirality_ratio,
            "parallax_p10_deg": self.parallax_p10_deg,
            "parallax_p50_deg": self.parallax_p50_deg,
            "bearing_parallax_p10_deg": self.parallax_p10_deg,
            "bearing_parallax_p50_deg": self.parallax_p50_deg,
            "parallax_kind": "BEARING_RAY_APPROXIMATION",
            "degeneracy_flags": list(self.degeneracy_flags),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "points_i": self.raw_points_i,
            "points_j": self.raw_points_j,
            "fundamental_mask": self.fundamental_mask.astype(np.uint8),
            "homography_mask": self.homography_mask.astype(np.uint8),
        }
        if self.essential_mask is not None:
            result["essential_mask"] = self.essential_mask.astype(np.uint8)
        if self.F is not None:
            result["F"] = self.F
        if self.E is not None:
            result["E"] = self.E
        if self.H is not None:
            result["H"] = self.H
        return result


def resize_pad_points(
    points: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    pad: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    scale = min(target_width / source_width, target_height / source_height)
    return ImageTransform(scale, scale, pad[0], pad[1]).to_source(points)


def verify_pair(
    points_i: np.ndarray,
    points_j: np.ndarray,
    *,
    image_shape_i: tuple[int, int] | None = None,
    image_shape_j: tuple[int, int] | None = None,
    K_i: np.ndarray | None = None,
    K_j: np.ndarray | None = None,
    threshold_px: float = 1.0,
    confidence: float = 0.999,
    max_iterations: int = 10000,
) -> PairGeometryResult:
    """Estimate F/E/H and diagnostics from original-pixel correspondences."""

    cv2 = _cv2()
    left = _points(points_i, "points_i")
    right = _points(points_j, "points_j")
    if left.shape != right.shape:
        raise ValueError("point arrays must have identical shape")
    if len(left) < 8:
        raise ValueError("at least eight correspondences are required")
    if threshold_px <= 0 or not np.isfinite(threshold_px):
        raise ValueError("threshold_px must be positive and finite")
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

    F, fundamental_mask_raw = cv2.findFundamentalMat(
        left,
        right,
        method,
        threshold_px,
        confidence,
        max_iterations,
    )
    F = _matrix3(F)
    fundamental_mask = (
        _mask(fundamental_mask_raw, len(left)) if F is not None else _empty_mask(len(left))
    )

    H, homography_mask_raw = cv2.findHomography(
        left,
        right,
        method=method,
        ransacReprojThreshold=threshold_px,
        confidence=confidence,
        maxIters=max_iterations,
    )
    H = _matrix3(H)
    homography_mask = (
        _mask(homography_mask_raw, len(left)) if H is not None else _empty_mask(len(left))
    )

    E = None
    essential_mask = None
    rotation = None
    translation = None
    cheirality_ratio = None
    parallax_p10 = None
    parallax_p50 = None
    if K_i is not None or K_j is not None:
        if K_i is None or K_j is None:
            raise ValueError("both K_i and K_j are required for essential geometry")
        intrinsic_i = _intrinsics(K_i, "K_i")
        intrinsic_j = _intrinsics(K_j, "K_j")
        normalized_i = cv2.undistortPoints(left.reshape(-1, 1, 2), intrinsic_i, None).reshape(-1, 2)
        normalized_j = cv2.undistortPoints(right.reshape(-1, 1, 2), intrinsic_j, None).reshape(
            -1, 2
        )
        focal_scale = float(
            min(
                intrinsic_i[0, 0],
                intrinsic_i[1, 1],
                intrinsic_j[0, 0],
                intrinsic_j[1, 1],
            )
        )
        normalized_threshold = threshold_px / max(focal_scale, 1e-9)
        essential_raw, essential_mask_raw = cv2.findEssentialMat(
            normalized_i,
            normalized_j,
            np.eye(3),
            method=method,
            prob=confidence,
            threshold=normalized_threshold,
            maxIters=max_iterations,
        )
        E = _matrix3(essential_raw)
        if E is not None:
            essential_mask = _mask(essential_mask_raw, len(left))
            _, rotation, translation, pose_mask_raw = cv2.recoverPose(
                E,
                normalized_i,
                normalized_j,
                np.eye(3),
                mask=essential_mask.astype(np.uint8).reshape(-1, 1),
            )
            pose_mask = _mask(pose_mask_raw, len(left))
            valid_count = max(int(essential_mask.sum()), 1)
            cheirality_ratio = float(np.logical_and(pose_mask, essential_mask).sum() / valid_count)
            essential_mask = np.logical_and(essential_mask, pose_mask)
            translation = np.asarray(translation, dtype=float).reshape(3)
            norm = float(np.linalg.norm(translation))
            translation = translation / norm if norm > 0 else translation
            parallax = _parallax_degrees(
                normalized_i[essential_mask], normalized_j[essential_mask], rotation
            )
            if len(parallax):
                parallax_p10 = float(np.percentile(parallax, 10))
                parallax_p50 = float(np.percentile(parallax, 50))

    coverage_i = _coverage(left[fundamental_mask], image_shape_i, cv2)
    coverage_j = _coverage(right[fundamental_mask], image_shape_j, cv2)
    occupancy_i = grid_occupancy(left[fundamental_mask], image_shape_i)
    occupancy_j = grid_occupancy(right[fundamental_mask], image_shape_j)
    median_sampson = _median_sampson(F, left[fundamental_mask], right[fundamental_mask])
    inliers_f = int(fundamental_mask.sum())
    inliers_e = None if essential_mask is None else int(essential_mask.sum())
    inliers_h = int(homography_mask.sum())
    flags: list[str] = []
    if F is None:
        flags.append("fundamental_failed")
    if K_i is None:
        flags.append("no_intrinsics")
    elif E is None:
        flags.append("essential_failed")
    if inliers_h >= 0.9 * max(inliers_e or inliers_f, 1):
        flags.append("homography_dominant")
    if parallax_p10 is not None and parallax_p10 < 1.0:
        flags.append("low_parallax")
    if cheirality_ratio is not None and cheirality_ratio < 0.5:
        flags.append("low_cheirality")
    if min(coverage_i, coverage_j) < 0.1:
        flags.append("low_spatial_coverage")

    return PairGeometryResult(
        raw_matches=len(left),
        inliers_F=inliers_f,
        inliers_E=inliers_e,
        inliers_H=inliers_h,
        inlier_ratio=float(inliers_f / len(left)),
        median_sampson_error=median_sampson,
        spatial_coverage=min(coverage_i, coverage_j),
        coverage_i=coverage_i,
        coverage_j=coverage_j,
        grid_occupancy_i=occupancy_i,
        grid_occupancy_j=occupancy_j,
        relative_rotation=None if rotation is None else np.asarray(rotation, dtype=float),
        translation_direction=translation,
        cheirality_ratio=cheirality_ratio,
        parallax_p10_deg=parallax_p10,
        parallax_p50_deg=parallax_p50,
        degeneracy_flags=tuple(flags),
        F=F,
        E=E,
        H=H,
        fundamental_mask=fundamental_mask,
        essential_mask=essential_mask,
        homography_mask=homography_mask,
        raw_points_i=left,
        raw_points_j=right,
    )


def grid_occupancy(
    points: np.ndarray,
    image_shape: tuple[int, int] | None,
    *,
    rows: int = 4,
    columns: int = 4,
) -> int:
    values = np.asarray(points, dtype=float).reshape(-1, 2)
    if image_shape is None or not len(values):
        return 0
    height, width = image_shape[:2]
    if min(height, width, rows, columns) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    x = np.clip((values[:, 0] * columns / width).astype(int), 0, columns - 1)
    y = np.clip((values[:, 1] * rows / height).astype(int), 0, rows - 1)
    return len({(int(col), int(row)) for col, row in zip(x, y, strict=True)})


def _points(value: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite")
    return points


def _intrinsics(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive")
    return matrix


def _matrix3(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size < 9 or not np.isfinite(array[:3, :3]).all():
        return None
    return array.reshape(-1, 3)[:3, :3]


def _mask(value: Any, size: int) -> np.ndarray:
    if value is None:
        return _empty_mask(size)
    mask = np.asarray(value).reshape(-1)[:size] > 0
    if len(mask) != size:
        result = _empty_mask(size)
        result[: len(mask)] = mask
        return result
    return mask


def _empty_mask(size: int) -> np.ndarray:
    return np.zeros(size, dtype=bool)


def _coverage(points: np.ndarray, shape: tuple[int, int] | None, cv2: Any) -> float:
    if shape is None or len(points) < 3:
        return 0.0
    height, width = shape[:2]
    if height <= 0 or width <= 0:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(np.clip(cv2.contourArea(hull) / (width * height), 0.0, 1.0))


def _median_sampson(F: np.ndarray | None, left: np.ndarray, right: np.ndarray) -> float | None:
    if F is None or not len(left):
        return None
    ones = np.ones((len(left), 1))
    x1 = np.hstack((left, ones))
    x2 = np.hstack((right, ones))
    fx1 = (F @ x1.T).T
    ftx2 = (F.T @ x2.T).T
    numerator = np.sum(x2 * fx1, axis=1) ** 2
    denominator = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
    valid = denominator > np.finfo(float).eps
    if not valid.any():
        return None
    return float(np.median(numerator[valid] / denominator[valid]))


def _parallax_degrees(
    normalized_i: np.ndarray, normalized_j: np.ndarray, rotation_i_to_j: np.ndarray
) -> np.ndarray:
    if not len(normalized_i):
        return np.empty(0, dtype=float)
    rays_i = np.column_stack((normalized_i, np.ones(len(normalized_i))))
    rays_j = np.column_stack((normalized_j, np.ones(len(normalized_j))))
    rays_i /= np.linalg.norm(rays_i, axis=1, keepdims=True)
    rays_j /= np.linalg.norm(rays_j, axis=1, keepdims=True)
    rays_j_in_i = (np.asarray(rotation_i_to_j).T @ rays_j.T).T
    cosine = np.clip(np.sum(rays_i * rays_j_in_i, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _cv2():
    try:
        import cv2  # type: ignore
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pair geometry requires optional opencv-python-headless; install .[video]"
        ) from error
    return cv2


__all__ = [
    "ImageTransform",
    "PairGeometryResult",
    "grid_occupancy",
    "resize_pad_points",
    "verify_pair",
]
