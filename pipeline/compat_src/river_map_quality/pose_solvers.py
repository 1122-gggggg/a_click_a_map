"""OpenCV-backed two-view and PnP solver controls used by the forensic runner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from river_map_quality.pose_attribution import (
    PlanarityDiagnostics,
    camera_center,
    planarity_diagnostics,
)


@dataclass(frozen=True)
class EssentialTrial:
    seed: int
    threshold_px: float
    rotation: np.ndarray
    translation_direction: np.ndarray
    inliers: int


@dataclass(frozen=True)
class PnPCandidate:
    solver: str
    pose: np.ndarray
    positive_depth_ratio: float
    reprojection_p50: float
    reprojection_p90: float


@dataclass(frozen=True)
class PnPEnvelope:
    planarity: PlanarityDiagnostics
    candidates: tuple[PnPCandidate, ...]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the EDM runtime
        raise RuntimeError(
            "OpenCV is required for Essential Matrix and PnP solver controls"
        ) from exc
    return cv2


def _camera_matrix(matrix: np.ndarray) -> np.ndarray:
    camera = np.asarray(matrix, dtype=np.float64)
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("camera matrix must be finite with shape (3, 3)")
    if camera[0, 0] <= 0 or camera[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return camera


def project_points(
    world_points: np.ndarray,
    pose: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through a world-to-camera pose."""

    points = np.asarray(world_points, dtype=float)
    transform = np.asarray(pose, dtype=float)
    camera = _camera_matrix(camera_matrix)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("invalid points or pose shape")
    camera_points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
    positive = camera_points[:, 2] > 1e-9
    projected = np.full((len(points), 2), np.nan, dtype=float)
    projected[positive, 0] = (
        camera[0, 0] * camera_points[positive, 0] / camera_points[positive, 2] + camera[0, 2]
    )
    projected[positive, 1] = (
        camera[1, 1] * camera_points[positive, 1] / camera_points[positive, 2] + camera[1, 2]
    )
    return projected, positive


def estimate_essential_trials(
    points1: np.ndarray,
    points2: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    seeds: Sequence[int] = tuple(range(10)),
    thresholds_px: Sequence[float] = (0.5, 1.0, 2.0),
    probability: float = 0.999,
    max_iterations: int = 10_000,
) -> tuple[EssentialTrial, ...]:
    """Estimate calibrated relative poses across deterministic RANSAC settings."""

    cv2 = _cv2()
    first = np.asarray(points1, dtype=np.float64)
    second = np.asarray(points2, dtype=np.float64)
    camera = _camera_matrix(camera_matrix)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 2 or len(first) < 5:
        raise ValueError("two-view points must have matching shape (N, 2), N >= 5")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("two-view points must be finite")
    output: list[EssentialTrial] = []
    for seed in seeds:
        for threshold in thresholds_px:
            cv2.setRNGSeed(int(seed))
            essential, mask = cv2.findEssentialMat(
                first,
                second,
                camera,
                method=cv2.RANSAC,
                prob=float(probability),
                threshold=float(threshold),
                maxIters=int(max_iterations),
            )
            if essential is None or mask is None:
                continue
            candidates = np.asarray(essential, dtype=float).reshape(-1, 3, 3)
            best = None
            for candidate in candidates:
                count, rotation, translation, recovered_mask = cv2.recoverPose(
                    candidate,
                    first,
                    second,
                    camera,
                    mask=np.asarray(mask, dtype=np.uint8).copy(),
                )
                recovered = int(np.count_nonzero(recovered_mask))
                score = (recovered, int(count))
                if best is None or score > best[0]:
                    best = (score, rotation, translation)
            if best is None:
                continue
            translation = np.asarray(best[2], dtype=float).reshape(3)
            norm = np.linalg.norm(translation)
            if norm <= np.finfo(float).eps:
                continue
            output.append(
                EssentialTrial(
                    seed=int(seed),
                    threshold_px=float(threshold),
                    rotation=np.asarray(best[1], dtype=float),
                    translation_direction=translation / norm,
                    inliers=best[0][0],
                )
            )
    if not output:
        raise RuntimeError("Essential Matrix estimation failed for every trial")
    return tuple(output)


def _pose_from_vectors(rotation_vector: np.ndarray, translation: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=float))
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return pose


def _candidate(
    solver: str,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> PnPCandidate:
    pose = _pose_from_vectors(rotation_vector, translation)
    projected, positive = project_points(world_points, pose, camera_matrix)
    valid = positive & np.isfinite(projected).all(axis=1)
    errors = np.linalg.norm(projected[valid] - image_points[valid], axis=1)
    return PnPCandidate(
        solver=solver,
        pose=pose,
        positive_depth_ratio=float(np.mean(positive)),
        reprojection_p50=float(np.percentile(errors, 50)) if len(errors) else float("inf"),
        reprojection_p90=float(np.percentile(errors, 90)) if len(errors) else float("inf"),
    )


def solve_pnp_envelope(
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    initial_poses: Mapping[str, np.ndarray] | None = None,
) -> PnPEnvelope:
    """Compare SQPnP/ITERATIVE and, for planar sets, the two IPPE solutions."""

    cv2 = _cv2()
    world = np.asarray(world_points, dtype=np.float64)
    image = np.asarray(image_points, dtype=np.float64)
    camera = _camera_matrix(camera_matrix)
    if world.ndim != 2 or world.shape[1] != 3 or image.shape != (len(world), 2) or len(world) < 6:
        raise ValueError("PnP envelope requires at least six 2D-3D correspondences")
    if not np.isfinite(world).all() or not np.isfinite(image).all():
        raise ValueError("PnP correspondences must be finite")
    initial_poses = dict(initial_poses or {})
    diagnostic_center = (
        camera_center(next(iter(initial_poses.values()))) if initial_poses else np.zeros(3)
    )
    planarity = planarity_diagnostics(world, camera_center_world=diagnostic_center)
    candidates: list[PnPCandidate] = []

    for name, flag in (
        ("SQPNP", cv2.SOLVEPNP_SQPNP),
        ("ITERATIVE_NO_GUESS", cv2.SOLVEPNP_ITERATIVE),
    ):
        success, rotation_vector, translation = cv2.solvePnP(
            world,
            image,
            camera,
            None,
            flags=flag,
        )
        if success:
            candidates.append(_candidate(name, rotation_vector, translation, world, image, camera))

    for label, pose in sorted(initial_poses.items()):
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("initial poses must be 4x4")
        rotation_vector, _ = cv2.Rodrigues(matrix[:3, :3])
        translation = matrix[:3, 3].reshape(3, 1).copy()
        success, rotation_vector, translation = cv2.solvePnP(
            world,
            image,
            camera,
            None,
            rotation_vector,
            translation,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            candidates.append(
                _candidate(
                    f"ITERATIVE_{label}",
                    rotation_vector,
                    translation,
                    world,
                    image,
                    camera,
                )
            )

    if planarity.is_planar:
        count, rotations, translations, _errors = cv2.solvePnPGeneric(
            world,
            image,
            camera,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if count:
            for index, (rotation_vector, translation) in enumerate(
                zip(rotations, translations, strict=True), 1
            ):
                candidates.append(
                    _candidate(
                        f"IPPE_{index}",
                        rotation_vector,
                        translation,
                        world,
                        image,
                        camera,
                    )
                )
    if not candidates:
        raise RuntimeError("every PnP control solver failed")
    return PnPEnvelope(planarity=planarity, candidates=tuple(candidates))
