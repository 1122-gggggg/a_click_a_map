"""Tiered leave-one-out planning and aligned camera stability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from sfm_diagnosis.sim3 import estimate_similarity_3d


@dataclass(frozen=True)
class LOOTarget:
    kind: str
    target_id: str
    reason: str


def tiered_loo_targets(
    keyframes: Iterable[Mapping[str, Any]], roles: Iterable[Mapping[str, Any]]
) -> tuple[LOOTarget, ...]:
    """Return every mapping video plus every load-bearing/high-risk segment."""

    frame_rows = list(keyframes)
    role_rows = list(roles)
    targets = {
        LOOTarget("video", str(row["video_id"]), "all_mapping_videos_exact")
        for row in frame_rows
        if row.get("evaluation_role", "MAPPING") == "MAPPING"
    }
    for row in role_rows:
        segment = str(row.get("segment_id") or row.get("segment") or "")
        if not segment:
            continue
        if row.get("post_sfm_role") == "BRIDGE":
            targets.add(LOOTarget("segment", segment, "verified_bridge"))
        elif row.get("risk") in {"HIGH", "CRITICAL"}:
            targets.add(LOOTarget("segment", segment, "high_influence"))
    return tuple(sorted(targets, key=lambda row: (row.kind, row.target_id)))


def aligned_camera_stability(
    reference: Mapping[str, Mapping[str, Any]],
    trial: Mapping[str, Mapping[str, Any]],
    *,
    scene_scale: float | None = None,
) -> dict[str, Any]:
    common = sorted(set(reference) & set(trial))
    if len(common) < 4:
        return {"status": "INSUFFICIENT_COMMON_CAMERAS", "common_cameras": len(common)}
    reference_centers = np.asarray([reference[name]["center"] for name in common], dtype=float)
    trial_centers = np.asarray([trial[name]["center"] for name in common], dtype=float)
    try:
        transform = estimate_similarity_3d(trial_centers, reference_centers)
    except ValueError as error:
        return {
            "status": "DEGENERATE_ALIGNMENT",
            "common_cameras": len(common),
            "error": str(error),
        }
    aligned = transform.apply(trial_centers)
    residuals = np.linalg.norm(aligned - reference_centers, axis=1)
    scale = scene_scale or _scene_scale(reference_centers)
    normalized = residuals / max(scale, 1e-9)
    rotation_residuals = []
    for name in common:
        reference_rotation = reference[name].get("rotation")
        trial_rotation = trial[name].get("rotation")
        if reference_rotation is None or trial_rotation is None:
            continue
        aligned_trial = np.asarray(trial_rotation, dtype=float) @ transform.rotation.T
        delta = np.asarray(reference_rotation, dtype=float).T @ aligned_trial
        rotation_residuals.append(_rotation_angle_deg(delta))
    return {
        "status": "OK",
        "common_cameras": len(common),
        "sim3_scale": transform.scale,
        "sim3_scale_log_abs": abs(float(np.log(transform.scale))),
        "position_median_normalized": float(np.median(normalized)),
        "position_p90_normalized": float(np.percentile(normalized, 90)),
        "rotation_median_deg": (
            None if not rotation_residuals else float(np.median(rotation_residuals))
        ),
        "rotation_p90_deg": (
            None if not rotation_residuals else float(np.percentile(rotation_residuals, 90))
        ),
    }


def loo_warning(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    warnings = []
    if metrics.get("status") != "OK":
        warnings.append(str(metrics.get("status")))
        return tuple(warnings)
    if float(metrics.get("rotation_p90_deg") or 0.0) > 2.0:
        warnings.append("LOO_ROTATION_GT_2_DEG")
    if float(metrics.get("position_p90_normalized") or 0.0) > 0.02:
        warnings.append("LOO_POSITION_GT_0_02")
    return tuple(warnings)


def _scene_scale(centers: np.ndarray) -> float:
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


__all__ = ["LOOTarget", "aligned_camera_stability", "loo_warning", "tiered_loo_targets"]
