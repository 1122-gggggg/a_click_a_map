"""Strict dual-layer localization fusion without ground-truth-based selection."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

import numpy as np


def combine_layer_results(
    robust: Mapping[str, Any],
    dense: Mapping[str, Any],
    *,
    scene_scale: float,
    maximum_position_normalized: float = 0.02,
    maximum_rotation_deg: float = 2.0,
    prefer_robust_on_agreement: bool = False,
) -> dict[str, Any]:
    """Fuse two internally gated poses and reject cross-layer disagreement."""

    _validate_identity(robust, dense)
    if scene_scale <= 0 or not math.isfinite(scene_scale):
        raise ValueError("scene_scale must be finite and positive")
    if maximum_position_normalized <= 0 or maximum_rotation_deg <= 0:
        raise ValueError("cross-layer thresholds must be positive")

    position_delta, rotation_delta = _pose_disagreement(robust, dense, scene_scale)
    robust_success, dense_success = bool(robust.get("success")), bool(dense.get("success"))
    selected_layer: str | None
    status: str
    accepted = False
    if robust_success and not dense_success:
        selected_layer, status, accepted = "robust", "SINGLE_LAYER_STRICT_ACCEPT", True
    elif dense_success and not robust_success:
        selected_layer, status, accepted = "dense", "SINGLE_LAYER_STRICT_ACCEPT", True
    elif robust_success and dense_success:
        agrees = (
            position_delta is not None
            and rotation_delta is not None
            and position_delta <= maximum_position_normalized
            and rotation_delta <= maximum_rotation_deg
        )
        if agrees:
            selected_layer = (
                "robust" if prefer_robust_on_agreement else _better_layer(robust, dense)
            )
            status, accepted = "CROSS_LAYER_AGREEMENT", True
        else:
            selected_layer, status = None, "REJECT_CROSS_LAYER_DISAGREEMENT"
    else:
        selected_layer, status = None, "NO_LAYER_STRICT_ACCEPT"

    preferred = robust if (selected_layer or _better_layer(robust, dense)) == "robust" else dense
    output = dict(preferred)
    output.update(
        success=accepted,
        selected_layer=selected_layer,
        pose_consistency=status,
        robust_success=robust_success,
        dense_success=dense_success,
        cross_layer_position_normalized=position_delta,
        cross_layer_rotation_deg=rotation_delta,
        cross_layer_thresholds={
            "maximum_position_normalized": float(maximum_position_normalized),
            "maximum_rotation_deg": float(maximum_rotation_deg),
        },
        layer_support={
            "robust": _support_summary(robust),
            "dense": _support_summary(dense),
        },
    )
    return output


def combine_result_sets(
    robust_results: Iterable[Mapping[str, Any]],
    dense_results: Iterable[Mapping[str, Any]],
    *,
    scene_scale: float,
    maximum_position_normalized: float = 0.02,
    maximum_rotation_deg: float = 2.0,
    prefer_robust_on_agreement: bool = False,
) -> list[dict[str, Any]]:
    robust = {str(row["query_id"]): row for row in robust_results}
    dense = {str(row["query_id"]): row for row in dense_results}
    if set(robust) != set(dense) or len(robust) == 0:
        raise ValueError("robust and dense query sets must be identical and non-empty")
    return [
        combine_layer_results(
            robust[query_id],
            dense[query_id],
            scene_scale=scene_scale,
            maximum_position_normalized=maximum_position_normalized,
            maximum_rotation_deg=maximum_rotation_deg,
            prefer_robust_on_agreement=prefer_robust_on_agreement,
        )
        for query_id in sorted(robust)
    ]


def _validate_identity(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    for key in ("query_id", "session_id"):
        if str(first.get(key)) != str(second.get(key)):
            raise ValueError(f"localization layer {key} identities disagree")
    if abs(float(first.get("timestamp") or 0.0) - float(second.get("timestamp") or 0.0)) > 1e-9:
        raise ValueError("localization layer timestamps disagree")


def _pose_disagreement(
    first: Mapping[str, Any], second: Mapping[str, Any], scene_scale: float
) -> tuple[float | None, float | None]:
    try:
        first_position = np.asarray(first["estimated_position"], dtype=float)
        second_position = np.asarray(second["estimated_position"], dtype=float)
        first_rotation = np.asarray(first["estimated_R_wc"], dtype=float)
        second_rotation = np.asarray(second["estimated_R_wc"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, None
    if (
        first_position.shape != (3,)
        or second_position.shape != (3,)
        or first_rotation.shape != (3, 3)
        or second_rotation.shape != (3, 3)
        or not all(
            np.isfinite(value).all()
            for value in (first_position, second_position, first_rotation, second_rotation)
        )
    ):
        return None, None
    position = float(np.linalg.norm(first_position - second_position) / scene_scale)
    delta = first_rotation.T @ second_rotation
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return position, float(np.degrees(np.arccos(cosine)))


def _better_layer(robust: Mapping[str, Any], dense: Mapping[str, Any]) -> str:
    def score(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row.get("ransac_inliers") or 0),
            float(row.get("inlier_ratio") or 0.0),
            float(row.get("valid_2d3d") or 0),
            -float(row.get("reprojection_p90") or math.inf),
        )

    return "dense" if score(dense) > score(robust) else "robust"


def _support_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(row.get("success")),
        "registration_success": bool(row.get("registration_success")),
        "raw_matches": row.get("raw_matches"),
        "valid_2d3d": row.get("valid_2d3d"),
        "ransac_inliers": row.get("ransac_inliers"),
        "inlier_ratio": row.get("inlier_ratio"),
        "reprojection_p90": row.get("reprojection_p90"),
        "pose_consistency": row.get("pose_consistency"),
    }


__all__ = ["combine_layer_results", "combine_result_sets"]
