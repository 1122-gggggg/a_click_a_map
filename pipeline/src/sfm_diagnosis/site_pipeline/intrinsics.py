"""Camera-profile validation and intrinsics scaling across resize/crop modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class IntrinsicsProfile:
    group_id: str
    camera_mode: str
    width: int
    height: int
    crop_state: str
    stabilization_state: str
    K: np.ndarray
    distortion: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        matrix = np.asarray(self.K, dtype=float)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("K must be a finite 3x3 matrix")
        if min(self.width, self.height, matrix[0, 0], matrix[1, 1]) <= 0:
            raise ValueError("image dimensions and focal lengths must be positive")
        object.__setattr__(self, "K", matrix)


def scale_intrinsics(
    K: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    crop_xywh: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Apply crop then resize to focal lengths and principal point."""

    source_width, source_height = source_size
    target_width, target_height = target_size
    if crop_xywh is None:
        crop_x, crop_y, crop_width, crop_height = 0.0, 0.0, source_width, source_height
    else:
        crop_x, crop_y, crop_width, crop_height = map(float, crop_xywh)
    if min(source_width, source_height, target_width, target_height, crop_width, crop_height) <= 0:
        raise ValueError("source, target and crop dimensions must be positive")
    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + crop_width > source_width
        or crop_y + crop_height > source_height
    ):
        raise ValueError("crop lies outside the source image")
    matrix = np.asarray(K, dtype=float).copy()
    if matrix.shape != (3, 3):
        raise ValueError("K must be 3x3")
    scale_x, scale_y = target_width / crop_width, target_height / crop_height
    matrix[0, 0] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[0, 2] = (matrix[0, 2] - crop_x) * scale_x
    matrix[1, 2] = (matrix[1, 2] - crop_y) * scale_y
    return matrix


def calibration_matrix_for_resolution(
    calibration: Mapping[str, object],
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale an undistorted PINHOLE calibration to a same-aspect resolution."""

    if calibration.get("images_are_undistorted") is not True:
        raise ValueError("calibration must confirm that input images are already undistorted")
    source_width = int(calibration.get("image_width") or 0)
    source_height = int(calibration.get("image_height") or 0)
    target_width, target_height = map(int, target_size)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("calibration and target dimensions must be positive")
    if source_width * target_height != source_height * target_width:
        raise ValueError("calibration and target resolution must have the same aspect ratio")
    return scale_intrinsics(
        np.asarray(calibration.get("K"), dtype=float),
        source_size=(source_width, source_height),
        target_size=(target_width, target_height),
    )


def validate_intrinsics_group(
    rows: Iterable[Mapping[str, object]],
    *,
    allow_unknown_metadata: bool = False,
) -> tuple[str, ...]:
    """Return warnings; unsafe static-profile assumptions raise immediately."""

    material = list(rows)
    if not material:
        raise ValueError("intrinsics group is empty")
    modes = {str(row.get("camera_mode") or "") for row in material}
    crop_states = {str(row.get("crop_state") or "") for row in material}
    stabilization = {str(row.get("stabilization_state") or "") for row in material}
    if "" in modes or "" in crop_states or "" in stabilization:
        raise ValueError("camera mode, crop and stabilization must be confirmed")
    unknown = {"unknown", "unavailable", "unspecified"}
    contains_unknown = bool((modes | crop_states | stabilization) & unknown)
    if contains_unknown:
        if not allow_unknown_metadata:
            raise ValueError("camera mode, crop and stabilization must be confirmed")
        if len(modes) != 1 or len(crop_states) != 1 or len(stabilization) != 1:
            raise ValueError("known and unknown camera metadata cannot share an intrinsics group")
        resolutions = {
            (int(row["width"]), int(row["height"]))
            for row in material
            if row.get("width") and row.get("height")
        }
        if len(resolutions) != 1 or len(resolutions) != len(
            {
                (str(row.get("width") or ""), str(row.get("height") or ""))
                for row in material
            }
        ):
            raise ValueError("unknown camera metadata may be shared only at one resolution")
        return ("UNKNOWN_CAMERA_METADATA_SHARED_BY_RESOLUTION",)
    if len(modes) != 1 or len(crop_states) != 1:
        raise ValueError("different camera/crop modes require different intrinsics groups")
    if any(value.lower() not in {"off", "none", "disabled"} for value in stabilization):
        raise ValueError("digital stabilization requires a separate per-frame warp model")
    warnings = []
    aspect_ratios = {
        round(float(row["width"]) / float(row["height"]), 6)
        for row in material
        if row.get("width") and row.get("height")
    }
    if len(aspect_ratios) > 1:
        warnings.append("ASPECT_RATIO_CHANGED_REQUIRES_EXPLICIT_CROP")
    return tuple(warnings)


def scaled_profile(
    profile: IntrinsicsProfile,
    *,
    target_size: tuple[int, int],
    crop_xywh: tuple[float, float, float, float] | None = None,
) -> IntrinsicsProfile:
    return IntrinsicsProfile(
        group_id=profile.group_id,
        camera_mode=profile.camera_mode,
        width=target_size[0],
        height=target_size[1],
        crop_state=profile.crop_state,
        stabilization_state=profile.stabilization_state,
        K=scale_intrinsics(
            profile.K,
            source_size=(profile.width, profile.height),
            target_size=target_size,
            crop_xywh=crop_xywh,
        ),
        distortion=profile.distortion,
    )


__all__ = [
    "IntrinsicsProfile",
    "calibration_matrix_for_resolution",
    "scale_intrinsics",
    "scaled_profile",
    "validate_intrinsics_group",
]
