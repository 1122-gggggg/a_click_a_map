"""Intrinsics validation and explicit calibration policy.

The mapping stage refuses to silently assume any calibration. Every run
needs either:

* ``--intrinsics <file.json>``: an explicit calibration file (see the
  ``CALIBRATION_SCHEMA`` below), or
* ``--auto_focal F``: an explicitly documented, approximate initial focal
  length in pixels at the stated calibration resolution, used only to seed
  reconstruction. The run receipt records ``calibration_source`` as
  ``APPROXIMATE_AUTO_FOCAL`` and the resulting map must NOT be presented
  as metrically calibrated.

The mapping intrinsics contract is: already-undistorted PINHOLE at a known
resolution, rescaled per source resolution. Distorted input is rejected;
undistortion is an explicit preprocessing step, not a silent default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CALIBRATION_SCHEMA = """\
Calibration file schema (JSON object):
{
  "image_width": <int, calibration resolution width>,
  "image_height": <int, calibration resolution height>,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "images_are_undistorted": true,
  "source_dist_reference_only": [<optional record of source distortion>],
  "camera_model": "PINHOLE"  (optional, defaults to PINHOLE)
}
"""


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    K: np.ndarray
    source: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_width": self.width,
            "image_height": self.height,
            "K": np.asarray(self.K, dtype=float).tolist(),
            "images_are_undistorted": True,
            "calibration_source": self.source,
        }


def auto_focal_for_resolution(width: int, height: int) -> float:
    """Documented video-only heuristic: 1.2 * max(width, height) pixels."""
    if min(int(width), int(height)) <= 0:
        raise ValueError("resolution dimensions must be positive")
    return 1.2 * float(max(int(width), int(height)))


def calibration_from_auto_focal(
    focal: float, resolution: tuple[int, int], *, source: str
) -> Calibration:
    width, height = int(resolution[0]), int(resolution[1])
    if min(width, height) <= 0:
        raise ValueError("auto resolution dimensions must be positive")
    if not focal or float(focal) <= 0:
        raise ValueError("auto focal length must be positive (pixels)")
    K = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return Calibration(width=width, height=height, K=K, source=source)


def load_calibration(
    path: str | Path | None,
    *,
    auto_focal: float | None = None,
    auto_resolution: tuple[int, int] | None = None,
) -> Calibration:
    """Load and validate an explicit calibration (no silent defaults)."""
    if path is not None and auto_focal is not None:
        raise ValueError("pass --intrinsics or --auto_focal, not both")
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return _from_payload(payload, source="EXPLICIT_CALIBRATION", path=str(path))
    if auto_focal is None:
        raise ValueError(
            "no calibration: pass --intrinsics <file.json> (recommended), "
            "explicitly opt into --auto_focal F [--auto_resolution WxH], or "
            "pass --auto_calibrate for the documented video-only heuristic "
            "(focal 1.2*max(W,H) at the first source resolution, refined by "
            "bundle adjustment). Arbitrary cameras never assume a focal "
            "length silently.\n" + CALIBRATION_SCHEMA
        )
    if auto_focal <= 0:
        raise ValueError("--auto_focal must be positive (pixels)")
    if auto_resolution is None:
        raise ValueError(
            "--auto_focal requires --auto_resolution WxH so the focal length "
            "has a defined resolution; it is rescaled per source resolution."
        )
    return calibration_from_auto_focal(
        auto_focal, auto_resolution, source="APPROXIMATE_AUTO_FOCAL"
    )


def _from_payload(payload: Mapping[str, Any], *, source: str, path: str | None) -> Calibration:
    if payload.get("images_are_undistorted") is not True:
        raise ValueError(
            "calibration must set images_are_undistorted=true: this stage maps "
            "already-undistorted PINHOLE frames only. Undistort explicitly first."
        )
    width = int(payload.get("image_width") or 0)
    height = int(payload.get("image_height") or 0)
    if min(width, height) <= 0:
        raise ValueError("calibration needs positive image_width/image_height")
    K = np.asarray(payload.get("K"), dtype=float)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("calibration K must be a finite 3x3 matrix")
    if K[1, 0] != 0.0 or K[2, 0] != 0.0 or K[2, 1] != 0.0 or K[2, 2] != 1.0:
        raise ValueError("calibration K must be upper-triangular with K[2,2]=1")
    if min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("calibration focal lengths must be positive")
    model = str(payload.get("camera_model") or "PINHOLE").upper()
    if model != "PINHOLE":
        raise ValueError(f"unsupported camera_model {model!r}: only PINHOLE is supported")
    return Calibration(width=width, height=height, K=K, source=source, path=path)


def scale_intrinsics(
    K: np.ndarray,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize-only rescaling of an undistorted PINHOLE matrix."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target dimensions must be positive")
    matrix = np.asarray(K, dtype=float).copy()
    if matrix.shape != (3, 3):
        raise ValueError("K must be 3x3")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    matrix[0, 0] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[0, 2] *= scale_x
    matrix[1, 2] *= scale_y
    return matrix


def calibration_matrix_for_resolution(
    calibration: Calibration,
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Scale the calibration to a same-aspect target resolution."""
    target_width, target_height = map(int, target_size)
    if calibration.width * target_height != calibration.height * target_width:
        raise ValueError(
            f"calibration {calibration.width}x{calibration.height} and target "
            f"{target_width}x{target_height} must share the same aspect ratio; "
            "crop/scale explicitly instead of silently stretching pixels"
        )
    return scale_intrinsics(
        calibration.K,
        source_size=(calibration.width, calibration.height),
        target_size=(target_width, target_height),
    )


def query_calibration_contract(
    calibration: Calibration, registered_size: tuple[int, int]
) -> dict[str, Any]:
    """Calibration contract downstream query stages must honor.

    Used for keyframes-only output where no reconstruction exists yet.
    Full runs overwrite output/intrinsics.json with refined_query_contract()
    built from the actual bundle-adjusted map cameras.
    """
    payload = calibration.as_dict()
    payload["contract"] = (
        "KEYFRAMES-ONLY seed contract: query frames MUST be undistorted to "
        "this PINHOLE model and rescaled from the stated calibration "
        "resolution; frames from another camera need their own calibration, "
        "never this map's K. A completed map replaces this with the refined "
        "camera contract."
    )
    payload["calibration_resolution"] = [calibration.width, calibration.height]
    payload["map_resolution_hint"] = [int(registered_size[0]), int(registered_size[1])]
    return payload


def refined_query_contract(
    model_cameras: list[dict[str, Any]],
    *,
    calibration_source: str,
    refined_intrinsics: bool,
) -> dict[str, Any]:
    """Build the query contract from the REFINED reconstruction cameras.

    Localization must use the actual bundle-adjusted map cameras, never the
    initial auto-focal seed. Each entry carries its own W/H/K so mixed
    resolutions stay exact; per-image camera assignment lives in the model.
    """
    if not model_cameras:
        raise ValueError("refined query contract needs at least one camera")
    return {
        "calibration_source": calibration_source,
        "refined_intrinsics": bool(refined_intrinsics),
        "camera_model": "PINHOLE",
        "images_are_undistorted": True,
        "cameras": [dict(entry) for entry in model_cameras],
        "contract": (
            "Localize with these REFINED map cameras at their stated W/H "
            "(rescale K linearly for other query sizes of the same camera). "
            "Query frames MUST be undistorted to PINHOLE first; frames from "
            "another camera need their own calibration, never this map's K. "
            + (
                "Approximate auto-calibrated map: treat scale/focal as "
                "unvalidated."
                if calibration_source != "EXPLICIT_CALIBRATION"
                else "Explicit-calibration map: intrinsics validated by seed."
            )
        ),
    }


__all__ = [
    "CALIBRATION_SCHEMA",
    "Calibration",
    "auto_focal_for_resolution",
    "calibration_from_auto_focal",
    "calibration_matrix_for_resolution",
    "load_calibration",
    "query_calibration_contract",
    "refined_query_contract",
    "scale_intrinsics",
]
