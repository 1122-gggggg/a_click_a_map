"""Query-calibration resolution for the portable localization stage.

Reads the mapping stage's ``output/intrinsics.json`` query contract
(``refined_cameras`` / ``query_contract.cameras``) and selects the single
PINHOLE model the query stream must honor. Stdlib + numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SUPPORTED_SOURCES = (
    "EXPLICIT_CALIBRATION",
    "APPROXIMATE_AUTO_FOCAL",
    "APPROXIMATE_AUTO_CALIBRATE",
)


@dataclass(frozen=True)
class QueryCalibration:
    width: int
    height: int
    K: np.ndarray
    camera_id: int
    calibration_source: str
    refined_intrinsics: bool

    def as_deploy_dict(self) -> dict[str, Any]:
        """Intrinsics dict in the deploy_loop scaled_pinhole_parameters format."""
        return {
            "schema_version": 1,
            "image_width": self.width,
            "image_height": self.height,
            "K": np.asarray(self.K, dtype=float).tolist(),
            "images_are_undistorted": True,
            "camera_id": self.camera_id,
            "calibration_source": self.calibration_source,
            "refined_intrinsics": self.refined_intrinsics,
            "policy": "Fixed PINHOLE intrinsics; downstream refinement disabled",
        }


def scale_pinhole(K: np.ndarray, *, source_size: tuple[int, int],
                  target_size: tuple[int, int]) -> np.ndarray:
    """Resize-only rescaling of an undistorted PINHOLE matrix."""
    source_width, source_height = (int(v) for v in source_size)
    target_width, target_height = (int(v) for v in target_size)
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


def scaled_fxfycxcy(calibration: QueryCalibration, *, width: int, height: int) -> tuple[float, float, float, float]:
    """fx/fy/cx/cy scaled to an exact query resolution (deploy_loop convention)."""
    matrix = scale_pinhole(
        calibration.K,
        source_size=(calibration.width, calibration.height),
        target_size=(int(width), int(height)),
    )
    return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])


def _entry_matrix(entry: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(entry.get("K"), dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"camera {entry.get('camera_id')} has an invalid K matrix")
    if matrix[1, 0] != 0.0 or matrix[2, 0] != 0.0 or matrix[2, 1] != 0.0 or matrix[2, 2] != 1.0:
        raise ValueError(f"camera {entry.get('camera_id')} K is not upper-triangular with K[2,2]=1")
    if min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError(f"camera {entry.get('camera_id')} has non-positive focal lengths")
    return matrix


def resolve_query_calibration(
    intrinsics_payload: Mapping[str, Any],
    *,
    camera_id: int | None = None,
    query_intrinsics: str | Path | None = None,
) -> QueryCalibration:
    """Select the single query calibration from a mapping intrinsics.json payload.

    Prefers ``query_contract.cameras`` (identical to ``refined_cameras``), i.e.
    the bundle-adjusted map cameras -- never the seed. Multiple cameras
    require ``camera_id``. A keyframes-only payload without ``cameras[]`` is
    rejected unless an explicit ``--query_intrinsics`` override file is given,
    in which case the override is validated and marked EXPLICIT_QUERY_OVERRIDE.
    """
    import json

    if query_intrinsics is not None:
        override = json.loads(Path(query_intrinsics).expanduser().read_text(encoding="utf-8"))
        if override.get("images_are_undistorted") is not True:
            raise ValueError("--query_intrinsics must set images_are_undistorted=true")
        width = int(override.get("image_width") or 0)
        height = int(override.get("image_height") or 0)
        if min(width, height) <= 0:
            raise ValueError("--query_intrinsics needs positive image_width/image_height")
        model = str(override.get("camera_model") or "PINHOLE").upper()
        if model != "PINHOLE":
            raise ValueError("--query_intrinsics only supports PINHOLE")
        entry = {"camera_id": override.get("camera_id", -1), "K": override.get("K")}
        return QueryCalibration(
            width=width, height=height, K=_entry_matrix(entry),
            camera_id=int(override.get("camera_id", -1)),
            calibration_source="EXPLICIT_QUERY_OVERRIDE", refined_intrinsics=False,
        )
    contract = intrinsics_payload.get("query_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("intrinsics.json has no query_contract object")
    if contract.get("images_are_undistorted") is not True:
        raise ValueError("query contract requires undistorted PINHOLE input")
    model = str(contract.get("camera_model") or "PINHOLE").upper()
    if model != "PINHOLE":
        raise ValueError(f"unsupported query camera_model {model!r}: only PINHOLE is supported")
    cameras = contract.get("cameras")
    if isinstance(cameras, Mapping):
        cameras = cameras.get("cameras", cameras)
    if not cameras:
        raise ValueError(
            "query_contract has no cameras[] (keyframes-only seed contract); "
            "localization needs a completed map's refined cameras, or pass "
            "--query_intrinsics with the query camera's own undistorted PINHOLE "
            "calibration; never substitute the map seed for another camera"
        )
    entries = list(cameras)
    if camera_id is None and len(entries) > 1:
        available = sorted(int(e.get("camera_id", i)) for i, e in enumerate(entries))
        raise ValueError(
            f"map has {len(entries)} cameras {available}; pass --camera_id to select "
            "the query camera (query frames must come from that camera, already "
            "undistorted to PINHOLE)"
        )
    if camera_id is None:
        entry = dict(entries[0])
    else:
        matches = [dict(e) for e in entries if int(e.get("camera_id", -1)) == int(camera_id)]
        if not matches:
            available = sorted(int(e.get("camera_id", i)) for i, e in enumerate(entries))
            raise ValueError(f"--camera_id {camera_id} not in map cameras {available}")
        entry = matches[0]
    width = int(entry.get("width") or 0)
    height = int(entry.get("height") or 0)
    if min(width, height) <= 0:
        raise ValueError(f"camera {entry.get('camera_id')} has invalid width/height")
    source = str(contract.get("calibration_source") or "UNKNOWN")
    if source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"query contract calibration_source {source!r} is not one of "
            f"{list(SUPPORTED_SOURCES)}; refusing to guess what it means"
        )
    return QueryCalibration(
        width=width, height=height, K=_entry_matrix(entry),
        camera_id=int(entry.get("camera_id", 0)),
        calibration_source=source,
        refined_intrinsics=bool(contract.get("refined_intrinsics", False)),
    )


__all__ = [
    "QueryCalibration",
    "SUPPORTED_SOURCES",
    "resolve_query_calibration",
    "scale_pinhole",
    "scaled_fxfycxcy",
]
