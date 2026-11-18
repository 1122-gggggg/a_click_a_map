"""Calibration format shared by the desktop and original CLI entry points."""
from __future__ import annotations

import json
import math
from pathlib import Path


def load_calibration(path: Path) -> dict:
    """Read an explicit, already-undistorted PINHOLE calibration for one camera mode."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    if data.get("images_are_undistorted") is not True:
        raise ValueError("images_are_undistorted must be true (undistort the footage first)")
    if str(data.get("camera_model", "PINHOLE")).upper() != "PINHOLE":
        raise ValueError("camera_model must be PINHOLE")
    for field in ("image_width", "image_height"):
        value = data.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    k = data.get("K")
    if not isinstance(k, list) or len(k) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in k):
        raise ValueError("K must be a 3×3 matrix")
    try:
        finite = all(type(x) in (int, float) and math.isfinite(x) for row in k for x in row)
    except OverflowError:
        finite = False
    if not finite or k[0][0] <= 0 or k[1][1] <= 0 or k[0][1] != 0 or k[1][0] != 0 or k[2] != [0, 0, 1]:
        raise ValueError("K must be [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] with finite values and positive fx/fy")
    return data
