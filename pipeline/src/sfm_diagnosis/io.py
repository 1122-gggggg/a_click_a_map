from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .models import CameraIntrinsics, MapData


def find_colmap_model(path: str | Path) -> Path:
    """Resolve a GlueMap/COLMAP sparse model directory.

    GlueMap writes COLMAP-compatible sparse reconstructions. The caller may pass
    the model directory itself, a project root containing sparse/0, or a GlueMap
    result root. If more than one reconstruction is found, the one with the most
    point file bytes is selected as a practical main-component heuristic.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    def is_model_dir(p: Path) -> bool:
        return (
            (p / "cameras.bin").exists()
            and (p / "images.bin").exists()
            and (p / "points3D.bin").exists()
        ) or (
            (p / "cameras.txt").exists()
            and (p / "images.txt").exists()
            and (p / "points3D.txt").exists()
        )

    if root.is_dir() and is_model_dir(root):
        return root
    common = [root / "sparse" / "0", root / "0", root / "sparse"]
    for candidate in common:
        if candidate.is_dir() and is_model_dir(candidate):
            return candidate

    candidates: list[Path] = []
    for marker in ("points3D.bin", "points3D.txt"):
        for point_file in root.rglob(marker):
            if is_model_dir(point_file.parent):
                candidates.append(point_file.parent)
    if not candidates:
        raise FileNotFoundError(
            f"No COLMAP sparse model found under {root}. Expected cameras/images/points3D .bin or .txt."
        )

    def score(p: Path) -> int:
        point_file = p / ("points3D.bin" if (p / "points3D.bin").exists() else "points3D.txt")
        return point_file.stat().st_size

    return max(set(candidates), key=score)


def load_gluemap(path: str | Path) -> MapData:
    """Load a GlueMap result via PyCOLMAP.

    GlueMap's reconstruction output is COLMAP format, so this loader deliberately
    uses PyCOLMAP rather than depending on GlueMap internals. This also makes the
    diagnostics usable with COLMAP and GLOMAP reconstructions.
    """
    model_dir = find_colmap_model(path)
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyCOLMAP is required to read GlueMap/COLMAP models. "
            "Install with `pip install -e '.[colmap]'` or `pip install pycolmap`."
        ) from exc

    rec = pycolmap.Reconstruction(str(model_dir))

    cameras: dict[int, CameraIntrinsics] = {}
    for camera_id, camera in rec.cameras.items():
        cameras[int(camera_id)] = CameraIntrinsics(
            camera_id=int(camera_id),
            model_name=_camera_model_name(camera),
            width=int(camera.width),
            height=int(camera.height),
            fx=float(camera.focal_length_x),
            fy=float(camera.focal_length_y),
            cx=float(camera.principal_point_x),
            cy=float(camera.principal_point_y),
        )

    image_items = sorted(rec.images.items(), key=lambda kv: int(kv[0]))
    image_ids: list[int] = []
    image_names: list[str] = []
    image_camera_ids: list[int] = []
    image_centers: list[np.ndarray] = []
    image_R_wc: list[np.ndarray] = []
    for image_id, image in image_items:
        if not bool(image.has_pose):
            continue
        cam_from_world = image.cam_from_world()
        R_cw = np.asarray(cam_from_world.rotation.matrix(), dtype=float)
        image_ids.append(int(image_id))
        image_names.append(str(image.name))
        image_camera_ids.append(int(image.camera_id))
        image_centers.append(np.asarray(image.projection_center(), dtype=float).reshape(3))
        image_R_wc.append(R_cw.T)

    point_items = sorted(rec.points3D.items(), key=lambda kv: int(kv[0]))
    point_ids: list[int] = []
    xyz: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    errors: list[float] = []
    track_lengths: list[int] = []
    track_image_ids: list[np.ndarray] = []
    for point_id, point in point_items:
        elems = list(point.track.elements)
        point_ids.append(int(point_id))
        xyz.append(np.asarray(point.xyz, dtype=float).reshape(3))
        rgb.append(np.asarray(point.color, dtype=np.uint8).reshape(3))
        errors.append(float(point.error) if float(point.error) >= 0 else 0.0)
        track_lengths.append(len(elems))
        track_image_ids.append(np.asarray([int(e.image_id) for e in elems], dtype=np.int64))

    return MapData(
        point_ids=np.asarray(point_ids, dtype=np.int64),
        points_xyz=np.asarray(xyz, dtype=float).reshape(-1, 3),
        point_rgb=np.asarray(rgb, dtype=np.uint8).reshape(-1, 3),
        point_errors=np.asarray(errors, dtype=float),
        track_lengths=np.asarray(track_lengths, dtype=np.int32),
        track_image_ids=track_image_ids,
        image_ids=np.asarray(image_ids, dtype=np.int64),
        image_names=image_names,
        image_camera_ids=np.asarray(image_camera_ids, dtype=np.int64),
        image_centers=np.asarray(image_centers, dtype=float).reshape(-1, 3),
        image_R_wc=np.asarray(image_R_wc, dtype=float).reshape(-1, 3, 3),
        cameras=cameras,
        metadata={
            "source": "gluemap/colmap",
            "model_dir": str(model_dir),
            "summary": str(rec.summary()),
        },
    )


def _camera_model_name(camera: object) -> str:
    """Normalize PyCOLMAP <=3.12 ``model_name`` and 3.13+ ``model.name``."""

    legacy = getattr(camera, "model_name", None)
    if legacy is not None:
        return str(legacy)
    model = getattr(camera, "model", None)
    name = getattr(model, "name", None)
    if name is not None:
        return str(name)
    text = str(model)
    return text.rsplit(".", 1)[-1]


def write_json(path: str | Path, payload: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def write_csv(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: object):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        try:
            return sorted(value)
        except TypeError:
            return list(value)
    raise TypeError(type(value).__name__)
