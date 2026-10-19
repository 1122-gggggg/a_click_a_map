from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from mapdoctor.model import Camera, ImageRecord, MapModel, Observation2D, Point3D, TrackElement


def _non_comment_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        raise ValueError("Invalid zero quaternion in images.txt")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def _transpose_mv(rotation: list[list[float]], vector: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = vector
    return (
        rotation[0][0] * x + rotation[1][0] * y + rotation[2][0] * z,
        rotation[0][1] * x + rotation[1][1] * y + rotation[2][1] * z,
        rotation[0][2] * x + rotation[1][2] * y + rotation[2][2] * z,
    )


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    n = math.sqrt(sum(value * value for value in vector))
    if n == 0:
        return (0.0, 0.0, 1.0)
    return (vector[0] / n, vector[1] / n, vector[2] / n)


def _looks_like_model(path: Path) -> bool:
    text_ok = all((path / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt"))
    binary_ok = all((path / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))
    return text_ok or binary_ok


def resolve_sparse_model_path(path: str | Path) -> Path:
    """Resolve either a model directory or its common parent containing 0/."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if _looks_like_model(root):
        return root
    candidate = root / "0"
    if candidate.is_dir() and _looks_like_model(candidate):
        return candidate
    raise ValueError(
        f"{root} does not look like a COLMAP sparse reconstruction. "
        "Expected cameras/images/points3D .txt or .bin files, optionally under a 0/ subdirectory."
    )


def detect_colmap_format(path: str | Path) -> str:
    root = resolve_sparse_model_path(path)
    return "colmap-text" if (root / "cameras.txt").exists() else "colmap-binary"


def _parse_text_model(path: Path, source: str) -> MapModel:
    cameras_path = path / "cameras.txt"
    images_path = path / "images.txt"
    points_path = path / "points3D.txt"
    model = MapModel(source=source, format="colmap-text", metadata={"resolved_path": str(path)})

    for line in _non_comment_lines(cameras_path):
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed camera line: {line}")
        camera_id = int(parts[0])
        model.cameras[camera_id] = Camera(
            id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(value) for value in parts[4:]),
        )

    raw_lines = images_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        pose = line.split(maxsplit=9)
        if len(pose) < 10:
            raise ValueError(f"Malformed image pose line: {line}")
        image_id = int(pose[0])
        qw, qx, qy, qz = map(float, pose[1:5])
        tx, ty, tz = map(float, pose[5:8])
        camera_id = int(pose[8])
        name = pose[9]

        i += 1
        while i < len(raw_lines) and raw_lines[i].lstrip().startswith("#"):
            i += 1
        obs_line = raw_lines[i].strip() if i < len(raw_lines) else ""
        obs_tokens = obs_line.split()
        if len(obs_tokens) % 3 != 0:
            raise ValueError(f"Malformed POINTS2D line for image {image_id}")
        observations: list[Observation2D] = []
        for j in range(0, len(obs_tokens), 3):
            point3d_id = int(obs_tokens[j + 2])
            observations.append(
                Observation2D(
                    float(obs_tokens[j]),
                    float(obs_tokens[j + 1]),
                    None if point3d_id < 0 else point3d_id,
                )
            )

        rotation = _quat_to_rot(qw, qx, qy, qz)
        rt_t = _transpose_mv(rotation, (tx, ty, tz))
        center = (-rt_t[0], -rt_t[1], -rt_t[2])
        viewing_direction = _normalize(_transpose_mv(rotation, (0.0, 0.0, 1.0)))
        model.images[image_id] = ImageRecord(
            id=image_id,
            camera_id=camera_id,
            name=name,
            center=center,
            viewing_direction=viewing_direction,
            observations=tuple(observations),
        )
        i += 1

    for line in _non_comment_lines(points_path):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"Malformed points3D line: {line}")
        point_id = int(parts[0])
        track_tokens = parts[8:]
        if len(track_tokens) % 2 != 0:
            raise ValueError(f"Malformed track for point {point_id}")
        track = tuple(
            TrackElement(int(track_tokens[j]), int(track_tokens[j + 1]))
            for j in range(0, len(track_tokens), 2)
        )
        model.points3d[point_id] = Point3D(
            id=point_id,
            xyz=(float(parts[1]), float(parts[2]), float(parts[3])),
            rgb=(int(parts[4]), int(parts[5]), int(parts[6])),
            error=float(parts[7]),
            track=track,
        )
    return model


def _is_valid_point3d_id(point3d_id: int, invalid_point3d: int, has_point3d) -> bool:
    if point3d_id < 0 or point3d_id == invalid_point3d:
        return False
    if callable(has_point3d):
        return bool(has_point3d())
    return True


def _parse_with_pycolmap(path: Path, source: str) -> MapModel:
    try:
        import pycolmap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Binary COLMAP model detected. Install binary support with: pip install 'mapdoctor-sfm[colmap]'"
        ) from exc

    reconstruction = pycolmap.Reconstruction(path)
    model = MapModel(source=source, format="colmap-binary", metadata={"resolved_path": str(path)})

    for camera_id, camera in reconstruction.cameras.items():
        model.cameras[int(camera_id)] = Camera(
            id=int(camera_id),
            model=str(camera.model_name),
            width=int(camera.width),
            height=int(camera.height),
            params=tuple(float(value) for value in camera.params),
        )

    invalid_point3d = getattr(pycolmap, "INVALID_POINT3D_ID", -1)
    for image_id, image in reconstruction.images.items():
        has_pose = getattr(image, "has_pose", True)
        if callable(has_pose):
            has_pose = has_pose()
        if not has_pose:
            continue
        observations = []
        for point2d in image.points2D:
            point3d_id = int(point2d.point3D_id)
            has_point3d = getattr(point2d, "has_point3D", None)
            is_valid = _is_valid_point3d_id(point3d_id, invalid_point3d, has_point3d)
            observations.append(
                Observation2D(float(point2d.xy[0]), float(point2d.xy[1]), point3d_id if is_valid else None)
            )
        center = tuple(float(value) for value in image.projection_center())
        direction = _normalize(tuple(float(value) for value in image.viewing_direction()))
        model.images[int(image_id)] = ImageRecord(
            id=int(image_id),
            camera_id=int(image.camera_id),
            name=str(image.name),
            center=(center[0], center[1], center[2]),
            viewing_direction=direction,
            observations=tuple(observations),
        )

    for point_id, point in reconstruction.points3D.items():
        track = tuple(
            TrackElement(int(element.image_id), int(element.point2D_idx)) for element in point.track.elements
        )
        xyz = tuple(float(value) for value in point.xyz)
        color = tuple(int(value) for value in point.color)
        model.points3d[int(point_id)] = Point3D(
            id=int(point_id),
            xyz=(xyz[0], xyz[1], xyz[2]),
            rgb=(color[0], color[1], color[2]),
            error=float(point.error),
            track=track,
        )
    return model


def load_colmap_format(path: str | Path, source: str) -> MapModel:
    """Decode a COLMAP-format sparse reconstruction for a named producer."""
    root = resolve_sparse_model_path(path)
    if (root / "cameras.txt").exists():
        return _parse_text_model(root, source)
    return _parse_with_pycolmap(root, source)


def load_colmap_compatible(path: str | Path, source: str = "colmap") -> MapModel:
    """Backward-compatible shared decoder used by the explicit adapters."""
    return load_colmap_format(path, source=source)
