"""Read-only 3D overlays for camera-pose localizability weak regions.

The spheres in this module live in the reconstruction coordinate frame, but
their semantics are attached to camera poses and route segments.  They do not
assert that enclosed sparse landmarks are themselves defective.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from .colmap_binary import iter_binary_image_observations, read_binary_points3d
from .health_export import status_rgb
from .point_color import ImageLoader, colorize_points_from_tracks, load_rgb_image

CSV_MARKER_SEMANTICS = "camera_pose_localizability_weak_region_not_point_defect"
HUMAN_MARKER_SEMANTICS = (
    "camera-pose localizability weak-region envelope; marks weak camera support, "
    "not defective 3D landmarks"
)

_STATUS_PRIORITY = {
    "green_healthy": 0,
    "gray_unknown": 1,
    "yellow_structural_warning": 2,
    "orange_pose_inaccurate": 3,
    "purple_m0_local_pose": 4,
    "blue_edm_reference_anchoring": 4,
    "magenta_pose_multimodal": 5,
    "red_acquisition_failure": 4,
}


@dataclass(frozen=True)
class WeakRegionSphere:
    """One adaptive envelope around a contiguous weak camera-route segment."""

    segment_id: int
    sequence: str
    start_frame: int | str | None
    end_frame: int | str | None
    center: tuple[float, float, float]
    radius: float
    status: str
    camera_count: int
    valid_center_count: int
    missing_center_count: int
    phenotypes: tuple[str, ...]
    causes: tuple[str, ...]
    conditioning: tuple[str, ...]
    decision_axis: str
    acquisition_failure_count: int
    pose_inaccurate_count: int
    image_names: tuple[str, ...]


@dataclass(frozen=True)
class WeakRegionField:
    """Scale estimate and all renderable weak-region spheres."""

    camera_spacing: float
    padding_factor: float
    minimum_radius_factor: float
    spheres: tuple[WeakRegionSphere, ...]


def _finite_center(row: Mapping[str, Any]) -> np.ndarray | None:
    value = row.get("camera_center", row.get("center"))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 3:
        return None
    try:
        center = np.asarray(value[:3], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return center if center.shape == (3,) and np.isfinite(center).all() else None


def _image_name(row: Mapping[str, Any]) -> str:
    return str(row.get("image_name", row.get("image", row.get("name", ""))))


def _sequence(row: Mapping[str, Any]) -> str:
    name = _image_name(row)
    value = row.get("sequence", row.get("route"))
    return str(value) if value is not None else name.split("/", 1)[0]


def _frame(row: Mapping[str, Any]) -> float | None:
    value = row.get("frame", row.get("frame_number"))
    try:
        frame = float(value)
    except (TypeError, ValueError):
        return None
    return frame if math.isfinite(frame) else None


def _mapping_rows(report: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    values = report.get(key)
    if isinstance(values, Mapping):
        values = list(values.values())
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"M0 health report must contain a {key} sequence")
    return [value for value in values if isinstance(value, Mapping)]


def estimate_camera_spacing(cameras: Sequence[Mapping[str, Any]]) -> float:
    """Estimate one robust map-unit-per-frame spacing from finite route poses."""

    by_sequence: dict[str, list[tuple[float, np.ndarray]]] = {}
    for row in cameras:
        center = _finite_center(row)
        frame = _frame(row)
        if center is None or frame is None:
            continue
        by_sequence.setdefault(_sequence(row), []).append((frame, center))
    steps: list[float] = []
    for rows in by_sequence.values():
        rows.sort(key=lambda item: item[0])
        for (first_frame, first), (second_frame, second) in pairwise(rows):
            frame_delta = second_frame - first_frame
            if frame_delta <= 0:
                continue
            distance = float(np.linalg.norm(second - first)) / frame_delta
            if math.isfinite(distance) and distance > np.finfo(np.float64).eps:
                steps.append(distance)
    if not steps:
        raise ValueError("cannot estimate camera spacing from fewer than two distinct route poses")
    return float(np.median(np.asarray(steps, dtype=np.float64)))


def _sequence_of_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _worst_status(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(row.get("heatmap_status", row.get("status", "gray_unknown"))) for row in rows]
    return max(statuses, key=lambda status: _STATUS_PRIORITY.get(status, 1), default="gray_unknown")


def derive_weak_region_spheres(
    report: Mapping[str, Any],
    *,
    padding_factor: float = 1.0,
    minimum_radius_factor: float = 1.0,
) -> WeakRegionField:
    """Turn report segments into adaptive camera-centre envelope spheres.

    A sphere centre is the arithmetic mean of all finite camera centres in the
    segment.  Its radius covers the furthest such centre plus ``padding_factor``
    times the global median camera spacing, with a spacing-scaled lower bound.
    """

    if not math.isfinite(padding_factor) or padding_factor < 0:
        raise ValueError("padding_factor must be a finite non-negative number")
    if not math.isfinite(minimum_radius_factor) or minimum_radius_factor <= 0:
        raise ValueError("minimum_radius_factor must be a finite positive number")
    cameras = _mapping_rows(report, "cameras")
    segments = _mapping_rows(report, "segments")
    spacing = estimate_camera_spacing(cameras)
    cameras_by_name = {_image_name(row): row for row in cameras}
    spheres: list[WeakRegionSphere] = []
    for segment in segments:
        try:
            segment_id = int(segment["segment_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each weak region segment must have an integer segment_id") from exc
        image_names = _sequence_of_strings(segment.get("image_names"))
        camera_count = int(segment.get("camera_count", len(image_names)))
        names_are_unique = len(set(image_names)) == len(image_names)
        names_are_known = all(name in cameras_by_name for name in image_names)
        if camera_count != len(image_names) or not names_are_unique or not names_are_known:
            raise ValueError(
                f"segment {segment_id} has inconsistent camera_count/image_names membership"
            )
        rows = [cameras_by_name[name] for name in image_names]
        centers = [center for row in rows if (center := _finite_center(row)) is not None]
        if not centers:
            raise ValueError(f"segment {segment_id} has no valid camera center")
        coordinates = np.stack(centers)
        center_array = coordinates.mean(axis=0)
        covering_radius = float(np.linalg.norm(coordinates - center_array, axis=1).max())
        radius_override = segment.get("radius_override")
        if radius_override is None:
            radius = max(
                covering_radius + padding_factor * spacing,
                minimum_radius_factor * spacing,
            )
        else:
            try:
                radius = float(radius_override)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"segment {segment_id} has invalid radius_override") from exc
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError(f"segment {segment_id} has invalid radius_override")
        spheres.append(
            WeakRegionSphere(
                segment_id=segment_id,
                sequence=str(segment.get("sequence", _sequence(rows[0]))),
                start_frame=segment.get("start_frame"),
                end_frame=segment.get("end_frame"),
                center=tuple(float(value) for value in center_array),
                radius=radius,
                status=_worst_status(rows),
                camera_count=camera_count,
                valid_center_count=len(centers),
                missing_center_count=max(camera_count - len(centers), 0),
                phenotypes=_sequence_of_strings(segment.get("phenotypes")),
                causes=_sequence_of_strings(segment.get("causes")),
                conditioning=_sequence_of_strings(segment.get("conditioning")),
                decision_axis=str(segment.get("decision_axis", "")),
                acquisition_failure_count=int(segment.get("acquisition_failure_count", 0)),
                pose_inaccurate_count=int(segment.get("pose_inaccurate_count", 0)),
                image_names=image_names,
            )
        )
    return WeakRegionField(
        camera_spacing=spacing,
        padding_factor=padding_factor,
        minimum_radius_factor=minimum_radius_factor,
        spheres=tuple(spheres),
    )


def _number(value: float) -> str:
    return format(value, ".10g")


_SPHERE_CSV_COLUMNS = (
    "segment_id",
    "sequence",
    "start_frame",
    "end_frame",
    "center_x",
    "center_y",
    "center_z",
    "radius",
    "status",
    "phenotypes",
    "causes",
    "conditioning",
    "decision_axis",
    "camera_count",
    "valid_center_count",
    "missing_center_count",
    "acquisition_failure_count",
    "pose_inaccurate_count",
    "image_names",
    "marker_semantics",
)


def render_sphere_csv(field: WeakRegionField) -> str:
    """Render one auditable metadata row for every segment sphere."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_SPHERE_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for sphere in field.spheres:
        writer.writerow(
            {
                "segment_id": sphere.segment_id,
                "sequence": sphere.sequence,
                "start_frame": sphere.start_frame,
                "end_frame": sphere.end_frame,
                "center_x": _number(sphere.center[0]),
                "center_y": _number(sphere.center[1]),
                "center_z": _number(sphere.center[2]),
                "radius": _number(sphere.radius),
                "status": sphere.status,
                "phenotypes": ";".join(sphere.phenotypes),
                "causes": ";".join(sphere.causes),
                "conditioning": ";".join(sphere.conditioning),
                "decision_axis": sphere.decision_axis,
                "camera_count": sphere.camera_count,
                "valid_center_count": sphere.valid_center_count,
                "missing_center_count": sphere.missing_center_count,
                "acquisition_failure_count": sphere.acquisition_failure_count,
                "pose_inaccurate_count": sphere.pose_inaccurate_count,
                "image_names": ";".join(sphere.image_names),
                "marker_semantics": CSV_MARKER_SEMANTICS,
            }
        )
    return stream.getvalue()


def _unit_uv_sphere(
    latitude_segments: int, longitude_segments: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    if latitude_segments < 3 or longitude_segments < 3:
        raise ValueError("sphere mesh requires at least 3 latitude and longitude segments")
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for latitude in range(1, latitude_segments):
        phi = math.pi * latitude / latitude_segments
        radial = math.sin(phi)
        z = math.cos(phi)
        for longitude in range(longitude_segments):
            theta = 2.0 * math.pi * longitude / longitude_segments
            vertices.append((radial * math.cos(theta), radial * math.sin(theta), z))
    vertices.append((0.0, 0.0, -1.0))
    bottom = len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for longitude in range(longitude_segments):
        current = 1 + longitude
        following = 1 + (longitude + 1) % longitude_segments
        faces.append((0, current, following))
    for latitude in range(latitude_segments - 2):
        first_ring = 1 + latitude * longitude_segments
        second_ring = first_ring + longitude_segments
        for longitude in range(longitude_segments):
            following = (longitude + 1) % longitude_segments
            a = first_ring + longitude
            b = first_ring + following
            c = second_ring + longitude
            d = second_ring + following
            faces.extend(((a, c, b), (b, c, d)))
    last_ring = bottom - longitude_segments
    for longitude in range(longitude_segments):
        current = last_ring + longitude
        following = last_ring + (longitude + 1) % longitude_segments
        faces.append((current, bottom, following))
    return np.asarray(vertices, dtype=np.float64), faces


def render_sphere_mesh_ply(
    field: WeakRegionField,
    *,
    latitude_segments: int = 12,
    longitude_segments: int = 24,
) -> str:
    """Render disconnected, coloured triangle spheres as an ASCII PLY mesh."""

    unit_vertices, unit_faces = _unit_uv_sphere(latitude_segments, longitude_segments)
    vertex_count = len(unit_vertices) * len(field.spheres)
    face_count = len(unit_faces) * len(field.spheres)
    lines = [
        "ply",
        "format ascii 1.0",
        f"comment {HUMAN_MARKER_SEMANTICS}",
        "comment vertex alpha is advisory; viewer support varies",
        f"element vertex {vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "property int segment_id",
        f"element face {face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for sphere in field.spheres:
        vertices = np.asarray(sphere.center) + sphere.radius * unit_vertices
        red, green, blue = status_rgb(sphere.status)
        lines.extend(
            f"{_number(float(x))} {_number(float(y))} {_number(float(z))} "
            f"{_number(float(nx))} {_number(float(ny))} {_number(float(nz))} "
            f"{red} {green} {blue} 96 {sphere.segment_id}"
            for (x, y, z), (nx, ny, nz) in zip(vertices, unit_vertices, strict=True)
        )
    for sphere_index in range(len(field.spheres)):
        offset = sphere_index * len(unit_vertices)
        lines.extend(f"3 {a + offset} {b + offset} {c + offset}" for a, b, c in unit_faces)
    return "\n".join(lines) + "\n"


_OVERLAY_DTYPE = np.dtype(
    [
        ("xyz", "<f4", (3,)),
        ("rgb", "u1", (3,)),
    ]
)


def _fibonacci_sphere(count: int) -> np.ndarray:
    if count < 4:
        raise ValueError("samples_per_sphere must be at least 4")
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = indices * (math.pi * (3.0 - math.sqrt(5.0)))
    return np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))


def _point_fields(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = points.dtype.names or ()
    if "xyz" not in names or "rgb" not in names:
        raise ValueError("point array must contain xyz and rgb fields")
    xyz = np.asarray(points["xyz"], dtype=np.float64)
    rgb = np.asarray(points["rgb"], dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("point xyz and rgb fields must both have shape (N, 3)")
    finite = np.isfinite(xyz).all(axis=1)
    return xyz[finite], rgb[finite]


def render_cloud_overlay_ply(
    points: np.ndarray,
    field: WeakRegionField,
    *,
    samples_per_sphere: int = 1024,
) -> bytes:
    """Render colored map points and bright sampled sphere shells in one binary PLY."""

    xyz, rgb = _point_fields(points)
    unit_samples = _fibonacci_sphere(samples_per_sphere)
    vertex_count = len(xyz) + samples_per_sphere * len(field.spheres)
    vertices = np.empty(vertex_count, dtype=_OVERLAY_DTYPE)
    vertices["xyz"][: len(xyz)] = xyz.astype(np.float32)
    vertices["rgb"][: len(xyz)] = rgb
    offset = len(xyz)
    for sphere in field.spheres:
        end = offset + samples_per_sphere
        vertices["xyz"][offset:end] = (
            np.asarray(sphere.center) + sphere.radius * unit_samples
        ).astype(np.float32)
        vertices["rgb"][offset:end] = status_rgb(sphere.status)
        offset = end
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"comment {HUMAN_MARKER_SEMANTICS}",
            "comment map point RGB recovered from mapping-image track observations",
            "comment no scalar fields are stored so CloudCompare displays RGB by default",
            f"element vertex {vertex_count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    return header + vertices.tobytes(order="C")


def select_focus_points(
    points: np.ndarray,
    field: WeakRegionField,
    *,
    context_margin: float,
) -> np.ndarray:
    """Select finite map points inside a sphere envelope plus spatial context."""

    if not math.isfinite(context_margin) or context_margin < 0:
        raise ValueError("context_margin must be a finite non-negative number")
    names = points.dtype.names or ()
    if "xyz" not in names:
        raise ValueError("point array must contain an xyz field")
    xyz = np.asarray(points["xyz"], dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    selected = np.zeros(len(points), dtype=bool)
    for sphere in field.spheres:
        distance = np.linalg.norm(xyz[finite] - np.asarray(sphere.center), axis=1)
        selected[np.flatnonzero(finite)] |= distance <= sphere.radius + context_margin
    return points[selected]


def _sample_preview_points(points: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    if maximum <= 0:
        raise ValueError("max_points must be positive")
    xyz, rgb = _point_fields(points)
    if len(xyz) <= maximum:
        return xyz, rgb
    indices = np.linspace(0, len(xyz) - 1, maximum, dtype=np.int64)
    return xyz[indices], rgb[indices]


def _project_preview_point(
    first_value: float,
    second_value: float,
    *,
    panel_left: float,
    panel_top: float,
    panel_size: float,
    first_mid: float,
    second_mid: float,
    scale: float,
) -> tuple[float, float]:
    return (
        panel_left + panel_size / 2.0 + (first_value - first_mid) * scale,
        panel_top + panel_size / 2.0 - (second_value - second_mid) * scale,
    )


def render_cloud_preview_svg(
    points: np.ndarray,
    field: WeakRegionField,
    *,
    max_points: int = 12_000,
) -> str:
    """Render compact XY/XZ/YZ previews with projected sphere envelopes."""

    xyz, rgb = _sample_preview_points(points, max_points)
    projections = (("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2))
    panel_size = 360.0
    panel_gap = 20.0
    panel_top = 72.0
    plot_padding = 26.0
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 520" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">M0 weak-region spheres on sparse point cloud</title>',
        f'<desc id="desc">{html.escape(HUMAN_MARKER_SEMANTICS)}</desc>',
        '<rect width="1160" height="520" fill="#f8fafc"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="20" '
        'font-weight="bold">M0 weak-region spheres on sparse point cloud</text>',
        '<text x="20" y="50" font-family="sans-serif" font-size="12" fill="#475569">'
        "Sphere = camera-pose weak region; it does not label enclosed landmarks as bad."
        "</text>",
    ]
    sphere_centers = np.asarray([sphere.center for sphere in field.spheres], dtype=np.float64)
    sphere_radii = np.asarray([sphere.radius for sphere in field.spheres], dtype=np.float64)
    for panel_index, (label, first, second) in enumerate(projections):
        panel_left = 20.0 + panel_index * (panel_size + panel_gap)
        first_values = xyz[:, first] if len(xyz) else np.asarray([], dtype=np.float64)
        second_values = xyz[:, second] if len(xyz) else np.asarray([], dtype=np.float64)
        if len(field.spheres):
            first_values = np.concatenate(
                (
                    first_values,
                    sphere_centers[:, first] - sphere_radii,
                    sphere_centers[:, first] + sphere_radii,
                )
            )
            second_values = np.concatenate(
                (
                    second_values,
                    sphere_centers[:, second] - sphere_radii,
                    sphere_centers[:, second] + sphere_radii,
                )
            )
        if not len(first_values):
            first_values = np.asarray([-0.5, 0.5])
            second_values = np.asarray([-0.5, 0.5])
        first_min, first_max = float(first_values.min()), float(first_values.max())
        second_min, second_max = float(second_values.min()), float(second_values.max())
        first_span = max(first_max - first_min, np.finfo(np.float64).eps)
        second_span = max(second_max - second_min, np.finfo(np.float64).eps)
        available = panel_size - 2.0 * plot_padding
        scale = min(available / first_span, available / second_span)
        first_mid = (first_min + first_max) / 2.0
        second_mid = (second_min + second_max) / 2.0

        lines.extend(
            [
                f'<g data-projection="{label}">',
                f'<rect x="{panel_left:g}" y="{panel_top:g}" width="{panel_size:g}" '
                f'height="{panel_size:g}" rx="5" fill="white" stroke="#cbd5e1"/>',
                f'<text x="{panel_left + 12:g}" y="{panel_top + 22:g}" '
                f'font-family="sans-serif" font-size="15" font-weight="bold">{label}</text>',
            ]
        )
        for point, color in zip(xyz, rgb, strict=True):
            x, y = _project_preview_point(
                float(point[first]),
                float(point[second]),
                panel_left=panel_left,
                panel_top=panel_top,
                panel_size=panel_size,
                first_mid=first_mid,
                second_mid=second_mid,
                scale=scale,
            )
            red, green, blue = (int(value) for value in color)
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="0.65" '
                f'fill="rgb({red},{green},{blue})" fill-opacity="0.72"/>'
            )
        for sphere in field.spheres:
            x, y = _project_preview_point(
                sphere.center[first],
                sphere.center[second],
                panel_left=panel_left,
                panel_top=panel_top,
                panel_size=panel_size,
                first_mid=first_mid,
                second_mid=second_mid,
                scale=scale,
            )
            radius = max(sphere.radius * scale, 2.5)
            red, green, blue = status_rgb(sphere.status)
            label_text = (
                f"S{sphere.segment_id} {sphere.sequence} {sphere.start_frame}-{sphere.end_frame}"
            )
            lines.extend(
                [
                    f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" '
                    f'fill="rgb({red},{green},{blue})" fill-opacity="0.22" '
                    f'stroke="rgb({red},{green},{blue})" stroke-width="1.8" '
                    f'data-segment-id="{sphere.segment_id}"><title>'
                    f"{html.escape(label_text)}"
                    "</title></circle>",
                    f'<text x="{x:.3f}" y="{y + 4:.3f}" text-anchor="middle" '
                    'font-family="sans-serif" font-size="9" font-weight="bold" '
                    f'fill="#0f172a">{sphere.segment_id}</text>',
                ]
            )
        lines.append("</g>")
    lines.extend(
        [
            '<g transform="translate(20,466)">',
            '<circle cx="8" cy="0" r="7" fill="rgb(234,179,8)" fill-opacity="0.5"/>',
            '<text x="20" y="4" font-family="sans-serif" font-size="11">'
            "yellow: structural warning</text>",
            '<circle cx="215" cy="0" r="7" fill="rgb(234,88,12)" fill-opacity="0.5"/>',
            '<text x="227" y="4" font-family="sans-serif" font-size="11">'
            "orange: pose inaccurate</text>",
            '<circle cx="415" cy="0" r="7" fill="rgb(220,38,38)" fill-opacity="0.5"/>',
            '<text x="427" y="4" font-family="sans-serif" font-size="11">'
            "red: acquisition failure</text>",
            f'<text x="680" y="4" font-family="sans-serif" font-size="11" fill="#475569">'
            f"Previewed map points: {len(xyz)}</text>",
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = isinstance(content, bytes)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="wb" if binary else "w",
            encoding=None if binary else "utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def export_weak_region_overlay(
    report_path: Path | str,
    points3d_path: Path | str,
    output_dir: Path | str,
    *,
    images_bin_path: Path | str | None = None,
    images_root: Path | str | None = None,
    image_loader: ImageLoader = load_rgb_image,
    padding_factor: float = 1.0,
    minimum_radius_factor: float = 1.0,
    samples_per_sphere: int = 1024,
    focus_margin_factor: float = 20.0,
    latitude_segments: int = 12,
    longitude_segments: int = 24,
    preview_max_points: int = 12_000,
) -> dict[str, Path]:
    """Write complete and focused point-cloud overlays plus sphere evidence."""

    report_source = Path(report_path)
    point_source = Path(points3d_path)
    images_bin_source = (
        Path(images_bin_path)
        if images_bin_path is not None
        else point_source.with_name("images.bin")
    )
    images_directory = (
        Path(images_root) if images_root is not None else point_source.parent.parent / "images"
    )
    if not math.isfinite(focus_margin_factor) or focus_margin_factor < 0:
        raise ValueError("focus_margin_factor must be a finite non-negative number")
    source_hashes = {
        "report_sha256": _sha256(report_source),
        "points3d_sha256": _sha256(point_source),
        "images_bin_sha256": _sha256(images_bin_source),
    }
    report = json.loads(report_source.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("M0 health report JSON must contain an object")
    points = read_binary_points3d(point_source)
    colorization = colorize_points_from_tracks(
        points,
        iter_binary_image_observations(images_bin_source),
        images_directory,
        image_loader=image_loader,
    )
    colored_points = colorization.points
    field = derive_weak_region_spheres(
        report,
        padding_factor=padding_factor,
        minimum_radius_factor=minimum_radius_factor,
    )
    context_margin = field.camera_spacing * focus_margin_factor
    focused_points = select_focus_points(colored_points, field, context_margin=context_margin)
    full_overlay = render_cloud_overlay_ply(
        colored_points, field, samples_per_sphere=samples_per_sphere
    )
    focus_overlay = render_cloud_overlay_ply(
        focused_points, field, samples_per_sphere=samples_per_sphere
    )
    contents: dict[str, bytes | str] = {
        "full_overlay": full_overlay,
        "focus_overlay": focus_overlay,
        "legacy_full_overlay": full_overlay,
        "legacy_focus_overlay": focus_overlay,
        "sphere_mesh": render_sphere_mesh_ply(
            field,
            latitude_segments=latitude_segments,
            longitude_segments=longitude_segments,
        ),
        "sphere_csv": render_sphere_csv(field),
        "preview_svg": render_cloud_preview_svg(
            focused_points, field, max_points=preview_max_points
        ),
    }
    source_hashes_after = {
        "report_sha256": _sha256(report_source),
        "points3d_sha256": _sha256(point_source),
        "images_bin_sha256": _sha256(images_bin_source),
    }
    if source_hashes_after != source_hashes:
        raise RuntimeError(
            "report, points3D, or images.bin source changed while rendering; no outputs written"
        )
    destination = Path(output_dir)
    paths = {
        "full_overlay": destination / "M0_track_RGB_cloud_with_weak_regions.ply",
        "focus_overlay": destination / "M0_track_RGB_weak_regions_focus.ply",
        "legacy_full_overlay": destination / "M0_cloud_with_weak_regions.ply",
        "legacy_focus_overlay": destination / "M0_weak_regions_focus.ply",
        "sphere_mesh": destination / "weak_region_spheres.ply",
        "sphere_csv": destination / "weak_region_spheres.csv",
        "preview_svg": destination / "weak_region_cloud_preview.svg",
        "manifest": destination / "weak_region_overlay_manifest.json",
    }
    for kind, content in contents.items():
        _atomic_write(paths[kind], content)
    output_hashes = {
        kind: hashlib.sha256(
            content if isinstance(content, bytes) else content.encode()
        ).hexdigest()
        for kind, content in contents.items()
    }
    manifest = {
        "schema_version": 1,
        "marker_semantics": HUMAN_MARKER_SEMANTICS,
        "sphere_count": len(field.spheres),
        "source_point_count": len(points),
        "focus_point_count": len(focused_points),
        "preview_point_count": min(len(focused_points), preview_max_points),
        "camera_spacing": field.camera_spacing,
        "radius_rule": "max_distance_from_segment_mean + padding_factor * camera_spacing",
        "padding_factor": padding_factor,
        "minimum_radius_factor": minimum_radius_factor,
        "samples_per_sphere": samples_per_sphere,
        "focus_margin_factor": focus_margin_factor,
        "focus_context_margin": context_margin,
        "background_rgb": "track_mean_nearest_pixel from mapping images",
        "colorization": {
            "method": colorization.method,
            "colored_point_count": colorization.colored_point_count,
            "sample_count": colorization.sample_count,
            "image_count": colorization.image_count,
            "input_points3d_unique_rgb_count": len(np.unique(points["rgb"], axis=0)),
            "output_unique_rgb_count": len(np.unique(colored_points["rgb"], axis=0)),
            "pure_black_point_count": int(np.all(colored_points["rgb"] == 0, axis=1).sum()),
        },
        "sources": {
            "report": str(report_source.resolve()),
            "report_sha256": source_hashes["report_sha256"],
            "points3d": str(point_source.resolve()),
            "points3d_sha256": source_hashes["points3d_sha256"],
            "images_bin": str(images_bin_source.resolve()),
            "images_bin_sha256": source_hashes["images_bin_sha256"],
            "images_root": str(images_directory.resolve()),
        },
        "outputs": {
            kind: {"path": path.name, "sha256": output_hashes[kind]}
            for kind, path in paths.items()
            if kind != "manifest"
        },
    }
    _atomic_write(paths["manifest"], json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return paths


__all__ = [
    "CSV_MARKER_SEMANTICS",
    "HUMAN_MARKER_SEMANTICS",
    "WeakRegionField",
    "WeakRegionSphere",
    "derive_weak_region_spheres",
    "estimate_camera_spacing",
    "export_weak_region_overlay",
    "render_cloud_overlay_ply",
    "render_cloud_preview_svg",
    "render_sphere_csv",
    "render_sphere_mesh_ply",
    "select_focus_points",
]
