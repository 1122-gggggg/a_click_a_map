"""Read-only visual exports for an ``M0_health_report.json``.

The report is already the provenance boundary for these exports.  This module only
reads its ``cameras`` rows and writes three compact, portable representations:
CSV evidence, a status-coloured camera-centre PLY, and a self-contained SVG.
"""

from __future__ import annotations

import csv
import html
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

# Keep this table deliberately explicit: the same RGB values are used by PLY and SVG.
STATUS_RGB: dict[str, tuple[int, int, int]] = {
    "red_acquisition_failure": (220, 38, 38),
    "magenta_pose_multimodal": (219, 39, 119),
    "purple_m0_local_pose": (147, 51, 234),
    "blue_edm_reference_anchoring": (37, 99, 235),
    "orange_pose_inaccurate": (234, 88, 12),
    "yellow_structural_warning": (234, 179, 8),
    "gray_unknown": (107, 114, 128),
    "green_healthy": (22, 163, 74),
}

_BASE_COLUMNS = (
    "image",
    "route",
    "frame",
    "x",
    "y",
    "z",
    "status",
    "primary",
    "causes",
    "segment",
)
_HG_COLUMNS = (
    "num_3d_observations",
    "track_length_p10",
    "track_length_p50",
    "track_length_p90",
    "track_ge3_observation_ratio",
    "covisibility_degree",
    "independent_neighbor_count",
    "independent_neighbor_sequence_count",
    "cross_route_neighbor_count",
    "shared_tracks_top3_total",
    "triangulation_angle_p10_deg",
    "triangulation_angle_p50_deg",
    "image_hull_coverage",
    "grid_4x4_occupancy",
    "point_reprojection_error_p90",
)
_HA_COLUMNS = (
    "matcher_raw_matches",
    "direction01_matches",
    "anchored_matches",
    "correspondences_before_spatial_cap",
    "correspondences_after_spatial_cap",
    "num_inliers",
    "retrieval_top1_score",
    "raw_total",
    "geometric_total",
    "weighted_geometric_ratio",
    "reliable_geometric_degree",
    "reliable_geometric_strength",
)
_HL_COLUMNS = (
    "production_acquisition_success",
    "ground_truth_pose_valid",
    "rotation_error_deg",
    "normalized_position_error",
    "inlier_hull_coverage",
    "inlier_occupancy_4x4",
    "inlier_fim_lambda_min",
    "inlier_fim_condition",
    "reprojection_p90",
    "positive_depth_ratio",
)
_METRIC_COLUMNS = tuple(
    dict.fromkeys(
        [
            *_HG_COLUMNS,
            *_HA_COLUMNS,
            *_HL_COLUMNS,
            *(f"hg_{name}" for name in _HG_COLUMNS),
            *(f"ha_{name}" for name in _HA_COLUMNS),
            *(f"hl_{name}" for name in _HL_COLUMNS),
        ]
    )
)
CSV_COLUMNS = _BASE_COLUMNS + _METRIC_COLUMNS


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _report_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = report.get("cameras")
    if isinstance(values, Mapping):
        values = values.values()
    if not isinstance(values, Sequence) and not hasattr(values, "__iter__"):
        raise ValueError("M0 health report must contain a cameras sequence")
    return [row for row in values if isinstance(row, Mapping)]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _center(row: Mapping[str, Any]) -> tuple[float, float, float] | None:
    value = row.get("camera_center", row.get("center"))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) < 3:
        return None
    values = tuple(_finite(value[index]) for index in range(3))
    return values if all(item is not None for item in values) else None  # type: ignore[return-value]


def _image_name(row: Mapping[str, Any]) -> str:
    return str(row.get("image_name", row.get("image", row.get("name", ""))))


def _route(row: Mapping[str, Any], image: str) -> str:
    value = row.get("route", row.get("sequence"))
    return str(value) if value is not None else image.split("/", 1)[0]


def _frame(row: Mapping[str, Any], image: str) -> int | str | None:
    value = row.get("frame", row.get("frame_number"))
    if value is None:
        token = image.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        digits = ""
        for character in reversed(token):
            if not character.isdigit():
                break
            digits = character + digits
        value = digits or None
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _status(row: Mapping[str, Any]) -> str:
    return str(row.get("heatmap_status", row.get("status", "gray_unknown")))


def status_rgb(status: str) -> tuple[int, int, int]:
    """Return the fixed RGB colour for a report heatmap status."""

    return STATUS_RGB.get(status, STATUS_RGB["gray_unknown"])


def _camera_row(row: Mapping[str, Any]) -> dict[str, Any]:
    image = _image_name(row)
    center = _center(row)
    classification = _mapping(row.get("classification"))
    static = _mapping(row.get("static"))
    edm = _mapping(row.get("edm"))
    matching = _mapping(row.get("matching_database"))
    actual = _mapping(row.get("actual"))
    causes = classification.get("causes", row.get("causes", ()))
    if isinstance(causes, str):
        cause_text = causes
    elif isinstance(causes, Sequence):
        cause_text = ";".join(str(cause) for cause in causes)
    else:
        cause_text = ""
    output: dict[str, Any] = {
        "image": image,
        "route": _route(row, image),
        "frame": _frame(row, image),
        "x": center[0] if center is not None else None,
        "y": center[1] if center is not None else None,
        "z": center[2] if center is not None else None,
        "status": _status(row),
        "primary": classification.get("primary", row.get("primary")),
        "causes": cause_text,
        "segment": classification.get("segment_id", row.get("segment_id")),
    }
    layers = {
        **{name: static.get(name) for name in _HG_COLUMNS},
        **{name: value for name, value in ((key, edm.get(key)) for key in _HA_COLUMNS)},
        **{name: actual.get(name) for name in _HL_COLUMNS},
    }
    # The matching database has its own HA namespace; merge it after EDM fields.
    for name in _HA_COLUMNS:
        if name in matching:
            layers[name] = matching[name]
    # Health reports keep acquisition and retrieval evidence under ``edm`` while
    # pose-quality evidence is under ``actual``; retain both in the HL export layer.
    for name in ("production_acquisition_success", "ground_truth_pose_valid"):
        if name in edm:
            layers[name] = edm[name]
    output.update(layers)
    output.update({f"hg_{name}": output[name] for name in _HG_COLUMNS})
    output.update({f"ha_{name}": output[name] for name in _HA_COLUMNS})
    output.update({f"hl_{name}": output[name] for name in _HL_COLUMNS})
    return output


def camera_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten report camera rows for all three visual export formats."""

    return [_camera_row(row) for row in _report_rows(report)]


def render_camera_csv(report: Mapping[str, Any]) -> str:
    """Render one CSV row per camera, including rows without a valid centre."""

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in camera_rows(report):
        writer.writerow({column: _text(row.get(column)) for column in CSV_COLUMNS})
    return stream.getvalue()


def _number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return format(value, ".9g")


def render_camera_ply(report: Mapping[str, Any]) -> str:
    """Render valid camera centres as an ASCII PLY with fixed status RGB."""

    rows = [row for row in camera_rows(report) if all(row.get(axis) is not None for axis in "xyz")]
    lines = [
        "ply",
        "format ascii 1.0",
        "comment river-map-quality M0 health camera centres",
        f"element vertex {len(rows)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for row in rows:
        red, green, blue = status_rgb(str(row["status"]))
        lines.append(
            f"{_number(float(row['x']))} {_number(float(row['y']))} "
            f"{_number(float(row['z']))} {red} {green} {blue}"
        )
    return "\n".join(lines) + "\n"


def _projection_coordinates(
    rows: Sequence[dict[str, Any]],
    first: str,
    second: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> dict[str, tuple[float, float]]:
    if not rows:
        return {}
    first_values = [float(row[first]) for row in rows]
    second_values = [float(row[second]) for row in rows]
    first_min, first_max = min(first_values), max(first_values)
    second_min, second_max = min(second_values), max(second_values)
    if first_min == first_max:
        first_min -= 0.5
        first_max += 0.5
    if second_min == second_max:
        second_min -= 0.5
        second_max += 0.5
    pad = 24.0
    return {
        row["image"]: (
            x0
            + pad
            + (float(row[first]) - first_min) / (first_max - first_min) * (width - 2 * pad),
            y0
            + height
            - pad
            - (float(row[second]) - second_min) / (second_max - second_min) * (height - 2 * pad),
        )
        for row in rows
    }


def render_camera_svg(report: Mapping[str, Any]) -> str:
    """Render an independent SVG with XY, XZ, and YZ camera projections."""

    rows = camera_rows(report)
    valid = [row for row in rows if all(row.get(axis) is not None for axis in "xyz")]
    panel_width, panel_height, panel_gap = 370.0, 350.0, 20.0
    projections = (("XY", "x", "y"), ("XZ", "x", "z"), ("YZ", "y", "z"))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 500" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">M0 camera health projections</title>',
        '<desc id="desc">Camera centres coloured by M0 heatmap status.</desc>',
        '<rect width="1200" height="500" fill="white"/>',
    ]
    for index, (label, first, second) in enumerate(projections):
        x0 = 10.0 + index * (panel_width + panel_gap)
        y0 = 10.0
        coordinates = _projection_coordinates(
            valid, first, second, x0, y0, panel_width, panel_height
        )
        lines.extend(
            [
                f'<g id="projection-{label.lower()}" data-projection="{label}">',
                f'<rect x="{x0:g}" y="{y0:g}" width="{panel_width:g}" height="{panel_height:g}" '
                'fill="#fafafa" stroke="#d1d5db"/>',
                f'<text x="{x0 + 12:g}" y="{y0 + 22:g}" font-family="sans-serif" '
                f'font-size="16" font-weight="bold">Projection {label}</text>',
                f'<text x="{x0 + panel_width - 18:g}" y="{y0 + panel_height - 8:g}" '
                'font-family="sans-serif" font-size="11">'
                f"{first}</text>",
                f'<text x="{x0 + 8:g}" y="{y0 + 38:g}" font-family="sans-serif" '
                'font-size="11">'
                f"{second}</text>",
            ]
        )
        by_route: dict[str, list[dict[str, Any]]] = {}
        for row in valid:
            by_route.setdefault(str(row["route"]), []).append(row)
        for route, route_rows in sorted(by_route.items()):
            route_rows.sort(key=lambda row: (row["frame"] is None, row["frame"] or 0, row["image"]))
            if len(route_rows) >= 2:
                points = " ".join(
                    f"{coordinates[row['image']][0]:.3f},{coordinates[row['image']][1]:.3f}"
                    for row in route_rows
                )
                lines.append(
                    f'<polyline points="{points}" fill="none" stroke="#d1d5db" '
                    'stroke-width="1.5" data-route="'
                    f'{html.escape(route, quote=True)}"/>'
                )
        for row in valid:
            x, y = coordinates[row["image"]]
            status = str(row["status"])
            red, green, blue = status_rgb(status)
            tooltip = html.escape(
                f"{row['image']} | status={status} | primary={row.get('primary') or ''} "
                f"| causes={row.get('causes') or ''}",
                quote=False,
            )
            lines.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="rgb({red},{green},{blue})" '
                f'data-image="{html.escape(str(row["image"]), quote=True)}" '
                f'data-status="{html.escape(status, quote=True)}">'
                f"<title>{tooltip}</title></circle>"
            )
        lines.append("</g>")
    lines.extend(
        [
            '<g id="legend" transform="translate(20,390)">',
            '<text x="0" y="0" font-family="sans-serif" font-size="14" '
            'font-weight="bold">Legend</text>',
        ]
    )
    for index, (status, rgb) in enumerate(STATUS_RGB.items()):
        x = index * 225.0
        red, green, blue = rgb
        lines.extend(
            [
                f'<rect x="{x:g}" y="12" width="14" height="14" fill="rgb({red},{green},{blue})"/>',
                f'<text x="{x + 20:g}" y="24" font-family="sans-serif" font-size="11">'
                f"{html.escape(status)}</text>",
            ]
        )
    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _load_report(source: Path | str | Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if isinstance(source, Mapping):
        return source, "M0_health_report"
    path = Path(source)
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping):
        raise ValueError("M0 health report JSON must contain an object")
    return report, path.stem


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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


def export_health_report(
    report_source: Path | str | Mapping[str, Any],
    output_dir: Path | str,
    *,
    csv_name: str = "cameras.csv",
    ply_name: str = "camera_centers.ply",
    svg_name: str = "camera_health.svg",
) -> dict[str, Path]:
    """Read one report and atomically write CSV, PLY, and SVG outputs."""

    report, _ = _load_report(report_source)
    # Validate before creating output files, so malformed reports cannot leave partial exports.
    _report_rows(report)
    destination = Path(output_dir)
    paths = {
        "csv": destination / csv_name,
        "ply": destination / ply_name,
        "svg": destination / svg_name,
    }
    _atomic_write(paths["csv"], render_camera_csv(report))
    _atomic_write(paths["ply"], render_camera_ply(report))
    _atomic_write(paths["svg"], render_camera_svg(report))
    return paths


__all__ = [
    "CSV_COLUMNS",
    "STATUS_RGB",
    "camera_rows",
    "export_health_report",
    "render_camera_csv",
    "render_camera_ply",
    "render_camera_svg",
    "status_rgb",
]
