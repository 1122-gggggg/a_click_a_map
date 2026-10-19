"""Colored risk-sphere PLY export from MapData plus canonical diagnosis JSON.

This module does not compute a second Fisher information matrix. Heatmap and
weak-region artifacts are consumed as already-diagnosed evidence.

Held-out success is the conjunction of outer ``DIRECT_STRONG`` and nested
``decision.status == ACCEPT``. Nested ``REJECT*`` always fails. Outer
``GEOMETRY_WEAK`` / ``PROVISIONAL`` plus nested ``ACCEPT`` remains a marker.
If either richer status is present, boolean ``success`` cannot override;
missing one side is not strict. Boolean ``success`` is used only when both
richer statuses are absent.

CloudCompare export writes a robust-clipped binary little-endian base cloud,
per-class / all-marker overlays, and an aggregated combined preview. Dense
held-out query failures are clustered only in the combined file. The
full-extent archival PLY keeps original coordinates, original map colors
(including black), and one shell per raw marker.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json
from .models import MapData


ISSUE_COLORS: dict[str, tuple[int, int, int]] = {
    "coverage_hole": (255, 0, 0),
    "fim_rank_deficient": (255, 0, 255),
    "fim_weak": (255, 140, 0),
    "direction_sensitive": (255, 215, 0),
    "unverified_bridge_pose": (148, 0, 211),
    "zero_triangulation": (0, 110, 70),
    "weak_region": (255, 64, 64),
    "heldout_geometry_weak": (180, 0, 0),
    "heldout_provisional": (0, 180, 200),
    "failure_retrieval_proxy": (30, 90, 255),
    "actloc_shadow": (160, 160, 160),
}

ISSUE_LEGEND: dict[str, str] = {
    "coverage_hole": "Heatmap pose has sparse visible support or occupancy below the diagnostic floor.",
    "fim_rank_deficient": "Consumed heatmap FIM rank is below 6; at least one pose direction is unobservable.",
    "fim_weak": "Consumed heatmap marks weak FIM isotropy/condition. This is observability, not success probability.",
    "direction_sensitive": "Consumed heatmap health changes sharply across orientations at one position.",
    "unverified_bridge_pose": "Registered camera has zero 3D observations; pose is not localization evidence.",
    "zero_triangulation": (
        "Registered triangulation camera has zero 3D observations; the pose is unsupported by map tracks."
    ),
    "weak_region": "Weak-region centroid from sfm-diagnosis analyze. Not a calibrated failure location.",
    "heldout_geometry_weak": (
        "Optional localization log has a pose but failed the strict conjunction "
        "(outer DIRECT_STRONG and nested decision.status ACCEPT). Nested REJECT* "
        "always fails. Outer GEOMETRY_WEAK/PROVISIONAL plus nested ACCEPT stays a marker."
    ),
    "heldout_provisional": (
        "Optional localization log is outer PROVISIONAL (even with nested ACCEPT); "
        "not a deployable success."
    ),
    "failure_retrieval_proxy": "Failed query has no pose; marker is a retrieval reference, not ground truth.",
    "actloc_shadow": "StructuralLocalizabilityProxy shadow marker only. Not an authorized ActLoc network.",
}

CAVEATS = (
    "FIM observability is not a calibrated localization success probability.",
    "No-pose failure retrieval proxies are not ground-truth query poses.",
    "StructuralLocalizabilityProxy and ExternalPredictorAdapter are ActLoc-style "
    "shadow diagnostics only; they are not an authorized ActLoc network and are "
    "not held-out calibrated.",
    "Held-out success is outer DIRECT_STRONG AND nested decision.status ACCEPT; "
    "nested REJECT always wins. Outer GEOMETRY_WEAK/PROVISIONAL plus nested "
    "ACCEPT remains a marker. If either richer status exists, boolean success "
    "cannot override; it is used only when both statuses are absent.",
)

CLOUDCOMPARE_FORMAT = "binary_little_endian 1.0"
CLIP_LO_QUANTILE = 0.005
CLIP_HI_QUANTILE = 0.995
CLIP_IQR_K = 8.0
MARKER_RADIUS_DIAG_FRACTION = 0.015
MARKER_RADIUS_NN_FRACTION = 2.0
MARKER_RADIUS_ABS_FLOOR = 0.05
MARKER_RADIUS_CAMERA_DIAG_CAP = 0.01
AGGREGATION_NN_FRACTION = 8.0
AGGREGATION_CAMERA_DIAG_FRACTION = 0.025
AGGREGATION_CAMERA_DIAG_CAP = 0.06
DEFAULT_SPHERE_SAMPLES = 96
VISIBLE_MAP_RGB = (210, 210, 210)
SOURCE_RGB_MISSING_MAX = 8
DENSE_COMBINED_CLASSES = frozenset(
    {
        "heldout_geometry_weak",
        "heldout_provisional",
        "failure_retrieval_proxy",
    }
)
CLASS_RADIUS_SCALE: dict[str, float] = {
    "coverage_hole": 0.55,
    "fim_rank_deficient": 0.65,
    "fim_weak": 0.65,
    "direction_sensitive": 0.65,
    "unverified_bridge_pose": 0.70,
    "zero_triangulation": 0.80,
    "weak_region": 0.55,
    "heldout_geometry_weak": 0.35,
    "heldout_provisional": 0.45,
    "failure_retrieval_proxy": 0.45,
    "actloc_shadow": 0.50,
}
VIEWER_INSTRUCTIONS = (
    "Load CloudCompare layers as separate entities; do not rely on the combined preview alone.",
    "1. File > Open the robust-clipped base cloud (*_cloudcompare_base.ply). Auto-fit this entity only. It is neutral gray.",
    "2. File > Open each needed class file (*_cloudcompare_<class>.ply) or the all-markers overlay (*_cloudcompare_markers.ply) as additional entities. Do not merge them into the base.",
    "3. Optional raw centers: *_cloudcompare_markers_raw.ply. Combined *_cloudcompare.ply is an aggregated preview of dense held-out classes plus the base.",
    "4. Colors: RGB. Turn every scalar field off. Accept Global Shift / scale if prompted.",
    "5. Point size: base 2–3 px; marker layers 6–8 px. Background dark navy or mid gray, not pure black.",
    "6. Do not auto-fit *_full.ply; outliers are retained there on purpose.",
    "Dense held-out failures are clustered only in the combined preview. Raw marker counts stay in the receipt, class files, and markers_raw.",
)

_JSONL_SUFFIXES = {".jsonl", ".ndjson"}
_HEATMAP_DIR_NAMES = (
    "pose_health.csv",
    "position_health.csv",
    "weak_regions.json",
    "summary.json",
)
_JSON_ROW_KEYS = (
    "images",
    "regions",
    "rows",
    "queries",
    "virtual_camera_rows",
    "frames",
)
_BRIDGE_ROLE_TOKENS = {
    "bridge",
    "bridge_only",
    "pure_rotation",
    "unverified_bridge_pose",
}
_TRIANGULATION_ROLE_TOKENS = {
    "core",
    "parallax",
    "triangulate",
    "triangulation",
    "zero_triangulation",
}
_ZERO_OBS_ISSUES = {"unverified_bridge_pose", "zero_triangulation"}


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _rgb(name: str) -> tuple[int, int, int]:
    return ISSUE_COLORS[name]


def sphere_points(center: Sequence[float], radius: float, count: int = 48) -> np.ndarray:
    """Deterministic spherical shell. Count is the number of sample points."""

    origin = np.asarray(center, dtype=float).reshape(3)
    n = max(8, int(count))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = np.zeros((n, 3), dtype=float)
    for index in range(n):
        y = 1.0 - (index / max(n - 1, 1)) * 2.0
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * index
        out[index] = (
            origin
            + radius
            * np.array([math.cos(theta) * radial, y, math.sin(theta) * radial], dtype=float)
        )
    return out


def _mapping_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    if isinstance(payload, Mapping):
        for key in _JSON_ROW_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(row) for row in value if isinstance(row, Mapping)]
        return [dict(payload)]
    return []


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Parse one JSON object per non-blank line. Non-objects fail closed."""

    target = Path(path)
    rows: list[dict[str, Any]] = []
    with target.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{target}:{lineno} is not valid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{target}:{lineno} is not a JSON object "
                    f"(got {type(payload).__name__})"
                )
            rows.append(dict(payload))
    return rows


def load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    """Load diagnosis or localization rows from JSON/JSONL/CSV or a directory."""

    if path is None:
        return []
    target = Path(path)
    if target.is_dir():
        for name in _HEATMAP_DIR_NAMES:
            candidate = target / name
            if candidate.is_file():
                target = candidate
                break
        else:
            jsonl_files = sorted(
                [
                    *target.glob("*.jsonl"),
                    *target.glob("*.ndjson"),
                ]
            )
            rows: list[dict[str, Any]] = []
            for candidate in jsonl_files:
                rows.extend(load_jsonl_rows(candidate))
            return rows
    if not target.is_file():
        return []
    suffix = target.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        return load_jsonl_rows(target)
    if suffix == ".json":
        return _mapping_rows(json.loads(target.read_text(encoding="utf-8")))
    with target.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_rows(path: str | Path | None) -> list[dict[str, Any]]:
    return load_rows(path)


def _is_row_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _coerce_rows(
    value: str | Path | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if _is_row_sequence(value):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return load_rows(value)


def _image_name(row: Mapping[str, Any]) -> str | None:
    for key in ("image_name", "output_name", "name"):
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return None


def _image_keys(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    name = _image_name(row)
    if name is not None:
        keys.add(("name", name))
        keys.add(("name", Path(name).name))
    image_id = _as_int(row.get("image_id"))
    if image_id is not None:
        keys.add(("id", str(image_id)))
    return keys


def _normalize_role(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _zero_obs_issue(role: Any) -> str:
    token = _normalize_role(role)
    if token in _TRIANGULATION_ROLE_TOKENS:
        return "zero_triangulation"
    if token in _BRIDGE_ROLE_TOKENS or token == "":
        return "unverified_bridge_pose"
    return "zero_triangulation"


def normalize_image_roles(
    value: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Accept a name→role map, row sequence, or JSON/JSONL/CSV path."""

    if value is None:
        return {}
    if isinstance(value, (str, bytes, Path)):
        value = load_rows(value)
    if isinstance(value, Mapping):
        if isinstance(value.get("frames"), list):
            value = value["frames"]
        elif any(key in value for key in ("role", "image_role", "motion_role")):
            value = [value]
        elif all(
            not isinstance(role, (Mapping, list, tuple)) for role in value.values()
        ):
            return {
                str(key): str(role)
                for key, role in value.items()
                if role is not None
            }
        else:
            return {}
    if not _is_row_sequence(value):
        return {}
    roles: dict[str, str] = {}
    for row in value:
        if not isinstance(row, Mapping):
            continue
        name = _image_name(row)
        role = row.get("role", row.get("image_role", row.get("motion_role")))
        if name is None or role is None:
            continue
        roles[name] = str(role)
        base = Path(name).name
        roles.setdefault(base, str(role))
    return roles


def _role_for_image(
    name: str,
    image_id: int,
    roles: Mapping[str, str],
) -> str | None:
    if name in roles:
        return roles[name]
    base = Path(name).name
    if base in roles:
        return roles[base]
    if str(image_id) in roles:
        return roles[str(image_id)]
    return None


def _observation_counts(map_data: MapData) -> np.ndarray:
    counts = np.zeros(map_data.num_images, dtype=int)
    lookup = map_data.image_index()
    for track in map_data.track_image_ids:
        for image_id in np.asarray(track).reshape(-1):
            index = lookup.get(int(image_id))
            if index is not None:
                counts[index] += 1
    return counts


def markers_from_map(
    map_data: MapData,
    image_roles: Mapping[str, str] | None = None,
    skip_image_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    counts = _observation_counts(map_data)
    roles = dict(image_roles or {})
    skip = skip_image_keys or set()
    markers: list[dict[str, Any]] = []
    for index, count in enumerate(counts.tolist()):
        if count > 0:
            continue
        name = str(map_data.image_names[index])
        image_id = int(map_data.image_ids[index])
        keys = {("name", name), ("name", Path(name).name), ("id", str(image_id))}
        if keys & skip:
            continue
        role = _role_for_image(name, image_id, roles)
        issue = _zero_obs_issue(role)
        center = map_data.image_centers[index]
        markers.append(
            {
                "issue_class": issue,
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "source": "map_zero_observations",
                "image_name": name,
                "image_id": image_id,
                "image_role": role,
            }
        )
    return markers


def markers_from_heatmap(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_position: dict[tuple[float, float, float], list[Mapping[str, Any]]] = {}
    markers: list[dict[str, Any]] = []
    for row in rows:
        x, y, z = _as_float(row.get("x")), _as_float(row.get("y")), _as_float(row.get("z"))
        if x is None or y is None or z is None:
            continue
        by_position.setdefault((round(x, 5), round(y, 5), round(z, 5)), []).append(row)
        codes = str(row.get("codes") or row.get("best_codes") or "")
        primary = str(row.get("primary") or row.get("best_primary") or "")
        visible = _as_int(row.get("visible_points"))
        occupancy = _as_float(row.get("grid_occupancy"))
        rank_proxy = _as_float(row.get("fim_lambda_min"))
        condition = _as_float(row.get("fim_condition"))
        issues: list[str] = []
        if "DATA_SPARSE" in codes or primary == "DATA_SPARSE" or (
            visible is not None and visible < 40
        ) or (occupancy is not None and occupancy < 6):
            issues.append("coverage_hole")
        if rank_proxy is not None and rank_proxy <= 1e-12:
            issues.append("fim_rank_deficient")
        if "GEOMETRY_WEAK" in codes or primary == "GEOMETRY_WEAK" or (
            condition is not None and condition > 1e6
        ):
            issues.append("fim_weak")
        for issue in issues:
            markers.append(
                {
                    "issue_class": issue,
                    "x": x,
                    "y": y,
                    "z": z,
                    "source": "heatmap",
                    "primary": primary,
                    "codes": codes,
                }
            )
    for (x, y, z), group in by_position.items():
        scores = [_as_float(row.get("health_score") or row.get("best_health")) for row in group]
        finite = [value for value in scores if value is not None]
        if len(finite) < 2:
            continue
        if max(finite) - min(finite) >= 0.25 and max(finite) >= 0.45:
            markers.append(
                {
                    "issue_class": "direction_sensitive",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "source": "heatmap_orientation_spread",
                }
            )
    return markers


def markers_from_weak_regions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else row
        x = _as_float(metrics.get("center_x") or metrics.get("x") or row.get("x"))
        y = _as_float(metrics.get("center_y") or metrics.get("y") or row.get("y"))
        z = _as_float(metrics.get("center_z") or metrics.get("z") or row.get("z"))
        if x is None or y is None or z is None:
            continue
        if row.get("weak") is False:
            continue
        markers.append(
            {
                "issue_class": "weak_region",
                "x": x,
                "y": y,
                "z": z,
                "source": "weak_regions",
                "region_id": row.get("region_id") or row.get("id"),
            }
        )
    return markers


def _pose_center(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x = _as_float(value.get("x"))
        y = _as_float(value.get("y"))
        z = _as_float(value.get("z"))
        if x is not None and y is not None and z is not None:
            return x, y, z
        value = value.get("pose") or value.get("T_cw") or value.get("matrix")
        if value is None:
            return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.shape == (4, 4):
        center = -matrix[:3, :3].T @ matrix[:3, 3]
        return float(center[0]), float(center[1]), float(center[2])
    if matrix.shape == (3,):
        return float(matrix[0]), float(matrix[1]), float(matrix[2])
    return None


def _nested_decision(row: Mapping[str, Any]) -> tuple[str, bool, bool]:
    decision = row.get("decision")
    if not isinstance(decision, Mapping):
        return "", False, False
    nested = str(decision.get("status") or "").strip().upper()
    accept_flag = decision.get("accept")
    if accept_flag is None:
        accept_flag = decision.get("accepted")
    accepted = nested == "ACCEPT" or accept_flag in {1, True, "1", "true", "True"}
    present = bool(nested) or accept_flag is not None
    return nested, accepted, present


def markers_from_localization(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Emit held-out failure and provisional markers.

    Precedence:
    1. Compute ``outer_strong`` from the outer ``status`` first. Only the
       exact token ``DIRECT_STRONG`` counts; ``STRONG`` and substring
       matches do not.
    2. When either richer status is present, success is
       ``outer_token == DIRECT_STRONG and nested_status == ACCEPT``.
       Missing nested status is failure. Any nested ``REJECT*`` is
       failure. Outer ``GEOMETRY_WEAK`` or ``PROVISIONAL`` plus nested
       ``ACCEPT`` stays a weak/provisional marker and is never hidden.
    3. Explicit boolean ``success`` is used only when the row lacks both
       richer outer and nested statuses. If either richer field exists,
       boolean cannot override the conjunction.
    """
    markers: list[dict[str, Any]] = []
    for row in rows:
        nested_status, _nested_accepted, has_nested = _nested_decision(row)
        outer_status = str(row.get("status") or "").strip()
        outer_token = outer_status.upper()
        outer_strong = outer_token == "DIRECT_STRONG"
        if has_nested or outer_status:
            strong = outer_strong and nested_status == "ACCEPT"
        else:
            success = row.get("success")
            strong = success in {1, True, "1", "true", "True"}
        provisional = bool(
            row.get("provisional")
            or "PROVISIONAL" in outer_token
            or outer_token in {"HELD_OUT_PROVISIONAL"}
        )
        if strong and not provisional:
            continue
        center = _pose_center(row) or _pose_center(row.get("pose"))
        if center is None:
            ref = row.get("retrieval_reference") or row.get("top1_reference")
            if isinstance(ref, Mapping):
                center = _pose_center(ref)
            if center is None:
                continue
            markers.append(
                {
                    "issue_class": "failure_retrieval_proxy",
                    "x": center[0],
                    "y": center[1],
                    "z": center[2],
                    "source": "localization_retrieval_proxy",
                    "query": row.get("query") or row.get("query_name"),
                    "status": outer_status or nested_status,
                    "nested_decision": nested_status or None,
                }
            )
            continue
        issue = "heldout_provisional" if provisional else "heldout_geometry_weak"
        markers.append(
            {
                "issue_class": issue,
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "source": "localization_log",
                "query": row.get("query") or row.get("query_name"),
                "status": outer_status or nested_status,
                "nested_decision": nested_status or None,
            }
        )
    return markers


def camera_nearest_spacing(centers: np.ndarray) -> float:
    """Median true nearest-neighbor spacing; KD-tree, order-independent at any n."""

    points = np.asarray(centers, dtype=float).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < 2:
        return 0.0
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(finite).query(finite, k=2)
    nearest = np.asarray(distances[:, 1], dtype=float)
    positive = nearest[np.isfinite(nearest) & (nearest > 0)]
    return float(np.median(positive)) if len(positive) else 0.0


def _robust_radius(
    xyz: np.ndarray,
    *,
    hi_quantile: float,
    iqr_k: float,
    guard_extrema: bool = False,
) -> tuple[np.ndarray, float, str, np.ndarray]:
    """Distance from coordinate-wise median; MAD-guarded when tails are extrema."""

    cloud = np.asarray(xyz, dtype=float).reshape(-1, 3)
    center = np.median(cloud, axis=0)
    dist = np.linalg.norm(cloud - center, axis=1)
    dmed = float(np.median(dist))
    dmad = float(np.median(np.abs(dist - dmed)))
    mad_radius = dmed + iqr_k * max(dmad, 1e-12)
    if len(cloud) >= 64:
        q_radius = float(np.quantile(dist, hi_quantile))
        if guard_extrema and q_radius > 4.0 * max(mad_radius, 1e-12):
            return center, mad_radius, "distance_mad_guarded", dist
        return center, q_radius, "distance_quantile", dist
    return center, mad_radius, "distance_mad", dist


def visible_map_rgb(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Replace black/near-black base colors with a CloudCompare-visible gray."""

    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(colors) == 0:
        return colors.copy(), {
            "applied": False,
            "override_count": 0,
            "source_max": 0,
            "fallback_rgb": list(VISIBLE_MAP_RGB),
        }
    source_max = int(colors.max())
    missing = colors.max(axis=1) <= SOURCE_RGB_MISSING_MAX
    override_count = int(missing.sum())
    out = colors.copy()
    applied = override_count > 0
    if applied:
        out[missing] = np.asarray(VISIBLE_MAP_RGB, dtype=np.uint8)
    return out, {
        "applied": applied,
        "override_count": override_count,
        "source_max": source_max,
        "fallback_rgb": list(VISIBLE_MAP_RGB),
        "threshold": SOURCE_RGB_MISSING_MAX,
    }


def _xyz_in_clip(xyz: Sequence[float], clip_min: np.ndarray, clip_max: np.ndarray) -> bool:
    point = np.asarray(xyz, dtype=float).reshape(3)
    if not np.isfinite(point).all():
        return False
    return bool(np.all(point >= clip_min) and np.all(point <= clip_max))


def class_marker_radius(issue: str, base_radius: float) -> float:
    """Smaller class-specific display radius; colors stay on ISSUE_COLORS."""

    scale = CLASS_RADIUS_SCALE.get(str(issue), 0.6)
    return max(float(base_radius) * float(scale), 1e-4)


def display_radius_from_clip(
    clip: Mapping[str, Any],
    *,
    override: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Camera-NN / local-spacing radius with a camera-path-diagonal cap.

    Robust point-cloud diagonal is a last-resort fallback only when the
    camera path is empty. It is never the sole driver when cameras exist.
    """

    camera_nn = float(clip.get("camera_nn") or 0.0)
    camera_diag = float(clip.get("camera_diagonal") or 0.0)
    robust_diag = float(clip.get("robust_diagonal") or 0.0)
    nn_term = MARKER_RADIUS_NN_FRACTION * camera_nn
    cap = MARKER_RADIUS_CAMERA_DIAG_CAP * camera_diag if camera_diag > 0.0 else None
    meta: dict[str, Any] = {
        "camera_nn": camera_nn,
        "camera_diagonal": camera_diag,
        "robust_diagonal": robust_diag,
        "nn_term": nn_term,
        "cap": cap,
        "override": override is not None,
    }
    if override is not None:
        radius = max(float(override), 1e-4)
        meta["source"] = "override"
        meta["radius"] = radius
        return radius, meta
    if cap is not None:
        uncapped = nn_term if nn_term > 0.0 else cap
        radius = min(max(uncapped, 1e-4), cap)
        if nn_term <= 0.0:
            meta["source"] = "camera_diagonal_cap"
        elif nn_term > cap:
            meta["source"] = "camera_diagonal_cap"
        else:
            meta["source"] = "camera_nn"
    else:
        radius = max(
            MARKER_RADIUS_ABS_FLOOR,
            MARKER_RADIUS_DIAG_FRACTION * robust_diag,
            nn_term,
            1e-4,
        )
        meta["source"] = "robust_fallback"
    meta["radius"] = float(radius)
    return float(radius), meta


def aggregation_radius_from_clip(clip: Mapping[str, Any]) -> float:
    """Deterministic neighborhood used to collapse dense held-out markers."""

    camera_nn = float(clip.get("camera_nn") or 0.0)
    camera_diag = float(clip.get("camera_diagonal") or 0.0)
    local = max(
        AGGREGATION_NN_FRACTION * camera_nn,
        AGGREGATION_CAMERA_DIAG_FRACTION * camera_diag,
    )
    if camera_diag > 0.0:
        cap = AGGREGATION_CAMERA_DIAG_CAP * camera_diag
        return float(min(max(local, 1e-4), cap))
    return float(max(local, MARKER_RADIUS_ABS_FLOOR, 1e-4))


def cluster_marker_indices(
    markers: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    radius: float,
) -> list[dict[str, Any]]:
    """Union-find clusters. Sorted seeds make the partition deterministic."""

    idxs = sorted(
        int(i) for i in indices if 0 <= int(i) < len(markers)
    )
    if not idxs:
        return []
    radius = max(float(radius), 0.0)
    points = {
        i: np.asarray(
            (markers[i]["x"], markers[i]["y"], markers[i]["z"]),
            dtype=float,
        )
        for i in idxs
    }
    parent = {i: i for i in idxs}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        root_l, root_r = find(left), find(right)
        if root_l == root_r:
            return
        if root_l < root_r:
            parent[root_r] = root_l
        else:
            parent[root_l] = root_r

    limit2 = radius * radius
    for offset, left in enumerate(idxs):
        origin = points[left]
        for right in idxs[offset + 1 :]:
            delta = origin - points[right]
            if float(delta @ delta) <= limit2:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in idxs:
        groups.setdefault(find(index), []).append(index)
    clusters: list[dict[str, Any]] = []
    for members in groups.values():
        stacked = np.vstack([points[i] for i in members])
        center = stacked.mean(axis=0)
        issue = str(markers[members[0]]["issue_class"])
        clusters.append(
            {
                "issue_class": issue,
                "center": [float(center[0]), float(center[1]), float(center[2])],
                "members": list(members),
                "member_count": int(len(members)),
                "aggregation_radius": float(radius),
            }
        )
    clusters.sort(
        key=lambda row: (
            row["center"][0],
            row["center"][1],
            row["center"][2],
            row["members"][0],
        )
    )
    return clusters



def robust_spatial_clip(
    xyz: np.ndarray,
    *,
    camera_xyz: np.ndarray | None = None,
    lo_quantile: float = CLIP_LO_QUANTILE,
    hi_quantile: float = CLIP_HI_QUANTILE,
    iqr_k: float = CLIP_IQR_K,
) -> dict[str, Any]:
    """Finite-point clip box from robust map/camera percentiles, never camera max extent."""

    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    cams = np.empty((0, 3), dtype=float)
    if camera_xyz is not None and len(np.asarray(camera_xyz)):
        raw_cams = np.asarray(camera_xyz, dtype=float).reshape(-1, 3)
        cams = raw_cams[np.isfinite(raw_cams).all(axis=1)]
    camera_nn = camera_nearest_spacing(cams) if len(cams) >= 2 else 0.0
    camera_diagonal_full = (
        float(np.linalg.norm(cams.max(axis=0) - cams.min(axis=0))) if len(cams) else 0.0
    )
    camera_radius = 0.0
    camera_method = "empty"
    if len(cams):
        _cam_center, camera_radius, camera_method, _cam_dist = _robust_radius(
            cams,
            hi_quantile=hi_quantile,
            iqr_k=iqr_k,
            guard_extrema=True,
        )
    # Robust cube diagonal from camera percentile/MAD radius; full AABB is archival only.
    camera_diagonal = float(2.0 * camera_radius * math.sqrt(3.0)) if camera_radius else 0.0
    work = points[finite]
    scale_source = "points"
    if len(work) == 0:
        work = cams
        scale_source = "cameras"
    if len(work) == 0:
        clip_min = np.array([-1.0, -1.0, -1.0], dtype=float)
        clip_max = np.array([1.0, 1.0, 1.0], dtype=float)
        keep = np.zeros(len(points), dtype=bool)
        return {
            "keep": keep,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "center": np.zeros(3, dtype=float),
            "limit": 1.0,
            "pad": 0.0,
            "quantile_min": clip_min.copy(),
            "quantile_max": clip_max.copy(),
            "lo_quantile": lo_quantile,
            "hi_quantile": hi_quantile,
            "iqr_k": iqr_k,
            "method": "empty",
            "scale_source": scale_source,
            "camera_nn": camera_nn,
            "camera_diagonal": camera_diagonal,
            "camera_diagonal_full": camera_diagonal_full,
            "camera_radius": camera_radius,
            "camera_method": camera_method,
            "robust_diagonal": float(np.linalg.norm(clip_max - clip_min)),
            "full_min": clip_min.copy(),
            "full_max": clip_max.copy(),
            "full_diagonal": 0.0,
            "finite_count": 0,
            "nonfinite_count": int((~finite).sum()),
            "retained_count": 0,
            "excluded_count": int(len(points)),
        }

    center, radius, method, _dist = _robust_radius(
        work,
        hi_quantile=hi_quantile,
        iqr_k=iqr_k,
        guard_extrema=scale_source == "cameras",
    )
    if len(cams):
        radius = max(radius, camera_radius)
    q_lo = np.quantile(work, lo_quantile, axis=0)
    q_hi = np.quantile(work, hi_quantile, axis=0)
    pad = max(0.15 * radius, 4.0 * camera_nn, 0.05 * camera_diagonal, 1e-6)
    limit = float(radius + pad)
    keep = finite & np.all(np.abs(points - center) <= limit, axis=1)
    clip_min = center - limit
    clip_max = center + limit
    full_min = points[finite].min(axis=0)
    full_max = points[finite].max(axis=0)
    return {
        "keep": keep,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "center": center,
        "limit": limit,
        "pad": float(pad),
        "quantile_min": q_lo,
        "quantile_max": q_hi,
        "lo_quantile": lo_quantile,
        "hi_quantile": hi_quantile,
        "iqr_k": iqr_k,
        "method": method,
        "scale_source": scale_source,
        "camera_nn": float(camera_nn),
        "camera_diagonal": float(camera_diagonal),
        "camera_diagonal_full": float(camera_diagonal_full),
        "camera_radius": float(camera_radius),
        "camera_method": camera_method,
        "robust_diagonal": float(np.linalg.norm(clip_max - clip_min)),
        "full_min": full_min,
        "full_max": full_max,
        "full_diagonal": float(np.linalg.norm(full_max - full_min)),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "retained_count": int(keep.sum()),
        "excluded_count": int((~keep).sum()),
    }


def sphere_mesh(
    center: Sequence[float],
    radius: float,
    stacks: int = 8,
    slices: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """UV-sphere vertices and triangle faces for a solid CloudCompare marker."""

    origin = np.asarray(center, dtype=float).reshape(3)
    stacks = max(3, int(stacks))
    slices = max(4, int(slices))
    verts = [origin + np.array([0.0, radius, 0.0], dtype=float)]
    for ring in range(1, stacks):
        polar = math.pi * ring / stacks
        y = math.cos(polar) * radius
        rad = math.sin(polar) * radius
        for spoke in range(slices):
            az = 2.0 * math.pi * spoke / slices
            verts.append(origin + np.array([rad * math.cos(az), y, rad * math.sin(az)], dtype=float))
    verts.append(origin + np.array([0.0, -radius, 0.0], dtype=float))
    vertices = np.asarray(verts, dtype=float)
    faces: list[tuple[int, int, int]] = []
    for spoke in range(slices):
        faces.append((0, 1 + spoke, 1 + (spoke + 1) % slices))
    for ring in range(stacks - 2):
        row = 1 + ring * slices
        nxt = row + slices
        for spoke in range(slices):
            a = row + spoke
            b = row + (spoke + 1) % slices
            c = nxt + (spoke + 1) % slices
            d = nxt + spoke
            faces.append((a, d, b))
            faces.append((b, d, c))
    bottom = len(vertices) - 1
    last = 1 + (stacks - 2) * slices
    for spoke in range(slices):
        faces.append((last + spoke, bottom, last + (spoke + 1) % slices))
    return vertices, np.asarray(faces, dtype=np.int32)


def write_binary_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    faces: np.ndarray | None = None,
    comments: Sequence[str] = (),
) -> Path:
    """Write a packed CloudCompare-readable little-endian RGB PLY."""

    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("xyz and rgb must have the same length")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "ply",
        "format binary_little_endian 1.0",
    ]
    for comment in comments:
        header.append(f"comment {comment}")
    header.extend(
        [
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    )
    face_array = None
    if faces is not None:
        face_array = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        header.append(f"element face {len(face_array)}")
        header.append("property list uchar int vertex_indices")
    header.append("end_header")
    vertex_dt = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    )
    if vertex_dt.itemsize != 15:
        raise RuntimeError(f"PLY vertex record must be 15 bytes, got {vertex_dt.itemsize}")
    records = np.empty(len(points), dtype=vertex_dt)
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["r"] = colors[:, 0]
    records["g"] = colors[:, 1]
    records["b"] = colors[:, 2]
    with dest.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        handle.write(records.tobytes(order="C"))
        if face_array is not None:
            face_dt = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
            if face_dt.itemsize != 13:
                raise RuntimeError(f"PLY face record must be 13 bytes, got {face_dt.itemsize}")
            face_records = np.empty(len(face_array), dtype=face_dt)
            face_records["n"] = 3
            face_records["i0"] = face_array[:, 0]
            face_records["i1"] = face_array[:, 1]
            face_records["i2"] = face_array[:, 2]
            handle.write(face_records.tobytes(order="C"))
    return dest


def _artifact_paths(
    output_dir: Path,
    filename: str,
    issues: Sequence[str] = (),
) -> dict[str, Any]:
    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".ply"
    return {
        "cloudcompare": output_dir / f"{stem}_cloudcompare{suffix}",
        "full": output_dir / f"{stem}_full{suffix}",
        "mesh": output_dir / f"{stem}_markers_mesh{suffix}",
        "base": output_dir / f"{stem}_cloudcompare_base{suffix}",
        "markers": output_dir / f"{stem}_cloudcompare_markers{suffix}",
        "markers_raw": output_dir / f"{stem}_cloudcompare_markers_raw{suffix}",
        "classes": {
            issue: output_dir / f"{stem}_cloudcompare_{issue}{suffix}" for issue in issues
        },
    }


def _stack_xyz_rgb(
    map_xyz: np.ndarray,
    map_rgb: np.ndarray,
    marker_xyz: list[np.ndarray],
    marker_rgb: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    chunks_xyz = [part for part in (map_xyz, *marker_xyz) if len(part)]
    chunks_rgb = [part for part in (map_rgb, *marker_rgb) if len(part)]
    if not chunks_xyz:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)
    return np.vstack(chunks_xyz), np.vstack(chunks_rgb)


def _write_legend_md(path: Path, receipt: Mapping[str, Any]) -> Path:
    clip = receipt.get("clip") or {}
    cc = receipt.get("cloudcompare") or {}
    aggregation = receipt.get("aggregation") or {}
    ply_classes = receipt.get("ply_classes") or {}
    lines = [
        "# Localization risk PLY",
        "",
        "## CloudCompare load workflow",
        "",
        "Load **separate entities**. Do not merge. RGB on, scalar fields off.",
        "",
        f"1. Open base (neutral gray, auto-fit this only): `{receipt.get('ply_base')}`",
        f"2. Open all-class markers as a new entity: `{receipt.get('ply_markers')}`",
        "3. Or open one class at a time:",
    ]
    for issue, class_path in ply_classes.items():
        lines.append(f"   - `{issue}`: `{class_path}`")
    if not ply_classes:
        lines.append("   - (no class files; no in-clip markers)")
    lines.extend(
        [
            f"4. Optional raw centers: `{receipt.get('ply_markers_raw')}`",
            f"5. Combined aggregated preview (not the primary view): `{receipt.get('ply')}`",
            f"6. Full-extent archival (do not auto-fit): `{receipt.get('ply_full')}`",
            f"7. Solid marker mesh: `{receipt.get('ply_mesh')}`",
            "",
            "- Point size: base 2–3 px; marker layers 6–8 px.",
            "- Background: dark navy or mid gray, not pure black.",
            "- Accept Global Shift / scale if CloudCompare prompts.",
            "",
            f"- Format: `{cc.get('format', CLOUDCOMPARE_FORMAT)}` with RGB uchar properties.",
            f"- Map vertices full/retained/excluded: `{receipt.get('map_vertices')}` / "
            f"`{receipt.get('map_vertices_retained')}` / `{receipt.get('map_vertices_excluded')}`",
            f"- Raw markers total/CloudCompare/excluded: `{receipt.get('marker_spheres')}` / "
            f"`{receipt.get('marker_spheres_cloudcompare')}` / `{receipt.get('marker_spheres_excluded')}`",
            f"- Combined display spheres (aggregated): `{receipt.get('display_spheres')}`",
            f"- Aggregation radius / dense cluster count: `{aggregation.get('radius')}` / "
            f"`{aggregation.get('cluster_count')}`",
            f"- Robust bounds diagonal: `{clip.get('robust_diagonal')}`",
            f"- Camera path diagonal / NN: `{clip.get('camera_diagonal')}` / `{clip.get('camera_nn')}`",
            f"- Full bounds diagonal: `{clip.get('full_diagonal')}`",
            f"- Marker radius / samples: `{receipt.get('sphere_radius')}` / `{receipt.get('sphere_samples')}`",
            f"- Radius source / camera-diag cap: `{receipt.get('sphere_radius_source')}` / "
            f"`{receipt.get('sphere_radius_cap')}`",
            f"- Visible base RGB fallback: `{cc.get('visible_rgb')}`",
            f"- Clipping receipt: `{receipt.get('clipping_receipt')}`",
            "",
            "## CloudCompare",
            "",
        ]
    )
    for item in receipt.get("viewer_instructions") or VIEWER_INSTRUCTIONS:
        lines.append(f"- {item}")
    lines.extend(["", "| class | RGB | meaning |", "|---|---|---|"])
    colors = receipt.get("colors_rgb") or {}
    legend = receipt.get("legend") or {}
    for name, meaning in legend.items():
        rgb = ",".join(str(v) for v in colors.get(name, ()))
        lines.append(f"| `{name}` | {rgb} | {meaning} |")
    lines.append("")
    for caveat in receipt.get("caveats") or ():
        lines.append(f"- {caveat}")
    dest = Path(path)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_ascii_ply(
    path: str | Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> Path:
    points = np.asarray(xyz, dtype=float).reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("xyz and rgb must have the same length")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    lines = [header]
    for (x, y, z), (r, g, b) in zip(points, colors):
        lines.append(f"{x:.9g} {y:.9g} {z:.9g} {int(r)} {int(g)} {int(b)}\n")
    dest.write_text("".join(lines), encoding="utf-8")
    return dest


def write_risk_ply(
    map_data: MapData,
    output_dir: str | Path,
    *,
    heatmap: str | Path | Sequence[Mapping[str, Any]] | None = None,
    weak_regions: str | Path | Sequence[Mapping[str, Any]] | None = None,
    localization: str | Path | Sequence[Mapping[str, Any]] | None = None,
    extra_markers: Sequence[Mapping[str, Any]] | None = None,
    image_roles: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    sphere_radius: float | None = None,
    sphere_samples: int = DEFAULT_SPHERE_SAMPLES,
    include_actloc_shadow: bool = False,
    filename: str = "localization_risk_spheres.ply",
) -> dict[str, Any]:
    """Write layered CloudCompare PLYs, archival full PLY, and a clip receipt."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    heatmap_rows = _coerce_rows(heatmap)
    weak_rows = _coerce_rows(weak_regions)
    loc_rows = _coerce_rows(localization)
    roles = normalize_image_roles(image_roles)
    skip_keys: set[tuple[str, str]] = set()
    accepted_extra: list[dict[str, Any]] = []
    for row in extra_markers or ():
        if not isinstance(row, Mapping):
            continue
        issue = str(row.get("issue_class") or "")
        if issue in _ZERO_OBS_ISSUES:
            skip_keys |= _image_keys(row)
        accepted_extra.append(dict(row))
    markers = markers_from_map(map_data, image_roles=roles, skip_image_keys=skip_keys)
    markers.extend(markers_from_heatmap(heatmap_rows))
    markers.extend(markers_from_weak_regions(weak_rows))
    markers.extend(markers_from_localization(loc_rows))
    for row in accepted_extra:
        issue = str(row.get("issue_class") or "")
        if issue not in ISSUE_COLORS:
            continue
        if issue == "actloc_shadow" and not include_actloc_shadow:
            continue
        x, y, z = _as_float(row.get("x")), _as_float(row.get("y")), _as_float(row.get("z"))
        if x is None or y is None or z is None:
            continue
        markers.append(dict(row))

    map_xyz = np.asarray(map_data.points_xyz, dtype=float).reshape(-1, 3)
    map_rgb = np.asarray(map_data.point_rgb, dtype=np.uint8).reshape(-1, 3)
    point_ids = np.asarray(map_data.point_ids)
    clip = robust_spatial_clip(map_xyz, camera_xyz=map_data.image_centers)
    keep = np.asarray(clip["keep"], dtype=bool)
    retained_xyz = map_xyz[keep]
    retained_rgb_src = map_rgb[keep]
    cc_map_rgb, rgb_fallback = visible_map_rgb(retained_rgb_src)
    excluded_idx = np.flatnonzero(~keep)
    if len(point_ids) == len(map_xyz):
        excluded_ids = [int(v) for v in point_ids[excluded_idx]]
    else:
        excluded_ids = [int(v) for v in excluded_idx]

    auto_radius, auto_meta = display_radius_from_clip(clip)
    if sphere_radius is None:
        sphere_radius = float(auto_radius)
        radius_meta = auto_meta
    else:
        sphere_radius, radius_meta = display_radius_from_clip(clip, override=sphere_radius)
        sphere_radius = float(sphere_radius)
    sphere_samples = max(8, int(sphere_samples))
    agg_radius = aggregation_radius_from_clip(clip)
    clip_min = np.asarray(clip["clip_min"], dtype=float)
    clip_max = np.asarray(clip["clip_max"], dtype=float)
    full_marker_xyz: list[np.ndarray] = []
    full_marker_rgb: list[np.ndarray] = []
    overlay_xyz: list[np.ndarray] = []
    overlay_rgb: list[np.ndarray] = []
    mesh_xyz: list[np.ndarray] = []
    mesh_rgb: list[np.ndarray] = []
    mesh_faces: list[np.ndarray] = []
    class_shells: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in ISSUE_COLORS
    }
    raw_pts: list[np.ndarray] = []
    raw_cols: list[np.ndarray] = []
    counts: dict[str, int] = {name: 0 for name in ISSUE_COLORS}
    excluded_markers: list[dict[str, Any]] = []
    excluded_marker_classes: dict[str, int] = {}
    in_clip_indices: list[int] = []
    class_indices: dict[str, list[int]] = {name: [] for name in ISSUE_COLORS}
    mesh_offset = 0
    unit_shell = sphere_points((0.0, 0.0, 0.0), 1.0, sphere_samples)
    unit_mesh_xyz, unit_mesh_faces = sphere_mesh((0.0, 0.0, 0.0), 1.0)
    radius_by_class = {
        name: class_marker_radius(name, sphere_radius) for name in ISSUE_COLORS
    }
    for index, row in enumerate(markers):
        issue = str(row["issue_class"])
        counts[issue] = counts.get(issue, 0) + 1
        color = np.asarray(_rgb(issue), dtype=np.uint8)
        center = (row["x"], row["y"], row["z"])
        in_clip = _xyz_in_clip(center, clip_min, clip_max)
        origin = np.asarray(center, dtype=float).reshape(3)
        marker_radius = radius_by_class.get(issue, class_marker_radius(issue, sphere_radius))
        shell = unit_shell * marker_radius + origin
        shell_rgb = np.repeat(color.reshape(1, 3), len(shell), axis=0)
        full_marker_xyz.append(shell)
        full_marker_rgb.append(shell_rgb)
        row["sphere_index"] = index
        row["in_cloudcompare_clip"] = in_clip
        row["display_radius"] = float(marker_radius)
        verts = unit_mesh_xyz * marker_radius + origin
        mesh_xyz.append(verts)
        mesh_rgb.append(np.repeat(color.reshape(1, 3), len(verts), axis=0))
        mesh_faces.append(unit_mesh_faces + mesh_offset)
        mesh_offset += len(verts)
        if not in_clip:
            excluded_marker_classes[issue] = excluded_marker_classes.get(issue, 0) + 1
            image_id = _as_int(row.get("image_id"))
            excluded_markers.append(
                {
                    "sphere_index": index,
                    "image_id": image_id,
                    "image_name": row.get("image_name"),
                    "issue_class": issue,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                }
            )
            continue
        in_clip_indices.append(index)
        class_indices.setdefault(issue, []).append(index)
        overlay_xyz.append(shell)
        overlay_rgb.append(shell_rgb)
        class_shells.setdefault(issue, []).append((shell, shell_rgb))
        raw_pts.append(origin.reshape(1, 3))
        raw_cols.append(color.reshape(1, 3))

    display_items: list[dict[str, Any]] = []
    aggregation_clusters: list[dict[str, Any]] = []
    for issue in ISSUE_COLORS:
        idxs = class_indices.get(issue) or []
        if not idxs:
            continue
        issue_radius = radius_by_class[issue]
        if issue in DENSE_COMBINED_CLASSES:
            clusters = cluster_marker_indices(markers, idxs, agg_radius)
            for cluster in clusters:
                cluster["display_radius"] = float(issue_radius)
                aggregation_clusters.append(cluster)
                display_items.append(
                    {
                        "issue_class": issue,
                        "center": cluster["center"],
                        "radius": issue_radius,
                        "member_count": cluster["member_count"],
                        "members": cluster["members"],
                        "aggregated": True,
                    }
                )
            continue
        for index in idxs:
            row = markers[index]
            display_items.append(
                {
                    "issue_class": issue,
                    "center": [float(row["x"]), float(row["y"]), float(row["z"])],
                    "radius": issue_radius,
                    "member_count": 1,
                    "members": [index],
                    "aggregated": False,
                }
            )

    display_xyz: list[np.ndarray] = []
    display_rgb: list[np.ndarray] = []
    for item in display_items:
        origin = np.asarray(item["center"], dtype=float).reshape(3)
        color = np.asarray(_rgb(str(item["issue_class"])), dtype=np.uint8)
        shell = unit_shell * float(item["radius"]) + origin
        display_xyz.append(shell)
        display_rgb.append(np.repeat(color.reshape(1, 3), len(shell), axis=0))

    finite_map = np.isfinite(map_xyz).all(axis=1)
    full_xyz, full_rgb = _stack_xyz_rgb(
        map_xyz[finite_map],
        map_rgb[finite_map],
        full_marker_xyz,
        full_marker_rgb,
    )
    cc_xyz, cc_rgb = _stack_xyz_rgb(retained_xyz, cc_map_rgb, display_xyz, display_rgb)
    overlay_stack_xyz, overlay_stack_rgb = _stack_xyz_rgb(
        np.zeros((0, 3), dtype=float),
        np.zeros((0, 3), dtype=np.uint8),
        overlay_xyz,
        overlay_rgb,
    )
    if raw_pts:
        raw_xyz = np.vstack(raw_pts)
        raw_rgb = np.vstack(raw_cols)
    else:
        raw_xyz = np.zeros((0, 3), dtype=float)
        raw_rgb = np.zeros((0, 3), dtype=np.uint8)
    base_rgb = np.repeat(
        np.asarray(VISIBLE_MAP_RGB, dtype=np.uint8).reshape(1, 3),
        len(retained_xyz),
        axis=0,
    )
    excluded_marker_ids = [
        int(item["image_id"]) if item["image_id"] is not None else int(item["sphere_index"])
        for item in excluded_markers
    ]
    marker_spheres_cc = int(len(in_clip_indices))
    present_issues = [name for name, idxs in class_indices.items() if idxs]
    paths = _artifact_paths(out, filename, present_issues)
    ply_path = write_binary_ply(
        paths["cloudcompare"],
        cc_xyz,
        cc_rgb,
        comments=(
            "sfm-diagnosis CloudCompare combined preview; dense held-out classes aggregated",
            "load *_cloudcompare_base.ply plus class files as separate entities",
        ),
    )
    base_path = write_binary_ply(
        paths["base"],
        retained_xyz,
        base_rgb,
        comments=("sfm-diagnosis CloudCompare base cloud; neutral gray; robust-clipped",),
    )
    markers_path = write_binary_ply(
        paths["markers"],
        overlay_stack_xyz,
        overlay_stack_rgb,
        comments=("sfm-diagnosis CloudCompare all-class markers; no base cloud",),
    )
    raw_path = write_binary_ply(
        paths["markers_raw"],
        raw_xyz,
        raw_rgb,
        comments=("sfm-diagnosis raw in-clip marker centers; one vertex per raw marker",),
    )
    ply_classes: dict[str, str] = {}
    class_vertex_counts: dict[str, int] = {}
    for issue in present_issues:
        shells = class_shells.get(issue) or []
        class_xyz, class_rgb = _stack_xyz_rgb(
            np.zeros((0, 3), dtype=float),
            np.zeros((0, 3), dtype=np.uint8),
            [shell for shell, _color in shells],
            [color for _shell, color in shells],
        )
        class_path = write_binary_ply(
            paths["classes"][issue],
            class_xyz,
            class_rgb,
            comments=(f"sfm-diagnosis CloudCompare {issue} markers; no base cloud",),
        )
        ply_classes[issue] = str(class_path)
        class_vertex_counts[issue] = int(len(class_xyz))
    full_path = write_binary_ply(
        paths["full"],
        full_xyz,
        full_rgb,
        comments=("sfm-diagnosis full-extent archival risk PLY; do not auto-fit",),
    )
    if mesh_xyz:
        mesh_v = np.vstack(mesh_xyz)
        mesh_c = np.vstack(mesh_rgb)
        mesh_f = np.vstack(mesh_faces)
    else:
        mesh_v = np.zeros((0, 3), dtype=float)
        mesh_c = np.zeros((0, 3), dtype=np.uint8)
        mesh_f = np.zeros((0, 3), dtype=np.int32)
    mesh_path = write_binary_ply(
        paths["mesh"],
        mesh_v,
        mesh_c,
        faces=mesh_f,
        comments=("sfm-diagnosis solid risk-marker mesh",),
    )

    robust_diag = float(clip["robust_diagonal"])
    full_diag = float(clip["full_diagonal"])
    camera_diag = float(clip["camera_diagonal"])
    radius_cap = (
        MARKER_RADIUS_CAMERA_DIAG_CAP * camera_diag if camera_diag > 0.0 else None
    )
    clip_payload = {
        "method": clip["method"],
        "scale_source": clip["scale_source"],
        "lo_quantile": clip["lo_quantile"],
        "hi_quantile": clip["hi_quantile"],
        "iqr_k": clip["iqr_k"],
        "center": [float(v) for v in np.asarray(clip["center"])],
        "limit": float(clip["limit"]),
        "pad": float(clip["pad"]),
        "clip_min": [float(v) for v in clip_min],
        "clip_max": [float(v) for v in clip_max],
        "robust_bounds": {
            "min": [float(v) for v in clip_min],
            "max": [float(v) for v in clip_max],
            "diagonal": robust_diag,
        },
        "full_bounds": {
            "min": [float(v) for v in clip["full_min"]],
            "max": [float(v) for v in clip["full_max"]],
            "diagonal": full_diag,
        },
        "quantile_bounds": {
            "min": [float(v) for v in clip["quantile_min"]],
            "max": [float(v) for v in clip["quantile_max"]],
        },
        "camera_nn": float(clip["camera_nn"]),
        "camera_diagonal": camera_diag,
        "camera_diagonal_full": float(clip.get("camera_diagonal_full") or 0.0),
        "camera_radius": float(clip.get("camera_radius") or 0.0),
        "camera_method": clip.get("camera_method"),
        "robust_diagonal": robust_diag,
        "full_diagonal": full_diag,
        "robust_diagonal_not_dominated_by_extrema": bool(
            full_diag <= 0.0 or robust_diag * 20.0 < full_diag or int(clip["excluded_count"]) > 0
        ),
        "finite_count": int(clip["finite_count"]),
        "nonfinite_count": int(clip["nonfinite_count"]),
        "retained_count": int(clip["retained_count"]),
        "excluded_count": int(clip["excluded_count"]),
        "excluded_indices": [int(v) for v in excluded_idx],
        "excluded_point_ids": excluded_ids,
        "excluded_marker_count": int(len(excluded_markers)),
        "excluded_marker_ids": excluded_marker_ids,
        "excluded_marker_classes": dict(excluded_marker_classes),
        "excluded_markers": excluded_markers,
    }
    display_by_class: dict[str, int] = {}
    for item in display_items:
        issue = str(item["issue_class"])
        display_by_class[issue] = display_by_class.get(issue, 0) + 1
    max_display_radius = max((float(item["radius"]) for item in display_items), default=0.0)
    base_n = int(len(retained_xyz))
    combined_n = int(len(cc_xyz))
    display_n = int(len(display_items))
    visual_balance = {
        "base_vertex_count": base_n,
        "combined_vertex_count": combined_n,
        "display_vertex_count": int(combined_n - base_n),
        "base_vertex_fraction": float(base_n / combined_n) if combined_n else 1.0,
        "base_rgb_majority": bool(base_n > (combined_n - base_n)),
        "display_sphere_count": display_n,
        "raw_marker_count": int(len(markers)),
        "raw_marker_count_in_clip": marker_spheres_cc,
        "max_display_radius": max_display_radius,
        "radius_bounded_by_camera_diagonal": bool(
            radius_cap is None or sphere_radius <= radius_cap + 1e-12
        ),
    }
    aggregation = {
        "classes": sorted(DENSE_COMBINED_CLASSES),
        "radius": float(agg_radius),
        "raw_marker_count": int(len(markers)),
        "cluster_count": int(len(aggregation_clusters)),
        "clusters": aggregation_clusters,
    }
    cloudcompare = {
        "ply": str(ply_path),
        "ply_base": str(base_path),
        "ply_markers": str(markers_path),
        "ply_markers_raw": str(raw_path),
        "ply_classes": dict(ply_classes),
        "format": "binary_little_endian",
        "rgb": True,
        "vertex_count": combined_n,
        "map_vertices_retained": base_n,
        "map_vertices_excluded": int(clip["excluded_count"]),
        "marker_spheres": marker_spheres_cc,
        "marker_spheres_total": int(len(markers)),
        "marker_spheres_excluded": int(len(excluded_markers)),
        "display_spheres": display_n,
        "display_spheres_by_class": display_by_class,
        "class_vertex_counts": class_vertex_counts,
        "sphere_radius": float(sphere_radius),
        "sphere_radius_by_class": {key: float(value) for key, value in radius_by_class.items()},
        "sphere_samples": int(sphere_samples),
        "robust_bounds": clip_payload["robust_bounds"],
        "visible_rgb": rgb_fallback,
        "payload_bytes": int(combined_n * 15),
        "visual_balance": visual_balance,
    }
    receipt = {
        "schema_version": 3,
        "artifact_type": "SFM_DIAGNOSIS_RISK_PLY",
        "ply": str(ply_path),
        "ply_base": str(base_path),
        "ply_markers": str(markers_path),
        "ply_markers_raw": str(raw_path),
        "ply_classes": dict(ply_classes),
        "ply_full": str(full_path),
        "ply_mesh": str(mesh_path),
        "format": "binary_little_endian",
        "map_vertices": int(len(map_xyz)),
        "map_vertices_retained": base_n,
        "map_vertices_excluded": int(clip["excluded_count"]),
        "marker_spheres": int(len(markers)),
        "marker_spheres_cloudcompare": marker_spheres_cc,
        "marker_spheres_excluded": int(len(excluded_markers)),
        "display_spheres": display_n,
        "display_spheres_by_class": display_by_class,
        "sphere_samples": int(sphere_samples),
        "sphere_radius": float(sphere_radius),
        "sphere_radius_auto": float(auto_radius),
        "sphere_radius_source": radius_meta["source"],
        "sphere_radius_cap": radius_cap,
        "sphere_radius_by_class": {key: float(value) for key, value in radius_by_class.items()},
        "vertex_count": combined_n,
        "vertex_count_full": int(len(full_xyz)),
        "mesh_vertex_count": int(len(mesh_v)),
        "mesh_face_count": int(len(mesh_f)),
        "counts": {key: value for key, value in counts.items() if value},
        "colors_rgb": {key: list(value) for key, value in ISSUE_COLORS.items()},
        "legend": ISSUE_LEGEND,
        "caveats": list(CAVEATS),
        "viewer_instructions": list(VIEWER_INSTRUCTIONS),
        "actloc": (
            "SHADOW_ONLY_IF_EXPLICITLY_INCLUDED"
            if include_actloc_shadow
            else "NOT_RUN_UNLICENSED_UNCALIBRATED_SHADOW_ONLY"
        ),
        "fim_recomputed": False,
        "inputs": {
            "heatmap_rows": len(heatmap_rows),
            "weak_region_rows": len(weak_rows),
            "localization_rows": len(loc_rows),
            "image_role_rows": len(roles),
        },
        "clip": clip_payload,
        "clipping_receipt": str(out / "risk_ply_clipping.json"),
        "cloudcompare": cloudcompare,
        "aggregation": aggregation,
        "visual_balance": visual_balance,
        "markers": markers,
    }
    write_json(out / "legend.json", {key: ISSUE_LEGEND[key] for key in ISSUE_COLORS})
    write_json(out / "risk_ply_clipping.json", clip_payload)
    write_json(out / "risk_ply_receipt.json", receipt)
    _write_legend_md(out / "LEGEND.md", receipt)
    return receipt



__all__ = [
    "CAVEATS",
    "CLASS_RADIUS_SCALE",
    "DENSE_COMBINED_CLASSES",
    "ISSUE_COLORS",
    "ISSUE_LEGEND",
    "MARKER_RADIUS_ABS_FLOOR",
    "MARKER_RADIUS_CAMERA_DIAG_CAP",
    "VIEWER_INSTRUCTIONS",
    "VISIBLE_MAP_RGB",
    "aggregation_radius_from_clip",
    "camera_nearest_spacing",
    "class_marker_radius",
    "cluster_marker_indices",
    "display_radius_from_clip",
    "load_jsonl_rows",
    "load_rows",
    "markers_from_heatmap",
    "markers_from_localization",
    "markers_from_map",
    "markers_from_weak_regions",
    "normalize_image_roles",
    "robust_spatial_clip",
    "sphere_mesh",
    "sphere_points",
    "visible_map_rgb",
    "write_ascii_ply",
    "write_binary_ply",
    "write_risk_ply",
]
