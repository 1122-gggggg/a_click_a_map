"""Query-time viewpoint gates for MegaLoc + EDM + PnP.

Implements the non-recapture repairs:
1. prefer side-looking mapping references during retrieval
2. abstain when the query viewpoint has empty image support
3. drop empty-side/sky/water views from the MegaLoc reference bank
5. never relax frozen PnP gates, especially inside the FIM∩LWTL volume
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

FROZEN_PNP_THRESHOLDS = MappingProxyType(
    {
        "strong_inliers": 80,
        "minimum_inlier_ratio": 0.25,
        "minimum_hull_coverage": 0.15,
        "minimum_occupancy_4x4": 6,
        "minimum_positive_depth_ratio": 0.99,
        "maximum_reprojection_p90_px": 3.0,
    }
)

# LWTL-style 30x30 occupancy. A view that fills fewer bins is looking at
# empty/sky/water even if a few tracks exist.
DEFAULT_MIN_REFERENCE_OCCUPIED_BINS = 40
DEFAULT_MIN_QUERY_OCCUPIED_FRAC_30 = 0.05
DEFAULT_BANK_OCCUPANCY_PERCENTILE = 10.0
INTERSECTION_RADIUS = 0.12
INTERSECTION_CELLS_NAME = "fim_lwtl_intersection_cells.json"


def occupied_bins(
    uv: np.ndarray, *, width: int, height: int, grid: int = 30
) -> tuple[int, float]:
    points = np.asarray(uv, dtype=float).reshape(-1, 2)
    cells = grid * grid
    if len(points) == 0 or width <= 0 or height <= 0 or grid <= 0:
        return 0, 0.0
    bx = np.clip((points[:, 0] / float(width)) * grid, 0, grid - 1e-9).astype(int)
    by = np.clip((points[:, 1] / float(height)) * grid, 0, grid - 1e-9).astype(int)
    count = int(np.unique(bx * grid + by).size)
    return count, float(count) / float(cells)


def reference_observation_occupancy(
    observation_xy: np.ndarray, *, width: int, height: int, grid: int = 30
) -> tuple[int, float]:
    return occupied_bins(observation_xy, width=width, height=height, grid=grid)


def prefer_side_looking_references(
    ranked: Sequence[int],
    occupied_bins_by_index: Sequence[int],
    *,
    min_occupied_bins: int,
    top_k: int,
) -> tuple[tuple[int, ...], bool]:
    """Keep retrieval order, but drop empty-side mapping views when possible."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    occupied = list(occupied_bins_by_index)
    usable = [index for index in ranked if occupied[index] >= int(min_occupied_bins)]
    if usable:
        return tuple(usable[:top_k]), False
    return tuple(list(ranked)[:top_k]), True


def viewpoint_should_abstain(
    *,
    occupancy_4x4: int,
    hull_coverage: float,
    occupied_frac_30: float,
    min_occupancy_4x4: int = 6,
    min_hull_coverage: float = 0.15,
    min_occupied_frac_30: float = DEFAULT_MIN_QUERY_OCCUPIED_FRAC_30,
) -> bool:
    return bool(
        int(occupancy_4x4) < int(min_occupancy_4x4)
        or float(hull_coverage) < float(min_hull_coverage)
        or float(occupied_frac_30) < float(min_occupied_frac_30)
    )


def decide_query_status(
    *,
    registration_success: bool,
    strong_success: bool,
    viewpoint_ok: bool,
) -> str:
    if (not registration_success) or (not viewpoint_ok):
        return "ABSTAINED"
    if strong_success:
        return "LOCALIZED_STRONG"
    return "POSE_ESTIMATED_WEAK"


def never_weaker_than_frozen(
    requested: Mapping[str, float | int] | None, *, in_intersection: bool
) -> dict[str, float | int]:
    """Never go below frozen PnP gates.

    ``in_intersection`` marks the 27 FIM∩LWTL cells so callers cannot special-case
    a weaker policy there. Stricter requested gates are still kept.
    """

    frozen = dict(FROZEN_PNP_THRESHOLDS)
    merged = {**frozen, **dict(requested or {})}
    return {
        "strong_inliers": max(int(merged["strong_inliers"]), int(frozen["strong_inliers"])),
        "minimum_inlier_ratio": max(
            float(merged["minimum_inlier_ratio"]), float(frozen["minimum_inlier_ratio"])
        ),
        "minimum_hull_coverage": max(
            float(merged["minimum_hull_coverage"]), float(frozen["minimum_hull_coverage"])
        ),
        "minimum_occupancy_4x4": max(
            int(merged["minimum_occupancy_4x4"]), int(frozen["minimum_occupancy_4x4"])
        ),
        "minimum_positive_depth_ratio": max(
            float(merged["minimum_positive_depth_ratio"]),
            float(frozen["minimum_positive_depth_ratio"]),
        ),
        "maximum_reprojection_p90_px": min(
            float(merged["maximum_reprojection_p90_px"]),
            float(frozen["maximum_reprojection_p90_px"]),
        ),
    }


def position_in_intersection(
    xyz: Sequence[float], cells: np.ndarray, *, radius: float = INTERSECTION_RADIUS
) -> bool:
    point = np.asarray(xyz, dtype=float).reshape(3)
    table = np.asarray(cells, dtype=float).reshape(-1, 3)
    if len(table) == 0:
        return False
    return bool(np.min(np.linalg.norm(table - point[None, :], axis=1)) <= float(radius))


def occupancy_percentile_threshold(
    occupied_bins_by_name: Sequence[int], percentile: float = DEFAULT_BANK_OCCUPANCY_PERCENTILE
) -> int:
    values = np.asarray(list(occupied_bins_by_name), dtype=float)
    if values.size == 0:
        raise ValueError("occupied bins are empty")
    return int(np.percentile(values, float(percentile)))


def bank_occupancy_cutoff(
    occupied_bins_by_name: Sequence[int],
    *,
    percentile: float = DEFAULT_BANK_OCCUPANCY_PERCENTILE,
    floor: int = DEFAULT_MIN_REFERENCE_OCCUPIED_BINS,
) -> int:
    return max(int(floor), occupancy_percentile_threshold(occupied_bins_by_name, percentile))


def filter_reference_bank(
    names: Sequence[str],
    occupied_bins_by_name: Sequence[int],
    *,
    min_occupied_bins: int,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Keep mapping views whose 30x30 occupancy is above the empty-side cutoff."""

    if len(names) != len(occupied_bins_by_name):
        raise ValueError("names and occupancy must align")
    if int(min_occupied_bins) <= 0:
        raise ValueError("min_occupied_bins must be positive")
    kept: list[int] = []
    dropped: list[str] = []
    for index, (name, occupied) in enumerate(zip(names, occupied_bins_by_name, strict=True)):
        if int(occupied) >= int(min_occupied_bins):
            kept.append(index)
        else:
            dropped.append(str(name))
    if not kept:
        raise RuntimeError("side-looking filter removed every MegaLoc reference")
    return tuple(kept), tuple(dropped)


def align_occupied_bins(
    all_names: Sequence[str],
    occupied_bins_by_name: Sequence[int],
    selected_names: Sequence[str],
) -> tuple[int, ...]:
    if len(all_names) != len(occupied_bins_by_name):
        raise ValueError("names and occupancy must align")
    index = {str(name): position for position, name in enumerate(all_names)}
    missing = [str(name) for name in selected_names if str(name) not in index]
    if missing:
        raise KeyError(f"selected reference identities are absent: {missing[:5]}")
    return tuple(int(occupied_bins_by_name[index[str(name)]]) for name in selected_names)


def resolve_megaloc_bank(localization_dir: str | Path) -> tuple[Path, Path, str]:
    """Prefer the occupancy-filtered sideview bank when both files exist."""

    root = Path(localization_dir)
    side_descriptors = root / "megaloc_references_sideview.npy"
    side_names = root / "megaloc_references_sideview.names.json"
    if side_descriptors.is_file() and side_names.is_file():
        return side_descriptors, side_names, "sideview"
    return (
        root / "megaloc_references.npy",
        root / "megaloc_references.names.json",
        "full",
    )


def load_intersection_cells(path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((0, 3), dtype=float)
    file = Path(path)
    if not file.is_file():
        return np.zeros((0, 3), dtype=float)
    payload = json.loads(file.read_text(encoding="utf-8"))
    cells = payload.get("cell_positions") or payload.get("cells") or []
    array = np.asarray(cells, dtype=float)
    if array.size == 0:
        return np.zeros((0, 3), dtype=float)
    return array.reshape(-1, 3)


def write_sideview_bank(
    output_dir: str | Path,
    names: Sequence[str],
    descriptors: np.ndarray,
    occupied_bins_by_name: Sequence[int],
    *,
    min_occupied_bins: int | None = None,
) -> dict[str, object]:
    cutoff = (
        int(min_occupied_bins)
        if min_occupied_bins is not None
        else bank_occupancy_cutoff(occupied_bins_by_name)
    )
    kept, dropped = filter_reference_bank(
        names, occupied_bins_by_name, min_occupied_bins=cutoff
    )
    occupied_by_name = {
        str(name): int(occupied)
        for name, occupied in zip(names, occupied_bins_by_name, strict=True)
    }
    selected_names = [str(names[index]) for index in kept]
    selected = np.ascontiguousarray(np.asarray(descriptors)[list(kept)], dtype=np.float32)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    descriptor_path = output / "megaloc_references_sideview.npy"
    names_path = output / "megaloc_references_sideview.names.json"
    drop_path = output / "megaloc_references_dropped.json"
    np.save(descriptor_path, selected, allow_pickle=False)
    names_path.write_text(
        json.dumps(selected_names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "MEGALOC_SIDEVIEW_DROPS",
        "min_occupied_bins": cutoff,
        "source_count": len(names),
        "kept_count": len(kept),
        "dropped": [
            {"image_name": name, "occupied_bins": occupied_by_name[name]} for name in dropped
        ],
    }
    drop_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "min_occupied_bins": cutoff,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "descriptors": str(descriptor_path),
        "names": str(names_path),
        "dropped": str(drop_path),
    }


__all__ = [
    "DEFAULT_BANK_OCCUPANCY_PERCENTILE",
    "DEFAULT_MIN_QUERY_OCCUPIED_FRAC_30",
    "DEFAULT_MIN_REFERENCE_OCCUPIED_BINS",
    "FROZEN_PNP_THRESHOLDS",
    "INTERSECTION_CELLS_NAME",
    "INTERSECTION_RADIUS",
    "align_occupied_bins",
    "bank_occupancy_cutoff",
    "decide_query_status",
    "filter_reference_bank",
    "load_intersection_cells",
    "never_weaker_than_frozen",
    "occupancy_percentile_threshold",
    "occupied_bins",
    "position_in_intersection",
    "prefer_side_looking_references",
    "reference_observation_occupancy",
    "resolve_megaloc_bank",
    "viewpoint_should_abstain",
    "write_sideview_bank",
]
