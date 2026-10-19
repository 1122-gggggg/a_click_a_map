from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import MapData, Pose


@dataclass(frozen=True)
class QueryGeometryStats:
    """Query-visible triangulation and observation-scale statistics.

    Separate from view-support redetectability and from map-level LOW_PARALLAX.
    """

    triangulation_angle_p10_deg: float | None
    triangulation_angle_p50_deg: float | None
    triangulation_angle_p90_deg: float | None
    low_parallax_visible_fraction: float
    query_range_p50_m: float | None
    historical_range_p50_m: float | None
    scale_ratio_p50: float | None
    scale_ratio_p90: float | None
    scale_extrapolated_fraction: float

    def as_dict(self) -> dict:
        return {
            "triangulation_angle_p10_deg": self.triangulation_angle_p10_deg,
            "triangulation_angle_p50_deg": self.triangulation_angle_p50_deg,
            "triangulation_angle_p90_deg": self.triangulation_angle_p90_deg,
            "low_parallax_visible_fraction": self.low_parallax_visible_fraction,
            "query_range_p50_m": self.query_range_p50_m,
            "historical_range_p50_m": self.historical_range_p50_m,
            "scale_ratio_p50": self.scale_ratio_p50,
            "scale_ratio_p90": self.scale_ratio_p90,
            "scale_extrapolated_fraction": self.scale_extrapolated_fraction,
        }


def compute_query_geometry_stats(
    map_data: MapData,
    pose: Pose,
    visible_idx: np.ndarray,
    range_expansion: float = 0.40,
    parallax_angle_deg: float = 2.0,
) -> QueryGeometryStats:
    """Compute triangulation-angle and scale statistics over visible landmarks."""
    visible = np.unique(np.asarray(visible_idx, dtype=int).reshape(-1))
    if len(visible) == 0:
        return _empty()

    thetas = np.asarray(map_data.triangulation_angles_deg(visible), dtype=float)
    xyz = map_data.points_xyz[visible]
    query_ranges = np.linalg.norm(xyz - pose.center_w.reshape(1, 3), axis=1)
    low_parallax = float(np.mean(thetas < float(parallax_angle_deg)))
    expansion = float(np.clip(range_expansion, 0.0, 0.95))
    image_lookup = map_data.image_index()

    rhos: list[float] = []
    hist_medians: list[float] = []
    extrapolated = 0
    counted = 0
    for i, pidx in enumerate(visible.tolist()):
        centers = [
            map_data.image_centers[image_lookup[int(image_id)]]
            for image_id in map_data.track_image_ids[int(pidx)]
            if int(image_id) in image_lookup
        ]
        if not centers:
            continue
        hist = np.linalg.norm(xyz[i] - np.asarray(centers, dtype=float), axis=1)
        hist = hist[hist > 1e-9]
        if len(hist) == 0:
            continue
        counted += 1
        median_hist = float(np.median(hist))
        hist_medians.append(median_hist)
        if median_hist > 1e-12:
            rhos.append(float(query_ranges[i] / median_hist))
        in_envelope = (1.0 - expansion) * float(np.min(hist)) <= float(
            query_ranges[i]
        ) <= (1.0 + expansion) * float(np.max(hist))
        extrapolated += int(not in_envelope)

    return QueryGeometryStats(
        triangulation_angle_p10_deg=_pct(thetas, 10),
        triangulation_angle_p50_deg=_pct(thetas, 50),
        triangulation_angle_p90_deg=_pct(thetas, 90),
        low_parallax_visible_fraction=low_parallax,
        query_range_p50_m=_pct(query_ranges, 50),
        historical_range_p50_m=_pct(hist_medians, 50),
        scale_ratio_p50=_pct(rhos, 50),
        scale_ratio_p90=_scale_ratio_p90(rhos),
        scale_extrapolated_fraction=extrapolated / max(counted, 1),
    )


def _empty() -> QueryGeometryStats:
    return QueryGeometryStats(
        triangulation_angle_p10_deg=None,
        triangulation_angle_p50_deg=None,
        triangulation_angle_p90_deg=None,
        low_parallax_visible_fraction=0.0,
        query_range_p50_m=None,
        historical_range_p50_m=None,
        scale_ratio_p50=None,
        scale_ratio_p90=None,
        scale_extrapolated_fraction=0.0,
    )


def _pct(values, q: float) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    return float(np.percentile(arr, q))


def _scale_ratio_p90(rhos: list[float]) -> float | None:
    arr = np.asarray(rhos, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return None
    return float(np.exp(np.percentile(np.abs(np.log(arr)), 90)))
