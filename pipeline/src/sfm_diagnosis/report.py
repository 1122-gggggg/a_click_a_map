from __future__ import annotations

import numpy as np

from .graph import build_covisibility_graph, nearest_neighbor_distances
from .models import MapData


def map_health_summary(
    map_data: MapData,
    *,
    covisibility_min_shared: int = 15,
    max_track_for_pair_expansion: int | None = None,
) -> dict:
    """Compute reconstruction-wide health statistics and a covisibility graph audit."""
    tl = map_data.track_lengths.astype(float)
    err = map_data.point_errors.astype(float)
    quality = map_data.point_quality_weights()
    tri_angle = map_data.triangulation_angles_deg()
    graph = build_covisibility_graph(
        map_data,
        min_shared_points=covisibility_min_shared,
        max_track_for_pair_expansion=max_track_for_pair_expansion,
    )

    camera_spacing = nearest_neighbor_distances(map_data.image_centers)
    point_min, point_max = map_data.bounds
    result = {
        "source": map_data.metadata.get("source"),
        "model_dir": map_data.metadata.get("model_dir"),
        "num_images": map_data.num_images,
        "num_points3D": map_data.num_points,
        "num_cameras": len(map_data.cameras),
        "bounds_min": point_min.tolist(),
        "bounds_max": point_max.tolist(),
        "track_length": _percentiles(tl),
        "reprojection_error_px": _percentiles(err),
        "triangulation_angle_deg": _percentiles(tri_angle),
        "point_quality_weight": _percentiles(quality),
        "two_view_track_fraction": float(np.mean(tl == 2)) if len(tl) else None,
        "short_track_fraction_lt3": float(np.mean(tl < 3)) if len(tl) else None,
        "high_reprojection_error_fraction_gt3px": (
            float(np.mean(err > 3.0)) if len(err) else None
        ),
        "low_triangulation_angle_fraction_lt2deg": (
            float(np.mean(tri_angle < 2.0)) if len(tri_angle) else None
        ),
        "image_point_support": _percentiles(graph.image_support.astype(float)),
        "mapping_camera_nearest_neighbor_distance": _percentiles(camera_spacing),
        "covisibility": {
            "min_shared_points": covisibility_min_shared,
            "support_mode": graph.support_mode,
            "omitted_long_track_count": graph.omitted_long_track_count,
            "pair_counts_threshold_retained": graph.support_mode == "exact",
            "strong_edges": graph.strong_edges,
            "connected_components": len(graph.components),
            "largest_component_images": max(
                (len(component) for component in graph.components),
                default=0,
            ),
            "isolated_images": (
                int(np.sum(graph.degrees == 0)) if len(graph.degrees) else 0
            ),
            "degree": _percentiles(graph.degrees.astype(float)),
        },
        "weak_image_ids": [
            int(map_data.image_ids[i])
            for i in np.flatnonzero(
                (
                    graph.image_support
                    < max(
                        25,
                        int(np.percentile(graph.image_support, 10))
                        if len(graph.image_support)
                        else 25,
                    )
                )
                | (graph.degrees == 0)
            )[:100]
        ],
    }
    return result


def _percentiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    p = np.percentile(values, [10, 50, 90, 95])
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p10": float(p[0]),
        "p50": float(p[1]),
        "p90": float(p[2]),
        "p95": float(p[3]),
    }
