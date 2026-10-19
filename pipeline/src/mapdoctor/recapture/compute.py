from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mapdoctor.metrics import convex_hull_area_fraction, grid_coverage

from .types import Availability, MetricValue


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("recapture geometry metrics require numpy; install mapdoctor-sfm[recapture]") from exc
    return np


def grid_occupancy(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
    rows: int = 4,
    cols: int = 4,
) -> tuple[int, float]:
    """Compatibility wrapper around the canonical map-health grid metric."""
    return grid_coverage(points, width, height, rows=rows, cols=cols)


def image_spatial_entropy(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
    rows: int = 4,
    cols: int = 4,
) -> float:
    if not points or width <= 0 or height <= 0 or rows <= 0 or cols <= 0:
        return 0.0
    counts = [0] * (rows * cols)
    total = 0
    for point in points:
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if 0.0 <= x <= width and 0.0 <= y <= height:
            c = min(cols - 1, int(x / width * cols))
            r = min(rows - 1, int(y / height * rows))
            counts[r * cols + c] += 1
            total += 1
    if total == 0:
        return 0.0
    entropy = -sum((n / total) * math.log(n / total) for n in counts if n)
    return entropy / math.log(rows * cols) if rows * cols > 1 else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    xs = sorted(float(v) for v in values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def depth_summary(depths: Sequence[float]) -> dict[str, float | None]:
    values = [float(v) for v in depths]
    positive = [v for v in values if v > 0 and math.isfinite(v)]
    ratio = len(positive) / len(values) if values else 0.0
    if not positive:
        return {
            "positive_depth_ratio": ratio,
            "depth_p10": None,
            "depth_p50": None,
            "depth_p90": None,
            "depth_iqr_ratio": None,
            "depth_entropy": 0.0,
        }
    p10, p25, p50, p75, p90 = (_percentile(positive, q) for q in (0.1, 0.25, 0.5, 0.75, 0.9))
    logs = [math.log(v) for v in positive]
    lo, hi = min(logs), max(logs)
    entropy = 0.0
    if hi > lo:
        bins = [0] * 8
        for value in logs:
            bins[min(7, int((value - lo) / (hi - lo) * 8))] += 1
        n = len(logs)
        entropy = -sum((count / n) * math.log(count / n) for count in bins if count) / math.log(8)
    return {
        "positive_depth_ratio": ratio,
        "depth_p10": p10,
        "depth_p50": p50,
        "depth_p90": p90,
        "depth_iqr_ratio": (p75 - p25) / max(abs(p50), 1e-12),
        "depth_entropy": entropy,
    }


def _valid_camera_points(points_camera: Sequence[Sequence[float]]):
    np = _np()
    pts = np.asarray(points_camera, dtype=float).reshape(-1, 3)
    if not len(pts):
        return pts
    finite = np.all(np.isfinite(pts), axis=1)
    nonzero = np.linalg.norm(pts, axis=1) > 1e-12
    return pts[finite & nonzero]


def bearing_fisher_information(
    points_camera: Sequence[Sequence[float]],
    *,
    noise_sigma_rad: float = 0.002,
    characteristic_length: float | None = None,
):
    np = _np()
    pts = _valid_camera_points(points_camera)
    if characteristic_length is None:
        radii = np.linalg.norm(pts, axis=1)
        characteristic_length = float(np.median(radii)) if len(radii) else 1.0
    if not math.isfinite(characteristic_length) or characteristic_length <= 0:
        raise ValueError("characteristic_length must be finite and > 0")
    if not math.isfinite(noise_sigma_rad) or noise_sigma_rad <= 0:
        raise ValueError("noise_sigma_rad must be finite and > 0")
    scale = np.diag([1 / characteristic_length] * 3 + [1.0] * 3)
    fim = np.zeros((6, 6), dtype=float)
    inv_var = 1.0 / noise_sigma_rad**2
    for point in pts:
        radius = float(np.linalg.norm(point))
        bearing = point / radius
        proj = (np.eye(3) - np.outer(bearing, bearing)) / radius
        skew = np.array(
            [
                [0.0, -point[2], point[1]],
                [point[2], 0.0, -point[0]],
                [-point[1], point[0], 0.0],
            ],
            dtype=float,
        )
        jacobian = proj @ np.concatenate((-np.eye(3), skew), axis=1) @ scale
        fim += inv_var * (jacobian.T @ jacobian)
    return fim


def fim_summary(fim, *, regularization: float = 1e-9) -> dict[str, Any]:
    np = _np()
    matrix = np.asarray(fim, dtype=float).reshape(6, 6)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("FIM must be finite")
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and > 0")
    vals, vecs = np.linalg.eigh((matrix + matrix.T) / 2)
    vals = np.maximum(vals, 0.0)
    vmax = float(vals[-1]) if len(vals) else 0.0
    tol = max(vmax * 1e-8, 1e-12)
    rank = int(np.sum(vals > tol))
    lmin = float(vals[0])
    # Use a rank-tolerance-capped effective condition number rather than inf.
    # Rank and lambda_min still preserve exact degeneracy information, while
    # this finite proxy remains valid in strict JSON and usable by thresholds.
    # An unobservable pose is not well-conditioned.
    if rank == 0 or vmax <= tol:
        condition = 1e12
    else:
        condition = max(1.0, float(vmax / max(lmin, tol)))
    positive = vals[vals > tol]
    if len(positive):
        probs = positive / positive.sum()
        effective_rank = float(math.exp(-float(np.sum(probs * np.log(probs)))))
    else:
        effective_rank = 0.0
    reg = matrix + np.eye(6) * regularization
    sign, logdet = np.linalg.slogdet(reg)
    if sign <= 0 or not math.isfinite(float(logdet)):
        raise ValueError("regularized FIM log-determinant is not finite")
    inv = np.linalg.pinv(reg)
    weakest = vecs[:, 0]
    labels = ("tx", "ty", "tz", "rx", "ry", "rz")
    mode = labels[int(np.argmax(np.abs(weakest)))]
    return {
        "fim_rank": rank,
        "fim_effective_rank": effective_rank,
        "fim_lambda_min": lmin,
        "fim_logdet": float(logdet),
        "fim_condition_number": condition,
        "fim_trace_inverse": float(np.trace(inv)),
        "fim_weakest_mode": {"dominant_dof": mode, "vector": [float(x) for x in weakest]},
        "crlb_translation_trace": float(np.trace(inv[:3, :3])),
        "crlb_rotation_trace": float(np.trace(inv[3:, 3:])),
    }


def compute_metric_bundle(payload: Mapping[str, Any]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    if "image_points" in payload and "image_size" in payload:
        width, height = payload["image_size"]
        points = payload["image_points"]
        count, ratio = grid_occupancy(points, width, height)
        metrics.update(
            {
                "inlier_convex_hull_coverage": MetricValue(
                    convex_hull_area_fraction(points, width, height),
                    Availability.DERIVED,
                    source="mapdoctor.metrics",
                ),
                "grid_occupancy_count": MetricValue(
                    count,
                    Availability.DERIVED,
                    source="mapdoctor.metrics",
                ),
                "grid_occupancy_ratio": MetricValue(
                    ratio,
                    Availability.DERIVED,
                    source="mapdoctor.metrics",
                ),
                "image_spatial_entropy": MetricValue(
                    image_spatial_entropy(points, width, height),
                    Availability.DERIVED,
                    source="recapture.compute",
                ),
            }
        )
    if "depths" in payload:
        for name, value in depth_summary(payload["depths"]).items():
            if value is None:
                metrics[name] = MetricValue(
                    status=Availability.UNAVAILABLE,
                    reason="no positive finite depths",
                    source="recapture.compute",
                )
            else:
                metrics[name] = MetricValue(value, Availability.DERIVED, source="recapture.compute")
    if "points_camera" in payload:
        valid_points = _valid_camera_points(payload["points_camera"])
        fim = bearing_fisher_information(
            valid_points,
            noise_sigma_rad=float(payload.get("noise_sigma_rad", 0.002)),
            characteristic_length=(
                None
                if payload.get("characteristic_length") is None
                else float(payload["characteristic_length"])
            ),
        )
        for name, value in fim_summary(fim).items():
            metrics[name] = MetricValue(value, Availability.DERIVED, source="recapture.compute")
        metrics["visible_landmark_count"] = MetricValue(
            int(len(valid_points)),
            Availability.DERIVED,
            source="recapture.compute",
        )
    return metrics
