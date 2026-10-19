from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from mapdoctor.config import HealthThresholds
from mapdoctor.model import ImageRecord, MapModel


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    twice_area = sum(
        poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
        for i in range(len(poly))
    )
    return abs(twice_area) / 2.0


def _valid_image_points(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    valid: list[tuple[float, float]] = []
    for point in points:
        if len(point) < 2:
            continue
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if 0.0 <= x <= width and 0.0 <= y <= height:
            valid.append((x, y))
    return valid


def convex_hull_area_fraction(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
) -> float:
    """Return image-normalized convex-hull coverage for valid 2D points.

    This is the canonical implementation shared by static map health and the
    targeted-recapture pipeline so the same evidence is not computed with two
    subtly different formulas. Non-finite and out-of-image coordinates are
    ignored rather than contaminating the hull with NaN/Inf.
    """
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        return 0.0
    valid = _valid_image_points(points, width, height)
    if len(valid) < 3:
        return 0.0
    return min(1.0, _polygon_area(_convex_hull(valid)) / float(width * height))


def grid_coverage(
    points: Sequence[Sequence[float]],
    width: float,
    height: float,
    *,
    rows: int = 4,
    cols: int = 4,
) -> tuple[int, float]:
    """Return occupied image-grid cells and normalized occupancy."""
    if (
        not math.isfinite(width)
        or not math.isfinite(height)
        or width <= 0
        or height <= 0
        or rows <= 0
        or cols <= 0
    ):
        return 0, 0.0
    occupied: set[tuple[int, int]] = set()
    for x, y in _valid_image_points(points, width, height):
        gx = min(cols - 1, max(0, int((x / width) * cols)))
        gy = min(rows - 1, max(0, int((y / height) * rows)))
        occupied.add((gx, gy))
    return len(occupied), len(occupied) / float(rows * cols)


def image_coverage(image: ImageRecord, width: int, height: int, grid: int = 4) -> tuple[float, float, int, int]:
    valid = [(obs.x, obs.y) for obs in image.observations if obs.point3d_id is not None]
    if not valid or width <= 0 or height <= 0:
        return 0.0, 0.0, len(valid), 0
    hull_ratio = convex_hull_area_fraction(valid, width, height)
    occupied_cells, grid_ratio = grid_coverage(valid, width, height, rows=grid, cols=grid)
    return hull_ratio, grid_ratio, len(valid), occupied_cells


class _UnionFind:
    def __init__(self, ids: Iterable[int]):
        self.parent = {item: item for item in ids}

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _largest_component_ratio(model: MapModel) -> float:
    if not model.images:
        return 0.0
    union_find = _UnionFind(model.images.keys())
    for point in model.points3d.values():
        image_ids = [element.image_id for element in point.track if element.image_id in model.images]
        if len(image_ids) >= 2:
            for other in image_ids[1:]:
                union_find.union(image_ids[0], other)
    counts: dict[int, int] = {}
    for image_id in model.images:
        root = union_find.find(image_id)
        counts[root] = counts.get(root, 0) + 1
    return max(counts.values()) / len(model.images)


def _nearest_baselines(model: MapModel) -> list[float]:
    centers = [image.center for image in model.images.values()]
    if len(centers) < 2:
        return []
    output: list[float] = []
    for i, a in enumerate(centers):
        best = math.inf
        for j, b in enumerate(centers):
            if i == j:
                continue
            distance = math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))
            best = min(best, distance)
        if math.isfinite(best):
            output.append(best)
    return output


def _view_direction_diversity(model: MapModel) -> float:
    directions = [image.viewing_direction for image in model.images.values()]
    if len(directions) < 2:
        return 0.0
    mean = tuple(sum(vector[axis] for vector in directions) / len(directions) for axis in range(3))
    resultant = min(1.0, math.sqrt(sum(value * value for value in mean)))
    return 1.0 - resultant


def _recapture_action(reasons: list[str]) -> str:
    actions: list[str] = []
    if "few_3d_observations" in reasons:
        actions.append("increase overlap and texture support around this viewpoint")
    if "low_spatial_hull_coverage" in reasons:
        actions.append("capture the scene with correspondences spread across a wider image area")
    if "low_4x4_grid_coverage" in reasons:
        actions.append("add a laterally offset view so landmarks occupy more image-grid cells")
    return "; ".join(actions)


def _producer_provenance(model: MapModel) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "adapter": model.metadata.get("adapter"),
        "producer": model.metadata.get("producer"),
    }
    if "gluemap_provenance" in model.metadata:
        provenance["gluemap"] = model.metadata["gluemap_provenance"]
    return {key: value for key, value in provenance.items() if value is not None}


@dataclass
class HealthMetrics:
    source: str
    format: str
    registered_images: int
    cameras: int
    points3d: int
    observations: int
    observations_per_image_median: float | None
    observations_per_image_p10: float | None
    track_length_median: float | None
    track_length_p10: float | None
    reprojection_error_median_px: float | None
    reprojection_error_p90_px: float | None
    reprojection_error_p95_px: float | None
    hull_coverage_median: float | None
    hull_coverage_p10: float | None
    grid4_coverage_median: float | None
    grid4_coverage_p10: float | None
    largest_covisibility_component_ratio: float
    nearest_camera_baseline_median: float | None
    view_direction_diversity: float
    weak_images: list[dict]
    recapture_suggestions: list[dict]
    producer_provenance: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def analyze(model: MapModel, thresholds: HealthThresholds | None = None) -> HealthMetrics:
    thresholds = thresholds or HealthThresholds()
    obs_counts: list[float] = []
    hull: list[float] = []
    grid: list[float] = []
    weak_images: list[dict] = []
    recapture_suggestions: list[dict] = []

    for image in model.images.values():
        camera = model.cameras.get(image.camera_id)
        if camera is None:
            hull_ratio, grid_ratio, observations, occupied_cells = 0.0, 0.0, 0, 0
            reasons = ["missing_camera"]
        else:
            hull_ratio, grid_ratio, observations, occupied_cells = image_coverage(
                image, camera.width, camera.height
            )
            reasons = []
            if observations < thresholds.min_observations_per_image:
                reasons.append("few_3d_observations")
            if hull_ratio < thresholds.min_hull_coverage:
                reasons.append("low_spatial_hull_coverage")
            if occupied_cells < thresholds.min_grid4_occupancy:
                reasons.append("low_4x4_grid_coverage")
        obs_counts.append(float(observations))
        hull.append(hull_ratio)
        grid.append(grid_ratio)
        if reasons:
            weak = {
                "image_id": image.id,
                "name": image.name,
                "reasons": reasons,
                "observations": observations,
                "hull_coverage": round(hull_ratio, 4),
                "grid4_occupancy": occupied_cells,
            }
            weak_images.append(weak)
            recapture_suggestions.append(
                {
                    "reference_image": image.name,
                    "camera_center": list(image.center),
                    "viewing_direction": list(image.viewing_direction),
                    "reasons": reasons,
                    "suggested_action": _recapture_action(reasons),
                }
            )

    track_lengths = [float(len(point.track)) for point in model.points3d.values() if point.track]
    errors = [
        float(point.error)
        for point in model.points3d.values()
        if point.error >= 0 and math.isfinite(point.error)
    ]
    baselines = _nearest_baselines(model)

    return HealthMetrics(
        source=model.source,
        format=model.format,
        registered_images=len(model.images),
        cameras=len(model.cameras),
        points3d=len(model.points3d),
        observations=int(sum(obs_counts)),
        observations_per_image_median=median(obs_counts),
        observations_per_image_p10=percentile(obs_counts, 0.10),
        track_length_median=median(track_lengths),
        track_length_p10=percentile(track_lengths, 0.10),
        reprojection_error_median_px=median(errors),
        reprojection_error_p90_px=percentile(errors, 0.90),
        reprojection_error_p95_px=percentile(errors, 0.95),
        hull_coverage_median=median(hull),
        hull_coverage_p10=percentile(hull, 0.10),
        grid4_coverage_median=median(grid),
        grid4_coverage_p10=percentile(grid, 0.10),
        largest_covisibility_component_ratio=_largest_component_ratio(model),
        nearest_camera_baseline_median=median(baselines),
        view_direction_diversity=_view_direction_diversity(model),
        weak_images=sorted(weak_images, key=lambda item: (-len(item["reasons"]), item["observations"])),
        recapture_suggestions=recapture_suggestions,
        producer_provenance=_producer_provenance(model),
    )
