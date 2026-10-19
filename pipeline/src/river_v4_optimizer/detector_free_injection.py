from __future__ import annotations

import math
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]

from .runner import atomic_json
from .rescue import maximum_triangulation_angle_deg


@dataclass(frozen=True)
class DetectorFreeMatch:
    image_i: str
    xy_i: tuple[float, float]
    image_j: str
    xy_j: tuple[float, float]


@dataclass(frozen=True)
class DetectorFreeInjectionConfig:
    required_videos: tuple[str, ...]
    anchor_radius_px: float = 2.0
    maximum_anchor_spread_px: float = 3.0
    minimum_distinct_videos: int = 3
    minimum_track_length: int = 3
    maximum_reprojection_error_px: float = 2.0
    maximum_reprojection_p90_px: float = 1.5
    minimum_triangulation_angle_deg: float = 1.0
    existing_observation_conflict_radius_px: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum_track_length < 3:
            raise ValueError("detector-free minimum track length must be at least three")


@dataclass(frozen=True)
class PlannedDetectorFreeTrack:
    xyz: tuple[float, float, float]
    observations: tuple[tuple[str, tuple[float, float]], ...]
    reprojection_errors_px: tuple[float, ...]
    maximum_triangulation_angle_deg: float


def validate_planned_track(
    plan: PlannedDetectorFreeTrack,
    config: DetectorFreeInjectionConfig,
) -> str | None:
    if len(plan.observations) < config.minimum_track_length:
        return "minimum_track_length"
    image_names = [str(name) for name, _ in plan.observations]
    if len(image_names) != len(set(image_names)):
        return "duplicate_image"
    videos = {name.split("/", 1)[0] for name in image_names}
    if not set(config.required_videos) <= videos:
        return "required_video_span"
    if len(videos) < config.minimum_distinct_videos:
        return "minimum_distinct_videos"
    errors = np.asarray(plan.reprojection_errors_px, dtype=np.float64)
    if len(errors) != len(plan.observations) or not np.isfinite(errors).all():
        return "invalid_reprojection_errors"
    if (
        float(errors.max()) > config.maximum_reprojection_error_px
        or float(np.percentile(errors, 90)) > config.maximum_reprojection_p90_px
    ):
        return "reprojection_error"
    if plan.maximum_triangulation_angle_deg < config.minimum_triangulation_angle_deg:
        return "triangulation_angle"
    return None


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def cluster_detector_free_matches(
    matches: Iterable[DetectorFreeMatch],
    *,
    anchor_radius_px: float,
    maximum_anchor_spread_px: float,
) -> tuple[list[dict[str, tuple[float, float]]], int]:
    """Cluster pairwise detector-free matches into coherent image tracks."""

    if anchor_radius_px <= 0 or maximum_anchor_spread_px <= 0:
        raise ValueError("anchor radius and maximum spread must be positive")
    rows = list(matches)
    occurrences: list[tuple[str, np.ndarray]] = []
    pair_edges: list[tuple[int, int]] = []
    for row in rows:
        if row.image_i == row.image_j:
            raise ValueError("detector-free self match is invalid")
        first = len(occurrences)
        occurrences.append((row.image_i, np.asarray(row.xy_i, dtype=np.float64)))
        second = len(occurrences)
        occurrences.append((row.image_j, np.asarray(row.xy_j, dtype=np.float64)))
        pair_edges.append((first, second))
    if not occurrences:
        return [], 0

    groups: dict[str, list[int]] = defaultdict(list)
    for index, (image, _) in enumerate(occurrences):
        groups[image].append(index)
    disjoint = _DisjointSet(len(occurrences))
    for left, right in pair_edges:
        disjoint.union(left, right)
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        points = np.asarray([occurrences[index][1] for index in indexes])
        for left, right in cKDTree(points).query_pairs(anchor_radius_px):
            disjoint.union(indexes[left], indexes[right])

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(occurrences)):
        components[disjoint.find(index)].append(index)
    tracks: list[dict[str, tuple[float, float]]] = []
    rejected_spread = 0
    for indexes in components.values():
        observations: dict[str, list[np.ndarray]] = defaultdict(list)
        for index in indexes:
            image, xy = occurrences[index]
            observations[image].append(xy)
        if len(observations) < 2:
            continue
        collapsed: dict[str, tuple[float, float]] = {}
        coherent = True
        for image, values in observations.items():
            points = np.asarray(values, dtype=np.float64)
            center = np.median(points, axis=0)
            if float(np.max(np.linalg.norm(points - center, axis=1))) > maximum_anchor_spread_px:
                coherent = False
                break
            collapsed[image] = (float(center[0]), float(center[1]))
        if not coherent:
            rejected_spread += 1
            continue
        tracks.append(dict(sorted(collapsed.items())))
    tracks.sort(key=lambda track: tuple(track.items()))
    return tracks, rejected_spread


def _matches_from_geometry(rows: Iterable[dict[str, Any]]) -> list[DetectorFreeMatch]:
    matches = []
    for row in rows:
        if str(row.get("admission") or "") != "VERIFIED":
            continue
        artifact = Path(str(row.get("match_artifact") or "")).resolve(strict=True)
        arrays = np.load(artifact)
        if not {"points_i", "points_j", "essential_mask"} <= set(arrays.files):
            raise ValueError(f"detector-free match artifact has an incomplete schema: {artifact}")
        mask = np.asarray(arrays["essential_mask"]).astype(bool)
        points_i = np.asarray(arrays["points_i"], dtype=np.float64)[mask]
        points_j = np.asarray(arrays["points_j"], dtype=np.float64)[mask]
        if len(points_i) != len(points_j):
            raise ValueError(f"detector-free match lengths differ: {artifact}")
        for left, right in zip(points_i, points_j, strict=True):
            matches.append(
                DetectorFreeMatch(
                    str(row["image_i"]),
                    (float(left[0]), float(left[1])),
                    str(row["image_j"]),
                    (float(right[0]), float(right[1])),
                )
            )
    return matches


def _camera_center(image: Any) -> np.ndarray:
    transform = image.cam_from_world()
    rotation = transform.rotation.matrix()
    return -rotation.T @ np.asarray(transform.translation, dtype=np.float64)


def _has_existing_observation_conflict(image: Any, xy: np.ndarray, radius_px: float) -> bool:
    if radius_px <= 0 or image.num_points2D() == 0:
        return False
    points = np.asarray([point.xy for point in image.points2D], dtype=np.float64)
    distance, index = cKDTree(points).query(xy)
    return bool(distance <= radius_px and image.points2D[int(index)].has_point3D())


def plan_detector_free_tracks(
    *,
    model: Path,
    keyframes: dict[str, dict[str, Any]],
    seed_geometry: Sequence[dict[str, Any]],
    support_geometry: Sequence[dict[str, Any]],
    selected_keyframes: set[str],
    config: DetectorFreeInjectionConfig,
) -> tuple[list[PlannedDetectorFreeTrack], dict[str, Any]]:
    """Plan strict three-video EDM tracks in a frozen COLMAP pose gauge."""

    import pycolmap

    if len(set(config.required_videos)) != 2:
        raise ValueError("detector-free injection requires two distinct required videos")
    seed = [row for row in seed_geometry if str(row.get("admission") or "") == "VERIFIED"]
    if not seed:
        raise RuntimeError("no VERIFIED detector-free seed pairs")
    seed_endpoints = {str(row[key]) for row in seed for key in ("image_i", "image_j")}
    support = [
        row
        for row in support_geometry
        if str(row.get("admission") or "") == "VERIFIED"
        and str(row.get("image_i")) in selected_keyframes
        and str(row.get("image_j")) in selected_keyframes
        and (str(row.get("image_i")) in seed_endpoints or str(row.get("image_j")) in seed_endpoints)
    ]
    geometry_by_pair = {
        tuple(sorted((str(row["image_i"]), str(row["image_j"])))): row for row in (*support, *seed)
    }
    matches = _matches_from_geometry(geometry_by_pair.values())
    clustered, rejected_spread = cluster_detector_free_matches(
        matches,
        anchor_radius_px=config.anchor_radius_px,
        maximum_anchor_spread_px=config.maximum_anchor_spread_px,
    )
    reconstruction = pycolmap.Reconstruction(str(model.resolve(strict=True)))
    images = {str(image.name): image for image in reconstruction.images.values() if image.has_pose}
    rejection: Counter[str] = Counter()
    plans: list[PlannedDetectorFreeTrack] = []
    signatures: set[tuple[tuple[str, tuple[float, float]], ...]] = set()
    for track in clustered:
        videos = {name.split(":", 1)[0] for name in track}
        if (
            not set(config.required_videos) <= videos
            or len(videos) < config.minimum_distinct_videos
        ):
            rejection["required_video_span"] += 1
            continue
        entries = []
        missing_image = False
        for keyframe_id, xy in sorted(track.items()):
            record = keyframes.get(keyframe_id)
            output_name = "" if record is None else str(record.get("output_name") or "")
            image = images.get(output_name)
            if image is None:
                missing_image = True
                break
            entries.append((keyframe_id, output_name, np.asarray(xy, dtype=np.float64), image))
        if missing_image or len(entries) < config.minimum_track_length:
            rejection["missing_registered_image"] += 1
            continue
        options = pycolmap.EstimateTriangulationOptions()
        options.min_tri_angle = math.radians(config.minimum_triangulation_angle_deg)
        options.ransac.max_error = math.radians(config.maximum_reprojection_error_px)
        options.ransac.random_seed = 0
        result = pycolmap.estimate_triangulation(
            np.asarray([entry[2] for entry in entries]),
            [entry[3].cam_from_world() for entry in entries],
            [entry[3].camera for entry in entries],
            options,
        )
        if result is None:
            rejection["triangulation_failed"] += 1
            continue
        inliers = np.asarray(result["inliers"]).astype(bool)
        inlier_entries = [entry for entry, keep in zip(entries, inliers, strict=True) if keep]
        inlier_videos = {entry[0].split(":", 1)[0] for entry in inlier_entries}
        if (
            len(inlier_entries) < config.minimum_track_length
            or not set(config.required_videos) <= inlier_videos
            or len(inlier_videos) < config.minimum_distinct_videos
        ):
            rejection["inlier_video_span"] += 1
            continue
        xyz = np.asarray(result["xyz"], dtype=np.float64)
        errors = []
        observations = []
        centers = []
        conflict = False
        for _, output_name, xy, image in inlier_entries:
            if _has_existing_observation_conflict(
                image, xy, config.existing_observation_conflict_radius_px
            ):
                conflict = True
                break
            projected = image.project_point(xyz)
            if projected is None:
                conflict = True
                break
            errors.append(float(np.linalg.norm(np.asarray(projected) - xy)))
            observations.append((output_name, (float(xy[0]), float(xy[1]))))
            centers.append(_camera_center(image))
        if conflict:
            rejection["observation_conflict_or_negative_depth"] += 1
            continue
        plan = PlannedDetectorFreeTrack(
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            observations=tuple(observations),
            reprojection_errors_px=tuple(errors),
            maximum_triangulation_angle_deg=maximum_triangulation_angle_deg(
                np.asarray(centers), xyz
            ),
        )
        reason = validate_planned_track(plan, config)
        if reason is not None:
            rejection[reason] += 1
            continue
        signature = tuple(sorted(plan.observations))
        if signature in signatures:
            rejection["duplicate_track"] += 1
            continue
        signatures.add(signature)
        plans.append(plan)
    plans.sort(key=lambda plan: tuple(sorted(plan.observations)))
    receipt = {
        "schema_version": 1,
        "artifact_type": "DETECTOR_FREE_TRACK_INJECTION_PLAN",
        "model": str(model.resolve()),
        "required_videos": list(config.required_videos),
        "selected_images": len(selected_keyframes),
        "seed_verified_pairs": len(seed),
        "support_verified_pairs": len(support),
        "geometry_pairs_loaded": len(geometry_by_pair),
        "essential_match_edges": len(matches),
        "clustered_tracks": len(clustered),
        "rejected_anchor_spread": rejected_spread,
        "rejections": dict(sorted(rejection.items())),
        "approved_tracks": len(plans),
    }
    return plans, receipt


def plan_observation_tracks(
    *,
    model: Path,
    observation_tracks: Sequence[dict[str, tuple[float, float]]],
    config: DetectorFreeInjectionConfig,
) -> tuple[list[PlannedDetectorFreeTrack], dict[str, Any]]:
    """Triangulate already-clustered multi-view observations in a frozen map gauge."""

    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model.resolve(strict=True)))
    images = {str(image.name): image for image in reconstruction.images.values() if image.has_pose}
    rejection: Counter[str] = Counter()
    plans: list[PlannedDetectorFreeTrack] = []
    signatures: set[tuple[tuple[str, tuple[float, float]], ...]] = set()
    for raw_track in observation_tracks:
        track = dict(sorted(raw_track.items()))
        if len(track) < config.minimum_track_length or len(track) != len(set(track)):
            rejection["minimum_track_length_or_duplicate_image"] += 1
            continue
        entries = []
        for name, xy in track.items():
            image = images.get(name)
            point = np.asarray(xy, dtype=np.float64)
            if image is None or point.shape != (2,) or not np.isfinite(point).all():
                entries = []
                break
            entries.append((name, point, image))
        if len(entries) < config.minimum_track_length:
            rejection["missing_registered_image_or_invalid_point"] += 1
            continue
        options = pycolmap.EstimateTriangulationOptions()
        options.min_tri_angle = math.radians(config.minimum_triangulation_angle_deg)
        options.ransac.max_error = math.radians(config.maximum_reprojection_error_px)
        options.ransac.random_seed = 0
        result = pycolmap.estimate_triangulation(
            np.asarray([entry[1] for entry in entries]),
            [entry[2].cam_from_world() for entry in entries],
            [entry[2].camera for entry in entries],
            options,
        )
        if result is None:
            rejection["triangulation_failed"] += 1
            continue
        inliers = np.asarray(result["inliers"], dtype=bool)
        selected = [entry for entry, keep in zip(entries, inliers, strict=True) if keep]
        if len(selected) < config.minimum_track_length:
            rejection["minimum_inlier_views"] += 1
            continue
        xyz = np.asarray(result["xyz"], dtype=np.float64)
        errors = []
        observations = []
        centers = []
        conflict = False
        for name, xy, image in selected:
            if _has_existing_observation_conflict(
                image, xy, config.existing_observation_conflict_radius_px
            ):
                conflict = True
                break
            projected = image.project_point(xyz)
            if projected is None:
                conflict = True
                break
            errors.append(float(np.linalg.norm(np.asarray(projected) - xy)))
            observations.append((name, (float(xy[0]), float(xy[1]))))
            centers.append(_camera_center(image))
        if conflict:
            rejection["observation_conflict_or_negative_depth"] += 1
            continue
        plan = PlannedDetectorFreeTrack(
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            observations=tuple(observations),
            reprojection_errors_px=tuple(errors),
            maximum_triangulation_angle_deg=maximum_triangulation_angle_deg(
                np.asarray(centers), xyz
            ),
        )
        reason = validate_planned_track(plan, config)
        if reason is not None:
            rejection[reason] += 1
            continue
        signature = tuple(sorted(plan.observations))
        if signature in signatures:
            rejection["duplicate_track"] += 1
            continue
        signatures.add(signature)
        plans.append(plan)
    plans.sort(key=lambda plan: tuple(sorted(plan.observations)))
    return plans, {
        "schema_version": 1,
        "artifact_type": "MULTIVIEW_OBSERVATION_TRACK_PLAN",
        "model": str(model.resolve()),
        "candidate_tracks": len(observation_tracks),
        "approved_tracks": len(plans),
        "rejections": dict(sorted(rejection.items())),
    }


def _point_snapshot(reconstruction: Any) -> dict[int, tuple[Any, ...]]:
    return {
        int(point_id): (
            tuple(float(value) for value in point.xyz),
            tuple(int(value) for value in point.color),
            float(point.error),
            tuple(
                (int(element.image_id), int(element.point2D_idx))
                for element in point.track.elements
            ),
        )
        for point_id, point in reconstruction.points3D.items()
    }


def inject_planned_tracks(
    input_model: Path,
    output_model: Path,
    plans: Sequence[PlannedDetectorFreeTrack],
) -> dict[str, Any]:
    """Append already-validated detector-free tracks without changing old geometry."""

    if not plans:
        raise RuntimeError("no detector-free tracks were approved for injection")
    import pycolmap

    source = input_model.resolve(strict=True)
    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(source))
    images = {str(image.name): image for image in reconstruction.images.values() if image.has_pose}
    points_before = reconstruction.num_points3D()
    observations_before = int(reconstruction.compute_num_observations())
    old_points = _point_snapshot(reconstruction)
    added_observations = 0
    added_point_ids = []
    for plan in plans:
        if len(plan.observations) < 3:
            raise ValueError("detector-free injection requires at least three observations")
        names = [str(name) for name, _ in plan.observations]
        if len(names) != len(set(names)):
            raise ValueError("detector-free track contains duplicate image observations")
        track = pycolmap.Track()
        for name, xy in plan.observations:
            if name not in images:
                raise ValueError(f"detector-free observation image is not registered: {name}")
            image = images[name]
            point2d_idx = image.num_points2D()
            image.points2D.append(pycolmap.Point2D(np.asarray(xy, dtype=np.float64)))
            track.add_element(int(image.image_id), int(point2d_idx))
            added_observations += 1
        added_point_ids.append(
            int(
                reconstruction.add_point3D(
                    np.asarray(plan.xyz, dtype=np.float64),
                    track,
                    np.zeros(3, dtype=np.uint8),  # ty: ignore[invalid-argument-type]
                )
            )
        )
    current = _point_snapshot(reconstruction)
    if any(current.get(point_id) != snapshot for point_id, snapshot in old_points.items()):
        raise RuntimeError("detector-free injection changed existing point geometry")
    output.mkdir(parents=True)
    reconstruction.write(output)
    written = pycolmap.Reconstruction(str(output))
    if written.num_points3D() != points_before + len(plans):
        raise RuntimeError("detector-free point count does not match the approved plan")
    if int(written.compute_num_observations()) != observations_before + added_observations:
        raise RuntimeError("detector-free observation count does not match the approved plan")
    receipt = {
        "schema_version": 1,
        "artifact_type": "DETECTOR_FREE_TRACK_INJECTION",
        "input_model": str(source),
        "output_model": str(output),
        "added_points3D": len(plans),
        "added_observations": added_observations,
        "added_point_ids": added_point_ids,
        "existing_geometry_unchanged": True,
        "points3D_before": points_before,
        "points3D_after": written.num_points3D(),
        "observations_before": observations_before,
        "observations_after": int(written.compute_num_observations()),
    }
    atomic_json(output.parent / "detector_free_injection.json", receipt)
    return receipt


__all__ = [
    "DetectorFreeInjectionConfig",
    "DetectorFreeMatch",
    "PlannedDetectorFreeTrack",
    "cluster_detector_free_matches",
    "inject_planned_tracks",
    "plan_detector_free_tracks",
    "plan_observation_tracks",
    "validate_planned_track",
]
