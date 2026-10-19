"""Append existing B0 observations, then triangulate only connected fringe tracks.

B0 cameras, intrinsics, and original Point3D XYZ stay frozen.  Historical images may
enter the child map only after a DIRECT_STRONG registration and a stable mask.
New points are admitted only when a track includes at least one frozen B0 view.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from river_map_quality.historical_experiment import (
    HistoricalExperimentError,
    assert_b0_unchanged,
    resolve_b0_model_root,
)
from river_map_quality.historical_inputs import DIRECT_STRONG, STABLE
from river_map_quality.official_edm_adapter import LiftedMatch, UnmappedPair
from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import write_new_json

APPEND_MAP_SCHEMA = "RIVER_EXISTING_POINT3D_APPEND_MAP_V1"
FRINGE_MAP_SCHEMA = "RIVER_CONNECTED_FRINGE_TRIANGULATION_V1"
APPEND_RELATIVE = "maps/map_observation_append"
FRINGE_RELATIVE = "maps/map_connected_fringe"
DEFAULT_REPROJECTION_MAX_PX = 3.0
DEFAULT_MIN_TRACK_VIEWS = 3
DEFAULT_QUANTIZE_PX = 2.0


class MapExtendError(HistoricalExperimentError):
    """Raised when observation append or fringe triangulation would mutate B0."""


@dataclass(frozen=True)
class AppendObservation:
    image_name: str
    point3d_id: int
    query_xy: tuple[float, float]
    reference_name: str
    confidence: float


@dataclass(frozen=True)
class FringeView:
    image_name: str
    xy: tuple[float, float]
    is_b0: bool
    pose: np.ndarray


def _make_tree_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        path.chmod(path.stat().st_mode | stat.S_IWUSR | (stat.S_IXUSR if path.is_dir() else 0))
    root.chmod(root.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)


def _as_pose(pose: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise MapExtendError("historical pose must be a finite 4x4 world-to-camera matrix")
    return matrix


def _rigid_from_pose(pose: np.ndarray):
    import pycolmap

    return pycolmap.Rigid3d(np.ascontiguousarray(pose[:3, :], dtype=np.float64))


def _pose_matrix_of_image(image: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
    return matrix


def _protected_b0_geometry(
    reconstruction: Any,
    original_image_ids: set[int],
    original_point_ids: set[int],
) -> dict[str, object]:
    poses = {
        int(image_id): _pose_matrix_of_image(reconstruction.image(image_id)).tolist()
        for image_id in sorted(original_image_ids)
    }
    xyz = {
        int(point_id): np.asarray(reconstruction.point3D(point_id).xyz, dtype=np.float64).tolist()
        for point_id in original_point_ids
    }
    return {"poses": poses, "xyz": xyz}


def _assert_protected_geometry(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    allowed_new_point_ids: set[int] | None = None,
) -> None:
    if before["poses"] != after["poses"]:
        raise MapExtendError("extend changed a frozen B0 camera pose")
    before_xyz = dict(before["xyz"])
    after_xyz = dict(after["xyz"])
    extra = set(after_xyz) - set(before_xyz)
    missing = set(before_xyz) - set(after_xyz)
    if missing:
        raise MapExtendError(f"extend deleted frozen Point3Ds: {sorted(missing)[:8]!r}")
    if allowed_new_point_ids is None and extra:
        raise MapExtendError(f"append created unexpected Point3Ds: {sorted(extra)[:8]!r}")
    if allowed_new_point_ids is not None and extra - allowed_new_point_ids:
        raise MapExtendError("fringe created Point3Ds outside the admitted track set")
    for point_id, xyz in before_xyz.items():
        if after_xyz[point_id] != xyz:
            raise MapExtendError(f"extend moved frozen Point3D {point_id}")


def select_append_rows(
    registrations: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Keep only stable DIRECT_STRONG historical cameras."""

    best: dict[str, dict[str, Any]] = {}
    for row in registrations:
        name = str(row["query_name"])
        if str(row.get("status")) != DIRECT_STRONG:
            continue
        if str(labels.get(name, "")) != STABLE:
            continue
        if row.get("pose") is None:
            raise MapExtendError(f"DIRECT_STRONG row is missing a pose: {name}")
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        score = int(metrics.get("inlier_count") or 0)
        current = best.get(name)
        current_score = int((current or {}).get("metrics", {}).get("inlier_count") or 0)
        if current is None or score > current_score:
            best[name] = dict(row)
    return tuple(best[name] for name in sorted(best))



def observations_from_row(row: Mapping[str, Any]) -> tuple[AppendObservation, ...]:
    """Read persisted PnP inliers; refuse to invent 2D locations."""

    payload = row.get("inlier_observations")
    if not isinstance(payload, list) or not payload:
        raise MapExtendError(
            f"{row.get('query_name')} has no persisted inlier observations; relift first"
        )
    observations: list[AppendObservation] = []
    seen_points: set[int] = set()
    for item in payload:
        point_id = int(item["point3d_id"])
        if point_id in seen_points:
            continue
        seen_points.add(point_id)
        xy = item["query_xy"]
        observations.append(
            AppendObservation(
                image_name=str(row["query_name"]),
                point3d_id=point_id,
                query_xy=(float(xy[0]), float(xy[1])),
                reference_name=str(item["reference_name"]),
                confidence=float(item.get("confidence") or 0.0),
            )
        )
    return tuple(observations)


def unmapped_pairs_from_row(row: Mapping[str, Any]) -> tuple[UnmappedPair, ...]:
    payload = row.get("unmapped_pairs")
    if not isinstance(payload, list):
        return ()
    return tuple(
        UnmappedPair(
            query_xy=(float(item["query_xy"][0]), float(item["query_xy"][1])),
            reference_xy=(float(item["reference_xy"][0]), float(item["reference_xy"][1])),
            reference_name=str(item["reference_name"]),
            confidence=float(item.get("confidence") or 0.0),
        )
        for item in payload
    )


def _copy_model(source_model: Path, destination: Path) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"extend destination already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model = temporary / "model"
        shutil.copytree(source_model, model, copy_function=shutil.copy)
        _make_tree_writable(model)
        return temporary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def append_existing_observations(
    *,
    source_model: Path,
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
    camera_id: int | None = None,
) -> dict[str, Any]:
    """Copy B0 and append historical cameras observing existing Point3Ds only."""

    import pycolmap

    staged = _copy_model(source_model, destination)
    try:
        model = staged / "model"
        reconstruction = pycolmap.Reconstruction(str(model))
        original_image_ids = set(int(image_id) for image_id in reconstruction.images)
        original_point_ids = set(int(point_id) for point_id in reconstruction.point3D_ids())
        before = _protected_b0_geometry(reconstruction, original_image_ids, original_point_ids)
        if camera_id is None:
            if len(reconstruction.cameras) != 1:
                raise MapExtendError("append requires a single shared B0 camera")
            camera_id = next(iter(reconstruction.cameras))
        next_image_id = max(original_image_ids) + 1
        appended: list[dict[str, Any]] = []
        for row in rows:
            observations = observations_from_row(row)
            unknown = [item.point3d_id for item in observations if item.point3d_id not in original_point_ids]
            if unknown:
                raise MapExtendError(f"append referenced unknown Point3Ds: {unknown[:8]!r}")
            keypoints = np.asarray([item.query_xy for item in observations], dtype=np.float64)
            image = pycolmap.Image(
                name=str(row["query_name"]),
                keypoints=keypoints,
                camera_id=int(camera_id),
                image_id=next_image_id,
            )
            reconstruction.add_image_with_trivial_frame(image, _rigid_from_pose(_as_pose(row["pose"])))
            for index, item in enumerate(observations):
                reconstruction.add_observation(
                    item.point3d_id,
                    pycolmap.TrackElement(next_image_id, index),
                )
            appended.append(
                {
                    "query_name": str(row["query_name"]),
                    "image_id": next_image_id,
                    "observation_count": len(observations),
                    "point3d_ids": [item.point3d_id for item in observations],
                }
            )
            next_image_id += 1
        after = _protected_b0_geometry(reconstruction, original_image_ids, original_point_ids)
        _assert_protected_geometry(before, after)
        reconstruction.write(str(model))
        manifest = {
            "schema_version": 1,
            "artifact_type": APPEND_MAP_SCHEMA,
            "source_model": fingerprint_file(source_model / "images.bin", sha256=True).as_dict(),
            "original_image_count": len(original_image_ids),
            "original_point_count": len(original_point_ids),
            "appended_image_count": len(appended),
            "appended_observation_count": int(sum(row["observation_count"] for row in appended)),
            "appended": appended,
            "b0_pose_unchanged": True,
            "b0_xyz_unchanged": True,
        }
        (staged / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, destination)
        return manifest
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[str, tuple[int, int]], tuple[str, tuple[int, int]]] = {}

    def add(self, key: tuple[str, tuple[int, int]]) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: tuple[str, tuple[int, int]]) -> tuple[str, tuple[int, int]]:
        parent = self.parent[key]
        if parent != key:
            parent = self.find(parent)
            self.parent[key] = parent
        return parent

    def union(
        self,
        left: tuple[str, tuple[int, int]],
        right: tuple[str, tuple[int, int]],
    ) -> None:
        self.add(left)
        self.add(right)
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _quantize(xy: tuple[float, float], bin_px: float) -> tuple[int, int]:
    return (int(round(xy[0] / bin_px)), int(round(xy[1] / bin_px)))


def cluster_connected_fringe_tracks(
    rows: Sequence[Mapping[str, Any]],
    *,
    b0_names: set[str],
    poses_by_name: Mapping[str, np.ndarray],
    bin_px: float = DEFAULT_QUANTIZE_PX,
    minimum_views: int = DEFAULT_MIN_TRACK_VIEWS,
) -> tuple[tuple[FringeView, ...], ...]:
    """Group unmapped 2D-2D pairs into tracks that still touch B0."""

    union = _UnionFind()
    views: dict[tuple[str, tuple[int, int]], FringeView] = {}
    for row in rows:
        query_name = str(row["query_name"])
        query_pose = _as_pose(row["pose"])
        for pair in unmapped_pairs_from_row(row):
            if pair.reference_name not in b0_names or pair.reference_name not in poses_by_name:
                continue
            query_key = (query_name, _quantize(pair.query_xy, bin_px))
            reference_key = (pair.reference_name, _quantize(pair.reference_xy, bin_px))
            views[query_key] = FringeView(query_name, pair.query_xy, False, query_pose)
            views[reference_key] = FringeView(
                pair.reference_name,
                pair.reference_xy,
                True,
                np.asarray(poses_by_name[pair.reference_name], dtype=np.float64),
            )
            union.union(query_key, reference_key)
    grouped: dict[tuple[str, tuple[int, int]], list[FringeView]] = defaultdict(list)
    for key, view in views.items():
        grouped[union.find(key)].append(view)
    tracks: list[tuple[FringeView, ...]] = []
    for members in grouped.values():
        by_image: dict[str, FringeView] = {}
        for view in members:
            current = by_image.get(view.image_name)
            if current is None or view.is_b0 and not current.is_b0:
                by_image[view.image_name] = view
        unique = tuple(by_image.values())
        b0_count = sum(1 for view in unique if view.is_b0)
        historical_count = len(unique) - b0_count
        if len(unique) < minimum_views:
            continue
        if b0_count < 1 or historical_count < 2:
            continue
        tracks.append(unique)
    return tuple(tracks)


def _project_reprojection(
    world: np.ndarray,
    pose: np.ndarray,
    camera_params: tuple[float, float, float, float],
    xy: tuple[float, float],
) -> tuple[float, bool]:
    camera_xyz = pose[:3, :3] @ world + pose[:3, 3]
    if camera_xyz[2] <= 1e-9:
        return float("inf"), False
    focal_x, focal_y, principal_x, principal_y = camera_params
    projected = (
        focal_x * camera_xyz[0] / camera_xyz[2] + principal_x,
        focal_y * camera_xyz[1] / camera_xyz[2] + principal_y,
    )
    residual = float(np.hypot(projected[0] - xy[0], projected[1] - xy[1]))
    return residual, True


def triangulate_connected_fringe(
    *,
    source_model: Path,
    destination: Path,
    tracks: Sequence[Sequence[FringeView]],
    camera_params: tuple[float, float, float, float],
    maximum_reprojection_px: float = DEFAULT_REPROJECTION_MAX_PX,
) -> dict[str, Any]:
    """Add only fringe points that reproject cleanly into B0 and historical views."""

    import pycolmap

    staged = _copy_model(source_model, destination)
    try:
        model = staged / "model"
        reconstruction = pycolmap.Reconstruction(str(model))
        original_image_ids = set(int(image_id) for image_id in reconstruction.images)
        original_point_ids = set(int(point_id) for point_id in reconstruction.point3D_ids())
        before = _protected_b0_geometry(reconstruction, original_image_ids, original_point_ids)
        camera = next(iter(reconstruction.cameras.values()))
        images_by_name = {image.name: int(image_id) for image_id, image in reconstruction.images.items()}
        admitted: list[dict[str, Any]] = []
        rejected = 0
        new_ids: set[int] = set()
        for track in tracks:
            points = np.asarray([view.xy for view in track], dtype=np.float64)
            poses = [_rigid_from_pose(view.pose) for view in track]
            cameras = [camera] * len(track)
            estimate = pycolmap.estimate_triangulation(points, poses, cameras)
            if estimate is None:
                rejected += 1
                continue
            xyz = np.asarray(estimate["xyz"], dtype=np.float64).reshape(3)
            inliers = np.asarray(estimate["inliers"], dtype=bool)
            if int(np.count_nonzero(inliers)) < DEFAULT_MIN_TRACK_VIEWS:
                rejected += 1
                continue
            residuals: list[float] = []
            ok = True
            kept: list[FringeView] = []
            for view, keep in zip(track, inliers, strict=True):
                if not keep:
                    continue
                residual, positive = _project_reprojection(xyz, view.pose, camera_params, view.xy)
                if not positive or residual > maximum_reprojection_px:
                    ok = False
                    break
                residuals.append(residual)
                kept.append(view)
            if not ok or not any(view.is_b0 for view in kept):
                rejected += 1
                continue
            point_id = int(reconstruction.add_point3D(xyz, pycolmap.Track()))
            new_ids.add(point_id)
            for view in kept:
                image_id = images_by_name.get(view.image_name)
                if image_id is None:
                    continue
                image = reconstruction.image(image_id)
                point_index = image.num_points2D()
                image.points2D.append(pycolmap.Point2D(xy=np.asarray(view.xy, dtype=np.float64)))
                reconstruction.add_observation(point_id, pycolmap.TrackElement(image_id, point_index))
            admitted.append(
                {
                    "point3d_id": point_id,
                    "xyz": xyz.tolist(),
                    "view_count": int(np.count_nonzero(inliers)),
                    "b0_view_count": int(
                        sum(
                            view.is_b0 and keep
                            for view, keep in zip(track, inliers, strict=True)
                        )
                    ),
                    "reprojection_max_px": float(max(residuals)),
                    "images": [view.image_name for view, keep in zip(track, inliers, strict=True) if keep],
                }
            )
        after = _protected_b0_geometry(reconstruction, original_image_ids, original_point_ids)
        _assert_protected_geometry(before, after, allowed_new_point_ids=new_ids)
        reconstruction.write(str(model))
        manifest = {
            "schema_version": 1,
            "artifact_type": FRINGE_MAP_SCHEMA,
            "source_model": fingerprint_file(source_model / "images.bin", sha256=True).as_dict(),
            "original_point_count": len(original_point_ids),
            "candidate_track_count": len(tracks),
            "admitted_point_count": len(admitted),
            "rejected_track_count": rejected,
            "admitted": admitted,
            "b0_pose_unchanged": True,
            "b0_xyz_unchanged": True,
        }
        (staged / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged, destination)
        return manifest
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def b0_poses_by_name(model: Path) -> dict[str, np.ndarray]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    return {
        image.name: _pose_matrix_of_image(image)
        for image in reconstruction.images.values()
    }


def persist_relifted_row(run_root: Path, row: Mapping[str, Any]) -> Path:
    """Write correspondences beside the immutable registration shard."""

    stem = Path(str(row["query_name"])).stem
    return write_new_json(
        run_root,
        f"historical/observation_append/{stem}.json",
        dict(row),
    )


def overlay_append_catalog(
    catalog: Any,
    rows: Sequence[Mapping[str, Any]],
) -> Any:
    """Return a catalog that can lift through appended historical observations."""

    extra_names: list[str] = []
    extra_descriptors: list[np.ndarray] = []
    extra_xy: list[np.ndarray] = []
    extra_ids: list[np.ndarray] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row["query_name"])
        if name in seen or name in catalog.names:
            continue
        descriptor = row.get("query_descriptor")
        observations = row.get("inlier_observations")
        if descriptor is None or not isinstance(observations, list) or not observations:
            continue
        seen.add(name)
        extra_names.append(name)
        extra_descriptors.append(np.asarray(descriptor, dtype=np.float32).reshape(-1))
        extra_xy.append(
            np.asarray([item["query_xy"] for item in observations], dtype=np.float64)
        )
        extra_ids.append(
            np.asarray([int(item["point3d_id"]) for item in observations], dtype=np.int64)
        )
    if not extra_names:
        return catalog
    new_offsets = list(np.asarray(catalog.observation_offsets, dtype=np.int64))
    all_xy = [np.asarray(catalog.observation_xy, dtype=np.float64)]
    all_ids = [np.asarray(catalog.observation_point3d_ids, dtype=np.int64)]
    for xy, ids in zip(extra_xy, extra_ids, strict=True):
        all_xy.append(xy)
        all_ids.append(ids)
        new_offsets.append(new_offsets[-1] + len(ids))
    descriptors = np.concatenate(
        [np.asarray(catalog.descriptors, dtype=np.float32), np.stack(extra_descriptors)],
        axis=0,
    )
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = np.ascontiguousarray(descriptors / np.clip(norms, 1e-12, None), dtype=np.float32)
    from river_map_quality.megaloc_edm_catalog import FrozenReferenceCatalog

    return FrozenReferenceCatalog(
        root=catalog.root,
        names=tuple(catalog.names) + tuple(extra_names),
        descriptors=descriptors,
        metadata=dict(catalog.metadata),
        observation_offsets=np.asarray(new_offsets, dtype=np.int64),
        observation_xy=np.concatenate(all_xy, axis=0),
        observation_point3d_ids=np.concatenate(all_ids, axis=0),
        point_ids=np.asarray(catalog.point_ids, dtype=np.int64),
        point_xyz=np.asarray(catalog.point_xyz, dtype=np.float64),
    )



def extend_from_registrations(
    *,
    run_root: Path,
    b0_root: Path,
    registrations: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    camera_params: tuple[float, float, float, float],
    b0_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the append child map, then a connected-fringe child if tracks exist."""

    frozen = resolve_b0_model_root(b0_root)
    assert_b0_unchanged(frozen.root, b0_receipt)
    selected = select_append_rows(registrations, labels)
    if not selected:
        raise MapExtendError("no stable DIRECT_STRONG historical cameras are available")
    (run_root / "maps").mkdir(parents=True, exist_ok=True)
    append_destination = run_root / APPEND_RELATIVE
    append_manifest = append_existing_observations(
        source_model=frozen.model,
        destination=append_destination,
        rows=selected,
    )
    assert_b0_unchanged(frozen.root, b0_receipt)
    poses = b0_poses_by_name(frozen.model)
    tracks = cluster_connected_fringe_tracks(
        selected,
        b0_names=set(poses),
        poses_by_name=poses,
    )
    fringe_manifest: dict[str, Any] | None = None
    if tracks:
        fringe_manifest = triangulate_connected_fringe(
            source_model=append_destination / "model",
            destination=run_root / FRINGE_RELATIVE,
            tracks=tracks,
            camera_params=camera_params,
        )
        assert_b0_unchanged(frozen.root, b0_receipt)
    return {
        "schema_version": 1,
        "artifact_type": "RIVER_MAP_EXTEND_RECEIPT_V1",
        "selected_image_count": len(selected),
        "append": append_manifest,
        "fringe_track_count": len(tracks),
        "fringe": fringe_manifest,
        "b0_unchanged": True,
    }


def matches_from_inliers(row: Mapping[str, Any]) -> tuple[LiftedMatch, ...]:
    return tuple(
        LiftedMatch(
            query_xy=(float(item["query_xy"][0]), float(item["query_xy"][1])),
            point3d_id=int(item["point3d_id"]),
            reference_name=str(item["reference_name"]),
            confidence=float(item.get("confidence") or 0.0),
            lift_distance_px=0.0,
        )
        for item in row.get("inlier_observations") or ()
    )


__all__ = [
    "APPEND_MAP_SCHEMA",
    "APPEND_RELATIVE",
    "AppendObservation",
    "DEFAULT_MIN_TRACK_VIEWS",
    "FRINGE_MAP_SCHEMA",
    "FRINGE_RELATIVE",
    "FringeView",
    "MapExtendError",
    "append_existing_observations",
    "cluster_connected_fringe_tracks",
    "extend_from_registrations",
    "observations_from_row",
    "overlay_append_catalog",
    "persist_relifted_row",
    "select_append_rows",
    "triangulate_connected_fringe",
]

