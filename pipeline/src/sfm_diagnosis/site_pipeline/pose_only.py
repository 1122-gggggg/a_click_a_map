"""Keep GLUEMAP bridge poses while excluding their triangulation observations."""

from __future__ import annotations

import copy
import inspect
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLMAP_MAX_IMAGE_ID = 2_147_483_647


@dataclass(frozen=True)
class PoseOnlyMaskStats:
    image_indexes: tuple[int, ...]
    prediction_groups: int
    suppressed_score_views: int
    suppressed_virtual_views: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_indexes": list(self.image_indexes),
            "prediction_groups": self.prediction_groups,
            "suppressed_score_views": self.suppressed_score_views,
            "suppressed_virtual_views": self.suppressed_virtual_views,
        }


def normalized_image_name(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"image name must be a safe relative path: {value!r}")
    return path.as_posix()


def pose_only_indexes(
    images_list: Sequence[str], pose_only_names: Iterable[str | Path]
) -> frozenset[int]:
    lookup = {normalized_image_name(name): index for index, name in enumerate(images_list)}
    names = {normalized_image_name(name) for name in pose_only_names}
    missing = sorted(names - set(lookup))
    if missing:
        raise ValueError(f"pose-only images are absent from GLUEMAP input: {missing}")
    return frozenset(lookup[name] for name in names)


def filter_predictions_for_triangulation(
    predictions: Mapping[str, Any], excluded: frozenset[int]
) -> tuple[dict[str, Any], PoseOnlyMaskStats]:
    if not excluded:
        return dict(predictions), PoseOnlyMaskStats((), 0, 0, 0)
    indexes = predictions.get("indexes")
    scores = predictions.get("scores")
    virtual = predictions.get("valid_virtual")
    if not isinstance(indexes, (Sequence, Mapping)):
        raise ValueError("predictions require indexed groups")
    if not isinstance(scores, (Sequence, Mapping)) or not isinstance(virtual, (Sequence, Mapping)):
        raise ValueError("predictions require scores and valid_virtual groups")
    masked_scores, suppressed_scores = _masked_series(scores, indexes, excluded, "scores")
    masked_virtual, suppressed_virtual = _masked_series(virtual, indexes, excluded, "valid_virtual")
    result = dict(predictions)
    result["scores"] = masked_scores
    result["valid_virtual"] = masked_virtual
    return result, PoseOnlyMaskStats(
        tuple(sorted(excluded)), len(indexes), suppressed_scores, suppressed_virtual
    )


def _masked_series(series, indexes, excluded, label):
    if len(series) != len(indexes):
        raise ValueError(f"{label} and indexes group counts differ")
    keys = sorted(series) if isinstance(series, Mapping) else range(len(series))
    result = {} if isinstance(series, Mapping) else []
    suppressed = 0
    for key in keys:
        group = series[key]
        image_indexes = tuple(int(value) for value in indexes[key])
        if getattr(group, "ndim", None) != 3 or int(group.shape[1]) != len(image_indexes):
            raise ValueError(f"{label} group shape does not match image indexes")
        clone = group.clone() if callable(getattr(group, "clone", None)) else copy.deepcopy(group)
        positions = [
            position
            for position, image_index in enumerate(image_indexes)
            if image_index in excluded
        ]
        if positions:
            clone[:, positions, :] = 0
            suppressed += len(positions)
        if isinstance(result, dict):
            result[key] = clone
        else:
            result.append(clone)
    if isinstance(series, tuple):
        return tuple(result), suppressed
    return result, suppressed


def purge_pose_only_pairs(database_path: str | Path, image_ids: Iterable[int]) -> dict[str, int]:
    excluded = frozenset(int(value) for value in image_ids)
    if not excluded:
        return {"matches": 0, "two_view_geometries": 0}
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    deleted = {}
    select_queries = {
        "matches": "SELECT pair_id FROM matches",
        "two_view_geometries": "SELECT pair_id FROM two_view_geometries",
    }
    delete_queries = {
        "matches": "DELETE FROM matches WHERE pair_id=?",
        "two_view_geometries": "DELETE FROM two_view_geometries WHERE pair_id=?",
    }
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in ("matches", "two_view_geometries"):
            if table not in tables:
                raise RuntimeError(f"triangulation database lacks {table}")
            pair_ids = [
                int(row[0])
                for row in connection.execute(select_queries[table])
                if excluded.intersection(divmod(int(row[0]), COLMAP_MAX_IMAGE_ID))
            ]
            connection.executemany(delete_queries[table], ((pair_id,) for pair_id in pair_ids))
            deleted[table] = len(pair_ids)
    return deleted


def install_gluemap_pose_only_mask(
    names: Iterable[str], *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Patch one pinned GLUEMAP process before refinement.

    Global pose/star inference still sees the original predictions. Every
    triangulation path receives a copy with pose-only observations masked, and
    SIFT pairs incident to those images are removed before triangulation.
    """

    from gluemap.controllers import global_refinement

    pose_names = tuple(normalized_image_name(name) for name in names)
    state: dict[str, Any] = {
        "pose_only_names": list(pose_names),
        "image_indexes": None,
        "image_ids": None,
        "prediction_mask": None,
        "database_pair_deletions": [],
    }
    cache: dict[int, dict[str, Any]] = {}
    original_prepare = global_refinement.prepare_glomap_prior
    original_establish = global_refinement.establish_tracks_from_predictions_dict
    original_initialize = global_refinement.initialize_world_points
    original_merge = global_refinement.merge_colmap_databases
    merge_signature = inspect.signature(original_merge)
    if "output_path" not in merge_signature.parameters:
        raise RuntimeError("GLUEMAP merged-database signature is unsupported")
    owned_root = None if workspace_root is None else Path(workspace_root).resolve()

    def prediction_from(args, kwargs, position):
        if "predictions_dict" in kwargs:
            return kwargs["predictions_dict"]
        if len(args) <= position:
            raise RuntimeError("GLUEMAP prediction argument contract changed")
        return args[position]

    def replace_prediction(args, kwargs, filtered, position):
        replacement_kwargs = dict(kwargs)
        if "predictions_dict" in replacement_kwargs:
            replacement_kwargs["predictions_dict"] = filtered
            return args, replacement_kwargs
        replacement_args = list(args)
        if len(replacement_args) <= position:
            raise RuntimeError("GLUEMAP prediction argument contract changed")
        replacement_args[position] = filtered
        return tuple(replacement_args), replacement_kwargs

    def filtered_for(predictions):
        existing = cache.get(id(predictions))
        if existing is not None:
            return existing
        indexes = state["image_indexes"]
        if indexes is None:
            raise RuntimeError("GLUEMAP triangulated before pose-only indexes resolved")
        filtered, stats = filter_predictions_for_triangulation(predictions, indexes)
        cache[id(predictions)] = filtered
        state["prediction_mask"] = stats.as_dict()
        return filtered

    def prepare_wrapper(*args, **kwargs):
        images = kwargs.get("images_list", args[2] if len(args) > 2 else None)
        if images is None:
            raise RuntimeError("GLUEMAP image-list argument contract changed")
        indexes = pose_only_indexes(images, pose_names)
        if state["image_indexes"] is not None and state["image_indexes"] != indexes:
            raise RuntimeError("GLUEMAP image order changed during refinement")
        state["image_indexes"] = indexes
        state["image_ids"] = tuple(index + 1 for index in sorted(indexes))
        predictions = prediction_from(args, kwargs, 4)
        replaced_args, replaced_kwargs = replace_prediction(
            args, kwargs, filtered_for(predictions), 4
        )
        return original_prepare(*replaced_args, **replaced_kwargs)

    def establish_wrapper(*args, **kwargs):
        predictions = prediction_from(args, kwargs, 0)
        replaced_args, replaced_kwargs = replace_prediction(
            args, kwargs, filtered_for(predictions), 0
        )
        return original_establish(*replaced_args, **replaced_kwargs)

    def initialize_wrapper(*args, **kwargs):
        predictions = prediction_from(args, kwargs, 0)
        replaced_args, replaced_kwargs = replace_prediction(
            args, kwargs, filtered_for(predictions), 0
        )
        return original_initialize(*replaced_args, **replaced_kwargs)

    def merge_wrapper(*args, **kwargs):
        result = original_merge(*args, **kwargs)
        try:
            bound = merge_signature.bind(*args, **kwargs)
        except TypeError as error:
            raise RuntimeError("GLUEMAP merged-database arguments changed") from error
        database = bound.arguments.get("output_path")
        image_ids = state["image_ids"]
        if database is None or image_ids is None:
            raise RuntimeError("GLUEMAP merged-database argument contract changed")
        resolved_database = Path(database).resolve()
        if owned_root is not None and owned_root not in resolved_database.parents:
            raise RuntimeError("refusing to purge a database outside the GLUEMAP workspace")
        state["database_pair_deletions"].append(purge_pose_only_pairs(resolved_database, image_ids))
        return result

    global_refinement.prepare_glomap_prior = prepare_wrapper
    global_refinement.establish_tracks_from_predictions_dict = establish_wrapper
    global_refinement.initialize_world_points = initialize_wrapper
    global_refinement.merge_colmap_databases = merge_wrapper
    return state


@contextmanager
def gluemap_pose_only_mask(names: Iterable[str], *, workspace_root: str | Path):
    """Install the process-local patch and always restore upstream functions."""

    from gluemap.controllers import global_refinement

    attributes = (
        "prepare_glomap_prior",
        "establish_tracks_from_predictions_dict",
        "initialize_world_points",
        "merge_colmap_databases",
    )
    originals = {name: getattr(global_refinement, name) for name in attributes}
    state = install_gluemap_pose_only_mask(names, workspace_root=workspace_root)
    try:
        yield state
    finally:
        for name, implementation in originals.items():
            setattr(global_refinement, name, implementation)


def assert_pose_only_observation_free(
    reconstruction: Any, pose_only_names: Iterable[str]
) -> dict[str, Any]:
    names = {normalized_image_name(name) for name in pose_only_names}
    images = dict(reconstruction.images.items())
    points = dict(reconstruction.points3D.items())
    matching = {
        int(image_id): image
        for image_id, image in images.items()
        if normalized_image_name(str(image.name)) in names
    }
    missing = names - {normalized_image_name(str(image.name)) for image in matching.values()}
    if missing:
        raise RuntimeError(f"pose-only cameras were not registered: {sorted(missing)}")
    per_image = {}
    observations = 0
    for image_id, image in matching.items():
        count = sum(int(point.point3D_id) in points for point in image.points2D)
        observations += count
        per_image[str(image_id)] = {"image_name": str(image.name), "observations": count}
    image_ids = set(matching)
    incident_tracks = sum(
        any(int(element.image_id) in image_ids for element in point.track.elements)
        for point in points.values()
    )
    summary = {
        "pose_only_images": len(matching),
        "point3D_observations": observations,
        "incident_tracks": incident_tracks,
        "per_image": per_image,
        "clean": observations == 0 and incident_tracks == 0,
    }
    if not summary["clean"]:
        raise RuntimeError(
            "pose-only triangulation invariant violated: "
            f"observations={observations}, tracks={incident_tracks}"
        )
    return summary


__all__ = [
    "PoseOnlyMaskStats",
    "filter_predictions_for_triangulation",
    "assert_pose_only_observation_free",
    "install_gluemap_pose_only_mask",
    "gluemap_pose_only_mask",
    "normalized_image_name",
    "pose_only_indexes",
    "purge_pose_only_pairs",
]
