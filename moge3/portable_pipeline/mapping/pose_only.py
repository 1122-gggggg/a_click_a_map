"""True pose-only enforcement for GlueMap refinement (no silent triangulation).

Pure-rotation keyframes keep their poses (preserving retrieval and sequential
graph connectivity) but must contribute zero 3D observations. This module
ports the proven mechanism against THIS checkout's refinement API
(verified identical call shapes to the older pinned checkout for the four
patched entry points): before refinement, prediction scores for pose-only
views are zeroed (blocking prior/virtual tracks) and SIFT match pairs
incident to pose-only images are purged from the merged database (blocking
SIFT triangulation). The finished model is then verified observation-free;
the run fails loudly if the invariant is violated.

The mask installs inside the GlueMap subprocess via the pair-patch
injection (see gluemap_runner); this module holds the importable logic so
both the stage process (verification) and the subprocess (mask) share it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import copy
import inspect
import sqlite3

COLMAP_MAX_IMAGE_ID = 2_147_483_647


def normalized_image_name(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"image name must be a safe relative path: {value!r}")
    return path.as_posix()


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
    missing = names - {
        normalized_image_name(str(image.name)) for image in matching.values()
    }
    if missing:
        raise RuntimeError(
            f"pose-only cameras were not registered: {sorted(missing)}"
        )
    per_image: dict[str, Any] = {}
    observations = 0
    for image_id, image in matching.items():
        count = sum(int(pt.point3D_id) in points for pt in image.points2D)
        observations += count
        per_image[str(image_id)] = {
            "image_name": str(image.name),
            "observations": count,
        }
    image_ids = set(matching)
    incident_tracks = sum(
        any(int(el.image_id) in image_ids for el in point.track.elements)
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


def pose_only_indexes(
    images_list: Sequence[str], pose_only_names: Iterable[str | Path]
) -> frozenset[int]:
    lookup = {normalized_image_name(name): i for i, name in enumerate(images_list)}
    names = {normalized_image_name(name) for name in pose_only_names}
    missing = sorted(names - set(lookup))
    if missing:
        raise ValueError(f"pose-only images absent from GlueMap input: {missing}")
    return frozenset(lookup[name] for name in names)


def filter_predictions_for_triangulation(
    predictions: Mapping[str, Any], excluded: frozenset[int]
) -> tuple[dict[str, Any], PoseOnlyMaskStats]:
    """Zero scores/valid_virtual for excluded views (copies, never mutates)."""
    if not excluded:
        return dict(predictions), PoseOnlyMaskStats((), 0, 0, 0)
    indexes = predictions.get("indexes")
    scores = predictions.get("scores")
    virtual = predictions.get("valid_virtual")
    if not isinstance(indexes, (Sequence, Mapping)):
        raise ValueError("predictions require indexed groups")
    if not isinstance(scores, (Sequence, Mapping)) or not isinstance(
        virtual, (Sequence, Mapping)
    ):
        raise ValueError("predictions require scores and valid_virtual groups")
    masked_scores, suppressed_scores = _masked_series(
        scores, indexes, excluded, "scores"
    )
    masked_virtual, suppressed_virtual = _masked_series(
        virtual, indexes, excluded, "valid_virtual"
    )
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
        image_indexes = tuple(int(v) for v in indexes[key])
        if getattr(group, "ndim", None) != 3 or int(group.shape[1]) != len(
            image_indexes
        ):
            raise ValueError(f"{label} group shape does not match image indexes")
        clone = (
            group.clone()
            if callable(getattr(group, "clone", None))
            else copy.deepcopy(group)
        )
        positions = [
            pos for pos, img in enumerate(image_indexes) if img in excluded
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


def purge_pose_only_pairs(
    database_path: str | Path, image_ids: Iterable[int]
) -> dict[str, int]:
    """Delete SIFT matches/two-view geometries incident to pose-only images."""
    excluded = frozenset(int(v) for v in image_ids)
    if not excluded:
        return {"matches": 0, "two_view_geometries": 0}
    path = Path(database_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    deleted: dict[str, int] = {}
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in ("matches", "two_view_geometries"):
            if table not in tables:
                raise RuntimeError(f"triangulation database lacks {table}")
            pair_ids = [
                int(row[0])
                for row in connection.execute(f"SELECT pair_id FROM {table}")
                if excluded.intersection(
                    divmod(int(row[0]), COLMAP_MAX_IMAGE_ID)
                )
            ]
            connection.executemany(
                f"DELETE FROM {table} WHERE pair_id=?",
                ((pair_id,) for pair_id in pair_ids),
            )
            deleted[table] = len(pair_ids)
    return deleted


def install_gluemap_pose_only_mask(
    names: Iterable[str], *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Patch one GlueMap process: poses flow, triangulation observations stop.

    Global pose/star inference still sees the original predictions. Every
    triangulation path (prior database, track establishment, world-point
    init) receives a masked copy, and SIFT pairs incident to pose-only
    images are purged from the merged database before triangulation.
    Raises immediately if this checkout's GlueMap call shapes drift.
    """
    from gluemap.controllers import global_refinement

    pose_names = tuple(normalized_image_name(n) for n in names)
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
        raise RuntimeError("GlueMap merged-database signature is unsupported")
    owned_root = None if workspace_root is None else Path(workspace_root).resolve()

    def prediction_from(args, kwargs, position):
        if "predictions_dict" in kwargs:
            return kwargs["predictions_dict"]
        if len(args) <= position:
            raise RuntimeError("GlueMap prediction argument contract changed")
        return args[position]

    def replace_prediction(args, kwargs, filtered, position):
        replacement_kwargs = dict(kwargs)
        if "predictions_dict" in replacement_kwargs:
            replacement_kwargs["predictions_dict"] = filtered
            return args, replacement_kwargs
        replacement_args = list(args)
        if len(replacement_args) <= position:
            raise RuntimeError("GlueMap prediction argument contract changed")
        replacement_args[position] = filtered
        return tuple(replacement_args), replacement_kwargs

    def filtered_for(predictions):
        existing = cache.get(id(predictions))
        if existing is not None:
            return existing
        indexes = state["image_indexes"]
        if indexes is None:
            raise RuntimeError(
                "GlueMap triangulated before pose-only indexes resolved"
            )
        filtered, stats = filter_predictions_for_triangulation(predictions, indexes)
        cache[id(predictions)] = filtered
        state["prediction_mask"] = stats.as_dict()
        return filtered

    def prepare_wrapper(*args, **kwargs):
        images = kwargs.get("images_list", args[2] if len(args) > 2 else None)
        if images is None:
            raise RuntimeError("GlueMap image-list argument contract changed")
        indexes = pose_only_indexes(images, pose_names)
        if state["image_indexes"] is not None and state["image_indexes"] != indexes:
            raise RuntimeError("GlueMap image order changed during refinement")
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
            raise RuntimeError("GlueMap merged-database arguments changed") from error
        database = bound.arguments.get("output_path")
        image_ids = state["image_ids"]
        if database is None or image_ids is None:
            raise RuntimeError("GlueMap merged-database argument contract changed")
        resolved_database = Path(database).resolve()
        if owned_root is not None and owned_root not in resolved_database.parents:
            raise RuntimeError("refusing to purge a database outside the workspace")
        state["database_pair_deletions"].append(
            purge_pose_only_pairs(resolved_database, image_ids)
        )
        return result

    global_refinement.prepare_glomap_prior = prepare_wrapper
    global_refinement.establish_tracks_from_predictions_dict = establish_wrapper
    global_refinement.initialize_world_points = initialize_wrapper
    global_refinement.merge_colmap_databases = merge_wrapper
    return state


@contextmanager
def gluemap_pose_only_mask(names: Iterable[str], *, workspace_root: str | Path):
    """Install the process-local patch; always restore upstream functions."""
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


__all__ = [
    "COLMAP_MAX_IMAGE_ID",
    "PoseOnlyMaskStats",
    "assert_pose_only_observation_free",
    "filter_predictions_for_triangulation",
    "gluemap_pose_only_mask",
    "install_gluemap_pose_only_mask",
    "normalized_image_name",
    "pose_only_indexes",
    "purge_pose_only_pairs",
]
