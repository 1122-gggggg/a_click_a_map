"""Strict held-out MegaLoc + EDM + COLMAP-PnP validation provider.

The provider builds retrieval rows only from the final registered map, matches
held-out query pixels with EDM, lifts them through final-map observations, and
solves an absolute pose.  Query images never enter the reference reconstruction.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from sfm_diagnosis.edm_risk.edm_loo import (
    EDMQuery,
    EDMQueryResult,
    EDMReferenceIndex,
)
from sfm_diagnosis.viewpoint_policy import (
    DEFAULT_MIN_REFERENCE_OCCUPIED_BINS,
    FROZEN_PNP_THRESHOLDS,
    INTERSECTION_CELLS_NAME,
    align_occupied_bins,
    bank_occupancy_cutoff,
    decide_query_status,
    filter_reference_bank,
    load_intersection_cells,
    never_weaker_than_frozen,
    occupied_bins,
    position_in_intersection,
    prefer_side_looking_references,
    viewpoint_should_abstain,
)


DEFAULT_THRESHOLDS: dict[str, float | int] = dict(FROZEN_PNP_THRESHOLDS)
TEMPORAL_REFERENCE_NAME = "temporal_anchor"
MIN_TEMPORAL_INLIERS = 30
# 8 was the ceiling a pairwise-flow transfer could actually carry; a point tracker
# survives older anchors (bench/locotrack_transfer.json), so the cap is tunable.
MAX_TEMPORAL_ANCHOR_AGE = int(os.environ.get("P172_MAX_ANCHOR_AGE", "8"))
KLT_WIN_SIZE = (31, 31)
KLT_MAX_LEVEL = 3
KLT_FB_MAX_PX = 2.0



def _resolve_audited_site_packages(
    explicit: str | None,
    *,
    prefix: Path | None = None,
) -> Path:
    """Resolve the exact site-packages tree approved for the EDM runtime."""

    if explicit:
        path = Path(explicit).expanduser().resolve(strict=True)
    else:
        runtime_prefix = Path(sys.prefix) if prefix is None else Path(prefix)
        path = (
            runtime_prefix
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        ).resolve(strict=False)
    if explicit and not path.is_dir():
        raise NotADirectoryError(path)
    return path


def _configure_audited_runtime_site_packages(path: Path) -> Path:
    """Point both MegaLoc and official-EDM admission at one audited runtime."""

    resolved = Path(path).resolve(strict=True)
    import river_map_quality.historical_experiment as historical_runtime
    import river_map_quality.official_edm_adapter_loo as official_edm

    historical_runtime.AUDITED_TORCH_SITE_PACKAGES = resolved
    official_edm.AUDITED_EDM_SITE_PACKAGES = resolved
    return resolved


def scaled_pinhole_parameters(
    calibration: Mapping[str, Any], *, width: int, height: int
) -> tuple[float, float, float, float]:
    """Scale one undistorted PINHOLE calibration to an exact image resolution."""

    source_width = int(calibration.get("image_width") or 0)
    source_height = int(calibration.get("image_height") or 0)
    matrix = np.asarray(calibration.get("K"), dtype=float)
    if min(source_width, source_height, width, height) <= 0 or matrix.shape != (3, 3):
        raise ValueError("intrinsics calibration and target dimensions must be valid")
    if calibration.get("images_are_undistorted") is not True:
        raise ValueError("localization inputs must already be undistorted")
    scale_x, scale_y = width / source_width, height / source_height
    return (
        float(matrix[0, 0] * scale_x),
        float(matrix[1, 1] * scale_y),
        float(matrix[0, 2] * scale_x),
        float(matrix[1, 2] * scale_y),
    )


def rank_reference_indices(
    reference_descriptors: np.ndarray,
    query_descriptor: np.ndarray,
    *,
    reference_sessions: Sequence[str],
    excluded_sessions: frozenset[str],
    top_k: int,
    occupied_bins: Sequence[int] | None = None,
    min_occupied_bins: int = 0,
    retrieve_pool: int | None = None,
) -> tuple[tuple[int, ...], bool]:
    """Apply session exclusion before cosine ranking, then drop empty-side views."""

    references = np.asarray(reference_descriptors, dtype=np.float32)
    query = np.asarray(query_descriptor, dtype=np.float32).reshape(-1)
    if references.ndim != 2 or references.shape[1:] != query.shape:
        raise ValueError("query/reference descriptor dimensions disagree")
    if len(references) != len(reference_sessions):
        raise ValueError("reference sessions must align with descriptor rows")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    query_norm = float(np.linalg.norm(query))
    reference_norms = np.linalg.norm(references, axis=1)
    if query_norm <= 0 or np.any(reference_norms <= 0):
        raise ValueError("retrieval descriptors must have non-zero norm")
    scores = (references @ query) / (reference_norms * query_norm)
    allowed = [
        index
        for index, session in enumerate(reference_sessions)
        if session not in excluded_sessions
    ]
    order = sorted(allowed, key=lambda index: (-float(scores[index]), index))
    if occupied_bins is None or int(min_occupied_bins) <= 0:
        return tuple(order[:top_k]), False
    if len(occupied_bins) != len(references):
        raise ValueError("occupied bins must align with descriptor rows")
    return prefer_side_looking_references(
        tuple(order),
        occupied_bins,
        min_occupied_bins=int(min_occupied_bins),
        top_k=top_k,
    )


def subset_reference_identities(
    *,
    names: Sequence[str],
    sessions: Sequence[str],
    paths: Sequence[Path],
    occupied: Sequence[int],
    observations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    selected_names: Sequence[str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[Path, ...],
    tuple[int, ...],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    selected = tuple(str(name) for name in selected_names)
    if len(names) != len(sessions) or len(names) != len(paths) or len(names) != len(occupied):
        raise ValueError("reference identity lists must align")
    index = {str(name): position for position, name in enumerate(names)}
    missing = [name for name in selected if name not in index]
    if missing:
        raise KeyError(f"selected reference identities are absent: {missing[:5]}")
    missing_obs = [name for name in selected if name not in observations]
    if missing_obs:
        raise KeyError(f"selected observations are absent: {missing_obs[:5]}")
    indexes = [index[name] for name in selected]
    return (
        selected,
        tuple(sessions[position] for position in indexes),
        tuple(Path(paths[position]) for position in indexes),
        align_occupied_bins(names, occupied, selected),
        {name: observations[name] for name in selected},
    )


def evaluate_localization_admission(
    *,
    registration_success: bool,
    metrics: Mapping[str, Any],
    decision_status: str,
    position: Sequence[float] | None,
    intersection_cells: np.ndarray,
    requested_thresholds: Mapping[str, Any] | None,
) -> tuple[bool, str, bool, bool]:
    """Apply frozen PnP gates and viewpoint abstain after a pose exists."""

    in_intersection = bool(
        position is not None and position_in_intersection(position, intersection_cells)
    )
    thresholds = never_weaker_than_frozen(
        requested_thresholds, in_intersection=in_intersection
    )
    metrics_present = all(
        metrics.get(key) is not None
        for key in ("occupancy_4x4", "convex_hull_coverage", "occupied_frac_30")
    )
    viewpoint_abstain = bool(registration_success) and metrics_present and viewpoint_should_abstain(
        occupancy_4x4=int(metrics.get("occupancy_4x4") or 0),
        hull_coverage=float(metrics.get("convex_hull_coverage") or 0.0),
        occupied_frac_30=float(metrics.get("occupied_frac_30") or 0.0),
        min_occupancy_4x4=int(thresholds["minimum_occupancy_4x4"]),
        min_hull_coverage=float(thresholds["minimum_hull_coverage"]),
    )
    strong = bool(registration_success) and localization_is_strong(
        metrics, decision_status=decision_status, thresholds=thresholds
    )
    status = decide_query_status(
        registration_success=bool(registration_success),
        strong_success=strong,
        viewpoint_ok=not viewpoint_abstain,
    )
    return status == "LOCALIZED_STRONG", status, in_intersection, viewpoint_abstain


def build_localization_payload(
    *,
    query: Path | str,
    map_name: str,
    result: EDMQueryResult,
    reference_bank: str,
) -> dict[str, Any]:
    status = result.query_status or decide_query_status(
        registration_success=result.registration_success,
        strong_success=bool(result.success),
        viewpoint_ok=not bool(result.viewpoint_abstain),
    )
    result_payload = result.to_dict()
    result_payload["loo_mode"] = "none"
    result_payload["ground_truth_source"] = "NONE"
    return {
        "status": status,
        "validation": "NONE",
        "map": map_name,
        "query": str(query),
        "reference_bank": reference_bank,
        "in_intersection": bool(result.in_intersection),
        "viewpoint_abstain": bool(result.viewpoint_abstain),
        "result": result_payload,
    }


def localization_is_strong(
    metrics: Mapping[str, Any],
    *,
    decision_status: str,
    thresholds: Mapping[str, Any],
) -> bool:
    """Apply the frozen, conservative held-out localization admission gates."""

    if decision_status != "ACCEPT":
        return False
    try:
        return bool(
            int(metrics.get("inlier_count") or 0) >= int(thresholds["strong_inliers"])
            and float(metrics.get("inlier_ratio") or 0.0)
            >= float(thresholds["minimum_inlier_ratio"])
            and float(metrics.get("convex_hull_coverage") or 0.0)
            >= float(thresholds["minimum_hull_coverage"])
            and int(metrics.get("occupancy_4x4") or 0) >= int(thresholds["minimum_occupancy_4x4"])
            and float(metrics.get("positive_depth_ratio") or 0.0)
            >= float(thresholds["minimum_positive_depth_ratio"])
            and float(metrics.get("reprojection_p90") or math.inf)
            <= float(thresholds["maximum_reprojection_p90_px"])
        )
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class _ReferenceSubset:
    indices: tuple[int, ...]
    excluded_sessions: frozenset[str]


@dataclass(frozen=True)
class MapTrackAnchor:
    """ACCEPT frame map refs + track inliers for the next query."""

    image_path: Path
    xy: np.ndarray
    point3d_ids: np.ndarray
    reference_names: tuple[str, ...]
    age: int = 0

    def next_age(self) -> "MapTrackAnchor | None":
        nxt = int(self.age) + 1
        if nxt > MAX_TEMPORAL_ANCHOR_AGE:
            return None
        return MapTrackAnchor(
            self.image_path, self.xy, self.point3d_ids, self.reference_names, nxt
        )


class FinalMapEDMProvider:
    """Deployment-owned implementation of the Stage-14 EDMProvider protocol."""

    def __init__(
        self,
        *,
        map_model: str,
        keyframes: str,
        query_manifest: str,
        cache_dir: str,
        edm_config: Mapping[str, Any],
        megaloc_source: str,
        megaloc_checkpoint: str,
        intrinsics_calibration: Mapping[str, Any],
        precomputed_query_descriptors: str | None = None,
        precomputed_query_names: str | None = None,
        top_k: int = 5,
        lift_distance_px: float = 2.0,
        thresholds: Mapping[str, Any] | None = None,
        descriptor_batch_size: int = 8,
        audited_edm_site_packages: str | None = None,
        intersection_cells_path: str | None = None,
        min_reference_occupied_bins: int | None = None,
        reference_depth_dir: str | None = None,
        matcher_backend: str = "edm",
        sparse_max_keypoints: int = 4096,
        sparse_ref_cache: int = 128,
    ) -> None:
        from .sparse_reloc import SUPPORTED_BACKENDS

        if matcher_backend not in ("edm",) + tuple(SUPPORTED_BACKENDS):
            raise ValueError(f"unsupported reloc matcher: {matcher_backend}")
        self.matcher_backend = str(matcher_backend)
        self.sparse_max_keypoints = int(sparse_max_keypoints)
        self.sparse_ref_cache = int(sparse_ref_cache)
        self._sparse: Any | None = None
        self.map_model = Path(map_model).resolve(strict=True)
        self.keyframes_path = Path(keyframes).resolve(strict=True)
        self.query_manifest_path = Path(query_manifest).resolve(strict=True)
        self.cache_dir = Path(cache_dir).absolute()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.edm_config = dict(edm_config)
        self.megaloc_source = Path(megaloc_source).resolve(strict=True)
        self.megaloc_checkpoint = Path(megaloc_checkpoint).resolve(strict=True)
        self.intrinsics_calibration = dict(intrinsics_calibration)
        self.precomputed_query_descriptors = _optional_file(precomputed_query_descriptors)
        self.precomputed_query_names = _optional_file(precomputed_query_names)
        self.top_k = int(top_k)
        self.lift_distance_px = float(lift_distance_px)
        self.thresholds = MappingProxyType(
            never_weaker_than_frozen(thresholds, in_intersection=False)
        )
        self._requested_min_reference_occupied_bins = (
            None if min_reference_occupied_bins is None else int(min_reference_occupied_bins)
        )
        self.min_reference_occupied_bins = (
            DEFAULT_MIN_REFERENCE_OCCUPIED_BINS
            if self._requested_min_reference_occupied_bins is None
            else self._requested_min_reference_occupied_bins
        )
        self.descriptor_batch_size = int(descriptor_batch_size)
        self.audited_edm_site_packages = _configure_audited_runtime_site_packages(
            _resolve_audited_site_packages(audited_edm_site_packages)
        )
        if self.top_k <= 0 or self.lift_distance_px <= 0 or self.descriptor_batch_size <= 0:
            raise ValueError("localizer top_k, lift distance, and batch size must be positive")

        discovered_cells = self.map_model.parent / "localization" / INTERSECTION_CELLS_NAME
        cells_path = (
            Path(intersection_cells_path).expanduser()
            if intersection_cells_path
            else discovered_cells
        )
        self._intersection_cells_path = cells_path if cells_path.is_file() else None
        self._intersection_cells = load_intersection_cells(self._intersection_cells_path)

        self._keyframes = _keyframe_index(self.keyframes_path)
        self._queries = _query_index(self.query_manifest_path)
        self._reference_names: tuple[str, ...] = ()
        self._reference_sessions: tuple[str, ...] = ()
        self._reference_paths: tuple[Path, ...] = ()
        self._reference_occupied_bins: tuple[int, ...] = ()
        self._observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._cam_from_world: dict[str, np.ndarray] = {}
        self._native_wh: dict[str, tuple[int, int]] = {}
        self._camera_params: dict[str, tuple[float, float, float, float]] = {}
        self._median_sparse_z: dict[str, float] = {}
        self._reference_depth_dir = (
            None if reference_depth_dir is None else Path(reference_depth_dir).expanduser()
        )
        self._depth_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        from river_map_quality.official_edm_adapter import PreparedReferenceCache
        self._prepared_reference_cache = PreparedReferenceCache()
        self._point_ids = np.empty(0, dtype=np.int64)
        self._point_track_len = np.empty(0, dtype=np.int32)
        self._point_xyz = np.empty((0, 3), dtype=np.float64)
        self._reference_descriptors = np.empty((0, 0), dtype=np.float32)
        self._query_descriptors: dict[str, np.ndarray] = {}
        self._subsets: dict[str, _ReferenceSubset] = {}
        self._matcher: Any | None = None
        self.last_retrieval_source = "megaloc"
        self.last_n_transferred = 0
        self.last_track_anchor: MapTrackAnchor | None = None
        self._prepared = False
        self._geometry_locked = False
        self._reference_descriptors_locked = False
        self.fingerprint = _canonical_sha256(
            {
                "implementation": "FINAL_MAP_MEGALOC_OFFICIAL_EDM_PNP_V4",
                "min_reference_occupied_bins": (
                    "auto_p10_floor_40"
                    if self._requested_min_reference_occupied_bins is None
                    else self._requested_min_reference_occupied_bins
                ),
                "intersection_cells": (
                    None
                    if self._intersection_cells_path is None
                    else _sha256_file(self._intersection_cells_path)
                ),
                "model": _model_hashes(self.map_model),
                "keyframes": _sha256_file(self.keyframes_path),
                "queries": _sha256_file(self.query_manifest_path),
                "query_images": _query_image_set_sha256(self._queries),
                "edm_checkpoint": _sha256_file(Path(str(self.edm_config["checkpoint"]))),
                "megaloc_checkpoint": _sha256_file(self.megaloc_checkpoint),
                "precomputed_query_descriptors": (
                    None
                    if self.precomputed_query_descriptors is None
                    else _sha256_file(self.precomputed_query_descriptors)
                ),
                "precomputed_query_names": (
                    None
                    if self.precomputed_query_names is None
                    else _sha256_file(self.precomputed_query_names)
                ),
                "intrinsics": self.intrinsics_calibration,
                "thresholds": dict(self.thresholds),
                "top_k": self.top_k,
                "lift_distance_px": self.lift_distance_px,
                "matcher_backend": self.matcher_backend,
                "sparse_max_keypoints": self.sparse_max_keypoints,
                "audited_edm_site_packages": str(self.audited_edm_site_packages),
            }
        )

    def build_reference_index(
        self, *, excluded_sessions: frozenset[str], strict: bool
    ) -> EDMReferenceIndex:
        self._prepare()
        indices = tuple(
            index
            for index, session in enumerate(self._reference_sessions)
            if session not in excluded_sessions
        )
        if not indices:
            raise RuntimeError("strict LOO removed every final-map reference")
        sessions = frozenset(self._reference_sessions[index] for index in indices)
        index_id = _canonical_sha256(
            {
                "provider": self.fingerprint,
                "excluded_sessions": sorted(excluded_sessions),
                "reference_names": [self._reference_names[index] for index in indices],
                "strict": bool(strict),
            }
        )
        self._subsets[index_id] = _ReferenceSubset(indices, excluded_sessions)
        return EDMReferenceIndex(
            index_id=index_id,
            reference_sessions=sessions,
            session_only_landmarks_removed=bool(strict),
            appearance_descriptors_rebuilt=bool(strict),
        )

    def _anchor_if_admissible(
        self,
        query_path: Path,
        lifted,
        inlier_mask: np.ndarray,
        metrics: Mapping[str, Any],
        decision_status: str,
        reference_ids: tuple[str, ...],
    ) -> MapTrackAnchor | None:
        if decision_status != "ACCEPT":
            return None
        if int(metrics.get("track_inliers") or 0) < int(self.thresholds["strong_inliers"]):
            return None
        if float(metrics.get("inlier_ratio") or 0.0) < float(self.thresholds["minimum_inlier_ratio"]):
            return None
        map_refs = tuple(
            name for name in reference_ids if name and name != TEMPORAL_REFERENCE_NAME
        )
        if not map_refs:
            return None
        xy: list[tuple[float, float]] = []
        ids: list[int] = []
        for match, keep in zip(lifted, inlier_mask, strict=True):
            if not bool(keep) or match.point3d_id is None:
                continue
            point_id = int(match.point3d_id)
            if point_id < 0:
                continue
            xy.append((float(match.query_xy[0]), float(match.query_xy[1])))
            ids.append(point_id)
        if len(ids) < MIN_TEMPORAL_INLIERS:
            return None
        return MapTrackAnchor(
            image_path=query_path,
            xy=np.asarray(xy, dtype=np.float64),
            point3d_ids=np.asarray(ids, dtype=np.int64),
            reference_names=map_refs,
            age=0,
        )

    def _reference_indices_for_names(self, names: tuple[str, ...]) -> tuple[int, ...]:
        index = {name: position for position, name in enumerate(self._reference_names)}
        return tuple(index[name] for name in names if name in index)



    def _pack_query_result(
        self,
        *,
        query: EDMQuery,
        started: float,
        lifted,
        raw_matches: int,
        result,
        metrics: Mapping[str, Any],
        decision_status: str,
        reference_ids: tuple[str, ...],
        viewpoint_pool_fallback: bool,
        runtime_edm: float,
        runtime_pnp: float,
        temporal_consistency: float | None,
    ) -> EDMQueryResult:
        inlier_mask = (
            np.zeros(len(lifted), dtype=bool)
            if result is None
            else np.asarray(result.inlier_mask, dtype=bool)
        )
        point_ids = tuple(
            -1 if match.point3d_id is None else int(match.point3d_id)
            for match, keep in zip(lifted, inlier_mask, strict=True)
            if bool(keep)
        )
        inlier_query_xy = tuple(
            (float(match.query_xy[0]), float(match.query_xy[1]))
            for match, keep in zip(lifted, inlier_mask, strict=True)
            if bool(keep)
        )
        inlier_conf = [
            float(match.confidence)
            for match, keep in zip(lifted, inlier_mask, strict=True)
            if bool(keep) and getattr(match, "confidence", None) is not None
        ]
        position, rotation, yaw, pitch = _pose_fields(None if result is None else result.pose)
        success, query_status, in_intersection, viewpoint_abstain = (
            evaluate_localization_admission(
                registration_success=result is not None,
                metrics=metrics,
                decision_status=decision_status,
                position=position,
                intersection_cells=self._intersection_cells,
                requested_thresholds=self.thresholds,
            )
        )
        self.last_track_anchor = self._anchor_if_admissible(
            Path(query.image_path or self._queries[query.query_id]["image_path"]).resolve(),
            lifted,
            inlier_mask,
            metrics,
            decision_status,
            reference_ids,
        )
        # Change 3: Removed redundant self._empty_cuda_cache() on the per-query path.
        # Calling torch.cuda.empty_cache() after every query forced a full CUDA driver
        # synchronization and memory deallocation stall (~8 ms on 5090, ~32 ms on 5060).
        # Removing it allows PyTorch's caching allocator to reuse allocated buffers across
        # queries without changing any query result, decision, or pose.
        # Note: self._empty_cuda_cache() is retained in _ensure_megaloc_descriptors for
        # model unloading when freeing the MegaLoc runtime.
        return EDMQueryResult(
            query_id=query.query_id,
            session_id=query.session_id,
            timestamp=query.timestamp,
            success=success,
            registration_success=result is not None,
            raw_matches=raw_matches,
            valid_2d3d=len(lifted),
            ransac_inliers=int(np.count_nonzero(inlier_mask)),
            inlier_ratio=metrics.get("inlier_ratio"),
            reprojection_mean=metrics.get("reprojection_mean"),
            reprojection_median=metrics.get("reprojection_median"),
            reprojection_p90=metrics.get("reprojection_p90"),
            positive_depth_ratio=metrics.get("positive_depth_ratio"),
            pose_consistency=decision_status,
            temporal_consistency=temporal_consistency,
            runtime_edm=runtime_edm,
            runtime_pnp=runtime_pnp,
            runtime_total=time.perf_counter() - started,
            reference_ids=reference_ids,
            point_ids=point_ids,
            inlier_query_xy=inlier_query_xy,
            inlier_confidence_mean=float(sum(inlier_conf) / len(inlier_conf))
            if inlier_conf
            else None,
            estimated_position=position,
            estimated_R_wc=rotation,
            estimated_yaw_deg=yaw,
            estimated_pitch_deg=pitch,
            ground_truth_source="HELDOUT_NO_ABSOLUTE_GT",
            loo_mode="strict",
            occupancy_4x4=metrics.get("occupancy_4x4"),
            convex_hull_coverage=metrics.get("convex_hull_coverage"),
            occupied_frac_30=metrics.get("occupied_frac_30"),
            viewpoint_pool_fallback=bool(viewpoint_pool_fallback),
            query_status=query_status,
            in_intersection=bool(in_intersection),
            viewpoint_abstain=bool(viewpoint_abstain),
            n_track_2d3d=metrics.get("n_track_2d3d"),
            n_depth_2d3d=metrics.get("n_depth_2d3d"),
            track_inliers=metrics.get("track_inliers"),
        )

    def _klt_track_anchor(self, query_path: Path, anchor: MapTrackAnchor):
        import cv2
        from river_map_quality.official_edm_adapter import LiftedMatch

        if anchor.xy.size == 0 or anchor.point3d_ids.size == 0:
            return (), 0
        previous = cv2.imread(str(anchor.image_path), cv2.IMREAD_GRAYSCALE)
        current = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
        if previous is None:
            raise FileNotFoundError(anchor.image_path)
        if current is None:
            raise FileNotFoundError(query_path)
        p0 = np.asarray(anchor.xy, dtype=np.float32).reshape(-1, 1, 2)
        ids = np.asarray(anchor.point3d_ids, dtype=np.int64).reshape(-1)
        if len(p0) != len(ids):
            raise RuntimeError("KLT anchor xy/point3d_ids length mismatch")
        lk_kwargs = {
            "winSize": KLT_WIN_SIZE,
            "maxLevel": KLT_MAX_LEVEL,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        p1, status, _err = cv2.calcOpticalFlowPyrLK(previous, current, p0, None, **lk_kwargs)
        if p1 is None or status is None:
            return (), 0
        p0r, status_back, _err_b = cv2.calcOpticalFlowPyrLK(current, previous, p1, None, **lk_kwargs)
        if p0r is None or status_back is None:
            return (), 0
        forward = p1.reshape(-1, 2)
        back = p0r.reshape(-1, 2)
        origin = p0.reshape(-1, 2)
        fb = np.linalg.norm(origin - back, axis=1)
        height, width = current.shape
        keep = (
            status.reshape(-1).astype(bool)
            & status_back.reshape(-1).astype(bool)
            & np.isfinite(forward).all(axis=1)
            & (fb <= KLT_FB_MAX_PX)
            & (forward[:, 0] >= 0.0)
            & (forward[:, 1] >= 0.0)
            & (forward[:, 0] < float(width))
            & (forward[:, 1] < float(height))
            & (ids >= 0)
        )
        lifted = tuple(
            LiftedMatch(
                query_xy=(float(xy[0]), float(xy[1])),
                point3d_id=int(point_id),
                reference_name=TEMPORAL_REFERENCE_NAME,
                confidence=1.0,
                lift_distance_px=float(max(error, 1e-6)),
            )
            for xy, point_id, error in zip(forward[keep], ids[keep], fb[keep], strict=True)
        )
        return lifted, int(keep.sum())

    def _localize_klt(
        self,
        query: EDMQuery,
        query_path: Path,
        started: float,
        anchor: MapTrackAnchor,
    ) -> EDMQueryResult | None:
        match_started = time.perf_counter()
        lifted, raw_matches = self._klt_track_anchor(query_path, anchor)
        runtime_klt = time.perf_counter() - match_started
        self.last_n_transferred = len(lifted)
        if len(lifted) < 6:
            return None
        pnp_started = time.perf_counter()
        result, metrics, decision_status = self._solve(
            query_path, lifted, session_groups=False
        )
        runtime_pnp = time.perf_counter() - pnp_started
        n_inliers = 0 if result is None else int(np.count_nonzero(result.inlier_mask))
        if n_inliers < MIN_TEMPORAL_INLIERS:
            return None
        self.last_retrieval_source = "klt"
        return self._pack_query_result(
            query=query,
            started=started,
            lifted=lifted,
            raw_matches=raw_matches,
            result=result,
            metrics=metrics,
            decision_status=decision_status,
            reference_ids=tuple(anchor.reference_names),
            viewpoint_pool_fallback=False,
            runtime_edm=runtime_klt,
            runtime_pnp=runtime_pnp,
            temporal_consistency=1.0,
        )

    @staticmethod
    def _prefer_klt_bridge(megaloc: EDMQueryResult, klt: EDMQueryResult) -> EDMQueryResult:
        if klt.query_status == "LOCALIZED_STRONG":
            return klt
        if megaloc.query_status == "ABSTAINED" and klt.registration_success:
            return klt
        if klt.pose_consistency == "ACCEPT" and megaloc.pose_consistency != "ACCEPT":
            return klt
        return megaloc




    def localize(
        self,
        query: EDMQuery,
        index: EDMReferenceIndex,
        *,
        anchor: MapTrackAnchor | None = None,
    ) -> EDMQueryResult:
        self._prepare()
        subset = self._subsets.get(index.index_id)
        if subset is None:
            raise RuntimeError("unknown or stale strict-LOO reference index")
        if query.query_id not in self._query_descriptors:
            raise KeyError(f"query descriptor is absent: {query.query_id}")
        query_path = Path(query.image_path or self._queries[query.query_id]["image_path"])
        query_path = query_path.resolve(strict=True)
        started = time.perf_counter()
        self.last_retrieval_source = "megaloc"
        self.last_n_transferred = 0
        self.last_track_anchor = None
        self.last_modes = ()
        self.last_mode_decision = None
        ranked, viewpoint_pool_fallback = rank_reference_indices(
            self._reference_descriptors,
            self._query_descriptors[query.query_id],
            reference_sessions=self._reference_sessions,
            excluded_sessions=subset.excluded_sessions,
            top_k=self.top_k,
            occupied_bins=self._reference_occupied_bins,
            min_occupied_bins=self.min_reference_occupied_bins,
            retrieve_pool=None,
        )
        allowed = set(subset.indices)
        ranked = tuple(index for index in ranked if index in allowed)
        if not ranked:
            raise RuntimeError("strict LOO retrieval returned no references")
        # Change 2: Decode query image once per call and pass it through to
        # _match_and_lift and _solve, eliminating redundant 2688x1512 disk reads.
        # Cannot change decision because the decoded pixel array is identical.
        import cv2

        query_image = cv2.imread(
            str(query_path),
            cv2.IMREAD_COLOR if self.matcher_backend != "edm" else cv2.IMREAD_GRAYSCALE,
        )
        if query_image is None:
            raise FileNotFoundError(query_path)
        if self.matcher_backend != "edm":
            self._ensure_sparse()
            assert self._sparse is not None
            self._sparse.reset_stage_ms()
        match_started = time.perf_counter()
        lifted, raw_matches = self._match_and_lift(
            query_path, ranked, query_image=query_image
        )
        runtime_edm = time.perf_counter() - match_started
        pnp_started = time.perf_counter()
        result, metrics, decision_status = self._solve(
            query_path, lifted, query_image=query_image
        )
        runtime_pnp = time.perf_counter() - pnp_started
        megaloc = self._pack_query_result(
            query=query,
            started=started,
            lifted=lifted,
            raw_matches=raw_matches,
            result=result,
            metrics=metrics,
            decision_status=decision_status,
            reference_ids=tuple(self._reference_names[value] for value in ranked),
            viewpoint_pool_fallback=bool(viewpoint_pool_fallback),
            runtime_edm=runtime_edm,
            runtime_pnp=runtime_pnp,
            temporal_consistency=None,
        )
        megaloc_anchor = self.last_track_anchor
        modes_megaloc = tuple(self.last_modes)
        decision_megaloc = self.last_mode_decision
        if megaloc.query_status == "LOCALIZED_STRONG":
            return megaloc
        if anchor is None or int(anchor.point3d_ids.size) < MIN_TEMPORAL_INLIERS:
            self.last_modes = modes_megaloc
            self.last_mode_decision = decision_megaloc
            return megaloc
        klt = self._localize_klt(query, query_path, started, anchor)
        if klt is None:
            self.last_retrieval_source = "megaloc"
            self.last_n_transferred = 0
            self.last_track_anchor = megaloc_anchor
            self.last_modes = modes_megaloc
            self.last_mode_decision = decision_megaloc
            return megaloc
        modes_klt = tuple(self.last_modes)
        decision_klt = self.last_mode_decision
        chosen = self._prefer_klt_bridge(megaloc, klt)
        if chosen is megaloc:
            self.last_retrieval_source = "megaloc"
            self.last_n_transferred = 0
            self.last_track_anchor = megaloc_anchor
            self.last_modes = modes_megaloc
            self.last_mode_decision = decision_megaloc
        else:
            self.last_modes = modes_klt
            self.last_mode_decision = decision_klt
        return chosen


    def apply_reference_bank(self, names: Sequence[str], descriptors: np.ndarray) -> None:
        """Replace the reconstruction-wide bank with a frozen MegaLoc subset."""

        if not self._reference_names:
            self._load_geometry()
        selected = tuple(str(name) for name in names)
        missing = sorted(set(selected) - set(self._reference_names))
        if missing:
            raise RuntimeError(
                f"bundle reference identities are absent from the map: {missing[:5]}"
            )
        array = np.ascontiguousarray(descriptors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(selected):
            raise RuntimeError("MegaLoc descriptor rows disagree with selected identities")
        self._select_reference_names(selected)
        self._reference_descriptors = array
        self._reference_descriptors_locked = True
        self._geometry_locked = True
        self._subsets.clear()

    def _select_reference_names(self, selected: Sequence[str]) -> None:
        selected_names = tuple(str(name) for name in selected)
        (
            self._reference_names,
            self._reference_sessions,
            self._reference_paths,
            self._reference_occupied_bins,
            self._observations,
        ) = subset_reference_identities(
            names=self._reference_names,
            sessions=self._reference_sessions,
            paths=self._reference_paths,
            occupied=self._reference_occupied_bins,
            observations=self._observations,
            selected_names=selected_names,
        )

    def _drop_empty_side_references(self) -> None:
        occupied = self._reference_occupied_bins
        if not occupied:
            return
        cutoff = self._requested_min_reference_occupied_bins
        if cutoff is None:
            positive = [value for value in occupied if int(value) > 0]
            cutoff = bank_occupancy_cutoff(positive or occupied)
        self.min_reference_occupied_bins = int(cutoff)
        kept, _dropped = filter_reference_bank(
            self._reference_names, occupied, min_occupied_bins=self.min_reference_occupied_bins
        )
        if len(kept) == len(self._reference_names):
            return
        selected = tuple(self._reference_names[index] for index in kept)
        self._select_reference_names(selected)

    def _prepare(self) -> None:
        if self._prepared:
            return
        self._load_geometry()
        self._load_descriptors()
        self._prepared = True

    def _load_geometry(self) -> None:
        if self._geometry_locked and self._reference_names:
            return
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(self.map_model))
        names = tuple(sorted(str(image.name) for image in reconstruction.images.values()))
        if not names:
            raise RuntimeError("final map contains no registered reference images")
        missing = sorted(set(names) - set(self._keyframes))
        if missing:
            raise RuntimeError(f"final map image identity is absent from keyframes: {missing[:5]}")
        self._reference_names = names
        self._reference_sessions = tuple(str(self._keyframes[name]["video_id"]) for name in names)
        self._reference_paths = tuple(
            Path(str(self._keyframes[name]["image_uri"])).resolve(strict=True) for name in names
        )
        images_by_name = {str(image.name): image for image in reconstruction.images.values()}
        observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in names:
            points = [point for point in images_by_name[name].points2D if point.has_point3D()]
            observations[name] = (
                np.asarray([point.xy for point in points], dtype=np.float64).reshape(-1, 2),
                np.asarray([point.point3D_id for point in points], dtype=np.int64),
            )
        point_rows = sorted(reconstruction.points3D.items(), key=lambda item: int(item[0]))
        self._point_ids = np.asarray([int(point_id) for point_id, _ in point_rows], dtype=np.int64)
        self._point_xyz = np.asarray([point.xyz for _, point in point_rows], dtype=np.float64)
        if self.matcher_backend != "edm":
            self._point_track_len = np.asarray(
                [p.track.length() for _, p in point_rows], dtype=np.int32
            )
        else:
            self._point_track_len = np.empty(0, dtype=np.int32)
        self._observations = observations
        occupied: list[int] = []
        cam_from_world: dict[str, np.ndarray] = {}
        native_wh: dict[str, tuple[int, int]] = {}
        camera_params: dict[str, tuple[float, float, float, float]] = {}
        median_sparse_z: dict[str, float] = {}
        for name in names:
            image = images_by_name[name]
            camera = reconstruction.cameras[image.camera_id]
            pose = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)[:3, :]
            cam_from_world[name] = pose
            native_wh[name] = (int(camera.width), int(camera.height))
            params = [float(value) for value in camera.params]
            camera_params[name] = (params[0], params[1], params[2], params[3])
            count, _frac = occupied_bins(
                observations[name][0],
                width=int(camera.width),
                height=int(camera.height),
                grid=30,
            )
            occupied.append(count)
            point_ids = observations[name][1]
            if len(point_ids) == 0:
                median_sparse_z[name] = float("nan")
                continue
            world = self._point_xyz_for_ids(point_ids)
            camera_xyz = pose[:, :3] @ world.T + pose[:, 3:4]
            median_sparse_z[name] = float(np.median(camera_xyz[2]))
        self._reference_occupied_bins = tuple(occupied)
        self._cam_from_world = cam_from_world
        self._native_wh = native_wh
        self._camera_params = camera_params
        self._median_sparse_z = median_sparse_z
        del reconstruction
        gc.collect()
        self._drop_empty_side_references()
        self._geometry_locked = True

    def _load_descriptors(self) -> None:
        descriptor_root = self.cache_dir / "descriptors"
        descriptor_root.mkdir(parents=True, exist_ok=True)
        reference_path = descriptor_root / f"references-{self.fingerprint}.npy"
        reference_names_path = reference_path.with_suffix(".names.json")
        query_path = descriptor_root / f"queries-{self.fingerprint}.npy"
        query_names_path = query_path.with_suffix(".names.json")
        query_names = tuple(sorted(self._queries))
        if self._reference_descriptors_locked and self._reference_descriptors.shape[0] == len(
            self._reference_names
        ):
            references = self._reference_descriptors
        elif reference_path.is_file() and reference_names_path.is_file():
            if json.loads(reference_names_path.read_text(encoding="utf-8")) != list(
                self._reference_names
            ):
                raise RuntimeError("cached final-map reference identities changed")
            references = np.load(reference_path, allow_pickle=False)
        else:
            references = None
        if query_path.is_file() and query_names_path.is_file():
            if json.loads(query_names_path.read_text(encoding="utf-8")) != list(query_names):
                raise RuntimeError("cached held-out query identities changed")
            queries = np.load(query_path, allow_pickle=False)
        else:
            queries = self._load_precomputed_queries(query_names)
        if references is None or queries is None:
            from river_map_quality.megaloc_edm_catalog import (
                extract_megaloc_descriptors,
                load_offline_megaloc_runtime,
            )

            runtime = load_offline_megaloc_runtime(
                source=self.megaloc_source,
                checkpoint=self.megaloc_checkpoint,
                device="cuda",
            )
            if references is None:
                references = extract_megaloc_descriptors(
                    runtime,
                    self._reference_paths,
                    batch_size=self.descriptor_batch_size,
                )
            if queries is None:
                queries = extract_megaloc_descriptors(
                    runtime,
                    [Path(str(self._queries[name]["image_path"])) for name in query_names],
                    batch_size=self.descriptor_batch_size,
                )
            del runtime
            gc.collect()
            self._empty_cuda_cache()
        references = np.ascontiguousarray(references, dtype=np.float32)
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if references.shape[0] != len(self._reference_names) or queries.shape[0] != len(
            query_names
        ):
            raise RuntimeError("MegaLoc descriptor rows disagree with frozen image identities")
        if references.ndim != 2 or queries.ndim != 2 or references.shape[1] != queries.shape[1]:
            raise RuntimeError("MegaLoc query/reference descriptor dimensions disagree")
        if not reference_path.is_file():
            np.save(reference_path, references, allow_pickle=False)
            reference_names_path.write_text(
                json.dumps(list(self._reference_names), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not query_path.is_file():
            np.save(query_path, queries, allow_pickle=False)
            query_names_path.write_text(
                json.dumps(list(query_names), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._reference_descriptors = references
        self._query_descriptors = dict(zip(query_names, queries, strict=True))

    def _load_precomputed_queries(self, query_names: tuple[str, ...]) -> np.ndarray | None:
        if self.precomputed_query_descriptors is None and self.precomputed_query_names is None:
            return None
        if self.precomputed_query_descriptors is None or self.precomputed_query_names is None:
            raise ValueError("precomputed query descriptors require their names sidecar")
        names = json.loads(self.precomputed_query_names.read_text(encoding="utf-8"))
        if names != list(query_names):
            raise RuntimeError("precomputed query descriptor identities changed")
        return np.load(self.precomputed_query_descriptors, allow_pickle=False)

    def _matcher_runtime(self) -> Any:
        if self._matcher is None:
            import river_map_quality.official_edm_adapter_loo as official_edm

            _configure_audited_runtime_site_packages(self.audited_edm_site_packages)
            self._matcher = official_edm.load_official_edm_runtime(
                edm_repo=Path(str(self.edm_config["repo"])),
                checkpoint=Path(str(self.edm_config["checkpoint"])),
                config_path=Path(str(self.edm_config["model_config"])),
                data_config_path=Path(str(self.edm_config["data_config"])),
                device="cuda",
            )
        return self._matcher

    def _match_and_lift(
        self,
        query_path: Path,
        ranked: tuple[int, ...],
        *,
        query_image: Any = None,
        prepared_query: Any = None,
    ):
        if self.matcher_backend != "edm":
            return self._sparse_match_and_lift(query_image, ranked)
        import cv2
        from river_map_quality.official_edm_adapter import (
            deduplicate_lifted_matches,
            lift_reference_matches,
            lift_unmapped_with_depth,
            prepare_official_megadepth_image_from_array,
        )
        from river_map_quality.official_edm_adapter_loo import (
            prepare_official_megadepth_image,
        )
        if query_image is None:
            query_image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
            if query_image is None:
                raise FileNotFoundError(query_path)
        lifted = []
        raw_matches = 0
        matcher = self._matcher_runtime()
        if prepared_query is None:
            # Change 2: Prepare query directly from in-memory array, avoiding redundant disk read.
            # Cannot change decision because preprocessing follows the deterministic official transform.
            prepared_query = prepare_official_megadepth_image_from_array(query_image, matcher)
        threshold = float(self.edm_config.get("confidence_threshold") or 0.0)

        # Change 2: Retrieve prepared references from in-memory LRU cache.
        # Removes repeated cv2.imread and resizing of 2688x1512 reference JPEGs across calls.
        # Cannot change decision because the prepared arrays and native shapes are deterministic
        # and bitwise identical to on-the-fly preparation.
        ref_items = []
        for reference_index in ranked:
            reference_path = self._reference_paths[reference_index]
            name = self._reference_names[reference_index]
            prep_ref, ref_shape = self._get_prepared_reference(name, reference_path, matcher)
            ref_items.append((reference_index, name, prep_ref, ref_shape))

        # Change 1: Batch top_k reference pairs into a single EDM forward when EDM_BATCH_REFS=1.
        # Removes sequential model forward passes (saving ~33 ms on 5090, ~132 ms on 5060).
        # Cannot change decision because EDM natively processes mini-batches with separable outputs.
        # Gated behind EDM_BATCH_REFS (default 0) to guard against GPU cuBLAS floating-point
        # reduction order differences across different batch sizes on GPU.
        batch_enabled = os.environ.get("EDM_BATCH_REFS", "0") == "1"
        if batch_enabled and len(ref_items) > 1:
            matched_list = _match_official_prepared_batch(
                matcher, prepared_query, [item[2] for item in ref_items]
            )
        else:
            matched_list = [
                _match_official_prepared(matcher, prepared_query, item[2])
                for item in ref_items
            ]

        for (reference_index, name, prepared_reference, reference_shape), matched in zip(
            ref_items, matched_list, strict=True
        ):
            query_points = np.asarray(matched["mkpts0_f"], dtype=float).reshape(-1, 2)
            reference_points = np.asarray(matched["mkpts1_f"], dtype=float).reshape(-1, 2)
            confidences = np.asarray(matched["mconf"], dtype=float).reshape(-1)
            valid = _valid_matches(
                query_points,
                reference_points,
                confidences,
                query_shape=query_image.shape,
                reference_shape=reference_shape,
                confidence_threshold=threshold,
            )
            query_points = query_points[valid]
            reference_points = reference_points[valid]
            confidences = confidences[valid]
            geometry = _two_view_inliers(query_points, reference_points, threshold_px=3.0)
            query_points = query_points[geometry]
            reference_points = reference_points[geometry]
            confidences = confidences[geometry]
            raw_matches += len(query_points)
            observation_xy, observation_ids = self._observations[name]
            lift_result = lift_reference_matches(
                query_points=query_points,
                reference_points=reference_points,
                observation_points=observation_xy,
                observation_point3d_ids=observation_ids,
                confidences=confidences,
                maximum_distance_px=self.lift_distance_px,
                reference_name=name,
            )
            lifted.extend(lift_result.matches)
            depth = self._depth_for_reference(name)
            if depth is not None and lift_result.unmapped_pairs:
                fx, fy, cx, cy = self._camera_params[name]
                lifted.extend(
                    lift_unmapped_with_depth(
                        unmapped=lift_result.unmapped_pairs,
                        depth=depth,
                        native_wh=self._native_wh[name],
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        cam_from_world_3x4=self._cam_from_world[name],
                        median_sparse_z=self._median_sparse_z[name],
                    )
                )
        deduplicated = deduplicate_lifted_matches(
            lifted,
            query_conflict_distance_px=1.0,
        )
        return deduplicated.matches, raw_matches

    def _ensure_sparse(self) -> None:
        if self._sparse is not None:
            return
        from .sparse_reloc import SparseRelocMatcher

        self._sparse = SparseRelocMatcher(
            self.matcher_backend,
            max_keypoints=self.sparse_max_keypoints,
            ref_cache=self.sparse_ref_cache,
        )

    def sparse_stage_ms(self) -> dict[str, float]:
        if self.matcher_backend == "edm" or self._sparse is None:
            return {}
        return self._sparse.stage_ms()

    def _sparse_match_and_lift(self, query_image, ranked) -> tuple:
        import cv2
        from river_map_quality.official_edm_adapter import (
            deduplicate_lifted_matches,
            lift_reference_matches,
        )

        self._ensure_sparse()
        assert self._sparse is not None
        if query_image is None:
            raise RuntimeError("sparse reloc requires a decoded query image")
        # Query may be color (H,W,3); work resolution is its first two dims.
        work_wh = (int(query_image.shape[1]), int(query_image.shape[0]))
        # Ensure color for the sparse detector; grayscale callers pass (H,W).
        if query_image.ndim == 2:
            query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2BGR)
        query_feat = self._sparse.prepare_query(query_image)
        lifted: list = []
        raw_matches = 0
        for reference_index in ranked:
            name = self._reference_names[reference_index]
            observation_xy, observation_ids = self._observations[name]
            q_xy, r_xy, conf = self._sparse.match_reference(
                query_feat,
                name=name,
                image_path=self._reference_paths[reference_index],
                observation_xy=observation_xy,
                track_lengths=self._track_lengths_for(observation_ids),
                native_wh=self._native_wh[name],
                work_wh=work_wh,
            )
            valid = _valid_matches(
                q_xy,
                r_xy,
                conf,
                query_shape=query_image.shape,
                reference_shape=(self._native_wh[name][1], self._native_wh[name][0]),
                confidence_threshold=0.0,
            )
            q_xy, r_xy, conf = q_xy[valid], r_xy[valid], conf[valid]
            geometry = _two_view_inliers(q_xy, r_xy, threshold_px=3.0)
            q_xy, r_xy, conf = q_xy[geometry], r_xy[geometry], conf[geometry]
            raw_matches += len(q_xy)
            lifted.extend(
                lift_reference_matches(
                    query_points=q_xy,
                    reference_points=r_xy,
                    observation_points=observation_xy,
                    observation_point3d_ids=observation_ids,
                    confidences=conf,
                    maximum_distance_px=self.lift_distance_px,
                    reference_name=name,
                ).matches
            )
        deduplicated = deduplicate_lifted_matches(
            lifted,
            query_conflict_distance_px=1.0,
        )
        return deduplicated.matches, raw_matches


    def _solve(
        self,
        query_path: Path,
        matches,
        *,
        session_groups: bool = True,
        query_image: Any = None,
        query_shape: tuple[int, int] | None = None,
    ):
        import cv2
        import pycolmap
        from river_map_quality.ambiguity_localization import (
            AmbiguityConfig,
            cluster_pose_modes,
            localization_decision,
        )
        from river_map_quality.historical_inputs import (
            CURRENT_ONLY,
            _hypothesis_from_matches,
        )
        from river_map_quality.loo_metrics import compute_loo_metrics
        from river_map_quality.pose_source_runner import _estimate_pnp

        if query_image is not None:
            height, width = query_image.shape[:2]
        elif query_shape is not None:
            height, width = query_shape
        else:
            image = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(query_path)
            height, width = image.shape
        camera_params = scaled_pinhole_parameters(
            self.intrinsics_calibration,
            width=width,
            height=height,
        )
        camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=list(camera_params),
        )
        if len(matches) < 6:
            return None, {}, "REJECT_INSUFFICIENT_SUPPORT"
        image_points = np.asarray([match.query_xy for match in matches], dtype=float)
        world_points = self._world_points_for_matches(matches)
        result = _estimate_pnp(
            image_points,
            world_points,
            camera,
            max_error=float(self.thresholds["maximum_reprojection_p90_px"]),
            seed=0,
            covariance=False,
        )
        if result is None:
            return None, {}, "REJECT_PNP_FAILED"
        metrics = compute_loo_metrics(
            world_points,
            image_points,
            result.inlier_mask,
            camera_params,
            result.pose,
            result.pose,
            [match.reference_name for match in matches],
            image_size=(width, height),
        )
        inlier_mask = np.asarray(result.inlier_mask, dtype=bool)
        inlier_xy = image_points[inlier_mask]
        _occupied_30, occupied_frac_30 = occupied_bins(
            inlier_xy, width=width, height=height, grid=30
        )
        track_mask = np.asarray([match.point3d_id is not None for match in matches], dtype=bool)
        if int(track_mask.sum()) >= 8:
            inlier_ratio = float(np.mean(inlier_mask[track_mask]))
        else:
            inlier_ratio = float(np.mean(inlier_mask))
        metrics = {
            **metrics,
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "inlier_ratio": inlier_ratio,
            "n_track_2d3d": int(track_mask.sum()),
            "n_depth_2d3d": int((~track_mask).sum()),
            "track_inliers": int(np.count_nonzero(inlier_mask & track_mask)),
            "occupied_bins_30": int(_occupied_30),
            "occupied_frac_30": float(occupied_frac_30),
        }
        hypotheses = [
            _hypothesis_from_matches(
                hypothesis_id=f"{query_path.name}:union",
                group_id="final_map_union",
                pose=result.pose,
                matches=matches,
                inlier_mask=result.inlier_mask,
                metrics=metrics,
                source_role=CURRENT_ONLY,
            )
        ]
        if session_groups:
            # Change 4: Skip redundant session-group PnP when union PnP already decisively
            # satisfies every STRONG gate.
            # Removes redundant per-session pycolmap PnP solves and LOO metric computations
            # (saving 20-55 ms on multi-session queries).
            # Cannot change decision on unimodal frames where union pose is decisively strong.
            # Guarded behind EDM_SKIP_SESSION_PNP (default 0) because in rare multimodal
            # scenarios, competing session poses could theoretically trigger REJECT_MULTIMODAL.
            skip_session_pnp = os.environ.get("EDM_SKIP_SESSION_PNP", "0") == "1"
            union_is_strong = (
                result is not None
                and localization_is_strong(
                    metrics,
                    decision_status="ACCEPT",
                    thresholds=self.thresholds,
                )
            )
            if not (skip_session_pnp and union_is_strong):
                groups: dict[str, list[Any]] = {}
                for match in matches:
                    name = match.reference_name
                    if name == TEMPORAL_REFERENCE_NAME or name not in self._keyframes:
                        continue
                    session = str(self._keyframes[name]["video_id"])
                    groups.setdefault(session, []).append(match)
                if len(groups) > 1:
                    for session, group_matches in sorted(groups.items()):
                        if len(group_matches) < 6:
                            continue
                        if len(group_matches) == len(matches):
                            continue
                        group_image = np.asarray([match.query_xy for match in group_matches], dtype=float)
                        group_world = self._world_points_for_matches(group_matches)
                        group_result = _estimate_pnp(
                            group_image,
                            group_world,
                            camera,
                            max_error=float(self.thresholds["maximum_reprojection_p90_px"]),
                            seed=0,
                            covariance=False,
                        )
                        if group_result is None:
                            continue
                        group_metrics = compute_loo_metrics(
                            group_world,
                            group_image,
                            group_result.inlier_mask,
                            camera_params,
                            group_result.pose,
                            group_result.pose,
                            [match.reference_name for match in group_matches],
                            image_size=(width, height),
                        )
                        hypotheses.append(
                            _hypothesis_from_matches(
                                 hypothesis_id=f"{query_path.name}:{session}",
                                 group_id=session,
                                 pose=group_result.pose,
                                 matches=group_matches,
                                 inlier_mask=group_result.inlier_mask,
                                 metrics=group_metrics,
                                 source_role=CURRENT_ONLY,
                            )
                        )
        config = AmbiguityConfig()
        modes = cluster_pose_modes(hypotheses, config=config)
        decision = localization_decision(modes, reject_multimodal=True, config=config)
        self.last_modes = tuple(modes)
        self.last_mode_decision = decision
        return result, metrics, decision.status


    def _world_points_for_matches(self, matches) -> np.ndarray:
        rows = np.empty((len(matches), 3), dtype=np.float64)
        sparse_ids: list[int] = []
        sparse_index: list[int] = []
        for index, match in enumerate(matches):
            if match.xyz is not None:
                rows[index] = np.asarray(match.xyz, dtype=np.float64)
                continue
            if match.point3d_id is None:
                raise RuntimeError("lifted match is missing both xyz and point3d_id")
            sparse_ids.append(int(match.point3d_id))
            sparse_index.append(index)
        if sparse_ids:
            rows[np.asarray(sparse_index, dtype=np.int64)] = self._point_xyz_for_ids(
                np.asarray(sparse_ids, dtype=np.int64)
            )
        return rows

    def _depth_for_reference(self, name: str) -> np.ndarray | None:
        if self._reference_depth_dir is None:
            return None
        cached = self._depth_cache.get(name)
        if cached is not None:
            self._depth_cache.move_to_end(name)
            return cached
        path = self._reference_depth_dir / f"{name}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as payload:
            depth = np.asarray(payload["depth"], dtype=np.float32)
        self._depth_cache[name] = depth
        while len(self._depth_cache) > 8:
            self._depth_cache.popitem(last=False)
        return depth

    def _get_prepared_reference(
        self,
        name: str,
        path: Path,
        runtime: Any,
    ) -> tuple[Any, tuple[int, int]]:
        """Retrieve prepared reference image and its native shape, using bounded LRU cache.

        Change 2: Removes repeated disk reads (cv2.imread) and preprocessing transforms
        of static 2688x1512 reference JPEGs across relocalizations.
        Cannot change the localization decision because the cached prepared representation
        is deterministic and bitwise identical to on-the-fly preparation.
        """
        return self._prepared_reference_cache.get_or_prepare(name, path, runtime)

    def _point_xyz_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(self._point_ids, point_ids)
        if np.any(indices >= len(self._point_ids)) or not np.array_equal(
            self._point_ids[indices], point_ids
        ):
            raise RuntimeError("lifted point ID is absent from the final map")
        return self._point_xyz[indices]

    def _track_lengths_for(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0,), dtype=np.int32)
        slot = np.searchsorted(self._point_ids, ids)
        clipped = np.clip(slot, 0, max(len(self._point_ids) - 1, 0))
        present = (
            len(self._point_ids) > 0 and np.array_equal(self._point_ids[clipped], ids)
        )
        if not present:
            # Fall back per-id; missing ids get length 0 (never selected over real tracks).
            out = np.zeros(ids.shape, dtype=np.int32)
            table = dict(
                zip(self._point_ids.tolist(), self._point_track_len.tolist(), strict=False)
            )
            for i, v in enumerate(ids.tolist()):
                out[i] = int(table.get(int(v), 0))
            return out
        return np.asarray(self._point_track_len[clipped], dtype=np.int32)

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def create_edm_provider(**kwargs: Any) -> FinalMapEDMProvider:
    """Factory allowlisted by ``localization_worker``."""

    return FinalMapEDMProvider(**kwargs)


def _prepare_official_image(
    path: Path,
    runtime: Any,
    *,
    prepare_image: Any = None,
    image: np.ndarray | None = None,
):
    import cv2

    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
    from river_map_quality.official_edm_adapter import prepare_official_image as _prep

    return _prep(path, runtime, prepare_image=prepare_image, image=image)


def _prepare_official_pair(
    query_path: Path,
    reference_path: Path,
    runtime: Any,
    *,
    prepare_image,
):
    """Prepare mixed-resolution pairs with independent official EDM transforms."""

    return (
        _prepare_official_image(query_path, runtime, prepare_image=prepare_image),
        _prepare_official_image(reference_path, runtime, prepare_image=prepare_image),
    )


def _match_official_prepared(runtime: Any, query_image: Any, reference_image: Any):
    """Run the official square-padding/mask path for independently scaled images."""
    from river_map_quality.official_edm_adapter import match_official_prepared

    return match_official_prepared(runtime, query_image, reference_image)


def _match_official_prepared_batch(
    runtime: Any, query_image: Any, reference_images: Sequence[Any]
):
    """Run batched official EDM forward pass for top_k reference pairs.

    Change 1: Batches top_k references into a single model forward pass.
    Removes sequential per-reference model forwards (saving ~33 ms on 5090, ~132 ms on 5060).
    Cannot change decision because EDM natively processes mini-batches with separable outputs.
    Gated behind EDM_BATCH_REFS (default 0) to guard against GPU cuBLAS floating-point
    reduction order differences across different batch sizes on GPU.
    """
    from river_map_quality.official_edm_adapter import match_official_prepared_batch

    return match_official_prepared_batch(runtime, query_image, reference_images)


def _valid_matches(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    confidences: np.ndarray,
    *,
    query_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    confidence_threshold: float,
) -> np.ndarray:
    query = np.asarray(query_points, dtype=float).reshape(-1, 2)
    reference = np.asarray(reference_points, dtype=float).reshape(-1, 2)
    scores = np.asarray(confidences, dtype=float).reshape(-1)
    if len(query) != len(reference) or len(query) != len(scores):
        raise RuntimeError("EDM correspondence arrays have different lengths")
    return (
        np.isfinite(query).all(axis=1)
        & np.isfinite(reference).all(axis=1)
        & np.isfinite(scores)
        & (scores >= confidence_threshold)
        & (query[:, 0] >= 0)
        & (query[:, 0] < query_shape[1])
        & (query[:, 1] >= 0)
        & (query[:, 1] < query_shape[0])
        & (reference[:, 0] >= 0)
        & (reference[:, 0] < reference_shape[1])
        & (reference[:, 1] >= 0)
        & (reference[:, 1] < reference_shape[0])
    )


def _two_view_inliers(query_points: np.ndarray, reference_points: np.ndarray, *, threshold_px: float = 3.0) -> np.ndarray:
    """Keep MAGSAC fundamental inliers so 3D lift runs only on geometrically consistent pairs."""
    import cv2

    query = np.asarray(query_points, dtype=np.float32).reshape(-1, 2)
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    n = len(query)
    if n < 8:
        return np.ones(n, dtype=bool)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    try:
        matrix, mask = cv2.findFundamentalMat(query, reference, method, threshold_px, 0.999, 10000)
    except Exception:
        return np.ones(n, dtype=bool)
    if matrix is None or mask is None:
        return np.ones(n, dtype=bool)
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    if int(keep.sum()) < 30:
        return np.ones(n, dtype=bool)
    return keep


def _keyframe_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["output_name"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("keyframe output names must be unique")
    return result


def _query_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = {str(row["query_id"]): row for row in rows}
    if not result or len(result) != len(rows):
        raise ValueError("held-out query IDs must be non-empty and unique")
    if any(not Path(str(row.get("image_path") or "")).is_file() for row in rows):
        raise FileNotFoundError("one or more held-out query images are unavailable")
    return result


def _pose_fields(pose: np.ndarray | None):
    if pose is None:
        return None, None, None, None
    matrix = np.asarray(pose, dtype=float)
    rotation_wc = matrix[:3, :3].T
    center = -rotation_wc @ matrix[:3, 3]
    forward = rotation_wc[:, 2]
    yaw = math.degrees(math.atan2(float(forward[1]), float(forward[0]))) % 360.0
    pitch = math.degrees(math.asin(float(np.clip(forward[2], -1.0, 1.0))))
    return (
        tuple(float(value) for value in center),
        tuple(tuple(float(value) for value in row) for row in rotation_wc),
        yaw,
        pitch,
    )


def _optional_file(value: str | None) -> Path | None:
    return None if value is None else Path(value).resolve(strict=True)


def _model_hashes(model: Path) -> dict[str, str]:
    names = ("cameras.bin", "images.bin", "points3D.bin")
    if any(not (model / name).is_file() for name in names):
        raise FileNotFoundError("final map is not a complete binary COLMAP model")
    return {name: _sha256_file(model / name) for name in names}


def _query_image_set_sha256(queries: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for query_id, row in sorted(queries.items()):
        path = Path(str(row["image_path"])).resolve(strict=True)
        digest.update(query_id.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


__all__ = [
    "FinalMapEDMProvider",
    "build_localization_payload",
    "create_edm_provider",
    "evaluate_localization_admission",
    "localization_is_strong",
    "rank_reference_indices",
    "scaled_pinhole_parameters",
    "subset_reference_identities",
]
