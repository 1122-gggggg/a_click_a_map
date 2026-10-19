"""Historical-view extraction, direct registration, exclusion, and stability masks.

P116/P117 remain update-only.  Runtime lifting accepts only calibrated `stable`
pixels; changed, dynamic, and uncertain pixels are hard-excluded before PnP.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from river_map_quality.adaptive_sampling import (
    MOTION_THRESHOLDS,
    SamplingInvariantError,
    scan_motion,
    select_sample_frames,
    validate_frame_records,
)
from river_map_quality.ambiguity_localization import (
    ATOMIC_SUPPORT_PROVENANCE,
    CURRENT_ONLY,
    HISTORICAL_ONLY,
    MIXED,
    AmbiguityConfig,
    PoseHypothesisRecord,
    cluster_pose_modes,
    localization_decision,
)
from river_map_quality.exclusion import parse_reference_name
from river_map_quality.historical_experiment import (
    HistoricalExperimentError,
    assert_b0_unchanged,
)
from river_map_quality.loo_metrics import compute_loo_metrics
from river_map_quality.megaloc_edm_catalog import (
    FrozenReferenceCatalog,
    MegaLocRuntime,
    extract_megaloc_descriptors,
)
from river_map_quality.official_edm_adapter import (
    LiftedMatch,
    UnmappedPair,
    deduplicate_lifted_matches,
    lift_reference_matches,
)
from river_map_quality.official_edm_adapter_loo import OfficialEdmRuntime
from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    BASE_VIDEO_NAMES,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    UPDATE_VIDEO_NAMES,
    write_native_frame,
    write_new_json,
)

HISTORICAL_EXTRACTION_SCHEMA = "RIVER_HISTORICAL_FRAME_EXTRACTION_V1"
DIRECT_REGISTRATION_SCHEMA = "RIVER_HISTORICAL_DIRECT_REGISTRATION_V1"
STABILITY_MASK_SCHEMA = "RIVER_HISTORICAL_STABILITY_MASK_V1"
HISTORICAL_NEIGHBOR_SECONDS = 5.0
STABLE = "stable"
CHANGED = "changed"
DYNAMIC = "dynamic"
UNCERTAIN = "uncertain"
DIRECT_STRONG = "DIRECT_STRONG"
DIRECT_PROVISIONAL_BRIDGE_ONLY = "DIRECT_PROVISIONAL_BRIDGE_ONLY"
RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
GEOMETRY_WEAK = "GEOMETRY_WEAK"
AMBIGUOUS_MULTIMODAL = "AMBIGUOUS_MULTIMODAL"
LOST = "LOST"


class HistoricalInputError(HistoricalExperimentError):
    """Raised when historical extraction or registration would cross a role gate."""


@dataclass(frozen=True)
class HistoricalExclusion:
    """Exact-byte, same-frame, ±5 s, and self-candidate exclusion for historical LOO."""

    query_name: str
    query_sha256: str
    source_video: str
    source_pts_seconds: float
    source_frame_index: int

    def excludes(
        self,
        *,
        name: str,
        image_sha256: str | None = None,
        source_video: str | None = None,
        source_pts_seconds: float | None = None,
        source_frame_index: int | None = None,
    ) -> bool:
        if name == self.query_name:
            return True
        if image_sha256 is not None and image_sha256 == self.query_sha256:
            return True
        if source_video == self.source_video and source_frame_index == self.source_frame_index:
            return True
        neighbor = (
            source_video == self.source_video
            and source_pts_seconds is not None
            and abs(float(source_pts_seconds) - self.source_pts_seconds)
            <= HISTORICAL_NEIGHBOR_SECONDS
        )
        return neighbor


def historical_exclusion_for(record: Mapping[str, Any]) -> HistoricalExclusion:
    return HistoricalExclusion(
        query_name=str(record["output_name"]),
        query_sha256=str(record["image_sha256"]),
        source_video=str(record.get("video") or Path(str(record["output_name"])).parts[0] + ".MP4"),
        source_pts_seconds=float(record["source_pts_seconds"]),
        source_frame_index=int(record["source_frame_index"]),
    )


def _locked_update_sources(run_root: Path, raw_root: Path) -> dict[str, Mapping[str, object]]:
    lock_path = run_root / "input_lock/source_videos.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    update = source_lock["update"]
    if tuple(update) != UPDATE_VIDEO_NAMES:
        raise HistoricalInputError("locked historical update corpus membership changed")
    for name in UPDATE_VIDEO_NAMES:
        path = raw_root / "update" / name
        current = fingerprint_file(path, sha256=True)
        expected = update[name]
        if current.sha256 != expected["sha256"] or current.size != expected["size"]:
            raise HistoricalInputError(f"locked historical source changed: {path}")
        if str(path) != expected["path"]:
            raise HistoricalInputError(f"historical source path differs from lock: {path}")
    return update


def extract_historical_frames(
    *,
    run_root: Path,
    raw_root: Path,
    camera_matrix: np.ndarray,
    b0_receipt: Mapping[str, Any],
    b0_root: Path,
) -> dict[str, object]:
    """Run the existing PTS/motion classifier on P116/P117 and publish native frames."""

    run_root = run_root.resolve(strict=True)
    raw_root = raw_root.resolve(strict=True)
    assert_b0_unchanged(b0_root, b0_receipt)
    image_root = run_root / "historical/images"
    if (run_root / "historical/frame_manifest.json").exists():
        raise FileExistsError("historical frame manifest already exists")
    if any(image_root.iterdir()):
        raise HistoricalInputError("historical image directory must start empty")
    source_lock = _locked_update_sources(run_root, raw_root)
    intervals_by_video: dict[str, list[dict[str, object]]] = {}
    selections: list[dict[str, object]] = []
    for name in UPDATE_VIDEO_NAMES:
        intervals = scan_motion(
            raw_root / "update" / name,
            camera_matrix=np.asarray(camera_matrix, dtype=float),
            analysis_interval_seconds=float(MOTION_THRESHOLDS["analysis_interval_seconds"]),
        )
        for row in intervals:
            row["video"] = name
        intervals_by_video[name] = intervals
        selections.extend(select_sample_frames(intervals))

    selected_by_video: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for selection in selections:
        selected_by_video[str(selection["video"])][int(selection["source_frame_index"])] = selection

    stage = Path(tempfile.mkdtemp(prefix=".historical-images.", dir=run_root / "historical"))
    try:
        records: list[dict[str, object]] = []
        for name in UPDATE_VIDEO_NAMES:
            selected = selected_by_video.get(name, {})
            if not selected:
                continue
            capture = cv2.VideoCapture(str(raw_root / "update" / name))
            if not capture.isOpened():
                raise FileNotFoundError(raw_root / "update" / name)
            frame_index = 0
            try:
                while selected:
                    ok, pixels = capture.read()
                    if not ok:
                        break
                    selection = selected.pop(frame_index, None)
                    if selection is not None:
                        pts_seconds = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                        source_pts = int(selection["source_pts"])
                        filename = f"pts_{source_pts:012d}_frame_{frame_index:06d}.jpg"
                        relative = Path(Path(name).stem) / filename
                        written = write_native_frame(
                            pixels,
                            stage / relative,
                            width=NATIVE_WIDTH,
                            height=NATIVE_HEIGHT,
                            writer=lambda path, image: cv2.imwrite(
                                path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                            ),
                        )
                        records.append(
                            {
                                "video": name,
                                "video_sha256": source_lock[name]["sha256"],
                                "source_pts": source_pts,
                                "source_pts_seconds": pts_seconds,
                                "source_frame_index": frame_index,
                                "output_name": relative.as_posix(),
                                "image_sha256": written["image_sha256"],
                                "motion_class": selection["motion_class"],
                                "role": selection["role"],
                                "source_role": "historical",
                                "width": NATIVE_WIDTH,
                                "height": NATIVE_HEIGHT,
                            }
                        )
                    frame_index += 1
            finally:
                capture.release()
            if selected:
                raise SamplingInvariantError(
                    f"historical source indexes were not decoded from {name}: {sorted(selected)}"
                )
        validate_frame_records(records)
        for record in records:
            destination = image_root / str(record["output_name"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / str(record["output_name"]), destination)
        manifest = {
            "schema_version": 1,
            "artifact_type": HISTORICAL_EXTRACTION_SCHEMA,
            "native_pixels_only": True,
            "resize_permitted": False,
            "undistort_permitted": False,
            "fast_motion_dropped": True,
            "videos": intervals_by_video,
            "frames": records,
        }
        write_new_json(run_root, "historical/frame_manifest.json", manifest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    assert_b0_unchanged(b0_root, b0_receipt)
    return manifest


def rank_current_references(
    query_descriptor: np.ndarray,
    catalog: FrozenReferenceCatalog,
    *,
    topk: int,
    excluded_names: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    query = np.asarray(query_descriptor, dtype=np.float32).reshape(-1)
    query = query / float(np.linalg.norm(query))
    scores = catalog.descriptors @ query
    blocked = set(excluded_names)
    order = [
        int(index)
        for index in np.argsort(-scores, kind="stable")
        if catalog.names[int(index)] not in blocked
    ][:topk]
    return tuple(catalog.names[index] for index in order), tuple(float(scores[index]) for index in order)

def _partition_current_references(names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        parsed = parse_reference_name(name)
        groups[parsed.sequence or "unknown"].append(name)
    return {key: tuple(value) for key, value in sorted(groups.items())}


def _hypothesis_from_matches(
    *,
    hypothesis_id: str,
    group_id: str,
    pose: np.ndarray,
    matches: Sequence[LiftedMatch],
    inlier_mask: np.ndarray,
    metrics: Mapping[str, Any],
    source_role: str,
) -> PoseHypothesisRecord:
    inliers = int(np.count_nonzero(inlier_mask))
    return PoseHypothesisRecord(
        hypothesis_id=hypothesis_id,
        reference_group_id=group_id,
        pose=np.asarray(pose, dtype=float),
        reference_ids=tuple(sorted({match.reference_name for match in matches})),
        raw_matches=len(matches),
        verified_matches=len(matches),
        unique_2d3d=len({match.point3d_id for match in matches}),
        pnp_inliers=inliers,
        pnp_inlier_ratio=inliers / len(matches) if matches else 0.0,
        positive_depth_ratio=float(metrics.get("positive_depth_ratio") or 0.0),
        reprojection_mean=metrics.get("reprojection_mean"),
        reprojection_median=metrics.get("reprojection_median"),
        reprojection_p90=metrics.get("reprojection_p90"),
        convex_hull_coverage=float(metrics.get("convex_hull_coverage") or 0.0),
        grid_occupancy=int(metrics.get("occupancy_4x4") or 0),
        fim_condition=metrics.get("scaled_fim_condition"),
        fim_lambda_min=metrics.get("scaled_fim_lambda_min"),
        track_statistics={},
        parallax_statistics={},
        support_provenance=ATOMIC_SUPPORT_PROVENANCE,
        source_role=source_role,
    )


def classify_direct_registration(
    *,
    metrics: Mapping[str, Any],
    decision_status: str,
    retrieved: int,
    lifted: int,
    thresholds: Mapping[str, Any],
) -> str:
    if retrieved <= 0:
        return LOST
    if lifted <= 0:
        return RETRIEVAL_ONLY
    if decision_status == "REJECT_MULTIMODAL":
        return AMBIGUOUS_MULTIMODAL
    if decision_status.startswith("REJECT"):
        return GEOMETRY_WEAK
    if decision_status != "ACCEPT":
        return GEOMETRY_WEAK
    inliers = int(metrics.get("inlier_count") or 0)
    if (
        inliers >= int(thresholds["strong_inliers"])
        and float(metrics.get("inlier_ratio") or 0.0) >= float(thresholds["minimum_inlier_ratio"])
        and float(metrics.get("convex_hull_coverage") or 0.0)
        >= float(thresholds["minimum_hull_coverage"])
        and int(metrics.get("occupancy_4x4") or 0) >= int(thresholds["minimum_occupancy_4x4"])
        and float(metrics.get("positive_depth_ratio") or 0.0)
        >= float(thresholds["minimum_positive_depth_ratio"])
        and float(metrics.get("reprojection_p90") or math.inf)
        <= float(thresholds["maximum_reprojection_p90_px"])
    ):
        return DIRECT_STRONG
    if inliers >= int(thresholds["provisional_bridge_inliers"]):
        return DIRECT_PROVISIONAL_BRIDGE_ONLY
    return GEOMETRY_WEAK


def register_historical_view(
    *,
    query_name: str,
    query_image: Path,
    query_descriptor: np.ndarray,
    catalog: FrozenReferenceCatalog,
    current_image_root: Path,
    runtime: OfficialEdmRuntime,
    camera_params: tuple[float, float, float, float],
    thresholds: Mapping[str, Any],
    estimate_pose,
    topk: int = 8,
    lift_distance_px: float = 2.0,
    historical_image_root: Path | None = None,
) -> dict[str, object]:
    """Retrieve current B0 references, lift only current observations, and classify."""

    names, scores = rank_current_references(
        query_descriptor,
        catalog,
        topk=topk,
        excluded_names=(query_name,),
    )
    groups = _partition_current_references(names)
    prepared_query = runtime.prepare_image(
        query_image.parent,
        query_image.name,
        native_width=NATIVE_WIDTH,
        native_height=NATIVE_HEIGHT,
    )
    lifted: list[LiftedMatch] = []
    unmapped: list[UnmappedPair] = []
    raw_matches = 0
    for reference_name in names:
        reference_root = current_image_root
        if not (current_image_root / reference_name).is_file() and historical_image_root is not None:
            reference_root = historical_image_root
        prepared_reference = runtime.prepare_image(
            reference_root,
            reference_name,
            native_width=NATIVE_WIDTH,
            native_height=NATIVE_HEIGHT,
        )
        query_pts, reference_pts, confidences = runtime.match(
            prepared_query,
            prepared_reference,
            native_width=NATIVE_WIDTH,
            native_height=NATIVE_HEIGHT,
        )
        raw_matches += len(query_pts)
        observation_xy, observation_ids = catalog.observation_slice(reference_name)
        lift_result = lift_reference_matches(
            query_points=query_pts,
            reference_points=reference_pts,
            observation_points=observation_xy,
            observation_point3d_ids=observation_ids,
            confidences=confidences,
            maximum_distance_px=lift_distance_px,
            reference_name=reference_name,
        )
        lifted.extend(lift_result.matches)
        unmapped.extend(lift_result.unmapped_pairs)
    deduped = deduplicate_lifted_matches(lifted, query_conflict_distance_px=1.0)
    hypotheses: list[PoseHypothesisRecord] = []
    union_estimate = None
    if deduped.matches:
        image_points = np.asarray([match.query_xy for match in deduped.matches], dtype=float)
        world_points = catalog.point_xyz_for_ids(
            np.asarray([match.point3d_id for match in deduped.matches], dtype=np.int64)
        )
        union_estimate = estimate_pose(image_points, world_points)
        if union_estimate is not None:
            metrics = compute_loo_metrics(
                world_points,
                image_points,
                union_estimate["inlier_mask"],
                camera_params,
                union_estimate["pose"],
                union_estimate["pose"],
                [match.reference_name for match in deduped.matches],
                image_size=(NATIVE_WIDTH, NATIVE_HEIGHT),
            )
            hypotheses.append(
                _hypothesis_from_matches(
                    hypothesis_id=f"{query_name}:union",
                    group_id="current_union",
                    pose=union_estimate["pose"],
                    matches=deduped.matches,
                    inlier_mask=union_estimate["inlier_mask"],
                    metrics=metrics,
                    source_role=CURRENT_ONLY,
                )
            )
        for group_id, group_names in groups.items():
            group_matches = [
                match for match in deduped.matches if match.reference_name in group_names
            ]
            if len(group_matches) < 6:
                continue
            group_image = np.asarray([match.query_xy for match in group_matches], dtype=float)
            group_world = catalog.point_xyz_for_ids(
                np.asarray([match.point3d_id for match in group_matches], dtype=np.int64)
            )
            estimate = estimate_pose(group_image, group_world)
            if estimate is None:
                continue
            group_metrics = compute_loo_metrics(
                group_world,
                group_image,
                estimate["inlier_mask"],
                camera_params,
                estimate["pose"],
                estimate["pose"],
                [match.reference_name for match in group_matches],
                image_size=(NATIVE_WIDTH, NATIVE_HEIGHT),
            )
            hypotheses.append(
                _hypothesis_from_matches(
                    hypothesis_id=f"{query_name}:{group_id}",
                    group_id=group_id,
                    pose=estimate["pose"],
                    matches=group_matches,
                    inlier_mask=estimate["inlier_mask"],
                    metrics=group_metrics,
                    source_role=CURRENT_ONLY,
                )
            )
    modes = cluster_pose_modes(hypotheses, config=AmbiguityConfig())
    decision = localization_decision(modes, reject_multimodal=True, config=AmbiguityConfig())
    selected_metrics = (
        compute_loo_metrics(
            catalog.point_xyz_for_ids(
                np.asarray([match.point3d_id for match in deduped.matches], dtype=np.int64)
            ),
            np.asarray([match.query_xy for match in deduped.matches], dtype=float),
            union_estimate["inlier_mask"],
            camera_params,
            union_estimate["pose"],
            union_estimate["pose"],
            [match.reference_name for match in deduped.matches],
            image_size=(NATIVE_WIDTH, NATIVE_HEIGHT),
        )
        if union_estimate is not None
        else {}
    )
    if union_estimate is not None:
        selected_metrics = {
            **selected_metrics,
            "inlier_count": int(np.count_nonzero(union_estimate["inlier_mask"])),
            "inlier_ratio": float(np.mean(union_estimate["inlier_mask"]))
            if len(union_estimate["inlier_mask"])
            else 0.0,
        }
    status = classify_direct_registration(
        metrics=selected_metrics,
        decision_status=decision.status,
        retrieved=len(names),
        lifted=len(deduped.matches),
        thresholds=thresholds,
    )
    inlier_mask = (
        np.asarray(union_estimate["inlier_mask"], dtype=bool)
        if union_estimate is not None
        else np.zeros(len(deduped.matches), dtype=bool)
    )
    inlier_observations = [
        {
            "query_xy": [float(match.query_xy[0]), float(match.query_xy[1])],
            "point3d_id": int(match.point3d_id),
            "reference_name": match.reference_name,
            "confidence": float(match.confidence),
        }
        for match, keep in zip(deduped.matches, inlier_mask, strict=True)
        if bool(keep)
    ]
    unmapped_pairs = [
        {
            "query_xy": [float(pair.query_xy[0]), float(pair.query_xy[1])],
            "reference_xy": [float(pair.reference_xy[0]), float(pair.reference_xy[1])],
            "reference_name": pair.reference_name,
            "confidence": float(pair.confidence),
        }
        for pair in unmapped
    ]
    return {
        "schema_version": 1,
        "artifact_type": DIRECT_REGISTRATION_SCHEMA,
        "query_name": query_name,
        "status": status,
        "retrieved_names": list(names),
        "retrieved_scores": list(scores),
        "raw_matches": raw_matches,
        "lifted_correspondences": len(deduped.matches),
        "unique_point3d_count": len({match.point3d_id for match in deduped.matches}),
        "decision": {
            "status": decision.status,
            "selected_mode_id": decision.selected_mode_id,
            "current_first_state": decision.current_first_state,
            "source_role": decision.source_role or CURRENT_ONLY,
        },
        "metrics": selected_metrics,
        "pose": None if union_estimate is None else np.asarray(union_estimate["pose"]).tolist(),
        "source_role": HISTORICAL_ONLY if status != DIRECT_STRONG else MIXED,
        "inlier_observations": inlier_observations,
        "unmapped_pairs": unmapped_pairs,
    }


def _patch_grid(height: int, width: int, *, patch: int = 14) -> tuple[int, int]:
    return max(1, height // patch), max(1, width // patch)


def calibrate_stability_thresholds(
    current_current_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Calibrate only on healthy pose-aligned current-current pairs."""

    if not current_current_pairs:
        raise HistoricalInputError("stability calibration requires current-current pairs")
    if any(str(row.get("source_role")) != "current" for row in current_current_pairs):
        raise HistoricalInputError("P116/P117/P157 cannot enter stability-threshold fitting")
    dino = np.asarray([float(row["dino_cosine_distance"]) for row in current_current_pairs])
    ssim = np.asarray([float(row["grayscale_ssim"]) for row in current_current_pairs])
    residual = np.asarray([float(row["warp_residual_px"]) for row in current_current_pairs])
    return {
        "stable_dino_max": float(np.percentile(dino, 90)),
        "changed_dino_min": float(np.percentile(dino, 99)),
        "stable_ssim_min": float(np.percentile(ssim, 10)),
        "changed_ssim_max": float(np.percentile(ssim, 1)),
        "stable_warp_residual_max": float(np.percentile(residual, 90)),
        "changed_warp_residual_min": float(np.percentile(residual, 99)),
    }


def classify_stability_patch(
    *,
    flight_evidence: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> str:
    """Aggregate independent current-flight evidence into one hard mask label."""

    stable_flights = 0
    changed_flights = 0
    dynamic = 0
    for row in flight_evidence:
        if not bool(row.get("positive_depth")) or int(row.get("independent_support") or 0) < 1:
            continue
        dino = float(row["dino_cosine_distance"])
        ssim = float(row["grayscale_ssim"])
        residual = float(row["warp_residual_px"])
        if bool(row.get("unsupported_local_motion")):
            dynamic += 1
            continue
        if (
            dino <= float(thresholds["stable_dino_max"])
            and ssim >= float(thresholds["stable_ssim_min"])
            and residual <= float(thresholds["stable_warp_residual_max"])
        ):
            stable_flights += 1
        if (
            dino >= float(thresholds["changed_dino_min"])
            and ssim <= float(thresholds["changed_ssim_max"])
            and residual >= float(thresholds["changed_warp_residual_min"])
        ):
            changed_flights += 1
    if stable_flights >= 2:
        return STABLE
    if changed_flights >= 2:
        return CHANGED
    if dynamic >= 2:
        return DYNAMIC
    return UNCERTAIN


def build_stability_mask(
    *,
    height: int,
    width: int,
    patch_evidence: Mapping[tuple[int, int], Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
) -> dict[str, object]:
    rows, cols = _patch_grid(height, width)
    labels = np.full((rows, cols), UNCERTAIN, dtype=object)
    for (row, col), evidence in patch_evidence.items():
        labels[row, col] = classify_stability_patch(
            flight_evidence=evidence,
            thresholds=thresholds,
        )
    return {
        "schema_version": 1,
        "artifact_type": STABILITY_MASK_SCHEMA,
        "runtime_accepts": STABLE,
        "hard_excluded": [CHANGED, DYNAMIC, UNCERTAIN],
        "shape": [rows, cols],
        "labels": labels.tolist(),
        "thresholds": dict(thresholds),
        "calibration_corpus": list(BASE_VIDEO_NAMES),
    }


def filter_matches_by_stable_mask(
    matches: Sequence[LiftedMatch],
    mask: Mapping[str, Any],
    *,
    image_size: tuple[int, int] = (NATIVE_WIDTH, NATIVE_HEIGHT),
) -> tuple[LiftedMatch, ...]:
    labels = np.asarray(mask["labels"], dtype=object)
    rows, cols = labels.shape
    width, height = image_size
    accepted: list[LiftedMatch] = []
    for match in matches:
        col = min(cols - 1, max(0, int(match.query_xy[0] * cols / width)))
        row = min(rows - 1, max(0, int(match.query_xy[1] * rows / height)))
        if labels[row, col] == STABLE:
            accepted.append(match)
    return tuple(accepted)


def extract_historical_descriptors(
    *,
    records: Sequence[Mapping[str, Any]],
    image_root: Path,
    runtime: MegaLocRuntime,
) -> np.ndarray:
    paths = [image_root / str(record["output_name"]) for record in records]
    return extract_megaloc_descriptors(runtime, paths)


__all__ = [
    "AMBIGUOUS_MULTIMODAL",
    "CHANGED",
    "DIRECT_PROVISIONAL_BRIDGE_ONLY",
    "DIRECT_STRONG",
    "DYNAMIC",
    "GEOMETRY_WEAK",
    "HISTORICAL_NEIGHBOR_SECONDS",
    "LOST",
    "RETRIEVAL_ONLY",
    "STABLE",
    "UNCERTAIN",
    "HistoricalExclusion",
    "HistoricalInputError",
    "build_stability_mask",
    "calibrate_stability_thresholds",
    "classify_direct_registration",
    "classify_stability_patch",
    "extract_historical_descriptors",
    "extract_historical_frames",
    "filter_matches_by_stable_mask",
    "historical_exclusion_for",
    "rank_current_references",
    "register_historical_view",
]
