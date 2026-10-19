"""Evidence contracts for the official-EDM + MegaLoc + COLMAP adapter.

This adapter is a new diagnostic path.  It is deliberately not the deleted
production-EDM runtime and its outputs must never be consumed as production HL
(localizer-behavior) evidence.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OFFICIAL_EDM_ADAPTER_LINEAGE = "OFFICIAL_EDM_MEGALOC_COLMAP_ADAPTER_LOO_V1"
ADAPTER_EVIDENCE_SCOPE = "ADAPTER_EVIDENCE_ONLY_NOT_PRODUCTION_HL"


@dataclass(frozen=True)
class AdapterThresholds:
    min_retrieved_refs: int = 1
    min_raw_matches: int = 1
    min_lifted_correspondences: int = 6
    min_pnp_inliers: int = 80
    max_normalized_position_error: float = 0.02
    max_rotation_error_deg: float = 2.0

    def __post_init__(self) -> None:
        if min(
            self.min_retrieved_refs,
            self.min_raw_matches,
            self.min_lifted_correspondences,
            self.min_pnp_inliers,
        ) <= 0:
            raise ValueError("adapter count thresholds must be positive")
        if self.max_normalized_position_error <= 0 or self.max_rotation_error_deg <= 0:
            raise ValueError("adapter pose thresholds must be positive")


@dataclass(frozen=True)
class AdapterEvidence:
    completed: bool
    retrieved_refs: int | None = None
    raw_matches: int | None = None
    lifted_correspondences: int | None = None
    pnp_succeeded: bool | None = None
    pnp_inliers: int | None = None
    ground_truth_pose_valid: bool | None = None
    normalized_position_error: float | None = None
    rotation_error_deg: float | None = None


@dataclass(frozen=True)
class AdapterAssessment:
    status: str
    action: str
    reasons: tuple[str, ...]
    evidence_lineage: str = OFFICIAL_EDM_ADAPTER_LINEAGE
    evidence_scope: str = ADAPTER_EVIDENCE_SCOPE
    is_production_hl: bool = False
    confirms_production_localization: bool = False


@dataclass(frozen=True)
class LiftedMatch:
    query_xy: tuple[float, float]
    point3d_id: int | None
    reference_name: str
    confidence: float
    lift_distance_px: float
    xyz: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class UnmappedPair:
    query_xy: tuple[float, float]
    reference_xy: tuple[float, float]
    reference_name: str
    confidence: float


@dataclass(frozen=True)
class ReferenceLiftResult:
    matches: tuple[LiftedMatch, ...]
    unmapped_match_count: int
    unmapped_pairs: tuple[UnmappedPair, ...] = ()


@dataclass(frozen=True)
class DeduplicatedMatches:
    matches: tuple[LiftedMatch, ...]
    conflicting_query_match_count: int
    duplicate_point3d_match_count: int


def _assessment(status: str, action: str, *reasons: str) -> AdapterAssessment:
    return AdapterAssessment(status=status, action=action, reasons=tuple(reasons))


def assess_adapter_evidence(
    evidence: AdapterEvidence,
    thresholds: AdapterThresholds,
) -> AdapterAssessment:
    """Classify only adapter execution stages, never production-localizer health."""

    if not evidence.completed:
        return _assessment(
            "ADAPTER_RUN_INCOMPLETE",
            "RERUN_ADAPTER",
            "adapter record is incomplete",
        )
    if (
        evidence.retrieved_refs is None
        or evidence.retrieved_refs < thresholds.min_retrieved_refs
    ):
        return _assessment(
            "ADAPTER_RETRIEVAL_INSUFFICIENT",
            "INSPECT_MEGALOC_RETRIEVAL",
            f"retrieved_refs={evidence.retrieved_refs}",
        )
    if evidence.raw_matches is None or evidence.raw_matches < thresholds.min_raw_matches:
        return _assessment(
            "ADAPTER_LOCAL_MATCHING_INSUFFICIENT",
            "INSPECT_OFFICIAL_EDM_LOCAL_MATCHING",
            f"raw_matches={evidence.raw_matches}",
        )
    if (
        evidence.lifted_correspondences is None
        or evidence.lifted_correspondences < thresholds.min_lifted_correspondences
    ):
        return _assessment(
            "ADAPTER_COLMAP_LIFT_INSUFFICIENT",
            "INSPECT_REFERENCE_OBSERVATION_LIFT",
            f"lifted_correspondences={evidence.lifted_correspondences}",
        )
    if evidence.pnp_succeeded is not True:
        return _assessment("ADAPTER_PNP_UNSOLVED", "INSPECT_PNP_GEOMETRY", "PnP did not return")
    if evidence.pnp_inliers is None or evidence.pnp_inliers < thresholds.min_pnp_inliers:
        return _assessment(
            "ADAPTER_PNP_INLIERS_BELOW_GATE",
            "INSPECT_ADAPTER_CORRESPONDENCE_GEOMETRY",
            f"pnp_inliers={evidence.pnp_inliers}",
        )
    if evidence.ground_truth_pose_valid is not True:
        return _assessment(
            "ADAPTER_POSE_UNVERIFIABLE",
            "RETAIN_AS_ADAPTER_ONLY",
            "M0 map-relative pose is unavailable",
        )
    if evidence.normalized_position_error is None or evidence.rotation_error_deg is None:
        return _assessment(
            "ADAPTER_POSE_UNVERIFIABLE",
            "RETAIN_AS_ADAPTER_ONLY",
            "M0 map-relative pose metric is unavailable",
        )
    if (
        evidence.normalized_position_error > thresholds.max_normalized_position_error
        or evidence.rotation_error_deg > thresholds.max_rotation_error_deg
    ):
        return _assessment(
            "ADAPTER_MAP_RELATIVE_POSE_UNSTABLE",
            "INSPECT_REFERENCE_DIVERSITY_AND_GEOMETRY",
            f"normalized_position_error={evidence.normalized_position_error}",
            f"rotation_error_deg={evidence.rotation_error_deg}",
        )
    return _assessment(
        "ADAPTER_MAP_RELATIVE_POSE_SUPPORTED",
        "SEPARATE_PRODUCTION_VALIDATION_REQUIRED",
        "adapter PnP agrees with its non-independent M0 map-relative pose gate",
    )


def _validated_arrays(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    observation_points: np.ndarray,
    observation_point3d_ids: np.ndarray,
    confidences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    observations = np.asarray(observation_points, dtype=np.float64)
    point_ids = np.asarray(observation_point3d_ids, dtype=np.int64)
    scores = np.asarray(confidences, dtype=np.float64)
    if query.ndim != 2 or query.shape[1:] != (2,) or reference.shape != query.shape:
        raise ValueError("query and reference points must both have shape (N, 2)")
    if scores.shape != (len(query),):
        raise ValueError("confidences must have shape (N,)")
    if (
        observations.ndim != 2
        or observations.shape[1:] != (2,)
        or point_ids.shape != (len(observations),)
    ):
        raise ValueError(
            "COLMAP observations must be aligned (N, 2) points and (N,) IDs"
        )
    if not all(np.isfinite(values).all() for values in (query, reference, observations, scores)):
        raise ValueError("adapter correspondences must be finite")
    return query, reference, observations, point_ids, scores


def lift_reference_matches(
    *,
    query_points: np.ndarray,
    reference_points: np.ndarray,
    observation_points: np.ndarray,
    observation_point3d_ids: np.ndarray,
    confidences: np.ndarray,
    maximum_distance_px: float,
    reference_name: str,
) -> ReferenceLiftResult:
    """Lift official-EDM 2D-2D matches only through nearby valid M0 observations.

    A deterministic spatial grid avoids materializing an all-pairs distance matrix.  A
    match without a valid M0 Point3D observation inside the declared radius is rejected.
    """

    if not math.isfinite(maximum_distance_px) or maximum_distance_px <= 0:
        raise ValueError("maximum_distance_px must be finite and positive")
    if not reference_name:
        raise ValueError("reference_name is required")
    query, reference, observations, point_ids, scores = _validated_arrays(
        query_points,
        reference_points,
        observation_points,
        observation_point3d_ids,
        confidences,
    )
    valid = point_ids >= 0
    observations, point_ids = observations[valid], point_ids[valid]
    cell_size = float(maximum_distance_px)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(observations):
        grid[(math.floor(point[0] / cell_size), math.floor(point[1] / cell_size))].append(index)

    max_distance_sq = cell_size * cell_size
    lifted: list[LiftedMatch] = []
    unmapped_pairs: list[UnmappedPair] = []
    for query_point, reference_point, confidence in zip(query, reference, scores, strict=True):
        x_cell = math.floor(reference_point[0] / cell_size)
        y_cell = math.floor(reference_point[1] / cell_size)
        candidates = [
            index
            for x_offset in (-1, 0, 1)
            for y_offset in (-1, 0, 1)
            for index in grid.get((x_cell + x_offset, y_cell + y_offset), ())
        ]
        if not candidates:
            unmapped_pairs.append(
                UnmappedPair(
                    query_xy=(float(query_point[0]), float(query_point[1])),
                    reference_xy=(float(reference_point[0]), float(reference_point[1])),
                    reference_name=reference_name,
                    confidence=float(confidence),
                )
            )
            continue
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        deltas = observations[candidate_indices] - reference_point
        distances_sq = np.einsum("ij,ij->i", deltas, deltas)
        within = distances_sq <= max_distance_sq
        if not within.any():
            unmapped_pairs.append(
                UnmappedPair(
                    query_xy=(float(query_point[0]), float(query_point[1])),
                    reference_xy=(float(reference_point[0]), float(reference_point[1])),
                    reference_name=reference_name,
                    confidence=float(confidence),
                )
            )
            continue
        candidate_indices, distances_sq = candidate_indices[within], distances_sq[within]
        # Equal-distance ties use the stable M0 Point3D ID rather than input order.
        selection = np.lexsort((point_ids[candidate_indices], distances_sq))[0]
        observation_index = int(candidate_indices[selection])
        lifted.append(
            LiftedMatch(
                query_xy=(float(query_point[0]), float(query_point[1])),
                point3d_id=int(point_ids[observation_index]),
                reference_name=reference_name,
                confidence=float(confidence),
                lift_distance_px=float(math.sqrt(distances_sq[selection])),
            )
        )
    return ReferenceLiftResult(
        matches=tuple(lifted),
        unmapped_match_count=len(unmapped_pairs),
        unmapped_pairs=tuple(unmapped_pairs),
    )


def deduplicate_lifted_matches(
    matches: Sequence[LiftedMatch],
    *,
    query_conflict_distance_px: float,
) -> DeduplicatedMatches:
    """Keep one Point3D per isolated query location; reject spatially conflicting lifts."""

    if not math.isfinite(query_conflict_distance_px) or query_conflict_distance_px <= 0:
        raise ValueError("query_conflict_distance_px must be finite and positive")
    ordered = sorted(
        matches,
        key=lambda match: (
            0 if match.point3d_id is not None else 1,
            -match.confidence,
            match.lift_distance_px,
            match.point3d_id if match.point3d_id is not None else -1,
            match.reference_name,
            match.query_xy,
        ),
    )
    grid: dict[tuple[int, int], list[LiftedMatch]] = defaultdict(list)
    accepted: list[LiftedMatch] = []
    point_ids: set[int] = set()
    conflicts = 0
    duplicates = 0
    radius_sq = query_conflict_distance_px * query_conflict_distance_px
    for match in ordered:
        x_cell = math.floor(match.query_xy[0] / query_conflict_distance_px)
        y_cell = math.floor(match.query_xy[1] / query_conflict_distance_px)
        nearby = [
            earlier
            for x_offset in (-1, 0, 1)
            for y_offset in (-1, 0, 1)
            for earlier in grid.get((x_cell + x_offset, y_cell + y_offset), ())
            if (earlier.query_xy[0] - match.query_xy[0]) ** 2
            + (earlier.query_xy[1] - match.query_xy[1]) ** 2
            <= radius_sq
        ]
        if any(earlier.point3d_id != match.point3d_id for earlier in nearby):
            conflicts += 1
            continue
        if nearby or (match.point3d_id is not None and match.point3d_id in point_ids):
            duplicates += 1
            continue
        accepted.append(match)
        if match.point3d_id is not None:
            point_ids.add(match.point3d_id)
        grid[(x_cell, y_cell)].append(match)
    return DeduplicatedMatches(
        matches=tuple(accepted),
        conflicting_query_match_count=conflicts,
        duplicate_point3d_match_count=duplicates,
    )


def read_colmap_bin_mat(path: Path) -> np.ndarray:
    """Read a COLMAP ``Mat`` file as ``(H, W)`` float32 depth."""

    raw = Path(path).read_bytes()
    pos = 0
    fields: list[int] = []
    for _ in range(3):
        amp = raw.find(b"&", pos)
        if amp < 0:
            raise ValueError(f"truncated COLMAP Mat header: {path}")
        token = raw[pos:amp]
        try:
            fields.append(int(token))
        except ValueError as exc:
            raise ValueError(f"truncated COLMAP Mat header: {path}") from exc
        pos = amp + 1
    width, height, channels = fields
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError(f"invalid COLMAP Mat shape: {path}")
    expected = width * height * channels * 4
    payload = raw[pos:]
    if len(payload) < expected:
        raise ValueError(f"truncated COLMAP Mat payload: {path}")
    array = np.frombuffer(payload[:expected], dtype="<f4").reshape(height, width, channels)
    return np.asarray(array[..., 0], dtype=np.float32)


def sample_depth_bilinear(
    depth: np.ndarray, uv: np.ndarray, native_wh: tuple[int, int]
) -> np.ndarray:
    """Bilinear-sample depth at native-resolution pixels. Out-of-bounds → NaN."""

    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("depth must have shape (H, W)")
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    native_w, native_h = int(native_wh[0]), int(native_wh[1])
    if native_w <= 0 or native_h <= 0:
        raise ValueError("native_wh must be positive")
    height, width = depth.shape
    x = uv[:, 0] * (width / native_w)
    y = uv[:, 1] * (height / native_h)
    out = np.full(len(uv), np.nan, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    in_bounds = finite & (x >= 0.0) & (y >= 0.0) & (x <= width - 1) & (y <= height - 1)
    if not in_bounds.any():
        return out
    xs = x[in_bounds]
    ys = y[in_bounds]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = xs - x0
    wy = ys - y0
    v00 = depth[y0, x0]
    v10 = depth[y0, x1]
    v01 = depth[y1, x0]
    v11 = depth[y1, x1]
    sampled = (
        v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy
    )
    four_finite = np.isfinite(v00) & np.isfinite(v10) & np.isfinite(v01) & np.isfinite(v11)
    out[in_bounds] = np.where(four_finite, sampled, np.nan)
    return out


def unproject_depth(
    uv_native: np.ndarray | Sequence[float],
    z: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_from_world_3x4: np.ndarray,
) -> np.ndarray:
    """Unproject a native pixel + camera-z through COLMAP w2c to world XYZ."""

    z_value = float(z)
    if not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("depth z must be finite and positive")
    uv = np.asarray(uv_native, dtype=np.float64).reshape(2)
    if not np.isfinite(uv).all():
        raise ValueError("uv must be finite")
    x_cam = np.asarray(
        [
            (float(uv[0]) - float(cx)) / float(fx) * z_value,
            (float(uv[1]) - float(cy)) / float(fy) * z_value,
            z_value,
        ],
        dtype=np.float64,
    )
    pose = np.asarray(cam_from_world_3x4, dtype=np.float64)
    if pose.shape != (3, 4):
        raise ValueError("cam_from_world_3x4 must have shape (3, 4)")
    rotation = pose[:, :3]
    translation = pose[:, 3]
    x_world = rotation.T @ (x_cam - translation)
    if not np.isfinite(x_world).all():
        raise ValueError("unprojected xyz is non-finite")
    return x_world


def lift_unmapped_with_depth(
    *,
    unmapped: Sequence[UnmappedPair],
    depth: np.ndarray,
    native_wh: tuple[int, int],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_from_world_3x4: np.ndarray,
    median_sparse_z: float,
    z_lo_mult: float = 0.25,
    z_hi_mult: float = 4.0,
) -> tuple[LiftedMatch, ...]:
    """Unproject unmapped EDM pairs through a per-reference depth map."""

    if not unmapped:
        return ()
    median_z = float(median_sparse_z)
    if not math.isfinite(median_z) or median_z <= 0.0:
        return ()
    z_lo = z_lo_mult * median_z
    z_hi = z_hi_mult * median_z
    uv = np.asarray([pair.reference_xy for pair in unmapped], dtype=np.float64)
    sampled_z = sample_depth_bilinear(depth, uv, native_wh)
    lifted: list[LiftedMatch] = []
    for pair, z_value in zip(unmapped, sampled_z, strict=True):
        if not math.isfinite(float(z_value)) or float(z_value) < z_lo or float(z_value) > z_hi:
            continue
        try:
            xyz = unproject_depth(
                pair.reference_xy,
                float(z_value),
                fx,
                fy,
                cx,
                cy,
                cam_from_world_3x4,
            )
        except ValueError:
            continue
        lifted.append(
            LiftedMatch(
                query_xy=pair.query_xy,
                point3d_id=None,
                reference_name=pair.reference_name,
                confidence=pair.confidence,
                lift_distance_px=0.0,
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
    return tuple(lifted)



def reject_adapter_records_from_production_health_report(
    records: Sequence[Mapping[str, object]],
) -> None:
    """Fail closed: adapter evidence must use its dedicated report builder."""

    if any(record.get("evidence_lineage") == OFFICIAL_EDM_ADAPTER_LINEAGE for record in records):
        raise ValueError(
            "official-EDM adapter evidence cannot be consumed as production HL; "
            "use the dedicated adapter report"
        )


DEFAULT_REF_CACHE_SIZE = int(os.environ.get("EDM_REF_CACHE_SIZE", "1024"))


class PreparedReferenceCache:
    """Bounded LRU cache for prepared reference images and native shapes.

    Removes repeated disk reads (cv2.imread) and preprocessing transformations of static
    2688x1512 reference JPEGs during relocalization passes.
    Cannot change localization decisions because the prepared image tensors/arrays and
    native shapes are deterministic and bitwise identical to decoding on the fly.
    """

    def __init__(self, maxsize: int | None = None) -> None:
        if maxsize is None:
            maxsize = int(os.environ.get("EDM_REF_CACHE_SIZE", "1024"))
        self.maxsize = max(0, int(maxsize))
        self._cache: OrderedDict[str, tuple[Any, tuple[int, int]]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, key: str) -> tuple[Any, tuple[int, int]] | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: tuple[Any, tuple[int, int]]) -> None:
        if self.maxsize <= 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        if len(self._cache) >= self.maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def get_or_prepare(
        self,
        key: str,
        path: Path,
        runtime: Any,
    ) -> tuple[Any, tuple[int, int]]:
        cached = self.get(key)
        if cached is not None:
            return cached
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        shape = (int(image.shape[0]), int(image.shape[1]))
        prepared = prepare_official_megadepth_image_from_array(image, runtime)
        self.put(key, (prepared, shape))
        return prepared, shape

    def clear(self) -> None:
        self._cache.clear()


def prepare_official_megadepth_image_from_array(
    image: np.ndarray,
    runtime: Any,
) -> Any:
    """Apply the packaged EDM MegaDepth preprocessing directly to an in-memory grayscale array.

    Removes redundant cv2.imread disk reads when the image is already decoded in memory.
    Cannot change the localization decision because resizing, square padding, and coarse mask
    generation follow the exact deterministic official MegaDepth transform byte-for-byte.
    """
    import cv2
    from river_map_quality.official_edm_adapter_loo import (
        PreparedMegadepthImage,
        _official_megadepth_shape,
    )

    native_height, native_width = image.shape[:2]
    content_width, content_height, pad_size = _official_megadepth_shape(
        native_width,
        native_height,
        long_edge=runtime.image_resize,
        divisor=runtime.image_divisor,
    )
    if (pad_size, pad_size) != (runtime.input_width, runtime.input_height):
        raise ValueError(
            "official EDM square padding disagrees with its configured test resolution"
        )
    resized = cv2.resize(
        image,
        (content_width, content_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pixels = np.zeros((pad_size, pad_size), dtype=resized.dtype)
    pixels[:content_height, :content_width] = resized
    coarse_mask = np.zeros(
        (pad_size // runtime.coarse_scale, pad_size // runtime.coarse_scale),
        dtype=bool,
    )
    coarse_mask[
        : content_height // runtime.coarse_scale,
        : content_width // runtime.coarse_scale,
    ] = True
    return PreparedMegadepthImage(
        pixels=pixels,
        coarse_mask=coarse_mask,
        scale=(native_width / content_width, native_height / content_height),
        content_width=content_width,
        content_height=content_height,
    )


def prepare_official_image(
    path: Path,
    runtime: Any,
    *,
    prepare_image: Any = None,
    image: np.ndarray | None = None,
) -> Any:
    """Prepare an official EDM image, reusing an already-decoded array if provided.

    Removes redundant cv2.imread disk reads.
    Cannot change the localization decision because the underlying transform is unchanged.
    """
    import cv2

    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
    height, width = image.shape[:2]
    if prepare_image is None:
        return prepare_official_megadepth_image_from_array(image, runtime)

    from river_map_quality.official_edm_adapter_loo import prepare_official_megadepth_image

    if prepare_image is prepare_official_megadepth_image:
        return prepare_official_megadepth_image_from_array(image, runtime)

    return prepare_image(
        path.parent,
        path.name,
        runtime,
        native_width=width,
        native_height=height,
    )


def match_official_prepared(
    runtime: Any,
    query_image: Any,
    reference_image: Any,
) -> dict[str, np.ndarray]:
    """Run the official square-padding/mask path for independently scaled images."""
    torch = runtime.torch
    batch = {
        "image0": torch.from_numpy(np.ascontiguousarray(query_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "image1": torch.from_numpy(np.ascontiguousarray(reference_image.pixels))[None, None]
        .to(runtime.device, dtype=torch.float32)
        .div_(255.0),
        "mask0": torch.from_numpy(query_image.coarse_mask)[None].to(runtime.device),
        "mask1": torch.from_numpy(reference_image.coarse_mask)[None].to(runtime.device),
        "scale0": torch.tensor([query_image.scale], dtype=torch.float32, device=runtime.device),
        "scale1": torch.tensor(
            [reference_image.scale],
            dtype=torch.float32,
            device=runtime.device,
        ),
    }
    with torch.inference_mode():
        runtime.matcher(batch)
    return {
        "mkpts0_f": batch["mkpts0_f"].detach().cpu().numpy(),
        "mkpts1_f": batch["mkpts1_f"].detach().cpu().numpy(),
        "mconf": batch["mconf"].detach().cpu().numpy(),
    }


def match_official_prepared_batch(
    runtime: Any,
    query_image: Any,
    reference_images: Sequence[Any],
) -> list[dict[str, np.ndarray]]:
    """Batch top_k reference pairs into a single official EDM matcher forward pass.

    Removes sequential per-reference model forwards and redundant query backbone passes.
    Cannot change the localization decision because EDM natively processes mini-batches
    with identical weights in eval mode, and individual pair matches are separated by
    batch ID (m_bids) into per-reference outputs that are downstream byte-identical.
    Gated behind EDM_BATCH_REFS (default 0) to guard against GPU cuBLAS floating-point
    summation order differences across different batch sizes on GPU.
    """
    torch = runtime.torch
    bs = len(reference_images)
    if bs == 0:
        return []
    if bs == 1:
        return [match_official_prepared(runtime, query_image, reference_images[0])]

    q_pix_np = np.ascontiguousarray(query_image.pixels)
    q_tensor = torch.from_numpy(q_pix_np)[None, None].to(
        runtime.device, dtype=torch.float32
    ).div_(255.0)
    image0 = q_tensor.repeat(bs, 1, 1, 1)

    r_tensors = [
        torch.from_numpy(np.ascontiguousarray(r.pixels))[None]
        for r in reference_images
    ]
    image1 = torch.stack(r_tensors, dim=0).to(
        runtime.device, dtype=torch.float32
    ).div_(255.0)

    q_mask_t = torch.from_numpy(query_image.coarse_mask).to(runtime.device)
    mask0 = torch.stack([q_mask_t] * bs, dim=0)

    mask1 = torch.stack(
        [torch.from_numpy(r.coarse_mask).to(runtime.device) for r in reference_images],
        dim=0,
    )

    scale0 = torch.tensor(
        [query_image.scale] * bs,
        dtype=torch.float32,
        device=runtime.device,
    )
    scale1 = torch.tensor(
        [r.scale for r in reference_images],
        dtype=torch.float32,
        device=runtime.device,
    )

    batch = {
        "image0": image0,
        "image1": image1,
        "mask0": mask0,
        "mask1": mask1,
        "scale0": scale0,
        "scale1": scale1,
    }

    with torch.inference_mode():
        runtime.matcher(batch)

    m_bids = batch["m_bids"].detach().cpu().numpy()
    mk0 = batch["mkpts0_f"].detach().cpu().numpy()
    mk1 = batch["mkpts1_f"].detach().cpu().numpy()
    mconf = batch["mconf"].detach().cpu().numpy()

    per_pair = []
    for b in range(bs):
        pair_mask = (m_bids == b)
        per_pair.append(
            {
                "mkpts0_f": mk0[pair_mask],
                "mkpts1_f": mk1[pair_mask],
                "mconf": mconf[pair_mask],
            }
        )
    return per_pair
