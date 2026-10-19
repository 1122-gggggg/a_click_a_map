"""Pure post-SfM evidence aggregation.

The module deliberately reports evidence and warnings only.  A warning never changes
the pre-SfM or post-SfM role of a segment.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..fisher import FisherMetrics, compute_fisher_metrics


def model_capabilities(
    *,
    evidence_level: str,
    registered_images: int,
    total_images: int,
    landmarks: int,
    observations: int,
) -> dict[str, object]:
    """Describe which downstream decisions a reconstruction can support."""

    if evidence_level not in {"coarse_pose", "refined_geometry"}:
        raise ValueError("evidence_level must be coarse_pose or refined_geometry")
    values = {
        "evidence_level": evidence_level,
        "registered_images": int(registered_images),
        "total_images": int(total_images),
        "landmarks": int(landmarks),
        "observations": int(observations),
    }
    values.update(
        has_camera_poses=values["registered_images"] > 0,
        has_landmarks=values["landmarks"] > 0,
        has_tracks=values["observations"] > 0,
    )
    values["role_assignment_ready"] = bool(
        evidence_level == "refined_geometry"
        and values["has_camera_poses"]
        and values["has_landmarks"]
        and values["has_tracks"]
    )
    return values


def sample_track_centers(
    centers: list[np.ndarray],
    *,
    max_centers: int = 6,
) -> list[np.ndarray]:
    """Keep deterministic baseline coverage while bounding angle combinations."""

    if max_centers < 2:
        raise ValueError("max_centers must be at least two")
    if len(centers) <= max_centers:
        return centers
    last = len(centers) - 1
    indexes = [(index * last) // (max_centers - 1) for index in range(max_centers)]
    return [centers[index] for index in indexes]


@dataclass(frozen=True)
class SegmentMetrics:
    input_images: int
    registered_images: int
    registration_ratio: float
    landmark_count: int
    median_track_length: float | None
    median_reprojection_error: float | None


@dataclass(frozen=True)
class PostSfMMetrics:
    registration_ratio: float
    track_length_ratios: dict[str, float]
    segment_metrics: dict[str, SegmentMetrics]
    covisibility: dict[tuple[str, str], int]
    triangulation_angle_deg: dict[str, float | None]
    reprojection: dict[str, float | None]
    coverage: dict[str, float | None]
    fim: FisherMetrics
    star_overlap: dict[tuple[str, str], float]
    cycle_residuals: list[dict[str, float]]
    sim3_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    warnings: tuple[str, ...] = ()
    role_override: None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "registration_ratio": self.registration_ratio,
            "track_length_ratios": self.track_length_ratios,
            "segment_metrics": {k: vars(v) for k, v in self.segment_metrics.items()},
            "covisibility": {"|".join(k): v for k, v in self.covisibility.items()},
            "triangulation_angle_deg": self.triangulation_angle_deg,
            "reprojection": self.reprojection,
            "coverage": self.coverage,
            "fim": self.fim.as_dict(),
            "star_overlap": {"|".join(k): v for k, v in self.star_overlap.items()},
            "cycle_residuals": self.cycle_residuals,
            "sim3_diagnostics": self.sim3_diagnostics,
            "warnings": list(self.warnings),
        }


def diagnose_post_sfm(
    cameras: Sequence[Mapping[str, Any]],
    landmarks: Iterable[Mapping[str, Any]],
    *,
    fim: np.ndarray | None = None,
    cycle_edges: Sequence[tuple[str, str, np.ndarray]] = (),
    sim3_groups: Sequence[Mapping[str, Any]] = (),
    thresholds: Mapping[str, float] | None = None,
) -> PostSfMMetrics:
    """Aggregate normalized camera/landmark observations into map diagnostics."""
    cams = {str(c["image_id"]): c for c in cameras}
    registered = [c for c in cams.values() if bool(c.get("registered", True))]
    ratio = len(registered) / max(len(cams), 1)
    segment_cameras: dict[str, list[Mapping[str, Any]]] = {}
    for camera in cams.values():
        segment_cameras.setdefault(str(camera.get("segment_id", "unknown")), []).append(camera)
    segment_landmarks = {segment: 0 for segment in segment_cameras}
    segment_track_lengths: dict[str, array[int]] = {
        segment: array("I") for segment in segment_cameras
    }
    segment_errors: dict[str, array[float]] = {
        segment: array("d") for segment in segment_cameras
    }
    length_values: array[int] = array("I")
    reprojection_errors: array[float] = array("d")
    angle_values: array[float] = array("d")
    uv_values: list[Sequence[float]] = []
    covis: dict[tuple[str, str], int] = {}
    image_landmark_counts: dict[str, int] = {image_id: 0 for image_id in cams}

    for landmark in landmarks:
        observations = tuple(landmark.get("observations", ()))
        track_length = len(observations)
        length_values.append(track_length)
        ids = sorted(
            {
                str(observation["image_id"])
                for observation in observations
                if str(observation["image_id"]) in cams
            }
        )
        for image_id in ids:
            image_landmark_counts[image_id] += 1
        for left, right in combinations(ids, 2):
            covis[(left, right)] = covis.get((left, right), 0) + 1

        track_errors = [
            float(observation["reprojection_error"])
            for observation in observations
            if "reprojection_error" in observation
        ]
        reprojection_errors.extend(track_errors)
        touched_segments = {
            str(cams[image_id].get("segment_id", "unknown")) for image_id in ids
        }
        for segment in touched_segments:
            segment_landmarks[segment] += 1
            segment_track_lengths[segment].append(track_length)
            segment_errors[segment].extend(track_errors)

        uv_values.extend(
            observation["uv"] for observation in observations if "uv" in observation
        )
        xyz = landmark.get("xyz")
        if xyz is not None:
            point = np.asarray(xyz, dtype=float)
            centers = [
                np.asarray(cams[str(observation["image_id"])].get("center"), dtype=float)
                for observation in observations
                if str(observation["image_id"]) in cams
                and cams[str(observation["image_id"])].get("center") is not None
            ]
            centers = sample_track_centers(centers)
            for first, second in combinations(centers, 2):
                ray_first, ray_second = point - first, point - second
                denominator = np.linalg.norm(ray_first) * np.linalg.norm(ray_second)
                if denominator > 0:
                    cosine = np.clip(ray_first @ ray_second / denominator, -1, 1)
                    angle_values.append(float(np.degrees(np.arccos(cosine))))

    lengths = np.asarray(length_values, dtype=float)
    track_ratios = {
        f"ge_{n}": float(np.mean(lengths >= n)) if len(lengths) else 0.0 for n in (3, 5)
    }

    segment_metrics: dict[str, SegmentMetrics] = {}
    for segment, subset in sorted(segment_cameras.items()):
        track_lengths = segment_track_lengths[segment]
        errors = segment_errors[segment]
        registered_images = sum(bool(camera.get("registered", True)) for camera in subset)
        segment_metrics[segment] = SegmentMetrics(
            len(subset),
            registered_images,
            registered_images / max(len(subset), 1),
            segment_landmarks[segment],
            float(np.median(track_lengths)) if track_lengths else None,
            float(np.median(errors)) if errors else None,
        )

    angles = {f"p{q}": _percentile(angle_values, q) for q in (10, 50, 90)}
    reproj = {
        "median": _percentile(reprojection_errors, 50),
        "p90": _percentile(reprojection_errors, 90),
    }
    coverage = {"median_observation_occupancy": _occupancy_values(uv_values)}
    fisher = compute_fisher_metrics(np.eye(6) if fim is None else fim)
    stars = _star_overlap_counts(image_landmark_counts, covis)
    cycles = _cycle_residuals(cycle_edges)
    sim3 = _sim3_diagnostics(sim3_groups)

    cfg = {
        "track_ratio_ge_3": 0.6,
        "median_track_length": 3.0,
        "fim_kappa": 1000.0,
        **(thresholds or {}),
    }
    warning: list[str] = []
    if track_ratios["ge_3"] < cfg["track_ratio_ge_3"] or (
        lengths.size and np.median(lengths) < cfg["median_track_length"]
    ):
        warning.append("LOW_TRACK_SUPPORT")
    if fisher.condition_number > cfg["fim_kappa"]:
        warning.append("LOW_FIM_OBSERVABILITY")
    if any(c["rotation_deg"] > 2.0 or c["translation_normalized"] > 0.02 for c in cycles):
        warning.append("CYCLE_INCONSISTENT")
    return PostSfMMetrics(
        ratio,
        track_ratios,
        segment_metrics,
        covis,
        angles,
        reproj,
        coverage,
        fisher,
        stars,
        cycles,
        sim3,
        tuple(warning),
    )


def _percentile(values: Sequence[float], q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    return float(np.percentile(array, q)) if len(array) else None


def _triangulation_angles(cams, tracks):
    values = []
    for landmark in tracks.values():
        centers = [
            np.asarray(cams[str(observation["image_id"])].get("center"), float)
            for observation in landmark
            if str(observation["image_id"]) in cams
            and cams[str(observation["image_id"])].get("center") is not None
        ]
        xyz = None
        for observation in landmark:
            xyz = observation.get("xyz", xyz)
        if xyz is None:
            continue
        point = np.asarray(xyz, float)
        for first, second in combinations(centers, 2):
            ray_first, ray_second = point - first, point - second
            denominator = np.linalg.norm(ray_first) * np.linalg.norm(ray_second)
            if denominator > 0:
                cosine = np.clip(ray_first @ ray_second / denominator, -1, 1)
                values.append(np.degrees(np.arccos(cosine)))
    return {f"p{q}": _percentile(values, q) for q in (10, 50, 90)}


def _occupancy_values(uv_values: Sequence[Sequence[float]]) -> float | None:
    if not uv_values:
        return None
    array = np.asarray(uv_values, float)
    span = np.maximum(np.ptp(array, axis=0), 1)
    cells = np.floor(array / span * 4)
    return float(len(np.unique(cells, axis=0)) / 16)


def _star_overlap(sets):
    return {
        (a, b): len(sets[a] & sets[b]) / max(min(len(sets[a]), len(sets[b])), 1)
        for a, b in combinations(sorted(sets), 2)
    }


def _star_overlap_counts(counts, covisibility):
    return {
        (left, right): covisibility.get((left, right), 0)
        / max(min(counts[left], counts[right]), 1)
        for left, right in combinations(sorted(counts), 2)
    }


def _cycle_residuals(edges):
    if not edges:
        return []
    product = np.eye(4)
    for _, _, transform in edges:
        product = product @ np.asarray(transform, float)
    rotation = product[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return [
        {
            "rotation_deg": float(angle),
            "translation_normalized": float(np.linalg.norm(product[:3, 3])),
        }
    ]


def _sim3_diagnostics(groups):
    return [dict(group) for group in groups]


__all__ = [
    "SegmentMetrics",
    "PostSfMMetrics",
    "diagnose_post_sfm",
    "model_capabilities",
    "sample_track_centers",
]
