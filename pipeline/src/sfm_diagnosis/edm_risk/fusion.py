"""Align held-out EDM evidence to the spatial grid and fuse calibrated risk.

The training feature extractor deliberately excludes the query's own EDM outcome
fields.  Otherwise a calibration model could learn the label from
``edm_loo_success_rate`` and report a misleadingly optimistic held-out score.
Cross-session empirical aggregates can be added later through an explicitly
group-excluded feature source.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .calibration import RiskTrainingSample
from .classifier import RiskThresholds, classify_spatial_risk
from .edm_loo import EDMQueryResult
from .schema import SpatialDiagnostic


_CALIBRATION_FEATURES = (
    "actloc_score",
    "visible_landmarks",
    "effective_landmarks",
    "median_track_length",
    "track_ge_3_ratio",
    "track_ge_5_ratio",
    "covisibility",
    "independent_observers",
    "parallax_p10_deg",
    "parallax_p25_deg",
    "parallax_median_deg",
    "viewpoint_entropy",
    "view_entropy",
    "convex_hull_coverage",
    "grid_occupancy",
    "landmark_spatial_entropy",
    "median_reprojection_error",
    "reprojection_error_p90",
    "positive_depth_ratio",
    "landmark_linearity",
    "landmark_planarity",
    "landmark_scattering",
    "mapping_camera_linearity",
    "mapping_camera_planarity",
    "mapping_camera_scattering",
    "fim_lambda_min",
    "fim_lambda_max",
    "fim_condition",
    "fim_logdet",
    "fim_trace",
    "fim_a_opt",
    "fim_d_opt",
    "fim_e_opt",
    "fim_translation_min_eigenvalue",
    "fim_rotation_min_eigenvalue",
    "num_pose_modes",
    "mode_support_margin",
    "mode_translation_separation",
    "mode_rotation_separation_deg",
    "mode_entropy",
    "loo_translation_jump",
    "loo_rotation_jump_deg",
)


def diagnostic_features(row: SpatialDiagnostic) -> dict[str, float | int | None]:
    """Return non-leaking predictors for one spatial orientation."""

    return {name: getattr(row, name) for name in _CALIBRATION_FEATURES}


def build_training_samples(
    rows: Sequence[SpatialDiagnostic],
    edm_results: Sequence[EDMQueryResult],
    *,
    ambiguity_by_query: Mapping[
        str, Mapping[str, float | int | None]
    ] | None = None,
    max_position_distance: float = math.inf,
    max_orientation_distance_deg: float = math.inf,
) -> tuple[list[RiskTrainingSample], dict[str, int | float]]:
    """Snap empirical query outcomes to the nearest grid pose.

    Alignment uses Euclidean position distance followed by circular yaw and
    linear pitch distance.  Session IDs remain the calibration groups.
    """

    if max_position_distance < 0 or max_orientation_distance_deg < 0:
        raise ValueError("alignment distance limits must be nonnegative")
    samples: list[RiskTrainingSample] = []
    unmatched = 0
    aligned: list[tuple[EDMQueryResult, SpatialDiagnostic]] = []
    for result in edm_results:
        if (
            result.estimated_position is None
            or result.estimated_yaw_deg is None
            or result.estimated_pitch_deg is None
            or not rows
        ):
            unmatched += 1
            continue
        nearest, position_distance, orientation_distance = _nearest_row(rows, result)
        if (
            position_distance > max_position_distance
            or orientation_distance > max_orientation_distance_deg
        ):
            unmatched += 1
            continue
        aligned.append((result, nearest))

    for result, nearest in aligned:
        peer_results = [
            peer
            for peer, peer_row in aligned
            if peer.session_id != result.session_id and peer_row is nearest
        ]
        features = diagnostic_features(nearest)
        features.update(_empirical_features(peer_results))
        features.update(
            _ambiguity_features(
                [
                    analysis
                    for peer in peer_results
                    if (
                        analysis := (ambiguity_by_query or {}).get(peer.query_id)
                    ) is not None
                ]
            )
        )
        samples.append(
            RiskTrainingSample(
                query_id=result.query_id,
                group=result.session_id,
                failure=not bool(result.success),
                features=features,
            )
        )
    return samples, {
        "query_count": len(edm_results),
        "matched_queries": len(samples),
        "unmatched_queries": unmatched,
        "max_position_distance": float(max_position_distance),
        "max_orientation_distance_deg": float(max_orientation_distance_deg),
    }


def apply_calibrated_risk(
    rows: list[SpatialDiagnostic],
    calibrator,
    *,
    edm_results: Sequence[EDMQueryResult] = (),
    thresholds: RiskThresholds | None = None,
) -> None:
    """Populate per-orientation probabilities, then classify each position."""

    empirical = _empirical_by_row(rows, edm_results)
    features: list[Mapping[str, float | int | None]] = []
    for row in rows:
        values = diagnostic_features(row)
        values.update(_empirical_features(empirical.get(id(row), ())))
        features.append(values)
    probabilities = np.asarray(calibrator.predict_proba(features), dtype=float)
    if probabilities.shape != (len(rows),):
        raise ValueError(
            "risk calibrator must return one failure probability per diagnostic row"
        )
    if np.any(~np.isfinite(probabilities)) or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("risk calibrator returned invalid probabilities")
    for row, probability in zip(rows, probabilities):
        row.failure_probability = float(probability)
    classify_spatial_risk(rows, thresholds)


def enrich_rows_with_edm(
    rows: Sequence[SpatialDiagnostic], edm_results: Sequence[EDMQueryResult]
) -> dict[str, int]:
    """Attach observed EDM aggregates to their nearest spatial orientation."""

    grouped = _empirical_by_row(rows, edm_results)
    for row in rows:
        evidence = list(grouped.get(id(row), ()))
        values = _empirical_features(evidence)
        row.edm_loo_success_rate = values["empirical_edm_success_rate"]
        row.edm_inliers_median = values["empirical_pnp_inliers"]
        row.edm_inlier_ratio_median = values["empirical_inlier_ratio"]
        row.edm_reprojection_median = values["empirical_reprojection_error"]
        modes = {result.loo_mode for result in evidence if result.loo_mode}
        row.edm_loo_mode = sorted(modes)[0] if len(modes) == 1 else ("mixed" if modes else None)
    return {
        "queries": len(edm_results),
        "queries_with_pose": sum(len(values) for values in grouped.values()),
        "rows_with_empirical_edm": len(grouped),
    }


def enrich_rows_with_temporal(
    rows: Sequence[SpatialDiagnostic], edm_results: Sequence[EDMQueryResult]
) -> dict[str, int]:
    """Attach causal per-session failure-burst evidence to visited grid rows."""

    from .temporal import summarize_temporal

    by_session: dict[str, list[EDMQueryResult]] = defaultdict(list)
    for result in edm_results:
        by_session[result.session_id].append(result)
    per_row: dict[int, list] = defaultdict(list)
    for session_results in by_session.values():
        summary = summarize_temporal(session_results)
        touched = set()
        for result in session_results:
            if (
                result.estimated_position is None
                or result.estimated_yaw_deg is None
                or result.estimated_pitch_deg is None
                or not rows
            ):
                continue
            row, _, _ = _nearest_row(rows, result)
            touched.add(id(row))
        for row_id in touched:
            per_row[row_id].append(summary)
    for row in rows:
        summaries = per_row.get(id(row), ())
        if not summaries:
            continue
        row.consecutive_failure_probability = max(
            summary.consecutive_failure_probability.get(2, 0.0)
            for summary in summaries
        )
        row.max_consecutive_failures = max(
            summary.max_consecutive_failures for summary in summaries
        )
        recovery = [
            summary.recovery_time_median
            for summary in summaries
            if summary.recovery_time_median is not None
        ]
        row.recovery_time = float(np.median(recovery)) if recovery else None
    return {
        "sessions": len(by_session),
        "rows_with_temporal_evidence": len(per_row),
    }


def enrich_rows_with_ambiguity(
    rows: Sequence[SpatialDiagnostic],
    edm_results: Sequence[EDMQueryResult],
    analyses: Mapping[str, Mapping[str, float | int | None]],
) -> dict[str, int]:
    """Attach precomputed repeated-hypothesis and group-LOO evidence."""

    result_by_id = {result.query_id: result for result in edm_results}
    grouped: dict[int, list[Mapping[str, float | int | None]]] = defaultdict(list)
    for query_id, analysis in analyses.items():
        result = result_by_id.get(query_id)
        if (
            result is None
            or result.estimated_position is None
            or result.estimated_yaw_deg is None
            or result.estimated_pitch_deg is None
            or not rows
        ):
            continue
        row, _, _ = _nearest_row(rows, result)
        grouped[id(row)].append(analysis)
    for row in rows:
        evidence = grouped.get(id(row), ())
        if not evidence:
            continue
        row.num_pose_modes = _aggregate(evidence, "num_pose_modes", max, integer=True)
        row.mode_support_margin = _aggregate(evidence, "support_margin", min)
        row.mode_translation_separation = _aggregate(
            evidence, "mode_translation_separation", max
        )
        row.mode_rotation_separation_deg = _aggregate(
            evidence, "mode_rotation_separation_deg", max
        )
        row.mode_entropy = _aggregate(evidence, "mode_entropy", max)
        row.loo_translation_jump = _aggregate(
            evidence, "loo_translation_jump", max
        )
        row.loo_rotation_jump_deg = _aggregate(
            evidence, "loo_rotation_jump_deg", max
        )
    return {
        "query_analyses": len(analyses),
        "rows_with_ambiguity_evidence": len(grouped),
    }


def _nearest_row(
    rows: Sequence[SpatialDiagnostic], result: EDMQueryResult
) -> tuple[SpatialDiagnostic, float, float]:
    query_position = np.asarray(result.estimated_position, dtype=float)
    candidates = []
    for row in rows:
        position_distance = float(
            np.linalg.norm(np.asarray(row.position, dtype=float) - query_position)
        )
        yaw_distance = _circular_distance_deg(row.yaw_deg, result.estimated_yaw_deg)
        pitch_distance = abs(float(row.pitch_deg) - float(result.estimated_pitch_deg))
        orientation_distance = float(math.hypot(yaw_distance, pitch_distance))
        candidates.append((position_distance, orientation_distance, row))
    position_distance, orientation_distance, nearest = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    return nearest, position_distance, orientation_distance


def _circular_distance_deg(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def _empirical_by_row(
    rows: Sequence[SpatialDiagnostic], edm_results: Sequence[EDMQueryResult]
) -> dict[int, list[EDMQueryResult]]:
    grouped: dict[int, list[EDMQueryResult]] = {}
    for result in edm_results:
        if (
            result.estimated_position is None
            or result.estimated_yaw_deg is None
            or result.estimated_pitch_deg is None
            or not rows
        ):
            continue
        row, _, _ = _nearest_row(rows, result)
        grouped.setdefault(id(row), []).append(result)
    return grouped


def _empirical_features(
    results: Sequence[EDMQueryResult],
) -> dict[str, float | None]:
    def median(name: str) -> float | None:
        values = [
            float(value)
            for result in results
            if (value := getattr(result, name)) is not None and np.isfinite(float(value))
        ]
        return float(np.median(values)) if values else None

    temporal = None
    if results:
        from .temporal import summarize_temporal

        by_session: dict[str, list[EDMQueryResult]] = defaultdict(list)
        for result in results:
            by_session[result.session_id].append(result)
        temporal = max(
            summarize_temporal(session).consecutive_failure_probability.get(2, 0.0)
            for session in by_session.values()
        )
    return {
        "empirical_edm_success_rate": (
            float(np.mean([result.success for result in results])) if results else None
        ),
        "empirical_pnp_inliers": median("ransac_inliers"),
        "empirical_inlier_ratio": median("inlier_ratio"),
        "empirical_reprojection_error": median("reprojection_median"),
        "consecutive_failure_probability": temporal,
    }


def _ambiguity_features(
    evidence: Sequence[Mapping[str, float | int | None]],
) -> dict[str, float | int | None]:
    return {
        "num_pose_modes": _aggregate(evidence, "num_pose_modes", max, integer=True),
        "mode_support_margin": _aggregate(evidence, "support_margin", min),
        "mode_translation_separation": _aggregate(
            evidence, "mode_translation_separation", max
        ),
        "mode_rotation_separation_deg": _aggregate(
            evidence, "mode_rotation_separation_deg", max
        ),
        "mode_entropy": _aggregate(evidence, "mode_entropy", max),
        "loo_translation_jump": _aggregate(evidence, "loo_translation_jump", max),
        "loo_rotation_jump_deg": _aggregate(
            evidence, "loo_rotation_jump_deg", max
        ),
    }


def _aggregate(evidence, key: str, operation, *, integer: bool = False):
    values = [
        float(value)
        for row in evidence
        if (value := row.get(key)) is not None and np.isfinite(float(value))
    ]
    if not values:
        return None
    result = operation(values)
    return int(result) if integer else float(result)
