from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sfm_diagnosis.models import MapData
from sfm_diagnosis.io import write_csv, write_json
from sfm_diagnosis.visibility import (
    image_grid_occupancy,
    normalized_convex_hull_area,
    visible_points,
)

from .grid import SpatialGridConfig, SpatialPoseGrid
from .fim import FIMConfig, matchability_aware_fim
from .degeneracy import diagnose_degeneracy
from .schema import SpatialDiagnostic
from .actloc_provider import ActLocProvider, DisabledActLocProvider


@dataclass(frozen=True)
class DiagnosisRun:
    mode: str
    grid: SpatialPoseGrid
    rows: tuple[SpatialDiagnostic, ...]
    metadata: dict[str, Any]
    map_data: MapData | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "EDM_LOCALIZATION_RISK_DIAGNOSIS",
            "mode": self.mode,
            "metadata": self.metadata,
            "grid": {
                "num_positions": self.grid.num_positions,
                "num_pose_samples": self.grid.num_pose_samples,
                "voxel_size": self.grid.config.voxel_size,
                "yaw_step_deg": self.grid.config.yaw_step_deg,
                "pitch_values_deg": list(self.grid.config.pitch_values_deg),
                "bounds": self.grid.config.bounds,
            },
            "rows": [row.to_dict() for row in self.rows],
        }

    def save(self, output_dir) -> dict[str, Any]:
        from pathlib import Path

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        risk_map = output / "risk_map.json"
        voxels = output / "risk_voxels.csv"
        write_json(risk_map, self.to_dict())
        csv_rows = []
        for row in self.rows:
            payload = row.to_dict()
            x, y, z = payload.pop("position")
            csv_rows.append({"x": x, "y": y, "z": z, **payload})
        write_csv(voxels, csv_rows)
        outputs = {"risk_map_json": risk_map, "risk_voxels_csv": voxels}
        if self.map_data is not None:
            from .visualization import write_diagnosis_artifacts

            outputs.update(
                write_diagnosis_artifacts(
                    self.map_data,
                    self.rows,
                    output,
                    voxel_size=self.grid.config.voxel_size,
                    display_radius=self.grid.config.max_landmark_distance,
                )
            )
        return outputs


class EDMRiskDiagnosis:
    """Deep module for spatial EDM-failure diagnosis.

    The map-only ``run_fast`` interface is intentionally stable.  Full-mode
    evidence providers are added behind the same module in later phases rather
    than leaking localizer/checkpoint details to callers.
    """

    def __init__(self, map_data: MapData):
        self.map_data = map_data

    def run_fast(
        self,
        config: SpatialGridConfig,
        *,
        matchability=None,
        fim_config: FIMConfig | None = None,
        actloc_provider: ActLocProvider | None = None,
        occlusion_proxy=None,
    ) -> DiagnosisRun:
        grid = SpatialPoseGrid.from_map(self.map_data, config)
        intrinsics = self.map_data.median_intrinsics
        image_lookup = self.map_data.image_index()
        actloc = actloc_provider or DisabledActLocProvider()
        rows: list[SpatialDiagnostic] = []
        for sample in grid:
            pose = sample.pose
            visible = visible_points(
                self.map_data,
                pose,
                intrinsics=intrinsics,
                max_distance=config.max_landmark_distance,
            )
            raw_visible_count = len(visible.point_indices)
            occlusion_result = None
            if occlusion_proxy is not None and raw_visible_count:
                occlusion_result = occlusion_proxy.filter(
                    pose, self.map_data.points_xyz[visible.point_indices]
                )
                keep = occlusion_result.visible
                visible = type(visible)(
                    visible.point_indices[keep],
                    visible.camera_points[keep],
                    visible.distances[keep],
                    visible.uv[keep],
                    visible.geometric_weights[keep],
                )
            indices = visible.point_indices
            track_lengths = self.map_data.track_lengths[indices]
            errors = self.map_data.point_errors[indices]
            angles = self.map_data.triangulation_angles_deg(indices)
            observer_counts = _observer_counts(self.map_data.track_image_ids, indices, image_lookup)
            observer_indices = np.asarray(
                [image_lookup[image_id] for image_id in observer_counts], dtype=int
            )
            yaw_std, pitch_std, view_entropy = _view_distribution(
                self.map_data.image_R_wc[observer_indices, :, 2]
                if len(observer_indices)
                else np.empty((0, 3))
            )
            covisibility = (
                max(observer_counts.values()) / max(len(indices), 1) if observer_counts else None
            )
            fim = matchability_aware_fim(
                self.map_data,
                visible,
                intrinsics,
                empirical=matchability,
                config=fim_config,
            )
            degeneracy = diagnose_degeneracy(
                landmarks=self.map_data.points_xyz[indices],
                observer_centers=(
                    self.map_data.image_centers[observer_indices]
                    if len(observer_indices)
                    else np.empty((0, 3))
                ),
                fim=fim.metrics,
                parallax_p10_deg=_percentile(angles, 10),
                parallax_median_deg=_percentile(angles, 50),
                view_entropy=view_entropy,
            )
            row = SpatialDiagnostic(
                position=sample.position,
                yaw_deg=sample.yaw_deg,
                pitch_deg=sample.pitch_deg,
                actloc_score=actloc.score(
                    np.asarray(sample.position, dtype=float),
                    sample.yaw_deg,
                    sample.pitch_deg,
                ),
                actloc_source=actloc.source,
                visible_landmarks=int(len(indices)),
                raw_visible_landmarks=raw_visible_count,
                occluded_count=0 if occlusion_result is None else occlusion_result.occluded_count,
                occlusion_uncertain_count=0
                if occlusion_result is None
                else occlusion_result.uncertain_count,
                visibility_source="point_cloud_depth_proxy"
                if occlusion_proxy is not None
                else "frustum_only",
                occlusion_verified=False,
                occlusion_proxy_applied=occlusion_proxy is not None,
                effective_landmarks=float(np.sum(fim.weights)),
                matchability_source=fim.matchability_source,
                median_track_length=_percentile(track_lengths, 50),
                track_ge_3_ratio=_ratio(track_lengths >= 3),
                track_ge_5_ratio=_ratio(track_lengths >= 5),
                covisibility=covisibility,
                independent_observers=len(observer_counts),
                parallax_p10_deg=_percentile(angles, 10),
                parallax_p25_deg=_percentile(angles, 25),
                parallax_median_deg=_percentile(angles, 50),
                mapping_view_yaw_std_deg=yaw_std,
                mapping_view_pitch_std_deg=pitch_std,
                view_entropy=view_entropy,
                convex_hull_coverage=normalized_convex_hull_area(visible.uv, intrinsics),
                grid_occupancy=image_grid_occupancy(visible.uv, intrinsics),
                landmark_spatial_entropy=_spatial_entropy(self.map_data.points_xyz[indices]),
                median_reprojection_error=_percentile(errors, 50),
                reprojection_error_p90=_percentile(errors, 90),
                positive_depth_ratio=1.0 if len(indices) else None,
                landmark_pca_eigenvalues=(
                    degeneracy.landmark_shape.eigenvalues_descending.tolist()
                ),
                mapping_camera_pca_eigenvalues=(
                    degeneracy.camera_shape.eigenvalues_descending.tolist()
                ),
                landmark_linearity=degeneracy.landmark_shape.linearity,
                landmark_planarity=degeneracy.landmark_shape.planarity,
                landmark_scattering=degeneracy.landmark_shape.scattering,
                mapping_camera_linearity=degeneracy.camera_shape.linearity,
                mapping_camera_planarity=degeneracy.camera_shape.planarity,
                mapping_camera_scattering=degeneracy.camera_shape.scattering,
                fim_lambda_min=fim.metrics.lambda_min,
                fim_lambda_max=fim.metrics.lambda_max,
                fim_condition=fim.metrics.condition_number,
                fim_logdet=fim.metrics.logdet,
                fim_trace=fim.metrics.trace,
                fim_a_opt=fim.metrics.fim_a_opt,
                fim_d_opt=fim.metrics.fim_d_opt,
                fim_e_opt=fim.metrics.fim_e_opt,
                fim_translation_min_eigenvalue=(fim.metrics.translation_min_eigenvalue),
                fim_rotation_min_eigenvalue=fim.metrics.rotation_min_eigenvalue,
                weakest_eigenvector=fim.metrics.weakest_eigenvector.tolist(),
                degeneracy_flags=list(degeneracy.flags),
            )
            rows.append(row)
        _enrich_actloc_position_statistics(rows)
        from .classifier import classify_spatial_risk

        classify_spatial_risk(rows)
        return DiagnosisRun(
            mode="fast",
            grid=grid,
            rows=tuple(rows),
            metadata={
                "map_points": self.map_data.num_points,
                "map_images": self.map_data.num_images,
                "map_source": self.map_data.metadata.get("source"),
                "probability_status": "UNCALIBRATED_UNKNOWN",
                "actloc_source": actloc.source,
                "occlusion": (
                    occlusion_proxy.metadata()
                    if occlusion_proxy is not None
                    else {"mode": "frustum_only", "source": "none"}
                ),
                "fim_config": {
                    "pixel_sigma": (fim_config or FIMConfig()).pixel_sigma,
                    "translation_scale": (fim_config or FIMConfig()).translation_scale,
                    "regularization": (fim_config or FIMConfig()).regularization,
                },
            },
            map_data=self.map_data,
        )

    def run_full(
        self,
        config: SpatialGridConfig,
        *,
        edm_results,
        matchability=None,
        fim_config: FIMConfig | None = None,
        actloc_provider: ActLocProvider | None = None,
        occlusion_proxy=None,
        calibration_config=None,
        ambiguity_by_query=None,
        risk_feature_set: str = "F",
    ) -> DiagnosisRun:
        """Run the map diagnostics and calibrate them against held-out EDM outcomes.

        Query outcomes are aligned to the same spatial grid.  Calibration folds
        remain session-disjoint and empirical neighborhood predictors exclude the
        query's own session.  Results without an estimated pose remain valid
        temporal/global failure evidence but cannot be assigned to a spatial voxel.
        """

        from .ablation import select_variant_samples
        from .calibration import RiskCalibrator, probability_status_from_calibration
        from .fusion import (
            apply_calibrated_risk,
            build_training_samples,
            enrich_rows_with_ambiguity,
            enrich_rows_with_edm,
            enrich_rows_with_temporal,
        )

        fast = self.run_fast(
            config,
            matchability=matchability,
            fim_config=fim_config,
            actloc_provider=actloc_provider,
            occlusion_proxy=occlusion_proxy,
        )
        rows = list(fast.rows)
        empirical_receipt = enrich_rows_with_edm(rows, edm_results)
        temporal_receipt = enrich_rows_with_temporal(rows, edm_results)
        samples, alignment = build_training_samples(
            rows,
            edm_results,
            ambiguity_by_query=ambiguity_by_query,
            max_position_distance=max(config.voxel_size * 2.0, 1e-6),
            max_orientation_distance_deg=max(config.yaw_step_deg, 30.0),
        )
        ambiguity_receipt = enrich_rows_with_ambiguity(rows, edm_results, ambiguity_by_query or {})
        calibration = RiskCalibrator(calibration_config).fit(
            select_variant_samples(samples, risk_feature_set)
        )
        status = probability_status_from_calibration(
            calibration.metrics,
            min_roc_auc=calibration.config.min_roc_auc,
        )
        if status == "CALIBRATED_EDM_LOO":
            apply_calibrated_risk(
                rows,
                calibration.calibrator,
                edm_results=edm_results,
            )
        else:
            from .classifier import classify_spatial_risk

            for row in rows:
                row.failure_probability = None
            classify_spatial_risk(rows)
        metadata = dict(fast.metadata)
        metadata.update(
            {
                "probability_status": status,
                "risk_feature_set": risk_feature_set,
                "edm_evidence": empirical_receipt,
                "temporal_evidence": temporal_receipt,
                "ambiguity_evidence": ambiguity_receipt,
                "edm_alignment": alignment,
                "calibration": calibration.to_dict(),
                "evidence_priority": [
                    "EDM_EMPIRICAL_HELD_OUT",
                    "MULTI_HYPOTHESIS_STABILITY",
                    "MATCHABILITY_AWARE_FIM",
                    "ACTLOC_PRIOR",
                    "MAP_STATISTICS",
                ],
            }
        )
        return DiagnosisRun(
            mode="full",
            grid=fast.grid,
            rows=tuple(rows),
            metadata=metadata,
            map_data=self.map_data,
        )


def _enrich_actloc_position_statistics(rows: list[SpatialDiagnostic]) -> None:
    grouped: dict[tuple[float, float, float], list[SpatialDiagnostic]] = {}
    for row in rows:
        grouped.setdefault(row.position, []).append(row)
    for group in grouped.values():
        available = [row for row in group if row.actloc_score is not None]
        if not available:
            continue
        scores = np.asarray([float(row.actloc_score) for row in available])
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=float)
        ranks[order] = np.arange(len(scores)) / max(len(scores) - 1, 1)
        probability = np.clip(scores, 0.0, None)
        probability = probability / np.sum(probability) if np.sum(probability) else probability
        entropy = (
            -float(np.sum(probability[probability > 0] * np.log(probability[probability > 0])))
            / np.log(len(probability))
            if len(probability) > 1
            else 0.0
        )
        best = available[int(np.argmax(scores))]
        worst = available[int(np.argmin(scores))]
        for index, row in enumerate(available):
            row.actloc_rank = float(ranks[index])
            row.viewpoint_entropy = entropy
            row.actloc_best_yaw_deg = best.yaw_deg
            row.actloc_best_pitch_deg = best.pitch_deg
            row.actloc_worst_yaw_deg = worst.yaw_deg
            row.actloc_worst_pitch_deg = worst.pitch_deg


def _percentile(values: np.ndarray, q: float) -> float | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, q)) if len(array) else None


def _ratio(mask: np.ndarray) -> float | None:
    values = np.asarray(mask, dtype=bool)
    return float(np.mean(values)) if len(values) else None


def _view_distribution(forward: np.ndarray) -> tuple[float | None, float | None, float | None]:
    vectors = np.asarray(forward, dtype=float).reshape(-1, 3)
    if not len(vectors):
        return None, None, None
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(vectors):
        return None, None, None
    yaw = np.arctan2(vectors[:, 1], vectors[:, 0])
    pitch = np.arcsin(np.clip(vectors[:, 2], -1.0, 1.0))
    resultant = np.hypot(np.mean(np.cos(yaw)), np.mean(np.sin(yaw)))
    yaw_std = np.degrees(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-12)), 0.0)))
    pitch_std = np.degrees(np.std(pitch))
    hist, _ = np.histogramdd(
        np.column_stack((yaw, pitch)),
        bins=(12, 6),
        range=((-np.pi, np.pi), (-np.pi / 2.0, np.pi / 2.0)),
    )
    probability = hist.reshape(-1)
    probability = probability[probability > 0] / np.sum(probability)
    entropy = -float(np.sum(probability * np.log(probability))) / np.log(72.0)
    return float(yaw_std), float(pitch_std), entropy


def _spatial_entropy(points: np.ndarray) -> float | None:
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(xyz):
        return None
    lo, hi = np.min(xyz, axis=0), np.max(xyz, axis=0)
    span = hi - lo
    normalized = np.zeros_like(xyz)
    active = span > 1e-12
    normalized[:, active] = (xyz[:, active] - lo[active]) / span[active]
    cells = np.clip((normalized * 4).astype(int), 0, 3)
    _, counts = np.unique(cells, axis=0, return_counts=True)
    probability = counts / np.sum(counts)
    return -float(np.sum(probability * np.log(probability))) / np.log(64.0)


def _observer_counts(track_image_ids, point_indices, image_lookup) -> dict[int, int]:
    tracks = [
        np.asarray(track_image_ids[int(index)], dtype=np.int64).reshape(-1)
        for index in np.asarray(point_indices, dtype=int)
        if len(track_image_ids[int(index)])
    ]
    if not tracks:
        return {}
    observer_ids = np.concatenate(tracks)
    registered = np.isin(
        observer_ids,
        np.fromiter(image_lookup, dtype=np.int64, count=len(image_lookup)),
        assume_unique=False,
    )
    unique, counts = np.unique(observer_ids[registered], return_counts=True)
    return {int(image_id): int(count) for image_id, count in zip(unique, counts)}
