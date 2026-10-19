from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .actloc import StructuralLocalizabilityProxy
from .diagnose import DiagnosticThresholds, diagnose_pose
from .io import write_csv, write_json
from .logs import LocalizationHistory
from .models import MapData, Pose
from .visibility import IlluminationModel, OcclusionModel


@dataclass(frozen=True)
class HeatmapConfig:
    spacing_m: float = 1.0
    padding_m: float = 0.0
    orientation_mode: str = "map"  # map | yaw_pitch
    sample_mode: str = "grid"  # grid | cameras
    orientations_per_position: int = 3
    yaw_step_deg: float = 45.0
    pitches_deg: tuple[float, ...] = (0.0,)
    max_positions: int = 25000
    robust_threshold: float = 0.65


def build_heatmap(
    map_data: MapData,
    *,
    config: HeatmapConfig | None = None,
    thresholds: DiagnosticThresholds | None = None,
    history: LocalizationHistory | None = None,
    illumination: IlluminationModel | None = None,
    occlusion: OcclusionModel | None = None,
    matchability=None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict], list[dict]]:
    cfg = config or HeatmapConfig()
    t = thresholds or DiagnosticThresholds()
    positions = sample_positions(map_data, cfg, bounds=bounds)
    detailed: list[dict] = []
    aggregate: list[dict] = []
    camera_tree = map_data.image_tree() if map_data.num_images else None
    predictor = StructuralLocalizabilityProxy()

    for position in positions:
        orientations = _orientations_for_position(map_data, position, cfg, camera_tree)
        pose_rows = []
        for orientation_id, R_wc in enumerate(orientations):
            pose = Pose(position, R_wc)
            d = diagnose_pose(
                map_data,
                pose,
                thresholds=t,
                history=history,
                illumination=illumination,
                occlusion=occlusion,
                predictor=predictor,
                matchability=matchability,
            )
            score = health_score(d, thresholds=t)
            weakest_translation_camera = (
                d.fim_uncertainty.weakest_translation_direction_camera
            )
            weakest_translation_world = pose.R_wc @ weakest_translation_camera
            row = {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "orientation_id": int(orientation_id),
                "forward_x": float(pose.forward_w[0]),
                "forward_y": float(pose.forward_w[1]),
                "forward_z": float(pose.forward_w[2]),
                "health_score": score,
                "primary": d.primary.value,
                "codes": ";".join(code.value for code in d.codes),
                "visible_points": d.visible_points,
                "effective_points": d.effective_points,
                "fim_isotropy": d.fim_isotropy,
                "fim_lambda_min": d.fim.lambda_min,
                "fim_logdet": d.fim.logdet,
                "fim_condition": d.fim.condition_number,
                "fim_sigma_pose_worst_normalized": (
                    d.fim_uncertainty.sigma_pose_worst_normalized
                ),
                "fim_sigma_translation_worst_m": (
                    d.fim_uncertainty.sigma_translation_worst_m
                ),
                "fim_sigma_rotation_worst_deg": (
                    d.fim_uncertainty.sigma_rotation_worst_deg
                ),
                "fim_weakest_translation_fraction": (
                    d.fim_uncertainty.weakest_translation_fraction
                ),
                "fim_weakest_translation_camera_x": float(weakest_translation_camera[0]),
                "fim_weakest_translation_camera_y": float(weakest_translation_camera[1]),
                "fim_weakest_translation_camera_z": float(weakest_translation_camera[2]),
                "fim_weakest_translation_world_x": float(weakest_translation_world[0]),
                "fim_weakest_translation_world_y": float(weakest_translation_world[1]),
                "fim_weakest_translation_world_z": float(weakest_translation_world[2]),
                "grid_occupancy": d.grid_occupancy,
                "hull_coverage": d.hull_coverage,
                "track_diversity": d.track_diversity,
                "view_support_fraction": d.view_support.weighted_visible_support_fraction,
                "view_support_effective_points": (
                    d.view_support.effective_redetectable_points
                ),
                "view_support_angle_p90_deg": d.view_support.observation_angle_p90_deg,
                "view_support_angle_extrapolated_fraction": (
                    d.view_support.angle_extrapolated_fraction
                ),
                "view_support_range_extrapolated_fraction": (
                    d.view_support.range_extrapolated_fraction
                ),
                "illumination_ratio": d.illumination_ratio,
                "structural_localizability": d.structural_localizability,
                "history_success_rate": d.history.success_rate,
                "history_reference_dispersion_m": d.history.reference_dispersion_m,
                "history_reference_consensus_sigma_m": (
                    d.history.reference_consensus_sigma_m
                ),
                "history_reference_rotation_dispersion_deg": (
                    d.history.reference_rotation_dispersion_deg
                ),
                "history_reference_covariance_eligible_ratio": (
                    d.history.reference_covariance_eligible_ratio
                ),
                "query_triangulation_angle_p50_deg": (
                    d.query_geometry.triangulation_angle_p50_deg
                ),
                "query_scale_ratio_p50": d.query_geometry.scale_ratio_p50,
                "query_scale_extrapolated_fraction": (
                    d.query_geometry.scale_extrapolated_fraction
                ),
                "query_low_parallax_visible_fraction": (
                    d.query_geometry.low_parallax_visible_fraction
                ),
                "matchability_mean": (
                    None if d.matchability is None else d.matchability.mean_matchability
                ),
                "matchability_evidenced_fraction": (
                    None
                    if d.matchability is None
                    else d.matchability.evidenced_visible_fraction
                ),
                "matchable_points": (
                    None if d.matchability is None else d.matchability.matchable_points
                ),
                "effective_matchable_points": (
                    None if d.matchability is None else d.matchability.effective_matchable
                ),
            }
            detailed.append(row)
            pose_rows.append(row)

        scores = np.asarray([r["health_score"] for r in pose_rows], dtype=float)
        best = pose_rows[int(np.argmax(scores))]
        aggregate.append(
            {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "best_health": float(np.max(scores)),
                "mean_health": float(np.mean(scores)),
                "worst_health": float(np.min(scores)),
                "robust_view_ratio": float(np.mean(scores >= cfg.robust_threshold)),
                "best_forward_x": best["forward_x"],
                "best_forward_y": best["forward_y"],
                "best_forward_z": best["forward_z"],
                "best_primary": best["primary"],
                "best_codes": best["codes"],
            }
        )
    return detailed, aggregate


def save_heatmap(output_dir: str | Path, detailed: list[dict], aggregate: list[dict]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "pose_health.csv", detailed)
    write_csv(out / "position_health.csv", aggregate)
    write_json(
        out / "summary.json",
        {
            "num_pose_samples": len(detailed),
            "num_position_samples": len(aggregate),
            "weak_positions": sorted(aggregate, key=lambda x: x["best_health"])[:100],
        },
    )
    _maybe_plot(out / "position_health.html", aggregate)


def sample_positions(
    map_data: MapData,
    cfg: HeatmapConfig,
    *,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if str(getattr(cfg, "sample_mode", "grid")) == "cameras":
        if map_data.num_images == 0:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(map_data.image_centers, dtype=float)
    if bounds is None:
        if map_data.num_images:
            lo = map_data.image_centers.min(axis=0) - cfg.padding_m
            hi = map_data.image_centers.max(axis=0) + cfg.padding_m
        else:
            lo, hi = map_data.bounds
    else:
        lo, hi = (np.asarray(bounds[0], dtype=float), np.asarray(bounds[1], dtype=float))
    axes = [np.arange(lo[i], hi[i] + 0.5 * cfg.spacing_m, cfg.spacing_m) for i in range(3)]
    n = int(np.prod([len(a) for a in axes]))
    if n > cfg.max_positions:
        raise ValueError(
            f"Heatmap would contain {n} positions (> {cfg.max_positions}). "
            "Increase --spacing or reduce --bounds."
        )
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.reshape(-1) for m in mesh])


def health_score(
    diagnosis,
    *,
    thresholds: DiagnosticThresholds | None = None,
) -> float:
    """Display/ranking score; diagnostic decisions still use separate metrics.

    The score is deliberately interpretable and does not replace the underlying
    exported metrics. Reference-consensus history is used only when available.
    """
    t = thresholds or DiagnosticThresholds()
    visible = min(diagnosis.effective_points / 50.0, 1.0)
    coverage = 0.5 * min(diagnosis.grid_occupancy / 8.0, 1.0) + 0.5 * min(
        diagnosis.hull_coverage / 0.25, 1.0
    )
    isotropy = float(
        np.clip(
            (np.log10(max(diagnosis.fim_isotropy, 1e-8)) + 6.0) / 4.0,
            0.0,
            1.0,
        )
    )
    structural = diagnosis.structural_localizability
    structural = 0.5 if structural is None else structural
    view_support = min(
        diagnosis.view_support.weighted_visible_support_fraction
        / max(t.min_view_support_fraction * 2.0, 0.20),
        1.0,
    )

    parts = [visible, coverage, isotropy, structural, view_support]
    weights = [0.25, 0.20, 0.22, 0.13, 0.20]

    history = diagnosis.history.success_rate
    if history is not None:
        parts.append(float(np.clip(history, 0.0, 1.0)))
        weights.append(0.20)

    reference_quality = _reference_history_quality(diagnosis.history, t)
    if reference_quality is not None:
        parts.append(reference_quality)
        weights.append(0.20)

    values = np.asarray(parts, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.clip(np.sum(w * values) / np.sum(w), 0.0, 1.0))


def _reference_history_quality(history, t: DiagnosticThresholds) -> float | None:
    scores = []
    if history.reference_dispersion_m is not None:
        ratio = history.reference_dispersion_m / max(t.max_reference_dispersion_m, 1e-12)
        scores.append(float(np.clip(1.5 - 0.5 * ratio, 0.0, 1.0)))
    if history.reference_consensus_sigma_m is not None:
        ratio = history.reference_consensus_sigma_m / max(
            t.max_reference_consensus_sigma_m,
            1e-12,
        )
        scores.append(float(np.clip(1.5 - 0.5 * ratio, 0.0, 1.0)))
    if history.reference_rotation_dispersion_deg is not None:
        ratio = history.reference_rotation_dispersion_deg / max(
            t.max_reference_rotation_dispersion_deg,
            1e-12,
        )
        scores.append(float(np.clip(1.5 - 0.5 * ratio, 0.0, 1.0)))
    if history.reference_covariance_eligible_ratio is not None:
        scores.append(
            float(
                np.clip(
                    history.reference_covariance_eligible_ratio
                    / max(t.min_reference_covariance_eligible_ratio, 1e-12),
                    0.0,
                    1.0,
                )
            )
        )
    return min(scores) if scores else None


def _orientations_for_position(
    map_data: MapData,
    position: np.ndarray,
    cfg: HeatmapConfig,
    camera_tree: cKDTree | None,
) -> list[np.ndarray]:
    if cfg.orientation_mode == "map":
        if camera_tree is None:
            return [np.eye(3)]
        k = min(max(cfg.orientations_per_position, 1), map_data.num_images)
        _, idx = camera_tree.query(position, k=k)
        idx = np.atleast_1d(idx).astype(int)
        return [map_data.image_R_wc[i].copy() for i in idx]
    if cfg.orientation_mode == "yaw_pitch":
        rotations = []
        for pitch in cfg.pitches_deg:
            for yaw in np.arange(0.0, 360.0, cfg.yaw_step_deg):
                rotations.append(yaw_pitch_rotation(float(yaw), float(pitch)))
        return rotations
    raise ValueError(f"Unknown orientation_mode={cfg.orientation_mode!r}")


def yaw_pitch_rotation(yaw_deg: float, pitch_deg: float, up_w: np.ndarray | None = None) -> np.ndarray:
    """Create R_wc for a Z-up map, camera axes x-right/y-down/z-forward."""
    if up_w is None:
        up_w = np.array([0.0, 0.0, 1.0])
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    forward = np.array(
        [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
        dtype=float,
    )
    return rotation_from_forward(forward, up_w)


def rotation_from_forward(forward_w: np.ndarray, up_w: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward_w, dtype=float).reshape(3)
    forward /= max(np.linalg.norm(forward), 1e-12)
    up = np.asarray(up_w, dtype=float).reshape(3)
    up /= max(np.linalg.norm(up), 1e-12)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        fallback = (
            np.array([0.0, 1.0, 0.0])
            if abs(forward[1]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        right = np.cross(forward, fallback)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    return np.column_stack((right, down, forward))


def _maybe_plot(path: Path, aggregate: list[dict]) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return
    if not aggregate:
        return
    x = [r["x"] for r in aggregate]
    y = [r["y"] for r in aggregate]
    z = [r["z"] for r in aggregate]
    score = [r["best_health"] for r in aggregate]
    fig = px.scatter_3d(x=x, y=y, z=z, color=score, range_color=[0.0, 1.0])
    fig.update_traces(marker={"size": 3})
    fig.update_layout(title="SfM diagnosis: best-view health")
    fig.write_html(str(path), include_plotlyjs="cdn")
