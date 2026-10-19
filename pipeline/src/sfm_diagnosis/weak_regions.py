from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .evidence import BuildEvidence
from .graph import build_covisibility_graph, nearest_neighbor_distances
from .io import write_csv, write_json
from .models import MapData
from .pair_geometry import pair_model_flags, reconstructed_relative_error


class WeakRegionCause(str, Enum):
    IMAGE_QUALITY_WEAK = "IMAGE_QUALITY_WEAK"
    VIEW_GRAPH_ISOLATION = "VIEW_GRAPH_ISOLATION"
    RETRIEVAL_GAP = "RETRIEVAL_GAP"
    MATCHING_SHORTAGE = "MATCHING_SHORTAGE"
    MATCHING_AMBIGUITY = "MATCHING_AMBIGUITY"
    PLANAR_PAIR_DOMINANCE = "PLANAR_PAIR_DOMINANCE"
    RELATIVE_POSE_INCONSISTENT = "RELATIVE_POSE_INCONSISTENT"
    TRACK_FRAGMENTATION = "TRACK_FRAGMENTATION"
    LOW_PARALLAX = "LOW_PARALLAX"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    MAP_SUPPORT_SPARSE = "MAP_SUPPORT_SPARSE"
    UNEXPLAINED_WEAKNESS = "UNEXPLAINED_WEAKNESS"


@dataclass(frozen=True)
class WeakRegionConfig:
    covisibility_min_shared: int = 15
    max_track_for_pair_expansion: int | None = None
    robust_z_threshold: float = 2.5
    weak_image_score_threshold: float = 0.35
    cluster_radius: float | None = None
    anchor_radius: float | None = None
    anchor_radius_multiplier: float = 2.5
    min_anchor_edges: int = 2
    min_track_length_p50: float = 3.0
    max_two_view_fraction: float = 0.65
    min_triangulation_angle_p50_deg: float = 2.0
    max_reprojection_p90_px: float = 3.0
    min_pair_inliers: float = 30.0
    min_pair_inlier_ratio: float = 0.25
    min_pair_matches_for_ambiguity: float = 80.0
    min_pair_candidates_for_retrieval: int = 4
    min_selected_ratio: float = 0.25
    min_homography_support: float = 0.60
    max_essential_support: float = 0.30
    max_relative_rotation_deg: float = 15.0
    max_relative_direction: float = 0.35
    min_planar_pair_fraction: float = 0.50
    min_pose_inconsistent_fraction: float = 0.30
    max_report_regions: int = 100


@dataclass
class WeakRegionAnalysis:
    summary: dict
    baselines: dict
    evidence: dict
    images: list[dict]
    regions: list[dict]

    def as_dict(self) -> dict:
        return {
            "summary": self.summary,
            "baselines": self.baselines,
            "evidence": self.evidence,
            "regions": self.regions,
        }


def analyze_weak_regions(
    map_data: MapData,
    *,
    evidence: BuildEvidence | None = None,
    config: WeakRegionConfig | None = None,
) -> WeakRegionAnalysis:
    """Find weak reconstruction regions, explain why, and rank repair actions."""
    cfg = config or WeakRegionConfig()
    ev = evidence or BuildEvidence()
    graph = build_covisibility_graph(
        map_data,
        min_shared_points=cfg.covisibility_min_shared,
        max_track_for_pair_expansion=cfg.max_track_for_pair_expansion,
    )
    image_points = _image_point_indices(map_data)
    angles = map_data.triangulation_angles_deg()
    nn = nearest_neighbor_distances(map_data.image_centers)
    quality = ev.image_by_name

    rows: list[dict] = []
    for i in range(map_data.num_images):
        pidx = image_points[i]
        tracks = map_data.track_lengths[pidx].astype(float) if len(pidx) else np.array([])
        errors = map_data.point_errors[pidx].astype(float) if len(pidx) else np.array([])
        tri = angles[pidx].astype(float) if len(pidx) else np.array([])
        row = {
            "image_index": i,
            "image_id": int(map_data.image_ids[i]),
            "image_name": map_data.image_names[i],
            "x": float(map_data.image_centers[i, 0]),
            "y": float(map_data.image_centers[i, 1]),
            "z": float(map_data.image_centers[i, 2]),
            "point_support": int(graph.image_support[i]),
            "covisibility_degree": int(graph.degrees[i]),
            "track_length_p50": _pct(tracks, 50),
            "two_view_fraction": float(np.mean(tracks == 2)) if len(tracks) else None,
            "reprojection_error_p90_px": _pct(errors, 90),
            "triangulation_angle_p50_deg": _pct(tri, 50),
            "camera_nn_distance": float(nn[i]) if i < len(nn) else None,
        }
        q = quality.get(map_data.image_names[i], {})
        for key in (
            "sharpness_laplacian_var",
            "tenengrad",
            "entropy",
            "dark_ratio",
            "bright_ratio",
            "texture_score",
            "route_id",
            "timestamp",
        ):
            if q.get(key) is not None:
                row[key] = q[key]
        rows.append(row)

    directions = {
        "point_support": "low",
        "covisibility_degree": "low",
        "track_length_p50": "low",
        "two_view_fraction": "high",
        "reprojection_error_p90_px": "high",
        "triangulation_angle_p50_deg": "low",
        "sharpness_laplacian_var": "low",
        "tenengrad": "low",
        "entropy": "low",
        "dark_ratio": "high",
        "bright_ratio": "high",
    }
    baselines = {key: _distribution([r.get(key) for r in rows]) for key in directions}
    for row in rows:
        _score_image(row, baselines, cfg)
        row["unregistered_seed_count"] = 0

    cluster_radius = _cluster_radius(nn, cfg.cluster_radius)
    anchor_radius = cfg.anchor_radius
    if anchor_radius is None:
        anchor_radius = max(cluster_radius, cluster_radius * cfg.anchor_radius_multiplier)
    failures = _assign_unregistered(map_data, ev.image_rows, float(anchor_radius))
    for i, failed in failures.items():
        rows[i]["unregistered_seed_count"] = len(failed)
        rows[i]["weakness_score"] = max(rows[i]["weakness_score"], 0.55)

    weak = {
        int(row["image_index"])
        for row in rows
        if row["weakness_score"] >= cfg.weak_image_score_threshold
        or row["point_support"] == 0
        or row["covisibility_degree"] == 0
        or row["unregistered_seed_count"] > 0
    }
    clusters = _clusters(map_data.image_centers, weak, cluster_radius)
    pair_baseline = _pair_baseline(ev.pair_rows)
    regions = [
        _region(
            k + 1,
            members,
            map_data,
            rows,
            image_points,
            angles,
            graph,
            ev,
            pair_baseline,
            weak,
            float(anchor_radius),
            failures,
            cfg,
        )
        for k, members in enumerate(clusters)
    ]
    regions.sort(key=lambda r: r["severity_score"], reverse=True)
    regions = regions[: cfg.max_report_regions]
    for rank, region in enumerate(regions, 1):
        region["rank"] = rank

    counts = Counter(c for region in regions for c in region["root_causes"])
    summary = {
        "source": map_data.metadata.get("source"),
        "model_dir": map_data.metadata.get("model_dir"),
        "num_registered_images": map_data.num_images,
        "num_points3D": map_data.num_points,
        "num_weak_images": len(weak),
        "weak_image_fraction": len(weak) / map_data.num_images if map_data.num_images else 0.0,
        "num_weak_regions": len(regions),
        "positioned_unregistered_images": sum(len(v) for v in failures.values()),
        "cluster_radius_map_units": cluster_radius,
        "anchor_radius_map_units": anchor_radius,
        "cause_counts": dict(counts),
        "diagnostic_mode": _mode(ev),
        "covisibility": {
            "min_shared_points": cfg.covisibility_min_shared,
            "support_mode": graph.support_mode,
            "omitted_long_track_count": graph.omitted_long_track_count,
            "pair_counts_threshold_retained": graph.support_mode == "exact",
        },
    }
    baselines["pair_evidence"] = pair_baseline
    return WeakRegionAnalysis(summary, baselines, ev.availability(), rows, regions)


def save_weak_region_analysis(output_dir: str | Path, analysis: WeakRegionAnalysis) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", analysis.as_dict())
    write_csv(out / "image_diagnostics.csv", analysis.images)
    write_json(out / "weak_regions.json", analysis.regions)
    write_csv(out / "weak_regions.csv", [_flatten(r) for r in analysis.regions])
    write_json(
        out / "repair_plan.json",
        [
            {
                "rank": r["rank"],
                "region_id": r["region_id"],
                "root_causes": r["root_causes"],
                "recapture": r["recapture"],
                "repair_sequence": r["repair_sequence"],
                "anchor_image_names": r["anchor_image_names"],
                "solution_plan": r["solution_plan"],
            }
            for r in analysis.regions
        ],
    )
    _plot(out / "weak_regions.html", analysis.images, analysis.regions)


def weak_region_config_from_dict(data: dict) -> WeakRegionConfig:
    allowed = set(asdict(WeakRegionConfig()))
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown weak-region config fields: {sorted(unknown)}")
    return WeakRegionConfig(**data)


def _score_image(row: dict, baselines: dict, cfg: WeakRegionConfig) -> None:
    weights = {
        "point_support": ("low", 0.22),
        "covisibility_degree": ("low", 0.20),
        "track_length_p50": ("low", 0.16),
        "two_view_fraction": ("high", 0.12),
        "reprojection_error_p90_px": ("high", 0.16),
        "triangulation_angle_p50_deg": ("low", 0.14),
    }
    parts = {
        key: _badness(row.get(key), baselines[key], direction, cfg.robust_z_threshold)
        for key, (direction, _) in weights.items()
    }
    if row.get("track_length_p50") is not None:
        if row["track_length_p50"] < cfg.min_track_length_p50:
            parts["track_length_p50"] = max(parts["track_length_p50"], 0.85)
    if row.get("two_view_fraction") is not None:
        if row["two_view_fraction"] > cfg.max_two_view_fraction:
            parts["two_view_fraction"] = max(parts["two_view_fraction"], 0.85)
    if row.get("triangulation_angle_p50_deg") is not None:
        if row["triangulation_angle_p50_deg"] < cfg.min_triangulation_angle_p50_deg:
            parts["triangulation_angle_p50_deg"] = 1.0
    if row.get("reprojection_error_p90_px") is not None:
        if row["reprojection_error_p90_px"] > cfg.max_reprojection_p90_px:
            parts["reprojection_error_p90_px"] = 1.0
    if row["point_support"] == 0:
        parts["point_support"] = 1.0
    if row["covisibility_degree"] == 0:
        parts["covisibility_degree"] = 1.0
    row["weakness_components"] = parts
    row["weakness_score"] = float(
        np.clip(sum(weights[k][1] * parts[k] for k in weights), 0.0, 1.0)
    )

    quality = []
    for key, direction in (
        ("sharpness_laplacian_var", "low"),
        ("tenengrad", "low"),
        ("entropy", "low"),
        ("dark_ratio", "high"),
        ("bright_ratio", "high"),
    ):
        if row.get(key) is not None:
            quality.append(_badness(row[key], baselines[key], direction, cfg.robust_z_threshold))
    row["image_quality_badness"] = max(quality) if quality else None
    if quality:
        row["weakness_score"] = max(row["weakness_score"], 0.35 * max(quality))


def _region(
    region_id: int,
    members: tuple[int, ...],
    map_data: MapData,
    rows: list[dict],
    image_points: list[np.ndarray],
    angles: np.ndarray,
    graph,
    evidence: BuildEvidence,
    pair_baseline: dict,
    weak: set[int],
    anchor_radius: float,
    failures: dict[int, list[dict]],
    cfg: WeakRegionConfig,
) -> dict:
    member_set = set(members)
    centers = map_data.image_centers[list(members)]
    anchors = _anchors(map_data.image_centers, members, weak, anchor_radius)
    anchor_set = set(anchors)
    internal = anchor_edges = 0
    for i in members:
        for j in graph.adjacency[i]:
            if j in member_set and i < j:
                internal += 1
            elif j in anchor_set:
                anchor_edges += 1

    chunks = [image_points[i] for i in members if len(image_points[i])]
    points = np.unique(np.concatenate(chunks)) if chunks else np.array([], dtype=int)
    tracks = map_data.track_lengths[points].astype(float) if len(points) else np.array([])
    errors = map_data.point_errors[points].astype(float) if len(points) else np.array([])
    tri = angles[points].astype(float) if len(points) else np.array([])
    anchor_ratio = _anchor_track_ratio(map_data, points, member_set, anchor_set)
    pair_stats = _pair_stats(evidence.pair_rows, map_data, member_set, anchor_set, cfg)
    quality_stats = _quality_stats([rows[i] for i in members])
    observed = [x for i in members for x in failures.get(i, [])]
    metrics = {
        "num_images": len(members),
        "num_points": int(len(points)),
        "positioned_unregistered_images": len(observed),
        "point_support_p50": _pct([rows[i]["point_support"] for i in members], 50),
        "covisibility_degree_p50": _pct(
            [rows[i]["covisibility_degree"] for i in members], 50
        ),
        "internal_strong_edges": internal,
        "anchor_strong_edges": anchor_edges,
        "anchor_track_ratio": anchor_ratio,
        "track_length_p50": _pct(tracks, 50),
        "two_view_fraction": float(np.mean(tracks == 2)) if len(tracks) else None,
        "reprojection_error_p90_px": _pct(errors, 90),
        "triangulation_angle_p10_deg": _pct(tri, 10),
        "triangulation_angle_p50_deg": _pct(tri, 50),
    }
    causes, cause_evidence = _causes(
        metrics,
        pair_stats,
        pair_baseline,
        quality_stats,
        len(anchors),
        cfg,
        [rows[i] for i in members],
    )
    actions, recapture = _actions(causes, bool(anchors))
    solution_plan = _solution_plan(actions, recapture)
    scores = [rows[i]["weakness_score"] for i in members]
    severity = 0.75 * max(scores) + 0.25 * float(np.mean(scores))
    if WeakRegionCause.LOW_PARALLAX.value in causes:
        severity = max(severity, 0.65)
    if observed:
        severity = max(severity, min(0.90, 0.55 + 0.05 * len(observed)))
    status = "CRITICAL" if severity >= 0.75 else "WEAK" if severity >= 0.5 else "WATCH"
    return {
        "region_id": region_id,
        "status": status,
        "severity_score": float(severity),
        "primary_cause": causes[0],
        "root_causes": causes,
        "cause_evidence": cause_evidence,
        "center": np.mean(centers, axis=0).tolist(),
        "bbox_min": np.min(centers, axis=0).tolist(),
        "bbox_max": np.max(centers, axis=0).tolist(),
        "image_ids": [int(map_data.image_ids[i]) for i in members],
        "image_names": [map_data.image_names[i] for i in members],
        "anchor_image_ids": [int(map_data.image_ids[i]) for i in anchors],
        "anchor_image_names": [map_data.image_names[i] for i in anchors],
        "observed_unregistered_images": [x.get("image_name") for x in observed],
        "metrics": metrics,
        "pair_evidence": pair_stats,
        "image_quality": quality_stats,
        "recapture": recapture,
        "repair_sequence": actions,
        "solution_plan": solution_plan,
    }


def _causes(metrics, pair, baseline, quality, num_anchors, cfg, images):
    causes: list[str] = []
    notes: list[dict] = []

    def add(cause: WeakRegionCause, confidence: str, text: str) -> None:
        if cause.value not in causes:
            causes.append(cause.value)
            notes.append({"cause": cause.value, "confidence": confidence, "evidence": text})

    if quality.get("quality_images", 0) >= 2 and quality.get("bad_fraction", 0) >= 0.5:
        add(WeakRegionCause.IMAGE_QUALITY_WEAK, "high", "Most measured frames are quality outliers.")
    if num_anchors and metrics["internal_strong_edges"] > 0:
        if metrics["anchor_strong_edges"] < cfg.min_anchor_edges:
            add(
                WeakRegionCause.VIEW_GRAPH_ISOLATION,
                "high",
                "Weak images connect internally but have too few strong edges to healthy anchors.",
            )
    if num_anchors and metrics.get("anchor_track_ratio") is not None:
        if metrics["anchor_track_ratio"] < 0.15:
            add(
                WeakRegionCause.VIEW_GRAPH_ISOLATION,
                "medium",
                f"Only {metrics['anchor_track_ratio']:.1%} of region tracks reach anchors.",
            )
    if pair.get("selection_available") and pair["candidate_pairs"] >= cfg.min_pair_candidates_for_retrieval:
        if pair.get("selected_ratio") is not None and pair["selected_ratio"] < cfg.min_selected_ratio:
            add(WeakRegionCause.RETRIEVAL_GAP, "high", "Too few preserved candidate pairs were selected.")
    if pair.get("matching_available") and pair.get("attempted_pairs", 0) >= 2:
        matches = pair.get("matches_p50")
        inliers = pair.get("inliers_p50")
        ratio = pair.get("inlier_ratio_p50")
        ambiguous = (
            matches is not None
            and matches >= cfg.min_pair_matches_for_ambiguity
            and ratio is not None
            and ratio < cfg.min_pair_inlier_ratio
        )
        if ambiguous:
            add(
                WeakRegionCause.MATCHING_AMBIGUITY,
                "high",
                f"Raw match p50={matches:.1f} but inlier-ratio p50={ratio:.3f}.",
            )
        elif inliers is not None and inliers < cfg.min_pair_inliers:
            add(
                WeakRegionCause.MATCHING_SHORTAGE,
                "high",
                f"Verified pair inlier p50={inliers:.1f}.",
            )
    if pair.get("attempted_pairs", 0) >= 2:
        planar_frac = pair.get("planar_pair_fraction")
        if planar_frac is not None and planar_frac >= cfg.min_planar_pair_fraction:
            add(
                WeakRegionCause.PLANAR_PAIR_DOMINANCE,
                "high",
                f"Planar/panoramic pair fraction={planar_frac:.1%}.",
            )
        pose_frac = pair.get("pose_inconsistent_fraction")
        if pose_frac is not None and pose_frac >= cfg.min_pose_inconsistent_fraction:
            add(
                WeakRegionCause.RELATIVE_POSE_INCONSISTENT,
                "high",
                f"Two-view vs reconstructed pose disagreement fraction={pose_frac:.1%}.",
            )
    track = metrics.get("track_length_p50")
    two = metrics.get("two_view_fraction")
    if track is not None and two is not None:
        if track < cfg.min_track_length_p50 and two > cfg.max_two_view_fraction:
            add(
                WeakRegionCause.TRACK_FRAGMENTATION,
                "high",
                f"Track p50={track:.2f}; two-view fraction={two:.1%}.",
            )
    parallax = metrics.get("triangulation_angle_p50_deg")
    if parallax is not None and parallax < cfg.min_triangulation_angle_p50_deg:
        add(
            WeakRegionCause.LOW_PARALLAX,
            "high",
            f"Maximum available triangulation-angle p50={parallax:.2f} deg.",
        )
    reproj = metrics.get("reprojection_error_p90_px")
    if reproj is not None and reproj > cfg.max_reprojection_p90_px:
        add(WeakRegionCause.HIGH_REPROJECTION_ERROR, "high", f"Reprojection p90={reproj:.2f}px.")
    support_bad = [r["weakness_components"].get("point_support", 0.0) for r in images]
    if support_bad and float(np.mean(support_bad)) >= 0.65:
        add(WeakRegionCause.MAP_SUPPORT_SPARSE, "medium", "3D support is a strong map-relative low outlier.")
    if not causes:
        add(
            WeakRegionCause.UNEXPLAINED_WEAKNESS,
            "low",
            "Structural weakness is present but current evidence cannot separate the root cause.",
        )
    return causes, notes


def _actions(causes: list[str], has_anchors: bool) -> tuple[list[dict], str]:
    actions: list[dict] = []

    def add(priority: int, action: str, reason: str) -> None:
        if action not in {x["action"] for x in actions}:
            expected_improvements, acceptance_checks = _action_contract(action)
            actions.append(
                {
                    "priority": priority,
                    "stage_id": action.lower(),
                    "action": action,
                    "reason": reason,
                    "expected_improvements": expected_improvements,
                    "acceptance_checks": acceptance_checks,
                }
            )

    if WeakRegionCause.IMAGE_QUALITY_WEAK.value in causes:
        add(1, "RECOVER_SOURCE_IMAGE_QUALITY", "Fix unusable source frames first.")
    if any(c in causes for c in (WeakRegionCause.VIEW_GRAPH_ISOLATION.value, WeakRegionCause.RETRIEVAL_GAP.value)):
        add(1, "TARGETED_BRIDGE_PAIR_SELECTION", "Connect weak images to healthy/cross-route anchors.")
    if WeakRegionCause.MATCHING_SHORTAGE.value in causes:
        add(2, "TARGETED_DENSE_REMATCHING", "Run stronger matching only on weak and bridge pairs.")
    if WeakRegionCause.MATCHING_AMBIGUITY.value in causes:
        add(2, "TIGHTEN_GEOMETRIC_VERIFICATION", "Reject repeated/symmetric false correspondences before BA.")
    if WeakRegionCause.TRACK_FRAGMENTATION.value in causes:
        add(3, "MULTIVIEW_TRACK_REPAIR", "Extend/merge geometrically consistent multi-view tracks.")
        add(4, "LOCAL_RETRIANGULATION_AND_BA", "Re-triangulate and optimize with healthy anchors constrained.")
    if WeakRegionCause.HIGH_REPROJECTION_ERROR.value in causes:
        add(3, "PRUNE_RETRIANGULATE_LOCAL_BA", "Prune inconsistent tracks, verify intrinsics, and re-optimize locally.")
    if WeakRegionCause.MAP_SUPPORT_SPARSE.value in causes:
        add(2, "EXPAND_EXISTING_CORRESPONDENCE_SUPPORT", "Search existing images before assuming new capture is needed.")
    if WeakRegionCause.LOW_PARALLAX.value in causes:
        if has_anchors:
            add(1, "SEARCH_EXISTING_LONG_BASELINE_ANCHORS", "Use existing long-baseline/cross-route views first.")
        add(5, "TARGETED_LATERAL_OBLIQUE_RECAPTURE", "If parallax is absent, capture a new independent baseline.")
    if WeakRegionCause.PLANAR_PAIR_DOMINANCE.value in causes:
        if has_anchors:
            add(
                1,
                "SEARCH_EXISTING_LONG_BASELINE_ANCHORS",
                "Use existing long-baseline/cross-route views first.",
            )
        add(5, "TARGETED_LATERAL_OBLIQUE_RECAPTURE", "Planar pair models cannot manufacture parallax.")
    if WeakRegionCause.RELATIVE_POSE_INCONSISTENT.value in causes:
        add(
            2,
            "TIGHTEN_GEOMETRIC_VERIFICATION",
            "Reject two-view poses that disagree with the reconstruction.",
        )
        add(3, "PRUNE_RETRIANGULATE_LOCAL_BA", "Prune inconsistent pairs and re-optimize locally.")
    if causes == [WeakRegionCause.UNEXPLAINED_WEAKNESS.value]:
        add(1, "COLLECT_BUILD_EVIDENCE", "Save retrieval, matching, verification and image-quality evidence.")
    actions.sort(key=lambda x: x["priority"])
    if WeakRegionCause.IMAGE_QUALITY_WEAK.value in causes:
        recapture = "RECOMMENDED_IF_SOURCE_FRAMES_CANNOT_BE_RECOVERED"
    elif (
        WeakRegionCause.LOW_PARALLAX.value in causes
        or WeakRegionCause.PLANAR_PAIR_DOMINANCE.value in causes
    ):
        recapture = "CONDITIONAL_AFTER_EXISTING_LONG_BASELINE_SEARCH"
    else:
        recapture = "NOT_FIRST_ACTION"
    return actions, recapture


def _action_contract(action: str) -> tuple[list[str], list[str]]:
    contracts = {
        "RECOVER_SOURCE_IMAGE_QUALITY": (
            ["usable source frames are recovered or their unavailability is documented"],
            ["re-measure image-quality evidence before considering capture"],
        ),
        "TARGETED_BRIDGE_PAIR_SELECTION": (
            ["stronger weak-to-anchor or cross-route pair support in existing data"],
            ["recompute selected/candidate pairs and anchor connectivity for the region"],
        ),
        "TARGETED_DENSE_REMATCHING": (
            ["more verified correspondences on targeted weak and bridge pairs"],
            ["review matches, inliers, and inlier ratios for targeted pairs"],
        ),
        "TIGHTEN_GEOMETRIC_VERIFICATION": (
            ["fewer geometrically inconsistent correspondences or pair poses"],
            ["rerun geometric verification and inspect retained and rejected pair evidence"],
        ),
        "MULTIVIEW_TRACK_REPAIR": (
            ["more consistent multi-view support for affected tracks"],
            ["recompute track-length and two-view diagnostics and inspect stable-region regressions"],
        ),
        "LOCAL_RETRIANGULATION_AND_BA": (
            ["better-supported local points with consistent reprojection evidence"],
            ["recompute triangulation, parallax, and reprojection diagnostics with anchors constrained"],
        ),
        "PRUNE_RETRIANGULATE_LOCAL_BA": (
            ["fewer outlier-supported points and more consistent local reprojection evidence"],
            ["verify residual and track diagnostics and compare frozen weak/stable holdouts"],
        ),
        "EXPAND_EXISTING_CORRESPONDENCE_SUPPORT": (
            ["more existing observations supporting the weak region"],
            ["recompute point support and track coverage from existing images"],
        ),
        "SEARCH_EXISTING_LONG_BASELINE_ANCHORS": (
            ["existing long-baseline or cross-route support for parallax-sensitive pairs"],
            ["record baseline evidence and recompute triangulation-angle and anchor metrics"],
        ),
        "TARGETED_LATERAL_OBLIQUE_RECAPTURE": (
            ["new independent baseline observations if the counterfactual leaves a measured deficit"],
            [
                "authorize only after existing_data_counterfactual_complete and validate new frames on frozen weak/stable holdouts"
            ],
        ),
        "COLLECT_BUILD_EVIDENCE": (
            ["evidence coverage sufficient to separate unresolved root causes"],
            ["confirm retrieval, matching, verification, and image-quality evidence availability"],
        ),
    }
    return contracts.get(
        action,
        (
            [f"observable evidence relevant to {action} is improved or its deficit is characterized"],
            [f"rerun weak-region diagnostics and record the observed result for {action}"],
        ),
    )


def _solution_plan(actions: list[dict], recapture: str) -> dict:
    counterfactual_stage = {
        "stage_id": "existing_data_counterfactual",
        "action": "existing_data_counterfactual",
        "reason": "Evaluate declared existing-data repairs on frozen weak and stable holdouts.",
        "expected_improvements": ["observed repairability of existing-data interventions is measured"],
        "acceptance_checks": [
            "all declared existing-data stages have weak and stable holdout comparisons"
        ],
    }
    existing_data_steps = [
        dict(item)
        for item in actions
        if item["action"] != "TARGETED_LATERAL_OBLIQUE_RECAPTURE"
    ]
    existing_data_steps.append(counterfactual_stage)

    recapture_steps = [
        {
            "stage_id": "recapture_decision",
            "decision": recapture,
            "after_stage_id": counterfactual_stage["stage_id"],
            "acceptance_checks": [
                "do not authorize recapture until the existing-data counterfactual is complete"
            ],
        }
    ]
    recapture_steps.extend(
        {
            **dict(item),
            "after_stage_id": "recapture_decision",
        }
        for item in actions
        if item["action"] == "TARGETED_LATERAL_OBLIQUE_RECAPTURE"
    )
    return {
        "schema_version": 1,
        "policy": "EXISTING_DATA_FIRST",
        "authorization_status": "NOT_AUTHORIZED",
        "counterfactual_status": "REQUIRED_NOT_RUN",
        "required_stages": [
            item["stage_id"]
            for item in existing_data_steps
            if item["stage_id"] != "existing_data_counterfactual"
        ],
        "counterfactual_trials": [],
        "counterfactual_result": None,
        "blocked_by": [
            "existing_data_counterfactual_complete",
            "heldout_provenance_verified",
            "stable_holdout_comparison",
            "structural_deficit_evidence",
        ],
        "existing_data_steps": existing_data_steps,
        "recapture_steps": recapture_steps,
        "counterfactual_required_before_recapture": True,
        "validation_contract": {
            "required_checks": [
                "re-run weak-region metrics after each repair stage",
                "compare frozen weak and stable holdouts with the same validation protocol",
                "record observed evidence before accepting a repair or recapture decision",
            ],
            "counterfactual_completion_metric": "existing_data_counterfactual_complete",
            "weak_region_holdout_required": True,
            "stable_region_holdout_required": True,
            "observed_evidence_only": True,
        },
    }


def _pair_stats(pair_rows, map_data, members, anchors, cfg) -> dict:
    name_index = {name: i for i, name in enumerate(map_data.image_names)}
    id_index = {int(v): i for i, v in enumerate(map_data.image_ids.tolist())}
    relevant = []
    internal = anchor_pairs = 0
    for row in pair_rows:
        i = _endpoint(row, "i", name_index, id_index)
        j = _endpoint(row, "j", name_index, id_index)
        if i is None or j is None:
            continue
        is_internal = i in members and j in members
        is_anchor = (i in members and j in anchors) or (j in members and i in anchors)
        if is_internal or is_anchor:
            relevant.append(row)
            internal += int(is_internal)
            anchor_pairs += int(is_anchor)
    selected = [r["selected"] for r in relevant if r.get("selected") is not None]
    attempted = [r for r in relevant if r.get("attempted") is True or r.get("num_matches") is not None]
    verified = [r for r in relevant if r.get("verified") is True or (r.get("num_inliers") or 0) > 0]
    planar_flags = []
    pose_flags = []
    degenerate_count = 0
    pose_pairs = 0
    for row in relevant:
        rel = reconstructed_relative_error(map_data, row)
        rot_err = None if rel is None else rel[0]
        trans_err = None if rel is None else rel[1]
        flags = pair_model_flags(
            row,
            rot_err_deg=rot_err,
            trans_err=trans_err,
            min_h=cfg.min_homography_support,
            max_e=cfg.max_essential_support,
            max_rel_rot_deg=cfg.max_relative_rotation_deg,
            max_rel_dir=cfg.max_relative_direction,
        )
        if flags["has_model"]:
            planar_flags.append(flags["planar"])
        if flags["degenerate"]:
            degenerate_count += 1
        if flags["has_pose"]:
            pose_pairs += 1
            pose_flags.append(flags["pose_bad"])
    return {
        "selection_available": bool(selected),
        "matching_available": any(r.get("num_matches") is not None for r in attempted),
        "verification_available": any(r.get("num_inliers") is not None for r in attempted),
        "candidate_pairs": len(relevant),
        "internal_pairs": internal,
        "anchor_pairs": anchor_pairs,
        "selected_pairs": sum(bool(v) for v in selected),
        "selected_ratio": float(np.mean(selected)) if selected else None,
        "attempted_pairs": len(attempted),
        "verified_pairs": len(verified),
        "matches_p50": _pct([r.get("num_matches") for r in attempted], 50),
        "inliers_p50": _pct([r.get("num_inliers") for r in attempted], 50),
        "inlier_ratio_p50": _pct([r.get("inlier_ratio") for r in attempted], 50),
        "planar_pair_fraction": float(np.mean(planar_flags)) if planar_flags else None,
        "pose_inconsistent_fraction": float(np.mean(pose_flags)) if pose_flags else None,
        "degenerate_pair_count": degenerate_count,
        "two_view_pose_pairs": pose_pairs,
    }


def _pair_baseline(rows) -> dict:
    attempted = [r for r in rows if r.get("attempted") is True or r.get("num_matches") is not None]
    return {
        "matches": _distribution([r.get("num_matches") for r in attempted]),
        "inliers": _distribution([r.get("num_inliers") for r in attempted]),
        "inlier_ratio": _distribution([r.get("inlier_ratio") for r in attempted]),
    }


def _quality_stats(rows) -> dict:
    measured = [r for r in rows if r.get("image_quality_badness") is not None]
    if not measured:
        return {"quality_images": 0, "bad_fraction": None}
    bad = [float(r["image_quality_badness"]) >= 0.65 for r in measured]
    return {"quality_images": len(measured), "bad_fraction": float(np.mean(bad))}


def _image_point_indices(map_data: MapData) -> list[np.ndarray]:
    lookup = map_data.image_index()
    buckets = [[] for _ in range(map_data.num_images)]
    for pidx, obs in enumerate(map_data.track_image_ids):
        for image_id in set(int(v) for v in obs.tolist()):
            if image_id in lookup:
                buckets[lookup[image_id]].append(pidx)
    return [np.asarray(v, dtype=int) for v in buckets]


def _anchor_track_ratio(map_data, points, members, anchors) -> float | None:
    if not len(points) or not anchors:
        return None
    lookup = map_data.image_index()
    count = 0
    for pidx in points:
        observed = {lookup[int(v)] for v in map_data.track_image_ids[pidx] if int(v) in lookup}
        count += int(bool(observed & members) and bool(observed & anchors))
    return count / len(points)


def _assign_unregistered(map_data, rows, max_distance) -> dict[int, list[dict]]:
    if not map_data.num_images or max_distance <= 0:
        return {}
    tree = cKDTree(map_data.image_centers)
    out: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("registered") is not False:
            continue
        xyz = [row.get("x"), row.get("y"), row.get("z")]
        if any(v is None for v in xyz):
            continue
        distance, index = tree.query(np.asarray(xyz, dtype=float), k=1)
        if float(distance) <= max_distance:
            out.setdefault(int(index), []).append(row)
    return out


def _anchors(centers, members, weak, radius) -> tuple[int, ...]:
    if radius <= 0:
        return ()
    tree = cKDTree(centers)
    out = set()
    for i in members:
        out.update(int(j) for j in tree.query_ball_point(centers[i], radius) if int(j) not in weak)
    return tuple(sorted(out))


def _clusters(centers, weak, radius) -> list[tuple[int, ...]]:
    if not weak:
        return []
    ids = sorted(weak)
    if radius <= 0:
        return [(i,) for i in ids]
    tree = cKDTree(centers[ids])
    adj = [set() for _ in ids]
    for a, b in tree.query_pairs(radius):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    seen = set()
    out = []
    for start in range(len(ids)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(ids[u])
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(tuple(sorted(comp)))
    return out


def _cluster_radius(nn, configured) -> float:
    if configured is not None:
        if configured < 0:
            raise ValueError("cluster_radius must be >= 0")
        return float(configured)
    finite = nn[np.isfinite(nn) & (nn > 0)]
    if not len(finite):
        return 0.0
    return max(2.5 * float(np.median(finite)), 1.5 * float(np.percentile(finite, 90)))


def _distribution(values) -> dict:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not values:
        return {"count": 0, "median": None, "mad": None, "p10": None, "p90": None}
    arr = np.asarray(values)
    median = float(np.median(arr))
    return {
        "count": len(arr),
        "median": median,
        "mad": float(np.median(np.abs(arr - median))),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _badness(value, baseline, direction, threshold) -> float:
    if value is None or baseline.get("count", 0) < 3:
        return 0.0
    median = baseline["median"]
    mad = baseline["mad"] or 0.0
    scale = 1.4826 * mad
    if scale < 1e-12:
        spread = (baseline["p90"] or median) - (baseline["p10"] or median)
        scale = max(spread / 2.563, abs(median) * 0.1, 1e-9)
    z = (median - float(value)) / scale if direction == "low" else (float(value) - median) / scale
    return float(np.clip(z / max(threshold, 1e-6), 0.0, 1.0))


def _pct(values, q) -> float | None:
    arr = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.percentile(arr, q)) if arr else None



def _endpoint(row, suffix, name_index, id_index) -> int | None:
    name = row.get(f"image_{suffix}")
    if name is not None and str(name) in name_index:
        return name_index[str(name)]
    image_id = row.get(f"image_id_{suffix}")
    return id_index.get(int(image_id)) if image_id is not None else None


def _mode(evidence: BuildEvidence) -> str:
    if evidence.has_pair_selection and evidence.has_matching and evidence.has_geometric_verification and evidence.has_image_quality:
        return "FULL_BUILD_EVIDENCE"
    if evidence.has_matching or evidence.has_geometric_verification or evidence.has_image_quality:
        return "MAP_PLUS_PARTIAL_BUILD_EVIDENCE"
    return "MAP_ONLY"


def _flatten(region) -> dict:
    metrics = region["metrics"]
    pair = region["pair_evidence"]
    return {
        "rank": region.get("rank"),
        "region_id": region["region_id"],
        "status": region["status"],
        "severity_score": region["severity_score"],
        "primary_cause": region["primary_cause"],
        "root_causes": "|".join(region["root_causes"]),
        "center_x": region["center"][0],
        "center_y": region["center"][1],
        "center_z": region["center"][2],
        "num_images": metrics.get("num_images"),
        "num_points": metrics.get("num_points"),
        "anchor_strong_edges": metrics.get("anchor_strong_edges"),
        "anchor_track_ratio": metrics.get("anchor_track_ratio"),
        "track_length_p50": metrics.get("track_length_p50"),
        "two_view_fraction": metrics.get("two_view_fraction"),
        "triangulation_angle_p50_deg": metrics.get("triangulation_angle_p50_deg"),
        "reprojection_error_p90_px": metrics.get("reprojection_error_p90_px"),
        "pair_inliers_p50": pair.get("inliers_p50"),
        "pair_inlier_ratio_p50": pair.get("inlier_ratio_p50"),
        "planar_pair_fraction": pair.get("planar_pair_fraction"),
        "pose_inconsistent_fraction": pair.get("pose_inconsistent_fraction"),
        "recapture": region["recapture"],
    }


def _plot(path: Path, images: list[dict], regions: list[dict]) -> None:
    try:
        import plotly.express as px
    except ImportError:
        return
    if not images:
        return
    cause = {}
    region_id = {}
    for region in regions:
        for image_id in region["image_ids"]:
            cause[int(image_id)] = region["primary_cause"]
            region_id[int(image_id)] = region["region_id"]
    plot_rows = [
        {
            **row,
            "primary_cause": cause.get(int(row["image_id"]), "HEALTHY/UNFLAGGED"),
            "region_id": region_id.get(int(row["image_id"]), 0),
        }
        for row in images
    ]
    fig = px.scatter_3d(
        plot_rows,
        x="x",
        y="y",
        z="z",
        color="weakness_score",
        hover_name="image_name",
        hover_data=["region_id", "primary_cause"],
        range_color=[0.0, 1.0],
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(title="SfM weak-region root-cause diagnosis")
    fig.write_html(str(path), include_plotlyjs="cdn")
