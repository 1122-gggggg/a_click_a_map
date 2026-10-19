"""Video-first session selection runner. Does not invent STRONG geometry."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

from sfm_qa.session_select.edges import build_session_edges
from sfm_qa.session_select.export import write_selection_outputs
from sfm_qa.session_select.image_qa import evaluate_video
from sfm_qa.session_select.ingest import discover_sessions
from sfm_qa.session_select.motion import scan_video
from sfm_qa.session_select.prebuild import propose_prebuild_set

try:
    from sfm_qa.session_select.types import ROLES, SessionQuality
except ImportError:  # sibling types may land later
    ROLES = (
        "BASE_CORE",
        "BASE_SUPPORT",
        "APPEARANCE_REF",
        "GEOMETRY_REINFORCEMENT",
        "UPDATE_CANDIDATE",
        "NEW_SUBMAP",
        "QUARANTINE",
        "REJECT",
        "VALIDATION_ONLY",
    )

    @dataclass
    class SessionQuality:  # type: ignore[no-redef]
        session_id: str
        timestamp: str | None = None
        num_frames: int = 0
        num_keyframes: int = 0
        registered_ratio: float | None = None
        sharpness_median: float | None = None
        sharpness_p10: float | None = None
        underexposed_ratio: float | None = None
        overexposed_ratio: float | None = None
        near_duplicate_ratio: float | None = None
        exposure_mean: float | None = None
        parallax_ratio: float | None = None
        low_parallax_ratio: float | None = None
        hover_ratio: float | None = None
        pure_rotation_ratio: float | None = None
        fast_motion_ratio: float | None = None
        unproven_ratio: float | None = None
        epipolar_outlier_ratio_median: float | None = None
        essential_inlier_ratio_median: float | None = None
        flow_median_px: float | None = None
        motion_parallax_median_px: float | None = None
        num_tracks: int | None = None
        num_observations: int | None = None
        median_track_length: float | None = None
        long_track_ratio: float | None = None
        reprojection_rmse: float | None = None
        reprojection_p90: float | None = None
        parallax_median_deg: float | None = None
        parallax_p10_deg: float | None = None
        positive_depth_ratio: float | None = None
        convex_hull_coverage: float | None = None
        grid_occupancy_4x4: float | None = None
        fim_condition_number: float | None = None
        fim_logdet: float | None = None
        rotation_cycle_error: float | None = None
        translation_consistency: float | None = None
        connected_components: int | None = None
        average_degree: float | None = None
        num_bridges: int | None = None
        num_articulation_points: int | None = None
        fiedler_value: float | None = None
        internal_quality_score: float = 0.0
        internal_status: str = "REJECT"
        reasons: tuple[str, ...] = ()


def _construct(cls: type, **kwargs: Any) -> Any:
    if is_dataclass(cls):
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in kwargs.items() if key in allowed})
    return cls(**kwargs)


def _lookup(config: Mapping[str, Any] | None, dotted: str, default: Any = None) -> Any:
    if config:
        try:
            from sfm_qa.session_select.config import lookup

            return lookup(dict(config), dotted, default)
        except Exception:
            node: Any = config
            for part in dotted.split("."):
                if not isinstance(node, Mapping) or part not in node:
                    return default
                node = node[part]
            return node
    return default


def _resolve_config(config: Any) -> dict[str, Any]:
    if config is None:
        try:
            from sfm_qa.session_select.config import load_config

            loaded = load_config()
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            return {}
    if isinstance(config, (str, Path)):
        path = Path(config)
        try:
            from sfm_qa.session_select.config import load_config

            loaded = load_config(path)
            return dict(loaded) if isinstance(loaded, Mapping) else {}
        except Exception:
            try:
                import yaml

                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return dict(payload) if isinstance(payload, Mapping) else {}
            except Exception:
                return {}
    if isinstance(config, Mapping):
        return dict(config)
    return {}


def _load_vpr_payload(path: str | Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"pairs": payload}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source} must contain a JSON object or list of pair objects")
    return payload


def _ratio(mapping: Mapping[str, Any], key: str) -> float:
    try:
        value = float(mapping.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _video_only_status(
    qa: Mapping[str, Any],
    motion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[str, float, list[str]]:
    """Video-only admission status.

    This status can be USABLE but never STRONG because no cross-session geometry
    exists yet. Motion/epipolar evidence is a pre-build filter, not merge proof.
    """

    reasons = [str(item) for item in (qa.get("reasons") or ())]
    reasons.extend(str(item) for item in (motion.get("reasons") or ()))
    reasons.append("video_only_never_strong_geometry")

    sharp = qa.get("sharpness_p10")
    reject_sharp = float(
        _lookup(config, "internal_status.reject_sharpness_p10_below", 15.0) or 15.0
    )
    if sharp is not None and float(sharp) < reject_sharp:
        reasons.append(f"sharpness_p10_below_heuristic_{reject_sharp}")
        return "REJECT", 0.05, reasons

    missing = sharp is None or "missing_opencv" in reasons or "unreadable_video" in reasons
    if missing:
        reasons.append("video_only_missing_geometry_fail_closed")
        return "WEAK", 0.2, reasons

    parallax = _ratio(motion, "parallax_ratio")
    low_parallax = _ratio(motion, "low_parallax_ratio")
    hover = _ratio(motion, "hover_ratio")
    pure_rotation = _ratio(motion, "pure_rotation_ratio")
    fast_motion = _ratio(motion, "fast_motion_ratio")
    unproven = _ratio(motion, "unproven_ratio")
    duplicate = _ratio(qa, "near_duplicate_ratio")
    under = _ratio(qa, "underexposed_ratio")
    over = _ratio(qa, "overexposed_ratio")
    bad_exposure = min(1.0, under + over)

    epi_raw = motion.get("epipolar_outlier_ratio_median")
    epi = None
    if epi_raw is not None:
        try:
            epi = max(0.0, min(1.0, float(epi_raw)))
        except (TypeError, ValueError):
            epi = None

    low_credit = float(_lookup(config, "prebuild.low_parallax_credit", 0.35) or 0.35)
    geometric_motion = min(1.0, parallax + low_credit * low_parallax)
    degenerate = min(1.0, hover + pure_rotation + fast_motion + 0.5 * low_parallax)

    min_geom = float(
        _lookup(config, "prebuild.min_parallax_ratio_for_usable", 0.20) or 0.20
    )
    max_degenerate = float(
        _lookup(config, "prebuild.max_degenerate_ratio_for_usable", 0.70) or 0.70
    )
    max_fast = float(_lookup(config, "prebuild.max_fast_motion_ratio_for_usable", 0.45) or 0.45)
    max_dup = float(
        _lookup(config, "prebuild.max_near_duplicate_ratio_for_usable", 0.75) or 0.75
    )
    max_exposure = float(
        _lookup(config, "prebuild.max_bad_exposure_ratio_for_usable", 0.75) or 0.75
    )
    max_epi = float(
        _lookup(config, "prebuild.max_epipolar_outlier_ratio_for_usable", 0.75) or 0.75
    )

    sharp_quality = min(1.0, float(sharp) / max(float(
        _lookup(config, "prebuild.sharpness_reference", 100.0) or 100.0
    ), 1e-9))
    epi_quality = 0.5 if epi is None else 1.0 - epi
    score = max(
        0.0,
        min(
            0.90,
            0.38 * geometric_motion
            + 0.22 * sharp_quality
            + 0.12 * (1.0 - duplicate)
            + 0.10 * (1.0 - bad_exposure)
            + 0.10 * epi_quality
            + 0.08 * (1.0 - unproven)
            - 0.12 * degenerate,
        ),
    )

    blockers: list[str] = []
    if geometric_motion < min_geom:
        blockers.append(f"geometric_motion_below_heuristic_{min_geom}")
    if degenerate > max_degenerate:
        blockers.append(f"degenerate_motion_above_heuristic_{max_degenerate}")
    if fast_motion > max_fast:
        blockers.append(f"fast_motion_above_heuristic_{max_fast}")
    if duplicate > max_dup:
        blockers.append(f"near_duplicate_above_heuristic_{max_dup}")
    if bad_exposure > max_exposure:
        blockers.append(f"bad_exposure_above_heuristic_{max_exposure}")
    if epi is not None and epi > max_epi:
        blockers.append(f"epipolar_outlier_above_heuristic_{max_epi}")

    if blockers:
        reasons.extend(blockers)
        return "WEAK", min(score, 0.45), reasons

    reasons.append("video_quality_and_motion_admissible_for_geometry_probe")
    return "USABLE", max(score, 0.46), reasons


def _session_quality(
    record: Any,
    qa: Mapping[str, Any],
    motion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    status, score, reasons = _video_only_status(qa, motion, config)
    sampled = int(qa.get("sampled") or 0)
    num_frames = int(getattr(record, "num_frames", 0) or 0)
    return _construct(
        SessionQuality,
        session_id=str(record.session_id),
        timestamp=getattr(record, "timestamp", None),
        num_frames=num_frames,
        num_keyframes=sampled,
        registered_ratio=None,
        sharpness_median=qa.get("sharpness_median"),
        sharpness_p10=qa.get("sharpness_p10"),
        underexposed_ratio=qa.get("underexposed_ratio"),
        overexposed_ratio=qa.get("overexposed_ratio"),
        near_duplicate_ratio=qa.get("near_duplicate_ratio"),
        exposure_mean=qa.get("exposure_mean"),
        parallax_ratio=motion.get("parallax_ratio"),
        low_parallax_ratio=motion.get("low_parallax_ratio"),
        hover_ratio=motion.get("hover_ratio"),
        pure_rotation_ratio=motion.get("pure_rotation_ratio"),
        fast_motion_ratio=motion.get("fast_motion_ratio"),
        unproven_ratio=motion.get("unproven_ratio"),
        epipolar_outlier_ratio_median=motion.get("epipolar_outlier_ratio_median"),
        essential_inlier_ratio_median=motion.get("essential_inlier_ratio_median"),
        flow_median_px=motion.get("flow_median_px"),
        motion_parallax_median_px=motion.get("motion_parallax_median_px"),
        num_tracks=None,
        num_observations=None,
        median_track_length=None,
        long_track_ratio=None,
        reprojection_rmse=None,
        reprojection_p90=None,
        parallax_median_deg=None,
        parallax_p10_deg=None,
        positive_depth_ratio=None,
        convex_hull_coverage=None,
        grid_occupancy_4x4=None,
        fim_condition_number=None,
        fim_logdet=None,
        rotation_cycle_error=None,
        translation_consistency=None,
        connected_components=None,
        average_degree=None,
        num_bridges=None,
        num_articulation_points=None,
        fiedler_value=None,
        internal_quality_score=float(score),
        internal_status=status,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _greedy_select(qualities: list[Any], edges: list[Any], config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from sfm_qa.session_select.select_core import greedy_select_core

        result = greedy_select_core(qualities, edges, config)
        if isinstance(result, Mapping):
            return dict(result)
    except Exception:
        pass
    eligible = [
        row for row in qualities if getattr(row, "internal_status", None) in {"STRONG", "USABLE"}
    ]
    if not eligible:
        return {
            "core": [],
            "support": [],
            "selected": [],
            "scores": {},
            "stop_reason": "no_strong_or_usable_seed",
            "seed": None,
        }
    # Timestamp is intentionally omitted from the ranking key.
    seed_row = max(eligible, key=lambda row: (float(row.internal_quality_score), row.session_id))
    return {
        "core": [seed_row.session_id],
        "support": [],
        "selected": [seed_row.session_id],
        "scores": {},
        "stop_reason": "video_only_seed_only_no_admissible_neighbor",
        "seed": seed_row.session_id,
    }


def _classify(
    qualities: list[Any],
    edges: list[Any],
    core: list[str],
    support: list[str],
    config: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, str]:
    try:
        from sfm_qa.session_select.classify_remainder import classify_remainder

        assigned = classify_remainder(qualities, edges, core, support, config, extra=extra)
        if isinstance(assigned, Mapping):
            return {str(key): str(value) for key, value in assigned.items()}
    except Exception:
        pass
    roles: dict[str, str] = {}
    for sid in core:
        roles[sid] = "BASE_CORE"
    for sid in support:
        roles[sid] = "BASE_SUPPORT"
    for row in qualities:
        sid = row.session_id
        if sid in roles:
            continue
        status = getattr(row, "internal_status", "WEAK")
        if status in {"REJECT", "INCONSISTENT"}:
            roles[sid] = "REJECT"
            continue
        if status == "WEAK" and row.sharpness_p10 is not None and float(row.sharpness_p10) <= 0:
            roles[sid] = "REJECT"
            continue
        if status in {"STRONG", "USABLE"}:
            roles[sid] = "NEW_SUBMAP"
        else:
            roles[sid] = "QUARANTINE"
    return roles


def _reason_map(qualities: list[Any], roles: Mapping[str, str]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in qualities:
        role = roles.get(row.session_id, "QUARANTINE")
        bits = [f"role={role}", f"status={row.internal_status}"]
        bits.extend(list(getattr(row, "reasons", ()) or ())[:8])
        if role == "NEW_SUBMAP":
            bits.append("no_reliable_geometric_edge")
        reasons[row.session_id] = "; ".join(bits)
    return reasons


def select_sessions(
    video_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    config: Any = None,
    maps_dir: str | Path | None = None,
    vpr_candidates: str | Path | None = None,
    edge_probes: str | Path | None = None,
    **aliases: Any,
) -> dict[str, Any]:
    """QA videos, propose a pre-build subset, then select only verified geometry."""

    if video_dir is None:
        video_dir = aliases.get("videos")
    if output_dir is None:
        output_dir = aliases.get("output")
    if maps_dir is None:
        maps_dir = aliases.get("maps")
    if vpr_candidates is None:
        vpr_candidates = aliases.get("vpr") or aliases.get("vpr_candidates")
    if edge_probes is None:
        edge_probes = aliases.get("edge_probe") or aliases.get("edge_probes")
    if isinstance(config, (str, Path)) and maps_dir is None and aliases.get("config") is None:
        # Positional (videos, output, maps) from callers that treat the third slot as maps.
        maybe_maps = Path(config)
        if maybe_maps.exists() and not str(maybe_maps).lower().endswith((".yaml", ".yml", ".json")):
            maps_dir = maybe_maps
            config = aliases.get("config")

    if video_dir is None or output_dir is None:
        raise TypeError("select_sessions requires video_dir/output_dir (or videos/output)")

    cfg = _resolve_config(config)
    records = discover_sessions(video_dir, cfg)
    qualities = []
    sample_limit = int(_lookup(cfg, "image_qa.sharpness_sample_limit", 32) or 32)
    motion_interval = float(_lookup(cfg, "prebuild.motion_interval_seconds", 0.5) or 0.5)
    for record in records:
        qa = evaluate_video(
            record.video_path,
            sample_limit=sample_limit,
            underexposure_mean=float(_lookup(cfg, "image_qa.underexposure_mean", 20.0) or 20.0),
            overexposure_mean=float(_lookup(cfg, "image_qa.overexposure_mean", 235.0) or 235.0),
            near_duplicate_hist_corr=float(
                _lookup(cfg, "image_qa.near_duplicate_hist_corr", 0.995) or 0.995
            ),
            blur_variance_reject=float(
                _lookup(cfg, "image_qa.blur_variance_reject", 25.0) or 25.0
            ),
        )
        motion = scan_video(record.video_path, interval_seconds=motion_interval)
        qualities.append(_session_quality(record, qa, motion, cfg))

    session_ids = [row.session_id for row in qualities]
    vpr_payload = _load_vpr_payload(vpr_candidates)
    probe_payload = _load_vpr_payload(edge_probes)
    edges = build_session_edges(
        session_ids,
        maps_dir=maps_dir,
        vpr_payload=vpr_payload,
        edge_probe_payload=probe_payload,
        config=cfg,
    )
    for edge in edges:
        if getattr(edge, "status", None) in {"STRONG", "USABLE"} and (
            int(getattr(edge, "num_verified_pairs", 0) or 0) <= 0
            or not getattr(edge, "independent_artifact", False)
            or str(getattr(edge, "evidence_scope", "")) != "exact_pair"
            or not getattr(edge, "geometry_complete", False)
            or not getattr(edge, "group_holdout_disjoint", False)
        ):
            edge.status = "AMBIGUOUS"
            if hasattr(edge, "reasons"):
                edge.reasons = tuple(edge.reasons) + (
                    "maps_absent_or_unverified_not_strong",
                )

    # Phase A: proposal-only. Candidate/VPR graph is allowed to rank expensive
    # geometric verification, never to authorize a merge.
    prebuild = propose_prebuild_set(qualities, edges, cfg)

    # Phase B: final Base roles remain fail-closed on verified geometry.
    extra = {
        "change_score": {row.session_id: 0.0 for row in qualities},
        "loc": {},
        "influences": {},
        "appearance_shift": {},
    }
    selection = _greedy_select(qualities, edges, cfg)
    core = list(selection.get("core") or [])
    support = list(selection.get("support") or [])
    if not core and support:
        core, support = [support[0]], support[1:]
    roles = _classify(qualities, edges, core, support, cfg, extra)
    for sid, role in list(roles.items()):
        if role not in ROLES:
            roles[sid] = "QUARANTINE"
    reasons = _reason_map(qualities, roles)

    dest = Path(output_dir)
    write_selection_outputs(
        dest,
        qualities=qualities,
        edges=edges,
        roles=roles,
        reasons=reasons,
        config=cfg,
        extra=extra,
        selection=selection,
        prebuild=prebuild,
    )
    role_counts = {name: 0 for name in ROLES}
    role_counts.update(Counter(roles.values()))
    assignments = [
        {"session_id": sid, "role": roles[sid], "reason": reasons.get(sid, "")}
        for sid in session_ids
    ]
    return {
        "sessions": session_ids,
        "roles": roles,
        "assignments": assignments,
        "role_counts": role_counts,
        "output": str(dest),
        "prebuild": prebuild,
        "selection": selection,
        "qualities": qualities,
        "edges": edges,
    }


__all__ = ["select_sessions"]
