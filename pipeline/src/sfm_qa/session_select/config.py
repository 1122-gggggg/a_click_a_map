"""YAML config: merge user overlay over heuristic defaults. No magic cutoffs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("defaults.yaml")

HEURISTIC_NOTES: dict[str, str] = {
    "prebuild.relative_admission": "cohort-relative proposal mode; geometry remains authoritative",
    "prebuild.relative_marginal_keep_ratio": "relative diminishing-return stop for proposal set",
    "prebuild.min_video_score": "legacy/reference score; not a default relative eligibility gate",
    "prebuild.min_marginal_gain": "legacy proposal setting retained for compatibility",
    "prebuild.max_no_graph_sessions": "heuristic cap when retrieval graph is unavailable",
    "prebuild.weights": "heuristic proposal objective weights; VPR remains candidate-only",
    "prebuild.video_weights": "heuristic video-admission weights; not a fitted classifier",
    "prebuild.min_parallax_ratio_for_usable": "heuristic motion-quality gate before geometry",
    "selection.min_information_gain": "heuristic stop gate on Δinformation",
    "selection.min_coverage_gain": "heuristic stop gate on Δcoverage",
    "selection.min_independent_bridges_for_strong_edge": "heuristic fail-closed independent support",
    "selection.weights": "heuristic objective weights; not a fitted score",
    "internal_status.reject_registered_ratio_below": "heuristic; missing metric fails closed",
    "internal_status.reject_positive_depth_below": "heuristic; missing metric fails closed",
    "internal_status.weak_parallax_p10_deg": "heuristic low-parallax WEAK gate",
    "internal_status.inconsistent_cycle_error_deg": "heuristic rotation-cycle INCONSISTENT gate",
    "edge.min_cross_tracks_for_verified": "heuristic; VPR alone never verifies",
    "edge.holdout_fraction": "heuristic explicit 80/20; never fit+validate on same points",
    "edge.require_exact_pair_scope": "fail-closed invariant; shared-map/VPR cannot be STRONG/USABLE",
    "edge.require_group_disjoint_holdout": "fail-closed invariant; fit and holdout must not share a group",
    "edge.require_complete_geometry": "fail-closed invariant; missing finite geometry cannot be STRONG/USABLE",
    "edge.min_holdout_inlier_ratio": "heuristic holdout quality; not a calibrated success probability",
    "edge.max_holdout_residual_px": "heuristic holdout residual; not a calibrated success probability",
    "image_qa.blur_variance_reject": "heuristic Laplacian variance",
    "influence.high_reproj_delta": "heuristic offline contrast; not a live LOO authority",
}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load defaults, then merge an optional overlay. Never injects video-name cores."""

    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        defaults = yaml.safe_load(handle) or {}
    if not isinstance(defaults, dict):
        raise ValueError("defaults.yaml must be a mapping")
    if path is None:
        return defaults
    overlay_path = Path(path)
    if not overlay_path.is_file():
        raise FileNotFoundError(overlay_path)
    with overlay_path.open(encoding="utf-8") as handle:
        overlay = yaml.safe_load(handle) or {}
    if not isinstance(overlay, dict):
        raise ValueError(f"{overlay_path} must be a mapping")
    return _deep_merge(defaults, overlay)


def heuristic_note(key: str) -> str:
    """Return the labeled-heuristic provenance string for a config key."""

    if key in HEURISTIC_NOTES:
        return HEURISTIC_NOTES[key]
    for prefix, note in HEURISTIC_NOTES.items():
        if key == prefix or key.startswith(prefix + "."):
            return note
    return "heuristic (see defaults.yaml); not an empirically fitted cutoff"


def lookup(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
