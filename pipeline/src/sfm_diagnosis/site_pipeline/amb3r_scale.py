"""Estimate metres per map unit from the metric baselines of every AMB3R star.

GlueMap pins star 0's scale to 1 and never rescales afterwards, so the final map
unit is whatever one star's AMB3R metres happened to be. Each star's metric
baseline ``||t_j||`` (centre camera is the identity gauge) is compared with the
same camera pair's distance in the final map; the robust aggregate over all stars
is the estimate. This is a learned-prior estimate with no external length truth.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .metric_anchors import weighted_median

SCHEMA = "amb3r-star-scale/v1"
MIN_EDGES_PER_STAR = 3
MIN_STARS = 30
MAX_SESSION_SCALE_RATIO = 1.20
STATUS_OK = "ESTIMATED_UNVALIDATED"
STATUS_DISAGREEMENT = "SCALE_DISAGREEMENT_UNVALIDATED"


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {"p10": float(np.quantile(values, 0.1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9))}


def estimate_star_scale(
    indexes: Sequence[Sequence[int]],
    translations: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    centers: Mapping[int, np.ndarray],
    sessions: Mapping[int, str],
) -> dict[str, Any]:
    """Aggregate per-star metric/map baseline ratios into ``meters_per_map_unit``.

    ``translations[k]`` is ``(N, 3)`` (star k, centre-relative w2c translation in
    metres), ``scores[k]`` is ``(N,)``, ``centers[i]`` the final-model camera centre
    of image index ``i``, ``sessions[i]`` its session id.
    """
    if not len(indexes) == len(translations) == len(scores):
        raise ValueError("star arrays have different lengths")
    star_scale: dict[int, float] = {}
    star_session: dict[int, str] = {}
    num_edges = 0
    for k, members in enumerate(indexes):
        translation = np.asarray(translations[k], dtype=np.float64)
        score = np.asarray(scores[k], dtype=np.float64)
        if translation.shape != (len(members), 3) or score.shape != (len(members),):
            raise ValueError(f"star {k} array shape does not match its members")
        centre = int(members[0])
        if centre not in centers or not np.isfinite(centers[centre]).all():
            continue
        logs, weights = [], []
        for j in range(1, len(members)):
            other = int(members[j])
            if not score[j] > 0 or other not in centers or not np.isfinite(centers[other]).all():
                continue
            metric = float(np.linalg.norm(translation[j]))
            model = float(np.linalg.norm(np.asarray(centers[other]) - np.asarray(centers[centre])))
            if not (np.isfinite(metric) and metric > 0 and model > 0):
                continue
            logs.append(math.log(metric / model))
            weights.append(float(score[j]))
        if len(logs) < MIN_EDGES_PER_STAR:
            continue
        star_scale[k] = math.exp(weighted_median(np.asarray(logs), np.asarray(weights)))
        star_session[k] = str(sessions[centre])
        num_edges += len(logs)
    if len(star_scale) < MIN_STARS:
        raise ValueError("insufficient AMB3R star scale support")

    values = np.asarray(list(star_scale.values()))
    scale = float(np.exp(np.median(np.log(values))))
    by_session: dict[str, list[float]] = {}
    for k, value in star_scale.items():
        by_session.setdefault(star_session[k], []).append(value)
    session_scale = {name: float(np.median(v)) for name, v in sorted(by_session.items())}
    ratio = max(session_scale.values()) / min(session_scale.values())
    parity = {}
    for label, remainder in (("even", 0), ("odd", 1)):
        group = [v for k, v in star_scale.items() if int(indexes[k][0]) % 2 == remainder]
        parity[label] = float(np.median(group)) if group else None
    parity_diff = (None if parity["even"] is None or parity["odd"] is None
                   else abs(parity["even"] / parity["odd"] - 1.0) * 100.0)
    return {
        "meters_per_map_unit": scale,
        "status": STATUS_DISAGREEMENT if ratio > MAX_SESSION_SCALE_RATIO else STATUS_OK,
        "absolute_accuracy": "NO_EXTERNAL_LENGTH_HOLDOUT",
        "diagnostics": {
            "num_stars": len(indexes),
            "num_stars_used": len(star_scale),
            "num_edges": num_edges,
            "star_scale_quantiles": _quantiles(values),
            "session_scale": session_scale,
            "session_scale_max_over_min": ratio,
            "max_session_scale_ratio": MAX_SESSION_SCALE_RATIO,
            "parity_median_scale": parity,
            "parity_difference_percent": parity_diff,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_amb3r_scale(*, star_cache: Path, map_model: Path, keyframes: Path, output: Path,
                    moge_sidecar: Path | None = None) -> dict[str, Any]:
    import pycolmap
    import torch

    from .metric_scale import _mapping_keyframe_entries, load_metric_scale, model_file_hashes

    entries, keyframes_hash = _mapping_keyframe_entries(keyframes)
    names = sorted(entries)
    sessions_by_index = {i: str(entries[name].get("session_id")) for i, name in enumerate(names)}
    if any(value == "None" for value in sessions_by_index.values()):
        raise ValueError("keyframes.jsonl rows need session_id")
    cache = torch.load(star_cache, map_location="cpu", weights_only=False)
    indexes = [[int(i) for i in members] for members in cache["indexes"]]
    translations = [extrinsic[0, :, :3, 3].double().numpy() for extrinsic in cache["extrinsics"]]
    scores = [score[0].double().numpy() for score in cache["pose_scores"]]

    rec = pycolmap.Reconstruction(str(map_model))
    used = sorted({i for members in indexes for i in members})
    centers: dict[int, np.ndarray] = {}
    for index in used:
        if index >= len(names):
            raise ValueError(f"star cache index {index} is outside the keyframe manifest")
        if index + 1 not in rec.images:
            continue  # unregistered in the final model: edge is skipped
        image = rec.images[index + 1]
        if image.name != names[index]:
            raise ValueError(f"image id {index + 1} is {image.name}, expected {names[index]}")
        centers[index] = np.asarray(image.projection_center(), dtype=np.float64)

    result = estimate_star_scale(indexes, translations, scores, centers, sessions_by_index)
    result.update({
        "schema": SCHEMA,
        "star_cache": str(star_cache), "star_cache_sha256": _sha256(star_cache),
        "map_model": str(map_model), "map_model_hashes": model_file_hashes(map_model),
        "keyframes": str(keyframes), "keyframes_sha256": keyframes_hash,
        "warning": "Estimated metres from the AMB3R learned scale prior; no external length truth.",
    })
    if moge_sidecar is not None:
        moge = load_metric_scale(moge_sidecar, map_model)
        moge_scale = float(moge["meters_per_map_unit"])
        ratio = moge_scale / result["meters_per_map_unit"]
        result["moge_cross_check"] = {
            "sidecar": str(moge_sidecar), "sidecar_sha256": _sha256(moge_sidecar),
            "meters_per_map_unit": moge_scale, "moge_over_amb3r": ratio,
            "note": "Both estimators are learned priors; shared bias is undetectable.",
        }
        if ratio > MAX_SESSION_SCALE_RATIO or ratio < 1.0 / MAX_SESSION_SCALE_RATIO:
            result["status"] = STATUS_DISAGREEMENT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
