"""Deterministic Stage 3 retrieval candidates.

Retrieval is deliberately evidence only: these records are never admitted as
SfM geometry edges until a later two-view verification stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .domain import KeyframeRecord


class RetrievalCategory(str, Enum):
    TEMPORAL = "temporal"
    SAME_SESSION = "same_session"
    CROSS_VIDEO = "cross_video"
    CROSS_SESSION = "cross_session"
    LOOP = "loop"


@dataclass(frozen=True)
class DescriptorArtifact:
    keyframe_ids: tuple[str, ...]
    matrix: np.ndarray
    model: str
    checkpoint: str
    sha256: str

    @classmethod
    def from_matrix(
        cls,
        matrix: np.ndarray,
        keyframe_ids: Sequence[str],
        model: str,
        checkpoint: str,
        sha256: str,
    ) -> "DescriptorArtifact":
        values = np.asarray(matrix, dtype=np.float32)
        ids = tuple(keyframe_ids)
        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("descriptor rows must match keyframe_ids")
        return cls(ids, values, model, checkpoint, sha256)


@dataclass(frozen=True)
class RetrievalCandidate:
    image_i: str
    image_j: str
    category: RetrievalCategory
    score: float
    admission: str = "CANDIDATE"
    is_geometry_edge: bool = False

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self.category.value, -self.score, self.image_i, self.image_j)


@dataclass(frozen=True)
class RetrievalResult:
    candidates: tuple[RetrievalCandidate, ...]
    excluded_holdouts: tuple[str, ...] = ()


def cache_identity(
    keyframes: Sequence[tuple[str, str]], model: str, checkpoint: str, config: Mapping[str, Any]
) -> str:
    payload = {
        "keyframes": list(keyframes),
        "model": model,
        "checkpoint": checkpoint,
        "config": config,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def generate_candidates(
    descriptors: np.ndarray,
    keyframes: Sequence[KeyframeRecord],
    *,
    temporal_radius: int = 3,
    top_k: int = 20,
    category_caps: Mapping[RetrievalCategory, int] | None = None,
    loop_min_time_delta: float = 10.0,
    holdout_ids: set[str] | frozenset[str] = frozenset(),
) -> RetrievalResult:
    matrix = np.asarray(descriptors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(keyframes):
        raise ValueError("descriptor rows must match keyframes")
    if top_k < 0 or temporal_radius < 0:
        raise ValueError("top_k and temporal_radius must be non-negative")
    caps = {category: top_k for category in RetrievalCategory}
    if category_caps:
        caps.update(
            {
                RetrievalCategory(k) if not isinstance(k, RetrievalCategory) else k: int(v)
                for k, v in category_caps.items()
            }
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    scores = normalized @ normalized.T
    excluded = tuple(sorted(k.keyframe_id for k in keyframes if k.keyframe_id in holdout_ids))
    selected: dict[tuple[str, str, RetrievalCategory], RetrievalCandidate] = {}
    for i, left in enumerate(keyframes):
        if left.keyframe_id in holdout_ids:
            continue
        per_category: dict[RetrievalCategory, list[RetrievalCandidate]] = {
            category: [] for category in RetrievalCategory
        }
        for j, right in enumerate(keyframes):
            if i == j or right.keyframe_id in holdout_ids:
                continue
            same_video = left.video_id == right.video_id
            same_session = _session(left) == _session(right)
            temporal = (
                same_video
                and abs(left.source_frame_index - right.source_frame_index) <= temporal_radius
            )
            long_loop = (
                same_video
                and abs(left.source_pts_seconds - right.source_pts_seconds) >= loop_min_time_delta
            ) or not same_video
            categories = []
            if temporal:
                categories.append(RetrievalCategory.TEMPORAL)
            if same_session and not temporal:
                categories.append(RetrievalCategory.SAME_SESSION)
            if not same_video:
                categories.append(RetrievalCategory.CROSS_VIDEO)
            if not same_session:
                categories.append(RetrievalCategory.CROSS_SESSION)
            if long_loop:
                categories.append(RetrievalCategory.LOOP)
            image_i, image_j = sorted((left.keyframe_id, right.keyframe_id))
            for category in categories:
                per_category[category].append(
                    RetrievalCandidate(image_i, image_j, category, float(scores[i, j]))
                )
        for category, options in per_category.items():
            options.sort(
                key=lambda candidate: (-candidate.score, candidate.image_i, candidate.image_j)
            )
            for candidate in options[: caps[category]]:
                key = (candidate.image_i, candidate.image_j, category)
                current = selected.get(key)
                if current is None or candidate.score > current.score:
                    selected[key] = candidate
    return RetrievalResult(
        tuple(sorted(selected.values(), key=lambda candidate: candidate.sort_key)), excluded
    )


def load_salad_descriptors(path: str | Path) -> np.ndarray:
    """Load a GLUEMAP SALAD ``.pt`` artifact without making torch mandatory."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("loading SALAD .pt descriptors requires torch") from exc
    value = torch.load(  # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
        Path(path), map_location="cpu", weights_only=True
    )
    if isinstance(value, Mapping):
        for key in ("descriptors", "global_descriptors", "embeddings", "features"):
            if key in value:
                value = value[key]
                break
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _session(frame: KeyframeRecord) -> str | None:
    return frame.metadata.get("session_id") or frame.segment_id
