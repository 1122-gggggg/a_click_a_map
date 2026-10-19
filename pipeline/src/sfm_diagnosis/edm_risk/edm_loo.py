from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol


class EDMLOOMode(str, Enum):
    STRICT = "strict"
    REFERENCE_EXCLUSION = "reference-exclusion"


@dataclass(frozen=True)
class EDMQuery:
    query_id: str
    session_id: str
    timestamp: float
    image_path: str | None = None
    pose_provenance: str | None = None


@dataclass(frozen=True)
class EDMReferenceIndex:
    index_id: str
    reference_sessions: frozenset[str]
    session_only_landmarks_removed: bool
    appearance_descriptors_rebuilt: bool


@dataclass(frozen=True)
class EDMQueryResult:
    query_id: str
    session_id: str
    timestamp: float
    success: bool
    registration_success: bool
    raw_matches: int | None = None
    valid_2d3d: int | None = None
    ransac_inliers: int | None = None
    inlier_ratio: float | None = None
    reprojection_mean: float | None = None
    reprojection_median: float | None = None
    reprojection_p90: float | None = None
    positive_depth_ratio: float | None = None
    pose_translation_error: float | None = None
    pose_rotation_error_deg: float | None = None
    pose_consistency: float | None = None
    temporal_consistency: float | None = None
    runtime_edm: float | None = None
    runtime_pnp: float | None = None
    runtime_total: float | None = None
    reference_ids: tuple[str, ...] = ()
    point_ids: tuple[int, ...] = ()
    estimated_position: tuple[float, float, float] | None = None
    estimated_R_wc: tuple[tuple[float, float, float], ...] | None = None
    estimated_yaw_deg: float | None = None
    estimated_pitch_deg: float | None = None
    ground_truth_source: str | None = None
    loo_mode: str | None = None
    occupancy_4x4: int | None = None
    convex_hull_coverage: float | None = None
    occupied_frac_30: float | None = None
    viewpoint_pool_fallback: bool = False
    query_status: str | None = None
    in_intersection: bool = False
    viewpoint_abstain: bool = False
    n_track_2d3d: int | None = None
    n_depth_2d3d: int | None = None
    track_inliers: int | None = None
    inlier_query_xy: tuple[tuple[float, float], ...] | None = None
    inlier_confidence_mean: float | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["reference_ids"] = list(self.reference_ids)
        payload["point_ids"] = list(self.point_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "EDMQueryResult":
        data = dict(payload)
        data["reference_ids"] = tuple(data.get("reference_ids") or ())
        data["point_ids"] = tuple(int(value) for value in data.get("point_ids") or ())
        position = data.get("estimated_position")
        data["estimated_position"] = None if position is None else tuple(position)
        rotation = data.get("estimated_R_wc")
        data["estimated_R_wc"] = (
            None if rotation is None else tuple(tuple(row) for row in rotation)
        )
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in data if key in allowed})


class EDMProvider(Protocol):
    """Deployment adapter loaded once and reused for all LOO queries."""

    fingerprint: str

    def build_reference_index(
        self, *, excluded_sessions: frozenset[str], strict: bool
    ) -> EDMReferenceIndex: ...

    def localize(
        self, query: EDMQuery, index: EDMReferenceIndex
    ) -> EDMQueryResult: ...


@dataclass(frozen=True)
class EDMLOOResult:
    mode: EDMLOOMode
    strict_loo: bool
    pseudo_loo: bool
    results: tuple[EDMQueryResult, ...]
    cache_hits: int
    cache_misses: int
    temporal_by_session: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "artifact_type": "EDM_LEAVE_ONE_SESSION_OUT",
            "loo_mode": self.mode.value,
            "strict_loo": self.strict_loo,
            "pseudo_loo": self.pseudo_loo,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "results": [result.to_dict() for result in self.results],
            "temporal_by_session": self.temporal_by_session,
        }


class EDMLOORunner:
    def __init__(self, *, provider: EDMProvider, cache_dir: str | Path):
        self.provider = provider
        self.cache_dir = Path(cache_dir)

    def run(
        self,
        queries_by_session: dict[str, list[EDMQuery]],
        *,
        mode: EDMLOOMode,
    ) -> EDMLOOResult:
        if not queries_by_session:
            raise ValueError("EDM LOO requires at least one session")
        strict = mode is EDMLOOMode.STRICT
        results: list[EDMQueryResult] = []
        hits = 0
        misses = 0
        temporal: dict[str, dict] = {}
        for session_id in sorted(queries_by_session):
            queries = sorted(
                queries_by_session[session_id], key=lambda query: query.timestamp
            )
            if any(query.session_id != session_id for query in queries):
                raise ValueError("query session key and EDMQuery.session_id disagree")
            index = self.provider.build_reference_index(
                excluded_sessions=frozenset({session_id}), strict=strict
            )
            _validate_reference_index(index, heldout_session=session_id, strict=strict)
            session_results: list[EDMQueryResult] = []
            for query in queries:
                cached = self._load_cached(query, index, mode)
                if cached is None:
                    result = self.provider.localize(query, index)
                    _validate_result(result, query=query, mode=mode)
                    self._store_cached(result, query, index, mode)
                    misses += 1
                else:
                    result = cached
                    hits += 1
                _validate_reference_ids(
                    result.reference_ids, heldout_session=session_id
                )
                session_results.append(result)
                results.append(result)
            from .temporal import summarize_temporal

            temporal[session_id] = summarize_temporal(session_results).to_dict()
        return EDMLOOResult(
            mode=mode,
            strict_loo=strict,
            pseudo_loo=not strict,
            results=tuple(results),
            cache_hits=hits,
            cache_misses=misses,
            temporal_by_session=temporal,
        )

    def _cache_path(
        self, query: EDMQuery, index: EDMReferenceIndex, mode: EDMLOOMode
    ) -> Path:
        identity = json.dumps(
            {
                "provider": self.provider.fingerprint,
                "index": index.index_id,
                "mode": mode.value,
                "query": asdict(query),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(identity).hexdigest()
        return self.cache_dir / mode.value / query.session_id / f"{digest}.json"

    def _load_cached(
        self, query: EDMQuery, index: EDMReferenceIndex, mode: EDMLOOMode
    ) -> EDMQueryResult | None:
        path = self._cache_path(query, index, mode)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("provider_fingerprint") != self.provider.fingerprint:
            return None
        result = EDMQueryResult.from_dict(payload["result"])
        _validate_result(result, query=query, mode=mode)
        return result

    def _store_cached(
        self,
        result: EDMQueryResult,
        query: EDMQuery,
        index: EDMReferenceIndex,
        mode: EDMLOOMode,
    ) -> None:
        path = self._cache_path(query, index, mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider_fingerprint": self.provider.fingerprint,
            "reference_index": index.index_id,
            "loo_mode": mode.value,
            "result": {**result.to_dict(), "loo_mode": mode.value},
        }
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)


def _validate_reference_index(
    index: EDMReferenceIndex, *, heldout_session: str, strict: bool
) -> None:
    if heldout_session in index.reference_sessions:
        raise RuntimeError(
            f"reference index leaks held-out session {heldout_session}"
        )
    if strict and not (
        index.session_only_landmarks_removed and index.appearance_descriptors_rebuilt
    ):
        raise RuntimeError(
            "strict LOO requires session-only landmarks removed and descriptors rebuilt"
        )


def _validate_reference_ids(
    reference_ids: tuple[str, ...], *, heldout_session: str
) -> None:
    leaked = [
        reference
        for reference in reference_ids
        if str(reference).split("/", 1)[0] == heldout_session
    ]
    if leaked:
        raise RuntimeError(
            f"localization result leaks held-out session references: {leaked!r}"
        )


def _validate_result(
    result: EDMQueryResult, *, query: EDMQuery, mode: EDMLOOMode
) -> None:
    if result.query_id != query.query_id or result.session_id != query.session_id:
        raise RuntimeError("EDM provider returned a result for another query")
    if abs(float(result.timestamp) - float(query.timestamp)) > 1e-9:
        raise RuntimeError("EDM provider changed the query timestamp")
    if result.loo_mode not in (None, mode.value):
        raise RuntimeError("cached EDM result has the wrong LOO mode")
