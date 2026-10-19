"""Fail-closed edge and fusion admission. Site IDs never enter this module."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any


GEOMETRY_METRIC_NAMES = (
    "rotation_consensus_deg",
    "translation_direction_consensus_deg",
    "scale_consensus",
    "parallax_deg",
    "edge_positive_depth_ratio",
    "spatial_coverage",
    "cross_session_reprojection_error",
    "holdout_inlier_ratio",
    "holdout_residual",
)

_GROUP_AXES = ("time", "image", "landmark", "region")

_SHARED_SCOPES = frozenset({"shared_map", "vpr", "raw_tracks", "unknown", ""})


@dataclass(frozen=True)
class BridgeGroupSplit:
    fit_group_ids: tuple[str, ...]
    holdout_group_ids: tuple[str, ...]
    independent_group_ids: tuple[str, ...]
    independent_group_count: int
    status: str
    reasons: tuple[str, ...]
    fit_holdout_group_disjoint: bool


def finite_metric(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def geometry_metrics_complete(
    values: Mapping[str, Any] | None = None,
    *legacy: Any,
    **named: Any,
) -> bool:
    """True only when every required geometry metric is finite."""

    payload = dict(values or {})
    if legacy and len(legacy) == len(GEOMETRY_METRIC_NAMES):
        payload = {name: item for name, item in zip(GEOMETRY_METRIC_NAMES, legacy)}
    payload.update(named)
    return all(finite_metric(payload.get(name)) for name in GEOMETRY_METRIC_NAMES)


def _as_id_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        text = str(value).strip()
        return (text,) if text else ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _axis_values(record: Mapping[str, Any], axis: str) -> frozenset[str]:
    aliases = {
        "time": (
            "time",
            "times",
            "timestamp",
            "timestamps",
            "query_frame_indices",
            "reference_frame_indices",
        ),
        "image": ("image", "images", "query_image_ids", "reference_image_ids"),
        "landmark": ("landmark", "landmarks", "landmark_ids"),
        "region": ("region", "regions", "region_ids"),
    }
    values: set[str] = set()
    for key in aliases[axis]:
        item = record.get(key)
        if item is None:
            continue
        if isinstance(item, (str, int, float)):
            values.add(str(item))
        elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            values.update(str(part) for part in item)
    return frozenset(values)


def _normalise_groups(
    groups: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        return {
            str(row.get("group_id") or f"group-{index}"): row
            for index, row in enumerate(groups)
            if isinstance(row, Mapping)
        }
    if isinstance(groups, Mapping):
        return {str(key): value for key, value in groups.items() if isinstance(value, Mapping)}
    return {}


def _numeric_values(record: Mapping[str, Any], key: str) -> tuple[float, ...]:
    raw = record.get(key)
    if raw is None:
        return ()
    values = raw if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes)) else (raw,)
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            parsed.append(number)
    return tuple(parsed)


def _maximum_independent_group_ids(
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    query_frame_separation: int | None = None,
    reference_frame_separation: int | None = None,
) -> tuple[str, ...]:
    names = sorted(metadata)
    if not names:
        return ()
    axes = {
        name: {axis: _axis_values(metadata[name], axis) for axis in _GROUP_AXES}
        for name in names
    }
    query_frames = {name: _numeric_values(metadata[name], "query_frame_indices") for name in names}
    reference_frames = {
        name: _numeric_values(metadata[name], "reference_frame_indices") for name in names
    }

    def separated(
        left: tuple[float, ...], right: tuple[float, ...], minimum: int | None
    ) -> bool:
        if minimum is None:
            return True
        if not left or not right:
            return False
        return min(abs(a - b) for a in left for b in right) >= float(minimum)

    def independent(left: str, right: str) -> bool:
        diverse = all(
            axes[left][axis]
            and axes[right][axis]
            and axes[left][axis].isdisjoint(axes[right][axis])
            for axis in _GROUP_AXES
        )
        return (
            diverse
            and separated(query_frames[left], query_frames[right], query_frame_separation)
            and separated(
                reference_frames[left],
                reference_frames[right],
                reference_frame_separation,
            )
        )

    if len(names) <= 12:
        for size in range(len(names), 0, -1):
            for subset in combinations(names, size):
                if all(independent(a, b) for a, b in combinations(subset, 2)):
                    return tuple(subset)
        return ()
    chosen: list[str] = []
    for name in names:
        if all(independent(name, other) for other in chosen):
            chosen.append(name)
    return tuple(chosen)


def assess_declared_bridge_groups(
    groups: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    fit_evidence_ids: Iterable[str] = (),
    holdout_evidence_ids: Iterable[str] = (),
    query_frame_separation: int = 30,
    reference_frame_separation: int = 30,
) -> BridgeGroupSplit:
    """Recompute independence. A claimed group count is never trusted alone."""

    mapped = _normalise_groups(groups)
    reasons: list[str] = []
    if not mapped:
        reasons.append("missing_bridge_group_metadata")
    independent_ids = _maximum_independent_group_ids(
        mapped,
        query_frame_separation=max(0, int(query_frame_separation)),
        reference_frame_separation=max(0, int(reference_frame_separation)),
    )
    if any(
        not _axis_values(raw, "time")
        or not raw.get("query_frame_indices")
        or not raw.get("reference_frame_indices")
        for raw in mapped.values()
    ):
        reasons.append("missing_numeric_bridge_frame_metadata")
    if mapped and len(independent_ids) < 2:
        reasons.append("fewer_than_two_independent_bridge_groups")
    fit_ids = {str(item) for item in fit_evidence_ids}
    holdout_ids = {str(item) for item in holdout_evidence_ids}
    fit_groups: set[str] = set()
    holdout_groups: set[str] = set()
    assigned: set[str] = set()
    for group_id, raw in mapped.items():
        evidence_ids = {str(item) for item in raw.get("evidence_ids") or ()}
        assigned.update(evidence_ids)
        if evidence_ids & fit_ids:
            fit_groups.add(group_id)
        if evidence_ids & holdout_ids:
            holdout_groups.add(group_id)
    disjoint = bool(fit_groups and holdout_groups and fit_groups.isdisjoint(holdout_groups))
    if (fit_ids | holdout_ids) - assigned:
        reasons.append("unassigned_fit_or_holdout_evidence")
        disjoint = False
    if fit_groups & holdout_groups:
        reasons.append("fit_holdout_share_bridge_group")
    if not disjoint:
        reasons.append("missing_group_disjoint_fit_holdout")
    status = "STRONG" if len(independent_ids) >= 2 and disjoint else "AMBIGUOUS"
    return BridgeGroupSplit(
        fit_group_ids=tuple(sorted(fit_groups)),
        holdout_group_ids=tuple(sorted(holdout_groups)),
        independent_group_ids=independent_ids,
        independent_group_count=len(independent_ids),
        status=status,
        reasons=tuple(dict.fromkeys(reasons)),
        fit_holdout_group_disjoint=disjoint,
    )


def usable_geometry_ready(
    *,
    independent_artifact: bool,
    evidence_scope: str,
    independent_groups: int,
    group_holdout_disjoint: bool,
    geometry_complete: bool,
    fit_evidence_ids: Iterable[str] = (),
    holdout_evidence_ids: Iterable[str] = (),
    min_groups: int = 1,
) -> bool:
    """STRONG/USABLE require exact-pair independent, disjoint, complete geometry."""

    fit_ids = {str(item) for item in fit_evidence_ids}
    holdout_ids = {str(item) for item in holdout_evidence_ids}
    return bool(
        independent_artifact
        and str(evidence_scope) == "exact_pair"
        and int(independent_groups) >= int(min_groups)
        and group_holdout_disjoint
        and geometry_complete
        and fit_ids
        and holdout_ids
        and fit_ids.isdisjoint(holdout_ids)
    )


def classify_fusion_authorization(
    *,
    role: str,
    has_holdout: bool = False,
    independent_bridge_groups: int = 0,
    geometry_complete: bool = False,
    group_holdout_disjoint: bool = False,
    loo_passed: bool = False,
    base_admitted: bool = False,
) -> str:
    """A role is not map-changing authority. Reinforcement stays local unless complete."""

    if role in {"BASE_CORE", "BASE_SUPPORT"} and base_admitted:
        return "GLOBAL_BA"
    if role == "BASE_CORE":
        return "GLOBAL_BA_PENDING_APPROVAL"
    if role == "BASE_SUPPORT":
        return "LOCAL_RELATION_ONLY"
    if role == "GEOMETRY_REINFORCEMENT":
        if (
            has_holdout
            and independent_bridge_groups >= 2
            and geometry_complete
            and group_holdout_disjoint
        ):
            return "LOCAL_FUSION"
        return "LOCAL_RELATION_ONLY"
    if role == "UPDATE_CANDIDATE":
        return "LOCAL_FUSION" if loo_passed else "LOCAL_FUSION_PENDING_LOO"
    if role == "APPEARANCE_REF":
        return "LOCALIZATION_ONLY"
    if role == "NEW_SUBMAP":
        return "SUBMAP_ONLY"
    if role == "VALIDATION_ONLY":
        return "EVALUATION_ONLY"
    return "NONE"


def normalize_evidence_scope(
    value: Any,
    *,
    independent_artifact: bool = False,
    shared_map: bool = False,
    source: Any = None,
) -> str:
    scope = str(value or "").strip().lower() or "unknown"
    aliases = {
        "exact-pair": "exact_pair",
        "pair": "exact_pair",
        "independent": "exact_pair",
        "shared": "shared_map",
        "map": "shared_map",
        "reconstruction": "shared_map",
        "retrieval": "vpr",
        "raw": "raw_tracks",
        "tracks": "raw_tracks",
    }
    scope = aliases.get(scope, scope)
    source_text = str(source or "").strip().lower()
    if shared_map or source_text in {"shared_map", "vpr", "map", "reconstruction"}:
        if scope == "exact_pair" and not independent_artifact:
            return "shared_map" if source_text != "vpr" else "vpr"
        if scope in _SHARED_SCOPES:
            return "vpr" if source_text == "vpr" or scope == "vpr" else "shared_map"
    if scope == "exact_pair" and not independent_artifact:
        return "unknown"
    return scope


__all__ = [
    "GEOMETRY_METRIC_NAMES",
    "BridgeGroupSplit",
    "assess_declared_bridge_groups",
    "classify_fusion_authorization",
    "finite_metric",
    "geometry_metrics_complete",
    "normalize_evidence_scope",
    "usable_geometry_ready",
]
