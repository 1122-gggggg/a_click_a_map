"""Frozen-base policy helpers.  No GlueMap, no site video IDs.

Principles
----------
* Freeze base poses and points.  After any fringe or append step the recorded
  frozen-bin hashes must still match.
* New points are allowed only when a track includes a frozen-base view.
  This helper does not hang geometry off newly admitted history.
* Localize leftovers against the frozen base, never against newly hung
  historical cameras.
* A later timestamp is not an update.  Update requires change evidence
  against frozen geometry, not recency.
* Uncertain metrics fail closed (do not admit; do not invent UPDATE).

Numeric gates below are labeled heuristic engineering defaults, not fitted
site parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Heuristic gates.  Missing or invalid evidence fails closed.
DEFAULT_FRINGE_HEURISTICS: dict[str, float | int] = {
    "min_core_hit_fraction": 0.50,  # heuristic; majority of hits from frozen core
    "min_independent_bridges": 2,  # heuristic fail-closed independent support
    "max_old_region_high_reproj_delta": 0.0,  # heuristic; old cameras must not worsen
}


class FrozenCoreError(ValueError):
    """Raised when a frozen-core policy check fails closed."""


class FrozenBinsChangedError(FrozenCoreError):
    """Raised when frozen pose/point bin hashes differ after an operation."""


def _as_hash_map(values: Mapping[Any, Any], *, name: str) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping of bin id to hash")
    hashed: dict[str, str] = {}
    for key, value in values.items():
        if value is None:
            raise FrozenBinsChangedError(f"{name} hash for {key!r} is missing")
        hashed[str(key)] = str(value)
    return hashed


def assert_frozen_bins_unchanged(
    before_hashes: Mapping[Any, Any],
    after_hashes: Mapping[Any, Any],
) -> None:
    """Require every frozen bin hash to be identical after an operation.

    Extra keys in ``after_hashes`` are ignored (new fringe bins may appear).
    A missing frozen key or a changed hash fails closed.
    """

    before = _as_hash_map(before_hashes, name="before_hashes")
    after = _as_hash_map(after_hashes, name="after_hashes")
    changed: list[str] = []
    for key, digest in before.items():
        if key not in after:
            changed.append(key)
        elif after[key] != digest:
            changed.append(key)
    if changed:
        raise FrozenBinsChangedError(
            "frozen bins changed: " + ", ".join(sorted(changed))
        )


def _lookup_heuristic(
    config: Mapping[str, Any] | None,
    key: str,
    default: float | int,
) -> float:
    if not config:
        return float(default)
    node: Any = config
    if key in node:
        node = node[key]
    elif isinstance(node.get("gates"), Mapping) and key in node["gates"]:
        node = node["gates"][key]
    elif isinstance(node.get("fringe"), Mapping) and key in node["fringe"]:
        node = node["fringe"][key]
    else:
        return float(default)
    try:
        value = float(node)
    except (TypeError, ValueError) as error:
        raise FrozenCoreError(f"heuristic {key} is not numeric") from error
    if not value == value:  # NaN
        raise FrozenCoreError(f"heuristic {key} is not numeric")
    return value


def admit_fringe_only_if(
    core_hit_fraction: float,
    independent_bridges: int,
    old_region_high_reproj_delta: float,
    *,
    config: Mapping[str, Any] | None = None,
) -> bool:
    """Admit local fringe only when frozen-core support stays clean.

    Fail-closed when the core-hit fraction is below the heuristic, fewer than
    two independent geometric bridges exist, or old-region high-reproj rose.
    Does not authorize map fusion or moving frozen poses/points.
    """

    min_core = _lookup_heuristic(
        config, "min_core_hit_fraction", DEFAULT_FRINGE_HEURISTICS["min_core_hit_fraction"]
    )
    min_bridges = int(
        _lookup_heuristic(
            config,
            "min_independent_bridges",
            DEFAULT_FRINGE_HEURISTICS["min_independent_bridges"],
        )
    )
    max_delta = _lookup_heuristic(
        config,
        "max_old_region_high_reproj_delta",
        DEFAULT_FRINGE_HEURISTICS["max_old_region_high_reproj_delta"],
    )
    try:
        fraction = float(core_hit_fraction)
        bridges = int(independent_bridges)
        delta = float(old_region_high_reproj_delta)
    except (TypeError, ValueError):
        return False
    if not (fraction == fraction and delta == delta):
        return False
    if fraction < min_core:
        return False
    if bridges < min_bridges:
        return False
    if delta > max_delta:
        return False
    return True


def classify_update_vs_appearance(
    loc_success: bool,
    geometry_covered: bool,
    change_evidence: bool,
) -> str:
    """Distinguish appearance leftover from a geometry update candidate.

    Later capture time is intentionally unused.  Timestamp never ranks a
    leftover as an update.  Uncertain / incomplete evidence is
    ``QUARANTINE``.
    """

    if loc_success is not True:
        return "QUARANTINE"
    if change_evidence is True:
        return "UPDATE_CANDIDATE"
    if geometry_covered is True and change_evidence is False:
        return "APPEARANCE_REF"
    return "QUARANTINE"


__all__ = [
    "DEFAULT_FRINGE_HEURISTICS",
    "FrozenBinsChangedError",
    "FrozenCoreError",
    "admit_fringe_only_if",
    "assert_frozen_bins_unchanged",
    "classify_update_vs_appearance",
]
