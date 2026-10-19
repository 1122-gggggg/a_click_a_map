"""Site-agnostic leftover-vs-frozen-base intake tree. Fail-closed."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import ROLES, SessionQuality

_CORE_OK = frozenset({"STRONG", "USABLE"})
_REJECT_STATUS = frozenset({"REJECT", "INCONSISTENT"})


def _as_float(value: Any) -> float | None:
    if value is None or value is False:
        return None
    if value is True:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _session_status(session: Any) -> str:
    if isinstance(session, SessionQuality):
        return session.internal_status
    if isinstance(session, Mapping):
        return str(session.get("internal_status") or "")
    return ""


def _session_sharpness_p10(session: Any) -> float | None:
    if isinstance(session, SessionQuality):
        return session.sharpness_p10
    if isinstance(session, Mapping):
        return _as_float(session.get("sharpness_p10"))
    return None


def loc_succeeds(loc_strong: Any, loc_queries: Any) -> bool | None:
    """True/False when both counts are present; None if localization evidence is missing."""

    if loc_strong is None or loc_queries is None:
        return None
    strong = _as_float(loc_strong)
    queries = _as_float(loc_queries)
    if strong is None or queries is None or queries <= 0:
        return None
    return strong > 0.0


def classify_leftover_vs_frozen_base(
    session: SessionQuality | Mapping[str, Any] | str,
    *,
    loc_strong: Any,
    loc_queries: Any,
    core_hit_fraction: Any,
    independent_bridges: Any,
    usable_edge: Any,
    change_score: Any,
    high_influence: Any,
    geometry_already_covered: Any,
) -> str:
    """Walk the leftover intake tree. Uncertain → QUARANTINE; no reliable edge → NEW_SUBMAP.

    Never force-merges on one critical or AMBIGUOUS bridge. UPDATE_CANDIDATE only
    when ``change_score > 0``. VPR is not a geometric edge: callers must pass
    ``usable_edge`` only for verified geometry.
    """

    status = _session_status(session)
    sharpness = _session_sharpness_p10(session)
    if status in _REJECT_STATUS:
        return "REJECT"
    if status == "WEAK" and sharpness is not None and sharpness <= 0:
        return "REJECT"

    good_internal = status in _CORE_OK
    loc_ok = loc_succeeds(loc_strong, loc_queries)
    bridges = _as_int(independent_bridges)
    change = _as_float(change_score)
    if change is None:
        change = 0.0
    has_usable = bool(usable_edge)
    covered = bool(geometry_already_covered)
    influential = bool(high_influence)

    no_reliable_edge = (not has_usable) or bridges == 0
    single_critical = has_usable and bridges == 1
    uncertain_edge = has_usable and bridges is None

    # Good internal geometry with no reliable geometric edge is a new component,
    # unless localization claims a connection the geometry cannot support.
    if good_internal and no_reliable_edge:
        if loc_ok is True:
            return "QUARANTINE"
        return "NEW_SUBMAP"

    if influential or single_critical or uncertain_edge:
        return "QUARANTINE"
    if loc_ok is True and not has_usable:
        return "QUARANTINE"

    if core_hit_fraction is not None:
        frac = _as_float(core_hit_fraction)
        if frac is None:
            return "QUARANTINE"
        if loc_ok is True and frac <= 0.0:
            return "QUARANTINE"

    if change > 0.0 and good_internal:
        return "UPDATE_CANDIDATE"

    if has_usable and not covered and good_internal:
        return "GEOMETRY_REINFORCEMENT"

    if loc_ok is True and covered and change <= 0.0 and good_internal:
        return "APPEARANCE_REF"

    if status and status not in _CORE_OK and status != "WEAK":
        return "QUARANTINE"
    return "QUARANTINE"


def role_or_quarantine(role: str) -> str:
    return role if role in ROLES else "QUARANTINE"
