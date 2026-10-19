"""Sequence-aware reference exclusion for leave-one-out evaluation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

_TRAILING_NUMBER = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class ReferenceFrame:
    """The route identity and numeric position encoded by a reference name."""

    name: str
    sequence: str
    frame_number: int | None


def parse_reference_name(name: str) -> ReferenceFrame:
    """Parse names such as ``P1180118/000100.jpg`` without assuming one extension."""

    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    match = _TRAILING_NUMBER.search(path.stem)
    frame_number = None if match is None else int(match.group(1))
    sequence = "" if str(path.parent) == "." else str(path.parent)
    return ReferenceFrame(name=name, sequence=sequence, frame_number=frame_number)


def loo_exclusion_indices(
    ref_names: Sequence[str], query_index: int, near_k: int
) -> tuple[int, ...]:
    """Return self plus same-sequence frames within ``±near_k``.

    A missing numeric suffix is deliberately conservative: only the exact query row is
    excluded because lexical adjacency is not evidence of temporal adjacency.
    """

    if not 0 <= query_index < len(ref_names):
        raise IndexError("query_index is outside ref_names")
    if near_k < 0:
        raise ValueError("near_k must be non-negative")

    query = parse_reference_name(ref_names[query_index])
    excluded = {query_index}
    if query.frame_number is None:
        return (query_index,)

    for index, name in enumerate(ref_names):
        candidate = parse_reference_name(name)
        if candidate.sequence != query.sequence or candidate.frame_number is None:
            continue
        if abs(candidate.frame_number - query.frame_number) <= near_k:
            excluded.add(index)
    return tuple(sorted(excluded))
