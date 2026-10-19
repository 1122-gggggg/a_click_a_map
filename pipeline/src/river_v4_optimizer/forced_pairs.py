from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


_ADMISSION_RANK = {
    "CANDIDATE": 0,
    "REJECTED": 1,
    "AMBIGUOUS": 2,
    "VERIFIED": 3,
}


def _canonical(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise ValueError(f"self pair is invalid: {left}")
    return (left, right) if left < right else (right, left)


def build_forced_cross_video_pairs(
    selected_keyframes: Sequence[str],
    *,
    left_video: str,
    right_video: str,
    category: str,
) -> list[dict[str, Any]]:
    """Build the exact cross product for a retrieval-blind video pair."""

    if not left_video or not right_video or left_video == right_video:
        raise ValueError("forced pair videos must be distinct and non-empty")
    left = sorted(
        str(value) for value in selected_keyframes if str(value).split(":", 1)[0] == left_video
    )
    right = sorted(
        str(value) for value in selected_keyframes if str(value).split(":", 1)[0] == right_video
    )
    if not left:
        raise ValueError(f"left video is absent from the selection: {left_video}")
    if not right:
        raise ValueError(f"right video is absent from the selection: {right_video}")
    rows = []
    for first in left:
        for second in right:
            image_i, image_j = _canonical(first, second)
            rows.append(
                {
                    "admission": "CANDIDATE",
                    "category": category,
                    "image_i": image_i,
                    "image_j": image_j,
                    "is_geometry_edge": False,
                    "score": 0.0,
                }
            )
    return sorted(rows, key=lambda row: (row["image_i"], row["image_j"]))


def _quality_key(row: Mapping[str, Any]) -> tuple[int, int, float, str]:
    admission = str(row.get("admission") or "CANDIDATE")
    return (
        _ADMISSION_RANK.get(admission, -1),
        int(row.get("inliers_E") or 0),
        float(row.get("spatial_coverage") or row.get("score") or 0.0),
        str(sorted(row.items())),
    )


def merge_pair_geometry(
    base: Iterable[Mapping[str, Any]], extra: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Merge pair geometry deterministically, preferring stronger evidence."""

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in (*list(base), *list(extra)):
        row = dict(raw)
        image_i, image_j = _canonical(str(row["image_i"]), str(row["image_j"]))
        row["image_i"], row["image_j"] = image_i, image_j
        key = (image_i, image_j)
        existing = merged.get(key)
        if existing is None or _quality_key(row) > _quality_key(existing):
            merged[key] = row
    return [merged[key] for key in sorted(merged)]


__all__ = ["build_forced_cross_video_pairs", "merge_pair_geometry"]
