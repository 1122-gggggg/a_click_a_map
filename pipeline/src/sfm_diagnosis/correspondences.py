from __future__ import annotations

import csv
import json
from pathlib import Path


def load_correspondence_events(path: str | Path) -> list[dict]:
    """Load query-landmark correspondence events from CSV/JSON/JSONL."""
    rows = _read_rows(path)
    events: list[dict] = []
    for row in rows:
        point_id = _int_or_none(_first(row, "point_id", "id", "landmark_id"))
        if point_id is None:
            continue
        observed = _bool_or_none(_first(row, "observed", "visible"))
        inlier = _bool_or_none(_first(row, "inlier", "is_inlier"))
        if observed is None and inlier is None:
            continue
        if inlier is True and observed is None:
            observed = True
        if observed is None:
            observed = False
        if inlier is None:
            inlier = False
        events.append(
            {
                "query_id": _first(row, "query_id", "query", "name"),
                "point_id": point_id,
                "observed": bool(observed),
                "inlier": bool(inlier) and bool(observed),
                "timestamp": _float_or_none(_first(row, "timestamp", "time")),
                "reference_id": _first(row, "reference_id", "ref_id"),
                "residual_px": _float_or_none(_first(row, "residual_px", "reproj_px")),
            }
        )
    return events


def _read_rows(path: str | Path) -> list[dict]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        return [
            dict(json.loads(line))
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("rows", "events", "correspondences"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise TypeError(f"Unsupported JSON table shape in {p}")
    raise ValueError(f"Unsupported correspondence table format: {p.suffix}")


def _first(row: dict, *keys: str):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None
