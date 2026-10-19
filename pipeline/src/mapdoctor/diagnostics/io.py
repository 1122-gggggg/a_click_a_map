from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _unique_nonempty(values: Sequence[Any], label: str) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _nonempty_string(value, f"{label} value")
        if text in seen:
            raise ValueError(f"{label} contains duplicate value: {text}")
        seen.add(text)
        output.append(text)
    if not output:
        raise ValueError(f"{label} cannot be empty")
    return output


def load_query_manifest(path: str | Path) -> list[str]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".json":
        payload = _read_json(source)
        if isinstance(payload, Mapping):
            keys = set(payload)
            if keys != {"queries"}:
                unknown = sorted(str(key) for key in keys - {"queries"})
                detail = ": " + ", ".join(unknown) if unknown else ""
                raise ValueError("query manifest object must contain only 'queries'" + detail)
            payload = payload["queries"]
        if not isinstance(payload, list):
            raise ValueError("query manifest JSON must be a list or {'queries': [...]}")
        return _unique_nonempty(payload, "query manifest")

    values = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _unique_nonempty(values, "query manifest")


def load_region_assignments(path: str | Path) -> dict[str, str]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    payload = _read_json(source)
    output: dict[str, str] = {}

    if isinstance(payload, Mapping):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for row in payload:
            if not isinstance(row, Mapping):
                raise ValueError("region assignment rows must be objects")
            if set(row) != {"query", "region"}:
                raise ValueError(
                    "region assignment rows must contain only 'query' and 'region'"
                )
            items.append((row["query"], row["region"]))
    else:
        raise ValueError("region assignments must be an object or list")

    for query, region in items:
        query_name = _nonempty_string(query, "region assignment query")
        region_id = _nonempty_string(region, "region assignment region")
        if query_name in output:
            raise ValueError(f"duplicate region assignment: {query_name}")
        output[query_name] = region_id
    if not output:
        raise ValueError("region assignments cannot be empty")
    return output


def load_risk_scores(path: str | Path) -> dict[str, float]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if len(fieldnames) != 2 or set(fieldnames) != {"query", "risk"}:
                raise ValueError("risk CSV header must contain exactly 'query' and 'risk'")
            rows: Any = list(reader)
    elif source.suffix.lower() == ".json":
        payload = _read_json(source)
        if isinstance(payload, Mapping):
            rows = [
                {"query": query, "risk": risk}
                for query, risk in payload.items()
            ]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("risk JSON must be an object or list")
    else:
        raise ValueError("risk scores must be .csv or .json")

    output: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("risk rows must be objects")
        if set(row) != {"query", "risk"}:
            raise ValueError("risk rows must contain only 'query' and 'risk'")
        query = _nonempty_string(row["query"], "risk query")
        if query in output:
            raise ValueError(f"duplicate risk score: {query}")
        if isinstance(row["risk"], bool):
            raise ValueError(f"{query}: risk must be numeric")
        try:
            risk = float(row["risk"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{query}: risk must be numeric") from exc
        if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"{query}: risk must be finite and in [0, 1]")
        output[query] = risk
    if not output:
        raise ValueError("risk scores cannot be empty")
    return output


def write_json(payload: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    return output
