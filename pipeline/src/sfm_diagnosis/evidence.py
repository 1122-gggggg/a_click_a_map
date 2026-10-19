from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import MapData

COLMAP_MAX_IMAGE_ID = 2_147_483_647


@dataclass
class BuildEvidence:
    """Optional build-time evidence used for root-cause attribution.

    A final sparse model is sufficient for track, geometry and covisibility
    diagnostics. Pair-selection/matching and image-quality causes are only
    emitted when the corresponding evidence is available.
    """

    image_rows: list[dict] = field(default_factory=list)
    pair_rows: list[dict] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def image_by_name(self) -> dict[str, dict]:
        return {
            str(row["image_name"]): row
            for row in self.image_rows
            if row.get("image_name") not in (None, "")
        }

    @property
    def has_pair_selection(self) -> bool:
        return any(row.get("selected") is not None for row in self.pair_rows)

    @property
    def has_matching(self) -> bool:
        return any(row.get("num_matches") is not None for row in self.pair_rows)

    @property
    def has_geometric_verification(self) -> bool:
        return any(row.get("num_inliers") is not None for row in self.pair_rows)

    @property
    def has_image_quality(self) -> bool:
        quality_keys = {
            "sharpness_laplacian_var",
            "tenengrad",
            "entropy",
            "dark_ratio",
            "bright_ratio",
            "texture_score",
        }
        return any(any(row.get(k) is not None for k in quality_keys) for row in self.image_rows)

    def availability(self) -> dict:
        return {
            "pair_selection": self.has_pair_selection,
            "matching": self.has_matching,
            "geometric_verification": self.has_geometric_verification,
            "image_quality": self.has_image_quality,
            "image_rows": len(self.image_rows),
            "pair_rows": len(self.pair_rows),
            "sources": self.metadata.get("sources", []),
        }


def load_build_evidence(
    map_data: MapData,
    *,
    database: str | Path | None = None,
    pairs: str | Path | None = None,
    images_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> BuildEvidence:
    """Load optional evidence and merge it by image/pair identity.

    ``database`` reads COLMAP's SQLite database without decoding descriptor or
    match blobs; the ``rows`` fields in ``matches`` and
    ``two_view_geometries`` already provide raw-match and verified-inlier counts.

    ``pairs`` may be CSV/JSON/JSONL and is the preferred way to preserve
    retrieval candidates, including pairs that were considered but not selected.
    """
    image_rows: list[dict] = []
    pair_rows: list[dict] = []
    sources: list[str] = []

    if database is not None:
        db_images, db_pairs = load_colmap_database(database)
        image_rows.extend(db_images)
        pair_rows.extend(db_pairs)
        sources.append(f"colmap_database:{Path(database)}")

    if images_manifest is not None:
        manifest = [_normalize_image_row(row) for row in _read_rows(images_manifest)]
        image_rows.extend(row for row in manifest if row)
        sources.append(f"image_manifest:{Path(images_manifest)}")

    if pairs is not None:
        pair_table = [_normalize_pair_row(row) for row in _read_rows(pairs)]
        pair_rows.extend(row for row in pair_table if row)
        sources.append(f"pair_table:{Path(pairs)}")

    if images_dir is not None:
        quality_rows = analyze_registered_image_quality(map_data, images_dir)
        image_rows.extend(quality_rows)
        sources.append(f"image_quality:{Path(images_dir)}")

    image_rows = _merge_image_rows(image_rows)
    pair_rows = _merge_pair_rows(pair_rows)

    registered_names = set(map_data.image_names)
    for row in image_rows:
        if row.get("registered") is None and row.get("image_name") is not None:
            row["registered"] = row["image_name"] in registered_names

    return BuildEvidence(
        image_rows=image_rows,
        pair_rows=pair_rows,
        metadata={"sources": sources},
    )


def load_colmap_database(path: str | Path) -> tuple[list[dict], list[dict]]:
    """Read image identities and pair statistics from a COLMAP database."""
    db_path = Path(path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "images" not in tables:
            raise ValueError(f"{db_path} is missing the COLMAP images table")

        image_name_by_id = {
            int(image_id): str(name)
            for image_id, name in conn.execute("SELECT image_id, name FROM images").fetchall()
        }
        image_rows = [
            {"image_id": image_id, "image_name": name, "registered": None}
            for image_id, name in sorted(image_name_by_id.items())
        ]

        merged: dict[int, dict] = {}
        if "matches" in tables:
            for pair_id, rows in conn.execute("SELECT pair_id, rows FROM matches"):
                pid = int(pair_id)
                row = merged.setdefault(pid, {"pair_id": pid})
                row["num_matches"] = int(rows)

        if "two_view_geometries" in tables:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(two_view_geometries)").fetchall()
            }
            select = ["pair_id", "rows"]
            optional = [
                name
                for name in (
                    "config",
                    "qvec",
                    "tvec",
                    "qvec0",
                    "qvec1",
                    "qvec2",
                    "qvec3",
                    "tvec0",
                    "tvec1",
                    "tvec2",
                )
                if name in columns
            ]
            select.extend(optional)
            query = f"SELECT {', '.join(select)} FROM two_view_geometries"
            for values in conn.execute(query):
                record = dict(zip(select, values))
                pid = int(record["pair_id"])
                row = merged.setdefault(pid, {"pair_id": pid})
                row["num_inliers"] = int(record["rows"])
                if record.get("config") is not None:
                    row["two_view_config"] = int(record["config"])
                qvec = record.get("qvec")
                tvec = record.get("tvec")
                if all(record.get(f"qvec{i}") is not None for i in range(4)):
                    qvec = [record[f"qvec{i}"] for i in range(4)]
                if all(record.get(f"tvec{i}") is not None for i in range(3)):
                    tvec = [record[f"tvec{i}"] for i in range(3)]
                if qvec is not None:
                    row["two_view_qvec"] = qvec
                if tvec is not None:
                    row["two_view_tvec"] = tvec

        pair_rows: list[dict] = []
        for pair_id, row in merged.items():
            image_id_i, image_id_j = decode_pair_id(pair_id)
            matches = row.get("num_matches")
            inliers = row.get("num_inliers")
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "image_id_i": image_id_i,
                    "image_id_j": image_id_j,
                    "image_i": image_name_by_id.get(image_id_i),
                    "image_j": image_name_by_id.get(image_id_j),
                    "selected": None,
                    "attempted": matches is not None,
                    "num_matches": matches,
                    "num_inliers": inliers,
                    "inlier_ratio": (
                        float(inliers) / float(matches)
                        if inliers is not None and matches not in (None, 0)
                        else None
                    ),
                    "verified": (inliers > 0) if inliers is not None else None,
                    "retrieval_score": None,
                    "source": "colmap_database",
                    "two_view_config": row.get("two_view_config"),
                    "two_view_qvec": row.get("two_view_qvec"),
                    "two_view_tvec": row.get("two_view_tvec"),
                }
            )
        return image_rows, pair_rows
    finally:
        conn.close()


def decode_pair_id(pair_id: int) -> tuple[int, int]:
    image_id_j = int(pair_id % COLMAP_MAX_IMAGE_ID)
    image_id_i = int((pair_id - image_id_j) // COLMAP_MAX_IMAGE_ID)
    return image_id_i, image_id_j


def analyze_registered_image_quality(
    map_data: MapData,
    images_dir: str | Path,
    *,
    max_side: int = 1024,
) -> list[dict]:
    """Compute lightweight, dataset-relative image-quality evidence.

    Metrics are intentionally reported raw. Weak-region classification compares
    them to the current map's healthy distribution instead of assuming one
    universal blur/exposure threshold.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pillow is required for --images-dir. Install with "
            "`pip install -e '.[images]'` or `pip install Pillow`."
        ) from exc

    from scipy import ndimage

    root = Path(images_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    rows: list[dict] = []
    for image_id, name in zip(map_data.image_ids.tolist(), map_data.image_names, strict=True):
        path = root / name
        if not path.exists():
            continue
        with Image.open(path) as image:
            gray = image.convert("L")
            if max(gray.size) > max_side:
                scale = max_side / max(gray.size)
                resized = (
                    max(1, int(round(gray.size[0] * scale))),
                    max(1, int(round(gray.size[1] * scale))),
                )
                gray = gray.resize(resized)
            arr = np.asarray(gray, dtype=np.float32) / 255.0

        lap = ndimage.laplace(arr)
        gx = ndimage.sobel(arr, axis=1, mode="reflect")
        gy = ndimage.sobel(arr, axis=0, mode="reflect")
        hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=False)
        probs = hist.astype(float)
        probs /= max(float(np.sum(probs)), 1.0)
        nz = probs > 0
        entropy = -float(np.sum(probs[nz] * np.log2(probs[nz])))
        rows.append(
            {
                "image_id": int(image_id),
                "image_name": name,
                "registered": True,
                "sharpness_laplacian_var": float(np.var(lap)),
                "tenengrad": float(np.mean(gx * gx + gy * gy)),
                "entropy": entropy,
                "dark_ratio": float(np.mean(arr <= 0.03)),
                "bright_ratio": float(np.mean(arr >= 0.97)),
                "source": "image_quality",
            }
        )
    return rows


def _read_rows(path: str | Path) -> list[dict]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(dict(json.loads(line)))
        return rows
    if suffix == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        if isinstance(payload, dict):
            for key in ("rows", "pairs", "images"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key]]
        raise TypeError(f"Unsupported JSON table shape in {p}")
    raise ValueError(f"Unsupported evidence table format: {p.suffix}")


def _normalize_image_row(row: dict) -> dict:
    image_name = _first(row, "image_name", "name", "image", "filename")
    image_id = _int_or_none(_first(row, "image_id", "id"))
    if image_name is None and image_id is None:
        return {}
    out = {
        "image_id": image_id,
        "image_name": str(image_name) if image_name is not None else None,
        "registered": _bool_or_none(_first(row, "registered", "is_registered")),
        "route_id": _first(row, "route_id", "route", "sequence"),
        "timestamp": _first(row, "timestamp", "time"),
        "x": _float_or_none(_first(row, "x", "center_x", "camera_x")),
        "y": _float_or_none(_first(row, "y", "center_y", "camera_y")),
        "z": _float_or_none(_first(row, "z", "center_z", "camera_z")),
    }
    for key in (
        "sharpness_laplacian_var",
        "laplacian_variance",
        "tenengrad",
        "entropy",
        "dark_ratio",
        "bright_ratio",
        "texture_score",
        "grid_coverage",
    ):
        value = _float_or_none(row.get(key))
        if value is not None:
            canonical = (
                "sharpness_laplacian_var" if key == "laplacian_variance" else key
            )
            out[canonical] = value
    return out


def _normalize_pair_row(row: dict) -> dict:
    image_i = _first(row, "image_i", "image1", "name1", "query", "src")
    image_j = _first(row, "image_j", "image2", "name2", "reference", "dst")
    image_id_i = _int_or_none(_first(row, "image_id_i", "image1_id", "src_id"))
    image_id_j = _int_or_none(_first(row, "image_id_j", "image2_id", "dst_id"))
    if image_i is None and image_j is None and image_id_i is None and image_id_j is None:
        return {}

    num_matches = _float_or_none(_first(row, "num_matches", "matches", "raw_matches"))
    num_inliers = _float_or_none(
        _first(row, "num_inliers", "inliers", "verified_inliers", "geometry_inliers")
    )
    ratio = _float_or_none(_first(row, "inlier_ratio", "verified_ratio"))
    if ratio is None and num_inliers is not None and num_matches not in (None, 0):
        ratio = num_inliers / num_matches

    selected = _bool_or_none(_first(row, "selected", "retrieved", "pair_selected"))
    attempted = _bool_or_none(_first(row, "attempted", "matched"))
    if attempted is None:
        attempted = num_matches is not None
    verified = _bool_or_none(_first(row, "verified", "geometry_verified"))
    if verified is None and num_inliers is not None:
        verified = num_inliers > 0

    return {
        "image_i": str(image_i) if image_i is not None else None,
        "image_j": str(image_j) if image_j is not None else None,
        "image_id_i": image_id_i,
        "image_id_j": image_id_j,
        "selected": selected,
        "attempted": attempted,
        "retrieval_score": _float_or_none(
            _first(row, "retrieval_score", "similarity", "global_score")
        ),
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": ratio,
        "verified": verified,
        "essential_support": _float_or_none(
            _first(row, "essential_support", "e_support")
        ),
        "homography_support": _float_or_none(
            _first(row, "homography_support", "h_support")
        ),
        "two_view_config": _int_or_none(_first(row, "two_view_config", "config")),
        "two_view_qvec": _first(row, "two_view_qvec", "qvec"),
        "two_view_tvec": _first(row, "two_view_tvec", "tvec"),
        "two_view_qw": _float_or_none(_first(row, "two_view_qw", "qw")),
        "two_view_qx": _float_or_none(_first(row, "two_view_qx", "qx")),
        "two_view_qy": _float_or_none(_first(row, "two_view_qy", "qy")),
        "two_view_qz": _float_or_none(_first(row, "two_view_qz", "qz")),
        "two_view_tx": _float_or_none(_first(row, "two_view_tx", "tx")),
        "two_view_ty": _float_or_none(_first(row, "two_view_ty", "ty")),
        "two_view_tz": _float_or_none(_first(row, "two_view_tz", "tz")),
    }


def _merge_image_rows(rows: list[dict]) -> list[dict]:
    merged: dict[tuple[str, object], dict] = {}
    anonymous: list[dict] = []
    for row in rows:
        if not row:
            continue
        if row.get("image_name") not in (None, ""):
            key = ("name", str(row["image_name"]))
        elif row.get("image_id") is not None:
            key = ("id", int(row["image_id"]))
        else:
            anonymous.append(dict(row))
            continue
        current = merged.setdefault(key, {})
        for field, value in row.items():
            if value is not None:
                current[field] = value
    return list(merged.values()) + anonymous


def _merge_pair_rows(rows: list[dict]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    anonymous: list[dict] = []
    for row in rows:
        if not row:
            continue
        key = _pair_key(row)
        if key is None:
            anonymous.append(dict(row))
            continue
        current = merged.setdefault(key, {})
        for field, value in row.items():
            if value is not None:
                current[field] = value
    return list(merged.values()) + anonymous


def _pair_key(row: dict) -> tuple | None:
    names = [row.get("image_i"), row.get("image_j")]
    if all(v not in (None, "") for v in names):
        return ("name",) + tuple(sorted(str(v) for v in names))
    ids = [row.get("image_id_i"), row.get("image_id_j")]
    if all(v is not None for v in ids):
        return ("id",) + tuple(sorted(int(v) for v in ids))
    return None


def _first(row: dict, *keys: str):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
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
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None
