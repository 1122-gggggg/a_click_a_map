"""Read-only diagnostics for a COLMAP matching database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_IMAGE_ID = 2_147_483_647


@dataclass(frozen=True)
class VerifiedPairPoints:
    image_id1: int
    image_id2: int
    name1: str
    name2: str
    points1: np.ndarray
    points2: np.ndarray
    indices1: np.ndarray
    indices2: np.ndarray


def _pair_id(a: int, b: int) -> int:
    return MAX_IMAGE_ID * min(a, b) + max(a, b)


def _decode(blob: bytes | None, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    if blob is None:
        return np.empty((0, *shape[1:]), dtype=dtype)
    arr = np.frombuffer(blob, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(f"invalid blob shape: expected {expected} values, got {arr.size}")
    return arr.reshape(shape)


def _coverage(points: np.ndarray, width: int, height: int) -> tuple[float, int]:
    if len(points) == 0 or width <= 0 or height <= 0:
        return 0.0, 0
    p = points[np.isfinite(points).all(axis=1)]
    p = p[(p[:, 0] >= 0) & (p[:, 0] < width) & (p[:, 1] >= 0) & (p[:, 1] < height)]
    if len(p) < 3:
        area = 0.0
    else:
        q = sorted(set(map(tuple, p.tolist())))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lo = []
        for x in q:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], x) <= 0:
                lo.pop()
            lo.append(x)
        up = []
        for x in reversed(q):
            while len(up) >= 2 and cross(up[-2], up[-1], x) <= 0:
                up.pop()
            up.append(x)
        hull = lo[:-1] + up[:-1]
        area = (
            abs(
                sum(
                    hull[i][0] * hull[(i + 1) % len(hull)][1]
                    - hull[(i + 1) % len(hull)][0] * hull[i][1]
                    for i in range(len(hull))
                )
            )
            / 2
        )
    cells = set(
        zip(
            np.clip((p[:, 0] * 4 / width).astype(int), 0, 3),
            np.clip((p[:, 1] * 4 / height).astype(int), 0, 3),
            strict=True,
        )
    )
    return float(area / (width * height)), len(cells)


def read_verified_pair_points(
    path: str | Path,
    name1: str,
    name2: str,
) -> VerifiedPairPoints:
    """Read one verified 2D-2D pair from COLMAP SQLite without a write lock."""

    if name1 == name2:
        raise ValueError("pair images must be distinct")
    uri = f"file:{Path(path).resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as database:
        rows = database.execute(
            "SELECT image_id,name FROM images WHERE name IN (?,?)",
            (name1, name2),
        ).fetchall()
        ids = {str(name): int(image_id) for image_id, name in rows}
        missing = {name1, name2} - set(ids)
        if missing:
            raise KeyError(f"images missing from matching database: {sorted(missing)!r}")
        requested_ids = (ids[name1], ids[name2])
        canonical_ids = tuple(sorted(requested_ids))
        pair_id = _pair_id(*canonical_ids)
        hit = database.execute(
            "SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?",
            (pair_id,),
        ).fetchone()
        if hit is None:
            raise KeyError(f"verified pair is missing: {name1!r}, {name2!r}")
        matches = _decode(hit[2], np.uint32, (hit[0], hit[1]))
        if matches.shape[1] != 2:
            raise ValueError("verified match table must contain two keypoint indices")
        keypoints: dict[int, np.ndarray] = {}
        for image_id in canonical_ids:
            keypoint_row = database.execute(
                "SELECT rows,cols,data FROM keypoints WHERE image_id=?", (image_id,)
            ).fetchone()
            if keypoint_row is None:
                raise KeyError(f"keypoints missing for image ID {image_id}")
            keypoints[image_id] = _decode(
                keypoint_row[2], np.float32, (keypoint_row[0], keypoint_row[1])
            )
        canonical_points = []
        for side, image_id in enumerate(canonical_ids):
            indices = matches[:, side]
            if np.any(indices >= len(keypoints[image_id])):
                raise ValueError("match index out of range")
            canonical_points.append(np.asarray(keypoints[image_id][indices, :2], dtype=float))
        canonical_indices = [matches[:, 0], matches[:, 1]]
        if requested_ids == canonical_ids:
            points1, points2 = canonical_points
            indices1, indices2 = canonical_indices
        else:
            points2, points1 = canonical_points
            indices2, indices1 = canonical_indices
        return VerifiedPairPoints(
            image_id1=requested_ids[0],
            image_id2=requested_ids[1],
            name1=name1,
            name2=name2,
            points1=points1,
            points2=points2,
            indices1=np.asarray(indices1, dtype=np.int64),
            indices2=np.asarray(indices2, dtype=np.int64),
        )


def read_matching_db(
    path: str | Path, *, include_coverage: bool = False, reliable_threshold: int = 1
) -> dict[str, list[dict[str, Any]]]:
    """Read matching health from SQLite without acquiring a write lock."""
    if reliable_threshold < 1:
        raise ValueError("reliable_threshold must be positive")
    uri = f"file:{Path(path).resolve()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as db:
        images = {
            r[0]: {"image_id": r[0], "name": r[1], "camera_id": r[2]}
            for r in db.execute("SELECT image_id,name,camera_id FROM images")
        }
        cameras = {
            r[0]: (r[1], r[2]) for r in db.execute("SELECT camera_id,width,height FROM cameras")
        }
        keypoints = {}
        if include_coverage:
            for image_id, rows, cols, blob in db.execute(
                "SELECT image_id,rows,cols,data FROM keypoints"
            ):
                keypoints[image_id] = _decode(blob, np.float32, (rows, cols))
        raw = {r[0]: r[1] for r in db.execute("SELECT pair_id,rows FROM matches")}
        geo = {r[0]: r[1] for r in db.execute("SELECT pair_id,rows FROM two_view_geometries")}
        pairs = []
        for pid in sorted(set(raw) | set(geo)):
            a, b = divmod(pid, MAX_IMAGE_ID)
            if b < a:
                a, b = b, a
            if a not in images or b not in images:
                continue
            rc, gc = int(raw.get(pid, 0)), int(geo.get(pid, 0))
            row: dict[str, Any] = {
                "image_id1": a,
                "image_id2": b,
                "name1": images[a]["name"],
                "name2": images[b]["name"],
                "raw_count": rc,
                "geometric_count": gc,
                "geometric_ratio": float(gc / rc) if rc else None,
            }
            if include_coverage:
                for prefix, table in (("raw", "matches"), ("geometric", "two_view_geometries")):
                    hit = db.execute(
                        f"SELECT rows,cols,data FROM {table} WHERE pair_id=?", (pid,)
                    ).fetchone()
                    pts = [[], []]
                    if hit:
                        m = _decode(hit[2], np.uint32, (hit[0], hit[1]))
                        for x, y in m:
                            if x >= len(keypoints.get(a, [])) or y >= len(keypoints.get(b, [])):
                                raise ValueError("match index out of range")
                            pts[0].append(keypoints[a][x, :2])
                            pts[1].append(keypoints[b][y, :2])
                    for side, image_id in enumerate((a, b), 1):
                        cov, occ = _coverage(
                            np.asarray(pts[side - 1]), *cameras[images[image_id]["camera_id"]]
                        )
                        (
                            row[f"{prefix}_image{side}_hull_coverage"],
                            row[f"{prefix}_image{side}_occupancy"],
                        ) = cov, occ
            pairs.append(row)
        out = []
        for image_id in sorted(images):
            ps = [p for p in pairs if image_id in (p["image_id1"], p["image_id2"])]
            total = sum(p["raw_count"] for p in ps)
            gtotal = sum(p["geometric_count"] for p in ps)
            strengths = [
                p["geometric_count"] for p in ps if p["geometric_count"] >= reliable_threshold
            ]
            raw_coverage: list[float] = []
            geometric_coverage: list[float] = []
            raw_occupancy: list[int] = []
            geometric_occupancy: list[int] = []
            if include_coverage:
                for pair in ps:
                    side = 1 if pair["image_id1"] == image_id else 2
                    raw_coverage.append(pair[f"raw_image{side}_hull_coverage"])
                    geometric_coverage.append(pair[f"geometric_image{side}_hull_coverage"])
                    raw_occupancy.append(pair[f"raw_image{side}_occupancy"])
                    geometric_occupancy.append(pair[f"geometric_image{side}_occupancy"])
            out.append(
                {
                    **images[image_id],
                    "raw_total": total,
                    "geometric_total": gtotal,
                    "weighted_geometric_ratio": gtotal / total if total else None,
                    "pair_count": len(ps),
                    "reliable_geometric_degree": len(strengths),
                    "reliable_geometric_strength": sum(strengths),
                    "raw_hull_coverage_median": (
                        float(np.median(raw_coverage)) if raw_coverage else None
                    ),
                    "raw_hull_coverage_max": max(raw_coverage, default=None),
                    "raw_grid_occupancy_max": max(raw_occupancy, default=None),
                    "geometric_hull_coverage_median": (
                        float(np.median(geometric_coverage)) if geometric_coverage else None
                    ),
                    "geometric_hull_coverage_max": max(geometric_coverage, default=None),
                    "geometric_grid_occupancy_max": max(geometric_occupancy, default=None),
                }
            )
        return {"pairs": pairs, "images": out}
