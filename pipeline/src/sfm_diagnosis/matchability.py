from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .correspondences import load_correspondence_events
from .io import write_csv
from .models import MapData, Pose


@dataclass(frozen=True)
class MatchabilityConfig:
    p_min: float = 0.20
    beta_a: float = 1.0
    beta_b: float = 1.0
    half_life: float | None = None
    reweight_fim: bool = False

    def __post_init__(self) -> None:
        if self.beta_a <= 0 or self.beta_b <= 0:
            raise ValueError("beta_a and beta_b must be positive")
        if self.half_life is not None and self.half_life <= 0:
            raise ValueError("half_life must be positive when set")


@dataclass
class LandmarkMatchability:
    point_ids: np.ndarray
    n_obs: np.ndarray
    n_inlier: np.ndarray
    p: np.ndarray
    last_t: np.ndarray

    def p_by_point_id(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for point_id, n_obs, value in zip(self.point_ids, self.n_obs, self.p, strict=True):
            if int(n_obs) <= 0 or not np.isfinite(value):
                continue
            out[int(point_id)] = float(value)
        return out

    def as_rows(self) -> list[dict]:
        rows = []
        for point_id, n_obs, n_inlier, value, last_t in zip(
            self.point_ids, self.n_obs, self.n_inlier, self.p, self.last_t, strict=True
        ):
            rows.append(
                {
                    "point_id": int(point_id),
                    "n_obs": int(n_obs),
                    "n_inlier": int(n_inlier),
                    "p": None if not np.isfinite(value) else float(value),
                    "last_t": None if not np.isfinite(last_t) else float(last_t),
                }
            )
        return rows


@dataclass(frozen=True)
class QueryMatchabilityMetrics:
    matchable_points: int
    evidenced_visible_fraction: float
    mean_matchability: float | None
    effective_matchable: float
    point_p: np.ndarray

    def as_dict(self) -> dict:
        return {
            "matchable_points": self.matchable_points,
            "evidenced_visible_fraction": self.evidenced_visible_fraction,
            "mean_matchability": self.mean_matchability,
            "effective_matchable": self.effective_matchable,
        }


def matchability_config_from_dict(data: dict) -> MatchabilityConfig:
    allowed = set(asdict(MatchabilityConfig()))
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown matchability config keys: {sorted(unknown)}")
    return MatchabilityConfig(**data)


def build_landmark_matchability(
    map_data: MapData,
    events: list[dict],
    *,
    config: MatchabilityConfig | None = None,
) -> LandmarkMatchability:
    """Accumulate per-landmark inlier rates. Unknown point_ids are ignored."""
    cfg = config or MatchabilityConfig()
    known = {int(v) for v in map_data.point_ids.tolist()}
    grouped: dict[int, list[tuple[float, int, int]]] = defaultdict(list)
    timestamps: list[float] = []
    for event in events:
        point_id = int(event["point_id"])
        if point_id not in known:
            continue
        observed = bool(event.get("observed"))
        inlier = bool(event.get("inlier")) and observed
        stamp = event.get("timestamp")
        time_value = float(stamp) if stamp is not None and stamp != "" else float("nan")
        if np.isfinite(time_value):
            timestamps.append(time_value)
        grouped[point_id].append((time_value, int(observed), int(inlier)))

    t_now = max(timestamps) if timestamps else 0.0
    point_ids = []
    n_obs = []
    n_inlier = []
    probs = []
    last_t = []
    for point_id in sorted(grouped):
        rows = grouped[point_id]
        observed_count = int(sum(item[1] for item in rows))
        inlier_count = int(sum(item[2] for item in rows))
        times = [item[0] for item in rows if item[1] and np.isfinite(item[0])]
        if observed_count <= 0:
            continue
        if cfg.half_life is not None:
            weights = []
            successes = []
            for time_value, observed, inlier in rows:
                if not observed:
                    continue
                if np.isfinite(time_value):
                    weight = 2.0 ** (-(t_now - time_value) / cfg.half_life)
                else:
                    weight = 1.0
                weights.append(weight)
                successes.append(inlier)
            weight_sum = float(np.sum(weights))
            success_sum = float(np.dot(weights, successes))
            p = (success_sum + cfg.beta_a) / (weight_sum + cfg.beta_a + cfg.beta_b)
        else:
            p = (inlier_count + cfg.beta_a) / (observed_count + cfg.beta_a + cfg.beta_b)
        point_ids.append(point_id)
        n_obs.append(observed_count)
        n_inlier.append(inlier_count)
        probs.append(float(p))
        last_t.append(max(times) if times else float("nan"))

    return LandmarkMatchability(
        point_ids=np.asarray(point_ids, dtype=np.int64),
        n_obs=np.asarray(n_obs, dtype=np.int32),
        n_inlier=np.asarray(n_inlier, dtype=np.int32),
        p=np.asarray(probs, dtype=float),
        last_t=np.asarray(last_t, dtype=float),
    )


def query_matchability(
    table: LandmarkMatchability,
    pose: Pose,
    visible_idx: np.ndarray,
    map_data: MapData,
    *,
    config: MatchabilityConfig | None = None,
) -> QueryMatchabilityMetrics:
    """Roll up per-landmark matchability over currently visible points."""
    del pose  # rollup is over the already-selected visible set
    cfg = config or MatchabilityConfig()
    visible = np.asarray(visible_idx, dtype=int).reshape(-1)
    lookup = table.p_by_point_id()
    point_p = np.full(len(visible), np.nan, dtype=float)
    if len(visible):
        ids = map_data.point_ids[visible]
        for i, point_id in enumerate(ids.tolist()):
            value = lookup.get(int(point_id))
            if value is not None:
                point_p[i] = value
    evidenced = np.isfinite(point_p)
    evidenced_count = int(np.sum(evidenced))
    matchable = int(np.sum(evidenced & (point_p >= cfg.p_min)))
    mean = float(np.mean(point_p[evidenced])) if evidenced_count else None
    effective = float(np.nansum(np.where(evidenced, point_p, 0.0)))
    return QueryMatchabilityMetrics(
        matchable_points=matchable,
        evidenced_visible_fraction=evidenced_count / max(len(visible), 1),
        mean_matchability=mean,
        effective_matchable=effective,
        point_p=point_p,
    )


def load_landmark_matchability(path: str | Path) -> LandmarkMatchability:
    p = Path(path)
    with p.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    point_ids = []
    n_obs = []
    n_inlier = []
    probs = []
    last_t = []
    for row in rows:
        if row.get("point_id") in (None, ""):
            continue
        observed = row.get("n_obs", "")
        value = row.get("p", "")
        if observed in (None, "") and value in (None, ""):
            continue
        n = int(float(observed)) if observed not in (None, "") else 0
        point_ids.append(int(row["point_id"]))
        n_obs.append(n)
        n_inlier.append(int(float(row["n_inlier"])) if row.get("n_inlier") not in (None, "") else 0)
        probs.append(float(value) if value not in (None, "") else float("nan"))
        stamp = row.get("last_t", "")
        last_t.append(float(stamp) if stamp not in (None, "") else float("nan"))
    return LandmarkMatchability(
        point_ids=np.asarray(point_ids, dtype=np.int64),
        n_obs=np.asarray(n_obs, dtype=np.int32),
        n_inlier=np.asarray(n_inlier, dtype=np.int32),
        p=np.asarray(probs, dtype=float),
        last_t=np.asarray(last_t, dtype=float),
    )


def load_matchability_source(
    path: str | Path,
    map_data: MapData,
    *,
    config: MatchabilityConfig | None = None,
) -> LandmarkMatchability:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix != ".csv":
        return build_landmark_matchability(
            map_data, load_correspondence_events(p), config=config
        )
    with p.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
    if "observed" in fields or "inlier" in fields:
        return build_landmark_matchability(
            map_data, load_correspondence_events(p), config=config
        )
    return load_landmark_matchability(p)


def save_landmark_matchability(output_dir: str | Path, table: LandmarkMatchability) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "landmark_matchability.csv"
    write_csv(path, table.as_rows())
    return path
