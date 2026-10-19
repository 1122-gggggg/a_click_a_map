from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .models import Pose


@dataclass(frozen=True)
class HistoricalStats:
    count: int = 0
    success_rate: float | None = None
    pnp_inliers: float | None = None
    inlier_ratio: float | None = None
    reproj_p90: float | None = None
    grid_occupancy: float | None = None
    hull_coverage: float | None = None
    positive_depth_ratio: float | None = None
    retrieval_score: float | None = None
    registration_confidence: float | None = None
    matches: float | None = None
    unique_tracks: float | None = None
    reference_count: float | None = None
    pose_consensus_translation_m: float | None = None
    pose_consensus_rotation_deg: float | None = None
    reference_dispersion_m: float | None = None
    reference_consensus_sigma_m: float | None = None
    reference_rotation_dispersion_deg: float | None = None
    reference_hypothesis_count: float | None = None
    reference_covariance_eligible_ratio: float | None = None
    reference_joint_gate_ratio: float | None = None
    mean_distance_m: float | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class LocalizationHistory:
    """Spatial/view-conditioned aggregation of actual localization logs.

    Expected columns (CSV or JSONL): x,y,z and optionally qx,qy,qz,qw defining
    camera-to-world orientation. Metrics are optional. Besides conventional
    retrieval/matching/PnP fields, logs may include RIC-Loc-style per-reference
    consensus diagnostics produced by ``sfm-diagnosis consensus``:

    ``reference_dispersion_m``
        RMS disagreement among per-reference query-center hypotheses.
    ``reference_consensus_sigma_m``
        Worst-direction sigma of the information-weighted consensus covariance.
    ``reference_rotation_dispersion_deg``
        RMS geodesic disagreement of per-reference rotation hypotheses.
    ``reference_covariance_eligible_ratio``
        Fraction of reference hypotheses with usable covariance information.

    These values are aggregated only over nearby, view-compatible records.
    """

    METRICS = (
        "success",
        "pnp_inliers",
        "inlier_ratio",
        "reproj_p90",
        "grid_occupancy",
        "hull_coverage",
        "positive_depth_ratio",
        "retrieval_score",
        "registration_confidence",
        "matches",
        "unique_tracks",
        "reference_count",
        "pose_consensus_translation_m",
        "pose_consensus_rotation_deg",
        "reference_dispersion_m",
        "reference_consensus_sigma_m",
        "reference_rotation_dispersion_deg",
        "reference_hypothesis_count",
        "reference_covariance_eligible_ratio",
        "reference_joint_gate_ratio",
    )

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.centers = np.asarray(
            [[_f(r, "x"), _f(r, "y"), _f(r, "z")] for r in rows], dtype=float
        ).reshape(-1, 3)
        self.forward = np.full((len(rows), 3), np.nan, dtype=float)
        for i, row in enumerate(rows):
            if all(k in row and str(row[k]).strip() != "" for k in ("qx", "qy", "qz", "qw")):
                quat = [_f(row, "qx"), _f(row, "qy"), _f(row, "qz"), _f(row, "qw")]
                self.forward[i] = Rotation.from_quat(quat).apply([0.0, 0.0, 1.0])
        self.values: dict[str, np.ndarray] = {}
        for metric in self.METRICS:
            vals = []
            for row in rows:
                raw = row.get(metric, "")
                if raw is None or str(raw).strip() == "":
                    vals.append(np.nan)
                elif metric == "success":
                    vals.append(_bool_float(raw))
                else:
                    vals.append(float(raw))
            self.values[metric] = np.asarray(vals, dtype=float)

    @classmethod
    def load(cls, path: str | Path) -> LocalizationHistory:
        p = Path(path)
        if p.suffix.lower() in {".jsonl", ".ndjson"}:
            rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            with p.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        if not rows:
            return cls([])
        missing = {"x", "y", "z"} - set(rows[0])
        if missing:
            raise ValueError(f"localization log is missing required columns: {sorted(missing)}")
        return cls(rows)

    def query(
        self,
        pose: Pose,
        *,
        radius_m: float = 2.0,
        max_view_angle_deg: float = 60.0,
        min_records: int = 1,
    ) -> HistoricalStats:
        if len(self.rows) == 0:
            return HistoricalStats()
        delta = self.centers - pose.center_w
        distance = np.linalg.norm(delta, axis=1)
        mask = distance <= radius_m
        valid_orientation = np.isfinite(self.forward[:, 0])
        if np.any(valid_orientation & mask):
            dots = np.einsum("ij,j->i", self.forward, pose.forward_w)
            angles = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
            mask &= (~valid_orientation) | (angles <= max_view_angle_deg)
        idx = np.flatnonzero(mask)
        if len(idx) < min_records:
            return HistoricalStats()
        # Closer records receive larger weights. A floor prevents singular weight at d=0.
        w = 1.0 / np.maximum(distance[idx], 0.25)
        w /= np.sum(w)

        def mean(metric: str) -> float | None:
            v = self.values[metric][idx]
            ok = np.isfinite(v)
            if not np.any(ok):
                return None
            ww = w[ok]
            ww /= np.sum(ww)
            return float(np.sum(ww * v[ok]))

        return HistoricalStats(
            count=len(idx),
            success_rate=mean("success"),
            pnp_inliers=mean("pnp_inliers"),
            inlier_ratio=mean("inlier_ratio"),
            reproj_p90=mean("reproj_p90"),
            grid_occupancy=mean("grid_occupancy"),
            hull_coverage=mean("hull_coverage"),
            positive_depth_ratio=mean("positive_depth_ratio"),
            retrieval_score=mean("retrieval_score"),
            registration_confidence=mean("registration_confidence"),
            matches=mean("matches"),
            unique_tracks=mean("unique_tracks"),
            reference_count=mean("reference_count"),
            pose_consensus_translation_m=mean("pose_consensus_translation_m"),
            pose_consensus_rotation_deg=mean("pose_consensus_rotation_deg"),
            reference_dispersion_m=mean("reference_dispersion_m"),
            reference_consensus_sigma_m=mean("reference_consensus_sigma_m"),
            reference_rotation_dispersion_deg=mean("reference_rotation_dispersion_deg"),
            reference_hypothesis_count=mean("reference_hypothesis_count"),
            reference_covariance_eligible_ratio=mean(
                "reference_covariance_eligible_ratio"
            ),
            reference_joint_gate_ratio=mean("reference_joint_gate_ratio"),
            mean_distance_m=float(np.mean(distance[idx])),
        )


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _bool_float(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "success", "ok"}:
        return 1.0
    if text in {"false", "no", "n", "failure", "fail"}:
        return 0.0
    return float(value)
