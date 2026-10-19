from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ReferenceHypothesis:
    """One query-pose hypothesis induced by one reference.

    ``covariance_w`` is a 3x3 covariance of the induced query camera center in
    map/world coordinates. It may come from RIC-Loc-style covariance propagation,
    per-reference PnP information, or another estimator.

    ``weight`` is a *relative* confidence inside one query. Consensus normalizes
    positive weights to mean one, so multiplying every weight by the same constant
    cannot make the reported covariance artificially more confident. Set weight to
    zero to disable a hypothesis.
    """

    query_id: str
    reference_id: str
    center_w: np.ndarray
    covariance_w: np.ndarray | None = None
    R_wc: np.ndarray | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center_w, dtype=float).reshape(3)
        if not np.all(np.isfinite(center)):
            raise ValueError("reference hypothesis center must be finite")
        object.__setattr__(self, "center_w", center)

        if self.covariance_w is not None:
            covariance = np.asarray(self.covariance_w, dtype=float).reshape(3, 3)
            covariance = 0.5 * (covariance + covariance.T)
            object.__setattr__(self, "covariance_w", covariance)

        if self.R_wc is not None:
            rotation = np.asarray(self.R_wc, dtype=float).reshape(3, 3)
            if not np.all(np.isfinite(rotation)):
                raise ValueError("reference hypothesis rotation must be finite")
            object.__setattr__(self, "R_wc", rotation)

        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("reference hypothesis weight must be finite and non-negative")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class ReferenceConsensusMetrics:
    query_id: str
    hypothesis_count: int
    covariance_eligible_count: int
    covariance_eligible_ratio: float
    consensus_center_w: np.ndarray
    consensus_covariance_w: np.ndarray | None
    sigma_disp_m: float
    sigma_cons_m: float | None
    heterogeneity_inflation: float | None
    normalized_residual_sum: float | None
    translation_residual_p90_m: float
    rotation_consensus_R_wc: np.ndarray | None
    rotation_dispersion_deg: float | None
    robust_weights: dict[str, float]
    mahalanobis_sq: dict[str, float]
    geometric_outlier_reference_ids: tuple[str, ...]
    invalid_covariance_reference_ids: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "hypothesis_count": self.hypothesis_count,
            "covariance_eligible_count": self.covariance_eligible_count,
            "covariance_eligible_ratio": self.covariance_eligible_ratio,
            "consensus_center_w": self.consensus_center_w.tolist(),
            "consensus_covariance_w": (
                self.consensus_covariance_w.tolist()
                if self.consensus_covariance_w is not None
                else None
            ),
            "sigma_disp_m": self.sigma_disp_m,
            "sigma_cons_m": self.sigma_cons_m,
            "heterogeneity_inflation": self.heterogeneity_inflation,
            "normalized_residual_sum": self.normalized_residual_sum,
            "translation_residual_p90_m": self.translation_residual_p90_m,
            "rotation_consensus_R_wc": (
                self.rotation_consensus_R_wc.tolist()
                if self.rotation_consensus_R_wc is not None
                else None
            ),
            "rotation_dispersion_deg": self.rotation_dispersion_deg,
            "robust_weights": self.robust_weights,
            "mahalanobis_sq": self.mahalanobis_sq,
            "geometric_outlier_reference_ids": list(self.geometric_outlier_reference_ids),
            "invalid_covariance_reference_ids": list(self.invalid_covariance_reference_ids),
            "calibration_note": (
                "sigma_cons is a local model-based ranking signal unless upstream "
                "covariances and shared correlations are calibrated on held-out queries."
            ),
        }

    def localization_log_row(self) -> dict:
        """Return fields understood by :class:`LocalizationHistory`."""
        return {
            "reference_hypothesis_count": self.hypothesis_count,
            "reference_covariance_eligible_ratio": self.covariance_eligible_ratio,
            "reference_dispersion_m": self.sigma_disp_m,
            "reference_consensus_sigma_m": self.sigma_cons_m,
            "reference_rotation_dispersion_deg": self.rotation_dispersion_deg,
        }


@dataclass(frozen=True)
class SelectiveNormalization:
    """Held-out medians for RIC-Loc σ_joint (Kang et al., arXiv:2607.04722 Eq. 22)."""

    median_sigma_cons_m: float
    median_sigma_disp_m: float
    n_covariance_eligible: int


class ReferenceConsensusStatus(str, Enum):
    HEALTHY = "HEALTHY"
    REFERENCE_DISAGREEMENT = "REFERENCE_DISAGREEMENT"
    OBSERVABILITY_WEAK = "OBSERVABILITY_WEAK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ReferenceConsensusAssessment:
    status: ReferenceConsensusStatus
    reasons: tuple[str, ...]
    joint_gate_ratio: float

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "joint_gate_ratio": self.joint_gate_ratio,
        }


def compute_reference_consensus(
    hypotheses: list[ReferenceHypothesis],
    *,
    student_t_nu: float = 5.0,
    covariance_floor_m: float = 1e-3,
    sigma_inflation: float = 1.0,
    max_iterations: int = 50,
    tolerance_m: float = 1e-7,
) -> ReferenceConsensusMetrics:
    """Fuse per-reference query hypotheses using robust information weighting.

    Translation follows the RIC-Loc diagnostic abstraction:

    ``m_i^2 = (C-C_i)^T Sigma_i^-1 (C-C_i)``.

    Student-t IRLS is used only to *downweight* inconsistent references; it never
    increases their nominal information above the supplied covariance. Positive
    upstream weights are normalized to mean one, so they represent relative rather
    than arbitrary absolute confidence.

    ``sigma_disp`` remains separate from ``sigma_cons`` so cross-reference
    disagreement and weak observability are never collapsed into one score.
    ``sigma_inflation`` is an explicit held-out calibration hook for correlated
    references/shared-model error.
    """
    if not hypotheses:
        raise ValueError("at least one reference hypothesis is required")
    if student_t_nu <= 0.0 or not np.isfinite(student_t_nu):
        raise ValueError("student_t_nu must be finite and > 0")
    if covariance_floor_m < 0.0 or not np.isfinite(covariance_floor_m):
        raise ValueError("covariance_floor_m must be finite and >= 0")
    if sigma_inflation <= 0.0 or not np.isfinite(sigma_inflation):
        raise ValueError("sigma_inflation must be finite and > 0")

    query_ids = {str(h.query_id) for h in hypotheses}
    if len(query_ids) != 1:
        raise ValueError("compute_reference_consensus expects one query_id at a time")

    active = [h for h in hypotheses if h.weight > 0.0]
    if not active:
        raise ValueError("all reference hypotheses are disabled by zero weight")
    reference_ids = [str(h.reference_id) for h in active]
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("reference_id values must be unique within one query")

    centers = np.asarray([h.center_w for h in active], dtype=float)
    floor2 = max(float(covariance_floor_m), 1e-9) ** 2

    eligible: list[ReferenceHypothesis] = []
    precisions: list[np.ndarray] = []
    invalid_covariance_ids: list[str] = []
    for h in active:
        if h.covariance_w is None:
            continue
        precision = _regularized_precision(h.covariance_w, floor2)
        if precision is None:
            invalid_covariance_ids.append(str(h.reference_id))
            continue
        eligible.append(h)
        precisions.append(precision)

    robust_weights: dict[str, float] = {}
    mahalanobis_sq: dict[str, float] = {}
    covariance_consensus: np.ndarray | None = None
    sigma_cons: float | None = None
    heterogeneity: float | None = None
    residual_sum: float | None = None

    if eligible:
        eligible_centers = np.asarray([h.center_w for h in eligible], dtype=float)
        precisions_arr = np.asarray(precisions, dtype=float)
        base_weights = np.asarray([h.weight for h in eligible], dtype=float)
        base_weights /= max(float(np.mean(base_weights)), 1e-12)

        consensus = _information_weighted_center(
            eligible_centers,
            precisions_arr,
            base_weights,
        )
        nu = float(student_t_nu)
        final_weights = base_weights.copy()
        final_m2 = np.zeros(len(eligible), dtype=float)

        for _ in range(max(int(max_iterations), 1)):
            delta = eligible_centers - consensus[None, :]
            m2 = np.einsum("ni,nij,nj->n", delta, precisions_arr, delta)
            robust = np.minimum(1.0, (nu + 3.0) / (nu + np.maximum(m2, 0.0)))
            weights = base_weights * robust
            updated = _information_weighted_center(
                eligible_centers,
                precisions_arr,
                weights,
            )
            final_weights = weights
            final_m2 = m2
            if np.linalg.norm(updated - consensus) <= max(float(tolerance_m), 0.0):
                consensus = updated
                break
            consensus = updated

        delta = eligible_centers - consensus[None, :]
        final_m2 = np.einsum("ni,nij,nj->n", delta, precisions_arr, delta)
        robust = np.minimum(1.0, (nu + 3.0) / (nu + np.maximum(final_m2, 0.0)))
        final_weights = base_weights * robust

        information = np.einsum("n,nij->ij", final_weights, precisions_arr)
        aggregate_covariance = np.linalg.pinv(information, rcond=1e-12)

        # Normalize only for the residual-consistency test. This prevents a global
        # confidence rescaling from changing Q while retaining relative confidence.
        q_weights = final_weights / max(float(np.mean(final_weights)), 1e-12)
        residual_sum = float(np.sum(q_weights * final_m2))
        effective_n = float(np.sum(q_weights) ** 2 / max(np.sum(q_weights**2), 1e-12))
        dof = max(3.0 * effective_n - 3.0, 1.0)
        heterogeneity = max(1.0, residual_sum / dof)

        covariance_consensus = (
            aggregate_covariance * heterogeneity * float(sigma_inflation) ** 2
        )
        covariance_consensus = 0.5 * (
            covariance_consensus + covariance_consensus.T
        )
        sigma_cons = _worst_sigma(covariance_consensus)

        for h, w, m2 in zip(eligible, final_weights, final_m2, strict=True):
            robust_weights[str(h.reference_id)] = float(w)
            mahalanobis_sq[str(h.reference_id)] = float(m2)
    else:
        consensus = np.median(centers, axis=0)

    residual = np.linalg.norm(centers - consensus[None, :], axis=1)
    sigma_disp = float(np.sqrt(np.mean(residual**2)))
    residual_p90 = float(np.percentile(residual, 90)) if len(residual) else 0.0
    geometric_outliers = _geometric_outliers(active, residual)
    rotation_consensus, rotation_dispersion = _rotation_consensus(active)

    return ReferenceConsensusMetrics(
        query_id=str(active[0].query_id),
        hypothesis_count=len(active),
        covariance_eligible_count=len(eligible),
        covariance_eligible_ratio=len(eligible) / max(len(active), 1),
        consensus_center_w=np.asarray(consensus, dtype=float),
        consensus_covariance_w=covariance_consensus,
        sigma_disp_m=sigma_disp,
        sigma_cons_m=sigma_cons,
        heterogeneity_inflation=heterogeneity,
        normalized_residual_sum=residual_sum,
        translation_residual_p90_m=residual_p90,
        rotation_consensus_R_wc=rotation_consensus,
        rotation_dispersion_deg=rotation_dispersion,
        robust_weights=robust_weights,
        mahalanobis_sq=mahalanobis_sq,
        geometric_outlier_reference_ids=geometric_outliers,
        invalid_covariance_reference_ids=tuple(invalid_covariance_ids),
    )


def fit_selective_normalization(
    metrics: Sequence[ReferenceConsensusMetrics],
) -> SelectiveNormalization | None:
    """Fit RIC-Loc fusion scales on a held-out, covariance-eligible set.

    In-sample medians are not this function's contract. Pass a disjoint query
    split (leave-one-scene-out, cross-building, or a frozen S9 calibration set).
    """
    eligible = [row for row in metrics if row.sigma_cons_m is not None]
    if not eligible:
        return None
    cons = np.asarray([row.sigma_cons_m for row in eligible], dtype=float)
    disp = np.asarray([row.sigma_disp_m for row in eligible], dtype=float)
    return SelectiveNormalization(
        median_sigma_cons_m=float(np.median(cons)),
        median_sigma_disp_m=float(np.median(disp)),
        n_covariance_eligible=len(eligible),
    )


def compute_sigma_joint(
    metrics: ReferenceConsensusMetrics,
    normalization: SelectiveNormalization | None,
) -> float | None:
    """Return max(σ_cons/median, σ_disp/median), or None if not covariance-eligible.

    Retrieval-score gaps are intentionally unused: Kang et al. Table 1 shows
    top-1−top-2 MegaLoc margins are near-random for pose-failure ranking.
    """
    if normalization is None or metrics.sigma_cons_m is None:
        return None
    cons = metrics.sigma_cons_m / max(float(normalization.median_sigma_cons_m), 1e-12)
    disp = metrics.sigma_disp_m / max(float(normalization.median_sigma_disp_m), 1e-12)
    return float(max(cons, disp))


def assess_reference_consensus(
    metrics: ReferenceConsensusMetrics,
    *,
    max_dispersion_m: float = 0.5,
    max_consensus_sigma_m: float = 0.5,
    max_rotation_dispersion_deg: float = 5.0,
    min_covariance_eligible_ratio: float = 0.5,
    min_hypothesis_count: int = 2,
) -> ReferenceConsensusAssessment:
    """Classify evidence sufficiency, disagreement, and observability separately."""
    reasons: list[str] = []
    ratios: list[float] = []

    minimum_count = max(int(min_hypothesis_count), 1)
    insufficient = metrics.hypothesis_count < minimum_count
    count_ratio = minimum_count / max(metrics.hypothesis_count, 1)
    ratios.append(float(count_ratio))
    if insufficient:
        reasons.append(
            "too few active reference hypotheses for consensus: "
            f"count={metrics.hypothesis_count}, required={minimum_count}"
        )

    disp_limit = max(float(max_dispersion_m), 1e-12)
    disp_ratio = metrics.sigma_disp_m / disp_limit
    ratios.append(disp_ratio)
    disagreement = disp_ratio > 1.0
    if disagreement:
        reasons.append(
            f"reference center hypotheses disagree: sigma_disp={metrics.sigma_disp_m:.4g}"
        )

    rotation_bad = False
    if metrics.rotation_dispersion_deg is not None:
        rot_limit = max(float(max_rotation_dispersion_deg), 1e-12)
        rot_ratio = metrics.rotation_dispersion_deg / rot_limit
        ratios.append(rot_ratio)
        rotation_bad = rot_ratio > 1.0
        if rotation_bad:
            reasons.append(
                "reference rotation hypotheses disagree: "
                f"dispersion={metrics.rotation_dispersion_deg:.4g} deg"
            )
    disagreement = disagreement or rotation_bad

    eligible_limit = float(np.clip(min_covariance_eligible_ratio, 0.0, 1.0))
    observability = metrics.covariance_eligible_ratio < eligible_limit
    if eligible_limit > 0.0:
        eligible_ratio = eligible_limit / max(metrics.covariance_eligible_ratio, 1e-12)
        ratios.append(eligible_ratio)
    if observability:
        reasons.append(
            "too few references have usable positive-semidefinite covariance: "
            f"ratio={metrics.covariance_eligible_ratio:.3f}"
        )

    if metrics.sigma_cons_m is None:
        observability = True
        ratios.append(1.0 + 1e-9)
        reasons.append("no covariance-eligible references; sigma_cons is unavailable")
    else:
        cons_limit = max(float(max_consensus_sigma_m), 1e-12)
        cons_ratio = metrics.sigma_cons_m / cons_limit
        ratios.append(cons_ratio)
        if cons_ratio > 1.0:
            observability = True
            reasons.append(
                "information-weighted consensus is weakly constrained: "
                f"sigma_cons={metrics.sigma_cons_m:.4g}"
            )

    if metrics.invalid_covariance_reference_ids:
        reasons.append(
            "invalid/non-PSD covariance rejected for references: "
            + ", ".join(metrics.invalid_covariance_reference_ids)
        )

    if insufficient:
        status = ReferenceConsensusStatus.INSUFFICIENT_EVIDENCE
    elif disagreement and observability:
        status = ReferenceConsensusStatus.CRITICAL
    elif disagreement:
        status = ReferenceConsensusStatus.REFERENCE_DISAGREEMENT
    elif observability:
        status = ReferenceConsensusStatus.OBSERVABILITY_WEAK
    else:
        status = ReferenceConsensusStatus.HEALTHY

    return ReferenceConsensusAssessment(
        status=status,
        reasons=tuple(reasons),
        joint_gate_ratio=float(max(ratios, default=0.0)),
    )


def load_reference_hypotheses(path: str | Path) -> list[ReferenceHypothesis]:
    """Load per-reference hypotheses from CSV, JSON, or JSONL.

    Required center fields are ``x,y,z`` (or ``center_x,center_y,center_z``).
    Covariance may be supplied as isotropic ``sigma_m``, diagonal
    ``sigma_x_m,sigma_y_m,sigma_z_m``, or six symmetric covariance entries.
    Optional ``qx,qy,qz,qw`` stores a camera-to-world rotation hypothesis.
    """
    rows = _read_rows(path)
    result: list[ReferenceHypothesis] = []
    for index, row in enumerate(rows):
        center = np.array(
            [
                _required_float(row, "x", "center_x"),
                _required_float(row, "y", "center_y"),
                _required_float(row, "z", "center_z"),
            ],
            dtype=float,
        )
        covariance = _covariance_from_row(row)
        R_wc = None
        if all(_has_value(row.get(k)) for k in ("qx", "qy", "qz", "qw")):
            quat = [float(row[k]) for k in ("qx", "qy", "qz", "qw")]
            R_wc = Rotation.from_quat(quat).as_matrix()
        query_id = str(row.get("query_id") or row.get("query") or "query")
        reference_id = str(
            row.get("reference_id") or row.get("ref_id") or row.get("reference") or index
        )
        weight_raw = row.get("weight", row.get("confidence", 1.0))
        weight = float(weight_raw) if _has_value(weight_raw) else 1.0
        result.append(
            ReferenceHypothesis(
                query_id=query_id,
                reference_id=reference_id,
                center_w=center,
                covariance_w=covariance,
                R_wc=R_wc,
                weight=weight,
            )
        )
    return result


def group_reference_hypotheses(
    hypotheses: list[ReferenceHypothesis],
) -> dict[str, list[ReferenceHypothesis]]:
    groups: dict[str, list[ReferenceHypothesis]] = {}
    for hypothesis in hypotheses:
        groups.setdefault(str(hypothesis.query_id), []).append(hypothesis)
    return groups


def analyze_reference_hypotheses(
    path: str | Path,
    *,
    student_t_nu: float = 5.0,
    covariance_floor_m: float = 1e-3,
    sigma_inflation: float = 1.0,
    max_dispersion_m: float = 0.5,
    max_consensus_sigma_m: float = 0.5,
    max_rotation_dispersion_deg: float = 5.0,
    min_covariance_eligible_ratio: float = 0.5,
    min_hypothesis_count: int = 2,
    held_out_path: str | Path | None = None,
    selective_normalization: SelectiveNormalization | None = None,
) -> list[dict]:
    hypotheses = load_reference_hypotheses(path)
    groups = group_reference_hypotheses(hypotheses)
    rows = []
    computed: list[tuple[str, ReferenceConsensusMetrics, ReferenceConsensusAssessment]] = []
    for query_id, group in groups.items():
        metrics = compute_reference_consensus(
            group,
            student_t_nu=student_t_nu,
            covariance_floor_m=covariance_floor_m,
            sigma_inflation=sigma_inflation,
        )
        assessment = assess_reference_consensus(
            metrics,
            max_dispersion_m=max_dispersion_m,
            max_consensus_sigma_m=max_consensus_sigma_m,
            max_rotation_dispersion_deg=max_rotation_dispersion_deg,
            min_covariance_eligible_ratio=min_covariance_eligible_ratio,
            min_hypothesis_count=min_hypothesis_count,
        )
        computed.append((query_id, metrics, assessment))

    normalization = selective_normalization
    if held_out_path is not None:
        held_groups = group_reference_hypotheses(load_reference_hypotheses(held_out_path))
        leaked = sorted(set(groups) & set(held_groups))
        if leaked:
            raise ValueError(
                "held-out query ids leak into the ranked set: " + ", ".join(leaked)
            )
        held_metrics = [
            compute_reference_consensus(
                group,
                student_t_nu=student_t_nu,
                covariance_floor_m=covariance_floor_m,
                sigma_inflation=sigma_inflation,
            )
            for group in held_groups.values()
        ]
        normalization = fit_selective_normalization(held_metrics)

    for query_id, metrics, assessment in computed:
        log_fields = metrics.localization_log_row()
        log_fields["reference_joint_gate_ratio"] = assessment.joint_gate_ratio
        sigma_joint = compute_sigma_joint(metrics, normalization)
        log_fields["reference_sigma_joint"] = sigma_joint
        rows.append(
            {
                "query_id": query_id,
                "metrics": metrics.as_dict(),
                "assessment": assessment.as_dict(),
                "sigma_joint": sigma_joint,
                "localization_log_fields": log_fields,
            }
        )
    return rows


def _information_weighted_center(
    centers: np.ndarray,
    precisions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    information = np.einsum("n,nij->ij", weights, precisions)
    rhs = np.einsum("n,nij,nj->i", weights, precisions, centers)
    return np.linalg.pinv(information, rcond=1e-12) @ rhs


def _regularized_precision(covariance: np.ndarray, floor2: float) -> np.ndarray | None:
    covariance = np.asarray(covariance, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(covariance)):
        return None
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), float(floor2), 1e-18)
    if float(eigenvalues[0]) < -1e-8 * scale:
        return None
    regularized = np.maximum(eigenvalues, 0.0) + floor2
    return (eigenvectors * (1.0 / regularized)[None, :]) @ eigenvectors.T


def _rotation_consensus(
    hypotheses: list[ReferenceHypothesis],
) -> tuple[np.ndarray | None, float | None]:
    rotations = [h for h in hypotheses if h.R_wc is not None]
    if not rotations:
        return None, None
    quaternions = np.asarray(
        [Rotation.from_matrix(h.R_wc).as_quat() for h in rotations],
        dtype=float,
    )
    weights = np.asarray([h.weight for h in rotations], dtype=float)
    mean_quaternion = _chordal_quaternion_mean(quaternions, weights)
    mean_rotation = Rotation.from_quat(mean_quaternion).as_matrix()
    residual = _rotation_residuals_deg(mean_rotation, rotations)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    threshold = max(2.0, median + 3.0 * 1.4826 * mad)
    inlier = residual <= threshold
    if np.any(inlier) and not np.all(inlier):
        mean_quaternion = _chordal_quaternion_mean(quaternions[inlier], weights[inlier])
        mean_rotation = Rotation.from_quat(mean_quaternion).as_matrix()
        residual = _rotation_residuals_deg(mean_rotation, rotations)
    dispersion = float(np.sqrt(np.mean(residual**2)))
    return mean_rotation, dispersion


def _chordal_quaternion_mean(quaternions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = np.einsum("n,ni,nj->ij", weights, quaternions, quaternions)
    _, vectors = np.linalg.eigh(matrix)
    quaternion = vectors[:, -1]
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return quaternion


def _rotation_residuals_deg(
    mean_rotation: np.ndarray,
    hypotheses: list[ReferenceHypothesis],
) -> np.ndarray:
    values = []
    for h in hypotheses:
        if h.R_wc is None:
            continue
        relative = mean_rotation.T @ h.R_wc
        values.append(np.degrees(Rotation.from_matrix(relative).magnitude()))
    return np.asarray(values, dtype=float)


def _geometric_outliers(
    hypotheses: list[ReferenceHypothesis],
    residual: np.ndarray,
) -> tuple[str, ...]:
    if len(residual) < 3:
        return ()
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        return ()
    threshold = median + 3.5 * scale
    return tuple(
        str(h.reference_id)
        for h, value in zip(hypotheses, residual, strict=True)
        if value > threshold
    )


def _covariance_from_row(row: dict) -> np.ndarray | None:
    full_keys = ("cov_xx", "cov_xy", "cov_xz", "cov_yy", "cov_yz", "cov_zz")
    if all(_has_value(row.get(k)) for k in full_keys):
        xx, xy, xz, yy, yz, zz = (float(row[k]) for k in full_keys)
        return np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]],
            dtype=float,
        )
    diag_keys = ("sigma_x_m", "sigma_y_m", "sigma_z_m")
    if all(_has_value(row.get(k)) for k in diag_keys):
        sigma = np.asarray([float(row[k]) for k in diag_keys], dtype=float)
        if np.any(sigma < 0.0):
            raise ValueError("diagonal sigma values must be non-negative")
        return np.diag(sigma**2)
    if _has_value(row.get("sigma_m")):
        sigma = float(row["sigma_m"])
        if sigma < 0.0:
            raise ValueError("sigma_m must be non-negative")
        return np.eye(3) * sigma**2
    return None


def _read_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("hypotheses", data.get("rows", []))
        if not isinstance(data, list):
            raise TypeError("JSON hypothesis input must be a list or contain 'hypotheses'")
        return [dict(row) for row in data]
    with p.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _required_float(row: dict, *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if _has_value(value):
            return float(value)
    raise ValueError(f"missing required numeric field; expected one of {keys}")


def _has_value(value: object) -> bool:
    return value is not None and str(value).strip() != ""


def _worst_sigma(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(float(eigenvalues[-1]), 0.0)))
