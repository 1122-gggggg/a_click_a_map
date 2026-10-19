"""Group-safe multivariate calibration of EDM localization failure.

Implementation follows the official scikit-learn estimator/split contracts:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mapdoctor.diagnostics.calibration import calibration_metrics


@dataclass(frozen=True)
class RiskTrainingSample:
    query_id: str
    group: str
    failure: bool
    features: Mapping[str, float | int | None]


@dataclass(frozen=True)
class RiskCalibrationConfig:
    model_type: str = "logistic"
    splitter: str = "leave_one_group_out"
    folds: int = 5
    target_recall: float = 0.95
    max_fnr: float | None = None
    random_state: int = 0
    min_roc_auc: float = 0.6

    def __post_init__(self) -> None:
        if self.model_type not in {"logistic", "hist_gradient_boosting"}:
            raise ValueError("model_type must be logistic or hist_gradient_boosting")
        if self.splitter not in {"group_kfold", "leave_one_group_out"}:
            raise ValueError("splitter must be group_kfold or leave_one_group_out")
        if self.folds < 2:
            raise ValueError("folds must be >= 2")
        if not 0.0 < self.target_recall <= 1.0:
            raise ValueError("target_recall must be in (0, 1]")
        if self.max_fnr is not None and not 0.0 <= self.max_fnr < 1.0:
            raise ValueError("max_fnr must be in [0, 1)")
        if not 0.0 <= self.min_roc_auc <= 1.0:
            raise ValueError("min_roc_auc must be in [0, 1]")

    @property
    def resolved_target_recall(self) -> float:
        return 1.0 - self.max_fnr if self.max_fnr is not None else self.target_recall


class FittedRiskCalibrator:
    def __init__(self, *, estimator: Any, feature_names: tuple[str, ...], model_type: str):
        self.estimator = estimator
        self.feature_names = feature_names
        self.model_type = model_type

    def predict_proba(self, features: Sequence[Mapping[str, float | int | None]]) -> np.ndarray:
        matrix = _feature_matrix(features, self.feature_names)
        probabilities = np.asarray(self.estimator.predict_proba(matrix), dtype=float)
        return probabilities[:, 1]


@dataclass(frozen=True)
class RiskCalibrationResult:
    config: RiskCalibrationConfig
    calibrator: FittedRiskCalibrator
    feature_names: tuple[str, ...]
    threshold: float
    metrics: dict[str, float | str]
    out_of_fold_probabilities: dict[str, float]
    folds: tuple[dict[str, Any], ...]
    group_leakage_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "EDM_RISK_CALIBRATION",
            "model_type": self.config.model_type,
            "splitter": self.config.splitter,
            "feature_names": list(self.feature_names),
            "threshold": self.threshold,
            "target_recall": self.config.resolved_target_recall,
            "metrics": self.metrics,
            "out_of_fold_probabilities": self.out_of_fold_probabilities,
            "folds": list(self.folds),
            "group_leakage_detected": self.group_leakage_detected,
            "evaluation_contract": (
                "Metrics use out-of-fold predictions with session/video-disjoint groups. "
                "The final fitted estimator is only for future untouched samples."
            ),
        }


def probability_status_from_calibration(
    metrics: Mapping[str, Any],
    *,
    min_roc_auc: float = 0.6,
) -> str:
    """Refuse the calibrated label when ranking is no better than a weak floor."""

    try:
        roc = float(metrics.get("roc_auc"))
    except (TypeError, ValueError):
        return "UNCALIBRATED_EDM_LOO"
    if not math.isfinite(roc) or roc < float(min_roc_auc):
        return "UNCALIBRATED_EDM_LOO"
    return "CALIBRATED_EDM_LOO"


class RiskCalibrator:
    def __init__(self, config: RiskCalibrationConfig | None = None):
        self.config = config or RiskCalibrationConfig()

    def fit(self, samples: Sequence[RiskTrainingSample]) -> RiskCalibrationResult:
        if not samples:
            raise ValueError("risk calibration requires samples")
        query_ids = [sample.query_id for sample in samples]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("risk calibration query_id values must be unique")
        groups = np.asarray([sample.group for sample in samples], dtype=object)
        if any(not str(group).strip() for group in groups):
            raise ValueError("risk calibration groups must be non-empty")
        labels = np.asarray([int(sample.failure) for sample in samples], dtype=int)
        if len(np.unique(labels)) < 2:
            raise ValueError("risk calibration requires success and failure outcomes")
        candidate_names = tuple(
            sorted({name for sample in samples for name in sample.features})
        )
        if not candidate_names:
            raise ValueError("risk calibration requires at least one feature")
        candidate_matrix = _feature_matrix(
            [sample.features for sample in samples], candidate_names
        )
        keep = np.any(np.isfinite(candidate_matrix), axis=0)
        feature_names = tuple(
            name for name, available in zip(candidate_names, keep) if available
        )
        if not feature_names:
            raise ValueError("risk calibration has no observed numeric features")
        matrix = candidate_matrix[:, keep]
        splits = list(_splits(matrix, labels, groups, self.config))
        oof = np.full(len(samples), np.nan, dtype=float)
        fold_rows = []
        leakage = False
        for fold, (train, test) in enumerate(splits):
            train_groups = sorted(set(groups[train].tolist()))
            test_groups = sorted(set(groups[test].tolist()))
            overlap = sorted(set(train_groups).intersection(test_groups))
            leakage |= bool(overlap)
            if overlap:
                raise RuntimeError(f"group leakage in fold {fold}: {overlap}")
            if len(np.unique(labels[train])) < 2:
                raise ValueError(
                    f"fold {fold} training data contains only one outcome class"
                )
            estimator = _estimator(self.config)
            estimator.fit(matrix[train], labels[train])
            oof[test] = np.asarray(estimator.predict_proba(matrix[test]))[:, 1]
            fold_rows.append(
                {
                    "fold": fold,
                    "train_groups": train_groups,
                    "test_groups": test_groups,
                    "train_count": int(len(train)),
                    "test_count": int(len(test)),
                }
            )
        if np.any(~np.isfinite(oof)):
            raise RuntimeError("cross-validation did not predict every sample")
        threshold = _threshold_for_recall(
            labels, oof, target_recall=self.config.resolved_target_recall
        )
        metrics = _metrics(labels, oof, threshold)
        final_estimator = _estimator(self.config)
        final_estimator.fit(matrix, labels)
        return RiskCalibrationResult(
            config=self.config,
            calibrator=FittedRiskCalibrator(
                estimator=final_estimator,
                feature_names=feature_names,
                model_type=self.config.model_type,
            ),
            feature_names=feature_names,
            threshold=threshold,
            metrics=metrics,
            out_of_fold_probabilities={
                query_id: float(probability)
                for query_id, probability in zip(query_ids, oof)
            },
            folds=tuple(fold_rows),
            group_leakage_detected=leakage,
        )


def _feature_matrix(
    rows: Sequence[Mapping[str, float | int | None]],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    matrix = np.full((len(rows), len(feature_names)), np.nan, dtype=float)
    for row_index, row in enumerate(rows):
        for column, name in enumerate(feature_names):
            value = row.get(name)
            if value is None or isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                matrix[row_index, column] = number
    return matrix


def _estimator(config: RiskCalibrationConfig):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Install the risk extra: pip install -e '.[risk]'"
        ) from exc
    if config.model_type == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=config.random_state,
            ),
        )
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        HistGradientBoostingClassifier(
            max_iter=200,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=config.random_state,
        ),
    )


def _splits(matrix, labels, groups, config: RiskCalibrationConfig):
    try:
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    except ImportError as exc:
        raise RuntimeError("Install the risk extra: pip install -e '.[risk]'") from exc
    unique_groups = np.unique(groups)
    if config.splitter == "leave_one_group_out":
        return LeaveOneGroupOut().split(matrix, labels, groups)
    if config.folds > len(unique_groups):
        raise ValueError(
            f"folds={config.folds} exceeds independent groups={len(unique_groups)}"
        )
    return GroupKFold(n_splits=config.folds).split(matrix, labels, groups)


def _threshold_for_recall(
    labels: np.ndarray, probabilities: np.ndarray, *, target_recall: float
) -> float:
    positives = int(np.sum(labels == 1))
    if positives == 0:
        raise ValueError("target-recall threshold requires failure examples")
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    feasible = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        recall = float(np.sum(predictions & (labels == 1)) / positives)
        if recall + 1e-12 >= target_recall:
            feasible.append(float(threshold))
    return max(feasible) if feasible else 0.0


def _metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | str]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("Install the risk extra: pip install -e '.[risk]'") from exc
    predictions = probabilities >= threshold
    recall = float(recall_score(labels, predictions, zero_division=0))
    calibration = calibration_metrics(probabilities, labels, bins=10)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "pr_metric": "average_precision",
        "brier_score": calibration["brier"],
        "ece": calibration["adaptive_ece"],
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_negative_rate": 1.0 - recall,
        "threshold": float(threshold),
    }
