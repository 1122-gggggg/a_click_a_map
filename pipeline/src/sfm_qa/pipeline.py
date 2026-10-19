"""One-command map screening and localization attribution."""

from __future__ import annotations

import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from mapdoctor.adapters import get_adapter
from mapdoctor.benchmark import load_localization_results, summarize_benchmark
from mapdoctor.config import load_settings
from mapdoctor.diagnostics.graph import analyze_covisibility_fragility
from mapdoctor.metrics import analyze as analyze_map_metrics
from mapdoctor.model import MapModel
from mapdoctor.scoring import score
from sfm_diagnosis.diagnose import DiagnosisCode, diagnose_pose
from sfm_diagnosis.logs import LocalizationHistory
from sfm_diagnosis.models import MapData, Pose
from sfm_diagnosis.report import map_health_summary
from sfm_diagnosis.evidence import load_build_evidence
from sfm_diagnosis.weak_regions import analyze_weak_regions

from .bridge import map_model_to_map_data, mapdoctor_rows_to_history_rows, nearest_mapping_rotation
from .relative_quality import percentile_ranks, weighted_observed_score

MAP_LIMITED_CODES = {
    DiagnosisCode.DATA_SPARSE.value,
    DiagnosisCode.GEOMETRY_WEAK.value,
    DiagnosisCode.QUERY_PARALLAX_WEAK.value,
    DiagnosisCode.OBSERVATION_SCALE_WEAK.value,
    DiagnosisCode.VIEW_COVERAGE_WEAK.value,
}
LOCALIZER_CODES = {
    DiagnosisCode.RETRIEVAL_WEAK.value,
    DiagnosisCode.MATCHING_WEAK.value,
    DiagnosisCode.LANDMARK_MATCHABILITY_WEAK.value,
}
ALIAS_CODES = {
    DiagnosisCode.REFERENCE_DISAGREEMENT.value,
    DiagnosisCode.PERCEPTUAL_ALIASING_SUSPECTED.value,
    DiagnosisCode.REFERENCE_OBSERVABILITY_WEAK.value,
    DiagnosisCode.REFERENCE_EVIDENCE_INSUFFICIENT.value,
}

_PASS_STATUSES = frozenset(
    {"READY", "READY_WITH_MAP_WARNINGS", "MAP_SCREENED_LOCALIZATION_UNCHECKED"}
)


def _relative_localization_quality(results, strict_by_query: dict[str, bool]) -> tuple[dict, dict]:
    """Rank the current query cohort and expose every risk--coverage tradeoff."""

    metric_specs = {
        "inliers": ({result.query: result.inliers for result in results}, True, 0.20),
        "inlier_ratio": (
            {result.query: result.inlier_ratio for result in results},
            True,
            0.15,
        ),
        "reprojection": (
            {result.query: result.reproj_p90_px for result in results},
            False,
            0.15,
        ),
        "hull_coverage": (
            {result.query: result.hull_coverage for result in results},
            True,
            0.10,
        ),
        "grid_coverage": (
            {result.query: result.grid4_occupancy for result in results},
            True,
            0.10,
        ),
        "positive_depth": (
            {result.query: result.positive_depth_ratio for result in results},
            True,
            0.05,
        ),
        "pose_consensus": (
            {result.query: result.pose_consensus for result in results},
            True,
            0.10,
        ),
    }
    ranks = {
        name: percentile_ranks(values, higher_is_better=higher)
        for name, (values, higher, _) in metric_specs.items()
    }
    weights = {name: weight for name, (_, _, weight) in metric_specs.items()}
    details: dict[str, dict] = {}
    for result in results:
        values = {name: ranks[name].get(result.query) for name in metric_specs}
        quality, completeness = weighted_observed_score(values, weights)
        quality *= 0.70 + 0.30 * completeness
        details[result.query] = {
            "relative_quality_score": float(quality),
            "relative_risk_score": float(1.0 - quality),
            "relative_evidence_completeness": float(completeness),
            "relative_metric_ranks": values,
        }

    quality_ranks = percentile_ranks(
        {query: row["relative_quality_score"] for query, row in details.items()}
    )
    for query, rank in quality_ranks.items():
        details[query]["relative_quality_rank"] = rank

    ordered = sorted(
        results,
        key=lambda result: (
            details[result.query]["relative_quality_score"],
            result.query,
        ),
        reverse=True,
    )
    curve: list[dict] = []
    accepted: list = []
    index = 0
    while index < len(ordered):
        threshold = details[ordered[index].query]["relative_quality_score"]
        end = index + 1
        while (
            end < len(ordered)
            and math.isclose(
                details[ordered[end].query]["relative_quality_score"],
                threshold,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
        ):
            end += 1
        accepted.extend(ordered[index:end])
        count = len(accepted)
        curve.append(
            {
                "accepted": count,
                "coverage": count / len(ordered),
                "minimum_relative_quality": float(threshold),
                "mean_relative_quality": float(
                    sum(details[item.query]["relative_quality_score"] for item in accepted)
                    / count
                ),
                "raw_success_rate": sum(item.success for item in accepted) / count,
                "strict_success_rate": (
                    sum(strict_by_query[item.query] for item in accepted) / count
                ),
            }
        )
        index = end

    scores = [row["relative_quality_score"] for row in details.values()]
    summary = {
        "selection_mode": "RELATIVE_RISK_COVERAGE",
        "calibrated_failure_probability": False,
        "ranking": [result.query for result in ordered],
        "coverage_curve": curve,
        "quality_p10": float(np.percentile(scores, 10)) if scores else None,
        "quality_median": float(np.median(scores)) if scores else None,
        "quality_p90": float(np.percentile(scores, 90)) if scores else None,
        "note": (
            "Cohort-relative scores prioritize diagnosis and expose availability/quality "
            "tradeoffs. The log is not proven held out here; external immutable split "
            "evidence remains release authority."
        ),
    }
    return summary, details


def jsonable(value: Any) -> Any:
    """Convert numpy / path / enum values so ``json.dumps`` can emit the report."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def attribute_query(strict_pass: bool, diagnosis_primary: str | None) -> str:
    """Map a gate result plus optional pose-diagnosis primary code onto one label."""
    if strict_pass:
        return "OK"
    if diagnosis_primary is None:
        return "UNATTRIBUTED"
    if diagnosis_primary in MAP_LIMITED_CODES:
        return "MAP_LIMITED"
    if diagnosis_primary in ALIAS_CODES or diagnosis_primary == DiagnosisCode.PNP_DEGENERATE.value:
        return "ALIAS_OR_PNP"
    if diagnosis_primary in LOCALIZER_CODES or diagnosis_primary == DiagnosisCode.HEALTHY.value:
        return "LOCALIZER_LIMITED"
    return "UNATTRIBUTED"


def _load_model(model_path: str | Path, backend: str, model: MapModel | None) -> MapModel:
    if model is not None:
        return model
    return get_adapter(backend).load(model_path)


def _bounded_health(metrics_dict: dict) -> dict:
    health = dict(metrics_dict)
    weak = list(health.get("weak_images") or [])
    health["weak_images_count"] = len(weak)
    health["weak_images"] = weak[:20]
    health["recapture_suggestions"] = list(health.get("recapture_suggestions") or [])[:20]
    return health


def _has_finite_xyz(result) -> bool:
    return (
        result.x is not None
        and result.y is not None
        and result.z is not None
        and math.isfinite(result.x)
        and math.isfinite(result.y)
        and math.isfinite(result.z)
    )


_ATTRIBUTION_ACTION = {
    "MAP_LIMITED": (
        "MAP_REPAIR: inspect weak-region coverage and geometry, then test repairs using "
        "existing images before collecting new data."
    ),
    "LOCALIZER_LIMITED": (
        "LOCALIZER_FIX: trace retrieval, matching, 2D-3D lifting, and PnP on the existing map."
    ),
    "ALIAS_OR_PNP": (
        "REFERENCE_GEOMETRY_VERIFY: inspect independent reference hypotheses, calibration, "
        "and repeated-structure aliasing before changing the map."
    ),
    "UNATTRIBUTED": (
        "FAILURE_FUNNEL_TRACE: instrument retrieval, matching, 2D-3D lifting, and PnP to "
        "identify the earliest failing stage."
    ),
}

_GATE_ACTION = {
    "localization_failed": "Replay retrieval through PnP and record the earliest failed stage.",
    "low_inliers": "Run targeted strong matching and rebuild unique 2D-3D support.",
    "low_inlier_ratio": "Tighten geometric verification and quarantine ambiguous matches.",
    "missing_reprojection_error": "Instrument reprojection residuals before accepting a pose.",
    "high_reprojection_error": "Prune inconsistent correspondences and re-estimate the pose.",
    "low_inlier_hull_coverage": "Balance correspondences across the image before rerunning PnP.",
    "low_grid_occupancy": "Expand reference/correspondence coverage into missing image cells.",
    "low_positive_depth_ratio": "Check calibration, 2D-3D associations, and pose-hypothesis degeneracy.",
    "low_pose_consensus": "Keep per-reference poses separate and reject incompatible hypotheses.",
}


def _dedupe_strings(values) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _query_recommendations(
    gate_failures: list[str],
    attribution: str,
    diagnosis_recommendations: list[str],
    *,
    has_xyz: bool,
) -> list[str]:
    actions = list(diagnosis_recommendations)
    if attribution == "OK":
        if not has_xyz:
            actions.append(
                "Strict gates passed, but finite query pose/location evidence was not "
                "provided for pose-local root-cause diagnosis."
            )
        return _dedupe_strings(actions)
    actions.append(_ATTRIBUTION_ACTION.get(attribution, _ATTRIBUTION_ACTION["UNATTRIBUTED"]))
    actions.extend(_GATE_ACTION.get(failure, f"Resolve strict gate failure: {failure}.") for failure in gate_failures)
    if not has_xyz:
        actions.append(
            "Provide finite query pose/location evidence so map-limited and localizer-limited "
            "failures can be separated."
        )
    return _dedupe_strings(actions)


def _resolution_plan(
    gate_failures: list[str],
    attribution: str,
    recommendations: list[str],
) -> dict[str, Any]:
    map_limited = attribution == "MAP_LIMITED"
    deferred_capture = [
        item
        for item in recommendations
        if "recapture" in item.lower() or "capture a " in item.lower()
    ]
    actions = [item for item in recommendations if item not in deferred_capture]
    actions.append(
        "COUNTERFACTUAL_VALIDATE: replay the intervention on frozen weak and stable "
        "query blocks before accepting it."
    )
    if map_limited:
        actions.extend(
            f"DEFERRED_CAPTURE_HYPOTHESIS: {item}" for item in deferred_capture
        )
        actions.append(
            "RECAPTURE_CONDITIONAL_AFTER_COUNTERFACTUAL: request targeted capture only "
            "if existing-data repairs leave a measured structural deficit."
        )
    priority = {
        "MAP_LIMITED": "MAP_COUNTERFACTUAL",
        "LOCALIZER_LIMITED": "LOCALIZER_FIX",
        "ALIAS_OR_PNP": "REFERENCE_GEOMETRY_VERIFY",
    }.get(attribution, "EXISTING_DATA_TRIAGE")
    required_stages = {
        "MAP_LIMITED": ["map_support", "geometry", "retrieval", "matching", "pnp"],
        "LOCALIZER_LIMITED": ["retrieval", "matching", "pnp"],
        "ALIAS_OR_PNP": ["reference_geometry", "geometric_verification", "pnp"],
    }.get(attribution, ["retrieval", "matching", "pnp", "map_support"])
    blocked_by = [
        "existing_data_counterfactual_complete",
        "heldout_provenance_verified",
        "stable_holdout_comparison",
    ]
    if map_limited:
        blocked_by.append("structural_deficit_evidence")
    return {
        "schema_version": 1,
        "policy": "EXISTING_DATA_FIRST",
        "priority": priority,
        "authorization_status": "NOT_AUTHORIZED",
        "counterfactual_status": "REQUIRED_NOT_RUN",
        "required_stages": required_stages,
        "counterfactual_trials": [],
        "counterfactual_result": None,
        "actions": _dedupe_strings(actions),
        "failed_gate_checks": list(gate_failures),
        "counterfactual_required_before_recapture": True,
        "recapture_policy": (
            "NOT_AUTHORIZED_PENDING_COUNTERFACTUAL"
            if map_limited
            else "NOT_AUTHORIZED_BY_QUERY_DIAGNOSIS"
        ),
        "blocked_by": blocked_by,
        "validation_contract": {
            "required_checks": [
                "rerun the same failed query block with frozen map/query/config identities",
                "close every reported strict gate failure",
                "confirm the stable holdout does not regress",
            ],
            "pass_condition": (
                "the weak query block improves and the frozen stable holdout remains non-regressed"
            ),
        },
    }


def _map_integrity_checks(model: MapModel, metrics) -> dict[str, bool]:
    def finite(values) -> bool:
        return all(math.isfinite(float(value)) for value in values)

    def track_element_is_linked(point, element) -> bool:
        image = model.images.get(element.image_id)
        if image is None or not 0 <= element.point2d_idx < len(image.observations):
            return False
        return image.observations[element.point2d_idx].point3d_id == point.id

    return {
        "has_registered_images": metrics.registered_images > 0,
        "has_cameras": metrics.cameras > 0,
        "has_points3d": metrics.points3d > 0,
        "has_observations": metrics.observations > 0,
        "all_images_have_known_cameras": all(
            image.camera_id in model.cameras for image in model.images.values()
        ),
        "camera_models_are_finite": all(
            camera.width > 0
            and camera.height > 0
            and bool(camera.params)
            and finite(camera.params)
            for camera in model.cameras.values()
        ),
        "image_poses_are_finite": all(
            finite(image.center)
            and finite(image.viewing_direction)
            and sum(value * value for value in image.viewing_direction) > 0.0
            for image in model.images.values()
        ),
        "points_are_finite": all(
            finite(point.xyz)
            and math.isfinite(float(point.error))
            and point.error >= 0.0
            for point in model.points3d.values()
        ),
        "observations_are_finite_and_linked": all(
            math.isfinite(float(observation.x))
            and math.isfinite(float(observation.y))
            and (
                observation.point3d_id is None
                or observation.point3d_id in model.points3d
            )
            for image in model.images.values()
            for observation in image.observations
        ),
        "tracks_reference_known_images": all(
            track_element_is_linked(point, element)
            for point in model.points3d.values()
            for element in point.track
        ),
    }


def check_map(
    model_path: str | Path,
    *,
    backend: str,
    config_path: str | Path | None = None,
    model: MapModel | None = None,
    map_data: MapData | None = None,
    database: str | Path | None = None,
    pairs: str | Path | None = None,
    images_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> dict:
    """Screen reconstruction health with MapDoctor and sfm-diagnosis map-only tools."""
    settings = load_settings(config_path)
    model = _load_model(model_path, backend, model)
    if map_data is None:
        map_data = map_model_to_map_data(model)
    metrics = analyze_map_metrics(model, settings.health)
    readiness = score(metrics, settings.health)
    fragility = analyze_covisibility_fragility(model)
    health_summary = map_health_summary(map_data)
    evidence = None
    if any(item is not None for item in (database, pairs, images_manifest, images_dir)):
        evidence = load_build_evidence(
            map_data,
            database=database,
            pairs=pairs,
            images_manifest=images_manifest,
            images_dir=images_dir,
        )
    weak = analyze_weak_regions(map_data, evidence=evidence)
    integrity_checks = _map_integrity_checks(model, metrics)
    map_integrity_ok = all(integrity_checks.values())
    map_ok = map_integrity_ok and all(
        check["pass"] for check in readiness.checks.values()
    )
    map_status = "PASS" if map_ok else "INVALID" if not map_integrity_ok else "FAIL"
    weak_summary = weak.summary
    return jsonable(
        {
            "backend": backend,
            "source": model.source,
            "format": model.format,
            "readiness": {
                "score": readiness.score,
                "grade": readiness.grade,
                "checks": readiness.checks,
                "integrity_checks": integrity_checks,
                "map_integrity_ok": map_integrity_ok,
                "map_ok": map_ok,
                "map_status": map_status,
            },
            "health": _bounded_health(metrics.to_dict()),
            "graph": fragility.to_dict(),
            "reconstruction": {
                "map_health": health_summary,
                "diagnostic_mode": weak_summary.get("diagnostic_mode"),
                "num_weak_images": weak_summary.get("num_weak_images"),
                "num_weak_regions": weak_summary.get("num_weak_regions"),
                "cause_counts": weak_summary.get("cause_counts", {}),
                "regions": list(weak.as_dict().get("regions") or [])[:20],
            },
        }
    )


def check_localize(
    model_path: str | Path,
    logs_path: str | Path,
    *,
    backend: str,
    config_path: str | Path | None = None,
    map_data: MapData | None = None,
    model: MapModel | None = None,
) -> dict:
    """Score localization logs and attribute failures to map vs localizer vs viewpoint."""
    settings = load_settings(config_path)
    if map_data is None:
        map_data = map_model_to_map_data(_load_model(model_path, backend, model))
    results = load_localization_results(logs_path)
    summary = summarize_benchmark(results, settings.localization)
    history_rows = mapdoctor_rows_to_history_rows(results)
    history = LocalizationHistory(history_rows) if history_rows else None
    strict_by_query = {
        result.query: result.passes(settings.localization) for result in results
    }
    relative_summary, relative_details = _relative_localization_quality(
        results, strict_by_query
    )

    queries = []
    for result in results:
        strict_pass = strict_by_query[result.query]
        gate_failures = result.failures(settings.localization)
        diagnosis_primary = None
        diagnosis_codes: list[str] = []
        diagnosis_recommendations: list[str] = []
        has_xyz = _has_finite_xyz(result)
        if has_xyz:
            center = (float(result.x), float(result.y), float(result.z))
            pose = Pose(center, nearest_mapping_rotation(map_data, center))
            diagnosis = diagnose_pose(map_data, pose, history=history)
            diagnosis_primary = diagnosis.primary.value
            diagnosis_codes = [code.value for code in diagnosis.codes]
            diagnosis_recommendations = list(diagnosis.recommendations)
        attribution = attribute_query(strict_pass, diagnosis_primary)
        diagnosis_recommendations = _query_recommendations(
            gate_failures,
            attribution,
            diagnosis_recommendations,
            has_xyz=has_xyz,
        )
        resolution_plan = (
            None
            if strict_pass
            else _resolution_plan(
                gate_failures,
                attribution,
                diagnosis_recommendations,
            )
        )
        queries.append(
            {
                "query": result.query,
                "strict_pass": strict_pass,
                "gate_failures": gate_failures,
                "attribution": attribution,
                "diagnosis_primary": diagnosis_primary,
                "diagnosis_codes": diagnosis_codes,
                "pose_evidence_status": (
                    "HYPOTHESIS_ONLY" if has_xyz else "UNAVAILABLE"
                ),
                "diagnosis_recommendations": diagnosis_recommendations,
                "resolution_plan": resolution_plan,
                **relative_details[result.query],
            }
        )

    required_rate = float(settings.localization.min_strict_success_rate)
    target = Fraction(str(required_rate))
    required_successes = (
        target.numerator * len(queries) + target.denominator - 1
    ) // target.denominator
    strict_successes = sum(row["strict_pass"] for row in queries)
    localization_ok = bool(queries) and strict_successes >= required_successes
    counts = Counter(row["attribution"] for row in queries)
    return jsonable(
        {
            "total": summary.total_queries,
            "strict_success_rate": summary.strict_success_rate,
            "strict_successes": strict_successes,
            "required_strict_success_rate": required_rate,
            "required_strict_successes": required_successes,
            "localization_ok": localization_ok,
            "localization_status": "PASS" if localization_ok else "FAIL",
            "attribution_counts": dict(sorted(counts.items())),
            "relative_quality": relative_summary,
            "heldout_provenance_verified": False,
            "evaluation_provenance": "UNVERIFIED_PROVIDED_LOG",
            "queries": queries,
        }
    )


def _overall_status(
    map_ok: bool,
    loc: dict | None,
    *,
    map_integrity_ok: bool = True,
) -> str:
    if not map_integrity_ok:
        if loc is not None and not bool(loc["localization_ok"]):
            return "BOTH_FAILED"
        return "MAP_SCREENING_FAILED"
    if loc is None:
        return "MAP_SCREENED_LOCALIZATION_UNCHECKED" if map_ok else "MAP_SCREENING_FAILED"
    loc_ok = bool(loc["localization_ok"])
    if map_ok and loc_ok:
        return "READY"
    if map_ok and not loc_ok:
        return "LOCALIZATION_FAILED"
    if not map_ok and loc_ok:
        return "READY_WITH_MAP_WARNINGS"
    return "BOTH_FAILED"


def analyze(
    model_path: str | Path,
    *,
    backend: str,
    logs_path: str | Path | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    database: str | Path | None = None,
    pairs: str | Path | None = None,
    images_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> dict:
    """Diagnose the map first, then optionally attribute localization logs."""
    model = get_adapter(backend).load(model_path)
    map_data = map_model_to_map_data(model)
    map_report = check_map(
        model_path,
        backend=backend,
        config_path=config_path,
        model=model,
        map_data=map_data,
        database=database,
        pairs=pairs,
        images_manifest=images_manifest,
        images_dir=images_dir,
    )
    loc_report = None
    if logs_path is not None:
        loc_report = check_localize(
            model_path,
            logs_path,
            backend=backend,
            config_path=config_path,
            map_data=map_data,
            model=model,
        )
    payload = jsonable(
        {
            "overall_status": _overall_status(
                map_report["readiness"]["map_ok"],
                loc_report,
                map_integrity_ok=map_report["readiness"]["map_integrity_ok"],
            ),
            "map": map_report,
            "localization": loc_report,
        }
    )
    if output_dir is not None:
        out = Path(output_dir)
        map_dir = out / "map"
        map_dir.mkdir(parents=True, exist_ok=True)
        (map_dir / "report.json").write_text(
            json.dumps(map_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if loc_report is not None:
            sfm_dir = out / "sfm"
            sfm_dir.mkdir(parents=True, exist_ok=True)
            (sfm_dir / "report.json").write_text(
                json.dumps(loc_report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        (out / "report.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return payload


def check(
    model_path: str | Path,
    *,
    backend: str,
    logs_path: str | Path | None = None,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    database: str | Path | None = None,
    pairs: str | Path | None = None,
    images_manifest: str | Path | None = None,
    images_dir: str | Path | None = None,
) -> dict:
    """Compatibility alias for ``analyze``."""
    return analyze(
        model_path,
        backend=backend,
        logs_path=logs_path,
        config_path=config_path,
        output_dir=output_dir,
        database=database,
        pairs=pairs,
        images_manifest=images_manifest,
        images_dir=images_dir,
    )


def is_success_status(overall_status: str) -> bool:
    return overall_status in _PASS_STATUSES
