from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from pathlib import Path

from sfm_diagnosis.site_pipeline.deployment_localizer import (
    DEFAULT_THRESHOLDS,
    _configure_audited_runtime_site_packages,
    _prepare_official_pair,
    _resolve_audited_site_packages,
    build_localization_payload,
    evaluate_localization_admission,
    localization_is_strong,
    rank_reference_indices,
    scaled_pinhole_parameters,
    subset_reference_identities,
)


def test_configure_audited_runtime_updates_megaloc_and_edm_admission(tmp_path, monkeypatch) -> None:
    audited = tmp_path / "site-packages"
    audited.mkdir()
    package = ModuleType("river_map_quality")
    package.__path__ = []
    historical = ModuleType("river_map_quality.historical_experiment")
    official = ModuleType("river_map_quality.official_edm_adapter_loo")
    historical.AUDITED_TORCH_SITE_PACKAGES = tmp_path / "retired"
    official.AUDITED_EDM_SITE_PACKAGES = tmp_path / "retired"
    monkeypatch.setitem(sys.modules, "river_map_quality", package)
    monkeypatch.setitem(sys.modules, "river_map_quality.historical_experiment", historical)
    monkeypatch.setitem(sys.modules, "river_map_quality.official_edm_adapter_loo", official)

    configured = _configure_audited_runtime_site_packages(audited)

    assert configured == audited
    assert historical.AUDITED_TORCH_SITE_PACKAGES == audited
    assert official.AUDITED_EDM_SITE_PACKAGES == audited


def test_explicit_audited_site_packages_overrides_missing_prefix_path(tmp_path) -> None:
    explicit = tmp_path / "audited"
    explicit.mkdir()

    assert _resolve_audited_site_packages(str(explicit), prefix=tmp_path / "missing") == explicit
    assert _resolve_audited_site_packages(None, prefix=tmp_path / "missing") == (
        tmp_path
        / "missing"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def test_scaled_pinhole_parameters_share_intrinsics_by_resolution() -> None:
    calibration = {
        "image_width": 1280,
        "image_height": 720,
        "K": [[960.0, 0.0, 640.0], [0.0, 958.0, 360.0], [0.0, 0.0, 1.0]],
        "images_are_undistorted": True,
    }

    assert scaled_pinhole_parameters(calibration, width=1920, height=1080) == (
        1440.0,
        1437.0,
        960.0,
        540.0,
    )


def test_rank_reference_indices_excludes_heldout_session_before_top_k() -> None:
    references = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    indices, fallback = rank_reference_indices(
        references,
        query,
        reference_sessions=("holdout", "map-a", "map-b"),
        excluded_sessions=frozenset({"holdout"}),
        top_k=2,
    )

    assert indices == (1, 2)
    assert fallback is False


def test_rank_reference_indices_prefers_side_looking_views() -> None:
    references = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.5, 0.5]], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    indices, fallback = rank_reference_indices(
        references,
        query,
        reference_sessions=("a", "b", "c"),
        excluded_sessions=frozenset(),
        top_k=1,
        occupied_bins=(2, 80, 70),
        min_occupied_bins=20,
        retrieve_pool=3,
    )

    assert indices == (1,)
    assert fallback is False


def test_rank_reference_indices_scans_full_ranking_for_side_looking_views() -> None:
    references = np.asarray(
        [
            [1.00, 0.00],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.50, 0.50],
        ],
        dtype=np.float32,
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    indices, fallback = rank_reference_indices(
        references,
        query,
        reference_sessions=("a", "b", "c", "d", "e"),
        excluded_sessions=frozenset(),
        top_k=1,
        occupied_bins=(2, 2, 2, 2, 80),
        min_occupied_bins=20,
        retrieve_pool=3,
    )

    assert indices == (4,)
    assert fallback is False


def test_subset_reference_identities_keeps_occupancy_aligned() -> None:
    names, sessions, paths, occupied, observations = subset_reference_identities(
        names=("empty.jpg", "side.jpg", "sky.jpg"),
        sessions=("v1", "v2", "v3"),
        paths=(Path("/a"), Path("/b"), Path("/c")),
        occupied=(5, 80, 3),
        observations={
            "empty.jpg": (np.zeros((1, 2)), np.zeros(1, dtype=np.int64)),
            "side.jpg": (np.ones((2, 2)), np.arange(2, dtype=np.int64)),
            "sky.jpg": (np.zeros((1, 2)), np.zeros(1, dtype=np.int64)),
        },
        selected_names=("side.jpg", "empty.jpg"),
    )
    assert names == ("side.jpg", "empty.jpg")
    assert sessions == ("v2", "v1")
    assert occupied == (80, 5)
    assert list(observations) == ["side.jpg", "empty.jpg"]


def test_localization_is_strong_requires_all_geometry_gates() -> None:
    metrics = {
        "inlier_count": 100,
        "inlier_ratio": 0.5,
        "convex_hull_coverage": 0.2,
        "occupancy_4x4": 8,
        "positive_depth_ratio": 1.0,
        "reprojection_p90": 2.0,
    }
    thresholds = {
        "strong_inliers": 80,
        "minimum_inlier_ratio": 0.25,
        "minimum_hull_coverage": 0.15,
        "minimum_occupancy_4x4": 6,
        "minimum_positive_depth_ratio": 0.99,
        "maximum_reprojection_p90_px": 3.0,
    }

    assert localization_is_strong(metrics, decision_status="ACCEPT", thresholds=thresholds)
    assert not localization_is_strong(
        {**metrics, "convex_hull_coverage": 0.1},
        decision_status="ACCEPT",
        thresholds=thresholds,
    )
    assert not localization_is_strong(
        metrics,
        decision_status="REJECT_MULTIMODAL",
        thresholds=thresholds,
    )


def test_official_edm_pair_preparation_uses_each_native_resolution(tmp_path) -> None:
    import cv2

    query = tmp_path / "query.jpg"
    reference = tmp_path / "reference.jpg"
    assert cv2.imwrite(str(query), np.zeros((9, 16), dtype=np.uint8))
    assert cv2.imwrite(str(reference), np.zeros((18, 32), dtype=np.uint8))
    calls = []

    def prepare(root, name, runtime, *, native_width, native_height):
        calls.append((root / name, runtime, native_width, native_height))
        return (native_width, native_height)

    prepared = _prepare_official_pair(query, reference, object(), prepare_image=prepare)

    assert prepared == ((16, 9), (32, 18))
    assert [(row[2], row[3]) for row in calls] == [(16, 9), (32, 18)]


def _strong_metrics() -> dict:
    return {
        "inlier_count": 100,
        "inlier_ratio": 0.5,
        "convex_hull_coverage": 0.2,
        "occupancy_4x4": 8,
        "occupied_frac_30": 0.2,
        "positive_depth_ratio": 1.0,
        "reprojection_p90": 2.0,
    }


def test_evaluate_admission_does_not_call_empty_metrics_a_viewpoint_fail() -> None:
    success, status, _in_intersection, viewpoint_abstain = evaluate_localization_admission(
        registration_success=False,
        metrics={},
        decision_status="REJECT_PNP_FAILED",
        position=None,
        intersection_cells=np.zeros((0, 3)),
        requested_thresholds=DEFAULT_THRESHOLDS,
    )
    assert success is False
    assert status == "ABSTAINED"
    assert viewpoint_abstain is False


def test_evaluate_admission_abstains_on_empty_query_support() -> None:
    success, status, in_intersection, viewpoint_abstain = evaluate_localization_admission(
        registration_success=True,
        metrics={**_strong_metrics(), "occupied_frac_30": 0.01, "occupancy_4x4": 2, "convex_hull_coverage": 0.04},
        decision_status="ACCEPT",
        position=(0.0, 0.0, 0.0),
        intersection_cells=np.zeros((0, 3)),
        requested_thresholds=DEFAULT_THRESHOLDS,
    )
    assert success is False
    assert status == "ABSTAINED"
    assert viewpoint_abstain is True
    assert in_intersection is False


def test_evaluate_admission_locks_frozen_gates_inside_intersection() -> None:
    weaker = {
        "strong_inliers": 10,
        "minimum_inlier_ratio": 0.05,
        "minimum_hull_coverage": 0.01,
        "minimum_occupancy_4x4": 1,
        "minimum_positive_depth_ratio": 0.5,
        "maximum_reprojection_p90_px": 20.0,
    }
    weak_but_viewable = {
        "inlier_count": 20,
        "inlier_ratio": 0.1,
        "convex_hull_coverage": 0.2,
        "occupancy_4x4": 8,
        "occupied_frac_30": 0.2,
        "positive_depth_ratio": 1.0,
        "reprojection_p90": 2.0,
    }
    success, status, in_intersection, viewpoint_abstain = evaluate_localization_admission(
        registration_success=True,
        metrics=weak_but_viewable,
        decision_status="ACCEPT",
        position=(0.0, 0.0, 0.0),
        intersection_cells=np.array([[0.0, 0.0, 0.0]]),
        requested_thresholds=weaker,
    )
    assert in_intersection is True
    assert viewpoint_abstain is False
    assert success is False
    assert status == "POSE_ESTIMATED_WEAK"


def test_evaluate_admission_accepts_strong_side_looking_pose() -> None:
    success, status, in_intersection, viewpoint_abstain = evaluate_localization_admission(
        registration_success=True,
        metrics=_strong_metrics(),
        decision_status="ACCEPT",
        position=(10.0, 0.0, 0.0),
        intersection_cells=np.array([[0.0, 0.0, 0.0]]),
        requested_thresholds=DEFAULT_THRESHOLDS,
    )
    assert success is True
    assert status == "LOCALIZED_STRONG"
    assert in_intersection is False
    assert viewpoint_abstain is False


def test_build_localization_payload_uses_query_status() -> None:
    result = EDMQueryResult(
        query_id="q",
        session_id="DEPLOYMENT_QUERY",
        timestamp=0.0,
        success=False,
        registration_success=True,
        query_status="ABSTAINED",
        viewpoint_abstain=True,
        in_intersection=True,
    )
    payload = build_localization_payload(
        query="/tmp/query.jpg",
        map_name="river_gluemap_all8_direct_20260831",
        result=result,
        reference_bank="sideview",
    )
    assert payload["status"] == "ABSTAINED"
    assert payload["reference_bank"] == "sideview"
    assert payload["in_intersection"] is True
    assert payload["result"]["loo_mode"] == "none"
