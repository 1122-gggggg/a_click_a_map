from __future__ import annotations

import numpy as np

from sfm_diagnosis.site_pipeline.cli import build_parser
from sfm_diagnosis.site_pipeline.edm_retriangulate import (
    accumulate_cell_keypoints,
    match_rows_for_pair,
    quantize_xy,
    select_retriangulation_pairs,
)


def test_quantize_xy_shares_a_two_pixel_cell() -> None:
    assert quantize_xy((0.05, 0.05), cell=2.0) == quantize_xy((1.95, 0.05), cell=2.0)
    assert quantize_xy((0.05, 0.05), cell=2.0) != quantize_xy((2.15, 0.05), cell=2.0)


def test_quantize_xy_one_pixel_cell_keeps_neighbors_apart() -> None:
    assert quantize_xy((0.05, 0.05), cell=1.0) != quantize_xy((1.05, 0.05), cell=1.0)
    assert quantize_xy((0.05, 0.05), cell=2.0) == quantize_xy((1.05, 0.05), cell=2.0)


def test_accumulate_cell_keypoints_keeps_highest_confidence() -> None:
    pairs = [
        (
            "a.jpg",
            "b.jpg",
            np.array([[0.4, 0.4], [0.6, 0.5]], dtype=np.float64),
            np.array([[10.1, 10.1], [12.0, 10.0]], dtype=np.float64),
            np.array([0.2, 0.9], dtype=np.float64),
        )
    ]
    keypoints, index_of, discarded = accumulate_cell_keypoints(pairs, pose_only=set(), cell_px=2.0)
    assert discarded == 0
    assert keypoints["a.jpg"].shape == (1, 2)
    np.testing.assert_allclose(keypoints["a.jpg"][0], [0.6, 0.5])
    assert (0, 0) in index_of["a.jpg"]


def test_match_rows_for_pair_skips_empty_cells() -> None:
    index_a = {(0, 0): 0}
    index_b = {(5, 5): 0}
    rows = match_rows_for_pair(
        np.array([[0.5, 0.5], [8.0, 8.0]], dtype=np.float64),
        np.array([[10.5, 10.5], [3.0, 3.0]], dtype=np.float64),
        index_a,
        index_b,
        cell_px=2.0,
    )
    assert rows.shape == (1, 2)
    assert tuple(rows[0]) == (0, 0)


def test_select_retriangulation_pairs_adds_temporal_and_covisibility() -> None:
    tracks = {
        "vid/frame_000000.jpg": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        "vid/frame_000001.jpg": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        "vid/frame_000002.jpg": {1, 2, 3},
    }
    keyframes = {
        "vid/frame_000000.jpg": {"frame_index": 0},
        "vid/frame_000001.jpg": {"frame_index": 1},
        "vid/frame_000002.jpg": {"frame_index": 2},
    }
    pairs = select_retriangulation_pairs(
        tracks, keyframes, covis_min_shared=10, covis_top_k=5, temporal_ordinal=2
    )
    reasons = {(row["name_a"], row["name_b"]): row["reason"] for row in pairs}
    covis_pair = ("vid/frame_000000.jpg", "vid/frame_000001.jpg")
    assert covis_pair in reasons
    assert reasons[covis_pair] in {"both", "covisibility"}
    assert any(row["reason"] in {"temporal", "both"} for row in pairs)


def test_cli_exposes_adopted_map_and_localize() -> None:
    parser = build_parser()
    mapped = parser.parse_args(
        [
            "map",
            "--site-name",
            "river",
            "--corpus",
            "raw",
            "--run",
            "run",
            "--intrinsics",
            "k.json",
            "--gluemap-root",
            "gm",
            "--gluemap-config",
            "gm.json",
            "--workspace-root",
            "ws",
            "--megaloc-source",
            "ml",
            "--megaloc-checkpoint",
            "ml.ckpt",
            "--edm-root",
            "edm",
            "--edm-checkpoint",
            "edm.ckpt",
        ]
    )
    assert mapped.command == "map"
    assert mapped.cell_px == 2.0
    localize = parser.parse_args(
        [
            "localize",
            "--map-model",
            "model",
            "--keyframes",
            "kf.jsonl",
            "--frames",
            "frames",
            "--output",
            "out",
            "--localizer-config",
            "loc.json",
            "--localization-dir",
            "loc",
            "--edm-root",
            "edm",
            "--edm-checkpoint",
            "edm.ckpt",
        ]
    )
    assert localize.command == "localize"
    assert localize.route == "stream"
