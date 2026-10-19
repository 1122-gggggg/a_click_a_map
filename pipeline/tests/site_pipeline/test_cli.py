from __future__ import annotations

from sfm_diagnosis.site_pipeline.cli import build_parser


def test_cli_has_init_run_status_approve_export_and_backfill_commands() -> None:
    parser = build_parser()

    assert (
        parser.parse_args(
            ["init", "--config", "site.toml", "--corpus", "raw", "--output", "run"]
        ).command
        == "init"
    )
    assert (
        parser.parse_args(["run", "--run", "run", "--to-stage", "stage11_reinforcement"]).command
        == "run"
    )
    assert parser.parse_args(["status", "--run", "run"]).command == "status"
    assert (
        parser.parse_args(
            [
                "approve-final",
                "--run",
                "run",
                "--decision-sha",
                "a" * 64,
                "--approver",
                "operator",
            ]
        ).command
        == "approve-final"
    )
    assert parser.parse_args(["export", "--run", "run"]).command == "export"
    assert (
        parser.parse_args(["backfill", "--run", "run", "--legacy-run", "legacy"]).command
        == "backfill"
    )
    assert (
        parser.parse_args(["normalize-selection", "--run", "run"]).command
        == "normalize-selection"
    )
    assert (
        parser.parse_args(
            [
                "enrich-bridge",
                "--run",
                "run",
                "--keyframe",
                "frame-a",
                "--attempt-name",
                "repair-1",
            ]
        ).command
        == "enrich-bridge"
    )
    assert (
        parser.parse_args(["materialize-layers", "--run", "run"]).command
        == "materialize-layers"
    )
    assert (
        parser.parse_args(["validate-layers", "--run", "run"]).command
        == "validate-layers"
    )
    assert (
        parser.parse_args(
            [
                "fuse-localization",
                "--run",
                "run",
                "--robust-validation",
                "robust.json",
                "--dense-validation",
                "dense.json",
            ]
        ).command
        == "fuse-localization"
    )
    refinement = parser.parse_args(
        [
            "refine-map",
            "--backend",
            "pixsfm",
            "--input-model",
            "model",
            "--images",
            "images",
            "--intrinsics",
            "intrinsics.json",
            "--run",
            "run",
            "--runtime-python",
            "/runtime/python",
            "--pixsfm-patch-size",
            "6",
        ]
    )
    assert refinement.command == "refine-map"
    assert refinement.backend == "pixsfm"
    assert refinement.pixsfm_patch_size == 6

    dense_full = parser.parse_args(
        [
            "refine-map",
            "--backend",
            "densesfm-full",
            "--input-model",
            "model",
            "--images",
            "images",
            "--pairs",
            "pairs.jsonl",
            "--intrinsics",
            "intrinsics.json",
            "--run",
            "run",
            "--runtime-python",
            "/runtime/python",
            "--runtime-root",
            "DenseSfM-Refine",
            "--continuation-receipt",
            "refinement-receipt.json",
        ]
    )
    assert dense_full.continuation_receipt.name == "refinement-receipt.json"

    mvroma = parser.parse_args(
        [
            "mvroma-augment",
            "--scope",
            "targeted",
            "--input-model",
            "model",
            "--images",
            "images",
            "--selection",
            "final_selection.json",
            "--intrinsics",
            "intrinsics.json",
            "--run",
            "run",
            "--runtime-python",
            "/runtime/python",
            "--runtime-root",
            "MV-RoMa",
            "--weight",
            "outdoor_final.pth",
        ]
    )
    assert mvroma.command == "mvroma-augment"
    assert mvroma.scope == "targeted"

    direct = parser.parse_args(
        [
            "direct-map",
            "--site-name",
            "river-all8",
            "--corpus",
            "mapping",
            "--run",
            "run",
            "--intrinsics",
            "intrinsics.json",
            "--gluemap-root",
            "gluemap",
            "--gluemap-config",
            "gluemap.json",
            "--workspace-root",
            "workspace",
            "--megaloc-source",
            "megaloc/source",
            "--megaloc-checkpoint",
            "megaloc/model.safetensors",
            "--hover-fps",
            "0.2",
            "--resume",
        ]
    )
    assert direct.command == "direct-map"
    assert direct.hover_fps == 0.2
    assert direct.resume is True
