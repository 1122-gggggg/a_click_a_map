from __future__ import annotations

import json
from pathlib import Path

import pytest

from acam import cli, hw, pack, plan, report, runner
from acam.config import Paths

GIB = 2**30


def test_choose_tracks_reproduces_the_two_measured_corpora() -> None:
    # 2043 images on ~30 GiB: K=1024 OOM-killed, K=512 completed -> must pick 512
    assert plan.choose_tracks(2043, 30 * GIB)[0] == 512
    # 1389 images completed at K=1024 (22 M points), but the budget only admits the proven region
    tracks, points, fits = plan.choose_tracks(1389, 30 * GIB)
    assert fits and 640 <= tracks <= 768 and points <= plan.point_budget(30 * GIB)


def test_choose_tracks_shrinks_with_less_ram_and_flags_hopeless_corpora() -> None:
    big = plan.choose_tracks(2043, 62 * GIB)[0]
    small = plan.choose_tracks(2043, 20 * GIB)[0]
    assert small < plan.choose_tracks(2043, 30 * GIB)[0] < big
    tracks, _, fits = plan.choose_tracks(20_000, 30 * GIB)
    assert tracks == plan.MIN_TRACKS and not fits


def test_reduce_tracks_terminates_at_the_floor() -> None:
    k, seen = 1024, []
    while k is not None:
        seen.append(k)
        k = plan.reduce_tracks(k)
    assert seen[0] == 1024 and seen[-1] >= plan.MIN_TRACKS and seen == sorted(seen, reverse=True)
    assert plan.reduce_tracks(plan.MIN_TRACKS) is None


def test_oom_classification() -> None:
    assert plan.is_oom_exit(-9, "") and plan.is_oom_exit(137, "")
    assert plan.is_oom_exit(1, "RuntimeError: std::bad_alloc")
    assert not plan.is_oom_exit(1, "ValueError: n_pairs=27225 > 25000")
    assert not plan.is_oom_exit(0, "std::bad_alloc handled upstream")


def test_memory_limits_leave_headroom_and_cap_swap() -> None:
    mem_max, swap_max = plan.memory_limits(30 * GIB, 20 * GIB)
    assert mem_max == int(28.5 * GIB) and swap_max == 6 * GIB
    assert plan.memory_limits(30 * GIB, 2 * GIB)[1] == 2 * GIB


def test_meminfo_parsing() -> None:
    text = "MemTotal: 1048576 kB\nMemAvailable: 524288 kB\nSwapTotal: 2097152 kB\nSwapFree: 1048576 kB\n"
    mem = hw.read_meminfo(text)
    assert (mem.total, mem.available, mem.swap_total, mem.swap_free) == (GIB, GIB // 2, 2 * GIB, GIB)


def test_video_components_flag_a_disconnected_video() -> None:
    shared = {("a", "b"): 400, ("b", "c"): 80, ("c", "d"): 3}
    assert report.video_components(shared, ["a", "b", "c", "d"], 50) == [["a", "b", "c"], ["d"]]
    assert report.video_components(shared, ["a", "b", "c", "d"], 1) == [["a", "b", "c", "d"]]


def test_default_command_is_map() -> None:
    parser_args = cli.build_parser().parse_args(["map", "/videos", "--tracks", "384"])
    assert parser_args.tracks == "384" and parser_args.out == Path("acam_out")
    assert cli.safe_name("烏來 建圖/2026") == "2026"
    assert cli.safe_name("a b") == "a_b"


def _paths(tmp: Path) -> Paths:
    return Paths(python=Path("python"), gluemap=tmp / "gm", gluemap_checkpoints=tmp / "ck", edm=tmp / "edm",
                 edm_checkpoint=tmp / "edm.ckpt", megaloc_source=tmp / "ml", megaloc_checkpoint=tmp / "ml.st")


def test_rendered_gluemap_config_points_at_checkpoints_and_sets_tracks(tmp_path: Path) -> None:
    cfg = json.loads(runner.render_gluemap_config(_paths(tmp_path), 448, tmp_path / "c.json").read_text())
    assert cfg["num_track_per_img"] == 448
    assert cfg["path_feedforward"] == str(tmp_path / "ck/pi3.safetensors")
    assert cfg["path_retrieval"].endswith("dino_salad.ckpt") and cfg["camera_model"] == "PINHOLE"
    assert cfg["use_gt_intrinsics"] is True and cfg["refine_intrinsics"] is False


def test_transplant_copies_only_k_independent_caches(tmp_path: Path) -> None:
    old, new = tmp_path / "old/gluemap", tmp_path / "new/gluemap"
    (old / "vid_a").mkdir(parents=True)
    (new).mkdir(parents=True)
    (old / "vid_a/salad_descriptors.pt").write_bytes(b"d")
    (old / "twoview_result.pth").write_bytes(b"t")
    (old / "star_result.pth").write_bytes(b"s")  # depends on num_track_per_img: must NOT move
    (old / "database_sift.db").write_bytes(b"x")
    copied = runner.transplant_caches(tmp_path / "old", tmp_path / "new")
    assert sorted(copied) == ["twoview_result.pth", "vid_a"]
    assert (new / "vid_a/salad_descriptors.pt").exists() and not (new / "star_result.pth").exists()
    assert not (new / "database_sift.db").exists()


def test_pack_templating_rewrites_known_paths_and_rejects_leftovers(tmp_path: Path) -> None:
    src = tmp_path / "keyframes.jsonl"
    src.write_text('{"image_uri": "/data/run/artifacts/keyframes/images/v/f.jpg"}\n')
    dst = tmp_path / "out.in"
    pack._render(src, dst, [("/data/run/artifacts/keyframes/images", "@PKG@/localization/keyframes/images")], ["/data/run"])
    assert "@PKG@/localization/keyframes/images/v/f.jpg" in dst.read_text()
    src.write_text('{"x": "/data/run/other/file"}\n')
    with pytest.raises(RuntimeError, match="unmapped absolute path"):
        pack._render(src, dst, [], ["/data/run"])
