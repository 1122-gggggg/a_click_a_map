from __future__ import annotations

import json
from pathlib import Path

from acam import cli, plan, runner, screen
from acam.plan import VideoBudget

GIB = 2**30


def _videos(*seconds: float, scores: tuple[float, ...] | None = None) -> list[VideoBudget]:
    scores = scores or tuple(1.0 for _ in seconds)
    return [VideoBudget(f"v{i}.mp4", s, sc) for i, (s, sc) in enumerate(zip(seconds, scores))]


def test_a_corpus_that_fits_is_used_whole_and_never_below_k512() -> None:
    result = plan.plan_budget(_videos(300, 300), 30 * GIB)
    assert result.sampling_scale == 1.0 and not result.dropped and result.tracks >= 512 and result.fits
    # the real Wulai corpus (1858 s) is within the estimator's tolerance of the validated 2043-keyframe run
    wulai = plan.plan_budget(_videos(109, 497, 108, 267, 182, 112, 301, 140, 142), 30 * GIB)
    assert wulai.sampling_scale == 1.0 and wulai.tracks == 512 and not wulai.dropped


def test_oversized_corpus_is_thinned_before_any_video_is_dropped() -> None:
    result = plan.plan_budget(_videos(1500, 1500), 30 * GIB)          # ~3450 keyframes at full density
    assert plan.MIN_SCALE <= result.sampling_scale < 1.0 and result.tracks == 512 and not result.dropped
    assert result.est_keyframes <= plan.keyframe_cap(30 * GIB) * 1.01


def test_far_oversized_corpus_drops_the_lowest_scoring_videos_first() -> None:
    videos = _videos(1000, 1000, 1000, 1000, 1000, 1000, scores=(1.0, 0.9, 0.3, 1.0, 0.8, 0.2))
    result = plan.plan_budget(videos, 30 * GIB)
    assert result.sampling_scale == plan.MIN_SCALE and result.dropped
    assert {"v2.mp4", "v5.mp4"} <= {name for name, _ in result.dropped}      # worst scores go first
    assert "v0.mp4" in result.keep and "v3.mp4" in result.keep
    assert result.keep == sorted(result.keep)                                  # original order preserved
    assert result.est_keyframes <= plan.keyframe_cap(30 * GIB)


def test_one_enormous_video_is_thinned_rather_than_refused() -> None:
    result = plan.plan_budget(_videos(20000), 30 * GIB)
    assert result.keep == ["v0.mp4"] and result.sampling_scale < plan.MIN_SCALE
    assert result.est_keyframes <= plan.keyframe_cap(30 * GIB)


def test_less_ram_means_a_smaller_cap_and_max_keyframes_is_honoured() -> None:
    assert plan.keyframe_cap(16 * GIB) < plan.keyframe_cap(30 * GIB) < plan.keyframe_cap(64 * GIB)
    assert plan.keyframe_cap(64 * GIB, max_keyframes=500) == 500
    small = plan.plan_budget(_videos(1000), 16 * GIB)
    assert small.sampling_scale < 1.0


def test_rescale_after_preprocess_only_triggers_when_the_real_count_exceeds_the_cap() -> None:
    cap = plan.keyframe_cap(30 * GIB)
    assert plan.rescale_after_preprocess(cap, 1.0, 30 * GIB) is None
    assert plan.rescale_after_preprocess(int(cap * 1.04), 1.0, 30 * GIB) is None
    retry = plan.rescale_after_preprocess(cap * 2, 1.0, 30 * GIB)
    assert retry is not None and 0.4 < retry < 0.5


def test_cuda_oom_is_not_confused_with_host_oom_retry() -> None:
    assert plan.is_cuda_oom("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate")
    assert not plan.is_cuda_oom("std::bad_alloc")


# ------------------------------------------------------------------ screening decisions
def _frames(n: int, **kw) -> list[screen.FrameStats]:
    base = dict(lap_var=1500.0, luma=110.0, dark_fraction=0.0, saturated_fraction=0.0, corners=4000)
    base.update(kw)
    return [screen.FrameStats(**base) for _ in range(n)]


def _judge(frames, **kw) -> screen.Verdict:
    args = dict(unreadable=0, width=1920, height=1080, fps=30.0, duration=120.0, static_motion=20.0)
    args.update(kw)
    return screen.judge(frames, **args)


def test_good_video_passes() -> None:
    verdict = _judge(_frames(48))
    assert verdict.verdict == "OK" and verdict.score == 1.0 and not verdict.reasons


def test_mostly_blurry_dark_or_textureless_video_is_rejected() -> None:
    assert _judge(_frames(48, lap_var=3.0, corners=10)).verdict == "REJECT"
    assert _judge(_frames(48, luma=8.0, dark_fraction=0.95)).verdict == "REJECT"
    assert _judge(_frames(48, luma=250.0, saturated_fraction=0.95)).verdict == "REJECT"


def test_partially_bad_video_warns_and_ranks_below_a_clean_one() -> None:
    mixed = _frames(34) + _frames(14, lap_var=2.0, corners=5)           # 29% bad
    verdict = _judge(mixed)
    assert verdict.verdict == "WARN" and 0.6 < verdict.score < 0.8
    assert verdict.score < _judge(_frames(48)).score


def test_unreadable_short_tiny_and_empty_videos() -> None:
    assert _judge(_frames(30), unreadable=18).verdict == "REJECT"       # >25% of sampled positions unreadable
    assert _judge(_frames(46), unreadable=2).verdict == "OK"            # a failed seek or two is normal
    assert _judge(_frames(42), unreadable=6).verdict == "WARN"
    assert _judge(_frames(48), duration=3.0).verdict == "REJECT"
    assert _judge(_frames(48), width=320, height=180).verdict == "REJECT"
    assert _judge([], unreadable=0, width=0, height=0, fps=0, duration=0, static_motion=None).verdict == "REJECT"


def test_static_camera_and_low_fps_only_warn() -> None:
    verdict = _judge(_frames(48), static_motion=0.1, fps=5.0)
    assert verdict.verdict == "WARN" and verdict.score < 1.0 and len(verdict.reasons) == 2


# ------------------------------------------------------------------ run-level gates
def test_weak_sources_flags_videos_the_engine_could_not_use(tmp_path: Path) -> None:
    (tmp_path / "receipts").mkdir()
    (tmp_path / "inputs").mkdir()
    (tmp_path / "receipts/direct_preprocessing.json").write_text(json.dumps({"per_source": [
        {"video_id": "a", "analyzed_frames": 400, "removed_frames": 5, "keyframes": 90},
        {"video_id": "b", "analyzed_frames": 400, "removed_frames": 300, "keyframes": 20},
        {"video_id": "c", "analyzed_frames": 400, "removed_frames": 0, "keyframes": 3},
    ]}))
    (tmp_path / "inputs/corpus_manifest.json").write_text(json.dumps({"sources": [
        {"video_id": v, "path": f"/data/{v}.mp4"} for v in "abc"]}))
    weak = runner.weak_sources(tmp_path)
    assert set(weak) == {"b.mp4", "c.mp4"} and "discarded" in weak["b.mp4"] and "3 usable" in weak["c.mp4"]


def test_stage_corpus_hardlinks_exactly_the_kept_videos(tmp_path: Path) -> None:
    sources = {n: tmp_path / n for n in ("a.mp4", "b.mp4", "c.mp4")}
    for path in sources.values():
        path.write_bytes(b"x")
    corpus = tmp_path / "corpus"
    assert cli.stage_corpus(corpus, ["a.mp4", "c.mp4"], sources) == "hardlink"
    assert sorted(p.name for p in corpus.iterdir()) == ["a.mp4", "c.mp4"]
    assert not (corpus / "a.mp4").is_symlink() and (corpus / "a.mp4").stat().st_ino == sources["a.mp4"].stat().st_ino
    cli.stage_corpus(corpus, ["b.mp4"], sources)  # restaging replaces the previous selection
    assert [p.name for p in corpus.iterdir()] == ["b.mp4"]


def test_sampling_scale_becomes_engine_flags_only_when_thinning(tmp_path: Path) -> None:
    from acam.config import Paths
    paths = Paths(Path("py"), tmp_path, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    job = runner.MapJob("s", tmp_path, tmp_path / "run", tmp_path / "ws", tmp_path / "m", tmp_path / "k.json")
    assert "--baseline-fps" not in runner.map_argv(paths, job, tmp_path / "c.json")
    thinned = runner.map_argv(paths, job, tmp_path / "c.json", sampling_scale=0.5)
    assert thinned[thinned.index("--baseline-fps") + 1] == "0.500" and thinned[thinned.index("--fast-fps") + 1] == "1.000"
