"""Free video picking for the original entry: files are staged into one folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs


def _video(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(os.urandom(64))
    return path


def test_stage_selection_keeps_names_and_dedupes(tmp_path: Path) -> None:
    a = _video(tmp_path / "a", "clip.mp4")
    b = _video(tmp_path / "b", "clip.mp4")
    c = _video(tmp_path / "b", "other.MP4")
    staged = jobs.stage_selection([a, b, c], tmp_path / "out")
    names = sorted(p.name for p in staged.iterdir())
    assert names == ["clip.mp4", "clip_2.mp4", "other.MP4"]
    assert os.path.samefile(staged / "clip.mp4", a)
    assert os.path.samefile(staged / "clip_2.mp4", b)


def test_build_map_original_stages_mixed_entries(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    folder = tmp_path / "folder"
    one = _video(folder, "one.mp4")
    two = _video(tmp_path / "elsewhere", "two.mp4")
    cmd, meta = jobs.build_map_command(
        {"engine": "original", "videos": [str(folder), str(two)], "name": "n",
         "out": str(tmp_path / "out"), "intrinsics": "", "skip_dense": False}, "job1")
    selection = Path(cmd[2])
    assert selection.parent.name == "job1"
    assert sorted(p.name for p in selection.iterdir()) == ["one.mp4", "two.mp4"]
    assert meta["result_dir"].endswith("n")
    assert os.path.samefile(selection / "one.mp4", one)


def test_build_map_original_needs_real_videos(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    request = {"engine": "original", "videos": [str(tmp_path / "nope")], "name": "n",
               "out": str(tmp_path / "out"), "intrinsics": "", "skip_dense": False}
    with pytest.raises(ValueError, match="路徑不存在"):
        jobs.build_map_command(request, "job1")
    empty = tmp_path / "empty"
    empty.mkdir()
    request["videos"] = [str(empty)]
    with pytest.raises(ValueError, match="沒有找到影片"):
        jobs.build_map_command(request, "job1")


def test_validate_original_counts_videos_not_entries(tmp_path: Path) -> None:
    _video(tmp_path / "d1", "a.mp4")
    _video(tmp_path / "d1", "b.mp4")
    _video(tmp_path, "c.mp4")
    ok, message = jobs.validate_map_request(
        {"engine": "original", "videos": [str(tmp_path / "d1"), str(tmp_path / "c.mp4")],
         "name": "n", "out": str(tmp_path), "intrinsics": ""})
    assert ok and "共 3 部影片" in message
