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


def test_find_engine_python_prefers_explicit_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ACAM_PYTHON", str(tmp_path / "custom" / "python"))
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom" / "python").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(jobs, "_probe_engine_python", lambda p: calls.append(p) or p.endswith("custom/python"))
    monkeypatch.setattr(jobs, "toolkit", lambda: tmp_path / "nope")
    jobs.reset_engine_python_cache()
    try:
        assert jobs.find_engine_python() == str(tmp_path / "custom" / "python")
        assert jobs.find_engine_python() == str(tmp_path / "custom" / "python")
        assert len(calls) == 1  # second call served from cache
    finally:
        jobs.reset_engine_python_cache()


def test_find_engine_python_none_when_nothing_works(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ACAM_PYTHON", raising=False)
    monkeypatch.setattr(jobs, "_probe_engine_python", lambda p: False)
    monkeypatch.setattr(jobs, "toolkit", lambda: tmp_path / "nope")
    jobs.reset_engine_python_cache()
    try:
        assert jobs.find_engine_python() is None
    finally:
        jobs.reset_engine_python_cache()


def test_parse_progress_stages_and_tqdm() -> None:
    log = ("a_click_a_map 0.1.0: 2 video(s)\n"
           "[0/4] screening videos\n  OK a.mp4\n"
           "[1/4] preprocessing: hashing\n"
           "[2/4] mapping: GlueMap poses -> EDM (~0.1 h; memory-guarded)\n"
           "  0%|          | 0/57 [00:00<?, ?it/s]\r"
           " 56%|###     | 32/57 [00:31<00:24, 2.15it/s]\r")
    info = jobs.parse_progress(log)
    assert info["stage"] == 2
    assert (info["percent"], info["done"], info["total"]) == (56, 32, 57)
    assert info["remaining"] == "00:24"
    assert info["estimate"] == "0.1 h"


def test_parse_progress_empty_and_done() -> None:
    assert jobs.parse_progress("")["stage"] is None
    assert jobs.parse_progress("starting...\n")["percent"] is None
    info = jobs.parse_progress("[0/4] x\n[1/4] y\n[2/4] z\n[3/4] r\n[4/4] p\ndone\n")
    assert info["stage"] == 4


def test_resume_request_rebuilds_submit() -> None:
    job = {"kind": "map", "engine": "original",
           "request": {"videos": ["a"], "out": "o", "name": "n", "irrelevant": 1}}
    req = jobs.resume_request(job)
    assert req["kind"] == "map" and req["engine"] == "original"
    assert req["videos"] == ["a"] and req["name"] == "n"


def _write_ply(path: Path, binary: bool, n: int = 5) -> None:
    import numpy as np
    xyz = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    rgb = (np.arange(n * 3, dtype=np.uint8).reshape(n, 3) * 8)
    header = ("ply\n" + ("format binary_little_endian 1.0\n" if binary else "format ascii 1.0\n")
              + f"element vertex {n}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    if binary:
        import numpy as _np
        dt = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        raw = _np.zeros(n, dtype=dt)
        raw["x"], raw["y"], raw["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        raw["red"], raw["green"], raw["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        path.write_bytes(header.encode() + raw.tobytes())
    else:
        lines = [header] + [f"{x} {y} {z} {r} {g} {b}\n" for (x, y, z), (r, g, b) in zip(xyz.tolist(), rgb.tolist())]
        path.write_text("".join(lines))


def test_read_ply_points_binary_and_ascii(tmp_path: Path) -> None:
    for binary in (True, False):
        p = tmp_path / ("b.ply" if binary else "a.ply")
        _write_ply(p, binary)
        xyz, rgb, total = jobs.read_ply_points(p)
        assert total == 5 and xyz.shape == (5, 3) and rgb.shape == (5, 3)
        assert abs(xyz[4, 0] - 12.0) < 1e-6 and rgb.dtype.name == "uint8"


def test_build_map_original_skips_pack(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    _video(tmp_path / "v", "a.mp4")
    cmd, _ = jobs.build_map_command(
        {"engine": "original", "videos": [str(tmp_path / "v")], "name": "n",
         "out": str(tmp_path / "out"), "intrinsics": "", "skip_dense": False}, "job9")
    assert "--no-pack" in cmd


def test_build_pack_command_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    base = tmp_path / "mymap"
    (base / "run").mkdir(parents=True)
    cmd, meta = jobs.build_pack_command(base, "job9")
    assert cmd[1] == "pack" and str(base / "run") in cmd
    assert cmd[cmd.index("--out") + 1] == str(base)
    assert meta["result_dir"] == str(base)
    with pytest.raises(ValueError, match="run 資料夾"):
        jobs.build_pack_command(tmp_path / "norun", "job9")


def test_pack_result_lists_bundle(tmp_path) -> None:
    assert jobs.pack_result({"result_dir": str(tmp_path)}) == []
    bundle = tmp_path / "mymap_bundle.tar.zst"
    bundle.write_bytes(b"x" * 10)
    rows = jobs.pack_result({"result_dir": str(tmp_path)})
    assert len(rows) == 1 and rows[0]["path"] == str(bundle)
    assert jobs.map_bundle(str(tmp_path)) == bundle


def test_densest_point_ignores_runaways(tmp_path: Path) -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    main = rng.normal([10.0, 20.0, 30.0], 1.0, (2000, 3))
    outliers = np.array([[1e6, 0.0, 0.0], [-1e6, 5.0, 5.0], [0.0, 1e7, 0.0]])
    xyz = np.vstack([main, outliers])
    qlo, qhi = np.percentile(xyz, [1.0, 99.0], axis=0)
    center = jobs.dense_center(xyz, qlo, qhi)
    assert np.linalg.norm(center - [10.0, 20.0, 30.0]) < 2.0
