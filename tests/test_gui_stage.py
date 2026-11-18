"""Free video picking for the original entry: files are staged into one folder."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs

CAMERA = {"intrinsics": str(jobs.BUILTIN_CAMERA_PROFILE), "camera_confirmed": True}


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
         "out": str(tmp_path / "out"), **CAMERA}, "job1")
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
         "name": "n", "out": str(tmp_path), **CAMERA})
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
    assert info["stage"] == 5


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
         "out": str(tmp_path / "out"), **CAMERA}, "job9")
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


def _finished_export_job(tmp_path: Path, payload: Path) -> dict:
    return {"id": "exported", "kind": "export", "engine": "original", "status": "done", "name": "mymap",
            "result_dir": str(payload)}


def test_export_result_routes_zip_and_import_folder(tmp_path: Path) -> None:
    archive = tmp_path / "場域交付" / "mymap.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"PCK" + bytes(1024))
    site = tmp_path / "場域交付" / "mymap"
    site.mkdir()
    (site / "site_profile.json").write_text("{}")
    for name in ("map", "localization", "model", "keyframes"):
        (site / name).mkdir()
    zip_rows = jobs.export_result(_finished_export_job(tmp_path, archive))
    folder_rows = jobs.export_result(_finished_export_job(tmp_path, site))
    nested = [row for row in folder_rows if row["path"] == str(site)]
    assert zip_rows[0]["path"] == str(archive) and "ZIP" in zip_rows[0]["label"]
    assert folder_rows[0]["path"] == str(site) and nested
    assert not jobs.export_result(_finished_export_job(tmp_path, tmp_path / "missing.zip"))
    assert jobs.delivery_archive(tmp_path / "othermap") == tmp_path / "othermap" / "場域交付" / "othermap.zip"


def test_densest_point_ignores_runaways(tmp_path: Path) -> None:
    import numpy as np
    rng = np.random.default_rng(0)
    main = rng.normal([10.0, 20.0, 30.0], 1.0, (2000, 3))
    outliers = np.array([[1e6, 0.0, 0.0], [-1e6, 5.0, 5.0], [0.0, 1e7, 0.0]])
    xyz = np.vstack([main, outliers])
    qlo, qhi = np.percentile(xyz, [1.0, 99.0], axis=0)
    center = jobs.dense_center(xyz, qlo, qhi)
    assert np.linalg.norm(center - [10.0, 20.0, 30.0]) < 2.0


def test_build_map_original_dense_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    _video(tmp_path / "v", "a.mp4")
    base = {"engine": "original", "videos": [str(tmp_path / "v")], "name": "n",
            "out": str(tmp_path / "out"), **CAMERA}
    cmd, _ = jobs.build_map_command({**base, "dense": True}, "job1")
    assert "--dense" in cmd
    cmd, _ = jobs.build_map_command({**base}, "job9")
    assert "--dense" not in cmd


def test_moge3_dense_defaults_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    _video(tmp_path / "v", "a.mp4")
    base = {"engine": "moge3", "videos": [str(tmp_path / "v")], "name": "n",
            "out": str(tmp_path / "out"), "camera_confirmed": True, "auto_calibrate": True}
    cmd, _ = jobs.build_map_command(base, "job1")
    assert "--auto_calibrate" in cmd
    assert "--skip-dense" not in cmd
    cmd, _ = jobs.build_map_command({**base, "dense": False}, "job2")
    assert "--skip-dense" in cmd


def test_dense_readiness_needs_weights(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ACAM_MVS_CHECKPOINT", raising=False)
    monkeypatch.delenv("ACAM_MVS_CONFIG", raising=False)
    monkeypatch.setattr(jobs, "REPO", tmp_path / "repo")
    _video(tmp_path, "a.mp4")
    req = {"engine": "original", "videos": [str(tmp_path / "a.mp4")], "name": "n",
           "out": str(tmp_path), "dense": True, **CAMERA}
    ok, message = jobs.validate_map_request(req)
    assert not ok and "MVSFormer++" in message
    (tmp_path / "repo/third_party/mvsformerpp/weights").mkdir(parents=True)
    (tmp_path / "repo/third_party/mvsformerpp/weights/model_best.pth").write_bytes(b"w")
    (tmp_path / "repo/third_party/mvsformerpp/weights/config.json").write_bytes(b"{}")
    ok, message = jobs.validate_map_request(req)
    assert ok and "稠密" in message


def test_parse_progress_densification_stage() -> None:
    log = "[2/4] mapping\n  0%| | 0/57\n=== densification (MVSFormer++) ===\n"
    assert jobs.parse_progress(log)["stage"] == 3
    log2 = "[2/4] mapping\n=== densification (MVSFormer++) ===\n[3/4] report\n"
    assert jobs.parse_progress(log2)["stage"] == 4


def test_map_connectivity_split_warns(tmp_path) -> None:
    import json
    report = {"images_registered": 226, "images_total": 226, "connected": False,
              "video_components": [["a"], ["b"]],
              "videos": {"a": {"source": "P1.MP4", "keyframes": 96, "registered": 96},
                         "b": {"source": "P2.MP4", "keyframes": 130, "registered": 130}}}
    (tmp_path / "map_report.json").write_text(json.dumps(report))
    conn = jobs.map_connectivity(tmp_path)
    assert conn is not None and not conn["connected"] and len(conn["components"]) == 2
    assert conn["images_registered"] == 226
    report["connected"] = True
    report["video_components"] = [["a", "b"]]
    (tmp_path / "map_report.json").write_text(json.dumps(report))
    assert jobs.map_connectivity(tmp_path)["connected"]
    assert jobs.map_connectivity(tmp_path / "missing") is None


def test_parse_mapping_phase_ba_and_retri() -> None:
    ba = "Refinement iteration 2/2\n=== Iteration 1/3 ===\nStarting iterative BA with 1 tracks\n"
    info = jobs.parse_mapping_phase(ba)
    assert info["in_ba"] and "BA" in info["phase"] and "2/2" in info["phase"]
    retri = "Refinement iteration 2/2\nRetriangulation and Global bundle adjustment\nTriangulating image #226 (225)\n"
    info = jobs.parse_mapping_phase(retri)
    assert info["in_ba"] and "226" in info["phase"]
    blank = jobs.parse_mapping_phase("")
    assert blank["phase"] is None and not blank["in_ba"]


def test_map_output_status_blocks_finished_name(tmp_path) -> None:
    import json
    site = tmp_path / "out" / "testtest"
    (site / "acam").mkdir(parents=True)
    (site / "acam" / "mapping_summary.json").write_text(json.dumps({"minutes": 26.9}))
    (site / "map_report.json").write_text(json.dumps(
        {"images_registered": 226, "images_total": 226, "connected": False,
         "video_components": [["a"], ["b"]]}))
    msg = jobs.map_output_status(str(tmp_path / "out"), "testtest")
    assert msg and "換個任務名稱" in msg and "多段不相連" in msg
    assert jobs.map_output_status(str(tmp_path / "out"), "fresh") is None
    assert jobs.map_output_status(str(tmp_path / "nope"), "x") is None
    # partial failed run without a report stays resumable
    (tmp_path / "out" / "half" / "run").mkdir(parents=True)
    assert jobs.map_output_status(str(tmp_path / "out"), "half") is None


def test_same_name_fresh_task_blocked_but_resume_passes(tmp_path, monkeypatch) -> None:
    import json
    from pathlib import Path as _P
    site = tmp_path / "out" / "dup"
    (site / "acam").mkdir(parents=True)
    (site / "acam" / "mapping_summary.json").write_text(json.dumps({"minutes": 1.0}))
    (site / "map_report.json").write_text(json.dumps(
        {"images_registered": 10, "images_total": 10, "connected": True, "video_components": [["a"]]}))
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"\x00" * 64)
    req = {"engine": "original", "videos": [str(vid)], "name": "dup",
           "out": str(tmp_path / "out"), "intrinsics": "", "dense": False}
    ok, message = jobs.validate_map_request(req)
    assert not ok and "換個任務名稱" in message
    try:
        jobs.build_map_command(req, "job1")
    except ValueError as error:
        assert "換個任務名稱" in str(error)
    else:
        raise AssertionError("fresh same-name build must fail")
    resumed = jobs.resume_request({"kind": "map", "engine": "original", "request": dict(req)})
    assert resumed["_resume"]
    monkeypatch.setattr(jobs, "stage_selection", lambda videos, dest: dest)
    _cmd, meta = jobs.build_map_command(resumed, "job2")
    assert meta["result_dir"] == str(_P(str(tmp_path / "out")) / "dup")


def test_camera_choice_requires_explicit_mode_and_matching_batch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    video = _video(tmp_path, "clip.mp4")
    base = {"engine": "original", "videos": [str(video)], "name": "new", "out": str(tmp_path / "out")}
    ready, reason = jobs.validate_map_request(base)
    assert not ready and "原始版不會自動標定" in reason
    with pytest.raises(ValueError, match="原始版不會自動標定"):
        jobs.build_map_command(base, "noCamera")
    with pytest.raises(ValueError, match="分批建圖"):
        jobs.build_map_command({**base, **CAMERA, "camera_confirmed": False}, "mixed")
    with pytest.raises(ValueError, match="只有 MoGe-3"):
        jobs.build_map_command({**base, "camera_confirmed": True, "auto_calibrate": True}, "invalid")
    with pytest.raises(ValueError, match="明確選擇"):
        jobs.build_map_command({**base, "engine": "moge3", "camera_confirmed": True}, "noAuto")
    assert not jobs.JOBS.exists()


def test_camera_profile_import_and_resume_use_job_snapshot(tmp_path, monkeypatch) -> None:
    import json

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    video = _video(tmp_path, "clip.mp4")
    calibration = jobs.load_calibration(jobs.BUILTIN_CAMERA_PROFILE)
    source = tmp_path / "camera-mode.json"
    source.write_text(json.dumps(calibration))
    saved = jobs.import_camera_profile(source)
    request = {"kind": "map", "engine": "original", "videos": [str(video)], "name": "new",
               "out": str(tmp_path / "out"), "intrinsics": str(saved), "camera_confirmed": True}
    cmd, meta = jobs.build_map_command(request, "old")
    assert cmd[cmd.index("--intrinsics") + 1] == meta["calibration_file"]
    changed = jobs.load_calibration(saved)
    changed["K"][0][0] += 100
    source.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="同名"):
        jobs.import_camera_profile(source)
    assert jobs.load_calibration(saved) == calibration

    saved.unlink()
    resumed = jobs.resume_request({**meta, "kind": "map", "engine": "original", "request": request})
    cmd, resumed_meta = jobs.build_map_command(resumed, "new")
    assert resumed_meta["result_dir"] == meta["result_dir"]
    assert Path(cmd[cmd.index("--intrinsics") + 1]).parent.name == "new"
    assert jobs.load_calibration(Path(resumed_meta["calibration_file"])) == calibration


def test_bad_calibration_is_not_imported_or_staged(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    profile = tmp_path / "invalid.json"
    profile.write_text('{"image_width": 1280, "image_height": 720, "images_are_undistorted": false}')
    with pytest.raises(ValueError, match="undistort the footage"):
        jobs.import_camera_profile(profile)
    video = _video(tmp_path, "clip.mp4")
    request = {"engine": "original", "videos": [str(video)], "out": str(tmp_path / "out"),
               "intrinsics": str(profile), "camera_confirmed": True}
    with pytest.raises(ValueError, match="標定檔無效"):
        jobs.build_map_command(request, "invalid")
    assert not jobs.camera_profiles_dir().exists()
    assert not jobs.JOBS.exists()


def test_measured_spkf_prefers_history(tmp_path) -> None:
    import json
    from acam import plan
    assert plan.measured_spkf(tmp_path / "empty") is None
    for name, minutes, images in (("a", 26.9, 226), ("b", 13.5, 113)):
        site = tmp_path / name
        (site / "acam").mkdir(parents=True)
        (site / "acam" / "mapping_summary.json").write_text(json.dumps({"minutes": minutes}))
        (site / "map_report.json").write_text(json.dumps({"images_total": images}))
    got = plan.measured_spkf(tmp_path)
    assert min(26.9 * 60 / 226, 13.5 * 60 / 113) <= got <= max(26.9 * 60 / 226, 13.5 * 60 / 113)
    assert plan.eta_hours(226, got) == pytest.approx(26.9 / 60, rel=0.02)
    assert plan.eta_hours(226) == pytest.approx(226 * plan.SECONDS_PER_KEYFRAME / 3600)


def test_phase_remaining_triangulation() -> None:
    head = ("[2026-11-18 17:37:15,341] INFO:gluemap.estimators.track_establishment: Established 26144 tracks\n"
            "Refinement iteration 2/2\n"
            "[2026-11-18 17:38:25,804] INFO:gluemap.controllers.augmented_bundle_adjustment: Starting iterative BA\n"
            "Created merged database with 226 images\n")
    body = "".join(
        f"I20261118 17:45:{40 + i:02d}.100000 1 incremental_pipeline.cc:751] "
        f"Triangulating image #{200 + i} ({199 + i})\n" for i in range(10))
    info = jobs.parse_mapping_phase(head + body)
    assert info["in_ba"] and info["done"] == 209 and info["total"] == 226
    assert info["rate_per_s"] == pytest.approx(1.0, rel=0.05)
    assert info["remaining_s"] == pytest.approx((226 - 209) / info["rate_per_s"], rel=0.01)


def test_phase_remaining_ba_uses_previous_round() -> None:
    log = ("Refinement iteration 1/2\n"
           "[2026-11-18 17:38:26,235] INFO: x === Iteration 1/3 ===\n"
           "[2026-11-18 17:39:54,428] INFO: x Ceres Solver Report: Iterations: 60\n"
           "[2026-11-18 17:40:04,755] INFO: x === Iteration 2/3 ===\n")
    now = jobs._log_epoch("2026", "11", "18", "17", "40", "44")
    info = jobs.parse_mapping_phase(log, now=now)
    assert info["in_ba"] and info["round"] == "2/3"
    assert info["elapsed_s"] == pytest.approx(40.0, abs=1.0)
    # round 1 took ~88s, 40s into round 2 -> ~48s left
    assert info["remaining_s"] == pytest.approx(88.0 - 40.0, abs=2.0)


def test_resume_plan_reports_skip_vs_redo(tmp_path) -> None:
    base = tmp_path / "site"
    job = {"kind": "map", "result_dir": str(base)}
    actions = dict(jobs.resume_plan(job))
    assert actions["前處理 keyframes"] == "redo"
    assert actions["GlueMap 匹配＋BA"] == "redo"
    (base / "run/receipts").mkdir(parents=True)
    (base / "run/receipts/direct_preprocessing.json").write_text("{}")
    glue = base / "workspace/work_x/gluemap"
    glue.mkdir(parents=True)
    (glue / "twoview_result.pth").write_bytes(b"x")
    (glue / "star_result.pth").write_bytes(b"x")
    (glue / "database_sift.db").write_bytes(b"x" * 100)
    actions = dict(jobs.resume_plan(job))
    assert actions["前處理 keyframes"] == "skip"
    assert actions["雙視角幾何快取"] == "reuse"
    assert actions["SIFT 資料庫"] == "reuse"
    assert actions["GlueMap 匹配＋BA"] == "redo"
    model = base / "run/artifacts/mapping/edm/model"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(b"x")
    (model / "points3D.bin").write_bytes(b"x")
    assert dict(jobs.resume_plan(job))["GlueMap 匹配＋BA"] == "skip"
    assert jobs.resume_plan({"kind": "cruise"}) == [("巡航定位", "redo")]
