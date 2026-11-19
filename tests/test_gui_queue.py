"""Exercise the desktop queue with gated child processes, without Gtk or a GPU."""
from __future__ import annotations

import shlex
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs


@pytest.fixture
def engine(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(jobs, "REPO", tmp_path)
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    monkeypatch.setattr(jobs, "find_engine_python", lambda: sys.executable)
    (tmp_path / "third_party").mkdir()
    (tmp_path / "video.mp4").touch()
    (tmp_path / "maps").mkdir()
    child = tmp_path / "engine.py"
    child.write_text(
        """\
import signal
import sys
import time
from pathlib import Path

option = "--name" if sys.argv[1] == "map" else "--site-id"
name = sys.argv[sys.argv.index(option) + 1]
release = Path(f"{name}.release")

def record(event):
    with Path("events.txt").open("a", encoding="utf-8") as f:
        f.write(f"{event} {name}\\n")

def wait_for_release():
    deadline = time.monotonic() + 30
    while not release.exists():
        if time.monotonic() >= deadline:
            sys.exit(2)
        time.sleep(0.01)

def stop(_signum, _frame):
    Path(f"{name}.signalled").touch()
    wait_for_release()
    sys.exit(7)

if Path(f"{name}.hold-stop").exists():
    signal.signal(signal.SIGTERM, stop)
record("start")
Path(f"{name}.started").touch()
wait_for_release()
record("end")
""",
        encoding="utf-8",
    )
    launcher = tmp_path / "run.sh"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.join([sys.executable, str(child)])} "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)
    yield tmp_path
    for hold in tmp_path.glob("*.hold-stop"):
        hold.with_suffix(".release").touch()
    jobs.stop_all()


def _map_request(root: Path, name: str) -> dict:
    return {"kind": "map", "engine": "original", "name": name, "out": str(root / "maps"),
            "videos": [str(root / "video.mp4")], "intrinsics": str(jobs.BUILTIN_CAMERA_PROFILE),
            "camera_confirmed": True}


def _submit_map(root: Path, name: str) -> str:
    return jobs.create_job(_map_request(root, name))["id"]


def _wait_for(predicate) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "job queue did not reach the expected state"
        time.sleep(0.01)


def _wait_started(root: Path, name: str, job_id: str) -> None:
    _wait_for(lambda: (root / f"{name}.started").exists() and jobs.read_job(job_id).get("pid"))


def test_jobs_run_fifo_and_reap_before_next_start(engine: Path) -> None:
    first = _submit_map(engine, "first")
    _wait_started(engine, "first", first)
    pid = jobs.read_job(first)["pid"]
    second = _submit_map(engine, "second")
    third = _submit_map(engine, "third")
    assert jobs.running_count() == 1
    assert jobs.active_count() == 3
    assert jobs.read_job(second)["status"] == jobs.read_job(third)["status"] == "queued"

    (engine / "first.release").touch()
    _wait_started(engine, "second", second)
    assert jobs.read_job(first)["status"] == "done"
    assert not Path(f"/proc/{pid}").exists()
    assert jobs.read_job(third)["status"] == "queued"
    (engine / "second.release").touch()
    _wait_started(engine, "third", third)
    (engine / "third.release").touch()
    _wait_for(lambda: jobs.active_count() == 0)
    assert (engine / "events.txt").read_text(encoding="utf-8").splitlines() == [
        "start first", "end first", "start second", "end second", "start third", "end third",
    ]


@pytest.mark.parametrize("kind", ["map", "map-normalized-name", "export"])
def test_resolved_output_reserved_until_queued_job_is_final(engine: Path, kind: str) -> None:
    first = _submit_map(engine, "first")
    _wait_started(engine, "first", first)
    base = engine / "maps" / "target"
    base.mkdir()
    alias = base.with_name("alias")
    alias.symlink_to(base, target_is_directory=True)
    if kind != "export":
        request = _map_request(engine, "target" if kind == "map" else "target地圖")
        duplicate = {**request, "name": "alias" if kind == "map" else "target", "_resume": True,
                     "videos": [str(engine / "missing.mp4")]}
    else:
        (base / "run").mkdir()
        jobs.approve_map(base)
        request = {"kind": "export", "engine": "original", "map": str(base)}
        duplicate = {**request, "map": str(alias)}
    queued = jobs.create_job(request)["id"]
    assert jobs.read_job(queued)["status"] == "queued"
    with pytest.raises(ValueError, match="輸出位置已有排隊或執行中的任務"):
        jobs.create_job(duplicate)
    assert {path.name for path in jobs.JOBS.iterdir()} == {first, queued}

    jobs.stop_job(queued)
    cancelled = jobs.read_job(queued)
    assert cancelled["status"] == "stopped"
    assert "pid" not in cancelled
    assert not (jobs.job_dir(queued) / "log.txt").exists()
    retry = jobs.create_job(jobs.resume_request(cancelled))["id"]
    assert jobs.read_job(retry)["result_dir"] == cancelled["result_dir"]
    (engine / "first.release").touch()
    _wait_started(engine, "target", retry)
    (engine / "target.release").touch()
    _wait_for(lambda: jobs.active_count() == 0)
    assert (engine / "events.txt").read_text(encoding="utf-8").splitlines() == [
        "start first", "end first", "start target", "end target",
    ]
    assert not (jobs.job_dir(queued) / "log.txt").exists()


@pytest.mark.parametrize("close", [False, True], ids=["stop-one", "stop-all"])
def test_stopping_job_holds_its_slot_until_reaped(engine: Path, close: bool) -> None:
    (engine / "first.hold-stop").touch()
    first = _submit_map(engine, "first")
    _wait_started(engine, "first", first)
    pid = jobs.read_job(first)["pid"]
    second = _submit_map(engine, "second")
    finished = threading.Event()
    counts = []

    def stop() -> None:
        if close:
            counts.append(jobs.stop_all())
        else:
            jobs.stop_job(first)
        finished.set()

    threading.Thread(target=stop, daemon=True).start()
    _wait_for(lambda: (engine / "first.signalled").exists())
    if close:
        assert not finished.is_set()
        assert jobs.read_job(second)["status"] == "stopped"
        assert jobs.active_count() == 1
    else:
        assert finished.wait(timeout=5), "stop_job blocked waiting for the subprocess"
        assert jobs.read_job(second)["status"] == "queued"
        assert jobs.active_count() == 2
    stopping = jobs.read_job(first)
    assert stopping["status"] == "running"
    assert stopping["stop_requested"] is True
    assert jobs.running_count() == 1
    assert not (engine / "second.started").exists()
    with pytest.raises(ValueError, match="輸出位置已有排隊或執行中的任務"):
        jobs.create_job(jobs.resume_request(stopping))

    (engine / "first.release").touch()
    if close:
        assert finished.wait(timeout=5)
        assert counts == [2]
        assert not (jobs.job_dir(second) / "log.txt").exists()
    else:
        _wait_started(engine, "second", second)
        (engine / "second.release").touch()
    stopped = jobs.read_job(first)
    assert stopped["status"] == "stopped"
    assert stopped["exit_code"] == 7
    assert "stop_requested" not in stopped
    assert not Path(f"/proc/{pid}").exists()
    _wait_for(lambda: jobs.active_count() == 0)


def test_restart_marks_queued_and_stopping_jobs_interrupted(engine: Path) -> None:
    queued = {"id": "queued", "status": "queued"}
    stopping = {"id": "stopping", "status": "running", "stop_requested": True}
    completed = {"id": "completed", "status": "done"}
    for job in (queued, stopping, completed):
        jobs.job_dir(job["id"]).mkdir(parents=True, exist_ok=True)
        jobs.write_job(job)

    jobs.mark_interrupted()

    assert jobs.read_job("queued") == {"id": "queued", "status": "interrupted"}
    assert jobs.read_job("stopping") == {"id": "stopping", "status": "interrupted"}
    assert jobs.read_job("completed") == completed
    assert jobs.active_count() == 0
