"""Single-instance lock for the desktop app (no display needed)."""
from __future__ import annotations

import pytest
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs


def test_lock_acquire_and_release(tmp_path: Path) -> None:
    lock = tmp_path / "gui.lock"
    assert jobs.acquire_gui_lock(lock) is True
    assert int(json.loads(lock.read_text(encoding="utf-8"))["pid"]) == os.getpid()
    jobs.release_gui_lock(lock)
    assert not lock.exists()
    jobs.release_gui_lock(lock)  # already gone: no error


def test_lock_blocks_when_owner_alive(tmp_path: Path, monkeypatch) -> None:
    lock = tmp_path / "gui.lock"
    assert jobs.acquire_gui_lock(lock) is True
    monkeypatch.setattr(jobs, "_gui_owner_alive", lambda pid: True)
    assert jobs.acquire_gui_lock(lock) is False
    jobs.release_gui_lock(lock)


def test_stale_lock_is_reclaimed(tmp_path: Path) -> None:
    lock = tmp_path / "gui.lock"
    lock.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    assert jobs._gui_owner_alive(999999999) is False
    assert jobs.acquire_gui_lock(lock) is True
    jobs.release_gui_lock(lock)


def test_release_ignores_foreign_lock(tmp_path: Path) -> None:
    lock = tmp_path / "gui.lock"
    assert jobs.acquire_gui_lock(lock) is True
    lock.write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    jobs.release_gui_lock(lock)  # pid mismatch: must leave the file alone
    assert lock.exists()


def _gone_or_zombie(pid: int, timeout: float = 10.0) -> bool:
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[2]
        except OSError:
            return True
        if state == "Z":
            return True
        time.sleep(0.2)
    return False


def test_kill_tree_reaps_escaped_grandchildren() -> None:
    import subprocess
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
    import jobs
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time; "
         "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True); "
         "time.sleep(60)"],
        start_new_session=True)
    try:
        time.sleep(0.5)
        grandchildren = jobs._descendants(parent.pid)
        assert grandchildren, "expected at least the grandchild"
        jobs._kill_tree(parent.pid, timeout=5.0)
        assert parent.wait(timeout=10) != 0, "parent should have been signalled"
        for pid in grandchildren:
            assert _gone_or_zombie(pid), f"escaped child {pid} survived the kill tree"
    finally:
        try:
            parent.kill()
        except OSError:
            pass


def test_delete_job_removes_record_but_requires_finished(tmp_path, monkeypatch) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
    import jobs
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    with pytest.raises((OSError, ValueError)):
        jobs.delete_job("nosuchid")
    d = jobs.job_dir("abc123")
    d.mkdir(parents=True)
    (d / "job.json").write_text('{"id": "abc123", "status": "running"}', encoding="utf-8")
    with pytest.raises(ValueError, match="停止"):
        jobs.delete_job("abc123")
    assert d.exists()
    (d / "job.json").write_text('{"id": "abc123", "status": "failed"}', encoding="utf-8")
    jobs.delete_job("abc123")
    assert not d.exists()
