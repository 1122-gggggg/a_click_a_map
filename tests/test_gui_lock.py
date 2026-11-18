"""Single-instance lock for the desktop app (no display needed)."""
from __future__ import annotations

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
