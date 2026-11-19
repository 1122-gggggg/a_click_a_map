"""Incremental job logs preserve byte offsets and bound each poll's I/O."""
from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))

import jobs


JOB_ID = "job1"
POLL_BYTES = 64 * 1024


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(jobs, "JOBS", tmp_path / "jobs")
    path = jobs.job_dir(JOB_ID) / "log.txt"
    path.parent.mkdir(parents=True)
    return path


def test_backfill_reads_only_from_saved_offset(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = b"old" * POLL_BYTES
    new = b"n" * POLL_BYTES + b"newest\n"
    offset = len(history)
    log_path.write_bytes(history + new)

    with log_path.open("rb", buffering=0) as stream, monkeypatch.context() as patch:
        reader = Mock(wraps=stream)
        patch.setattr(Path, "open", lambda *_args, **_kwargs: nullcontext(reader))
        first, offset = jobs.read_log(JOB_ID, offset)
        assert (first, offset) == ("n" * POLL_BYTES, len(history) + POLL_BYTES)
        second, offset = jobs.read_log(JOB_ID, offset)
        assert (second, offset) == ("newest\n", len(history) + len(new))
        assert jobs.read_log(JOB_ID, offset) == ("", offset)
        assert reader.read.call_args_list == [call(POLL_BYTES), call(len(b"newest\n"))]


def test_selection_tails_without_showing_a_cut_utf8_character(log_path: Path) -> None:
    tail = "n" * (POLL_BYTES - 2)
    data = ("old history\n" * POLL_BYTES + "🚀" + tail).encode()
    log_path.write_bytes(data)

    assert jobs.read_log(JOB_ID, None) == (tail, len(data))
    assert log_path.read_bytes() == data


@pytest.mark.parametrize("character", ["é", "圖", "🚀"])
def test_poll_boundary_round_trips_utf8_even_with_finished_job(
    log_path: Path, character: str
) -> None:
    prefix = "a" * (POLL_BYTES - 1)
    text = prefix + character + "tail\n"
    data = text.encode()
    log_path.write_bytes(data)

    first, offset = jobs.read_log(JOB_ID, 0, final=True)
    assert (first, offset) == (prefix, len(prefix))
    second, offset = jobs.read_log(JOB_ID, offset, final=True)
    assert first + second == text
    assert offset == len(data)


@pytest.mark.parametrize("character", ["é", "圖", "🚀"])
def test_partial_write_is_held_until_remaining_bytes_arrive(log_path: Path, character: str) -> None:
    encoded = character.encode()
    log_path.write_bytes(b"prefix " + encoded[:1])
    text, offset = jobs.read_log(JOB_ID, None)
    assert (text, offset) == ("prefix ", len(b"prefix "))

    for byte in encoded[1:-1]:
        assert jobs.read_log(JOB_ID, offset) == ("", offset)
        with log_path.open("ab") as stream:
            stream.write(bytes([byte]))
    assert jobs.read_log(JOB_ID, offset) == ("", offset)

    with log_path.open("ab") as stream:
        stream.write(encoded[-1:] + b"\n")
    assert jobs.read_log(JOB_ID, offset) == (character + "\n", log_path.stat().st_size)


def test_final_poll_flushes_incomplete_utf8_at_eof(log_path: Path) -> None:
    data = b"ok\n\xf0\x9f"
    log_path.write_bytes(data)

    assert jobs.read_log(JOB_ID, 0) == ("ok\n", 3)
    assert jobs.read_log(JOB_ID, 3, final=True) == ("\ufffd", len(data))
    assert jobs.read_log(JOB_ID, len(data), final=True) == ("", len(data))


def test_missing_and_truncated_logs_recover(log_path: Path) -> None:
    assert jobs.read_log(JOB_ID, None) == ("", None)
    assert jobs.read_log(JOB_ID, 50) == ("", 50)
    log_path.write_bytes(b"old\n" * POLL_BYTES)
    _, offset = jobs.read_log(JOB_ID, None)

    data = b"discarded" + b"r" * (POLL_BYTES - 6) + b"reset\n"
    log_path.write_bytes(data)
    assert jobs.read_log(JOB_ID, offset) == ("r" * (POLL_BYTES - 6) + "reset\n", len(data))
    log_path.write_bytes(b"")
    assert jobs.read_log(JOB_ID, len(data)) == ("", 0)
    log_path.write_bytes(b"new\n")
    assert jobs.read_log(JOB_ID, 0) == ("new\n", 4)
