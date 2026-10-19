"""SQLite ledger with immutable raw records and run-scoped evidence."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .domain import (
    ArtifactRef,
    KeyframeRecord,
    PairEdgeRecord,
    PostSfmRole,
    RoleAssignment,
    SegmentRecord,
    record_dict,
)


_RUN_TABLES = frozenset(
    {"stages", "segments", "keyframes", "pairs", "evidence", "artifacts", "roles", "approvals"}
)
_RUN_QUERIES = {
    "stages": "SELECT * FROM stages WHERE run_id=? ORDER BY rowid",
    "segments": "SELECT * FROM segments WHERE run_id=? ORDER BY rowid",
    "keyframes": "SELECT * FROM keyframes WHERE run_id=? ORDER BY rowid",
    "pairs": "SELECT * FROM pairs WHERE run_id=? ORDER BY rowid",
    "evidence": "SELECT * FROM evidence WHERE run_id=? ORDER BY rowid",
    "artifacts": "SELECT * FROM artifacts WHERE run_id=? ORDER BY rowid",
    "roles": "SELECT * FROM roles WHERE run_id=? ORDER BY rowid",
    "approvals": "SELECT * FROM approvals WHERE run_id=? ORDER BY rowid",
}


class Ledger:
    """Single-writer provenance ledger.

    Raw sources are append-only at the database level. Derived records are
    versioned by ``run_id`` and may be deterministically replaced when the same
    stage fingerprint is recomputed.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = None if str(path) == ":memory:" else Path(path).expanduser().resolve()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def initialize(self) -> "Ledger":
        return self

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN")
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _migrate(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_sources(
                video_id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL DEFAULT 'video',
                source_uri TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                evaluation_role TEXT,
                payload TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS raw_sources_no_update
            BEFORE UPDATE ON raw_sources BEGIN
                SELECT RAISE(ABORT, 'raw_sources are append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS raw_sources_no_delete
            BEFORE DELETE ON raw_sources BEGIN
                SELECT RAISE(ABORT, 'raw_sources are append-only');
            END;

            CREATE TABLE IF NOT EXISTS stages(
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                fingerprint TEXT,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, stage)
            );
            CREATE TABLE IF NOT EXISTS segments(
                run_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, segment_id)
            );
            CREATE TABLE IF NOT EXISTS keyframes(
                run_id TEXT NOT NULL,
                keyframe_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, keyframe_id)
            );
            CREATE TABLE IF NOT EXISTS pairs(
                run_id TEXT NOT NULL,
                pair_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, pair_id)
            );
            CREATE TABLE IF NOT EXISTS evidence(
                run_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, evidence_id)
            );
            CREATE TABLE IF NOT EXISTS artifacts(
                run_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                uri TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                media_type TEXT,
                PRIMARY KEY(run_id, kind, uri)
            );
            CREATE TABLE IF NOT EXISTS roles(
                run_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                post_role TEXT,
                reason TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, segment_id)
            );
            CREATE TABLE IF NOT EXISTS approvals(
                run_id TEXT NOT NULL,
                decision_sha TEXT NOT NULL,
                approver TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(run_id, decision_sha)
            );
            """
        )
        self.connection.commit()

    def add_raw_source(
        self,
        video_id: str,
        source_uri: str,
        sha256: str,
        payload: Mapping[str, Any] | None = None,
        *,
        source_kind: str = "video",
        evaluation_role: str | None = None,
    ) -> None:
        self.connection.execute(
            "INSERT INTO raw_sources VALUES (?, ?, ?, ?, ?, ?)",
            (
                video_id,
                source_kind,
                source_uri,
                sha256,
                evaluation_role,
                _json(payload or {}),
            ),
        )
        self.connection.commit()

    def register_source(self, source: Mapping[str, Any]) -> None:
        source_id = str(source.get("source_id") or source.get("video_id") or "")
        if not source_id:
            raise ValueError("source requires source_id or video_id")
        self.add_raw_source(
            source_id,
            str(source.get("path") or source.get("source_uri") or ""),
            str(source.get("sha256") or source.get("content_fingerprint") or source_id),
            source,
            source_kind=str(source.get("source_kind") or "video"),
            evaluation_role=(
                None if source.get("evaluation_role") is None else str(source["evaluation_role"])
            ),
        )

    def add_segment(self, record: SegmentRecord, run_id: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO segments VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                record.segment_id,
                record.video_id,
                record.session_id,
                record.source_uri,
                _json(record_dict(record)),
            ),
        )
        self.connection.commit()

    def add_keyframe(self, record: KeyframeRecord, run_id: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO keyframes VALUES (?, ?, ?, ?)",
            (run_id, record.keyframe_id, record.segment_id, _json(record_dict(record))),
        )
        self.connection.commit()

    def add_pair(self, record: PairEdgeRecord, run_id: str) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO pairs VALUES (?, ?, ?)",
            (run_id, record.pair_id, _json(record_dict(record))),
        )
        self.connection.commit()

    def add_artifact(self, run_id: str, kind: str, artifact: ArtifactRef) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?, ?)",
            (run_id, kind, artifact.uri, artifact.sha256, artifact.media_type),
        )
        self.connection.commit()

    def add_role(
        self,
        run_id: str,
        segment_id: str,
        role: PostSfmRole | str,
        reason: str = "",
        payload: Mapping[str, Any] | RoleAssignment | None = None,
    ) -> None:
        role_value = PostSfmRole(role).value
        body = record_dict(payload) if isinstance(payload, RoleAssignment) else dict(payload or {})
        self.connection.execute(
            "INSERT OR REPLACE INTO roles VALUES (?, ?, ?, ?, ?)",
            (run_id, segment_id, role_value, reason, _json(body)),
        )
        self.connection.commit()

    def add_approval(
        self, run_id: str, decision_sha: str, approver: str, payload: Mapping[str, Any]
    ) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO approvals VALUES (?, ?, ?, ?)",
            (run_id, decision_sha, approver, _json(payload)),
        )
        self.connection.commit()

    def record_stage(self, run_id: str, stage: str, payload: Mapping[str, Any]) -> None:
        fingerprint = payload.get("fingerprint")
        self.connection.execute(
            "INSERT OR REPLACE INTO stages VALUES (?, ?, ?, ?)",
            (run_id, stage, fingerprint, _json(payload)),
        )
        self.connection.commit()

    def record_json(self, run_id: str, record_id: str, payload: Mapping[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?)",
            (run_id, record_id, _json(payload)),
        )
        self.connection.commit()

    def get_segments(self, run_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM segments WHERE run_id=? ORDER BY segment_id", (run_id,)
            )
        ]

    def rows(self, run_id: str, table: str) -> list[dict[str, Any]]:
        if table not in _RUN_TABLES:
            raise ValueError(f"unsupported export table {table!r}")
        records = [dict(row) for row in self.connection.execute(_RUN_QUERIES[table], (run_id,))]
        return [_canonical_export_row(row) for row in records]

    def export_jsonl(self, run_id: str, table: str, output: str | Path | None = None) -> Path:
        path = self._export_path(run_id, table, "jsonl", output)
        path.write_text(
            "".join(_json(row) + "\n" for row in self.rows(run_id, table)),
            encoding="utf-8",
        )
        return path

    def export_csv(self, run_id: str, table: str, output: str | Path | None = None) -> Path:
        rows = self.rows(run_id, table)
        path = self._export_path(run_id, table, "csv", output)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(buffer.getvalue(), encoding="utf-8")
        return path

    def _export_path(
        self,
        run_id: str,
        table: str,
        extension: str,
        output: str | Path | None,
    ) -> Path:
        if output is not None:
            path = Path(output)
        else:
            base = self.path.parent if self.path is not None else Path.cwd()
            path = base / "exports" / run_id / f"{table}.{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def _canonical_export_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.pop("payload", None)
    decoded = json.loads(payload) if payload else {}
    merged = {"run_id": row.get("run_id"), **decoded}
    for key, value in row.items():
        merged.setdefault(key, value)
    return merged


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


PipelineStore = Ledger

__all__ = ["Ledger", "PipelineStore"]
