from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .hashing import sha256_file


def _now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sources_sha256 ON sources(sha256);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL,
    queue_position INTEGER,
    archived_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(id);
"""


class Catalog:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "catalog.sqlite"
        self.import_root = root / "imports"
        self.import_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add_source(self, original_name: str, source_path: Path, kind: str) -> dict[str, Any]:
        digest = sha256_file(source_path)
        with self._lock, self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM sources WHERE sha256=?", (digest,)
            ).fetchone()
            if existing is not None:
                source_path.unlink(missing_ok=True)
                result = dict(existing)
                result["duplicate"] = True
                return result
            source_id = uuid.uuid4().hex
            target = self.import_root / f"{source_id}{source_path.suffix.casefold()}"
            source_path.replace(target)
            size = target.stat().st_size
            connection.execute(
                "INSERT INTO sources VALUES(?,?,?,?,?,?,?)",
                (source_id, original_name, str(target), kind, size, digest, _now()),
            )
        return {
            "id": source_id,
            "original_name": original_name,
            "kind": kind,
            "size": size,
            "sha256": digest,
            "duplicate": False,
        }

    def source(self, source_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown source: {source_id}")
        return dict(row)

    def group_images(self, source_ids: list[str], display_name: str) -> Path:
        group_id = uuid.uuid4().hex
        target = self.import_root / "groups" / group_id
        target.mkdir(parents=True, exist_ok=False)
        for index, source_id in enumerate(source_ids):
            source = self.source(source_id)
            source_path = Path(source["path"])
            suffix = source_path.suffix.casefold()
            shutil.copy2(source_path, target / f"page_{index:05d}{suffix}")
        (target / ".paneltone.json").write_text(
            json.dumps({"display_name": display_name, "sources": source_ids}, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def upsert_job(self, job_id: str, display_name: str, status: str) -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(id,display_name,status,created_at,updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  display_name=excluded.display_name,status=excluded.status,updated_at=excluded.updated_at
                """,
                (job_id, display_name, status, now, now),
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {"display_name", "status", "queue_position", "archived_at"}
        fields = {key: value for key, value in values.items() if key in allowed}
        if not fields:
            return
        fields["updated_at"] = _now()
        assignments = ",".join(f"{key}=?" for key in fields)
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?", [*fields.values(), job_id]
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job: {job_id}")

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def jobs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += (
            " ORDER BY CASE WHEN queue_position IS NULL THEN 1 ELSE 0 END,"
            " queue_position, updated_at DESC"
        )
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    def add_event(self, kind: str, payload: dict[str, Any], job_id: str | None = None) -> int:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events(job_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
                (job_id, kind, json.dumps(payload, ensure_ascii=False), _now()),
            )
            return int(cursor.lastrowid)

    def events_after(self, event_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE id>? ORDER BY id LIMIT ?", (event_id, limit)
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def latest_event_id(self) -> int:
        with self.connect() as connection:
            row = connection.execute("SELECT COALESCE(MAX(id), 0) AS id FROM events").fetchone()
        return int(row["id"])

    def delete_job(self, job_id: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("DELETE FROM events WHERE job_id=?", (job_id,))
            cursor = connection.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job: {job_id}")

    def prune_trash(self, days: int = 7) -> list[str]:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        with self.connect() as connection:
            return [
                row["id"]
                for row in connection.execute(
                    "SELECT id FROM jobs WHERE archived_at IS NOT NULL AND archived_at<?", (cutoff,)
                ).fetchall()
            ]
