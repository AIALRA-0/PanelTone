from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import JobSpec, JobStatus, PanelBox, QAResult, UnitStatus

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    output_path TEXT,
    UNIQUE(job_id, page_index)
);

CREATE TABLE IF NOT EXISTS units (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    unit_index INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt INTEGER NOT NULL DEFAULT 0,
    engine TEXT NOT NULL,
    params_hash TEXT NOT NULL,
    source_path TEXT NOT NULL,
    mask_path TEXT,
    generated_path TEXT,
    final_path TEXT,
    qa_json TEXT,
    error TEXT,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE(page_id, unit_index)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_units_status ON units(status);
CREATE INDEX IF NOT EXISTS idx_pages_job ON pages(job_id, page_index);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def create_job(self, job_id: str, spec: JobSpec) -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO jobs(id,status,spec_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (job_id, JobStatus.CREATED.value, json.dumps(spec.to_json_dict()), now, now),
            )
        self.add_event(job_id, "job_created", {"status": JobStatus.CREATED.value})

    def set_job_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status=?,updated_at=?,error=? WHERE id=?",
                (status.value, _now(), error, job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job: {job_id}")
        self.add_event(job_id, "job_status", {"status": status.value, "error": error})

    def add_page(
        self,
        job_id: str,
        page_index: int,
        source_path: Path,
        source_sha256: str,
        width: int,
        height: int,
    ) -> int:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pages(job_id,page_index,source_path,source_sha256,width,height)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(job_id,page_index) DO UPDATE SET
                  source_path=excluded.source_path,
                  source_sha256=excluded.source_sha256,
                  width=excluded.width,
                  height=excluded.height
                """,
                (job_id, page_index, str(source_path), source_sha256, width, height),
            )
            row = connection.execute(
                "SELECT id FROM pages WHERE job_id=? AND page_index=?", (job_id, page_index)
            ).fetchone()
        return int(row["id"])

    def add_unit(
        self,
        page_id: int,
        unit_index: int,
        box: PanelBox,
        engine: str,
        params_hash: str,
        source_path: Path,
        mask_path: Path,
    ) -> int:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO units(
                    page_id,unit_index,x,y,width,height,engine,params_hash,source_path,mask_path
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(page_id,unit_index) DO UPDATE SET
                  x=excluded.x,y=excluded.y,width=excluded.width,height=excluded.height,
                  engine=excluded.engine,params_hash=excluded.params_hash,
                  source_path=excluded.source_path,mask_path=excluded.mask_path
                """,
                (
                    page_id,
                    unit_index,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    engine,
                    params_hash,
                    str(source_path),
                    str(mask_path),
                ),
            )
            row = connection.execute(
                "SELECT id FROM units WHERE page_id=? AND unit_index=?", (page_id, unit_index)
            ).fetchone()
        return int(row["id"])

    def pending_units(self, job_id: str) -> list[dict[str, Any]]:
        query = """
        SELECT u.*, p.page_index, p.job_id, p.source_path AS page_source_path,
               p.output_path AS page_output_path
        FROM units u JOIN pages p ON p.id=u.page_id
        WHERE p.job_id=? AND u.status NOT IN ('qa_passed')
        ORDER BY p.page_index,u.unit_index
        """
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, (job_id,)).fetchall()]

    def mark_unit_running(self, unit_id: int) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE units SET status=?,attempt=attempt+1,started_at=?,error=NULL WHERE id=?
                """,
                (UnitStatus.RUNNING.value, _now(), unit_id),
            )

    def finish_unit(
        self,
        unit_id: int,
        generated_path: Path,
        final_path: Path,
        qa: QAResult,
    ) -> None:
        status = UnitStatus.QA_PASSED if qa.passed else UnitStatus.QA_FAILED
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE units SET status=?,generated_path=?,final_path=?,qa_json=?,finished_at=?
                WHERE id=?
                """,
                (
                    status.value,
                    str(generated_path),
                    str(final_path),
                    json.dumps(qa.to_json_dict()),
                    _now(),
                    unit_id,
                ),
            )

    def rename_job(self, job_id: str, display_name: str) -> None:
        with self._lock, self.connect() as connection:
            row = connection.execute("SELECT spec_json FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown job: {job_id}")
            spec = json.loads(row["spec_json"])
            spec["display_name"] = display_name
            connection.execute(
                "UPDATE jobs SET spec_json=?,updated_at=? WHERE id=?",
                (json.dumps(spec, ensure_ascii=False), _now(), job_id),
            )

    def set_queued(self, job_id: str) -> None:
        self.set_job_status(job_id, JobStatus.QUEUED)

    def fail_unit(self, unit_id: int, error: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE units SET status=?,error=?,finished_at=? WHERE id=?",
                (UnitStatus.FAILED.value, error, _now(), unit_id),
            )

    def retry_page(self, job_id: str, page_index: int) -> int:
        with self._lock, self.connect() as connection:
            page = connection.execute(
                "SELECT id FROM pages WHERE job_id=? AND page_index=?",
                (job_id, page_index),
            ).fetchone()
            if page is None:
                raise KeyError(f"Unknown page: {page_index}")
            cursor = connection.execute(
                """
                UPDATE units SET status='pending',attempt=0,generated_path=NULL,
                  final_path=NULL,qa_json=NULL,error=NULL,started_at=NULL,finished_at=NULL
                WHERE page_id=? AND status!='qa_passed'
                """,
                (int(page["id"]),),
            )
            connection.execute(
                "UPDATE pages SET status='pending',output_path=NULL WHERE id=?",
                (int(page["id"]),),
            )
        if cursor.rowcount == 0:
            raise ValueError("这一页没有需要重试的处理单元")
        self.set_job_status(job_id, JobStatus.READY)
        return int(cursor.rowcount)

    def recover_interrupted(self, job_id: str) -> bool:
        with self._lock, self.connect() as connection:
            job = connection.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["status"] != JobStatus.RUNNING.value:
                return False
            connection.execute(
                "UPDATE units SET status='pending',error=NULL WHERE status='running' "
                "AND page_id IN (SELECT id FROM pages WHERE job_id=?)",
                (job_id,),
            )
            connection.execute(
                "UPDATE jobs SET status=?,updated_at=?,error=? WHERE id=?",
                (
                    JobStatus.PAUSED.value,
                    _now(),
                    "Recovered after the previous local app process stopped",
                    job_id,
                ),
            )
        self.add_event(job_id, "job_recovered", {"status": JobStatus.PAUSED.value})
        return True

    def finish_page(self, page_id: int, output_path: Path) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE pages SET status='qa_passed',output_path=? WHERE id=?",
                (str(output_path), page_id),
            )

    def add_event(self, job_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO events(job_id,created_at,kind,payload_json) VALUES(?,?,?,?)",
                (job_id, _now(), kind, json.dumps(payload, ensure_ascii=False)),
            )

    def summary(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            counts = connection.execute(
                """
                SELECT u.status,COUNT(*) AS count FROM units u
                JOIN pages p ON p.id=u.page_id WHERE p.job_id=? GROUP BY u.status
                """,
                (job_id,),
            ).fetchall()
            pages = connection.execute(
                "SELECT COUNT(*) AS count FROM pages WHERE job_id=?", (job_id,)
            ).fetchone()["count"]
        result = dict(job)
        result["spec"] = json.loads(result.pop("spec_json"))
        result["unit_counts"] = {row["status"]: row["count"] for row in counts}
        result["page_count"] = pages
        return result

    def pages(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM pages WHERE job_id=? ORDER BY page_index", (job_id,)
                ).fetchall()
            ]

    def page_units(self, page_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM units WHERE page_id=? ORDER BY unit_index", (page_id,)
                ).fetchall()
            ]

    def progress(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            job_row = connection.execute(
                "SELECT status,created_at,updated_at FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            unit_rows = connection.execute(
                """
                SELECT u.status,u.started_at,u.finished_at,u.width,u.height,p.page_index
                FROM units u JOIN pages p ON p.id=u.page_id
                WHERE p.job_id=? ORDER BY p.page_index,u.unit_index
                """,
                (job_id,),
            ).fetchall()
            page_rows = connection.execute(
                "SELECT page_index,status FROM pages WHERE job_id=? ORDER BY page_index", (job_id,)
            ).fetchall()
        total = len(unit_rows)
        completed = sum(row["status"] == UnitStatus.QA_PASSED.value for row in unit_rows)
        failed = sum(row["status"] in {"failed", "qa_failed"} for row in unit_rows)
        running = next((row for row in unit_rows if row["status"] == "running"), None)
        durations_per_mp: list[float] = []
        for row in unit_rows:
            if not row["started_at"] or not row["finished_at"] or row["status"] != "qa_passed":
                continue
            started = datetime.fromisoformat(row["started_at"])
            finished = datetime.fromisoformat(row["finished_at"])
            megapixels = max(0.01, int(row["width"]) * int(row["height"]) / 1_000_000)
            durations_per_mp.append(max(0.0, (finished - started).total_seconds()) / megapixels)
        recent = sorted(durations_per_mp[-8:])
        seconds_per_mp = recent[len(recent) // 2] if len(recent) >= 2 else None
        pending_mp = sum(
            int(row["width"]) * int(row["height"]) / 1_000_000
            for row in unit_rows
            if row["status"] != "qa_passed"
        )
        eta = round(seconds_per_mp * pending_mp) if seconds_per_mp is not None else None
        ready_pages = [int(row["page_index"]) for row in page_rows if row["status"] == "qa_passed"]
        current_page = int(running["page_index"]) if running else None
        elapsed = max(
            0,
            round(
                (datetime.now(UTC) - datetime.fromisoformat(job_row["created_at"])).total_seconds()
            ),
        )
        return {
            "stage": "generating" if running else job_row["status"],
            "completed_units": completed,
            "total_units": total,
            "failed_units": failed,
            "completed_pages": len(ready_pages),
            "total_pages": len(page_rows),
            "current_page": current_page,
            "percent": round(completed / total * 100, 1) if total else 0.0,
            "seconds_per_megapixel": seconds_per_mp,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "ready_page_indices": ready_pages,
        }
