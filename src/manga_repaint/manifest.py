from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
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
    error TEXT,
    ingest_progress_json TEXT NOT NULL DEFAULT '{}'
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
    asset_revision TEXT,
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
CREATE TABLE IF NOT EXISTS semantic_masks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    version TEXT NOT NULL,
    descriptor_json TEXT NOT NULL,
    confidence_path TEXT,
    uncertain_path TEXT,
    cached_at TEXT NOT NULL,
    UNIQUE(page_id, provider, version)
);
CREATE TABLE IF NOT EXISTS identity_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    identity_id TEXT NOT NULL,
    label TEXT NOT NULL,
    region TEXT NOT NULL,
    color TEXT,
    shadow_color TEXT,
    confidence REAL NOT NULL DEFAULT 0,
    locked INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, identity_id, region)
);
CREATE TABLE IF NOT EXISTS mask_corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    corrections_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_checkpoints (
    job_id TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    current INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    bytes_processed INTEGER NOT NULL DEFAULT 0,
    bytes_total INTEGER NOT NULL DEFAULT 0,
    current_file TEXT,
    latest_message TEXT,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', '3');
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
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "ingest_progress_json" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN ingest_progress_json TEXT NOT NULL DEFAULT '{}'"
                )
            page_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(pages)").fetchall()
            }
            if "asset_revision" not in page_columns:
                connection.execute("ALTER TABLE pages ADD COLUMN asset_revision TEXT")
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('schema_version','3') "
                "ON CONFLICT(key) DO UPDATE SET value=CASE "
                "WHEN CAST(metadata.value AS INTEGER) < 3 THEN '3' ELSE metadata.value END"
            )

    @property
    def schema_version(self) -> int:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
        try:
            return int(row["value"]) if row else 1
        except (TypeError, ValueError):
            return 1

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

    def backup_to(self, destination: Path) -> None:
        """Create a consistent SQLite backup, including committed WAL content."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            source = sqlite3.connect(self.path, timeout=30)
            target = sqlite3.connect(destination, timeout=30)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()

    def apply_repaired_outputs(
        self,
        job_id: str,
        units: list[dict[str, Any]],
        pages: list[dict[str, Any]],
        *,
        backup_path: Path,
    ) -> None:
        """Commit a fully staged repair in one manifest transaction."""
        with self._lock, self.connect() as connection:
            for unit in units:
                qa = unit["qa"]
                if not isinstance(qa, QAResult) or not qa.passed:
                    raise ValueError("A staged unit does not have passing QA")
                qa_data = qa.to_json_dict()
                qa_data["repaired"] = True
                cursor = connection.execute(
                    """
                    UPDATE units SET status='qa_passed',generated_path=?,final_path=?,
                      qa_json=?,error=NULL WHERE id=?
                    """,
                    (
                        str(unit["generated_path"]) if unit["generated_path"] else None,
                        str(unit["final_path"]),
                        json.dumps(qa_data, ensure_ascii=False),
                        int(unit["id"]),
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown repair unit: {unit['id']}")
            for page in pages:
                cursor = connection.execute(
                    """
                    UPDATE pages SET status='qa_passed',output_path=?,asset_revision=?
                    WHERE id=? AND job_id=?
                    """,
                    (
                        str(page["output_path"]),
                        str(page["asset_revision"]),
                        int(page["id"]),
                        job_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"Unknown repair page: {page['id']}")
            now = _now()
            connection.execute(
                "UPDATE jobs SET updated_at=?,error=NULL WHERE id=?", (now, job_id)
            )
            connection.execute(
                "INSERT INTO events(job_id,created_at,kind,payload_json) VALUES(?,?,?,?)",
                (
                    job_id,
                    now,
                    "results_repaired",
                    json.dumps(
                        {
                            "units": len(units),
                            "pages": len(pages),
                            "backup_path": str(backup_path),
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

    def create_job(self, job_id: str, spec: JobSpec) -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT INTO jobs(id,status,spec_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (job_id, JobStatus.CREATED.value, json.dumps(spec.to_json_dict()), now, now),
            )
        self.add_event(job_id, "job_created", {"status": JobStatus.CREATED.value})

    def set_ingest_progress(self, job_id: str, progress: dict[str, Any]) -> None:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET ingest_progress_json=?,updated_at=? WHERE id=?",
                (json.dumps(progress, ensure_ascii=False), _now(), job_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown job: {job_id}")
            connection.execute(
                """
                INSERT INTO ingest_checkpoints(
                    job_id,stage,current,total,bytes_processed,bytes_total,current_file,
                    latest_message,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    stage=excluded.stage,current=excluded.current,total=excluded.total,
                    bytes_processed=excluded.bytes_processed,bytes_total=excluded.bytes_total,
                    current_file=excluded.current_file,latest_message=excluded.latest_message,
                    updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    str(progress.get("stage") or "ingesting"),
                    int(progress.get("current") or 0),
                    int(progress.get("total") or 0),
                    int(progress.get("bytes_processed") or 0),
                    int(progress.get("bytes_total") or 0),
                    progress.get("current_file"),
                    progress.get("latest_message"),
                    _now(),
                ),
            )

    def ingest_progress(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT ingest_progress_json FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown job: {job_id}")
        try:
            value = json.loads(row["ingest_progress_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def ingest_checkpoint(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingest_checkpoints WHERE job_id=?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

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

    def save_semantic_mask(
        self,
        page_id: int,
        *,
        provider: str,
        version: str,
        descriptor: dict[str, Any],
        confidence_path: Path | None = None,
        uncertain_path: Path | None = None,
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO semantic_masks(
                    page_id,provider,version,descriptor_json,confidence_path,uncertain_path,cached_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(page_id,provider,version) DO UPDATE SET
                  descriptor_json=excluded.descriptor_json,
                  confidence_path=excluded.confidence_path,
                  uncertain_path=excluded.uncertain_path,
                  cached_at=excluded.cached_at
                """,
                (
                    page_id,
                    provider,
                    version,
                    json.dumps(descriptor, ensure_ascii=False),
                    str(confidence_path) if confidence_path else None,
                    str(uncertain_path) if uncertain_path else None,
                    _now(),
                ),
            )

    def semantic_mask(self, page_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM semantic_masks WHERE page_id=? ORDER BY id DESC LIMIT 1",
                (page_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["descriptor"] = json.loads(result.pop("descriptor_json") or "{}")
        return result

    def semantic_masks_for_job(self, job_id: str) -> dict[int, dict[str, Any]]:
        """Return the newest persisted semantic descriptor for each page."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT sm.*, p.page_index
                FROM semantic_masks sm JOIN pages p ON p.id=sm.page_id
                WHERE p.job_id=?
                ORDER BY sm.id
                """,
                (job_id,),
            ).fetchall()
        result: dict[int, dict[str, Any]] = {}
        for row in rows:
            item = dict(row)
            page_index = int(item.pop("page_index"))
            item["descriptor"] = json.loads(item.pop("descriptor_json") or "{}")
            result[page_index] = item
        return result

    def save_mask_correction(self, page_id: int, corrections: list[dict[str, Any]]) -> int:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO mask_corrections(page_id,corrections_json,created_at) VALUES(?,?,?)",
                (page_id, json.dumps(corrections, ensure_ascii=False), _now()),
            )
        return int(cursor.lastrowid)

    def mask_corrections(self, page_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT corrections_json FROM mask_corrections WHERE page_id=? ORDER BY id",
                (page_id,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            value = json.loads(row["corrections_json"] or "[]")
            if isinstance(value, list):
                result.extend(item for item in value if isinstance(item, dict))
        return result

    def save_identity(
        self,
        job_id: str,
        *,
        identity_id: str,
        label: str,
        region: str,
        color: str | None,
        shadow_color: str | None,
        confidence: float,
        locked: bool,
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO identity_records(
                    job_id,identity_id,label,region,color,shadow_color,confidence,locked,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id,identity_id,region) DO UPDATE SET
                  label=excluded.label,color=excluded.color,shadow_color=excluded.shadow_color,
                  confidence=excluded.confidence,locked=excluded.locked,updated_at=excluded.updated_at
                """,
                (
                    job_id,
                    identity_id,
                    label,
                    region,
                    color,
                    shadow_color,
                    max(0.0, min(1.0, confidence)),
                    int(locked),
                    _now(),
                ),
            )

    def identities(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM identity_records WHERE job_id=? ORDER BY identity_id,region",
                (job_id,),
            ).fetchall()
        result = [dict(row) for row in rows]
        for item in result:
            item["locked"] = bool(item["locked"])
        return result

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

    def touch_job(self, job_id: str) -> None:
        """Refresh the worker heartbeat without changing the public status."""
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET updated_at=? WHERE id=? AND status IN (?,?,?)",
                (
                    _now(),
                    job_id,
                    JobStatus.INGESTING.value,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )

    def mark_unit_running(self, unit_id: int) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE units SET status=?,attempt=attempt+1,started_at=?,error=NULL WHERE id=?
                """,
                (UnitStatus.RUNNING.value, _now(), unit_id),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
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
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
            )

    def finish_bypassed_unit(
        self, unit_id: int, final_path: Path, qa: QAResult
    ) -> None:
        """Finish a source-pass-through unit without a generated asset."""
        status = UnitStatus.QA_PASSED if qa.passed else UnitStatus.QA_FAILED
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE units SET status=?,generated_path=NULL,final_path=?,qa_json=?,finished_at=?
                WHERE id=?
                """,
                (
                    status.value,
                    str(final_path),
                    json.dumps(qa.to_json_dict(), ensure_ascii=False),
                    _now(),
                    unit_id,
                ),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
            )

    def replace_unit_output(
        self, unit_id: int, generated_path: Path, final_path: Path
    ) -> None:
        """Update a deterministic recomposition without changing QA state.

        Result repair must not turn a previously accepted unit into a failed
        one just because the repair worker uses a different diagnostic
        threshold. The original QA record remains the source of truth while
        the generated and final asset paths are replaced atomically in the
        manifest transaction.
        """
        with self._lock, self.connect() as connection:
            existing = connection.execute(
                "SELECT qa_json FROM units WHERE id=?", (unit_id,)
            ).fetchone()
            try:
                qa_data = json.loads(existing["qa_json"] or "{}") if existing else {}
            except (TypeError, json.JSONDecodeError):
                qa_data = {}
            if not isinstance(qa_data, dict):
                qa_data = {}
            qa_data["repaired"] = True
            connection.execute(
                """
                UPDATE units SET generated_path=?,final_path=?,qa_json=?,finished_at=?
                WHERE id=?
                """,
                (
                    str(generated_path),
                    str(final_path),
                    json.dumps(qa_data, ensure_ascii=False),
                    _now(),
                    unit_id,
                ),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=? )
                """,
                (_now(), unit_id),
            )

    def accept_repaired_unit(
        self,
        unit_id: int,
        generated_path: Path | None,
        final_path: Path,
        qa: QAResult,
    ) -> None:
        """Accept a repaired asset without counting repair time as inference time.

        Result repair runs on the CPU after the model has already generated an
        asset.  Reusing ``finish_unit`` would replace the original finish time
        and make the UI report a false, multi-hour seconds-per-megapixel rate.
        Keep the historical timing while recording the new QA decision.
        """
        with self._lock, self.connect() as connection:
            qa_data = qa.to_json_dict()
            qa_data["repaired"] = True
            connection.execute(
                """
                UPDATE units SET status='qa_passed',generated_path=?,final_path=?,
                  qa_json=?,error=NULL
                WHERE id=?
                """,
                (
                    str(generated_path) if generated_path else None,
                    str(final_path),
                    json.dumps(qa_data, ensure_ascii=False),
                    unit_id,
                ),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
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
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
            )

    def defer_unit(self, unit_id: int, error: str) -> None:
        """Return a unit to pending when the shared model is unavailable."""
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                UPDATE units SET status='pending',error=?,finished_at=NULL,started_at=NULL
                WHERE id=?
                """,
                (error, unit_id),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?)
                """,
                (_now(), unit_id),
            )

    def reset_running_unit(self, unit_id: int, error: str | None = None) -> bool:
        """Return an interrupted unit to pending without counting a failure."""
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE units SET status='pending',generated_path=NULL,final_path=NULL,
                  qa_json=NULL,error=?,started_at=NULL,finished_at=NULL
                WHERE id=? AND status='running'
                """,
                (error, unit_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    UPDATE pages SET status='pending',output_path=NULL
                    WHERE id=(SELECT page_id FROM units WHERE id=?)
                    """,
                    (unit_id,),
                )
                connection.execute(
                    """
                    UPDATE jobs SET updated_at=?
                    WHERE id=(
                        SELECT p.job_id FROM pages p JOIN units u ON u.page_id=p.id WHERE u.id=?
                    )
                    """,
                    (_now(), unit_id),
                )
        return bool(cursor.rowcount)

    def reset_running_units(
        self, job_id: str, error: str | None = None
    ) -> int:
        """Return stale running units to pending for a non-running job.

        A pause request can arrive after the worker has left the process but
        before its final unit checkpoint is persisted.  Only units belonging
        to the requested job are touched, and completed outputs are never
        changed.  The operation is idempotent, so it is safe to run during
        startup recovery or when a user presses pause again.
        """
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id,page_id FROM units
                WHERE status='running'
                  AND page_id IN (SELECT id FROM pages WHERE job_id=?)
                """,
                (job_id,),
            ).fetchall()
            if not rows:
                return 0
            connection.execute(
                """
                UPDATE units SET status='pending',generated_path=NULL,final_path=NULL,
                  qa_json=NULL,error=?,started_at=NULL,finished_at=NULL
                WHERE status='running'
                  AND page_id IN (SELECT id FROM pages WHERE job_id=?)
                """,
                (error, job_id),
            )
            page_ids = sorted({int(row["page_id"]) for row in rows})
            connection.executemany(
                "UPDATE pages SET status='pending',output_path=NULL WHERE id=?",
                ((page_id,) for page_id in page_ids),
            )
            connection.execute(
                "UPDATE jobs SET updated_at=? WHERE id=?",
                (_now(), job_id),
            )
        self.add_event(
            job_id,
            "stale_units_reset",
            {"count": len(rows), "status": "pending"},
        )
        return len(rows)

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

    def recover_interrupted(
        self, job_id: str, stale_after_seconds: float | None = None
    ) -> bool:
        with self._lock, self.connect() as connection:
            job = connection.execute(
                "SELECT status,updated_at FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            if job is None or job["status"] != JobStatus.RUNNING.value:
                return False
            if stale_after_seconds is not None:
                try:
                    age = (
                        datetime.now(UTC) - datetime.fromisoformat(str(job["updated_at"]))
                    ).total_seconds()
                except (TypeError, ValueError):
                    age = float("inf")
                if age < max(0.0, float(stale_after_seconds)):
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

    def finish_page(
        self, page_id: int, output_path: Path, asset_revision: str | None = None
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "UPDATE pages SET status='qa_passed',output_path=?,asset_revision=? WHERE id=?",
                (str(output_path), asset_revision, page_id),
            )
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT job_id FROM pages WHERE id=?)
                """,
                (_now(), page_id),
            )

    def set_page_status(self, page_id: int, status: str) -> None:
        with self._lock, self.connect() as connection:
            connection.execute("UPDATE pages SET status=? WHERE id=?", (status, page_id))
            connection.execute(
                """
                UPDATE jobs SET updated_at=?
                WHERE id=(SELECT job_id FROM pages WHERE id=?)
                """,
                (_now(), page_id),
            )

    def set_page_asset_revision(self, page_id: int, asset_revision: str) -> None:
        """Persist the version shared by a page's preview and derived assets."""
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "UPDATE pages SET asset_revision=? WHERE id=?",
                (str(asset_revision), page_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown page: {page_id}")

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

    def page_by_index(self, job_id: str, page_index: int) -> dict[str, Any]:
        """Read one page row without scanning the rest of the book."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM pages WHERE job_id=? AND page_index=?",
                (job_id, page_index),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown page {page_index} for job {job_id}")
        return dict(row)

    def page_asset_revision(self, page_id: int) -> str | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT asset_revision FROM pages WHERE id=?", (page_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown page: {page_id}")
        value = row["asset_revision"]
        return str(value) if value else None

    def page_units(self, page_id: int) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM units WHERE page_id=? ORDER BY unit_index", (page_id,)
                ).fetchall()
            ]

    def page_units_for_job(self, job_id: str) -> dict[int, list[dict[str, Any]]]:
        """Read every unit for a job with one query."""
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT u.*, p.page_index
                FROM units u JOIN pages p ON p.id=u.page_id
                WHERE p.job_id=?
                ORDER BY p.page_index,u.unit_index
                """,
                (job_id,),
            ).fetchall()
        result: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            item = dict(row)
            page_index = int(item.pop("page_index"))
            result.setdefault(page_index, []).append(item)
        return result

    def progress(self, job_id: str) -> dict[str, Any]:
        with self.connect() as connection:
            job_row = connection.execute(
                "SELECT status,created_at,updated_at,ingest_progress_json FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            unit_rows = connection.execute(
                """
                SELECT u.status,u.started_at,u.finished_at,u.width,u.height,u.qa_json,
                       p.page_index,u.unit_index
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
        terminal_job = job_row["status"] in {
            "needs_attention",
            "failed",
            "cancelled",
        }
        failed = sum(
            row["status"] in ({"failed", "qa_failed"} if terminal_job else {"failed"})
            for row in unit_rows
        )
        running = next((row for row in unit_rows if row["status"] == "running"), None)
        try:
            ingest = json.loads(job_row["ingest_progress_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            ingest = {}
        if not isinstance(ingest, dict):
            ingest = {}
        durations_per_mp: list[float] = []
        for row in unit_rows:
            if not row["started_at"] or not row["finished_at"] or row["status"] != "qa_passed":
                continue
            try:
                qa_data = json.loads(row["qa_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                qa_data = {}
            if isinstance(qa_data, dict) and qa_data.get("repaired"):
                continue
            started = datetime.fromisoformat(row["started_at"])
            finished = datetime.fromisoformat(row["finished_at"])
            megapixels = max(0.01, int(row["width"]) * int(row["height"]) / 1_000_000)
            durations_per_mp.append(max(0.0, (finished - started).total_seconds()) / megapixels)
        # Use the true median of the most recent eight successful units.  The
        # previous upper-middle shortcut biased even-sized samples toward a
        # slower request and made the ETA jump after a fast page completed.
        recent = durations_per_mp[-8:]
        seconds_per_mp = float(median(recent)) if len(recent) >= 2 else None
        pending_mp = sum(
            int(row["width"]) * int(row["height"]) / 1_000_000
            for row in unit_rows
            if row["status"] != "qa_passed"
        )
        eta = round(seconds_per_mp * pending_mp) if seconds_per_mp is not None else None
        ready_pages = [int(row["page_index"]) for row in page_rows if row["status"] == "qa_passed"]
        current_page = int(running["page_index"]) if running else ingest.get("current_page")
        current_unit = int(running["unit_index"]) if running else ingest.get("current_unit")
        page_states: list[dict[str, Any]] = []
        with self.connect() as connection:
            page_state_rows = connection.execute(
                """
                SELECT p.page_index,p.status AS page_status,
                       COUNT(u.id) AS total_units,
                       SUM(CASE WHEN u.status='qa_passed' THEN 1 ELSE 0 END) AS completed_units,
                       MAX(u.error) AS error
                FROM pages p LEFT JOIN units u ON u.page_id=p.id
                WHERE p.job_id=? GROUP BY p.id ORDER BY p.page_index
                """,
                (job_id,),
            ).fetchall()
        for row in page_state_rows:
            page_states.append(
                {
                    "page_index": int(row["page_index"]),
                    "status": str(row["page_status"]),
                    "completed_units": int(row["completed_units"] or 0),
                    "total_units": int(row["total_units"] or 0),
                    "error": row["error"],
                }
            )
        known_total_pages = int(ingest.get("total_pages") or 0)
        total_pages = max(len(page_rows), known_total_pages)
        status = str(job_row["status"])
        # Terminal jobs have no active cursor.  Clearing the page/unit fields
        # keeps maintenance gates and reconnecting clients from mistaking the
        # last processed page for work that is still in flight.
        if status in {
            JobStatus.COMPLETED.value,
            JobStatus.NEEDS_ATTENTION.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        }:
            current_page = None
            current_unit = None
        stage_by_status = {
            JobStatus.QUEUED.value: "queued",
            JobStatus.RUNNING.value: "generating",
            JobStatus.WAITING_MODEL.value: "waiting_model",
            JobStatus.PAUSED.value: "paused",
            JobStatus.COMPLETED.value: "completed",
            JobStatus.NEEDS_ATTENTION.value: "needs_attention",
            JobStatus.FAILED.value: "failed",
            JobStatus.CANCELLED.value: "cancelled",
        }
        # A process can be stopped between the model call and the unit
        # checkpoint, leaving one unit marked ``running`` while the job itself
        # is already paused after recovery.  The persisted job status is the
        # user-visible truth in that case, otherwise the UI reports a paused
        # task as if it were still generating.
        stage = (
            "generating"
            if running and status == JobStatus.RUNNING.value
            else stage_by_status.get(status, str(ingest.get("stage") or status))
        )
        stage_percent = float(ingest.get("stage_percent") or 0.0)
        if job_row["status"] == JobStatus.READY.value:
            stage_percent = 100.0
        elif job_row["status"] not in {JobStatus.INGESTING.value, JobStatus.CREATED.value}:
            stage_percent = round(completed / total * 100, 1) if total else 0.0
        latest_message = ingest.get("latest_message")
        if running:
            latest_message = (
                f"正在处理第 {int(running['page_index']) + 1} 页"
                f" · 第 {int(running['unit_index']) + 1} 个处理单元"
            )
        elif status == JobStatus.QUEUED.value:
            latest_message = "已进入处理队列，等待 GPU"
        elif status == JobStatus.WAITING_MODEL.value:
            latest_message = "等待本地模型服务恢复"
        elif status == JobStatus.PAUSED.value:
            latest_message = "任务已暂停，可以继续处理"
        elif status == JobStatus.COMPLETED.value:
            latest_message = "整本漫画已完成，可以导出"
        elif status == JobStatus.NEEDS_ATTENTION.value:
            latest_message = "部分页面需要检查或重试"
        elif status == JobStatus.FAILED.value:
            latest_message = "任务处理失败，请查看日志"
        elif status == JobStatus.CANCELLED.value:
            latest_message = "任务已取消"
        elapsed = max(
            0,
            round(
                (datetime.now(UTC) - datetime.fromisoformat(job_row["created_at"])).total_seconds()
            ),
        )
        return {
            "stage": stage,
            "stage_percent": round(max(0.0, min(100.0, stage_percent)), 1),
            "completed_units": completed,
            "total_units": total,
            "failed_units": failed,
            "completed_pages": len(ready_pages),
            "total_pages": total_pages,
            "current_page": current_page,
            "percent": round(completed / total * 100, 1) if total else 0.0,
            "seconds_per_megapixel": seconds_per_mp,
            "eta_seconds": eta,
            "elapsed_seconds": elapsed,
            "ready_page_indices": ready_pages,
            "bytes_processed": int(ingest.get("bytes_processed") or 0),
            "bytes_total": int(ingest.get("bytes_total") or 0),
            "current_file": ingest.get("current_file"),
            "current_unit": current_unit,
            "latest_message": latest_message,
            "page_states": page_states,
        }
