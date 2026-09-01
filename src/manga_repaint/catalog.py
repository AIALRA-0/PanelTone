from __future__ import annotations

import json
import os
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
    folder_id TEXT,
    library_order INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS folders (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    name TEXT NOT NULL,
    sort_order INTEGER NOT NULL DEFAULT 0,
    archived_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES folders(id) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_folders_parent_sort ON folders(parent_id, sort_order, name);
CREATE TABLE IF NOT EXISTS uploads (
    upload_id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    relative_path TEXT,
    temp_path TEXT NOT NULL,
    total_bytes INTEGER,
    uploaded_bytes INTEGER NOT NULL DEFAULT 0,
    source_id TEXT,
    status TEXT NOT NULL DEFAULT 'uploading',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploads_updated ON uploads(updated_at);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(id);
CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operations_job ON operations(job_id, updated_at);
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
INSERT OR IGNORE INTO metadata(key, value) VALUES('schema_version', '2');
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
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Additive catalog migration with a recoverable backup."""
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            try:
                version = int(row["value"]) if row else 1
            except (TypeError, ValueError):
                version = 1
            columns = {
                str(item["name"])
                for item in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            source_columns = {
                str(item["name"])
                for item in connection.execute("PRAGMA table_info(sources)").fetchall()
            }
            required_columns = {"folder_id", "library_order"}
            required_source_columns = {"relative_path", "import_batch_id"}
            needs_migration = (
                version < 3
                or not required_columns.issubset(columns)
                or not required_source_columns.issubset(source_columns)
            )
            if not needs_migration:
                # ``SCHEMA`` has already run the idempotent table/index
                # creation above, so a fully migrated catalog needs no new
                # backup on every application start.
                return
            backup = self.path.with_name(f"{self.path.name}.pre-v3.bak")
            if self.path.is_file() and not backup.exists():
                shutil.copy2(self.path, backup)
            if "folder_id" not in columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN folder_id TEXT")
            if "library_order" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN library_order INTEGER NOT NULL DEFAULT 0"
                )
            if "relative_path" not in source_columns:
                connection.execute("ALTER TABLE sources ADD COLUMN relative_path TEXT")
            if "import_batch_id" not in source_columns:
                connection.execute("ALTER TABLE sources ADD COLUMN import_batch_id TEXT")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS folders ("
                "id TEXT PRIMARY KEY, parent_id TEXT, name TEXT NOT NULL, "
                "sort_order INTEGER NOT NULL DEFAULT 0, archived_at TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "FOREIGN KEY(parent_id) REFERENCES folders(id) ON DELETE RESTRICT)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_folders_parent_sort "
                "ON folders(parent_id, sort_order, name)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS uploads ("
                "upload_id TEXT PRIMARY KEY, original_name TEXT NOT NULL, "
                "relative_path TEXT, temp_path TEXT NOT NULL, total_bytes INTEGER, "
                "uploaded_bytes INTEGER NOT NULL DEFAULT 0, source_id TEXT, "
                "status TEXT NOT NULL DEFAULT 'uploading', created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploads_updated ON uploads(updated_at)"
            )
            connection.execute(
                "INSERT INTO metadata(key,value) VALUES('schema_version','3') "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
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

    def add_source(
        self,
        original_name: str,
        source_path: Path,
        kind: str,
        *,
        relative_path: str | None = None,
        import_batch_id: str | None = None,
    ) -> dict[str, Any]:
        digest = sha256_file(source_path)
        with self._lock, self.connect() as connection:
            existing = connection.execute(
                "SELECT * FROM sources WHERE sha256=?", (digest,)
            ).fetchone()
            if existing is not None:
                existing_path = Path(existing["path"])
                if existing_path.is_file():
                    source_path.unlink(missing_ok=True)
                    result = dict(existing)
                    result["duplicate"] = True
                    if relative_path is not None:
                        result["relative_path"] = relative_path
                    return result
                connection.execute("DELETE FROM sources WHERE id=?", (existing["id"],))
            source_id = uuid.uuid4().hex
            target = self.import_root / f"{source_id}{source_path.suffix.casefold()}"
            source_path.replace(target)
            size = target.stat().st_size
            connection.execute(
                "INSERT INTO sources"
                "(id,original_name,path,kind,size,sha256,created_at,relative_path,import_batch_id) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    source_id,
                    original_name,
                    str(target),
                    kind,
                    size,
                    digest,
                    _now(),
                    relative_path,
                    import_batch_id,
                ),
            )
        return {
            "id": source_id,
            "original_name": original_name,
            "kind": kind,
            "size": size,
            "sha256": digest,
            "duplicate": False,
            "relative_path": relative_path,
        }

    def save_upload(
        self,
        upload_id: str,
        *,
        original_name: str,
        relative_path: str | None,
        temp_path: Path,
        total_bytes: int | None,
        uploaded_bytes: int,
        status: str = "uploading",
        source_id: str | None = None,
    ) -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO uploads(
                    upload_id,original_name,relative_path,temp_path,total_bytes,
                    uploaded_bytes,source_id,status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(upload_id) DO UPDATE SET
                    original_name=excluded.original_name,
                    relative_path=excluded.relative_path,
                    temp_path=excluded.temp_path,
                    total_bytes=excluded.total_bytes,
                    uploaded_bytes=excluded.uploaded_bytes,
                    source_id=excluded.source_id,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    upload_id,
                    original_name,
                    relative_path,
                    str(temp_path),
                    total_bytes,
                    uploaded_bytes,
                    source_id,
                    status,
                    now,
                    now,
                ),
            )

    def upload(self, upload_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM uploads WHERE upload_id=?", (upload_id,)
            ).fetchone()
        return dict(row) if row else None

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
            destination = target / f"page_{index:05d}{suffix}"
            # Sources and virtual book directories normally share the same
            # data volume.  A hard link keeps a large batch responsive and
            # avoids doubling disk usage; copy only when the filesystem does
            # not support linking (for example a cross-volume import).
            try:
                os.link(source_path, destination)
            except (FileExistsError, OSError):
                shutil.copy2(source_path, destination)
        (target / ".paneltone.json").write_text(
            json.dumps({"display_name": display_name, "sources": source_ids}, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    def upsert_job(self, job_id: str, display_name: str, status: str) -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            existing = connection.execute(
                "SELECT id FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
            library_order = 0
            if existing is None:
                order_row = connection.execute(
                    "SELECT COALESCE(MAX(library_order), -1) + 1 AS next_order "
                    "FROM jobs WHERE folder_id IS NULL"
                ).fetchone()
                library_order = int(order_row["next_order"])
            connection.execute(
                """
                INSERT INTO jobs(id,display_name,status,library_order,created_at,updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  display_name=excluded.display_name,status=excluded.status,updated_at=excluded.updated_at
                """,
                (job_id, display_name, status, library_order, now, now),
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "display_name",
            "status",
            "queue_position",
            "archived_at",
            "folder_id",
            "library_order",
        }
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

    def create_folder(self, name: str, parent_id: str | None = None) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 120:
            raise ValueError("目录名称必须为 1 到 120 个字符")
        now = _now()
        folder_id = uuid.uuid4().hex
        with self._lock, self.connect() as connection:
            if parent_id is not None:
                parent = connection.execute(
                    "SELECT id,archived_at FROM folders WHERE id=?", (parent_id,)
                ).fetchone()
                if parent is None:
                    raise KeyError(f"Unknown folder: {parent_id}")
                if parent["archived_at"]:
                    raise ValueError("回收站中的目录不能新建子目录")
            duplicate = connection.execute(
                "SELECT id FROM folders WHERE COALESCE(parent_id,'')=COALESCE(?, '') "
                "AND name=? AND archived_at IS NULL",
                (parent_id, clean_name),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("同级目录已经存在同名目录")
            row = connection.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_order "
                "FROM folders WHERE COALESCE(parent_id,'')=COALESCE(?, '')",
                (parent_id,),
            ).fetchone()
            sort_order = int(row["next_order"])
            connection.execute(
                "INSERT INTO folders(id,parent_id,name,sort_order,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?)",
                (folder_id, parent_id, clean_name, sort_order, now, now),
            )
        return self.folder(folder_id) or {}

    def ensure_folder_path(self, parts: list[str]) -> str | None:
        """Create or reuse virtual folder nodes for an uploaded relative path."""
        parent_id: str | None = None
        for raw_part in parts:
            name = raw_part.strip()
            if not name:
                continue
            with self._lock, self.connect() as connection:
                row = connection.execute(
                    "SELECT * FROM folders WHERE COALESCE(parent_id,'')=COALESCE(?, '') "
                    "AND name=? AND archived_at IS NULL",
                    (parent_id, name),
                ).fetchone()
            if row is None:
                row = self.create_folder(name, parent_id)
            parent_id = str(row["id"])
        return parent_id

    def folder(self, folder_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM folders WHERE id=?", (folder_id,)
            ).fetchone()
        return dict(row) if row else None

    def folders(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM folders"
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY COALESCE(parent_id,''), sort_order, name COLLATE NOCASE"
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    def update_folder(
        self,
        folder_id: str,
        *,
        name: str | None = None,
        parent_id: str | None | object = ...,
    ) -> dict[str, Any]:
        current = self.folder(folder_id)
        if current is None:
            raise KeyError(f"Unknown folder: {folder_id}")
        clean_name = current["name"] if name is None else name.strip()
        if not clean_name or len(clean_name) > 120:
            raise ValueError("目录名称必须为 1 到 120 个字符")
        target_parent = current["parent_id"] if parent_id is ... else parent_id
        if target_parent == folder_id:
            raise ValueError("目录不能移动到自身")
        with self._lock, self.connect() as connection:
            if target_parent is not None:
                parent = connection.execute(
                    "SELECT id,archived_at FROM folders WHERE id=?", (target_parent,)
                ).fetchone()
                if parent is None:
                    raise KeyError(f"Unknown folder: {target_parent}")
                if parent["archived_at"]:
                    raise ValueError("不能移动到回收站目录")
                ancestor = target_parent
                while ancestor is not None:
                    if ancestor == folder_id:
                        raise ValueError("目录不能移动到自己的子目录")
                    row = connection.execute(
                        "SELECT parent_id FROM folders WHERE id=?", (ancestor,)
                    ).fetchone()
                    ancestor = row["parent_id"] if row else None
            duplicate = connection.execute(
                "SELECT id FROM folders WHERE id<>? "
                "AND COALESCE(parent_id,'')=COALESCE(?, '') AND name=? "
                "AND archived_at IS NULL",
                (folder_id, target_parent, clean_name),
            ).fetchone()
            if duplicate is not None:
                raise ValueError("同级目录已经存在同名目录")
            connection.execute(
                "UPDATE folders SET name=?, parent_id=?, updated_at=? WHERE id=?",
                (clean_name, target_parent, _now(), folder_id),
            )
        return self.folder(folder_id) or {}

    def archive_folder(self, folder_id: str) -> dict[str, Any]:
        current = self.folder(folder_id)
        if current is None:
            raise KeyError(f"Unknown folder: {folder_id}")
        now = _now()
        with self._lock, self.connect() as connection:
            descendants = connection.execute(
                """
                WITH RECURSIVE descendants(id) AS (
                    SELECT id FROM folders WHERE id=?
                    UNION ALL
                    SELECT folders.id FROM folders
                    JOIN descendants ON folders.parent_id=descendants.id
                )
                SELECT id FROM descendants
                """,
                (folder_id,),
            ).fetchall()
            descendant_ids = [str(row["id"]) for row in descendants]
            placeholders = ",".join("?" for _ in descendant_ids)
            active = connection.execute(
                f"SELECT COUNT(*) AS count FROM jobs WHERE folder_id IN ({placeholders}) "
                "AND archived_at IS NULL AND status IN "
                "('created','running','queued','ingesting','waiting_model')",
                descendant_ids,
            ).fetchone()
            if int(active["count"] or 0) > 0:
                raise ValueError("运行中的目录不能移到回收站")
            connection.execute(
                f"UPDATE folders SET archived_at=?, updated_at=? WHERE id IN ({placeholders})",
                [now, now, *descendant_ids],
            )
            # A folder archive is a library-level move.  Mark all contained
            # jobs as archived in the same transaction so the trash view never
            # exposes orphaned active children.  Restoring the folder restores
            # those jobs without changing their manifest processing status.
            connection.execute(
                f"UPDATE jobs SET archived_at=?, updated_at=? WHERE folder_id IN ({placeholders})",
                [now, now, *descendant_ids],
            )
        return self.folder(folder_id) or {}

    def restore_folder(self, folder_id: str) -> dict[str, Any]:
        current = self.folder(folder_id)
        if current is None:
            raise KeyError(f"Unknown folder: {folder_id}")
        with self._lock, self.connect() as connection:
            parent_id = current["parent_id"]
            if parent_id is not None:
                parent = connection.execute(
                    "SELECT archived_at FROM folders WHERE id=?", (parent_id,)
                ).fetchone()
                if parent is None or parent["archived_at"]:
                    raise ValueError("请先恢复父目录")
            descendants = connection.execute(
                """
                WITH RECURSIVE descendants(id) AS (
                    SELECT id FROM folders WHERE id=?
                    UNION ALL
                    SELECT folders.id FROM folders
                    JOIN descendants ON folders.parent_id=descendants.id
                )
                SELECT id FROM descendants
                """,
                (folder_id,),
            ).fetchall()
            descendant_ids = [str(row["id"]) for row in descendants]
            placeholders = ",".join("?" for _ in descendant_ids)
            now = _now()
            connection.execute(
                f"UPDATE folders SET archived_at=NULL, updated_at=? WHERE id IN ({placeholders})",
                [now, *descendant_ids],
            )
            connection.execute(
                f"UPDATE jobs SET archived_at=NULL, updated_at=? "
                f"WHERE folder_id IN ({placeholders})",
                [now, *descendant_ids],
            )
        return self.folder(folder_id) or {}

    def delete_folder(self, folder_id: str, *, confirmation: str) -> None:
        if confirmation != "永久删除":
            raise ValueError("永久删除需要明确确认")
        with self._lock, self.connect() as connection:
            folder = connection.execute(
                "SELECT archived_at FROM folders WHERE id=?", (folder_id,)
            ).fetchone()
            if folder is None:
                raise KeyError(f"Unknown folder: {folder_id}")
            child = connection.execute(
                "SELECT COUNT(*) AS count FROM folders WHERE parent_id=?", (folder_id,)
            ).fetchone()
            jobs = connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE folder_id=?", (folder_id,)
            ).fetchone()
            if not folder["archived_at"]:
                raise ValueError("目录必须先移到回收站")
            if int(child["count"] or 0) or int(jobs["count"] or 0):
                raise ValueError("只有空目录可以永久删除")
            connection.execute("DELETE FROM folders WHERE id=?", (folder_id,))

    def library_jobs(self, include_archived: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM jobs"
        if not include_archived:
            query += " WHERE archived_at IS NULL"
        query += " ORDER BY COALESCE(folder_id,''), library_order, updated_at DESC"
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    def move_job_library(
        self,
        job_id: str,
        folder_id: str | None,
        before_job_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock, self.connect() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            if job["archived_at"]:
                raise ValueError("回收站任务不能移动")
            if folder_id is not None:
                folder = connection.execute(
                    "SELECT archived_at FROM folders WHERE id=?", (folder_id,)
                ).fetchone()
                if folder is None:
                    raise KeyError(f"Unknown folder: {folder_id}")
                if folder["archived_at"]:
                    raise ValueError("不能移动到回收站目录")
            siblings = connection.execute(
                "SELECT id FROM jobs WHERE COALESCE(folder_id,'')=COALESCE(?, '') "
                "AND id<>? AND archived_at IS NULL ORDER BY library_order, updated_at DESC",
                (folder_id, job_id),
            ).fetchall()
            ordered = [str(row["id"]) for row in siblings]
            if before_job_id and before_job_id in ordered:
                ordered.insert(ordered.index(before_job_id), job_id)
            else:
                ordered.append(job_id)
            for index, sibling_id in enumerate(ordered):
                connection.execute(
                    "UPDATE jobs SET folder_id=?, library_order=?, updated_at=? WHERE id=?",
                    (folder_id, index, _now(), sibling_id),
                )
        return self.job(job_id) or {}

    def reorder_library(self, folder_id: str | None, job_ids: list[str]) -> None:
        with self._lock, self.connect() as connection:
            rows = connection.execute(
                "SELECT id FROM jobs WHERE COALESCE(folder_id,'')=COALESCE(?, '') "
                "AND archived_at IS NULL ORDER BY library_order, updated_at DESC",
                (folder_id,),
            ).fetchall()
            expected = {str(row["id"]) for row in rows}
            if set(job_ids) != expected or len(job_ids) != len(expected):
                raise ValueError("书库排序必须包含同级全部任务")
            for index, job_id in enumerate(job_ids):
                connection.execute(
                    "UPDATE jobs SET library_order=?, updated_at=? WHERE id=?",
                    (index, _now(), job_id),
                )

    def reorder_folders(self, parent_id: str | None, folder_ids: list[str]) -> None:
        """Persist the visual order of direct folder siblings only.

        Folder ordering is deliberately separate from the GPU queue and from
        job ordering.  The complete-sibling check makes a stale drag/drop
        request fail instead of silently losing a folder from the tree.
        Archived folders are excluded so the trash view cannot alter the
        active library order.
        """
        with self._lock, self.connect() as connection:
            if parent_id is not None:
                parent = connection.execute(
                    "SELECT id FROM folders WHERE id=?", (parent_id,)
                ).fetchone()
                if parent is None:
                    raise KeyError(f"Unknown folder: {parent_id}")
            rows = connection.execute(
                "SELECT id FROM folders WHERE COALESCE(parent_id,'')=COALESCE(?, '') "
                "AND archived_at IS NULL ORDER BY sort_order, name COLLATE NOCASE",
                (parent_id,),
            ).fetchall()
            expected = {str(row["id"]) for row in rows}
            if set(folder_ids) != expected or len(folder_ids) != len(expected):
                raise ValueError("目录排序必须包含同级全部目录")
            for index, folder_id in enumerate(folder_ids):
                connection.execute(
                    "UPDATE folders SET sort_order=?, updated_at=? WHERE id=?",
                    (index, _now(), folder_id),
                )

    def add_event(self, kind: str, payload: dict[str, Any], job_id: str | None = None) -> int:
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events(job_id,kind,payload_json,created_at) VALUES(?,?,?,?)",
                (job_id, kind, json.dumps(payload, ensure_ascii=False), _now()),
            )
            return int(cursor.lastrowid)

    def create_operation(self, operation_id: str, job_id: str, stage: str = "accepted") -> None:
        now = _now()
        with self._lock, self.connect() as connection:
            connection.execute(
                """
                INSERT INTO operations(id,job_id,status,stage,progress_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  status=excluded.status,stage=excluded.stage,updated_at=excluded.updated_at
                """,
                (operation_id, job_id, "accepted", stage, "{}", now, now),
            )

    def update_operation(
        self,
        operation_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        values: dict[str, Any] = {"updated_at": _now()}
        if status is not None:
            values["status"] = status
        if stage is not None:
            values["stage"] = stage
        if progress is not None:
            values["progress_json"] = json.dumps(progress, ensure_ascii=False)
        if error is not None:
            values["error"] = error
        elif status in {"accepted", "running", "completed"}:
            # A resumed operation must not keep showing the previous failure
            # after it has successfully reached the next checkpoint.
            values["error"] = None
        assignments = ",".join(f"{key}=?" for key in values)
        with self._lock, self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE operations SET {assignments} WHERE id=?",
                [*values.values(), operation_id],
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown operation: {operation_id}")

    def operation(self, operation_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE id=?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["progress"] = json.loads(result.pop("progress_json") or "{}")
        return result

    def latest_operation_for_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE job_id=? ORDER BY updated_at DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["progress"] = json.loads(result.pop("progress_json") or "{}")
        return result

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

    def recent_events(
        self, job_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        query = "SELECT * FROM events"
        parameters: list[Any] = []
        if job_id is not None:
            query += " WHERE job_id=?"
            parameters.append(job_id)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)
        with self.connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result: list[dict[str, Any]] = []
        for row in reversed(rows):
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
