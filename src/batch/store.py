"""SQLite-backed batch job store for async transcription jobs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


@dataclass
class BatchJob:
    """Represents a batch transcription job."""

    job_id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    model: str = ""
    files: list[str] = field(default_factory=list)
    options: dict[str, Any] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_summary(self) -> dict[str, Any]:
        """Return job metadata without results blob."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "model": self.model,
            "file_count": len(self.files),
            "error": self.error,
        }

    def to_detail(self) -> dict[str, Any]:
        """Return full job detail including results."""
        d = self.to_summary()
        d["files"] = self.files
        d["options"] = self.options
        d["results"] = self.results
        return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS batch_jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    payload TEXT NOT NULL
);
"""


class BatchJobStore:
    """SQLite-backed store for batch transcription jobs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "batch_jobs.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executescript(_SCHEMA)
            conn.commit()

    def _job_from_row(self, row: sqlite3.Row) -> BatchJob:
        payload = json.loads(row["payload"])
        return BatchJob(
            job_id=row["job_id"],
            status=row["status"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            model=payload.get("model", ""),
            files=payload.get("files", []),
            options=payload.get("options", {}),
            results=payload.get("results", []),
            error=payload.get("error"),
        )

    def _payload(self, job: BatchJob) -> str:
        return json.dumps({
            "model": job.model,
            "files": job.files,
            "options": job.options,
            "results": job.results,
            "error": job.error,
        })

    def create(self, job: BatchJob) -> BatchJob:
        """Insert a new job into the store."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO batch_jobs (job_id, status, created_at, started_at, finished_at, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (job.job_id, job.status, job.created_at, job.started_at, job.finished_at, self._payload(job)),
            )
            conn.commit()
        return job

    def get(self, job_id: str) -> BatchJob | None:
        """Get a job by ID, or None if not found."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return self._job_from_row(row)

    def update(self, job_id: str, **fields: Any) -> bool:
        """Update specific fields of a job. Returns True if updated."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM batch_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return False

            job = self._job_from_row(row)

            # Direct DB column fields
            col_updates: list[str] = []
            col_params: list[Any] = []
            for col in ("status", "started_at", "finished_at"):
                if col in fields:
                    setattr(job, col, fields.pop(col))
                    col_updates.append(f"{col} = ?")
                    col_params.append(getattr(job, col))

            # Everything else goes into the payload
            for k, v in fields.items():
                if hasattr(job, k):
                    setattr(job, k, v)

            payload = self._payload(job)
            col_updates.append("payload = ?")
            col_params.append(payload)

            col_params.append(job_id)
            conn.execute(
                f"UPDATE batch_jobs SET {', '.join(col_updates)} WHERE job_id = ?",
                tuple(col_params),
            )
            conn.commit()
        return True

    def list_jobs(self, limit: int = 50, status: str | None = None) -> list[BatchJob]:
        """List jobs, optionally filtered by status."""
        with self._lock:
            conn = self._get_conn()
            if status:
                rows = conn.execute(
                    "SELECT * FROM batch_jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM batch_jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._job_from_row(r) for r in rows]

    def delete(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if deleted."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM batch_jobs WHERE job_id = ?", (job_id,))
            conn.commit()
        return cursor.rowcount > 0
