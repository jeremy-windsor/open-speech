"""Tests for the Batch Transcription API."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
import threading
import time
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import Headers, UploadFile

from src import main as main_module
from src import storage as storage_module
from src.backends.faster_whisper import FasterWhisperBackend
from src.batch import worker as worker_module
from src.batch.store import BatchJob, BatchJobStore
from src.batch.worker import INTERRUPTED_JOB_ERROR, BatchWorker, recover_interrupted_jobs
from src.config import settings
from src.main import app
from src.services import batch as batch_service
from src.services import stt as stt_service
from src.utils import audio as audio_module
from tests.test_audio_conversion import _silent_flac

# ── Helpers ──────────────────────────────────────────────────────────────────


def _wav_bytes(amplitude: int = 0, rate: int = 16000) -> bytes:
    """Generate minimal valid WAV audio."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(amplitude.to_bytes(2, "little", signed=True) * 1600)
    return buf.getvalue()


def _make_store(tmp_path) -> BatchJobStore:
    """Create a fresh BatchJobStore with temp DB."""
    return BatchJobStore(db_path=tmp_path / "batch_test.db")


def _make_job(**kwargs) -> BatchJob:
    """Create a BatchJob with defaults."""
    defaults = {
        "job_id": "test-job-1",
        "status": "queued",
        "created_at": time.time(),
        "model": "test-model",
        "files": ["a.wav", "b.wav"],
        "options": {"model": "test-model", "language": "en"},
    }
    defaults.update(kwargs)
    return BatchJob(**defaults)


def _reset_db(tmp_path):
    """Reset DB for API tests."""
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_history_enabled = True
    main_module.settings.os_history_max_entries = 1000
    storage_module._conn = None
    storage_module.init_db()


def _mock_backend():
    """Create a mock STT backend router."""
    mock = MagicMock()
    mock.transcribe.return_value = {
        "text": "hello world",
        "language": "en",
        "duration": 1.0,
        "segments": [],
    }
    return mock


async def _submit_task(worker, job_id, audio_files, options):
    await worker.submit(job_id, audio_files, options)
    return worker._tasks[job_id]


class _GuardedChunkedUpload:
    """Upload fake that rejects unbounded reads and yields fixed-size chunks."""

    def __init__(self, total_bytes: int) -> None:
        self._remaining = total_bytes
        self.read_sizes: list[int] = []
        self.filename = "oversized.wav"

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size < 1:
            raise AssertionError("upload reads must specify a positive size")
        chunk_size = min(size, self._remaining)
        self._remaining -= chunk_size
        return b"x" * chunk_size


class _BlockingRouter:
    """STT router fake whose transcriptions block until released, recording overlap."""

    def __init__(self, *names: bytes) -> None:
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.any_started = threading.Event()
        self.started = {name: threading.Event() for name in names}
        self.release = {name: threading.Event() for name in names}

    def transcribe(self, audio, model, **kwargs):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.started[audio].set()
        self.any_started.set()
        try:
            self.release[audio].wait(10)
            return {"text": audio.decode(), "language": "en", "duration": 1.0, "segments": []}
        finally:
            with self._lock:
                self.in_flight -= 1

    def release_all(self) -> None:
        for event in self.release.values():
            event.set()


def _form_request(*uploads):
    form = MagicMock()
    form.getlist.return_value = list(uploads)
    form.close = AsyncMock()
    request = MagicMock()
    request.form = AsyncMock(return_value=form)
    return request


def _wav_upload(data: bytes, filename: str = "a.wav") -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=filename, headers=Headers({"content-type": "audio/wav"}))


_LIMITS = SimpleNamespace(os_batch_max_pending=10, os_max_upload_mb=1, os_batch_max_total_mb=2)


async def _submit_batch(request, *, store, worker, language=None):
    return await batch_service.submit_batch_transcription(
        request=request, model="test-model", language=language, response_format="json",
        temperature=0.0, settings=_LIMITS, batch_worker=worker, batch_store=store,
    )


# ── Store Tests ──────────────────────────────────────────────────────────────


def test_store_create(tmp_path):
    """1. BatchJobStore.create creates row with queued status."""
    store = _make_store(tmp_path)
    job = _make_job()
    result = store.create(job)
    assert result.job_id == "test-job-1"
    assert result.status == "queued"


def test_store_get(tmp_path):
    """2. BatchJobStore.get returns job by ID."""
    store = _make_store(tmp_path)
    store.create(_make_job())
    got = store.get("test-job-1")
    assert got is not None
    assert got.job_id == "test-job-1"
    assert got.model == "test-model"
    assert got.files == ["a.wav", "b.wav"]


def test_store_get_unknown(tmp_path):
    """3. BatchJobStore.get returns None for unknown ID."""
    store = _make_store(tmp_path)
    assert store.get("nonexistent") is None


def test_store_update(tmp_path):
    """4. BatchJobStore.update updates status field."""
    store = _make_store(tmp_path)
    store.create(_make_job())
    store.update("test-job-1", status="running", started_at=time.time())
    got = store.get("test-job-1")
    assert got.status == "running"
    assert got.started_at is not None


def test_store_list_jobs(tmp_path):
    """5. BatchJobStore.list_jobs returns list, respects limit."""
    store = _make_store(tmp_path)
    for i in range(5):
        store.create(_make_job(job_id=f"job-{i}"))
    all_jobs = store.list_jobs()
    assert len(all_jobs) == 5
    limited = store.list_jobs(limit=2)
    assert len(limited) == 2


def test_store_delete(tmp_path):
    """6. BatchJobStore.delete removes from DB, returns True."""
    store = _make_store(tmp_path)
    store.create(_make_job())
    assert store.delete("test-job-1") is True
    assert store.get("test-job-1") is None


def test_store_delete_unknown(tmp_path):
    """7. BatchJobStore.delete returns False for unknown ID."""
    store = _make_store(tmp_path)
    assert store.delete("nonexistent") is False


def test_store_write_that_fails_does_not_keep_the_database_locked(tmp_path):
    # A failed write left its transaction open, holding SQLite's write lock until a later commit.
    store = _make_store(tmp_path)
    store.create(_make_job())
    with pytest.raises(sqlite3.IntegrityError):
        store.create(_make_job())  # the same job id again

    other = sqlite3.connect(str(tmp_path / "batch_test.db"), timeout=0)  # no waiting for a lock
    try:
        other.execute("DELETE FROM batch_jobs")
        other.commit()
    finally:
        other.close()
    assert store.get("test-job-1") is None


def test_recover_interrupted_jobs_fails_queued_and_running(tmp_path):
    store = _make_store(tmp_path)
    queued_ids = ["queued-job-1", "queued-job-2", "queued-job-3"]
    for job_id in queued_ids:
        store.create(_make_job(job_id=job_id, status="queued"))
    store.create(_make_job(job_id="running-job", status="running"))
    store.create(_make_job(job_id="done-job", status="done"))

    assert recover_interrupted_jobs(store, batch_size=2) == 4

    for job_id in (*queued_ids, "running-job"):
        job = store.get(job_id)
        assert job.status == "failed"
        assert job.error == INTERRUPTED_JOB_ERROR
        assert job.finished_at is not None
    assert store.get("done-job").status == "done"


def test_recover_interrupted_jobs_rejects_invalid_batch_size(tmp_path):
    store = _make_store(tmp_path)

    with pytest.raises(ValueError, match="at least 1"):
        recover_interrupted_jobs(store, batch_size=0)


def test_recover_interrupted_jobs_stops_when_updates_make_no_progress(tmp_path):
    store = _make_store(tmp_path)
    store.create(_make_job(job_id="stuck-job", status="queued"))

    with patch.object(store, "update", return_value=False):
        assert recover_interrupted_jobs(store, batch_size=1) == 0

    assert store.get("stuck-job").status == "queued"


# ── Worker Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_worker_submit_and_process(tmp_path):
    """8. BatchWorker submits job and processes (mock STT router)."""
    store = _make_store(tmp_path)
    job = _make_job()
    store.create(job)

    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=2)

    audio = _wav_bytes()
    task = await _submit_task(
        worker, job.job_id, [("a.wav", audio), ("b.wav", audio)], job.options
    )
    await task

    updated = store.get(job.job_id)
    assert updated.status == "done"
    assert len(updated.results) == 2
    assert updated.results[0]["filename"] == "a.wav"
    assert updated.results[0]["text"] == "hello world"
    assert updated.results[1]["filename"] == "b.wav"


@pytest.mark.asyncio
async def test_worker_per_file_failure(tmp_path):
    """9. BatchWorker — per-file failure doesn't abort whole job."""
    store = _make_store(tmp_path)
    job = _make_job(files=["good.wav", "bad.wav"])
    store.create(job)

    call_count = 0
    def mock_transcribe(audio, model, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Transcription failed for this file")
        return {"text": "success", "language": "en", "duration": 1.0, "segments": []}

    mock_router = MagicMock()
    mock_router.transcribe.side_effect = mock_transcribe

    worker = BatchWorker(store, mock_router, max_concurrent=2)
    task = await _submit_task(
        worker,
        job.job_id,
        [("good.wav", _wav_bytes()), ("bad.wav", _wav_bytes())],
        job.options,
    )
    await task

    updated = store.get(job.job_id)
    assert updated.status == "done"  # Job still completes
    assert updated.results[0]["status"] == "done"
    assert updated.results[1]["status"] == "failed"
    assert "error" in updated.results[1]


@pytest.mark.asyncio
async def test_worker_semaphore(tmp_path):
    """10. BatchWorker respects max_concurrent semaphore."""
    store = _make_store(tmp_path)
    concurrent_count = 0
    max_seen = 0
    first_started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_process(*args, **kwargs):
        nonlocal concurrent_count, max_seen
        concurrent_count += 1
        max_seen = max(max_seen, concurrent_count)
        first_started.set()
        try:
            await release.wait()
        finally:
            concurrent_count -= 1

    # max_concurrent=1 means only 1 job at a time
    worker = BatchWorker(store, _mock_backend(), max_concurrent=1)
    tasks = []

    with patch.object(worker, "_process_job", side_effect=controlled_process):
        for i in range(3):
            job = _make_job(job_id=f"sem-job-{i}", files=[f"f{i}.wav"])
            store.create(job)
            tasks.append(
                await _submit_task(
                    worker, job.job_id, [(f"f{i}.wav", _wav_bytes())], job.options
                )
            )

        try:
            await asyncio.wait_for(first_started.wait(), timeout=2)
            await asyncio.sleep(0)
            assert max_seen == 1
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert max_seen == 1


@pytest.mark.asyncio
async def test_worker_prepares_audio_like_single_transcription(tmp_path, monkeypatch):
    # STT_NORMALIZE / STT_NOISE_REDUCE apply to batch files too, not only single uploads.
    monkeypatch.setattr(audio_module.subprocess, "run", MagicMock(side_effect=FileNotFoundError("ffmpeg")))
    quiet = _wav_bytes(amplitude=50)
    upload = UploadFile(file=io.BytesIO(quiet), filename="a.wav", headers=Headers({"content-type": "audio/wav"}))
    store = _make_store(tmp_path)
    job = _make_job(files=["a.wav"])
    store.create(job)
    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=1)

    with patch.object(settings, "stt_normalize", True), patch.object(settings, "stt_noise_reduce", False):
        single_audio = await stt_service.read_and_prepare_upload(
            file=upload, settings=settings, allow_filename_override=True,
        )
        task = await _submit_task(worker, job.job_id, [("a.wav", quiet)], job.options)
        await asyncio.wait_for(task, timeout=10)

    assert single_audio != quiet
    assert mock_router.transcribe.call_args.kwargs["audio"] == single_audio


@pytest.mark.asyncio
async def test_worker_fails_a_file_that_decodes_to_zero_samples_as_empty(tmp_path, monkeypatch):
    # Batch shares prepare_audio with single uploads, so a header-only
    # WAV fails with "Empty audio file" and never reaches the model; the job's other files still run.
    monkeypatch.setattr(audio_module.subprocess, "run", MagicMock(side_effect=FileNotFoundError("ffmpeg")))
    header_only = io.BytesIO()
    with wave.open(header_only, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")
    store = _make_store(tmp_path)
    job = _make_job(files=["empty.wav", "good.wav"])
    store.create(job)
    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=1)

    task = await _submit_task(
        worker, job.job_id, [("empty.wav", header_only.getvalue()), ("good.wav", _wav_bytes())], job.options,
    )
    await asyncio.wait_for(task, timeout=10)

    results = store.get(job.job_id).results
    assert (results[0]["status"], results[0]["error"]) == ("failed", "Empty audio file")
    assert results[1]["status"] == "done"
    assert mock_router.transcribe.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "text"),
    [("json", "Hello there"), ("srt", "1\n00:00:00,000 --> 00:00:12,500\nHello there\n")],
    ids=["json", "srt"],
)
async def test_worker_results_report_file_duration_and_language(tmp_path, response_format, text):
    # Every per-file result carries language and duration, whatever the requested format.
    segment = SimpleNamespace(
        seek=0, start=0.0, end=12.5, text=" Hello there", tokens=[], temperature=0.0,
        avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
    )
    backend = FasterWhisperBackend()
    backend._models["test-model"] = SimpleNamespace(
        transcribe=lambda path, **kwargs: (iter([segment]), SimpleNamespace(language="en", duration=12.5)),
    )
    store = _make_store(tmp_path)
    job = _make_job(files=["a.wav"], options={"model": "test-model", "response_format": response_format})
    store.create(job)
    worker = BatchWorker(store, backend, max_concurrent=1)

    task = await _submit_task(worker, job.job_id, [("a.wav", _wav_bytes())], job.options)
    await asyncio.wait_for(task, timeout=10)

    result = store.get(job.job_id).results[0]
    assert (result["text"], result["duration"], result["language"]) == (text, 12.5, "en")


# ── API Tests ────────────────────────────────────────────────────────────────


def test_api_batch_no_files(tmp_path):
    """11. POST /v1/audio/transcriptions/batch — 422 if no files."""
    _reset_db(tmp_path)
    client = TestClient(app)
    # Create a store/worker in tmp
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post("/v1/audio/transcriptions/batch", data={"model": "test"})
    assert resp.status_code == 422


def test_api_batch_no_model(tmp_path):
    """12. POST /v1/audio/transcriptions/batch — uses default model when none provided."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    # When model is not explicitly provided, endpoint uses settings.stt_model default
    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
            # No model field — should use default from settings
        )
    # Default model from settings is used — request succeeds
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"


def test_api_batch_too_many_files(tmp_path):
    """13. POST /v1/audio/transcriptions/batch — 422 if >20 files."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    files = [("file", (f"f{i}.wav", _wav_bytes(), "audio/wav")) for i in range(21)]
    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post("/v1/audio/transcriptions/batch", files=files, data={"model": "test"})
    assert resp.status_code == 422
    body = resp.json()
    msg = body.get("error", {}).get("message", "") or body.get("detail", "")
    assert "20" in msg


def test_api_batch_success(tmp_path):
    """14. POST /v1/audio/transcriptions/batch — success returns job_id + queued."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
            data={"model": "test-model"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["file_count"] == 1


def test_api_list_jobs(tmp_path):
    """15. GET /v1/audio/jobs — returns list."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="list-1"))
    tmp_store.create(_make_job(job_id="list-2"))

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["jobs"]) == 2
    assert data["total"] == 2


def test_api_list_jobs_total_counts_all_matching_jobs(tmp_path):
    # Like history/composer/conversations, "total" counts every match, not the page.
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    for i in range(3):
        tmp_store.create(_make_job(job_id=f"queued-{i}", created_at=1000.0 + i))
    for i in range(2):
        tmp_store.create(_make_job(job_id=f"done-{i}", status="done", created_at=2000.0 + i))

    with patch.object(main_module, "batch_store", tmp_store):
        page = client.get("/v1/audio/jobs?limit=2").json()
        done = client.get("/v1/audio/jobs?status=done&limit=1").json()

    assert (len(page["jobs"]), page["total"]) == (2, 5)
    assert (len(done["jobs"]), done["total"]) == (1, 2)


@pytest.mark.parametrize("limit", [-1, 0, 201])
def test_api_list_jobs_rejects_out_of_range_limit(tmp_path, limit):
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get(f"/v1/audio/jobs?limit={limit}")

    assert resp.status_code == 422


def test_api_list_jobs_filter(tmp_path):
    """16. GET /v1/audio/jobs?status=queued — filters by status."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="q1", status="queued"))
    tmp_store.create(_make_job(job_id="d1", status="done"))

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs?status=queued")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["jobs"]) == 1
    assert data["jobs"][0]["job_id"] == "q1"


def test_api_get_job(tmp_path):
    """17. GET /v1/audio/jobs/{id} — returns job detail."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="detail-1"))

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs/detail-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "detail-1"
    assert "files" in data
    assert "options" in data


def test_api_get_job_404(tmp_path):
    """18. GET /v1/audio/jobs/{id} — 404 for unknown."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs/nonexistent")
    assert resp.status_code == 404


def test_api_result_not_done(tmp_path):
    """19. GET /v1/audio/jobs/{id}/result — 409 if not done."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="running-1", status="running"))

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs/running-1/result")
    assert resp.status_code == 409
    data = resp.json()
    assert data["status"] == "running"
    assert data["retry_after"] == 5


def test_api_result_of_failed_job_has_no_retry_hint(tmp_path):
    # "failed" is terminal; a retry_after hint would keep pollers waiting forever.
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="failed-1", status="failed", finished_at=time.time(), error="Canceled"))

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs/failed-1/result")
    assert resp.status_code == 409
    data = resp.json()
    assert data["status"] == "failed"
    assert data["reason"] == "Canceled"
    assert "retry_after" not in data


def test_api_result_done(tmp_path):
    """20. GET /v1/audio/jobs/{id}/result — returns results array when done."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    job = _make_job(job_id="done-1", status="done")
    job.results = [{"filename": "a.wav", "status": "done", "text": "hello"}]
    tmp_store.create(job)

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.get("/v1/audio/jobs/done-1/result")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["text"] == "hello"


def test_api_delete_job(tmp_path):
    """21. DELETE /v1/audio/jobs/{id} — 204 on success."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_store.create(_make_job(job_id="del-1"))
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.delete("/v1/audio/jobs/del-1")
    assert resp.status_code == 204
    assert tmp_store.get("del-1") is None


def test_api_delete_job_404(tmp_path):
    """22. DELETE /v1/audio/jobs/{id} — 404 for unknown."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)

    with patch.object(main_module, "batch_store", tmp_store):
        resp = client.delete("/v1/audio/jobs/nonexistent")
    assert resp.status_code == 404


# ── Integration Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_job_lifecycle(tmp_path):
    """23. Job status lifecycle: queued → running → done."""
    store = _make_store(tmp_path)
    job = _make_job(job_id="lifecycle-1")
    store.create(job)

    assert store.get("lifecycle-1").status == "queued"

    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=2)

    task = await _submit_task(worker, job.job_id, [("a.wav", _wav_bytes())], job.options)
    await task

    final = store.get("lifecycle-1")
    assert final.status == "done"
    assert final.started_at is not None
    assert final.finished_at is not None
    assert final.finished_at >= final.started_at


@pytest.mark.asyncio
async def test_history_integration(tmp_path):
    """24. Batch jobs no longer log history (API-only, no X-History header)."""
    store = _make_store(tmp_path)
    job = _make_job(job_id="hist-1")
    store.create(job)

    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=2)

    # Reset the shared DB for history
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_history_enabled = True
    storage_module._conn = None
    storage_module.init_db()

    audio = _wav_bytes()
    task = await _submit_task(
        worker, job.job_id, [("a.wav", audio), ("b.wav", audio)], job.options
    )
    await task

    from src.history import HistoryManager
    hm = HistoryManager()
    entries = hm.list_entries(type_filter="stt")
    # Batch worker no longer logs history — only web UI with X-History header does
    stt_items = entries["items"]
    batch_items = [i for i in stt_items if i.get("input_filename") in ("a.wav", "b.wav")]
    assert len(batch_items) == 0


@pytest.mark.asyncio
async def test_concurrent_submission(tmp_path):
    """25. Concurrent submission: two jobs don't interfere."""
    store = _make_store(tmp_path)
    job1 = _make_job(job_id="conc-1", files=["x.wav"])
    job2 = _make_job(job_id="conc-2", files=["y.wav"])
    store.create(job1)
    store.create(job2)

    mock_router = _mock_backend()
    worker = BatchWorker(store, mock_router, max_concurrent=2)

    task1 = await _submit_task(worker, job1.job_id, [("x.wav", _wav_bytes())], job1.options)
    task2 = await _submit_task(worker, job2.job_id, [("y.wav", _wav_bytes())], job2.options)
    await asyncio.gather(task1, task2)

    j1 = store.get("conc-1")
    j2 = store.get("conc-2")
    assert j1.status == "done"
    assert j2.status == "done"
    assert j1.results[0]["filename"] == "x.wav"
    assert j2.results[0]["filename"] == "y.wav"


# --- Validation tests ---

def test_api_batch_empty_file(tmp_path):
    """26. POST batch — 400 if any file is empty (0 bytes)."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("empty.wav", b"", "audio/wav"))],
            data={"model": "large-v3"},
        )
    assert resp.status_code == 400
    assert "empty" in resp.json()["error"]["message"].lower()


def test_api_batch_text_field_named_file_is_422(tmp_path):
    """POST batch — a plain form field named "file" is a client error, not a 500."""
    _reset_db(tmp_path)
    client = TestClient(app, raise_server_exceptions=False)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    with patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            data={"model": "test-model", "file": "not-a-file"},
        )
    assert resp.status_code == 422
    assert tmp_store.list_jobs() == []


def test_api_batch_aggregate_size_limit(tmp_path):
    """27. POST batch — 413 if total upload exceeds aggregate limit."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    # Patch the aggregate limit to 1 byte so we trigger it easily
    from src.config import settings as _settings
    with patch.object(_settings, "os_batch_max_total_mb", 0), \
         patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
            data={"model": "large-v3"},
        )
    assert resp.status_code == 413
    assert "aggregate" in resp.json()["error"]["message"].lower()


@pytest.mark.asyncio
async def test_batch_upload_limit_uses_bounded_reads():
    """Per-file limits must be enforced without reading a whole upload at once."""
    max_bytes = 1024 * 1024
    upload = _GuardedChunkedUpload(max_bytes + 1)
    form = MagicMock()
    form.getlist.return_value = [upload]
    form.close = AsyncMock()
    request = MagicMock()
    request.form = AsyncMock(return_value=form)
    worker = MagicMock()
    worker._tasks = {}
    worker.submit = AsyncMock()
    store = MagicMock()
    settings = SimpleNamespace(
        os_batch_max_pending=10,
        os_max_upload_mb=1,
        os_batch_max_total_mb=2,
    )

    with pytest.raises(HTTPException) as exc_info:
        await batch_service.submit_batch_transcription(
            request=request,
            model="test-model",
            language=None,
            response_format="json",
            temperature=0.0,
            settings=settings,
            batch_worker=worker,
            batch_store=store,
        )

    assert exc_info.value.status_code == 413
    assert upload.read_sizes
    assert all(0 < size <= max_bytes + 1 for size in upload.read_sizes)
    form.close.assert_awaited_once()
    store.create.assert_not_called()
    worker.submit.assert_not_awaited()


def test_api_batch_invalid_response_format_rejected_before_scheduling(tmp_path):
    """Unsupported formats must not create or schedule a batch job."""
    _reset_db(tmp_path)
    client = TestClient(app)
    store = _make_store(tmp_path)
    worker = MagicMock()
    worker._tasks = {}
    worker.submit = AsyncMock()

    with (
        patch.object(main_module, "batch_store", store),
        patch.object(main_module, "batch_worker", worker),
    ):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
            data={"model": "test-model", "response_format": "bogus"},
        )

    assert resp.status_code == 400
    assert "response_format" in resp.json()["error"]["message"]
    assert store.list_jobs() == []
    worker.submit.assert_not_awaited()


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_api_batch_non_finite_temperature_rejected_before_scheduling(tmp_path, value):
    # A NaN temperature was queued and decoded, then every read of the job answered 500.
    _reset_db(tmp_path)
    client = TestClient(app)
    store = _make_store(tmp_path)
    worker = MagicMock()
    worker._tasks = {}
    worker.submit = AsyncMock()

    with (
        patch.object(main_module, "batch_store", store),
        patch.object(main_module, "batch_worker", worker),
    ):
        resp = client.post(
            "/v1/audio/transcriptions/batch",
            files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
            data={"model": "test-model", "temperature": value},
        )

    assert resp.status_code == 422, resp.text
    assert "temperature" in resp.json()["error"]["message"]
    assert store.list_jobs() == []
    worker.submit.assert_not_awaited()


def test_api_job_stored_with_a_nan_temperature_can_still_be_read_and_deleted(tmp_path):
    # A job queued before the fix holds NaN in its options and in each segment; its
    # detail and result answered 500 (JSON has no NaN). They now read it as null.
    _reset_db(tmp_path)
    client = TestClient(app)
    store = _make_store(tmp_path)
    job = _make_job(job_id="nan-1", status="done", finished_at=time.time())
    job.options = {"model": "test-model", "language": None, "response_format": "verbose_json",
                   "temperature": float("nan")}
    job.results = [{"filename": "a.wav", "status": "done", "text": "hi",
                    "segments": [{"id": 0, "text": "hi", "temperature": float("inf")}]}]
    store.create(job)
    worker = BatchWorker(store, _mock_backend(), max_concurrent=1)

    with patch.object(main_module, "batch_store", store), patch.object(main_module, "batch_worker", worker):
        detail = client.get("/v1/audio/jobs/nan-1")
        result = client.get("/v1/audio/jobs/nan-1/result")
        deleted = client.delete("/v1/audio/jobs/nan-1")

    assert detail.status_code == 200, detail.text
    assert detail.json()["options"]["temperature"] is None
    assert result.status_code == 200, result.text
    assert result.json()["results"][0]["segments"][0]["temperature"] is None
    assert result.json()["results"][0]["text"] == "hi"
    assert deleted.status_code == 204
    assert store.get("nan-1") is None


def test_api_batch_max_pending_limit(tmp_path):
    """28. POST batch — 429 when pending job backlog is full."""
    _reset_db(tmp_path)
    client = TestClient(app)
    tmp_store = _make_store(tmp_path)
    tmp_worker = BatchWorker(tmp_store, _mock_backend(), max_concurrent=2)

    # Fake a full _tasks dict by patching it
    fake_tasks = {f"job-{i}": MagicMock() for i in range(10)}
    with patch.object(tmp_worker, "_tasks", fake_tasks), \
         patch.object(main_module, "batch_store", tmp_store), \
         patch.object(main_module, "batch_worker", tmp_worker):
        from src.config import settings as _settings
        with patch.object(_settings, "os_batch_max_pending", 10):
            resp = client.post(
                "/v1/audio/transcriptions/batch",
                files=[("file", ("a.wav", _wav_bytes(), "audio/wav"))],
                data={"model": "large-v3"},
            )
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_batch_pending_limit_holds_under_concurrent_submissions():
    """28b. OS_BATCH_MAX_PENDING holds when submissions overlap while uploads are read."""
    reads_started = asyncio.Event()
    release = asyncio.Event()
    reads = 0

    class SlowUpload:
        # Starlette reads disk-spooled (>1 MB) parts in a threadpool, so reads yield.
        filename = "a.wav"

        def __init__(self):
            self._sent = False

        async def read(self, size=-1):
            nonlocal reads
            if self._sent:
                return b""
            reads += 1
            if reads == 2:
                reads_started.set()
            await release.wait()
            self._sent = True
            return _wav_bytes()

    worker = MagicMock()
    worker._tasks = {}

    async def submit(job_id, audio_files, options):
        worker._tasks[job_id] = MagicMock()  # the job stays queued

    worker.submit = submit
    limits = SimpleNamespace(os_batch_max_pending=1, os_max_upload_mb=1, os_batch_max_total_mb=2)

    async def submit_batch():
        form = MagicMock()
        form.getlist.return_value = [SlowUpload()]
        form.close = AsyncMock()
        request = MagicMock()
        request.form = AsyncMock(return_value=form)
        return await batch_service.submit_batch_transcription(
            request=request, model="test-model", language=None, response_format="json",
            temperature=0.0, settings=limits, batch_worker=worker, batch_store=MagicMock(),
        )

    submissions = [asyncio.create_task(submit_batch()) for _ in range(2)]
    try:
        await asyncio.wait_for(reads_started.wait(), timeout=5)
    finally:
        release.set()
    outcomes = await asyncio.wait_for(asyncio.gather(*submissions, return_exceptions=True), timeout=5)

    assert len(worker._tasks) == 1
    assert [o.status_code for o in outcomes if isinstance(o, HTTPException)] == [429]


@pytest.mark.asyncio
async def test_worker_cancelled_error_marks_failed(tmp_path):
    """29. CancelledError during job processing marks job as failed (not zombie)."""
    store = _make_store(tmp_path)

    async def exploding_transcribe(*args, **kwargs):
        raise asyncio.CancelledError()

    mock_router = MagicMock()
    job = _make_job(job_id="cancel-test", files=["x.wav"])
    store.create(job)
    worker = BatchWorker(store, mock_router, max_concurrent=2)

    # Process directly, wrapping the cancellation
    with pytest.raises(asyncio.CancelledError):
        with patch.object(worker, "_transcribe_file", side_effect=asyncio.CancelledError()):
            await worker._process_job("cancel-test", [("x.wav", _wav_bytes())], job.options)

    final = store.get("cancel-test")
    assert final.status == "failed"
    assert final.error == "Canceled"   # US spelling
    assert final.finished_at is not None


# ── Cancellation, inference priority, spooled audio and loop offloading ─────


def _passthrough_audio(monkeypatch):
    monkeypatch.setattr(worker_module, "prepare_audio", lambda audio, *, suffix, settings: audio)


@pytest.mark.asyncio
async def test_cancelled_running_job_keeps_its_worker_slot_until_its_transcription_ends(tmp_path, monkeypatch):
    # OS_BATCH_WORKERS is the max number of concurrent batch jobs (CHANGELOG 0.7.0) and
    # OS_BATCH_MAX_PENDING counts queued + running jobs (README). The executor thread of a
    # cancelled job cannot be interrupted, so the job keeps its slot until the thread ends.
    _passthrough_audio(monkeypatch)
    store = _make_store(tmp_path)
    router = _BlockingRouter(b"job-a", b"job-b")
    worker = BatchWorker(store, router, max_concurrent=1)
    options = {"model": "test-model", "response_format": "json"}
    for job_id in ("a", "b"):
        store.create(_make_job(job_id=job_id, files=[f"{job_id}.wav"]))

    tasks = []
    try:
        tasks.append(await _submit_task(worker, "a", [("a.wav", b"job-a")], options))
        assert await asyncio.to_thread(router.started[b"job-a"].wait, 5)
        assert await worker.cancel("a")
        tasks.append(await _submit_task(worker, "b", [("b.wav", b"job-b")], options))

        assert not await asyncio.to_thread(router.started[b"job-b"].wait, 0.5)
        assert set(worker._tasks) == {"a", "b"}
    finally:
        router.release_all()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)

    assert router.max_in_flight == 1
    assert tasks[0].cancelled()
    assert (store.get("a").status, store.get("a").error) == ("failed", "Canceled")
    assert store.get("b").status == "done"


@pytest.mark.asyncio
async def test_cancelled_job_stops_after_the_file_being_transcribed(tmp_path, monkeypatch):
    # A running file cannot be interrupted, so the job ends after it, even when cancelled
    # again (shutdown right after DELETE), and its later files never start.
    _passthrough_audio(monkeypatch)
    store = _make_store(tmp_path)
    router = _BlockingRouter(b"first", b"second")
    worker = BatchWorker(store, router, max_concurrent=1)
    store.create(_make_job(job_id="multi", files=["1.wav", "2.wav"]))

    task = await _submit_task(
        worker, "multi", [("1.wav", b"first"), ("2.wav", b"second")], {"model": "test-model"},
    )
    try:
        assert await asyncio.to_thread(router.started[b"first"].wait, 5)
        assert await worker.cancel("multi")
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.2)
        assert not task.done()
        assert worker._tasks == {"multi": task}
    finally:
        router.release_all()
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)

    assert task.cancelled()
    assert not router.started[b"second"].is_set()
    assert (store.get("multi").status, store.get("multi").error) == ("failed", "Canceled")


@pytest.mark.asyncio
async def test_batch_jobs_run_one_inference_at_a_time(tmp_path, monkeypatch):
    # A batch file holds the backend's inference lock for its whole run, so live STT
    # (streams, realtime, Wyoming) waits behind every batch inference queued on it.
    # Whatever OS_BATCH_WORKERS allows, only one batch inference may be in flight.
    _passthrough_audio(monkeypatch)
    store = _make_store(tmp_path)
    router = _BlockingRouter(b"job-a", b"job-b")
    worker = BatchWorker(store, router, max_concurrent=2)
    options = {"model": "test-model", "response_format": "json"}
    tasks = []
    try:
        for job_id in ("a", "b"):
            store.create(_make_job(job_id=job_id, files=[f"{job_id}.wav"]))
            tasks.append(await _submit_task(worker, job_id, [(f"{job_id}.wav", f"job-{job_id}".encode())], options))
        assert await asyncio.to_thread(router.any_started.wait, 5)
        await asyncio.sleep(0.5)
        assert router.in_flight == 1
    finally:
        router.release_all()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10)

    assert router.max_in_flight == 1
    assert [store.get(job_id).status for job_id in ("a", "b")] == ["done", "done"]


@pytest.mark.asyncio
async def test_queued_batch_audio_waits_on_disk_and_is_deleted_when_the_job_ends(tmp_path, monkeypatch):
    # Up to OS_BATCH_MAX_PENDING jobs of OS_BATCH_MAX_TOTAL_MB each may wait in the queue;
    # their uploads are spooled under the batch data dir, not kept in RAM, and removed
    # once the job has run.
    monkeypatch.setattr(audio_module.subprocess, "run", MagicMock(side_effect=FileNotFoundError("ffmpeg")))
    store = _make_store(tmp_path)
    router = _mock_backend()
    worker = BatchWorker(store, router, max_concurrent=1)
    queued: dict[str, list] = {}
    submit = worker.submit

    async def recording_submit(job_id, audio_files, options):
        queued[job_id] = [(name, audio, audio.read_bytes()) for name, audio in audio_files]
        await submit(job_id, audio_files, options)

    worker.submit = recording_submit
    audio = _wav_bytes(amplitude=100)

    response = await _submit_batch(_form_request(_wav_upload(audio)), store=store, worker=worker)
    job_id = json.loads(response.body)["job_id"]
    [(filename, path, spooled)] = queued[job_id]
    assert (filename, path.parent, spooled) == ("a.wav", store.audio_dir(job_id), audio)
    assert path.parent.parent == tmp_path / "batch_audio"

    await asyncio.wait_for(worker._tasks[job_id], timeout=10)

    assert store.get(job_id).results[0]["text"] == "hello world"
    assert not store.audio_dir(job_id).exists()


@pytest.mark.asyncio
async def test_cancelling_a_job_before_it_starts_deletes_its_spooled_audio(tmp_path):
    store = _make_store(tmp_path)
    audio_dir = store.audio_dir("q")
    audio_dir.mkdir(parents=True)
    (audio_dir / "0").write_bytes(_wav_bytes())
    store.create(_make_job(job_id="q", files=["a.wav"]))
    worker = BatchWorker(store, _mock_backend(), max_concurrent=1)

    task = await _submit_task(worker, "q", [("a.wav", audio_dir / "0")], {"model": "test-model"})
    assert await worker.cancel("q")  # before the task's coroutine ever ran
    await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=10)

    assert task.cancelled()
    assert not audio_dir.exists()


@pytest.mark.asyncio
async def test_rejected_batch_upload_leaves_no_spooled_audio(tmp_path):
    store = _make_store(tmp_path)
    worker = BatchWorker(store, _mock_backend(), max_concurrent=1)
    request = _form_request(_wav_upload(_wav_bytes()), _wav_upload(b"", filename="empty.wav"))

    with pytest.raises(HTTPException) as exc_info:
        await _submit_batch(request, store=store, worker=worker)

    assert exc_info.value.status_code == 400
    assert list(store.audio_root.glob("*")) == []
    assert store.list_jobs() == []
    assert worker._tasks == {}


def test_recover_interrupted_jobs_deletes_their_spooled_audio(tmp_path):
    store = _make_store(tmp_path)
    store.create(_make_job(job_id="queued-job", status="queued"))
    audio_dir = store.audio_dir("queued-job")
    audio_dir.mkdir(parents=True)
    (audio_dir / "0").write_bytes(_wav_bytes())

    assert recover_interrupted_jobs(store) == 1

    assert not store.audio_root.exists()


@pytest.mark.parametrize("job_id", ["", ".", "..", "../x", "a/b"])
def test_store_rejects_job_ids_that_escape_the_audio_dir(tmp_path, job_id):
    with pytest.raises(ValueError):
        _make_store(tmp_path).audio_dir(job_id)


@pytest.mark.asyncio
async def test_batch_store_writes_run_off_the_event_loop(tmp_path, monkeypatch):
    # Each results update re-serializes every result so far (segments and tokens included).
    monkeypatch.setattr(audio_module.subprocess, "run", MagicMock(side_effect=FileNotFoundError("ffmpeg")))
    store = _make_store(tmp_path)
    loop_thread = threading.get_ident()
    on_loop: list[tuple[str, bool]] = []

    def recording(name):
        original = getattr(store, name)

        def call(*args, **kwargs):
            on_loop.append((name, threading.get_ident() == loop_thread))
            return original(*args, **kwargs)

        return call

    for name in ("create", "update"):
        monkeypatch.setattr(store, name, recording(name))
    worker = BatchWorker(store, _mock_backend(), max_concurrent=1)

    response = await _submit_batch(_form_request(_wav_upload(_wav_bytes())), store=store, worker=worker)
    await asyncio.wait_for(worker._tasks[json.loads(response.body)["job_id"]], timeout=10)

    assert {name for name, _ in on_loop} == {"create", "update"}
    assert not any(loop for _, loop in on_loop), on_loop


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/v1/audio/jobs"),
    ("GET", "/v1/audio/jobs/slow"),
    ("GET", "/v1/audio/jobs/slow/result"),
    ("DELETE", "/v1/audio/jobs/slow"),
])
async def test_batch_job_reads_do_not_block_the_event_loop(tmp_path, method, path):
    # A finished job's row holds every file's segments: reading and decoding it on the event
    # loop would stall every other request and live stream until SQLite and json are done.
    import httpx
    from fastapi import FastAPI

    from src.routes import batch as batch_routes

    store = _make_store(tmp_path)
    job = _make_job(job_id="slow", status="done")
    job.results = [{"filename": "a.wav", "status": "done", "text": "hello"}]
    store.create(job)
    reading, release = threading.Event(), threading.Event()
    released: list[bool] = []

    def blocking(read):
        def call(*args, **kwargs):
            reading.set()
            released.append(release.wait(timeout=5))  # False: the loop could not release it
            return read(*args, **kwargs)

        return call

    store.get = blocking(store.get)
    store.list_jobs = blocking(store.list_jobs)
    app = FastAPI()
    app.include_router(batch_routes.create_router(
        get_settings=lambda: settings, get_batch_worker=lambda: None, get_batch_store=lambda: store,
    ))

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow = asyncio.create_task(client.request(method, path))
        try:
            assert await asyncio.to_thread(reading.wait, 5)
            other = await asyncio.wait_for(client.get("/ping"), timeout=5)
        finally:
            release.set()
        response = await asyncio.wait_for(slow, timeout=10)

    assert other.status_code == 200
    assert released == [True], "the store read blocked the event loop"
    assert response.status_code == (204 if method == "DELETE" else 200)
    if path.endswith("/result"):
        assert response.json() == {"results": job.results}


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["english", "xx"])
async def test_batch_rejects_unknown_language_before_reading_uploads(language):
    # OpenAI's transcription API takes an ISO-639-1 language and answers an unknown one
    # with a 400; faster-whisper would only fail every file after decoding it.
    request = _form_request(_wav_upload(_wav_bytes()))
    worker = MagicMock()
    worker._tasks = {}
    worker.submit = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await _submit_batch(request, store=MagicMock(), worker=worker, language=language)

    assert exc_info.value.status_code == 400
    assert "ISO-639-1" in exc_info.value.detail
    request.form.assert_not_awaited()
    worker.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_file_with_implausible_sample_rate_fails_before_decoding(tmp_path):
    # Decoding resamples to 16 kHz: 1600 samples declared at 1 Hz would become 25.6M samples.
    store = _make_store(tmp_path)
    store.create(_make_job(job_id="rate", files=["a.wav"]))
    router = _mock_backend()
    worker = BatchWorker(store, router, max_concurrent=1)

    task = await _submit_task(worker, "rate", [("a.wav", _wav_bytes(rate=1))], {"model": "test-model"})
    await asyncio.wait_for(task, timeout=10)

    result = store.get("rate").results[0]
    assert result["status"] == "failed"
    assert "sample rate 1 Hz" in result["error"]
    router.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_batch_file_longer_than_the_limit_fails_before_decoding(tmp_path, monkeypatch):
    # A few KB of FLAC at a legitimate 8 kHz can declare hours of silence, which the
    # backend would decode into one 16 kHz float32 array.
    monkeypatch.setattr(audio_module, "MAX_INPUT_AUDIO_SECONDS", 60)
    store = _make_store(tmp_path)
    store.create(_make_job(job_id="long", files=["a.flac", "b.wav"]))
    router = _mock_backend()
    worker = BatchWorker(store, router, max_concurrent=1)

    files = [("a.flac", _silent_flac(10 * 60)), ("b.wav", _wav_bytes())]
    task = await _submit_task(worker, "long", files, {"model": "test-model"})
    await asyncio.wait_for(task, timeout=10)

    long_file, short_file = store.get("long").results
    assert long_file["status"] == "failed"
    assert "longer than 60 seconds" in long_file["error"]
    assert short_file["status"] == "done"
    router.transcribe.assert_called_once()
