"""Tests for Phase 7: Batch Transcription API."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.batch.store import BatchJobStore, BatchJob
from src.batch.worker import BatchWorker, INTERRUPTED_JOB_ERROR, recover_interrupted_jobs
from src.main import app
from src import main as main_module
from src import storage as storage_module
from src.services import batch as batch_service


# ── Helpers ──────────────────────────────────────────────────────────────────


def _wav_bytes() -> bytes:
    """Generate minimal valid WAV audio."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
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
    main_module.settings.os_history_max_mb = 2000
    main_module.settings.os_history_retain_audio = True
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


# --- Post-review validation tests ---

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
    assert final.error == "Cancelled"
    assert final.finished_at is not None
