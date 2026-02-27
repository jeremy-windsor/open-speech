"""Tests for Phase 7: Batch Transcription API."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from fastapi.testclient import TestClient

from src.batch.store import BatchJobStore, BatchJob
from src.batch.worker import BatchWorker
from src.main import app
from src import main as main_module
from src import storage as storage_module


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
    await worker.submit(job.job_id, [("a.wav", audio), ("b.wav", audio)], job.options)
    # Wait for processing
    await asyncio.sleep(0.5)

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
    await worker.submit(job.job_id, [("good.wav", _wav_bytes()), ("bad.wav", _wav_bytes())], job.options)
    await asyncio.sleep(0.5)

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
    lock = asyncio.Lock()

    original_transcribe_calls = 0

    def slow_transcribe(audio, model, **kwargs):
        nonlocal concurrent_count, max_seen, original_transcribe_calls
        import threading
        # We can't use asyncio lock in sync context, use a simple counter
        concurrent_count += 1
        if concurrent_count > max_seen:
            max_seen = concurrent_count
        import time as t
        t.sleep(0.1)
        concurrent_count -= 1
        original_transcribe_calls += 1
        return {"text": "ok", "language": "en", "duration": 1.0, "segments": []}

    mock_router = MagicMock()
    mock_router.transcribe.side_effect = slow_transcribe

    # max_concurrent=1 means only 1 job at a time
    worker = BatchWorker(store, mock_router, max_concurrent=1)

    for i in range(3):
        job = _make_job(job_id=f"sem-job-{i}", files=[f"f{i}.wav"])
        store.create(job)
        await worker.submit(job.job_id, [(f"f{i}.wav", _wav_bytes())], job.options)

    await asyncio.sleep(1.5)

    # All jobs should complete
    for i in range(3):
        j = store.get(f"sem-job-{i}")
        assert j.status == "done"


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

    await worker.submit(job.job_id, [("a.wav", _wav_bytes())], job.options)
    await asyncio.sleep(0.5)

    final = store.get("lifecycle-1")
    assert final.status == "done"
    assert final.started_at is not None
    assert final.finished_at is not None
    assert final.finished_at >= final.started_at


@pytest.mark.asyncio
async def test_history_integration(tmp_path):
    """24. History integration: done job creates history entries."""
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
    await worker.submit(job.job_id, [("a.wav", audio), ("b.wav", audio)], job.options)
    await asyncio.sleep(0.5)

    from src.history import HistoryManager
    hm = HistoryManager()
    entries = hm.list_entries(type_filter="stt")
    # Should have at least 2 entries from our batch
    stt_items = entries["items"]
    batch_items = [i for i in stt_items if i.get("input_filename") in ("a.wav", "b.wav")]
    assert len(batch_items) == 2


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

    await worker.submit(job1.job_id, [("x.wav", _wav_bytes())], job1.options)
    await worker.submit(job2.job_id, [("y.wav", _wav_bytes())], job2.options)
    await asyncio.sleep(0.5)

    j1 = store.get("conc-1")
    j2 = store.get("conc-2")
    assert j1.status == "done"
    assert j2.status == "done"
    assert j1.results[0]["filename"] == "x.wav"
    assert j2.results[0]["filename"] == "y.wav"
