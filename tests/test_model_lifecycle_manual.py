from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app, model_manager
from src.model_manager import ModelLifecycleError


client = TestClient(app, raise_server_exceptions=False)


def test_provider_install_endpoint_returns_job_id_immediately():
    with patch.object(model_manager, "install_provider", side_effect=lambda **_: time.sleep(0.2) or {"status": "already_installed", "provider": "kokoro", "stdout": "", "stderr": ""}):
        r = client.post("/api/providers/install", json={"model": "kokoro"})

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "installing"
    assert body["job_id"]


def test_provider_install_status_endpoint_done_flow():
    with patch.object(model_manager, "install_provider", return_value={"status": "already_installed", "provider": "kokoro", "stdout": "Requirement already satisfied\n", "stderr": ""}):
        start = client.post("/api/providers/install", json={"model": "kokoro"}).json()

    job_id = start["job_id"]
    deadline = time.time() + 2
    status = None
    while time.time() < deadline:
        poll = client.get(f"/api/providers/install/{job_id}")
        assert poll.status_code == 200
        status = poll.json()
        if status["status"] in {"done", "failed"}:
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "done"
    assert "Requirement already satisfied" in status["output"]
    assert status["error"] == ""


def test_provider_install_status_endpoint_surfaces_error_output():
    err = ModelLifecycleError(
        message="Failed to install provider 'kokoro'.",
        code="provider_install_failed",
        model_id="kokoro",
        provider="kokoro",
        action="install_provider",
        details={"stdout": "Collecting foo\nERROR: No matching distribution found for foo"},
    )
    with patch.object(model_manager, "install_provider", side_effect=err):
        start = client.post("/api/providers/install", json={"model": "kokoro"}).json()

    job_id = start["job_id"]
    deadline = time.time() + 2
    status = None
    while time.time() < deadline:
        poll = client.get(f"/api/providers/install/{job_id}")
        assert poll.status_code == 200
        status = poll.json()
        if status["status"] == "failed":
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "failed"
    assert "No matching distribution found" in status["error"]


def test_provider_install_status_endpoint_404_for_unknown_job():
    r = client.get("/api/providers/install/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "install_job_not_found"


def test_load_endpoint_returns_actionable_provider_missing_error():
    err = ModelLifecycleError(
        message="Provider 'kokoro' is not installed for model 'kokoro'. Install it first.",
        code="provider_missing",
        model_id="kokoro",
        provider="kokoro",
        action="load",
    )
    with patch.object(model_manager, "load", side_effect=err):
        r = client.post("/api/models/kokoro/load")
    assert r.status_code == 400
    body = r.json()["error"]
    assert body["code"] == "provider_missing"
    assert "Install" in body["message"]


def test_download_endpoint_exists_and_returns_downloaded_state():
    with patch.object(model_manager, "download") as mock_download:
        from src.model_manager import ModelInfo, ModelState
        mock_download.return_value = ModelInfo(id="kokoro", type="tts", provider="kokoro", state=ModelState.DOWNLOADED)
        r = client.post("/api/models/kokoro/download")
    assert r.status_code == 200
    assert r.json()["state"] == "downloaded"


def test_delete_artifacts_scoped_to_managed_dirs(tmp_path: Path):
    managed = tmp_path / "hub"
    managed.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = managed / "models--Systran--faster-whisper-base"
    target.mkdir()

    with patch.object(model_manager, "_hf_cache_roots", return_value=[managed]):
        with patch.object(model_manager, "_candidate_artifact_paths", return_value=[outside, target]):
            result = model_manager.delete_artifacts("Systran/faster-whisper-base")

    assert result["status"] == "deleted"
    assert target.exists() is False
    assert outside.exists() is True


def test_status_transitions_include_provider_states():
    with patch("src.model_manager._check_provider", return_value=False):
        info = model_manager.status("kokoro")
        assert info.state.value == "provider_missing"
