from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app, model_manager
from src.model_manager import ModelLifecycleError


client = TestClient(app, raise_server_exceptions=False)


def test_provider_install_endpoint_idempotent():
    with patch.object(model_manager, "install_provider", return_value={"status": "already_installed", "provider": "kokoro"}):
        r = client.post("/api/providers/install", json={"model": "kokoro"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "already_installed"
    assert body["provider"] == "kokoro"


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
