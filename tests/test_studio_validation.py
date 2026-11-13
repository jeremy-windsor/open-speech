"""Reject invalid studio parameters before starting synthesis or allocating audio."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app


@pytest.mark.parametrize("endpoint,manager,payload", [
    ("/api/conversations/example/render", "conversation_manager", {"sample_rate": 0}),
    ("/api/conversations/example/render", "conversation_manager", {"sample_rate": -16000}),
    ("/api/conversations/example/render", "conversation_manager", {"format": "unknown"}),
    # Buffers are sized by the sample rate and offsets; one request must not demand terabytes.
    ("/api/conversations/example/render", "conversation_manager", {"sample_rate": 10**12}),
    ("/api/conversations/example/render", "conversation_manager", {"sample_rate": 7999}),
    ("/api/conversations/example/render", "conversation_manager", {"sample_rate": 48001}),
    ("/api/composer/render", "composer_manager", {"sample_rate": 0, "tracks": []}),
    ("/api/composer/render", "composer_manager", {"sample_rate": -16000, "tracks": []}),
    ("/api/composer/render", "composer_manager", {"sample_rate": 10**9, "tracks": [{"source_path": "a.wav"}]}),
    ("/api/composer/render", "composer_manager", {"format": "flac", "tracks": []}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "volume": "NaN"}]}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "offset_s": "Infinity"}]}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "offset_s": 365 * 86400.0}]}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav"}] * 65}),
])
def test_invalid_render_parameters_never_reach_renderer(monkeypatch, endpoint, manager, payload):
    renderer = MagicMock(return_value={"duration_ms": 1})
    monkeypatch.setattr(getattr(main_module, manager), "render", renderer)

    response = TestClient(app).post(endpoint, json=payload)

    assert response.status_code == 422
    renderer.assert_not_called()


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("endpoint,manager,payload", [
    ("/api/conversations/example/render", "conversation_manager", {}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "offset_s": 7200}]}),
])
def test_common_render_sample_rates_reach_renderer(monkeypatch, endpoint, manager, payload, sample_rate):
    renderer = MagicMock(return_value={"duration_ms": 1})
    monkeypatch.setattr(getattr(main_module, manager), "render", renderer)

    response = TestClient(app).post(endpoint, json={**payload, "sample_rate": sample_rate})

    assert response.status_code == 200
    assert renderer.call_args.kwargs["sample_rate"] == sample_rate
