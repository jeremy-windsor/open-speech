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
    ("/api/composer/render", "composer_manager", {"sample_rate": 0, "tracks": []}),
    ("/api/composer/render", "composer_manager", {"sample_rate": -16000, "tracks": []}),
    ("/api/composer/render", "composer_manager", {"format": "flac", "tracks": []}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "volume": "NaN"}]}),
    ("/api/composer/render", "composer_manager", {"tracks": [{"source_path": "a.wav", "offset_s": "Infinity"}]}),
])
def test_invalid_render_parameters_never_reach_renderer(monkeypatch, endpoint, manager, payload):
    renderer = MagicMock(return_value={"duration_ms": 1})
    monkeypatch.setattr(getattr(main_module, manager), "render", renderer)

    response = TestClient(app).post(endpoint, json=payload)

    assert response.status_code == 422
    renderer.assert_not_called()
