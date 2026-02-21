"""Tests for TTS UI capability-based feature gating in src/static/app.js."""

from __future__ import annotations

from pathlib import Path


def _app_js() -> str:
    return Path("src/static/app.js").read_text(encoding="utf-8")


def test_tts_capability_gates_are_rendered_dynamically():
    js = _app_js()
    assert "function renderAdvancedControls(caps)" in js
    assert "if (caps.voice_clone)" in js
    assert "caps.voice_blend" in js
    assert "if (caps.instructions)" in js
    assert "byId('tts-stream-group').hidden = !caps.streaming;" in js


def test_tts_model_change_fetches_capabilities_and_voices():
    js = _app_js()
    assert "state.ttsCaps = await fetchTTSCapabilities(model);" in js
    assert "state.ttsVoices = await fetchVoices(model);" in js
    assert "renderAdvancedControls(state.ttsCaps);" in js
