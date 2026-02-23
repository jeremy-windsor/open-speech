"""UI orchestration regression checks for core model flows."""

from pathlib import Path


def _app_js() -> str:
    return Path("src/static/app.js").read_text(encoding="utf-8")


def test_ensure_model_ready_function_exists_and_chains_states():
    js = _app_js()
    assert "async function ensureModelReady(modelId, kind = 'tts')" in js
    assert "status.state === 'provider_missing'" in js
    assert "status2.state === 'downloaded' || status2.state === 'ready'" in js
    assert "await downloadModel(modelId)" in js
    assert "await loadModel(modelId)" in js


def test_generate_and_transcribe_paths_use_ensure_model_ready():
    js = _app_js()
    assert "await ensureModelReady(model, 'tts');" in js
    assert js.count("await ensureModelReady(model, 'stt');") >= 2


def test_defaults_and_provider_focus():
    js = _app_js()
    assert "m.provider === 'faster-whisper'" in js
    assert "m.provider === 'kokoro'" in js
    assert "m.provider === 'piper'" in js
    assert "const providerRank = { kokoro: 0, piper: 1 };" in js
