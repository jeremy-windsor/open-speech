"""UI orchestration regression checks for core model flows."""

from pathlib import Path


def _index_html() -> str:
    return Path("src/static/index.html").read_text(encoding="utf-8")


def test_ensure_model_ready_function_exists_and_chains_states():
    html = _index_html()
    assert "async function ensureModelReady(modelId, kind='tts')" in html
    assert "status.state === 'provider_missing'" in html
    assert "status.state === 'provider_installed' || status.state === 'available'" in html
    assert "status.state === 'downloaded'" in html
    assert "installProvider(modelId)" in html
    assert "await downloadModel(modelId);" in html
    assert "await loadModel(modelId);" in html


def test_generate_and_transcribe_paths_use_ensure_model_ready():
    html = _index_html()
    assert "ensureModelReady(model, 'tts')" in html
    assert html.count("ensureModelReady(model, 'stt')") >= 2


def test_defaults_and_provider_focus():
    html = _index_html()
    assert "m.provider === 'faster-whisper'" in html
    assert "m.provider === 'kokoro' || m.provider === 'piper'" in html
    assert "const providerRank = { kokoro: 0, piper: 1 };" in html


def test_advanced_controls_are_grouped_and_capability_gated():
    html = _index_html()
    assert 'id="tts-advanced"' in html
    assert "const hasAnyAdvanced = supportsVoiceBlend || supportsVoiceDesign || supportsVoiceClone" in html
    assert "advancedDetails.style.display = hasAnyAdvanced ? '' : 'none';" in html
