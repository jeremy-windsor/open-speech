"""Browser microphone regression checks in src/static/app.js."""

from pathlib import Path


def _app_js() -> str:
    return Path("src/static/app.js").read_text(encoding="utf-8")


def test_browser_mic_stream_uses_selected_model_and_actual_audio_context_rate():
    js = _app_js()
    assert "const audioCtx = new AudioContext();" in js
    assert "new AudioContext({ sampleRate: 16000 })" not in js
    assert "const sampleRate = Math.round(audioCtx.sampleRate || 16000);" in js
    assert "sample_rate: String(sampleRate)" in js
    assert "if (model) qs.set('model', model);" in js


def test_browser_mic_history_only_records_speech_final_transcripts():
    js = _app_js()
    assert "if (msg.speech_final === true)" in js
    assert "finalSegments: (state.sttSession?.finalSegments || 0) + 1" in js
    assert "pushHistory(HISTORY_KEYS.stt, {" in js


def test_browser_mic_stop_sends_graceful_stop_before_socket_close():
    js = _app_js()
    assert "const MIC_STOP_GRACE_MS = 4000;" in js
    assert "ws.send(JSON.stringify({ type: 'stop' }));" in js
    assert "function clearMicStopTimer(ws = null)" in js
    assert "if (ws && state.micStopTimer.ws !== ws) return;" in js
    assert "setTimeout(() => {" in js
    assert "state.micStopTimer = { id: timerId, ws };" in js
    assert "}, MIC_STOP_GRACE_MS);" in js
