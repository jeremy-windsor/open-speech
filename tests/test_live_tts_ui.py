"""Static checks for browser-side Live Reader behavior."""

from pathlib import Path


HTML = Path("src/static/index.html").read_text(encoding="utf-8")
JS = Path("src/static/app.js").read_text(encoding="utf-8")


def test_live_reader_controls_are_present():
    for element_id in (
        "live-reader-start",
        "live-reader-read-all",
        "live-reader-pause",
        "live-reader-stop",
        "live-reader-clear",
        "live-reader-mode",
        "live-reader-status",
    ):
        assert f'id="{element_id}"' in HTML


def test_live_reader_uses_wss_and_browser_audio_context():
    assert "location.protocol === 'https:' ? 'wss' : 'ws'" in JS
    assert "/v1/audio/speech/stream" in JS
    assert "reader.audioCtx.createBuffer" in JS
    assert "reader.audioCtx.destination" in JS
    assert "new Audio(" not in JS


def test_live_reader_ack_is_gated_on_local_playback_completion():
    assert "source.onended = () =>" in JS
    assert "type: 'playback.ack'" in JS
    assert "reader.stopping" in JS
    assert "source.start(startAt);" in JS


def test_live_reader_sends_text_in_bounded_chunks():
    assert "Math.min(2048, room)" in JS
    assert "type: 'input_text.append'" in JS
    assert "max_buffer_chars" in JS
    assert "reader.inflight.push(text)" in JS
    assert "reader.outbound.unshift({ type: 'text', text: rejected })" in JS


def test_live_reader_pauses_when_the_browser_is_hidden():
    assert "document.addEventListener('visibilitychange'" in JS
    assert "type: 'playback.pause'" in JS
    assert "reader.visibilityPaused" in JS
    assert "if (document.hidden)" in JS


def test_live_reader_snapshots_voice_settings_before_async_model_load():
    config_position = JS.index("const sessionConfig = {")
    model_load_position = JS.index("await ensureModelReady(model, 'tts')", config_position)
    assert config_position < model_load_position
    assert "session: reader.sessionConfig" in JS


def test_live_reader_resets_acceptance_window_after_cancel():
    assert "reader.acceptedChars = event.accepted_chars || 0" in JS
    assert "reader.inflight.length = 0" in JS
