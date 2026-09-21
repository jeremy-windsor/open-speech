"""Voice Lab browser contract checks for the dependency-free web UI."""

from pathlib import Path


def _source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _function(js: str, name: str, next_name: str) -> str:
    return js[js.index(f"function {name}"):js.index(f"function {next_name}")]


def test_voice_lab_is_a_first_class_accessible_tab():
    html = _source("src/static/index.html")
    assert 'data-tab="voicelab"' in html
    assert 'id="panel-voicelab"' in html
    assert 'aria-labelledby="tab-voicelab"' in html
    assert html.index('id="tab-speak"') < html.index('id="tab-voicelab"') < html.index('id="tab-models"')
    for element_id in (
        "vl-record", "vl-file", "vl-draft-audio", "vl-name", "vl-transcript",
        "vl-transcript-confirm", "vl-save", "vl-assets-body", "vl-clone-model",
    ):
        assert f'id="{element_id}"' in html


def test_voice_lab_normalizes_recordings_and_uploads_to_pcm_wav():
    js = _source("src/static/app.js")
    assert "function encodeWavPcm16(" in js
    assert "writeText(0, 'RIFF');" in js
    assert "writeText(8, 'WAVE');" in js
    assert "decodeAudioData" in js
    assert "MediaRecorder" not in js


def test_voice_lab_recording_is_isolated_from_transcribe_mic_state():
    js = _source("src/static/app.js")
    start = _function(js, "startVoiceLabRecording", "toggleVoiceLabRecording")
    assert "if (state.sttRecording)" in start
    assert "state.voiceLab.recording = recording;" in start
    assert "state.audioCtx =" not in start
    assert "state.mediaStream =" not in start
    assert "showToast('Stop the Voice Lab recording first'" in js


def test_voice_lab_save_requires_verified_exact_transcript_or_explicit_skip():
    js = _source("src/static/app.js")
    gate = _function(js, "updateVoiceLabSaveState", "setVoiceLabDraft")
    save = _function(js, "saveVoiceLabDraft", "suggestVoiceLabTranscript")
    assert "lab.transcriptSkipped || (transcript && lab.transcriptVerified)" in gate
    assert "Confirm that the transcript matches the recording word-for-word" in save
    assert "form.append('transcript'" in save


def test_clone_test_uses_saved_asset_and_omits_unsupported_speed():
    js = _source("src/static/app.js")
    clone = _function(js, "cloneTestVoiceLabAsset", "openVoiceLabTranscriptEditor")
    assert "fetch('/v1/audio/speech'" in clone
    assert "voice_library_ref: name" in clone
    assert "reference_audio" not in clone
    assert "speed:" not in clone


def test_voice_lab_profile_and_delete_paths_preserve_reference_identity():
    js = _source("src/static/app.js")
    assert "profile.reference_audio_id === name" in js
    assert "reference_audio_id: name" in js
    assert "Saved profile reference ${profile.reference_audio_id} is missing" in js


def test_speak_reference_selector_is_visible_outside_advanced_controls():
    html = _source("src/static/index.html")
    js = _source("src/static/app.js")
    reference_row = html[html.index('id="tts-reference-row"'):html.index('id="effects-panel"')]
    advanced = html[html.index('id="tts-advanced"'):html.index('id="audio-player"')]
    assert 'id="tts-voice-library-ref"' in reference_row
    assert 'id="tts-voice-library-ref"' not in advanced
    assert "state.ttsCaps.clone_transcript_required && !selectedReference" in js
