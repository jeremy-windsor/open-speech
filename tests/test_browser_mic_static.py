"""Browser microphone regression checks in src/static/app.js."""

from pathlib import Path


def _app_js() -> str:
    return Path("src/static/app.js").read_text(encoding="utf-8")


def _index_html() -> str:
    return Path("src/static/index.html").read_text(encoding="utf-8")


def _app_css() -> str:
    return Path("src/static/app.css").read_text(encoding="utf-8")


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


def test_browser_mic_selector_is_rendered_near_start_button():
    html = _index_html()
    assert 'id="mic-select"' in html
    assert "Default microphone" in html
    assert html.index('id="mic-btn"') < html.index('id="mic-select"') < html.index('id="mic-waveform"')

    css = _app_css()
    assert ".mic-device-field" in css
    assert "text-overflow: ellipsis;" in css


def test_browser_mic_devices_are_enumerated_from_audio_inputs():
    js = _app_js()
    assert "async function loadMicDevices()" in js
    assert "navigator.mediaDevices.enumerateDevices()" in js
    assert "device.kind === 'audioinput'" in js
    assert "device.label || `Microphone ${index + 1}`" in js
    assert "await refreshMicDevicesAfterPermission();" in js


def test_browser_mic_selected_device_id_is_passed_to_get_user_media():
    js = _app_js()
    assert "async function getMicStream(selectId = 'mic-select')" in js
    assert "const selectedDeviceId = byId(selectId)?.value || '';" in js
    assert "audio: { deviceId: { exact: selectedDeviceId } }," in js
    assert "const stream = await getMicStream();" in js
    assert "const stream = await navigator.mediaDevices.getUserMedia({ audio: true });" in js


def test_browser_mic_selected_device_persists_as_opaque_device_id_only():
    js = _app_js()
    assert "const MIC_DEVICE_STORAGE_KEY = 'open-speech-mic-device';" in js
    assert "localStorage.getItem(MIC_DEVICE_STORAGE_KEY)" in js
    assert "localStorage.setItem(MIC_DEVICE_STORAGE_KEY, deviceId)" in js
    assert "localStorage.removeItem(MIC_DEVICE_STORAGE_KEY)" in js
    assert "byId('mic-select')?.addEventListener('change', (e) => saveMicDeviceId(e.target.value));" in js


def test_browser_mic_refreshes_on_devicechange_when_supported():
    js = _app_js()
    assert "navigator.mediaDevices.addEventListener('devicechange', () => loadMicDevices().catch(() => {}));" in js
    assert "navigator.mediaDevices.ondevicechange = () => loadMicDevices().catch(() => {});" in js
    assert "loadMicDevices()," in js


def test_browser_mic_device_list_is_shared_with_voice_lab():
    js = _app_js()
    assert "return ['mic-select', 'vl-mic-select'].map(byId).filter(Boolean);" in js
    assert "const stream = await getMicStream('vl-mic-select');" in js
    assert "if (state.voiceLab.recording)" in js


def test_browser_mic_unavailable_saved_or_selected_device_falls_back_to_default():
    js = _app_js()
    assert "const hasSavedDevice = !!savedDeviceId && audioInputs.some((device) => device.deviceId === savedDeviceId);" in js
    assert "if (savedDeviceId && !hasSavedDevice) saveMicDeviceId('');" in js
    assert "catch (selectedDeviceError)" in js
    assert "showToast('Selected microphone unavailable; using default microphone');" in js
    assert "throw selectedDeviceError" not in js


def test_frontend_code_ready_accessibility_and_hidden_layout_fixes():
    html = _index_html()
    css = _app_css()
    js = _app_js()

    assert "[hidden] { display: none !important; }" in css
    assert "--text2: #9a9ab2;" in css
    assert ".visually-hidden" in css
    assert '<input id="stt-file" type="file" accept="audio/*" class="visually-hidden">' in html
    assert '<input id="tts-upload" type="file" accept=".txt,text/plain" class="visually-hidden">' in html
    assert '<input id="stt-file" type="file" accept="audio/*" hidden>' not in html
    assert '<input id="tts-upload" type="file" accept=".txt,text/plain" hidden>' not in html

    for name in ("transcribe", "speak", "models", "history", "studio", "settings"):
        assert f'id="tab-{name}"' in html
        assert f'aria-controls="panel-{name}"' in html
        assert f'id="panel-{name}"' in html
        assert f'aria-labelledby="tab-{name}"' in html

    for selector in (
        ".tab:focus-visible",
        ".models-tab:focus-visible",
        ".btn:focus-visible",
        "input:focus-visible",
        "select:focus-visible",
        "textarea:focus-visible",
        "summary:focus-visible",
        ".dropzone:focus-within",
        ".provider-card-toggle:focus-visible",
        ".visually-hidden:focus-visible + .btn",
    ):
        assert selector in css

    assert '@media (prefers-reduced-motion: reduce)' in css
    assert "animation: none !important;" in css
    assert "transition: none !important;" in css

    assert 'role="tablist" aria-label="Model categories"' in html
    assert 'id="models-tab-tts"' in html
    assert 'aria-controls="models-tts-panel"' in html
    assert 'role="tabpanel" aria-labelledby="models-tab-tts"' in html
    assert "ttsPanel.hidden = !active;" in js
    assert "sttPanel.hidden = !active;" in js
    assert "t.setAttribute('aria-selected', active ? 'true' : 'false');" in js

    assert "function toggleProviderCard(button)" in js
    assert 'class="provider-card-toggle" type="button" aria-expanded="true" aria-controls="${bodyId}"' in js
    assert '<span class="chevron" aria-hidden="true">▼</span>' in js
    assert "const header = button.closest('.provider-card-header');" in js
    assert "button.setAttribute('aria-expanded', collapsed ? 'false' : 'true');" in js
    assert "if (body) body.hidden = collapsed;" in js
