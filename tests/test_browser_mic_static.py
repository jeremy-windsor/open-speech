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
    assert "const selectedDeviceId = byId('mic-select')?.value || '';" in js
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


def test_browser_mic_unavailable_saved_or_selected_device_falls_back_to_default():
    js = _app_js()
    assert "const hasSavedDevice = !!savedDeviceId && audioInputs.some((device) => device.deviceId === savedDeviceId);" in js
    assert "if (savedDeviceId && !hasSavedDevice) saveMicDeviceId('');" in js
    assert "catch (selectedDeviceError)" in js
    assert "showToast('Selected microphone unavailable; using default microphone');" in js
    assert "throw selectedDeviceError" not in js
