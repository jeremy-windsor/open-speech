"""Behavior checks for the Speak model switch and reading default."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


JS = Path("src/static/app.js").read_text(encoding="utf-8")
HTML = Path("src/static/index.html").read_text(encoding="utf-8")


def run_node(source: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser behavior tests")
    subprocess.run([node, "-e", source], check=True, timeout=5)


def test_configured_default_wins_over_loaded_canary_until_user_selects_it():
    helpers = JS[JS.index("function selectTTSProvider"):JS.index("async function loadTTSProviders")]
    run_node(helpers + """
const models = [
  {id: 'qwen3/0.6b-custom-voice', provider: 'qwen3', state: 'loaded'},
  {id: 'kokoro', provider: 'kokoro', state: 'downloaded'},
];
const providers = ['kokoro', 'qwen3'];
if (selectTTSProvider(models, providers, '', 'kokoro') !== 'kokoro') process.exit(1);
if (selectTTSModel(models.filter(m => m.provider === 'kokoro'), '', 'kokoro') !== 'kokoro') process.exit(2);
if (selectTTSProvider(models, providers, 'qwen3', 'kokoro') !== 'qwen3') process.exit(3);
if (selectTTSModel(models.filter(m => m.provider === 'qwen3'), 'qwen3/0.6b-custom-voice', 'kokoro') !== 'qwen3/0.6b-custom-voice') process.exit(4);
""")


def test_switch_confirmation_can_cancel_before_eviction():
    ensure_ready = JS[JS.index("async function ensureModelReady"):JS.index("function pushHistory")]
    run_node(ensure_ready + """
const state = {modelsCache: [], defaultTtsModel: 'kokoro'};
let loaded = false;
let loadCalls = 0;
let refreshCalls = 0;
let confirmCalls = 0;
const window = {confirm() { confirmCalls++; return false; }};
function setButtonState() {}
function providerFromModel() { return 'qwen3'; }
async function loadModel() { loadCalls++; loaded = true; }
async function downloadModel() { throw new Error('Unexpected download'); }
async function refreshModels() { refreshCalls++; }
async function api(url) {
  if (url === '/api/models') return {
    default_tts_model: 'kokoro',
    models: [{id: 'kokoro', type: 'tts', state: 'loaded'},
             {id: 'qwen3/0.6b-custom-voice', type: 'tts', state: 'provider_installed'}],
  };
  if (url.endsWith('/status')) return {state: loaded ? 'loaded' : 'provider_installed'};
  throw new Error('Unexpected API request: ' + url);
}
async function check() {
  if (await ensureModelReady('qwen3/0.6b-custom-voice', 'tts') !== false) process.exit(1);
  if (loadCalls !== 0 || confirmCalls !== 1) process.exit(2);
  window.confirm = () => { confirmCalls++; return true; };
  if (await ensureModelReady('qwen3/0.6b-custom-voice', 'tts') !== true) process.exit(3);
  if (loadCalls !== 1 || confirmCalls !== 2 || refreshCalls !== 1) process.exit(4);
}
check().catch(() => process.exit(5));
""")


def test_unavailable_worker_is_not_described_as_uninstalled():
    badge = JS[JS.index("function getStateBadge"):JS.index("function getModelHint")]
    run_node(badge + """
const result = getStateBadge({state: 'provider_unavailable', provider_available: false});
if (result.text !== '✗ Worker unavailable') process.exit(1);
""")
    assert 'id="tts-restore-default"' in HTML
    assert "renderUnavailableWorkerCard(p, ms)" in JS
    assert "Check its health and manifest; rebuilding the core image will not fix this state" in JS


def test_saved_profile_does_not_silently_fall_back_to_kokoro():
    apply_profile = JS[JS.index("async function applyProfile"):JS.index("async function saveAsProfile")]
    run_node(apply_profile + """
async function api() { return {model: 'qwen3/0.6b-custom-voice'}; }
function getTTSModels() { return [{id: 'kokoro'}]; }
function byId() { throw new Error('Changed the UI before checking availability'); }
async function check() {
  try {
    await applyProfile('unavailable');
    process.exit(1);
  } catch (error) {
    if (!error.message.includes('is unavailable')) process.exit(2);
  }
}
check().catch(() => process.exit(3));
""")


def test_restore_action_loads_default_and_reselects_it():
    restore = JS[JS.index("async function restoreDefaultTTSModel"):JS.index("async function deleteModel")]
    run_node(restore + """
const state = {defaultTtsModel: 'kokoro', liveReader: null, ttsPreferredProvider: 'qwen3', ttsPreferredModel: 'qwen3/0.6b-custom-voice'};
const button = {disabled: false};
const PROVIDER_DISPLAY = {kokoro: 'Kokoro'};
let loadCalls = 0;
function getTTSModels() { return [{id: 'kokoro', provider: 'kokoro', state: 'downloaded'}]; }
function byId() { return button; }
async function loadModel(modelId) { if (modelId !== 'kokoro') process.exit(1); loadCalls++; }
async function refreshModels() {}
async function loadTTSProviders() {
  if (state.ttsPreferredProvider !== 'kokoro' || state.ttsPreferredModel !== 'kokoro') process.exit(2);
}
function updateRestoreDefaultButton() {}
function showToast() {}
async function check() {
  await restoreDefaultTTSModel();
  if (loadCalls !== 1 || button.disabled) process.exit(3);
}
check().catch(() => process.exit(4));
""")


def test_restore_action_reloads_default_even_when_inventory_is_stale():
    restore = JS[JS.index("async function restoreDefaultTTSModel"):JS.index("async function deleteModel")]
    run_node(restore + """
const state = {defaultTtsModel: 'kokoro', liveReader: null};
const button = {disabled: false};
const PROVIDER_DISPLAY = {kokoro: 'Kokoro'};
let loadCalls = 0;
function getTTSModels() { return [{id: 'kokoro', provider: 'kokoro', state: 'loaded'}]; }
function byId() { return button; }
async function loadModel() { loadCalls++; }
async function refreshModels() {}
async function loadTTSProviders() {}
function updateRestoreDefaultButton() {}
function showToast() {}
async function check() {
  await restoreDefaultTTSModel();
  if (loadCalls !== 1 || button.disabled) process.exit(1);
}
check().catch(() => process.exit(2));
""")


def test_restore_button_is_available_when_selection_differs_from_loaded_default():
    update = JS[JS.index("function updateRestoreDefaultButton"):JS.index("function selectTTSProvider")]
    run_node(update + """
const state = {defaultTtsModel: 'kokoro'};
const PROVIDER_DISPLAY = {kokoro: 'Kokoro'};
const button = {hidden: true};
const modelSelect = {value: 'qwen3/0.6b-custom-voice'};
function byId(id) { return id === 'tts-restore-default' ? button : modelSelect; }
function getTTSModels() { return [{id: 'kokoro', provider: 'kokoro', state: 'loaded'}]; }
updateRestoreDefaultButton();
if (button.hidden) process.exit(1);
modelSelect.value = 'kokoro';
updateRestoreDefaultButton();
if (!button.hidden) process.exit(2);
""")


def test_late_voice_response_cannot_replace_newer_model_controls():
    load_voices = JS[JS.index("async function loadTTSVoices"):JS.index("async function downloadModel")]
    run_node(load_voices + """
const state = {ttsPreferredModel: '', ttsCaps: {}, ttsVoices: [], ttsLibraryVoices: [], ttsVoiceRequestId: 0};
let blendVoices = [];
let resolveKokoro;
let lastStatus = '';
const kokoroCaps = new Promise((resolve) => { resolveKokoro = resolve; });
const modelSelect = {value: 'kokoro'};
const voiceSelect = {innerHTML: '', value: '', options: []};
function byId(id) { return id === 'tts-model' ? modelSelect : voiceSelect; }
function esc(value) { return value; }
function renderAdvancedControls() {}
function updateTTSModelStatus(model) { lastStatus = model; }
async function fetchTTSCapabilities(model) {
  return model === 'kokoro' ? kokoroCaps : {model, voice_clone: false, voice_blend: false};
}
async function fetchVoices(model) { return [{id: model === 'kokoro' ? 'af_heart' : 'Vivian'}]; }
async function check() {
  const first = loadTTSVoices();
  modelSelect.value = 'qwen3/0.6b-custom-voice';
  const second = loadTTSVoices();
  await second;
  resolveKokoro({model: 'kokoro', voice_clone: false, voice_blend: true});
  await first;
  if (state.ttsCaps.model !== 'qwen3/0.6b-custom-voice') process.exit(1);
  if (state.ttsVoices[0].id !== 'Vivian' || lastStatus !== 'qwen3/0.6b-custom-voice') process.exit(2);
}
check().catch(() => process.exit(3));
""")
