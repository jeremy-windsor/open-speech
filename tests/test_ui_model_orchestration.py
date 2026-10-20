"""UI orchestration regression checks for core model flows."""

import shutil
import subprocess
from pathlib import Path

import pytest


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
    assert js.count("if (!await ensureModelReady(model, 'tts'))") == 2
    assert js.count("await ensureModelReady(model, 'stt');") >= 2


def test_defaults_and_provider_focus():
    js = _app_js()
    assert "m.provider === 'faster-whisper'" in js
    assert "m.provider === 'kokoro'" in js
    assert "const providerRank = { kokoro: 0, piper: 1 };" in js
    assert "state.defaultTtsModel = data.default_tts_model" in js
    assert "selectTTSProvider(models, providers, state.ttsPreferredProvider, state.defaultTtsModel)" in js


def test_model_loads_show_a_notice_until_they_finish():
    js = _app_js()
    ensure = js[js.index("async function ensureModelReadyWithButton"):js.index("function readLocalHistory")]
    assert "announce('Loading');" in ensure and "announce('Downloading');" in ensure
    assert "} finally {\n    if (closeNotice) closeNotice();" in ensure
    run_op = js[js.index("async function runModelOp"):js.index("function initTheme")]
    assert "const closeNotice = showModelNotice(modelId" in run_op and "} finally {\n    closeNotice();" in run_op
    restore = js[js.index("async function restoreDefaultTTSModel"):js.index("async function deleteModel")]
    assert "showModelNotice(modelId" in restore
    css = Path("src/static/app.css").read_text(encoding="utf-8")
    # Plain toasts fade out after a few seconds; the loading notice only animates in.
    rule = css[css.index(".toast.toast-loading {"):]
    rule = rule[:rule.index("}")]
    assert "toast-in" in rule and "toast-out" not in rule


def test_model_notice_is_shared_per_model_and_closes_with_its_last_caller():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser behavior tests")
    js = _app_js()
    source = (
        js[js.index("const PROVIDER_DISPLAY = {"):js.index("function byId")]
        + js[js.index("function providerFromModel"):js.index("function getTTSModels")]
        + js[js.index("function formatModelName"):js.index("function updateTTSModelStatus")]
        + js[js.index("function modelLabel"):js.index("function setButtonState")]
        + r"""
const check = (cond, label) => { if (!cond) { console.error(`failed: ${label}`); process.exit(1); } };
class FakeEl {
  constructor() { this.children = []; this.parts = {}; this.removed = false; }
  set innerHTML(html) {
    for (const name of ['strong', '.toast-detail', '.toast-elapsed']) this.parts[name] = { textContent: '' };
  }
  querySelector(selector) { return this.parts[selector]; }
  appendChild(el) { this.children.push(el); }
  remove() { this.removed = true; }
}
const root = new FakeEl();
const document = { createElement: () => new FakeEl() };
function byId() { return root; }
const state = { modelsCache: [
  { id: 'kokoro', type: 'tts', state: 'loaded' },
  { id: 'cosyvoice/3-0.5b', type: 'tts', state: 'available' },
  { id: 'deepdml/faster-whisper-large-v3-turbo-ct2', type: 'stt', state: 'downloaded' },
] };
function classifyKind(model) { return model.type; }
global.setInterval = () => 1;
global.clearInterval = () => {};

check(modelLabel('kokoro') === 'Kokoro', 'provider name');
check(modelLabel('cosyvoice/3-0.5b') === 'CosyVoice 3-0.5b', 'provider name and variant');
check(modelLabel('deepdml/faster-whisper-large-v3-turbo-ct2') === 'faster-whisper-large-v3-turbo-ct2', 'unknown provider');
check(otherLoadedTts('cosyvoice/3-0.5b')?.id === 'kokoro', 'loading a voice model unloads the loaded one');
check(otherLoadedTts('kokoro') === null, 'nothing else loaded');
check(otherLoadedTts('deepdml/faster-whisper-large-v3-turbo-ct2') === null, 'a transcription model unloads no voice');

const first = showModelNotice('cosyvoice/3-0.5b', { unloads: 'kokoro' });
const el = root.children[0];
check(el.parts.strong.textContent === 'Loading CosyVoice 3-0.5b…', 'names the model');
check(el.parts['.toast-detail'].textContent.startsWith('Unloading Kokoro first.'), 'says what it unloads');
const second = showModelNotice('cosyvoice/3-0.5b', { verb: 'Downloading' });
check(root.children.length === 1, 'one notice per model');
check(el.parts.strong.textContent === 'Downloading CosyVoice 3-0.5b…', 'a second caller updates it');
first();
first();
check(!el.removed, 'stays while a caller still waits');
second();
check(el.removed, 'closes with the last caller');
showModelNotice('cosyvoice/3-0.5b')();
check(root.children.length === 2 && root.children[1].removed, 'a later load gets a fresh notice');
"""
    )
    result = subprocess.run([node, "-e", source], check=False, timeout=10, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
