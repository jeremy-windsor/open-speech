const state = {
  ttsCaps: {},
  ttsVoices: [],
  ttsAudioBlob: null,
  ttsAudioUrl: null,
  mediaRecorder: null,
  mediaStream: null,
  ws: null,
  sttRecording: false,
  sttChunkTimer: null,
  profiles: [],
  defaultProfileId: null,
  history: { items: [], total: 0, limit: 50, offset: 0, type: "" },
  modelsCache: [],
  modelOps: {},
  modelsBusy: false,
  ttsPreferredProvider: '',
  ttsPreferredModel: '',
  currentConversationId: null,
  currentConversation: null,
};
let composerTracks = [];
let blendVoices = [];

const HISTORY_KEYS = {
  tts: 'open-speech-tts-history',
  stt: 'open-speech-stt-history',
};
const BTN_STATES = {
  idle: { text: '▶ Generate', loading: false },
  checking: { text: 'Checking model…', loading: true },
  installing: { text: 'Installing provider…', loading: true },
  downloading: { text: 'Downloading model…', loading: true },
  loading: { text: 'Loading model…', loading: true },
  generating: { text: 'Generating…', loading: true },
};
const PROVIDER_DISPLAY = {
  'kokoro': 'Kokoro',
  'piper': 'Piper',
  'pocket-tts': 'Pocket TTS',
  'qwen3-tts': 'Qwen3 TTS',
  'fish-speech': 'Fish Speech',
  'f5-tts': 'F5 TTS',
  'xtts': 'XTTS v2',
};
function byId(id) { return document.getElementById(id); }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function formatSize(mb) {
  if (!mb) return '';
  if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`;
  return `${mb} MB`;
}
async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      msg = payload?.error?.message
        || payload?.detail?.message
        || payload?.detail
        || (typeof payload?.error === 'string' ? payload.error : null)
        || msg;
    } catch { }
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res;
}
function showToast(message, type = '') {
  const root = byId('toast-root');
  const el = document.createElement('div');
  el.className = `toast ${type ? `toast-${type}` : ''}`;
  el.textContent = message;
  root.appendChild(el);
  setTimeout(() => el.remove(), 4200);
  return el;
}
function setButtonState(btnId, stateKey, text) {
  const btn = byId(btnId);
  const stateDef = BTN_STATES[stateKey] || BTN_STATES.idle;
  btn.textContent = text || stateDef.text;
  btn.classList.toggle('loading', !!stateDef.loading);
  btn.disabled = !!stateDef.loading;
}
function statusSuffix(stateName) {
  if (stateName === 'loaded') return '● Loaded';
  if (stateName === 'downloaded' || stateName === 'ready') return '○ Cached';
  if (stateName === 'provider_installed' || stateName === 'available') return '○ Installed (cache on load)';
  if (stateName === 'provider_missing') return '⚠ Missing';
  return '○ Unknown';
}
function classifyKind(model) {
  if (model.type) return model.type;
  const id = model.id || '';
  if (id.includes('whisper') || id.includes('vosk') || id.includes('moonshine')) return 'stt';
  return 'tts';
}
function providerFromModel(modelId) {
  const id = modelId || '';
  if (!id) return 'kokoro';
  if (id.includes('/')) return id.split('/')[0];
  if (id.startsWith('kokoro-')) return 'kokoro';
  return id;
}
function getTTSModels() {
  return (state.modelsCache || []).filter((m) => classifyKind(m) === 'tts' && m.state !== 'provider_missing');
}
function formatModelName(model) {
  const id = model?.id || '';
  const provider = model?.provider || providerFromModel(id);
  if (!id) return '—';
  if (id.includes('/')) return id.split('/').slice(1).join('/');
  if (provider && id.startsWith(`${provider}-`)) return `${id.slice(provider.length + 1)} (${provider})`;
  return id;
}
function updateTTSModelStatus(modelId) {
  const statusEl = byId('tts-model-status');
  if (!statusEl) return;
  const model = getTTSModels().find((m) => m.id === modelId);
  statusEl.classList.remove('loaded', 'downloaded', 'available');
  if (!model) {
    statusEl.textContent = '';
    return;
  }
  if (model.state === 'loaded') {
    statusEl.textContent = '● Loaded';
    statusEl.classList.add('loaded');
    return;
  }
  if (model.state === 'downloaded' || model.state === 'ready') {
    statusEl.textContent = '○ Downloaded';
    statusEl.classList.add('downloaded');
    return;
  }
  statusEl.textContent = '○ Available';
  statusEl.classList.add('available');
}
async function loadTTSProviders() {
  if (!state.modelsCache.length) {
    try {
      const data = await api('/api/models');
      state.modelsCache = data.models || [];
    } catch (e) { /* non-fatal */ }
  }
  const models = getTTSModels();
  const providerRank = { kokoro: 0, piper: 1 };
  const providers = [...new Set(models.map((m) => m.provider || providerFromModel(m.id)).filter(Boolean))]
    .sort((a, b) => (providerRank[a] ?? 99) - (providerRank[b] ?? 99) || a.localeCompare(b));
  const providerSel = byId('tts-provider');
  providerSel.innerHTML = providers.map((provider) => `<option value="${esc(provider)}">${esc(PROVIDER_DISPLAY[provider] || provider)}</option>`).join('');

  const loaded = models.find((m) => m.state === 'loaded');
  const downloaded = models.find((m) => m.state === 'downloaded' || m.state === 'ready');
  const kokoro = models.find((m) => m.provider === 'kokoro');
  const piper = models.find((m) => m.provider === 'piper');
  const preferredProvider = state.ttsPreferredProvider && providers.includes(state.ttsPreferredProvider)
    ? state.ttsPreferredProvider
    : (loaded?.provider || downloaded?.provider || kokoro?.provider || piper?.provider || providers[0] || '');
  providerSel.value = preferredProvider;
  state.ttsPreferredProvider = providerSel.value;
  state.ttsPreferredModel = state.ttsPreferredModel || loaded?.id || downloaded?.id || '';
  await loadTTSModels();
}
async function loadTTSModels() {
  const providerSel = byId('tts-provider');
  const modelSel = byId('tts-model');
  const provider = providerSel.value;
  const models = getTTSModels().filter((m) => (m.provider || providerFromModel(m.id)) === provider);
  modelSel.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(formatModelName(m))}</option>`).join('');
  const loaded = models.find((m) => m.state === 'loaded');
  const downloaded = models.find((m) => m.state === 'downloaded' || m.state === 'ready');
  const preferredModel = models.find((m) => m.id === state.ttsPreferredModel)?.id
    || loaded?.id
    || downloaded?.id
    || models[0]?.id
    || '';
  modelSel.value = preferredModel;
  state.ttsPreferredProvider = provider;
  state.ttsPreferredModel = modelSel.value;
  await loadTTSVoices();
}
async function loadSTTModels() {
  const data = { models: state.modelsCache };
  const rank = { loaded: 0, downloaded: 1, ready: 1, provider_installed: 2, available: 2, provider_missing: 3 };
  const models = (data.models || [])
    .filter((m) => classifyKind(m) === 'stt' && (m.provider === 'faster-whisper' || (m.id || '').includes('faster-whisper')))
    .sort((a, b) => {
      const sr = (rank[a.state] ?? 9) - (rank[b.state] ?? 9);
      return sr || (a.id || '').localeCompare(b.id || '');
    });
  const sel = byId('stt-model');
  sel.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)} ${statusSuffix(m.state)}</option>`).join('');
  if (models[0]?.id) sel.value = models[0].id;
}
async function fetchTTSCapabilities(model) {
  const url = model ? `/api/tts/capabilities?model=${encodeURIComponent(model)}` : '/api/tts/capabilities';
  const data = await api(url);
  const caps = data.capabilities || {};
  if (Array.isArray(caps)) {
    return Object.fromEntries(caps.map((k) => [k, true]));
  }
  return caps;
}
async function fetchVoices(model) {
  try {
    const data = await api(`/v1/audio/voices?model=${encodeURIComponent(model)}`);
    if (Array.isArray(data)) return data;
    if (Array.isArray(data.voices)) return data.voices;
  } catch {}
  return [];
}
function renderBlendUI(voices) {
  const chips = blendVoices.map((b, i) =>
    `<span class="blend-chip">${esc(b.voice)} <span class="blend-weight">${esc(b.weight)}</span> <button type="button" onclick="removeBlendVoice(${i})">✕</button></span>`
  ).join('');
  const opts = (voices || []).map((v) => {
    const id = v.voice_id || v.id || v.name || v;
    const label = v.name || id;
    return `<option value="${esc(id)}">${esc(label)}</option>`;
  }).join('');
  return `<div class="blend-chips">${chips}</div>
    <div class="blend-add-row">
      <select id="blend-voice-picker">${opts}</select>
      <input id="blend-weight" type="number" value="1.0" min="0.1" max="2.0" step="0.1" style="width:60px">
      <button type="button" onclick="addBlendVoice()">+ Add</button>
    </div>`;
}

function rerenderBlendSection() {
  const holder = byId('tts-blend-ui');
  if (!holder) return;
  holder.innerHTML = renderBlendUI(state.ttsVoices || []);
}

function addBlendVoice() {
  const picker = document.getElementById('blend-voice-picker');
  const weight = document.getElementById('blend-weight');
  const v = picker?.value;
  const w = parseFloat(weight?.value) || 1.0;
  if (v && !blendVoices.find((b) => b.voice === v)) {
    blendVoices.push({ voice: v, weight: w });
    rerenderBlendSection();
  }
}

function removeBlendVoice(i) {
  blendVoices.splice(i, 1);
  rerenderBlendSection();
}

window.addBlendVoice = addBlendVoice;
window.removeBlendVoice = removeBlendVoice;

function renderAdvancedControls(caps) {
  const details = byId('tts-advanced');
  const wrap = byId('tts-advanced-content');
  wrap.innerHTML = '';
  const rows = [];
  if (caps.voice_clone) {
    rows.push('<div class="field"><label for="tts-clone-file">Voice Clone</label><input id="tts-clone-file" type="file" accept="audio/*"></div>');
  }
  if (caps.voice_blend === true) {
    rows.push('<div class="field"><label>Voice Blend</label><div id="tts-blend-ui"></div></div>');
  }
  if (caps.instructions) {
    rows.push('<div class="field"><label for="tts-instructions">Instructions</label><input id="tts-instructions" type="text" placeholder="Style / direction"></div>');
  }
  details.hidden = rows.length === 0;
  if (rows.length > 0) {
    wrap.innerHTML = rows.join('');
    details.open = false;
  }
  if (caps.voice_blend === true) rerenderBlendSection();
  byId('tts-stream-group').hidden = !caps.streaming;
}
async function loadTTSVoices(preferredVoice = '') {
  const model = byId('tts-model').value;
  state.ttsPreferredModel = model;
  state.ttsCaps = await fetchTTSCapabilities(model);
  state.ttsVoices = await fetchVoices(model);
  renderAdvancedControls(state.ttsCaps);
  if (state.ttsCaps.voice_blend !== true) blendVoices = [];
  const voiceSel = byId('tts-voice');
  const options = (state.ttsVoices || []).map((v) => {
    const id = v.id || v.name || v;
    return `<option value="${esc(id)}">${esc(id)}</option>`;
  }).join('');
  voiceSel.innerHTML = options;
  const nextVoice = preferredVoice || voiceSel.value;
  if (nextVoice && [...voiceSel.options].some((o) => o.value === nextVoice)) {
    voiceSel.value = nextVoice;
  }
  if (!voiceSel.value && voiceSel.options.length) voiceSel.selectedIndex = 0;
  updateTTSModelStatus(model);
}
async function installProvider(modelOrProvider, onCancel) {
  const provider = (modelOrProvider || '').includes('/') ? providerFromModel(modelOrProvider) : modelOrProvider;
  const result = await api('/api/providers/install', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });
  const note = showToast(`Installing provider ${provider}...`, '');
  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn btn-ghost btn-sm';
  cancelBtn.textContent = 'Cancel';
  note.appendChild(document.createElement('br'));
  note.appendChild(cancelBtn);
  let canceled = false;
  cancelBtn.onclick = () => {
    canceled = true;
    note.remove();
    if (onCancel) onCancel();
  };
  while (!canceled) {
    const status = await api(`/api/providers/install/${encodeURIComponent(result.job_id)}`);
    if (status.status === 'done') {
      showToast(`Provider ${provider} installed`, 'success');
      return true;
    }
    if (status.status === 'failed') {
      throw new Error(status.error || `Failed to install provider ${provider}`);
    }
    await new Promise((r) => setTimeout(r, 1500));
  }
  return false;
}
async function downloadModel(modelId) {
  await api(`/api/models/${encodeURIComponent(modelId)}/download`, { method: 'POST' });
}
async function prefetchModel(modelId) {
  await api(`/api/models/${encodeURIComponent(modelId)}/prefetch`, { method: 'POST' });
}
async function loadModel(modelId) {
  try {
    await api(`/api/models/${encodeURIComponent(modelId)}/load`, { method: 'POST' });
    return;
  } catch {}
  await api('/v1/audio/models/load', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelId }),
  });
}
async function unloadModel(modelId) {
  try {
    await api(`/v1/audio/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
    return;
  } catch {}
  try {
    await api(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
    return;
  } catch {}
  await api('/v1/audio/models/unload', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelId }),
  });
}
async function ensureModelReady(modelId, kind = 'tts') {
  setButtonState('tts-generate', 'checking');
  const status = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (status.state === 'loaded') return true;

  if (status.state === 'provider_missing') {
    setButtonState('tts-generate', 'installing');
    const ok = await installProvider(modelId);
    if (!ok) return false;
  }

  const status2 = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (status2.state === 'provider_installed' || status2.state === 'available') {
    if (kind === 'tts') {
      setButtonState('tts-generate', 'loading');
      await loadModel(modelId);
    } else {
      setButtonState('tts-generate', 'downloading');
      await downloadModel(modelId);
    }
  }

  const status3 = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (status3.state === 'downloaded' || status3.state === 'ready') {
    setButtonState('tts-generate', 'loading');
    await loadModel(modelId);
  }

  const final = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (final.state !== 'loaded') throw new Error(`${kind.toUpperCase()} model not ready: state=${final.state}`);
  return true;
}
function pushHistory(key, item) {
  const curr = JSON.parse(localStorage.getItem(key) || '[]');
  curr.unshift({ ...item, ts: Date.now() });
  localStorage.setItem(key, JSON.stringify(curr.slice(0, 5)));
}
function renderHistory(key, elId, mapFn) {
  const arr = JSON.parse(localStorage.getItem(key) || '[]');
  byId(elId).innerHTML = arr.map(mapFn).join('') || '<p class="history-item">No recent items</p>';
}
function refreshHistory() {
  renderHistory(HISTORY_KEYS.tts, 'tts-history', (h) => `<div class="history-item">${esc(h.text)}<small>${new Date(h.ts).toLocaleString()}</small></div>`);
  renderHistory(HISTORY_KEYS.stt, 'stt-history', (h) => `<div class="history-item">${esc(h.text)}<small>${new Date(h.ts).toLocaleString()}</small></div>`);
}
function buildEffectsPayload() {
  const effects = [];
  document.querySelectorAll('#effects-panel [data-effect]').forEach((cb) => {
    if (cb.checked) {
      const type = cb.dataset.effect;
      const fx = { type };
      if (type === 'reverb') fx.room = document.querySelector('[data-effect-param="reverb-room"]').value;
      if (type === 'pitch') fx.semitones = parseInt(document.querySelector('[data-effect-param="pitch-semitones"]').value, 10) || 0;
      effects.push(fx);
    }
  });
  return effects.length ? effects : null;
}
async function doSpeak() {
  const provider = byId('tts-provider')?.value;
  const model = byId('tts-model').value;
  const voice = byId('tts-voice').value;
  const input = byId('tts-input').value.trim();
  if (!input) return showToast('Enter text first', 'error');
  try {
    await ensureModelReady(model, 'tts');
    updateTTSModelStatus(model);
    setButtonState('tts-generate', 'generating');
    const payload = {
      model, voice, input,
      speed: Number(byId('tts-speed').value),
      response_format: byId('tts-format').value,
      effects: buildEffectsPayload(),
    };
    const instructions = byId('tts-instructions');
    if (instructions) payload.instructions = instructions.value;
    if (blendVoices.length > 0) {
      payload.voice_blend = blendVoices.map((b) => `${b.voice}(${b.weight})`).join('+');
    }
    const doStream = !byId('tts-stream-group').hidden && byId('tts-stream').checked;
    const endpoint = '/v1/audio/speech' + (doStream ? '?stream=true' : '');
    const res = await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    state.ttsAudioBlob = await res.blob();
    if (state.ttsAudioUrl) URL.revokeObjectURL(state.ttsAudioUrl);
    state.ttsAudioUrl = URL.createObjectURL(state.ttsAudioBlob);
    byId('audio-player').hidden = false;
    byId('tts-audio').src = state.ttsAudioUrl;
    byId('tts-audio').play().catch(() => {});
    byId('tts-download').disabled = false;
    pushHistory(HISTORY_KEYS.tts, { text: input.slice(0, 120) });
    refreshHistory();
    showToast('Speech generated', 'success');
  } catch (e) {
    showToast(`TTS failed: ${e.message}`, 'error');
  } finally {
    setButtonState('tts-generate', 'idle');
  }
}
async function transcribeFile(file) {
  const model = byId('stt-model').value;
  const format = byId('stt-format').value;
  const t0 = performance.now();
  try {
    await ensureModelReady(model, 'stt');
    const fd = new FormData();
    fd.append('file', file);
    fd.append('model', model);
    fd.append('response_format', format);
    const res = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    const text = data.text || '';
    byId('stt-final').textContent = text || '—';
    byId('stt-partial').textContent = '—';
    byId('stt-duration').textContent = data.duration ? `${data.duration}s` : '—';
    byId('stt-processing').textContent = `${((performance.now() - t0) / 1000).toFixed(2)}s`;
    pushHistory(HISTORY_KEYS.stt, { text: text.slice(0, 200) || file.name });
    refreshHistory();
  } catch (e) {
    showToast(`Transcription failed: ${e.message}`, 'error');
  }
}
async function toggleMic() {
  const btn = byId('mic-btn');
  if (state.sttRecording) {
    state.mediaRecorder?.stop();
    state.mediaStream?.getTracks().forEach((t) => t.stop());
    state.ws?.close();
    state.sttRecording = false;
    btn.textContent = '🎙 Start Mic';
    btn.classList.remove('btn-danger', 'mic-recording');
    byId('vad-dot').className = 'status-dot';
    byId('vad-text').textContent = 'Silence';
    return;
  }
  const model = byId('stt-model').value;
  try {
    await ensureModelReady(model, 'stt');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaStream = stream;
    state.mediaRecorder = new MediaRecorder(stream);
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/v1/audio/stream`;
    state.ws = new WebSocket(wsUrl);
    state.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'partial') byId('stt-partial').textContent = msg.text || '…';
        if (msg.type === 'final') {
          byId('stt-final').textContent = msg.text || '—';
          pushHistory(HISTORY_KEYS.stt, { text: msg.text || '' });
          refreshHistory();
        }
        if (msg.type === 'vad') {
          const speaking = !!msg.speaking;
          byId('vad-dot').className = `status-dot ${speaking ? 'loaded' : ''}`;
          byId('vad-text').textContent = speaking ? 'Speech' : 'Silence';
        }
      } catch {}
    };
    state.mediaRecorder.ondataavailable = (e) => {
      if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(e.data);
    };
    state.mediaRecorder.start(250);
    state.sttRecording = true;
    btn.textContent = '⏹ Stop';
    btn.classList.add('btn-danger', 'mic-recording');
    showToast('Mic recording started');
  } catch (e) {
    showToast(`Mic failed: ${e.message}`, 'error');
  }
}
function getStateBadge(model) {
  if (model.state === 'loaded') return { text: '● Loaded', cls: 'loaded' };
  if (model.state === 'downloaded') return { text: '● Cached', cls: 'downloaded' };
  if (model.state === 'provider_installed' || model.state === 'available') return { text: '○ Installed (weights on demand)', cls: 'available' };
  return { text: '○ Available', cls: 'available' };
}
function getModelHint(model) {
  if (model.state === 'provider_installed' || model.state === 'available') {
    const size = formatSize(model.size_mb);
    return size
      ? `Installed — Load to cache weights (${size}) or use Cache`
      : 'Installed — Load to cache weights or use Cache';
  }
  if (model.state === 'downloaded') return 'Cached on disk, ready to load into memory';
  if (model.id === 'kokoro') return 'Kokoro is pre-baked in the default GPU image';
  return '';
}
function actionSortRank(m) {
  if (m.state === 'downloaded') return 0;
  return 1;
}
function renderModelRow(m) {
  const op = state.modelOps[m.id];
  const badge = getStateBadge(m);
  const busy = op && (op.kind === 'downloading' || op.kind === 'loading' || op.kind === 'prefetch');
  let actions = '';
  if (busy) {
    const label = op.kind === 'loading' ? 'Loading…' : 'Caching…';
    actions = `<span class="row-status"><span class="spin-dot"></span>${esc(op.text || label)}</span>`;
  } else if (op?.error) {
    actions = `<span class="row-status error">${esc(op.error)}</span> <button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Cache</button>`;
  } else if (m.state === 'available') {
    actions = `<button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Cache</button>`;
  } else if (m.state === 'provider_installed') {
    actions = `<button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Cache</button> <button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load</button>`;
  } else if (m.state === 'downloaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  } else if (m.state === 'loaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-unload="${esc(m.id)}">Unload</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  }
  const sizeTxt = m.size_mb ? `<span class="model-size">${esc(formatSize(m.size_mb))}</span>` : '';
  const descTxt = m.description ? `<span class="model-desc">${esc(m.description)}</span>` : '';
  const hint = getModelHint(m);
  const hintTxt = hint ? `<span class="model-desc">${esc(hint)}</span>` : '';
  return `<div class="model-row" data-model-row="${esc(m.id)}">
    <div class="model-main">
      <span class="model-id">${esc(m.id)}</span>${descTxt}${hintTxt}
      <span class="state-badge ${badge.cls}">${badge.text}</span>${sizeTxt}
    </div>
    <div>${actions}</div>
  </div>`;
}
function renderProviderRow(provider, installed, kind) {
  const action = installed
    ? '<button class="btn btn-ghost btn-sm" disabled title="TODO">Uninstall</button>'
    : `<button class="btn btn-ghost btn-sm" data-install-provider="${esc(provider)}" data-provider-kind="${esc(kind)}">Install</button>`;
  return `<div class="provider-row"><span>${esc(provider)} <span class="state-badge ${installed ? 'loaded' : 'available'}">${installed ? '● installed' : '○ not installed'}</span></span><span>${action}</span></div>`;
}
function renderModelsView() {
  const models = state.modelsCache || [];
  const loaded = models.filter((m) => m.state === 'loaded');
  byId('loaded-count').textContent = `${loaded.length} models loaded`;
  byId('loaded-models-body').innerHTML = loaded.map((m) => `
    <tr>
      <td>${esc(m.id)}</td>
      <td>${esc((m.type || '').toUpperCase())}</td>
      <td>${esc(m.device || '—')}</td>
      <td><button class="btn btn-ghost btn-sm" data-unload="${esc(m.id)}">Unload</button></td>
    </tr>
  `).join('') || '<tr><td colspan="4">No loaded models</td></tr>';

  const installed = models.filter((m) => m.state !== 'provider_missing' && m.state !== 'loaded');
  const sttModels = installed.filter((m) => m.type === 'stt').sort((a, b) => actionSortRank(a) - actionSortRank(b) || (a.id || '').localeCompare(b.id || ''));
  const ttsModels = installed.filter((m) => m.type === 'tts').sort((a, b) => actionSortRank(a) - actionSortRank(b) || (a.id || '').localeCompare(b.id || ''));
  if (!models.length) {
    byId('stt-models-list').innerHTML = '<p class="legend">Loading…</p>';
    byId('tts-models-list').innerHTML = '<p class="legend">Loading…</p>';
  } else {
    byId('stt-models-list').innerHTML = sttModels.map(renderModelRow).join('') || '<p class="legend">No STT models for installed providers.</p>';
    byId('tts-models-list').innerHTML = ttsModels.map(renderModelRow).join('') || '<p class="legend">No TTS models for installed providers.</p>';
  }

  const sttProvidersKnown = ['faster-whisper', 'moonshine', 'vosk'];
  const ttsProvidersKnown = ['kokoro', 'piper', 'pocket-tts', 'qwen3-tts', 'fish-speech', 'f5-tts', 'xtts'];
  const knownProviders = [...new Set([...sttProvidersKnown, ...ttsProvidersKnown, ...models.map((m) => m.provider).filter(Boolean)])];
  const installedProviders = new Set(models.filter((m) => m.state !== 'provider_missing').map((m) => m.provider).filter(Boolean));
  const sttProviders = knownProviders.filter((p) => sttProvidersKnown.includes(p));
  const ttsProviders = knownProviders.filter((p) => ttsProvidersKnown.includes(p));
  byId('stt-providers-list').innerHTML = sttProviders.map((p) => renderProviderRow(p, installedProviders.has(p), 'stt')).join('');
  byId('tts-providers-list').innerHTML = ttsProviders.map((p) => renderProviderRow(p, installedProviders.has(p), 'tts')).join('');
}
async function refreshModels({ silent = false } = {}) {
  if (state.modelsBusy) return;
  state.modelsBusy = true;
  try {
    const data = await api('/api/models');
    state.modelsCache = data.models || [];
    renderModelsView();
  } catch (e) {
    if (!silent) throw e;
  } finally {
    state.modelsBusy = false;
  }
}
async function deleteModel(modelId) {
  await api(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
}
async function runModelOp(modelId, kind) {
  const opLabel = kind === 'loading' ? 'Loading…' : 'Caching…';
  state.modelOps[modelId] = { kind, text: opLabel };
  renderModelsView();
  try {
    if (kind === 'downloading') {
      await downloadModel(modelId);
    } else if (kind === 'prefetch') {
      await prefetchModel(modelId);
    } else {
      await loadModel(modelId);
    }
    while (true) {
      await new Promise((r) => setTimeout(r, 2000));
      const status = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
      const nextText = status.progress ? `${opLabel} ${status.progress}` : opLabel;
      state.modelOps[modelId] = { kind, text: nextText };
      const idx = state.modelsCache.findIndex((m) => m.id === modelId);
      if (idx >= 0) state.modelsCache[idx] = { ...state.modelsCache[idx], ...status };
      renderModelsView();
      if (kind === 'loading' && status.state === 'loaded') break;
      if ((kind === 'downloading' || kind === 'prefetch') && (status.state === 'downloaded' || status.state === 'loaded')) break;
      if (status.state === 'provider_missing') throw new Error('Provider missing');
    }
    delete state.modelOps[modelId];
    await refreshModels({ silent: true });
  } catch (e) {
    state.modelOps[modelId] = { kind: 'error', error: e.message || 'Action failed' };
    renderModelsView();
  }
}
function initTheme() {
  const key = 'open-speech-theme';
  const saved = localStorage.getItem(key) || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  byId('theme-toggle').textContent = saved === 'dark' ? '☀️' : '🌙';
  byId('theme-toggle').onclick = () => {
    const now = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', now);
    localStorage.setItem(key, now);
    byId('theme-toggle').textContent = now === 'dark' ? '☀️' : '🌙';
  };
}
function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const name = tab.dataset.tab;
      document.querySelectorAll('.tab').forEach((t) => {
        const active = t.dataset.tab === name;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      document.querySelectorAll('.panel').forEach((p) => {
        const active = p.id === `panel-${name}`;
        p.classList.toggle('active', active);
        p.hidden = !active;
      });
      if (name === 'history') loadHistory(byId('history-type')?.value || '', state.history.limit, state.history.offset).catch((e) => showToast(e.message, 'error'));
      if (name === 'studio') {
        loadConversations().catch((e) => showToast(e.message, 'error'));
        loadComposerHistory().catch((e) => showToast(e.message, 'error'));
      }
      if (name === 'settings') loadProfiles().catch((e) => showToast(e.message, 'error'));
    });
  });
}
function bindEvents() {
  byId('tts-input').addEventListener('input', (e) => {
    byId('tts-counter').textContent = `${e.target.value.length} / 5,000`;
  });
  byId('tts-speed').addEventListener('input', (e) => {
    byId('tts-speed-value').textContent = `${Number(e.target.value).toFixed(1)}x`;
  });
  byId('tts-provider')?.addEventListener('change', () => {
    state.ttsPreferredProvider = byId('tts-provider').value;
    state.ttsPreferredModel = '';
    loadTTSModels().catch((err) => showToast(err.message, 'error'));
  });
  byId('tts-model').addEventListener('change', () => loadTTSVoices().catch((err) => showToast(err.message, 'error')));
  byId('tts-voice')?.addEventListener('change', () => {
    const presetSel = byId('tts-preset');
    if (presetSel) presetSel.value = '';
  });
  byId('tts-preset')?.addEventListener('change', (e) => applyProfile(e.target.value).catch((err) => showToast(err.message, 'error')));
  byId('tts-save-profile')?.addEventListener('click', () => saveAsProfile().catch((err) => showToast(err.message, 'error')));
  byId('history-type')?.addEventListener('change', (e) => loadHistory(e.target.value, state.history.limit, 0));
  byId('history-search')?.addEventListener('input', () => loadHistory(state.history.type, state.history.limit, state.history.offset));
  byId('history-prev')?.addEventListener('click', () => loadHistory(state.history.type, state.history.limit, Math.max(0, state.history.offset - state.history.limit)));
  byId('history-next')?.addEventListener('click', () => loadHistory(state.history.type, state.history.limit, state.history.offset + state.history.limit));
  byId('history-clear')?.addEventListener('click', () => clearHistory().catch((err) => showToast(err.message, 'error')));
  byId('settings-clear-history')?.addEventListener('click', () => clearHistory().catch((err) => showToast(err.message, 'error')));
  byId('tts-generate').addEventListener('click', doSpeak);
  byId('tts-download').addEventListener('click', () => {
    if (!state.ttsAudioUrl) return;
    const a = document.createElement('a');
    a.href = state.ttsAudioUrl;
    a.download = `open-speech-${Date.now()}.${byId('tts-format').value}`;
    a.click();
  });
  byId('tts-upload').addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    byId('tts-input').value = await file.text();
    byId('tts-counter').textContent = `${byId('tts-input').value.length} / 5,000`;
  });
  const dz = byId('stt-dropzone');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('dragover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('dragover');
    const file = e.dataTransfer.files?.[0];
    if (file) transcribeFile(file);
  });
  byId('stt-file').addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) transcribeFile(file);
  });
  byId('mic-btn').addEventListener('click', toggleMic);
  byId('copy-transcript').addEventListener('click', async () => {
    const text = byId('stt-final').textContent || '';
    if (text && text !== '—') await navigator.clipboard.writeText(text);
  });
  byId('save-transcript').addEventListener('click', () => {
    const text = byId('stt-final').textContent || '';
    const blob = new Blob([text], { type: 'text/plain' });
    const u = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = u; a.download = `transcript-${Date.now()}.txt`; a.click();
    setTimeout(() => URL.revokeObjectURL(u), 500);
  });
  byId('models-refresh').addEventListener('click', () => refreshModels().catch((e) => showToast(e.message, 'error')));
  byId('studio-new-conversation')?.addEventListener('click', () => createConversation().catch((e) => showToast(e.message, 'error')));
  byId('studio-add-turn')?.addEventListener('click', () => addTurn().catch((e) => showToast(e.message, 'error')));
  byId('studio-render')?.addEventListener('click', () => renderConversation('wav').catch((e) => { byId('studio-status').textContent = `Render status: ${e.message}`; showToast(e.message, 'error'); }));
  byId('studio-download-wav')?.addEventListener('click', () => downloadConversationAudio('wav'));
  byId('studio-download-mp3')?.addEventListener('click', async () => { await renderConversation('mp3'); downloadConversationAudio('mp3'); });
  byId('studio-load')?.addEventListener('click', async () => {
    const id = byId('studio-past').value;
    if (!id) return;
    state.currentConversationId = id;
    state.currentConversation = await api(`/api/conversations/${encodeURIComponent(id)}`);
    byId('studio-name').value = state.currentConversation.name || '';
    renderStudioTurns();
  });
  byId('studio-delete')?.addEventListener('click', async () => {
    const id = byId('studio-past').value || state.currentConversationId;
    if (!id) return;
    await api(`/api/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' });
    if (state.currentConversationId === id) {
      state.currentConversationId = null;
      state.currentConversation = null;
      renderStudioTurns();
    }
    await loadConversations();
  });
  byId('composer-add-track')?.addEventListener('click', () => addComposerTrack());
  byId('composer-render-btn')?.addEventListener('click', () => renderComposerMix().catch((e) => showToast(e.message, 'error')));
  byId('composer-download')?.addEventListener('click', () => {
    const src = byId('composer-audio')?.src;
    if (!src) return;
    const a = document.createElement('a');
    a.href = src;
    a.download = `composer-mix-${Date.now()}.${byId('composer-format')?.value || 'wav'}`;
    a.click();
  });
  byId('composer-tracks')?.addEventListener('input', (e) => {
    const idx = Number(e.target?.dataset?.idx);
    const field = e.target?.dataset?.field;
    if (Number.isNaN(idx) || !field || !composerTracks[idx]) return;
    if (field === 'offset_s' || field === 'volume') {
      composerTracks[idx][field] = Number(e.target.value || 0);
    } else {
      composerTracks[idx][field] = e.target.value;
    }
  });
  byId('composer-tracks')?.addEventListener('change', (e) => {
    const idx = Number(e.target?.dataset?.idx);
    const field = e.target?.dataset?.field;
    if (Number.isNaN(idx) || !field || !composerTracks[idx]) return;
    if (field === 'muted' || field === 'solo') composerTracks[idx][field] = !!e.target.checked;
  });

  document.addEventListener('click', async (e) => {
    const unload = e.target.closest('[data-unload]');
    const download = e.target.closest('[data-download]');
    const prefetch = e.target.closest('[data-prefetch]');
    const load = e.target.closest('[data-load]');
    const deleteBtn = e.target.closest('[data-delete-model]');
    const installProviderBtn = e.target.closest('[data-install-provider]');
    try {
      if (unload) {
        await unloadModel(unload.dataset.unload);
        await refreshModels();
      }
      if (download) {
        await runModelOp(download.dataset.download, 'downloading');
      }
      if (prefetch) {
        await runModelOp(prefetch.dataset.prefetch, 'prefetch');
      }
      if (load) {
        await runModelOp(load.dataset.load, 'loading');
      }
      if (deleteBtn) {
        await deleteModel(deleteBtn.dataset.deleteModel);
        await refreshModels();
      }
      if (installProviderBtn) {
        const provider = installProviderBtn.dataset.installProvider;
        installProviderBtn.disabled = true;
        installProviderBtn.classList.add('loading');
        installProviderBtn.textContent = 'Installing…';
        await installProvider(provider);
        await refreshModels();
        await loadTTSProviders();
      }
      const profileDelete = e.target.closest('[data-profile-delete]');
      const profileDefault = e.target.closest('[data-profile-default]');
      const histDelete = e.target.closest('[data-history-delete]');
      const histRegen = e.target.closest('[data-history-regen]');
      const turnDelete = e.target.closest('[data-turn-delete]');
      const composerDelete = e.target.closest('[data-composer-delete]');
      if (profileDelete) await deleteProfile(profileDelete.dataset.profileDelete);
      if (profileDefault) await setDefaultProfile(profileDefault.dataset.profileDefault);
      if (histDelete) await deleteHistoryEntry(histDelete.dataset.historyDelete);
      if (histRegen) await reGenerateTTS(JSON.parse(histRegen.dataset.historyRegen));
      if (turnDelete) await deleteTurn(turnDelete.dataset.turnDelete);
      if (composerDelete) removeComposerTrack(Number(composerDelete.dataset.composerDelete));
    } catch (err) {
      showToast(err.message, 'error');
    }
  });
}

async function loadProfiles() {
  const data = await api('/api/profiles');
  state.profiles = data.profiles || [];
  state.defaultProfileId = data.default_profile_id || null;

  const presetSel = byId('tts-preset');
  if (presetSel) {
    presetSel.innerHTML = '<option value="">— Custom —</option>' + state.profiles.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
    if (state.defaultProfileId) presetSel.value = state.defaultProfileId;
  }

  const studioProfile = byId('studio-profile');
  if (studioProfile) {
    studioProfile.innerHTML = '<option value="">— None —</option>' + state.profiles.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  }

  const body = byId('profiles-body');
  if (body) {
    body.innerHTML = state.profiles.map((p) => `
      <tr>
        <td>${esc(p.name)}</td><td>${esc(p.backend)}</td><td>${esc(p.model || '—')}</td><td>${esc(p.voice)}</td><td>${esc(p.speed)}</td>
        <td>${p.id === state.defaultProfileId ? '✓' : ''}</td>
        <td>
          <button class="btn btn-ghost btn-sm" data-profile-default="${esc(p.id)}">Default</button>
          <button class="btn btn-ghost btn-sm" data-profile-delete="${esc(p.id)}">Delete</button>
        </td>
      </tr>
    `).join('') || '<tr><td colspan="7">No profiles</td></tr>';
  }
}

async function applyProfile(profileId) {
  if (!profileId) return;
  const profile = await api(`/api/profiles/${encodeURIComponent(profileId)}`);
  const providerSel = byId('tts-provider');
  const modelSel = byId('tts-model');
  const provider = profile.provider || providerFromModel(profile.model);

  if (provider && [...providerSel.options].some((o) => o.value === provider)) {
    providerSel.value = provider;
    state.ttsPreferredProvider = provider;
  }
  state.ttsPreferredModel = profile.model || '';
  await loadTTSModels();
  if (profile.model && [...modelSel.options].some((o) => o.value === profile.model)) {
    modelSel.value = profile.model;
  }
  await loadTTSVoices(profile.voice || '');

  byId('tts-speed').value = Number(profile.speed || 1.0);
  byId('tts-speed-value').textContent = `${Number(byId('tts-speed').value).toFixed(1)}x`;
  byId('tts-format').value = profile.format || byId('tts-format').value;
  blendVoices = [];
  const blend = profile.blend || '';
  if (blend) {
    blend.split('+').forEach((part) => {
      const m = part.match(/^(.+)\(([^)]+)\)$/);
      if (m) blendVoices.push({ voice: m[1], weight: parseFloat(m[2]) || 1.0 });
    });
  }
  rerenderBlendSection();
}

async function saveAsProfile() {
  const modelId = byId('tts-model').value;
  const providerId = byId('tts-provider')?.value || providerFromModel(modelId);
  const payload = {
    name: window.prompt('Profile name?'),
    backend: providerId,
    provider: providerId,
    model: modelId,
    voice: byId('tts-voice').value,
    speed: Number(byId('tts-speed').value),
    format: byId('tts-format').value,
    blend: blendVoices.length ? blendVoices.map((b) => `${b.voice}(${b.weight})`).join('+') : null,
    reference_audio_id: null,
    effects: [],
  };
  if (!payload.name) return;
  await api('/api/profiles', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  await loadProfiles();
}

async function deleteProfile(id) {
  await api(`/api/profiles/${encodeURIComponent(id)}`, { method: 'DELETE' });
  await loadProfiles();
}

async function setDefaultProfile(id) {
  await api(`/api/profiles/${encodeURIComponent(id)}/default`, { method: 'POST' });
  await loadProfiles();
}

async function loadHistory(type = '', limit = 50, offset = 0) {
  const qs = new URLSearchParams();
  if (type) qs.set('type', type);
  qs.set('limit', String(limit));
  qs.set('offset', String(offset));
  const data = await api(`/api/history?${qs.toString()}`);
  state.history = { ...data, type };
  const search = (byId('history-search')?.value || '').toLowerCase();
  const items = (data.items || []).filter((i) => !search || `${i.text_preview || ''} ${i.input_filename || ''}`.toLowerCase().includes(search));
  byId('history-body').innerHTML = items.map((h) => `
    <tr>
      <td>${esc((h.type || '').toUpperCase())}</td>
      <td>${esc(new Date(h.created_at).toLocaleString())}</td>
      <td>${esc(h.text_preview || h.input_filename || '—')}</td>
      <td>${esc(h.model || '—')}</td>
      <td>${esc(h.voice || '—')}</td>
      <td>
        <button class="btn btn-ghost btn-sm" data-history-regen='${esc(JSON.stringify(h))}'>Re-generate</button>
        <button class="btn btn-ghost btn-sm" data-history-delete="${esc(h.id)}">Delete</button>
      </td>
    </tr>
  `).join('') || '<tr><td colspan="6">No history</td></tr>';
  byId('history-count').textContent = `${Math.min(offset + limit, data.total || 0)} / ${data.total || 0}`;
}

async function deleteHistoryEntry(id) {
  await api(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
  await loadHistory(state.history.type, state.history.limit, state.history.offset);
}

async function clearHistory() {
  if (!window.confirm('Clear all history?')) return;
  await api('/api/history', { method: 'DELETE' });
  await loadHistory(state.history.type, state.history.limit, 0);
}

async function loadConversations() {
  const data = await api('/api/conversations?limit=100&offset=0');
  const sel = byId('studio-past');
  if (!sel) return;
  const items = data.items || [];
  sel.innerHTML = items.map((c) => `<option value="${esc(c.id)}">${esc(c.name || c.id)}</option>`).join('');
}

function renderStudioTurns() {
  const wrap = byId('studio-turns');
  if (!wrap) return;
  const turns = state.currentConversation?.turns || [];
  wrap.innerHTML = turns.map((t, idx) => `<div class="history-item">Turn ${idx + 1}: ${esc(t.speaker)} — "${esc(t.text)}" <button class="btn btn-ghost btn-sm" data-turn-delete="${esc(t.id)}">Delete</button></div>`).join('') || '<p class="legend">No turns yet.</p>';
}

async function createConversation() {
  const name = byId('studio-name').value.trim() || `Conversation ${new Date().toLocaleString()}`;
  const created = await api('/api/conversations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, turns: [] }) });
  state.currentConversationId = created.id;
  state.currentConversation = created;
  renderStudioTurns();
  await loadConversations();
}

async function addTurn() {
  if (!state.currentConversationId) await createConversation();
  const speaker = byId('studio-speaker').value.trim() || 'Speaker';
  const text = byId('studio-text').value.trim();
  if (!text) return showToast('Enter turn text', 'error');
  const profile_id = byId('studio-profile').value || null;
  await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}/turns`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speaker, text, profile_id, effects: null }),
  });
  state.currentConversation = await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}`);
  byId('studio-text').value = '';
  renderStudioTurns();
  await loadConversations();
}

async function deleteTurn(turnId) {
  if (!state.currentConversationId) return;
  await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}/turns/${encodeURIComponent(turnId)}`, { method: 'DELETE' });
  state.currentConversation = await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}`);
  renderStudioTurns();
}

async function renderConversation(format = 'wav') {
  if (!state.currentConversationId) return showToast('Create or load a conversation first', 'error');
  byId('studio-status').textContent = 'Render status: Rendering...';
  const data = await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}/render`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format, sample_rate: 24000, save_turn_audio: true }),
  });
  state.currentConversation = await api(`/api/conversations/${encodeURIComponent(state.currentConversationId)}`);
  byId('studio-status').textContent = 'Render status: Done ✓';
  byId('studio-download-wav').disabled = false;
  byId('studio-download-mp3').disabled = false;
  return data;
}

function downloadConversationAudio(format = 'wav') {
  if (!state.currentConversationId) return;
  const a = document.createElement('a');
  a.href = `/api/conversations/${encodeURIComponent(state.currentConversationId)}/audio`;
  a.download = `conversation-${state.currentConversationId}.${format}`;
  a.click();
}

function addComposerTrack() {
  composerTracks.push({ source_path: '', offset_s: 0.0, volume: 1.0, muted: false, solo: false, effects: [] });
  renderComposerTrackList();
}

function removeComposerTrack(idx) {
  composerTracks = composerTracks.filter((_, i) => i !== idx);
  renderComposerTrackList();
}

function renderComposerTrackList() {
  const wrap = byId('composer-tracks');
  if (!wrap) return;
  wrap.innerHTML = composerTracks.map((t, idx) => `
    <div class="composer-track-row">
      <span>Track ${idx + 1}</span>
      <input type="text" placeholder="data/voices/example.wav" value="${esc(t.source_path)}" data-idx="${idx}" data-field="source_path">
      <label>Offset <input class="composer-num" type="number" step="0.1" value="${Number(t.offset_s || 0)}" data-idx="${idx}" data-field="offset_s">s</label>
      <label>Vol <input class="composer-num" type="number" min="0.1" max="2.0" step="0.1" value="${Number(t.volume || 1)}" data-idx="${idx}" data-field="volume"></label>
      <label><input type="checkbox" ${t.muted ? 'checked' : ''} data-idx="${idx}" data-field="muted"> Mute</label>
      <label><input type="checkbox" ${t.solo ? 'checked' : ''} data-idx="${idx}" data-field="solo"> Solo</label>
      <button class="btn btn-ghost btn-sm" data-composer-delete="${idx}" type="button">Delete</button>
    </div>
  `).join('') || '<p class="legend">No tracks yet.</p>';
}

async function renderComposerMix() {
  if (!composerTracks.length) return showToast('Add at least one track', 'error');
  byId('composer-status').innerHTML = '<span class="spin-dot"></span> Rendering...';
  const payload = {
    name: `Composition ${new Date().toLocaleString()}`,
    format: byId('composer-format')?.value || 'wav',
    sample_rate: 24000,
    tracks: composerTracks,
  };
  const data = await api('/api/composer/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const audio = byId('composer-audio');
  audio.src = data.download_url;
  byId('composer-result').style.display = '';
  byId('composer-status').textContent = 'Done ✓';
  await loadComposerHistory();
}

async function loadComposerHistory() {
  const data = await api('/api/composer/renders?limit=50&offset=0');
  const wrap = byId('composer-history');
  if (!wrap) return;
  const items = data.items || [];
  wrap.innerHTML = items.map((r) => `
    <div class="history-item">
      <strong>${esc(r.name || r.id)}</strong>
      <small>${esc(new Date(r.created_at).toLocaleString())}</small>
      <div class="form-row">
        <a class="btn btn-ghost btn-sm" href="/api/composer/render/${encodeURIComponent(r.id)}/audio">Play/Download</a>
      </div>
    </div>
  `).join('') || '<p class="legend">No renders yet.</p>';
}

async function reGenerateTTS(entry) {
  byId('tts-input').value = entry.full_text || '';
  byId('tts-counter').textContent = `${byId('tts-input').value.length} / 5,000`;

  const providerSel = byId('tts-provider');
  const provider = entry.provider || providerFromModel(entry.model);
  if (provider && [...providerSel.options].some((o) => o.value === provider)) {
    providerSel.value = provider;
    state.ttsPreferredProvider = provider;
  }
  state.ttsPreferredModel = entry.model || '';
  await loadTTSModels();
  if (entry.model && [...byId('tts-model').options].some((o) => o.value === entry.model)) {
    byId('tts-model').value = entry.model;
  }
  await loadTTSVoices(entry.voice || '');

  if (entry.speed) byId('tts-speed').value = entry.speed;
  if (entry.format) byId('tts-format').value = entry.format;
  byId('tts-speed-value').textContent = `${Number(byId('tts-speed').value).toFixed(1)}x`;
  byId('tts-preset').value = '';
  document.querySelector('.tab[data-tab="speak"]').click();
}

async function init() {
  initTheme();
  initTabs();
  bindEvents();
  refreshHistory();
  api('/health').then((h) => {
    const v = h.version ? `v${h.version}` : '';
    const el = byId('app-version');
    if (el) el.textContent = v;
    if (v) document.title = `Open Speech ${v}`;
  }).catch(() => {});

  // Step 1: fetch models ONCE, populate cache, render Models tab immediately
  try {
    const data = await api('/api/models');
    state.modelsCache = data.models || [];
  } catch (e) {
    // non-fatal — cache stays empty, renderModelsView shows empty state
  }
  renderModelsView();

  // Step 2: parallel init for everything else (use cached models, don't re-fetch)
  await Promise.allSettled([
    loadTTSProviders(),
    loadSTTModels(),
    loadProfiles(),
  ]);

  // Step 3: non-critical background loaders (don't block UI)
  Promise.allSettled([loadConversations(), loadComposerHistory()]).catch(() => {});

  if (!composerTracks.length) addComposerTrack();
}
document.addEventListener('DOMContentLoaded', () => {
  init().catch((e) => showToast(`Init failed: ${e.message}`, 'error'));
});
