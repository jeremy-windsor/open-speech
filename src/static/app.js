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
};
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
function byId(id) { return document.getElementById(id); }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const payload = await res.json();
      msg = payload?.detail?.message || payload?.detail || payload?.error || msg;
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
  if (stateName === 'downloaded' || stateName === 'ready') return '○ Ready';
  if (stateName === 'provider_installed' || stateName === 'available') return '○ Installed';
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
  if ((modelId || '').includes('/')) return modelId.split('/')[0];
  return modelId || 'kokoro';
}
async function loadTTSModels() {
  const data = await api('/api/models');
  const models = (data.models || []).filter((m) => classifyKind(m) === 'tts' && !(m.id || '').includes('pocket-tts'));
  const providerRank = { kokoro: 0, piper: 1 };
  models.sort((a, b) => {
    const ar = providerRank[a.provider] ?? 99;
    const br = providerRank[b.provider] ?? 99;
    if (ar !== br) return ar - br;
    return (a.id || '').localeCompare(b.id || '');
  });
  const sel = byId('tts-model');
  sel.innerHTML = models.map((m) => `<option value="${esc(m.id)}">${esc(m.id)} ${statusSuffix(m.state)}</option>`).join('');
  const kokoro = models.find((m) => m.provider === 'kokoro');
  const piper = models.find((m) => m.provider === 'piper');
  sel.value = kokoro?.id || piper?.id || models[0]?.id || '';
  await onTTSModelChanged();
}
async function loadSTTModels() {
  const data = await api('/api/models');
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
  const urls = [
    `/api/voices?model=${encodeURIComponent(model)}`,
    `/v1/audio/voices?model=${encodeURIComponent(model)}`,
    '/v1/audio/voices',
  ];
  for (const u of urls) {
    try {
      const data = await api(u);
      if (Array.isArray(data)) return data;
      if (Array.isArray(data.voices)) return data.voices;
    } catch {}
  }
  return [];
}
function renderAdvancedControls(caps) {
  const details = byId('tts-advanced');
  const wrap = byId('tts-advanced-content');
  wrap.innerHTML = '';
  const rows = [];
  if (caps.voice_clone) {
    rows.push('<div class="field"><label for="tts-clone-file">Voice Clone</label><input id="tts-clone-file" type="file" accept="audio/*"></div>');
  }
  if (caps.voice_blend) {
    rows.push('<div class="field"><label for="tts-blend">Voice Blend</label><input id="tts-blend" type="text" placeholder="voiceA:0.5,voiceB:0.5"></div>');
  }
  if (caps.instructions) {
    rows.push('<div class="field"><label for="tts-instructions">Instructions</label><input id="tts-instructions" type="text" placeholder="Style / direction"></div>');
  }
  details.hidden = rows.length === 0;
  if (rows.length > 0) {
    wrap.innerHTML = rows.join('');
    details.open = false;
  }
  byId('tts-stream-group').hidden = !caps.streaming;
}
async function onTTSModelChanged() {
  const model = byId('tts-model').value;
  state.ttsCaps = await fetchTTSCapabilities(model);
  state.ttsVoices = await fetchVoices(model);
  renderAdvancedControls(state.ttsCaps);
  const voiceSel = byId('tts-voice');
  voiceSel.innerHTML = (state.ttsVoices || []).map((v) => {
    const id = v.id || v.name || v;
    return `<option value="${esc(id)}">${esc(id)}</option>`;
  }).join('');
  if (!voiceSel.value && voiceSel.options.length) voiceSel.selectedIndex = 0;
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
async function loadModel(modelId) {
  try {
    await api(`/v1/audio/models/${encodeURIComponent(modelId)}`, { method: 'POST' });
    return;
  } catch {}
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
    setButtonState('tts-generate', 'downloading');
    await downloadModel(modelId);
  }
  const status3 = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (status3.state === 'downloaded') {
    setButtonState('tts-generate', 'loading');
    await loadModel(modelId);
  }
  const final = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (final.state !== 'loaded') throw new Error(`${kind.toUpperCase()} model not ready`);
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
async function doSpeak() {
  const model = byId('tts-model').value;
  const voice = byId('tts-voice').value;
  const input = byId('tts-input').value.trim();
  if (!input) return showToast('Enter text first', 'error');
  try {
    await ensureModelReady(model, 'tts');
    setButtonState('tts-generate', 'generating');
    const payload = {
      model, voice, input,
      speed: Number(byId('tts-speed').value),
      response_format: byId('tts-format').value,
    };
    const instructions = byId('tts-instructions');
    if (instructions) payload.instructions = instructions.value;
    const blend = byId('tts-blend');
    if (blend) payload.voice_blend = blend.value;
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
  if (model.state === 'downloaded') return { text: '● Downloaded', cls: 'downloaded' };
  return { text: '○ Available', cls: 'available' };
}
function actionSortRank(m) {
  if (m.state === 'loaded') return 0;
  if (m.state === 'downloaded') return 1;
  return 2;
}
function renderModelRow(m) {
  const op = state.modelOps[m.id];
  const badge = getStateBadge(m);
  const busy = op && (op.kind === 'downloading' || op.kind === 'loading');
  let actions = '';
  if (busy) {
    const label = op.kind === 'loading' ? 'Loading…' : 'Downloading…';
    actions = `<span class="row-status"><span class="spin-dot"></span>${esc(op.text || label)}</span>`;
  } else if (op?.error) {
    actions = `<span class="row-status error">${esc(op.error)}</span> <button class="btn btn-ghost btn-sm" data-download="${esc(m.id)}">Download</button>`;
  } else if (m.state === 'available' || m.state === 'provider_installed') {
    actions = `<button class="btn btn-ghost btn-sm" data-download="${esc(m.id)}">Download</button>`;
  } else if (m.state === 'downloaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  } else if (m.state === 'loaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-unload="${esc(m.id)}">Unload</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  }
  return `<div class="model-row" data-model-row="${esc(m.id)}"><div class="model-main"><span class="model-id">${esc(m.id)}</span><span class="state-badge ${badge.cls}">${badge.text}</span></div><div>${actions}</div></div>`;
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

  const installed = models.filter((m) => m.state !== 'provider_missing');
  const sttModels = installed.filter((m) => m.type === 'stt').sort((a, b) => actionSortRank(a) - actionSortRank(b) || (a.id || '').localeCompare(b.id || ''));
  const ttsModels = installed.filter((m) => m.type === 'tts').sort((a, b) => actionSortRank(a) - actionSortRank(b) || (a.id || '').localeCompare(b.id || ''));
  byId('stt-models-list').innerHTML = sttModels.map(renderModelRow).join('') || '<p class="legend">No STT models for installed providers.</p>';
  byId('tts-models-list').innerHTML = ttsModels.map(renderModelRow).join('') || '<p class="legend">No TTS models for installed providers.</p>';

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
  state.modelOps[modelId] = { kind, text: kind === 'loading' ? 'Loading…' : 'Downloading…' };
  renderModelsView();
  try {
    if (kind === 'downloading') {
      await downloadModel(modelId);
    } else {
      await loadModel(modelId);
    }
    while (true) {
      await new Promise((r) => setTimeout(r, 2000));
      const status = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
      const nextText = status.progress ? `${kind === 'loading' ? 'Loading…' : 'Downloading…'} ${status.progress}` : (kind === 'loading' ? 'Loading…' : 'Downloading…');
      state.modelOps[modelId] = { kind, text: nextText };
      const idx = state.modelsCache.findIndex((m) => m.id === modelId);
      if (idx >= 0) state.modelsCache[idx] = { ...state.modelsCache[idx], ...status };
      renderModelsView();
      if (kind === 'loading' && status.state === 'loaded') break;
      if (kind === 'downloading' && (status.state === 'downloaded' || status.state === 'loaded')) break;
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
  byId('tts-model').addEventListener('change', onTTSModelChanged);
  byId('tts-profile')?.addEventListener('change', (e) => applyProfile(e.target.value).catch((err) => showToast(err.message, 'error')));
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
  document.addEventListener('click', async (e) => {
    const unload = e.target.closest('[data-unload]');
    const download = e.target.closest('[data-download]');
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
      }
      const profileDelete = e.target.closest('[data-profile-delete]');
      const profileDefault = e.target.closest('[data-profile-default]');
      const histDelete = e.target.closest('[data-history-delete]');
      const histRegen = e.target.closest('[data-history-regen]');
      if (profileDelete) await deleteProfile(profileDelete.dataset.profileDelete);
      if (profileDefault) await setDefaultProfile(profileDefault.dataset.profileDefault);
      if (histDelete) await deleteHistoryEntry(histDelete.dataset.historyDelete);
      if (histRegen) reGenerateTTS(JSON.parse(histRegen.dataset.historyRegen));
    } catch (err) {
      showToast(err.message, 'error');
    }
  });
}

async function loadProfiles() {
  const data = await api('/api/profiles');
  state.profiles = data.profiles || [];
  state.defaultProfileId = data.default_profile_id || null;

  const profileSel = byId('tts-profile');
  if (profileSel) {
    profileSel.innerHTML = '<option value="">— No Profile —</option>' + state.profiles.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
    if (state.defaultProfileId) profileSel.value = state.defaultProfileId;
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
  byId('tts-model').value = profile.model || byId('tts-model').value;
  await onTTSModelChanged();
  byId('tts-voice').value = profile.voice || byId('tts-voice').value;
  byId('tts-speed').value = Number(profile.speed || 1.0);
  byId('tts-speed-value').textContent = `${Number(byId('tts-speed').value).toFixed(1)}x`;
  byId('tts-format').value = profile.format || byId('tts-format').value;
  const blend = byId('tts-blend');
  if (blend) blend.value = profile.blend || '';
}

async function saveAsProfile() {
  const modelId = byId('tts-model').value;
  const payload = {
    name: window.prompt('Profile name?'),
    backend: providerFromModel(modelId),
    model: modelId,
    voice: byId('tts-voice').value,
    speed: Number(byId('tts-speed').value),
    format: byId('tts-format').value,
    blend: byId('tts-blend')?.value || null,
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

function reGenerateTTS(entry) {
  byId('tts-input').value = entry.full_text || '';
  byId('tts-counter').textContent = `${byId('tts-input').value.length} / 5,000`;
  if (entry.model) byId('tts-model').value = entry.model;
  if (entry.voice) byId('tts-voice').value = entry.voice;
  if (entry.speed) byId('tts-speed').value = entry.speed;
  if (entry.format) byId('tts-format').value = entry.format;
  document.querySelector('.tab[data-tab="speak"]').click();
}

async function init() {
  initTheme();
  initTabs();
  bindEvents();
  refreshHistory();
  await Promise.all([loadTTSModels(), loadSTTModels(), refreshModels(), loadProfiles()]);
}
document.addEventListener('DOMContentLoaded', () => {
  init().catch((e) => showToast(`Init failed: ${e.message}`, 'error'));
});
