const state = {
  ttsCaps: {},
  ttsVoices: [],
  ttsAudioBlob: null,
  ttsAudioUrl: null,
  mediaStream: null,
  audioCtx: null,
  audioSource: null,
  scriptProcessor: null,
  ws: null,
  micStopTimer: null,
  sttRecording: false,
  sttSession: null,
  sttChunkTimer: null,
  profiles: [],
  defaultProfileId: null,
  history: { items: [], total: 0, limit: 50, offset: 0, type: "" },
  modelsCache: [],
  defaultSttModel: '',
  modelOps: {},
  modelsBusy: false,
  ttsPreferredProvider: '',
  ttsPreferredModel: '',
  currentConversationId: null,
  currentConversation: null,
  liveReader: null,
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
  downloading: { text: 'Downloading model…', loading: true },
  loading: { text: 'Loading model…', loading: true },
  generating: { text: 'Generating…', loading: true },
};
const MIC_STOP_GRACE_MS = 4000;
const MIC_DEVICE_STORAGE_KEY = 'open-speech-mic-device';
const PROVIDER_DISPLAY = {
  'kokoro': 'Kokoro',
  'piper': 'Piper',
  'pocket-tts': 'Pocket TTS',
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
  if (stateName === 'downloaded' || stateName === 'ready') return '○ Downloaded';
  if (stateName === 'provider_installed' || stateName === 'available') return '○ Ready';
  if (stateName === 'provider_missing') return '✗ Not installed';
  return '○ Ready';
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
  return (state.modelsCache || []).filter((m) => classifyKind(m) === 'tts' && m.state !== 'provider_missing' && m.provider_available !== false);
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
      state.defaultSttModel = data.default_stt_model || '';
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
  const defaultModel = state.defaultSttModel && models.find((m) => m.id === state.defaultSttModel);
  if (defaultModel?.id) {
    sel.value = defaultModel.id;
  } else if (models[0]?.id) {
    sel.value = models[0].id;
  }
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
  setTTSSpeed(byId('tts-speed').value);
}
function setTTSSpeed(value) {
  const control = byId('tts-speed');
  const supported = state.ttsCaps.speed_control !== false;
  const requested = Number(value) || 1.0;
  control.disabled = !supported;
  control.title = supported ? '' : 'The selected model does not support speed control';
  control.value = supported ? requested : 1.0;
  byId('tts-speed-value').textContent = supported
    ? `${Number(control.value).toFixed(1)}x`
    : '1.0x (unsupported)';
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
    const provider = providerFromModel(modelId);
    throw new Error(`Provider not installed — rebuild image with BAKED_PROVIDERS=${provider}`);
  }

  if (status.state === 'provider_installed' || status.state === 'available') {
    if (kind === 'tts') {
      setButtonState('tts-generate', 'loading');
      await loadModel(modelId);
    } else {
      setButtonState('tts-generate', 'downloading');
      await downloadModel(modelId);
    }
  }

  const status2 = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
  if (status2.state === 'downloaded' || status2.state === 'ready') {
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
  if (input.length > 4096) {
    return showToast('Generate accepts up to 4,096 characters. Use Live Reader for longer text.', 'error');
  }
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
    const instructions = byId('tts-instructions')?.value.trim();
    if (instructions) payload.instructions = instructions;
    if (blendVoices.length > 0) {
      payload.voice = blendVoices.map((b) => `${b.voice}(${b.weight})`).join('+');
    }
    const doStream = !byId('tts-stream-group').hidden && byId('tts-stream').checked;
    const fmt = byId('tts-format').value;
    const canStreamFormat = ['mp3'].includes(fmt); // mp3 is most reliable for MSE
    const shouldStream = doStream && canStreamFormat;
    const endpoint = '/v1/audio/speech' + (shouldStream ? '?stream=true' : '');
    const res = await fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'X-History': 'true' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());

    if (shouldStream && window.MediaSource && MediaSource.isTypeSupported('audio/mpeg') && res.body) {
      // Progressive streaming playback via MediaSource API
      const audioEl = byId('tts-audio');
      const ms = new MediaSource();
      const streamUrl = URL.createObjectURL(ms);
      if (state.ttsAudioUrl) URL.revokeObjectURL(state.ttsAudioUrl);
      state.ttsAudioBlob = null;
      state.ttsAudioUrl = streamUrl;

      byId('audio-player').hidden = false;
      audioEl.src = streamUrl;

      await new Promise((resolve, reject) => {
        ms.addEventListener('sourceopen', async () => {
          let sb;
          try {
            sb = ms.addSourceBuffer('audio/mpeg');
          } catch (e) {
            reject(e);
            return;
          }

          const reader = res.body.getReader();
          const appendNext = async () => {
            try {
              const { done, value } = await reader.read();
              if (done) {
                if (ms.readyState === 'open') ms.endOfStream();
                resolve();
                return;
              }
              if (sb.updating) {
                await new Promise((r) => sb.addEventListener('updateend', r, { once: true }));
              }
              sb.appendBuffer(value);
              sb.addEventListener('updateend', appendNext, { once: true });
            } catch (err) {
              reject(err);
            }
          };

          audioEl.addEventListener('canplay', () => audioEl.play().catch(() => {}), { once: true });
          appendNext();
        }, { once: true });
        ms.addEventListener('error', reject, { once: true });
      });
    } else {
      // Non-streaming: collect all audio then play (original behavior)
      const blob = await res.blob();
      if (state.ttsAudioUrl) URL.revokeObjectURL(state.ttsAudioUrl);
      state.ttsAudioBlob = blob;
      state.ttsAudioUrl = URL.createObjectURL(blob);
      byId('audio-player').hidden = false;
      byId('tts-audio').src = state.ttsAudioUrl;
      byId('tts-audio').play().catch(() => {});
    }
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

function setLiveReaderStatus(message, kind = '') {
  const status = byId('live-reader-status');
  status.textContent = message;
  status.classList.toggle('connected', kind === 'connected');
  status.classList.toggle('error', kind === 'error');
}

function setLiveReaderControls(active, paused = false) {
  byId('live-reader-start').disabled = active;
  byId('live-reader-read-all').disabled = active;
  byId('live-reader-pause').disabled = !active;
  byId('live-reader-stop').disabled = !active;
  byId('live-reader-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
  byId('live-reader-mode').disabled = active;
}

function liveReaderWsUrl() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${protocol}://${location.host}/v1/audio/speech/stream`;
}

function decodePcm16(base64) {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function sendLiveReaderEvent(reader, event) {
  if (reader.ws.readyState !== WebSocket.OPEN) return false;
  reader.ws.send(JSON.stringify(event));
  return true;
}

function sliceLiveReaderText(text, maxCharacters) {
  let end = 0;
  let count = 0;
  for (const character of text) {
    if (count >= maxCharacters) break;
    end += character.length;
    count += 1;
  }
  return text.slice(0, end);
}

function pumpLiveReaderInput(reader) {
  if (state.liveReader !== reader || reader.ws.readyState !== WebSocket.OPEN || !reader.configured) return;
  if (reader.inflight.length) return;
  while (reader.outbound.length) {
    const next = reader.outbound[0];
    if (next.type === 'commit') {
      if (reader.inflight.length) return;
      sendLiveReaderEvent(reader, { type: 'input_text.commit' });
      reader.outbound.shift();
      continue;
    }
    const room = Math.max(0, reader.maxBufferChars - reader.serverBufferedChars);
    if (!room) return;
    const text = sliceLiveReaderText(next.text, Math.min(2048, room));
    if (!text) return;
    sendLiveReaderEvent(reader, { type: 'input_text.append', text });
    reader.inflight.push(text);
    next.text = next.text.slice(text.length);
    if (!next.text) reader.outbound.shift();
    return;
  }
}

function queueLiveReaderText(reader, text, commit = false) {
  if (text) reader.outbound.push({ type: 'text', text });
  if (commit) reader.outbound.push({ type: 'commit' });
  pumpLiveReaderInput(reader);
}

function flushLiveReaderTextBatch(reader) {
  clearTimeout(reader.inputBatchTimer);
  reader.inputBatchTimer = null;
  const text = reader.inputBatch;
  reader.inputBatch = '';
  queueLiveReaderText(reader, text);
}

function scheduleLiveReaderAudio(reader, event) {
  if (state.liveReader !== reader || reader.stopping || event.generation !== reader.generation) return;
  const samples = decodePcm16(event.delta);
  const buffer = reader.audioCtx.createBuffer(1, samples.length, event.sample_rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] / 32768;
  const source = reader.audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(reader.audioCtx.destination);
  const startAt = Math.max(reader.nextStartTime, reader.audioCtx.currentTime + 0.025);
  reader.nextStartTime = startAt + buffer.duration;
  reader.sources.add(source);
  source.onended = () => {
    reader.sources.delete(source);
    source.disconnect();
    if (state.liveReader !== reader || reader.stopping || event.generation !== reader.generation) return;
    sendLiveReaderEvent(reader, {
      type: 'playback.ack',
      response_id: event.response_id,
      sequence: event.sequence,
    });
  };
  source.start(startAt);
}

function handleLiveReaderEvent(reader, event) {
  if (state.liveReader !== reader) return;
  if (event.type === 'session.created') {
    reader.maxBufferChars = event.session?.limits?.max_buffer_chars || 8192;
    reader.generation = 0;
    sendLiveReaderEvent(reader, {
      type: 'session.update',
      session: reader.sessionConfig,
    });
    return;
  }
  if (event.type === 'session.updated') {
    reader.configured = true;
    if (document.hidden) {
      reader.visibilityPaused = true;
      reader.audioCtx.suspend().catch(() => {});
      sendLiveReaderEvent(reader, { type: 'playback.pause' });
      setLiveReaderStatus('Paused in background');
    } else {
      setLiveReaderStatus('Listening', 'connected');
      byId('live-reader-now').textContent = 'Listening for text…';
    }
    queueLiveReaderText(reader, reader.initialText, Boolean(reader.initialText));
    reader.initialText = '';
    pumpLiveReaderInput(reader);
    return;
  }
  if (event.type === 'input_text.accepted') {
    reader.serverBufferedChars = event.buffered_chars || 0;
    reader.inflight.shift();
    if (Number.isInteger(event.generation)) reader.generation = event.generation;
    pumpLiveReaderInput(reader);
    return;
  }
  if (event.type === 'session.status') {
    reader.serverBufferedChars = event.buffered_chars || 0;
    if (Number.isInteger(event.generation)) reader.generation = event.generation;
    pumpLiveReaderInput(reader);
    return;
  }
  if (event.type === 'response.created') {
    const spoken = event.response?.text || '';
    const now = byId('live-reader-now');
    now.textContent = spoken ? `Reading: ${spoken}` : 'Reading…';
    now.classList.add('active');
    return;
  }
  if (event.type === 'response.output_audio.delta') {
    scheduleLiveReaderAudio(reader, event);
    return;
  }
  if (event.type === 'response.done') {
    byId('live-reader-now').textContent = 'Listening for text…';
    byId('live-reader-now').classList.remove('active');
    return;
  }
  if (event.type === 'response.cancelled') {
    reader.generation = Number.isInteger(event.generation) ? event.generation + 1 : reader.generation + 1;
    reader.inflight.length = 0;
    reader.outbound.length = 0;
    reader.inputBatch = '';
    clearTimeout(reader.inputBatchTimer);
    reader.inputBatchTimer = null;
    return;
  }
  if (event.type === 'error') {
    const message = event.error?.message || 'Live Reader error';
    if (event.error?.code === 'buffer_overflow' && reader.inflight.length) {
      const rejected = reader.inflight.shift();
      reader.outbound.unshift({ type: 'text', text: rejected });
      reader.serverBufferedChars = reader.maxBufferChars;
      return;
    }
    reader.fatalStatus = event.error?.code === 'session_busy' ? 'Busy' : 'Error';
    reader.fatalMessage = message;
    setLiveReaderStatus('Error', 'error');
    byId('live-reader-now').textContent = message;
    showToast(message, 'error');
  }
}

function releaseLiveReader(reader, message = 'Stopped', detail = '') {
  if (state.liveReader !== reader) return;
  clearInterval(reader.keepaliveTimer);
  clearTimeout(reader.inputBatchTimer);
  reader.stopping = true;
  reader.sources.forEach((source) => { try { source.stop(); } catch {} });
  reader.sources.clear();
  if (reader.audioCtx.state !== 'closed') reader.audioCtx.close().catch(() => {});
  state.liveReader = null;
  setLiveReaderControls(false);
  setLiveReaderStatus(message, ['Busy', 'Error', 'Disconnected'].includes(message) ? 'error' : '');
  byId('live-reader-now').textContent = detail || 'Start once, then type or paste. Completed words are sent automatically.';
  byId('live-reader-now').classList.remove('active');
}

async function stopLiveReader(message = 'Stopped') {
  const reader = state.liveReader;
  if (!reader) return;
  reader.stopping = true;
  reader.sources.forEach((source) => { try { source.stop(); } catch {} });
  reader.sources.clear();
  sendLiveReaderEvent(reader, { type: 'response.cancel' });
  if (reader.ws.readyState < WebSocket.CLOSING) reader.ws.close(1000, 'Stopped by user');
  releaseLiveReader(reader, message);
}

async function startLiveReader({ readAll = false } = {}) {
  if (state.liveReader) return;
  const input = byId('tts-input');
  const startOffset = readAll ? 0 : (input.selectionStart ?? input.value.length);
  const model = byId('tts-model').value;
  const voice = blendVoices.length
    ? blendVoices.map((item) => `${item.voice}(${item.weight})`).join('+')
    : byId('tts-voice').value;
  const sessionConfig = {
    model,
    voice,
    speed: Number(byId('tts-speed').value),
    latency_mode: byId('live-reader-mode').value,
  };
  setLiveReaderControls(true);
  setLiveReaderStatus('Preparing…');
  byId('live-reader-now').textContent = 'Preparing the selected voice…';
  const audioCtx = new AudioContext();
  try {
    // Resume from the button gesture before model/network waits consume browser activation.
    await audioCtx.resume();
    await ensureModelReady(model, 'tts');
    const ws = new WebSocket(liveReaderWsUrl());
    const reader = {
      ws,
      audioCtx,
      sources: new Set(),
      nextStartTime: audioCtx.currentTime,
      outbound: [],
      inflight: [],
      maxBufferChars: 8192,
      serverBufferedChars: 0,
      generation: 0,
      sessionConfig,
      configured: false,
      stopping: false,
      paused: false,
      visibilityPaused: false,
      inputBatch: '',
      inputBatchTimer: null,
      fatalStatus: '',
      fatalMessage: '',
      lastValue: input.value,
      startOffset,
      initialText: input.value.slice(startOffset),
      keepaliveTimer: null,
    };
    state.liveReader = reader;
    setLiveReaderControls(true);
    setLiveReaderStatus('Connecting…');
    byId('live-reader-now').textContent = 'Connecting to the remote voice engine…';
    ws.onmessage = (message) => {
      try { handleLiveReaderEvent(reader, JSON.parse(String(message.data))); }
      catch { showToast('Live Reader received invalid data', 'error'); }
    };
    ws.onerror = () => {
      if (!reader.stopping) setLiveReaderStatus('Connection error', 'error');
    };
    ws.onclose = () => {
      if (state.liveReader !== reader) return;
      const status = reader.stopping ? 'Stopped' : (reader.fatalStatus || 'Disconnected');
      releaseLiveReader(reader, status, reader.fatalMessage);
    };
    reader.keepaliveTimer = setInterval(() => {
      sendLiveReaderEvent(reader, { type: 'session.keepalive' });
    }, 30000);
  } catch (error) {
    if (state.liveReader) {
      await stopLiveReader('Stopped');
    } else {
      if (audioCtx.state !== 'closed') await audioCtx.close().catch(() => {});
      setLiveReaderControls(false);
      setLiveReaderStatus('Error', 'error');
    }
    showToast(`Live Reader failed: ${error.message}`, 'error');
  } finally {
    setButtonState('tts-generate', 'idle');
  }
}

function handleLiveReaderTextChange(value) {
  const reader = state.liveReader;
  if (!reader) return;
  if (!value.startsWith(reader.lastValue)) {
    stopLiveReader('Text changed').catch(() => {});
    showToast('Live Reader stopped because earlier text was edited. Start again at the new cursor.', 'error');
    return;
  }
  const delta = value.slice(reader.lastValue.length);
  reader.lastValue = value;
  if (!reader.configured) {
    reader.initialText += delta;
    return;
  }
  reader.inputBatch += delta;
  if (!reader.inputBatchTimer) {
    reader.inputBatchTimer = setTimeout(() => flushLiveReaderTextBatch(reader), 40);
  }
}

async function toggleLiveReaderPause() {
  const reader = state.liveReader;
  if (!reader) return;
  if (reader.paused) {
    await reader.audioCtx.resume();
    reader.paused = false;
    sendLiveReaderEvent(reader, { type: 'playback.resume' });
    setLiveReaderStatus('Listening', 'connected');
  } else {
    await reader.audioCtx.suspend();
    reader.paused = true;
    sendLiveReaderEvent(reader, { type: 'playback.pause' });
    setLiveReaderStatus('Paused');
  }
  setLiveReaderControls(true, reader.paused);
}

async function handleLiveReaderVisibilityChange() {
  const reader = state.liveReader;
  if (!reader || reader.stopping) return;
  if (document.hidden && !reader.paused && !reader.visibilityPaused) {
    reader.visibilityPaused = true;
    await reader.audioCtx.suspend();
    sendLiveReaderEvent(reader, { type: 'playback.pause' });
    setLiveReaderStatus('Paused in background');
  } else if (!document.hidden && reader.visibilityPaused) {
    reader.visibilityPaused = false;
    await reader.audioCtx.resume();
    sendLiveReaderEvent(reader, { type: 'playback.resume' });
    setLiveReaderStatus(reader.paused ? 'Paused' : 'Listening', reader.paused ? '' : 'connected');
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
    const res = await fetch('/v1/audio/transcriptions', { method: 'POST', headers: { 'X-History': 'true' }, body: fd });
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

function getSavedMicDeviceId() {
  try {
    return localStorage.getItem(MIC_DEVICE_STORAGE_KEY) || '';
  } catch {
    return '';
  }
}

function saveMicDeviceId(deviceId) {
  try {
    if (deviceId) localStorage.setItem(MIC_DEVICE_STORAGE_KEY, deviceId);
    else localStorage.removeItem(MIC_DEVICE_STORAGE_KEY);
  } catch {}
}

function resetMicDeviceSelect(disabled = false) {
  const sel = byId('mic-select');
  if (!sel) return;
  sel.innerHTML = '<option value="">Default microphone</option>';
  sel.value = '';
  sel.disabled = disabled;
}

async function loadMicDevices() {
  const sel = byId('mic-select');
  if (!sel) return;
  if (!navigator.mediaDevices?.enumerateDevices) {
    resetMicDeviceSelect(true);
    return;
  }

  let audioInputs = [];
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    audioInputs = devices.filter((device) => device.kind === 'audioinput');
  } catch {
    resetMicDeviceSelect(true);
    return;
  }

  const savedDeviceId = getSavedMicDeviceId();
  const hasSavedDevice = !!savedDeviceId && audioInputs.some((device) => device.deviceId === savedDeviceId);
  if (savedDeviceId && !hasSavedDevice) saveMicDeviceId('');

  const options = ['<option value="">Default microphone</option>']
    .concat(audioInputs.map((device, index) => {
      const label = device.label || `Microphone ${index + 1}`;
      const value = device.deviceId || '';
      const disabled = value ? '' : ' disabled';
      return `<option value="${esc(value)}"${disabled}>${esc(label)}</option>`;
    }));
  sel.innerHTML = options.join('');
  sel.disabled = audioInputs.length === 0;
  sel.value = hasSavedDevice ? savedDeviceId : '';
}

async function refreshMicDevicesAfterPermission() {
  await loadMicDevices().catch(() => {});
}

async function getMicStream() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Browser microphone capture is not available');
  }

  const selectedDeviceId = byId('mic-select')?.value || '';
  if (selectedDeviceId) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { deviceId: { exact: selectedDeviceId } },
      });
      await refreshMicDevicesAfterPermission();
      return stream;
    } catch (selectedDeviceError) {
      saveMicDeviceId('');
      const sel = byId('mic-select');
      if (sel) sel.value = '';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      await refreshMicDevicesAfterPermission();
      showToast('Selected microphone unavailable; using default microphone');
      return stream;
    }
  }

  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  await refreshMicDevicesAfterPermission();
  return stream;
}

function setMicUiIdle() {
  const btn = byId('mic-btn');
  state.sttRecording = false;
  btn.textContent = '🎙 Start Mic';
  btn.classList.remove('btn-danger', 'mic-recording');
  byId('vad-dot').className = 'status-dot';
  byId('vad-text').textContent = 'Silence';
}

function clearMicStopTimer(ws = null) {
  if (!state.micStopTimer) return;
  if (ws && state.micStopTimer.ws !== ws) return;
  clearTimeout(state.micStopTimer.id);
  state.micStopTimer = null;
}

function stopMicCapture() {
  stopMicWaveform();
  state.scriptProcessor?.disconnect();
  state.audioSource?.disconnect();
  state.mediaStream?.getTracks().forEach((t) => t.stop());
  if (state.audioCtx) state.audioCtx.close().catch(() => {});
  state.scriptProcessor = null;
  state.audioSource = null;
  state.mediaStream = null;
  state.audioCtx = null;
  state.sttRecording = false;
  setMicUiIdle();
}

function forgetMicSocket(ws) {
  if (state.ws === ws) state.ws = null;
  clearMicStopTimer(ws);
}

function stopMicSession({ closeWs = true, graceful = false } = {}) {
  const ws = state.ws;
  stopMicCapture();
  if (!closeWs || !ws) {
    if (!closeWs) forgetMicSocket(ws);
    return;
  }

  if (ws.readyState >= WebSocket.CLOSING) {
    forgetMicSocket(ws);
    return;
  }

  if (graceful && ws.readyState === WebSocket.OPEN) {
    try {
      ws.send(JSON.stringify({ type: 'stop' }));
      state.sttSession = { ...(state.sttSession || {}), stoppingAt: Date.now() };
    } catch { }
    clearMicStopTimer(ws);
    const timerId = setTimeout(() => {
      if (ws.readyState < WebSocket.CLOSING) ws.close();
      forgetMicSocket(ws);
    }, MIC_STOP_GRACE_MS);
    state.micStopTimer = { id: timerId, ws };
    return;
  }

  ws.close();
  forgetMicSocket(ws);
}

async function toggleMic() {
  if (state.sttRecording) {
    stopMicSession({ closeWs: true, graceful: true });
    return;
  }
  const btn = byId('mic-btn');
  const model = byId('stt-model').value;
  try {
    await ensureModelReady(model, 'stt');
    const stream = await getMicStream();
    state.mediaStream = stream;

    const audioCtx = new AudioContext();
    const sampleRate = Math.round(audioCtx.sampleRate || 16000);
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioCtx.destination);
    state.audioCtx = audioCtx;
    state.audioSource = source;
    state.scriptProcessor = processor;

    const qs = new URLSearchParams({
      sample_rate: String(sampleRate),
      interim_results: 'true',
    });
    if (model) qs.set('model', model);
    const wsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/v1/audio/stream?${qs.toString()}`;
    const ws = new WebSocket(wsUrl);
    state.ws = ws;
    state.sttSession = {
      startedAt: Date.now(),
      model,
      sampleRate,
      finalSegments: 0,
      errors: 0,
      disconnected: false,
    };

    processor.onaudioprocess = (e) => {
      if (state.ws !== ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const int16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i += 1) {
        int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
      }
      ws.send(int16.buffer);
    };

    ws.onmessage = (ev) => {
      if (state.ws !== ws) return;
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'session.begin') {
          state.sttSession = {
            ...(state.sttSession || {}),
            sessionId: msg.session_id,
            model: msg.model || state.sttSession?.model || model,
            sampleRate: msg.sample_rate || state.sttSession?.sampleRate || sampleRate,
            internalSampleRate: msg.internal_sample_rate,
          };
        }
        if (msg.type === 'transcript') {
          if (!msg.is_final) {
            byId('stt-partial').textContent = msg.text || '…';
          } else {
            byId('stt-partial').textContent = '';
            byId('stt-final').textContent = msg.text || '—';
            if (msg.speech_final === true) {
              state.sttSession = {
                ...(state.sttSession || {}),
                finalSegments: (state.sttSession?.finalSegments || 0) + 1,
              };
              pushHistory(HISTORY_KEYS.stt, {
                text: msg.text || '',
                model: state.sttSession?.model || model,
                sampleRate: state.sttSession?.sampleRate || sampleRate,
                sessionId: state.sttSession?.sessionId || '',
                startedAt: state.sttSession?.startedAt || Date.now(),
              });
              refreshHistory();
            }
          }
        }
        if (msg.type === 'vad') {
          const speaking = msg.state === 'speech_start';
          byId('vad-dot').className = `status-dot ${speaking ? 'loaded' : ''}`;
          byId('vad-text').textContent = speaking ? 'Speech' : 'Silence';
        }
      } catch {}
    };
    ws.onerror = () => {
      if (state.ws !== ws) return;
      const wasRecording = state.sttRecording;
      state.sttSession = {
        ...(state.sttSession || {}),
        errors: (state.sttSession?.errors || 0) + 1,
        lastError: 'socket error',
        lastErrorAt: Date.now(),
      };
      if (wasRecording) showToast('Mic stream socket error', 'error');
      stopMicSession({ closeWs: false });
    };
    ws.onclose = (ev) => {
      if (state.ws !== ws) return;
      const wasRecording = state.sttRecording;
      state.sttSession = {
        ...(state.sttSession || {}),
        disconnected: true,
        lastDisconnect: {
          at: Date.now(),
          code: ev.code,
          reason: ev.reason || '',
          wasClean: ev.wasClean,
        },
      };
      forgetMicSocket(ws);
      if (state.sttRecording) {
        showToast('Mic stream disconnected', 'error');
        stopMicSession({ closeWs: false });
      }
      if (!wasRecording) stopMicCapture();
    };

    state.sttRecording = true;
    btn.textContent = '⏹ Stop';
    btn.classList.add('btn-danger', 'mic-recording');
    startMicWaveform();
    showToast('Mic recording started');
  } catch (e) {
    stopMicSession({ closeWs: true });
    showToast(`Mic failed: ${e.message}`, 'error');
  }
}
function getStateBadge(model) {
  if (model.state === 'provider_missing' || model.provider_available === false) return { text: '✗ Not installed', cls: 'error' };
  if (model.state === 'loaded') return { text: '● Loaded', cls: 'loaded' };
  if (model.state === 'downloaded') return { text: '● Downloaded', cls: 'downloaded' };
  if (model.state === 'provider_installed' || model.state === 'available') return { text: '○ Ready', cls: 'available' };
  return { text: '○ Ready', cls: 'available' };
}
function getModelHint(model) {
  if (model.state === 'provider_missing' || model.provider_available === false) return 'Provider not installed — rebuild image with this provider baked in';
  if (model.state === 'provider_installed' || model.state === 'available') {
    const size = formatSize(model.size_mb);
    return size
      ? `Ready — Download weights (${size}), then Load to GPU`
      : 'Ready — Download weights, then Load to GPU';
  }
  if (model.state === 'downloaded') return 'Downloaded to disk — click Load to GPU to activate';
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
  const unavailable = m.state === 'provider_missing' || m.provider_available === false;
  let actions = '';

  if (unavailable) {
    actions = `<span class="row-status muted" style="opacity:0.6">Not installed — rebuild with BAKED_PROVIDERS including ${esc(m.provider || 'provider')}</span>`;
  } else if (busy) {
    const label = op.kind === 'loading' ? 'Loading…' : 'Downloading…';
    actions = `<span class="row-status"><span class="spin-dot"></span>${esc(op.text || label)}</span>`;
  } else if (op?.error) {
    actions = `<span class="row-status error">${esc(op.error)}</span> <button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Download</button>`;
  } else if (m.state === 'available') {
    actions = `<button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Download</button> <button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load to GPU</button>`;
  } else if (m.state === 'provider_installed') {
    actions = `<button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Download</button> <button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load to GPU</button>`;
  } else if (m.state === 'downloaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load to GPU</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  } else if (m.state === 'loaded') {
    actions = `<button class="btn btn-ghost btn-sm" data-unload="${esc(m.id)}">Unload</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  }
  const sizeTxt = m.size_mb ? `<span class="model-size">${esc(formatSize(m.size_mb))}</span>` : '';
  const descTxt = m.description ? `<span class="model-desc">${esc(m.description)}</span>` : '';
  const hint = getModelHint(m);
  const hintTxt = hint ? `<span class="model-desc">${esc(hint)}</span>` : '';
  const rowCls = `model-row${unavailable ? ' unavailable' : ''}`;
  return `<div class="${rowCls}" data-model-row="${esc(m.id)}">
    <div class="model-main">
      <span class="model-id">${esc(m.id)}</span>${descTxt}${hintTxt}
      <span class="state-badge ${badge.cls}">${badge.text}</span>${sizeTxt}
    </div>
    <div class="model-actions">${actions}</div>
  </div>`;
}
// Piper voice display name parser
function parsePiperVoice(modelId) {
  const id = (modelId || '').replace(/^piper\//, '');
  // Pattern: lang_CC-name-quality or lang-name-quality
  const m = id.match(/^([a-z]{2})_([A-Z]{2})-(.+)-(low|medium|high)$/);
  if (m) {
    const name = m[3].charAt(0).toUpperCase() + m[3].slice(1);
    const cc = m[2];
    const quality = m[4];
    const stars = quality === 'high' ? '★★★' : quality === 'medium' ? '★★' : '★';
    return { name: `${name} (${cc})`, quality, stars, qualityLabel: quality.charAt(0).toUpperCase() + quality.slice(1), sortKey: `${m[3]}-${cc}` };
  }
  return { name: id, quality: 'medium', stars: '★★', qualityLabel: 'Medium', sortKey: id };
}

function stripSttPrefix(modelId) {
  return (modelId || '')
    .replace(/^Systran\/faster-distil-whisper-/, 'distil-')
    .replace(/^Systran\/faster-whisper-/, '')
    .replace(/^deepdml\/faster-whisper-/, '')
    .replace(/-ct2$/, '');
}

function getProviderOverallStatus(models) {
  if (models.some((m) => m.state === 'loaded')) return { text: 'Loaded ●', cls: 'loaded' };
  if (models.some((m) => m.state === 'downloaded')) return { text: 'Downloaded', cls: 'downloaded' };
  return { text: 'Available', cls: 'available' };
}

function renderModelActions(m) {
  const op = state.modelOps[m.id];
  const busy = op && (op.kind === 'downloading' || op.kind === 'loading' || op.kind === 'prefetch');
  if (busy) {
    const label = op.kind === 'loading' ? 'Loading…' : 'Downloading…';
    return `<span class="row-status"><span class="spin-dot"></span>${esc(op.text || label)}</span>`;
  }
  if (op?.error) {
    return `<span class="row-status error">${esc(op.error)}</span>`;
  }
  if (m.state === 'loaded') return `<button class="btn btn-ghost btn-sm" data-unload="${esc(m.id)}">Unload</button>`;
  if (m.state === 'downloaded') return `<button class="btn btn-ghost btn-sm" data-load="${esc(m.id)}">Load</button> <button class="btn btn-ghost btn-sm" data-delete-model="${esc(m.id)}">Delete</button>`;
  if (m.state === 'available' || m.state === 'provider_installed') return `<button class="btn btn-ghost btn-sm" data-prefetch="${esc(m.id)}">Download</button>`;
  return '';
}

function renderStatusDot(m) {
  if (m.state === 'loaded') return '<span style="color:var(--success)">● Loaded</span>';
  if (m.state === 'downloaded') return '<span style="color:#60a5fa">● Cached</span>';
  return '<span style="color:var(--text2)">—</span>';
}

function renderKokoroCard(models) {
  const m = models[0]; // kokoro is one model
  if (!m) return '';
  const status = getProviderOverallStatus(models);
  const voices = (state.ttsVoices || []).filter(() => true); // kokoro voices come from voice API
  const voiceTags = voices.length
    ? voices.slice(0, 20).map((v) => {
        const id = v.id || v.name || v;
        return `<span class="kokoro-voice-tag">${esc(id)}</span>`;
      }).join('') + (voices.length > 20 ? `<span class="kokoro-voice-tag">+${voices.length - 20} more</span>` : '')
    : '<span class="legend">Voices load when model is active</span>';
  return `<div class="provider-card">
    <div class="provider-card-header">
      <h3><button class="provider-card-toggle" type="button" aria-expanded="true" onclick="toggleProviderCard(this)"><span class="chevron" aria-hidden="true">▼</span> Kokoro TTS</button></h3>
      <span class="provider-status ${status.cls}">${status.text}</span>
    </div>
    <div class="provider-card-body">
      <div class="kokoro-info">
        ${renderModelActions(m)}
      </div>
      <div class="kokoro-voices">${voiceTags}</div>
    </div>
  </div>`;
}

function renderPiperCard(models) {
  if (!models.length) return '';
  const status = getProviderOverallStatus(models);
  // Sort: loaded first, then downloaded, then by name
  const rank = { loaded: 0, downloaded: 1, ready: 2, available: 3, provider_installed: 3 };
  const sorted = [...models].sort((a, b) => {
    // Loaded models first
    if (a.state === 'loaded' && b.state !== 'loaded') return -1;
    if (b.state === 'loaded' && a.state !== 'loaded') return 1;
    const sr = (rank[a.state] ?? 9) - (rank[b.state] ?? 9);
    if (sr !== 0) return sr;
    // US before GB
    const aUS = a.id.includes('en_US');
    const bUS = b.id.includes('en_US');
    if (aUS && !bUS) return -1;
    if (bUS && !aUS) return 1;
    // Alphabetical
    return (a.id || '').localeCompare(b.id || '');
  });
  const showAllKey = 'piperShowAll';
  const showAll = state[showAllKey];
  const visible = showAll ? sorted : sorted.slice(0, 5);
  const rows = visible.map((m) => {
    const p = parsePiperVoice(m.id);
    return `<tr>
      <td>${esc(p.name)}</td>
      <td><span class="quality-stars">${p.stars}</span><span class="quality-label">${esc(p.qualityLabel)}</span></td>
      <td>${esc(formatSize(m.size_mb))}</td>
      <td>${renderStatusDot(m)}</td>
      <td>${renderModelActions(m)}</td>
    </tr>`;
  }).join('');
  const showAllBtn = sorted.length > 5
    ? `<button class="provider-show-all" onclick="state.piperShowAll=!state.piperShowAll;renderModelsView()">
        ${showAll ? `Show Less` : `Showing 5 of ${sorted.length} voices — Show All`}
      </button>`
    : '';
  return `<div class="provider-card">
    <div class="provider-card-header">
      <h3><button class="provider-card-toggle" type="button" aria-expanded="true" onclick="toggleProviderCard(this)"><span class="chevron" aria-hidden="true">▼</span> Piper TTS</button></h3>
      <span class="provider-status ${status.cls}">${status.text}</span>
    </div>
    <div class="provider-card-body">
      <table class="provider-table">
        <thead><tr><th>Voice</th><th>Quality</th><th>Size</th><th>Status</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${showAllBtn}
    </div>
  </div>`;
}

function renderNotInstalledCard(providerName, displayName, description) {
  const cmd = `docker build --build-arg BAKED_PROVIDERS=kokoro,piper,${providerName} .`;
  return `<div class="provider-card">
    <div class="provider-card-header">
      <h3><button class="provider-card-toggle" type="button" aria-expanded="true" onclick="toggleProviderCard(this)"><span class="chevron" aria-hidden="true">▼</span> ${esc(displayName)}</button></h3>
      <span class="provider-status not-installed">Not Installed ✗</span>
    </div>
    <div class="provider-card-body install-card-body">
      <p>${esc(description)}</p>
      <p style="color:var(--text2);font-size:.85rem;margin-bottom:6px">To install, rebuild your image:</p>
      <div class="install-cmd" id="install-cmd-${esc(providerName)}">${esc(cmd)}</div>
      <button class="copy-cmd-btn" onclick="navigator.clipboard.writeText(document.getElementById('install-cmd-${esc(providerName)}').textContent);this.textContent='Copied!';setTimeout(()=>this.textContent='Copy Command',1500)">Copy Command</button>
    </div>
  </div>`;
}

const PROVIDER_DESCRIPTIONS = {
  'pocket-tts': 'CPU-first low-latency TTS with streaming support',
  'fish-speech': 'High-quality neural TTS with voice cloning',
  'f5-tts': 'F5 TTS — flow-matching text-to-speech',
  'xtts': 'XTTS v2 — multilingual TTS with voice cloning',
};

function renderGenericProviderCard(providerName, models) {
  const status = getProviderOverallStatus(models);
  const rows = models.map((m) => `<tr>
    <td>${esc(formatModelName(m))}</td>
    <td>${esc(formatSize(m.size_mb))}</td>
    <td>${renderStatusDot(m)}</td>
    <td>${renderModelActions(m)}</td>
  </tr>`).join('');
  return `<div class="provider-card">
    <div class="provider-card-header">
      <h3><button class="provider-card-toggle" type="button" aria-expanded="true" onclick="toggleProviderCard(this)"><span class="chevron" aria-hidden="true">▼</span> ${esc(PROVIDER_DISPLAY[providerName] || providerName)}</button></h3>
      <span class="provider-status ${status.cls}">${status.text}</span>
    </div>
    <div class="provider-card-body">
      <table class="provider-table">
        <thead><tr><th>Model</th><th>Size</th><th>Status</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  </div>`;
}

function renderSTTPanel(models) {
  const rank = { loaded: 0, downloaded: 1, ready: 2, available: 3, provider_installed: 3 };
  const sorted = [...models].sort((a, b) => (rank[a.state] ?? 9) - (rank[b.state] ?? 9) || (a.id || '').localeCompare(b.id || ''));
  // Detect default model (first loaded or first in list)
  const defaultModel = sorted.find((m) => m.state === 'loaded') || sorted[0];
  const showAll = state.sttShowAll;
  const visible = showAll ? sorted : sorted.slice(0, 5);
  const rows = visible.map((m) => {
    const shortName = stripSttPrefix(m.id);
    const isDefault = m.id === defaultModel?.id;
    const metaParts = [m.source, m.model_format].filter(Boolean);
    const metaLabel = metaParts.length ? ` <span class="stt-meta">${esc(metaParts.join(' · '))}</span>` : '';
    return `<tr>
      <td><span class="stt-short-name">${esc(shortName)}</span>${metaLabel}${isDefault ? '<span class="stt-default-badge">(default)</span>' : ''}</td>
      <td>${esc(formatSize(m.size_mb))}</td>
      <td>${renderStatusDot(m)}</td>
      <td>${renderModelActions(m)}</td>
    </tr>`;
  }).join('');
  const showAllBtn = sorted.length > 5
    ? `<button class="provider-show-all" onclick="state.sttShowAll=!state.sttShowAll;renderModelsView()">
        ${showAll ? `Show Less` : `Showing 5 of ${sorted.length} models — Show All`}
      </button>`
    : '';
  return `<div class="provider-card">
    <div class="provider-card-header">
      <h3><button class="provider-card-toggle" type="button" aria-expanded="true" onclick="toggleProviderCard(this)"><span class="chevron" aria-hidden="true">▼</span> faster-whisper</button></h3>
      <span class="provider-status ${getProviderOverallStatus(models).cls}">${getProviderOverallStatus(models).text}</span>
    </div>
    <div class="provider-card-body">
      <table class="provider-table">
        <thead><tr><th>Model</th><th>Size</th><th>Status</th><th>Action</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${showAllBtn}
    </div>
  </div>`;
}

function renderModelsView() {
  const models = state.modelsCache || [];
  const loaded = models.filter((m) => m.state === 'loaded');
  byId('loaded-count').textContent = `${loaded.length} models loaded`;

  // Group TTS models by provider
  const ttsModels = models.filter((m) => classifyKind(m) === 'tts');
  const sttModels = models.filter((m) => classifyKind(m) === 'stt');

  const providerGroups = {};
  const notInstalled = {};
  ttsModels.forEach((m) => {
    const provider = m.provider || providerFromModel(m.id);
    if (m.state === 'provider_missing' || m.provider_available === false) {
      if (!notInstalled[provider]) notInstalled[provider] = [];
      notInstalled[provider].push(m);
    } else {
      if (!providerGroups[provider]) providerGroups[provider] = [];
      providerGroups[provider].push(m);
    }
  });

  // Render TTS panel
  let ttsHtml = '';
  const providerOrder = ['kokoro', 'piper'];
  // Installed providers first in order
  for (const p of providerOrder) {
    if (providerGroups[p]) {
      if (p === 'kokoro') ttsHtml += renderKokoroCard(providerGroups[p]);
      else if (p === 'piper') ttsHtml += renderPiperCard(providerGroups[p]);
      else ttsHtml += renderGenericProviderCard(p, providerGroups[p]);
      delete providerGroups[p];
    }
  }
  // Remaining installed providers
  for (const [p, ms] of Object.entries(providerGroups)) {
    if (p === 'kokoro') ttsHtml += renderKokoroCard(ms);
    else if (p === 'piper') ttsHtml += renderPiperCard(ms);
    else ttsHtml += renderGenericProviderCard(p, ms);
  }
  // Not-installed providers as install cards
  for (const [p, ms] of Object.entries(notInstalled)) {
    if (providerGroups[p]) continue; // skip if also has installed models
    const desc = PROVIDER_DESCRIPTIONS[p] || `${PROVIDER_DISPLAY[p] || p} TTS provider`;
    ttsHtml += renderNotInstalledCard(p, PROVIDER_DISPLAY[p] || p, desc);
  }
  if (!ttsHtml) ttsHtml = '<p class="legend">No TTS models available.</p>';

  const ttsPanel = byId('models-tts-panel');
  if (ttsPanel) ttsPanel.innerHTML = ttsHtml;

  // Render STT panel
  const sttPanel = byId('models-stt-panel');
  if (sttPanel) {
    sttPanel.innerHTML = sttModels.length ? renderSTTPanel(sttModels) : '<p class="legend">No STT models available.</p>';
  }
}
async function refreshModels({ silent = false } = {}) {
  if (state.modelsBusy) return;
  state.modelsBusy = true;
  try {
    const data = await api('/api/models');
    state.modelsCache = data.models || [];
    state.defaultSttModel = data.default_stt_model || state.defaultSttModel || '';
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
  const opLabel = kind === 'loading' ? 'Loading…' : 'Downloading…';
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
    let pollCount = 0;
    while (true) {
      await new Promise((r) => setTimeout(r, 3000));
      const status = await api(`/api/models/${encodeURIComponent(modelId)}/status`);
      const nextText = status.progress ? `${opLabel} ${status.progress}` : opLabel;
      state.modelOps[modelId] = { kind, text: nextText };
      const idx = state.modelsCache.findIndex((m) => m.id === modelId);
      if (idx >= 0) state.modelsCache[idx] = { ...state.modelsCache[idx], ...status };
      renderModelsView();

      // Don't break on queued/downloading/loading — operation is in progress
      if (status.state === 'queued' || status.state === 'downloading' || status.state === 'loading') {
        pollCount = 0; // reset timeout while actively making progress
        continue;
      }

      // Break on successful terminal states
      if ((kind === 'loading' || kind === 'load') && status.state === 'loaded') break;
      if ((kind === 'downloading' || kind === 'prefetch') && (status.state === 'downloaded' || status.state === 'loaded')) break;

      // Break on terminal failure states (model reverted or provider missing)
      if (status.state === 'available' || status.state === 'provider_missing') {
        throw new Error(
          status.state === 'provider_missing'
            ? 'Provider not installed — rebuild image with this provider baked'
            : 'Model reverted to available — load failed',
        );
      }

      // Safety timeout: max 3 minutes of polling (60 * 3s intervals)
      if (++pollCount > 60) {
        throw new Error('Timed out waiting for model operation');
      }
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
function toggleProviderCard(button) {
  const header = button.closest('.provider-card-header');
  if (!header) return;
  const collapsed = header.classList.toggle('collapsed');
  button.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
}
function bindEvents() {
  document.addEventListener('visibilitychange', () => {
    handleLiveReaderVisibilityChange().catch(() => {});
  });
  byId('tts-input').addEventListener('input', (e) => {
    byId('tts-counter').textContent = `${e.target.value.length.toLocaleString()} characters`;
    handleLiveReaderTextChange(e.target.value);
  });
  byId('tts-speed').addEventListener('input', (e) => {
    byId('tts-speed-value').textContent = `${Number(e.target.value).toFixed(1)}x`;
  });
  byId('tts-provider')?.addEventListener('change', () => {
    if (state.liveReader) stopLiveReader('Voice changed').catch(() => {});
    state.ttsPreferredProvider = byId('tts-provider').value;
    state.ttsPreferredModel = '';
    loadTTSModels().catch((err) => showToast(err.message, 'error'));
  });
  byId('tts-model').addEventListener('change', () => {
    if (state.liveReader) stopLiveReader('Voice changed').catch(() => {});
    loadTTSVoices().catch((err) => showToast(err.message, 'error'));
  });
  byId('tts-voice')?.addEventListener('change', () => {
    if (state.liveReader) stopLiveReader('Voice changed').catch(() => {});
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
  byId('live-reader-start').addEventListener('click', () => startLiveReader());
  byId('live-reader-read-all').addEventListener('click', () => startLiveReader({ readAll: true }));
  byId('live-reader-pause').addEventListener('click', () => toggleLiveReaderPause().catch((error) => showToast(error.message, 'error')));
  byId('live-reader-stop').addEventListener('click', () => stopLiveReader());
  byId('live-reader-clear').addEventListener('click', async () => {
    await stopLiveReader();
    byId('tts-input').value = '';
    byId('tts-counter').textContent = '0 characters';
  });
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
    if (state.liveReader) await stopLiveReader('Text changed');
    byId('tts-input').value = await file.text();
    byId('tts-counter').textContent = `${byId('tts-input').value.length.toLocaleString()} characters`;
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
  byId('mic-select')?.addEventListener('change', (e) => saveMicDeviceId(e.target.value));
  byId('mic-btn').addEventListener('click', toggleMic);
  if (navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', () => loadMicDevices().catch(() => {}));
  } else if (navigator.mediaDevices && 'ondevicechange' in navigator.mediaDevices) {
    navigator.mediaDevices.ondevicechange = () => loadMicDevices().catch(() => {});
  }
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
  document.querySelectorAll('.models-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.models-tab').forEach((t) => {
        const active = t === tab;
        t.classList.toggle('active', active);
        t.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      const which = tab.dataset.modelsTab;
      const ttsPanel = byId('models-tts-panel');
      const sttPanel = byId('models-stt-panel');
      if (ttsPanel) {
        const active = which === 'tts';
        ttsPanel.classList.toggle('active', active);
        ttsPanel.hidden = !active;
      }
      if (sttPanel) {
        const active = which === 'stt';
        sttPanel.classList.toggle('active', active);
        sttPanel.hidden = !active;
      }
    });
  });
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

  setTTSSpeed(profile.speed || 1.0);
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
  if (state.liveReader) await stopLiveReader('Text changed');
  byId('tts-input').value = entry.full_text || '';
  byId('tts-counter').textContent = `${byId('tts-input').value.length.toLocaleString()} characters`;

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

  setTTSSpeed(entry.speed || 1.0);
  if (entry.format) byId('tts-format').value = entry.format;
  byId('tts-preset').value = '';
  document.querySelector('.tab[data-tab="speak"]').click();
}

async function init() {
  initTheme();
  initTabs();
  bindEvents();
  byId('tts-provider').innerHTML = '<option>Loading…</option>';
  byId('tts-model').innerHTML = '<option>Loading…</option>';
  byId('stt-model').innerHTML = '<option>Loading…</option>';
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
    state.defaultSttModel = data.default_stt_model || '';
  } catch (e) {
    // non-fatal — cache stays empty, renderModelsView shows empty state
  }
  renderModelsView();

  // Step 2: parallel init for everything else (use cached models, don't re-fetch)
  await Promise.allSettled([
    loadTTSProviders(),
    loadSTTModels(),
    loadProfiles(),
    loadMicDevices(),
  ]);

  // Step 3: non-critical background loaders (don't block UI)
  Promise.allSettled([loadConversations(), loadComposerHistory()]).catch(() => {});

  if (!composerTracks.length) addComposerTrack();
}
/* ── Waveform Visualizer ── */
const waveform = {
  playbackCtx: null,
  playbackAnalyser: null,
  playbackSource: null,
  playbackRaf: null,
  micAnalyser: null,
  micRaf: null,
};

function drawWaveform(canvas, analyser, color) {
  const ctx = canvas.getContext('2d');
  const bufLen = analyser.frequencyBinCount;
  const data = new Uint8Array(bufLen);
  const draw = () => {
    analyser.getByteTimeDomainData(data);
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.beginPath();
    const step = w / bufLen;
    for (let i = 0; i < bufLen; i++) {
      const y = (data[i] / 255) * h;
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * step, y);
    }
    ctx.stroke();
  };
  return draw;
}

function startPlaybackWaveform() {
  const audio = byId('tts-audio');
  const canvas = byId('tts-waveform');
  if (!canvas || !audio.src) return;
  // Resize canvas to actual pixel size
  canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);

  if (!waveform.playbackCtx || waveform.playbackCtx.state === 'closed') {
    waveform.playbackCtx = new AudioContext();
    waveform.playbackSource = null; // invalidate old source for new context
  }
  if (waveform.playbackCtx.state === 'suspended') waveform.playbackCtx.resume();

  if (!waveform.playbackSource || waveform.playbackSource.context !== waveform.playbackCtx) {
    if (waveform.playbackSource) { try { waveform.playbackSource.disconnect(); } catch {} }
    waveform.playbackSource = waveform.playbackCtx.createMediaElementSource(audio);
  }
  if (!waveform.playbackAnalyser || waveform.playbackAnalyser.context !== waveform.playbackCtx) {
    waveform.playbackAnalyser = waveform.playbackCtx.createAnalyser();
    waveform.playbackAnalyser.fftSize = 1024;
  }
  try { waveform.playbackSource.disconnect(); } catch {}
  waveform.playbackSource.connect(waveform.playbackAnalyser);
  waveform.playbackAnalyser.connect(waveform.playbackCtx.destination);

  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const drawFn = drawWaveform(canvas, waveform.playbackAnalyser, accent || '#6366f1');
  const loop = () => { drawFn(); waveform.playbackRaf = requestAnimationFrame(loop); };
  if (waveform.playbackRaf) cancelAnimationFrame(waveform.playbackRaf);
  loop();

  audio.addEventListener('ended', stopPlaybackWaveform);
  audio.addEventListener('pause', stopPlaybackWaveform);
}

function stopPlaybackWaveform() {
  if (waveform.playbackRaf) { cancelAnimationFrame(waveform.playbackRaf); waveform.playbackRaf = null; }
}

function updatePlaybackTime() {
  const audio = byId('tts-audio');
  const el = byId('tts-time');
  if (!audio || !el) return;
  const fmt = (s) => { const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`; };
  el.textContent = `${fmt(audio.currentTime || 0)} / ${fmt(audio.duration || 0)}`;
}

function initPlaybackControls() {
  const audio = byId('tts-audio');
  const playBtn = byId('tts-play-btn');
  if (!audio || !playBtn) return;
  playBtn.addEventListener('click', () => {
    if (audio.paused) {
      audio.play().then(() => startPlaybackWaveform()).catch(() => {});
      playBtn.textContent = '⏸';
    } else {
      audio.pause();
      playBtn.textContent = '▶';
    }
  });
  audio.addEventListener('play', () => { playBtn.textContent = '⏸'; startPlaybackWaveform(); });
  audio.addEventListener('pause', () => playBtn.textContent = '▶');
  audio.addEventListener('ended', () => playBtn.textContent = '▶');
  audio.addEventListener('timeupdate', updatePlaybackTime);
  audio.addEventListener('loadedmetadata', updatePlaybackTime);
}

function startMicWaveform() {
  const canvas = byId('mic-waveform');
  if (!canvas || !state.audioCtx || !state.audioSource) return;
  canvas.hidden = false;
  canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
  canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
  waveform.micAnalyser = state.audioCtx.createAnalyser();
  waveform.micAnalyser.fftSize = 512;
  state.audioSource.connect(waveform.micAnalyser);
  const drawFn = drawWaveform(canvas, waveform.micAnalyser, '#f59e0b');
  const loop = () => { drawFn(); waveform.micRaf = requestAnimationFrame(loop); };
  loop();
}

function stopMicWaveform() {
  if (waveform.micRaf) { cancelAnimationFrame(waveform.micRaf); waveform.micRaf = null; }
  const canvas = byId('mic-waveform');
  if (canvas) { canvas.hidden = true; canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height); }
}

document.addEventListener('DOMContentLoaded', () => {
  init().then(() => initPlaybackControls()).catch((e) => showToast(`Init failed: ${e.message}`, 'error'));
});
window.addEventListener('beforeunload', () => {
  const reader = state.liveReader;
  if (reader?.ws?.readyState < WebSocket.CLOSING) reader.ws.close();
});
