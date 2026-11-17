/**
 * Model inventory, labels, the state vocabulary, and loading (ensureReady, runOp), ported from the old UI
 * with the same semantics.
 * @module core/models
 */
import { get as apiGet, request } from './api.js';
import { getKey } from './auth.js';
import { app } from './store.js';
import { confirm } from './dialogs.js';
import { beginJob } from './status.js';
import { toast } from './toast.js';

export const PROVIDER_DISPLAY = Object.freeze({
  kokoro: 'Kokoro', piper: 'Piper', 'pocket-tts': 'Pocket TTS', supertonic: 'Supertonic', qwen3: 'Qwen3 TTS', chatterbox: 'Chatterbox',
  fish: 'Fish Audio S2', breeze: 'Breeze TTS 2', 'f5-tts': 'F5 TTS', xtts: 'XTTS v2',
  'faster-whisper': 'faster-whisper', parakeet: 'Parakeet',
});

export const PROVIDER_DESCRIPTIONS = Object.freeze({
  kokoro: 'Fast, natural speech with weighted voice blends',
  piper: 'Lightweight local voices in many languages, fast on a CPU',
  'pocket-tts': 'CPU-first low-latency TTS with streaming support',
  supertonic: 'Fast ONNX voices on a GPU or CPU, with voice blends and quality steps',
  qwen3: 'Qwen3 voices, voice design and voice cloning',
  chatterbox: 'English voice cloning with regular and Turbo models',
  fish: 'Fish Audio S2 Pro voice cloning from a recording and its transcript (NF4)',
  breeze: 'Breeze TTS 2 voices designed from a description, and cloning from a recording and its transcript',
  'f5-tts': 'F5 TTS — flow-matching text-to-speech',
  xtts: 'XTTS v2 — multilingual TTS with voice cloning',
  'faster-whisper': 'Speech recognition with Whisper models (CTranslate2)',
  parakeet: 'Fast English speech recognition with NVIDIA Parakeet (ONNX)',
});

/** Providers built into the image; everything else runs as an isolated GPU worker. */
export const BAKED_PROVIDERS = Object.freeze(['kokoro', 'piper', 'pocket-tts', 'supertonic']);

const FAST_PATH_MS = 20000;
const POLL_MS = 2000;
const NO_PROGRESS_TIMEOUT_MS = 180000;
const REQUEST_TIMEOUT_MS = 15000;   // /api/models and the status checks: a stalled request never blocks callers
const IN_FLIGHT = new Set(['queued', 'downloading', 'loading']);

let inflightRefresh = null;    // {mark, key, promise}: the inventory request now in flight
let lastOk = new Map();        // modelId -> ms of the last successful use or status check
let fastPathOff = new Set();   // models whose last request failed
let opPromises = new Map();    // modelId -> running runOp promise
let readyPromises = new Map(); // modelId -> running ensureReady promise
// Freshness: every row merged from a status check gets a mark. An inventory answer keeps the rows marked
// after its request started (it may predate them), and an older answer never replaces a newer one.
let mark = 0;
let rowMarks = new Map();      // modelId -> mark of its last merge
let fetchSeq = 0;
let appliedSeq = 0;

const enc = (id) => encodeURIComponent(String(id));

// The server reports phases, not measured progress: 0 queued, 0.1 downloading, 0.5 loading, 1 done
// (src/services/models.py). Showing those as "10%" would be invented, so only another value counts.
const PHASE_MARKERS = new Set([0, 0.1, 0.5, 1]);

/** A measured progress fraction (0 < p < 1), or null when the server only reported a phase. */
export function knownProgress(value) {
  return typeof value === 'number' && value > 0 && value < 1 && !PHASE_MARKERS.has(value) ? value : null;
}

/** 'stt' or 'tts': model.type, else id heuristics (whisper/parakeet/vosk/moonshine are STT). */
export function classifyKind(model) {
  if (model && model.type) return model.type;
  const id = String((model && model.id) || '');
  if (id.includes('whisper') || id.includes('parakeet') || id.includes('vosk') || id.includes('moonshine')) return 'stt';
  return 'tts';
}

/** 'piper/en_US-amy-medium' -> 'piper'; 'kokoro' -> 'kokoro'; '' -> 'kokoro' (today's rule). */
export function providerOf(modelId) {
  const id = String(modelId || '');
  if (!id) return 'kokoro';
  if (id.includes('/')) return id.split('/')[0];
  if (id.startsWith('kokoro-')) return 'kokoro';
  return id;
}

function formatModelName(model) {
  const id = String(model?.id || '');
  const provider = model?.provider || providerOf(id);
  if (!id) return '—';
  if (id.includes('/')) return id.split('/').slice(1).join('/');
  if (provider && id.startsWith(`${provider}-`)) return `${id.slice(provider.length + 1)} (${provider})`;
  return id;
}

/** Provider display name plus variant ('Chatterbox turbo'); an unknown provider gives the model name. */
export function modelLabel(modelId) {
  const id = String(modelId || '');
  const name = PROVIDER_DISPLAY[providerOf(id)];
  if (!name) return formatModelName({ id });
  const variant = id.includes('/') ? id.split('/').slice(1).join('/') : '';
  // 'breeze/tts-2' is 'Breeze TTS 2', not 'Breeze TTS 2 tts-2'.
  if (!variant || name.toLowerCase().replace(/ /g, '-').endsWith(variant.toLowerCase())) return name;
  return `${name} ${variant}`;
}

/** Piper voice ids: 'piper/en_US-amy-medium' -> {name: 'Amy (US)', quality: 'medium', qualityLabel}. */
export function parsePiperVoice(modelId) {
  const id = String(modelId || '').replace(/^piper\//, '');
  const m = id.match(/^([a-z]{2})_([A-Z]{2})-(.+)-(low|medium|high|x_low)$/);
  if (m) {
    const words = m[3].split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    const quality = m[4];
    const qualityLabel = quality === 'x_low' ? 'Extra low' : quality.charAt(0).toUpperCase() + quality.slice(1);
    return { name: `${words} (${m[2]})`, lang: `${m[1]}_${m[2]}`, quality, qualityLabel };
  }
  return { name: id, lang: '', quality: 'medium', qualityLabel: 'Medium' };
}

function stripSttPrefix(modelId) {
  return String(modelId || '')
    .replace(/^nvidia\/parakeet-/, 'parakeet-')
    .replace(/^Systran\/faster-distil-whisper-/, 'distil-')
    .replace(/^Systran\/faster-whisper-/, '')
    .replace(/^deepdml\/faster-whisper-/, '')
    .replace(/-ct2$/, '');
}

/** The short name in a select: STT 'base'; Piper 'Amy (US)'; Kokoro 'Kokoro'; others the variant. */
export function shortName(model) {
  const m = typeof model === 'string' ? { id: model } : (model || {});
  const id = String(m.id || '');
  const provider = m.provider || providerOf(id);
  if (classifyKind(m) === 'stt') return stripSttPrefix(id);
  if (provider === 'piper') return parsePiperVoice(id).name;
  if (!id.includes('/') && PROVIDER_DISPLAY[provider] && id === provider) return PROVIDER_DISPLAY[provider];
  return formatModelName(m);
}

/** An STT model as people say it: 'whisper base', 'whisper large-v3-turbo', 'parakeet-tdt-0.6b-v2'. */
export function sttLabel(model) {
  const m = typeof model === 'string' ? { id: model } : (model || {});
  const id = String(m.id || '');
  if (!id) return '';
  const short = stripSttPrefix(id);
  if (m.provider === 'parakeet' || /parakeet/i.test(short)) return short;
  return `whisper ${short}`;
}

/**
 * An STT model named with its family, for a list that mixes them: 'Whisper base', 'Whisper large-v3-turbo',
 * 'Distil-Whisper small.en', 'Parakeet TDT 0.6B v2'; any other model its short name.
 */
export function sttName(model) {
  const m = typeof model === 'string' ? { id: model } : (model || {});
  const id = String(m.id || '');
  const short = stripSttPrefix(id);
  if (m.provider === 'parakeet' || /^parakeet-/i.test(short)) {
    return short.split('-').map((word, i) => {
      if (i === 0) return 'Parakeet';
      return /^(tdt|ctc|rnnt)$/i.test(word) || /^\d+(\.\d+)?[bm]$/i.test(word) ? word.toUpperCase() : word;
    }).join(' ');
  }
  if (/^distil-/.test(short)) return `Distil-Whisper ${short.slice('distil-'.length)}`;
  if ((m.provider === 'faster-whisper' || /faster-whisper/.test(id)) && !short.includes('/')) return `Whisper ${short}`;
  return short;
}

/** 'cuda…' or 'mps' -> 'GPU', 'cpu' -> 'CPU', else ''. */
export function deviceLabel(device) {
  const d = String(device || '').toLowerCase();
  if (d.startsWith('cuda') || d === 'mps' || d === 'gpu') return 'GPU';
  if (d === 'cpu') return 'CPU';
  return '';
}

const providerRank = (p) => ({ kokoro: 0, piper: 1 }[p] ?? 2);

/**
 * Usable TTS models: not provider_missing and provider_available !== false. Ordered Kokoro, Piper, then the
 * other providers alphabetically; the server's order within a provider is kept.
 */
export function ttsModels(list) {
  const usable = (list || []).filter((m) => classifyKind(m) === 'tts' && m.state !== 'provider_missing' && m.provider_available !== false);
  return usable
    .map((m, i) => ({ m, i, p: m.provider || providerOf(m.id) }))
    .sort((a, b) => providerRank(a.p) - providerRank(b.p) || (providerRank(a.p) === 2 ? a.p.localeCompare(b.p) : 0) || a.i - b.i)
    .map(({ m }) => m);
}

/**
 * A model that only designs voices (voice_design, no voices of its own, no cloning): every request makes a new voice
 * from its description. Takes a model row or a capabilities object.
 */
export function designOnly(rowOrCaps) {
  const raw = rowOrCaps && typeof rowOrCaps === 'object' && 'capabilities' in rowOrCaps ? rowOrCaps.capabilities : rowOrCaps;
  const c = raw && typeof raw === 'object' && !Array.isArray(raw) ? raw : {};
  return c.voice_design === true && c.voice_clone !== true && !(Array.isArray(c.speakers) && c.speakers.length);
}

/** The usable TTS models a view that speaks many requests in one voice can offer (the Reader, Studio, Talk). */
export function steadyTtsModels(list) {
  return ttsModels(list).filter((m) => !designOnly(m));
}

/** The model select value of the Piper voice models: each Piper voice is a model of its own, listed as one entry. */
export const PIPER_GROUP = 'piper';

/** PURE: what a model select holds for a model: PIPER_GROUP for a Piper voice model, the id itself for any other. */
export function selectValue(modelId) {
  const id = String(modelId || '');
  return id.includes('/') && providerOf(id) === 'piper' ? PIPER_GROUP : id;
}

/** PURE: whether a model select value stands for the Piper voice models rather than one model. */
export function isGroupValue(value) {
  return value === PIPER_GROUP;
}

/**
 * PURE: a model select's entries for `rows` (models in the order they are offered): one per model, except the Piper
 * voice models, which are one 'Piper' entry where the first of them is.
 * @param {object[]} rows
 * @returns {{id: string, provider: string, label: string, models: string[]}[]} `models`: the ids behind the entry
 */
export function selectEntries(rows) {
  const out = [];
  for (const m of rows || []) {
    const id = String(m?.id || '');
    if (!id) continue;
    const value = selectValue(id);
    const known = out.find((entry) => entry.id === value);
    if (known) {
      known.models.push(id);
      continue;
    }
    const provider = m.provider || providerOf(id);
    out.push({ id: value, provider, label: isGroupValue(value) ? PROVIDER_DISPLAY.piper : shortName(m), models: [id] });
  }
  return out;
}

/** PURE: the ids of `rows` a model select value stands for: the Piper voice models for PIPER_GROUP, else the one model. */
export function groupIds(value, rows) {
  return (rows || []).map((m) => String(m?.id || '')).filter((id) => id && selectValue(id) === value);
}

/**
 * PURE: the model to use when a model select value is chosen: the value itself for one model; for the Piper entry
 * the first of `preferred` it offers, else a loaded Piper voice, else a downloaded one, else the first.
 * @param {string} value
 * @param {object[]} rows the models offered (ttsModels() or steadyTtsModels())
 * @param {{preferred?: string[]}} [options] `preferred`: ids in order (the voice used last, the configured default)
 * @returns {string} '' when the entry has no model in `rows`
 */
export function modelForValue(value, rows, { preferred = [] } = {}) {
  if (!isGroupValue(value)) return String(value || '');
  const ids = groupIds(value, rows);
  const offered = (rows || []).filter((m) => ids.includes(m.id));
  const wanted = (preferred || []).find((id) => id && ids.includes(id));
  if (wanted) return wanted;
  const inState = (state) => offered.find((m) => m.state === state);
  return (inState('loaded') || inState('downloaded') || offered[0] || {}).id || '';
}

/** The hint where design-only models are left out. */
export const DESIGN_ONLY_HINT = 'Models that only design voices make a new one each time, so they aren’t listed. Make a voice to keep in Voices › Design a voice.';

const STT_RANK = { loaded: 0, downloaded: 1, ready: 1, provider_installed: 2, available: 2, provider_missing: 3 };

const STT_PROVIDERS = new Set(['faster-whisper', 'parakeet']);

/** faster-whisper and Parakeet models, sorted loaded > downloaded > available > missing, then id. */
export function sttModels(list) {
  return (list || [])
    .filter((m) => classifyKind(m) === 'stt' && (STT_PROVIDERS.has(m.provider) || /faster-whisper|parakeet/.test(String(m.id || ''))))
    .sort((a, b) => ((STT_RANK[a.state] ?? 9) - (STT_RANK[b.state] ?? 9)) || String(a.id || '').localeCompare(String(b.id || '')));
}

/**
 * The GPU workers docker-compose.voice-models.yml has a service for, each under a Compose profile named after it.
 * Layering the file alone starts none of them (docs/CONFIGURATION.md), so a command names the profile.
 */
const COMPOSE_WORKERS = Object.freeze(['chatterbox', 'qwen3', 'fish', 'breeze']);

/**
 * The install hint for a missing provider: the image rebuild for a baked provider, or the command that turns on a
 * worker's own Compose profile. A worker provider the Compose file has no service for gets no command (command ''):
 * there is no profile to name, and no command could start it.
 */
export function installHint(provider) {
  const p = String(provider || '');
  if (BAKED_PROVIDERS.includes(p)) {
    return {
      kind: 'rebuild',
      text: 'To install, rebuild your image:',
      command: `docker build --build-arg BAKED_PROVIDERS=${BAKED_PROVIDERS.join(',')} .`,
    };
  }
  const name = PROVIDER_DISPLAY[p] || p;
  if (!COMPOSE_WORKERS.includes(p)) {
    return { kind: 'worker', text: `${name} runs as an isolated voice worker, and docker-compose.voice-models.yml has none for it.`, command: '' };
  }
  return {
    kind: 'worker',
    text: `${name} runs as an isolated voice worker. Turn it on with its Compose profile:`,
    command: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile ${p} up -d --build`,
  };
}

function missingMessage(modelId) {
  const provider = providerOf(modelId);
  if (BAKED_PROVIDERS.includes(provider)) return `Provider not installed — rebuild the image with ${provider} in BAKED_PROVIDERS`;
  return COMPOSE_WORKERS.includes(provider)
    ? `The ${provider} voice worker is not configured — turn it on with docker-compose.voice-models.yml and --profile ${provider}`
    : `The ${provider} voice worker is not configured — docker-compose.voice-models.yml has none for it`;
}

/**
 * A loaded model whose GPU worker stopped answering: the server keeps state 'loaded' and only sets
 * provider_available false (src/model_manager.py list_all), so it is an offline worker, never "not
 * configured" (MD-02). Baked providers run in the server process and have no worker.
 */
function workerWentOffline(m) {
  return m.state === 'loaded' && m.provider_available === false && !BAKED_PROVIDERS.includes(m.provider || providerOf(m.id));
}

/**
 * The reason a configured worker gave for not running, or '' when the row has none. A worker that answers its
 * /health with status "failed" reaches the row as provider_failure {code, message} (src/model_manager.py):
 * the worker's own text, one printable line of at most 300 characters.
 * Without it (null: the worker doesn't answer) the worker is offline. Plain text: shown only as text.
 */
export function workerFailure(model) {
  const failure = model && model.provider_failure;
  const message = failure && typeof failure === 'object' && typeof failure.message === 'string' ? failure.message : '';
  return message.replace(/\s+/g, ' ').trim();
}

/** "The voice worker can't run: <reason>." (a reason that ends its own sentence keeps its stop). */
function cantRunText(reason) {
  return `The voice worker can't run: ${/[.!?…]$/.test(reason) ? reason : `${reason}.`}`;
}

/**
 * PURE: the single source of the model state vocabulary.
 * @param {object} model a /api/models row
 * @param {object|null} [op] app.ops[model.id]
 * @param {{list?: object[]}} [options] the inventory, for the "Unloads Kokoro" note
 * @returns {{key: string, label: string, tone: 'ok'|'idle'|'busy'|'warn'|'error',
 *   action: null|{kind: 'unload'|'load'|'download'|'copy'|'retry', label: string, op?: string, command?: string},
 *   overflow: string[], busy: boolean, progress: number|null, elapsedFrom: number, note: string,
 *   hint: null|{kind: string, text: string, command?: string}, error: string, signature: string}}
 */
export function uiState(model, op = null, { list = [] } = {}) {
  const m = model || {};
  const state = m.state || 'available';
  const base = { busy: false, progress: null, elapsedFrom: 0, note: '', hint: null, error: '', overflow: [] };
  let out;
  if (op && op.error) {
    const verb = op.kind === 'load' ? 'load' : op.kind === 'unload' ? 'unload' : op.kind === 'delete' ? 'delete' : 'download';
    out = { ...base, key: 'error', label: `Couldn't ${verb}`, tone: 'error', action: { kind: 'retry', label: 'Retry', op: op.kind }, error: String(op.error) };
  } else if ((op && op.kind && !op.error) || IN_FLIGHT.has(state)) {
    const kind = op?.kind;
    const progress = knownProgress(op?.progress) ?? (IN_FLIGHT.has(state) ? knownProgress(m.progress) : null);
    const startedAt = op?.startedAt || 0;
    if (state === 'queued' || op?.phase === 'queued') {
      out = { ...base, key: 'queued', label: 'Queued', tone: 'busy', action: null, busy: true, elapsedFrom: startedAt };
    } else if (kind === 'unload') {
      out = { ...base, key: 'unloading', label: 'Unloading', tone: 'busy', action: null, busy: true, elapsedFrom: startedAt };
    } else if (kind === 'delete') {
      out = { ...base, key: 'deleting', label: 'Deleting files', tone: 'busy', action: null, busy: true, elapsedFrom: startedAt };
    } else if (kind === 'download' || kind === 'prefetch' || state === 'downloading') {
      const pct = progress !== null ? ` · ${Math.round(progress * 100)}%` : '';
      out = { ...base, key: 'downloading', label: `Downloading${pct}`, tone: 'busy', action: null, busy: true, progress, elapsedFrom: startedAt };
    } else {
      out = { ...base, key: 'loading', label: 'Loading', tone: 'busy', action: null, busy: true, elapsedFrom: startedAt };
    }
  } else if (state === 'provider_unavailable' || workerWentOffline(m)) {
    // A worker that answers but reports itself failed can't run, which is not offline.
    const reason = workerFailure(m);
    out = reason
      ? { ...base, key: 'worker-failed', label: "Worker can't run", tone: 'error', action: null, hint: { kind: 'failed', text: cantRunText(reason) } }
      : { ...base, key: 'worker-offline', label: 'Worker offline', tone: 'error', action: null, hint: { kind: 'offline', text: "The voice worker isn't responding. Check the worker service." } };
  } else if (state === 'provider_missing' || m.provider_available === false) {
    const provider = m.provider || providerOf(m.id);
    const hint = installHint(provider);
    out = {
      ...base,
      key: hint.kind === 'rebuild' ? 'not-installed' : 'worker-missing',
      label: hint.kind === 'rebuild' ? 'Not installed' : 'Worker not configured',
      tone: 'warn',
      action: hint.command ? { kind: 'copy', label: 'Copy command', command: hint.command } : null,
      hint,
    };
  } else if (state === 'loaded') {
    const device = deviceLabel(m.device);
    out = { ...base, key: 'loaded', label: device ? `Loaded · ${device}` : 'Loaded', tone: 'ok', action: { kind: 'unload', label: 'Unload' }, overflow: ['delete'] };
  } else if (state === 'downloaded' || state === 'ready') {
    out = {
      ...base, key: 'downloaded', label: 'Downloaded', tone: 'idle', action: { kind: 'load', label: 'Load' }, overflow: ['delete'],
      note: m.notice || unloadsNote(m.id, list),
    };
  } else if (m.model_format === 'external') {
    out = { ...base, key: 'on-first-load', label: 'Downloads on first load', tone: 'idle', action: { kind: 'load', label: 'Load' }, note: unloadsNote(m.id, list) };
  } else {
    out = { ...base, key: 'not-downloaded', label: 'Not downloaded', tone: 'idle', action: { kind: 'download', label: 'Download', op: 'prefetch' } };
  }
  out.signature = [out.key, out.label, out.tone, out.action ? out.action.kind : '', out.progress ?? '', out.note, out.error].join('|');
  return out;
}

/** The other loaded TTS models, for a TTS `modelId`; empty for STT models. */
function otherLoadedTtsAll(modelId, list) {
  const models = list || [];
  const model = models.find((m) => m.id === modelId) || { id: modelId };
  if (classifyKind(model) !== 'tts') return [];
  return models.filter((m) => classifyKind(m) === 'tts' && m.state === 'loaded' && m.id !== modelId);
}

/**
 * The loaded TTS models that loading `modelId` would unload: an explicit load unloads every other loaded
 * speech model except the pinned default (docs/MODELS.md "Model residency"), and a Reader, Live or API request can
 * leave more than one loaded. Empty for STT models or when nothing else is loaded.
 */
export function otherLoadedTtsModels(modelId, list) {
  return otherLoadedTtsAll(modelId, list).filter((m) => m.pinned !== true);
}

/** The first of otherLoadedTtsModels(), or null. */
export function otherLoadedTts(modelId, list) {
  return otherLoadedTtsModels(modelId, list)[0] || null;
}

/** Names for one sentence: "Kokoro", "Kokoro and Piper", "Kokoro, Piper and Qwen3". */
export function joinNames(names) {
  const all = names.filter(Boolean);
  return all.length > 1 ? `${all.slice(0, -1).join(', ')} and ${all[all.length - 1]}` : (all[0] || '');
}

/**
 * What loading an STT model does to the other loaded STT models. The web app loads STT models with
 * keep_pinned, so a pinned default (the row's `pinned`, OS_PIN_DEFAULT_MODELS) stays loaded next to it; every
 * other loaded STT model is unloaded. Unloading the default itself is its own row's Unload.
 * @returns {{unloads: object[], stays: object[]}} both empty for a TTS model
 */
export function otherLoadedSttModels(modelId, list) {
  const models = list || [];
  const model = models.find((m) => m.id === modelId) || { id: modelId };
  if (classifyKind(model) !== 'stt') return { unloads: [], stays: [] };
  const others = models.filter((m) => classifyKind(m) === 'stt' && m.state === 'loaded' && m.id !== modelId);
  return { unloads: others.filter((m) => m.pinned !== true), stays: others.filter((m) => m.pinned === true) };
}

/** The models loading `modelId` would unload: the other loaded TTS models, or the unpinned loaded STT ones. */
function unloadedBy(modelId, list) {
  const row = (list || []).find((m) => m.id === modelId) || { id: modelId };
  return classifyKind(row) === 'stt' ? otherLoadedSttModels(modelId, list).unloads : otherLoadedTtsModels(modelId, list);
}

/**
 * A model row's note: "Unloads Pocket" (or every model a load would unload), then the pinned default it loads
 * next to ("Unloads Whisper small · Parakeet TDT 0.6B v2 stays loaded", "Kokoro stays loaded"); '' when there
 * is nothing to say.
 */
function unloadsNote(modelId, list) {
  const row = (list || []).find((m) => m.id === modelId) || { id: modelId };
  const stt = classifyKind(row) === 'stt';
  const { unloads, stays } = stt ? otherLoadedSttModels(modelId, list) : {
    unloads: otherLoadedTtsModels(modelId, list),
    stays: otherLoadedTtsAll(modelId, list).filter((m) => m.pinned === true),
  };
  const name = stt ? sttName : shortName;
  const parts = [];
  if (unloads.length) parts.push(`Unloads ${joinNames(unloads.map((o) => name(o)))}`);
  if (stays.length) parts.push(`${joinNames(stays.map((o) => name(o)))} stays loaded`);
  return parts.join(' · ');
}

/**
 * Pick the TTS model: the user's preferred > the configured default > Kokoro > a loaded one > a downloaded
 * one > the first. A loaded canary never beats the configured default until the user picks it (SP-02).
 * @param {object[]} models usable TTS models (ttsModels())
 * @param {{preferred?: string, configuredDefault?: string}} [options]
 * @returns {string} '' when there is none
 */
export function pickTtsModel(models, { preferred = '', configuredDefault = '' } = {}) {
  const list = models || [];
  const has = (id) => !!id && list.some((m) => m.id === id);
  if (has(preferred)) return preferred;
  if (has(configuredDefault)) return configuredDefault;
  const kokoro = list.find((m) => (m.provider || providerOf(m.id)) === 'kokoro');
  if (kokoro) return kokoro.id;
  const loaded = list.find((m) => m.state === 'loaded');
  if (loaded) return loaded.id;
  const downloaded = list.find((m) => m.state === 'downloaded' || m.state === 'ready');
  if (downloaded) return downloaded.id;
  return list[0]?.id || '';
}

/**
 * GET /api/models into app.models. Callers share the request in flight, unless a model's row changed since
 * it started (then a new request goes out: a load that just finished must not be overwritten by an older
 * answer), or the API key changed since it was sent (then a new request carries the key in use now: an
 * answer for the old key says nothing about the new one; AU-09). An answer keeps the rows merged
 * after its request started, and an older answer that lands after a newer one is dropped; the failure of a
 * request that a newer one replaced is not recorded (the newer answer is the one that counts). The request
 * times out after 15 s.
 * @param {{silent?: boolean}} [options] silent: failures are recorded in app.models.error, not thrown
 * @returns {Promise<object|null>}
 */
export function refresh({ silent = false } = {}) {
  const key = getKey();
  if (!inflightRefresh || inflightRefresh.mark !== mark || inflightRefresh.key !== key) {
    const startMark = mark;
    const seq = (fetchSeq += 1);
    const entry = { mark: startMark, key, promise: null };
    entry.promise = (async () => {
      try {
        const data = await apiGet('/api/models', { accept: 'json', timeoutMs: REQUEST_TIMEOUT_MS });
        if (seq < appliedSeq) return app.get().models;   // a newer answer is already in
        appliedSeq = seq;
        const current = app.get().models;
        const fresh = Array.isArray(data?.models) ? data.models : [];
        const list = fresh.map((row) => {
          if ((rowMarks.get(row.id) || 0) <= startMark) return row;
          const kept = current.list.find((m) => m.id === row.id);
          return kept ? { ...row, ...pickStatus(kept) } : row;
        });
        app.set({
          models: {
            list,
            defaults: {
              tts: data?.default_tts_model || current.defaults.tts || '',
              stt: data?.default_stt_model || current.defaults.stt || '',
            },
            fetchedAt: Date.now(),
            error: '',
          },
        });
        return app.get().models;
      } catch (error) {
        if (error?.kind !== 'aborted' && seq >= appliedSeq && seq === fetchSeq) app.set({ models: { ...app.get().models, error: error?.message || String(error) } });
        throw error;
      } finally {
        if (inflightRefresh === entry) inflightRefresh = null;
      }
    })();
    inflightRefresh = entry;
  }
  const { promise } = inflightRefresh;
  return silent ? promise.catch(() => null) : promise;
}

// provider_failure comes with provider_available, so a status check never leaves a stale reason.
const STATUS_KEYS = ['state', 'device', 'progress', 'provider_available', 'provider_failure'];

function pickStatus(row) {
  const out = {};
  for (const key of STATUS_KEYS) if (key in row) out[key] = row[key];
  return out;
}

/** GET /api/models/{id}/status (the id is encodeURIComponent-ed). */
export async function getStatus(modelId) {
  return apiGet(`/api/models/${enc(modelId)}/status`, { accept: 'json', timeoutMs: REQUEST_TIMEOUT_MS });
}

/**
 * The TTS capability flags for a model (GET /api/tts/capabilities?model=): an object of flags plus
 * `max_input_length`, the lower of the server's and the model's input limits.
 */
export async function capabilities(modelId, { signal } = {}) {
  const url = modelId ? `/api/tts/capabilities?model=${enc(modelId)}` : '/api/tts/capabilities';
  const data = await apiGet(url, { signal, accept: 'json' });
  const caps = data?.capabilities || {};
  const flags = Array.isArray(caps) ? Object.fromEntries(caps.map((k) => [k, true])) : { ...caps };
  const limits = [data?.max_input_length, data?.max_input_chars].filter((n) => Number.isInteger(n) && n > 0);
  if (limits.length) flags.max_input_length = Math.min(...limits);
  return flags;
}

function mergeRow(modelId, status) {
  if (!status || typeof status !== 'object') return;
  const models = app.get().models;
  const idx = models.list.findIndex((m) => m.id === modelId);
  if (idx < 0) return;
  const row = { ...models.list[idx], notice: status.notice || '' };
  for (const key of STATUS_KEYS) {
    if (key in status) row[key] = status[key];
  }
  const list = models.list.slice();
  list[idx] = row;
  mark += 1;
  rowMarks.set(modelId, mark);
  app.set({ models: { ...models, list } });
}

/** Report a successful synthesis or transcription: keeps the ensureReady fast path open for 20 s. */
export function markUsed(modelId) {
  lastOk.set(String(modelId), Date.now());
  fastPathOff.delete(String(modelId));
}

/** Report a failed request for a model: the fast path stays off until a status check succeeds. */
export function markFailed(modelId) {
  fastPathOff.add(String(modelId));
  lastOk.delete(String(modelId));
}

function jobName(modelId, list) {
  const row = (list || []).find((m) => m.id === modelId) || { id: modelId };
  if (classifyKind(row) === 'stt') return sttLabel(row);
  return PROVIDER_DISPLAY[providerOf(modelId)] || modelLabel(modelId);
}

/** Start the status job for a model operation: "Loading Kokoro…", detail "Unloads Piper". */
function modelJob(verb, modelId, others = []) {
  const list = app.get().models.list;
  const detail = others.length ? `Unloads ${joinNames(others.map((o) => jobName(o.id, list)))}` : '';
  return beginJob(`model:${modelId}`, `${verb} ${jobName(modelId, list)}…`, { detail });
}

/**
 * Make sure a model is loaded before a request that needs it.
 *
 * Fast path: when app.models lists it as 'loaded' and either the inventory or its last
 * successful use is under 20 s old, return true with no request at all. Otherwise: a runOp() already running
 * for the model is joined (its failure is thrown); GET status; queued/downloading/loading (another tab, or
 * a load started elsewhere) -> wait for it, polling every 2 s; 'loaded' -> true; provider_missing /
 * provider_unavailable -> throw with the install advice; a TTS load that would unload another model asks
 * first (false when cancelled); available -> load (TTS) or download then load (STT); downloaded -> load. The
 * final status must be 'loaded'. A status job shows throughout. Concurrent calls for one model share one run.
 * @param {string} modelId
 * @param {{kind?: 'tts'|'stt', confirmSwitch?: boolean, fastPath?: boolean}} [options]
 * @returns {Promise<boolean>} false only when the listener cancelled the switch
 */
export async function ensureReady(modelId, { kind = 'tts', confirmSwitch = true, fastPath = true } = {}) {
  const id = String(modelId || '');
  if (!id) throw new Error(`No ${kind.toUpperCase()} model selected`);
  const models = app.get().models;
  const row = models.list.find((m) => m.id === id);
  const now = Date.now();
  if (fastPath && row && row.state === 'loaded' && !fastPathOff.has(id) && !opPromises.has(id)
      && (now - models.fetchedAt < FAST_PATH_MS || now - (lastOk.get(id) ?? -Infinity) < FAST_PATH_MS)) {
    return true;
  }
  if (readyPromises.has(id)) return readyPromises.get(id);
  const run = ensureReadyInner(id, { kind, confirmSwitch }).finally(() => {
    if (readyPromises.get(id) === run) readyPromises.delete(id);
  });
  readyPromises.set(id, run);
  return run;
}

/** Poll the status while a model is queued, downloading or loading; resolves with the first other status. */
async function waitWhileInFlight(id, status) {
  let current = status;
  if (!IN_FLIGHT.has(current?.state)) return current;
  const end = modelJob(current.state === 'loading' ? 'Loading' : 'Downloading', id);
  try {
    while (IN_FLIGHT.has(current?.state)) {
      await sleep(POLL_MS);
      current = await getStatus(id);
      mergeRow(id, current);
    }
    return current;
  } finally {
    end();
  }
}

async function ensureReadyInner(id, { kind, confirmSwitch }) {
  if (opPromises.has(id)) {
    // Models › Load (or Download) is already running for this model: wait for it instead of failing.
    const ok = await opPromises.get(id);
    if (!ok) {
      markFailed(id);
      throw new Error(app.get().ops[id]?.error || `Couldn't load ${modelLabel(id)}`);
    }
  }
  let status;
  try {
    status = await getStatus(id);
    mergeRow(id, status);
    status = await waitWhileInFlight(id, status);
  } catch (error) {
    markFailed(id);
    throw error;
  }
  fastPathOff.delete(id);
  if (status?.state === 'loaded') {
    lastOk.set(id, Date.now());
    return true;
  }
  if (status?.state === 'provider_missing') throw new Error(missingMessage(id));
  if (status?.state === 'provider_unavailable') {
    // A worker that answers but can't run says why (provider_failure); one that doesn't answer is
    // offline.
    const reason = workerFailure(status);
    throw new Error(reason ? cantRunText(reason) : 'The selected voice provider is offline');
  }

  let others = [];
  if (kind === 'tts') {
    if (Date.now() - app.get().models.fetchedAt > FAST_PATH_MS) await refresh({ silent: true });
    others = otherLoadedTtsModels(id, app.get().models.list);
    if (others.length && confirmSwitch) {
      const ok = await confirm({
        title: 'Switch speech model?',
        message: `Loading ${modelLabel(id)} will unload ${joinNames(others.map((o) => modelLabel(o.id)))}. Continue?`,
        confirmLabel: 'Load',
      });
      if (!ok) return false;
    }
  } else {
    others = otherLoadedSttModels(id, app.get().models.list).unloads;
  }

  const downloadsFirst = kind === 'stt' && (status?.state === 'available' || status?.state === 'provider_installed');
  let end = modelJob(downloadsFirst ? 'Downloading' : 'Loading', id, others);
  try {
    let state = status?.state;
    if (state === 'provider_installed' || state === 'available') {
      if (kind === 'tts') {
        await request(`/api/models/${enc(id)}/load`, { method: 'POST', accept: 'json' });
      } else {
        await request(`/api/models/${enc(id)}/download`, { method: 'POST', accept: 'json' });
      }
      const next = await getStatus(id);
      mergeRow(id, next);
      state = next?.state;
    }
    if (state === 'downloaded' || state === 'ready') {
      const endLoading = modelJob('Loading', id, others);
      end();
      end = endLoading;
      // An STT model loads next to the pinned STT_MODEL, as a transcription request naming it would load it.
      const keep = kind === 'stt' ? '?keep_pinned=true' : '';
      await request(`/api/models/${enc(id)}/load${keep}`, { method: 'POST', accept: 'json' });
    }
    const final = await getStatus(id);
    mergeRow(id, final);
    if (final?.state !== 'loaded') throw new Error(`${kind.toUpperCase()} model not ready: state=${final?.state}`);
    lastOk.set(id, Date.now());
  } catch (error) {
    markFailed(id);
    throw error;
  } finally {
    end();
  }
  await refresh({ silent: true });
  return true;
}

function setOp(modelId, op) {
  const ops = { ...app.get().ops };
  if (op) ops[modelId] = Object.freeze(op);
  else delete ops[modelId];
  app.set({ ops });
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run a model operation and own its polling until a terminal state, whatever view is showing: POST
 * download/prefetch/load, poll the status right away and then every 2 s (progress updates app.ops), reset
 * the 3-minute no-progress timeout while queued/downloading/loading. Failures are kept in
 * app.ops[modelId].error (and toasted when the Models view is not showing). refresh() runs at the end.
 * @param {string} modelId
 * @param {'download'|'prefetch'|'load'} kind
 * @returns {Promise<boolean>} true on success
 */
export function runOp(modelId, kind) {
  const id = String(modelId);
  if (!['download', 'prefetch', 'load'].includes(kind)) throw new TypeError(`Unknown model operation: ${kind}`);
  if (opPromises.has(id)) return opPromises.get(id);
  const promise = runOpInner(id, kind).finally(() => opPromises.delete(id));
  opPromises.set(id, promise);
  return promise;
}

async function runOpInner(id, kind) {
  const loading = kind === 'load';
  const list = app.get().models.list;
  const others = loading ? unloadedBy(id, list) : [];
  // Models › Load of an STT model loads it next to the pinned STT_MODEL, as Transcribe does (ensureReady): the
  // default speech-to-text model stays loaded unless its own row unloads it.
  const row = list.find((m) => m.id === id) || { id };
  // A TTS load from here is a person's own (explicit): it may unload the pinned STT model to make room.
  const keep = !loading ? '' : classifyKind(row) === 'stt' ? '?keep_pinned=true' : '?explicit=true';
  const startedAt = Date.now();
  setOp(id, { kind, startedAt, progress: null, error: '', phase: '' });
  const end = modelJob(loading ? 'Loading' : 'Downloading', id, others);
  let postDone = false;
  let postError = null;
  let wake = null;
  const post = request(`/api/models/${enc(id)}/${kind}${keep}`, { method: 'POST', accept: 'json' })
    .then(() => { postDone = true; }, (error) => { postDone = true; postError = error; })
    .finally(() => { if (wake) wake(); });
  try {
    let lastProgressAt = Date.now();
    for (;;) {
      if (postDone && postError) throw postError;
      let status = null;
      try {
        status = await getStatus(id);
      } catch (error) {
        if (postDone && !postError) throw error;
      }
      if (postDone && postError) throw postError;
      if (status) {
        mergeRow(id, status);
        const state = status.state;
        const progress = typeof status.progress === 'number' ? status.progress : null;
        setOp(id, { kind, startedAt, progress, error: '', phase: state === 'queued' ? 'queued' : '' });
        if (IN_FLIGHT.has(state)) {
          lastProgressAt = Date.now();
        } else if (postDone) {
          if (loading && state === 'loaded') break;
          if (!loading && (state === 'downloaded' || state === 'ready' || state === 'loaded')) break;
          if (state === 'provider_missing') throw new Error(missingMessage(id));
          if (state === 'provider_unavailable') throw new Error('Voice provider is offline');
          if (state === 'available' || state === 'provider_installed') throw new Error('Model reverted to available — load failed');
          throw new Error(`Model not ready: state=${state}`);
        }
      }
      if (postDone && Date.now() - lastProgressAt > NO_PROGRESS_TIMEOUT_MS) throw new Error('Timed out waiting for model operation');
      await Promise.race([sleep(POLL_MS), new Promise((resolve) => { wake = resolve; if (postDone) resolve(); })]);
      wake = null;
      if (!postDone) lastProgressAt = Date.now();
    }
    setOp(id, null);
    if (loading) lastOk.set(id, Date.now());
    return true;
  } catch (error) {
    const message = error?.message || 'Action failed';
    setOp(id, { kind, startedAt, progress: null, error: message, phase: '' });
    if (app.get().route.view !== 'models') {
      toast(`Couldn't ${loading ? 'load' : 'download'} ${modelLabel(id)}: ${message}`, { kind: 'error', id: `model-op-${id}` });
    }
    return false;
  } finally {
    end();
    await post.catch(() => {});
    await refresh({ silent: true });
  }
}

/** Forget a finished operation's error (the row returns to its plain state). */
export function clearOp(modelId) {
  if (app.get().ops[modelId]?.error) setOp(String(modelId), null);
}

/** DELETE /api/models/{id} (STT and TTS). On failure it refreshes silently, then rethrows. */
export async function unload(modelId) {
  const id = String(modelId);
  setOp(id, { kind: 'unload', startedAt: Date.now(), progress: null, error: '', phase: '' });
  try {
    await request(`/api/models/${enc(id)}`, { method: 'DELETE', accept: 'json' });
  } catch (error) {
    setOp(id, null);
    await refresh({ silent: true });
    throw error;
  }
  setOp(id, null);
  lastOk.delete(id);
  await refresh({ silent: true });
  return true;
}

/** The confirm copy for Delete files. */
export function deleteFilesMessage(modelId) {
  return `Delete downloaded files for ${modelLabel(modelId)}? This unloads the model and removes its cached files.`;
}

/**
 * DELETE /api/models/{id}/artifacts, then refresh(). The caller confirms first, or passes {confirm: true}
 * to show deleteFilesMessage() here (false when cancelled; nothing is sent).
 */
export async function deleteArtifacts(modelId, { confirm: ask = false } = {}) {
  const id = String(modelId);
  if (ask) {
    const ok = await confirm({ title: 'Delete files?', message: deleteFilesMessage(id), confirmLabel: 'Delete files', danger: true });
    if (!ok) return false;
  }
  setOp(id, { kind: 'delete', startedAt: Date.now(), progress: null, error: '', phase: '' });
  try {
    await request(`/api/models/${enc(id)}/artifacts`, { method: 'DELETE', accept: 'json' });
  } finally {
    setOp(id, null);
  }
  lastOk.delete(id);
  await refresh({ silent: true });
  return true;
}

/**
 * Load the configured default TTS model through runOp('load'), even when the cached inventory says it is
 * loaded (SP-11), then refresh().
 * @returns {Promise<string>} the default model id
 */
export async function restoreDefault() {
  const id = app.get().models.defaults.tts;
  if (!id || !ttsModels(app.get().models.list).some((m) => m.id === id)) {
    throw new Error('The configured default model is unavailable');
  }
  const ok = await runOp(id, 'load');
  if (!ok) throw new Error(app.get().ops[id]?.error || `Couldn't load ${modelLabel(id)}`);
  return id;
}

/** Test hook. */
export function __resetForTests() {
  inflightRefresh = null;
  lastOk = new Map();
  fastPathOff = new Set();
  opPromises = new Map();
  readyPromises = new Map();
  mark = 0;
  rowMarks = new Map();
  fetchSeq = 0;
  appliedSeq = 0;
}
