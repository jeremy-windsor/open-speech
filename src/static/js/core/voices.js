/**
 * Voices: descriptions, lists, blends, favourites and previews.
 * @module core/voices
 */
import { get as apiGet, request, ApiError } from './api.js';
import { KEYS, readJson, writeJson } from './storage.js';
import { PROVIDER_DISPLAY, modelLabel, parsePiperVoice, providerOf } from './models.js';

export const KOKORO_LANGUAGES = Object.freeze({
  a: 'American English', b: 'British English', e: 'Spanish', f: 'French', h: 'Hindi', i: 'Italian',
  j: 'Japanese', p: 'Portuguese', z: 'Mandarin',
});

/** The two Kokoro voices its authors grade highest. */
export const RECOMMENDED_VOICES = Object.freeze(['af_heart', 'af_bella']);

export const PREVIEW_TEXT = Object.freeze({
  a: 'Hello! This is how I sound.', b: 'Hello! This is how I sound.', e: '¡Hola! Así suena mi voz.',
  f: 'Bonjour ! Voici ma voix.', h: 'नमस्ते! मेरी आवाज़ ऐसी है।', i: 'Ciao! Questa è la mia voce.',
  j: 'こんにちは。これが私の声です。', p: 'Olá! Esta é a minha voz.', z: '你好！这是我的声音。',
  default: 'Hello! This is how I sound.',
});

const KOKORO_ID = /^([a-z])([fm])_([a-z0-9]+)$/;
const PIPER_VOICE = /^piper\/[^/]+$/;
const PIPER_SPEAKER = /^(piper\/[^/]+)\/(.+)$/;

/**
 * Describe one voice for pickers and chips (today's describeVoice). Kokoro 'af_heart' ->
 * {label: 'Heart', detail: 'American English · female', group: 'Recommended', lang: 'a', gender: 'female'}.
 * @param {object|string} voice a /v1/audio/voices entry or an id
 * @returns {{id: string, label: string, detail: string, group: string, lang: string, language: string,
 *   gender: string, tags: string[]}}
 */
export function describeVoice(voice) {
  const id = String(voice?.id || voice?.voice_id || voice?.name || (typeof voice === 'string' ? voice : '') || '');
  const name = voice?.name && voice.name !== id ? String(voice.name) : '';
  const kokoro = KOKORO_ID.exec(id);
  if (kokoro) {
    const language = KOKORO_LANGUAGES[kokoro[1]] || 'Other languages';
    const gender = kokoro[2] === 'f' ? 'female' : 'male';
    const recommended = RECOMMENDED_VOICES.includes(id);
    return {
      id,
      label: name || kokoro[3].charAt(0).toUpperCase() + kokoro[3].slice(1),
      detail: `${language} · ${gender}`,
      group: recommended ? 'Recommended' : language,
      lang: kokoro[1],
      language,
      gender,
      tags: recommended ? ['recommended'] : [],
    };
  }
  const rawGender = String(voice?.gender || '').toLowerCase();
  const gender = rawGender && rawGender !== 'unknown' ? rawGender : '';
  // A Piper voice is its model: 'piper/en_US-amy-medium' -> 'Amy (US)', 'Medium quality'. A speaker of one
  // ('piper/en_US-arctic-medium/awb') keeps its own name, with its model's name and quality as detail.
  const piper = PIPER_VOICE.test(id) ? parsePiperVoice(id) : null;
  const speaker = PIPER_SPEAKER.exec(id);
  const owner = speaker ? parsePiperVoice(speaker[1]) : null;
  let label = name || id;
  let detail = gender;
  if (piper && piper.lang) {
    label = piper.name;
    detail = [`${piper.qualityLabel} quality`, gender].filter(Boolean).join(' · ');
  } else if (owner && owner.lang) {
    label = name || speaker[2];
    detail = [owner.name, `${owner.qualityLabel} quality`, gender].filter(Boolean).join(' · ');
  }
  return {
    id,
    label,
    detail,
    group: 'Voices',
    lang: '',
    language: typeof voice?.language === 'string' ? voice.language : '',
    gender,
    tags: Array.isArray(voice?.tags) ? voice.tags.map(String) : [],
  };
}

/** The name a model's voices carry outside its own list: 'Kokoro', 'Supertonic', 'Piper'. */
export function voiceModelName(model) {
  const id = String(model || '');
  return id ? PROVIDER_DISPLAY[providerOf(id)] || modelLabel(id) : '';
}

/** A label with its model's name: 'Supertonic · M1'; the label alone without a model. */
export function withModel(model, label) {
  const name = voiceModelName(model);
  const text = String(label || '');
  return name && text ? `${name} · ${text}` : text;
}

/** A label without its model's name: 'Heart' for 'Kokoro · Heart'. */
export function withoutModel(model, label) {
  const name = voiceModelName(model);
  const text = String(label || '');
  return name && text.startsWith(`${name} · `) ? text.slice(name.length + 3) : text;
}

/**
 * A voice as it is shown outside its model's own list: 'Supertonic · M1', 'Kokoro · Heart + Bella'. A named voice
 * (voice:<id>) keeps its own name, `label`.
 * @param {string} model
 * @param {string} voice
 * @param {string} [label] the voice's label when known (a picker row's)
 */
export function voiceTitle(model, voice, label = '') {
  const v = String(voice || '');
  if (v.startsWith('voice:')) return String(label || 'Named voice');
  let own = String(label || '');
  if (!own && isBlend(v)) {
    try { own = parseBlend(v).map((item) => describeVoice(item.voice).label).join(' + '); } catch { own = v; }
  }
  return withModel(model, own || describeVoice(v).label || v);
}

/**
 * describeVoice over a list. When two rows share a label (Kokoro's duplicate display names), the id is
 * appended to their detail so they can be told apart.
 */
export function describeVoices(voices) {
  const rows = (voices || []).map(describeVoice);
  // Piper voices of one name differ in quality, which their detail already says.
  const key = (row) => (PIPER_VOICE.test(row.id) ? `${row.label}|${row.detail}` : row.label).toLowerCase();
  const counts = new Map();
  for (const row of rows) counts.set(key(row), (counts.get(key(row)) || 0) + 1);
  return rows.map((row) => (counts.get(key(row)) > 1
    ? { ...row, detail: row.detail ? `${row.detail} · ${row.id}` : row.id }
    : row));
}

const PIPER_LANGUAGES = Object.freeze({ en_US: 'American English', en_GB: 'British English' });
const PIPER_QUALITY_RANK = Object.freeze({ x_low: 0, low: 1, medium: 2, high: 3 });

/**
 * PURE: Piper voice models as voices for the voice picker: 'piper/en_US-amy-medium' ->
 * {id: 'piper/en_US-amy-medium', name, model: 'piper/en_US-amy-medium', language: 'American English'}, which
 * describeVoice shows as 'Amy (US)', 'Medium quality'. In the order the languages first appear, then by name,
 * then low to high quality.
 * @param {string[]} ids
 * @returns {{id: string, name: string, model: string, language: string}[]}
 */
export function modelVoices(ids) {
  const entries = (ids || []).map(String).filter(Boolean).map((id, index) => ({ id, index, parsed: parsePiperVoice(id) }));
  const languages = [...new Set(entries.map((e) => e.parsed.lang))];
  const rank = (e) => PIPER_QUALITY_RANK[e.parsed.quality] ?? 2;
  return entries
    .sort((a, b) => languages.indexOf(a.parsed.lang) - languages.indexOf(b.parsed.lang)
      || a.parsed.name.localeCompare(b.parsed.name) || rank(a) - rank(b) || a.index - b.index)
    .map(({ id, parsed }) => ({ id, name: id, model: id, language: PIPER_LANGUAGES[parsed.lang] || parsed.lang }));
}

let speakerCache = new Map();   // model -> its speakers ([] for a model with one voice)

/**
 * A voice model's speakers (Piper's multi-speaker models): GET /v1/audio/voices?model= when it lists voices other
 * than the model itself, else []. Kept for the page once loaded; a failed load throws (the ApiError) and is asked
 * again next time.
 * @param {string} model
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<object[]>}
 */
export async function speakersOf(model, { signal } = {}) {
  const id = String(model || '');
  if (speakerCache.has(id)) return speakerCache.get(id);
  const list = await listVoices(id, { signal, strict: true });
  const own = (v) => String(v?.id || v?.voice_id || v?.name || '');
  const speakers = list.some((v) => own(v) && own(v) !== id) ? list : [];
  speakerCache.set(id, speakers);
  return speakers;
}

/**
 * GET /v1/audio/voices?model= -> the model's built-in voices[]. Named-voice rows (kind 'named_voice', listed
 * there for OpenAI clients) are left out: the app lists those with listNamed. Any error gives [] (the caller
 * shows its empty state), unless `strict` is set: then the ApiError is thrown, so the caller can tell "no
 * voices" from "didn't load".
 * @param {string} model
 * @param {{signal?: AbortSignal, strict?: boolean}} [options]
 */
export async function listVoices(model, { signal, strict = false } = {}) {
  const builtIn = (list) => list.filter((v) => v?.kind !== 'named_voice');
  try {
    const url = model ? `/v1/audio/voices?model=${encodeURIComponent(model)}` : '/v1/audio/voices';
    const data = await apiGet(url, { signal, accept: 'json' });
    if (Array.isArray(data)) return builtIn(data);
    if (Array.isArray(data?.voices)) return builtIn(data.voices);
  } catch (error) {
    if (strict) throw error;
  }
  return [];
}

/**
 * Named voices: GET /api/voices/identities?model= (with availability for that model), or with no model
 * the whole catalogue; check:false adds check_availability=false. Each carries `design`: null, or for a designed
 * voice {description, model, seed, options, sample, transcript, duration_s} (`sample` is its library name).
 * @returns {Promise<object[]>}
 */
export async function listNamed({ model = '', check = true } = {}) {
  const params = new URLSearchParams();
  if (model) params.set('model', model);
  if (!check) params.set('check_availability', 'false');
  const qs = params.toString();
  const data = await apiGet(`/api/voices/identities${qs ? `?${qs}` : ''}`, { accept: 'json' });
  const voices = Array.isArray(data?.voices) ? data.voices : [];
  return voices.map((voice) => (voice && typeof voice === 'object' ? { ...voice, design: voice.design || null } : voice));
}

/**
 * POST /api/voices/designed (multipart): name, model, description, transcript, audio (WAV, at most 30 s), and
 * optional seed, options (a JSON string) and speak_with (repeated). Resolves the new named voice, its `design`
 * included.
 * @param {FormData} formData
 * @returns {Promise<object>}
 */
export function saveDesigned(formData) {
  return request('/api/voices/designed', { method: 'POST', body: formData, accept: 'json' });
}

/**
 * PUT /api/voices/designed/{id} (multipart): a new sample and design for a designed voice (the saveDesigned fields
 * without name and speak_with); its versions keep pointing at the same sample. Resolves the named voice.
 * @param {string} id
 * @param {FormData} formData
 * @returns {Promise<object>}
 */
export function replaceDesigned(id, formData) {
  return request(`/api/voices/designed/${encodeURIComponent(id)}`, { method: 'PUT', body: formData, accept: 'json' });
}

/** GET /api/profiles -> {presets, defaultId}. */
export async function listPresets() {
  const data = await apiGet('/api/profiles', { accept: 'json' });
  return { presets: Array.isArray(data?.profiles) ? data.profiles : [], defaultId: data?.default_profile_id || null };
}

/** GET /api/voices/library -> recordings[]. */
export async function listRecordings() {
  const data = await apiGet('/api/voices/library', { accept: 'json' });
  return Array.isArray(data) ? data : [];
}

/**
 * GET /api/voices/library-config -> {maxSeconds}: 0 means no limit; 60 when the server doesn't say
 * (a rolling upgrade from a server without the route). The server stays authoritative.
 */
export async function recordingConfig() {
  try {
    const data = await apiGet('/api/voices/library-config', { accept: 'json' });
    const max = Number(data?.max_seconds);
    return { maxSeconds: Number.isFinite(max) && max > 0 ? max : 0 };
  } catch {
    return { maxSeconds: 60 };
  }
}

/** A blend recipe that can't be parsed. */
export class BlendError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BlendError';
  }
}

const COMPONENT = /^([a-zA-Z0-9_]+)(?:\((\d+(?:\.\d+)?)\))?$/;

/**
 * Parse a blend recipe with the server's grammar: 'af_bella(2)+af_sky' ->
 * [{voice: 'af_bella', weight: 2}, {voice: 'af_sky', weight: 1}]. A single voice gives one item.
 * @param {string} spec
 * @returns {{voice: string, weight: number}[]}
 * @throws {BlendError} on an empty recipe, a bad part, a duplicate voice or a weight <= 0
 */
export function parseBlend(spec) {
  const text = String(spec ?? '').trim();
  if (!text) throw new BlendError('Enter at least one voice.');
  const seen = new Set();
  return text.split('+').map((raw) => {
    const part = raw.trim();
    const m = COMPONENT.exec(part);
    if (!m) throw new BlendError(`“${part || raw}” isn't a voice. Use name or name(weight), joined with +.`);
    const voice = m[1];
    const weight = m[2] === undefined ? 1 : Number(m[2]);
    if (!(weight > 0)) throw new BlendError(`The weight for ${voice} must be more than 0.`);
    if (seen.has(voice)) throw new BlendError(`${voice} appears more than once in the blend.`);
    seen.add(voice);
    return { voice, weight };
  });
}

// A weight exactly as the number it is (SP-27: blends round-trip exactly): the shortest decimal that reads back
// as the same number, so 0.725 stays 0.725 and an edited 0.3 is "0.3". Never in exponent form ("1e-7" is not in
// the server's grammar) and never "0" for a positive weight (the server would silently mute that voice).
function weightText(weight) {
  const n = Number(weight);
  const text = String(n);
  if (!/e/i.test(text)) return text;
  const plain = n.toFixed(20).replace(/\.?0+$/, '');
  return plain === '0' && n > 0 ? '0.00000000000000000001' : plain;
}

/**
 * The canonical recipe: one voice -> 'af_bella'; two or more -> 'af_bella(2)+af_sky(1)', each weight the
 * shortest decimal for its number (no rounding, no trailing zeros).
 */
export function formatBlend(items) {
  const list = (items || []).filter((item) => item && item.voice);
  if (!list.length) return '';
  if (list.length === 1) return String(list[0].voice);
  return list.map((item) => `${item.voice}(${weightText(item.weight ?? 1)})`).join('+');
}

/** @returns {boolean} whether a voice string is a blend recipe */
export function isBlend(spec) {
  return /[+(]/.test(String(spec ?? ''));
}

/** Integer percents for the blend's weights, summing to 100 (largest-remainder rounding). */
export function blendPercents(items) {
  const list = items || [];
  if (!list.length) return [];
  const weights = list.map((item) => Math.max(0, Number(item?.weight) || 0));
  const total = weights.reduce((a, b) => a + b, 0);
  const shares = total > 0 ? weights.map((w) => (w / total) * 100) : weights.map(() => 100 / list.length);
  const floors = shares.map(Math.floor);
  let left = 100 - floors.reduce((a, b) => a + b, 0);
  const order = shares.map((s, i) => ({ i, r: s - Math.floor(s) })).sort((a, b) => b.r - a.r || a.i - b.i);
  for (let k = 0; left > 0; k = (k + 1) % order.length, left -= 1) floors[order[k].i] += 1;
  return floors;
}

const favKey = (model, id) => `${model}|${id}`;

/** @returns {string[]} the favourite voices as "model|voiceId" */
export function favorites() {
  const list = readJson(KEYS.favorites, [], Array.isArray);
  return list.filter((item) => typeof item === 'string');
}

/** @returns {boolean} */
export function isFavorite(model, id) {
  return favorites().includes(favKey(model, id));
}

/**
 * Add or remove a favourite.
 * @returns {boolean} whether it is a favourite now
 */
export function toggleFavorite(model, id) {
  const key = favKey(model, id);
  const list = favorites();
  const at = list.indexOf(key);
  if (at >= 0) list.splice(at, 1);
  else list.push(key);
  writeJson(KEYS.favorites, list);
  return at < 0;
}

const PREVIEW_CACHE_MAX = 20;
let previewCache = new Map();   // key -> {url}
let previewInflight = new Map(); // key -> {controller, waiters, promise}
let pagehideBound = false;

function revokePreviews() {
  for (const entry of previewCache.values()) URL.revokeObjectURL(entry.url);
  previewCache = new Map();
}

function bindPagehide() {
  if (pagehideBound || typeof window === 'undefined' || !window.addEventListener) return;
  pagehideBound = true;
  // Only a real unload: a page restored from the back/forward cache (event.persisted) keeps its cached previews.
  window.addEventListener('pagehide', (event) => { if (!event || !event.persisted) revokePreviews(); });
}

function abortError() {
  return new ApiError('Canceled', { code: 'aborted', kind: 'aborted' });
}

/**
 * A short preview of a voice: POST /v1/audio/speech as MP3 WITHOUT X-History (previews stay out of
 * History). Results are kept in an LRU cache of 20 object URLs; evicted URLs are revoked, and all of them
 * on pagehide. Callers asking for the same preview share one request: each caller's signal only
 * cancels its own wait, and the request is cancelled once every waiting caller has.
 * @param {string} model
 * @param {string} voice
 * @param {{text?: string, signal?: AbortSignal}} [options]
 * @returns {Promise<{url: string}>}
 */
export async function preview(model, voice, { text, signal } = {}) {
  const lang = describeVoice(voice).lang;
  const input = text || PREVIEW_TEXT[lang] || PREVIEW_TEXT.default;
  const key = `${model}|${voice}|${input}`;
  if (signal?.aborted) throw abortError();
  const cached = previewCache.get(key);
  if (cached) {
    previewCache.delete(key);
    previewCache.set(key, cached);
    return cached;
  }
  let shared = previewInflight.get(key);
  if (!shared) {
    const controller = new AbortController();
    shared = { controller, waiters: 0, promise: null };
    shared.promise = (async () => {
      try {
        const blob = await request('/v1/audio/speech', {
          method: 'POST', json: { model, voice, input, response_format: 'mp3' }, signal: controller.signal, accept: 'blob',
        });
        bindPagehide();
        const entry = { url: URL.createObjectURL(blob) };
        previewCache.set(key, entry);
        while (previewCache.size > PREVIEW_CACHE_MAX) {
          const [oldKey, oldEntry] = previewCache.entries().next().value;
          previewCache.delete(oldKey);
          URL.revokeObjectURL(oldEntry.url);
        }
        return entry;
      } finally {
        if (previewInflight.get(key) === shared) previewInflight.delete(key);
      }
    })();
    shared.promise.catch(() => {});
    previewInflight.set(key, shared);
  }
  shared.waiters += 1;
  const task = shared;
  let onAbort = null;
  try {
    return await new Promise((resolve, reject) => {
      task.promise.then(resolve, reject);
      if (signal) {
        onAbort = () => reject(abortError());
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  } finally {
    if (signal && onAbort) signal.removeEventListener('abort', onAbort);
    task.waiters -= 1;
    if (task.waiters <= 0 && previewInflight.get(key) === task) {
      // Nobody waits for it any more: stop the synthesis.
      task.controller.abort();
      previewInflight.delete(key);
    }
  }
}

/** Test hook. */
export function __resetForTests() {
  speakerCache = new Map();
  previewCache = new Map();
  previewInflight = new Map();
  pagehideBound = false;
}
