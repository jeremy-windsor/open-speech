/**
 * Speak's voice for the rest of the app: `app.speakVoice` = {model, voice, label, kind}.
 * The Reader reads it for RE-04's last fallback and its "Same as Speak" row.
 *
 * - While the Speak view runs, its updateSpeakVoice() publishes the live setup (`voice` is the string Speak
 *   sends; `kind` comes from voiceKind()).
 * - Before it has run (a page that opens straight into the Reader, a deep link, a reload), main.js starts
 *   startBootVoice() after the first view mounted. Once the model list is known it publishes
 *   what Speak would select, without running the view: its saved setup (KEYS.speak) on the model it would
 *   pick, else that model's default voice. Kokoro's is known (its recommended voice); another provider's is
 *   the first voice of its list, which costs one GET /v1/audio/voices in the background, off every path to
 *   first audio. Speak's own publish refines it once the view runs (claim() stops this one).
 * - `kind` says what the voice is: 'named' (voice:<id>), 'blend' (a recipe), 'voice' (a
 *   provider voice), 'recording' (a raw reference recording, which a Reader render can't use: REV-09), or ''
 *   when there is none.
 *
 * Importing this module has no side effects; it imports only core modules main.js already preloads.
 * @module views/speak/speak-voice
 */
import { app } from '../../core/store.js';
import * as models from '../../core/models.js';
import { KEYS, readJson } from '../../core/storage.js';
import { describeVoice, isBlend, parseBlend, listVoices, withModel, RECOMMENDED_VOICES } from '../../core/voices.js';

/** How long the boot publisher waits for a provider's voice list before giving up (it publishes nothing then). */
export const VOICE_LIST_TIMEOUT_MS = 8000;

/* ───────────────────────────── pure helpers (tested) ───────────────────────────── */

/** Validate a stored setup (KEYS.speak) field by field; anything unusable is left out. */
export function savedSetup(value) {
  const v = value && typeof value === 'object' && !Array.isArray(value) ? value : null;
  if (!v) return null;
  const out = {};
  if (typeof v.model === 'string') out.model = v.model;
  if (typeof v.voice === 'string') out.voice = v.voice;
  if (typeof v.identity === 'string') out.identity = v.identity;
  if (typeof v.reference === 'string') out.reference = v.reference;
  if (Number.isFinite(Number(v.speed))) out.speed = Number(v.speed);
  if (typeof v.format === 'string') out.format = v.format;
  if (typeof v.stream === 'boolean') out.stream = v.stream;
  if (Array.isArray(v.effects) || v.effects === null) out.effects = v.effects;
  if (typeof v.instructions === 'string') out.instructions = v.instructions;
  if (v.options && typeof v.options === 'object' && !Array.isArray(v.options)) out.options = { ...v.options };
  return out;
}

/**
 * PURE: what Speak's voice is, in the order its voice string is built (request.voiceSpec): a named voice, a
 * blend of two or more voices, a provider voice, else the reference recording alone.
 * @param {{identity?: string, blend?: boolean, voice?: string, reference?: string}} s
 * @returns {'named'|'blend'|'voice'|'recording'|''}
 */
export function voiceKind({ identity = '', blend = false, voice = '', reference = '' } = {}) {
  if (identity) return 'named';
  if (blend) return 'blend';
  if (voice) return 'voice';
  if (reference) return 'recording';
  return '';
}

/** PURE: a blend's label as the voice button shows it ("Bella + Sky", "Bella + Sky + 1"). */
export function blendLabel(spec, describe = describeVoice) {
  let items = [];
  try { items = parseBlend(spec); } catch { items = []; }
  const labels = items.map((item) => describe(item.voice).label || item.voice);
  return labels.length > 2 ? `${labels.slice(0, 2).join(' + ')} + ${labels.length - 2}` : labels.join(' + ');
}

/**
 * PURE: the voice Speak would select before its view has run.
 * - The model: models.pickTtsModel over the usable TTS models, preferring the saved setup's model (SP-02).
 * - On the saved setup's own model, its voice: a named voice as voice:<id>, a blend, a provider voice, or a
 *   reference recording alone (kind 'recording').
 * - Otherwise the model's default: Kokoro's recommended voice; for another provider `voice` is '' and
 *   `needsList` is true (Speak selects the first voice of the model's list).
 * @param {{saved?: object|null, list?: object[], configuredDefault?: string}} input `list` is app.models.list
 * @returns {{model: string, voice: string, label: string, kind: string, needsList?: boolean}|null} null without a model
 */
export function bootVoice({ saved = null, list = [], configuredDefault = '' } = {}) {
  const usable = models.ttsModels(list || []);
  const s = savedSetup(saved);
  const model = models.pickTtsModel(usable, { preferred: s && s.model ? s.model : '', configuredDefault });
  if (!model) return null;
  if (s && s.model === model) {
    const voice = s.voice || '';
    const reference = s.reference || '';
    if (s.identity) return { model, voice: `voice:${s.identity}`, label: 'Named voice', kind: 'named' };
    if (voice && isBlend(voice)) return { model, voice, label: withModel(model, blendLabel(voice) || voice), kind: 'blend' };
    if (voice && voice !== reference) return { model, voice, label: withModel(model, describeVoice(voice).label || voice), kind: 'voice' };
    if (reference) return { model, voice: reference, label: reference, kind: 'recording' };
  }
  const row = usable.find((m) => m.id === model);
  if ((row && row.provider ? row.provider : models.providerOf(model)) === 'kokoro') {
    const voice = RECOMMENDED_VOICES[0];
    return { model, voice, label: withModel(model, describeVoice(voice).label), kind: 'voice' };
  }
  return { model, voice: '', label: '', kind: '', needsList: true };
}

/** A voice list entry's id (/v1/audio/voices rows, or plain strings). */
function voiceIdOf(entry) {
  return String((entry && (entry.id || entry.voice_id || entry.name)) || (typeof entry === 'string' ? entry : '') || '');
}

/* ───────────────────────────── the boot publisher ───────────────────────────── */

let claimed = false;
let stopBoot = null;

/**
 * Speak's view runs and publishes its own voice from now on: the boot publisher stops (its request in flight,
 * if any, is aborted and never publishes). Speak's init() calls it.
 */
export function claim() {
  claimed = true;
  if (stopBoot) stopBoot();
}

/**
 * Publish what Speak would select into app.speakVoice once the model list is known, until Speak's view runs.
 * Called once by main.js after the first view mounted; a no-op once Speak claimed the voice.
 * It follows later model lists (a key fixed, a worker that came up) and writes only when the answer changes.
 * @param {{timeoutMs?: number}} [options]
 * @returns {() => void} stop
 */
export function startBootVoice({ timeoutMs = VOICE_LIST_TIMEOUT_MS } = {}) {
  if (claimed) return () => {};
  if (stopBoot) return stopBoot;
  let published = '';
  let pending = null;   // {model, controller, timer}: one voice list request at a time
  const listed = new Map();   // model -> its first voice ('' when the list failed or was empty): asked once per page
  let off = null;

  const cancel = () => {
    if (!pending) return;
    clearTimeout(pending.timer);
    try { pending.controller.abort(); } catch { /* done */ }
    pending = null;
  };

  const publish = (value) => {
    const next = { model: value.model, voice: value.voice, label: value.label, kind: value.kind };
    const key = JSON.stringify(next);
    if (key === published) return;
    published = key;
    app.set({ speakVoice: next });
  };

  const firstVoice = (model) => {
    if (listed.has(model)) {
      cancel();
      const known = listed.get(model);
      if (known) publish(known);
      return;
    }
    if (pending && pending.model === model) return;
    cancel();
    const controller = new AbortController();
    const mine = { model, controller, timer: setTimeout(() => controller.abort(), timeoutMs) };
    pending = mine;
    // Non-strict: a failure or the timeout reads as an empty list, and nothing is published (nor asked again).
    listVoices(model, { signal: controller.signal }).then((list) => {
      if (pending !== mine) return;   // cancelled: another model, stop() or Speak's claim() (which aborts it)
      clearTimeout(mine.timer);
      pending = null;
      const rows = Array.isArray(list) ? list : [];
      const voice = rows.map(voiceIdOf).find(Boolean);
      const entry = voice ? rows.find((v) => voiceIdOf(v) === voice) : null;
      const value = voice ? { model, voice, label: withModel(model, describeVoice(entry || voice).label || voice), kind: 'voice' } : '';
      listed.set(model, value);
      if (value) publish(value);
    }).catch((error) => { if (globalThis.console) globalThis.console.error(error); });
  };

  const update = () => {
    if (claimed) return;
    const st = app.get();
    if (!st.models.fetchedAt) return;
    const choice = bootVoice({
      saved: readJson(KEYS.speak, null), list: st.models.list, configuredDefault: (st.models.defaults && st.models.defaults.tts) || '',
    });
    if (!choice) return;
    if (choice.needsList) {
      firstVoice(choice.model);
      return;
    }
    cancel();
    publish(choice);
  };

  const stop = () => {
    if (off) off();
    off = null;
    cancel();
    if (stopBoot === stop) stopBoot = null;
  };
  stopBoot = stop;
  off = app.select((st) => st.models, () => update(), { immediate: true });
  return stop;
}

/** Test hook. */
export function __resetForTests() {
  if (stopBoot) stopBoot();
  stopBoot = null;
  claimed = false;
}
