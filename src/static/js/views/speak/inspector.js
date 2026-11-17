/**
 * Speak: the inspector. One model select, the model state pill and "Restore Kokoro", the
 * voice button (the shared voice picker), the blend editor, the reference recording, speed, format, effect
 * chips, instructions and the Stream switch. It owns the voice setup Speak sends and Live reads.
 *
 * The setup is one plain object (see currentSetup()): {model, voice, identity, reference, speed, format,
 * stream, effects, instructions, options}, where `voice` is the recipe (a provider voice id or a canonical blend) and
 * `identity` a named voice id. Everything a model offers (capabilities, voices, named voices, recordings) is
 * loaded once per model change, with a request token, so a late answer for an earlier model never replaces a
 * newer choice (inventory SP-05). Capabilities, voices and the input limit are therefore in hand before
 * Generate.
 * @module views/speak/inspector
 */
import { html, render, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app } from '../../core/store.js';
import * as models from '../../core/models.js';
import * as voicesApi from '../../core/voices.js';
import * as audio from '../../core/audio.js';
import { explain } from '../../core/api.js';
import { prompt } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { copyText } from '../../core/clipboard.js';
import { announce } from '../../core/a11y.js';
import { speed as speedText, spokenSpeed } from '../../core/format.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { bindRange } from '../../components/range.js';
import { mountModelOptions } from '../../components/model-options.js';
import { openVoicePicker } from '../../components/voice-picker.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import * as blend from './blend.js';
import {
  FORMATS, FORMAT_MIME, emptyEffects, effectsPayload, effectsState, effectOn, clampSpeed, voiceSpec, streamBlocker,
  mseSupported, STREAM_HINTS, SPEED_STEP, PITCH_MIN, PITCH_MAX, REVERB_ROOMS, DEFAULT_LIMIT,
  EFFECT_TYPES, formatLabel, modelOptions, speedRange,
} from './request.js';

/** Model data stays fresh this long; voices:changed and auth:changed drop it earlier. */
export const DATA_TTL_MS = 60000;

/* ───────────────────────────── pure helpers (tested) ───────────────────────────── */

/** A voice entry's id (/v1/audio/voices rows, or plain strings). */
export function voiceIdOf(v) {
  return String((v && (v.id || v.voice_id || v.name)) || (typeof v === 'string' ? v : '') || '');
}

/**
 * The provider voice to select (inventory SP-09): the preferred one when the model offers it, else the
 * current one when offered, else the first; '' when the model has no stock voices (a clone model).
 */
export function pickVoice(ids, { preferred = '', current = '' } = {}) {
  const list = ids || [];
  if (preferred && list.includes(preferred)) return preferred;
  if (current && list.includes(current)) return current;
  return list[0] || '';
}

/**
 * The model select's groups: one per provider, Kokoro then Piper then the rest (models.ttsModels order). The Piper
 * voice models are one option, models.PIPER_GROUP, whose voices the voice picker lists.
 */
export function modelGroups(list) {
  const groups = [];
  for (const entry of models.selectEntries(models.ttsModels(list || []))) {
    const provider = entry.provider;
    const label = models.PROVIDER_DISPLAY[provider] || provider;
    let group = groups.find((g) => g.provider === provider);
    if (!group) {
      group = { provider, label, options: [] };
      groups.push(group);
    }
    group.options.push({ id: entry.id, text: entry.label });
  }
  return groups;
}

/**
 * The model state pill: {tone, text}. "Loaded · GPU"; "Loads when you generate · unloads
 * Piper"; "Downloads, then loads when you generate"; busy, offline and install states from models.uiState().
 */
export function modelHint(row, { list = [], op = null } = {}) {
  if (!row) return { tone: 'idle', text: '' };
  const ui = models.uiState(row, op, { list });
  const others = models.otherLoadedTtsModels(row.id, list);
  const unloads = others.length ? ` · unloads ${models.joinNames(others.map((o) => models.shortName(o)))}` : '';
  switch (ui.key) {
    case 'loaded': return { tone: 'ok', text: ui.label };
    case 'downloaded': return { tone: 'idle', text: `Loads when you generate${unloads}` };
    case 'on-first-load':
    case 'not-downloaded': return { tone: 'idle', text: `Downloads, then loads when you generate${unloads}` };
    case 'queued':
    case 'downloading':
    case 'loading':
    case 'unloading':
    case 'deleting': return { tone: 'busy', text: `${ui.label}…`.replace(/……$/, '…') };
    case 'worker-offline': return { tone: 'error', text: 'Worker offline' };
    case 'not-installed':
    case 'worker-missing': return { tone: 'warn', text: ui.label };
    case 'error': return { tone: 'error', text: ui.label };
    default: return { tone: ui.tone || 'idle', text: ui.label || '' };
  }
}

/** The default's name in the Restore action: "Kokoro". */
function defaultName(defaultId) {
  const provider = models.providerOf(defaultId);
  return models.PROVIDER_DISPLAY[provider] || models.modelLabel(defaultId);
}

/**
 * The Restore action's label (inventory SP-11): "Restore Kokoro" while another model is selected; "Load
 * Kokoro now" when the default is already the selected model and only isn't loaded, where "Restore" would
 * read as doing nothing (a deliberate wording change; the action is the same: load and select).
 */
export function restoreLabel(defaultId, selectedId = '') {
  const name = defaultName(defaultId);
  return selectedId && selectedId === defaultId ? `Load ${name} now` : `Restore ${name}`;
}

/** Restore shows when the configured default exists and another model is selected or it isn't loaded. */
export function showRestore(defaultId, selectedId, list) {
  if (!defaultId) return false;
  const row = models.ttsModels(list || []).find((m) => m.id === defaultId);
  if (!row) return false;
  return selectedId !== defaultId || row.state !== 'loaded';
}

/** The reference select's options (inventory SP-36): "None", then each recording with its transcript state. */
export function referenceOptions(recordings) {
  const out = [{ value: '', label: 'None' }];
  for (const rec of recordings || []) {
    const name = rec && (rec.name || rec.id);
    if (!name) continue;
    out.push({ value: String(name), label: `${name} · ${rec.transcript ? 'transcript saved' : 'no transcript'}` });
  }
  return out;
}

/** The realization of a named voice for a model ('' when it has none that is available). */
export function realizationFor(identity, model, { available = true } = {}) {
  const list = (identity && identity.realizations) || [];
  return list.find((r) => r.model === model && (!available || r.available !== false)) || null;
}

/** The designed named voice whose sample is `sample`, when it has an available version for `model`; else null. */
export function designedVoiceFor(named, sample, model) {
  const row = (named || []).find((n) => n && n.design && n.design.sample === sample) || null;
  return row && row.available !== false && realizationFor(row, model) ? row : null;
}

/**
 * What the voice button shows: {name, detail, seed}. A named voice shows its name, a blend its voices, a
 * provider voice its label and language, a clone model with only a reference the recording's name, and a model
 * that designs a voice from the description (`designing`) "Designed voice".
 */
export function voiceDisplay(s, describe = (id) => voicesApi.describeVoice(id), { designing = false } = {}) {
  const mix = s.mix || [];
  if (s.identity) {
    const what = mix.length >= 2 ? `Blend of ${mix.length} voices` : (s.voice ? describe(s.voice).label : (s.reference || ''));
    return { name: s.identityName || 'Named voice', detail: what ? `Named voice · ${what}` : 'Named voice', seed: s.identity };
  }
  if (mix.length >= 2) {
    const labels = mix.map((item) => describe(item.voice).label);
    const name = labels.length > 2 ? `${labels.slice(0, 2).join(' + ')} + ${labels.length - 2}` : labels.join(' + ');
    return { name, detail: `Blend of ${mix.length} voices`, seed: blend.toSpec(mix) };
  }
  if (s.voice) {
    const d = describe(s.voice);
    return { name: d.label || s.voice, detail: d.detail || '', seed: s.voice };
  }
  if (s.reference) return { name: s.reference, detail: 'Reference recording', seed: s.reference };
  if (designing) return { name: 'Designed voice', detail: 'From the voice description', seed: '' };
  return { name: 'Choose a voice', detail: '', seed: '' };
}

/**
 * Whether Speak designs the voice from the description: the model designs voices and no voice, blend, named voice
 * or reference recording is chosen.
 */
export function designsVoice(caps, s) {
  return !!caps && caps.voice_design === true && !(s.identity || (s.mix || []).length >= 2 || s.voice || s.reference);
}

/** The Instructions field's label and hint: "Voice description" while Speak designs the voice. */
export function instructionsCopy(designing, { fixedSpeed = false } = {}) {
  if (designing) {
    return { label: 'Voice description', hint: 'Who speaks, such as “a calm woman in her forties with a low, warm voice”. Each take is a new voice.' };
  }
  const pace = fixedSpeed ? ' This model has no speed slider: ask for slower or faster here.' : '';
  return { label: 'Direction', hint: `How to say it: pace, mood and emphasis, such as “Speak slowly, calm and warm”.${pace} Sent only when filled in.` };
}

/** The voice recipe Speak would save (a blend's canonical string, a provider voice, or the reference). */
export function recipeOf(s) {
  const mix = s.mix || [];
  if (mix.length >= 2) return blend.toSpec(mix);
  return (mix[0] && mix[0].voice) || s.voice || s.reference || '';
}

/* ───────────────────────────── banners (shared by the view) ───────────────────────────── */

/**
 * Render an inline banner into `host`. tone: error|warning|info|success. Errors use
 * role=alert, the rest role=status; `polite` makes an error role=status too (a failure nobody's action
 * caused, such as one on page load: role=alert is kept for blocking failures after an action); live:
 * false leaves the role to the host (a persistent live region). `detail` goes in a <details> through
 * textContent. actions: [{label, run(event, button), primary?}]. onDismiss adds a Dismiss button.
 */
export function renderBanner(host, { tone = 'error', title = '', message = '', detail = '', actions = [], onDismiss = null, live = true, polite = false } = {}) {
  if (!host) return;
  const role = !live ? null : tone === 'error' && !polite ? 'alert' : 'status';
  const glyph = tone === 'error' ? 'alert' : tone === 'success' ? 'check' : tone === 'warning' ? 'alert' : 'info';
  render(host, html`<div class="banner banner--${tone}"${role ? html` role="${role}"` : ''}>${icon(glyph, { className: 'ico banner__icon' })}<div class="banner__body">${title ? html`<p class="banner__title">${title}</p>` : ''}${message ? html`<p class="banner__msg">${message}</p>` : ''}${detail ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}</div>${actions.length ? html`<div class="banner__actions">${actions.map((a, i) => html`<button type="button" class="btn ${a.primary ? 'btn--primary' : 'btn--secondary'} btn--sm" data-banner-act="${i}"><span class="btn__label">${a.label}</span></button>`)}</div>` : ''}${onDismiss ? html`<button type="button" class="icon-btn" data-banner-act="dismiss" aria-label="Dismiss" data-tooltip="Dismiss">${icon('x')}</button>` : ''}</div>`);
  const pre = host.querySelector('pre');
  if (pre) pre.textContent = String(detail);
  for (const button of host.querySelectorAll('[data-banner-act]')) {
    const act = button.getAttribute('data-banner-act');
    button.addEventListener('click', (event) => {
      if (act === 'dismiss') {
        onDismiss(event);
        return;
      }
      const action = actions[Number(act)];
      if (!action) return;
      try {
        const result = action.run(event, button);
        if (result && typeof result.then === 'function') result.catch(() => {});
      } catch { /* the action reports its own errors */ }
    });
  }
  host.hidden = false;
}

/**
 * Banner props for an error, with Retry / Open Settings as api.explain() suggests. A refused or locked key
 * reads "API key needed" (or "Too many wrong keys") with Open Settings and no message: the shell's
 * connection banner carries the details, so the sentence isn't repeated.
 * Details show explain()'s `detailText`: the server's message and code, never its JSON
 * envelope; a proxy's page or a validation repr as it came.
 */
export function errorBanner(error, title, { retry = null } = {}) {
  const info = explain(error, title);
  const actions = [];
  const settings = info.action === 'open-settings';
  if (settings) actions.push({ label: 'Open Settings', run: () => openConnectionSettings() });
  else if (retry && (info.retryable || info.action === 'retry' || !error || error.kind !== 'auth')) actions.push({ label: 'Retry', run: retry });
  // A dropped connection or a silent server: what happened, then what to do, under the view's
  // title ("Couldn't generate speech · Can't reach the server. Check that Open Speech is running.").
  const kind = error && error.kind;
  const said = (kind === 'network' || kind === 'timeout') && info.cause && info.cause !== info.title
    ? `${info.cause}. ${info.message}` : info.message;
  const message = settings ? '' : said;
  const detail = info.detailText;
  return {
    tone: 'error',
    title: settings ? info.cause : info.title,
    message,
    detail: detail && detail !== message ? detail : '',   // Details that only repeat the message add nothing
    actions,
  };
}

/** PURE: what keeps a banner on screen unchanged (a poll that changes nothing re-renders nothing). */
export function bannerSignature(banner) {
  if (!banner) return '';
  return JSON.stringify([banner.tone || '', banner.title || '', banner.message || '', banner.detail || '', (banner.actions || []).map((a) => a.label)]);
}

/* ───────────────────────────── the inspector ───────────────────────────── */

const els = (root) => {
  const q = (id) => root.querySelector(`#${id}`);
  return {
    root: q('sp-inspector'), summary: q('sp-summary'), summaryText: q('sp-summary-text'), summaryAvatar: q('sp-summary-avatar'),
    body: q('sp-inspector-body'), title: q('sp-inspector-title'), error: q('sp-inspector-error'), loading: q('sp-inspector-loading'),
    fields: q('sp-inspector-fields'),
    model: q('sp-model'), hint: q('sp-model-hint'), hintText: q('sp-model-hint-text'), restore: q('sp-restore-default'),
    voice: q('sp-voice'), voiceName: q('sp-voice-name'), voiceDetail: q('sp-voice-detail'), voiceAvatar: q('sp-voice-avatar'),
    blendToggle: q('sp-blend-toggle'), blend: q('sp-blend'), blendBar: q('sp-blend-bar'), blendRows: q('sp-blend-rows'),
    blendString: q('sp-blend-string'), blendCopy: q('sp-blend-copy'), blendAdd: q('sp-blend-add'), blendEqual: q('sp-blend-equal'),
    blendPaste: q('sp-blend-paste'), blendPreview: q('sp-blend-preview'), blendCount: q('sp-blend-count'), blendHint: q('sp-blend-hint'),
    blendError: q('sp-blend-error'), blendErrorText: q('sp-blend-error-text'),
    referenceField: q('sp-reference-field'), reference: q('sp-reference'), referenceError: q('sp-reference-error'),
    referenceErrorText: q('sp-reference-error-text'),
    speed: q('sp-speed'), speedOut: q('sp-speed-out'), speedDown: q('sp-speed-down'), speedUp: q('sp-speed-up'), speedHint: q('sp-speed-hint'),
    options: q('sp-options'),
    format: q('sp-format'), formatHint: q('sp-format-hint'),
    effects: q('sp-effects'), reverbField: q('sp-reverb-field'), reverbRoom: q('sp-reverb-room'), pitchField: q('sp-pitch-field'),
    pitch: q('sp-pitch'), pitchDown: q('sp-pitch-down'), pitchUp: q('sp-pitch-up'),
    advanced: q('sp-advanced'), instructions: q('sp-instructions'), instructionsLabel: q('sp-instructions-label'),
    instructionsHint: q('sp-instructions-hint'),
    streamWrap: q('sp-stream-wrap'), stream: q('sp-stream'), streamHint: q('sp-stream-hint'),
  };
};

/**
 * Build the inspector over the rendered partial.
 * @param {Element} root the view section
 * @param {{onChange?: (kind: string, info?: object) => void, stopLive?: (reason: string) => void,
 *   onNotice?: (banner: object|null) => void, isEditing?: () => boolean}} hooks
 *   onChange kinds: user edits 'model', 'voice', 'named', 'blend', 'reference', 'speed', 'options', 'format',
 *   'effects', 'instructions', 'stream'; and 'commit' (a setup applied), 'load' (loading or loaded), 'models'
 */
export function createInspector(root, hooks = {}) {
  const el = els(root);
  const on = (target, type, fn) => { if (target) target.addEventListener(type, fn); offs.push(() => target && target.removeEventListener(type, fn)); };
  const offs = [];
  const notify = (kind, info) => { if (typeof hooks.onChange === 'function') hooks.onChange(kind, info); };
  const stopLive = (reason) => { if (typeof hooks.stopLive === 'function') hooks.stopLive(reason); };
  const notice = (banner) => { if (typeof hooks.onNotice === 'function') hooks.onNotice(banner); };

  const s = {
    model: '', data: null, loading: false, error: null,
    voice: '', mix: [], blendOpen: false, identity: '', identityName: '', reference: '',
    speed: 1, format: 'mp3', stream: true, effects: emptyEffects(), instructions: '', options: {},
    open: false, modelsReady: false,
  };
  // Who owns the inspector: every selectModel() and every committed hand-off (commitPrepared) takes a new
  // `seq`; an older one that is still loading then changes nothing. A hand-off that is only prepared (a
  // preset that fails its checks, a History entry whose model didn't load) takes none, so it never strands
  // a selection that is still loading.
  let seq = 0;
  let prepSeq = 0;   // hand-offs started (prepared): only the newest may commit
  let settledWaiters = [];   // callers waiting for the model data in flight (whenSettled)
  let cache = new Map();   // model -> {data, at}
  let pickerOpen = false;
  let renderedGroups = '';
  let rowsKey = '';
  let loadBannerSig = '';   // the inspector banner on screen (bannerSignature)
  let optionsUi = null;     // the model options' sliders (mountModelOptions), for the model whose options they show
  let optionsKey = '';
  let retrying = false;     // a banner Retry is reloading: the banner stays, its Retry busy
  let lastInGroup = '';     // the Piper voice model used last: choosing the Piper entry again picks it

  const speedSlider = el.speed ? bindRange(el.speed, { output: el.speedOut, format: speedText, spoken: spokenSpeed }) : null;

  /* ── data ── */

  async function fetchModelData(model) {
    const [caps, voiceList, named] = await Promise.all([
      models.capabilities(model),
      voicesApi.listVoices(model, { strict: true }),
      voicesApi.listNamed({ model }).catch(() => null),
    ]);
    const recordings = caps.voice_clone ? await voicesApi.listRecordings() : [];
    const list = Array.isArray(voiceList) ? voiceList : [];
    return {
      model,
      caps,
      voices: list,
      ids: list.map(voiceIdOf).filter(Boolean),
      described: voicesApi.describeVoices(list),
      named: named || [],
      namedLoaded: !!named,
      recordings,
    };
  }

  /**
   * Everything a model offers, from the cache when fresh (one load per model, shared by callers).
   * @returns {Promise<object>}
   */
  function prepare(model, { fresh = false } = {}) {
    const hit = cache.get(model);
    if (hit && !fresh && (hit.pending || Date.now() - hit.at < DATA_TTL_MS)) return hit.promise;
    const entry = { at: Date.now(), pending: true, promise: null };
    entry.promise = fetchModelData(model).then((data) => {
      entry.pending = false;
      entry.at = Date.now();
      return data;
    }, (error) => {
      if (cache.get(model) === entry) cache.delete(model);
      throw error;
    });
    cache.set(model, entry);
    return entry.promise;
  }

  function describe(id) {
    const row = s.data && s.data.described.find((d) => d.id === id);
    return row || voicesApi.describeVoice(id);
  }

  function caps() {
    return (s.data && s.data.caps) || {};
  }

  /* ── the setup ── */

  function currentSetup() {
    return {
      model: s.model,
      voice: recipeOf(s),
      identity: s.identity,
      reference: s.reference,
      speed: s.speed,
      format: s.format,
      stream: s.stream,
      effects: effectsPayload(s.effects),
      instructions: s.instructions,
      options: { ...s.options },
    };
  }

  /** Resolve every caller waiting for the model data in flight (whenSettled). */
  function settle() {
    const waiting = settledWaiters;
    settledWaiters = [];
    for (const resolve of waiting) resolve();
  }

  /** @returns {Promise<void>} resolved once no model data is loading (at once when none is) */
  function whenSettled() {
    if (!s.loading) return Promise.resolve();
    return new Promise((resolve) => { settledWaiters.push(resolve); });
  }

  /**
   * Apply a setup to loaded model data (no network). Returns {warnings}: 'identity-unavailable',
   * {kind: 'missing-reference', name}, 'blend-dropped', 'speed-fixed'.
   * A field the setup leaves out (undefined) keeps what the listener has at this moment, not at the time the
   * load started: a speed, effect, format, voice or reference changed while the model data loaded is kept.
   */
  function commit(data, setup) {
    const warnings = [];
    const c = data.caps || {};
    const ids = data.ids || [];
    const wantedIdentity = setup.identity === undefined ? s.identity : setup.identity;
    let identityRow = null;
    if (wantedIdentity) {
      identityRow = (data.named || []).find((n) => n.id === wantedIdentity) || null;
      const realization = identityRow ? realizationFor(identityRow, data.model) : null;
      if (!identityRow || identityRow.available === false || !realization) {
        warnings.push('identity-unavailable');
        identityRow = null;
      }
    }
    let spec = String((setup.voice === undefined ? recipeOf(s) : setup.voice) || '');
    let wantedReference = setup.reference === undefined ? s.reference : String(setup.reference || '');
    if (!identityRow && wantedReference) {
      // A designed voice's hidden sample as the reference (a History entry): that named voice speaks.
      identityRow = designedVoiceFor(data.named, wantedReference, data.model);
    }
    if (identityRow) {
      const realization = realizationFor(identityRow, data.model);
      spec = String(realization.voice || '');
      wantedReference = String(realization.reference_audio_id || '');
    }
    let mix = [];
    let voice = '';
    if (spec && voicesApi.isBlend(spec)) {
      let items = [];
      try { items = blend.fromSpec(spec); } catch { items = []; }
      const offered = items.length && (!ids.length || items.every((item) => ids.includes(item.voice)));
      if (c.voice_blend === true && offered && items.length >= 2) {
        mix = items;
        voice = items[0].voice;
      } else {
        warnings.push('blend-dropped');
        voice = pickVoice(ids, { preferred: items[0] && items[0].voice, current: s.voice });
      }
    } else {
      voice = pickVoice(ids, { preferred: spec, current: s.voice });
    }
    let reference = '';
    if (c.voice_clone) {
      const names = (data.recordings || []).map((r) => String(r.name || r.id || ''));
      // A designed voice's sample is not a recording: the named voice carries it.
      const ownSample = !!identityRow && !!identityRow.design && identityRow.design.sample === wantedReference;
      if (wantedReference && names.includes(wantedReference)) reference = wantedReference;
      else if (wantedReference && !ownSample) warnings.push({ kind: 'missing-reference', name: wantedReference });
    }
    const wantedSpeed = setup.speed === undefined ? s.speed : Number(setup.speed);
    if (c.speed_control === false && Number.isFinite(wantedSpeed) && wantedSpeed !== 1) warnings.push('speed-fixed');

    s.model = data.model;
    if (models.isGroupValue(models.selectValue(data.model))) lastInGroup = data.model;
    s.data = data;
    s.loading = false;
    s.error = null;
    s.voice = voice;
    s.mix = mix;
    s.blendOpen = mix.length > 0;
    s.identity = identityRow ? identityRow.id : '';
    s.identityName = identityRow ? String(identityRow.name || '') : '';
    s.reference = reference;
    s.speed = c.speed_control === false ? 1 : clampSpeed(Number.isFinite(wantedSpeed) ? wantedSpeed : 1, c);
    if (setup.format !== undefined) s.format = FORMATS.some((f) => f.value === setup.format) ? setup.format : 'mp3';
    if (setup.effects !== undefined) s.effects = effectsState(setup.effects);
    if (setup.instructions !== undefined) s.instructions = String(setup.instructions || '');
    if (setup.stream !== undefined) s.stream = setup.stream !== false;
    // The model's own options: a setup that has them says all of them (one it leaves out is at its default or
    // Auto); without them, what was set stays where the model has the same option.
    s.options = modelOptions(c, setup.options && typeof setup.options === 'object' ? setup.options : s.options);
    renderAll();
    if (el.instructions && el.instructions.value !== s.instructions) el.instructions.value = s.instructions;
    // A model that only designs voices needs the description to speak at all.
    if (el.advanced && models.designOnly(c)) el.advanced.open = true;
    notify('commit', { warnings });
    settle();
    return { warnings };
  }

  /**
   * Select a model and apply `setup`; what it leaves out keeps the listener's choices where the model offers
   * them (read when the data lands, see commit()). The newest selection or committed hand-off wins: an
   * earlier selection that is still loading resolves {stale: true} and changes nothing.
   */
  async function selectModel(model, setup = null, { fresh = false } = {}) {
    const token = (seq += 1);
    const wanted = setup ? { ...setup, model } : { model };
    s.model = model;
    s.loading = true;
    s.error = null;
    renderModelSelect();
    renderState();
    notify('load', { loading: true });
    try {
      const data = await prepare(model, { fresh });
      if (token !== seq) return { stale: true, warnings: [] };
      return commit(data, wanted);
    } catch (error) {
      if (token !== seq) return { stale: true, warnings: [] };
      s.loading = false;
      s.error = error;
      s.data = null;
      renderState();
      notify('load', { loading: false, error });
      settle();
      return { error, warnings: [] };
    }
  }

  /**
   * Load a model's data for a caller that validates before committing (presets, History, hand-offs). It
   * takes no turn of its own: a selection still loading carries on unless this one commits. `current()`
   * is false once a newer selection, a newer hand-off or a committed one came along.
   */
  async function prepared(model) {
    const base = seq;
    const mine = (prepSeq += 1);
    const data = await prepare(model);
    return { data, current: () => base === seq && mine === prepSeq };
  }

  /** Commit prepared data unless a newer selection happened meanwhile; a selection still loading is then dropped. */
  function commitPrepared(p, setup) {
    if (!p.current()) return { stale: true, warnings: [] };
    seq += 1;
    return commit(p.data, { ...setup, model: p.data.model });
  }

  /* ── rendering ── */

  function renderModelSelect() {
    if (!el.model) return;
    const groups = modelGroups(app.get().models.list);
    const key = JSON.stringify(groups);
    if (key !== renderedGroups) {
      renderedGroups = key;
      render(el.model, groups.map((g) => html`<optgroup label="${g.label}">${g.options.map((o) => html`<option value="${o.id}">${o.text}</option>`)}</optgroup>`));
    }
    const value = models.selectValue(s.model);
    if (value && el.model.value !== value) el.model.value = value;
  }

  function renderModelState() {
    const state = app.get();
    const list = state.models.list;
    const row = list.find((m) => m.id === s.model);
    const hint = modelHint(row, { list, op: state.ops[s.model] || null });
    if (el.hint) {
      el.hint.className = `pill pill--${hint.tone}`;
      el.hint.hidden = !hint.text;
      if (el.hintText) el.hintText.textContent = hint.text;
    }
    const def = state.models.defaults.tts;
    if (el.restore) {
      const show = showRestore(def, s.model, list);
      el.restore.hidden = !show;
      const label = el.restore.querySelector('.btn__label');
      if (label && el.restore.getAttribute('aria-busy') !== 'true') label.textContent = restoreLabel(def, s.model);
    }
  }

  function designing() {
    return designsVoice(caps(), s);
  }

  function renderVoice() {
    const d = voiceDisplay(s, describe, { designing: designing() });
    if (el.voiceName) el.voiceName.textContent = d.name;
    if (el.voiceDetail) {
      el.voiceDetail.textContent = d.detail;
      el.voiceDetail.hidden = !d.detail;
    }
    for (const av of [el.voiceAvatar, el.summaryAvatar]) {
      if (!av) continue;
      av.textContent = d.seed ? monogram(d.name) : '';
      if (d.seed) applyTile(av, d.seed);
      av.hidden = !d.seed;
    }
    if (el.summaryText) el.summaryText.textContent = summaryText();
    renderAdvanced();
  }

  function summaryText() {
    const st = app.get();
    if (!st.models.fetchedAt) {
      if (!st.models.error) return 'Loading…';
      return keyRefused() ? explain(modelsError()).cause : "Couldn't load the models";
    }
    if (!s.model) return 'No speech model';
    if (!s.data && s.error && !s.loading) {
      const info = explain(s.error);
      return info.action === 'open-settings' ? info.cause : "Couldn't load the voices";
    }
    const design = designing();
    const d = voiceDisplay(s, describe, { designing: design });
    const row = app.get().models.list.find((m) => m.id === s.model);
    // The model as the select names it: "Piper" for a Piper voice, whose name the voice already gives.
    const modelName = models.isGroupValue(models.selectValue(s.model)) ? models.PROVIDER_DISPLAY.piper : (row ? models.shortName(row) : models.modelLabel(s.model));
    const parts = [s.voice || s.mix.length || s.identity || s.reference || design ? d.name : '', modelName, speedText(s.speed)];
    return parts.filter(Boolean).join(' · ');
  }

  function renderBlend() {
    const c = caps();
    const can = c.voice_blend === true && !!s.data;
    // "Mix voices" is the editor's disclosure while the mix has one voice (so the editor can be closed
    // again); a real blend of two or more voices keeps the editor open and the toggle away.
    if (el.blendToggle) el.blendToggle.hidden = !can || s.mix.length >= 2;
    if (el.blendToggle) el.blendToggle.setAttribute('aria-expanded', String(can && s.blendOpen));
    if (!el.blend) return;
    el.blend.hidden = !can || !s.blendOpen;
    if (!can || !s.blendOpen) {
      blendError('');
      return;
    }
    const segs = blend.segments(s.mix, (id) => describe(id).label);
    if (el.blendBar) {
      render(el.blendBar, segs.map((seg) => html`<span class="blend-bar__seg" data-i="${seg.index % 6}">${seg.showLabel ? `${seg.percent}%` : ''}</span>`));
      const parts = el.blendBar.querySelectorAll('.blend-bar__seg');
      segs.forEach((seg, i) => { if (parts[i]) parts[i].style.setProperty('--w', `${seg.percent}%`); });
      el.blendBar.setAttribute('aria-label', blend.barLabel(s.mix, (id) => describe(id).label) || 'Blend');
    }
    if (el.blendRows) {
      const single = s.mix.length <= 1;
      const key = s.mix.map((item) => item.voice).join('|');
      if (key !== rowsKey || el.blendRows.querySelectorAll('[data-voice]').length !== segs.length) {
        // Voices added or removed: rebuild the rows. A weight change updates them in place (focus stays).
        rowsKey = key;
        render(el.blendRows, segs.map((seg) => html`<li class="sp-blend__row" data-voice="${seg.voice}">
          <span class="swatch" data-i="${seg.index % 6}" aria-hidden="true"></span>
          <span class="sp-blend__name"><span class="sp-blend__label">${seg.label}</span><span class="sp-blend__id mono">${seg.voice}</span></span>
          <span class="stepper sp-blend__stepper">
            <button type="button" class="icon-btn" data-blend-act="down" aria-label="${`Decrease weight for ${seg.label}`}">${icon('minus')}</button>
            <input class="input stepper__value sp-blend__weight" type="text" inputmode="decimal" value="${blend.weightText(seg.weight)}" aria-label="${`Weight for ${seg.label}`}" data-blend-act="weight">
            <button type="button" class="icon-btn" data-blend-act="up" aria-label="${`Increase weight for ${seg.label}`}">${icon('plus')}</button>
          </span>
          <span class="sp-blend__pct tabular"></span>
          <button type="button" class="icon-btn sp-blend__remove" data-blend-act="remove" aria-label="${`Remove ${seg.label}`}" data-tooltip="${`Remove ${seg.label}`}">${icon('x')}</button>
        </li>`));
      }
      for (const seg of segs) {
        const row = el.blendRows.querySelector(`[data-voice="${seg.voice}"]`);
        if (!row) continue;
        const w = Number(seg.weight);
        const input = row.querySelector('[data-blend-act="weight"]');
        if (input && document.activeElement !== input) input.value = blend.weightText(w);
        const pct = row.querySelector('.sp-blend__pct');
        if (pct) pct.textContent = `${seg.percent}%`;
        const down = row.querySelector('[data-blend-act="down"]');
        if (down) down.disabled = w <= blend.MIN_WEIGHT;
        const up = row.querySelector('[data-blend-act="up"]');
        if (up) up.disabled = w >= blend.MAX_WEIGHT;
        const remove = row.querySelector('[data-blend-act="remove"]');
        if (remove) remove.disabled = single;
      }
    }
    const spec = blend.toSpec(s.mix);
    // A break opportunity after each "+", so a long recipe wraps between voices, not inside one.
    if (el.blendString) render(el.blendString, html`${spec.split('+').map((part, i) => html`${i ? html`+<wbr>` : ''}${part}`)}`);
    if (el.blendCount) el.blendCount.textContent = s.mix.length >= 2 ? `${s.mix.length} voices` : '1 voice';
    if (el.blendAdd) {
      const full = s.mix.length >= blend.MAX_VOICES;
      el.blendAdd.setAttribute('aria-disabled', String(full));
      if (full) el.blendAdd.setAttribute('data-full', 'true');
      else el.blendAdd.removeAttribute('data-full');
    }
    if (el.blendEqual) el.blendEqual.disabled = s.mix.length < 2;
    if (el.blendPreview) el.blendPreview.disabled = !spec;
    if (el.blendCopy) el.blendCopy.disabled = !spec;
  }

  function renderReference() {
    const c = caps();
    const show = !!c.voice_clone && !!s.data;
    if (el.referenceField) el.referenceField.hidden = !show;
    if (!el.reference || !show) return;
    const opts = referenceOptions(s.data.recordings);
    render(el.reference, opts.map((o) => html`<option value="${o.value}">${o.label}</option>`));
    el.reference.value = s.reference;
  }

  function renderSpeed() {
    const fixed = caps().speed_control === false;
    const range = speedRange(caps());
    if (el.speed) {
      el.speed.min = String(range.min);
      el.speed.max = String(range.max);
    }
    if (speedSlider) speedSlider.set(s.speed);
    if (el.speed) el.speed.disabled = fixed;
    if (el.speedDown) el.speedDown.disabled = fixed || s.speed <= range.min;
    if (el.speedUp) el.speedUp.disabled = fixed || s.speed >= range.max;
    if (el.speedHint) el.speedHint.hidden = !fixed;
    if (el.speed) {
      if (fixed) el.speed.setAttribute('aria-describedby', 'sp-speed-hint');
      else el.speed.removeAttribute('aria-describedby');
    }
  }

  /** The model's own options (capabilities.options: Supertonic's steps, Breeze's Guidance and Variation). */
  function renderOptions() {
    if (!el.options) return;
    const declared = s.data ? caps().options || null : null;
    const key = declared ? `${s.model}|${JSON.stringify(declared)}` : '';
    if (key !== optionsKey) {
      optionsKey = key;
      if (optionsUi) optionsUi.destroy();
      optionsUi = null;
      if (key) {
        optionsUi = mountModelOptions(el.options, caps(), {
          values: s.options,
          idPrefix: 'sp-option',
          onChange: (values) => {
            s.options = values;
            notify('options');
          },
        });
      }
      el.options.hidden = !optionsUi || !optionsUi.has();
      return;
    }
    if (optionsUi) optionsUi.set(s.options);
  }

  function renderFormat() {
    if (el.format && el.format.value !== s.format) el.format.value = s.format;
    if (!el.formatHint) return;
    let hint = '';
    if (s.format === 'pcm') hint = 'PCM is raw audio: takes download and never play.';
    else if (FORMAT_MIME[s.format] && !audio.canPlay(FORMAT_MIME[s.format])) hint = `This browser can't play ${formatLabel(s.format)}, so takes download instead.`;
    el.formatHint.textContent = hint;
    el.formatHint.hidden = !hint;
  }

  function renderEffects() {
    if (!el.effects) return;
    for (const chip of el.effects.querySelectorAll('[data-effect]')) {
      chip.setAttribute('aria-pressed', String(effectOn(s.effects, chip.getAttribute('data-effect'))));
    }
    if (el.reverbField) el.reverbField.hidden = !s.effects.reverb.on;
    if (el.reverbRoom && el.reverbRoom.value !== s.effects.reverb.room) el.reverbRoom.value = s.effects.reverb.room;
    if (el.pitchField) el.pitchField.hidden = !s.effects.pitch.on;
    if (el.pitch && document.activeElement !== el.pitch) el.pitch.value = String(s.effects.pitch.semitones);
    if (el.pitchDown) el.pitchDown.disabled = s.effects.pitch.semitones <= PITCH_MIN;
    if (el.pitchUp) el.pitchUp.disabled = s.effects.pitch.semitones >= PITCH_MAX;
  }

  function renderAdvanced() {
    if (!el.advanced) return;
    // Rebuilding never touches the typed instructions or the open state (inventory SP-08, VL-19).
    el.advanced.hidden = !caps().instructions;
    const copy = instructionsCopy(designing(), { fixedSpeed: caps().speed_control === false });
    if (el.instructionsLabel) el.instructionsLabel.textContent = copy.label;
    if (el.instructionsHint) el.instructionsHint.textContent = copy.hint;
  }

  function renderStream() {
    const c = caps();
    if (el.streamWrap) el.streamWrap.hidden = c.streaming !== true;
    if (!el.stream) return;
    el.stream.checked = !!s.stream;
    const blocker = streamBlocker({ caps: c, format: s.format, effects: effectsPayload(s.effects), mse: mseSupported(), on: true });
    const hint = STREAM_HINTS[blocker] || '';
    el.stream.disabled = !!hint;
    if (el.streamHint) {
      el.streamHint.textContent = hint;
      el.streamHint.hidden = !hint || c.streaming !== true;
    }
  }

  /** The inspector's load failure, or null: the models list, else this model's capabilities or voices. */
  function loadBanner() {
    const st = app.get();
    if (!st.models.fetchedAt && st.models.error) return errorBanner(modelsError(), "Couldn't load the models", { retry: (event, button) => retry(() => models.refresh({ silent: true }), button) });
    if (s.error && !s.loading) return errorBanner(s.error, "Couldn't load the voices", { retry: (event, button) => retry(() => selectModel(s.model, null, { fresh: true }), button) });
    return null;
  }

  /**
   * #sp-inspector-error is a persistent role=status region: the banner inside is rebuilt only when what it
   * says changes, so the 60 s models poll and every model-op tick neither re-announce it nor take the focus
   * off its Retry (live regions never tick). A load failure isn't a blocking failure after an
   * action, so it is polite.
   */
  function renderLoadBanner() {
    if (!el.error) return;
    const banner = loadBanner();
    if (!banner && retrying) return;   // the reload runs: the banner stays, with its Retry busy
    const sig = bannerSignature(banner);
    if (sig === loadBannerSig) return;
    loadBannerSig = sig;
    if (banner) renderBanner(el.error, { ...banner, live: false });
    else render(el.error, html``);
    el.error.hidden = false;
  }

  /**
   * A banner's Retry: the banner stays on screen with the Retry busy while it reloads (focus stays on it);
   * when the reload worked and the banner goes, the focus moves to the model select (or the inspector).
   */
  async function retry(load, button) {
    const hadFocus = !!button && typeof document !== 'undefined' && document.activeElement === button;
    retrying = true;
    if (button) setBusy(button, true);
    try {
      await load();
    } catch { /* the banner shows what failed */ }
    retrying = false;
    renderState();
    if (button && button.isConnected) setBusy(button, false);
    if (hadFocus && !(el.root && el.root.contains(document.activeElement))) focusInspector();
  }

  /** Put the focus back into the inspector after its banner went away: a new Retry, the model select, the heading. */
  function focusInspector() {
    const again = el.error ? el.error.querySelector('button') : null;
    if (again) {
      again.focus({ preventScroll: true });
      return;
    }
    if (el.fields && !el.fields.hidden && el.model) {
      el.model.focus({ preventScroll: true });
      return;
    }
    const wide = typeof globalThis.matchMedia === 'function' && globalThis.matchMedia('(min-width: 960px)').matches;
    const target = !wide && el.summary ? el.summary : el.title;
    if (!target) return;
    if (target === el.title) target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
  }

  function renderState() {
    const st = app.get();
    const fetched = st.models.fetchedAt > 0;
    s.modelsReady = fetched;
    const noModels = fetched && !models.ttsModels(st.models.list).length;
    const modelsFailed = !fetched && !!st.models.error;
    const busy = (!fetched && !modelsFailed) || (s.loading && !s.data);
    renderLoadBanner();
    // While a Retry reloads, its busy banner is the progress shown; the skeleton would only add noise.
    if (el.loading) el.loading.hidden = !busy || noModels || (retrying && !!loadBannerSig);
    if (el.fields) el.fields.hidden = busy || noModels || modelsFailed || (!!s.error && !s.data);
    if (el.root) el.root.setAttribute('aria-busy', String(busy));
    renderModelState();
    renderVoice();
  }

  function keyRefused() {
    const state = app.get().connection.state;
    return state === 'unauthorized' || state === 'locked';
  }

  /** The models list's failure as an error explain() reads: a refused key, a lockout, or anything else. */
  function modelsError() {
    const message = app.get().models.error || "Couldn't load the models";
    const state = app.get().connection.state;
    if (state === 'locked') return { kind: 'locked', status: 429, message };
    return state === 'unauthorized' ? { kind: 'auth', status: 401, message } : { message, kind: 'http', status: 0 };
  }

  function renderAll() {
    renderModelSelect();
    renderState();
    renderBlend();
    renderReference();
    renderSpeed();
    renderOptions();
    renderFormat();
    renderEffects();
    renderAdvanced();
    renderStream();
  }

  function setOpen(open) {
    s.open = !!open;
    if (el.root) el.root.setAttribute('data-open', String(s.open));
    if (el.summary) el.summary.setAttribute('aria-expanded', String(s.open));
  }

  /* ── user edits ── */

  function edited(kind, { live = true } = {}) {
    if (live) stopLive('Voice changed');
    renderAll();
    notify(kind);
  }

  function clearIdentity() {
    s.identity = '';
    s.identityName = '';
  }

  /**
   * Wait for the model data in flight (a model change), with the trigger marked busy meanwhile, so the
   * voice controls never act on the previous model's voices. @returns {Promise<boolean>} whether the
   * selected model's data is in hand afterwards
   */
  async function afterLoad(trigger) {
    if (s.loading) {
      if (trigger) trigger.setAttribute('aria-busy', 'true');
      try {
        await whenSettled();
      } finally {
        if (trigger) trigger.removeAttribute('aria-busy');
      }
    }
    return !!s.data && s.data.model === s.model && !s.loading;
  }

  /**
   * The voice picker: a provider voice, a named voice or a recording. On the Piper entry it lists every Piper voice
   * model; choosing another one selects that model with the voice.
   */
  async function openPicker(trigger = el.voice) {
    if (pickerOpen || !s.model) return null;
    pickerOpen = true;
    let choice = null;
    let model = '';
    try {
      if (!(await afterLoad(trigger))) return null;
      model = s.model;
      const value = s.identity ? `voice:${s.identity}` : (s.mix.length >= 2 ? '' : (s.voice || s.reference));
      const group = models.selectValue(model);
      const grouped = models.isGroupValue(group);
      // modelLabel: for a model with no voices of its own (a cloning model), the picker says so and lists
      // the recordings first.
      choice = await openVoicePicker({
        trigger, model: grouped ? group : model, modelLabel: models.modelLabel(grouped ? group : model), value, title: 'Choose a voice',
        include: { named: true, recordings: !!caps().voice_clone, voices: true },
        ...(grouped ? { voiceModels: groupModels(group, model) } : {}),
      });
    } finally {
      pickerOpen = false;
    }
    if (!choice) return null;
    // A choice for a model that is no longer the selected one (it changed while the picker was open) is dropped.
    if (s.loading) await whenSettled();
    if (s.model !== model || !s.data || s.data.model !== model) return null;
    if (choice.model && choice.model !== model) {
      await applyOnModel(choice);
      return choice;
    }
    await applyChoice(choice);
    return choice;
  }

  /** The Piper voice models the picker lists for the Piper entry, the selected one first (its named voice versions). */
  function groupModels(group, current) {
    const ids = models.groupIds(group, models.ttsModels(app.get().models.list));
    return [current, ...ids.filter((id) => id !== current)];
  }

  /** A choice that speaks with another Piper voice model: select that model with the voice chosen. */
  async function applyOnModel(choice) {
    let setup;
    if (choice.kind === 'named') setup = { identity: choice.id };
    else if (choice.kind === 'recording') setup = { reference: choice.voice, identity: '' };
    else setup = { voice: choice.voice, identity: '' };
    stopLive('Voice changed');
    const result = await selectModel(choice.model, setup);
    if (result && result.stale) return;
    notify(choice.kind === 'named' ? 'named' : 'model', result);
    announce(`Voice: ${choice.label}`);
  }

  /** The blend editor's inline error (a failed preview), '' to clear it. */
  function blendError(message) {
    if (!el.blendError) return;
    if (el.blendErrorText) el.blendErrorText.textContent = message || '';
    el.blendError.hidden = !message;
  }

  async function applyChoice(choice) {
    if (choice.kind === 'voice' || choice.kind === 'extra') {
      s.voice = choice.voice;
      s.mix = [];
      s.blendOpen = false;
      clearIdentity();
      edited('voice');
      announce(`Voice: ${choice.label}`);
    } else if (choice.kind === 'recording') {
      s.reference = choice.voice;
      clearIdentity();
      if (el.referenceError) el.referenceError.hidden = true;
      edited('reference');
      announce(`Reference recording: ${choice.label}`);
    } else if (choice.kind === 'named') {
      await applyNamed(choice.id);
    }
  }

  /** Pick a named voice: apply its version for this model (recipe and reference), keep the speed (SP-30). */
  async function applyNamed(id) {
    const model = s.model;
    let row = s.data && s.data.named.find((n) => n.id === id);
    if (!row) {
      try {
        const list = await voicesApi.listNamed({ model });
        if (s.model !== model) return;
        if (s.data) s.data.named = list;
        row = list.find((n) => n.id === id);
      } catch (error) {
        notice(errorBanner(error, "Couldn't load the named voices"));
        return;
      }
    }
    const realization = row ? realizationFor(row, model) : null;
    if (!row || row.available === false || !realization) {
      notice({ tone: 'error', title: "This named voice can't be used", message: 'This named voice has no available version for the selected model.' });
      return;
    }
    stopLive('Voice changed');
    const result = commit(s.data, { ...currentSetup(), voice: realization.voice, reference: realization.reference_audio_id || '', identity: id });
    notify('named', result);
    announce(`Voice: ${row.name}`);
  }

  function setMix(items, kind = 'blend') {
    s.mix = items;
    if (items.length) s.voice = items[0].voice;
    s.blendOpen = true;
    clearIdentity();
    blendError('');
    edited(kind);
  }

  async function addBlendVoice(trigger) {
    if (s.mix.length >= blend.MAX_VOICES) {
      announce(`A blend mixes up to ${blend.MAX_VOICES} voices.`);
      return;
    }
    if (pickerOpen || !(await afterLoad(trigger)) || caps().voice_blend !== true) return;
    const model = s.model;
    pickerOpen = true;
    let choice = null;
    try {
      choice = await openVoicePicker({
        trigger, model, value: '', mode: 'blend-add', title: 'Add a voice to the blend',
        exclude: s.mix.map((item) => item.voice),
      });
    } finally {
      pickerOpen = false;
    }
    if (!choice || !choice.voice || s.model !== model || s.loading) return;
    const base = s.mix.length ? s.mix : (s.voice ? [{ voice: s.voice, weight: 1 }] : []);
    setMix(blend.addVoice(base, choice.voice));
    announce(`Added ${choice.label}. ${blend.barLabel(s.mix, (v) => describe(v).label)}.`);
  }

  async function pasteBlend() {
    const answer = await prompt({
      title: 'Paste blend', label: 'Blend recipe', value: blend.toSpec(s.mix), maxLength: 600,
      hint: 'Voices joined with +, each with an optional weight: af_bella(2)+af_sky(1)',
      confirmLabel: 'Use blend',
      validate: (text) => {
        const result = blend.validateRecipe(text, { voices: s.data ? s.data.ids : [] });
        return result.error || null;
      },
    });
    if (answer === null) return;
    const result = blend.validateRecipe(answer, { voices: s.data ? s.data.ids : [] });
    if (result.error) return;
    setMix(result.items);
    announce(`Blend: ${blend.barLabel(s.mix, (v) => describe(v).label)}.`);
  }

  async function previewBlend() {
    const spec = blend.toSpec(s.mix);
    if (!spec) return;
    audio.prime();
    blendError('');
    setBusy(el.blendPreview, true);
    try {
      const { url } = await voicesApi.preview(s.model, spec);
      await audio.playClip({ id: 'preview', owner: 'preview', title: 'Blend preview', subtitle: spec, src: url, ephemeral: true });
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      // Next to the control that failed, not a toast.
      const message = error && error.name === 'CaptureActiveError' ? error.message : (explain(error).message || String(error?.message || error));
      blendError(`Couldn't preview the blend. ${message}`);
    } finally {
      setBusy(el.blendPreview, false);
    }
  }

  async function copyBlend() {
    const spec = blend.toSpec(s.mix);
    if (!spec) return;
    const ok = await copyText(spec);
    toast(ok ? 'Copied the blend recipe' : "Couldn't copy. Select the recipe and copy it.", { kind: ok ? 'success' : 'error' });
  }

  function onBlendRowClick(event) {
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = target && target.closest ? target.closest('button[data-blend-act]') : null;
    if (!button || button.disabled) return;
    const row = button.closest('[data-voice]');
    const voice = row ? row.getAttribute('data-voice') : '';
    const act = button.getAttribute('data-blend-act');
    if (act === 'up' || act === 'down') {
      setMix(blend.stepWeight(s.mix, voice, act === 'up' ? 1 : -1));
      const again = el.blendRows.querySelector(`[data-voice="${voice}"] [data-blend-act="${act}"]`);
      if (again && !again.disabled) again.focus();
      else {
        const input = el.blendRows.querySelector(`[data-voice="${voice}"] [data-blend-act="weight"]`);
        if (input) input.focus();
      }
    } else if (act === 'remove') {
      const index = s.mix.findIndex((item) => item.voice === voice);
      const label = describe(voice).label;
      setMix(blend.removeVoice(s.mix, voice));
      const rows = el.blendRows.querySelectorAll('[data-voice]');
      const next = rows[Math.min(index, rows.length - 1)];
      const focusTarget = next && s.mix.length > 1 ? next.querySelector('[data-blend-act="remove"]') : el.blendAdd;
      if (focusTarget) focusTarget.focus();
      announce(`Removed ${label}.`);
    }
  }

  function onBlendWeight(event) {
    const input = event.target;
    if (!input || input.getAttribute('data-blend-act') !== 'weight') return;
    const row = input.closest('[data-voice]');
    const voice = row ? row.getAttribute('data-voice') : '';
    const w = blend.clampWeight(input.value);
    if (Number.isNaN(w)) {
      const item = s.mix.find((i) => i.voice === voice);
      if (item) input.value = blend.weightText(item.weight);
      return;
    }
    setMix(blend.setWeight(s.mix, voice, w));
    const again = el.blendRows.querySelector(`[data-voice="${voice}"] [data-blend-act="weight"]`);
    if (again) again.value = blend.weightText(w);
  }

  function setSpeed(value) {
    if (caps().speed_control === false) return;
    const next = clampSpeed(value, caps());
    if (next === s.speed) {
      renderSpeed();
      return;
    }
    s.speed = next;
    renderSpeed();
    renderVoice();
    notify('speed');
  }

  function toggleEffect(type) {
    if (!EFFECT_TYPES.includes(type)) return;
    const fx = { ...s.effects };
    if (type === 'reverb') fx.reverb = { ...fx.reverb, on: !fx.reverb.on };
    else if (type === 'pitch') fx.pitch = { ...fx.pitch, on: !fx.pitch.on };
    else fx[type] = !fx[type];
    s.effects = fx;
    edited('effects');
  }

  function setPitch(value) {
    const n = Math.round(Number(value));
    const semitones = Number.isFinite(n) ? Math.min(PITCH_MAX, Math.max(PITCH_MIN, n)) : s.effects.pitch.semitones;
    s.effects = { ...s.effects, pitch: { ...s.effects.pitch, semitones } };
    edited('effects');
  }

  async function restoreDefault() {
    const def = app.get().models.defaults.tts;
    if (!def || el.restore.getAttribute('aria-busy') === 'true') return;
    const reselect = s.model !== def;   // "Restore Kokoro"; otherwise "Load Kokoro now" (the default is selected)
    stopLive('Voice changed');
    setBusy(el.restore, true, { label: reselect ? 'Restoring…' : 'Loading…' });
    try {
      const id = await models.restoreDefault();
      const result = await selectModel(id, { identity: '' });
      if (!result || !result.stale) notify('model');
      toast(`${defaultName(id)} ${reselect ? 'restored' : 'loaded'}`, { kind: 'success' });
    } catch (error) {
      notice(errorBanner(error, "Couldn't restore the default model"));
    } finally {
      setBusy(el.restore, false);
      renderModelState();
    }
  }

  /* ── wiring ── */

  on(el.summary, 'click', () => setOpen(!s.open));
  on(el.model, 'change', () => {
    const value = el.model.value;
    // The Piper entry selects the Piper voice used last, else a loaded, downloaded or the first one; its voice
    // picker lists the others.
    const st = app.get();
    const id = value === models.selectValue(s.model) ? s.model
      : models.modelForValue(value, models.ttsModels(st.models.list), { preferred: [lastInGroup, st.models.defaults.tts] });
    // Already this model (loaded or loading): nothing to do. Anything else, an inconsistent state included,
    // selects it.
    if (!id || (id === s.model && (s.loading || (s.data && s.data.model === id)))) return;
    stopLive('Voice changed');
    clearIdentity();
    // Only the named voice is cleared here; the voice, reference, speed, format, effects and instructions are
    // read when the model's data lands, so an edit made while it loads is kept (commit()).
    selectModel(id, { identity: '' }).then((result) => { if (!result || !result.stale) notify('model'); });
    notify('model-pending');
  });
  on(el.restore, 'click', () => { restoreDefault(); });
  on(el.voice, 'click', () => { openPicker(el.voice).catch(() => {}); });
  on(el.blendToggle, 'click', () => {
    if (!s.data || caps().voice_blend !== true) return;
    if (s.blendOpen && s.mix.length < 2) {
      // Close the editor again: a mix of one voice is that voice, so nothing the take uses changes.
      s.blendOpen = false;
      s.mix = [];
      renderBlend();
      return;
    }
    s.blendOpen = true;
    if (!s.mix.length && s.voice) s.mix = [{ voice: s.voice, weight: 1 }];
    renderBlend();
    if (el.blendAdd) el.blendAdd.focus();
  });
  on(el.blendAdd, 'click', () => { if (el.blendAdd.getAttribute('aria-disabled') !== 'true') addBlendVoice(el.blendAdd).catch(() => {}); });
  on(el.blendEqual, 'click', () => { setMix(blend.equalize(s.mix)); announce(`Equalized. ${blend.barLabel(s.mix, (v) => describe(v).label)}.`); });
  on(el.blendPaste, 'click', () => { pasteBlend().catch(() => {}); });
  on(el.blendPreview, 'click', () => { previewBlend(); });
  on(el.blendCopy, 'click', () => { copyBlend(); });
  on(el.blendRows, 'click', onBlendRowClick);
  on(el.blendRows, 'change', onBlendWeight);
  on(el.blendRows, 'keydown', (event) => { if (event.key === 'Enter' && event.target && event.target.getAttribute('data-blend-act') === 'weight') { event.preventDefault(); onBlendWeight(event); } });
  on(el.reference, 'change', () => {
    s.reference = el.reference.value;
    clearIdentity();
    if (el.referenceError) el.referenceError.hidden = true;
    if (el.reference) el.reference.removeAttribute('aria-invalid');
    edited('reference');
  });
  on(el.speed, 'input', () => setSpeed(el.speed.value));
  on(el.speedDown, 'click', () => setSpeed(s.speed - SPEED_STEP));
  on(el.speedUp, 'click', () => setSpeed(s.speed + SPEED_STEP));
  on(el.format, 'change', () => {
    s.format = FORMATS.some((f) => f.value === el.format.value) ? el.format.value : 'mp3';
    renderFormat();
    renderStream();
    notify('format');
  });
  on(el.effects, 'click', (event) => {
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const chip = target && target.closest ? target.closest('[data-effect]') : null;
    if (chip) toggleEffect(chip.getAttribute('data-effect'));
  });
  on(el.reverbRoom, 'change', () => {
    const room = REVERB_ROOMS.includes(el.reverbRoom.value) ? el.reverbRoom.value : 'small';
    s.effects = { ...s.effects, reverb: { ...s.effects.reverb, room } };
    edited('effects');
  });
  on(el.pitchDown, 'click', () => setPitch(s.effects.pitch.semitones - 1));
  on(el.pitchUp, 'click', () => setPitch(s.effects.pitch.semitones + 1));
  on(el.pitch, 'change', () => setPitch(el.pitch.value));
  on(el.instructions, 'input', () => {
    s.instructions = el.instructions.value;
    notify('instructions');
  });
  on(el.stream, 'change', () => {
    s.stream = !!el.stream.checked;
    notify('stream');
  });

  setOpen(false);
  renderAll();

  return {
    /** The live state (read-only by convention). */
    state: () => s,
    caps,
    currentSetup,
    describe,
    prepare,
    prepared,
    commitPrepared,
    selectModel,
    commit,
    openPicker,
    setOpen,
    isOpen: () => s.open,
    /** The input limit for this model (inventory SP-06). */
    limit: () => Number(caps().max_input_length) || DEFAULT_LIMIT,
    ready: () => !!s.data && !s.loading && s.data.model === s.model,
    loading: () => s.loading,
    /** The voice string Speak sends (named -> voice:<id>, blend -> canonical, else the voice or reference). */
    voiceSpec: () => voiceSpec({ identity: s.identity, blendSpec: s.mix.length >= 2 ? blend.toSpec(s.mix) : '', voice: s.voice, reference: s.reference }),
    voiceLabel: () => voiceDisplay(s, describe, { designing: designing() }).name,
    designing,
    /** Show the voice description field and focus it (Generate without a voice on a model that designs one). */
    focusDescription() {
      setOpen(true);
      if (el.advanced) el.advanced.open = true;
      if (el.instructions) el.instructions.focus();
    },
    summaryText,
    /** Show the reference field error (a clone model that needs one, inventory SP-16). */
    referenceError(message) {
      if (!el.referenceError) return;
      if (message) {
        setOpen(true);
        if (el.referenceErrorText) el.referenceErrorText.textContent = message;
        el.referenceError.hidden = false;
        el.reference.setAttribute('aria-invalid', 'true');
        el.reference.focus();
      } else {
        el.referenceError.hidden = true;
        el.reference.removeAttribute('aria-invalid');
      }
    },
    /** The blend editor with a given recipe (the screenshot hook). */
    showBlend(spec) {
      if (caps().voice_blend !== true) return false;
      setOpen(true);
      try {
        const items = blend.fromSpec(spec).filter((item) => !s.data.ids.length || s.data.ids.includes(item.voice));
        if (items.length) setMix(items, 'blend');
      } catch { /* the hook's sample is valid */ }
      return true;
    },
    /** app.models changed: rebuild the select, the state pill and the loading/empty states. */
    modelsChanged() {
      renderModelSelect();
      renderState();
    },
    /** Recordings or named voices changed elsewhere (voices:changed): drop the cache, refresh the lists. */
    async refreshAssets(what = '') {
      cache = new Map();
      if (!s.model || !s.data) return;
      const model = s.model;
      try {
        if (what === 'named' || !what) {
          const named = await voicesApi.listNamed({ model });
          if (s.model === model && s.data) {
            s.data.named = named;
            // A named voice deleted (or made unusable) elsewhere: keep its recipe, drop the stale selection.
            if (s.identity && !named.some((n) => n.id === s.identity && n.available !== false)) {
              clearIdentity();
              renderVoice();
              notify('commit', { warnings: ['identity-unavailable'] });
            }
          }
        }
        if ((what === 'recordings' || !what) && caps().voice_clone) {
          const recordings = await voicesApi.listRecordings();
          if (s.model !== model || !s.data) return;
          s.data.recordings = recordings;
          if (s.reference && !recordings.some((r) => (r.name || r.id) === s.reference)) s.reference = '';
          // Only the reference options change: typed instructions and the Advanced state stay (VL-19).
          renderReference();
          renderVoice();
        }
      } catch { /* a refresh failure keeps what is shown */ }
    },
    /** Forget cached model data (auth:changed, a stale view). */
    invalidate() { cache = new Map(); },
    render: renderAll,
    destroy() {
      for (const off of offs.splice(0)) off();
      if (speedSlider) speedSlider.destroy();
      if (optionsUi) optionsUi.destroy();
      optionsUi = null;
    },
  };
}
