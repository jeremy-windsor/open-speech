/**
 * Speak: presets, named voices and the History hand-off.
 *
 * - The presets row (#sp-presets): one chip per preset (the Default one with a star and "Default" in its
 *   name), pressed while it is the applied preset; any manual change clears it; "Save preset…" at the end.
 * - applyPreset(): GET /api/profiles/{id}/resolve, then the checks in today's order (model, reference,
 *   named voice, speed, instructions) BEFORE anything changes; a failure shows "Preset “X” can't be used"
 *   and changes nothing, so Speak never silently falls back to Kokoro (inventory SP-33).
 * - The Default preset applies once per page session, never over a preset the listener picked (SP-34).
 * - savePreset() and saveNamedVoice() ask first (dialogs.prompt), then build the payload (BUG-15).
 * - updatePreset() and updateNamedVoice(): the editing bar's Update preset and Update voice, with no prompt.
 * - historySetup(): an /api/history entry as a Speak setup (inventory HI-09).
 * @module views/speak/presets
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { get as apiGet, post as apiPost, put as apiPut, explain } from '../../core/api.js';
import * as voicesApi from '../../core/voices.js';
import * as models from '../../core/models.js';
import { prompt, confirm } from '../../core/dialogs.js';
import { emit } from '../../core/store.js';
import { toast } from '../../core/toast.js';
import { announce } from '../../core/a11y.js';

export const PRESET_NAME_MAX = 100;
export const VOICE_NAME_MAX = 200;
export const DUPLICATE_PRESET = 'A preset with this name already exists';

const enc = (v) => encodeURIComponent(String(v));

/** Per page session (SP-34): whether the Default preset was tried, and whether the listener picked one. */
let session = { defaultTried: false, picked: false };

/** @returns {{defaultTried: boolean, picked: boolean}} the page-session flags */
export function sessionFlags() {
  return session;
}

/** The listener chose a setup themselves (a preset chip, an intent): the Default preset no longer applies. */
export function markPicked() {
  session.picked = true;
}

/** Test hook. */
export function __resetForTests() {
  session = { defaultTried: false, picked: false };
}

/* ───────────────────────────── pure helpers ───────────────────────────── */

/** The provider a model belongs to (the preset's `backend`). */
export function providerFor(modelId) {
  return models.providerOf(modelId);
}

/**
 * PURE: the POST /api/profiles payload (inventory SP-35, today's shape).
 * @param {string} name
 * @param {{model: string, voice?: string, blend?: string, identity?: string, reference?: string, speed?: number,
 *   format?: string, effects?: object[]|null, instructions?: string}} s voice is the provider voice; blend the
 *   canonical recipe of a 2+ voice blend
 */
export function presetPayload(name, s) {
  const provider = providerFor(s.model);
  return {
    name: String(name || '').trim(),
    backend: provider,
    provider,
    model: s.model,
    voice: s.voice || s.reference || '',
    speed: Number(s.speed) || 1,
    format: s.format || 'mp3',
    blend: s.blend || null,
    reference_audio_id: s.reference || null,
    voice_identity_id: s.identity || null,
    instructions: String(s.instructions || '').trim() || null,
    effects: Array.isArray(s.effects) ? s.effects : [],
  };
}

/**
 * PURE: the model a new preset with a named voice starts on: the model Speak is on when the voice
 * has a version for it, else the voice's first model (in Speak's model order) that has one; '' when none can.
 * @param {object|null} identity a named voice with its realizations
 * @param {string[]} ttsIds the usable models, ranked
 * @param {string} [current] the model Speak is on
 */
export function newPresetModel(identity, ttsIds = [], current = '') {
  const versions = (identity && identity.realizations) || [];
  const has = (model) => versions.some((r) => r.model === model && r.available !== false);
  if (current && ttsIds.includes(current) && has(current)) return current;
  return ttsIds.find(has) || '';
}

/** PURE: how many presets follow a named voice's version for a model (they are linked to it, on that model). */
export function followingCount(presetList, identityId, model) {
  return (presetList || []).filter((p) => p && p.voice_identity_id === identityId && (p.model || p.backend) === model).length;
}

/** PURE: the toast after Update voice: "Updated Narrator for Kokoro. 2 presets follow it." */
export function updatedVoiceMessage(name, modelLabel, count) {
  const head = `Updated ${name} for ${modelLabel}`;
  if (!count) return head;
  return `${head}. ${count === 1 ? '1 preset follows it' : `${count} presets follow it`}.`;
}

/** PURE: a preset name problem, or null (non-empty, at most 100 characters). */
export function presetNameProblem(text) {
  const name = String(text || '').trim();
  if (!name) return 'Enter a name for the preset.';
  if (Array.from(name).length > PRESET_NAME_MAX) return `Use ${PRESET_NAME_MAX} characters or fewer.`;
  return null;
}

/**
 * PURE: why a resolved preset can't be used, checked in today's order, or '' when it can (inventory SP-33).
 * @param {object} profile the /resolve answer
 * @param {{ttsIds: string[], referenceMissing?: boolean, named?: object[]|null, caps?: object|null}} ctx
 */
export function presetProblem(profile, { ttsIds = [], referenceMissing = false, named = null, caps = null } = {}) {
  const model = profile.model || profile.backend || '';
  if (model && !ttsIds.includes(model)) return `the model ${models.modelLabel(model)} is unavailable`;
  if (profile.reference_audio_id && referenceMissing) return `the reference recording ${profile.reference_audio_id} is missing`;
  if (profile.voice_identity_id && named) {
    const identity = named.find((n) => n.id === profile.voice_identity_id);
    if (!identity || identity.available === false) return 'its named voice is unavailable for this model';
  }
  if (caps && caps.speed_control === false && Number(profile.speed ?? 1) !== 1) return `${models.modelLabel(model)} has a fixed speed`;
  if (caps && profile.instructions && !caps.instructions) return `${models.modelLabel(model)} doesn't take instructions`;
  return '';
}

/** PURE: a resolved preset as a Speak setup. The resolved `voice` is already the blend or the named voice's recipe. */
export function presetSetup(profile) {
  return {
    model: profile.model || profile.backend || '',
    voice: String(profile.voice || ''),
    identity: profile.voice_identity_id || '',
    reference: profile.reference_audio_id || '',
    speed: Number(profile.speed ?? 1),
    format: profile.format || 'mp3',
    effects: Array.isArray(profile.effects) ? profile.effects : [],
    instructions: profile.instructions || '',
  };
}

/**
 * PURE: an /api/history entry as a Speak setup (inventory HI-09, today's reGenerateTTS):
 * {error} for an entry that can't be used, else {text, setup, referenceFallback}. The reference comes from
 * meta.reference_voice; an older entry without it uses its voice as the reference only on a clone model with
 * no stock voices (referenceFallback, decided once the model's voices are known). Instructions are the
 * entry's, never what is typed now; effects only when the entry recorded them.
 */
export function historySetup(entry, { ttsIds = [] } = {}) {
  const e = entry || {};
  if (e.type && e.type !== 'tts') return { error: 'Only speech entries can be re-generated.' };
  if (e.model && !ttsIds.includes(e.model)) return { error: `This entry's model ${models.modelLabel(e.model)} is unavailable.` };
  const meta = e.meta && typeof e.meta === 'object' ? e.meta : {};
  const voice = String(e.voice || '');
  const identity = voice.startsWith('voice:') ? voice.slice(6) : '';
  const setup = {
    model: e.model || '',
    voice: identity ? '' : voice,
    identity,
    reference: 'reference_voice' in meta ? String(meta.reference_voice || '') : undefined,
    instructions: String(meta.instructions || ''),
    speed: Number(e.speed) || 1,
  };
  if (e.format) setup.format = e.format;
  if ('effects' in meta) setup.effects = Array.isArray(meta.effects) ? meta.effects : [];
  return {
    text: String(e.full_text ?? e.text_preview ?? ''),
    setup,
    referenceFallback: 'reference_voice' in meta || identity ? null : voice,
  };
}

/* ───────────────────────────── network steps ───────────────────────────── */

/** True when GET /api/voices/library/{name} answers 404. Other failures throw. */
async function referenceMissing(name) {
  try {
    await apiGet(`/api/voices/library/${enc(name)}`, { accept: 'json' });
    return false;
  } catch (error) {
    if (error && error.status === 404) return true;
    throw error;
  }
}

/**
 * Apply a preset: resolve, check, then change everything at once.
 * @param {string} id
 * @param {{inspector: object, ttsIds: string[], stopLive: (reason: string) => void, name?: string}} ctx
 * @returns {Promise<{ok: true, profile: object, warnings: Array}|{ok: false, reason: string, error?: Error, stale?: boolean}>}
 */
export async function applyPreset(id, { inspector, ttsIds, stopLive, name = '' }) {
  let profile;
  try {
    profile = await apiGet(`/api/profiles/${enc(id)}/resolve`, { accept: 'json' });
  } catch (error) {
    return { ok: false, reason: explain(error).message || String(error?.message || error), error };
  }
  const label = profile.name || name || 'this preset';
  const model = profile.model || profile.backend || '';
  let problem = presetProblem(profile, { ttsIds });
  if (problem) return { ok: false, reason: problem, name: label };
  try {
    if (profile.reference_audio_id && await referenceMissing(profile.reference_audio_id)) {
      return { ok: false, reason: presetProblem(profile, { ttsIds, referenceMissing: true }), name: label };
    }
    const prepared = await inspector.prepared(model);
    problem = presetProblem(profile, { ttsIds, named: prepared.data.named, caps: prepared.data.caps });
    if (problem) return { ok: false, reason: problem, name: label };
    if (!prepared.current()) return { ok: false, stale: true, reason: '' };
    if (typeof stopLive === 'function') stopLive('Preset changed');
    const result = inspector.commitPrepared(prepared, presetSetup(profile));
    if (result.stale) return { ok: false, stale: true, reason: '' };
    return { ok: true, profile, warnings: result.warnings };
  } catch (error) {
    return { ok: false, reason: explain(error).message || String(error?.message || error), error, name: label };
  }
}

/**
 * Save preset… (inventory SP-35): ask the name first, then POST; a 409 asks again with the message.
 * @param {{setup: object, onSaved?: (profile: object) => void, value?: string}} ctx value: the name the prompt
 *   starts with (a new preset with a named voice starts with the voice's name)
 * @returns {Promise<object|null>} the created preset, or null when cancelled
 */
export async function savePreset({ setup, onSaved, value: start = '' }) {
  let error = '';
  let value = String(start || '');
  for (;;) {
    const name = await prompt({
      title: 'Save preset', label: 'Preset name', value, maxLength: PRESET_NAME_MAX, error,
      hint: 'Saves the model, voice, speed, format, effects and instructions.',
      validate: presetNameProblem, confirmLabel: 'Save preset',
    });
    if (name === null) return null;
    try {
      const created = await apiPost('/api/profiles', presetPayload(name, setup), { accept: 'json' });
      emit('voices:changed', { what: 'presets' });
      toast(`Saved preset “${name}”`, { kind: 'success' });
      if (typeof onSaved === 'function') onSaved(created || null);
      return created || null;
    } catch (failure) {
      if (failure && failure.status === 409) {
        error = DUPLICATE_PRESET;
        value = name;
        continue;
      }
      throw failure;
    }
  }
}

/**
 * Update preset (the editing bar): PUT /api/profiles/{id} with the full payload of Speak's current
 * setup, under the preset's own name (the server replaces the preset with it), then tell the other views.
 * @param {string} id
 * @param {string} name the preset's name
 * @param {object} setup as presetPayload() takes it
 * @returns {Promise<object>} the updated preset
 */
export async function updatePreset(id, name, setup) {
  const updated = await apiPut(`/api/profiles/${enc(id)}`, presetPayload(name, setup), { accept: 'json' });
  emit('voices:changed', { what: 'presets' });
  return updated;
}

/**
 * Update voice (the editing bar): PUT the named voice's version for this model with Speak's
 * recipe and reference, no name prompt. The server moves the presets linked to that version with it.
 * @param {{id: string, model: string, recipe: string, reference?: string}} version
 * @returns {Promise<object>} the server's answer
 */
export async function updateNamedVoice({ id, model, recipe, reference = '' }) {
  const body = { model, voice: recipe, reference_audio_id: reference || null };
  const saved = await apiPut(`/api/voices/identities/${enc(id)}/realizations/${enc(model)}`, body, { accept: 'json' });
  emit('voices:changed', { what: 'named' });
  return saved;
}

/**
 * Save named voice… (inventory SP-31): ask the name (an existing name, matched case-insensitively, adds or
 * updates this model's version), create the identity when new, confirm before replacing a version, then
 * POST or PUT the version, reselect it and announce it.
 * @param {{model: string, recipe: string, reference: string, current?: string,
 *   onSaved?: (identity: object) => (void|Promise<void>)}} ctx
 * @returns {Promise<object|null>} the identity, or null when cancelled
 */
export async function saveNamedVoice({ model, recipe, reference = '', current = '', onSaved }) {
  let error = '';
  let value = current;
  for (;;) {
    const name = await prompt({
      title: 'Save named voice', label: 'Voice name', value, maxLength: VOICE_NAME_MAX, error,
      hint: "An existing name adds or updates this model's version",
      confirmLabel: 'Save voice',
    });
    if (name === null) return null;
    const catalog = await voicesApi.listNamed({ check: false });
    let identity = catalog.find((item) => String(item.name || '').toLowerCase() === name.toLowerCase()) || null;
    if (!identity) {
      try {
        identity = await apiPost('/api/voices/identities', { name }, { accept: 'json' });
      } catch (failure) {
        if (failure && failure.status === 409) {
          error = failure.message || 'A named voice with this name already exists';
          value = name;
          continue;
        }
        throw failure;
      }
    }
    const replace = (identity.realizations || []).some((item) => item.model === model);
    if (replace) {
      const ok = await confirm({
        title: 'Update named voice?',
        message: `Update “${identity.name}” for ${models.modelLabel(model)}? Linked presets will use the updated voice.`,
        confirmLabel: 'Update',
      });
      if (!ok) return null;
    }
    const body = { model, voice: recipe, reference_audio_id: reference || null };
    const base = `/api/voices/identities/${enc(identity.id)}/realizations`;
    if (replace) await apiPut(`${base}/${enc(model)}`, body, { accept: 'json' });
    else await apiPost(base, body, { accept: 'json' });
    emit('voices:changed', { what: 'named' });
    if (typeof onSaved === 'function') await onSaved(identity);
    toast(`Saved ${identity.name} for ${models.modelLabel(model)}`, { kind: 'success' });
    return identity;
  }
}

/* ───────────────────────────── the presets row ───────────────────────────── */

/** A chip's identity across renders: ['id', presetId] or ['act', 'save'|'retry']. */
function chipKey(button) {
  if (!button || !button.getAttribute) return null;
  const id = button.getAttribute('data-preset-id');
  if (id !== null) return ['id', id];
  const act = button.getAttribute('data-preset-act');
  return act !== null ? ['act', act] : null;
}

/**
 * The presets row.
 * @param {HTMLElement} group #sp-presets (role=group)
 * @param {{onApply: (id: string, name: string) => void, onSave: () => void}} hooks
 */
export function createPresetsRow(group, hooks = {}) {
  const st = {
    list: [], defaultId: null, applied: '', loaded: false, loading: false, error: null, loadedAt: 0, token: 0,
    saveBlocked: false, saveHint: '',
  };
  let renderedSig = null;

  const chipsIn = () => (group ? [...group.querySelectorAll('button[data-preset-id], button[data-preset-act]')] : []);

  /** Pressed state on the chips that are there: applying a preset never rebuilds the row (focus stays). */
  function syncPressed() {
    for (const chip of chipsIn()) {
      const id = chip.getAttribute('data-preset-id');
      if (id !== null) chip.setAttribute('aria-pressed', String(id === st.applied));
    }
  }

  /** "Save preset…" is aria-disabled, described by the reason, while Speak has no voice setup to save. */
  function syncSave() {
    const save = group ? group.querySelector('[data-preset-act="save"]') : null;
    if (!save) return;
    save.setAttribute('aria-disabled', String(st.saveBlocked));
    if (st.saveBlocked && st.saveHint) save.setAttribute('aria-describedby', st.saveHint);
    else save.removeAttribute('aria-describedby');
  }

  function render_() {
    if (!group) return;
    const sig = JSON.stringify([st.list.map((p) => [p.id, p.name]), st.defaultId, !st.loaded && st.loading, !!st.error && !st.loaded]);
    if (sig === renderedSig) {
      // Nothing to rebuild (a refresh with the same presets): the chips, and the focus on one, stay.
      syncPressed();
      syncSave();
      return;
    }
    renderedSig = sig;
    // A rebuild (presets added, renamed or removed) puts the keyboard focus back on the same chip.
    const active = typeof document !== 'undefined' ? document.activeElement : null;
    const before = chipsIn();
    const focusedIndex = before.indexOf(active);
    const key = focusedIndex >= 0 ? chipKey(active) : null;
    const chips = st.list.map((p) => {
      const isDefault = p.id === st.defaultId;
      return html`<button type="button" class="chip sp-preset" aria-pressed="${String(p.id === st.applied)}" data-preset-id="${p.id}">${isDefault ? icon('star-fill', { className: 'ico sp-preset__star' }) : ''}<span class="sp-preset__name">${p.name}</span>${isDefault ? html`<span class="visually-hidden">, Default</span>` : ''}</button>`;
    });
    let status = '';
    if (!st.loaded && st.loading) status = html`<span class="sp-presets__status" aria-hidden="true">Loading presets…</span>`;
    else if (st.error && !st.loaded) status = html`<span class="sp-presets__status">Presets didn't load.</span><button type="button" class="chip chip--action" data-preset-act="retry">${icon('retry')}Retry</button>`;
    render(group, html`${chips}${status}<button type="button" class="chip chip--action" data-preset-act="save">${icon('plus')}Save preset…</button>`);
    syncSave();
    if (!key) return;
    const after = chipsIn();
    const same = after.find((chip) => {
      const k = chipKey(chip);
      return k && k[0] === key[0] && k[1] === key[1];
    });
    // The focused chip went away (a deleted preset, a Retry that worked): its neighbour takes the focus.
    const target = same || after[Math.min(focusedIndex, after.length - 1)] || null;
    if (target) target.focus({ preventScroll: true });
  }

  async function load() {
    const token = (st.token += 1);
    st.loading = true;
    if (!st.loaded) render_();
    try {
      const { presets, defaultId } = await voicesApi.listPresets();
      if (token !== st.token) return { stale: true };
      st.list = presets;
      st.defaultId = defaultId;
      st.loaded = true;
      st.error = null;
      st.loadedAt = Date.now();
      // Refreshing keeps the applied preset; a deleted one falls back to none (inventory SP-32).
      if (st.applied && !presets.some((p) => p.id === st.applied)) st.applied = '';
      return { presets, defaultId };
    } catch (error) {
      if (token !== st.token) return { stale: true };
      st.error = error;
      return { error };
    } finally {
      if (token === st.token) {
        st.loading = false;
        render_();
      }
    }
  }

  const onClick = (event) => {
    const target = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const chip = target && target.closest ? target.closest('[data-preset-id], [data-preset-act]') : null;
    if (!chip || chip.disabled) return;
    const act = chip.getAttribute('data-preset-act');
    if (act === 'save') {
      if (st.saveBlocked) {
        if (typeof hooks.onBlocked === 'function') hooks.onBlocked(chip);
        return;
      }
      if (typeof hooks.onSave === 'function') hooks.onSave(chip);
      return;
    }
    if (act === 'retry') {
      load();
      return;
    }
    const id = chip.getAttribute('data-preset-id');
    const preset = st.list.find((p) => p.id === id);
    if (preset && typeof hooks.onApply === 'function') hooks.onApply(id, preset.name);
  };
  if (group) group.addEventListener('click', onClick);
  render_();

  return {
    load,
    list: () => st.list.slice(),
    defaultId: () => st.defaultId,
    loaded: () => st.loaded,
    loadedAt: () => st.loadedAt,
    error: () => st.error,
    find: (id) => st.list.find((p) => p.id === id) || null,
    applied: () => st.applied,
    /** Mark a preset applied ('' for none). Only aria-pressed changes, so a focused chip keeps the focus. */
    setApplied(id) {
      const next = id || '';
      if (next === st.applied) return;
      st.applied = next;
      syncPressed();
    },
    /**
     * Block "Save preset…" while there is no voice setup to save; `hintId` is the element that says why
     * (aria-describedby). A click then calls hooks.onBlocked instead of hooks.onSave.
     */
    setSaveBlocked(blocked, hintId = '') {
      st.saveBlocked = !!blocked;
      st.saveHint = hintId;
      syncSave();
    },
    destroy() { if (group) group.removeEventListener('click', onClick); },
  };
}

/** The announcement after a preset applied. */
export function appliedMessage(name) {
  return `Preset ${name} applied`;
}

/** Announce politely (kept here so the view and tests share the copy). */
export function announceApplied(name) {
  announce(appliedMessage(name));
}
