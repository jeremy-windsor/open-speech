/**
 * Presets in Voices' one list (#voices, inventory SE-05): a preset sits on
 * the card of its named voice, or is a card of its own in the group "Presets without a named voice" (list.js
 * places them). Each shows its name, a summary ("1.1× · MP3 · Podcast EQ" on a voice's card, "Kokoro · Heart ·
 * 1.1× · MP3" on its own), a "Default" badge on the default preset, [Use in Speak] (the `apply-preset` intent),
 * [Edit] (the `edit-preset` intent: Speak applies it and shows its editing bar), a "Set as default"
 * aria-pressed toggle (POST /api/profiles/{id}/default; its label never changes with its state), [Rename] and a
 * menu with Rename… (PUT with the full payload, which the server requires) and Delete….
 * @module views/voices/presets
 */
import { html, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import { confirm, prompt } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { setIntent } from '../../core/store.js';
import { describeVoice, isBlend, parseBlend } from '../../core/voices.js';
import { speed as speedText } from '../../core/format.js';
import { idPart, modelName, cardMenuMarkup } from './shared.js';

export const NAME_CONFLICT = 'A preset with this name already exists. Choose another name.';
export const NAME_MAX = 100;

/** Effect names as Speak labels them. */
export const EFFECT_LABELS = Object.freeze({ normalize: 'Normalize', podcast_eq: 'Podcast EQ', reverb: 'Reverb', pitch: 'Pitch', robot: 'Robot' });

const enc = encodeURIComponent;

/** The fields PUT /api/profiles/{id} needs (the server replaces the preset with them). */
export const PRESET_FIELDS = Object.freeze(['name', 'backend', 'model', 'voice', 'speed', 'format', 'blend', 'reference_audio_id', 'voice_identity_id', 'instructions', 'effects']);

/**
 * The full payload of a preset with a new name: every stored field kept as it is (a rename must not reset
 * the voice, effects or instructions).
 */
export function renamePayload(preset, name) {
  const out = {};
  for (const key of PRESET_FIELDS) out[key] = preset[key] === undefined ? null : preset[key];
  out.name = name;
  if (!Array.isArray(out.effects)) out.effects = [];
  if (out.speed === null) out.speed = 1;
  if (!out.format) out.format = 'mp3';
  return out;
}

function voiceText(preset, namedById) {
  if (preset.voice_identity_id) {
    const identity = namedById.get(preset.voice_identity_id);
    return identity ? identity.name : 'Named voice';
  }
  const recipe = String(preset.blend || preset.voice || '');
  if (preset.reference_audio_id && (!recipe || recipe === preset.reference_audio_id)) return `Recording ${preset.reference_audio_id}`;
  if (isBlend(recipe)) {
    try {
      return `Blend of ${parseBlend(recipe).map((item) => describeVoice(item.voice).label).join(' and ')}`;
    } catch {
      return recipe;
    }
  }
  return describeVoice(recipe).label || recipe;
}

/**
 * The one-line summary of a preset: model · voice · speed · format · effects (· Instructions). On its named
 * voice's card (`identity`) the voice is left out, since the card names it, and so is the model when the voice
 * has only that one version: "1.1× · MP3 · Podcast EQ".
 * @param {object} preset a /api/profiles row
 * @param {object[]} [named] the named voices, to name a linked one
 * @param {{identity?: object|null}} [options] the named voice whose card shows the preset
 */
export function presetSummary(preset, named = [], { identity = null } = {}) {
  const namedById = new Map((named || []).map((item) => [item.id, item]));
  const model = preset.model || preset.backend;
  const parts = [];
  if (identity) {
    const versions = identity.realizations || [];
    if (!(versions.length === 1 && versions[0].model === model)) parts.push(modelName(model));
  } else {
    parts.push(modelName(model), voiceText(preset, namedById));
  }
  const speed = Number(preset.speed);
  if (Number.isFinite(speed) && speed > 0) parts.push(speedText(speed));
  if (preset.format) parts.push(String(preset.format).toUpperCase());
  for (const effect of preset.effects || []) {
    const type = typeof effect === 'string' ? effect : effect?.type;
    if (type) parts.push(EFFECT_LABELS[type] || type);
  }
  if (preset.instructions) parts.push('Instructions');
  return parts.filter(Boolean).join(' · ');
}

/** A preset's More actions menu. */
const ROW_MENU = Object.freeze([
  { action: 'rename', label: 'Rename…', icon: 'edit' },
  { action: 'delete', label: 'Delete…', icon: 'trash', danger: true },
]);

/**
 * One preset: a row on its named voice's card (`identity`), or a card of its own (`card`) in the group of presets
 * without a named voice. Its name is an h3 either way (under the voice's h2 or the group's).
 * @param {object} preset a /api/profiles row
 * @param {{isDefault?: boolean, named?: object[], identity?: object|null, card?: boolean}} [options]
 */
export function presetItemMarkup(preset, { isDefault = false, named = [], identity = null, card = false } = {}) {
  const key = `preset-${idPart(preset.id)}`;
  const inner = html`<div class="vo-preset__main">
      <div class="vo-preset__head"><h3 class="vo-preset__name" id="vo-${key}-title" tabindex="-1">${preset.name}</h3>${isDefault ? html`<span class="badge badge--accent vo-preset__badge">Default</span>` : ''}</div>
      <p class="vo-preset__meta">${presetSummary(preset, named, { identity })}</p>
    </div>
    <div class="vo-preset__actions">
      <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-use" data-action="preset-use" data-id="${preset.id}" aria-label="Use in Speak: ${preset.name}"><span class="btn__label">Use in Speak</span></button>
      <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-edit" data-action="preset-edit" data-id="${preset.id}" aria-label="Edit: ${preset.name}">${icon('edit')}<span class="btn__label">Edit</span></button>
      <button type="button" class="chip vo-default" id="vo-${key}-default" data-action="preset-default" data-id="${preset.id}" aria-pressed="${isDefault ? 'true' : 'false'}" aria-label="Set as default: ${preset.name}">${icon('check', { className: 'ico chip__check' })}<span>Set as default</span></button>
      <button type="button" class="btn btn--ghost btn--sm" id="vo-${key}-rename" data-action="preset-rename" data-id="${preset.id}" aria-label="Rename: ${preset.name}"><span class="btn__label">Rename</span></button>
      ${cardMenuMarkup(key, preset.name, ROW_MENU)}
    </div>
    <div class="vo-slot vo-preset__error" id="vo-${key}-error" hidden></div>`;
  if (card) return html`<li class="vo-grid__item"><article class="card vo-card vo-card--preset vo-preset" id="vo-${key}" aria-labelledby="vo-${key}-title">${inner}</article></li>`;
  return html`<li class="vo-preset" id="vo-${key}">${inner}</li>`;
}

/**
 * The presets' actions in the one list.
 * @param {{lists: object, load: (what: string) => Promise<boolean>, put: Function, changed: (what: string) => void,
 *   navigate: Function, listEl: HTMLElement, render: (options?: object) => void,
 *   error: (slot: string, error: *, title: string) => void, clearError: (slot: string) => void}} ctx
 */
export function createPresets(ctx) {
  const { listEl } = ctx;
  const list = () => ctx.lists.presets;
  const find = (id) => (list().items || []).find((p) => p.id === id) || null;
  const render = (options) => ctx.render(options);

  function rowError(id, error, title) {
    ctx.error(`vo-preset-${idPart(id)}-error`, error, title);
  }

  function clearRowError(id) {
    ctx.clearError(`vo-preset-${idPart(id)}-error`);
  }

  function fallbackAfter(index) {
    return () => {
      const names = listEl.querySelectorAll('.vo-preset__name');
      if (names.length) return names[Math.min(Math.max(index, 0), names.length - 1)];
      return listEl.querySelector('.vo-card__title, .empty__title') || null;
    };
  }

  /** Use in Speak: Speak resolves and validates the preset (intent apply-preset). */
  function use(button) {
    const id = button.getAttribute('data-id');
    if (!find(id)) return;
    setIntent('speak', { type: 'apply-preset', id });
    ctx.navigate('speak');
  }

  /** Edit: Speak applies the preset and shows its editing bar, with Update preset (intent edit-preset). */
  function edit(button) {
    const id = button.getAttribute('data-id');
    if (!find(id)) return;
    setIntent('speak', { type: 'edit-preset', id });
    ctx.navigate('speak');
  }

  /** Default toggle: POST …/default; its answer is the new list. The default can't be unset, only moved. */
  async function makeDefault(button) {
    const id = button.getAttribute('data-id');
    const preset = find(id);
    if (!preset) return;
    if (list().defaultId === id) {
      toast(`${preset.name} is already the default preset`, { kind: 'info', id: 'vo-default' });
      return;
    }
    clearRowError(id);
    setBusy(button, true);
    let data;
    try {
      data = await api.post(`/api/profiles/${enc(id)}/default`, undefined, { accept: 'json' });
    } catch (error) {
      setBusy(button, false);
      if (error?.kind === 'aborted') return;
      rowError(id, error, "Couldn't set the default preset");
      return;
    }
    setBusy(button, false);
    if (data && Array.isArray(data.profiles)) ctx.put('presets', data.profiles, { defaultId: data.default_profile_id || id });
    else await ctx.load('presets');
    render({ restore: button.id });
    toast(`${preset.name} is now the default preset`, { kind: 'success' });
    ctx.changed('presets');
  }

  /** Rename… : PUT with the full payload; a 409 re-opens the prompt with a field error. */
  async function rename(id, trigger) {
    const preset = find(id);
    if (!preset) return;
    let value = preset.name;
    let error = '';
    for (;;) {
      const name = await prompt({
        title: 'Rename preset', label: 'Preset name', value, error, maxLength: NAME_MAX, confirmLabel: 'Rename',
        validate: (v) => (v.length > NAME_MAX ? `Use at most ${NAME_MAX} characters.` : null),
      });
      if (name === null || name === preset.name) return;
      clearRowError(id);
      if (trigger) setBusy(trigger, true);
      try {
        await api.put(`/api/profiles/${enc(id)}`, renamePayload(preset, name), { accept: 'json' });
      } catch (err) {
        if (trigger) setBusy(trigger, false);
        if (err?.status === 409) {
          value = name;
          error = NAME_CONFLICT;
          continue;
        }
        if (err?.kind === 'aborted') return;
        rowError(id, err, "Couldn't rename the preset");
        return;
      }
      if (trigger) setBusy(trigger, false);
      toast(`Renamed ${preset.name} to ${name}`, { kind: 'success' });
      ctx.changed('presets');
      await ctx.load('presets');
      render({ fallback: fallbackAfter(0), restore: trigger ? trigger.id : '' });
      return;
    }
  }

  /** Delete… : confirm "Delete preset “N”?", then DELETE. */
  async function remove(id, trigger) {
    const preset = find(id);
    if (!preset) return;
    const isDefault = list().defaultId === id;
    const ok = await confirm({
      title: `Delete preset “${preset.name}”?`,
      message: isDefault ? 'It is the default preset, so Speak will open with your last setup instead.' : 'Named voices and recordings it uses are kept.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    const order = Array.from(listEl.querySelectorAll('.vo-preset')).map((row) => row.id);
    const index = Math.max(0, order.indexOf(`vo-preset-${idPart(id)}`));
    clearRowError(id);
    if (trigger) setBusy(trigger, true);
    try {
      await api.del(`/api/profiles/${enc(id)}`);
    } catch (error) {
      if (trigger) setBusy(trigger, false);
      if (error?.kind === 'aborted') return;
      rowError(id, error, "Couldn't delete");
      return;
    }
    toast(`Deleted preset ${preset.name}`, { kind: 'success' });
    ctx.changed('presets');
    await Promise.all([ctx.load('presets'), ctx.load('named')]);
    render({ fallback: fallbackAfter(index), restore: trigger ? trigger.id : '' });
  }

  function onMenu(action, item, button) {
    const key = (button.id.match(/^vo-preset-(.+)-more$/) || [])[1];
    const preset = (list().items || []).find((p) => idPart(p.id) === key);
    if (!preset) return;
    const run = action === 'rename' ? rename(preset.id, button) : action === 'delete' ? remove(preset.id, button) : null;
    if (run) run.catch((error) => toast(error?.message || String(error), { kind: 'error' }));
  }

  /** The visible Rename: the same prompt as the menu's Rename…. */
  function renameFrom(button) {
    const id = button.getAttribute('data-id');
    if (!find(id)) return;
    rename(id, button).catch((error) => toast(error?.message || String(error), { kind: 'error' }));
  }

  return {
    onMenu,
    actions: {
      'preset-use': (button) => use(button),
      'preset-edit': (button) => edit(button),
      'preset-default': (button) => makeDefault(button),
      'preset-rename': (button) => renameFrom(button),
    },
  };
}
