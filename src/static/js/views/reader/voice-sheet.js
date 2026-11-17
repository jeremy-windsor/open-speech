/**
 * The voice sheet #rdp-voice-sheet: a model select (TTS models grouped by provider, Kokoro
 * first; the Piper voice models are one entry, Piper, whose voices the picker lists), the current choice as a row
 * (avatar, name, detail; a blend shows its canonical string in mono), and
 * [Choose voice], which opens the shared voice picker with named voices and provider voices plus one
 * suggested row, "Same as Speak · <voice>": the voice Speak has selected now, blends included, when its
 * model is the one chosen here (any Piper voice on the Piper entry). Raw recordings are not offered (REV-09): Reader renders send only
 * {model, voice}, so a cloned voice reaches the Reader as a named voice, which the server resolves with its
 * reference. [Use this voice] calls engine.setVoice(); without a model and a voice it says so (RE-66).
 * A model with no voices of its own and no named voice for it (Breeze TTS 2 before any is saved) gets a note with
 * a link to Voices › Design a voice.
 * @module views/reader/voice-sheet
 */
import { html, render } from '../../core/dom.js';
import { app } from '../../core/store.js';
import * as models from '../../core/models.js';
import { isBlend, withoutModel, listVoices, listNamed } from '../../core/voices.js';
import { openSheet, closeSheet } from '../../core/dialogs.js';
import { announce } from '../../core/a11y.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { openVoicePicker } from '../../components/voice-picker.js';
import { COPY, voiceLabel, voiceDetail, isRecordingChoice } from './play-logic.js';

/**
 * PURE: a detail line whose ' · ' separators never start a line: "American English · female" wraps after the
 * dot, not before it (a non-breaking space ties each dot to the word before it).
 */
export function keepSeparators(text) {
  return String(text || '').replace(/ · /g, '\u00a0· ');
}

/** What the picker may offer the Reader: named voices and provider voices, never raw recordings (REV-09). */
export const PICKER_INCLUDE = Object.freeze({ named: true, recordings: false, voices: true });

/**
 * PURE: the model select's groups: usable TTS models grouped by provider, in models.ttsModels() order (Kokoro,
 * Piper, then the other providers A–Z), without the models that only design voices (models.steadyTtsModels). The
 * Piper voice models are one option, models.PIPER_GROUP.
 * @returns {{label: string, options: {id: string, label: string}[]}[]}
 */
export function modelGroups(list) {
  const groups = [];
  for (const entry of models.selectEntries(models.steadyTtsModels(list || []))) {
    const label = models.PROVIDER_DISPLAY[entry.provider] || entry.provider;
    let group = groups.find((g) => g.label === label);
    if (!group) {
      group = { label, options: [] };
      groups.push(group);
    }
    group.options.push({ id: entry.id, label: entry.label });
  }
  return groups;
}

/**
 * PURE: the "Same as Speak" picker row for app.speakVoice, or null when Speak has no voice, uses another model
 * than the one chosen in the sheet (on the Piper entry, a voice that isn't Piper's), or reads with a raw reference
 * recording (REV-09: a Reader render would send the recording's name as a voice). The row carries Speak's model.
 * @param {{model?: string, voice?: string, label?: string, kind?: string}} speak
 * @param {string} model a model, or a model select value
 */
export function sameAsSpeakRow(speak, model) {
  if (!speak || !speak.model || !speak.voice || models.selectValue(speak.model) !== models.selectValue(model)) return null;
  if (isRecordingChoice(speak)) return null;
  const name = speak.label || voiceLabel({ model: speak.model, voice: speak.voice });
  return {
    id: 'same-as-speak',
    label: `Same as Speak · ${name}`,
    detail: isBlend(speak.voice) ? speak.voice : voiceDetail({ voice: speak.voice }),
    voice: speak.voice,
    group: 'Suggested',
    model: speak.model,
  };
}

/**
 * PURE: the note for a model with nothing to read with: no voices of its own and no named voice with an
 * available version for it; '' otherwise.
 * @param {string} label the model's display name
 * @param {{model: string, voices: object[], named: object[]}} loaded
 */
export function noVoicesNote(label, { model, voices = [], named = [] }) {
  if ((voices || []).length) return '';
  const usable = (named || []).some((identity) => identity && identity.available !== false
    && (identity.realizations || []).some((r) => r.model === model && r.available !== false));
  return usable ? '' : `${label} has no voices of its own. Design one in Voices, or save a take in Speak.`;
}

/**
 * @param {HTMLElement} root the player slot
 * @param {object} engine
 * @param {{onOpenModels?: () => void}} [options]
 * @returns {{open: (trigger?: Element, options?: {message?: string}) => void, close: () => void, isOpen: () => boolean,
 *   pending: () => {model: string, voice: string}, destroy: () => void}}
 */
export function bindVoiceSheet(root, engine, { onOpenModels = () => {} } = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const dialog = $('rdp-voice-sheet');
  const select = $('rdp-voice-model');
  const modelHint = $('rdp-voice-model-hint');
  const banner = $('rdp-voice-banner');
  const note = $('rdp-voice-note');
  const noteMsg = $('rdp-voice-note-msg');
  const avatar = $('rdp-voice-avatar');
  const name = $('rdp-voice-name');
  const detail = $('rdp-voice-detail');
  const choose = $('rdp-voice-choose');
  const use = $('rdp-voice-use');
  const error = $('rdp-voice-error');
  const errorText = $('rdp-voice-error-text');
  const modelsBtn = $('rdp-voice-models');
  const empty = $('rdp-voice-empty');
  const emptyText = $('rdp-voice-empty-text');
  const designLink = $('rdp-voice-design');
  // pending.model: the model the voice speaks with ('' on the Piper entry until a voice is chosen).
  let pending = { model: '', voice: '', label: '' };
  let picking = false;
  let checkToken = 0;
  const offs = [];
  const listen = (el, type, fn) => {
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  function fillModels(selected) {
    const list = app.get().models.list;
    const groups = modelGroups(list);
    if (modelHint) {
      modelHint.textContent = models.DESIGN_ONLY_HINT;
      modelHint.hidden = !models.ttsModels(list || []).some(models.designOnly);
    }
    render(select, groups.map((g) => html`<optgroup label="${g.label}">${g.options.map((o) => html`<option value="${o.id}">${o.label}</option>`)}</optgroup>`));
    const ids = groups.flatMap((g) => g.options.map((o) => o.id));
    const value = models.selectValue(selected);
    select.value = ids.includes(value) ? value : (ids[0] || '');
    select.disabled = !ids.length;
    return ids;
  }

  /** The pending choice for a select value: the Reader's voice when it is on that entry and offered, else none yet. */
  function pendingFor(value) {
    const current = engine.state.voice;
    const offered = !!current && models.steadyTtsModels(app.get().models.list || []).some((m) => m.id === current.model);
    if (offered && models.selectValue(current.model) === value) return { model: current.model, voice: current.voice, label: '' };
    return { model: models.isGroupValue(value) ? '' : value, voice: '', label: '' };
  }

  function describeChoose() {
    const ids = ['rdp-voice-name'];
    if (!error.hidden) ids.push('rdp-voice-error');
    if (empty && !empty.hidden) ids.push('rdp-voice-empty');
    choose.setAttribute('aria-describedby', ids.join(' '));
  }

  function showError(text) {
    errorText.textContent = text;
    error.hidden = !text;
    describeChoose();
  }

  function showEmpty(text) {
    if (!empty) return;
    if (emptyText) emptyText.textContent = text;
    empty.hidden = !text;
    describeChoose();
  }

  /** Whether the chosen model has anything to read with; the note says so when it has nothing. */
  async function checkVoices(model) {
    const token = (checkToken += 1);
    showEmpty('');
    if (!model || !empty) return;
    const [voices, named] = await Promise.allSettled([listVoices(model, { strict: true }), listNamed({ model })]);
    if (token !== checkToken || !dialog.open || voices.status !== 'fulfilled' || named.status !== 'fulfilled') return;
    const text = noVoicesNote(models.modelLabel(model), { model, voices: voices.value, named: named.value });
    showEmpty(text);
    if (text) announce(text);
  }

  function paintCurrent() {
    const chosen = !!pending.voice;
    const label = chosen ? (pending.label || engine.voiceLabel({ model: pending.model, voice: pending.voice })) : 'No voice chosen';
    name.textContent = label;
    const sub = chosen ? voiceDetail(pending) : 'Choose a voice to read with.';
    detail.textContent = keepSeparators(sub);
    detail.classList.toggle('mono', chosen && isBlend(pending.voice));
    render(avatar, html`${chosen ? monogram(withoutModel(pending.model, label)) : '?'}`);
    applyTile(avatar, pending.voice || 'none');
    avatar.classList.toggle('is-empty', !chosen);
    const chooseLabel = choose.querySelector('.btn__label');
    if (chooseLabel) chooseLabel.textContent = chosen ? 'Change voice' : 'Choose voice';
  }

  listen(select, 'change', () => {
    pending = pendingFor(select.value);
    showError('');
    paintCurrent();
    checkVoices(models.isGroupValue(select.value) ? '' : pending.model);
  });

  listen(choose, 'click', async () => {
    const value = select.value;
    if (picking || !value) {
      if (!value) showError(COPY.chooseBoth);
      return;
    }
    picking = true;
    try {
      const grouped = models.isGroupValue(value);
      const model = grouped ? value : pending.model;
      const extra = sameAsSpeakRow(app.get().speakVoice, model);
      // modelLabel: a model with no voices of its own gets the picker's note, as in Speak. On the Piper entry the
      // picker lists the Piper voice models as its voices, the one in use first (a named voice's version for it).
      const inUse = grouped && pending.model ? [pending.model] : [];
      const voiceModels = grouped
        ? [...inUse, ...models.groupIds(value, models.steadyTtsModels(app.get().models.list || [])).filter((id) => !inUse.includes(id))]
        : [];
      const choice = await openVoicePicker({
        trigger: choose,
        model,
        modelLabel: models.modelLabel(model),
        value: pending.voice,
        title: 'Choose a reading voice',
        include: PICKER_INCLUDE,
        extraRows: extra ? [extra] : [],
        ...(grouped ? { voiceModels } : {}),
      });
      if (choice && choice.voice && select.value === value) {
        const speak = app.get().speakVoice;
        const label = choice.kind === 'named' ? choice.label : (choice.kind === 'extra' && speak?.label ? speak.label : '');
        pending = { model: choice.model || model, voice: choice.voice, label };
        showError('');
        paintCurrent();
      }
    } finally {
      picking = false;
    }
  });

  listen(use, 'click', () => {
    if (!pending.model || !pending.voice) {
      showError(COPY.chooseBoth);
      choose.focus();
      return;
    }
    engine.setVoice({ model: pending.model, voice: pending.voice });
    closeSheet(dialog);
  });

  listen(modelsBtn, 'click', () => {
    closeSheet(dialog);
    onOpenModels();
  });

  // Design a voice: the sheet closes; the link's own hash change opens Voices › Design a voice.
  if (designLink) listen(designLink, 'click', () => closeSheet(dialog));

  return {
    open(trigger, { message = '' } = {}) {
      const ids = fillModels(engine.state.voice?.model || '');
      pending = pendingFor(select.value);
      banner.hidden = ids.length > 0;
      noteMsg.textContent = ids.length ? message : '';
      note.hidden = !ids.length || !message;
      showError('');
      paintCurrent();
      const focus = !ids.length ? modelsBtn : (pending.voice ? use : choose);
      openSheet(dialog, { trigger: trigger || undefined, initialFocus: focus });
      checkVoices(ids.length && !models.isGroupValue(select.value) ? pending.model : '');
    },
    close() { if (dialog.open) closeSheet(dialog); },
    isOpen: () => !!dialog.open,
    pending: () => ({ model: pending.model, voice: pending.voice }),
    destroy() { for (const off of offs.splice(0)) off(); },
  };
}
