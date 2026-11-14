/**
 * The voice sheet #rdp-voice-sheet: a model select (TTS models grouped by provider, Kokoro
 * first), the current choice as a row (avatar, name, detail; a blend shows its canonical string in mono), and
 * [Choose voice], which opens the shared voice picker with named voices and provider voices plus one
 * suggested row, "Same as Speak · <voice>": the voice Speak has selected now, blends included, when its
 * model is the one chosen here. Raw recordings are not offered (REV-09): Reader renders send only
 * {model, voice}, so a cloned voice reaches the Reader as a named voice, which the server resolves with its
 * reference. [Use this voice] calls engine.setVoice(); without a model and a voice it says so (RE-66).
 * @module views/reader/voice-sheet
 */
import { html, render } from '../../core/dom.js';
import { app } from '../../core/store.js';
import * as models from '../../core/models.js';
import { isBlend } from '../../core/voices.js';
import { openSheet, closeSheet } from '../../core/dialogs.js';
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
 * Piper, then the other providers A–Z).
 * @returns {{label: string, options: {id: string, label: string}[]}[]}
 */
export function modelGroups(list) {
  const groups = [];
  for (const model of models.ttsModels(list || [])) {
    const provider = model.provider || models.providerOf(model.id);
    const label = models.PROVIDER_DISPLAY[provider] || provider;
    let group = groups.find((g) => g.label === label);
    if (!group) {
      group = { label, options: [] };
      groups.push(group);
    }
    group.options.push({ id: model.id, label: models.shortName(model) });
  }
  return groups;
}

/**
 * PURE: the "Same as Speak" picker row for app.speakVoice, or null when Speak has no voice, uses another model
 * than the one chosen in the sheet, or reads with a raw reference recording (REV-09: a Reader render would send
 * the recording's name as a voice).
 * @param {{model?: string, voice?: string, label?: string, kind?: string}} speak
 * @param {string} model
 */
export function sameAsSpeakRow(speak, model) {
  if (!speak || !speak.model || !speak.voice || speak.model !== model) return null;
  if (isRecordingChoice(speak)) return null;
  const name = speak.label || voiceLabel({ model: speak.model, voice: speak.voice });
  return {
    id: 'same-as-speak',
    label: `Same as Speak · ${name}`,
    detail: isBlend(speak.voice) ? speak.voice : voiceDetail({ voice: speak.voice }),
    voice: speak.voice,
    group: 'Suggested',
  };
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
  let pending = { model: '', voice: '', label: '' };
  let picking = false;
  const offs = [];
  const listen = (el, type, fn) => {
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  function fillModels(selected) {
    const groups = modelGroups(app.get().models.list);
    render(select, groups.map((g) => html`<optgroup label="${g.label}">${g.options.map((o) => html`<option value="${o.id}">${o.label}</option>`)}</optgroup>`));
    const ids = groups.flatMap((g) => g.options.map((o) => o.id));
    select.value = ids.includes(selected) ? selected : (ids[0] || '');
    select.disabled = !ids.length;
    return ids;
  }

  function showError(text) {
    errorText.textContent = text;
    error.hidden = !text;
    if (text) choose.setAttribute('aria-describedby', 'rdp-voice-name rdp-voice-error');
    else choose.setAttribute('aria-describedby', 'rdp-voice-name');
  }

  function paintCurrent() {
    const chosen = !!pending.voice;
    const label = chosen ? (pending.label || engine.voiceLabel({ model: pending.model, voice: pending.voice })) : 'No voice chosen';
    name.textContent = label;
    const sub = chosen ? voiceDetail(pending) : 'Choose a voice to read with.';
    detail.textContent = keepSeparators(sub);
    detail.classList.toggle('mono', chosen && isBlend(pending.voice));
    render(avatar, html`${chosen ? monogram(label) : '?'}`);
    applyTile(avatar, pending.voice || 'none');
    avatar.classList.toggle('is-empty', !chosen);
    const chooseLabel = choose.querySelector('.btn__label');
    if (chooseLabel) chooseLabel.textContent = chosen ? 'Change voice' : 'Choose voice';
  }

  listen(select, 'change', () => {
    const current = engine.state.voice;
    pending = { model: select.value, voice: current && current.model === select.value ? current.voice : '', label: '' };
    showError('');
    paintCurrent();
  });

  listen(choose, 'click', async () => {
    if (picking || !pending.model) {
      if (!pending.model) showError(COPY.chooseBoth);
      return;
    }
    picking = true;
    try {
      const extra = sameAsSpeakRow(app.get().speakVoice, pending.model);
      // modelLabel: a model with no voices of its own gets the picker's note, as in Speak.
      const choice = await openVoicePicker({
        trigger: choose,
        model: pending.model,
        modelLabel: models.modelLabel(pending.model),
        value: pending.voice,
        title: 'Choose a reading voice',
        include: PICKER_INCLUDE,
        extraRows: extra ? [extra] : [],
      });
      if (choice && choice.voice) {
        const speak = app.get().speakVoice;
        const label = choice.kind === 'named' ? choice.label : (choice.kind === 'extra' && speak?.label ? speak.label : '');
        pending = { model: pending.model, voice: choice.voice, label };
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

  return {
    open(trigger, { message = '' } = {}) {
      const current = engine.state.voice;
      const ids = fillModels(current?.model || '');
      pending = { model: select.value, voice: current && current.model === select.value ? current.voice : '', label: '' };
      banner.hidden = ids.length > 0;
      noteMsg.textContent = ids.length ? message : '';
      note.hidden = !ids.length || !message;
      showError('');
      paintCurrent();
      const focus = !ids.length ? modelsBtn : (pending.voice ? use : choose);
      openSheet(dialog, { trigger: trigger || undefined, initialFocus: focus });
    },
    close() { if (dialog.open) closeSheet(dialog); },
    isOpen: () => !!dialog.open,
    pending: () => ({ model: pending.model, voice: pending.voice }),
    destroy() { for (const off of offs.splice(0)) off(); },
  };
}
