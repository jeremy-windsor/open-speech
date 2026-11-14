/**
 * Recordings (#voices/recordings, inventory VL-11…VL-18): one card per reference recording with
 * "8.4 s · 24 kHz · mono", a two-line transcript snippet (or "No transcript" in warn tone), and a badge per
 * clone model (greyed with a reason when the model needs a transcript and there is none). Actions:
 * [Preview] (the recording itself, ephemeral), [Use in Speak] (the `use-recording` intent with the model
 * from #vo-clone-model), [Test clone] (POST /v1/audio/speech with voice and voice_library_ref set to the
 * name, WAV, no speed; failures show on the card), and a menu with Edit transcript… (#vo-transcript-sheet,
 * PATCH), Save as preset… (POST /api/profiles) and Delete… (the confirm names the presets that use it).
 *
 * While the microphone records (any view), Preview and Test clone are unavailable and point to #vo-rec-note,
 * "Stop recording to listen." Playback and capture changes patch the cards' buttons in place,
 * so an open card menu, focus and the clone-model select are never rebuilt under the listener.
 * @module views/voices/recordings
 */
import { html, render as renderInto, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as audio from '../../core/audio.js';
import * as models from '../../core/models.js';
import { confirm, prompt, openSheet, closeSheet } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { app, setIntent } from '../../core/store.js';
import { applyTile, monogram } from '../../core/tiles.js';
import {
  TEST_SENTENCE, idPart, modelName, cloneModels, matches, listNames, loadingMarkup, loadErrorMarkup,
  staleErrorMarkup, fillDetails, loadErrorDetail, renderKeepingFocus, focusElement, cardMenuMarkup, bindMenus,
  openMenuFocus, restoreMenuFocus, actionError, showBanner, clearBanner, createSlots,
} from './shared.js';
import { NAME_CONFLICT, NAME_MAX } from './presets.js';

const enc = encodeURIComponent;
/** What the list is called in its loading and error copy. */
const WHAT = 'recordings';

export const NO_CLONE_MODEL = 'No clone model available';

/** The id of the panel's "Stop recording to listen." note (voices.html). */
export const REC_NOTE = 'vo-rec-note';

/** "6.0 s · 24 kHz · mono" (VL-11). */
export function recordingMeta(rec) {
  const parts = [];
  const d = Number(rec?.duration_s);
  parts.push(Number.isFinite(d) && rec?.duration_s !== null && rec?.duration_s !== undefined ? `${d.toFixed(1)} s` : 'Length unknown');
  if (rec?.sample_rate) parts.push(`${Math.round(Number(rec.sample_rate) / 100) / 10} kHz`);
  if (rec?.channels) parts.push(Number(rec.channels) === 1 ? 'mono' : `${rec.channels} channels`);
  return parts.join(' · ');
}

/**
 * The compatibility of a recording with each clone model: ok, or not with the reason (VL-14: a model that
 * needs the exact words can't use a recording without a transcript).
 * @returns {{id: string, label: string, ok: boolean, reason: string}[]}
 */
export function compatibility(rec, cloneList) {
  const hasTranscript = !!String(rec?.transcript || '').trim();
  return (cloneList || []).map((m) => {
    const needs = !!(m.capabilities && m.capabilities.clone_transcript_required);
    const ok = !needs || hasTranscript;
    return { id: m.id, label: modelName(m.id), ok, reason: ok ? '' : 'needs a transcript' };
  });
}

/** The presets that use a recording as their reference (VL-16). */
export function presetsUsing(name, presets) {
  return (presets || []).filter((p) => p && p.reference_audio_id === name);
}

/**
 * The JSON body of a saved recording's clone test (VL-14): voice and voice_library_ref are the name, WAV, the
 * fixed sentence, and no speed (the server's 1.0 passes models without speed control).
 */
export function cloneTestBody(name, model) {
  return { model, voice: name, input: TEST_SENTENCE, response_format: 'wav', voice_library_ref: name };
}

/** POST /api/profiles for "Save as preset…" (VL-17): today's clone preset payload. */
export function clonePresetPayload(presetName, recordingName, model) {
  return {
    name: presetName,
    backend: models.providerOf(model),
    model,
    voice: recordingName,
    speed: 1.0,
    format: 'wav',
    blend: null,
    reference_audio_id: recordingName,
    effects: [],
  };
}

/**
 * What a card's Preview and Test clone buttons say, from what plays now (exported for tests).
 * playback.preview / playback.test: {name, playing} of the recording whose preview or clone test is the
 * current audio session ('' when none). A session that is paused (autoplay refused, or paused from the lock
 * screen) offers Play; a playing one offers Stop. The accessible name always starts with the visible label.
 */
export function playbackButtons(name, { testing = '', playback = {} } = {}) {
  const preview = playback.preview || {};
  const test = playback.test || {};
  const previewOn = preview.name === name && !!preview.playing;
  let testState = 'idle';
  if (testing === name) testState = 'busy';
  else if (test.name === name) testState = test.playing ? 'playing' : 'paused';
  return {
    preview: previewOn
      ? { label: 'Stop', aria: `Stop preview: ${name}`, icon: 'stop' }
      : { label: 'Preview', aria: `Preview: ${name}`, icon: 'play' },
    test: {
      busy: { label: 'Testing…', aria: `Testing: ${name}`, icon: '' },
      playing: { label: 'Stop test', aria: `Stop test: ${name}`, icon: 'stop' },
      paused: { label: 'Play test', aria: `Play test: ${name}`, icon: 'play' },
      idle: { label: 'Test clone', aria: `Test clone: ${name}`, icon: 'waveform' },
    }[testState],
    testState,
  };
}

/**
 * The state of a card's buttons and its hint (exported for tests): label, accessible name, icon, busy,
 * disabled and description of Preview and Test clone, whether Use in Speak can run, and the reason Test
 * clone can't (shown on the card when a clone model is chosen).
 * @param {{name: string, transcript?: string}} rec
 * @param {{cloneList?: object[], chosen?: string, testing?: string, playback?: object, pending?: string,
 *   recording?: boolean}} [options] pending: the recording whose preview audio is loading; recording: the
 *   microphone is open (any view)
 */
export function cardControls(rec, { cloneList = [], chosen = '', testing = '', playback = {}, pending = '', recording = false } = {}) {
  const name = rec.name;
  const transcript = String(rec.transcript || '').trim();
  const buttons = playbackButtons(name, { testing, playback });
  const busy = buttons.testState === 'busy';
  const live = buttons.testState === 'playing' || buttons.testState === 'paused';
  const chosenModel = cloneList.find((m) => m.id === chosen);
  const blocked = live ? '' : (!chosenModel ? NO_CLONE_MODEL
    : (chosenModel.capabilities?.clone_transcript_required && !transcript ? `${modelName(chosen)} requires an exact reference transcript` : ''));
  const hintId = `vo-rec-${idPart(name)}-hint`;
  const loading = pending === name;
  // Nothing plays while the microphone records; stopping what plays is always allowed.
  const previewMuted = recording && buttons.preview.icon !== 'stop';
  const testMuted = recording && !busy && buttons.testState !== 'playing';
  return {
    preview: {
      ...buttons.preview,
      icon: loading ? '' : buttons.preview.icon,
      busy: loading,
      disabled: previewMuted,
      describedby: previewMuted ? REC_NOTE : '',
    },
    use: { disabled: !chosenModel },
    test: {
      ...buttons.test,
      busy,
      disabled: (!!testing && !busy) || !!blocked || testMuted,
      describedby: [testMuted ? REC_NOTE : '', blocked ? hintId : ''].filter(Boolean).join(' '),
    },
    hint: { id: hintId, text: blocked, visible: !!blocked && !!chosenModel },
  };
}

function buttonInner(state) {
  return html`${state.icon ? icon(state.icon) : ''}<span class="btn__label">${state.label}</span>`;
}

function stateAttrs(state) {
  return html` aria-label="${state.aria}" data-icon="${state.icon}"${state.busy ? html` aria-busy="true"` : ''}${state.disabled ? html` aria-disabled="true"` : ''}${state.describedby ? html` aria-describedby="${state.describedby}"` : ''}`;
}

/** A card's More actions menu: Save as preset… stays listed without a clone model, with the reason. */
function cardMenu(hasModel) {
  return [
    { action: 'transcript', label: 'Edit transcript…', icon: 'edit' },
    { action: 'preset', label: 'Save as preset…', icon: 'plus', disabled: hasModel ? '' : NO_CLONE_MODEL },
    { action: 'delete', label: 'Delete…', icon: 'trash', danger: true },
  ];
}

function cardMarkup(rec, options) {
  const key = `rec-${idPart(rec.name)}`;
  const transcript = String(rec.transcript || '').trim();
  const compat = compatibility(rec, options.cloneList);
  const c = cardControls(rec, options);
  return html`<li class="vo-grid__item"><article class="card vo-card vo-rec" id="vo-${key}" aria-labelledby="vo-${key}-title">
    <div class="vo-card__head">
      <span class="avatar vo-card__avatar" data-tile="${rec.name}" aria-hidden="true">${monogram(rec.name.replace(/_/g, ' '))}</span>
      <div class="vo-card__heading">
        <h2 class="vo-card__title mono" id="vo-${key}-title" tabindex="-1">${rec.name}</h2>
        <p class="vo-card__meta tabular">${recordingMeta(rec)}</p>
      </div>
      ${cardMenuMarkup(key, rec.name, cardMenu(options.cloneList.length > 0))}
    </div>
    ${transcript
    ? html`<p class="vo-rec__transcript">${transcript}</p>`
    : html`<p class="vo-rec__transcript vo-rec__transcript--missing">${icon('alert', { className: 'ico ico--sm' })}<span>No transcript. Clone models that need the exact words can't use it.</span></p>`}
    ${compat.length ? html`<p class="vo-rec__compat"><span class="visually-hidden">Clone models: </span>${compat.map((m) => html`<span class="badge${m.ok ? ' badge--ok' : ' vo-badge--off'}">${m.ok ? icon('check') : ''}${m.label}${m.ok ? '' : html` · ${m.reason}`}</span>`)}</p>` : ''}
    <div class="vo-card__actions">
      <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-preview" data-action="rec-preview" data-name="${rec.name}"${stateAttrs(c.preview)}>${buttonInner(c.preview)}</button>
      <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-use" data-action="rec-use" data-name="${rec.name}" aria-label="Use in Speak: ${rec.name}"${c.use.disabled ? html` aria-disabled="true" aria-describedby="vo-clone-model-hint"` : ''}><span class="btn__label">Use in Speak</span></button>
      <button type="button" class="btn btn--ghost btn--sm" id="vo-${key}-test" data-action="rec-test" data-name="${rec.name}"${stateAttrs(c.test)}>${buttonInner(c.test)}</button>
    </div>
    <p class="${c.hint.visible ? 'field__hint vo-card__hint' : 'visually-hidden'}" id="${c.hint.id}"${c.hint.text ? '' : html` hidden`}>${c.hint.text}</p>
    <p class="status-text vo-card__status" id="vo-${key}-status" hidden></p>
    <div class="vo-slot vo-card__error" id="vo-${key}-error" hidden></div>
  </article></li>`;
}

/**
 * The markup of the recordings list for a state (exported for tests).
 * @param {{status: string, items: object[], error: *}} list
 * @param {{query?: string, cloneList?: object[], testing?: string, chosen?: string, pending?: string,
 *   recording?: boolean,
 *   playback?: {preview?: {name: string, playing: boolean}, test?: {name: string, playing: boolean}}}} [options]
 */
export function recordingsListMarkup(list, { query = '', cloneList = [], testing = '', playback = {}, chosen = '', pending = '', recording = false } = {}) {
  if (!list.at && (list.status === 'loading' || list.status === 'idle')) return loadingMarkup(WHAT, { kind: 'card', rows: 2 });
  if (list.status === 'error' && !list.at) return loadErrorMarkup(list.error, { what: WHAT, list: 'recordings' });
  const items = list.items || [];
  const banner = list.status === 'error' ? staleErrorMarkup(list.error, { what: WHAT, list: 'recordings' }) : '';
  if (!items.length) {
    return html`${banner}<div class="card vo-empty-card"><div class="empty">${icon('mic', { className: 'ico empty__icon' })}<h2 class="empty__title">No recordings yet</h2><p class="empty__text">Record or upload a short clip of a voice, then clone it with a clone model.</p><button type="button" class="btn btn--primary" data-action="clone">${icon('plus')}<span class="btn__label">Clone your first voice</span></button></div></div>`;
  }
  const shown = items.filter((r) => matches(query, r.name, r.transcript));
  if (!shown.length) {
    return html`${banner}<div class="empty vo-nomatch">${icon('search', { className: 'ico empty__icon' })}<h2 class="empty__title">No recordings match “${query.trim()}”</h2><button type="button" class="btn btn--secondary" data-action="clear-search"><span class="btn__label">Clear search</span></button></div>`;
  }
  const options = { cloneList, testing, playback, chosen, pending, recording };
  return html`${banner}<ul class="vo-grid" aria-label="Recordings">${shown.map((rec) => cardMarkup(rec, options))}</ul>`;
}

/** The clone-test session of the Recordings cards (the wizard's own test has homeRoute voices/clone). */
const TEST_HOME = 'voices/recordings';

/**
 * Which of this panel's sessions is the current audio session, and whether it plays: '' when none, else
 * "preview:1", "test:0"… A store selector, so the cards are patched only when it changes (not at 4 Hz).
 */
export function playbackKey(player) {
  if (!player || player.owner !== 'voices') return '';
  if (player.id === 'preview') return `preview:${player.playing ? 1 : 0}`;
  if (player.id === 'clone-test' && player.homeRoute === TEST_HOME) return `test:${player.playing ? 1 : 0}`;
  return '';
}

/** Screenshot hook only: stand-in clone models (the preview server has none running). */
const SHOT_MODELS = Object.freeze([
  { id: 'chatterbox', capabilities: { voice_clone: true, clone_transcript_required: true } },
  { id: 'fish', capabilities: { voice_clone: true } },
]);
/** Screenshot hook only: a recording without a transcript, shown when the library has none. */
const SHOT_RECORDING = Object.freeze({ name: 'sample_take', duration_s: 12.4, sample_rate: 48000, channels: 1, transcript: null });

function setAttr(el, name, value) {
  if (value) el.setAttribute(name, value);
  else el.removeAttribute(name);
}

/**
 * The recordings panel.
 * @param {HTMLElement} panel #vo-recs
 * @param {HTMLDialogElement} sheet #vo-transcript-sheet
 * @param {{lists: object, load: Function, models: () => object, query: () => string, changed: Function,
 *   navigate: Function, announce: (text: string) => void}} ctx
 */
export function createRecordings(panel, sheet, ctx) {
  const listEl = panel.querySelector('#vo-rec-list');
  const select = panel.querySelector('#vo-clone-model');
  const selectHint = panel.querySelector('#vo-clone-model-hint');
  const note = panel.querySelector(`#${REC_NOTE}`);
  const sheetEls = {
    form: sheet ? sheet.querySelector('#vo-transcript-form') : null,
    text: sheet ? sheet.querySelector('#vo-transcript-edit') : null,
    forName: sheet ? sheet.querySelector('#vo-transcript-for') : null,
    error: sheet ? sheet.querySelector('#vo-transcript-error') : null,
    save: sheet ? sheet.querySelector('#vo-transcript-save') : null,
  };
  let unbind = () => {};
  let chosen = '';          // the clone model the listener picked (kept across refreshes, VL-13)
  let testing = '';         // the recording whose clone test is being generated
  let testPlaying = '';     // the recording whose clone test is the current audio session
  let previewing = '';      // the recording whose preview is the current audio session
  let previewPending = '';  // the recording whose preview audio is loading
  let previewSeq = 0;       // the newest Preview press wins
  let editing = '';         // the recording in the transcript sheet
  let editingStart = '';
  let sheetSession = 0;     // which opening of the transcript sheet a save belongs to
  let modelSig = '';
  let selectSig = '';
  let shot = false;         // ?shot=with-clone-models is showing (view-local stand-ins; actions are inert)
  const slots = createSlots(panel);

  const list = () => ctx.lists.recordings;
  const find = (name) => (list().items || []).find((r) => r.name === name) || null;
  const cloneList = () => (shot ? SHOT_MODELS : cloneModels(ctx.models().list));
  const recordingNow = () => !!app.get().recording;
  /** The recordings on screen: the library, plus the screenshot hook's stand-in (never written to the list). */
  const displayed = () => {
    const items = list().items || [];
    return shot && !items.some((r) => !String(r.transcript || '').trim()) ? [...items, SHOT_RECORDING] : items;
  };

  /** Fill #vo-clone-model: the clone models, keeping the listener's choice (VL-13). Untouched when they didn't change. */
  function renderSelect() {
    if (!select) return;
    const options = cloneList();
    const loaded = !!(ctx.models().fetchedAt || ctx.models().list.length);
    const sig = `${loaded ? 1 : 0}|${options.map((m) => m.id).join(',')}`;
    if (sig === selectSig) return;   // re-rendering the options would close the select if it is open
    selectSig = sig;
    const keep = chosen || select.value;
    if (!options.length) {
      renderInto(select, html`<option value="">${loaded ? NO_CLONE_MODEL : 'Loading models…'}</option>`);
      select.value = '';
      select.disabled = true;
      if (selectHint) {
        renderInto(selectHint, loaded
          ? html`No model that can clone a voice is running. <a href="#models/tts">Open Models</a> to add one.`
          : html`Use in Speak and Test clone use this model.`);
      }
    } else {
      renderInto(select, html`${options.map((m) => html`<option value="${m.id}">${modelName(m.id)}</option>`)}`);
      select.value = options.some((m) => m.id === keep) ? keep : options[0].id;
      select.disabled = false;
      if (selectHint) selectHint.textContent = 'Use in Speak and Test clone use this model.';
    }
    if (select.value) chosen = select.value;
  }

  /** The preview and clone test this panel started, and whether each plays now (app.player). */
  function playback() {
    const key = playbackKey(app.get().player);
    return {
      preview: { name: previewing, playing: key === 'preview:1' },
      test: { name: testPlaying, playing: key === 'test:1' },
    };
  }

  function controlOptions() {
    return {
      cloneList: cloneList(), chosen: select ? select.value : chosen, testing, playback: playback(),
      pending: previewPending, recording: recordingNow(),
    };
  }

  /**
   * Re-render the list. restore: the id of the control that had focus before an async action (focus may
   * have dropped to <body> while it was disabled); when it is gone, fallback() picks the next target. A card
   * menu that had focus opens again on the same item.
   */
  function render({ fallback, restore = '' } = {}) {
    const menuFocus = openMenuFocus(listEl);
    unbind();
    renderSelect();
    const state = list();
    const view = shot ? { ...state, items: displayed() } : state;
    listEl.setAttribute('aria-busy', state.status === 'loading' ? 'true' : 'false');
    renderKeepingFocus(listEl, recordingsListMarkup(view, { query: ctx.query(), ...controlOptions() }), { fallback, wasFocused: restore });
    if (state.status === 'error' && !state.at) fillDetails(listEl, loadErrorDetail(state.error, WHAT));
    for (const tile of listEl.querySelectorAll('[data-tile]')) applyTile(tile, tile.getAttribute('data-tile'));
    slots.restore(listEl);
    if (note) note.hidden = !recordingNow();
    modelSig = modelsSignature();
    unbind = bindMenus(listEl, onMenu);
    restoreMenuFocus(unbind, menuFocus, listEl, fallback);
  }

  function applyButton(button, state) {
    if (!button) return;
    const label = button.querySelector('.btn__label');
    if (!label || label.textContent !== state.label || (button.getAttribute('data-icon') || '') !== state.icon) {
      renderInto(button, buttonInner(state));
      button.setAttribute('data-icon', state.icon);
    }
    button.setAttribute('aria-label', state.aria);
    setAttr(button, 'aria-busy', state.busy ? 'true' : '');
    setAttr(button, 'aria-disabled', state.disabled ? 'true' : '');
    setAttr(button, 'aria-describedby', state.describedby);
  }

  /**
   * Bring every card's Preview, Use in Speak and Test clone buttons (and its hint) up to date in place: a
   * preview or test starting or ending, the microphone opening or closing, or another clone model chosen
   * never rebuilds the list (an open card menu, focus and the select stay as they are).
   */
  function patchControls() {
    const options = controlOptions();
    const shown = displayed();
    for (const button of listEl.querySelectorAll('[data-action="rec-preview"]')) {
      const name = button.getAttribute('data-name');
      const rec = shown.find((r) => r.name === name);
      if (!rec) continue;
      const c = cardControls(rec, options);
      const key = `vo-rec-${idPart(name)}`;
      applyButton(button, c.preview);
      applyButton(listEl.querySelector(`#${key}-test`), c.test);
      const use = listEl.querySelector(`#${key}-use`);
      if (use) {
        setAttr(use, 'aria-disabled', c.use.disabled ? 'true' : '');
        setAttr(use, 'aria-describedby', c.use.disabled ? 'vo-clone-model-hint' : '');
      }
      const hint = listEl.querySelector(`#${c.hint.id}`);
      if (hint) {
        hint.textContent = c.hint.text;
        hint.hidden = !c.hint.text;
        hint.className = c.hint.visible ? 'field__hint vo-card__hint' : 'visually-hidden';
      }
    }
    if (note) note.hidden = !options.recording;
  }

  function modelsSignature() {
    return cloneList().map((m) => `${m.id}:${m.capabilities?.clone_transcript_required ? 1 : 0}`).join(',')
      + (ctx.models().fetchedAt ? '|loaded' : '');
  }

  /** Re-render when the clone models change (not on every inventory poll). */
  function onModels() {
    if (modelsSignature() === modelSig) return;
    render();
  }

  /**
   * The session this panel started for `name`, when it is still the current one: `kind` 'preview' or 'test'.
   * Stop when it plays; play it (synchronously, inside the click) when it is paused, for example after the
   * browser refused to start it on its own.
   * @returns {boolean} true when the click was handled
   */
  function toggleOwn(kind, name) {
    const session = audio.current();
    const ours = session && session.owner === 'voices' && session.controls && (kind === 'preview'
      ? session.id === 'preview'
      : session.id === 'clone-test' && session.homeRoute === TEST_HOME);
    if (!ours) return false;
    const playing = typeof session.getState === 'function' && !!session.getState().playing;
    if (playing) session.controls.stop();
    else session.controls.play();
    return true;
  }

  function cardEl(name, part) {
    return listEl.querySelector(`#vo-rec-${idPart(name)}-${part}`);
  }

  /** A card's status line (kept across re-renders; announced politely). */
  function cardStatus(name, text) {
    slots.text(listEl, `vo-rec-${idPart(name)}-status`, text);
    if (text) ctx.announce(text);
  }

  function cardError(name, error, title) {
    const e = actionError(error, title);
    if (cardEl(name, 'error')) slots.error(listEl, `vo-rec-${idPart(name)}-error`, e);
    else toast(`${e.title}: ${e.message}`, { kind: 'error' });
    return e;
  }

  function clearCardError(name) {
    slots.clear(listEl, `vo-rec-${idPart(name)}-error`);
  }

  /* ───────────── Preview (VL-12) ───────────── */

  async function preview(button) {
    const name = button.getAttribute('data-name');
    if (!find(name)) return;
    if (previewPending === name) return;   // its audio is still loading: one download per recording
    if (previewing === name) {
      if (toggleOwn('preview', name)) return;
      previewing = '';
    }
    if (recordingNow()) {   // nothing plays while the microphone records (the button says so)
      patchControls();
      return;
    }
    audio.prime();   // synchronously in the click: the clip plays after a network wait (iOS)
    clearCardError(name);
    const token = (previewSeq += 1);
    previewPending = name;
    patchControls();
    let media = null;
    try {
      media = await api.mediaUrl(`/api/voices/library/${enc(name)}/audio`);
      if (token !== previewSeq) {   // a later Preview press took over while this audio loaded
        media.revoke();
        return;
      }
      previewPending = '';
      previewing = name;
      const release = media;
      await audio.playClip({
        id: 'preview', owner: 'voices', title: name, subtitle: 'Recording', src: media.url, ephemeral: true,
        onRelease: () => {
          release.revoke();
          if (previewing === name) {
            previewing = '';
            patchControls();
          }
        },
      });
      media = null;
      patchControls();
    } catch (error) {
      if (media) media.revoke();
      if (token !== previewSeq) return;
      if (previewPending === name) previewPending = '';
      if (previewing === name) previewing = '';
      patchControls();
      if (error?.kind === 'aborted') return;
      cardError(name, error, error?.name === 'CaptureActiveError' ? "Couldn't play" : "Couldn't play the recording");
    }
  }

  /* ───────────── Use in Speak (VL-18) ───────────── */

  function use(button) {
    const name = button.getAttribute('data-name');
    if (!find(name)) return;
    const model = select ? select.value : chosen;
    if (!model || !cloneList().some((m) => m.id === model)) {
      cardError(name, new Error(model ? `Clone model ${model} is unavailable` : NO_CLONE_MODEL), "Couldn't open in Speak");
      return;
    }
    setIntent('speak', { type: 'use-recording', name, model });
    ctx.navigate('speak');
  }

  /* ───────────── Test clone (VL-14) ───────────── */

  async function test(button) {
    const name = button.getAttribute('data-name');
    const rec = find(name);
    if (!rec || testing) return;
    if (testPlaying === name) {
      if (toggleOwn('test', name)) return;   // Stop test / Play test
      testPlaying = '';
    }
    if (recordingNow()) {   // checked before prime(), the model load and the synthesis
      patchControls();
      return;
    }
    clearCardError(name);
    cardStatus(name, '');
    const modelId = select ? select.value : chosen;
    const model = cloneList().find((m) => m.id === modelId);
    if (!model) {
      cardError(name, new Error(NO_CLONE_MODEL), 'Clone test failed');
      return;
    }
    if (model.capabilities?.clone_transcript_required && !String(rec.transcript || '').trim()) {
      cardError(name, new Error(`${modelName(modelId)} requires an exact reference transcript`), 'Clone test failed');
      return;
    }
    // Synchronously in the click, once the test will run (a refused click leaves what plays alone): the
    // result plays after the synthesis, outside the gesture (iOS).
    audio.prime();
    const label = modelName(modelId);
    testing = name;
    patchControls();
    cardStatus(name, `Generating a clone test with ${label}…`);
    let url = '';
    try {
      const ok = await models.ensureReady(modelId, { kind: 'tts' });
      if (!ok) {
        testing = '';
        patchControls();
        cardStatus(name, 'Clone test canceled.');
        return;
      }
      const blob = await api.request('/v1/audio/speech', { method: 'POST', json: cloneTestBody(name, modelId), accept: 'blob' });
      models.markUsed(modelId);
      url = URL.createObjectURL(blob);
      const playUrl = url;
      testing = '';
      testPlaying = name;   // before playClip: the session it claims patches the card to "Stop test"
      patchControls();
      cardStatus(name, `Clone test complete with ${label}.`);
      await audio.playClip({
        id: 'clone-test', owner: 'voices', title: 'Clone test', subtitle: `${name} · ${label}`, homeRoute: TEST_HOME,
        src: playUrl,
        onRelease: () => {
          URL.revokeObjectURL(playUrl);
          if (testPlaying === name) {
            testPlaying = '';
            patchControls();
          }
        },
      });
      url = '';
    } catch (error) {
      if (url) URL.revokeObjectURL(url);
      testing = '';
      if (testPlaying === name) testPlaying = '';
      patchControls();
      cardStatus(name, '');   // "Generating…" is over; a failure is said once, by the card's alert banner
      if (error?.kind === 'aborted') return;
      if (error?.name !== 'CaptureActiveError') models.markFailed(modelId);
      cardError(name, error, 'Clone test failed');
    }
  }

  /* ───────────── Edit transcript… (VL-15) ───────────── */

  function openTranscript(name, trigger) {
    const rec = find(name);
    if (!rec || !sheet) return;
    editing = name;
    editingStart = String(rec.transcript || '');
    sheetSession += 1;
    if (sheetEls.forName) sheetEls.forName.textContent = name;
    if (sheetEls.text) sheetEls.text.value = editingStart;
    clearBanner(sheetEls.error);
    openSheet(sheet, {
      trigger,
      initialFocus: sheetEls.text,
      backdropClose: () => !sheetEls.text || sheetEls.text.value === editingStart,
      onClose: () => { editing = ''; },
    });
  }

  async function saveTranscript(event) {
    if (event) event.preventDefault();
    const name = editing;
    if (!name) return;
    const session = sheetSession;
    const value = sheetEls.text ? sheetEls.text.value.trim() : '';
    clearBanner(sheetEls.error);
    if (value === editingStart.trim()) {   // nothing changed: nothing to send
      closeSheet(sheet, 'cancel');
      return;
    }
    // Cancel can close the sheet while the request runs; the outcome is then said by a toast, never by a
    // banner in a closed sheet.
    const stillOpen = () => !!(sheet && sheet.open) && sheetSession === session && editing === name;
    setBusy(sheetEls.save, true);
    try {
      await api.patch(`/api/voices/library/${enc(name)}`, { transcript: value || null }, { accept: 'json' });
    } catch (error) {
      setBusy(sheetEls.save, false);
      if (error?.kind === 'aborted') return;
      if (stillOpen()) {
        showBanner(sheetEls.error, actionError(error, "Couldn't save the transcript"));
      } else {
        const e = actionError(error, `Couldn't save the transcript for ${name}`);
        toast(`${e.title}: ${e.message}`, { kind: 'error' });
      }
      return;
    }
    setBusy(sheetEls.save, false);
    if (stillOpen()) closeSheet(sheet, 'saved');
    toast(value ? `Updated the transcript for ${name}` : `Removed the transcript for ${name}`, { kind: 'success' });
    ctx.changed('recordings');
    await ctx.load('recordings');
    render();
  }

  /* ───────────── Save as preset… (VL-17) ───────────── */

  async function saveAsPreset(name, trigger) {
    const modelId = select ? select.value : chosen;
    if (!modelId || !cloneList().some((m) => m.id === modelId)) {
      cardError(name, new Error(NO_CLONE_MODEL), "Couldn't save the preset");
      return;
    }
    let value = `${name} clone`;
    let error = '';
    for (;;) {
      const presetName = await prompt({
        title: 'Save as preset', label: 'Preset name', value, error, maxLength: NAME_MAX,
        hint: `Uses ${name} with ${modelName(modelId)}.`,
        validate: (v) => (v.length > NAME_MAX ? `Use at most ${NAME_MAX} characters.` : null),
      });
      if (presetName === null) return;
      if (trigger) setBusy(trigger, true);
      try {
        await api.post('/api/profiles', clonePresetPayload(presetName, name, modelId), { accept: 'json' });
      } catch (err) {
        if (trigger) setBusy(trigger, false);
        if (err?.status === 409) {
          value = presetName;
          error = NAME_CONFLICT;
          continue;
        }
        if (err?.kind === 'aborted') return;
        cardError(name, err, "Couldn't save the preset");
        return;
      }
      if (trigger) setBusy(trigger, false);
      toast(`Saved preset ${presetName}`, { kind: 'success' });
      ctx.changed('presets');
      ctx.load('presets');
      return;
    }
  }

  /* ───────────── Delete… (VL-16) ───────────── */

  async function remove(name, trigger) {
    const rec = find(name);
    if (!rec) return;
    const using = presetsUsing(name, ctx.lists.presets.items);
    const ok = await confirm({
      title: `Delete recording ${name}?`,
      message: using.length
        ? `Presets using it: ${listNames(using.map((p) => p.name))}. They won't work until you choose another recording.`
        : 'Its audio and transcript are removed.',
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    const order = Array.from(listEl.querySelectorAll('.vo-rec')).map((card) => card.id);
    const index = Math.max(0, order.indexOf(`vo-rec-${idPart(name)}`));
    if (trigger) setBusy(trigger, true);
    try {
      await api.del(`/api/voices/library/${enc(name)}`);
    } catch (error) {
      if (trigger) setBusy(trigger, false);
      if (error?.kind === 'aborted') return;
      cardError(name, error, "Couldn't delete");
      return;
    }
    if (previewing === name) {
      const s = audio.current();
      if (s && s.id === 'preview' && s.controls) s.controls.stop();
    }
    toast(`Deleted recording ${name}`, { kind: 'success' });
    ctx.changed('recordings');
    await ctx.load('recordings');
    render({
      restore: trigger ? trigger.id : '',
      fallback: () => {
        const titles = listEl.querySelectorAll('.vo-card__title');
        if (titles.length) return titles[Math.min(index, titles.length - 1)];
        return listEl.querySelector('.empty__title');
      },
    });
  }

  function onMenu(action, item, button) {
    if (shot) return;   // the screenshot state's stand-ins are not real recordings or models
    const key = (button.id.match(/^vo-rec-(.+)-more$/) || [])[1];
    const rec = (list().items || []).find((r) => idPart(r.name) === key);
    if (!rec) return;
    let run = null;
    if (action === 'transcript') openTranscript(rec.name, button);
    else if (action === 'preset') run = saveAsPreset(rec.name, button);
    else if (action === 'delete') run = remove(rec.name, button);
    if (run) run.catch((error) => toast(error?.message || String(error), { kind: 'error' }));
  }

  const onSelectChange = () => {
    chosen = select.value;
    patchControls();
  };
  const onSubmit = (event) => { saveTranscript(event).catch((error) => toast(error?.message || String(error), { kind: 'error' })); };
  if (select) select.addEventListener('change', onSelectChange);
  if (sheetEls.form) sheetEls.form.addEventListener('submit', onSubmit);
  // Preview and Test clone say Stop while their audio plays and Play when it is paused, and they are
  // unavailable while the microphone records: patch the buttons on those transitions only (the selectors
  // change a few times per clip, never per timeupdate).
  const offPlayer = app.select((s) => playbackKey(s.player), () => patchControls());
  const offRecording = app.select((s) => !!s.recording, () => patchControls());

  return {
    render,
    onModels,
    actions: {
      'rec-preview': (button) => (shot ? undefined : preview(button)),
      'rec-use': (button) => (shot ? undefined : use(button)),
      'rec-test': (button) => (shot ? undefined : test(button)),
    },
    /** Close the transcript sheet (the view is being hidden). */
    closeSheets() {
      if (sheet && sheet.open) closeSheet(sheet, 'cancel');
    },
    /**
     * The panel is no longer on screen: stop its preview. A preview is ephemeral (no mini-player), so once its
     * Stop button is out of sight nothing could stop it. A clone test is a normal clip and keeps playing, with
     * the mini-player.
     */
    stopPreview() {
      previewSeq += 1;   // a preview still loading never starts off screen
      previewPending = '';
      const session = audio.current();
      if (previewing && session && session.id === 'preview' && session.owner === 'voices' && session.controls) session.controls.stop();
    },
    /**
     * Screenshot hook (?shot=with-clone-models): the cards as they look with clone models running, one that
     * needs a transcript and one that doesn't, plus a recording without a transcript. Display only: the
     * stand-ins never reach the recordings list, the model inventory or the server, and the cards' actions do
     * nothing until the route shows without the shot (clearShot()).
     */
    applyShot(name) {
      if (name !== 'with-clone-models') return false;
      shot = true;
      selectSig = '';
      render();
      return true;
    },
    /** Leave the screenshot state: the real clone models and recordings again. */
    clearShot() {
      if (!shot) return;
      shot = false;
      selectSig = '';
      render();
    },
    /** Screenshot hook and tests: open the transcript sheet for a recording (the first by default). */
    openTranscript(name) {
      const rec = name ? find(name) : (list().items || [])[0];
      if (!rec) return false;
      openTranscript(rec.name, cardEl(rec.name, 'more') || null);
      return true;
    },
    focusCard(name) {
      focusElement(listEl.querySelector(`#vo-rec-${idPart(name)}-title`));
    },
    state: () => ({ chosen, testing, testPlaying, previewing, previewPending, editing, shot }),
    destroy() {
      unbind();
      offPlayer();
      offRecording();
      if (select) select.removeEventListener('change', onSelectChange);
      if (sheetEls.form) sheetEls.form.removeEventListener('submit', onSubmit);
      slots.destroy();
    },
  };
}
