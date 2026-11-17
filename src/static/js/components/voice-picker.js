/**
 * The voice picker: an APG combobox (#vp-search) over a grouped listbox (#vp-list) inside the
 * shell dialog #app-voice-picker. Speak, the Reader's voice sheet and the blend editor open it.
 *
 *   openVoicePicker({trigger, model, modelLabel, value, title, mode: 'select'|'blend-add',
 *                    include: {named, recordings, voices}, exclude, extraRows, voiceModels}) -> Promise<Choice|null>
 *   Choice: {kind: 'voice'|'named'|'recording'|'extra', id, voice, label, detail, model}
 *     named -> voice = `voice:${identityId}`; recording -> voice = the recording name.
 *
 * `voiceModels`: the Piper voice models, for a model select's Piper entry (`model` is then models.PIPER_GROUP).
 * Each model is listed as a voice ("Amy (US)", "Medium quality") with no request; the named voices offered are those
 * with a version for one of them. A Choice carries the model it speaks with (the row's own). Choosing a model with
 * several speakers (voices.speakersOf) lists its speakers in the same picker, and the speaker chosen there is the
 * Choice.
 *
 * Groups, in order: extra rows, Favorites, Named voices (unavailable ones aria-disabled with the reason as
 * detail), Recordings (when included), Recommended, then one group per language (Kokoro) or "Voices"; the model's
 * own voice groups are headed with its name ("Kokoro · Recommended", "Supertonic · Voices").
 * A model with no voices of its own (a cloning model such as Chatterbox: its own voice list loads empty), when
 * the opener passes `modelLabel` (Speak, the Reader, Talk and Ask AI),
 * the note #vp-model-note says
 * "<Model> has no built-in voices. It speaks in the voice of a recording." ("No recordings yet." when there are
 * none) with the link Clone a voice… (#voices/clone, which also closes the picker; an opener that offers no
 * recordings, as the Reader, Talk and Ask AI, gets the first sentence only), and the order is extra rows,
 * Favorites, Recordings, Named voices (those with a version for the model), then "Named voices with no <Model>
 * version" last, aria-disabled with "No <Model> version" as detail. A favorite the model can't use is
 * listed only in that last group. Without `modelLabel`, and for a model with voices, nothing changes.
 * Keys in the search field: ArrowUp/ArrowDown/Home/End/PageUp/PageDown move, Enter chooses, Shift+Enter
 * previews, Esc closes. There is no favourite key (Alt+F types "ƒ" on macOS, REV-18): the Favourite button
 * is one Tab away. The star and play glyphs in a row are pointer targets only (no buttons inside options).
 * A failed load (refused key, server unreachable, a 5xx) shows #vp-failed with the cause and Retry, never
 * "no voices"; when only named voices or recordings failed, the list shows and #vp-error says what is missing.
 * A search or filter change announces how many voices match ("3 voices", "No voices match.") once it pauses.
 * A preview plays as an ephemeral clip with no mini-player, so closing the picker stops it (its object URL is
 * revoked once the element has let go).
 * @module components/voice-picker
 */
import { html, render, setBusy, nextFrame } from '../core/dom.js';
import { icon } from '../core/icons.js';
import * as voices from '../core/voices.js';
import * as audio from '../core/audio.js';
import { openSheet, closeSheet } from '../core/dialogs.js';
import { mediaUrl, explain } from '../core/api.js';
import { app } from '../core/store.js';
import { announce } from '../core/a11y.js';
import { applyTile, monogram } from '../core/tiles.js';
import { createListbox } from './listbox.js';
import { bindMenuButton } from './menu.js';

const DEFAULT_INCLUDE = Object.freeze({ named: true, recordings: false, voices: true });
// The match count is announced once typing (or a filter change) pauses, not after every key (WCAG 4.1.3).
const COUNT_ANNOUNCE_MS = 600;
let session = null;

function fold(text) {
  return String(text || '').normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
}

/** The last group for a model with no voices of its own: the named voices that have no version for it. */
const withoutVersion = (label) => `Named voices with no ${label} version`;

/**
 * PURE: the picker rows for the loaded data.
 * @param {{voices?: object[], named?: object[], recordings?: object[]}} data
 * @param {{model?: string, include?: object, exclude?: string[], extraRows?: object[], mode?: string,
 *   favorites?: string[], noBuiltIn?: string, voiceModels?: string[]}} options `noBuiltIn`: the label of a model with
 *   no voices of its own ('' for any other model); a named voice with no version for it then says so (noVersion).
 *   An extra row may come `disabled`, with its reason as detail, and with the `model` it speaks with.
 *   `voiceModels`: the models listed as voices (each voice entry carries its `model`); a named voice then uses its
 *   version for the first of them that has one (the opener puts the model in use first), and is unavailable without
 *   one.
 * @returns {object[]} rows: {key, kind, id, voice, label, detail, group, language, gender, recommended,
 *   favourite, disabled, seed, noVersion}, plus `model` on a row that speaks with a model of its own
 */
export function pickerRows(data = {}, { model = '', include = DEFAULT_INCLUDE, exclude = [], extraRows = [], mode = 'select', favorites = [], noBuiltIn = '', voiceModels = [] } = {}) {
  const inc = mode === 'blend-add' ? { named: false, recordings: false, voices: true } : { ...DEFAULT_INCLUDE, ...include };
  const skip = new Set(exclude || []);
  const favs = new Set(favorites || []);
  const group = new Set(voiceModels || []);
  const isFav = (voice, own = model) => favs.has(`${own}|${voice}`);
  const ownModel = (id) => (id ? { model: String(id) } : {});
  const rows = [];
  if (mode !== 'blend-add') {
    for (const x of extraRows || []) {
      if (!x || !x.voice) continue;
      rows.push({ key: `x|${x.id || x.voice}`, kind: 'extra', id: String(x.id || x.voice), voice: String(x.voice), label: String(x.label || x.voice), detail: String(x.detail || ''), group: String(x.group || 'Suggested'), seed: String(x.voice), favourite: false, disabled: !!x.disabled, ...ownModel(x.model) });
    }
  }
  if (inc.named) {
    for (const identity of data.named || []) {
      if (!identity || !identity.id) continue;
      const voice = identity.voice || `voice:${identity.id}`;
      if (skip.has(voice)) continue;
      const realizations = identity.realizations || [];
      const version = realizations.find((r) => r.model === model)
        || (voiceModels || []).map((id) => realizations.find((r) => r.model === id)).find(Boolean);
      const available = group.size
        ? !!version && identity.available !== false && version.available !== false
        : identity.available !== false && (!version || version.available !== false);
      const noVersion = !!noBuiltIn && !version && !available;
      let detail;
      if (noVersion) detail = `No ${noBuiltIn} version`;
      else if (!available) detail = (version && version.reason) || 'Unavailable for this model';
      else if (version && voices.isBlend(version.voice)) detail = `Blend · ${version.voice}`;
      else if (version && version.voice) detail = voices.describeVoice(version.voice).label;
      else detail = 'Named voice';
      const own = group.size && version ? version.model : '';
      rows.push({ key: `n|${identity.id}`, kind: 'named', id: String(identity.id), voice, label: String(identity.name || identity.id), detail, group: noVersion ? withoutVersion(noBuiltIn) : 'Named voices', seed: String(identity.id), favourite: isFav(voice, own || model), disabled: !available, noVersion, ...ownModel(own) });
    }
  }
  if (inc.recordings) {
    for (const rec of data.recordings || []) {
      const name = rec && (rec.name || rec.id);
      if (!name || skip.has(name)) continue;
      rows.push({ key: `r|${name}`, kind: 'recording', id: String(name), voice: String(name), label: String(name), detail: rec.transcript ? 'Transcript saved' : 'No transcript', group: 'Recordings', seed: String(name), favourite: isFav(name), disabled: false });
    }
  }
  if (inc.voices) {
    const list = data.voices || [];
    voices.describeVoices(list).forEach((d, i) => {
      if (!d.id || skip.has(d.id)) return;
      const recommended = d.tags.includes('recommended');
      const own = group.size ? String(list[i]?.model || '') : '';
      rows.push({ key: `v|${d.id}`, kind: 'voice', id: d.id, voice: d.id, label: d.label, detail: d.detail, group: recommended ? 'Recommended' : (d.lang ? d.language : 'Voices'), language: d.language || '', gender: d.gender || '', recommended, seed: d.id, favourite: isFav(d.id, own || model), disabled: false, ...ownModel(own) });
    });
  }
  return rows;
}

/**
 * PURE: filter rows and group them in order: extra rows, Favorites, Named voices,
 * Recordings, Recommended, then one group per language (or "Voices").
 * @param {object[]} rows
 * @param {{query?: string, lang?: string, female?: boolean, male?: boolean, favsOnly?: boolean,
 *   noBuiltIn?: string, modelName?: string}} filters `noBuiltIn`: the label of a model with no voices of its own,
 *   as given to pickerRows: Recordings then come before Named voices, and the named voices with no version for the
 *   model go in a last group only (never in Favorites). `modelName` heads the model's own voice groups.
 * @returns {{label: string, options: object[]}[]}
 */
export function pickerGroups(rows, { query = '', lang = '', female = false, male = false, favsOnly = false, noBuiltIn = '', modelName = '' } = {}) {
  const q = fold(query).trim();
  const gender = female || male;
  const keep = (row) => {
    if (q && !fold(`${row.label} ${row.detail} ${row.id}`).includes(q)) return false;
    if (row.kind === 'extra') return !favsOnly;
    if (favsOnly && !row.favourite) return false;
    if (lang && (row.kind !== 'voice' || row.language !== lang)) return false;
    if (gender && (row.kind !== 'voice' || !((female && row.gender === 'female') || (male && row.gender === 'male')))) return false;
    return true;
  };
  const shown = rows.filter(keep);
  const groups = [];
  const add = (label, options) => { if (options.length) groups.push({ label, options }); };
  const extraGroups = [...new Set(shown.filter((r) => r.kind === 'extra').map((r) => r.group))];
  for (const g of extraGroups) add(g, shown.filter((r) => r.kind === 'extra' && r.group === g));
  // For a model with no voices of its own, the named voices it has no version of are listed last and only there.
  const last = noBuiltIn ? shown.filter((r) => r.noVersion) : [];
  const usable = noBuiltIn ? shown.filter((r) => !r.noVersion) : shown;
  const favourites = usable.filter((r) => r.favourite && r.kind !== 'extra');
  if (favsOnly) {
    add('Favorites', favourites);
    add(withoutVersion(noBuiltIn), last);
    return groups;
  }
  add('Favorites', favourites.map((r) => ({ ...r, inFavourites: true })));
  const named = usable.filter((r) => r.kind === 'named');
  const recordings = shown.filter((r) => r.kind === 'recording');
  if (noBuiltIn) {
    add('Recordings', recordings);   // what such a model speaks in comes first
    add('Named voices', named);
  } else {
    add('Named voices', named);
    add('Recordings', recordings);
  }
  const own = (label) => (modelName ? `${modelName} · ${label}` : label);
  add(own('Recommended'), shown.filter((r) => r.kind === 'voice' && r.recommended));
  const languages = [];
  for (const r of shown) if (r.kind === 'voice' && !r.recommended && !languages.includes(r.group)) languages.push(r.group);
  for (const g of languages) add(own(g), shown.filter((r) => r.kind === 'voice' && !r.recommended && r.group === g));
  add(withoutVersion(noBuiltIn), last);
  return groups;
}

/**
 * PURE: whether a remembered voice can still speak (Talk and Ask AI remember one):
 * its model is one of the usable TTS models, and the picker would still offer it for that model (a named voice
 * available for it, or one of the model's voices). A list that isn't given (it didn't load) says nothing, so the
 * voice stays.
 * @param {{model: string, voice: string}|null} chosen
 * @param {{models?: object[], voices?: object[], named?: object[], include?: object}} known
 *   `models`: ttsModels(app.models.list); `include` as the picker is opened with (named voices and the model's
 *   voices)
 * @returns {boolean}
 */
export function voiceOffered(chosen, { models = [], voices, named, include = DEFAULT_INCLUDE } = {}) {
  if (!chosen || !chosen.model || !chosen.voice) return false;
  if (!(models || []).some((m) => m && m.id === chosen.model)) return false;
  const isNamed = String(chosen.voice).startsWith('voice:');
  const given = isNamed ? named : voices;
  if (!Array.isArray(given)) return true;
  const rows = pickerRows(isNamed ? { named: given } : { voices: given }, { model: chosen.model, include });
  return rows.some((row) => row.voice === chosen.voice && !row.disabled);
}

/**
 * PURE: the key of the row a voice string selects (named, then provider voice, recording, extra row); a speaker of a
 * model listed as a voice ('piper/en_US-arctic-medium/awb') selects that model's row.
 */
export function selectedKey(rows, value) {
  if (!value) return null;
  const v = String(value);
  const order = v.startsWith('voice:') ? ['named', 'voice', 'recording', 'extra'] : ['voice', 'named', 'recording', 'extra'];
  for (const kind of order) {
    const row = rows.find((r) => r.kind === kind && r.voice === v);
    if (row) return row.key;
  }
  const owner = rows.find((r) => r.kind === 'voice' && r.model && v.startsWith(`${r.model}/`));
  return owner ? owner.key : null;
}

function renderOption(row) {
  const states = [row.favourite ? 'favorite' : '', row.recommended ? 'recommended' : '', row.disabled ? 'unavailable' : ''].filter(Boolean);
  return html`<span class="avatar" data-seed="${row.seed}" aria-hidden="true">${monogram(row.label)}</span><span class="option__text"><span class="option__label">${row.label}</span>${row.detail ? html`<span class="option__detail">${row.detail}</span>` : ''}${states.length ? html`<span class="visually-hidden">${`, ${states.join(', ')}`}</span>` : ''}</span>${row.kind === 'extra' ? '' : html`<span class="option__glyph${row.favourite ? ' is-on' : ''}" data-act="fav" aria-hidden="true">${icon(row.favourite ? 'star-fill' : 'star')}</span>`}${row.disabled ? '' : html`<span class="option__glyph" data-act="preview" aria-hidden="true">${icon('play')}</span>`}<span class="option__check" aria-hidden="true">${icon('check')}</span>`;
}

function micOpen() {
  return !!app.get().recording;
}

function refs(dialog) {
  const q = (id) => dialog.querySelector(`#${id}`);
  return {
    title: q('vp-title'), search: q('vp-search'), list: q('vp-list'), empty: q('vp-empty'), hint: q('vp-hint'),
    lang: q('vp-lang'), langLabel: q('vp-lang-label'), langMenu: q('vp-lang-menu'), female: q('vp-female'),
    male: q('vp-male'), favs: q('vp-favs'), summaryLabel: q('vp-summary-label'), summaryDetail: q('vp-summary-detail'),
    preview: q('vp-preview'), fav: q('vp-fav'), use: q('vp-use'), error: q('vp-error'), note: q('vp-note'),
    failed: q('vp-failed'), failedMsg: q('vp-failed-msg'), retry: q('vp-retry'),
    modelNote: q('vp-model-note'), modelNoteText: q('vp-model-note-text'), clone: q('vp-clone'),
  };
}

/**
 * Bring the chosen option into view when the picker opens: an option in the first screenful leaves the
 * list at the top; otherwise its group label goes to the top (a long group centres the option instead).
 * The scroller is the list itself, or the sheet body on short screens where the whole body scrolls.
 */
function scrollToChoice(list, option, stickyEl) {
  const scroller = list.scrollHeight > list.clientHeight + 1 ? list : (list.closest('.sheet__body') || list);
  const view = scroller.getBoundingClientRect();
  const offset = (node) => node.getBoundingClientRect().top - view.top + scroller.scrollTop;
  const height = option.getBoundingClientRect().height;
  const sticky = scroller !== list && stickyEl ? stickyEl.getBoundingClientRect().height + 8 : 0;
  const visible = scroller.clientHeight - sticky;
  const optTop = offset(option) - sticky;
  if (optTop + height <= visible) {
    scroller.scrollTop = 0;
    return;
  }
  const group = option.closest('[role="group"]');
  const label = group ? group.querySelector('.listbox__label') : null;
  let target = (label ? offset(label) : optTop + sticky) - sticky;
  if (optTop + height - target > visible) target = optTop - (visible - height) / 2;
  scroller.scrollTop = Math.max(0, Math.round(target));
}

function createSession(dialog, opts, resolve) {
  const { trigger = null, modelLabel = '', title = 'Choose a voice', mode = 'select', exclude = [] } = opts;
  // Choosing a model with several speakers lists its speakers instead (speakersIn): these then change.
  let { model = '', value = '', include = {}, extraRows = [] } = opts;
  let voiceModels = Array.isArray(opts.voiceModels) ? opts.voiceModels.map(String).filter(Boolean) : [];
  let speakerList = null;   // the speakers being listed, or null
  const el = refs(dialog);
  // noBuiltIn: the model's label once its own voice list loaded empty (a cloning model), '' otherwise.
  const state = { query: '', lang: '', female: false, male: false, favsOnly: false, rows: [], loading: true, shown: 0, noBuiltIn: '', modelName: voices.voiceModelName(model) };
  const offs = [];
  let done = false;
  let choosing = false;     // a chosen voice model's speakers are loading
  let previewToken = 0;
  let revokeMedia = null;
  let previewHandle = null;   // the picker's own playing preview (audio session handle)
  let countTimer = null;

  el.title.textContent = title;
  el.use.querySelector('.btn__label').textContent = mode === 'blend-add' ? 'Add voice' : 'Use voice';
  el.search.value = '';
  for (const chip of [el.female, el.male, el.favs]) chip.setAttribute('aria-pressed', 'false');
  el.langLabel.textContent = 'All languages';
  el.lang.removeAttribute('data-active');
  el.error.hidden = true;
  el.error.textContent = '';
  if (el.failed) el.failed.hidden = true;

  const listbox = createListbox(el.list, {
    combobox: el.search,
    groups: [],
    idPrefix: 'vp-opt',
    renderOption,
    onActive: (row) => syncFooter(row),
    onSelect: (row) => { choose(rowFor(row)).catch(failedLoad); },
  });

  function choiceFor(row) {
    return { kind: row.kind, id: row.id, voice: row.voice, label: row.label, detail: row.detail, model: row.model || model };
  }

  /** A row chosen: the Choice, or for a voice model with several speakers, its speakers to choose from. */
  async function choose(row) {
    if (done || choosing || !row) return;
    if (!voiceModels.length || row.kind !== 'voice' || !row.model) {
      finish(choiceFor(row));
      return;
    }
    choosing = true;
    setBusy(el.use, true);
    let speakers = [];
    try {
      speakers = await voices.speakersOf(row.model);
    } catch {
      speakers = [];   // the model's own voice still speaks: its default speaker
    } finally {
      choosing = false;
      setBusy(el.use, false);
    }
    if (done) return;
    if (!speakers.length) finish(choiceFor(row));
    else speakersIn(row, speakers);
  }

  /** List the speakers of a voice model (row) in place of the voices; Esc still closes without a choice. */
  function speakersIn(row, speakers) {
    model = row.model;
    state.modelName = voices.voiceModelName(model);
    value = String(value).startsWith(`${row.model}/`) ? value : '';
    voiceModels = [];
    speakerList = speakers;
    include = { named: false, recordings: false, voices: true };
    extraRows = [];
    state.query = '';
    state.lang = '';
    state.female = false;
    state.male = false;
    state.favsOnly = false;
    el.search.value = '';
    for (const chip of [el.female, el.male, el.favs]) chip.setAttribute('aria-pressed', 'false');
    el.langLabel.textContent = 'All languages';
    el.lang.removeAttribute('data-active');
    el.title.textContent = `${row.label} speakers`;
    // Use this voice is disabled while the speakers load: the search field keeps focus.
    if (document.activeElement !== el.search) el.search.focus();
    announce(`${row.label} has ${speakers.length} speakers. Choose one.`);
    load().catch(failedLoad);
  }

  function rowFor(option) {
    return option ? state.rows.find((r) => r.key === option.key) || option : null;
  }

  function syncFooter(option) {
    const row = rowFor(option);
    el.summaryLabel.textContent = row ? row.label : '';
    el.summaryDetail.textContent = row ? row.detail : '';
    const none = !row;
    el.use.disabled = none || !!row.disabled;
    el.fav.disabled = none || row.kind === 'extra';
    el.fav.setAttribute('aria-pressed', row && row.favourite ? 'true' : 'false');
    render(el.fav, html`${icon(row && row.favourite ? 'star-fill' : 'star')}<span class="btn__label">Favorite</span>`);
    const blocked = micOpen();
    el.preview.disabled = none || !!row.disabled;
    if (blocked) el.preview.setAttribute('aria-disabled', 'true');
    else el.preview.removeAttribute('aria-disabled');
    if (el.note) {
      el.note.hidden = !blocked;
      if (blocked) el.preview.setAttribute('aria-describedby', 'vp-note');
      else el.preview.removeAttribute('aria-describedby');
    }
  }

  function decorate() {
    for (const avatar of el.list.querySelectorAll('.avatar[data-seed]')) applyTile(avatar, avatar.getAttribute('data-seed'));
  }

  function refresh({ keepActive = true } = {}) {
    if (state.loading) return;
    const groups = pickerGroups(state.rows, state);
    const options = groups.flatMap((g) => g.options);
    listbox.setGroups(groups.map((g) => ({ label: g.label, options: g.options.map((r) => ({ ...r, value: r.key })) })), {
      value: selectedKey(state.rows, value), keepActive,
    });
    decorate();
    // With nothing to list for a model with no voices of its own, the note above already says so.
    el.empty.hidden = options.length > 0 || (!!state.noBuiltIn && !state.rows.length);
    el.empty.textContent = state.rows.length ? 'No voices match.' : 'This model has no voices to choose from.';
    if (!options.length) syncFooter(null);
    state.shown = new Set(options.map((o) => o.key)).size;   // a favourite is listed twice, counted once
  }

  /** After a search or filter change: announce the number of matches once the listener pauses. */
  function announceCount() {
    clearTimeout(countTimer);
    countTimer = setTimeout(() => {
      countTimer = null;
      if (done || state.loading) return;
      const n = state.shown || 0;
      // A hidden line is never read out: with nothing to list for a model with no voices of its own, the note
      // explains and a search says "No voices match.".
      announce(n ? `${n} ${n === 1 ? 'voice' : 'voices'}` : (el.empty.hidden ? 'No voices match.' : el.empty.textContent));
    }, COUNT_ANNOUNCE_MS);
  }

  function buildLanguageMenu() {
    const langs = [...new Set(state.rows.filter((r) => r.kind === 'voice' && r.language).map((r) => r.language))].sort();
    el.lang.hidden = langs.length < 2;
    const genders = new Set(state.rows.filter((r) => r.kind === 'voice').map((r) => r.gender));
    el.female.hidden = !genders.has('female');
    el.male.hidden = !genders.has('male');
    const items = ['', ...langs];
    render(el.langMenu, html`${items.map((lang) => html`<button type="button" class="menu__item" role="menuitemradio" aria-checked="${lang === state.lang ? 'true' : 'false'}" data-lang="${lang}">${lang || 'All languages'}<span class="menu__check" aria-hidden="true">${icon('check')}</span></button>`)}`);
  }

  /**
   * The note for a model with no voices of its own (state.noBuiltIn is its label), hidden for any other model.
   * `clones`: the opener offers recordings, so a recording is what the model speaks in and Clone a voice… shows.
   */
  function syncModelNote({ clones = false, noRecordings = false } = {}) {
    if (!el.modelNote) return;
    let text = '';
    if (state.noBuiltIn) {
      text = `${state.noBuiltIn} has no built-in voices.`;
      if (clones) text += ` It speaks in the voice of a recording.${noRecordings ? ' No recordings yet.' : ''}`;
    }
    if (el.modelNoteText) el.modelNoteText.textContent = text;
    if (el.clone) el.clone.hidden = !(text && clones);
    el.modelNote.hidden = !text;
    if (text) announce(text);
  }

  function showFailed(error) {
    const info = explain(error, "Couldn't load the voices");
    if (el.failedMsg) el.failedMsg.textContent = info.message ? `${info.title}. ${info.message}` : info.title;
    if (el.failed) el.failed.hidden = false;
  }

  async function load() {
    state.loading = true;
    state.noBuiltIn = '';
    syncModelNote();
    if (el.failed) el.failed.hidden = true;
    showError('');
    el.list.hidden = false;
    el.list.setAttribute('aria-busy', 'true');
    render(el.list, html`${[0, 1, 2, 3].map(() => html`<div class="skeleton skeleton--row" aria-hidden="true"></div>`)}<span class="visually-hidden">Loading voices…</span>`);
    el.empty.hidden = true;
    syncFooter(null);
    const inc = { ...DEFAULT_INCLUDE, ...include };
    const blendAdd = mode === 'blend-add';
    const wanted = { voices: !!(inc.voices || blendAdd), named: !!(inc.named && !blendAdd), recordings: !!(inc.recordings && !blendAdd) };
    // Voice models are listed with no request; the named voices with a version for one of them come unchecked.
    let ownVoices;
    if (speakerList) ownVoices = Promise.resolve(speakerList);
    else if (voiceModels.length) ownVoices = Promise.resolve(voices.modelVoices(voiceModels));
    else ownVoices = voices.listVoices(model, { strict: true });
    const [list, named, recordings] = await Promise.allSettled([
      wanted.voices ? ownVoices : Promise.resolve([]),
      wanted.named ? voices.listNamed(voiceModels.length ? { check: false } : { model }) : Promise.resolve([]),
      wanted.recordings ? voices.listRecordings() : Promise.resolve([]),
    ]);
    if (done) return;
    const results = { voices: list, named, recordings };
    const failed = Object.keys(wanted).filter((k) => wanted[k] && results[k].status === 'rejected');
    const loaded = Object.keys(wanted).filter((k) => wanted[k] && results[k].status === 'fulfilled');
    state.loading = false;
    el.list.removeAttribute('aria-busy');
    if (failed.includes('voices') || (failed.length && !loaded.length)) {
      // Nothing usable came back: say why (and offer Retry) instead of "no voices".
      state.rows = [];
      listbox.setGroups([], {});
      el.list.hidden = true;
      el.empty.hidden = true;
      syncFooter(null);
      showFailed(results[failed[0]].reason);
      return;
    }
    if (failed.length) {
      const names = failed.map((k) => (k === 'named' ? 'Named voices' : 'Recordings')).join(' and ');
      showError(`${names} didn't load. ${explain(results[failed[0]].reason).message || ''}`.trim());
    }
    const data = {
      voices: list.status === 'fulfilled' ? list.value : [],
      named: named.status === 'fulfilled' ? named.value : [],
      recordings: recordings.status === 'fulfilled' ? recordings.value : [],
    };
    // The model's own voice list loaded and is empty: with the model's label from the opener, say so and put
    // what it can speak in first.
    state.noBuiltIn = modelLabel && inc.voices && !blendAdd && !data.voices.length ? String(modelLabel) : '';
    state.rows = pickerRows(data, { model, include, exclude, extraRows, mode, favorites: voices.favorites(), noBuiltIn: state.noBuiltIn, voiceModels });
    buildLanguageMenu();
    syncModelNote({ clones: wanted.recordings, noRecordings: recordings.status === 'fulfilled' && !data.recordings.length });
    refresh({ keepActive: false });
    // Once the sheet has its size, bring the current voice into view (its group label at the top).
    await nextFrame();
    const active = listbox.optionElement(listbox.activeIndex());
    if (active && !done) scrollToChoice(el.list, active, dialog.querySelector('.vp__search'));
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = !message;
  }

  function preview(option) {
    const row = rowFor(option);
    if (!row || row.disabled) return;
    if (micOpen()) {
      showError('Stop recording to listen.');
      return;
    }
    showError('');
    audio.prime();   // synchronously inside the click or key press, so iOS lets the result play
    const token = (previewToken += 1);
    setBusy(el.preview, true);
    (async () => {
      let src;
      let revoke = null;
      if (row.kind === 'recording') {
        const media = await mediaUrl(`/api/voices/library/${encodeURIComponent(row.id)}/audio`);
        src = media.url;
        revoke = media.revoke;
      } else {
        src = (await voices.preview(row.model || model, row.voice)).url;
      }
      if (token !== previewToken || done) {
        if (revoke) revoke();
        return;
      }
      revokeMedia = revoke;
      // Revoked when the session ends however it ends (played out, the next preview, closed with the picker):
      // the element has let go by then (WebKit reads blob media lazily).
      const handle = await audio.playClip({
        id: 'preview', owner: 'preview', title: `Preview: ${row.label}`, src, ephemeral: true,
        onRelease: () => { if (revoke) revoke(); if (revokeMedia === revoke) revokeMedia = null; },
      });
      if (token === previewToken && !done) previewHandle = handle;
      else if (handle.active) audio.closeCurrent();   // the picker closed while it started
    })().catch((error) => {
      if (token !== previewToken) return;
      const info = explain(error, "Couldn't play the preview");
      if (info.message) showError(`${info.title}: ${info.message}`);
    }).finally(() => {
      if (token === previewToken) setBusy(el.preview, false);
    });
  }

  function toggleFavourite(option) {
    const row = rowFor(option);
    if (!row || row.kind === 'extra') return;
    const own = row.model || model;
    const on = voices.toggleFavorite(own, row.voice);
    for (const r of state.rows) if (r.voice === row.voice && r.kind !== 'extra' && (r.model || model) === own) r.favourite = on;
    announce(on ? `${row.label} added to favorites` : `${row.label} removed from favorites`);
    refresh();
    syncFooter(listbox.activeOption());
  }

  function finish(choice) {
    if (done) return;
    done = true;
    previewToken += 1;
    clearTimeout(countTimer);
    // A preview has no mini-player or lock-screen control: it ends with the picker (its onRelease revokes it).
    if (previewHandle && previewHandle.active) audio.closeCurrent();
    previewHandle = null;
    for (const off of offs) off();
    listbox.destroy();
    langMenu.destroy();
    if (session && session.dialog === dialog) session = null;
    if (dialog.open) closeSheet(dialog, choice ? 'ok' : 'cancel');
    setBusy(el.preview, false);
    resolve(choice);
  }

  const langMenu = bindMenuButton(el.lang, el.langMenu, {
    onSelect: (item) => {
      state.lang = item.getAttribute('data-lang') || '';
      el.langLabel.textContent = state.lang || 'All languages';
      if (state.lang) el.lang.setAttribute('data-active', 'true');
      else el.lang.removeAttribute('data-active');
      refresh({ keepActive: false });
      announceCount();
    },
  });

  const listen = (target, type, fn, options) => {
    target.addEventListener(type, fn, options);
    offs.push(() => target.removeEventListener(type, fn, options));
  };
  listen(el.search, 'input', () => {
    state.query = el.search.value;
    refresh({ keepActive: false });
    announceCount();
  });
  listen(el.search, 'keydown', (event) => {
    if (event.key === 'Escape') {
      // Esc closes the picker (and never just clears the search field).
      event.preventDefault();
      finish(null);
    } else if (event.key === 'Enter' && event.shiftKey && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      preview(listbox.activeOption());
    }
  });
  const chip = (button, key) => listen(button, 'click', () => {
    state[key] = !state[key];
    button.setAttribute('aria-pressed', state[key] ? 'true' : 'false');
    refresh({ keepActive: false });
    announceCount();
  });
  chip(el.female, 'female');
  chip(el.male, 'male');
  chip(el.favs, 'favsOnly');
  listen(el.list, 'click', (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const act = start && start.closest ? start.closest('[data-act]') : null;
    if (!act) return;
    const optionEl = act.closest('[role="option"]');
    const option = optionEl ? listbox.options()[Number(optionEl.getAttribute('data-index'))] : null;
    if (act.getAttribute('data-act') === 'preview') preview(option);
    else if (act.getAttribute('data-act') === 'fav') toggleFavourite(option);
  });
  const failedLoad = (error) => {
    if (done) return;
    state.loading = false;
    state.rows = [];
    el.list.removeAttribute('aria-busy');
    el.list.hidden = true;
    syncFooter(null);
    showFailed(error);
  };
  if (el.retry) listen(el.retry, 'click', () => { el.search.focus(); load().catch(failedLoad); });
  // Clone a voice…: the picker closes without a choice; the link's own hash change then opens Voices › Clone.
  if (el.clone) listen(el.clone, 'click', () => finish(null));
  listen(el.preview, 'click', () => preview(listbox.activeOption()));
  listen(el.fav, 'click', () => toggleFavourite(listbox.activeOption()));
  listen(el.use, 'click', () => {
    const index = listbox.activeIndex();
    if (index >= 0) listbox.select(index);
  });
  offs.push(app.select((s) => s.recording, () => syncFooter(listbox.activeOption())));
  offs.push(() => { if (revokeMedia) { revokeMedia(); revokeMedia = null; } });

  openSheet(dialog, { trigger: trigger || undefined, initialFocus: el.search, onClose: () => finish(null) });
  load().catch(failedLoad);
  return { dialog, finish, state, listbox };
}

/**
 * Open the voice picker.
 * @param {{trigger?: HTMLElement, model: string, modelLabel?: string, value?: string, title?: string,
 *   mode?: 'select'|'blend-add', include?: {named?: boolean, recordings?: boolean, voices?: boolean},
 *   exclude?: string[], extraRows?: {id: string, label: string, detail?: string, voice: string, group?: string,
 *   disabled?: boolean, model?: string}[], voiceModels?: string[]}}
 *   options `modelLabel`: the model's display name; with it, a model whose own voice list is empty gets the
 *   "no built-in voices" note and order (see the module header). `voiceModels`: the models listed as voices (the
 *   Piper entry of a model select, see the module header).
 * @returns {Promise<{kind: string, id: string, voice: string, label: string, detail: string, model: string}|null>}
 */
export function openVoicePicker(options = {}) {
  if (session) session.finish(null);
  const dialog = document.getElementById('app-voice-picker');
  if (!dialog) return Promise.reject(new Error('The voice picker is missing from this page.'));
  return new Promise((resolve) => {
    session = createSession(dialog, options, resolve);
  });
}

/** @returns {boolean} whether the picker is open */
export function isOpen() {
  return !!session;
}

/** Test hook. */
export function __resetForTests() {
  if (session) session.finish(null);
  session = null;
}
