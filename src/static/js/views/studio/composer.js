/**
 * Studio Composer (inventory CO-01…CO-05, defect REV-04).
 *
 * - Tracks (`ol#st-tracks`): each has a source select listing only WAV files the Composer can read (REV-04):
 *   conversation renders (each conversation's render_output_path, by name) and previous mixes, plus "Other
 *   server path…", which reveals a path field. Offset is a stepper (0.1 s), volume a range (0.1–2.0), Mute
 *   and Solo are toggle chips. One empty track exists from the start (CO-01). Every edit updates the track
 *   model at once (CO-02).
 * - Render mix POSTs /api/composer/render; a refused path (403) or a missing source (400) shows on the
 *   track's field when the message names it, else in the mix banner. The result plays on the shared element
 *   and downloads as composer-mix-<ts>.<format that was rendered> (CO-03, CO-04), fetched with the key.
 * - A mix that ends while the Composer is not on screen (the Conversations section, another view) is told by a
 *   toast with Open; its banner stays for the return.
 * - Past mixes (`#st-mixes`): play, download and a menu with Delete… (CO-05).
 * - While a mix plays, an inline transport (a seek range and the time) sits where it was started: under the
 *   result, or in its past-mix row. The mini-player hides on the Composer, the mixes' home route.
 * @module views/studio/composer
 */
import { html, render, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { announce } from '../../core/a11y.js';
import * as audio from '../../core/audio.js';
import { app } from '../../core/store.js';
import { downloadFromApi } from '../../core/download.js';
import { absoluteTime, relativeTime, clock } from '../../core/format.js';
import { bindRange } from '../../components/range.js';
import {
  idPart, seg, errorCopy, bannerActions, showBanner, clearBanner, fillDetails, loadingMarkup, loadErrorMarkup,
  focusElement, hideKeepingFocus, renderKeepingFocus, menuButtonMarkup, menuMarkup, bindMenus, setPlayButton,
  playerKey, bindTransport, transportMarkup, startPress,
} from './shared.js';
import { SAMPLE_RATE, renderFormat } from './conversations.js';

export const OFFSET_STEP = 0.1;
export const OFFSET_MAX = 2 * 60 * 60;   // src/composer.py MAX_RENDER_SECONDS
export const VOLUME_MIN = 0.1;
export const VOLUME_MAX = 2;
const OTHER = 'other';

/* ───────────────────────────── pure helpers ───────────────────────────── */

/** Whether a server path is a WAV file (the Composer reads WAV sources only, REV-04). */
export function isWav(path) {
  return renderFormat(path) === 'wav';
}

/**
 * PURE: the source options: conversation renders and previous mixes that are WAV files.
 * @param {object[]} conversations GET /api/conversations items
 * @param {object[]} mixes GET /api/composer/renders items
 * @returns {{conversations: {path: string, label: string}[], mixes: {path: string, label: string}[]}}
 */
export function wavSources(conversations, mixes) {
  return {
    conversations: (conversations || []).filter((c) => isWav(c.render_output_path))
      .map((c) => ({ path: String(c.render_output_path), label: c.name || 'Untitled conversation' })),
    mixes: (mixes || []).filter((m) => isWav(m.render_output_path))
      .map((m) => ({ path: String(m.render_output_path), label: m.name || 'Untitled mix' })),
  };
}

/** A new, empty track (CO-01). */
export function newTrack(key) {
  return { key, source_path: '', other: false, offset_s: 0, volume: 1, muted: false, solo: false };
}

/** The tracks the server mixes: the unmuted ones, or only the soloed ones when any is soloed. */
export function activeTracks(tracks) {
  const unmuted = (tracks || []).filter((t) => !t.muted);
  return unmuted.some((t) => t.solo) ? unmuted.filter((t) => t.solo) : unmuted;
}

/** One decimal, clamped to 0…OFFSET_MAX; text that isn't a number gives 0. */
export function clampOffset(value) {
  const n = Number(String(value ?? '').replace(',', '.'));
  if (!Number.isFinite(n) || n < 0) return 0;
  return Math.min(OFFSET_MAX, Math.round(n * 10) / 10);
}

/** One decimal, clamped to 0.1…2.0. */
export function clampVolume(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 1;
  return Math.min(VOLUME_MAX, Math.max(VOLUME_MIN, Math.round(n * 10) / 10));
}

/** '1.0×' and '1.0 times' for the volume range. */
export const volumeText = (v) => `${clampVolume(v).toFixed(1)}×`;
export const volumeSpoken = (v) => `${clampVolume(v).toFixed(1)} times`;

/** The POST /api/composer/render track (the server's ComposerTrack). */
export function trackPayload(track) {
  return {
    source_path: String(track.source_path || '').trim(),
    offset_s: clampOffset(track.offset_s),
    volume: clampVolume(track.volume),
    muted: !!track.muted,
    solo: !!track.solo,
    effects: [],
  };
}

/**
 * PURE: the index of the track a server message names ("Track 2: …", or a track's own source path), else -1.
 * @param {string} message
 * @param {object[]} tracks
 */
export function trackNamedIn(message, tracks) {
  const text = String(message || '');
  const m = /\btrack\s+(\d+)\b/i.exec(text);
  if (m) {
    const index = Number(m[1]) - 1;
    if (index >= 0 && index < (tracks || []).length) return index;
  }
  const byPath = (tracks || []).findIndex((t) => t.source_path && text.includes(String(t.source_path).trim()));
  return byPath;
}

/** composer-mix-<ts>.<format> (CO-04). */
export function mixFileName(format, now = Date.now()) {
  return `composer-mix-${now}.${format || 'wav'}`;
}

/**
 * PURE: what the source selects can't list because a list didn't load ('' when nothing is missing). A list
 * that loaded once keeps its items through a failed refresh, so only a list that never loaded counts.
 * @param {{status: string, at: number}} conversations
 * @param {{status: string, at: number}} mixes
 */
export function sourcesError(conversations, mixes) {
  const failed = (list) => !!list && list.status === 'error' && !list.at;
  const convs = failed(conversations);
  const past = failed(mixes);
  if (convs && past) return 'Conversation renders and previous mixes didn’t load, so they aren’t listed.';
  if (convs) return 'Conversation renders didn’t load, so they aren’t listed.';
  if (past) return 'Previous mixes didn’t load, so they aren’t listed.';
  return '';
}

/* ───────────────────────────── markup ───────────────────────────── */

/**
 * @param {object} track
 * @param {number} n
 * @param {{conversations: object[], mixes: object[], known: boolean, errorId: string}} sources known: both lists
 *   loaded (the "render as WAV" hint is true only then); errorId: the sources error to describe the select with
 * @param {object} errors
 */
function trackMarkup(track, n, sources, errors) {
  const k = track.key;
  const base = `st-track-${k}`;
  const known = [...sources.conversations, ...sources.mixes].some((s) => s.path === track.source_path);
  const value = track.other || (track.source_path && !known) ? OTHER : (track.source_path ? `p:${track.source_path}` : '');
  const other = value === OTHER;
  const err = errors[k] || null;
  const srcError = err && err.field === 'src' ? err.message : '';
  const pathError = err && err.field === 'path' ? err.message : '';
  // The hint shows only when both lists loaded and nothing can be listed (the intro says the Composer reads WAV
  // renders); a list that didn't load is said by the sources error above the tracks instead.
  const none = sources.known && !sources.conversations.length && !sources.mixes.length;
  const opt = (s) => html`<option value="p:${s.path}"${value === `p:${s.path}` ? html` selected` : ''}>${s.label}</option>`;
  const describedBy = [none ? `${base}-src-hint` : '', sources.errorId, srcError ? `${base}-src-error` : ''].filter(Boolean).join(' ');
  return html`<li class="st-track" id="${base}" data-key="${k}">
    <div class="st-track__head">
      <h3 class="st-track__title" id="${base}-title">Track ${n}</h3>
      <button type="button" class="btn btn--ghost btn--sm st-track__remove" id="${base}-remove" data-action="remove-track" data-key="${k}" aria-label="Remove track ${n}">${icon('trash')}<span class="btn__label">Remove</span></button>
    </div>
    <div class="st-track__grid">
      <div class="field st-track__source">
        <label class="field__label" for="${base}-src">Source<span class="visually-hidden"> for track ${n}</span></label>
        <select class="select st-src" id="${base}-src" data-key="${k}" data-field="source"${describedBy ? html` aria-describedby="${describedBy}"` : ''}${srcError ? html` aria-invalid="true"` : ''}>
          <option value=""${value === '' ? html` selected` : ''}>Choose a WAV file</option>
          ${sources.conversations.length ? html`<optgroup label="Conversation renders">${sources.conversations.map(opt)}</optgroup>` : ''}
          ${sources.mixes.length ? html`<optgroup label="Previous mixes">${sources.mixes.map(opt)}</optgroup>` : ''}
          <option value="${OTHER}"${other ? html` selected` : ''}>Other server path…</option>
        </select>
        ${none ? html`<p class="field__hint" id="${base}-src-hint">Render a conversation as WAV to list it here.</p>` : ''}
        <p class="field__error" id="${base}-src-error"${srcError ? '' : html` hidden`}>${icon('alert', { className: 'ico ico--sm' })}<span>${srcError}</span></p>
      </div>
      <div class="field st-track__path" id="${base}-path-field"${other ? '' : html` hidden`}>
        <label class="field__label" for="${base}-path">Server path<span class="visually-hidden"> for track ${n}</span></label>
        <input class="input mono st-path" id="${base}-path" type="text" data-key="${k}" data-field="path" autocomplete="off" spellcheck="false" placeholder="data/voices/example.wav" value="${other ? track.source_path : ''}" aria-describedby="${base}-path-hint${pathError ? html` ${base}-path-error` : ''}"${pathError ? html` aria-invalid="true"` : ''}>
        <p class="field__hint" id="${base}-path-hint">A WAV file inside the server’s data folder.</p>
        <p class="field__error" id="${base}-path-error"${pathError ? '' : html` hidden`}>${icon('alert', { className: 'ico ico--sm' })}<span>${pathError}</span></p>
      </div>
      <div class="st-track__controls">
        <div class="field st-track__offset">
          <label class="field__label" for="${base}-offset">Offset<span class="visually-hidden"> for track ${n}, in seconds</span></label>
          <div class="stepper">
            <button type="button" class="icon-btn icon-btn--sm" data-action="offset-step" data-key="${k}" data-step="-1" aria-label="Decrease offset for track ${n}"${track.offset_s <= 0 ? html` disabled` : ''}>${icon('minus')}</button>
            <input class="input stepper__value" id="${base}-offset" type="text" inputmode="decimal" data-key="${k}" data-field="offset" value="${track.offset_s.toFixed(1)}" autocomplete="off">
            <button type="button" class="icon-btn icon-btn--sm" data-action="offset-step" data-key="${k}" data-step="1" aria-label="Increase offset for track ${n}">${icon('plus')}</button>
            <span class="st-track__unit" aria-hidden="true">s</span>
          </div>
        </div>
        <div class="field st-track__volume">
          <label class="field__label" for="${base}-volume">Volume<span class="visually-hidden"> for track ${n}</span></label>
          <div class="slider-row">
            <input type="range" class="slider" id="${base}-volume" data-key="${k}" data-field="volume" min="${VOLUME_MIN}" max="${VOLUME_MAX}" step="0.1" value="${track.volume}">
            <output class="tabular" id="${base}-volume-out" for="${base}-volume">${volumeText(track.volume)}</output>
          </div>
        </div>
        <div class="chips st-track__toggles" role="group" aria-label="Track ${n} in the mix">
          <button type="button" class="chip" id="${base}-mute" data-action="toggle-track" data-key="${k}" data-field="muted" aria-pressed="${track.muted ? 'true' : 'false'}" aria-label="Mute track ${n}">${icon('check', { className: 'ico chip__check' })}<span>Mute</span></button>
          <button type="button" class="chip" id="${base}-solo" data-action="toggle-track" data-key="${k}" data-field="solo" aria-pressed="${track.solo ? 'true' : 'false'}" aria-label="Solo track ${n}">${icon('check', { className: 'ico chip__check' })}<span>Solo</span></button>
        </div>
      </div>
    </div>
  </li>`;
}

function mixRowMarkup(mix, playingId, transportId) {
  const key = idPart(mix.id);
  const name = mix.name || 'Untitled mix';
  const fmt = renderFormat(mix.render_output_path).toUpperCase();
  const ms = Number(mix.meta?.duration_ms);
  const parts = [];
  if (Number.isFinite(ms) && ms > 0) parts.push(clock(ms / 1000));
  if (fmt) parts.push(fmt);
  const when = relativeTime(mix.created_at);
  const isPlaying = playingId === `mix:${mix.id}`;
  const transport = transportId === mix.id;
  return html`<li class="row row--dense st-mix-row${transport ? ' has-transport' : ''}" id="st-mix-${key}">
    <span class="row__lead"><button type="button" class="icon-btn st-mix-row__play" id="st-mix-${key}-play" data-action="play-past" data-id="${mix.id}" aria-label="${isPlaying ? 'Pause' : 'Play'} ${name}" data-tooltip="${isPlaying ? 'Pause' : 'Play'} ${name}" data-playing="${isPlaying ? 'true' : 'false'}">${icon(isPlaying ? 'pause' : 'play')}</button></span>
    <div class="st-mix-row__main"><span class="row__title">${name}</span><span class="row__meta">${parts.join(' · ')}${parts.length && when ? ' · ' : ''}${when ? html`<span title="${absoluteTime(mix.created_at)}">${when}</span>` : ''}</span></div>
    <span class="row__trail">
      <button type="button" class="icon-btn" id="st-mix-${key}-download" data-action="download-past" data-id="${mix.id}" aria-label="Download ${name}" data-tooltip="Download ${name}">${icon('download')}</button>
      ${menuButtonMarkup(`st-mix-${key}-more`, `st-mix-${key}-menu`, name)}
      ${menuMarkup(`st-mix-${key}-menu`, name, [{ cmd: 'delete-mix', arg: mix.id, label: 'Delete mix…', icon: 'trash', danger: true }])}
    </span>
    ${transport ? html`<div class="st-transport st-mix-row__transport">${transportMarkup(`st-mix-${key}-seek`, `Position in ${name}`)}</div>` : ''}
  </li>`;
}

/**
 * The past mixes list for a state (exported for tests).
 * @param {{status: string, items: object[], error: *, at: number}} list
 * @param {string} [playingId] the session that plays now ('mix:<id>')
 * @param {string} [transportId] the mix whose row carries the transport (it was played from its row)
 */
export function mixesMarkup(list, playingId = '', transportId = '') {
  if (!list.at && (list.status === 'loading' || list.status === 'idle')) return loadingMarkup('mixes', { rows: 2 });
  if (list.status === 'error' && !list.at) return loadErrorMarkup(list.error, { what: 'mixes', list: 'mixes', level: 3 });
  const items = list.items || [];
  if (!items.length) return html`<p class="st-mixes__empty">No mixes yet.</p>`;
  return html`<ul class="rows st-mix-list" aria-labelledby="st-mixes-title">${items.map((m) => mixRowMarkup(m, playingId, transportId))}</ul>`;
}

/* ───────────────────────────── the panel ───────────────────────────── */

/**
 * The Composer panel.
 * @param {HTMLElement} panel #st-composer
 * @param {{lists: object, load: (name: string) => Promise<boolean>, navigate: Function, isShown: () => boolean,
 *   isCurrentSub: () => boolean}} ctx isCurrentSub: the Composer is the section on show
 */
export function createComposer(panel, ctx) {
  const q = (id) => panel.querySelector(`#${id}`);
  const el = {
    title: q('st-mix-title'),
    tracks: q('st-tracks'),
    tracksEmpty: q('st-tracks-empty'),
    sourcesError: q('st-sources-error'),
    addTrack: q('st-add-track'),
    format: q('st-mix-format'),
    render: q('st-mix-render'),
    play: q('st-mix-play'),
    statusText: q('st-mix-status-text'),
    elapsed: q('st-mix-elapsed'),
    download: q('st-mix-download'),
    transport: q('st-mix-transport'),
    seek: q('st-mix-seek'),
    seekTime: q('st-mix-seek-time'),
    banner: q('st-mix-banner'),
    mixes: q('st-mixes'),
  };
  let keySeq = 0;
  const state = {
    tracks: [newTrack((keySeq += 1))],
    errors: {},          // track key -> {field: 'src'|'path', message}
    mixing: null,        // {startedAt, format}
    lastMix: null,       // {id, name, format, durationMs, url}
    origin: { id: '', from: '' },   // the mix played last and where from ('result' | 'row'): its transport sits there
  };
  let ranges = [];
  let elapsedTimer = null;
  let unbindMenus = () => {};
  let rowTransport = null;   // {id, binding} for the past mix played from its row
  let shownRowTransport = '';
  let sourcesErrorText = '';
  const offs = [];
  const media = new Map();   // session id -> revoke()
  const pending = new Map();   // session id -> the press whose audio is still downloading (shared.js startPress)

  const findTrack = (key) => state.tracks.find((t) => String(t.key) === String(key)) || null;
  const indexOf = (key) => state.tracks.findIndex((t) => String(t.key) === String(key));
  const selectedFormat = () => {
    const checked = el.format ? el.format.querySelector('input[name="st-mix-format"]:checked') : null;
    return checked ? checked.value : 'wav';
  };
  const sources = () => wavSources(ctx.lists.conversations.items, ctx.lists.mixes.items);
  const sourcesKnown = () => !!(ctx.lists.conversations.at && ctx.lists.mixes.at);
  const setText = (node, text) => { if (node && node.textContent !== text) node.textContent = text; };
  /** Whether the Composer is what the user sees now: the view and its Composer section. */
  const onScreen = () => !!(ctx.isShown() && (typeof ctx.isCurrentSub !== 'function' || ctx.isCurrentSub()));
  const mixPlayer = (player) => (player && String(player.id).startsWith('mix:') ? player : null);
  const resultTransport = el.seek ? bindTransport(el.seek, el.seekTime, { sessionId: () => (state.lastMix ? `mix:${state.lastMix.id}` : '') }) : null;

  /** Open the Composer (from a toast); focus goes to `target()` once it shows. */
  function openComposer(target) {
    const go = onScreen() ? Promise.resolve() : Promise.resolve(ctx.navigate('studio/composer', { focus: 'keep' }));
    return go.then(() => {
      if (onScreen()) focusElement(target());
    });
  }
  const openAction = (target) => ({ label: 'Open', run: () => openComposer(target) });

  /* ── tracks ── */

  /**
   * The line above the tracks when a list of sources didn't load, with Retry: the selects can't list
   * those renders, and the "render as WAV" hint would blame the wrong cause. Returns its id while it shows.
   */
  function renderSourcesError() {
    if (!el.sourcesError) return '';
    const text = sourcesError(ctx.lists.conversations, ctx.lists.mixes);
    if (text && (text !== sourcesErrorText || el.sourcesError.hidden)) {
      render(el.sourcesError, html`${icon('alert', { className: 'ico ico--sm' })}<span>${text}</span><button type="button" class="btn btn--ghost btn--sm" data-action="retry-sources"><span class="btn__label">Retry</span></button>`);
    }
    sourcesErrorText = text;
    hideKeepingFocus(el.sourcesError, !text, () => (state.tracks.length ? q(`st-track-${state.tracks[0].key}-src`) : el.addTrack));
    return text ? el.sourcesError.id : '';
  }

  function renderTracks({ fallback } = {}) {
    for (const r of ranges.splice(0)) r.destroy();
    const errorId = renderSourcesError();
    const src = { ...sources(), known: sourcesKnown(), errorId };
    renderKeepingFocus(el.tracks, html`${state.tracks.map((t, i) => trackMarkup(t, i + 1, src, state.errors))}`, {
      fallback: fallback || (() => el.addTrack),
    });
    for (const input of el.tracks.querySelectorAll('input[type="range"]')) {
      const out = el.tracks.querySelector(`#${input.id}-out`);
      ranges.push(bindRange(input, { output: out, format: volumeText, spoken: volumeSpoken }));
    }
    el.tracks.hidden = !state.tracks.length;
    el.tracksEmpty.hidden = !!state.tracks.length;
  }

  function setFieldError(key, field, message) {
    if (message) state.errors[key] = { field, message };
    else delete state.errors[key];
    const base = `st-track-${key}`;
    for (const which of ['src', 'path']) {
      const control = q(`${base}-${which}`);
      const errorEl = q(`${base}-${which}-error`);
      const text = message && field === which ? message : '';
      if (errorEl) {
        const span = errorEl.querySelector('span');
        if (span) span.textContent = text;
        errorEl.hidden = !text;
      }
      if (control) {
        const hint = q(`${base}-${which}-hint`) ? `${base}-${which}-hint` : '';
        const shared = which === 'src' && sourcesErrorText && el.sourcesError ? el.sourcesError.id : '';
        const ids = [hint, shared, text ? `${base}-${which}-error` : ''].filter(Boolean).join(' ');
        if (text) control.setAttribute('aria-invalid', 'true');
        else control.removeAttribute('aria-invalid');
        if (ids) control.setAttribute('aria-describedby', ids);
        else control.removeAttribute('aria-describedby');
      }
    }
  }

  function addTrack() {
    const track = newTrack((keySeq += 1));
    state.tracks.push(track);
    renderTracks();
    clearBanner(el.banner);
    announce(`Track ${state.tracks.length} added`);
    focusElement(q(`st-track-${track.key}-src`));
  }

  function removeTrack(key) {
    const index = indexOf(key);
    if (index < 0) return;
    state.tracks.splice(index, 1);
    delete state.errors[key];
    renderTracks();
    announce(`Track ${index + 1} removed`);
    const next = state.tracks[Math.min(index, state.tracks.length - 1)];
    focusElement(next ? q(`st-track-${next.key}-remove`) : el.addTrack);
  }

  function stepOffset(key, direction) {
    const track = findTrack(key);
    if (!track) return;
    track.offset_s = clampOffset(track.offset_s + direction * OFFSET_STEP);
    const input = q(`st-track-${key}-offset`);
    if (input) input.value = track.offset_s.toFixed(1);
    const dec = el.tracks.querySelector(`[data-action="offset-step"][data-key="${key}"][data-step="-1"]`);
    if (dec) {
      const wasFocused = document.activeElement === dec;
      dec.disabled = track.offset_s <= 0;
      if (wasFocused && dec.disabled) focusElement(input);
    }
  }

  function toggle(key, field) {
    const track = findTrack(key);
    if (!track || (field !== 'muted' && field !== 'solo')) return;
    track[field] = !track[field];
    const button = q(`st-track-${key}-${field === 'muted' ? 'mute' : 'solo'}`);
    if (button) button.setAttribute('aria-pressed', track[field] ? 'true' : 'false');
  }

  /** CO-02: every edit updates the track model at once (numbers for offset and volume). */
  function onInput(event) {
    const target = event.target && event.target.closest ? event.target.closest('[data-field]') : null;
    if (!target) return;
    const track = findTrack(target.getAttribute('data-key'));
    if (!track) return;
    const field = target.getAttribute('data-field');
    if (field === 'offset') {
      track.offset_s = clampOffset(target.value);
    } else if (field === 'volume') {
      track.volume = clampVolume(target.value);
    } else if (field === 'path') {
      track.source_path = String(target.value || '');
      if (state.errors[track.key]) setFieldError(track.key, '', '');
    }
  }

  function onChange(event) {
    const target = event.target && event.target.closest ? event.target.closest('[data-field]') : null;
    if (!target) return;
    const track = findTrack(target.getAttribute('data-key'));
    if (!track) return;
    const field = target.getAttribute('data-field');
    if (field === 'source') {
      const value = target.value;
      const pathField = q(`st-track-${track.key}-path-field`);
      const pathInput = q(`st-track-${track.key}-path`);
      if (value === OTHER) {
        track.other = true;
        track.source_path = pathInput ? String(pathInput.value || '') : '';
      } else {
        track.other = false;
        track.source_path = value.startsWith('p:') ? value.slice(2) : '';
      }
      if (pathField) pathField.hidden = !track.other;
      setFieldError(track.key, '', '');
    } else if (field === 'offset') {
      track.offset_s = clampOffset(target.value);
      target.value = track.offset_s.toFixed(1);
      const dec = el.tracks.querySelector(`[data-action="offset-step"][data-key="${track.key}"][data-step="-1"]`);
      if (dec) dec.disabled = track.offset_s <= 0;
    } else if (field === 'volume') {
      track.volume = clampVolume(target.value);
    }
  }
  el.tracks.addEventListener('input', onInput);
  el.tracks.addEventListener('change', onChange);
  offs.push(() => el.tracks.removeEventListener('input', onInput), () => el.tracks.removeEventListener('change', onChange));

  /* ── the mix ── */

  function stopElapsed() {
    if (elapsedTimer !== null) clearInterval(elapsedTimer);
    elapsedTimer = null;
    setText(el.elapsed, '');
  }

  function tick() {
    if (!state.mixing) return;
    setText(el.elapsed, ` ${clock((Date.now() - state.mixing.startedAt) / 1000)}`);
  }

  function renderMixSection() {
    const mix = state.lastMix;
    if (state.mixing) {
      setBusy(el.render, true, { label: 'Rendering…' });
      setText(el.statusText, 'Rendering…');
      tick();
      if (elapsedTimer === null && onScreen()) elapsedTimer = setInterval(tick, 1000);
    } else {
      setBusy(el.render, false);
      stopElapsed();
      const ms = mix ? Number(mix.durationMs) : NaN;
      setText(el.statusText, mix ? (Number.isFinite(ms) && ms > 0 ? `Mixed · ${clock(ms / 1000)}` : 'Mixed') : 'Ready');
    }
    const result = !!mix && !state.mixing;
    hideKeepingFocus(el.play, !result, el.render);
    hideKeepingFocus(el.download, !result, el.render);
    if (mix) setText(el.download.querySelector('.btn__label'), `Download ${String(mix.format || 'wav').toUpperCase()}`);
    syncPlayButtons();
    syncResultTransport();
  }

  /** Render mix (CO-03): checks the tracks first, then POSTs; errors land on the track that caused them. */
  async function renderMix() {
    if (state.mixing) return;
    clearBanner(el.banner);
    for (const key of Object.keys(state.errors)) setFieldError(key, '', '');
    if (!state.tracks.length) {
      showBanner(el.banner, { title: 'Add at least one track', message: 'A mix needs a track with a WAV source.', dismiss: true });
      return;
    }
    const active = activeTracks(state.tracks);
    if (!active.length) {
      showBanner(el.banner, { title: 'Every track is muted', message: 'Unmute a track to render the mix.' });
      return;
    }
    let first = null;
    for (const track of active) {
      if (String(track.source_path || '').trim()) continue;
      if (track.other) setFieldError(track.key, 'path', 'Enter a server path.');
      else setFieldError(track.key, 'src', 'Choose a source for this track.');
      if (!first) first = q(`st-track-${track.key}-${track.other ? 'path' : 'src'}`);
    }
    if (first) {
      focusElement(first);
      return;
    }
    const format = selectedFormat();
    const name = `Composition ${absoluteTime(Date.now())}`;
    state.mixing = { startedAt: Date.now(), format };
    renderMixSection();
    try {
      const data = await api.post('/api/composer/render', {
        name, format, sample_rate: SAMPLE_RATE, tracks: state.tracks.map(trackPayload),
      }, { accept: 'json' });
      state.mixing = null;
      state.lastMix = {
        id: data?.composition_id || '', name, format, durationMs: Number(data?.duration_ms),
        url: data?.download_url || (data?.composition_id ? `/api/composer/render/${seg(data.composition_id)}/audio` : ''),
      };
      renderMixSection();
      ctx.load('mixes');
      // Seen here, the status line says it (role=status); from elsewhere a toast does, with the way back.
      if (!onScreen()) toast('The mix is ready', { kind: 'success', action: openAction(() => (el.play.hidden ? el.title : el.play)) });
    } catch (error) {
      state.mixing = null;
      renderMixSection();
      if (error && error.kind === 'aborted') return;
      showMixError(error);
    }
  }

  /**
   * A 403 or 400 names a track (or its path): the error goes on that track's field, else in the banner. The
   * banner stays for the user's return; when the Composer isn't on screen a persistent error toast says it too,
   * with Open.
   */
  function showMixError(error) {
    const copy = errorCopy(error, "Couldn't render the mix");
    const status = error ? error.status : 0;
    let index = status === 400 || status === 403 ? trackNamedIn(error.message, state.tracks) : -1;
    if (index < 0 && status === 403) {
      // The only track with a typed path is the one the server refused.
      const typed = activeTracks(state.tracks).filter((t) => t.other);
      if (typed.length === 1) index = indexOf(typed[0].key);
    }
    const here = onScreen();
    if (index >= 0) {
      const track = state.tracks[index];
      const field = track.other ? 'path' : 'src';
      const message = status === 403 ? 'The server doesn’t allow this path. Use a WAV file inside its data folder.' : copy.message;
      setFieldError(track.key, field, message);
      showBanner(el.banner, { title: copy.title, message: `Check track ${index + 1}.`, detail: copy.detail });
      const target = () => q(`st-track-${track.key}-${field}`);
      if (here) {
        // Focus goes to the field from Render mix (or from nowhere), never out of a control the user moved to.
        const active = document.activeElement;
        if (!active || active === document.body || active === el.render) focusElement(target());
      } else {
        toast(`${copy.title}: check track ${index + 1}.`, { kind: 'error', action: openAction(target) });
      }
      return;
    }
    showBanner(el.banner, { title: copy.title, message: copy.message, detail: copy.detail, actions: bannerActions(copy, { cmd: 'render-mix', label: 'Retry' }) });
    if (!here) toast(`${copy.title}: ${copy.message}`, { kind: 'error', action: openAction(() => el.title) });
  }

  /* ── playback and downloads ── */

  function syncPlayButtons() {
    const player = app.get().player;
    const playingId = player && player.playing ? player.id : '';
    if (state.lastMix && !el.play.hidden) setPlayButton(el.play, { playing: playingId === `mix:${state.lastMix.id}`, label: 'mix' });
    const names = new Map((ctx.lists.mixes.items || []).map((m) => [m.id, m.name]));
    for (const button of el.mixes.querySelectorAll('[data-action="play-past"]')) {
      const id = button.getAttribute('data-id');
      setPlayButton(button, { playing: playingId === `mix:${id}`, label: names.get(id) || 'Untitled mix' });
    }
  }

  /**
   * Whether a mix's session knows its length (a transport from 0:00 to 0:00 says nothing). core/audio's playClip
   * reports it as soon as the clip's metadata loads, also while a refused autoplay waits paused.
   */
  const knowsLength = (player) => !!player && Number(player.duration) > 0;

  /** The result's transport: while the last mix, played from the result, is the current session. */
  function syncResultTransport(player = app.get().player) {
    if (!el.transport || !resultTransport) return;
    const mix = state.lastMix;
    const own = mix && !el.play.hidden && state.origin.from === 'result' && state.origin.id === mix.id
      && player && player.id === `mix:${mix.id}` && knowsLength(player) ? player : null;
    hideKeepingFocus(el.transport, !own, el.play);
    if (own) resultTransport.paint(own);
  }

  /** The past mix whose row carries the transport: played from its row, the current session, its length known. */
  function rowTransportId(player = app.get().player) {
    const id = state.origin.from === 'row' ? state.origin.id : '';
    return id && player && player.id === `mix:${id}` && knowsLength(player) ? id : '';
  }

  function bindRowTransport(player = app.get().player) {
    if (rowTransport) rowTransport.binding.destroy();
    rowTransport = null;
    const id = rowTransportId(player);
    const input = id ? q(`st-mix-${idPart(id)}-seek`) : null;
    if (!input) return;
    rowTransport = { id, binding: bindTransport(input, input.nextElementSibling, { sessionId: () => `mix:${id}` }) };
    rowTransport.binding.paint(player);
  }

  /**
   * Play a mix on the shared element: prime() inside the click, then the (keyed) media URL. The newest press
   * wins (shared.js startPress()): a slow download that lands after another press, or after other audio started,
   * is dropped; a repeat press while its audio loads makes that press the newest again, with one download.
   */
  async function playMix({ id, name, url }, from) {
    const sessionId = `mix:${id}`;
    state.origin = { id, from };
    const waiting = pending.get(sessionId);
    if (waiting) {
      waiting.renew();
      return;
    }
    const current = audio.current();
    if (current && current.id === sessionId) {
      current.controls.toggle();
      return;
    }
    audio.prime();
    const press = startPress();
    pending.set(sessionId, press);
    let handle = null;
    let started = false;
    try {
      handle = await api.mediaUrl(url);
      if (pending.get(sessionId) === press) pending.delete(sessionId);
      if (!press.current()) {
        handle.revoke();
        return;
      }
      started = true;
      const revoke = handle.revoke;
      media.set(sessionId, revoke);
      await audio.playClip({
        id: sessionId, owner: 'studio', title: name || 'Mix', subtitle: 'Composer mix', homeRoute: 'studio/composer', src: handle.url,
        onRelease: () => { revoke(); media.delete(sessionId); },
      });
    } catch (error) {
      if (pending.get(sessionId) === press) pending.delete(sessionId);
      if (handle) handle.revoke();
      if (error && error.kind === 'aborted') return;
      if (!started && !press.current()) return;   // a download nobody waits for any more failed: nothing to say
      const copy = errorCopy(error, "Couldn't play");
      toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
    }
  }

  async function downloadMix(button, path, filename) {
    setBusy(button, true);
    try {
      await downloadFromApi(path, filename);
    } catch (error) {
      const copy = errorCopy(error, "Couldn't download");
      toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
    } finally {
      setBusy(button, false);
    }
  }

  /* ── past mixes ── */

  function renderMixes({ fallback } = {}) {
    unbindMenus();
    const list = ctx.lists.mixes;
    el.mixes.setAttribute('aria-busy', list.status === 'loading' && !list.at ? 'true' : 'false');
    const player = app.get().player;
    const transportId = rowTransportId(player);
    renderKeepingFocus(el.mixes, mixesMarkup(list, player && player.playing ? player.id : '', transportId), {
      fallback: fallback || (() => el.mixes.querySelector('.st-mix-row__play') || q('st-mixes-title')),
    });
    shownRowTransport = transportId;
    bindRowTransport(player);
    if (list.status === 'error' && !list.at) fillDetails(el.mixes, errorCopy(list.error, "Couldn't load mixes").detail);
    unbindMenus = bindMenus(el.mixes, (cmd, item) => {
      if (cmd === 'delete-mix') deleteMix(item.getAttribute('data-arg'));
    });
  }

  async function deleteMix(id) {
    const list = ctx.lists.mixes;
    const mix = (list.items || []).find((m) => m.id === id);
    if (!mix) return;
    const name = mix.name || 'Untitled mix';
    const ok = await confirm({ title: `Delete the mix “${name}”?`, message: 'This can’t be undone.', confirmLabel: 'Delete', danger: true });
    if (!ok) return;
    const index = (list.items || []).indexOf(mix);
    try {
      await api.del(`/api/composer/render/${seg(id)}`);
    } catch (error) {
      if (!(error && error.status === 404)) {
        const copy = errorCopy(error, "Couldn't delete");
        toast(`${copy.title}: ${copy.message}`, { kind: 'error' });
        return;
      }
    }
    const current = audio.current();
    if (current && current.id === `mix:${id}`) audio.closeCurrent();
    if (pending.has(`mix:${id}`)) pending.get(`mix:${id}`).cancel();   // its audio is gone: it never plays
    if (state.lastMix && state.lastMix.id === id) {
      state.lastMix = null;
      renderMixSection();
    }
    list.items = (list.items || []).filter((m) => m.id !== id);
    renderMixes({
      fallback: () => {
        const plays = el.mixes.querySelectorAll('.st-mix-row__play');
        return plays.length ? plays[Math.min(index, plays.length - 1)] : q('st-mixes-title');
      },
    });
    renderTracks();
    toast(`Deleted “${name}”`, { kind: 'success' });
    const active = document.activeElement;
    if (!active || active === document.body) {
      const plays = el.mixes.querySelectorAll('.st-mix-row__play');
      focusElement(plays.length ? plays[Math.min(index, plays.length - 1)] : q('st-mixes-title'));
    }
    ctx.load('mixes');
  }

  // Play buttons follow which session plays and whether it does, never its 4 Hz position updates; the row that
  // carries a transport follows the playing mix; the transports follow a mix's position, and only a mix's.
  offs.push(app.select(playerKey, () => syncPlayButtons()));
  offs.push(app.select((s) => rowTransportId(s.player), (id) => { if (id !== shownRowTransport) renderMixes(); }));
  offs.push(app.select((s) => mixPlayer(s.player), (player) => {
    syncResultTransport(player);
    if (rowTransport && player && player.id === `mix:${rowTransport.id}`) rowTransport.binding.paint(player);
  }));
  if (resultTransport) offs.push(() => resultTransport.destroy());

  const actions = {
    'add-track': () => addTrack(),
    'remove-track': (button) => removeTrack(button.getAttribute('data-key')),
    'offset-step': (button) => stepOffset(button.getAttribute('data-key'), Number(button.getAttribute('data-step')) || 0),
    'toggle-track': (button) => toggle(button.getAttribute('data-key'), button.getAttribute('data-field')),
    'render-mix': () => renderMix(),
    'play-mix': () => (state.lastMix ? playMix(state.lastMix, 'result') : undefined),
    'download-mix': (button) => (state.lastMix ? downloadMix(button, state.lastMix.url, mixFileName(state.lastMix.format)) : undefined),
    'play-past': (button) => {
      const id = button.getAttribute('data-id');
      const mix = (ctx.lists.mixes.items || []).find((m) => m.id === id);
      if (mix) return playMix({ id, name: mix.name, url: `/api/composer/render/${seg(id)}/audio` }, 'row');
      return undefined;
    },
    'download-past': (button) => {
      const id = button.getAttribute('data-id');
      const mix = (ctx.lists.mixes.items || []).find((m) => m.id === id);
      if (!mix) return undefined;
      const ext = renderFormat(mix.render_output_path) || 'wav';
      return downloadMix(button, `/api/composer/render/${seg(id)}/audio`, `composer-${id}.${ext}`);
    },
    'retry-sources': () => Promise.all(['conversations', 'mixes']
      .filter((name) => ctx.lists[name].status === 'error')
      .map((name) => ctx.load(name))),
  };

  renderTracks();
  renderMixSection();

  // The selects are rebuilt only when what they can list (or why they can't) changes.
  const signature = () => {
    const src = sources();
    return JSON.stringify([src.conversations, src.mixes, sourcesKnown(), sourcesError(ctx.lists.conversations, ctx.lists.mixes)]);
  };
  let sourcesSig = signature();

  return {
    actions,
    /** The source lists changed (conversations or mixes loaded or failed): the selects follow when they must. */
    renderSources() {
      const sig = signature();
      if (sig === sourcesSig) return;
      sourcesSig = sig;
      renderTracks();
    },
    renderMixes,
    show() {
      if (state.mixing && elapsedTimer === null) elapsedTimer = setInterval(tick, 1000);
      tick();
    },
    hide() {
      stopElapsed();
    },
    /** Screenshot states. */
    applyShot(shot) {
      const src = sources();
      if (shot === 'tracks') {
        const first = src.conversations[0] || src.mixes[0];
        state.tracks = [
          { ...newTrack((keySeq += 1)), source_path: first ? first.path : '' },
          { ...newTrack((keySeq += 1)), source_path: (src.mixes[0] || src.conversations[0] || {}).path || '', offset_s: 1.5, volume: 0.7, muted: true },
          { ...newTrack((keySeq += 1)), other: true, source_path: 'data/voices/intro.wav', solo: false },
        ];
        renderTracks();
      } else if (shot === 'mix-error') {
        const track = state.tracks[0];
        if (track) {
          track.other = true;
          track.source_path = '/etc/music.wav';
          renderTracks();
          setFieldError(track.key, 'path', 'The server doesn’t allow this path. Use a WAV file inside its data folder.');
          showBanner(el.banner, { title: "Couldn't render the mix", message: 'Check track 1.' });
        }
      } else if (shot === 'mix-result' || shot === 'mix-playing') {
        const mix = (ctx.lists.mixes.items || [])[0];
        state.lastMix = { id: mix ? mix.id : 'demo', name: mix ? mix.name : 'Composition', format: 'wav', durationMs: Number(mix?.meta?.duration_ms) || 9000, url: mix ? `/api/composer/render/${seg(mix.id)}/audio` : '' };
        renderMixSection();
        if (shot === 'mix-playing' && mix) playMix(state.lastMix, 'result');
      } else if (shot === 'past-playing') {
        const mix = (ctx.lists.mixes.items || [])[0];
        if (mix) playMix({ id: mix.id, name: mix.name, url: `/api/composer/render/${seg(mix.id)}/audio` }, 'row');
      } else if (shot === 'mixing') {
        state.mixing = { startedAt: Date.now() - 4000, format: selectedFormat() };
        renderMixSection();
        stopElapsed();
        setText(el.elapsed, ' 0:04');
      }
    },
    destroy() {
      unbindMenus();
      stopElapsed();
      if (rowTransport) rowTransport.binding.destroy();
      rowTransport = null;
      for (const r of ranges.splice(0)) r.destroy();
      for (const off of offs.splice(0)) off();
      for (const revoke of media.values()) revoke();
      media.clear();
    },
    __state: state,
  };
}
