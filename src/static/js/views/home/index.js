/**
 * Home: the landing page. A greeting with the server's status, the four main shortcuts (static links in
 * the partial), the Reader document to pick up again and the newest History entries.
 *
 * Data (GS-12): the first show() loads both lists (GET /v1/reader/documents and GET /api/history); a later show
 * reloads when they are older than 30 s, or after 'library:changed', 'history:changed' or 'auth:changed'. Each
 * list has its own request token, loading state, empty state and error with Retry. Nothing here writes to the
 * server or asks the AI assistant.
 * @module views/home/index
 */
import { html, render, actions as bindActions } from '../../core/dom.js';
import * as api from '../../core/api.js';
import { app, on as onEvent } from '../../core/store.js';
import { chipModel } from '../../core/status.js';
import { percent, relativeTime, plural } from '../../core/format.js';
import { applyTile, monogram } from '../../core/tiles.js';
import { icon } from '../../core/icons.js';
import { navigate, format as formatRoute } from '../../core/router.js';
import { openStatusSheet } from '../../shell/status-chip.js';

export const DOCUMENTS_PATH = '/v1/reader/documents';
export const RECENT_LIMIT = 3;
export const HISTORY_PATH = `/api/history?limit=${RECENT_LIMIT}`;
export const STALE_MS = 30000;

export default {
  id: 'home',
  title: 'Home',
  partial: '/static/partials/home.html',
  init(root) {
    return createHomeView(root);
  },
};

/** PURE: the greeting for a local time of day. */
export function greeting(now = new Date()) {
  const hour = now.getHours();
  if (hour >= 5 && hour < 12) return 'Good morning';
  if (hour >= 12 && hour < 18) return 'Good afternoon';
  return 'Good evening';
}

const savedAt = (doc) => Date.parse((doc.progress && doc.progress.updated_at) || '') || 0;
const addedAt = (doc) => Date.parse(doc.updated_at || doc.created_at || '') || 0;

/**
 * PURE: what "Pick up where you left off" shows. The unfinished document whose place was saved last; with
 * none started, the newest document to begin; null for an empty library.
 * @param {object[]} documents GET /v1/reader/documents
 * @returns {{id: string, title: string, started: boolean, pct: number|null, meta: string}|null}
 */
export function continueModel(documents) {
  const docs = (Array.isArray(documents) ? documents : []).filter((d) => d && d.id);
  if (!docs.length) return null;
  const pctOf = (d) => (d.progress ? percent(d.progress.block_index, d.block_count) : null);
  const started = docs.filter((d) => d.progress && pctOf(d) !== 100).sort((a, b) => savedAt(b) - savedAt(a));
  const doc = started[0] || [...docs].sort((a, b) => addedAt(b) - addedAt(a))[0];
  const isStarted = doc === started[0];
  const pct = isStarted ? pctOf(doc) : null;
  const chapters = Number(doc.chapter_count) > 0 ? plural(doc.chapter_count, 'chapter') : '';
  const meta = isStarted
    ? [pct === null ? '' : `${pct}% listened`, relativeTime(savedAt(doc))].filter(Boolean).join(' · ')
    : [chapters, 'Not started'].filter(Boolean).join(' · ');
  return { id: String(doc.id), title: doc.title || 'Untitled', started: isStarted, pct, meta };
}

/**
 * PURE: the newest History entries as rows.
 * @param {object[]} items GET /api/history items, newest first
 * @returns {{id: string, type: 'tts'|'stt', text: string, meta: string}[]}
 */
export function recentModel(items, now = Date.now()) {
  return (Array.isArray(items) ? items : []).filter((item) => item && item.id).slice(0, RECENT_LIMIT).map((item) => {
    const type = item.type === 'stt' ? 'stt' : 'tts';
    const text = String(item.text_preview || item.input_filename || '').trim() || (type === 'stt' ? 'A transcript' : 'Speech');
    const meta = [type === 'stt' ? 'Transcript' : 'Speech', type === 'tts' ? item.voice : item.input_filename, relativeTime(item.created_at, now)]
      .filter(Boolean).join(' · ');
    return { id: String(item.id), type, text, meta };
  });
}

/**
 * Build the controller for the Home section (exported for tests).
 * @param {HTMLElement} root the view section, with the partial rendered into it
 * @param {{now?: () => Date}} [options]
 */
export function createHomeView(root, { now = () => new Date() } = {}) {
  const el = {
    title: root.querySelector('#hm-title'),
    status: root.querySelector('#hm-status'),
    pill: root.querySelector('#hm-status-pill'),
    pillLabel: root.querySelector('#hm-status-label'),
    cont: root.querySelector('#hm-continue'),
    recent: root.querySelector('#hm-recent'),
    all: root.querySelector('#hm-all'),
  };
  const st = { shown: false, dirty: true, loadedAt: 0, docSeq: 0, recentSeq: 0 };
  const offs = [];

  const failed = (host, error, action) => {
    host.removeAttribute('aria-busy');
    render(host, html`<div class="hm-note" role="alert"><p class="hm-note__text">${api.explain(error).message}</p><button type="button" class="btn btn--secondary btn--sm" data-action="${action}">${icon('retry')}<span class="btn__label">Retry</span></button></div>`);
  };

  function paintContinue(model) {
    el.cont.removeAttribute('aria-busy');
    if (!model) {
      render(el.cont, html`<div class="hm-note"><p class="hm-note__text">Your library is empty. Add a book, an article or pasted text to listen to it.</p><a class="btn btn--secondary btn--sm" href="#reader">${icon('plus')}<span class="btn__label">Open the Reader</span></a></div>`);
      return;
    }
    render(el.cont, html`<button type="button" class="hm-doc" data-action="open-doc" data-id="${model.id}">
      <span class="tile hm-doc__tile" data-tile="${model.id}" aria-hidden="true">${monogram(model.title)}</span>
      <span class="hm-doc__text">
        <span class="hm-doc__title">${model.title}</span>
        <span class="hm-doc__meta">${model.meta}</span>
        ${model.pct === null ? '' : html`<progress class="progress progress--thin hm-doc__bar" max="100" value="${model.pct}" aria-hidden="true"></progress>`}
      </span>
      <span class="hm-doc__go">${icon('play')}<span>${model.started ? 'Resume' : 'Start'}</span></span>
    </button>`);
    for (const tile of el.cont.querySelectorAll('[data-tile]')) applyTile(tile, tile.getAttribute('data-tile'));
  }

  function paintRecent(rows) {
    el.recent.removeAttribute('aria-busy');
    el.all.hidden = !rows.length;
    if (!rows.length) {
      render(el.recent, html`<div class="hm-note"><p class="hm-note__text">Nothing yet. What you generate in Speak and transcribe in Transcribe shows up here.</p></div>`);
      return;
    }
    render(el.recent, html`<ul class="hm-items">${rows.map((row) => html`<li><button type="button" class="hm-item" data-action="open-history">
      <span class="hm-item__icon" data-type="${row.type}">${icon(row.type === 'stt' ? 'mic' : 'speak')}</span>
      <span class="hm-item__text"><span class="hm-item__title">${row.text}</span><span class="hm-item__meta">${row.meta}</span></span>
      ${icon('chevron-right', { className: 'ico hm-item__chevron' })}
    </button></li>`)}</ul>`);
  }

  async function loadContinue() {
    const seq = ++st.docSeq;
    try {
      const data = await api.get(DOCUMENTS_PATH);
      if (seq === st.docSeq) paintContinue(continueModel(data && data.documents));
    } catch (error) {
      if (seq === st.docSeq && !(error && error.kind === 'aborted')) failed(el.cont, error, 'retry-continue');
    }
  }

  async function loadRecent() {
    const seq = ++st.recentSeq;
    try {
      const data = await api.get(HISTORY_PATH, { accept: 'json' });
      if (seq === st.recentSeq) paintRecent(recentModel(data && data.items, now().getTime()));
    } catch (error) {
      if (seq === st.recentSeq && !(error && error.kind === 'aborted')) failed(el.recent, error, 'retry-recent');
    }
  }

  function load() {
    st.dirty = false;
    st.loadedAt = now().getTime();
    return Promise.all([loadContinue(), loadRecent()]);
  }

  function paintStatus() {
    const model = chipModel(app.get());
    if (el.pillLabel.textContent !== model.label) el.pillLabel.textContent = model.label;
    el.pill.className = `pill pill--${model.tone}`;
    el.status.setAttribute('aria-label', model.ariaLabel);
  }

  offs.push(bindActions(root, {
    status: (button) => openStatusSheet(button),
    'open-doc': (button) => navigate(formatRoute({ view: 'reader', id: button.getAttribute('data-id') })),
    'open-history': () => navigate('history'),
    'retry-continue': () => loadContinue(),
    'retry-recent': () => loadRecent(),
  }));
  offs.push(app.select((s) => { const m = chipModel(s); return `${m.tone}|${m.label}|${m.ariaLabel}`; }, () => paintStatus(), { immediate: true }));
  const changed = () => {
    if (st.shown) load();
    else st.dirty = true;
  };
  for (const type of ['library:changed', 'history:changed', 'auth:changed']) offs.push(onEvent(type, changed));

  return {
    show() {
      st.shown = true;
      const title = greeting(now());
      if (el.title.textContent !== title) el.title.textContent = title;
      if (st.dirty || now().getTime() - st.loadedAt > STALE_MS) load();
      return undefined;
    },
    hide() {
      st.shown = false;
    },
    destroy() {
      for (const off of offs) off();
    },
    /** Test hooks. */
    __internals: { st, load },
  };
}
