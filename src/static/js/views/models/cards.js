/**
 * The Models view's provider cards.
 *
 * One card per provider: a disclosure header (aria-expanded/aria-controls; the body is `hidden` while
 * collapsed, so it leaves the tab order and the accessibility tree) and a body that is one of
 * - a model list: one `.mo-row[data-model-id]` per model, in the state vocabulary of `models.uiState`
 *   (one primary action, "Delete files…" in an overflow menu), with a filter, "Downloaded only" and
 *   "Show all N" once a provider has more than five models (Piper, faster-whisper); what a filter or
 *   "Downloaded only" leaves on screen is announced once the listener pauses ("No voices match “zzz”");
 * - an install card (provider not installed / worker not configured) with the exact command and Copy;
 * - an offline card (the worker isn't responding, or it answers but can't run: then it says so, with the
 *   worker's own reason as text): never described as "not installed" (MD-02).
 *
 * Rows are rendered once and patched in place (BUG-10): an update from polling touches only the rows whose
 * state signature changed, never replaces a focused element, keeps collapse, filter and "show all" in
 * module state, and keeps the order rows were first shown in, so nothing jumps under the pointer. A row that
 * leaves the list while it holds focus hands focus to its neighbour's action (never to <body>); a busy action
 * button is kept only while it has focus. Under "Downloaded only" a failed operation keeps its row until its
 * error is dismissed. Polling itself belongs to `models.runOp` (core); this module only renders app.models
 * and app.ops.
 * @module views/models/cards
 */
import { html, render, setHidden, debounce } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as models from '../../core/models.js';
import * as perf from '../../core/perf.js';
import { sizeMb, plural, clock } from '../../core/format.js';
import { copyText } from '../../core/clipboard.js';
import { announce } from '../../core/a11y.js';
import { toast } from '../../core/toast.js';
import { bindMenuButton } from '../../components/menu.js';

/** Rows shown before "Show all N" (MD-10, MD-11). */
export const LIST_LIMIT = 5;
const COPIED_MS = 1500;
// A list's filter result is said once typing pauses (the same pause as the Voices search).
export const FILTER_ANNOUNCE_MS = 700;

const ON_DISK = new Set(['loaded', 'downloaded', 'ready']);
const IN_FLIGHT = new Set(['queued', 'downloading', 'loading']);
const BUSY_TEXT = { queued: 'Queued', downloading: 'Downloading…', loading: 'Loading…', unloading: 'Unloading…', deleting: 'Deleting…' };
const QUALITY_RANK = { x_low: 0, low: 1, medium: 2, high: 3 };

/* ───────────────────────────── pure helpers ───────────────────────────── */

/** The provider key of a /api/models row. */
export function providerKey(model) {
  return (model && model.provider) || models.providerOf(model && model.id);
}

/** 'Kokoro', 'Piper', 'faster-whisper', or the raw provider name. */
export function providerName(provider) {
  return models.PROVIDER_DISPLAY[provider] || String(provider || '');
}

/** True for a provider whose only model is the provider itself (Kokoro, Pocket TTS): the card is the row. */
export function isSolo(card) {
  return !!card && card.type === 'models' && card.models.length === 1 && card.models[0].id === card.provider;
}

/** The row title: STT short name, Piper "Amy (US)", "Kokoro", or a worker's variant ("regular"). */
export function rowTitle(model) {
  const m = model || {};
  const id = String(m.id || '');
  const provider = providerKey(m);
  if (models.classifyKind(m) === 'stt') return models.shortName(m);
  if (provider === 'piper') return models.parsePiperVoice(id).name;
  if (id === provider) return providerName(provider);
  return id.includes('/') ? id.split('/').slice(1).join('/') : id;
}

/** Capability words for a TTS model: '54 voices', 'Blends', 'Streaming', 'Cloning' (MD-12: a count, not chips). */
export function features(model) {
  const caps = (model && model.capabilities) || {};
  const out = [];
  const voices = Array.isArray(caps.speakers) ? caps.speakers.length : 0;
  if (voices) out.push(plural(voices, 'voice'));
  if (caps.voice_blend) out.push('Blends');
  if (caps.streaming) out.push('Streaming');
  if (caps.voice_clone) out.push('Cloning');
  return out;
}

/** The meta line under a row title: quality and size, description, source · format (MD-10, MD-11). */
export function rowMeta(model) {
  const m = model || {};
  const provider = providerKey(m);
  const parts = [];
  if (provider === 'piper') parts.push(`${models.parsePiperVoice(m.id).qualityLabel} quality`);
  else if (m.id === provider) parts.push(...features(m));
  else if (m.description) parts.push(String(m.description));
  if (models.classifyKind(m) === 'stt') {
    if (m.source) parts.push(String(m.source));
    if (m.model_format) parts.push(String(m.model_format));
  }
  const size = sizeMb(m.size_mb);
  if (size) parts.push(size);
  return parts.join(' · ');
}

/** The name the summary line uses: 'Kokoro', 'Piper Amy (US)', 'whisper base'. */
export function loadedName(model) {
  const m = model || {};
  const provider = providerKey(m);
  if (models.classifyKind(m) === 'stt') return `whisper ${models.shortName(m)}`;
  if (provider === 'piper') return `Piper ${models.parsePiperVoice(m.id).name}`;
  return models.modelLabel(m.id);
}

/**
 * PURE: the #mo-summary line (MD-01): what is loaded, speech first. It changes only when the loaded set
 * changes, so the polite status region speaks once per finished operation, never per poll.
 */
export function summaryText(list) {
  const loaded = (list || []).filter((m) => m && m.state === 'loaded');
  if (!loaded.length) return 'No model loaded';
  const ordered = [...loaded.filter((m) => models.classifyKind(m) === 'tts'), ...loaded.filter((m) => models.classifyKind(m) === 'stt')];
  const names = ordered.map((m) => {
    const device = models.deviceLabel(m.device);
    return `${loadedName(m)}${device ? ` on ${device}` : ''}`;
  });
  return `${loaded.length} loaded · ${names.join(', ')}`;
}

const stateRank = (m) => (m.state === 'loaded' ? 0 : ON_DISK.has(m.state) ? 1 : 2);

/**
 * PURE: the order rows are first shown in. Loaded, then downloaded, then the rest; Piper puts en_US before
 * other locales, then the voice name, then quality (MD-10); STT and the others sort by short name (MD-11).
 */
export function sortModels(provider, rows) {
  const list = (rows || []).slice();
  if (provider === 'piper') {
    const info = (m) => models.parsePiperVoice(m.id);
    const us = (m) => (info(m).lang === 'en_US' ? 0 : 1);
    return list.sort((a, b) => stateRank(a) - stateRank(b) || us(a) - us(b)
      || info(a).name.localeCompare(info(b).name) || (QUALITY_RANK[info(a).quality] ?? 9) - (QUALITY_RANK[info(b).quality] ?? 9)
      || String(a.id).localeCompare(String(b.id)));
  }
  return list.sort((a, b) => stateRank(a) - stateRank(b) || rowTitle(a).localeCompare(rowTitle(b)) || String(a.id).localeCompare(String(b.id)));
}

const MISSING_KEYS = new Set(['not-installed', 'worker-missing']);
// A configured worker that can do no work: it doesn't answer, or it answers but can't run.
const OFFLINE_KEYS = new Set(['worker-offline', 'worker-failed']);

/**
 * PURE: the cards for one model kind ('tts' | 'stt'): [{key, provider, type, models}], type 'models',
 * 'install' (every model's provider is missing) or 'offline' (the worker isn't responding: core uiState()
 * also reads a model the server still lists as loaded under a worker that stopped answering as
 * 'worker-offline', never as not configured; MD-02; or it answers but can't run: 'worker-failed').
 * Ordered: cards with a loaded model, then Kokoro, Piper
 * (faster-whisper for STT), other installed providers, offline workers, then not-installed providers.
 */
export function groupCards(list, kind, { ops = {} } = {}) {
  const byProvider = new Map();
  for (const m of list || []) {
    if (!m || !m.id || models.classifyKind(m) !== kind) continue;
    const p = providerKey(m);
    if (!byProvider.has(p)) byProvider.set(p, []);
    byProvider.get(p).push(m);
  }
  const cards = [];
  for (const [provider, rows] of byProvider) {
    const keys = rows.map((m) => models.uiState(m, ops[m.id], { list }).key);
    let type = 'models';
    if (keys.every((k) => MISSING_KEYS.has(k))) type = 'install';
    else if (keys.every((k) => MISSING_KEYS.has(k) || OFFLINE_KEYS.has(k))) type = 'offline';
    cards.push({ key: `${kind}:${provider}`, provider, type, models: rows });
  }
  const typeRank = { offline: 4, install: 5 };
  const rank = (c) => {
    if (typeRank[c.type]) return typeRank[c.type];
    if (c.models.some((m) => m.state === 'loaded')) return 0;
    return 1;
  };
  const fixed = (p) => ({ kokoro: 0, piper: 1, 'faster-whisper': 0 }[p] ?? 2);
  return cards.sort((a, b) => rank(a) - rank(b) || fixed(a.provider) - fixed(b.provider)
    || providerName(a.provider).localeCompare(providerName(b.provider)));
}

/** PURE: whether a model matches the filter text (every word, in its id, title, meta or description). */
export function matchesFilter(model, text) {
  const words = String(text || '').trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return true;
  const hay = [model.id, rowTitle(model), rowMeta(model), model.description || ''].join(' ').toLowerCase();
  return words.every((w) => hay.includes(w));
}

/**
 * PURE: on disk, or busy getting there (a running download stays in "Downloaded only"), or its operation
 * failed: the row keeps its inline error and Retry on screen until the listener dismisses it.
 */
export function isOnDisk(model, op) {
  return ON_DISK.has(model.state) || IN_FLIGHT.has(model.state) || !!(op && op.kind);
}

/**
 * PURE: the ids a list card shows: its frozen order, filtered, then limited to LIST_LIMIT unless the
 * listener asked for all (a filter or "Downloaded only" shows every match).
 * @returns {{ids: string[], total: number, matched: number, limited: boolean}}
 */
export function visibleIds(order, byId, { filter = '', onDisk = false, showAll = false, ops = {} } = {}) {
  const ids = order.filter((id) => byId.has(id));
  const matched = ids.filter((id) => {
    const m = byId.get(id);
    return matchesFilter(m, filter) && (!onDisk || isOnDisk(m, ops[id]));
  });
  const narrowing = !!String(filter || '').trim() || onDisk;
  const limited = !narrowing && !showAll && matched.length > LIST_LIMIT;
  return { ids: limited ? matched.slice(0, LIST_LIMIT) : matched, total: ids.length, matched: matched.length, limited };
}

/**
 * PURE: what a list card shows after the listener narrowed it, said once typing pauses (WCAG 4.1.3; the
 * visible empty state alone is silent): "2 voices match “gb”", "1 voice matches “us high”", "No voices match
 * “zzz”"; with the filter emptied, "Showing 5 of 28 voices", or under "Downloaded only" "2 voices on disk".
 * @param {string} provider
 * @param {{ids: string[], matched: number, limited: boolean}} shown visibleIds()
 * @param {{filter?: string, onDisk?: boolean}} [options]
 */
export function filterSummary(provider, shown, { filter = '', onDisk = false } = {}) {
  const q = String(filter || '').trim();
  const n = Number(shown && shown.matched) || 0;
  const what = noun(provider, n);
  if (q) {
    if (!n) return `No ${noun(provider)} match “${q}”`;
    return `${n} ${what} ${n === 1 ? 'matches' : 'match'} “${q}”`;
  }
  if (onDisk) return n ? `${n} ${what} on disk` : `No ${noun(provider)} are downloaded yet`;
  return shown && shown.limited ? `Showing ${shown.ids.length} of ${n} ${what}` : `${n} ${what}`;
}

/** '6 s', '1:05' for an elapsed time in ms. */
export function elapsedText(ms) {
  const s = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
  return s < 60 ? `${s} s` : clock(s);
}

/** The noun for a provider's list: Piper lists voices, the rest list models. */
function noun(provider, count = 2) {
  const word = provider === 'piper' ? 'voice' : 'model';
  return count === 1 ? word : `${word}s`;
}

/* ───────────────────────────── module state ───────────────────────────── */

// The listener's choices survive polling, re-renders, sub switches and hide/show (BUG-10).
let ui = freshUi();
let serial = 0;

function freshUi() {
  return {
    collapsed: new Map(),   // card key -> true while collapsed
    showAll: new Map(),     // card key -> true
    filter: new Map(),      // card key -> text
    onDisk: new Map(),      // card key -> true ("Downloaded only")
    rowErrors: new Map(),   // model id -> {verb, message}: an unload or delete the server refused
  };
}

const nid = (prefix) => `mo-${prefix}${(serial += 1)}`;

/** Test hook. */
export function __resetForTests() {
  ui = freshUi();
  serial = 0;
}

/** The listener's choices (read-only view, for the view module and tests). */
export function uiState() {
  return ui;
}

function setText(el, text) {
  if (el && el.textContent !== text) el.textContent = text;
}

function setAttr(el, name, value) {
  if (!el) return;
  if (value === null || value === undefined) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
  } else if (el.getAttribute(name) !== String(value)) {
    el.setAttribute(name, String(value));
  }
}

function hasFocus(el) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  return !!el && !!active && (el === active || (typeof el.contains === 'function' && el.contains(active)));
}

function pillMarkup(className = '') {
  return html`<span class="${`pill pill--idle ${className}`.trim()}"><span class="pill__dot" aria-hidden="true"></span><span class="pill__label"></span> <span class="pill__elapsed" aria-hidden="true"></span></span>`;
}

function paintPill(pill, { tone, label, elapsedFrom = 0, extraClass = '' }) {
  if (!pill) return;
  const cls = `pill pill--${tone}${extraClass ? ` ${extraClass}` : ''}`;
  if (pill.className !== cls) pill.className = cls;
  setText(pill.querySelector('.pill__label'), label);
  setAttr(pill, 'data-from', elapsedFrom ? String(elapsedFrom) : null);
  const elapsed = pill.querySelector('.pill__elapsed');
  setText(elapsed, elapsedFrom ? `· ${elapsedText(Date.now() - elapsedFrom)}` : '');
}

/* ───────────────────────────── the renderer ───────────────────────────── */

/**
 * Render and patch the cards of the visible sub into `listEl`.
 * @param {HTMLElement} listEl #mo-list
 * @param {{menusHost: HTMLElement}} options the overflow menus live outside the cards (a card is a size
 *   container, which must not become the menus' containing block)
 * @returns {{update: (view: {sub: string, list: object[], ops: object, defaults: object}, options?: {resort?: boolean}) => void,
 *   tick: (now?: number) => boolean, busy: () => boolean, row: (id: string) => object|null, card: (key: string) => object|null,
 *   isShown: (id: string) => boolean, firstMenu: () => object|null, destroy: () => void}}
 */
export function createCards(listEl, { menusHost } = {}) {
  const cards = new Map();   // key -> card record
  let order = [];            // card keys, in the order shown
  let sub = '';
  let view = { sub: 'tts', list: [], ops: {}, defaults: { tts: '', stt: '' } };
  const offs = [];
  const timers = new Map();  // copy button -> timeout

  const listen = (el, type, fn) => {
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  // The result of a filter, "Downloaded only" or "Clear filter" is said once the listener pauses, never on a
  // poll or an operation's re-render. A card that left the screen (sub switch, hidden view) says nothing.
  const announceList = debounce((key) => {
    const rec = cards.get(key);
    if (!rec || rec.type !== 'models' || !rec.el.isConnected || rec.el.closest('[hidden]') || ui.collapsed.get(key)) return;
    const options = { filter: ui.filter.get(key) || '', onDisk: !!ui.onDisk.get(key) };
    const shown = visibleIds(rec.order, rec.byId, { ...options, showAll: !!ui.showAll.get(key), ops: view.ops });
    announce(filterSummary(rec.provider, shown, options));
  }, FILTER_ANNOUNCE_MS);
  offs.push(() => announceList.cancel());

  /* ── row records ── */

  function rowContext(row) {
    const op = view.ops[row.id] || null;
    const model = row.model;
    const state = models.uiState(model, op, { list: view.list });
    const err = ui.rowErrors.get(row.id) || null;
    const errorText = state.error ? state.error : (err ? `Couldn't ${err.verb}: ${err.message}` : '');
    const p = models.classifyKind(model) === 'tts' ? perf.forModel(model.id) : null;
    const perfLabel = perf.label(p);
    const isDefault = !!model.id && model.id === (view.defaults[models.classifyKind(model)] || '');
    const signature = [state.signature, state.elapsedFrom, errorText, perfLabel, perf.isSlow(p), isDefault, rowTitle(model), rowMeta(model)].join('|');
    return { state, errorText, perfLabel, slow: perf.isSlow(p), isDefault, signature };
  }

  function createRow(model, card) {
    const id = model.id;
    const rid = nid('r');
    const title = rowTitle(model);
    const solo = card.solo;
    const el = document.createElement('li');
    el.className = solo ? 'mo-row mo-row--solo' : 'mo-row';
    el.setAttribute('data-model-id', id);
    el.id = rid;
    render(el, html`
      <div class="mo-row__text">
        <p class="mo-row__title"><span class="${solo ? 'mo-row__name visually-hidden' : 'mo-row__name'}" id="${`${rid}-name`}">${title}</span>${solo ? '' : html` <span class="badge badge--accent mo-row__default" hidden>Default</span>`}</p>
        <p class="mo-row__meta">${rowMeta(model)}</p>
        <p class="mo-row__perf" hidden><span class="mo-row__perf-text"></span> <span class="badge badge--warn mo-row__slow" hidden>Slower than real time</span></p>
        <p class="mo-row__note" hidden></p>
        <progress class="progress progress--thin mo-row__progress" max="100" value="0" aria-labelledby="${`${rid}-name`}" hidden></progress>
      </div>
      <div class="mo-row__side">
        ${solo ? '' : pillMarkup('mo-row__pill')}
        <span class="mo-row__actions">
          <button type="button" class="btn btn--secondary mo-row__primary" data-action="primary"><span class="btn__label">Load</span></button>
          <button type="button" class="icon-btn mo-row__more" id="${`${rid}-more`}" aria-label="${`More actions for ${title}`}" data-tooltip="More actions" hidden>${icon('dots')}</button>
        </span>
      </div>
      <div class="mo-row__error" hidden><p class="mo-row__error-text" role="alert"></p><button type="button" class="icon-btn mo-row__dismiss" data-action="dismiss" aria-label="${`Dismiss the error for ${title}`}" data-tooltip="Dismiss">${icon('x')}</button></div>`);
    const menu = document.createElement('div');
    menu.className = 'menu mo-menu';
    menu.id = `${rid}-menu`;
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-labelledby', `${rid}-more`);
    menu.hidden = true;
    render(menu, html`<button type="button" class="menu__item menu__item--danger" role="menuitem" data-menu="delete">${icon('trash')}Delete files…</button>`);
    if (menusHost) menusHost.appendChild(menu);
    const row = {
      id, el, menu, model, card, title, sig: '',
      parts: {
        pill: solo ? card.headPill : el.querySelector('.mo-row__pill'),
        name: el.querySelector('.mo-row__name'),
        badge: solo ? card.headBadge : el.querySelector('.mo-row__default'),
        meta: el.querySelector('.mo-row__meta'),
        perf: el.querySelector('.mo-row__perf'),
        perfText: el.querySelector('.mo-row__perf-text'),
        slow: el.querySelector('.mo-row__slow'),
        note: el.querySelector('.mo-row__note'),
        progress: el.querySelector('.mo-row__progress'),
        primary: el.querySelector('.mo-row__primary'),
        primaryLabel: el.querySelector('.mo-row__primary .btn__label'),
        more: el.querySelector('.mo-row__more'),
        error: el.querySelector('.mo-row__error'),
        errorText: el.querySelector('.mo-row__error-text'),
      },
      menuCtl: null,
      action: null,
    };
    row.menuCtl = bindMenuButton(row.parts.more, menu, {
      // The overflow button sits at the row's end (a solo card's too, once the card is wider than 34rem), so
      // the menu opens towards the card; on a phone, where a stacked row puts it near the start, placeMenu
      // keeps the menu on screen.
      placement: 'bottom-end',
      onSelect: (item) => {
        if (item.getAttribute('data-menu') === 'delete') deleteFiles(row).catch(() => {});
      },
    });
    return row;
  }

  function destroyRow(row) {
    if (row.menuCtl) row.menuCtl.destroy();
    row.menu.remove();
    row.el.remove();
  }

  /** Patch one row from the store. Never replaces an element; a focused control keeps focus. */
  function patchRow(row, { force = false } = {}) {
    const ctx = rowContext(row);
    if (!force && ctx.signature === row.sig) return false;
    row.sig = ctx.signature;
    const { state } = ctx;
    const p = row.parts;
    row.el.setAttribute('data-state', state.key);
    paintPill(p.pill, { tone: state.tone, label: state.label, elapsedFrom: state.busy ? state.elapsedFrom : 0, extraClass: row.card.solo ? 'mo-card__pill' : 'mo-row__pill' });
    setText(p.name, rowTitle(row.model));
    setHidden(p.badge, !ctx.isDefault);
    setText(p.meta, rowMeta(row.model));
    setHidden(p.meta, !p.meta.textContent);
    setText(p.perfText, ctx.perfLabel ? `${ctx.perfLabel} (this machine)` : '');
    setHidden(p.slow, !ctx.slow);
    setHidden(p.perf, !ctx.perfLabel);
    setText(p.note, state.note);
    setHidden(p.note, !state.note);
    if (state.progress !== null && state.busy) {
      const pct = Math.round(state.progress * 100);
      if (Number(p.progress.value) !== pct) p.progress.value = pct;
      setAttr(p.progress, 'aria-valuetext', `${pct} percent`);
      setHidden(p.progress, false);
    } else {
      setHidden(p.progress, true);
    }

    // The primary action. Busy: the button stays (as a busy, aria-disabled button) only while it, the
    // overflow button or its menu holds focus, so keyboard focus is never dropped; else it hides.
    const keepBusy = state.busy && (hasFocus(p.primary) || hasFocus(p.more) || row.menuCtl.isOpen());
    if (state.busy && row.menuCtl.isOpen()) row.menuCtl.close({ restoreFocus: true });
    row.action = state.busy ? null : state.action;
    if (state.action && !state.busy) {
      setText(p.primaryLabel, state.action.label);
      setAttr(p.primary, 'aria-label', `${state.action.label} ${row.title}`);
      setAttr(p.primary, 'aria-busy', null);
      setAttr(p.primary, 'aria-disabled', null);
      setAttr(p.primary, 'data-op', state.action.kind);
      setHidden(p.primary, false);
    } else if (keepBusy) {
      const text = BUSY_TEXT[state.key] || 'Working…';
      setText(p.primaryLabel, text);
      setAttr(p.primary, 'aria-label', `${text.replace(/…$/, '')} ${row.title}`);
      setAttr(p.primary, 'aria-busy', 'true');
      setAttr(p.primary, 'aria-disabled', 'true');
      setAttr(p.primary, 'data-op', null);
      setHidden(p.primary, false);
      if (hasFocus(p.more)) p.primary.focus({ preventScroll: true });
    } else {
      if (hasFocus(p.primary)) row.card.toggle.focus({ preventScroll: true });
      setHidden(p.primary, true);
      setAttr(p.primary, 'aria-busy', null);
      setAttr(p.primary, 'aria-disabled', null);
    }
    const overflow = !state.busy && state.overflow.includes('delete');
    if (!overflow && hasFocus(p.more)) (p.primary.hidden ? row.card.toggle : p.primary).focus({ preventScroll: true });
    setHidden(p.more, !overflow);

    setText(p.errorText, ctx.errorText);
    if (!ctx.errorText && hasFocus(p.error)) (p.primary.hidden ? row.card.toggle : p.primary).focus({ preventScroll: true });
    setHidden(p.error, !ctx.errorText);
    return true;
  }

  /* ── card records ── */

  function cardMarkup(card, ids) {
    const name = providerName(card.provider);
    const desc = models.PROVIDER_DESCRIPTIONS[card.provider] || '';
    const head = html`<h2 class="mo-card__head"><button type="button" class="mo-card__toggle" id="${ids.toggle}" aria-expanded="true" aria-controls="${ids.body}">${icon('chevron-down', { className: 'ico mo-card__chev' })}<span class="mo-card__title"><span class="mo-card__name">${name}</span>${card.solo ? html` <span class="badge badge--accent mo-card__default" hidden>Default</span>` : ''}</span> <span class="mo-card__status"><span class="mo-card__count"></span> ${pillMarkup('mo-card__pill')}</span></button></h2>`;
    if (card.type === 'install' || card.type === 'offline') {
      const hint = models.installHint(card.provider);
      const offline = card.type === 'offline';
      return html`${head}<div class="mo-card__body" id="${ids.body}">
        ${desc ? html`<p class="mo-card__desc">${desc}</p>` : ''}
        ${offline ? html`<p class="mo-card__hint mo-card__reason"></p>
          <p class="mo-card__ids">${card.models.length === 1 ? 'Model' : 'Models'}: ${card.models.map((m, i) => html`${i ? ', ' : ''}<span class="mono">${m.id}</span>`)}</p>` : ''}
        <p class="mo-card__hint">${offline && hint.command ? 'To turn the worker on, name its Compose profile:' : hint.text}</p>
        ${hint.command ? html`<div class="mo-cmd"><code class="mo-cmd__text" id="${ids.cmd}">${hint.command}</code><button type="button" class="btn btn--secondary mo-cmd__copy" data-action="copy" aria-describedby="${ids.cmd}">${icon('copy')}<span class="btn__label">Copy command</span></button></div>` : ''}
      </div>`;
    }
    const tools = card.models.length > LIST_LIMIT;
    const what = noun(card.provider);
    return html`${head}<div class="mo-card__body" id="${ids.body}">
      ${desc ? html`<p class="mo-card__desc">${desc}</p>` : ''}
      ${tools ? html`<div class="mo-tools">
        <div class="mo-filter">${icon('search', { className: 'ico mo-filter__icon' })}<label class="visually-hidden" for="${ids.filter}">${`Filter ${name} ${what}`}</label><input class="input mo-filter__input" type="search" id="${ids.filter}" placeholder="${`Filter ${what}`}" autocomplete="off" spellcheck="false"></div>
        <button type="button" class="chip mo-ondisk" data-action="on-disk" aria-pressed="false">${icon('check', { className: 'ico chip__check' })}Downloaded only</button>
      </div>` : ''}
      <ul class="${card.solo ? 'mo-rows mo-rows--solo' : 'mo-rows'}" id="${ids.rows}" aria-label="${`${name} ${what}`}"></ul>
      ${tools ? html`<div class="mo-card__empty" hidden><p class="mo-card__empty-text"></p><button type="button" class="btn btn--ghost" data-action="clear-filter"><span class="btn__label">Clear filter</span></button></div>
      <div class="mo-card__foot"><button type="button" class="btn btn--ghost mo-card__more" data-action="show-all" aria-expanded="false" aria-controls="${ids.rows}"><span class="btn__label"></span></button></div>` : ''}
    </div>`;
  }

  function createCard(card) {
    const solo = isSolo(card);
    const base = nid('c');
    const ids = { toggle: `${base}-toggle`, body: `${base}-body`, rows: `${base}-rows`, filter: `${base}-filter`, cmd: `${base}-cmd` };
    const el = document.createElement('article');
    el.className = `card mo-card mo-card--${solo ? 'solo' : card.type}`;
    el.id = base;
    el.setAttribute('data-provider', card.provider);
    el.setAttribute('data-type', card.type);
    render(el, cardMarkup({ ...card, solo }, ids));
    const rec = {
      key: card.key, provider: card.provider, type: card.type, solo, el, ids,
      toggle: el.querySelector('.mo-card__toggle'),
      body: el.querySelector('.mo-card__body'),
      count: el.querySelector('.mo-card__count'),
      headPill: el.querySelector('.mo-card__pill'),
      headBadge: el.querySelector('.mo-card__default'),
      reason: el.querySelector('.mo-card__reason'),
      rowsEl: el.querySelector('.mo-rows'),
      filter: el.querySelector('.mo-filter__input'),
      onDisk: el.querySelector('.mo-ondisk'),
      empty: el.querySelector('.mo-card__empty'),
      emptyText: el.querySelector('.mo-card__empty-text'),
      more: el.querySelector('.mo-card__more'),
      models: card.models,
      byId: new Map(card.models.map((m) => [m.id, m])),
      order: sortModels(card.provider, card.models).map((m) => m.id),
      rows: new Map(),
    };
    if (rec.filter) rec.filter.value = ui.filter.get(card.key) || '';
    applyCollapse(rec);
    return rec;
  }

  function destroyCard(rec) {
    for (const row of rec.rows.values()) destroyRow(row);
    rec.rows.clear();
    rec.el.remove();
  }

  function applyCollapse(rec) {
    const collapsed = !!ui.collapsed.get(rec.key);
    setAttr(rec.toggle, 'aria-expanded', collapsed ? 'false' : 'true');
    rec.el.classList.toggle('is-collapsed', collapsed);
    setHidden(rec.body, collapsed);
  }

  /** Keep the first-shown order; models that appear later slot in at their sorted place. */
  function mergeOrder(rec, rows, { resort = false } = {}) {
    rec.models = rows;
    rec.byId = new Map(rows.map((m) => [m.id, m]));
    const sorted = sortModels(rec.provider, rows).map((m) => m.id);
    if (resort) {
      rec.order = sorted;
      return;
    }
    const kept = rec.order.filter((id) => rec.byId.has(id));
    for (const id of sorted) {
      if (kept.includes(id)) continue;
      const after = sorted.slice(sorted.indexOf(id) + 1).find((next) => kept.includes(next));
      if (after) kept.splice(kept.indexOf(after), 0, id);
      else kept.push(id);
    }
    rec.order = kept;
  }

  function renderRows(rec) {
    if (rec.type !== 'models') return;
    const shown = visibleIds(rec.order, rec.byId, {
      filter: ui.filter.get(rec.key) || '', onDisk: !!ui.onDisk.get(rec.key), showAll: !!ui.showAll.get(rec.key), ops: view.ops,
    });
    const active = document.activeElement;
    const focused = active && rec.rowsEl.contains(active) ? active : null;
    // The rows on screen before this render, in order: where a row that leaves while it holds focus was.
    const before = Array.from(rec.rowsEl.children, (li) => li.getAttribute('data-model-id'));
    let lostAt = -1;
    for (const [id, row] of rec.rows) {
      if (!shown.ids.includes(id)) {
        if (hasFocus(row.el) || hasFocus(row.menu)) lostAt = before.indexOf(id);
        destroyRow(row);
        rec.rows.delete(id);
      }
    }
    let prev = null;
    for (const id of shown.ids) {
      let row = rec.rows.get(id);
      if (!row) {
        row = createRow(rec.byId.get(id), rec);
        rec.rows.set(id, row);
      }
      row.model = rec.byId.get(id);
      const want = prev ? prev.el.nextSibling : rec.rowsEl.firstChild;
      if (row.el !== want) rec.rowsEl.insertBefore(row.el, want);
      prev = row;
    }
    if (focused && focused.isConnected && document.activeElement !== focused) focused.focus({ preventScroll: true });
    for (const row of rec.rows.values()) patchRow(row);
    if (lostAt >= 0) handOff(rec, before, lostAt);
    if (rec.empty) {
      const filterText = String(ui.filter.get(rec.key) || '').trim();
      const what = noun(rec.provider);
      const nothing = shown.matched === 0;
      setText(rec.emptyText, filterText ? `No ${what} match “${filterText}”.` : `No ${what} are downloaded yet.`);
      const clear = rec.empty.querySelector('[data-action="clear-filter"]');
      setText(clear && clear.querySelector('.btn__label'), filterText ? 'Clear filter' : 'Show all');
      setHidden(rec.empty, !nothing);
    }
    if (rec.more) {
      const narrowing = !!String(ui.filter.get(rec.key) || '').trim() || !!ui.onDisk.get(rec.key);
      const all = !!ui.showAll.get(rec.key);
      setText(rec.more.querySelector('.btn__label'), all ? 'Show fewer' : `Show all ${shown.total} ${noun(rec.provider, shown.total)}`);
      setAttr(rec.more, 'aria-expanded', all ? 'true' : 'false');
      if (narrowing && hasFocus(rec.more)) (rec.filter || rec.toggle).focus({ preventScroll: true });
      setHidden(rec.more.parentElement, narrowing || shown.total <= LIST_LIMIT);
    }
    if (rec.onDisk) setAttr(rec.onDisk, 'aria-pressed', ui.onDisk.get(rec.key) ? 'true' : 'false');
  }

  /**
   * A row that held focus left the list ("Delete files…" or a dismissed error under "Downloaded only", a
   * filter): focus goes to the next row's action, else the previous row's, else the list's own controls
   * ("Downloaded only" while it narrows the list, the filter, the card header), never to <body> (BUG-10).
   */
  function handOff(rec, before, lostAt) {
    const active = document.activeElement;
    if (active && active !== document.body && active.isConnected) return;   // focus already went somewhere
    const actionOf = (row) => {
      if (!row) return null;
      if (!row.parts.primary.hidden) return row.parts.primary;
      return row.parts.more.hidden ? null : row.parts.more;
    };
    const after = before.slice(lostAt + 1).map((id) => rec.rows.get(id));
    const earlier = before.slice(0, lostAt).reverse().map((id) => rec.rows.get(id));
    for (const row of [...after, ...earlier, ...rec.rows.values()]) {
      const target = actionOf(row);
      if (target) {
        target.focus({ preventScroll: true });
        return;
      }
    }
    const chip = rec.onDisk && ui.onDisk.get(rec.key) ? rec.onDisk : null;
    (chip || rec.filter || rec.toggle).focus({ preventScroll: true });
  }

  function patchHead(rec) {
    if (rec.solo) {
      const row = rec.rows.values().next().value;
      if (row) {
        setHidden(rec.headPill, false);
        setText(rec.count, '');
        setHidden(rec.count, true);
      }
      return;
    }
    if (rec.type === 'install') {
      const hint = models.installHint(rec.provider);
      paintPill(rec.headPill, { tone: 'warn', label: hint.kind === 'rebuild' ? 'Not installed' : 'Worker not configured', extraClass: 'mo-card__pill' });
      setHidden(rec.count, true);
      return;
    }
    if (rec.type === 'offline') {
      // A worker that answers but can't run says so and why (its reason, set as text); one that doesn't answer
      // is offline. Painted on every update: a poll can bring or end a failure without changing the card's type.
      const states = rec.models.map((m) => models.uiState(m, view.ops[m.id], { list: view.list }));
      const state = states.find((s) => s.key === 'worker-failed') || states.find((s) => s.key === 'worker-offline');
      paintPill(rec.headPill, { tone: 'error', label: state ? state.label : 'Worker offline', extraClass: 'mo-card__pill' });
      setText(rec.reason, state && state.hint ? state.hint.text : "The voice worker isn't responding. Check the worker service.");
      setHidden(rec.count, true);
      return;
    }
    const states = rec.models.map((m) => models.uiState(m, view.ops[m.id], { list: view.list }));
    const busy = states.filter((s) => s.busy);
    const loaded = rec.models.filter((m) => m.state === 'loaded');
    const onDisk = rec.models.filter((m) => ON_DISK.has(m.state)).length;
    setText(rec.count, `${onDisk} of ${plural(rec.models.length, noun(rec.provider, 1))} on disk`);
    setHidden(rec.count, false);
    if (busy.length) {
      paintPill(rec.headPill, { tone: 'busy', label: busy.length === 1 ? busy[0].label : `${busy.length} in progress`, extraClass: 'mo-card__pill' });
      setHidden(rec.headPill, false);
    } else if (loaded.length) {
      const device = loaded.length === 1 ? models.deviceLabel(loaded[0].device) : '';
      paintPill(rec.headPill, { tone: 'ok', label: device ? `Loaded · ${device}` : `${loaded.length} loaded`, extraClass: 'mo-card__pill' });
      setHidden(rec.headPill, false);
    } else {
      setHidden(rec.headPill, true);
    }
  }

  /**
   * Render the cards for `next` (app.models + app.ops for one sub). Cards and rows that already exist are
   * patched; only a row whose signature changed is touched. `resort` re-sorts cards and rows (a new sub,
   * an explicit Refresh); otherwise the order shown first is kept.
   */
  function update(next, { resort = false } = {}) {
    view = { sub: next.sub, list: next.list || [], ops: next.ops || {}, defaults: next.defaults || { tts: '', stt: '' } };
    const newSub = view.sub !== sub;
    sub = view.sub;
    const groups = groupCards(view.list, sub, { ops: view.ops });
    const wanted = new Map(groups.map((g) => [g.key, g]));
    const active = document.activeElement;
    const focusedCard = active && listEl.contains(active) ? [...cards.values()].find((c) => c.el.contains(active)) : null;
    // Remove cards that went away or changed type (a worker came online: offline -> models).
    for (const [key, rec] of cards) {
      const g = wanted.get(key);
      if (!g || g.type !== rec.type || isSolo(g) !== rec.solo) {
        destroyCard(rec);
        cards.delete(key);
      }
    }
    let keys;
    if (resort || newSub || !order.length) {
      keys = groups.map((g) => g.key);
    } else {
      keys = order.filter((k) => wanted.has(k));
      for (const g of groups) {
        if (keys.includes(g.key)) continue;
        const sorted = groups.map((x) => x.key);
        const after = sorted.slice(sorted.indexOf(g.key) + 1).find((k) => keys.includes(k));
        if (after) keys.splice(keys.indexOf(after), 0, g.key);
        else keys.push(g.key);
      }
    }
    order = keys;
    let prev = null;
    for (const key of keys) {
      const g = wanted.get(key);
      let rec = cards.get(key);
      if (!rec) {
        rec = createCard(g);
        cards.set(key, rec);
      }
      mergeOrder(rec, g.models, { resort: resort || newSub });
      const want = prev ? prev.el.nextSibling : listEl.firstChild;
      if (rec.el !== want) listEl.insertBefore(rec.el, want);
      prev = rec;
      renderRows(rec);
      patchHead(rec);
    }
    // A card that was replaced while it held focus hands focus to its successor's header.
    if (focusedCard && !focusedCard.el.isConnected) {
      const successor = cards.get(focusedCard.key);
      if (successor) successor.toggle.focus({ preventScroll: true });
    } else if (active && active.isConnected && document.activeElement !== active && listEl.contains(active)) {
      active.focus({ preventScroll: true });
    }
  }

  /** Update the aria-hidden elapsed counters ("· 6 s"). Returns whether any is running. */
  function tick(now = Date.now()) {
    let running = false;
    for (const pill of listEl.querySelectorAll('.pill[data-from]')) {
      const from = Number(pill.getAttribute('data-from'));
      if (!from) continue;
      running = true;
      setText(pill.querySelector('.pill__elapsed'), `· ${elapsedText(now - from)}`);
    }
    return running;
  }

  function busy() {
    return !!listEl.querySelector('.pill[data-from]');
  }

  /* ── actions ── */

  function rowOf(el) {
    const host = el.closest('[data-model-id]');
    if (!host) return null;
    for (const rec of cards.values()) {
      const row = rec.rows.get(host.getAttribute('data-model-id'));
      if (row && row.el === host) return row;
    }
    return null;
  }

  function cardOf(el) {
    const host = el.closest('.mo-card');
    return host ? [...cards.values()].find((c) => c.el === host) || null : null;
  }

  function refreshRow(row) {
    row.sig = '';
    patchRow(row);
    patchHead(row.card);
  }

  async function primary(row) {
    const action = row.action;
    if (!action) return;
    ui.rowErrors.delete(row.id);
    if (action.kind === 'unload') {
      try {
        await models.unload(row.id);
      } catch (error) {
        if (error && error.kind === 'aborted') return;
        // The row may be stale (the idle TTL unloads models too): core refreshed it; say why inline.
        ui.rowErrors.set(row.id, { verb: 'unload', message: error?.message || String(error) });
        refreshRow(row);
      }
      return;
    }
    if (action.kind === 'copy') {
      await copyCommand(row.parts.primary, action.command, action.label);
      return;
    }
    if (action.kind === 'retry') {
      models.clearOp(row.id);
      await models.runOp(row.id, action.op === 'download' || action.op === 'prefetch' ? action.op : 'load');
      return;
    }
    if (action.kind === 'load') {
      await models.runOp(row.id, 'load');
      return;
    }
    if (action.kind === 'download') await models.runOp(row.id, action.op || 'prefetch');
  }

  async function deleteFiles(row) {
    ui.rowErrors.delete(row.id);
    refreshRow(row);
    let deleted = false;
    try {
      deleted = await models.deleteArtifacts(row.id, { confirm: true });
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      ui.rowErrors.set(row.id, { verb: 'delete the files', message: error?.message || String(error) });
      if (row.el.isConnected) refreshRow(row);
      // Show what the server has now (a failure refreshes the row and says why inline).
      await models.refresh({ silent: true });
      return;
    }
    if (deleted) toast(`Deleted the downloaded files for ${row.title}.`, { kind: 'success' });
  }

  async function copyCommand(button, command, idleLabel = 'Copy command') {
    const label = button.querySelector('.btn__label');
    const ok = await copyText(command);
    if (!ok) {
      toast("Couldn't copy here. Select the command and copy it instead.", { kind: 'error' });
      return;
    }
    setText(label, 'Copied');
    announce('Command copied');
    clearTimeout(timers.get(button));
    timers.set(button, setTimeout(() => {
      timers.delete(button);
      setText(label, idleLabel);
    }, COPIED_MS));
  }

  listen(listEl, 'click', (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    if (!start || typeof start.closest !== 'function') return;
    const toggle = start.closest('.mo-card__toggle');
    if (toggle && listEl.contains(toggle)) {
      const rec = cardOf(toggle);
      if (!rec) return;
      ui.collapsed.set(rec.key, !ui.collapsed.get(rec.key));
      applyCollapse(rec);
      return;
    }
    const el = start.closest('[data-action]');
    if (!el || !listEl.contains(el) || el.disabled || el.getAttribute('aria-disabled') === 'true') return;
    const action = el.getAttribute('data-action');
    const rec = cardOf(el);
    const fail = (error) => { if (error && error.kind !== 'aborted') toast(error.message || String(error), { kind: 'error' }); };
    if (action === 'primary') {
      const row = rowOf(el);
      if (row) primary(row).catch(fail);
    } else if (action === 'dismiss') {
      const row = rowOf(el);
      if (!row) return;
      ui.rowErrors.delete(row.id);
      // Clearing a failed download can take the row out of "Downloaded only": renderRows() then hands focus on.
      models.clearOp(row.id);
      if (!row.el.isConnected) return;
      refreshRow(row);
      (row.parts.primary.hidden ? row.card.toggle : row.parts.primary).focus({ preventScroll: true });
    } else if (action === 'copy' && rec) {
      const code = rec.el.querySelector('.mo-cmd__text');
      copyCommand(el, code ? code.textContent : '').catch(fail);
    } else if (action === 'on-disk' && rec) {
      ui.onDisk.set(rec.key, !ui.onDisk.get(rec.key));
      mergeOrder(rec, rec.models, { resort: true });
      renderRows(rec);
      announceList(rec.key);
    } else if (action === 'show-all' && rec) {
      ui.showAll.set(rec.key, !ui.showAll.get(rec.key));
      mergeOrder(rec, rec.models, { resort: true });
      renderRows(rec);
    } else if (action === 'clear-filter' && rec) {
      ui.filter.set(rec.key, '');
      ui.onDisk.set(rec.key, false);
      if (rec.filter) rec.filter.value = '';
      renderRows(rec);
      (rec.filter || rec.toggle).focus({ preventScroll: true });
      announceList(rec.key);
    }
  });

  // A busy action button stays only while it holds focus (patchRow); once focus moves on, the row shows only
  // its pill ("Loading · 6 s") until the operation ends.
  listen(listEl, 'focusout', (event) => {
    const button = event.target;
    if (!button || !button.classList || !button.classList.contains('mo-row__primary') || button.getAttribute('aria-busy') !== 'true') return;
    const row = rowOf(button);
    const next = event.relatedTarget;
    if (!row || (next && (row.el.contains(next) || row.menu.contains(next)))) return;
    patchRow(row, { force: true });
  });

  listen(listEl, 'input', (event) => {
    const input = event.target;
    if (!input || !input.classList || !input.classList.contains('mo-filter__input')) return;
    const rec = cardOf(input);
    if (!rec) return;
    ui.filter.set(rec.key, input.value);
    mergeOrder(rec, rec.models, { resort: true });
    renderRows(rec);
    announceList(rec.key);
  });

  return {
    update,
    tick,
    busy,
    row(id) {
      for (const rec of cards.values()) if (rec.rows.has(id)) return rec.rows.get(id);
      return null;
    },
    card(key) {
      return cards.get(key) || null;
    },
    /** Whether a model's row is rendered (in the sub last rendered, not filtered out) with its card expanded. */
    isShown(id) {
      for (const rec of cards.values()) {
        const row = rec.rows.get(id);
        if (row) return row.el.isConnected && !ui.collapsed.get(rec.key);
      }
      return false;
    },
    /** The first row whose overflow menu is available (the screenshot hook opens it). */
    firstMenu() {
      for (const key of order) {
        const rec = cards.get(key);
        if (!rec || ui.collapsed.get(key)) continue;
        for (const row of rec.rows.values()) if (!row.parts.more.hidden) return row;
      }
      return null;
    },
    destroy() {
      for (const off of offs.splice(0)) off();
      for (const timer of timers.values()) clearTimeout(timer);
      timers.clear();
      for (const rec of cards.values()) destroyCard(rec);
      cards.clear();
      order = [];
    },
  };
}
