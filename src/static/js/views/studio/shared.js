/**
 * Shared pieces of the Studio view: the list store (request tokens, staleness), loading and error states,
 * inline banners, item menus, focus kept across a re-render, the inline transport of a clip on its own route,
 * and the plain-language copy for a failed request. No module-level DOM access. History imports the banner,
 * menu and focus helpers from here too.
 * @module views/studio/shared
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { explain } from '../../core/api.js';
import * as audio from '../../core/audio.js';
import { clock, spokenDuration } from '../../core/format.js';
import { bindMenuButton } from '../../components/menu.js';
import { bindRange } from '../../components/range.js';

/** A list is reloaded on show when it is older than this (GS-12). */
export const STALE_MS = 30000;

const enc = encodeURIComponent;

/** A safe id fragment for any key (uuids keep their shape). */
export function idPart(key) {
  return String(key ?? '').replace(/[^A-Za-z0-9_-]/g, '_') || 'x';
}

/** The path segment for an id: encodeURIComponent. */
export function seg(id) {
  return enc(String(id ?? ''));
}

/** Whether an error is the API key's (the shell's connection banner carries the details). */
export function isAuthError(error) {
  return !!error && (error.kind === 'auth' || error.kind === 'locked' || error.status === 401);
}

/**
 * The copy for a failed request: a banner title, the plain message (the server's own words when it says why),
 * Details (api.explain()'s detailText: the server's message and its code, never the JSON envelope; none when
 * they would only repeat the message), whether Retry makes sense, and whether it is the API key's.
 * @param {*} error
 * @param {string} title e.g. "Couldn't render"
 * @param {{notFound?: string}} [options]
 */
export function errorCopy(error, title, { notFound = '' } = {}) {
  const e = explain(error, notFound ? { title, notFound } : title);
  const auth = isAuthError(error);
  const message = auth
    ? 'Enter the server’s API key in Settings › Connection, then try again.'
    : (e.message || String(error?.message || '') || 'Something went wrong.');
  return {
    title: auth ? 'API key needed' : (e.title || title),
    message,
    detail: auth ? '' : e.detailText,
    retryable: !auth && !!e.retryable,
    auth,
  };
}

/**
 * The buttons of a failed action's banner: [Open Settings] for an API-key refusal, else `retry` when the
 * failure is worth retrying.
 * @param {{auth: boolean, retryable: boolean}} copy errorCopy()
 * @param {{cmd: string, label: string}|null} [retry]
 */
export function bannerActions(copy, retry = null) {
  if (copy && copy.auth) return [{ cmd: 'open-settings', label: 'Open Settings' }];
  return retry && copy && copy.retryable ? [retry] : [];
}

/* ───────────────────────────── banners ───────────────────────────── */

/**
 * An inline banner: an icon, a title, the message, a Details disclosure (filled with textContent),
 * optional action buttons and a Dismiss button. Each action button is data-action="<cmd>", run by the view's
 * action map like any other button. The banner is .banner--stacked: its actions sit under the message, aligned
 * with its text, at every width (Studio's and History's banners sit in cards and narrow columns).
 * @param {HTMLElement} host replaced content; shown
 * @param {{tone?: 'error'|'warning'|'info', role?: 'alert'|'status', title: string, message?: string,
 *   detail?: string, actions?: {cmd: string, label: string, primary?: boolean}[], dismiss?: boolean,
 *   prefix?: string}} options prefix: the class prefix of the view ('st' or 'hi')
 */
export function showBanner(host, { tone = 'error', role = 'alert', title, message = '', detail = '', actions = [], dismiss = true, prefix = 'st' }) {
  if (!host) return null;
  const glyph = tone === 'info' ? 'info' : 'alert';
  render(host, html`<div class="banner banner--${tone} banner--stacked ${prefix}-banner" role="${role}">${icon(glyph, { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${title}</p>${message ? html`<p class="banner__msg">${message}</p>` : ''}${detail ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}</div>${actions.length ? html`<div class="banner__actions">${actions.map((a) => html`<button type="button" class="btn ${a.primary ? 'btn--primary' : 'btn--secondary'} btn--sm" data-action="${a.cmd}"><span class="btn__label">${a.label}</span></button>`)}</div>` : ''}${dismiss ? html`<button type="button" class="icon-btn icon-btn--sm" data-action="dismiss-banner" aria-label="Dismiss">${icon('x')}</button>` : ''}</div>`);
  if (detail) fillDetails(host, detail);
  host.hidden = false;
  return host.firstElementChild;
}

/** Remove an inline banner. */
export function clearBanner(host) {
  if (!host) return;
  if (host.firstElementChild) render(host, html``);
  host.hidden = true;
}

/** Fill the <pre> of a Details disclosure with text (textContent only). */
export function fillDetails(root, detail) {
  const pre = root ? root.querySelector('details pre') : null;
  if (pre) pre.textContent = String(detail || '');
}

/* ───────────────────────────── list states ───────────────────────────── */

/**
 * Loading placeholder: flat skeleton rows inside an aria-busy container with a visually hidden "Loading…".
 * @param {string} what e.g. 'conversations'
 * @param {{rows?: number, prefix?: string}} [options]
 */
export function loadingMarkup(what, { rows = 3, prefix = 'st' } = {}) {
  return html`<div class="${prefix}-loading" aria-busy="true"><span class="visually-hidden">Loading ${what}…</span>${Array.from({ length: rows }, () => html`<div class="skeleton skeleton--row ${prefix}-skeleton" aria-hidden="true"></div>`)}</div>`;
}

/**
 * The error state of a list that didn't load: the plain cause, Retry and Details; an auth refusal says
 * "API key needed" with [Open Settings]. Retry is data-action="retry-load" data-list="<list>". Call
 * fillDetails() after rendering it.
 * @param {*} error
 * @param {{what: string, list: string, prefix?: string, level?: 2|3}} options
 */
export function loadErrorMarkup(error, { what, list, prefix = 'st', level = 2 }) {
  const copy = errorCopy(error, `Couldn't load ${what}`);
  const title = copy.auth ? 'API key needed' : `Couldn't load ${what}`;
  const heading = level === 3 ? html`<h3 class="empty__title">${title}</h3>` : html`<h2 class="empty__title">${title}</h2>`;
  return html`<div class="empty empty--error ${prefix}-state">${icon('alert', { className: 'ico empty__icon' })}${heading}<p class="empty__text">${copy.message}</p>${copy.auth
    ? html`<button type="button" class="btn btn--secondary" data-action="open-settings"><span class="btn__label">Open Settings</span></button>`
    : html`<button type="button" class="btn btn--secondary" data-action="retry-load" data-list="${list}">${icon('retry')}<span class="btn__label">Retry</span></button>`}${copy.detail ? html`<details class="banner__details ${prefix}-state__details"><summary>Details</summary><pre></pre></details>` : ''}</div>`;
}

/**
 * The lists a view shows. Each load takes a request token, so a late answer never overwrites a newer one;
 * `onChange(name)` runs when a list starts loading, loads or fails.
 * @param {Record<string, () => Promise<object[]>>} loaders name -> async () => items
 * @param {(name: string) => void} [onChange]
 */
export function createLists(loaders, onChange = () => {}) {
  const lists = {};
  for (const name of Object.keys(loaders)) {
    lists[name] = { items: [], status: 'idle', error: null, at: 0, seq: 0 };
  }
  async function load(name) {
    const list = lists[name];
    if (!list) throw new TypeError(`Unknown list: ${name}`);
    const seq = (list.seq += 1);
    list.status = 'loading';
    onChange(name);
    try {
      const items = await loaders[name]();
      if (seq !== list.seq) return false;
      list.items = Array.isArray(items) ? items : [];
      list.status = 'ready';
      list.error = null;
      list.at = Date.now();
      onChange(name);
      return true;
    } catch (error) {
      if (seq !== list.seq) return false;
      list.status = 'error';
      list.error = error;
      onChange(name);
      return false;
    }
  }
  return {
    lists,
    load,
    /** True when the list never loaded or is older than STALE_MS. */
    stale(name, now = Date.now()) {
      const list = lists[name];
      return !list.at || now - list.at > STALE_MS;
    },
  };
}

/* ───────────────────────────── focus ───────────────────────────── */

/** Move focus to an element (a heading gets tabindex="-1"); nothing for a missing one. */
export function focusElement(target, { preventScroll = false } = {}) {
  if (!target || typeof target.focus !== 'function') return false;
  if (/^h[1-6]$/i.test(target.tagName) && !target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
  target.focus({ preventScroll });
  return true;
}

/**
 * Show or hide `node`; when hiding it takes keyboard focus with it, focus goes to `fallback` (an element or a
 * function returning one) instead of dropping to <body>.
 */
export function hideKeepingFocus(node, hidden, fallback = null) {
  if (!node) return;
  const next = !!hidden;
  if (node.hidden === next) return;
  const had = next && typeof document !== 'undefined' && node.contains(document.activeElement);
  node.hidden = next;
  if (had) focusElement(typeof fallback === 'function' ? fallback() : fallback);
}

/** The id of the focused element when it sits inside `el` (or `also`), else ''. */
export function focusedIdIn(el, also = null) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  if (!active || active === el) return '';
  if (el.contains(active) || (also && also.contains(active))) return active.id || '';
  return '';
}

/**
 * Re-render `el` and keep keyboard focus on the same control: the focused element's id is looked up again in
 * the new markup; when it is gone, `fallback()` picks where focus goes. `wasFocused` is an id that had focus
 * before an async action (focus may have dropped to <body> meanwhile).
 */
export function renderKeepingFocus(el, markup, { fallback, also = null, wasFocused = '' } = {}) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  const inside = !!(active && active !== el && (el.contains(active) || (also && also.contains(active))));
  const lost = !active || active === document.body || active === document.documentElement || active.isConnected === false;
  const id = inside ? active.id : (lost ? wasFocused : '');
  render(el, markup);
  if (!inside && !(lost && wasFocused)) return;
  const again = id ? document.getElementById(id) : null;
  const target = again && (el.contains(again) || (also && also.contains(again))) ? again : (typeof fallback === 'function' ? fallback() : null);
  focusElement(target);
}

/* ───────────────────────────── menus ───────────────────────────── */

/**
 * A "More actions" icon button for an item. Its menu (menuMarkup) is written right after it, also inside a
 * .card: components/menu.js moves an open menu out of a containing-block trap and back, and Tab goes on from
 * the button.
 * @param {string} id the button id
 * @param {string} menuId the menu id
 * @param {string} label the item's name, for the accessible name
 * @param {{disabled?: boolean, describedBy?: string, className?: string}} [options]
 */
export function menuButtonMarkup(id, menuId, label, { className = 'st-more' } = {}) {
  return html`<button type="button" class="icon-btn ${className}" id="${id}" data-menu-btn="${menuId}" aria-label="More actions for ${label}" data-tooltip="More actions">${icon('dots')}</button>`;
}

/**
 * A menu's items. Items carry data-menu-cmd (and data-arg); a disabled item stays listed with aria-disabled and
 * its reason as a title.
 * @param {{cmd: string, label: string, icon: string, danger?: boolean, disabled?: string, arg?: string}[]} items
 */
export function menuItemsMarkup(items) {
  return html`${items.map((item) => html`<button type="button" role="menuitem" tabindex="-1" class="menu__item${item.danger ? ' menu__item--danger' : ''}" data-menu-cmd="${item.cmd}" data-arg="${item.arg ?? ''}"${item.disabled ? html` aria-disabled="true" title="${item.disabled}"` : ''}>${icon(item.icon)}<span>${item.label}</span></button>`)}`;
}

/**
 * A menu, written right after its menuButtonMarkup() button.
 * @param {string} menuId
 * @param {string} label
 * @param {{cmd: string, label: string, icon: string, danger?: boolean, disabled?: string, arg?: string}[]} items
 */
export function menuMarkup(menuId, label, items) {
  return html`<div class="menu" id="${menuId}" role="menu" aria-label="${label}" hidden>${menuItemsMarkup(items)}</div>`;
}

/**
 * Bind every `[data-menu-btn]` button in `root` to its menu, written after it in `root`. Call the returned
 * function before `root` re-renders: it closes an open menu, which first goes back to its place.
 * onSelect(cmd, item, button) runs for clicks and keys alike.
 * @returns {() => void} unbind all
 */
export function bindMenus(root, onSelect) {
  const bound = [];
  const menus = new Map([...root.querySelectorAll('.menu[id]')].map((m) => [m.id, m]));
  for (const button of root.querySelectorAll('[data-menu-btn]')) {
    const menu = menus.get(button.getAttribute('data-menu-btn'));
    if (!menu) continue;
    bound.push(bindMenuButton(button, menu, {
      placement: 'bottom-end',
      onSelect: (item) => onSelect(item.getAttribute('data-menu-cmd'), item, button),
    }));
  }
  return () => {
    for (const m of bound.splice(0)) m.destroy();
  };
}

/* ───────────────────────────── play presses ───────────────────────────── */

let pressSeq = 0;

/**
 * A Play press whose clip arrives after an await (with an API key, a render or a mix is downloaded whole before
 * it plays): call it inside the click, right after audio.prime(). The newest press in Studio wins, and a late
 * answer never displaces audio started meanwhile anywhere else (a take, the Reader, Live, a preview): before
 * playing, ask current().
 * @returns {{current: () => boolean, renew: () => object, cancel: () => void}} current(): this press may still
 *   start its clip; renew(): the same press again (a repeat press while its audio loads), now the newest, against
 *   what plays now; cancel(): it never plays (its item was deleted)
 */
export function startPress() {
  let token = 0;
  let before = null;
  const press = {
    renew() {
      token = (pressSeq += 1);
      before = audio.current();
      return press;
    },
    cancel() {
      token = 0;
    },
    current() {
      if (!token || token !== pressSeq) return false;
      const now = audio.current();
      return !now || now === before;
    },
  };
  return press.renew();
}

/** A play/pause icon button's state: the icon and its label follow whether the session plays now. */
export function setPlayButton(button, { playing, label }) {
  if (!button) return;
  const name = playing ? `Pause ${label}` : `Play ${label}`;
  if (button.getAttribute('aria-label') !== name) {
    button.setAttribute('aria-label', name);
    button.setAttribute('data-tooltip', name);
    render(button, icon(playing ? 'pause' : 'play'));
  }
  const flag = playing ? 'true' : 'false';
  if (button.getAttribute('data-playing') !== flag) button.setAttribute('data-playing', flag);
}

/**
 * What a play button shows of app.player: which session, and whether it plays. Selecting this (not the player
 * object, which is replaced 4 times a second while anything plays) runs play-button syncs only when they change.
 * @param {object} state app.get()
 */
export const playerKey = (state) => (state && state.player ? `${state.player.id}|${state.player.playing ? 1 : 0}` : '');

/* ───────────────────────────── inline transport ───────────────────────────── */

/** The keyboard step of a transport's seek range, in seconds. */
export const SEEK_KEY_STEP = 10;

/** The markup of a transport built at runtime (a past mix's row): the seek range and the time. */
export function transportMarkup(id, label) {
  return html`<input type="range" class="seek st-transport__seek" id="${id}" min="0" max="0" step="1" value="0" aria-label="${label}"><span class="tabular st-transport__time" aria-hidden="true"></span>`;
}

/**
 * An inline transport for a clip on its own route, where the mini-player hides: a native seek
 * range (bindRange; aria-valuetext "12 seconds of 1 minute 12 seconds") and the time "0:12 / 1:12" beside it.
 * paint(player) follows app.player; a change seeks the session, when it is still `sessionId()`. With the
 * keyboard on the range it steps SEEK_KEY_STEP seconds and moves only when that step does, so a focused range
 * is not re-announced every second; a pointer drags to the second.
 * @param {HTMLInputElement} input
 * @param {HTMLElement|null} timeEl
 * @param {{sessionId: () => string}} options
 * @returns {{paint: (player: object) => void, destroy: () => void}}
 */
export function bindTransport(input, timeEl, { sessionId }) {
  let duration = 0;
  let position = 0;
  let dragging = false;
  let keyboard = false;
  let pointer = false;
  const range = bindRange(input, {
    format: (value) => clock(value),
    spoken: (value) => `${spokenDuration(value)} of ${spokenDuration(duration)}`,
  });
  const showTime = (at) => {
    const text = `${clock(at)} / ${clock(duration)}`;
    if (timeEl && timeEl.textContent !== text) timeEl.textContent = text;
  };
  function place() {
    const step = String(keyboard ? SEEK_KEY_STEP : 1);
    if (input.getAttribute('step') !== step) input.setAttribute('step', step);
    if (dragging) return;
    const at = Math.max(0, Math.min(duration, position));
    const value = String(keyboard ? Math.floor(at / SEEK_KEY_STEP) * SEEK_KEY_STEP : Math.floor(at));
    if (input.value !== value) input.value = value;
    range.update();
  }
  const listeners = {
    pointerdown: () => { pointer = true; keyboard = false; place(); },
    focus: () => { if (!pointer) { keyboard = true; place(); } },
    blur: () => { pointer = false; keyboard = false; dragging = false; place(); showTime(position); },
    keydown: (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey || keyboard) return;
      keyboard = true;   // before the browser's own arrow step, which reads the step attribute
      place();
    },
    input: () => { dragging = true; showTime(Number(input.value)); },
    change: () => {
      dragging = false;
      const target = Number(input.value) || 0;
      const session = audio.current();
      if (session && session.id === sessionId() && session.controls && typeof session.controls.seekTo === 'function') {
        session.controls.seekTo(target);
        position = target;
      }
      showTime(position);
    },
    pointercancel: () => { dragging = false; place(); },
  };
  for (const [type, fn] of Object.entries(listeners)) input.addEventListener(type, fn);
  return {
    paint(player) {
      if (!player) return;
      duration = Number(player.duration) > 0 ? Number(player.duration) : 0;
      position = Number(player.position) > 0 ? Number(player.position) : 0;
      const max = String(Math.floor(duration));
      if (input.getAttribute('max') !== max) input.setAttribute('max', max);
      const disabled = !(player.canSeek && duration > 0);
      if (input.disabled !== disabled) input.disabled = disabled;
      place();
      if (!dragging) showTime(position);
    },
    destroy() {
      for (const [type, fn] of Object.entries(listeners)) input.removeEventListener(type, fn);
      range.destroy();
    },
  };
}
