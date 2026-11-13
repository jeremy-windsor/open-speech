/**
 * Shared pieces of the Voices view: the list store (request tokens, staleness), load and error states,
 * inline banners, stable ids, focus kept across a list re-render, card menus, and copy used by more than one
 * panel. No module-level DOM access.
 * @module views/voices/shared
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { explain } from '../../core/api.js';
import * as voices from '../../core/voices.js';
import * as modelsApi from '../../core/models.js';
import { bindMenuButton } from '../../components/menu.js';

/** The fixed sentence a saved recording's clone test speaks. */
export const TEST_SENTENCE = 'This is an Open Speech voice-cloning test using the saved recording.';

/** The fixed sentence the wizard's test speaks: the recording isn't saved yet. */
export const WIZARD_TEST_SENTENCE = 'This is an Open Speech voice-cloning test using your new recording.';

/** The hint on previews and clone tests while the microphone records. */
export const STOP_RECORDING = 'Stop recording to listen.';

/** The search result, said once typing pauses: "2 named voices match “jo”", "No presets match “x”", "5 recordings". */
export function searchSummary(what, count, query) {
  const nouns = { named: ['named voice', 'named voices'], presets: ['preset', 'presets'], recordings: ['recording', 'recordings'] }[what];
  if (!nouns) return '';
  const n = Number(count) || 0;
  const q = String(query || '').trim();
  const noun = n === 1 ? nouns[0] : nouns[1];
  if (!q) return `${n} ${noun}`;
  if (!n) return `No ${nouns[1]} match “${q}”`;
  return n === 1 ? `1 ${noun} matches “${q}”` : `${n} ${noun} match “${q}”`;
}

/** A list is reloaded on show when it is older than this (GS-12). */
export const STALE_MS = 30000;

/** A safe id fragment for any key (uuids and recording names keep their shape). */
export function idPart(key) {
  return String(key ?? '').replace(/[^A-Za-z0-9_-]/g, '_') || 'x';
}

/** "1 preset uses it", "2 presets use it", "No presets use it". */
export function presetCountText(n) {
  const count = Number(n) || 0;
  if (!count) return 'No presets use it';
  return count === 1 ? '1 preset uses it' : `${count} presets use it`;
}

/** The display name of a model id ('Kokoro', 'Piper en_US-amy-medium', 'Chatterbox'). */
export function modelName(modelId) {
  const id = String(modelId || '');
  if (!id) return 'Unknown model';
  try {
    return modelsApi.modelLabel(id) || id;
  } catch {
    return id;
  }
}

/** TTS models that can clone a voice (capabilities.voice_clone), in the app's model order (VL-13). */
export function cloneModels(list) {
  return modelsApi.ttsModels(list || []).filter((m) => m && m.capabilities && m.capabilities.voice_clone === true);
}

/**
 * Whether a model can be used now: {known: false} until the inventory has loaded (nothing is blocked on a
 * guess), else {known: true, ok} from models.ttsModels() (installed, worker reachable).
 * @param {string} modelId
 * @param {{list: object[], fetchedAt: number}} modelsState app.models
 */
export function modelAvailability(modelId, modelsState) {
  const state = modelsState || { list: [], fetchedAt: 0 };
  if (!state.fetchedAt && !(state.list || []).length) return { known: false, ok: true };
  return { known: true, ok: modelsApi.ttsModels(state.list).some((m) => m.id === modelId) };
}

/** Lower-cased, trimmed search text. */
export function normaliseQuery(text) {
  return String(text || '').trim().toLowerCase();
}

/** Whether any of `fields` contains the query (an empty query matches everything). */
export function matches(query, ...fields) {
  const q = normaliseQuery(query);
  if (!q) return true;
  return fields.some((f) => String(f ?? '').toLowerCase().includes(q));
}

/** Copy a list of names: "A", "A and B", "A, B and C". */
export function listNames(names) {
  const list = (names || []).map(String);
  if (list.length <= 1) return list.join('');
  return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
}

/* ───────────────────────────── the list store ───────────────────────────── */

const LOADERS = {
  named: async () => ({ items: await voices.listNamed({ check: false }) }),
  presets: async () => {
    const { presets, defaultId } = await voices.listPresets();
    return { items: presets, defaultId };
  },
  recordings: async () => ({ items: await voices.listRecordings() }),
};

/**
 * The three lists the view shows. Each load takes a request token, so a late answer never overwrites a
 * newer one; `onChange(what)` runs when a list starts loading, loads or fails.
 * @param {(what: string) => void} onChange
 */
export function createLists(onChange = () => {}) {
  const lists = {};
  for (const what of Object.keys(LOADERS)) {
    lists[what] = { items: [], defaultId: null, status: 'idle', error: null, at: 0, seq: 0 };
  }
  async function load(what) {
    const list = lists[what];
    if (!list) throw new TypeError(`Unknown list: ${what}`);
    const seq = (list.seq += 1);
    list.status = 'loading';
    onChange(what);
    try {
      const result = await LOADERS[what]();
      if (seq !== list.seq) return false;
      list.items = Array.isArray(result.items) ? result.items : [];
      if ('defaultId' in result) list.defaultId = result.defaultId || null;
      list.status = 'ready';
      list.error = null;
      list.at = Date.now();
      onChange(what);
      return true;
    } catch (error) {
      if (seq !== list.seq) return false;
      list.status = 'error';
      list.error = error;
      onChange(what);
      return false;
    }
  }
  return {
    lists,
    load,
    /** Apply an answer the server already gave (POST …/default returns the whole preset list). */
    put(what, items, extra = {}) {
      const list = lists[what];
      list.seq += 1;
      list.items = items;
      Object.assign(list, extra);
      list.status = 'ready';
      list.error = null;
      list.at = Date.now();
      onChange(what);
    },
    /** True when the list has never loaded or is older than STALE_MS. */
    stale(what, now = Date.now()) {
      const list = lists[what];
      return !list.at || now - list.at > STALE_MS;
    },
  };
}

/* ───────────────────────────── states ───────────────────────────── */

/**
 * Loading placeholder: flat skeleton blocks inside an aria-busy container with a visually hidden "Loading…".
 * @param {string} what e.g. 'named voices'
 * @param {{rows?: number, kind?: 'card'|'row'}} [options]
 */
export function loadingMarkup(what, { rows = 3, kind = 'row' } = {}) {
  return html`<div class="vo-loading vo-loading--${kind}" aria-busy="true"><span class="visually-hidden">Loading ${what}…</span>${Array.from({ length: rows }, () => html`<div class="skeleton skeleton--row vo-skeleton" aria-hidden="true"></div>`)}</div>`;
}

/** An auth refusal or the wrong-key lockout: the fix is in Settings › Connection, never a retry. */
export function isAuthError(error) {
  return !!error && (error.kind === 'auth' || error.kind === 'locked' || error.status === 401);
}

/**
 * The error state of a list that didn't load: plain cause, Retry, and Details; an auth refusal says
 * "API key needed" with [Open Settings] (the shell's banner carries the rest). Call fillDetails() after
 * rendering it. It sits in a card, like the empty states. Details are `explain().detailText`: the
 * server's message and code, never its JSON envelope, and none when they would only repeat the message.
 * @param {*} error
 * @param {{what: string, list: string}} options `list` is the list Retry reloads ('named', 'presets',
 *   'recordings'): one action, data-action="retry-list", with the list as its parameter (data-list), the way
 *   the item actions carry data-id.
 */
export function loadErrorMarkup(error, { what, list }) {
  const e = explain(error, `Couldn't load ${what}`);
  const auth = isAuthError(error);
  const title = auth ? 'API key needed' : `Couldn't load ${what}`;
  const message = auth ? 'Enter the server’s API key in Settings › Connection, then try again.' : (e.message || 'Something went wrong.');
  const detail = auth ? '' : e.detailText;
  return html`<div class="card vo-empty-card vo-state"><div class="empty empty--error">${icon('alert', { className: 'ico empty__icon' })}<h2 class="empty__title">${title}</h2><p class="empty__text">${message}</p>${auth
    ? html`<button type="button" class="btn btn--secondary" data-action="open-settings"><span class="btn__label">Open Settings</span></button>`
    : html`<button type="button" class="btn btn--secondary" data-action="retry-list" data-list="${list}">${icon('retry')}<span class="btn__label">Retry</span></button>`}${detail ? html`<details class="banner__details vo-state__details"><summary>Details</summary><pre></pre></details>` : ''}</div></div>`;
}

/** The Details text loadErrorMarkup() shows for an error (what fillDetails() puts in its <pre>). */
export function loadErrorDetail(error, what) {
  if (isAuthError(error)) return '';
  return explain(error, `Couldn't load ${what}`).detailText;
}

/**
 * A refresh that failed while older data is still on screen: a small banner above the list.
 * @param {*} error
 * @param {{what: string, list: string}} options
 */
export function staleErrorMarkup(error, { what, list }) {
  const e = explain(error, `Couldn't refresh ${what}`);
  return html`<div class="banner banner--warning vo-banner" role="status">${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">Couldn't refresh ${what}</p><p class="banner__msg">${e.message || 'Showing the last list that loaded.'}</p></div><div class="banner__actions"><button type="button" class="btn btn--secondary btn--sm" data-action="retry-list" data-list="${list}"><span class="btn__label">Retry</span></button></div></div>`;
}

/** Fill the <pre> of a Details disclosure with text (textContent only). */
export function fillDetails(root, detail) {
  const pre = root ? root.querySelector('details pre') : null;
  if (pre) pre.textContent = String(detail || '');
}

/**
 * An inline error banner next to the control that failed: role=alert, a title, the plain message, a
 * Details disclosure for the server text, optional actions and a Dismiss button.
 * @param {HTMLElement} host where it renders (replaces its content)
 * @param {{title: string, message: string, detail?: string, actions?: {label: string, action: string}[],
 *   dismiss?: boolean}} banner each action button is data-action="<action>", run by the view's dom.actions()
 *   map like any other button; `action` is always a name from the code, never server text
 */
export function showBanner(host, { title, message, detail = '', actions = [], dismiss = true }) {
  if (!host) return null;
  render(host, html`<div class="banner banner--error vo-banner" role="alert">${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">${title}</p>${message ? html`<p class="banner__msg">${message}</p>` : ''}${detail ? html`<details class="banner__details"><summary>Details</summary><pre></pre></details>` : ''}</div>${actions.length ? html`<div class="banner__actions">${actions.map((a) => html`<button type="button" class="btn btn--secondary btn--sm" data-action="${a.action}"><span class="btn__label">${a.label}</span></button>`)}</div>` : ''}${dismiss ? html`<button type="button" class="icon-btn icon-btn--sm" data-action="dismiss-banner" aria-label="Dismiss">${icon('x')}</button>` : ''}</div>`);
  if (detail) fillDetails(host, detail);
  host.hidden = false;
  return host.firstElementChild;
}

/** Remove an inline banner. */
export function clearBanner(host) {
  if (!host) return;
  render(host, html``);
  host.hidden = true;
}

/**
 * Inline errors and status lines of list items that outlive a re-render (a list re-renders when it reloads
 * or the models change): they are kept by element id and put back after every render. A Dismiss in a slot
 * bubbles a 'vo-dismiss' event (from the view), which forgets it.
 * @param {HTMLElement} panel the panel that listens for 'vo-dismiss'
 */
export function createSlots(panel) {
  const errors = new Map();   // slot id -> banner
  const texts = new Map();    // element id -> text
  const onDismiss = (event) => {
    const id = event && event.detail ? event.detail.id : '';
    if (id) errors.delete(id);
  };
  if (panel) panel.addEventListener('vo-dismiss', onDismiss);
  const find = (root, id) => (root && root.querySelector ? root.querySelector(`#${id}`) : null) || document.getElementById(id);
  return {
    /** Show a banner in the slot `id` (inside `root`) and keep it. */
    error(root, id, banner) {
      errors.set(id, banner);
      const host = find(root, id);
      if (host) showBanner(host, banner);
    },
    /** Remove the banner of slot `id`. */
    clear(root, id) {
      errors.delete(id);
      const host = find(root, id);
      if (host) clearBanner(host);
    },
    /** Set (or clear, with '') the text of a status line and keep it. */
    text(root, id, value) {
      if (value) texts.set(id, value);
      else texts.delete(id);
      const el = find(root, id);
      if (el) {
        el.textContent = value || '';
        el.hidden = !value;
      }
    },
    /** Put every kept banner and status back after a render (items that are gone are forgotten). */
    restore(root) {
      for (const [id, banner] of [...errors]) {
        const host = root.querySelector(`#${id}`);
        if (host) showBanner(host, banner);
        else errors.delete(id);
      }
      for (const [id, value] of [...texts]) {
        const el = root.querySelector(`#${id}`);
        if (el) {
          el.textContent = value;
          el.hidden = false;
        } else {
          texts.delete(id);
        }
      }
    },
    destroy() {
      if (panel) panel.removeEventListener('vo-dismiss', onDismiss);
    },
  };
}

/**
 * The copy for a failed action: `explain()` with a banner title, and the server's own message kept (it says
 * why, for example "requires an exact reference transcript"). An auth refusal or lockout is titled by its
 * cause ("API key needed") and offers [Open Settings]; Details are `explain().detailText`, the readable
 * server message and code, and none when they would only repeat the message.
 * @returns {{title: string, message: string, detail: string, actions: {label: string, action: string}[]}}
 */
export function actionError(error, title) {
  const e = explain(error, title);
  const auth = isAuthError(error);
  const message = e.message || String(error?.message || '');
  return {
    title: auth ? (e.cause || 'API key needed') : (e.title || title),
    message,
    detail: auth ? '' : e.detailText,
    actions: auth ? [{ label: 'Open Settings', action: 'open-settings' }] : [],
  };
}

/* ───────────────────────────── focus ───────────────────────────── */

function focusTarget(target) {
  if (!target || typeof target.focus !== 'function') return;
  if (/^h[1-6]$/i.test(target.tagName) && !target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
  target.focus({ preventScroll: false });
}

/**
 * Re-render a list and keep keyboard focus on the same control: the focused element's id is looked up again
 * in the new markup; when it is gone (the item was deleted), `fallback()` picks where focus goes.
 * @param {HTMLElement} el
 * @param {import('../../core/dom.js').SafeHtml} markup
 * @param {{fallback?: () => HTMLElement|null, wasFocused?: string}} [options]
 *   wasFocused: an id that had focus before an async action (focus may have moved to <body> meanwhile)
 */
export function renderKeepingFocus(el, markup, { fallback, wasFocused = '' } = {}) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  const inside = !!(active && active !== el && el.contains(active));
  const lost = !active || active === document.body || active === document.documentElement || active.isConnected === false;
  const id = inside ? active.id : (lost ? wasFocused : '');
  render(el, markup);
  if (!inside && !(lost && wasFocused)) return;
  const again = id ? document.getElementById(id) : null;
  const target = again && el.contains(again) ? again : (typeof fallback === 'function' ? fallback() : null);
  focusTarget(target);
}

/** Move focus to an element (a heading gets tabindex="-1"). */
export function focusElement(target) {
  focusTarget(target);
}

/* ───────────────────────────── card menus ───────────────────────────── */

/**
 * A card's "More actions" button with its menu written right after it, inside the card. While
 * the menu is open, components/menu.js moves it to the end of the view section (a .card is a size container,
 * which would become the containing block of the fixed-position menu and misplace it) and puts it back when
 * it closes; Tab goes on from the button and Shift+Tab stops on it, so focus order follows the button.
 * @param {string} key a stable id fragment
 * @param {string} label the item's name, for the button's accessible name and the menu's label
 * @param {{action: string, label: string, icon: string, danger?: boolean, disabled?: string}[]} items
 *   disabled: the reason the item can't be used now (it stays listed, aria-disabled)
 */
export function cardMenuMarkup(key, label, items) {
  return html`<button type="button" class="icon-btn vo-more" id="vo-${key}-more" data-menu-btn="vo-${key}-menu" aria-label="More actions for ${label}" data-tooltip="More actions">${icon('dots')}</button><div class="menu" id="vo-${key}-menu" role="menu" aria-label="${label}" hidden>${items.map((item) => html`<button type="button" role="menuitem" tabindex="-1" class="menu__item${item.danger ? ' menu__item--danger' : ''}" data-menu-action="${item.action}"${item.disabled ? html` aria-disabled="true" title="${item.disabled}"` : ''}>${icon(item.icon)}<span>${item.label}</span></button>`)}</div>`;
}

/**
 * The open card menu of `listEl` that has keyboard focus: its id and the index of the focused item, or null.
 * The menu may sit at the end of the view section while it is open (menu.js re-hosts it), so it belongs to
 * the list when one of the list's buttons opens it. Read it before a re-render replaces the cards (unbinding
 * closes the menus, which drops focus to <body>), then hand it to `bindMenus(…).reopen()`.
 * @param {HTMLElement|null} listEl
 * @returns {{menuId: string, index: number}|null}
 */
export function openMenuFocus(listEl) {
  const active = typeof document !== 'undefined' ? document.activeElement : null;
  const menu = active && active.closest ? active.closest('[role="menu"][id]') : null;
  if (!listEl || !menu || menu.hidden) return null;
  const ours = listEl.contains(menu)
    || Array.from(listEl.querySelectorAll('[data-menu-btn]')).some((b) => b.getAttribute('data-menu-btn') === menu.id);
  if (!ours) return null;
  const items = Array.from(menu.querySelectorAll('[role="menuitem"]'));
  return { menuId: menu.id, index: Math.max(0, items.indexOf(active)) };
}

/**
 * Bind every `[data-menu-btn]` button in `root` to its menu, written right after it (cardMenuMarkup()).
 * Items carry data-menu-action; onSelect(action, item, button) runs for clicks and keys alike.
 * @returns {(() => void) & {reopen: (memo: {menuId: string, index: number}|null) => boolean}} unbind all
 *   (an open menu closes and goes back into its card); `reopen(memo)` opens the menu of an openMenuFocus()
 *   memo again, with focus on the same item (false when that menu is gone)
 */
export function bindMenus(root, onSelect) {
  const bound = new Map();   // menu id -> {controller, menu}
  for (const button of root.querySelectorAll('[data-menu-btn]')) {
    const id = button.getAttribute('data-menu-btn');
    const next = button.nextElementSibling;
    const menu = next && next.id === id ? next : document.getElementById(id);
    if (!menu) continue;
    const controller = bindMenuButton(button, menu, {
      placement: 'bottom-end',
      onSelect: (item) => onSelect(item.getAttribute('data-menu-action'), item, button),
    });
    bound.set(id, { controller, menu });
  }
  const unbind = () => {
    for (const entry of bound.values()) entry.controller.destroy();
    bound.clear();
  };
  unbind.reopen = (memo) => {
    const entry = memo ? bound.get(memo.menuId) : null;
    if (!entry) return false;
    entry.controller.open('first');
    const items = Array.from(entry.menu.querySelectorAll('[role="menuitem"]'));
    const item = items[Math.min(memo.index, items.length - 1)];
    if (item && document.activeElement !== item) {
      for (const other of items) other.setAttribute('tabindex', other === item ? '0' : '-1');
      item.focus();
    }
    return true;
  };
  return unbind;
}

/**
 * After a list re-render: open the card menu that had focus again (see openMenuFocus()), or, when its card is
 * gone, put focus on `fallback()` (else the first card title in `listEl`), never on <body>.
 * @param {ReturnType<typeof bindMenus>} menus
 * @param {{menuId: string, index: number}|null} memo
 * @param {HTMLElement} listEl
 * @param {(() => HTMLElement|null)|undefined} fallback
 */
export function restoreMenuFocus(menus, memo, listEl, fallback) {
  if (!memo || menus.reopen(memo)) return;
  const target = (typeof fallback === 'function' ? fallback() : null)
    || listEl.querySelector('.vo-card__title, .vo-preset__name, .empty__title');
  focusTarget(target);
}
