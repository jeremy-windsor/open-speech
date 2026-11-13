/**
 * The connection banner (#app-conn-banner, role=status) and the boot error (#app-boot-error).
 * - offline: "Can't reach Open Speech." with "Retrying in 8 s." and [Retry now]
 * - unauthorized: the refusal in two parts, [Open Settings]: "This server needs an API key." / "Enter it in
 *   Settings › Connection." with no key, "The server didn't accept the API key." / "Check it in Settings ›
 *   Connection." for a refused one
 * - locked: "Too many wrong API keys from this device." with the time left and [Open Settings]
 * One banner, never repeated as a toast: a refused page load shows exactly this notice (AU-10). The
 * countdown ticks in an aria-hidden span; the live region changes only when the state changes.
 * While online the banner stays in the accessibility tree, empty and visually hidden (never display:none),
 * so screen readers already track the live region when its first message arrives and announce it.
 * @module shell/banners
 */
import { html, render } from '../core/dom.js';
import { icon } from '../core/icons.js';
import { app } from '../core/store.js';
import { refreshHealth } from '../core/status.js';
import { openConnectionSettings } from './status-chip.js';

let st = null;

/**
 * PURE: what the connection banner shows for a connection state, or null when it is hidden.
 * @param {object} connection app.connection
 * @returns {{tone: 'warning'|'error', title: string, message: string, spoken: string, countdown: boolean,
 *   action: 'retry'|'settings'}|null}
 */
export function bannerModel(connection = {}) {
  const c = connection || {};
  if (c.state === 'offline') {
    return { tone: 'warning', title: "Can't reach Open Speech.", message: 'Retrying in', spoken: 'Retrying automatically.', countdown: true, action: 'retry' };
  }
  if (c.state === 'unauthorized') {
    // The refusal message's first sentence is the title and the rest the line under it, so each is said once
    // and the title never contradicts the message: no key at all reads "This server needs an API key." /
    // "Enter it in Settings › Connection."; a refused key "The server didn't accept the API key." / "Check it
    // in Settings › Connection."; an unsendable key names its character the same way.
    const { title, message } = refusalCopy(c.error);
    return { tone: 'warning', title, message, spoken: '', countdown: false, action: 'settings' };
  }
  if (c.state === 'locked') {
    return { tone: 'error', title: 'Too many wrong API keys from this device.', message: 'The server will accept requests again in', spoken: 'The server will accept requests again shortly.', countdown: true, action: 'settings' };
  }
  return null;
}

const NEEDS_KEY = 'This server needs an API key.';

/**
 * PURE: the banner title and message for an API key refusal (auth.refusalMessage(), as app.connection.error).
 * @param {string} [error]
 * @returns {{title: string, message: string}}
 */
export function refusalCopy(error) {
  const text = String(error || '').trim();
  const split = /^(.+?[.!?])\s+(\S[\s\S]*)$/.exec(text);
  const title = split ? split[1] : (text || NEEDS_KEY);
  const message = split ? split[2] : (title === NEEDS_KEY ? 'Enter it in Settings › Connection.' : 'Check it in Settings › Connection.');
  return { title, message };
}

function secondsLeft(connection) {
  const at = Number(connection.retryAt) || 0;
  if (at) return Math.max(0, Math.ceil((at - Date.now()) / 1000));
  return Math.max(0, Math.ceil((Number(connection.retryInMs) || 0) / 1000));
}

function paintCountdown() {
  if (!st || !st.el) return;
  const span = st.el.querySelector('.banner__countdown');
  if (!span) return;
  const text = `${secondsLeft(app.get().connection)} s.`;
  if (span.textContent !== text) span.textContent = text;
}

/** The online state: an empty live region that is in the accessibility tree but takes no space. */
function idle(el) {
  el.hidden = false;
  el.className = 'banner banner--warning visually-hidden';
  el.setAttribute('data-state', '');
  for (const svg of el.querySelectorAll('svg[data-icon]')) svg.setAttribute('hidden', '');
  const content = el.querySelector('#app-conn-banner-content') || el;
  if (content.textContent !== '' || content.children.length) render(content, html``);
}

function paint(connection) {
  const el = st && st.el;
  if (!el) return;
  const model = bannerModel(connection);
  const key = model ? `${connection.state}|${model.message}` : '';
  clearInterval(st.timer);
  st.timer = null;
  if (!model) {
    if (st.key !== '' || el.getAttribute('data-state') !== '') idle(el);
    st.key = '';
    return;
  }
  if (key !== st.key) {
    // The live region changes only when the state or its message changes.
    st.key = key;
    el.className = `banner banner--${model.tone}`;
    el.setAttribute('data-state', connection.state);
    // The icons are static elements of the shell, resolved while the page loaded: an icon created now, with the
    // server unreachable, could not fetch the sprite and would stay blank.
    const iconName = model.tone === 'error' ? 'lock' : (connection.state === 'offline' ? 'offline' : 'key');
    // (an SVG element has no hidden property: toggle the attribute)
    for (const svg of el.querySelectorAll('svg[data-icon]')) svg.toggleAttribute('hidden', svg.getAttribute('data-icon') !== iconName);
    const content = el.querySelector('#app-conn-banner-content') || el;
    render(content, html`
      <div class="banner__body"><p class="banner__title">${model.title}</p><p class="banner__msg">${model.countdown ? html`<span aria-hidden="true">${model.message} <span class="banner__countdown tabular"></span></span><span class="visually-hidden">${model.spoken}</span>` : model.message}</p></div>
      <div class="banner__actions">${model.action === 'retry'
        ? html`<button type="button" class="btn btn--sm btn--secondary" data-banner-action="retry"><span class="btn__label">Retry now</span></button>`
        : html`<button type="button" class="btn btn--sm btn--secondary" data-banner-action="settings"><span class="btn__label">Open Settings</span></button>`}</div>`);
  }
  if (model.countdown) {
    paintCountdown();
    st.timer = setInterval(paintCountdown, 1000);
  }
}

/**
 * Show the boot error: "Open Speech didn't start: <message>" and [Reload] (it replaces the old
 * "Init failed" toast, GS-17).
 * @param {*} error
 */
export function showBootError(error) {
  const el = document.getElementById('app-boot-error');
  const message = (error && (error.message || String(error))) || 'Unknown error';
  if (!el) return;
  render(el, html`${icon('alert', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">Open Speech didn't start: ${message}</p><p class="banner__msg">Reload the page to try again. If it keeps happening, check that the server is up to date.</p></div><div class="banner__actions"><button type="button" class="btn btn--sm btn--primary" id="app-boot-reload"><span class="btn__label">Reload</span></button></div>`);
  el.hidden = false;
  const reload = document.getElementById('app-boot-reload');
  if (reload) reload.addEventListener('click', () => location.reload());
}

/**
 * Start the connection banner.
 * @returns {{destroy: () => void}}
 */
export function initBanners() {
  destroy();
  const el = document.getElementById('app-conn-banner');
  st = { el, key: '', timer: null, offs: [] };
  if (!el) return { destroy };
  idle(el);
  const onClick = (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const button = start && start.closest ? start.closest('[data-banner-action]') : null;
    if (!button) return;
    const action = button.getAttribute('data-banner-action');
    if (action === 'retry') refreshHealth().catch(() => {});
    else if (action === 'settings') openConnectionSettings();
  };
  el.addEventListener('click', onClick);
  st.offs.push(() => el.removeEventListener('click', onClick));
  st.offs.push(app.select((s) => s.connection, paint, { immediate: true }));
  return { destroy };
}

/** Stop the banner. */
export function destroy() {
  if (!st) return;
  clearInterval(st.timer);
  for (const off of st.offs) off();
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
