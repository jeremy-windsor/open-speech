/**
 * Navigation: links, not tabs. The rail (>= 960 px) and the bottom bar (< 960 px) are
 * <nav aria-label="Main"> lists of <a href="#view">; the current link gets aria-current="page". There is no
 * tablist and no arrow-key roving between them (BUG-19): arrow keys never navigate or add history entries.
 * More (#nav-more) opens the More sheet; it shows the current style, and its name says so, when the
 * current view lives in the sheet. The rail's theme button cycles System → Light → Dark.
 * @module shell/nav
 */
import { app } from '../core/store.js';
import * as theme from '../core/theme.js';
import { html, render } from '../core/dom.js';
import { icon } from '../core/icons.js';

// Below 960 px the bottom bar holds Reader, Speak, Transcribe, Talk and More; these views live in the More sheet
// (Home is also behind the top bar's brand).
export const MORE_VIEWS = Object.freeze(['home', 'voices', 'studio', 'history', 'models', 'settings']);
const TITLES = { home: 'Home', reader: 'Reader', speak: 'Speak', transcribe: 'Transcribe', talk: 'Talk', voices: 'Voices', studio: 'Studio', history: 'History', models: 'Models', settings: 'Settings' };
const THEME_ORDER = ['system', 'light', 'dark'];
const THEME_WORDS = { system: 'System', light: 'Light', dark: 'Dark' };
const THEME_ICONS = { system: 'monitor', light: 'sun', dark: 'moon' };

let st = null;

/** PURE: the More button's accessible name for a view. */
export function moreLabel(view) {
  return MORE_VIEWS.includes(view) ? `More, current: ${TITLES[view]}` : 'More';
}

/** Mark the current view in every nav (rail, bottom bar, More sheet). */
export function markCurrent(view) {
  for (const link of document.querySelectorAll('.nav-link[data-view]')) {
    if (link.getAttribute('data-view') === view) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  const more = document.getElementById('nav-more');
  if (more) {
    more.classList.toggle('is-current', MORE_VIEWS.includes(view));
    more.setAttribute('aria-label', moreLabel(view));
  }
}

function paintThemeToggle() {
  const button = document.getElementById('app-theme-toggle');
  if (!button) return;
  const choice = theme.getTheme();
  const label = `Theme: ${THEME_WORDS[choice]}. Change theme`;
  button.setAttribute('aria-label', label);
  button.setAttribute('data-tooltip', `Theme: ${THEME_WORDS[choice]}`);
  const use = button.querySelector('use');
  if (!use || !use.getAttribute('href').endsWith(`#i-${THEME_ICONS[choice]}`)) render(button, html`${icon(THEME_ICONS[choice])}`);
}

/** PURE: the next theme in the rail's cycle. */
export function nextTheme(current) {
  const i = THEME_ORDER.indexOf(current);
  return THEME_ORDER[(i + 1) % THEME_ORDER.length];
}

/**
 * Start the navs.
 * @param {{openMore?: (trigger: HTMLElement) => void}} [options]
 * @returns {{destroy: () => void}}
 */
export function initNav({ openMore = () => {} } = {}) {
  destroy();
  st = { offs: [] };
  const on = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    st.offs.push(() => el.removeEventListener(type, fn));
  };
  const more = document.getElementById('nav-more');
  on(more, 'click', () => openMore(more));
  on(document.getElementById('app-theme-toggle'), 'click', () => {
    theme.setTheme(nextTheme(theme.getTheme()));
  });
  // The skip link moves focus to <main> without touching the hash (the hash is the route).
  for (const link of document.querySelectorAll('.skip-link')) {
    on(link, 'click', (event) => {
      event.preventDefault();
      const main = document.getElementById('main');
      if (main) main.focus();
    });
  }
  st.offs.push(app.select((s) => s.route && s.route.view, (view) => markCurrent(view || ''), { immediate: true }));
  st.offs.push(theme.onThemeChange(() => paintThemeToggle()));
  paintThemeToggle();
  return { destroy };
}

/** Stop the navs. */
export function destroy() {
  if (!st) return;
  for (const off of st.offs) off();
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
