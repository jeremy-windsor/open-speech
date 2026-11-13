/**
 * The "?" shortcuts sheet (#app-shortcuts-sheet): every registered shortcut, grouped by scope,
 * with its keys as <kbd>. It also says whether single-key shortcuts are on (Settings › Shortcuts, WCAG 2.1.4).
 * @module shell/shortcuts-sheet
 */
import { html, render } from '../core/dom.js';
import { list, modLabel } from '../core/shortcuts.js';
import { openSheet } from '../core/dialogs.js';
import { app } from '../core/store.js';

const SCOPE_TITLES = { global: 'Anywhere', player: 'While audio plays' };
const VIEW_TITLES = { reader: 'Reader', speak: 'Speak', transcribe: 'Transcribe', voices: 'Voices', studio: 'Studio', history: 'History', models: 'Models', settings: 'Settings' };
const KEY_NAMES = { ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓', Escape: 'Esc', Space: 'Space', ' ': 'Space', Enter: 'Enter' };

/** PURE: the title of a scope group. */
export function scopeTitle(scope) {
  if (SCOPE_TITLES[scope]) return SCOPE_TITLES[scope];
  if (scope.startsWith('view:')) return VIEW_TITLES[scope.slice(5)] || scope.slice(5);
  return '';
}

/**
 * PURE: one key combination as a list of steps, each a list of key caps:
 * 'g r' -> [['g'], ['r']], 'Mod+Enter' -> [['Ctrl', 'Enter']].
 */
export function keyCaps(combo, mod = 'Ctrl') {
  return String(combo).trim().split(/\s+/).map((step) => {
    const parts = step.split('+');
    let key = parts.pop();
    if (key === '' && parts.length) key = '+';
    const caps = parts.map((p) => (p.toLowerCase() === 'mod' ? mod : p.charAt(0).toUpperCase() + p.slice(1)));
    caps.push(KEY_NAMES[key] || key);
    return caps;
  });
}

function comboMarkup(combo, mod) {
  const steps = keyCaps(combo, mod);
  return html`${steps.map((caps, i) => html`${i ? html` <span>then</span> ` : ''}${caps.map((cap, j) => html`${j ? '+' : ''}<kbd class="kbd">${cap}</kbd>`)}`)}`;
}

/** PURE: the shortcuts grouped for the sheet, in a stable order (Anywhere, While audio plays, views). */
export function groupShortcuts(entries) {
  const groups = new Map();
  for (const e of entries) {
    if (e.scope.startsWith('dialog:')) continue;
    const title = scopeTitle(e.scope);
    if (!title) continue;
    if (!groups.has(title)) groups.set(title, []);
    groups.get(title).push(e);
  }
  const order = ['Anywhere', 'While audio plays', ...Object.values(VIEW_TITLES)];
  return [...groups.entries()].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0])).map(([title, items]) => ({ title, items }));
}

function markup() {
  const mod = modLabel();
  const groups = groupShortcuts(list());
  const on = (app.get().prefs || {}).shortcuts !== false;
  return html`<div class="shortcuts">${groups.map((g, gi) => html`<section class="shortcuts__group" aria-labelledby="${`app-sc-g${gi}`}"><h3 class="section-label" id="${`app-sc-g${gi}`}">${g.title}</h3><ul class="shortcuts__list">${g.items.map((item) => html`<li class="shortcuts__row"><span>${item.description}</span><span class="shortcuts__keys">${item.keys.map((k, i) => html`${i ? html` <span>or</span> ` : ''}${comboMarkup(k, mod)}`)}</span></li>`)}</ul></section>`)}</div>
    <p class="shortcuts__note">${on ? 'Single-key shortcuts can be turned off in Settings › Shortcuts.' : 'Single-key shortcuts are off. Turn them on in Settings › Shortcuts.'}</p>`;
}

/**
 * Open the sheet.
 * @param {HTMLElement} [trigger]
 */
export function openShortcuts(trigger) {
  const dialog = document.getElementById('app-shortcuts-sheet');
  const body = document.getElementById('app-shortcuts-body');
  if (!dialog || !body || dialog.open) return;
  render(body, markup());
  openSheet(dialog, { trigger });
}
