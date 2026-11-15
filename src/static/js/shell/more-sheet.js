/**
 * The More sheet (#app-more-sheet): Voices, Studio, History, Models and Settings as links that close
 * the sheet, the status summary (it opens the status sheet) and the theme segmented control.
 * @module shell/more-sheet
 */
import { openSheet, closeSheet } from '../core/dialogs.js';
import * as theme from '../core/theme.js';

let st = null;

function dialogEl() {
  return document.getElementById('app-more-sheet');
}

function syncTheme() {
  const choice = theme.getTheme();
  for (const input of document.querySelectorAll('#app-more-sheet input[name="app-theme"]')) input.checked = input.value === choice;
}

/**
 * Open the More sheet.
 * @param {HTMLElement} [trigger]
 */
export function openMore(trigger) {
  const dialog = dialogEl();
  if (!dialog || dialog.open) return;
  syncTheme();
  openSheet(dialog, { trigger });
}

/** Close it. */
export function closeMore() {
  const dialog = dialogEl();
  if (dialog && dialog.open) closeSheet(dialog);
}

/**
 * Start the sheet.
 * @param {{openShortcuts?: (trigger: HTMLElement) => void}} [options]
 * @returns {{destroy: () => void}}
 */
export function initMoreSheet({ openShortcuts = () => {} } = {}) {
  destroy();
  st = { offs: [] };
  const dialog = dialogEl();
  if (!dialog) return { destroy };
  const onClick = (event) => {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    if (!start || !start.closest) return;
    if (start.closest('a.more-link')) {
      closeMore();   // the link's hash change then routes and focuses the view heading
      return;
    }
    if (start.closest('#app-more-shortcuts')) {
      closeMore();
      openShortcuts(document.getElementById('nav-more') || undefined);
    }
  };
  const onChange = (event) => {
    const input = event.target;
    if (input && input.name === 'app-theme' && input.checked) theme.setTheme(input.value);
  };
  dialog.addEventListener('click', onClick);
  dialog.addEventListener('change', onChange);
  st.offs.push(() => dialog.removeEventListener('click', onClick));
  st.offs.push(() => dialog.removeEventListener('change', onChange));
  st.offs.push(theme.onThemeChange(syncTheme));
  return { destroy };
}

/** Stop the sheet. */
export function destroy() {
  if (!st) return;
  for (const off of st.offs) off();
  st = null;
}

/** Test hook. */
export function __resetForTests() {
  destroy();
}
