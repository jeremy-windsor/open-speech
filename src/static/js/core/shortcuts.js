/**
 * Keyboard shortcuts. One document keydown listener (bubble phase, so element handlers
 * that call preventDefault() win), installed on the first register().
 * @module core/shortcuts
 */
import { isEditable } from './dom.js';
import { app } from './store.js';
import { topSheet } from './dialogs.js';

const SEQUENCE_MS = 1000;
// Space and Enter activate these natively; editable fields are covered by isEditable(). Keys pressed in a native
// media player's own controls (its shadow root) reach the page with the <audio>/<video> element as their target.
const ACTIVATABLE = 'button, a[href], summary, audio[controls], video[controls], [role="button"], [role="link"], [role="menuitem"], [role="menuitemradio"], [role="option"], [role="tab"], [role="checkbox"], [role="radio"], [role="switch"]';

let entries = [];
let seq = 0;
let bound = null;
let pending = null;   // {prefix, at}

function isApple() {
  try {
    const platform = navigator.userAgentData?.platform || navigator.platform || '';
    return /mac|iphone|ipad|ipod/i.test(platform);
  } catch {
    return false;
  }
}

function normaliseKeyName(name) {
  const k = String(name);
  if (k === ' ' || k.toLowerCase() === 'space' || k === 'Spacebar') return 'Space';
  if (k.toLowerCase() === 'esc') return 'Escape';
  if (k.length === 1) return k.toLowerCase();
  return k;
}

/** Parse 'Shift+ArrowLeft' / 'Mod+Enter' / 'g r' into a list of steps. */
function parseCombo(combo) {
  return String(combo).trim().split(/\s+/).map((step) => {
    const parts = step.split('+');
    let key = parts.pop();
    if (key === '' && parts.length) key = '+';   // 'Shift++'
    const mods = new Set(parts.map((p) => p.toLowerCase()));
    return {
      key: normaliseKeyName(key),
      mod: mods.has('mod'), ctrl: mods.has('ctrl'), alt: mods.has('alt'), shift: mods.has('shift'), meta: mods.has('meta'),
    };
  });
}

function isCharacter(step) {
  return step.key === 'Space' || step.key.length === 1;
}

function altGraphDown(event) {
  try {
    return typeof event.getModifierState === 'function' && !!event.getModifierState('AltGraph');
  } catch {
    return false;
  }
}

function stepMatches(step, event) {
  const key = normaliseKeyName(event.key);
  if (key !== step.key) return false;
  const apple = isApple();
  // A printed character typed with AltGr (which Windows reports as Ctrl+Alt) or, on Apple, with Option is the
  // character itself, not a chord: many layouts type [ and ] that way. Meta and a real Ctrl+Alt still count.
  const typed = step.key.length === 1 && !step.mod && !step.ctrl && !step.alt && !step.meta;
  const altGraph = typed && altGraphDown(event);
  const ctrlKey = !!event.ctrlKey && !altGraph;
  const altKey = !!event.altKey && !altGraph && !(typed && apple);
  const modDown = apple ? event.metaKey : event.ctrlKey;
  const wantCtrl = step.ctrl || (step.mod && !apple);
  const wantMeta = step.meta || (step.mod && apple);
  if (ctrlKey !== !!wantCtrl) return false;
  if (!!event.metaKey !== !!wantMeta) return false;
  if (step.mod && !modDown) return false;
  if (altKey !== !!step.alt) return false;
  // Shift produces characters such as '?', so it only counts for non-character keys.
  if (!isCharacter(step) && !!event.shiftKey !== !!step.shift) return false;
  if (isCharacter(step) && step.shift && !event.shiftKey) return false;
  return true;
}

function scopeRank(scope) {
  if (scope.startsWith('dialog:')) return 4;
  if (scope.startsWith('view:')) return 3;
  if (scope === 'player') return 2;
  return 1;
}

function scopeActive(scope) {
  if (scope === 'global') return true;
  if (scope === 'player') return !!(app.get().player || app.get().resumable);
  if (scope.startsWith('view:')) return app.get().route.view === scope.slice(5);
  return false;
}

function allowed(entry, event, target) {
  const dialog = topSheet();
  if (dialog) {
    if (entry.scope !== `dialog:${dialog.id}`) return false;
  } else if (entry.scope.startsWith('dialog:') || !scopeActive(entry.scope)) {
    return false;
  }
  if (entry.singleKey && app.get().prefs?.shortcuts === false) return false;
  if (isEditable(target) && !entry.allowInEditable) return false;
  try {
    return !!entry.when(event);
  } catch {
    return false;
  }
}

function pick(candidates) {
  return candidates.sort((a, b) => scopeRank(b.scope) - scopeRank(a.scope) || b.order - a.order)[0] || null;
}

function onKeydown(event) {
  if (event.defaultPrevented || event.isComposing) return;
  const target = event.target;
  const now = Date.now();
  const live = pending && now - pending.at <= SEQUENCE_MS ? pending : null;
  pending = null;

  if (live) {
    // A pending 'g' sequence wins over a single-key shortcut on its second key.
    const seqMatches = entries.filter((e) => e.steps.some((steps) => steps.length === 2
      && stepMatches(steps[0], live.event) && stepMatches(steps[1], event)) && allowed(e, event, target));
    const chosen = pick(seqMatches);
    if (chosen) {
      event.preventDefault();
      run(chosen, event);
      return;
    }
  }

  const prefixes = entries.filter((e) => e.steps.some((steps) => steps.length === 2 && stepMatches(steps[0], event)) && allowed(e, event, target));
  const singles = entries.filter((e) => e.steps.some((steps) => steps.length === 1 && stepMatches(steps[0], event)) && allowed(e, event, target));
  if (prefixes.length && !singles.length) {
    const altGraph = altGraphDown(event);
    pending = {
      at: now,
      event: {
        key: event.key, ctrlKey: event.ctrlKey, metaKey: event.metaKey, altKey: event.altKey, shiftKey: event.shiftKey,
        getModifierState: (name) => name === 'AltGraph' && altGraph,
      },
    };
    return;
  }
  const key = normaliseKeyName(event.key);
  const plain = !event.ctrlKey && !event.metaKey && !event.altKey;
  if (plain && (key === 'Space' || key === 'Enter') && target && target.closest && target.closest(ACTIVATABLE)) return;
  const chosen = pick(singles);
  if (!chosen) return;
  event.preventDefault();
  run(chosen, event);
}

function run(entry, event) {
  try {
    const result = entry.handler(event);
    if (result && typeof result.then === 'function') result.then(null, (error) => { if (globalThis.console) globalThis.console.error(error); });
  } catch (error) {
    if (globalThis.console) globalThis.console.error(error);
  }
}

/**
 * Register a shortcut.
 * keys: ['?'], ['Space', 'k'], ['Shift+ArrowLeft'], ['Mod+Enter'] (Mod = Ctrl, or Cmd on Apple), ['g r'] (a
 * sequence within 1 s). scope: 'global' | 'player' (while a session exists) | 'view:<id>' (while that view
 * is shown) | 'dialog:<id>' (only while that dialog is the top sheet). A shortcut never fires when the event
 * was defaultPrevented, while a modal sheet is open (unless it is that sheet's), for single-key shortcuts
 * while prefs.shortcuts === false, or while focus is in an editable control unless allowInEditable (allowed
 * only for Mod/Alt chords and Escape). Space and Enter never fire on a focused button, link or native media
 * player (<audio controls>). Character keys are always single-key shortcuts.
 * @returns {() => void} off
 */
export function register({
  id, keys, scope = 'global', singleKey = false, allowInEditable = false, description = '',
  when = () => true, handler,
} = {}) {
  if (!Array.isArray(keys) || !keys.length) throw new TypeError('register() needs keys');
  if (typeof handler !== 'function') throw new TypeError('register() needs a handler');
  if (!(scope === 'global' || scope === 'player' || /^view:[a-z]+$/.test(scope) || /^dialog:[\w-]+$/.test(scope))) {
    throw new TypeError(`Unknown shortcut scope: ${scope}`);
  }
  const steps = keys.map(parseCombo);
  const character = steps.some((s) => s.some((step) => isCharacter(step) && !step.mod && !step.ctrl && !step.meta && !step.alt));
  if (allowInEditable) {
    const ok = steps.every((s) => s.length === 1 && (s[0].mod || s[0].ctrl || s[0].meta || s[0].alt || s[0].key === 'Escape'));
    if (!ok) throw new TypeError('allowInEditable is only for Mod or Alt chords and Escape');
  }
  const entry = {
    id: id || `shortcut-${seq + 1}`, keys: keys.slice(), steps, scope, singleKey: !!singleKey || character,
    allowInEditable: !!allowInEditable, description, when, handler, order: (seq += 1),
  };
  entries.push(entry);
  if (!bound && typeof document !== 'undefined') {
    bound = onKeydown;
    document.addEventListener('keydown', bound);
  }
  return () => {
    entries = entries.filter((e) => e !== entry);
  };
}

/** @returns {{id: string, keys: string[], description: string, scope: string, singleKey: boolean}[]} for the "?" sheet */
export function list() {
  return entries.map(({ id, keys, description, scope, singleKey }) => ({ id, keys: keys.slice(), description, scope, singleKey }));
}

/** @returns {string} 'Ctrl' or '⌘' for Mod in labels */
export function modLabel() {
  return isApple() ? '⌘' : 'Ctrl';
}

/** Test hook. */
export function __resetForTests() {
  if (bound && typeof document !== 'undefined') {
    try { document.removeEventListener('keydown', bound); } catch { /* old env */ }
  }
  bound = null;
  entries = [];
  seq = 0;
  pending = null;
}
