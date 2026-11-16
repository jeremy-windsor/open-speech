/**
 * Settings › Playback: the default reading speed (KEYS.readerSpeed, for documents with
 * no speed of their own), the skip intervals, what previous/next move by in the app and on the lock screen,
 * and the switches for sentence snapping, the sleep-timer fade and the wake lock. Every control applies at
 * once; the Reader reads these when it plays.
 * @module views/settings/playback
 */
import { setHidden } from '../../core/dom.js';
import { KEYS, read, write } from '../../core/storage.js';
import * as prefs from '../../core/prefs.js';
import { speed as speedText, spokenSpeed } from '../../core/format.js';
import { bindRoving } from '../../components/roving.js';
import { bindRange } from '../../components/range.js';
import { MIN_SPEED, MAX_SPEED, SPEED_STEP, clampSpeed, storedSpeed, presetIndex } from './logic.js';

/** Segmented controls bound to a preference: input name -> [pref field, parse]. */
export const PREF_SEGS = Object.freeze({
  'se-skip-back': ['skipBack', Number],
  'se-skip-forward': ['skipForward', Number],
  'se-step': ['stepUnit', String],
  'se-track': ['trackUnit', String],
  'se-start': ['startOn', String],   // Appearance › Open on
});

/** Switches bound to a preference: input id -> pref field. */
export const PREF_SWITCHES = Object.freeze({
  'se-snap': 'snapToSentence',
  'se-fade': 'sleepFade',
  'se-wake': 'wakeLock',
  'se-shortcuts': 'shortcuts',
  'se-show-live': 'showLive',   // Developer › Show Speak › Live
});

/** Whether this browser can keep the screen on (the Screen Wake Lock API; it needs a secure page). */
export function wakeLockSupported() {
  try {
    return typeof navigator !== 'undefined' && !!navigator.wakeLock && typeof navigator.wakeLock.request === 'function';
  } catch {
    return false;
  }
}

/**
 * Bind the Playback controls and the preference switches.
 * @param {HTMLElement} root the view section
 * @returns {{sync: () => void, syncSpeed: () => void, destroy: () => void}}
 */
export function bindPlayback(root) {
  const $ = (id) => root.querySelector(`#${id}`);
  const presets = $('se-speed-presets');
  const range = $('se-speed');
  const output = $('se-speed-value');
  const dec = $('se-speed-dec');
  const inc = $('se-speed-inc');
  const wakeHint = $('se-wake-hint');
  const offs = [];
  const listen = (el, type, fn) => {
    if (!el) return;
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  const roving = bindRoving(presets, {
    items: '[role=radio]',
    orientation: 'both',
    onChange: (item) => setSpeed(Number(item.getAttribute('data-speed'))),
  });
  const slider = bindRange(range, { output, format: speedText, spoken: spokenSpeed });

  function paintSpeed(value) {
    const s = clampSpeed(value) ?? 1;
    if (Number(range.value) !== s) slider.set(s);
    else slider.update();
    const index = presetIndex(s);
    presets.querySelectorAll('[role=radio]').forEach((chip, i) => {
      const checked = i === index ? 'true' : 'false';
      if (chip.getAttribute('aria-checked') !== checked) chip.setAttribute('aria-checked', checked);
    });
    roving.refresh();
    dec.setAttribute('aria-disabled', s <= MIN_SPEED ? 'true' : 'false');
    inc.setAttribute('aria-disabled', s >= MAX_SPEED ? 'true' : 'false');
  }

  function setSpeed(value) {
    const s = clampSpeed(value);
    if (s === null) return;
    write(KEYS.readerSpeed, String(s));
    paintSpeed(s);
  }

  listen(range, 'input', () => setSpeed(Number(range.value)));
  // At 0.5× and 3.5× the steppers are aria-disabled, not disabled, so keyboard focus stays on them.
  const usable = (el) => el.getAttribute('aria-disabled') !== 'true';
  const current = () => storedSpeed(read(KEYS.readerSpeed, null));
  listen(dec, 'click', () => { if (usable(dec)) setSpeed(current() - SPEED_STEP); });
  listen(inc, 'click', () => { if (usable(inc)) setSpeed(current() + SPEED_STEP); });

  listen(root, 'change', (event) => {
    const input = event.target;
    if (!input || input.localName !== 'input') return;
    if (input.type === 'radio' && PREF_SEGS[input.name] && input.checked) {
      const [field, parse] = PREF_SEGS[input.name];
      prefs.set({ [field]: parse(input.value) });
    } else if (input.type === 'checkbox' && PREF_SWITCHES[input.id]) {
      prefs.set({ [PREF_SWITCHES[input.id]]: !!input.checked });
    }
  });

  function syncPrefs() {
    const p = prefs.get();
    for (const [name, [field]] of Object.entries(PREF_SEGS)) {
      for (const input of root.querySelectorAll(`input[name="${name}"]`)) {
        const want = String(input.value) === String(p[field]);
        if (input.checked !== want) input.checked = want;
      }
    }
    for (const [id, field] of Object.entries(PREF_SWITCHES)) {
      const input = $(id);
      if (input && input.checked !== !!p[field]) input.checked = !!p[field];
    }
    if (wakeHint) {
      const supported = wakeLockSupported();
      wakeHint.textContent = supported ? 'Only while the Reader is open and playing.' : "This browser can't keep the screen on (it needs a secure connection), so this has no effect here.";
      setHidden(wakeHint, false);
    }
  }

  // Another tab changed the default speed.
  if (typeof window !== 'undefined') {
    listen(window, 'storage', (event) => {
      if (event && event.key === KEYS.readerSpeed) paintSpeed(current());
    });
  }

  return {
    sync() {
      paintSpeed(current());
      syncPrefs();
    },
    /** Repaint the default speed from storage (the Reader changed it). */
    syncSpeed() {
      paintSpeed(current());
    },
    destroy() {
      for (const off of offs.splice(0)) off();
      roving.destroy();
      slider.destroy();
    },
  };
}
