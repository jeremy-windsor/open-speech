/**
 * The sleep sheet #rdp-sleep-sheet: a native radio group (Off, 5–60 min, End of chapter,
 * End of [N] chapters with a 2–10 stepper), the "Fade out" switch (prefs.sleepFade) and the time left, shown
 * live in plain text (never in a live region, so nothing ticks for screen readers). A choice applies at once;
 * the change is announced once.
 * @module views/reader/sleep-sheet
 */
import { openSheet, closeSheet } from '../../core/dialogs.js';
import { announce } from '../../core/a11y.js';
import * as prefs from '../../core/prefs.js';
import { MIN_CHAPTERS, MAX_CHAPTERS, optionValue, specFor, remainingText, spokenRemaining, start } from './sleep-timer.js';

/**
 * @param {HTMLElement} root the player slot
 * @param {object} engine
 * @returns {{open: (trigger?: Element) => void, close: () => void, paint: (state: object) => void, isOpen: () => boolean, destroy: () => void}}
 */
export function bindSleepSheet(root, engine) {
  const $ = (id) => root.querySelector(`#${id}`);
  const dialog = $('rdp-sleep-sheet');
  const options = $('rdp-sleep-options');
  const remaining = $('rdp-sleep-remaining');
  const count = $('rdp-sleep-n');
  const fewer = $('rdp-sleep-n-dec');
  const more = $('rdp-sleep-n-inc');
  const fade = $('rdp-sleep-fade');
  const radios = () => Array.from(options.querySelectorAll('input[name="rdp-sleep"]'));
  let chapters = MIN_CHAPTERS;
  const offs = [];
  const listen = (el, type, fn) => {
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  function apply(value) {
    const sleep = engine.setSleep(specFor(value, chapters)) || start(specFor(value, chapters));
    announce(spokenRemaining(sleep));
  }

  listen(options, 'change', (event) => {
    const input = event.target && event.target.closest ? event.target.closest('input[name="rdp-sleep"]') : null;
    if (input && input.checked) apply(input.value);
  });

  function setCount(n) {
    chapters = Math.min(MAX_CHAPTERS, Math.max(MIN_CHAPTERS, n));
    count.textContent = String(chapters);
    fewer.setAttribute('aria-disabled', chapters <= MIN_CHAPTERS ? 'true' : 'false');
    more.setAttribute('aria-disabled', chapters >= MAX_CHAPTERS ? 'true' : 'false');
    fewer.setAttribute('aria-label', `Fewer chapters (now ${chapters})`);
    more.setAttribute('aria-label', `More chapters (now ${chapters})`);
  }

  const stepCount = (delta, button) => {
    if (button.getAttribute('aria-disabled') === 'true') return;
    setCount(chapters + delta);
    const chosen = radios().find((r) => r.value === 'chapters');
    if (chosen) {
      // The stepper also picks "End of N chapters" (the listener clearly means it).
      chosen.checked = true;
      apply('chapters');
    }
  };
  listen(fewer, 'click', () => stepCount(-1, fewer));
  listen(more, 'click', () => stepCount(1, more));
  listen(fade, 'change', () => { prefs.set({ sleepFade: !!fade.checked }); });

  function paint(state) {
    const sleep = state?.sleep || { mode: 'off' };
    const value = optionValue(sleep);
    if (sleep.mode === 'chapters' && sleep.chapters > 1 && sleep.chapters !== chapters) setCount(sleep.chapters);
    for (const radio of radios()) {
      const on = radio.value === value;
      if (radio.checked !== on) radio.checked = on;
    }
    const text = remainingText(sleep, sleep.remainingMs);
    if (remaining.textContent !== text) remaining.textContent = text;
    const mode = sleep.mode || 'off';   // paint runs every frame while open: write only a change
    if (remaining.getAttribute('data-mode') !== mode) remaining.setAttribute('data-mode', mode);
    const wanted = prefs.get().sleepFade !== false;
    if (fade.checked !== wanted) fade.checked = wanted;
  }

  setCount(chapters);
  return {
    open(trigger) {
      paint(engine.state);
      const checked = radios().find((r) => r.checked) || radios()[0];
      openSheet(dialog, { trigger: trigger || undefined, initialFocus: checked });
    },
    close() { if (dialog.open) closeSheet(dialog); },
    isOpen: () => !!dialog.open,
    paint,
    destroy() { for (const off of offs.splice(0)) off(); },
  };
}
