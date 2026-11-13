/**
 * The speed sheet #rdp-speed-sheet: preset chips in a radiogroup with roving focus (0.75 … 3×),
 * a fine slider 0.5–3.5× in 0.05 steps with steppers (a drag-free path, WCAG 2.5.7), Reset to 1×, and the
 * listening time left in the document at the chosen speed. Every change applies at once through
 * engine.setSpeed(), which keeps it per document and as the default for new ones.
 * @module views/reader/speed-sheet
 */
import { openSheet, closeSheet } from '../../core/dialogs.js';
import { speed as speedText, spokenSpeed } from '../../core/format.js';
import { bindRoving } from '../../components/roving.js';
import { bindRange } from '../../components/range.js';
import { MIN_SPEED, MAX_SPEED, SPEED_STEP, SPEED_PRESETS, clampSpeed } from './play-logic.js';
import { leftLabel } from './timeline.js';

/** The preset index for a speed, or -1 when the speed sits between presets (no radio is checked then). */
export function presetIndex(speed) {
  const s = clampSpeed(speed);
  return SPEED_PRESETS.findIndex((p) => p === s);
}

/** One 0.05 step slower or faster, clamped to 0.5–3.5. */
export function stepSpeed(speed, delta) {
  return clampSpeed((Number(speed) || 1) + delta * SPEED_STEP);
}

/**
 * Bind the sheet inside the player root.
 * @param {HTMLElement} root the player slot
 * @param {object} engine
 * @returns {{open: (trigger?: Element) => void, close: () => void, paint: (state: object) => void, isOpen: () => boolean, destroy: () => void}}
 */
export function bindSpeedSheet(root, engine) {
  const $ = (id) => root.querySelector(`#${id}`);
  const dialog = $('rdp-speed-sheet');
  const presets = $('rdp-speed-presets');
  const range = $('rdp-speed-range');
  const value = $('rdp-speed-value');
  const left = $('rdp-speed-left');
  const dec = $('rdp-speed-dec');
  const inc = $('rdp-speed-inc');
  const reset = $('rdp-speed-reset');
  const offs = [];
  const listen = (el, type, fn) => {
    el.addEventListener(type, fn);
    offs.push(() => el.removeEventListener(type, fn));
  };

  const roving = bindRoving(presets, {
    items: '[role=radio]',
    orientation: 'both',
    onChange: (item) => engine.setSpeed(Number(item.getAttribute('data-speed'))),
  });
  const slider = bindRange(range, { format: speedText, spoken: spokenSpeed });
  listen(range, 'input', () => engine.setSpeed(Number(range.value)));
  // At 0.5× and 3.5× the steppers are aria-disabled, not disabled, so keyboard focus stays on them.
  const usable = (el) => el.getAttribute('aria-disabled') !== 'true';
  listen(dec, 'click', () => { if (usable(dec)) engine.setSpeed(stepSpeed(engine.state.speed, -1)); });
  listen(inc, 'click', () => { if (usable(inc)) engine.setSpeed(stepSpeed(engine.state.speed, 1)); });
  listen(reset, 'click', () => engine.setSpeed(1));

  const setAttr = (el, name, text) => { if (el.getAttribute(name) !== text) el.setAttribute(name, text); };
  let paintedIndex = null;   // the roving tab stop moves only when the checked preset changes

  // Runs on every frame while the sheet is open and the Reader plays: unchanged values write nothing.
  function paint(state) {
    const s = Number(state?.speed) || 1;
    const text = speedText(s);
    if (value.textContent !== text) value.textContent = text;
    if (Number(range.value) !== s) slider.set(s);
    const index = presetIndex(s);
    if (index !== paintedIndex) {
      paintedIndex = index;
      presets.querySelectorAll('[role=radio]').forEach((item, i) => setAttr(item, 'aria-checked', i === index ? 'true' : 'false'));
      roving.refresh();
    }
    setAttr(dec, 'aria-disabled', s <= MIN_SPEED ? 'true' : 'false');
    setAttr(inc, 'aria-disabled', s >= MAX_SPEED ? 'true' : 'false');
    const line = state?.docId && state.remainingMs > 0 ? leftLabel(state.remainingMs, s) : '';
    if (left.textContent !== line) left.textContent = line;
  }

  return {
    open(trigger) {
      paintedIndex = null;
      paint(engine.state);
      const checked = presets.querySelector('[role=radio][aria-checked="true"]');
      openSheet(dialog, { trigger: trigger || undefined, initialFocus: checked || range });
    },
    close() { if (dialog.open) closeSheet(dialog); },
    isOpen: () => !!dialog.open,
    paint,
    destroy() {
      for (const off of offs.splice(0)) off();
      roving.destroy();
      slider.destroy();
    },
  };
}
