/**
 * Reading appearance: the sheet `#rd-appearance-sheet` and the attributes it sets on
 * `#rd-reading` (`data-rd-paper`, `-size`, `-face`, `-leading`, `-measure`, `-justify`, `-hl`). Every control
 * applies at once and persists through prefs.setReading (KEYS.reading); Reset restores the defaults.
 *
 * The spoken sentence is marked with a tint plus an underline by default (`highlight: 'both'`), so it is never
 * marked by colour alone (WCAG 1.4.1, REV-35). The sheet's backdrop is a light scrim, so the text stays
 * visible while it changes.
 * @module views/reader/appearance
 */
import { app } from '../../core/store.js';
import * as prefs from '../../core/prefs.js';
import { openSheet, closeSheet } from '../../core/dialogs.js';
import { SIZES_PX, sizeStep, sizeLabel } from './text-logic.js';

const GROUPS = Object.freeze({ 'rd-face': 'face', 'rd-leading': 'leading', 'rd-measure': 'measure', 'rd-paper': 'paper', 'rd-hl': 'highlight' });

/**
 * PURE: the attributes for #rd-reading from a reading-appearance object (prefs.READING_DEFAULTS shape).
 * `size` is null for the responsive default (no data-rd-size attribute).
 * @returns {{paper: string, size: string|null, face: string, leading: string, measure: string, justify: string, hl: string}}
 */
export function readingAttrs(reading) {
  const r = { ...prefs.READING_DEFAULTS, ...(reading || {}) };
  return {
    paper: r.paper,
    size: Number.isInteger(r.size) ? String(r.size) : null,
    face: r.face,
    leading: r.leading,
    measure: r.measure,
    justify: r.justify ? 'true' : 'false',
    hl: r.highlight,
  };
}

function isDesktop() {
  try {
    return typeof matchMedia === 'function' && matchMedia('(min-width: 960px)').matches;
  } catch {
    return false;
  }
}

/**
 * Apply a reading appearance to the reading surface.
 * @param {HTMLElement} el #rd-reading
 * @param {object} reading
 */
export function applyReading(el, reading) {
  if (!el) return;
  const a = readingAttrs(reading);
  el.setAttribute('data-rd-paper', a.paper);
  if (a.size === null) el.removeAttribute('data-rd-size');
  else el.setAttribute('data-rd-size', a.size);
  el.setAttribute('data-rd-face', a.face);
  el.setAttribute('data-rd-leading', a.leading);
  el.setAttribute('data-rd-measure', a.measure);
  el.setAttribute('data-rd-justify', a.justify);
  el.setAttribute('data-rd-hl', a.hl);
}

/**
 * Bind the appearance sheet inside the Reader's root.
 * @param {HTMLElement} root
 * @param {{onChange?: (reading: object) => void}} [options]
 */
export function createAppearance(root, { onChange = () => {} } = {}) {
  const reading = root.querySelector('#rd-reading');
  const sheet = root.querySelector('#rd-appearance-sheet');
  const dec = root.querySelector('#rd-size-dec');
  const inc = root.querySelector('#rd-size-inc');
  const value = root.querySelector('#rd-size-value');
  const autoscroll = root.querySelector('#rd-autoscroll');
  const justify = root.querySelector('#rd-justify');
  const reset = root.querySelector('#rd-appearance-reset');
  const offs = [];

  function step() {
    return sizeStep(prefs.getReading().size, isDesktop());
  }

  /** Bring the controls in line with the saved appearance (it can change from Settings too). */
  function syncControls(r = prefs.getReading()) {
    for (const [name, field] of Object.entries(GROUPS)) {
      for (const input of sheet.querySelectorAll(`input[name="${name}"]`)) input.checked = input.value === r[field];
    }
    if (autoscroll) autoscroll.checked = !!r.autoScroll;
    if (justify) justify.checked = !!r.justify;
    const s = sizeStep(r.size, isDesktop());
    if (value) value.textContent = sizeLabel(s);
    if (dec) dec.setAttribute('aria-disabled', s <= 0 ? 'true' : 'false');
    if (inc) inc.setAttribute('aria-disabled', s >= SIZES_PX.length - 1 ? 'true' : 'false');
  }

  function apply(r = prefs.getReading()) {
    applyReading(reading, r);
    syncControls(r);
  }

  function save(patch) {
    return prefs.setReading(patch);   // app.reading changes: the subscription below applies it
  }

  function onInput(event) {
    const target = event.target;
    if (!target || !target.matches) return;
    if (target.matches('input[type="radio"]') && GROUPS[target.name] && target.checked) {
      save({ [GROUPS[target.name]]: target.value });
    } else if (target === autoscroll) {
      save({ autoScroll: !!autoscroll.checked });
    } else if (target === justify) {
      save({ justify: !!justify.checked });
    }
  }

  function resize(delta) {
    const next = Math.max(0, Math.min(SIZES_PX.length - 1, step() + delta));
    if (next === step() && prefs.getReading().size !== null) return;
    save({ size: next });
  }

  const onDec = (event) => {
    if (dec.getAttribute('aria-disabled') === 'true') { event.preventDefault(); return; }
    resize(-1);
  };
  const onInc = (event) => {
    if (inc.getAttribute('aria-disabled') === 'true') { event.preventDefault(); return; }
    resize(1);
  };
  const onReset = () => { prefs.resetReading(); };

  sheet.addEventListener('change', onInput);
  dec.addEventListener('click', onDec);
  inc.addEventListener('click', onInc);
  reset.addEventListener('click', onReset);
  offs.push(() => {
    sheet.removeEventListener('change', onInput);
    dec.removeEventListener('click', onDec);
    inc.removeEventListener('click', onInc);
    reset.removeEventListener('click', onReset);
  });
  offs.push(app.select((s) => s.reading, (r) => {
    apply(r);
    try { onChange(r); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }));
  // The responsive default size differs below and above 960 px; keep the "19 px" label honest.
  try {
    if (typeof matchMedia === 'function') {
      const mq = matchMedia('(min-width: 960px)');
      const onMq = () => syncControls();
      if (mq.addEventListener) mq.addEventListener('change', onMq);
      offs.push(() => { if (mq.removeEventListener) mq.removeEventListener('change', onMq); });
    }
  } catch { /* no matchMedia */ }

  apply();

  return {
    apply,
    /** Open the sheet over the reading view (the text stays visible behind its light scrim). */
    open(trigger) {
      if (!sheet || sheet.open) return;
      syncControls();
      openSheet(sheet, { trigger: trigger || document.activeElement });
    },
    close() {
      if (sheet && sheet.open) closeSheet(sheet);
    },
    get isOpen() { return !!(sheet && sheet.open); },
    autoScroll: () => !!prefs.getReading().autoScroll,
    destroy() {
      for (const off of offs.splice(0)) off();
    },
  };
}
