/**
 * Input level meter: bars plus a visible text label, so the level is never shown by
 * colour alone. Zones: quiet below -40 dBFS, good -40 to -12, loud above -12, clip when peak >= 0.999.
 * The bars update at most every 66 ms (15 Hz) and the label at most every 250 ms; nothing flashes.
 * @module components/meter
 */
import { html, render } from '../core/dom.js';

export const LABELS = Object.freeze({ idle: '', quiet: 'Too quiet', good: 'Good level', loud: 'Too loud', clip: 'Clipping' });
const BAR_MS = 66;
const LABEL_MS = 250;
const FLOOR_DB = -60;

/**
 * PURE: the zone for a level.
 * @param {{db?: number, peak?: number}} level
 * @returns {'quiet'|'good'|'loud'|'clip'}
 */
export function zoneFor({ db = -Infinity, peak = 0 } = {}) {
  if (Number(peak) >= 0.999) return 'clip';
  if (db > -12) return 'loud';
  if (db >= -40) return 'good';
  return 'quiet';
}

/**
 * PURE: the 0–100 position of a dBFS level on the meter (-60 dBFS and below is 0).
 * @param {number} db
 */
export function levelPercent(db) {
  if (!Number.isFinite(db)) return 0;
  return Math.round(Math.min(1, Math.max(0, (db - FLOOR_DB) / -FLOOR_DB)) * 100);
}

/**
 * Turn an element (role=meter) into a bar meter. The text label is the next sibling with class
 * meter__label, or a new one inserted after it.
 * @param {HTMLElement} el
 * @param {{bars?: number}} [options]
 * @returns {{set: (level: {db: number, peak?: number}) => void, reset: () => void, destroy: () => void}}
 */
export function createLevelMeter(el, { bars = 9 } = {}) {
  render(el, html`${Array.from({ length: bars }, (_, i) => html`<span class="meter__bar" data-i="${i}"></span>`)}`);
  if (!el.getAttribute('role')) el.setAttribute('role', 'meter');
  if (!el.hasAttribute('aria-valuemin')) el.setAttribute('aria-valuemin', '0');
  if (!el.hasAttribute('aria-valuemax')) el.setAttribute('aria-valuemax', '100');
  let label = el.nextElementSibling && el.nextElementSibling.classList.contains('meter__label') ? el.nextElementSibling : null;
  if (!label) {
    label = document.createElement('span');
    label.className = 'meter__label';
    el.after(label);
  }
  const barEls = Array.from(el.querySelectorAll('.meter__bar'));
  let lastBars = -Infinity;
  let lastLabel = -Infinity;
  let pending = null;
  let timer = null;
  let shownLit = -1;
  let shownZone = '';

  function paintBars(level) {
    const pct = levelPercent(level.db);
    const zone = zoneFor(level);
    const lit = Math.round((pct / 100) * bars);
    if (lit !== shownLit) {
      barEls.forEach((bar, i) => bar.classList.toggle('is-on', i < lit));
      shownLit = lit;
    }
    if (zone !== shownZone) {
      el.setAttribute('data-zone', zone);
      shownZone = zone;
    }
    el.setAttribute('aria-valuenow', String(pct));
    lastBars = Date.now();
  }

  function paintLabel(level) {
    const text = LABELS[zoneFor(level)];
    if (label.textContent !== text) label.textContent = text;
    el.setAttribute('aria-valuetext', text);
    lastLabel = Date.now();
  }

  function barsDiffer(level) {
    return Math.round((levelPercent(level.db) / 100) * bars) !== shownLit || zoneFor(level) !== shownZone;
  }

  function labelDiffers(level) {
    return label.textContent !== LABELS[zoneFor(level)];
  }

  function flush() {
    timer = null;
    if (!pending) return;
    const level = pending;
    const now = Date.now();
    let waitBars = lastBars + BAR_MS - now;
    let waitLabel = lastLabel + LABEL_MS - now;
    if (waitBars <= 0) { paintBars(level); waitBars = Infinity; }
    if (waitLabel <= 0) { paintLabel(level); waitLabel = Infinity; }
    // Whatever could not show this level yet gets a trailing update, so the last level always shows.
    const barsLate = waitBars !== Infinity && barsDiffer(level);
    const labelLate = waitLabel !== Infinity && labelDiffers(level);
    if (barsLate || labelLate) {
      timer = setTimeout(flush, Math.max(1, Math.min(barsLate ? waitBars : Infinity, labelLate ? waitLabel : Infinity)));
    } else {
      pending = null;
    }
  }

  return {
    set(level) {
      pending = { db: Number(level?.db), peak: Number(level?.peak) || 0 };
      if (timer === null) flush();
    },
    reset() {
      clearTimeout(timer);
      timer = null;
      pending = null;
      barEls.forEach((bar) => bar.classList.remove('is-on'));
      shownLit = 0;
      shownZone = 'idle';
      el.setAttribute('data-zone', 'idle');
      el.setAttribute('aria-valuenow', '0');
      el.removeAttribute('aria-valuetext');
      label.textContent = LABELS.idle;
      lastBars = -Infinity;
      lastLabel = -Infinity;
    },
    destroy() {
      clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}
