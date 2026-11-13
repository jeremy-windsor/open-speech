/**
 * Settings: the pure parts. No DOM, no storage, no requests.
 * @module views/settings/logic
 */
import { READING_DEFAULTS } from '../../core/prefs.js';

/** Reading speeds (the Reader's speed sheet uses the same range and presets). */
export const MIN_SPEED = 0.5;
export const MAX_SPEED = 3.5;
export const SPEED_STEP = 0.05;
export const SPEED_PRESETS = Object.freeze([0.75, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]);

/** A speed rounded to 0.05 within 0.5–3.5; null for anything that isn't a number. */
export function clampSpeed(value) {
  const n = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return null;
  const rounded = Math.round(n / SPEED_STEP) * SPEED_STEP;
  return Number(Math.min(MAX_SPEED, Math.max(MIN_SPEED, rounded)).toFixed(2));
}

/** The stored default reading speed (KEYS.readerSpeed): a missing or out-of-range value reads as 1. */
export function storedSpeed(text) {
  const n = Number(text);
  if (text === null || text === undefined || text === '' || !Number.isFinite(n) || n < MIN_SPEED || n > MAX_SPEED) return 1;
  return clampSpeed(n);
}

/** The preset index for a speed, or -1 between presets (no chip is checked then). */
export function presetIndex(speed) {
  return SPEED_PRESETS.indexOf(clampSpeed(speed));
}

/** Reading text sizes in px for data-rd-size 0…6. */
export const READING_SIZES = Object.freeze([15, 17, 19, 20, 22, 25, 28]);
const FACE_WORDS = { serif: 'Serif', sans: 'Sans', legible: 'Legible' };
const MEASURE_WORDS = { narrow: 'Narrow width', medium: 'Medium width', wide: 'Wide width' };
const PAPER_WORDS = { auto: 'Match app', paper: 'Paper', sepia: 'Sepia', night: 'Night', dusk: 'Dusk' };

/**
 * PURE: the reading-appearance summary, e.g. "Serif · 19 px · Medium width · Match app". With no size of
 * its own the Reader uses 19 px below 960 px and 20 px from 960 px.
 * @param {object} reading prefs.getReading()
 * @param {{desktop?: boolean}} [options]
 */
export function readingSummary(reading, { desktop = false } = {}) {
  const r = { ...READING_DEFAULTS, ...(reading || {}) };
  const step = Number.isInteger(r.size) && r.size >= 0 && r.size < READING_SIZES.length ? r.size : (desktop ? 3 : 2);
  return [FACE_WORDS[r.face] || FACE_WORDS.serif, `${READING_SIZES[step]} px`, MEASURE_WORDS[r.measure] || MEASURE_WORDS.medium, PAPER_WORDS[r.paper] || PAPER_WORDS.auto].join(' · ');
}

/** What #se-key-status says after Test (or a save). */
export const KEY_COPY = Object.freeze({
  accepted: 'Connected. The server accepted the key.',
  open: "This server doesn't require a key.",
  refused: "The server didn't accept the key.",
  needed: 'This server needs an API key.',
  unsendable: "The key can't be sent: it contains a character such as a curly quote.",
  offline: "Can't reach the server.",
  retrying: 'Trying the key again…',
});

/**
 * PURE: the status line for the answer to GET /api/models.
 * @param {{ok: boolean, error?: object, hasKey: boolean, sendable?: boolean}} outcome
 * @returns {{text: string, tone: ''|'ok'|'error', kind: 'accepted'|'open'|'unsendable'|'locked'|'refused'|'needed'|'offline'|'error'}}
 *   `kind` says which answer it is (the line follows later connection changes by it)
 */
export function keyCheckMessage({ ok, error = null, hasKey = false, sendable = true } = {}) {
  if (ok) return hasKey ? { text: KEY_COPY.accepted, tone: 'ok', kind: 'accepted' } : { text: KEY_COPY.open, tone: 'ok', kind: 'open' };
  const e = error || {};
  if (hasKey && !sendable) return { text: KEY_COPY.unsendable, tone: 'error', kind: 'unsendable' };
  if (e.kind === 'locked') {
    const wait = Number.isFinite(e.retryAfter) && e.retryAfter !== null ? `${e.retryAfter} s` : 'a while';
    return { text: `Locked out for ${wait} after too many wrong keys.`, tone: 'error', kind: 'locked' };
  }
  if (e.kind === 'auth' || e.status === 401) return hasKey ? { text: KEY_COPY.refused, tone: 'error', kind: 'refused' } : { text: KEY_COPY.needed, tone: 'error', kind: 'needed' };
  if (e.kind === 'network' || e.kind === 'timeout') return { text: KEY_COPY.offline, tone: 'error', kind: 'offline' };
  const detail = String(e.message || '').trim();
  return { text: `The server answered with an error${detail ? `: ${detail.replace(/\.$/, '')}` : ''}.`, tone: 'error', kind: 'error' };
}

/** The connection state in words (the status sheet uses the same words). */
export const STATE_WORDS = Object.freeze({
  connecting: 'Connecting…', online: 'Connected', offline: "Can't reach the server", unauthorized: 'API key needed', locked: 'Locked out',
});

/** The in-page sections, in order (their h2 ids are se-<name>-title). */
export const SECTIONS = Object.freeze(['appearance', 'playback', 'connection', 'server', 'assistant', 'history', 'shortcuts', 'developer', 'about']);
