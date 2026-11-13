/**
 * Formatting helpers (pure). English only (the UI is lang="en").
 * @module core/format
 */

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const CHARS_PER_SECOND = 15;   // a rough 1× speaking rate, for length estimates only

const whole = (n) => (Number.isFinite(n) && n > 0 ? Math.floor(n) : 0);

/** '4:03', '1:02:03' (negative or invalid -> '0:00'). */
export function clock(seconds) {
  const total = whole(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

/** '4 minutes 3 seconds', '1 hour 2 minutes', '0 seconds'. */
export function spokenDuration(seconds) {
  const total = whole(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const parts = [];
  if (h) parts.push(plural(h, 'hour'));
  if (m) parts.push(plural(m, 'minute'));
  if (s || !parts.length) parts.push(plural(s, 'second'));
  return parts.join(' ');
}

function trimNumber(x, decimals = 2) {
  const n = Number(x);
  if (!Number.isFinite(n)) return '0';
  return String(Number(n.toFixed(decimals)));
}

/** '1.25×', '1×', '0.75×' (at most 2 decimals). */
export function speed(x) {
  return `${trimNumber(x)}×`;
}

/** '1.25 times' (for aria-valuetext and labels). */
export function spokenSpeed(x) {
  return `${trimNumber(x)} times`;
}

/** Today's formatSize: '82 MB', '1.2 GB'; '' for nothing. */
export function sizeMb(mb) {
  const n = Number(mb);
  if (!n) return '';
  if (n >= 1000) return `${(n / 1000).toFixed(1)} GB`;
  return `${Math.round(n)} MB`;
}

/** Bytes in decimal units: '820 B', '14 KB', '1.4 MB', '2.1 GB'. */
export function bytes(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v < 0) return '';
  if (v < 1000) return `${Math.round(v)} B`;
  if (v < 1e6) return `${Math.round(v / 1e3)} KB`;
  if (v < 1e9) return `${(v / 1e6).toFixed(1)} MB`;
  return `${(v / 1e9).toFixed(1)} GB`;
}

/** Milliseconds from a Date, an ISO string, or a number in seconds (< 1e12) or milliseconds. */
export function toMs(value) {
  if (value instanceof Date) return value.getTime();
  if (typeof value === 'number') return value < 1e12 ? value * 1000 : value;
  const parsed = Date.parse(String(value || ''));
  return Number.isFinite(parsed) ? parsed : 0;
}

/** 'just now', '5 min ago', '3 h ago', 'Yesterday', 'Mar 3', 'Mar 3, 2025'. */
export function relativeTime(date, now = Date.now()) {
  const at = toMs(date);
  if (!at) return '';
  const diff = now - at;
  if (diff < 45000) return 'just now';
  if (diff < 3600000) return `${Math.max(1, Math.round(diff / 60000))} min ago`;
  const then = new Date(at);
  const today = new Date(now);
  const startOfToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  if (at >= startOfToday) return `${Math.max(1, Math.floor(diff / 3600000))} h ago`;
  if (at >= startOfToday - 86400000) return 'Yesterday';
  const label = `${MONTHS[then.getMonth()]} ${then.getDate()}`;
  return then.getFullYear() === today.getFullYear() ? label : `${label}, ${then.getFullYear()}`;
}

/** The full local date and time, for title attributes and screen readers. */
export function absoluteTime(date) {
  const at = toMs(date);
  if (!at) return '';
  const d = new Date(at);
  try {
    return d.toLocaleString('en', { dateStyle: 'medium', timeStyle: 'short' });
  } catch {
    return d.toLocaleString();
  }
}

/** '1 chapter', '3 chapters', '1,204 chars' (pluralWord for irregular plurals). */
export function plural(n, word, pluralWord) {
  const count = Number(n) || 0;
  const shown = Math.abs(count) >= 1000 ? count.toLocaleString('en-US') : String(count);
  return `${shown} ${count === 1 ? word : (pluralWord || `${word}s`)}`;
}

/** The length in code points (the server counts code points, not UTF-16 units). */
export function codePointLength(text) {
  const s = String(text ?? '');
  return /[\uD800-\uDFFF]/.test(s) ? Array.from(s).length : s.length;
}

/** The first `max` code points. */
export function sliceCodePoints(text, max) {
  const s = String(text ?? '');
  if (!/[\uD800-\uDFFF]/.test(s)) return s.slice(0, Math.max(0, max));
  return Array.from(s).slice(0, Math.max(0, max)).join('');
}

/** Today's ~15 chars/s estimate: 'under a minute', '~18 min', '~2 h 05 m'; `rate` is the playback speed. */
export function lengthLabel(chars, rate = 1) {
  const r = Number(rate) > 0 ? Number(rate) : 1;
  const minutes = Math.round((Number(chars) || 0) / CHARS_PER_SECOND / 60 / r);
  if (minutes < 1) return 'under a minute';
  if (minutes < 60) return `~${minutes} min`;
  return `~${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, '0')} m`;
}

/**
 * round(n / d × 100), clamped to 0–100; null when there is nothing to measure (no progress, d = 0).
 * The Reader's percentOf is percent(progress.block_index, block_count).
 */
export function percent(n, d) {
  if (n === null || n === undefined || !Number(d)) return null;
  const value = Math.round((Number(n) / Number(d)) * 100);
  if (!Number.isFinite(value)) return null;
  return Math.min(100, Math.max(0, value));
}
