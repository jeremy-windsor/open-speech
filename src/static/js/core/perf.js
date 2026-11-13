/**
 * Speed measured on this machine, per model: time to first audio and real-time factor.
 * @module core/perf
 */
import { KEYS, readJson, writeJson } from './storage.js';

/** RTF at or above this gets the 'Slower than real time' tag. */
export const SLOW_RTF = 1.0;
const MAX_MODELS = 50;

const isObject = (v) => !!v && typeof v === 'object' && !Array.isArray(v);

function all() {
  return readJson(KEYS.perf, {}, isObject);
}

/**
 * Record a finished synthesis: rtf = renderMs / 1000 / durationS. Keeps the 50 most recent models.
 * @param {string} modelId
 * @param {{ttfaMs?: number, renderMs?: number, durationS?: number}} sample
 * @returns {object|null} the stored entry
 */
export function record(modelId, { ttfaMs, renderMs, durationS } = {}) {
  if (!modelId) return null;
  const entry = { at: Date.now() };
  if (Number.isFinite(ttfaMs) && ttfaMs >= 0) entry.ttfaMs = Math.round(ttfaMs);
  if (Number.isFinite(renderMs) && renderMs >= 0) entry.renderMs = Math.round(renderMs);
  if (Number.isFinite(durationS) && durationS > 0) entry.durationS = durationS;
  if (entry.renderMs !== undefined && entry.durationS) entry.rtf = entry.renderMs / 1000 / entry.durationS;
  const data = all();
  data[modelId] = { ...(isObject(data[modelId]) ? data[modelId] : {}), ...entry };
  const ids = Object.keys(data).sort((a, b) => (data[b]?.at || 0) - (data[a]?.at || 0));
  for (const id of ids.slice(MAX_MODELS)) delete data[id];
  writeJson(KEYS.perf, data);
  return data[modelId];
}

/** @returns {{ttfaMs?: number, renderMs?: number, durationS?: number, rtf?: number, at: number}|null} */
export function forModel(modelId) {
  const entry = all()[modelId];
  return isObject(entry) ? entry : null;
}

/** 'First audio 140 ms · RTF 0.04' (missing parts are left out; '' for nothing). */
export function label(perf) {
  if (!perf) return '';
  const parts = [];
  if (Number.isFinite(perf.ttfaMs)) parts.push(`First audio ${Math.round(perf.ttfaMs)} ms`);
  if (Number.isFinite(perf.rtf)) parts.push(`RTF ${perf.rtf < 10 ? perf.rtf.toFixed(2) : perf.rtf.toFixed(1)}`);
  return parts.join(' · ');
}

/** @returns {boolean} whether the model renders slower than real time */
export function isSlow(perf) {
  return !!perf && Number.isFinite(perf.rtf) && perf.rtf >= SLOW_RTF;
}
