/**
 * Speak: the blend editor's model (the .blend-bar of components.css). Pure: no DOM. Items are
 * [{voice: 'af_bella', weight: 2}, …]; the canonical recipe is voices.formatBlend(items).
 *
 * Rules (inventory SP-26, SP-27): at most 6 voices, no duplicates, weights stepped by 0.1 within 0.1–3.0
 * (a weight that arrives from a saved recipe outside that range is kept exactly until it is edited, because
 * blends must round-trip), one voice left is not a blend.
 * @module views/speak/blend
 */
import { parseBlend, formatBlend, blendPercents, BlendError } from '../../core/voices.js';

export const MAX_VOICES = 6;
export const MIN_WEIGHT = 0.1;
export const MAX_WEIGHT = 3;
export const WEIGHT_STEP = 0.1;
/** A segment carries its percentage inside the bar only when it is at least this wide. */
export const LABEL_MIN_PERCENT = 18;

const round1 = (n) => Math.round(n * 10) / 10;

/** A typed or stepped weight on the 0.1 grid inside 0.1–3.0; NaN for text that isn't a number. */
export function clampWeight(value) {
  const n = typeof value === 'string' ? Number(value.replace(',', '.').trim()) : Number(value);
  if (!Number.isFinite(n) || (typeof value === 'string' && !value.trim())) return NaN;
  return Math.min(MAX_WEIGHT, Math.max(MIN_WEIGHT, round1(n)));
}

/** Items for a voice spec: a blend recipe, or one voice as a single item, or [] for ''. */
export function fromSpec(spec) {
  const text = String(spec ?? '').trim();
  if (!text) return [];
  return parseBlend(text).map((item) => ({ voice: item.voice, weight: item.weight }));
}

/** The canonical recipe ('af_bella(2)+af_sky(1)'; one voice gives just its id). */
export function toSpec(items) {
  return formatBlend(items || []);
}

/** @returns {boolean} two or more voices: a real blend */
export function isMix(items) {
  return Array.isArray(items) && items.length >= 2;
}

/** Add a voice at weight 1 (ignored when present or at MAX_VOICES). */
export function addVoice(items, voice, weight = 1) {
  const list = (items || []).slice();
  if (!voice || list.some((item) => item.voice === voice) || list.length >= MAX_VOICES) return list;
  list.push({ voice, weight });
  return list;
}

/** Remove a voice (the last one is kept: an empty blend has nothing to speak with). */
export function removeVoice(items, voice) {
  const list = items || [];
  if (list.length <= 1) return list.slice();
  return list.filter((item) => item.voice !== voice);
}

/** Set one voice's weight (clamped); invalid text leaves the items unchanged. */
export function setWeight(items, voice, weight) {
  const w = clampWeight(weight);
  if (Number.isNaN(w)) return (items || []).slice();
  return (items || []).map((item) => (item.voice === voice ? { ...item, weight: w } : item));
}

/** Step a weight by ±0.1 within the range. */
export function stepWeight(items, voice, direction) {
  return (items || []).map((item) => {
    if (item.voice !== voice) return item;
    const base = Math.min(MAX_WEIGHT, Math.max(MIN_WEIGHT, Number(item.weight) || 1));
    return { ...item, weight: clampWeight(base + (direction < 0 ? -WEIGHT_STEP : WEIGHT_STEP)) };
  });
}

/** Every weight back to 1. */
export function equalize(items) {
  return (items || []).map((item) => ({ ...item, weight: 1 }));
}

/** The weight shown in a stepper: up to 2 decimals, no trailing zeros. */
export function weightText(weight) {
  return String(Number((Math.round(Number(weight) * 100) / 100).toFixed(2)));
}

/**
 * The rows and bar segments: [{voice, label, weight, percent, index, showLabel}] with integer percents that
 * sum to 100 (voices.blendPercents).
 * @param {object[]} items
 * @param {(voice: string) => string} [labelFor] the display name ('Bella')
 */
export function segments(items, labelFor = (v) => v) {
  const list = items || [];
  const percents = blendPercents(list);
  return list.map((item, index) => ({
    voice: item.voice,
    label: labelFor(item.voice) || item.voice,
    weight: item.weight,
    percent: percents[index] ?? 0,
    index,
    showLabel: (percents[index] ?? 0) >= LABEL_MIN_PERCENT,
  }));
}

/** The bar's accessible name: 'Bella 67 percent, Sky 33 percent'. */
export function barLabel(items, labelFor) {
  return segments(items, labelFor).map((s) => `${s.label} ${s.percent} percent`).join(', ');
}

/**
 * Check a pasted recipe for this model: the server's grammar (voices.parseBlend), at most 6 voices, and
 * every voice offered by the model (when the model lists its voices).
 * @param {string} text
 * @param {{voices?: string[]}} [options] the model's voice ids
 * @returns {{items: object[]} | {error: string}}
 */
export function validateRecipe(text, { voices = [] } = {}) {
  let items;
  try {
    items = fromSpec(text);
  } catch (error) {
    return { error: error instanceof BlendError ? error.message : String(error?.message || error) };
  }
  if (!items.length) return { error: 'Enter at least one voice.' };
  if (items.length > MAX_VOICES) return { error: `A blend mixes up to ${MAX_VOICES} voices.` };
  if (voices.length) {
    const missing = items.filter((item) => !voices.includes(item.voice)).map((item) => item.voice);
    if (missing.length) return { error: `This model has no voice named ${missing.join(', ')}.` };
  }
  return { items };
}
