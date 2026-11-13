/**
 * The sleep timer: a pure reducer. The engine ticks it from the media events (timeupdate, about
 * 4 Hz, which keeps firing on a locked iPhone, plus pause and ended) and, while the page is visible, from its
 * animation-frame ticker. Nothing here uses setTimeout, because background timers are throttled on a locked
 * phone. The countdown only runs while audio plays: paused or waiting for a part, it holds.
 *
 * State: {mode: 'off'|'minutes'|'chapters', remainingMs, minutes, boundaryBlock, chapters, fading, lastAt}
 *   minutes: remainingMs counts down in wall-clock time while playing; the last 10 s fade the volume
 *   chapters: playback stops before a part whose first block is at or past boundaryBlock (the first block of
 *     chapter current + n). Server parts never span two chapters, so it stops exactly at the chapter end.
 * @module views/reader/sleep-timer
 */
import { clock, spokenDuration } from '../../core/format.js';

export const FADE_MS = 10000;
export const MINUTE_OPTIONS = Object.freeze([5, 10, 15, 30, 45, 60]);
export const MIN_CHAPTERS = 2;     // "End of [N] chapters" (one chapter is its own option)
export const MAX_CHAPTERS = 10;

export const OFF = Object.freeze({ mode: 'off', remainingMs: 0, minutes: 0, boundaryBlock: -1, chapters: 0, fading: false, lastAt: null });

const whole = (n, min, max) => Math.min(max, Math.max(min, Math.round(Number(n) || 0)));

/**
 * A new timer from a sheet choice.
 * @param {{minutes?: number, chapters?: number}|null} spec
 * @param {{boundaryBlock?: number}} [context] the first block of chapter current + n (chapters mode)
 */
export function start(spec, { boundaryBlock = -1 } = {}) {
  if (!spec) return OFF;
  const minutes = Number(spec.minutes);
  if (Number.isFinite(minutes) && minutes > 0) {
    return Object.freeze({ ...OFF, mode: 'minutes', minutes, remainingMs: Math.round(minutes * 60000) });
  }
  const chapters = Number(spec.chapters);
  if (Number.isFinite(chapters) && chapters > 0) {
    return Object.freeze({ ...OFF, mode: 'chapters', chapters: whole(chapters, 1, MAX_CHAPTERS), boundaryBlock });
  }
  return OFF;
}

/**
 * Count down by the wall-clock time audio played: `lastAt` is set only while playing, so the time since it
 * counts (up to the tick that reports the pause), and a pause or a wait for a part holds the countdown (the
 * first tick after it only notes the time).
 * @param {object} state
 * @param {{playing: boolean, now: number, fade?: boolean}} input playing: audio is playing now
 */
export function tick(state, { playing, now, fade = true }) {
  if (!state || state.mode !== 'minutes') return state || OFF;
  let remaining = state.remainingMs;
  if (state.lastAt !== null && Number.isFinite(state.lastAt)) remaining = Math.max(0, remaining - Math.max(0, now - state.lastAt));
  return Object.freeze({ ...state, remainingMs: remaining, lastAt: playing ? now : null, fading: !!fade && remaining <= FADE_MS });
}

/** The volume for the fade (1 → 0 over the last 10 s), or null when the timer doesn't touch the volume. */
export function volumeFor(state, { fade = true } = {}) {
  if (!state || state.mode !== 'minutes' || !fade || state.remainingMs > FADE_MS) return null;
  return Math.max(0, Math.min(1, state.remainingMs / FADE_MS));
}

/** True once a minutes timer has run out. */
export function expired(state) {
  return !!state && state.mode === 'minutes' && state.remainingMs <= 0;
}

/** True when playback must stop instead of starting the block `nextBlock` (chapters mode). */
export function stopsBefore(state, nextBlock) {
  return !!state && state.mode === 'chapters' && state.boundaryBlock >= 0 && nextBlock >= state.boundaryBlock;
}

/** The value of the sheet's radio group for a state: 'off', '5'…'60', 'chapter' or 'chapters'. */
export function optionValue(state) {
  if (!state || state.mode === 'off') return 'off';
  if (state.mode === 'chapters') return state.chapters > 1 ? 'chapters' : 'chapter';
  return String(state.minutes);
}

/** The timer a radio value asks for: 'off' -> null, '15' -> {minutes: 15}, 'chapter' -> {chapters: 1}. */
export function specFor(value, chapters = MIN_CHAPTERS) {
  if (value === 'chapter') return { chapters: 1 };
  if (value === 'chapters') return { chapters: whole(chapters, MIN_CHAPTERS, MAX_CHAPTERS) };
  const minutes = Number(value);
  return Number.isFinite(minutes) && minutes > 0 ? { minutes } : null;
}

/** The chip's short text: 'Sleep', '12:40', 'End of ch.' or 'End of 3 ch.'. */
export function chipLabel(state) {
  if (!state || state.mode === 'off') return 'Sleep';
  if (state.mode === 'chapters') return state.chapters > 1 ? `End of ${state.chapters} ch.` : 'End of ch.';
  return clock(Math.ceil(state.remainingMs / 1000));
}

/**
 * The chip's accessible name ("Sleep timer: off. Set sleep timer", "Sleep timer: 13 minutes left. Change sleep
 * timer"). It changes once a minute at most, never each second (a focused control's name must not tick).
 */
export function chipName(state) {
  if (!state || state.mode === 'off') return 'Sleep timer: off. Set sleep timer';
  if (state.mode === 'chapters') {
    return state.chapters > 1 ? `Sleep timer: end of ${state.chapters} chapters. Change sleep timer` : 'Sleep timer: end of chapter. Change sleep timer';
  }
  const minutes = Math.max(1, Math.ceil(state.remainingMs / 60000));
  return `Sleep timer: ${minutes} ${minutes === 1 ? 'minute' : 'minutes'} left. Change sleep timer`;
}

/**
 * The sheet's live line: 'Off', 'Stops in 12:40', 'Stops at the end of this chapter, in about 4:10'.
 * @param {object} state
 * @param {number} [remainingMs] for chapters mode: the listening time left until the boundary (at speed)
 */
export function remainingText(state, remainingMs = 0) {
  if (!state || state.mode === 'off') return 'Off';
  const at = clock(Math.ceil(Math.max(0, state.mode === 'minutes' ? state.remainingMs : remainingMs) / 1000));
  if (state.mode === 'minutes') return `Stops in ${at}`;
  const where = state.chapters > 1 ? `the end of ${state.chapters} chapters` : 'the end of this chapter';
  return remainingMs > 0 ? `Stops at ${where}, in about ${at}` : `Stops at ${where}`;
}

/** The spoken form of remainingText for an announcement when the timer is set. */
export function spokenRemaining(state) {
  if (!state || state.mode === 'off') return 'Sleep timer off';
  if (state.mode === 'chapters') return state.chapters > 1 ? `Sleep timer set to the end of ${state.chapters} chapters` : 'Sleep timer set to the end of this chapter';
  return `Sleep timer set: ${spokenDuration(Math.ceil(state.remainingMs / 1000))}`;
}
