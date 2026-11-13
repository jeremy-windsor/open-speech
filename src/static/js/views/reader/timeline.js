/**
 * The chapter-wide timeline across server-rendered parts. Pure: the engine passes the document's
 * blocks, the ready parts it has seen for the current voice and where playback is.
 *
 * - msPerChar is calibrated per model|voice from ready parts (Σ duration_ms / Σ characters of their blocks);
 *   until one is ready it is 1000/15 (the old 15 characters a second at 1×).
 * - A chapter is a list of segments: each ready part is one segment of its real duration, every other block
 *   one segment of chars × msPerChar. `estimated` is true while any block lacks a rendered part.
 * - Display values are divided by the playback rate ("−8:12" is the time left at the current speed; "~" marks
 *   an estimated elapsed time).
 * @module views/reader/timeline
 */
import { charsOf, clamp, trackPosition } from './play-logic.js';
import { clock, spokenDuration, speed as speedText } from '../../core/format.js';

export const DEFAULT_MS_PER_CHAR = 1000 / 15;
const MINUS = '−';

function partsOf(known) {
  if (!known) return [];
  if (typeof known.values === 'function') return Array.from(known.values());
  return Array.isArray(known) ? known : [];
}

/**
 * ms per character from ready parts: Σ duration_ms / Σ characters of their blocks.
 * @param {Map<number, object>|object[]} known ready parts (with duration_ms)
 * @param {object[]} blocks the document's blocks
 */
export function msPerChar(known, blocks) {
  let ms = 0;
  let chars = 0;
  for (const part of partsOf(known)) {
    if (!(part && part.duration_ms > 0)) continue;
    let partChars = 0;
    for (let b = part.first_block; b <= part.last_block; b += 1) partChars += charsOf(blocks?.[b]);
    if (!partChars) continue;
    ms += part.duration_ms;
    chars += partChars;
  }
  return chars > 0 && ms > 0 ? ms / chars : DEFAULT_MS_PER_CHAR;
}

/**
 * Add ready parts to the known map (first_block -> part). A newer part replaces any known part it overlaps
 * (the server plans the first part after a start short, so windows from different starts can overlap).
 * @returns {Map<number, object>} the same map
 */
export function remember(known, parts) {
  for (const part of parts || []) {
    if (!part || part.status !== 'ready' || !(part.duration_ms > 0)) continue;
    for (const [first, other] of known) {
      if (other.id !== part.id && first <= part.last_block && other.last_block >= part.first_block) known.delete(first);
    }
    known.set(part.first_block, part);
  }
  return known;
}

/**
 * The segments of blocks [start, end): a ready part is one segment, any other block one estimated segment.
 * @returns {{from: number, to: number, startMs: number, ms: number, part: object|null}[]}
 */
export function segments({ blocks = [], start = 0, end = 0, known = new Map(), perChar = DEFAULT_MS_PER_CHAR }) {
  const out = [];
  let at = 0;
  const map = known instanceof Map ? known : new Map(partsOf(known).map((p) => [p.first_block, p]));
  for (let b = start; b < end;) {
    const part = map.get(b);
    if (part && part.duration_ms > 0 && part.last_block >= b && part.last_block < end) {
      out.push({ from: b, to: part.last_block, startMs: at, ms: part.duration_ms, part });
      at += part.duration_ms;
      b = part.last_block + 1;
    } else {
      const ms = charsOf(blocks[b]) * perChar;
      out.push({ from: b, to: b, startMs: at, ms, part: null });
      at += ms;
      b += 1;
    }
  }
  return out;
}

function segmentIndexOf(segs, block) {
  return segs.findIndex((s) => s.from <= block && block <= s.to);
}

/** Where a block starts on the timeline (ms from the chapter start). */
export function timeOfBlock(segs, block) {
  const i = segmentIndexOf(segs, block);
  if (i < 0) return segs.length && block >= segs[segs.length - 1].to ? total(segs) : 0;
  const seg = segs[i];
  if (!seg.part || block === seg.from) return seg.startMs;
  const mark = (seg.part.sentences || []).find((m) => m.block === block);
  return seg.startMs + (mark ? Math.min(seg.ms, mark.start_ms || 0) : 0);
}

function total(segs) {
  const last = segs[segs.length - 1];
  return last ? last.startMs + last.ms : 0;
}

/**
 * The timeline of one chapter.
 * @param {object} input
 * @param {object[]} input.blocks
 * @param {number} input.start first block of the chapter
 * @param {number} input.end one past its last block
 * @param {Map<number, object>} input.known ready parts for the current voice
 * @param {number} input.perChar ms per character
 * @param {object|null} [input.active] the part on the active element
 * @param {number} [input.activeMs] the position inside it (ms)
 * @param {number} [input.cursor] the current block (when no part is active)
 * @param {number} [input.offsetMs] a saved offset inside the cursor block
 * @param {boolean} [input.finished] the document was heard to its end: the position is the chapter's end
 * @returns {{durationMs: number, elapsedMs: number, bufferedMs: number, estimated: boolean, segments: object[]}}
 */
export function chapterTimeline({ blocks = [], start = 0, end = 0, known = new Map(), perChar = DEFAULT_MS_PER_CHAR, active = null, activeMs = 0, cursor = 0, offsetMs = 0, finished = false }) {
  const segs = segments({ blocks, start, end, known, perChar });
  const durationMs = total(segs);
  if (!segs.length) return { durationMs: 0, elapsedMs: 0, bufferedMs: 0, estimated: false, segments: segs };
  let head;
  let elapsedMs;
  if (active && active.first_block >= start && active.first_block < end) {
    head = active.first_block;
    elapsedMs = timeOfBlock(segs, head) + Math.max(0, activeMs || 0);
  } else {
    head = clamp(cursor, start, end - 1);
    elapsedMs = finished ? durationMs : timeOfBlock(segs, head) + Math.max(0, offsetMs || 0);
  }
  elapsedMs = clamp(elapsedMs, 0, durationMs);
  let bufferedMs = elapsedMs;
  for (let i = Math.max(0, segmentIndexOf(segs, head)); i < segs.length && segs[i].part; i += 1) {
    bufferedMs = segs[i].startMs + segs[i].ms;
  }
  return {
    durationMs,
    elapsedMs,
    bufferedMs: clamp(Math.max(bufferedMs, elapsedMs), 0, durationMs),
    estimated: segs.some((s) => !s.part),
    segments: segs,
  };
}

/**
 * Map a chapter position to a seek target: inside a ready part, the part and the offset in it (and
 * the block there with its own offset); elsewhere, the block's start.
 * @returns {{block: number, part: object|null, partOffsetMs: number, blockOffsetMs: number}|null}
 */
export function seekTarget(segs, ms, { blocks = [] } = {}) {
  if (!segs || !segs.length) return null;
  const t = clamp(Number(ms) || 0, 0, total(segs));
  const seg = segs.find((s) => t < s.startMs + s.ms) || segs[segs.length - 1];
  if (!seg.part) return { block: seg.from, part: null, partOffsetMs: 0, blockOffsetMs: 0 };
  const partOffsetMs = clamp(t - seg.startMs, 0, seg.ms);
  const { block } = trackPosition(seg.part, partOffsetMs, { blocks, durationMs: seg.ms });
  const mark = (seg.part.sentences || []).find((m) => m.block === block);
  const blockStart = block === seg.part.first_block ? 0 : (mark ? mark.start_ms || 0 : 0);
  return { block, part: seg.part, partOffsetMs, blockOffsetMs: Math.max(0, partOffsetMs - blockStart) };
}

/**
 * Where a step of the scrubber (an arrow, PageUp/PageDown, an assistive increment) lands, so it always moves
 * (REV-13). A position inside a block without a rendered part maps to that block's start (seekTarget), so a step
 * that stays inside a long unrendered paragraph would seek back to where it started and the thumb would not move.
 * Forward, it goes on to the next block start that is past where playback is and at least `atLeastMs` (the first
 * value the keyboard's step grid shows as a move: landing on a short paragraph that starts inside the same grid step
 * would change the paragraph but not the value a screen reader hears). A backward step always lands behind (the
 * start of the block it reaches is never after it), and a step that reaches a rendered part lands at its exact
 * offset: both stand. Past the chapter's last block start there is nowhere further: the step stands, as End does.
 * @param {object[]} segs the chapter's segments (chapterTimeline().segments)
 * @param {number} fromMs where playback is on the chapter timeline
 * @param {number} toMs where the step asks to go
 * @param {number} direction 1 forward, -1 back, 0 not a step (a pointer drag): toMs as it is
 * @param {{atLeastMs?: number}} [options]
 * @returns {number} ms on the chapter timeline
 */
export function stepSeekMs(segs, fromMs, toMs, direction, { atLeastMs = 0 } = {}) {
  const to = Number(toMs) || 0;
  if (!segs || !segs.length || !(direction > 0)) return to;
  const from = Number(fromMs) || 0;
  const floor = Number(atLeastMs) || 0;
  const t = clamp(to, 0, total(segs));
  let reached = segs.findIndex((s) => t < s.startMs + s.ms);
  if (reached < 0) reached = segs.length - 1;
  if (segs[reached].part) return to;
  // An unrendered block is entered at its start: move on while that start is not past playback and the floor.
  let i = reached;
  while (segs[i].startMs <= from || segs[i].startMs < floor) {
    if (i + 1 >= segs.length) return to;
    i += 1;
  }
  return i === reached ? to : segs[i].startMs;
}

/**
 * What the player shows for a timeline at a playback rate: '~4:03', '−8:12', and the scrubber's
 * aria-valuetext ('4 minutes 3 seconds of about 12 minutes'). An estimate is marked once, on the elapsed time:
 * '−~8:12' ran together into '--8:12' at small sizes; the spoken text still says "about".
 */
export function display(timeline, rate = 1) {
  const r = Number(rate) > 0 ? Number(rate) : 1;
  const t = timeline || { durationMs: 0, elapsedMs: 0, estimated: false };
  const elapsed = (t.elapsedMs || 0) / r / 1000;
  const duration = (t.durationMs || 0) / r / 1000;
  const left = Math.max(0, duration - elapsed);
  const about = t.estimated ? '~' : '';
  return {
    elapsed: `${about}${clock(elapsed)}`,
    remaining: `${MINUS}${clock(Math.ceil(left))}`,
    valuetext: `${spokenDuration(elapsed)} of ${t.estimated ? 'about ' : ''}${spokenDuration(duration)}`,
  };
}

/** The spoken position for a scrubber value that is being dragged (ms of audio). */
export function spokenAt(ms, timeline, rate = 1) {
  return display({ ...(timeline || {}), elapsedMs: ms }, rate).valuetext;
}

/* ───────────── the scrubber's value ───────────── */

/**
 * The keyboard (and VoiceOver) step of the scrubber: the skip interval, but never more than a quarter of a
 * short chapter (whole seconds, at least 1 s), so the arrows can still reach several places in a 9 s note.
 */
export function keyStepMs(skipMs, durationMs) {
  const skip = Math.max(1000, Math.round(Number(skipMs) || 15000));
  const duration = Number(durationMs) || 0;
  if (duration <= 0) return skip;
  return Math.min(skip, Math.max(1000, Math.round(duration / 4000) * 1000));
}

/** How finely the thumb follows playback: about 1/600 of the chapter, between one frame (16 ms) and 1 s. */
export function scrubQuantumMs(durationMs) {
  return clamp(Math.round((Number(durationMs) || 0) / 600), 16, 1000);
}

/** A value on a step grid from 0, as a browser stores a range value: the nearest step, never past max. */
export function snapToStep(ms, step, max) {
  const top = Math.max(0, Number(max) || 0);
  const value = clamp(Number(ms) || 0, 0, top);
  if (!(step > 0)) return Math.round(value);
  const snapped = Math.round(value / step) * step;
  return snapped > top ? Math.max(0, snapped - step) : snapped;
}

/**
 * What the scrubber holds for a timeline. Normally there is no step (the thumb follows the exact position,
 * quantised to scrubQuantumMs); while the keyboard or a screen reader is on it, the step is keyStepMs() and the
 * value sits on that grid, so the arrows move by the skip interval and a focused slider isn't re-announced
 * every second.
 * @param {{elapsedMs?: number, durationMs?: number}} timeline
 * @param {{keyboard?: boolean, skipMs?: number}} [options]
 * @returns {{value: number, step: number|null}} ms; step null means "no step attribute"
 */
export function scrubPosition(timeline, { keyboard = false, skipMs = 15000 } = {}) {
  const duration = Math.max(0, Math.round(Number(timeline?.durationMs) || 0));
  const elapsed = clamp(Math.round(Number(timeline?.elapsedMs) || 0), 0, duration);   // a fractional end is the end
  if (keyboard) {
    const step = keyStepMs(skipMs, duration);
    return { value: snapToStep(elapsed, step, duration), step };
  }
  if (elapsed >= duration) return { value: duration, step: null };
  const quantum = scrubQuantumMs(duration);
  return { value: Math.min(duration, Math.round(Math.floor(elapsed / quantum) * quantum)), step: null };
}

/**
 * Where a scrubber change that no pointer and no key made lands. An assistive technology that adjusts
 * the slider without moving the page's focus to it (Android TalkBack moves its own cursor) steps by the browser's
 * default step of 1 (1 ms here) or a few percent, since the step attribute is only there while the slider has
 * focus. A move smaller than one key step becomes one key step in that direction; anything larger stands.
 * @param {number} value the scrubber's new value (ms)
 * @param {number} shown the value it was last painted at (ms)
 * @param {{skipMs?: number, durationMs?: number}} [options] durationMs: the scrubber's max
 * @returns {number} ms
 */
export function assistedTarget(value, shown, { skipMs = 15000, durationMs = 0 } = {}) {
  const to = Number(value) || 0;
  const from = Number(shown) || 0;
  const delta = to - from;
  const step = keyStepMs(skipMs, durationMs);
  if (!delta || Math.abs(delta) >= step) return to;
  return clamp(from + Math.sign(delta) * step, 0, Math.max(0, Number(durationMs) || 0));
}

/** A share of the chapter as a CSS percentage ('42.5%'), for --fill and --buffer. */
export function percentOf(ms, durationMs) {
  const duration = Number(durationMs) || 0;
  const pct = duration > 0 ? clamp(((Number(ms) || 0) / duration) * 100, 0, 100) : 0;
  return `${Math.round(pct * 100) / 100}%`;
}

/**
 * The listening time left in the whole document: the rest of this chapter on its timeline, plus the later
 * blocks estimated from their characters.
 */
export function documentRemainingMs({ blocks = [], chapterEnd = 0, perChar = DEFAULT_MS_PER_CHAR, timeline = null }) {
  let ms = timeline ? Math.max(0, timeline.durationMs - timeline.elapsedMs) : 0;
  for (let b = chapterEnd; b < blocks.length; b += 1) ms += charsOf(blocks[b]) * perChar;
  return ms;
}

/** '~2 h 10 min left at 1.25×', '~18 min left at 1×', 'Less than a minute left at 1.5×'. */
export function leftLabel(ms, rate = 1) {
  const r = Number(rate) > 0 ? Number(rate) : 1;
  const minutes = Math.round((Number(ms) || 0) / r / 60000);
  const at = `at ${speedText(r)}`;
  if (minutes < 1) return `Less than a minute left ${at}`;
  if (minutes < 60) return `~${minutes} min left ${at}`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return `~${h} h${m ? ` ${m} min` : ''} left ${at}`;
}
