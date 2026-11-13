/**
 * Reader playback: the pure helpers the engine is built from, ported from the old reader.js with
 * the same behaviour. No DOM queries: the seek helpers only read and write a media element's dataset and
 * currentTime, so tests drive them with fake elements.
 * @module views/reader/play-logic
 */
import { describeVoice, isBlend, parseBlend } from '../../core/voices.js';

export const POLL_MS = 500;                // first /parts poll, and the backoff base
export const MAX_BACKOFF_MS = 4000;
export const MAX_FAILURES = 10;            // consecutive poll or audio failures before playback gives up
export const LOOKAHEAD_POLL_MS = 1000;     // the lookahead polls at least this far apart
export const RERENDER_AFTER_MS = 3000;     // re-request the render when our window vanished for this long
export const RERENDER_MAX_MS = 24000;      // ...then wait twice as long after each re-request, up to this
export const PREPARING_DELAY_MS = 1000;    // "Preparing audio…" after this long
export const SLOW_PREPARING_MS = 20000;    // "Still preparing audio…" after this long
export const MODELS_WAIT_MS = 5000;        // a deep link waits this long for the model list
export const SAVE_EVERY_MS = 10000;        // progress saves while playing (driven by tracking, not a timer)
export const RESTART_MS = 3000;            // previous track restarts the paragraph (or chapter) when this far in
export const REQUEST_TIMEOUT_MS = 15000;   // render, parts and progress requests
export const NOTIFY_MS = 250;              // the coordinator (mini-player, lock screen) hears about position at 4 Hz

export const MIN_SPEED = 0.5;
export const MAX_SPEED = 3.5;
export const SPEED_STEP = 0.05;
export const SPEED_PRESETS = Object.freeze([0.75, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]);

/** The status line and toast copy. */
export const COPY = Object.freeze({
  preparing: 'Preparing audio…',
  slow: 'Still preparing audio. The model may still be loading, or Live is busy.',
  reconnecting: 'Reconnecting…',
  finished: 'Finished',
  sleepEnded: 'Sleep timer ended',
  tapPlay: 'Tap play to start listening',
  voiceChanged: 'Voice changed. Re-rendering from here.',
  noVoice: 'Choose a voice for the Reader first',
  noModel: 'No speech model is ready yet.',   // the toast carries [Open Models]
  openModels: 'Open Models',
  partFailed: 'The audio for this paragraph could not be rendered',
  nextFailed: 'The next part could not be rendered',
  unreachable: "the server can't be reached",
  audioFailed: 'The audio could not be loaded',
  chooseBoth: 'Choose a model and a voice',
  recording: 'Stop recording to listen.',
});

export const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

/** Where block `index` starts inside `parts`: the part beginning with it, or a sentence mark. Never a guess (RE-09). */
export function locate(index, parts) {
  const list = Array.isArray(parts) ? parts : [];
  const exact = list.find((part) => part && part.first_block === index);
  if (exact) return { part: exact, offsetMs: 0 };
  const part = list.find((item) => item && item.first_block <= index && index <= item.last_block);
  const mark = (part?.sentences || []).find((item) => item.block === index);
  return mark ? { part, offsetMs: mark.start_ms || 0 } : null;
}

/**
 * Polling and audio retries (RE-10): network errors, timeouts, 5xx, 408 and 429 are worth retrying; other
 * 4xx are final. A refused or locked-out API key is final too: retrying it would only count more failures.
 */
export function isRetryable(error) {
  if (!error) return true;
  if (error.kind === 'auth' || error.kind === 'locked' || error.kind === 'aborted') return false;
  const status = error.status;
  return !status || status >= 500 || status === 408 || status === 429;
}

/** 500 → 1000 → 2000 → 4000 ms (cap); `failures` consecutive failures so far. */
export function backoffMs(failures) {
  return failures > 0 ? Math.min(MAX_BACKOFF_MS, POLL_MS * 2 ** (failures - 1)) : POLL_MS;
}

// A seek made before metadata loads is parked on the element; a new src clears it, so it can't hit the next
// part (RE-14).
export function parkSeek(el, seconds) { el.dataset.seek = String(seconds); }
export function clearParkedSeek(el) { delete el.dataset.seek; }
export function takeParkedSeek(el) {
  if (el.dataset.seek == null) return null;
  const seconds = Number(el.dataset.seek);
  delete el.dataset.seek;
  return Number.isFinite(seconds) ? seconds : null;
}
/** Where the element plays from: a parked seek (a resume still loading) counts, as currentTime is still 0 (RE-15). */
export function positionOf(el) {
  const parked = el.dataset.seek == null ? NaN : Number(el.dataset.seek);
  return Number.isFinite(parked) ? parked : (el.currentTime || 0);
}

/** Lock-screen "previous": restart the paragraph when well into it, otherwise go back one (RE-45). */
export function previousTarget(cursor, elapsedMs) {
  return elapsedMs > RESTART_MS ? cursor : Math.max(0, cursor - 1);
}

/** The document's saved voice first, then the last Reader choice, then whatever Speak has selected (RE-04). */
export function resolveVoice({ progress, stored, speak, modelIds }) {
  const ids = Array.isArray(modelIds) ? modelIds : [];
  const known = (choice) => !!(choice?.model && choice.voice && ids.includes(choice.model));
  if (known(progress)) return { model: progress.model, voice: progress.voice };
  if (known(stored)) return { model: stored.model, voice: stored.voice };
  return known(speak) ? { model: speak.model, voice: speak.voice } : null;
}

/**
 * Whether Speak's published voice is a raw reference recording, which Reader renders can't use (REV-09): Speak
 * says so with `kind: 'recording'` (app.speakVoice, views/speak/speak-voice.js voiceKind()).
 * @param {{model?: string, voice?: string, kind?: string}|null} speak app.speakVoice
 */
export function isRecordingChoice(speak) {
  return !!(speak && speak.voice && speak.kind === 'recording');
}

/** The saved offset only applies to the saved paragraph, read with the voice that made it (RE-32). */
export function resumeOffset(progress, voice, cursor) {
  if (!progress || !voice || progress.block_index !== cursor) return 0;
  if (progress.model !== voice.model || progress.voice !== voice.voice) return 0;
  return Math.max(0, Number(progress.offset_ms) || 0);
}

/** A playback speed rounded to 0.05 within 0.5–3.5, or null for anything that isn't a number. */
export function clampSpeed(value) {
  const n = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(n)) return null;
  const rounded = Math.round(n / SPEED_STEP) * SPEED_STEP;
  return Number(clamp(rounded, MIN_SPEED, MAX_SPEED).toFixed(2));
}

/** The stored global default speed (KEYS.readerSpeed): unknown or out-of-range values give 1 (RE-46). */
export function storedSpeed(text) {
  const n = Number(text);
  if (text === null || text === undefined || text === '' || !Number.isFinite(n) || n < MIN_SPEED || n > MAX_SPEED) return 1;
  return clampSpeed(n);
}

/** The speed a document opens at: its saved speed, else the global default. */
export function documentSpeed(progress, fallback) {
  const n = Number(progress?.speed);
  return progress && progress.speed !== null && progress.speed !== undefined && Number.isFinite(n) && n >= MIN_SPEED && n <= MAX_SPEED
    ? clampSpeed(n)
    : fallback;
}

/**
 * block index -> chapter position: the last chapter starting at or before the block (blocks before the first
 * chapter, the front matter, count as the first one). The same rule as chapterRange().
 */
export function chapterMap(blockCount, chapters) {
  const map = new Array(Math.max(0, blockCount)).fill(0);
  const list = chapters || [];
  for (let position = 1; position < list.length; position += 1) {
    const end = position + 1 < list.length ? Math.min(blockCount, list[position + 1].first_block) : blockCount;
    for (let i = Math.max(0, list[position].first_block); i < end; i += 1) map[i] = position;
  }
  return map;
}

/** [start, end) blocks of chapter `position`; a document without chapters is one chapter. */
export function chapterRange(doc, position) {
  const count = doc?.blocks?.length || 0;
  const chapters = doc?.chapters || [];
  if (!chapters.length) return [0, count];
  const p = clamp(position, 0, chapters.length - 1);
  const chapter = chapters[p];
  const start = p <= 0 ? 0 : clamp(chapter.first_block, 0, count);
  const next = chapters[p + 1];
  const end = next ? clamp(next.first_block, start, count) : count;
  return [start, Math.max(start, end)];
}

/** The first block of chapter `position` (past the last chapter: the block count). */
export function chapterStart(doc, position) {
  const chapters = doc?.chapters || [];
  if (position >= Math.max(1, chapters.length)) return doc?.blocks?.length || 0;
  return chapterRange(doc, Math.max(0, position))[0];
}

/** The characters of a block (the timeline's unit before a part is rendered). */
export function charsOf(block) {
  return String(block?.text || '').length;
}

/**
 * Which block and sentence a part is at, `ms` into it: the last sentence mark at or before `ms`; a
 * multi-block part without marks estimates the block from its share of the characters (RE-51).
 * @returns {{block: number, sentence: number}}
 */
export function trackPosition(part, ms, { blocks = [], durationMs = 0 } = {}) {
  const marks = Array.isArray(part?.sentences) ? part.sentences : [];
  let block = part.first_block;
  let sentence = -1;
  if (marks.length) {
    for (const mark of marks) {
      if (mark.start_ms > ms) break;
      block = mark.block;
      sentence = mark.index;
    }
  } else if (part.last_block > part.first_block && durationMs > 0) {
    const inPart = blocks.slice(part.first_block, part.last_block + 1);
    const total = inPart.reduce((sum, item) => sum + charsOf(item), 0) || 1;
    let reached = (ms / durationMs) * total;
    for (let i = 0; i < inPart.length; i += 1) {
      reached -= charsOf(inPart[i]);
      if (reached <= 0) {
        block = part.first_block + i;
        break;
      }
    }
  }
  return { block, sentence };
}

/** The start of the sentence that contains `ms` in a part (snapToSentence), or `ms` when the part has no marks. */
export function sentenceStartMs(part, ms) {
  let start = null;
  for (const mark of part?.sentences || []) {
    if (mark.start_ms > ms) break;
    start = mark.start_ms;
  }
  return start === null ? ms : start;
}

/**
 * The sentence mark `delta` sentences from (block, sentence) inside a part, or null when it is outside it.
 * sentence -1 means "at the block start": the next sentence is the block's first.
 */
export function sentenceStep(part, block, sentence, delta) {
  const marks = (part?.sentences || []).slice().sort((a, b) => a.start_ms - b.start_ms);
  if (!marks.length) return null;
  let here = marks.findIndex((m) => m.block === block && m.index === sentence);
  if (here < 0) {
    const first = marks.findIndex((m) => m.block === block);
    if (first < 0) return null;
    here = delta > 0 ? first - 1 : first;
  }
  const target = marks[here + delta];
  return target || null;
}

/**
 * What a voice is called on the chip and the lock screen (RE-63): a named voice's name (or "Named voice"
 * until the names load), a Kokoro label ("Heart"), a blend's labels joined ("Bella + Sky").
 * @param {{voice?: string}|null} choice
 * @param {Map<string, string>} [names] named voice id -> name
 */
export function voiceLabel(choice, names = new Map()) {
  const voice = String(choice?.voice || '');
  if (!voice) return '';
  if (voice.startsWith('voice:')) return names.get(voice.slice(6)) || 'Named voice';
  if (isBlend(voice)) {
    try {
      return parseBlend(voice).map((item) => describeVoice(item.voice).label).join(' + ');
    } catch {
      return voice;
    }
  }
  return describeVoice(voice).label;
}

/** The detail line of a voice in the voice sheet: the canonical blend string, a Kokoro detail, or the id. */
export function voiceDetail(choice) {
  const voice = String(choice?.voice || '');
  if (!voice) return '';
  if (voice.startsWith('voice:')) return 'Named voice';
  if (isBlend(voice)) return voice;
  return describeVoice(voice).detail || voice;
}

let silent = '';

/** 0.1 s of 8 kHz unsigned 8-bit silence as a data: WAV (the keep-alive loop and the iOS unlock). */
export function silentWavUrl() {
  if (silent) return silent;
  const samples = 800;
  const bytes = new Uint8Array(44 + samples);
  const view = new DataView(bytes.buffer);
  const ascii = (offset, text) => { for (let i = 0; i < text.length; i += 1) bytes[offset + i] = text.charCodeAt(i); };
  ascii(0, 'RIFF'); view.setUint32(4, 36 + samples, true); ascii(8, 'WAVE'); ascii(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, 8000, true); view.setUint32(28, 8000, true); view.setUint16(32, 1, true); view.setUint16(34, 8, true);
  ascii(36, 'data'); view.setUint32(40, samples, true);
  bytes.fill(128, 44);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  silent = `data:audio/wav;base64,${btoa(binary)}`;
  return silent;
}
