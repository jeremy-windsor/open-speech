/**
 * Media Session values for the Reader. Pure: the engine computes the values here and
 * core/audio.js (the only module that touches navigator.mediaSession) applies them through the session's
 * mediaMetadata() and getState().
 *
 * - Metadata: title = the chapter title (the document title when the document has 0 or 1 chapters), artist =
 *   the author or else the voice, album = the document title, artwork = the document tile at 96, 256 and 512.
 * - Position state is chapter-relative (POSITION_SCOPE). Estimated durations are smoothed: the reported
 *   duration changes only when it moves by more than 2% or a part becomes ready, so the lock-screen scrubber
 *   doesn't jitter. If an on-device check shows that iOS mishandles it, set POSITION_SCOPE = 'part'.
 * @module views/reader/media-session
 */
import { RESTART_MS, previousTarget } from './play-logic.js';

/** 'chapter' (Apple Books style) or 'part' (the old per-part position). */
export const POSITION_SCOPE = 'chapter';
export const ARTWORK_SIZES = Object.freeze([96, 256, 512]);
export const SMOOTH_RATIO = 0.02;

/** The chapter's title, or '' when the document has 0 or 1 chapters (then the document title is the track). */
export function chapterTitle(doc, chapterIndex) {
  const chapters = doc?.chapters || [];
  if (chapters.length <= 1) return '';
  const chapter = chapters[Math.max(0, Math.min(chapters.length - 1, chapterIndex || 0))];
  return String(chapter?.title || `Chapter ${(chapterIndex || 0) + 1}`);
}

/**
 * The lock-screen metadata.
 * @param {{doc: object, chapterIndex?: number, voiceLabel?: string, artwork?: object[]}} input
 * @returns {{title: string, artist: string, album: string, artwork: object[]}}
 */
export function metadata({ doc, chapterIndex = 0, voiceLabel = '', artwork = [] } = {}) {
  const title = String(doc?.title || '');
  return {
    title: chapterTitle(doc, chapterIndex) || title,
    artist: String(doc?.author || '') || String(voiceLabel || ''),
    album: title,
    artwork: Array.isArray(artwork) ? artwork : [],
  };
}

/** The mini-player's title and subtitle for the same metadata (chapter title over document title). */
export function sessionText(md) {
  const title = md?.title || '';
  const subtitle = md?.album && md.album !== title ? md.album : (md?.artist || '');
  return { title, subtitle };
}

/**
 * Artwork entries from data URLs by size: {96: url, 256: url, 512: url} -> [{src, sizes, type}], skipping
 * sizes whose canvas failed ('').
 */
export function artworkList(urls) {
  const out = [];
  for (const size of ARTWORK_SIZES) {
    const src = urls && urls[size];
    if (src) out.push({ src, sizes: `${size}x${size}`, type: 'image/png' });
  }
  return out;
}

/**
 * The duration to report: a change of 2% or less is ignored (the estimate wobbles as parts render), unless
 * a part just became ready or nothing was reported yet.
 * @param {number} previousMs the duration reported last (0 = none)
 * @param {number} nextMs the timeline's duration now
 * @param {{partReady?: boolean}} [options]
 */
export function smoothDuration(previousMs, nextMs, { partReady = false } = {}) {
  if (!(previousMs > 0) || partReady || !(nextMs > 0)) return nextMs;
  return Math.abs(nextMs - previousMs) > previousMs * SMOOTH_RATIO ? nextMs : previousMs;
}

/**
 * The position state (seconds), or null when it must not be set (duration > 0, position <= duration and
 * rate > 0, all finite: setPositionState throws otherwise).
 * @param {{scope?: string, durationMs?: number, elapsedMs?: number, partPosition?: number, partDuration?: number,
 *   rate?: number}} input chapter scope uses durationMs/elapsedMs; part scope uses partPosition/partDuration
 */
export function positionState({ scope = POSITION_SCOPE, durationMs = 0, elapsedMs = 0, partPosition = 0, partDuration = 0, rate = 1 } = {}) {
  const duration = scope === 'part' ? Number(partDuration) : Number(durationMs) / 1000;
  const raw = scope === 'part' ? Number(partPosition) : Number(elapsedMs) / 1000;
  const r = Number(rate);
  if (!Number.isFinite(duration) || !(duration > 0) || !Number.isFinite(r) || !(r > 0) || !Number.isFinite(raw)) return null;
  return { duration, position: Math.max(0, Math.min(duration, raw)), rate: r };
}

/**
 * Where lock-screen previous/next track goes. Paragraph: previous restarts the paragraph after 3 s of
 * listening, else goes back one; next goes forward one (null at the end). Chapter: previous restarts the
 * chapter after 3 s, else goes to the previous chapter's start; next goes to the next chapter (null at the
 * last one).
 * @param {'paragraph'|'chapter'} unit
 * @param {'prev'|'next'} direction
 * @param {{cursor: number, blockCount: number, elapsedInBlockMs?: number, chapterIndex?: number,
 *   chapterStarts?: number[], elapsedInChapterMs?: number}} where elapsed values are listening time
 * @returns {number|null} the block to jump to
 */
export function trackTarget(unit, direction, { cursor = 0, blockCount = 0, elapsedInBlockMs = 0, chapterIndex = 0, chapterStarts = [0], elapsedInChapterMs = 0 } = {}) {
  if (unit === 'chapter' && chapterStarts.length) {
    if (direction === 'prev') {
      if (elapsedInChapterMs > RESTART_MS) return chapterStarts[chapterIndex] ?? 0;
      return chapterStarts[Math.max(0, chapterIndex - 1)] ?? 0;
    }
    return chapterIndex + 1 < chapterStarts.length ? chapterStarts[chapterIndex + 1] : null;
  }
  if (direction === 'prev') return previousTarget(cursor, elapsedInBlockMs);
  return cursor + 1 < blockCount ? cursor + 1 : null;
}
