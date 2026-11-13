/**
 * Transcript exports, pure: plain text, SRT, WebVTT and JSON from the transcript's segments.
 * SRT puts a comma before the milliseconds, VTT a period. File names are transcript-<timestamp>.<ext>.
 * @module views/transcribe/export
 */

/** The export formats, in menu order. */
export const FORMATS = Object.freeze({
  txt: Object.freeze({ ext: 'txt', mime: 'text/plain;charset=utf-8', label: 'Text (.txt)' }),
  srt: Object.freeze({ ext: 'srt', mime: 'application/x-subrip;charset=utf-8', label: 'SRT subtitles (.srt)' }),
  vtt: Object.freeze({ ext: 'vtt', mime: 'text/vtt;charset=utf-8', label: 'WebVTT captions (.vtt)' }),
  json: Object.freeze({ ext: 'json', mime: 'application/json;charset=utf-8', label: 'JSON with timings (.json)' }),
});

const MIN_CUE_S = 0.5;

function seconds(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

/**
 * A cue time: '00:00:03,120' (SRT, comma) or '00:00:03.120' (VTT, period). Hours are never dropped.
 * @param {number} value seconds
 * @param {','|'.'} [separator]
 */
export function cueTime(value, separator = ',') {
  const totalMs = Math.round(seconds(value) * 1000);
  const ms = totalMs % 1000;
  const totalS = Math.floor(totalMs / 1000);
  const s = totalS % 60;
  const m = Math.floor(totalS / 60) % 60;
  const h = Math.floor(totalS / 3600);
  const two = (n) => String(n).padStart(2, '0');
  return `${two(h)}:${two(m)}:${two(s)}${separator}${String(ms).padStart(3, '0')}`;
}

/** Cue text: no blank lines (they end a cue) and no '-->' (it starts a timing line). */
function cueText(text) {
  return String(text ?? '').replace(/\r\n?/g, '\n').replace(/\n\s*\n+/g, '\n').replace(/-->/g, '->').trim();
}

/**
 * WebVTT cue text: cueText, then '&' (a character reference) and '<' (a tag, which hides the rest of the cue)
 * escaped, '&' first, and '>' too, as the server's formatter does (src/formatters.py _vtt_escape).
 * SRT has no escape syntax and keeps the text as written.
 */
function vttText(text) {
  return cueText(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/**
 * Segments with usable times: text trimmed, empty ones dropped, every end after its start (a missing or
 * backwards end becomes the next segment's start, else start + 0.5 s), so subtitle players accept them.
 * @param {{text: string, start?: number, end?: number}[]} segments
 * @returns {{text: string, start: number, end: number}[]}
 */
export function cues(segments) {
  const list = (segments || [])
    .map((s) => ({ text: String(s?.text ?? '').trim(), start: seconds(s?.start), end: seconds(s?.end) }))
    .filter((s) => s.text);
  return list.map((s, i) => {
    let end = s.end;
    if (end <= s.start) {
      const next = list[i + 1];
      end = next && next.start > s.start ? next.start : s.start + MIN_CUE_S;
    }
    return { text: s.text, start: s.start, end };
  });
}

/** Plain text: every segment joined with spaces (the old Final text). */
export function toTxt(segments) {
  const text = cues(segments).map((s) => s.text).join(' ');
  return text ? `${text}\n` : '';
}

/** SubRip: numbered cues, '00:00:03,120 --> 00:00:05,400'. */
export function toSrt(segments) {
  return cues(segments)
    .map((s, i) => `${i + 1}\n${cueTime(s.start, ',')} --> ${cueTime(s.end, ',')}\n${cueText(s.text)}\n\n`)
    .join('');
}

/** WebVTT: the header, then '00:00:03.120 --> 00:00:05.400' cues, their text escaped (vttText). */
export function toVtt(segments) {
  return `WEBVTT\n\n${cues(segments)
    .map((s) => `${cueTime(s.start, '.')} --> ${cueTime(s.end, '.')}\n${vttText(s.text)}\n\n`)
    .join('')}`;
}

/**
 * JSON: {text, segments: [{start, end, text}], model, duration}.
 * @param {{segments: object[], model?: string, duration?: number}} transcript
 * @returns {string}
 */
export function toJson(transcript) {
  const t = transcript || {};
  const list = cues(t.segments);
  const round = (n) => Math.round(n * 1000) / 1000;
  const duration = seconds(t.duration) || (list.length ? list[list.length - 1].end : 0);
  return `${JSON.stringify({
    text: list.map((s) => s.text).join(' '),
    segments: list.map((s) => ({ start: round(s.start), end: round(s.end), text: s.text })),
    model: String(t.model || ''),
    duration: round(duration),
  }, null, 2)}\n`;
}

/** 'transcript-20261022-140512.srt' (local time). */
export function fileName(ext, date = new Date()) {
  const two = (n) => String(n).padStart(2, '0');
  const stamp = `${date.getFullYear()}${two(date.getMonth() + 1)}${two(date.getDate())}-${two(date.getHours())}${two(date.getMinutes())}${two(date.getSeconds())}`;
  return `transcript-${stamp}.${ext}`;
}

/**
 * One export, ready to save.
 * @param {'txt'|'srt'|'vtt'|'json'} kind
 * @param {{segments: object[], model?: string, duration?: number}} transcript
 * @param {Date} [date]
 * @returns {{content: string, mime: string, filename: string}}
 */
export function exportFile(kind, transcript, date = new Date()) {
  const format = FORMATS[kind];
  if (!format) throw new TypeError(`Unknown export format: ${kind}`);
  const segments = transcript?.segments || [];
  const content = kind === 'txt' ? toTxt(segments)
    : kind === 'srt' ? toSrt(segments)
      : kind === 'vtt' ? toVtt(segments)
        : toJson(transcript);
  return { content, mime: format.mime, filename: fileName(format.ext, date) };
}
