/**
 * The transcript reducer, pure: what the stream and the upload produced, and nothing about
 * the page. Segments are committed units; `confirmed` is the utterance confirmed so far (is_final without
 * speech_final), shown as a provisional last line; `interim` is the live hypothesis.
 *
 *   transcript is_final:false        -> interim
 *   transcript is_final, !speech_final -> confirmed (replaces the pending utterance), not yet in the log
 *   transcript speech_final            -> appends a segment, clears confirmed and interim
 *
 * A new microphone session continues a microphone transcript (its times follow on from the earlier
 * sessions), so pressing Record again never throws away what was said; a file, or a recording after a
 * file, starts over. Copy and export include every utterance (the segments plus a pending one).
 * @module views/transcribe/transcript
 */
import { spokenDuration, plural } from '../../core/format.js';

/** @returns {object} an empty transcript */
export function createState() {
  return Object.freeze({
    source: '',          // '' | 'mic' | 'file'
    segments: Object.freeze([]),
    confirmed: null,     // {text, start, end} | null
    interim: '',
    vad: '',             // '' | 'speech' | 'silence'
    offset: 0,           // seconds added to the times of the current microphone session
    duration: 0,         // seconds of audio (file) or of recording (every microphone session)
    model: '',
    processingMs: null,
    fileName: '',
  });
}

const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);

function segment(text, start, end, source) {
  return Object.freeze({ text: String(text ?? '').trim(), start: Math.max(0, num(start)), end: Math.max(0, num(end)), source });
}

function freeze(state, patch) {
  return Object.freeze({ ...state, ...patch });
}

/**
 * @param {object} state
 * @param {{type: string}} action
 *   {type: 'session-start', model}            a microphone session began
 *   {type: 'transcript', message}             a /v1/audio/stream transcript event
 *   {type: 'vad', state: 'speech_start'|…}    a vad event
 *   {type: 'session-end', duration}           the session ended (stopped, failed): a pending utterance is kept
 *   {type: 'file-result', segments, text, duration, model, processingMs, fileName}
 *   {type: 'clear'} | {type: 'restore', state}
 * @returns {object} the next state (frozen); the same object when nothing changed
 */
export function reduce(state, action) {
  const s = state || createState();
  const a = action || {};
  switch (a.type) {
    case 'session-start': {
      const continuing = s.source === 'mic' && hasContent(s);
      const base = continuing ? s : createState();
      return freeze(base, {
        source: 'mic',
        segments: continuing ? exportSegments(s) : base.segments,
        confirmed: null,
        interim: '',
        vad: '',
        offset: continuing ? Math.max(s.duration, lastEnd(s)) : 0,
        model: a.model || base.model,
        processingMs: null,
        fileName: '',
      });
    }
    case 'transcript': {
      const m = a.message || {};
      const text = String(m.text ?? '');
      if (m.is_final !== true) {
        return text === s.interim ? s : freeze(s, { interim: text });
      }
      const start = s.offset + num(m.start);
      const end = s.offset + num(m.end);
      if (m.speech_final === true) {
        const committed = text.trim() ? [...s.segments, segment(text, start, end, 'mic')]
          : (s.confirmed ? [...s.segments, s.confirmed] : s.segments);
        return freeze(s, { segments: Object.freeze(committed), confirmed: null, interim: '' });
      }
      return freeze(s, { confirmed: text.trim() ? segment(text, start, end, 'mic') : null, interim: '' });
    }
    case 'vad': {
      const vad = a.state === 'speech_start' ? 'speech' : 'silence';
      return vad === s.vad ? s : freeze(s, { vad });
    }
    case 'session-end': {
      const segments = s.confirmed ? Object.freeze([...s.segments, s.confirmed]) : s.segments;
      const duration = Math.max(s.duration, s.offset + num(a.duration), segments.length ? segments[segments.length - 1].end : 0);
      return freeze(s, { segments, confirmed: null, interim: '', vad: '', duration });
    }
    case 'file-result': {
      const list = (a.segments || []).map((x) => segment(x.text, x.start, x.end, 'file')).filter((x) => x.text);
      const segments = list.length || !String(a.text ?? '').trim()
        ? list
        : [segment(a.text, 0, num(a.duration), 'file')];
      const duration = num(a.duration) || (segments.length ? segments[segments.length - 1].end : 0);
      return freeze(createState(), {
        source: 'file',
        segments: Object.freeze(segments),
        duration,
        model: a.model || '',
        processingMs: Number.isFinite(a.processingMs) ? a.processingMs : null,
        fileName: String(a.fileName || ''),
      });
    }
    case 'clear':
      return createState();
    case 'restore':
      return a.state || createState();
    default:
      return s;
  }
}

function lastEnd(state) {
  const list = state.segments;
  return list.length ? list[list.length - 1].end : 0;
}

/** Every utterance: the segments plus the pending (confirmed) one. */
export function exportSegments(state) {
  const s = state || createState();
  return Object.freeze(s.confirmed ? [...s.segments, s.confirmed] : [...s.segments]);
}

/** Whether there is anything to copy or export. */
export function hasContent(state) {
  return exportSegments(state).some((x) => x.text);
}

/** Every utterance joined with spaces (Copy and the TXT export). */
export function fullText(state) {
  return exportSegments(state).map((x) => x.text).filter(Boolean).join(' ');
}

/**
 * What a screen reader hears when a file transcript lands, instead of the whole text:
 * 'Transcript ready: 12 segments, 2 minutes 13 seconds'.
 */
export function summary(state) {
  const s = state || createState();
  const parts = [plural(s.segments.length, 'segment')];
  if (s.duration > 0) parts.push(spokenDuration(s.duration));
  return `Transcript ready: ${parts.join(', ')}`;
}

/**
 * The provisional line under the log: the pending utterance, then the part of the live hypothesis that
 * goes beyond it (the server's hypothesis repeats the confirmed words). With interim text hidden only the
 * confirmed words show.
 * @returns {{confirmed: string, interim: string}}
 */
export function pendingLine(state, { showInterim = true } = {}) {
  const s = state || createState();
  const confirmed = s.confirmed ? s.confirmed.text : '';
  if (!showInterim || !s.interim) return { confirmed, interim: '' };
  if (!confirmed) return { confirmed: '', interim: s.interim.trim() };
  const tail = s.interim.startsWith(confirmed) ? s.interim.slice(confirmed.length) : '';
  return { confirmed, interim: tail.trim() };
}

/** The index of the segment playing at `position` seconds, or -1. */
export function segmentAt(state, position) {
  const list = (state || createState()).segments;
  const t = num(position);
  for (let i = list.length - 1; i >= 0; i -= 1) {
    if (t >= list[i].start) return t < Math.max(list[i].end, list[i].start + 0.5) ? i : -1;
  }
  return -1;
}
