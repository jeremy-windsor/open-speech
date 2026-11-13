/**
 * Screenshot hook states for Transcribe: `#transcribe?shot=<state>` shows a state that
 * otherwise needs a microphone, a server stream or a long upload. Display only: no microphone opens, no
 * request is sent. Pure data (the titles are the recorder's own); index.js applies it.
 * @module views/transcribe/shots
 */
import { START_FAILED_TITLE } from './recorder.js';

/** The states this view handles (the shell ignores them). */
export const SHOTS = Object.freeze(['recording', 'done', 'uploading', 'error', 'nomodel']);

const MODEL = 'Systran/faster-whisper-base';

/** A microphone session 42 s in: two segments, a confirmed utterance and the live hypothesis. */
export function recordingShot(now) {
  return {
    rec: Object.freeze({
      phase: 'recording', model: MODEL, startedAt: now - 42000, stoppedAt: 0, sampleRate: 48000, sessionId: 'shot',
      vadEnabled: true, vad: 'speech', error: null,
    }),
    level: { db: -22, peak: 0.32 },
    transcript: {
      segments: [
        { text: 'Okay, so the plan for today is to finish the reader and then look at the speak view.', start: 3.1, end: 9.4 },
        { text: 'We should ship the reader first, because people listen on their phones every evening.', start: 11.2, end: 17.8 },
      ],
      confirmed: { text: 'And then the speak view', start: 19.6, end: 21.4 },
      interim: 'And then the speak view gets the new blend editor',
    },
  };
}

/** A finished file transcript (2:13, eight segments). */
export function doneShot() {
  const lines = [
    [0.0, 6.2, 'Welcome back. Today we are looking at how people listen to long documents on their phones.'],
    [6.2, 14.9, 'Most of them start in the evening, and most of them lock the screen within the first minute.'],
    [14.9, 27.3, 'That means the lock screen controls are the player, so they have to be right: play, pause, skip and the chapter scrubber.'],
    [27.3, 41.0, 'The second thing we heard is that voices matter more than speed. A calm voice at one and a quarter beats a fast flat one.'],
    [41.0, 58.6, 'Blends helped here. Two voices mixed two to one gave the narrator a warmer sound without losing clarity.'],
    [58.6, 79.4, 'Third, transcripts. People want timestamps they can click, and subtitles they can drop into a video editor.'],
    [79.4, 104.8, 'So the plan is simple: keep the reader fast, keep the voices good, and make every transcript easy to export.'],
    [104.8, 133.2, 'Thanks for listening. The notes and the full transcript are in the show description.'],
  ];
  return {
    segments: lines.map(([start, end, text]) => ({ text, start, end })),
    text: lines.map((l) => l[2]).join(' '),
    duration: 133.2,
    model: MODEL,
    processingMs: 4100,
    fileName: 'listening-research.m4a',
  };
}

/** A file 12 s into its upload. */
export function uploadingShot(now) {
  return { phase: 'uploading', fileName: 'interview-2026-10-21.m4a', startedAt: now - 12000, error: null };
}

/**
 * A recording the server ended with an error event (the model didn't load: BUG-02), and an upload behind a
 * proxy that answered with HTML.
 */
export function errorShot() {
  return {
    rec: {
      kind: 'stream', title: START_FAILED_TITLE,
      message: 'Failed to load model Systran/faster-whisper-large-v3: CUDA out of memory.', detail: '',
    },
    upload: {
      title: "Couldn't transcribe", message: 'The server returned an unexpected response (HTTP 502).',
      detail: '<html><head><title>502 Bad Gateway</title></head><body>nginx</body></html>', retryable: true, action: 'retry',
    },
  };
}
