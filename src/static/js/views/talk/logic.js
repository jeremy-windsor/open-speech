/**
 * Talk: the pure parts. No DOM, no storage, no requests.
 *
 * Talk speaks GA /v1/realtime (src/realtime/server.py with no subprotocol): 24 kHz mono PCM16 both ways, server
 * turn detection that replies after each turn (create_response) and can be talked over (interrupt_response), the
 * session's voice and model, and the AI preset (`assistant: {preset}`). A typed question is
 * conversation.item.create with a user input_text message. An empty session.update is the
 * keep-alive: the server's idle timer (OS_REALTIME_IDLE_TIMEOUT_S, 120 s by default) restarts on any client
 * event, and an update that names nothing changes nothing.
 * @module views/talk/logic
 */
import { floatToInt16 } from '../../core/pcm.js';
import { voiceTitle, RECOMMENDED_VOICES } from '../../core/voices.js';
import { providerOf } from '../../core/models.js';

export const REALTIME_PATH = '/v1/realtime';
/** The only PCM rate /v1/realtime takes and sends (src/realtime/session.py parse_audio_format). */
export const AUDIO_RATE = 24000;
/** A typed question's limit: the server's ASSISTANT_PROMPT_MAX_CHARS. */
export const TEXT_MAX = 8000;
/** The microphone goes out in appends of 100 ms. */
export const SEND_SAMPLES = 2400;
/** A keep-alive goes out once nothing was sent for this long; the check runs every KEEPALIVE_CHECK_MS. */
export const KEEPALIVE_MS = 30000;
export const KEEPALIVE_CHECK_MS = 10000;
/** The waits before each try after a dropped connection; then Talk stops and says so. */
export const RECONNECT_MS = Object.freeze([500, 2000, 5000]);
/** The log keeps the newest entries only. */
export const MAX_LOG = 200;
/** Turning the microphone off mid-turn sends this much silence, so the server's VAD (500 ms) ends the turn. */
export const END_TURN_SILENCE_MS = 800;
/** Delete chat with Talk stopped offers Undo for this long. */
export const UNDO_MS = 5000;
export const DEFAULT_PRESET = 'voice';
/** The preset a book job uses; it isn't offered for talking. */
export const BOOK_PRESET = 'book';

export const COPY = Object.freeze({
  title: 'Talk stopped',
  reconnected: 'Reconnected: the conversation started over.',
  newSession: "A new conversation started. The AI doesn't remember the one above.",
  dropped: 'The connection to the server dropped.',
  settingsRefused: "The session's settings weren't accepted.",
  disabled: 'Talking to the AI is turned off on this server (OS_REALTIME_ENABLED).',
  busy: 'Too many conversations are open on this server. Try again shortly.',
  noAudio: "This browser can't play the AI's voice.",
  vad: "The server couldn't load voice detection, so it can't tell when you stop talking. Type your questions instead.",
  transcriptionFailed: "Couldn't write down what you said.",
  ttsError: "Couldn't speak the answer.",
  serverError: 'Something went wrong on the server.',
  micTitle: 'Microphone problem',
  canType: 'You can still type your questions.',
  recordingStatus: 'Stop recording to listen',
  interrupted: 'Interrupted',
  stopped: 'Stopped',
  serverDefault: 'Server default',
  chatDeleted: 'Chat deleted',
  chatDeletedRunning: "Chat deleted. The AI doesn't remember it.",
  chatRestored: 'Chat restored',
  undo: 'Undo',
  cutOff: 'The answer was cut off.',
  cutOffLimit: 'The answer was cut off: the AI reached its token limit.',
});

/** The code of a streamed answer the token limit cut off (src/assistant/errors.py ASSISTANT_REPLY_CUT_OFF). */
export const CUT_OFF_CODE = 'assistant_reply_cut_off';

const OWNER_NAMES = Object.freeze({ transcribe: 'Transcribe', voices: 'Voices', talk: 'Talk' });

/** PURE: why another view's capture blocks Talk ('Stop the Transcribe recording first.'). */
export function recordingRefusal(recording) {
  const owner = recording && recording.owner;
  if (!owner || owner === 'talk') return '';
  return `Stop the ${OWNER_NAMES[owner] || 'other'} recording first.`;
}

/**
 * A stateful resampler for the microphone's frames (its AudioContext runs at the device's rate, usually 48 kHz or
 * 44.1 kHz). Going down it averages the input over each output sample's span (a box filter, which keeps most of
 * the aliasing out of speech); going up it interpolates linearly. Samples a frame leaves over join the next one.
 * @param {number} fromRate
 * @param {number} toRate
 * @returns {(input: Float32Array) => Float32Array} always a new array
 */
export function createResampler(fromRate, toRate) {
  const from = Number(fromRate) > 0 ? Number(fromRate) : toRate;
  if (from === toRate) return (input) => Float32Array.from(input || []);
  const ratio = from / toRate;
  let carry = new Float32Array(0);
  let pos = 0;   // where the next output sample starts, in input samples from the start of `carry`
  return (input) => {
    const data = new Float32Array(carry.length + (input ? input.length : 0));
    data.set(carry, 0);
    if (input) data.set(input, carry.length);
    const out = [];
    if (ratio > 1) {
      while (pos + ratio <= data.length + 1e-9) {
        const end = pos + ratio;
        let acc = 0;
        let x = pos;
        while (x < end - 1e-9) {
          const i = Math.floor(x + 1e-9);
          const next = Math.min(end, i + 1);
          acc += (data[i] || 0) * (next - x);
          x = next;
        }
        out.push(acc / ratio);
        pos = end;
      }
    } else {
      while (pos + 1 < data.length) {
        const i = Math.floor(pos);
        const frac = pos - i;
        out.push(data[i] * (1 - frac) + data[i + 1] * frac);
        pos += ratio;
      }
    }
    const keep = Math.min(Math.floor(pos + 1e-9), data.length);
    carry = data.slice(keep);
    pos -= keep;
    return Float32Array.from(out);
  };
}

/**
 * PURE: little-endian PCM16 as base64.
 * @param {Int16Array} samples
 * @returns {string}
 */
export function int16ToBase64(samples) {
  const n = samples ? samples.length : 0;
  const bytes = new Uint8Array(n * 2);
  for (let i = 0; i < n; i += 1) {
    const v = samples[i];
    bytes[2 * i] = v & 0xff;
    bytes[2 * i + 1] = (v >> 8) & 0xff;
  }
  let binary = '';
  const STEP = 0x8000;
  for (let i = 0; i < bytes.length; i += STEP) binary += String.fromCharCode.apply(null, bytes.subarray(i, i + STEP));
  return btoa(binary);
}

/** PURE: an input_audio_buffer.append of float samples at 24 kHz (clamped to PCM16). */
export function appendEvent(samples) {
  return { type: 'input_audio_buffer.append', audio: int16ToBase64(floatToInt16(samples || new Float32Array(0))) };
}

/**
 * PURE: the presets Talk offers: every AI preset but the book preset, in the server's order ({id, name}).
 * @param {object[]} list GET /api/assistant/presets' `presets`
 */
export function voicePresets(list) {
  const out = (Array.isArray(list) ? list : [])
    .filter((p) => p && typeof p.id === 'string' && p.id && p.id !== BOOK_PRESET)
    .map((p) => ({ id: p.id, name: String(p.name || p.id) }));
  return out.length ? out : [{ id: DEFAULT_PRESET, name: 'Voice' }];
}

/**
 * PURE: the preset Talk uses: `wanted` (the remembered or chosen one) while the list
 * offers it; else the default (`voice`); else the first preset offered.
 * @param {object[]} list GET /api/assistant/presets' `presets`
 * @param {string} wanted
 * @returns {string}
 */
export function pickPreset(list, wanted) {
  const ids = voicePresets(list).map((p) => p.id);
  if (wanted && ids.includes(wanted)) return wanted;
  return ids.includes(DEFAULT_PRESET) ? DEFAULT_PRESET : ids[0];
}

/**
 * PURE: the session.update for Talk's choices.
 * @param {{model?: string, voice?: string, preset?: string, interrupt?: boolean}} choices
 * @param {{turnDetection?: boolean, eventId?: string}} [options] turnDetection false leaves turn detection as the
 *   server has it (after it said it has none: asking again would only fail again)
 */
export function sessionUpdate(choices = {}, { turnDetection = true, eventId = '' } = {}) {
  const c = choices || {};
  const input = { format: { type: 'audio/pcm', rate: AUDIO_RATE } };
  if (turnDetection) input.turn_detection = { type: 'server_vad', create_response: true, interrupt_response: c.interrupt !== false };
  const output = { format: { type: 'audio/pcm', rate: AUDIO_RATE } };
  if (c.voice) output.voice = String(c.voice);
  const session = { type: 'realtime' };
  if (c.model) session.model = String(c.model);
  session.audio = { input, output };
  session.assistant = { preset: c.preset ? String(c.preset) : DEFAULT_PRESET };
  return eventId ? { type: 'session.update', event_id: eventId, session } : { type: 'session.update', session };
}

/** PURE: the keep-alive: a session.update that changes nothing. */
export function keepalive() {
  return { type: 'session.update', session: {} };
}

/**
 * PURE: a typed question: the item and event ids match the answer to it.
 * @param {string} text
 * @param {number} n this session's count
 */
export function typedQuestion(text, n) {
  const itemId = `typed_${n}`;
  const eventId = `evt_typed_${n}`;
  return {
    itemId,
    eventId,
    event: { type: 'conversation.item.create', event_id: eventId, item: { id: itemId, type: 'message', role: 'user', content: [{ type: 'input_text', text: String(text) }] } },
  };
}

function sentence(text) {
  const s = String(text || '').trim();
  if (!s) return '';
  return /[.!?…]$/.test(s) ? s : `${s}.`;
}

/**
 * PURE: what a microphone failure (core/mic.js MicError or MicBusyError) says, and that typing still works.
 * @returns {{title: string, message: string}}
 */
export function micProblem(error) {
  const text = sentence(error && error.message) || "The microphone couldn't start.";
  return { title: COPY.micTitle, message: `${text} ${COPY.canType}` };
}

/**
 * PURE: the log entry for a server error during an answer. An AI error (assistant_*) reads as the preset's
 * spoken sentence when the preset speaks errors, with the server's words as detail; otherwise the server's words.
 * @param {{code?: string, message?: string}} err the error event's `error`
 * @param {{speak_errors?: boolean, error_sentences?: object}|null} preset
 * @returns {{text: string, detail: string}}
 */
export function errorEntry(err, preset) {
  const code = String((err && err.code) || '');
  const message = String((err && err.message) || '');
  if (code === 'tts_error') return { text: COPY.ttsError, detail: message };
  if (code.startsWith('assistant_') && preset && preset.speak_errors !== false) {
    const said = preset.error_sentences && typeof preset.error_sentences[code] === 'string' ? preset.error_sentences[code].trim() : '';
    if (said) return { text: said, detail: message };
  }
  return { text: message || COPY.serverError, detail: '' };
}

/**
 * PURE: the note under an answer that was cut off after some of it was spoken: a
 * streamed reply that an error or the token limit stopped ends `incomplete`, with the code in
 * status_details.error.code. The token limit reads in plain words; another code reads as its error
 * entry would (the preset's sentence when it speaks errors, else the plain message); no code, that it was cut off.
 * @param {string} code
 * @param {{speak_errors?: boolean, error_sentences?: object}|null} preset
 * @returns {string}
 */
export function cutOffNote(code, preset) {
  const c = String(code || '');
  if (c === CUT_OFF_CODE) return COPY.cutOffLimit;
  if (!c) return COPY.cutOff;
  return errorEntry({ code: c, message: '' }, preset).text;
}

/**
 * PURE: the voice Talk replies in: the one chosen in Talk, else Speak's (app.speakVoice; never a raw reference
 * recording, which the session can't use), else the default model's (Kokoro's recommended voice; for another
 * model the server's default).
 * @param {{chosen?: object|null, speak?: object|null, defaultModel?: string}} input
 * @returns {{model: string, voice: string, label: string, source: 'chosen'|'speak'|'default'}}
 */
export function talkVoice({ chosen = null, speak = null, defaultModel = '' } = {}) {
  if (chosen && chosen.voice) {
    return { model: String(chosen.model || ''), voice: String(chosen.voice), label: voiceTitle(chosen.model, chosen.voice, chosen.label), source: 'chosen' };
  }
  const s = speak || {};
  if (s.model && s.voice && s.kind !== 'recording') {
    return { model: String(s.model), voice: String(s.voice), label: String(s.label || voiceTitle(s.model, s.voice)), source: 'speak' };
  }
  const model = String(defaultModel || '');
  const voice = model && providerOf(model) === 'kokoro' ? RECOMMENDED_VOICES[0] : '';
  return { model, voice, label: voice ? voiceTitle(model, voice) : COPY.serverDefault, source: 'default' };
}

function lowerFirst(text) {
  return /^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text;
}

/**
 * PURE: the status pill and the spoken status. The label follows every turn ("Hearing you…", "Thinking…",
 * "Speaking"); the spoken status changes with the phase only (Connecting, Listening or Connected, Paused,
 * Stopped), so a screen reader never talks over the answer or the listener.
 * @param {object} state the engine's state
 * @returns {{label: string, tone: string, spoken: string}}
 */
export function statusOf(state) {
  const st = state || {};
  if (st.phase === 'stopped' || !st.phase) {
    const reason = String(st.stopReason || '').trim();
    const label = reason && reason !== 'Stopped' ? `Stopped: ${lowerFirst(reason)}` : 'Stopped';
    return { label, tone: st.error ? 'error' : 'idle', spoken: label };
  }
  if (st.phase === 'preparing' || st.phase === 'connecting') return { label: 'Connecting…', tone: 'busy', spoken: 'Connecting…' };
  if (st.phase === 'reconnecting') return { label: 'Reconnecting…', tone: 'busy', spoken: 'Reconnecting…' };
  const listening = st.mic === 'on' || st.mic === 'starting';
  const spoken = st.paused ? 'Paused' : listening ? 'Listening' : 'Connected';
  if (st.paused) return { label: 'Paused', tone: 'idle', spoken };
  if (st.reply === 'thinking') return { label: 'Thinking…', tone: 'busy', spoken };
  if (st.reply === 'speaking') return { label: 'Speaking', tone: 'ok', spoken };
  if (st.hearing) return { label: 'Hearing you…', tone: 'rec', spoken };
  if (st.transcribing) return { label: 'Writing down your words…', tone: 'busy', spoken };
  if (st.mic === 'starting') return { label: 'Starting the microphone…', tone: 'busy', spoken };
  if (st.mic === 'on') return { label: 'Listening', tone: 'ok', spoken };
  return { label: 'Connected', tone: 'ok', spoken };
}
