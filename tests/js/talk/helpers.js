// Shared setup for the Talk tests: the real shell markup with the Talk partial, the
// real microphone module (core/mic.js) on the harness's fake media devices and AudioWorklet, the real audio
// coordinator (core/audio.js), and a fake WebSocket that plays the GA /v1/realtime events the server sends
// (src/realtime/server.py). No real server, AI or network is ever reached.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, flushMicrotasks, STATIC_JS, REPO_ROOT } from '../harness/env.js';
import { json, text } from '../harness/fakes.js';

export { json, text };

/** Import a module from src/static/js: the same instance the code under test uses. */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

export const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/talk.html'), 'utf8');

export const PRESETS = [
  { id: 'voice', name: 'Voice', builtin: true, speak_errors: true, show_thinking: true,
    error_sentences: { assistant_unreachable: "I couldn't reach the AI server.", assistant_timeout: 'The AI took too long to answer.' } },
  { id: 'book', name: 'Book', builtin: true, speak_errors: true, error_sentences: {} },
  { id: 'short', name: 'Short answers', builtin: false, speak_errors: false, error_sentences: { assistant_unreachable: 'No AI.' } },
];

export const STATUS_ON = { enabled: true, host: '127.0.0.1:9', model: 'your-model', server_settings: {}, locked: false, sources: {} };

export const MODELS = [
  { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
  { id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' },
];

/** The routes Talk calls: the assistant's status and presets, model status, and WebSocket tickets. */
export function routes(extra = {}) {
  return {
    'GET /api/assistant': () => json(200, STATUS_ON),
    'GET /api/assistant/presets': () => json(200, { presets: PRESETS }),
    'GET /api/models/:id/status': () => json(200, { state: 'loaded' }),
    'POST /api/auth/ws-ticket': () => json(200, { ticket: `ticket-${Math.random().toString(36).slice(2, 8)}`, expires_in: 60 }),
    ...extra,
  };
}

/** Let promises, timers due now and frames settle; `ms` advances the fake clock. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/** The fake browser with the real shell markup (its live regions, dialogs and audio host). */
export function talkEnv({ routes: extra = {}, ...options } = {}) {
  return installBrowserEnv({ document: loadHtml('src/static/index.html'), routes: routes(extra), audio: { sampleRate: 48000 }, ...options });
}

export const CHOICES = Object.freeze({ model: 'kokoro', voice: 'af_bella', label: 'Bella', preset: 'voice', interrupt: true });

/**
 * A Talk engine on the fake browser.
 * @param {object} [options] {choices, env (installBrowserEnv options), engine (more createTalkEngine options)}
 */
export async function engineSetup({ choices = CHOICES, env: envOptions = {}, engine: engineOptions = {} } = {}) {
  const env = talkEnv(envOptions);
  const mod = await importFresh('views/talk/session.js');
  const audio = await load('core/audio.js');
  const mic = await load('core/mic.js');
  const { app } = await load('core/store.js');
  app.set({ models: { list: MODELS, defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' } });
  const t = { env, mod, audio, mic, app, choices: { ...choices }, states: [], readyCalls: [] };
  t.engine = mod.createTalkEngine({
    getChoices: () => ({ ...t.choices, presetInfo: PRESETS.find((p) => p.id === t.choices.preset) || null }),
    onChange: (state) => t.states.push(state),
    ensureReady: async (model, opts) => { t.readyCalls.push([model, opts]); return true; },
    ...engineOptions,
  });
  t.state = () => t.engine.getState();
  return t;
}

/** The newest socket. */
export function lastSocket(env) {
  return env.sockets[env.sockets.length - 1] || null;
}

/** The JSON events a socket was sent, without the microphone's appends unless asked. */
export function sent(ws, { appends = false } = {}) {
  return ws.sentJson.filter((e) => appends || e.type !== 'input_audio_buffer.append');
}

export function sentTypes(ws, options) {
  return sent(ws, options).map((e) => e.type);
}

/** The microphone's AudioContext: the one with the capture worklet. */
export function micContext(env) {
  return env.audioContexts.find((ctx) => ctx.__worklets.length && ctx.state !== 'closed') || null;
}

/** The AudioContext Talk plays the replies on: the first one it made, which has no worklet. */
export function playbackContext(env) {
  const open = env.audioContexts.filter((ctx) => !ctx.__worklets.length && !ctx.__modules.length);
  return open[open.length - 1] || null;
}

/** One frame from the microphone (the AudioWorklet posts it). */
export function micFrame(env, samples = new Float32Array(2048).fill(0.25)) {
  const ctx = micContext(env);
  const node = ctx && ctx.__worklets[ctx.__worklets.length - 1];
  if (!node) throw new Error('micFrame(): the microphone is not open');
  node.__emit(samples);
}

/** Base64 PCM16 of `n` samples of one value (the server's audio deltas, 24 kHz). */
export function pcm(n = 2400, value = 1000) {
  const bytes = new Uint8Array(n * 2);
  for (let i = 0; i < n; i += 1) {
    bytes[2 * i] = value & 0xff;
    bytes[2 * i + 1] = (value >> 8) & 0xff;
  }
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

/** Decode a base64 PCM16 append back to its sample count. */
export function samplesIn(base64) {
  return atob(base64).length / 2;
}

/** The GA server events (src/realtime/server.py with a GA session: no subprotocol). */
export const ga = {
  created: (extra = {}) => ({ type: 'session.created', session: session(extra) }),
  updated: (extra = {}) => ({ type: 'session.updated', session: session(extra) }),
  speechStarted: (item = 'item_1') => ({ type: 'input_audio_buffer.speech_started', audio_start_ms: 100, item_id: item }),
  speechStopped: (item = 'item_1') => ({ type: 'input_audio_buffer.speech_stopped', audio_end_ms: 900, item_id: item }),
  committed: (item = 'item_1') => ({ type: 'input_audio_buffer.committed', previous_item_id: null, item_id: item }),
  audioItem: (item = 'item_1') => ({ type: 'conversation.item.added', previous_item_id: null, item: { id: item, object: 'realtime.item', type: 'message', role: 'user', status: 'completed', content: [{ type: 'input_audio', transcript: null }] } }),
  transcript: (item = 'item_1', transcript = 'What is the weather tomorrow?') => ({ type: 'conversation.item.input_audio_transcription.completed', item_id: item, content_index: 0, transcript }),
  transcriptFailed: (item = 'item_1') => ({ type: 'conversation.item.input_audio_transcription.failed', item_id: item, content_index: 0, error: { type: 'transcription_error', code: 'transcription_error', message: 'whisper fell over' } }),
  typedAdded: (id, words) => ({ type: 'conversation.item.added', previous_item_id: null, item: { id, object: 'realtime.item', type: 'message', role: 'user', status: 'completed', content: [{ type: 'input_text', text: words }] } }),
  typedDone: (id, words) => ({ type: 'conversation.item.done', previous_item_id: null, item: { id, object: 'realtime.item', type: 'message', role: 'user', status: 'completed', content: [{ type: 'input_text', text: words }] } }),
  responseCreated: (id = 'resp_1') => ({ type: 'response.created', response: { id, object: 'realtime.response', status: 'in_progress', output: [] } }),
  itemAdded: (id = 'resp_1') => ({ type: 'response.output_item.added', response_id: id, output_index: 0, item: { id: `ai_${id}`, type: 'message', role: 'assistant', status: 'in_progress', content: [] } }),
  partAdded: (id = 'resp_1') => ({ type: 'response.content_part.added', response_id: id, item_id: `ai_${id}`, output_index: 0, content_index: 0, part: { type: 'output_audio', transcript: '' } }),
  filler: (id = 'resp_1', words = 'Hmm.') => ({ type: 'open_speech.assistant.filler', response_id: id, item_id: `ai_${id}`, text: words }),
  thinking: (id = 'resp_1', words = 'They want the weather.') => ({ type: 'open_speech.assistant.thinking', response_id: id, item_id: `ai_${id}`, text: words }),
  textDelta: (id = 'resp_1', delta = 'Sunny, with a light wind.') => ({ type: 'response.output_audio_transcript.delta', response_id: id, item_id: `ai_${id}`, output_index: 0, content_index: 0, delta }),
  audio: (id = 'resp_1', n = 2400) => ({ type: 'response.output_audio.delta', response_id: id, item_id: `ai_${id}`, output_index: 0, content_index: 0, delta: pcm(n) }),
  audioDone: (id = 'resp_1') => ({ type: 'response.output_audio.done', response_id: id, item_id: `ai_${id}`, output_index: 0, content_index: 0 }),
  textDone: (id = 'resp_1', transcript = 'Sunny, with a light wind.') => ({ type: 'response.output_audio_transcript.done', response_id: id, item_id: `ai_${id}`, output_index: 0, content_index: 0, transcript }),
  done: (id = 'resp_1', status = 'completed', details = null) => ({ type: 'response.done', response: { id, object: 'realtime.response', status, output: [], ...(details ? { status_details: details } : {}) } }),
  error: (code, message, { type = 'invalid_request_error', event_id: eventId } = {}) => ({ type: 'error', error: { type, message, code, ...(eventId ? { event_id: eventId } : {}) } }),
};

function session({ turnDetection = { type: 'server_vad', threshold: 0.5, prefix_padding_ms: 300, silence_duration_ms: 500, create_response: true, interrupt_response: true }, voice = 'af_bella', model = 'kokoro', preset = 'voice' } = {}) {
  return {
    id: 'sess_1', type: 'realtime', model, output_modalities: ['audio'], instructions: null, assistant: { preset },
    audio: {
      input: { format: { type: 'audio/pcm', rate: 24000 }, transcription: { model: 'whisper-1' }, turn_detection: turnDetection },
      output: { format: { type: 'audio/pcm', rate: 24000 }, voice, speed: 1 },
    },
  };
}

/** The server accepts the socket and answers the client's session.update, as /v1/realtime does. */
export async function accept(env, ws, updated = {}) {
  ws.open();
  ws.serverSend(ga.created());
  await settle(env);
  ws.serverSend(ga.updated(updated));
  await settle(env);
}

/** Start (a click) and get to a configured session; returns the socket. */
export async function startConnected(t, { mic = true, updated = {} } = {}) {
  const ok = t.env.withGesture(() => t.engine.start({ mic }));
  assert.equal(ok, true, 'start() accepted');
  await settle(t.env);
  const ws = lastSocket(t.env);
  assert.ok(ws, 'a socket opened');
  await accept(t.env, ws, updated);
  return ws;
}

/** A whole spoken answer for one response, as the AI path sends it, in its order. */
export async function answer(env, ws, id = 'resp_1', { text: words = 'Sunny, with a light wind.', fillers = [], chunks = 2 } = {}) {
  ws.serverSend(ga.responseCreated(id));
  ws.serverSend(ga.itemAdded(id));
  ws.serverSend(ga.partAdded(id));
  for (const f of fillers) {
    ws.serverSend(ga.filler(id, f));
    ws.serverSend(ga.audio(id, 1200));
  }
  ws.serverSend(ga.textDelta(id, words));
  for (let i = 0; i < chunks; i += 1) ws.serverSend(ga.audio(id));
  ws.serverSend(ga.audioDone(id));
  ws.serverSend(ga.textDone(id, words));
  ws.serverSend(ga.done(id));
  await settle(env);
}

/** The engine's log, as [kind, text] pairs. */
export function logOf(state) {
  return state.log.map((e) => [e.kind, e.text]);
}

/** Every timer still waiting on the fake clock. */
export function pendingTimers(env) {
  return env.clock.pending();
}
