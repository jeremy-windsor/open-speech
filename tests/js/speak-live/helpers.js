// Shared setup for the Speak: Live tests: a fake browser with a Speak section holding a fake editor (Speak's view
// owns the real #sp-input) and the #sp-live-slot, Kokoro listed as loaded (so models.ensureReady takes its
// no-request fast path), the real speak-live partial served from disk, and a scripted Live server.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, resetAll, STATIC_JS, REPO_ROOT, flushMicrotasks } from '../harness/env.js';
import { json, text, FakeAudioContext } from '../harness/fakes.js';

export { json, text, FakeAudioContext, flushMicrotasks };

export const KOKORO_CAPS = Object.freeze({ live_reader: true, streaming: false, voice_blend: true });

export const SPEAK_HTML = `
<main id="main" tabindex="-1">
  <section class="view" id="view-speak" data-view="speak">
    <h1 id="sp-title" data-view-heading tabindex="-1">Speak</h1>
    <label for="sp-input">Text to speak</label>
    <textarea id="sp-input"></textarea>
    <p id="sp-counter" aria-live="off">0 characters</p>
    <div id="sp-live-slot" data-slot="speak-live"></div>
  </section>
</main>`;

/** Import a module from src/static/js (the same instance the code under test uses). */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

/** Serve the real partials from disk. */
export function partialRoute() {
  return (req) => {
    const name = req.path.split('/').pop();
    return text(200, readFileSync(resolve(REPO_ROOT, 'src/static/partials', name), 'utf8'), { 'content-type': 'text/html; charset=utf-8' });
  };
}

/**
 * The fake browser for a Live test.
 * @param {object} [options] installBrowserEnv options; `routes` are added to the defaults; `key` stores an API key
 */
export async function liveEnv({ routes = {}, key = '', seed = {}, ...options } = {}) {
  let tickets = 0;
  const env = installBrowserEnv({
    html: SPEAK_HTML,
    seed: key ? { 'open-speech-api-key': key, ...seed } : seed,
    routes: {
      'GET /static/partials/:name': partialRoute(),
      'POST /api/auth/ws-ticket': () => { tickets += 1; return json(200, { ticket: `t${tickets}`, expires_in: 60 }); },
      'GET /api/models/:id/status': () => json(200, { id: 'kokoro', state: 'loaded' }),
      ...routes,
    },
    ...options,
  });
  await resetAll();
  const auth = await load('core/auth.js');
  auth.init();
  const { app } = await load('core/store.js');
  app.set({
    route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' },
    connection: { ...app.get().connection, state: 'online' },
    models: {
      list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' }],
      defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '',
    },
  });
  env.app = app;
  env.editor = document.getElementById('sp-input');
  env.slot = document.getElementById('sp-live-slot');
  return env;
}

/** Let promises, due timers and animation frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/** A voice config like Speak's (the Speak view's getVoiceConfig()). */
export function voiceConfig(extra = {}) {
  return { model: 'kokoro', voice: 'af_heart', speed: 1, instructions: '', voice_library_ref: '', caps: { ...KOKORO_CAPS }, ...extra };
}

/** Base64 of `n` PCM16 samples (a ramp), as the server sends them. */
export function pcm(n = 2400, value = 1000) {
  const buf = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i += 1) buf.writeInt16LE(Math.max(-32768, Math.min(32767, value + i)), i * 2);
  return buf.toString('base64');
}

/** The JSON events a socket sent, optionally of one type. */
export function sent(ws, type) {
  const all = ws.sentJson;
  return type ? all.filter((e) => e.type === type) : all;
}

/** The scripted server side of one Live socket. */
export function server(ws) {
  return {
    ws,
    open() { ws.open(); },
    created(limits = { max_buffer_chars: 8192 }) {
      ws.serverSend({ type: 'session.created', session: { id: 'sess1', model: 'kokoro', limits } });
    },
    updated() { ws.serverSend({ type: 'session.updated', session: { id: 'sess1' } }); },
    accepted(bufferedChars = 0, generation = 0) {
      ws.serverSend({ type: 'input_text.accepted', accepted_chars: bufferedChars, buffered_chars: bufferedChars, generation });
    },
    status(bufferedChars = 0, generation = 0) {
      ws.serverSend({ type: 'session.status', buffered_chars: bufferedChars, generation, pending_segments: 0, unacked_seconds: 0 });
    },
    /** `range`: the response's source range in the generation's text (code points, as the server counts). */
    response(id, textValue, generation = 0, range = {}) {
      const start = range.start ?? 0;
      const end = range.end ?? start + textValue.length;
      ws.serverSend({ type: 'response.created', response: { id, status: 'in_progress', generation, source_start: start, source_end: end, text: textValue } });
    },
    delta(id, sequence, { generation = 0, samples = 2400, sampleRate = 24000, textValue } = {}) {
      const event = { type: 'response.output_audio.delta', response_id: id, generation, sequence, sample_rate: sampleRate, delta: pcm(samples) };
      if (textValue !== undefined) event.text = textValue;
      ws.serverSend(event);
    },
    done(id, generation = 0, status = 'completed') {
      ws.serverSend({ type: 'response.done', response: { id, status, generation } });
    },
    cancelled(generation = 0) { ws.serverSend({ type: 'response.cancelled', generation, accepted_chars: 0, response_id: null }); },
    error(code, message = 'Something failed', extra = {}) {
      ws.serverSend({ type: 'error', error: { code, message, ...extra } });
    },
    close(code, reason = '') { ws.serverClose(code, reason); },
  };
}

/** The toast messages on screen. */
export function toasts() {
  return Array.from(document.querySelectorAll('.toast__msg')).map((el) => el.textContent);
}

/** A promise with its resolve function (for an ensureReady the test controls). */
export function deferred() {
  let resolveFn;
  let rejectFn;
  const promise = new Promise((res, rej) => { resolveFn = res; rejectFn = rej; });
  return { promise, resolve: resolveFn, reject: rejectFn };
}
