// Shared setup for the Transcribe tests: the real shell markup (live regions, dialogs, the shared
// audio element) with the real partial mounted in #view-transcribe, the store seeded with a model
// inventory, and small drivers for the microphone, the socket and the upload.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { installBrowserEnv, loadHtml, importFresh, flushMicrotasks, STATIC_JS, REPO_ROOT } from '../harness/env.js';
import { json, text } from '../harness/fakes.js';
import { FakeEvent } from '../harness/dom.js';

export { json, text, FakeEvent };

/**
 * A short, bounded description of a node for a failure message: `<button#tr-record data-action="record">`,
 * `#text "Hello"`, `#document`, `null`.
 * @param {*} n
 * @returns {string}
 */
export function describeNode(n) {
  if (n === null || n === undefined) return String(n);
  if (typeof n !== 'object') return `${typeof n} ${String(n).slice(0, 60)}`;
  if (n.nodeType === 9) return '#document';
  if (n.nodeType === 3) return `#text ${JSON.stringify(String(n.textContent || '').slice(0, 40))}`;
  if (n.nodeType !== 1) return `node (type ${n.nodeType})`;
  const attr = (name) => (typeof n.getAttribute === 'function' ? n.getAttribute(name) : null);
  const id = attr('id');
  const action = attr('data-action');
  const cls = attr('class');
  const words = String(n.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 40);
  return `<${n.localName || '?'}${id ? `#${id}` : ''}${action ? ` data-action="${action}"` : ''}${!id && cls ? ` class="${cls}"` : ''}>${words ? ` "${words}"` : ''}`;
}

/**
 * Node identity checks for DOM nodes. Never pass a harness node to assert.equal/deepEqual: when it fails,
 * node:assert inspects actual and expected (depth 1000, getters on), and a harness node reaches the whole
 * document, window and env through its enumerable links, so a single failure allocates without bound (a
 * mutation run reached the 2 GB cap with RangeError "Array buffer allocation failed"). These fail with a short
 * description of each node instead.
 * @param {*} actual
 * @param {*} expected
 * @param {string} [message]
 */
export function sameNode(actual, expected, message = '') {
  if (actual === expected) return;
  assert.fail(`${message ? `${message}: ` : ''}got ${describeNode(actual)}, expected ${describeNode(expected)}`);
}

/** No such node (null or undefined), with the same bounded failure message as sameNode(). */
export function noNode(actual, message = '') {
  if (actual === null || actual === undefined) return;
  assert.fail(`${message ? `${message}: ` : ''}expected no node, got ${describeNode(actual)}`);
}

export const STT = 'Systran/faster-whisper-base';
export const ROUTE = Object.freeze({ view: 'transcribe', sub: '', id: '', path: 'transcribe', shot: '' });

export const MODELS = [
  { id: 'Systran/faster-whisper-small', type: 'stt', provider: 'faster-whisper', state: 'downloaded', device: 'cpu' },
  { id: STT, type: 'stt', provider: 'faster-whisper', state: 'loaded', device: 'cpu' },
  { id: 'Systran/faster-whisper-tiny', type: 'stt', provider: 'faster-whisper', state: 'available' },
  { id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded', device: 'cuda:0' },
];

/** Import a module from src/static/js: the same instance the view uses. */
export function load(rel) {
  return import(pathToFileURL(resolve(STATIC_JS, rel)).href);
}

export const PARTIAL = readFileSync(resolve(REPO_ROOT, 'src/static/partials/transcribe.html'), 'utf8');

/** Let promises, timers due now and frames settle. */
export async function settle(env, ms = 0) {
  await flushMicrotasks();
  env.raf.flush();
  await env.clock.tickAsync(ms);
  env.raf.flush();
  await flushMicrotasks();
}

/**
 * The fake browser with the shell markup and the Transcribe partial, the view initialised and shown.
 * @param {object} [options]
 * @param {object} [options.routes] fetch routes (a model status route is always added)
 * @param {object[]} [options.models] the inventory in app.models
 * @param {string} [options.sttDefault]
 * @param {object} [options.env] more installBrowserEnv options
 * @param {boolean} [options.show]
 */
export async function mountView({ routes = {}, models = MODELS, sttDefault = STT, env: envOptions = {}, show = true } = {}) {
  const allRoutes = {
    'GET /api/models/:id/status': (req) => json(200, { state: (models.find((m) => m.id === req.params.id) || {}).state || 'loaded' }),
    'POST /api/auth/ws-ticket': () => json(200, { ticket: `ticket-${Math.random().toString(36).slice(2, 8)}`, expires_in: 60 }),
    ...routes,
  };
  const env = installBrowserEnv({ document: loadHtml('src/static/index.html'), routes: allRoutes, ...envOptions });
  const mod = await importFresh('views/transcribe/index.js');
  const { app } = await load('core/store.js');
  const dom = await load('core/dom.js');
  app.set({
    models: { list: models, defaults: { tts: 'kokoro', stt: sttDefault }, fetchedAt: Date.now(), error: '' },
    connection: { ...app.get().connection, state: 'online' },
    route: ROUTE,   // view-scoped shortcuts ("r", Esc) fire only while Transcribe shows
  });
  const root = env.document.getElementById('view-transcribe');
  dom.render(root, dom.raw(PARTIAL));
  root.hidden = false;
  const ctl = mod.default.init(root);
  if (show) ctl.show(ROUTE, { focus: 'none', scrolled: false, reason: 'route' });
  await settle(env);
  const $ = (id) => env.document.getElementById(id);
  return { env, mod, ctl, app, root, $, doc: env.document };
}

/** The socket the recorder opened (the newest one). */
export function lastSocket(env) {
  return env.sockets[env.sockets.length - 1] || null;
}

/** Feed one frame of samples from the microphone (the AudioWorklet posts it). */
export function micFrame(env, samples = new Float32Array(2048).fill(0.1)) {
  const ctx = env.audioContexts[env.audioContexts.length - 1];
  const node = ctx && ctx.__worklets[ctx.__worklets.length - 1];
  if (!node) throw new Error('micFrame(): no worklet node');
  node.__emit(samples);
}

/**
 * The server begins the session (its model and VAD are loaded): what makes a start a recording. Until then the
 * socket may be open while the server still loads, and a failure there is a start that failed.
 */
export function begin(ws, extra = {}) {
  ws.serverSend({ type: 'session.begin', session_id: 's1', model: STT, sample_rate: 48000, internal_sample_rate: 16000, vad_enabled: true, ...extra });
}

/** Click Record, accept the socket and begin the session: the session is recording. */
export async function startRecording(t, { begin: beginWith = { vad_enabled: true } } = {}) {
  t.env.click(t.$('tr-record'));
  await settle(t.env);
  const ws = lastSocket(t.env);
  if (!ws) throw new Error('startRecording(): no socket');
  ws.open();
  if (beginWith) begin(ws, beginWith);
  await settle(t.env);
  return ws;
}

/**
 * Every message the polite announcement region is given, in order (a11y.announce clears it, then writes the
 * message a frame later). Announcements less than 500 ms apart coalesce: space the steps a test compares.
 */
export function trackSpoken(t, id = 'app-sr-status') {
  const el = t.$(id);
  let proto = Object.getPrototypeOf(el);
  while (proto && !Object.getOwnPropertyDescriptor(proto, 'textContent')) proto = Object.getPrototypeOf(proto);
  const inner = Object.getOwnPropertyDescriptor(proto, 'textContent');
  const said = [];
  Object.defineProperty(el, 'textContent', {
    configurable: true,
    get() { return inner.get.call(this); },
    set(v) { if (String(v)) said.push(String(v)); inner.set.call(this, v); },
  });
  return said;
}

/** Count the writes of an element's markup (dom.render sets innerHTML): a banner rewritten is read out again. */
export function countWrites(el) {
  let proto = Object.getPrototypeOf(el);
  while (proto && !Object.getOwnPropertyDescriptor(proto, 'innerHTML')) proto = Object.getPrototypeOf(proto);
  const inner = Object.getOwnPropertyDescriptor(proto, 'innerHTML');
  const box = { n: 0 };
  Object.defineProperty(el, 'innerHTML', {
    configurable: true,
    get() { return inner.get.call(this); },
    set(v) { box.n += 1; inner.set.call(this, v); },
  });
  return box;
}

/** The text a screen reader reads from a node: its text without the aria-hidden parts. */
export function spokenText(node) {
  if (!node) return '';
  if (node.nodeType === 3) return node.textContent;
  if (node.nodeType !== 1 || node.getAttribute('aria-hidden') === 'true') return '';
  return Array.from(node.childNodes).map(spokenText).join('');
}

/** A server transcript event. */
export function said(ws, textValue, { isFinal = true, speechFinal = true, start = 0, end = 1 } = {}) {
  ws.serverSend({ type: 'transcript', text: textValue, is_final: isFinal, speech_final: speechFinal, start, end, confidence: 0.95 });
}

/** Choose a file in the hidden file input. */
export function chooseFile(t, file) {
  const input = t.$('tr-file');
  input.files = [file];
  input.dispatchEvent(new FakeEvent('change', { bubbles: true }));
}

/** A small audio File. */
export function audioFile(name = 'interview.mp3', bytes = 2048, type = 'audio/mpeg') {
  return new File([new Uint8Array(bytes).fill(7)], name, { type });
}

/** The verbose_json answer of the transcription endpoint. */
export function verbose(segments = [[0, 3.12, 'Hello there.'], [3.12, 5.4, 'General Kenobi.']], extra = {}) {
  return {
    text: segments.map((s) => s[2]).join(' '),
    language: 'en',
    duration: segments.length ? segments[segments.length - 1][1] : 0,
    segments: segments.map(([start, end, t], id) => ({ id, start, end, text: ` ${t}` })),
    ...extra,
  };
}

/** The text of the polite announcement region (after its frame). */
export function announced(t) {
  t.env.raf.flush();
  return t.$('app-sr-status').textContent;
}

/** Answer the open confirm dialog. */
export async function answerConfirm(t, ok) {
  const dialog = t.$('app-confirm');
  if (!dialog.open) throw new Error('answerConfirm(): no confirm is open');
  t.env.click(t.$(ok ? 'app-confirm-ok' : 'app-confirm-cancel'));
  await settle(t.env);
}
