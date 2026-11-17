// The Generate path (inventory SP-13, SP-16, SP-22).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { phaseText, savedSetup, restoreFor, STOP_GUARD_MS } from '../../../src/static/js/views/speak/index.js';
import { bannerSignature } from '../../../src/static/js/views/speak/inspector.js';
import { readableDetail } from '../../../src/static/js/core/api.js';
import { mountSpeak, ready, type, settle, callsTo, json, delay, modelsPayload, answerConfirm, load, bytes, speechBytes, networkError } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const statusText = () => t.$('sp-gen-status-text').textContent;
const label = () => t.$('sp-generate').querySelector('.btn__label').textContent;
/** Method and path of every request since the last reset. */
const sequence = () => t.env.fetch.calls.map((c) => `${c.method} ${c.path}`);

describe('views/speak generate (pure)', () => {
  it('[SP-13] the status line names each phase', () => {
    assert.equal(phaseText('checking'), 'Checking model…');
    assert.equal(phaseText('loading', { job: 'Loading Kokoro…' }), 'Loading Kokoro…');
    assert.equal(phaseText('generating'), 'Generating…');
    assert.equal(phaseText('streaming', { ttfaMs: 139.6 }), 'Streaming · first audio 140 ms');
    assert.equal(phaseText('done', { rtf: 0.0412 }), 'Done · RTF 0.04');
    assert.equal(phaseText('done', { rtf: 0.001, cached: true }), 'Done · cached · RTF < 0.01');
    assert.equal(phaseText('done', { loadMs: 4100 }), 'Done · model loaded in 4.1 s');
    assert.equal(phaseText('stopped'), 'Stopped');
  });

  it('[SP-13] error details read as the server message, never the raw JSON envelope (core api.readableDetail)', () => {
    // Speak's banners show explain().detailText, which core reads with the same rules (the view test below
    // shows it through the real request path).
    assert.equal(readableDetail('{"error":{"message":"CUDA out of memory","code":"tts_error"}}'), 'CUDA out of memory (tts_error)');
    assert.equal(readableDetail('{"detail":"Profile not found"}'), 'Profile not found');
    assert.equal(readableDetail('<html>502 Bad Gateway</html>'), '<html>502 Bad Gateway</html>');
    assert.equal(readableDetail('{"detail":[{"loc":["body","speed"],"msg":"too big"}]}'), 'speed: too big', "FastAPI's list reads as field: message");
  });

  it('the stored setup is validated field by field and only model-independent parts move to another model', () => {
    assert.equal(savedSetup('nope'), null);
    assert.deepEqual(savedSetup({ model: 'kokoro', voice: 3, speed: '1.2', effects: 'x', stream: false }), { model: 'kokoro', speed: 1.2, stream: false });
    const saved = { model: 'kokoro', voice: 'af_sky', identity: 'n1', speed: 1.2, format: 'wav', stream: false, effects: [], instructions: 'x' };
    assert.deepEqual(restoreFor(saved, 'kokoro'), saved);
    assert.deepEqual(restoreFor(saved, 'piper/x'), { model: 'piper/x', speed: 1.2, format: 'wav', stream: false, effects: [], instructions: 'x' });
  });
});

describe('views/speak generate (view)', () => {
  it('on the fast path nothing goes out before POST /v1/audio/speech', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Near-instant speech.');
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(sequence()[0], 'POST /v1/audio/speech', 'the POST is the first request');
    assert.equal(statusText().startsWith('Checking'), false, 'no "Checking model…" on the fast path');
  });

  it('off the fast path exactly one request (the status check) goes out before the POST', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'The inventory is stale now.');
    await t.env.clock.tickAsync(21000);   // app.models is older than 20 s and nothing was spoken yet
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const calls = sequence();
    const post = calls.indexOf('POST /v1/audio/speech');
    assert.ok(post > 0);
    assert.deepEqual(calls.slice(0, post), ['GET /api/models/kokoro/status']);
  });

  it('[SP-13] Generate shows Checking model…, Loading Kokoro… (elapsed aria-hidden), Generating…, then Done; it becomes Stop meanwhile', async () => {
    let releaseLoad;
    let releaseSpeech;
    t = await mountSpeak({
      models: modelsPayload({ kokoroState: 'downloaded' }),
      routes: {
        'POST /api/models/:id/load': () => new Promise((resolve) => { releaseLoad = () => resolve(json(200, { status: 'loaded' })); }),
        'POST /v1/audio/speech': () => new Promise((resolve) => { releaseSpeech = () => resolve(bytes(200, speechBytes(), 'audio/mpeg')); }),
      },
    });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');   // no streaming: one plain response
    type(t, 'Watch the phases.');
    t.env.click(t.$('sp-generate'));
    assert.equal(label(), 'Stop', 'the button turns into Stop at once');
    await settle(t.env);
    // ensureReady: status says downloaded, so it loads (the status job drives the label).
    assert.equal(statusText(), 'Loading Kokoro…');
    assert.equal(t.$('sp-gen-elapsed').getAttribute('aria-hidden'), 'true', 'the elapsed counter never ticks in the live region');
    await t.env.clock.tickAsync(3000);
    assert.equal(t.$('sp-gen-elapsed').textContent.trim(), '3 s');
    assert.equal(statusText(), 'Loading Kokoro…', 'the status text itself does not change per second');
    assert.equal(t.$('sp-gen-status').getAttribute('role'), 'status');
    // The server loads; the status check now says loaded.
    const statusRoute = t.env.fetch;
    statusRoute.route('GET /api/models/:id/status', (req) => json(200, { id: req.params.id, state: 'loaded' }));
    releaseLoad();
    await settle(t.env);
    await settle(t.env);
    assert.equal(statusText(), 'Generating…');
    releaseSpeech();
    await settle(t.env);
    await settle(t.env);
    assert.match(statusText(), /^Done/);
    assert.match(statusText(), /model loaded in \d+\.\d s/);
    assert.equal(label(), 'Generate');
  });

  it('[SP-13] another model\'s work (Voices, Models) never drives the Speak status', async () => {
    t = await mountSpeak();
    await ready(t);
    const status = await load('core/status.js');
    const end = status.beginJob('model:chatterbox', 'Loading Chatterbox…');
    await settle(t.env);
    assert.equal(statusText(), '');
    assert.equal(label(), 'Generate');
    end();
  });

  it('[SP-13] a cancelled model switch returns to idle silently', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-model'), 'piper');
    await settle(t.env);
    await settle(t.env);
    type(t, 'Switch first.');
    t.env.click(t.$('sp-generate'));
    await answerConfirm(t, false);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 0);
    assert.equal(statusText(), '');
    assert.equal(t.$('sp-gen-error').hidden, true, 'no error');
    assert.equal(label(), 'Generate');
  });

  it('[SP-13] Stop aborts the request; nothing is left behind', async () => {
    let signal = null;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': (req) => { signal = req.signal; return new Promise(() => {}); } } });
    await ready(t);
    type(t, 'Stop me.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(statusText(), 'Generating…');
    await t.env.clock.tickAsync(STOP_GUARD_MS);   // a deliberate click, not the tail of a double click
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(signal.aborted, true, 'the fetch is aborted');
    assert.equal(statusText(), 'Stopped');
    assert.equal(label(), 'Generate');
    assert.equal(t.$('sp-gen-error').hidden, true);
    assert.equal(t.parts.takes.all().length, 0);
  });

  it('[SP-13] Mod+Enter generates from the editor and Esc stops', async () => {
    let signal = null;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': (req) => { signal = req.signal; return new Promise(() => {}); } } });
    await ready(t);
    type(t, 'Keyboard.');
    t.$('sp-input').focus();
    t.env.key('Enter', { ctrlKey: true });
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 1);
    assert.equal(t.$('sp-generate').getAttribute('aria-keyshortcuts'), 'Escape');
    t.env.key('Escape');
    await settle(t.env);
    assert.equal(signal.aborted, true);
    assert.equal(statusText(), 'Stopped');
    assert.equal(t.$('sp-generate').getAttribute('aria-keyshortcuts'), 'Control+Enter');
  });

  it('[SP-13] the click primes the shared audio element synchronously (iOS plays the take after the await)', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Unlock.');
    const audioEl = t.$('app-audio');
    t.env.click(t.$('sp-generate'));
    assert.equal(audioEl.__plays.length, 1, 'played inside the click');
    assert.equal(audioEl.__plays[0].gesture, true);
    assert.match(audioEl.__plays[0].src, /^data:audio\/wav;base64,/);
  });

  it('[SP-13] a failed request shows "Couldn\'t generate speech" with the message, Details and Retry; Retry tries again', async () => {
    let fail = true;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => (fail ? json(500, { error: { message: 'CUDA out of memory', code: 'tts_error' } }) : bytes(200, speechBytes(), 'audio/mpeg')) } });
    await ready(t);
    type(t, 'This fails.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('sp-gen-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't generate speech");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'CUDA out of memory');
    assert.equal(banner.querySelector('details pre').textContent, 'CUDA out of memory (tts_error)', 'the server message goes in Details, not the JSON envelope');
    assert.ok(!banner.querySelector('.banner__msg').textContent.includes('"error"'), 'the message is never the JSON envelope');
    assert.equal(t.parts.takes.all().length, 0);
    fail = false;
    const retry = [...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry');
    retry.focus();
    t.env.click(retry);
    assert.equal(document.activeElement, t.$('sp-generate'), 'the focus moves to Generate (now Stop), never to <body>');
    assert.equal(label(), 'Stop');
    await settle(t.env);
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 2);
    assert.equal(banner.hidden, true);
    assert.equal(t.parts.takes.all().length, 1);
    assert.equal(document.activeElement, t.$('sp-generate'), 'and stays there when the take is in');
  });

  it('[SP-22] a take the server logged (X-History) tells History to reload: history:changed fires once its answer arrives, streamed or not; a failed request logs nothing and says nothing', async () => {
    let fail = false;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => (fail ? json(500, { error: { message: 'CUDA out of memory', code: 'tts_error' } }) : bytes(200, speechBytes(), 'audio/mpeg')) } });
    await ready(t);
    const { on } = await load('core/store.js');
    const events = [];
    on('history:changed', (detail) => events.push({ detail, takes: t.parts.takes.all().length }));
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Logged by the server.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.headers['x-history'], 'true', 'the request asks the server to log it');
    assert.equal(events.length, 1, 'one event per logged take');
    assert.equal(events[0].takes, 0, 'sent when the answer arrives, before the take is even listed (nothing waits for it)');
    assert.equal(t.parts.takes.all().length, 1);
    fail = true;
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-gen-error').hidden, false);
    assert.equal(events.length, 1, 'a refused request was not logged: no event');
  });

  // A voice worker that answers but can't run (its /health "failed",
  // provider_failure in the status) is not offline: Generate says the worker's reason.
  it('[SP-13] a voice worker that answers but can\'t run says why under "Couldn\'t generate speech", never "offline"', async () => {
    const reason = 'Torch wheel has no code that runs on sm_120 (built for: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90)';
    t = await mountSpeak({
      routes: {
        'GET /api/models/:id/status': (req) => json(200, {
          id: req.params.id, state: 'provider_unavailable', provider_available: false,
          provider_failure: { code: 'cuda_arch_unsupported', message: reason },
        }),
      },
    });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'The worker answers.');
    await t.env.clock.tickAsync(21000);   // off the fast path: the status check runs first
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('sp-gen-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't generate speech");
    assert.equal(banner.querySelector('.banner__msg').textContent, `The voice worker can't run: ${reason}.`);
    assert.doesNotMatch(banner.textContent, /offline/i);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 0, 'nothing was sent to the worker');
  });

  it('[SP-13] a dropped connection says what happened, then what to do, under "Couldn\'t generate speech", with Retry', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => networkError() } });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Unplugged.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('sp-gen-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't generate speech");
    assert.equal(banner.querySelector('.banner__msg').textContent, "Can't reach the server. Check that Open Speech is running.");
    assert.ok([...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Retry'));
    assert.equal(t.parts.takes.all().length, 0);
  });

  it('[SP-13] a double click on Generate starts one take and never stops it; a deliberate later click does', async () => {
    let signal = null;
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': (req) => { signal = req.signal; return delay(1000, bytes(200, speechBytes(), 'audio/mpeg')); } } });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Double click.');
    const button = t.$('sp-generate');
    t.env.click(button);                                                                       // detail 1
    button.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, detail: 2 }));   // the second click of the pair
    await settle(t.env);
    assert.equal(label(), 'Stop', 'still generating');
    assert.equal(signal.aborted, false);
    t.env.click(button);   // a second tap right after the first (touch screens report detail 1)
    await settle(t.env);
    assert.equal(signal.aborted, false, 'a tap within the guard is not a Stop');
    await t.env.clock.tickAsync(1000);
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.takes.all().length, 1, 'the take is in');
    assert.match(statusText(), /^Done/);
    assert.equal(callsTo(t, 'POST /v1/audio/speech').length, 1, 'one request');
    // A deliberate click later on does stop the next take.
    type(t, 'Stop this one.');
    t.env.click(button);
    await t.env.clock.tickAsync(STOP_GUARD_MS);
    t.env.click(button);
    await settle(t.env);
    assert.equal(signal.aborted, true);
    assert.equal(statusText(), 'Stopped');
  });

  it('[SP-13] the status line announces each phase once: "Done" is said once, the real-time factor joins it silently', async () => {
    t = await mountSpeak();
    await ready(t);
    // Every write to the announced text (the status region reads it out on each change).
    const text = t.$('sp-gen-status-text');
    let accessor = null;
    for (let p = Object.getPrototypeOf(text); p && !accessor; p = Object.getPrototypeOf(p)) accessor = Object.getOwnPropertyDescriptor(p, 'textContent');
    const seen = [];
    Object.defineProperty(text, 'textContent', {
      configurable: true,
      get() { return accessor.get.call(this); },
      set(value) { seen.push(String(value)); accessor.set.call(this, value); },
    });
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Count the announcements.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    delete text.textContent;
    assert.deepEqual(seen.filter((s) => s.startsWith('Done')), ['Done'], `one "Done" in the live region: ${JSON.stringify(seen)}`);
    assert.equal(t.$('sp-gen-status').getAttribute('role'), 'status');
    const extra = t.$('sp-gen-status-extra');
    assert.equal(extra.getAttribute('aria-hidden'), 'true', 'the late part is never announced');
    assert.match(extra.textContent, /^ · RTF /, 'the RTF still shows on screen');
    assert.match(t.$('sp-gen-status').textContent, /^Done · RTF /);
  });

  it('[SP-13] Generate while a recording runs elsewhere fetches the take whole (no stream) and finishes; it plays later', async () => {
    t = await mountSpeak();
    await ready(t);
    t.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    type(t, 'Streamed while recording.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.search, '', 'nothing can play while the microphone is open: no stream');
    assert.equal(label(), 'Generate', 'the generation finished');
    assert.match(statusText(), /^Done/);
    assert.equal(t.ctx.gen, null);
    const take = t.parts.takes.all()[0];
    assert.equal(take.hint, 'Stop recording to listen.');
    assert.ok(take.blob && take.blob.size > 0, 'the take keeps its audio');
    assert.equal(t.$('sp-takes').querySelector('[data-take-act="play"]').disabled, false, 'it plays from its button later');
  });

  it('[REV-33] a take that will only download (PCM) leaves the clip playing now alone: no prime()', async () => {
    t = await mountSpeak();
    await ready(t);
    const audio = await load('core/audio.js');
    await audio.playClip({ id: 'studio:render', owner: 'studio', title: 'A render', src: 'blob:http://127.0.0.1:8100/render', homeRoute: 'studio' });
    const before = t.$('app-audio').__plays.length;
    t.env.change(t.$('sp-format'), 'pcm');
    type(t, 'Raw samples.');
    t.env.click(t.$('sp-generate'));
    assert.equal(t.$('app-audio').__plays.length, before, 'no silent unlock clip');
    assert.equal(audio.current() && audio.current().id, 'studio:render', 'the render keeps playing');
    await settle(t.env);
    await settle(t.env);
    assert.equal(audio.current() && audio.current().id, 'studio:render');
  });

  it('[SP-13] a refused key shows "API key needed" with Open Settings', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => json(401, { error: { message: 'Invalid API key', code: 'unauthorized' } }) } });
    await ready(t);
    type(t, 'Refused.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('sp-gen-error');
    assert.equal(banner.querySelector('.banner__title').textContent, 'API key needed');
    assert.ok([...banner.querySelectorAll('button')].some((b) => b.textContent.trim() === 'Open Settings'));
  });

  it('[SP-16] a clone model that needs a transcript asks for a reference recording on the field, before any request', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-model'), 'chatterbox');
    await settle(t.env);
    await settle(t.env);
    type(t, 'Clone me.');
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(t.env.fetch.calls.length, 0, 'no status check, no POST');
    assert.equal(t.$('sp-reference-error').hidden, false);
    assert.equal(t.$('sp-reference-error-text').textContent, 'Choose a reference recording for this model');
    assert.equal(t.$('sp-reference').getAttribute('aria-invalid'), 'true');
    assert.equal(document.activeElement, t.$('sp-reference'));
    assert.equal(t.$('sp-inspector').getAttribute('data-open'), 'true', 'the phone inspector opens to show it');
    t.env.change(t.$('sp-reference'), 'demo_reference');
    assert.equal(t.$('sp-reference-error').hidden, true);
    assert.equal(t.$('sp-reference').hasAttribute('aria-invalid'), false);
  });

  it('[SP-22] after success the take can be downloaded; History is the record (X-History), not a local list', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Kept in History.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    const [ms] = (await import('../harness/fakes.js')).FakeMediaSource.instances;
    if (ms) ms.__open();
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.headers['x-history'], 'true');
    const download = t.$('sp-takes').querySelector('[data-take-act="download"]');
    assert.equal(download.disabled, false, 'Download is enabled');
    assert.equal(localStorage.getItem('open-speech-tts-history'), null, 'no local Recent list');
    assert.equal(document.getElementById('app-toasts').textContent.includes('Speech generated'), false, 'the take is the confirmation');
  });

  it('while the models load the inspector shows skeletons and Generate is aria-disabled with "Loading models…"', async () => {
    t = await mountSpeak({ loaded: false });
    assert.equal(t.$('sp-inspector-loading').hidden, false);
    assert.equal(t.$('sp-inspector-loading').getAttribute('aria-busy'), 'true');
    assert.equal(t.$('sp-inspector-fields').hidden, true);
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('sp-gen-hint').textContent, 'Loading models…');
    const models = modelsPayload();
    t.app.set({ models: { list: models.models, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await ready(t);
    assert.equal(t.$('sp-inspector-loading').hidden, true);
    assert.equal(t.$('sp-inspector-fields').hidden, false);
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false');
    assert.equal(t.$('sp-gen-hint').hidden, true);
  });

  it('with no speech model the view shows "No speech model is available" and Open Models', async () => {
    t = await mountSpeak({ models: { models: [{ id: 'Systran/faster-whisper-base', type: 'stt', provider: 'faster-whisper', state: 'loaded' }], default_tts_model: '' } });
    await ready(t);
    assert.equal(t.$('sp-empty').hidden, false);
    assert.equal(t.$('sp-empty').querySelector('.empty__title').textContent, 'No speech model is available');
    assert.equal(t.$('sp-empty').querySelector('a').getAttribute('href'), '#models/tts');
    assert.equal(t.$('sp-inspector').hidden, true);
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
  });

  it('offline, Generate is disabled and says why', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Offline.');
    t.app.set({ connection: { ...t.app.get().connection, state: 'offline' } });
    await settle(t.env);
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
    assert.match(t.$('sp-gen-hint').textContent, /^Offline/);
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(t.env.fetch.calls.length, 0);
    t.app.set({ connection: { ...t.app.get().connection, state: 'online' } });
    await settle(t.env);
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false');
  });

  it('a failed capabilities or voices load shows an inspector banner with Retry; a refused key says API key needed', async () => {
    let fail = true;
    t = await mountSpeak({ routes: { 'GET /v1/audio/voices': () => (fail ? json(503, { error: { message: 'Worker busy' } }) : json(200, { voices: [{ id: 'af_heart' }] })) } });
    await ready(t);
    const banner = t.$('sp-inspector-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'status', 'a load failure is polite: a persistent status region');
    assert.equal(banner.querySelector('[role]'), null, 'the banner inside carries no second live role');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't load the voices");
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('sp-summary-text').textContent, "Couldn't load the voices", 'the phone summary says it too');
    fail = false;
    t.env.click([...banner.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(banner.textContent.trim(), '', 'the banner is gone');
    assert.equal(banner.hidden, false, 'the region itself stays, ready for the next message');
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false');
    t.env.restore();
    t = await mountSpeak({ routes: { 'GET /api/tts/capabilities': () => json(401, { error: { message: 'Invalid API key' } }) } });
    await ready(t);
    assert.equal(t.$('sp-inspector-error').querySelector('.banner__title').textContent, 'API key needed');
    assert.equal(t.$('sp-inspector-error').querySelector('.banner__msg'), null, 'the shell banner carries the sentence; it is not repeated');
    assert.ok([...t.$('sp-inspector-error').querySelectorAll('button')].some((b) => b.textContent.trim() === 'Open Settings'));
  });

  it('the inspector banner is not rebuilt by the models poll or model-op ticks: no re-announcement, Retry keeps the focus', async () => {
    t = await mountSpeak({ routes: { 'GET /v1/audio/voices': () => json(503, { error: { message: 'Worker busy' } }) } });
    await ready(t);
    const host = t.$('sp-inspector-error');
    const banner = host.firstElementChild;
    const retry = [...host.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry');
    retry.focus();
    t.app.set({ models: { ...t.app.get().models, fetchedAt: Date.now() } });   // the 60 s poll
    await settle(t.env);
    t.app.set({ ops: { ...t.app.get().ops, 'piper/en_US-amy-medium': { kind: 'load', startedAt: Date.now() } } });   // a 2 s op tick
    await settle(t.env);
    assert.equal(host.firstElementChild, banner, 'the same banner element: nothing to announce again');
    assert.equal(document.activeElement, retry, 'the focused Retry is still there');
    assert.equal(bannerSignature(null), '');
  });

  it('Retry keeps its banner (busy) while it reloads, then hands the focus to the model select', async () => {
    let fail = true;
    t = await mountSpeak({ routes: { 'GET /v1/audio/voices': () => (fail ? json(503, { error: { message: 'Worker busy' } }) : delay(500, json(200, { voices: [{ id: 'af_heart' }] }))) } });
    await ready(t);
    const host = t.$('sp-inspector-error');
    let retry = [...host.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry');
    retry.focus();
    fail = false;
    t.env.click(retry);
    await settle(t.env);
    assert.equal(host.contains(retry), true, 'the banner stays while it reloads');
    assert.equal(retry.getAttribute('aria-busy'), 'true');
    assert.equal(t.$('sp-inspector-loading').hidden, true, 'the busy Retry is the progress; no skeleton under it');
    await t.env.clock.tickAsync(500);
    await settle(t.env);
    await settle(t.env);
    assert.equal(host.textContent.trim(), '');
    assert.equal(document.activeElement, t.$('sp-model'), 'the banner went: the focus is on the model select, not <body>');
    // A Retry that fails the same way keeps its banner, and the focus on its Retry.
    fail = true;
    t.env.change(t.$('sp-model'), 'piper');
    await settle(t.env);
    await settle(t.env);
    retry = [...host.querySelectorAll('button')].find((b) => b.textContent.trim() === 'Retry');
    retry.focus();
    t.env.click(retry);
    await settle(t.env);
    await settle(t.env);
    assert.equal(retry.isConnected, true, 'the same failure: the same banner');
    assert.equal(retry.hasAttribute('aria-busy'), false);
    assert.equal(document.activeElement, retry, 'and the focus stays on its Retry');
  });

  it('[RE-04] Speak keeps app.speakVoice current for the Reader ("Same as Speak"), saying what the voice is (kind)', async () => {
    t = await mountSpeak();
    await ready(t);
    const spoken = () => t.app.get().speakVoice;
    assert.equal(spoken().model, 'kokoro');
    assert.equal(spoken().voice, t.parts.inspector.voiceSpec());
    assert.equal(spoken().kind, 'voice');
    t.parts.inspector.commit(t.parts.inspector.state().data, { ...t.parts.inspector.currentSetup(), identity: 'n1' });
    await settle(t.env);
    assert.deepEqual(spoken(), { model: 'kokoro', voice: 'voice:n1', label: 'Narrator', kind: 'named' });
  });

  it('the last setup is saved (KEYS.speak) and restored on the next page load when there is no Default preset', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-format'), 'flac');
    t.env.input(t.$('sp-speed'), '1.35');
    await t.env.clock.tickAsync(400);
    const saved = JSON.parse(localStorage.getItem('open-speech-speak'));
    assert.equal(saved.format, 'flac');
    assert.equal(saved.speed, 1.35);
    assert.equal(saved.model, 'kokoro');
    t.env.restore();
    t = await mountSpeak({ seed: { 'open-speech-speak': JSON.stringify({ ...saved, voice: 'am_adam' }) } });
    await ready(t);
    assert.equal(t.$('sp-format').value, 'flac');
    assert.equal(t.parts.inspector.state().speed, 1.35);
    assert.equal(t.parts.inspector.voiceSpec(), 'am_adam');
  });
});
