// The shared "Ask AI" button and sheet (components/ask-ai.js, its pure parts in
// components/ask-ai-logic.js), on POST /api/assistant/ask. A fake of the assistant's routes, the
// Reader's job routes and its paste route answers here; nothing reaches an AI server.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames } from '../harness/a11y.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { shellEnv, load, settle, json } from '../shell/helpers.js';

let env;
afterEach(() => env?.restore());

const HOST = '127.0.0.1:9';
const PRESETS = [
  { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000, show_thinking: true },
  { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000, show_thinking: true },
  { id: 'summarize', name: 'Summarize', label: 'Summarize', inputs: ['text', 'transcript', 'document'], output: 'text', chunk_chars: 12000, show_thinking: true },
];
const TRANSCRIPT = 'We should ship the reader first, because people listen on their phones every evening.';
const AUDIO = { format: 'mp3', content_type: 'audio/mpeg', base64: 'SUQzBAAAAAAA', bytes: 9 };

/** A transcript source, as Transcribe gives it. */
function transcript(text = TRANSCRIPT) {
  return { id: 'transcript', label: 'The transcript', input: { kind: 'transcript', text }, text, chars: Array.from(text).length };
}

/** The fake server: GET /api/assistant, the presets, ask, the job routes and the Reader's paste route. */
function fakeServer({ enabled = true, presets = PRESETS, ask = null } = {}) {
  const state = { asks: [], jobGets: 0, jobDeletes: [], pastes: [], job: null };
  const routes = {
    'GET /api/assistant': () => json(200, { enabled, host: enabled ? HOST : null, model: 'qwen3', server_settings: {} }),
    'GET /api/assistant/presets': () => (enabled ? json(200, { presets }) : json(404, { error: { message: 'off', code: 'assistant_disabled' } })),
    'POST /api/assistant/ask': (req) => {
      state.asks.push(req);
      if (ask) return ask(req, state);
      return json(200, { reply: 'Ship the Reader first.', thinking: 'Two tasks.', finish_reason: 'stop', elapsed_ms: 1840, preset: req.json.preset || 'voice', input: { kind: req.json.input.kind, chars: 10 } });
    },
    'GET /v1/reader/assistant-jobs/:id': () => {
      state.jobGets += 1;
      return json(200, state.job);
    },
    'DELETE /v1/reader/assistant-jobs/:id': (req) => {
      state.jobDeletes.push(req.params.id);
      state.job = { ...state.job, status: 'cancelled' };
      return json(200, state.job);
    },
    'POST /v1/reader/documents': (req) => {
      state.pastes.push(req.json);
      return json(201, { id: 'saved-1', title: req.json.title });
    },
  };
  return { routes, state };
}

async function setup({ server = {}, routes = {}, seed = {} } = {}) {
  const fake = fakeServer(server);
  env = await shellEnv({ routes: { ...fake.routes, ...routes }, seed });
  const ask = await load('components/ask-ai.js');
  const audio = await load('core/audio.js');
  const { app } = await load('core/store.js');
  app.set({
    models: { list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded' }], defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' },
    speakVoice: { model: 'kokoro', voice: 'af_bella', label: 'Bella', kind: 'voice' },
  });
  const button = document.createElement('button');
  button.setAttribute('type', 'button');
  button.textContent = 'Ask AI';
  button.hidden = true;
  document.getElementById('main').appendChild(button);
  return { ask, audio, app, button, server: fake.state, dialog: document.getElementById('app-ask-ai') };
}

const $ = (id) => document.getElementById(id);
const textOf = (el) => String(el ? el.textContent : '').replace(/\s+/g, ' ').trim();
const shown = (id) => {
  for (let el = $(id); el; el = el.parentElement) if (el.hidden) return false;
  return true;
};

/** Open the sheet over `sources` and let it render. Returns {closed}: the open promise. */
async function open(ctx, sources = [transcript()], options = {}) {
  const closed = ctx.ask.openAskAi({ trigger: ctx.button, sources, ...options });
  await settle(env);
  await settle(env);
  return { closed };
}

async function send() {
  env.click($('ask-send'));
  await settle(env);
  await settle(env);
}

describe('components/ask-ai-logic: the pure parts', () => {
  it('presetChoices: the presets whose inputs suit the kind, by their label (and name when it differs); every preset when none suits', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    assert.deepEqual(L.presetChoices(PRESETS, 'transcript').map((c) => [c.id, c.text, c.output]), [['voice', 'Answer · Voice', 'speech'], ['summarize', 'Summarize', 'text']]);
    assert.deepEqual(L.presetChoices(PRESETS, 'document').map((c) => c.id), ['book', 'summarize']);
    assert.deepEqual(L.presetChoices([{ id: 'x', name: 'X', inputs: ['text'] }], 'document').map((c) => c.id), ['x'], 'none suits: all of them');
    assert.equal(L.defaultPreset(L.presetChoices(PRESETS, 'document'), 'document'), 'book');
    assert.equal(L.defaultPreset(L.presetChoices(PRESETS, 'text'), 'text'), 'voice');
    assert.equal(L.defaultPreset(L.presetChoices(PRESETS, 'text'), 'text', 'summarize'), 'summarize', 'the last one used wins while offered');
    assert.deepEqual(L.defaultOutputs({ output: 'speech' }), { show: true, speak: true, save: false });
    assert.deepEqual(L.defaultOutputs({ output: 'document' }), { show: true, speak: false, save: true });
  });

  it('isLong and privacyLine: more than one piece (or chapter) is a job; a long text or transcript says the Reader keeps it', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    const choice = { chunk: 100 };
    assert.equal(L.isLong({ input: { kind: 'text' }, chars: 100 }, choice), false);
    assert.equal(L.isLong({ input: { kind: 'text' }, chars: 101 }, choice), true);
    assert.equal(L.isLong({ input: { kind: 'document' }, chars: 10, chapters: 2 }, choice), true);
    assert.equal(L.privacyLine(HOST), `This is sent to ${HOST}.`);
    assert.equal(L.privacyLine(HOST, { long: true, kind: 'transcript' }), `This is sent to ${HOST}. A long text is kept in the Reader, where you can delete it.`);
    assert.match(L.privacyLine(HOST, { long: true, kind: 'document' }), /answers are saved in the Reader/);
  });

  it('requestBody: the input as given, the preset, speech in the chosen voice as mp3, a document, and text: false only beside another output', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    const source = transcript('Hi there.');
    assert.deepEqual(L.requestBody({ preset: 'voice', source, outputs: { show: true, speak: false, save: false } }), { input: { kind: 'transcript', text: 'Hi there.' }, preset: 'voice' });
    assert.deepEqual(L.requestBody({ preset: 'voice', source, outputs: { show: false, speak: true, save: true }, voice: { model: 'kokoro', voice: 'af_bella' } }), {
      input: { kind: 'transcript', text: 'Hi there.' }, preset: 'voice',
      output: { text: false, speech: { format: 'mp3', model: 'kokoro', voice: 'af_bella' }, document: {} },
    });
  });

  it('failureCopy and audioNote: the server\'s codes in plain words; a 422 by its param; never the AI server\'s own answer', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    assert.equal(L.failureCopy({ status: 502, code: 'assistant_unreachable', message: 'connect ECONNREFUSED 127.0.0.1:9' }).title, "Couldn't reach the AI server");
    assert.doesNotMatch(JSON.stringify(L.failureCopy({ status: 502, code: 'assistant_http_error', message: 'upstream said <secret>' })), /secret/);
    assert.equal(L.failureCopy({ status: 504, code: 'assistant_timeout' }).title, 'The AI took too long to answer');
    const voice = L.failureCopy({ status: 422, code: 'validation_error', fields: [{ field: 'output.speech.voice', message: "voice 'zz' is not a voice of kokoro" }] });
    assert.deepEqual([voice.title, voice.message], ["That voice wasn't accepted", "Voice 'zz' is not a voice of kokoro."]);
    assert.equal(L.failureCopy({ status: 404, code: 'document_not_found' }).title, 'That document is no longer in the library');
    assert.equal(L.failureCopy({ status: 503, code: 'speech_unavailable' }).title, "This server can't speak the answer");
    assert.match(L.audioNote({ code: 'speech_too_long', message: '…' }), /Save it as a document/);
    assert.equal(L.audioNote(null), '');
  });

  it('audioNote keeps the server\'s message as written: a first word that is a name or an acronym keeps its case', async () => {
    // The note lower-cased the first letter ("qwen3's", "tTS provider", "cUDA").
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    const cases = [
      ['dtype_unsupported', "Qwen3's float16 weights do not run on this GPU (set QWEN3_DTYPE=auto); without it the model can't load",
        "Couldn't speak the answer: Qwen3's float16 weights do not run on this GPU (set QWEN3_DTYPE=auto); without it the model can't load."],
      ['provider_unavailable', "TTS provider 'fish' is unavailable: <urlopen error [Errno 111] Connection refused>",
        "Couldn't speak the answer: TTS provider 'fish' is unavailable: <urlopen error [Errno 111] Connection refused>."],
      ['cuda_unavailable', 'CUDA is unavailable', "Couldn't speak the answer: CUDA is unavailable."],
      ['tts_error', 'Kokoro fell over', "Couldn't speak the answer: Kokoro fell over."],
    ];
    for (const [code, message, note] of cases) {
      assert.equal(L.answerOf({ reply: 'An answer.', audio_error: { code, message } }).audioNote, note, code);
    }
  });

  it('jobProgress, jobText and jobFailure: pieces done of all; the chapters\' texts joined with blank lines; a failed job by its code', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    const job = { status: 'done', chapters: [{ pieces: 2, pieces_done: 2, text: 'One.' }, { pieces: 1, pieces_done: 1, text: ' Two. ' }] };
    assert.deepEqual(L.jobProgress(job), { done: 3, total: 3, label: '3 of 3 pieces' });
    assert.equal(L.jobText(job), 'One.\n\nTwo.');
    assert.equal(L.jobFailure({ status: 'failed', error: 'interrupted' }).title, 'The job stopped when the server restarted');
  });
});

describe('components/ask-ai: the button', () => {
  it('shows only while GET /api/assistant says the assistant is on; binding sends nothing; refresh() loads once per 30 s', async () => {
    const ctx = await setup();
    let clicked = 0;
    const bound = ctx.ask.bindAskButton(ctx.button, { onClick: () => { clicked += 1; } });
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /api/assistant').length, 0, 'binding sends nothing');
    assert.equal(ctx.button.hidden, true);
    bound.refresh();
    await settle(env);
    assert.equal(ctx.button.hidden, false, 'on: shown');
    assert.equal(env.fetch.callsTo('GET /api/assistant/presets').length, 1, 'the presets load with it');
    bound.refresh();
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /api/assistant').length, 1, 'fresh: not asked again');
    await settle(env, 30001);
    bound.refresh();
    await settle(env);
    assert.equal(env.fetch.callsTo('GET /api/assistant').length, 2, 'stale: asked again');
    env.click(ctx.button);
    assert.equal(clicked, 1);
    bound.destroy();
  });

  it('stays hidden while the assistant is off, and when the status can\'t load', async () => {
    const ctx = await setup({ server: { enabled: false } });
    const bound = ctx.ask.bindAskButton(ctx.button, { onClick: () => {} });
    bound.refresh();
    await settle(env);
    assert.equal(ctx.button.hidden, true);
    assert.equal(env.fetch.callsTo('GET /api/assistant/presets').length, 0, 'off: no presets asked for');
    env.restore();
    const other = await setup({ routes: { 'GET /api/assistant': () => json(500, { error: { message: 'boom', code: 'internal' } }) } });
    const again = other.ask.bindAskButton(other.button, { onClick: () => {} });
    again.refresh();
    await settle(env);
    assert.equal(other.button.hidden, true);
  });
});

describe('components/ask-ai: the sheet', () => {
  it('shows what is sent (a summary and its size), the presets that suit it by their label, the outputs the preset starts with, and the privacy line', async () => {
    const ctx = await setup();
    await open(ctx);
    assert.equal(ctx.dialog.open, true);
    assert.equal(textOf($('ask-title')), 'Ask AI');
    assert.match(textOf($('ask-sources')), /The transcript/);
    assert.match(textOf($('ask-sources')), /We should ship the reader first/);
    assert.match(textOf($('ask-sources')), new RegExp(`${Array.from(TRANSCRIPT).length} characters`));
    assert.deepEqual($('ask-preset').querySelectorAll('option').map((o) => [o.value, textOf(o)]), [['voice', 'Answer · Voice'], ['summarize', 'Summarize']]);
    assert.equal($('ask-preset').value, 'voice');
    assert.equal($('ask-out-show').checked, true);
    assert.equal($('ask-out-speak').checked, true, "the voice preset's output is speech");
    assert.equal($('ask-out-save').checked, false);
    assert.equal(textOf($('ask-voice-name')), 'Voice: Same as Speak · Bella');
    assert.equal(textOf($('ask-privacy-text')), `This is sent to ${HOST}.`);
    assert.equal(shown('ask-long'), false);
    assert.equal(document.activeElement, $('ask-send'), 'Send has focus');
    // Another preset: its own first outputs.
    env.change($('ask-preset'), 'summarize');
    assert.equal($('ask-out-speak').checked, false);
    // No output at all: Send waits, and says why.
    env.change($('ask-out-show'), false);
    assert.equal($('ask-send').disabled, true);
    assert.match(textOf($('ask-out-error')), /Choose at least one/);
    assertAccessibleNames(ctx.dialog);
  });

  it('a long transcript goes as a job: the privacy line says the Reader keeps it, Speak is off and Save is on', async () => {
    const ctx = await setup();
    const long = 'Word '.repeat(2600).trim();
    await open(ctx, [transcript(long)]);
    assert.equal(textOf($('ask-privacy-text')), `This is sent to ${HOST}. A long text is kept in the Reader, where you can delete it.`);
    assert.equal($('ask-out-speak').disabled, true);
    assert.equal($('ask-out-speak').checked, false);
    assert.equal($('ask-out-save').checked, true);
    assert.equal(shown('ask-long'), true);
    assert.match(textOf($('ask-long')), /as a job/);
  });

  it('Send asks with the preset, the input and the voice; the answer shows as text (never markup), the thinking collapsed, and it plays through the audio coordinator', async () => {
    const reply = '<img src=x onerror="alert(1)"> Ship the <b>Reader</b> first.';
    const ctx = await setup({ server: { ask: (req) => json(200, { reply, thinking: 'Two <i>tasks</i>.', finish_reason: 'stop', elapsed_ms: 1840, preset: 'voice', input: { kind: 'transcript', chars: 86 }, audio: AUDIO }) } });
    await open(ctx);
    await send();
    assert.equal(ctx.server.asks.length, 1);
    assert.deepEqual(ctx.server.asks[0].json, {
      input: { kind: 'transcript', text: TRANSCRIPT }, preset: 'voice',
      output: { speech: { format: 'mp3', model: 'kokoro', voice: 'af_bella' } },
    });
    assert.equal(shown('ask-result'), true);
    assert.equal($('ask-reply').textContent, reply, 'the reply as it came, as text');
    assert.equal(ctx.dialog.querySelector('img'), null, 'no markup from the AI');
    assert.equal(ctx.dialog.querySelector('#ask-reply b'), null);
    assert.equal($('ask-thinking').hidden, false);
    assert.equal($('ask-thinking').open, false, 'the thinking is collapsed');
    assert.equal(ctx.dialog.querySelector('#ask-thinking i'), null);
    assert.match(textOf($('ask-meta')), /Answer · Voice · 1\.8 s/);
    const cur = ctx.audio.current();
    assert.ok(cur, 'a session plays');
    assert.equal(cur.id, 'ask-ai:answer');
    assert.equal(cur.ephemeral, false, 'the mini-player and the lock screen have it');
    assert.ok(String(ctx.audio.element().src).startsWith('blob:'), 'from the base64 audio');
    assert.equal(shown('ask-play'), true);
    assert.equal(shown('ask-again'), true);
    assert.equal(shown('ask-send'), false);
    assert.equal(textOf($('ask-cancel')), 'Done');
  });

  it('Copy, Save as a document (then Open in the Reader), and Send again', async () => {
    const ctx = await setup();
    await open(ctx);
    env.change($('ask-out-speak'), false);
    await send();
    assert.deepEqual(ctx.server.asks[0].json, { input: { kind: 'transcript', text: TRANSCRIPT }, preset: 'voice' }, 'Show only: no output field');
    assert.equal(shown('ask-play'), false, 'nothing to play');
    env.click($('ask-copy'));
    await settle(env);
    assert.deepEqual(env.clipboard.written, ['Ship the Reader first.']);
    env.click($('ask-again'));
    await settle(env);
    assert.equal(shown('ask-compose'), true);
    assert.equal(shown('ask-result'), false);
    assert.equal($('ask-out-speak').checked, false, 'the choices made stay');
    await send();
    assert.equal(ctx.server.asks.length, 2, 'sent again with the same choices');
    assert.deepEqual(ctx.server.asks[1].json, ctx.server.asks[0].json);
    env.click($('ask-save'));
    await settle(env);
    assert.equal(ctx.server.pastes.length, 1);
    assert.equal(ctx.server.pastes[0].type, 'text');
    assert.equal(ctx.server.pastes[0].text, 'Ship the Reader first.');
    assert.match(ctx.server.pastes[0].title, /^Answer: We should ship/);
    assert.equal(shown('ask-save'), false);
    assert.equal(shown('ask-open'), true);
    env.click($('ask-open'));
    await settle(env);
    assert.equal(ctx.dialog.open, false);
    assert.equal(location.hash, '#reader/saved-1');
  });

  it('a saved answer (output.document) offers Open in the Reader at once', async () => {
    const ctx = await setup({ server: { ask: () => json(200, { reply: 'Done.', thinking: null, finish_reason: 'stop', elapsed_ms: 10, preset: 'voice', input: {}, document_id: 'doc-9' }) } });
    await open(ctx);
    env.change($('ask-out-speak'), false);
    env.change($('ask-out-save'), true);
    await send();
    assert.deepEqual(ctx.server.asks[0].json.output, { document: {} });
    assert.equal(shown('ask-open'), true);
    assert.equal(shown('ask-save'), false);
    assert.equal($('ask-thinking').hidden, true, 'no thinking sent: none shown');
  });

  it('errors are plain words by the server\'s code, in an alert next to Send, with Try again; the spoken error sentence plays when speech was asked', async () => {
    const ctx = await setup({
      server: {
        ask: () => json(502, { error: { message: 'connect ECONNREFUSED 127.0.0.1:9', code: 'assistant_unreachable' }, elapsed_ms: 3, sentence: "I couldn't reach the AI server.", audio: AUDIO }),
      },
    });
    await open(ctx);
    await send();
    const banner = $('ask-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.match(textOf(banner), /Couldn't reach the AI server/);
    assert.match(textOf(banner), /I couldn't reach the AI server\./, 'the sentence the preset speaks');
    assert.doesNotMatch(textOf(banner), /ECONNREFUSED/);
    assert.equal(shown('ask-compose'), true, 'the choices stay for another try');
    assert.equal(ctx.audio.current() && ctx.audio.current().id, 'ask-ai:answer', 'the sentence is spoken');
    env.click(banner.querySelector('[data-ask="retry"]'));
    await settle(env);
    await settle(env);
    assert.equal(ctx.server.asks.length, 2);
  });

  // A real spoken sentence is tens of KB of base64, far more than the 4,000 characters
  // ApiError.detail keeps for display; the sheet reads the sentence and its audio from the whole parsed body.
  it('the spoken error sentence shows and plays with a clip of real size (32 KB, about 2 s of MP3)', async () => {
    const bytes = 32000;
    const raw = new Uint8Array(bytes);
    raw.set([0x49, 0x44, 0x33]);   // an ID3 header, then zeros
    let binary = '';
    for (let i = 0; i < raw.length; i += 1) binary += String.fromCharCode(raw[i]);
    const clip = { format: 'mp3', content_type: 'audio/mpeg', base64: btoa(binary), bytes };
    assert.ok(clip.base64.length > 40000);
    const ctx = await setup({
      server: {
        ask: () => json(502, { error: { message: 'connect ECONNREFUSED 127.0.0.1:9', code: 'assistant_unreachable' }, elapsed_ms: 3, sentence: "I couldn't reach the AI server.", audio: clip }),
      },
    });
    await open(ctx);
    assert.equal($('ask-out-speak').checked, true, 'Speak it in a voice is on');
    await send();
    const banner = $('ask-error');
    assert.match(textOf(banner), /Couldn't reach the AI server/);
    assert.match(textOf(banner), /I couldn't reach the AI server\./, 'the sentence the preset speaks is in the banner');
    assert.equal(ctx.audio.current() && ctx.audio.current().id, 'ask-ai:answer', 'the sentence is spoken');
    assert.ok(String(ctx.audio.element().src).startsWith('blob:'), 'from the whole base64 audio');
  });

  it('failureBody reads the parsed body (ApiError.body); without speech asked nothing is played', async () => {
    env = installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    const body = L.failureBody({ sentence: 'Sorry.', audio: AUDIO, elapsed_ms: 12 });
    assert.deepEqual([body.sentence, body.audio && body.audio.base64, body.elapsedMs], ['Sorry.', AUDIO.base64, 12]);
    assert.equal(L.failureBody(null), null);
    assert.equal(L.failureBody('{"sentence":"cut'), null, 'a cut text never parses');
    env.restore();
    const ctx = await setup({ server: { ask: () => json(502, { error: { message: 'x', code: 'assistant_unreachable' }, sentence: 'Sorry.', audio: AUDIO }) } });
    await open(ctx);
    env.change($('ask-out-speak'), false);
    await send();
    assert.doesNotMatch(textOf($('ask-error')), /Sorry\./, 'no speech asked: no spoken sentence');
    assert.equal(ctx.audio.current(), null);
  });

  it('the screenshot state ask-ai-error shows the banner with the spoken sentence, sending and playing nothing', async () => {
    const ctx = await setup();
    assert.ok(ctx.ask.SHOTS.includes('ask-ai-error'));
    assert.equal(await ctx.ask.openSample('ask-ai-error', { trigger: ctx.button }), true);
    await settle(env);
    assert.equal(ctx.dialog.open, true);
    const banner = $('ask-error');
    assert.equal(banner.hidden, false);
    assert.match(textOf(banner), /Couldn't reach the AI server/);
    assert.match(textOf(banner), /I couldn't reach the AI server\./);
    assert.equal(shown('ask-compose'), true, 'over the choices, as after a failed Send');
    assert.equal(ctx.server.asks.length, 0, 'nothing sent');
    assert.equal(ctx.audio.current(), null, 'nothing plays');
  });

  it('a refused voice (422 output.speech) says so with the server\'s reason; an AI timeout says so', async () => {
    let answer = () => json(422, { error: { message: "voice 'zz' is not a voice of kokoro", code: 'validation_error', param: 'output.speech.voice' } });
    const ctx = await setup({ server: { ask: () => answer() } });
    await open(ctx);
    await send();
    assert.match(textOf($('ask-error')), /That voice wasn't accepted/);
    assert.match(textOf($('ask-error')), /is not a voice of kokoro/);
    answer = () => json(504, { error: { message: 'timed out', code: 'assistant_timeout' }, elapsed_ms: 120000 });
    env.change($('ask-out-speak'), false);
    await send();
    assert.match(textOf($('ask-error')), /The AI took too long to answer/);
  });

  it('a long input answers 202: the job\'s pieces show with Cancel, polled until done; then its answers and its document', async () => {
    const job = (status, done, extra = {}) => ({
      id: 'job-1', document_id: 'kept-1', preset: 'voice', status, error: null, error_message: null, result_document_id: null, blocks_requested: null,
      chapters: [{ index: 0, title: 'Transcript', status, pieces: 3, pieces_done: done, text: null, thinking: null, error: null }], ...extra,
    });
    const ctx = await setup({ server: { ask: (req, state) => { state.job = job('queued', 0); return json(202, { job: state.job, preset: 'voice', input: { kind: 'transcript', chars: 13000 } }); } } });
    await open(ctx, [transcript('Word '.repeat(2600).trim())]);
    await send();
    assert.deepEqual(ctx.server.asks[0].json.output, { document: {} }, 'a long input is saved, never spoken');
    assert.equal(shown('ask-busy'), true);
    assert.equal(shown('ask-progress-row'), true);
    assert.equal(textOf($('ask-progress-text')), '0 of 3 pieces');
    assert.equal(textOf($('ask-status')), 'Waiting for the AI…');
    ctx.server.job = job('running', 2);
    await settle(env, 1500);
    await settle(env);
    assert.equal(textOf($('ask-progress-text')), '2 of 3 pieces');
    assert.equal($('ask-progress').getAttribute('value'), '2');
    assert.equal($('ask-progress').getAttribute('max'), '3');
    ctx.server.job = job('done', 3, { result_document_id: 'answers-1', chapters: [{ index: 0, title: 'Transcript', status: 'done', pieces: 3, pieces_done: 3, text: 'Part one.\n\nPart two.', thinking: null, error: null }] });
    await settle(env, 1500);
    await settle(env);
    assert.equal(shown('ask-result'), true);
    assert.equal($('ask-reply').textContent, 'Part one.\n\nPart two.');
    assert.equal(shown('ask-open'), true);
    assert.equal(shown('ask-save'), false, 'already saved');
    const gets = ctx.server.jobGets;
    await settle(env, 5000);
    assert.equal(ctx.server.jobGets, gets, 'no polling once it ended');
  });

  it('Cancel stops a job (DELETE) and goes back to the choices; Cancel during a short request aborts it', async () => {
    const ctx = await setup({ server: { ask: (req, state) => { state.job = { id: 'job-2', status: 'running', chapters: [{ pieces: 4, pieces_done: 1 }] }; return json(202, { job: state.job }); } } });
    await open(ctx, [transcript('Word '.repeat(2600).trim())]);
    await send();
    env.click($('ask-cancel'));
    await settle(env);
    assert.deepEqual(ctx.server.jobDeletes, ['job-2']);
    assert.equal(shown('ask-compose'), true);
    env.restore();
    const slow = await setup({ server: { ask: () => new Promise(() => {}) } });
    await open(slow);
    await send();
    assert.equal(shown('ask-busy'), true);
    assert.equal(textOf($('ask-status')), 'Asking the AI…');
    env.click($('ask-cancel'));
    await settle(env);
    assert.equal(slow.server.asks[0].signal.aborted, true, 'the request is aborted');
    assert.equal(shown('ask-compose'), true);
    assert.equal($('ask-error').hidden, true, 'a cancel is no error');
  });

  it('closing the sheet aborts a request in flight; a running job goes on (a toast says so) and is no longer polled', async () => {
    const ctx = await setup({ server: { ask: (req, state) => { state.job = { id: 'job-3', status: 'running', chapters: [{ pieces: 4, pieces_done: 1 }] }; return json(202, { job: state.job }); } } });
    const { closed } = await open(ctx, [transcript('Word '.repeat(2600).trim())]);
    await send();
    env.key('Escape');
    await closed;
    await settle(env);
    assert.equal(ctx.dialog.open, false);
    assert.match(textOf(document.getElementById('app-toasts')), /The AI job goes on/);
    const gets = ctx.server.jobGets;
    await settle(env, 5000);
    assert.equal(ctx.server.jobGets, gets);
    assert.deepEqual(ctx.server.jobDeletes, [], 'closing never cancels the job');
  });

  it('several sources: a choice of what is sent, each with its size; the kind decides the presets (a document gets the book preset)', async () => {
    const ctx = await setup();
    const sources = [
      { id: 'selection', label: 'The selected paragraph', input: { kind: 'document', document_id: 'd1', block_start: 3, block_end: 3 }, text: 'A paragraph.', chars: 12, chapters: 1 },
      { id: 'document', label: 'The whole document: Doc', input: { kind: 'document', document_id: 'd1' }, text: 'All of it.', chars: 900, chapters: 3 },
    ];
    await open(ctx, sources);
    const radios = ctx.dialog.querySelectorAll('input[name="ask-source"]');
    assert.deepEqual(radios.map((r) => [r.value, r.checked]), [['selection', true], ['document', false]]);
    assert.equal($('ask-preset').value, 'book');
    assert.equal(textOf($('ask-privacy-text')), `This is sent to ${HOST}.`);
    env.change(radios[1], true);
    assert.match(textOf($('ask-privacy-text')), /answers are saved in the Reader/, 'the whole document is a job');
    await send();
    assert.deepEqual(ctx.server.asks[0].json.input, { kind: 'document', document_id: 'd1' });
    assert.equal(ctx.server.asks[0].json.preset, 'book');
  });

  it('the screenshot states show the sheet without sending anything', async () => {
    const ctx = await setup();
    for (const state of ['ask-ai', 'ask-ai-answer', 'ask-ai-job']) {
      await ctx.ask.openSample(state, { trigger: ctx.button });
      await settle(env);
      assert.equal(ctx.dialog.open, true, state);
    }
    assert.equal(shown('ask-progress-row'), true);
    assert.equal(ctx.server.asks.length, 0, 'nothing sent');
    assert.equal(ctx.audio.current(), null, 'nothing plays');
  });
});

// The sheet remembers its choices in this browser (prefs.askAi, as Talk does), a
// remembered preset or voice that no longer exists falls back without an error, and a job whose sheet was closed
// says when it ends, with Open for its answers. Also the one-part job of a long text.
describe('components/ask-ai: remembered choices and closed jobs', () => {
  const VOICES = Object.freeze({
    'GET /v1/audio/voices': () => json(200, { voices: [{ id: 'af_bella', name: 'Bella' }, { id: 'af_sky', name: 'Sky' }] }),
    'GET /api/voices/identities': () => json(200, { voices: [{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', voice: 'af_sky', available: true }] }] }),
  });
  const stored = () => JSON.parse(env.localStorage.__data['open-speech-prefs'] || '{}').askAi;
  const seedAsk = (askAi) => ({ 'open-speech-prefs': JSON.stringify({ v: 1, askAi: { v: 1, ...askAi } }) });
  const LONG = 'Word '.repeat(2600).trim();
  const quiet = () => {
    assert.equal($('ask-error').hidden, true, 'no error banner');
    assert.equal(textOf($('app-toasts-alert')), '', 'no error toast');
    assert.deepEqual(env.errors, []);
  };

  async function pick(label) {
    env.click($('ask-voice'));
    await settle(env);
    await settle(env);
    const option = [...document.querySelectorAll('#vp-list [role=option]')].find((o) => o.querySelector('.option__label').textContent === label);
    assert.ok(option, `option ${label}`);
    env.click(option);
    await settle(env);
    await settle(env);
  }

  /** A job as the server reads it: a long text's job has one part, index 0, titled as the kept document. */
  function job(id, status, done, extra = {}) {
    return {
      id, document_id: 'kept-1', preset: 'voice', title: 'Pasted text: Voice', status, error: null, error_message: null,
      result_document_id: null, chapters_requested: null, blocks_requested: null,
      chapters: [{ index: 0, title: 'Pasted text', status, pieces: 4, pieces_done: done, text: null, thinking: null, error: null }], ...extra,
    };
  }

  /** A server whose long asks each start a job, job-1, job-2, …; the routes answer the newest one. */
  function jobServer() {
    let n = 0;
    return { ask: (req, state) => { n += 1; state.job = job(`job-${n}`, 'queued', 0); return json(202, { job: state.job, preset: 'voice', input: { kind: 'transcript', chars: LONG.length } }); } };
  }

  async function sendAndClose(ctx) {
    const { closed } = await open(ctx, [transcript(LONG)]);
    await send();
    env.key('Escape');
    await closed;
    await settle(env);
  }

  // The answer-voice picker shows the cloning-model note too, as Speak's
  // does. Ask AI offers no raw recordings, so the note is its first sentence, without Clone a voice….
  it('the answer-voice picker for a model with no voices of its own says so and lists the named voices that have no version for it last', async () => {
    const ctx = await setup({ routes: {
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'chatterbox' ? [] : [{ id: 'af_bella', name: 'Bella' }] }),
      'GET /api/voices/identities': () => json(200, { voices: [
        { id: 'narrator', name: 'Narrator', available: false, realizations: [{ model: 'kokoro', voice: 'af_sky', available: false, reason: "Select this realization's model" }] },
        { id: 'butler', name: 'Butler', available: true, realizations: [{ model: 'chatterbox', voice: 'butler_take', available: true }] },
      ] }),
    } });
    ctx.app.set({
      models: { list: [{ id: 'kokoro', type: 'tts', provider: 'kokoro', state: 'loaded' }, { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded' }], defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' },
      speakVoice: { model: 'chatterbox', voice: 'voice:butler', label: 'Butler', kind: 'named' },
    });
    await open(ctx);
    env.click($('ask-voice'));
    await settle(env);
    await settle(env);
    assert.equal($('app-voice-picker').open, true);
    assert.equal($('vp-model-note').hidden, false, 'the note shows');
    assert.equal($('vp-model-note-text').textContent, 'Chatterbox has no built-in voices.');
    assert.equal($('vp-clone').hidden, true, 'no raw recordings here: no Clone a voice…');
    const groupNames = [...document.querySelectorAll('#vp-list [role=group] .listbox__label')].map((g) => g.textContent);
    assert.deepEqual(groupNames, ['Named voices', 'Named voices with no Chatterbox version']);
  });

  it('the preset per kind, the outputs and the answer voice are remembered in this browser, and the next page starts with them', async () => {
    let ctx = await setup({ routes: VOICES });
    await open(ctx);
    env.change($('ask-preset'), 'summarize');
    env.change($('ask-out-speak'), true);
    env.change($('ask-out-save'), true);
    await pick('Sky');
    assert.equal(textOf($('ask-voice-name')), 'Voice: Kokoro · Sky');
    assert.deepEqual(stored().voice, { model: 'kokoro', voice: 'af_sky', label: 'Sky' }, 'a chosen voice is kept at once');
    await send();
    assert.deepEqual(stored(), {
      v: 1, presets: { transcript: 'summarize' }, outputs: { kind: 'transcript', preset: 'summarize', show: true, speak: true, save: true },
      voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky' }, questions: [],   // no question asked
    }, 'kept with the other preferences (open-speech-prefs) when sent');
    const seed = { 'open-speech-prefs': env.localStorage.__data['open-speech-prefs'] };
    env.restore();

    // The next page load.
    ctx = await setup({ routes: VOICES, seed });
    await open(ctx);
    assert.equal($('ask-preset').value, 'summarize');
    assert.deepEqual([$('ask-out-show').checked, $('ask-out-speak').checked, $('ask-out-save').checked], [true, true, true]);
    assert.equal(textOf($('ask-voice-name')), 'Voice: Kokoro · Sky', 'the voice the server still offers');
    await send();
    assert.deepEqual(ctx.server.asks[0].json, {
      input: { kind: 'transcript', text: TRANSCRIPT }, preset: 'summarize',
      output: { speech: { format: 'mp3', model: 'kokoro', voice: 'af_sky' }, document: {} },
    });
    quiet();
    env.click($('ask-cancel'));
    await settle(env);

    // Another kind keeps its own preset: a document starts with the server's default for it.
    await open(ctx, [{ id: 'document', label: 'The whole document: Doc', input: { kind: 'document', document_id: 'd1' }, text: 'All of it.', chars: 10, chapters: 1 }]);
    assert.equal($('ask-preset').value, 'book');
    assert.equal($('ask-out-speak').checked, false, "the book preset's own outputs");
  });

  it("a remembered preset that no longer exists, or a voice the server can't speak with any more, falls back without an error", async () => {
    const gone = { model: 'kokoro', voice: 'af_gone', label: 'Gone' };
    const memory = { presets: { transcript: 'deleted' }, outputs: { kind: 'transcript', preset: 'deleted', show: false, speak: false, save: true }, voice: gone, questions: [] };
    let ctx = await setup({ routes: VOICES, seed: seedAsk(memory) });
    await open(ctx);
    assert.equal($('ask-preset').value, 'voice', "the server's default");
    assert.equal($('ask-out-speak').checked, true, "the voice preset's own outputs");
    assert.equal(textOf($('ask-voice-name')), 'Voice: Same as Speak · Bella', 'a voice the model no longer has');
    quiet();
    assert.deepEqual(stored(), { v: 1, ...memory }, 'a fallback writes nothing: the choice stays until the listener makes another');
    await send();
    assert.equal(ctx.server.asks[0].json.output.speech.voice, 'af_bella');
    quiet();
    env.restore();

    // A model this server can't speak with now: no voice list is asked for.
    ctx = await setup({ routes: VOICES, seed: seedAsk({ voice: { model: 'qwen3-tts', voice: 'Ryan', label: 'Ryan' } }) });
    await open(ctx);
    assert.equal(textOf($('ask-voice-name')), 'Voice: Same as Speak · Bella');
    assert.equal(env.fetch.callsTo('GET /v1/audio/voices').length, 0);
    quiet();
    env.restore();

    // A named voice: kept while the server offers it for the model; gone or unavailable, Speak's.
    for (const [named, want] of [
      [null, 'Voice: Narrator'],
      [[], 'Voice: Same as Speak · Bella'],
      [[{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', available: false, reason: 'No Kokoro version' }] }], 'Voice: Same as Speak · Bella'],
    ]) {
      const routes = named ? { ...VOICES, 'GET /api/voices/identities': () => json(200, { voices: named }) } : VOICES;
      ctx = await setup({ routes, seed: seedAsk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }) });
      await open(ctx);
      assert.equal(textOf($('ask-voice-name')), want);
      quiet();
      env.restore();
    }

    // The voice list fails to load: nothing says the voice is gone, so it stays.
    ctx = await setup({ routes: { ...VOICES, 'GET /v1/audio/voices': () => json(503, { error: { message: 'busy' } }) }, seed: seedAsk({ voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky' } }) });
    await open(ctx);
    assert.equal(textOf($('ask-voice-name')), 'Voice: Kokoro · Sky');
    quiet();
  });

  // Send pressed before the check of a remembered voice ended asked for a deleted
  // voice, and the server refused it ("That voice wasn't accepted").
  it('Send pressed while the remembered voice is still being checked waits for the check: a deleted voice is never asked for, a kept one is', async () => {
    for (const [named, want] of [[[], 'af_bella'], [null, 'voice:narrator']]) {
      let release;
      const held = new Promise((resolve) => { release = resolve; });
      const list = named || [{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', voice: 'af_sky', available: true }] }];
      const ctx = await setup({
        routes: { ...VOICES, 'GET /api/voices/identities': () => held.then(() => json(200, { voices: list })) },
        seed: seedAsk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }),
      });
      await open(ctx);
      assert.equal($('ask-out-speak').checked, true);
      assert.equal(env.fetch.callsTo('GET /api/voices/identities').length, 1, 'the check waits for the list');
      await send();
      assert.equal(ctx.server.asks.length, 0, 'nothing is asked before the check ends');
      assert.equal(shown('ask-busy'), true, 'the sheet says it is asking meanwhile');
      assert.equal(shown('ask-send'), false, 'and Send is not offered twice');
      release();
      await settle(env);
      await settle(env);
      assert.equal(ctx.server.asks.length, 1);
      assert.equal(ctx.server.asks[0].json.output.speech.voice, want, named ? "deleted: Speak's voice, quietly" : 'still offered: kept');
      quiet();
      env.restore();
    }
  });

  it('Cancel while Send waits for the voice check asks nothing', async () => {
    let release;
    const held = new Promise((resolve) => { release = resolve; });
    const ctx = await setup({
      routes: { ...VOICES, 'GET /api/voices/identities': () => held.then(() => json(200, { voices: [] })) },
      seed: seedAsk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }),
    });
    await open(ctx);
    await send();
    env.click($('ask-cancel'));
    await settle(env);
    release();
    await settle(env);
    await settle(env);
    assert.equal(ctx.server.asks.length, 0, 'canceled before it was asked');
    assert.equal($('ask-send').hidden, false, 'back to the question');
    quiet();
  });

  it('a job whose sheet was closed: a toast says when it is done, with Open for its answers', async () => {
    const ctx = await setup({ server: jobServer() });
    const { on } = await load('core/store.js');
    const changed = [];
    const off = on('library:changed', (detail) => changed.push(detail && detail.docId));
    await sendAndClose(ctx);
    assert.match(textOf($('app-toasts')), /The AI job goes on\. A notice says when it is done/);
    const gets = ctx.server.jobGets;
    await settle(env, 9000);
    assert.equal(ctx.server.jobGets, gets, 'not at the sheet\'s pace once closed');
    ctx.server.job = job('job-1', 'running', 2);
    await settle(env, 1000);
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 1, 'read every 10 s while it runs');
    assert.doesNotMatch(textOf($('app-toasts')), /finished/);
    ctx.server.job = job('job-1', 'done', 4, { result_document_id: 'answers-1' });
    await settle(env, 10000);
    await settle(env);
    const toastEl = document.querySelector('#app-toasts-polite .toast');
    assert.ok(toastEl, 'a status toast, not an alert');
    assert.equal(textOf(toastEl.querySelector('.toast__msg')), 'The AI finished “Pasted text: Voice”. Its answers are in the Reader.');
    assert.equal(textOf($('app-toasts-alert')), '');
    assert.ok(changed.includes('answers-1'), "the Reader's library hears of the new document");
    const after = ctx.server.jobGets;
    await settle(env, 3000);
    assert.equal(ctx.server.jobGets, after, 'no reads once it ended');
    const openButton = [...toastEl.querySelectorAll('button')].find((b) => textOf(b) === 'Open');
    assert.ok(openButton, 'Open for its answers');
    env.click(openButton);
    await settle(env);
    assert.equal(location.hash, '#reader/answers-1');
    off();
  });

  it('a job that fails while its sheet is closed: an error toast says why, in plain words, and stays', async () => {
    const ctx = await setup({ server: jobServer() });
    await sendAndClose(ctx);
    ctx.server.job = job('job-1', 'failed', 1, { error: 'assistant_unreachable', error_message: 'connect ECONNREFUSED 127.0.0.1:9' });
    await settle(env, 10000);
    await settle(env);
    const alert = textOf($('app-toasts-alert'));
    assert.match(alert, /The AI job “Pasted text: Voice” stopped: Couldn't reach the AI server\. Check that it is running/);
    assert.doesNotMatch(alert, /ECONNREFUSED/, "never the AI server's own words");
    assert.equal([...document.querySelectorAll('#app-toasts-alert .toast button')].some((b) => textOf(b) === 'Open'), false, 'no answers to open');
    const gets = ctx.server.jobGets;
    await settle(env, 60000);
    assert.match(textOf($('app-toasts-alert')), /stopped/, 'an error toast stays');
    assert.equal(ctx.server.jobGets, gets, 'no reads once it ended');
  });

  it('a closed job is not read while the page is hidden, is read at once when it shows again, and no longer once the page is left', async () => {
    const ctx = await setup({ server: jobServer() });
    const listeners = () => [window, document].map((target) => ['visibilitychange', 'pagehide', 'pageshow']
      .map((type) => ((target.__listeners && target.__listeners.get(type)) || []).length).join(',')).join(' ');
    const before = listeners();
    await sendAndClose(ctx);
    let gets = ctx.server.jobGets;
    env.setHidden(true);
    await settle(env, 60000);
    assert.equal(ctx.server.jobGets, gets, 'hidden: no reads');
    ctx.server.job = job('job-1', 'done', 4, { result_document_id: 'answers-1' });
    env.setHidden(false);
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 1, 'shown again: read at once');
    assert.match(textOf($('app-toasts-polite')), /The AI finished/);

    // A page kept in the back/forward cache comes back to its job; a page that is left stops reading it.
    await sendAndClose(ctx);
    gets = ctx.server.jobGets;
    env.pageHide({ persisted: true });
    await settle(env, 30000);
    assert.equal(ctx.server.jobGets, gets);
    env.pageShow({ persisted: true });
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 1, 'back from the cache: read at once, once');
    assert.notEqual(listeners(), before, 'a job being watched listens to the page');
    env.pageHide();
    ctx.server.job = job('job-2', 'done', 4, { result_document_id: 'answers-2' });
    await settle(env, 60000);
    assert.equal(ctx.server.jobGets, gets + 1, 'the page is left: no more reads');
    assert.doesNotMatch(textOf($('app-toasts')), /The AI finished/, 'and nothing to tell');
    assert.equal(listeners(), before, 'no page listener is left behind');
  });

  it('a long text\'s one-part job shows progress by piece; a job that reports no pieces shows no made-up fraction', async () => {
    const ctx = await setup({ server: jobServer() });
    await open(ctx, [transcript(LONG)]);
    await send();
    assert.equal(textOf($('ask-progress-text')), '0 of 4 pieces');
    ctx.server.job = job('job-1', 'running', 1);
    await settle(env, 1500);
    await settle(env);
    assert.equal(textOf($('ask-progress-text')), '1 of 4 pieces');
    assert.deepEqual([$('ask-progress').getAttribute('value'), $('ask-progress').getAttribute('max')], ['1', '4']);
    ctx.server.job = job('job-1', 'done', 4, { result_document_id: 'answers-1', chapters: [{ index: 0, title: 'Pasted text', status: 'done', pieces: 4, pieces_done: 4, text: 'One.\n\nTwo.', thinking: null, error: null }] });
    await settle(env, 1500);
    await settle(env);
    assert.equal($('ask-reply').textContent, 'One.\n\nTwo.', 'the one part\'s answers, without its title');
    assert.equal(shown('ask-open'), true);
    env.click($('ask-cancel'));
    await settle(env);

    // No pieces reported (a job read before its plan): the status says it works; there is no bar to read.
    const bare = { ...job('job-9', 'running', 0), chapters: [] };
    env.restore();
    const other = await setup({ server: { ask: (req, state) => { state.job = bare; return json(202, { job: bare }); } } });
    await open(other, [transcript(LONG)]);
    await send();
    assert.equal(textOf($('ask-status')), 'The AI is working through it…');
    assert.equal(shown('ask-progress-row'), false, 'no "0 of 0" and no empty bar');
    other.server.job = job('job-9', 'running', 1);
    await settle(env, 1500);
    await settle(env);
    assert.equal(shown('ask-progress-row'), true);
    assert.equal(textOf($('ask-progress-text')), '1 of 4 pieces');
  });
});

// A question box in the shared sheet, for every entry point (the server
// takes it as `question`): one line that grows, the quick picks, the last few questions again (kept in
// this browser, prefs.askAi.questions); and the "what is sent" line naming what goes with a source (the Reader's
// document name and chapter).
describe('components/ask-ai: the question box', () => {
  const stored = () => JSON.parse(env.localStorage.__data['open-speech-prefs'] || '{}').askAi;
  const seedQuestions = (questions) => ({ 'open-speech-prefs': JSON.stringify({ v: 1, askAi: { v: 1, questions } }) });
  const chips = (id) => [...$(id).querySelectorAll('button')].map((b) => textOf(b));

  it('questionOf and rememberQuestion: a question is sent trimmed, up to 2,000 characters by code point; blank is none; the newest three are kept', async () => {
    installBrowserEnv();
    const L = await importFresh('components/ask-ai-logic.js');
    assert.deepEqual([L.QUESTION_MAX_CHARS, L.RECENT_QUESTIONS], [2000, 3]);
    assert.deepEqual(L.QUICK_QUESTIONS, ['Explain this', 'Summarize', 'Define the hard words']);
    assert.equal(L.questionOf('  Explain this \n'), 'Explain this');
    assert.equal(L.questionOf('   \n '), '');
    assert.equal(L.questionOf(null), '');
    assert.equal(L.questionOf('x'.repeat(2500)).length, 2000);
    const emoji = L.questionOf('😀'.repeat(2001));
    assert.equal(Array.from(emoji).length, 2000, 'cut by code point, as the server counts');
    assert.equal(/[\uD800-\uDBFF]$/.test(emoji), false, 'never a lone surrogate');
    const source = transcript('Hi there.');
    assert.deepEqual(L.requestBody({ preset: 'voice', source, outputs: { show: true }, question: 'Explain this' }), { input: { kind: 'transcript', text: 'Hi there.' }, preset: 'voice', question: 'Explain this' });
    assert.deepEqual(L.requestBody({ preset: 'voice', source, outputs: { show: true }, question: '  ' }), { input: { kind: 'transcript', text: 'Hi there.' }, preset: 'voice' }, 'a blank question is left out (the server refuses one)');
    assert.deepEqual(L.rememberQuestion(['B', 'A', 'C'], 'A'), ['A', 'B', 'C'], 'asked again: first');
    assert.deepEqual(L.rememberQuestion(['B', 'A', 'C'], 'D'), ['D', 'B', 'A'], 'the newest three');
    assert.deepEqual(L.rememberQuestion(['B'], ''), ['B'], 'no question: unchanged');
    assert.deepEqual(L.recentChoices(['Summarize', 'What does it mean?', 'Explain this']), ['What does it mean?'], 'a quick pick is not offered twice');
  });

  it('the box: one labelled line that grows, up to 2,000 characters, with the placeholder; a quick pick fills it; Send sends it as `question`, and a blank box sends none', async () => {
    const ctx = await setup();
    await open(ctx);
    const box = $('ask-question');
    assert.equal(box.localName, 'textarea');
    assert.deepEqual([box.getAttribute('rows'), box.getAttribute('maxlength'), box.getAttribute('placeholder')], ['1', '2000', 'Ask something about it (optional)']);
    assert.equal(textOf(ctx.dialog.querySelector('label[for="ask-question"]')), 'Your question');
    assert.equal(box.value, '', 'empty: the question is optional');
    assert.deepEqual(chips('ask-picks'), ['Explain this', 'Summarize', 'Define the hard words']);
    assert.equal(shown('ask-recent'), false, 'no recent questions yet');
    assertAccessibleNames(ctx.dialog);
    env.input(box, '   ');
    await send();
    assert.equal('question' in ctx.server.asks[0].json, false, 'only spaces: no question');
    env.click($('ask-again'));
    await settle(env);
    env.click([...$('ask-picks').querySelectorAll('button')].find((b) => textOf(b) === 'Explain this'));
    assert.equal(box.value, 'Explain this', 'a quick pick is just text in the box');
    assert.equal(document.activeElement, box, 'the box has focus, to add to it');
    await send();
    assert.deepEqual(ctx.server.asks[1].json, { input: { kind: 'transcript', text: TRANSCRIPT }, preset: 'voice', question: 'Explain this', output: { speech: { format: 'mp3', model: 'kokoro', voice: 'af_bella' } } });
    env.click($('ask-again'));
    await settle(env);
    assert.equal(box.value, 'Explain this', 'Send again keeps the question');
    env.input(box, '  What does “first” mean here?\n');
    await send();
    assert.equal(ctx.server.asks[2].json.question, 'What does “first” mean here?', 'sent trimmed');
  });

  it('the last three questions are offered again on the next page (kept in this browser); a pick of one fills the box', async () => {
    let ctx = await setup();
    await open(ctx);
    for (const q of ['Who is speaking?', 'Explain this', 'What does “first” mean here?', 'Is this a promise?']) {
      env.input($('ask-question'), q);
      await send();
      env.click($('ask-again'));
      await settle(env);
    }
    assert.deepEqual(stored().questions, ['Is this a promise?', 'What does “first” mean here?', 'Explain this'], 'newest first, three at most');
    assert.deepEqual(chips('ask-recent'), ['Is this a promise?', 'What does “first” mean here?'], 'offered at once; a quick pick is not offered twice');
    const seed = { 'open-speech-prefs': env.localStorage.__data['open-speech-prefs'] };
    env.restore();

    ctx = await setup({ seed });
    await open(ctx);
    assert.equal($('ask-question').value, '', 'a new sheet starts empty');
    assert.equal(shown('ask-recent'), true);
    assert.equal($('ask-recent').getAttribute('role'), 'group');
    assert.equal($('ask-recent').getAttribute('aria-label'), 'Recent questions');
    assert.deepEqual(chips('ask-recent'), ['Is this a promise?', 'What does “first” mean here?']);
    env.click([...$('ask-recent').querySelectorAll('button')][1]);
    assert.equal($('ask-question').value, 'What does “first” mean here?');
    await send();
    assert.equal(ctx.server.asks[0].json.question, 'What does “first” mean here?');
    assert.deepEqual(stored().questions, ['What does “first” mean here?', 'Is this a promise?', 'Explain this'], 'asked again: first');
    env.restore();

    // A long remembered question is shown short; the whole of it goes.
    const long = `Why ${'very '.repeat(30)}long?`;
    ctx = await setup({ seed: seedQuestions([long]) });
    await open(ctx);
    const chip = $('ask-recent').querySelector('button');
    assert.ok(textOf(chip).length <= 60 && textOf(chip).endsWith('…'), textOf(chip));
    env.click(chip);
    assert.equal($('ask-question').value, long);
  });

  it('a question goes with every kind and with a long input\'s job; the "what is sent" line shows what goes with a source', async () => {
    const ctx = await setup({ server: { ask: (req, state) => { state.job = { id: 'job-q', status: 'running', chapters: [{ pieces: 4, pieces_done: 0 }] }; return json(202, { job: state.job }); } } });
    const doc = { id: 'document', label: 'The whole document: Doc', input: { kind: 'document', document_id: 'd1' }, text: 'All of it.', chars: 900, chapters: 3, named: 'The AI is told it\'s from “Doc”, and the chapter of each part.' };
    await open(ctx, [doc]);
    assert.equal(textOf($('ask-named')), 'The AI is told it\'s from “Doc”, and the chapter of each part.');
    env.input($('ask-question'), 'Summarize');
    await send();
    assert.deepEqual(ctx.server.asks[0].json, { input: { kind: 'document', document_id: 'd1' }, preset: 'book', question: 'Summarize', output: { document: {} } });
    env.click($('ask-cancel'));
    await settle(env);
    env.click($('ask-cancel'));
    await settle(env);
    await open(ctx);
    assert.equal(shown('ask-named'), false, 'a source with nothing more to say shows no line');
  });

  it('the screenshot state ask-ai-question shows a question in the box, sending nothing', async () => {
    const ctx = await setup();
    assert.ok(ctx.ask.SHOTS.includes('ask-ai-question'));
    assert.equal(await ctx.ask.openSample('ask-ai-question', { trigger: ctx.button }), true);
    await settle(env);
    assert.equal(ctx.dialog.open, true);
    assert.equal($('ask-question').value, 'Explain this to me.');
    assert.equal(shown('ask-compose'), true);
    assert.equal(ctx.server.asks.length, 0, 'nothing sent');
    assert.equal(stored(), undefined, 'nothing remembered');
  });
});

// On a short viewport the error banner sits below the sheet's fold, after the choices, so a
// failure looked like nothing happened; and the open sheet read its job every 1.5 s while the page was hidden (only
// the closed-sheet watch paused). The banner, and the answer, are brought into view inside the sheet
// (no smooth scroll under reduced motion); the open sheet's polling pauses while the page is hidden and reads at once
// when it shows again.
describe('components/ask-ai: the banner in view, and no polling while hidden', () => {
  const LONG = 'Word '.repeat(2600).trim();
  const running = (id, done = 0, extra = {}) => ({ id, document_id: 'kept-1', preset: 'voice', title: 'Pasted text: Voice', status: 'running', error: null, error_message: null, result_document_id: null, chapters: [{ index: 0, title: 'Pasted text', status: 'running', pieces: 4, pieces_done: done, text: null, thinking: null, error: null }], ...extra });

  it('a failed Send brings the error banner into view (the nearest edge, smooth unless motion is reduced); an answer is brought into view too', async () => {
    let answer = () => json(502, { error: { message: 'connect ECONNREFUSED 127.0.0.1:9', code: 'assistant_unreachable' } });
    const ctx = await setup({ server: { ask: () => answer() } });
    await open(ctx);
    const banner = $('ask-error');
    assert.deepEqual(banner.__scrolledIntoView, [], 'nothing to show yet');
    await send();
    assert.equal(banner.hidden, false);
    assert.deepEqual(banner.__scrolledIntoView, [{ block: 'nearest', behavior: 'smooth' }], 'the banner, not only the top of the choices');
    env.setMedia('prefers-reduced-motion: reduce', true);
    env.click(banner.querySelector('[data-ask="retry"]'));
    await settle(env);
    await settle(env);
    assert.deepEqual(banner.__scrolledIntoView.at(-1), { block: 'nearest', behavior: 'auto' }, 'reduced motion: no smooth scroll');
    answer = () => json(200, { reply: 'Ship the Reader first.', thinking: null, finish_reason: 'stop', elapsed_ms: 10, preset: 'voice', input: {} });
    env.change($('ask-out-speak'), false);
    await send();
    assert.equal(shown('ask-result'), true);
    assert.deepEqual($('ask-result').__scrolledIntoView, [{ block: 'nearest', behavior: 'auto' }], 'the answer comes into view');
  });

  it('a job that fails while the sheet shows it brings its banner into view', async () => {
    const ctx = await setup({ server: { ask: (req, state) => { state.job = running('job-f'); return json(202, { job: state.job }); } } });
    await open(ctx, [transcript(LONG)]);
    await send();
    ctx.server.job = running('job-f', 1, { status: 'failed', error: 'assistant_unreachable' });
    await settle(env, 1500);
    await settle(env);
    assert.match(textOf($('ask-error')), /Couldn't reach the AI server/);
    assert.deepEqual($('ask-error').__scrolledIntoView, [{ block: 'nearest', behavior: 'smooth' }]);
  });

  it('the open sheet does not read its job while the page is hidden, reads it at once when it shows again, and leaves no page listener behind', async () => {
    const ctx = await setup({ server: { ask: (req, state) => { state.job = running('job-h'); return json(202, { job: state.job }); } } });
    const listeners = () => ((document.__listeners && document.__listeners.get('visibilitychange')) || []).length;
    const before = listeners();
    const { closed } = await open(ctx, [transcript(LONG)]);
    await send();
    await settle(env, 1500);
    await settle(env);
    let gets = ctx.server.jobGets;
    assert.equal(gets, 1, 'read every 1.5 s while shown');
    env.setHidden(true);
    await settle(env, 60000);
    assert.equal(ctx.server.jobGets, gets, 'hidden: not read');
    ctx.server.job = running('job-h', 3);
    env.setHidden(false);
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 1, 'shown again: read at once');
    assert.equal(textOf($('ask-progress-text')), '3 of 4 pieces');
    await settle(env, 1500);
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 2, 'then every 1.5 s again');

    // Hidden before the first read; a page kept in the back/forward cache comes back to it.
    gets = ctx.server.jobGets;
    env.pageHide({ persisted: true });
    await settle(env, 30000);
    assert.equal(ctx.server.jobGets, gets);
    ctx.server.job = running('job-h', 4, { status: 'done', result_document_id: 'answers-h', chapters: [{ index: 0, title: 'Pasted text', status: 'done', pieces: 4, pieces_done: 4, text: 'All done.', thinking: null, error: null }] });
    env.pageShow({ persisted: true });
    await settle(env);
    assert.equal(ctx.server.jobGets, gets + 1, 'back from the cache: read at once, once');
    assert.equal($('ask-reply').textContent, 'All done.');
    await settle(env, 5000);
    assert.equal(ctx.server.jobGets, gets + 1, 'ended: no more reads');
    env.click($('ask-cancel'));
    await closed;
    assert.equal(listeners(), before, 'the sheet listens to the page only while it is open');
  });
});
