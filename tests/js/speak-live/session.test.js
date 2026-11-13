// Speak: Live session engine: the protocol against a scripted server, with the
// harness's FakeSocket and FakeAudioContext. Inventory LR-02…LR-08, LR-10, LR-11, LR-13…LR-18, LR-22, LR-23
// and BUG-20.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  liveEnv, load, settle, voiceConfig, server, sent, toasts, deferred, json, FakeAudioContext,
} from './helpers.js';

let env;
afterEach(() => env?.restore());

/**
 * A Live engine on a fake editor. `config` may be a function (read at every getVoiceConfig() call).
 * @returns {Promise<{live, engine, states, calls, setConfig}>}
 */
async function setup({ key = '', config = voiceConfig(), text = '', caret, ensureReady, mode = 'natural', routes = {}, clock } = {}) {
  env = await liveEnv({ key, routes });
  const live = await load('views/speak/live-session.js');
  env.editor.value = text;
  env.editor.selectionStart = caret ?? text.length;
  let cfg = config;
  const calls = { config: 0 };
  const states = [];
  const engine = live.createLiveEngine({
    editor: env.editor,
    getVoiceConfig: () => { calls.config += 1; return typeof cfg === 'function' ? cfg() : cfg; },
    getMode: () => mode,
    onChange: (s) => states.push(s),
    ...(ensureReady ? { ensureReady } : {}),
    ...(clock ? { clock } : {}),
  });
  return { live, engine, states, calls, setConfig: (c) => { cfg = c; } };
}

/** Start, open the socket and configure the session; returns the scripted server. */
async function connect(engine, { readAll = false } = {}) {
  assert.equal(env.withGesture(() => engine.start({ readAll })), true, 'start() accepted');
  await settle(env);
  const ws = env.sockets[env.sockets.length - 1];
  assert.ok(ws, 'a socket was opened');
  const srv = server(ws);
  srv.open();
  srv.created();
  srv.updated();
  await settle(env);
  return srv;
}

describe('Live: start', () => {
  it('[LR-02] Start refuses a model without Live support and a missing voice, opens nothing, and a second Start while running is a no-op', async () => {
    const { live, engine, setConfig } = await setup({ config: voiceConfig({ caps: { live_reader: false } }) });
    assert.equal(engine.start(), false);
    assert.equal(engine.getState().refusal, "This model needs a reference recording and isn't available in Live yet");
    assert.equal(env.sockets.length, 0, 'no socket');
    assert.equal(FakeAudioContext.instances.length, 0, 'no AudioContext');
    assert.equal(engine.isActive(), false, 'the controls stay usable');

    setConfig(voiceConfig({ voice: '' }));
    assert.equal(engine.start(), false);
    assert.equal(engine.getState().refusal, 'Choose a voice or a reference recording first');
    assert.equal(env.sockets.length, 0);

    setConfig(voiceConfig());
    env.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    assert.equal(engine.start(), false, 'nothing plays while the microphone records');
    assert.equal(engine.getState().refusal, 'Stop recording to listen.');
    env.app.set({ recording: null });

    assert.equal(engine.start(), true);
    assert.equal(engine.getState().refusal, '', 'a successful start clears the refusal');
    assert.equal(engine.start(), false, 'already running: no-op');
    await settle(env);
    assert.equal(env.sockets.length, 1, 'one socket only');
    assert.equal(FakeAudioContext.instances.length, 1);
    assert.equal(live.refusalFor({ config: { model: 'kokoro', voice: '' }, caps: {} }), 'Choose a voice or a reference recording first');
    engine.stop();
  });

  it('[LR-03] Start reads from the caret; Read all reads from the beginning', async () => {
    const text = 'First sentence. Second sentence.';
    const { engine } = await setup({ text, caret: 16 });
    const srv = await connect(engine);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Second sentence.']);
    srv.accepted(16);
    await settle(env);
    assert.equal(sent(srv.ws).at(-1).type, 'input_text.commit', 'the initial text is committed once accepted');
    engine.stop();
    await settle(env);

    const srv2 = await connect(engine, { readAll: true });
    assert.deepEqual(sent(srv2.ws, 'input_text.append').map((e) => e.text), [text], 'Read all ignores the caret');
    engine.stop();
  });

  it('[LR-04] the session config is snapshotted before any await and sent unchanged', async () => {
    const ready = deferred();
    const { engine, calls, setConfig } = await setup({
      config: voiceConfig({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', speed: 1.25, instructions: '  Calm.  ', voice_library_ref: 'demo_reference' }),
      ensureReady: () => ready.promise,
      mode: 'responsive',
    });
    assert.equal(engine.start(), true);
    assert.equal(calls.config, 1, 'read once, synchronously, inside the click');
    setConfig(voiceConfig({ voice: 'am_adam', speed: 2 }));   // the user changes the voice while the model loads
    ready.resolve(true);
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created();
    await settle(env);
    assert.deepEqual(sent(srv.ws, 'session.update'), [{
      type: 'session.update',
      session: { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', speed: 1.25, latency_mode: 'responsive', instructions: 'Calm.', voice_library_ref: 'demo_reference' },
    }]);
    engine.stop();
    await settle(env);

    // Empty instructions and no reference are left out.
    setConfig(voiceConfig({ instructions: '   ', voice_library_ref: '' }));
    const srv2 = await connect(engine);
    assert.deepEqual(sent(srv2.ws, 'session.update')[0].session, { model: 'kokoro', voice: 'af_heart', speed: 1, latency_mode: 'responsive' });
    engine.stop();
  });

  it('[LR-05] the AudioContext is created and resumed inside the click, before the model check or the socket', async () => {
    const ready = deferred();
    const order = [];
    const { engine } = await setup({ ensureReady: () => { order.push('ensureReady'); return ready.promise; } });
    FakeAudioContext.defaults.state = 'suspended';   // a new context starts suspended until resume()
    const resumes = [];
    const original = FakeAudioContext.prototype.resume;
    FakeAudioContext.prototype.resume = function resume() { resumes.push(order.length); order.push('resume'); return original.call(this); };
    try {
      engine.start();
      assert.equal(FakeAudioContext.instances.length, 1, 'constructed synchronously');
      assert.equal(FakeAudioContext.instances[0].state, 'running', 'resumed synchronously');
      assert.deepEqual(order, ['resume', 'ensureReady'], 'resume() runs before the model check starts');
      assert.equal(env.sockets.length, 0);
      ready.resolve(true);
      await settle(env);
      assert.equal(env.sockets.length, 1);
      assert.equal(FakeAudioContext.instances[0].__sources.length, 0, 'Web Audio buffers only; no media element is routed through it');
    } finally {
      FakeAudioContext.prototype.resume = original;
      engine.stop();
    }
  });

  it('[LR-06] the model is made ready first, then the socket opens with a fresh ticket; a cancelled switch stops and closes the context', async () => {
    const ready = deferred();
    const seen = [];
    const { engine } = await setup({ key: 'secret-key', ensureReady: (model, opts) => { seen.push([model, opts.kind]); return ready.promise; } });
    engine.start();
    await settle(env);
    assert.deepEqual(seen, [['kokoro', 'tts']]);
    assert.equal(env.sockets.length, 0, 'no socket while the model loads');
    assert.equal(engine.getState().status, 'Preparing…');
    ready.resolve(true);
    await settle(env);
    const url = new URL(env.sockets[0].url);
    assert.equal(url.pathname, '/v1/audio/speech/stream');
    assert.equal(url.protocol, 'ws:');
    assert.equal(url.searchParams.get('ticket'), 't1');
    assert.equal(url.searchParams.has('api_key'), false, 'the key never goes in the URL');
    assert.ok(!env.sockets[0].url.includes('secret-key'));
    engine.stop();
    await settle(env);

    // A second session gets its own ticket.
    const again = deferred();
    const second = (await load('views/speak/live-session.js')).createLiveEngine({
      editor: env.editor, getVoiceConfig: () => voiceConfig(), ensureReady: () => again.promise,
    });
    second.start();
    again.resolve(true);
    await settle(env);
    assert.equal(new URL(env.sockets[1].url).searchParams.get('ticket'), 't2');
    second.stop();
    await settle(env);

    // The listener cancels the "Switch speech model?" confirm: Stopped, the context closed, no socket.
    const cancelled = deferred();
    const third = (await load('views/speak/live-session.js')).createLiveEngine({
      editor: env.editor, getVoiceConfig: () => voiceConfig(), ensureReady: () => cancelled.promise,
    });
    third.start();
    const ctx = FakeAudioContext.instances.at(-1);
    cancelled.resolve(false);
    await settle(env);
    assert.equal(third.isActive(), false);
    assert.equal(third.getState().status, 'Stopped');
    assert.equal(third.getState().error, null);
    assert.equal(ctx.state, 'closed');
    assert.equal(env.sockets.length, 2, 'no socket for the cancelled start');
  });

  it('[LR-06] a model that fails to load releases the session with the reason (no socket)', async () => {
    const { engine } = await setup({ ensureReady: async () => { throw new Error('The selected voice provider is offline'); } });
    engine.start();
    await settle(env);
    const state = engine.getState();
    assert.equal(state.active, false);
    assert.equal(state.status, 'Error');
    assert.equal(state.error.title, 'Live stopped');
    assert.equal(state.error.message, 'The selected voice provider is offline');
    assert.equal(env.sockets.length, 0);
    assert.equal(FakeAudioContext.instances[0].state, 'closed');
  });
});

describe('Live: protocol', () => {
  it('[LR-08] session.created sets the buffer limit and sends session.update once; session.updated configures and sends the text', async () => {
    const { engine } = await setup({ text: 'Hello there.', caret: 0 });
    engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created({ max_buffer_chars: 5 });
    srv.created({ max_buffer_chars: 5 });
    await settle(env);
    assert.equal(sent(srv.ws, 'session.update').length, 1, 'sent once');
    assert.equal(sent(srv.ws, 'input_text.append').length, 0, 'no text before session.updated');
    assert.equal(engine.getState().status, 'Connecting…');
    srv.updated();
    await settle(env);
    assert.equal(engine.getState().status, 'Listening');
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Hello'], 'chunks fit max_buffer_chars');
    engine.stop();
    await settle(env);

    // Without limits the default is 8192.
    const live = await load('views/speak/live-session.js');
    const flow = live.createTextFlow();
    flow.setLimits(undefined);
    assert.equal(flow.state.maxBufferChars, 8192);
  });

  it('[LR-08] a page that is hidden when the session is configured pauses at once', async () => {
    const { engine } = await setup({ text: 'Text.', caret: 0 });
    engine.start();
    await settle(env);
    env.setHidden(true);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created();
    srv.updated();
    await settle(env);
    assert.equal(engine.getState().status, 'Paused in background');
    assert.equal(FakeAudioContext.instances[0].state, 'suspended');
    assert.deepEqual(sent(srv.ws).map((e) => e.type).slice(0, 2), ['session.update', 'playback.pause']);
    env.setHidden(false);
    await settle(env);
    assert.equal(engine.getState().status, 'Listening');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');
    engine.stop();
  });

  it('[LR-10] an input method\'s composition is read once it ends; text not sent yet may still change; text the server has but hasn\'t spoken goes out again as it now reads', async () => {
    const { engine } = await setup({ text: '', caret: 0 });
    const srv = await connect(engine);
    const compose = (value) => {
      env.editor.value = value;
      const event = new InputEvent('input', { bubbles: true, inputType: 'insertCompositionText' });
      event.isComposing = true;
      env.editor.dispatchEvent(event);
    };
    compose('k');                                  // the interim romaji…
    compose('か');                                 // …replaced by the composed character
    await settle(env, 50);
    assert.equal(engine.isActive(), true, 'replacing interim text is not an edit');
    assert.equal(sent(srv.ws, 'input_text.append').length, 0, 'nothing goes out while composing');
    env.editor.dispatchEvent(new Event('compositionend'));
    await settle(env, 50);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['か']);
    srv.accepted(1);
    await settle(env);

    // Typed and corrected inside the 40 ms batch (a quick backspace, an autocorrect): nothing was sent yet.
    env.input(env.editor, 'かTeh');
    await settle(env, 10);
    env.input(env.editor, 'かThe');
    await settle(env, 40);
    assert.equal(engine.isActive(), true);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['か', 'The']);
    srv.accepted(4);
    await settle(env);

    // The server has "かThe" but has spoken none of it. An edit never stops Live: the
    // server drops what it has not spoken, and gets the text again as it now reads.
    env.input(env.editor, 'かTha');
    await settle(env);
    assert.equal(engine.isActive(), true, 'an edit never stops Live');
    assert.equal(sent(srv.ws).at(-1).type, 'response.cancel');
    srv.cancelled(0);
    await settle(env);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['か', 'The', 'かTha']);
    assert.equal(engine.getState().status, 'Listening');
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-10] typing is batched for 40 ms; text typed before configuration joins the initial text; an edit before the start offset changes nothing Live reads', async () => {
    const { engine } = await setup({ text: 'One.' });
    engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created();
    env.input(env.editor, 'One. Two');            // before session.updated
    srv.updated();
    await settle(env);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), [' Two'], 'the caret was at the end: only the new text');
    srv.accepted(4);
    await settle(env);
    env.input(env.editor, 'One. Two th');
    await settle(env, 10);
    env.input(env.editor, 'One. Two three.');
    await settle(env, 10);
    assert.equal(sent(srv.ws, 'input_text.append').length, 1, 'nothing yet: the batch waits 40 ms');
    await settle(env, 40);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), [' Two', ' three.'], 'one batched append');
    srv.accepted(11);
    await settle(env);

    // "One." lies before the start offset (Live began after it): an edit there never
    // stops Live and takes nothing back, and what is typed next goes out where the text now ends.
    env.input(env.editor, 'Uno. Two three.');
    await settle(env, 50);
    assert.equal(engine.isActive(), true, 'an edit never stops Live');
    assert.equal(sent(srv.ws, 'response.cancel').length, 0, 'nothing Live reads changed');
    env.input(env.editor, 'Uno. Two three. Four.');
    await settle(env, 50);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), [' Two', ' three.', ' Four.']);
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    assert.equal(srv.ws.closeCode, null, 'the socket stays open');
    engine.stop();
  });

  // An edit (Backspace too) never stops Live (only
  // Stop, leaving the page or a take do). Text already spoken stays said and Live reads on. Text the server has
  // but hasn't spoken is cancelled (response.cancel: the protocol's only way to drop queued text) and goes out
  // again as it now reads, from the end of what was spoken, while the sentence playing plays on.
  it('[LR-10] Backspace in text the server has but hasn\'t spoken: Live goes on, the sentence playing plays on, and the corrected words are what the server gets', async () => {
    const { engine } = await setup({ text: '', caret: 0 });
    const srv = await connect(engine);
    env.input(env.editor, 'Hello world. ');
    await settle(env, 50);
    srv.accepted(13);
    srv.response('r1', 'Hello world.', 0, { start: 0, end: 12 });
    srv.delta('r1', 0, { samples: 24000 });       // 1 s: it plays while the next sentence is typed
    srv.done('r1');
    await settle(env);
    const [playing] = FakeAudioContext.instances[0].__sources;
    env.input(env.editor, 'Hello world. Thsi');
    await settle(env, 50);
    srv.accepted(4);
    await settle(env);
    env.input(env.editor, 'Hello world. Ths');    // Backspace: the server holds "Thsi" and has spoken none of it
    await settle(env);
    assert.equal(engine.isActive(), true, 'Live goes on');
    assert.equal(sent(srv.ws, 'response.cancel').length, 1, 'the server drops the text it has not spoken');
    assert.equal(playing.stopped, false, 'the sentence playing plays on');
    env.input(env.editor, 'Hello world. This');   // typed before the server confirms the cancel
    await settle(env, 50);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Hello world. ', 'Thsi'], 'nothing goes out until the cancel is confirmed');
    srv.cancelled(0);
    await settle(env);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Hello world. ', 'Thsi', ' This'], 'from the end of what was spoken, as it now reads');
    assert.equal(sent(srv.ws, 'input_text.commit').length, 0, 'typed text is not committed: the server waits for the sentence to end');
    srv.accepted(5, 1);
    srv.response('r2', 'This', 1, { start: 0, end: 5 });
    srv.delta('r2', 0, { generation: 1 });
    await settle(env);
    const sources = FakeAudioContext.instances[0].__sources;
    assert.equal(sources.length, 2);
    assert.ok(sources[1].startedAt >= playing.startedAt + 1 - 1e-6, 'the corrected words follow the sentence that kept playing');
    assert.equal(engine.getState().now.text, 'Hello world.', 'the Now line still shows the sentence playing');
    await settle(env, 1200);
    assert.equal(playing.stopped, false);
    assert.equal(engine.getState().now.text, 'This', 'then the corrected words');
    assert.equal(engine.isActive(), true);
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-10] an edit to text already spoken takes nothing back: Live reads on, and what is typed next goes out where the text now ends', async () => {
    const { engine } = await setup({ text: '', caret: 0 });
    const srv = await connect(engine);
    env.input(env.editor, 'Helo there. ');
    await settle(env, 50);
    srv.accepted(12);
    srv.response('r1', 'Helo there.', 0, { start: 0, end: 11 });
    srv.delta('r1', 0);
    srv.done('r1');
    await settle(env, 200);                       // said
    env.input(env.editor, 'Hello there. ');       // the typo fixed after it was spoken
    await settle(env, 50);
    env.input(env.editor, 'Hello there ');        // and a Backspace in it
    await settle(env, 50);
    assert.equal(engine.isActive(), true, 'an edit never stops Live');
    assert.equal(sent(srv.ws, 'response.cancel').length, 0, 'what was said stays said: nothing to take back');
    env.input(env.editor, 'Hello there Next.');
    await settle(env, 50);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Helo there. ', 'Next.'], 'only the new text');
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-10] what is spoken is found from the server\'s source offsets, which count code points; an edit never splits a surrogate pair', async () => {
    const { live, engine } = await setup({ text: '', caret: 0 });
    assert.deepEqual(live.changedRange('a😀b', 'a😁b'), { start: 1, oldEnd: 3, newEnd: 3 }, 'the whole emoji changed, not half of it');
    assert.deepEqual(live.changedRange('Hello', 'Hello there'), { start: 5, oldEnd: 5, newEnd: 11 });
    assert.deepEqual(live.changedRange('aa', 'a'), { start: 1, oldEnd: 2, newEnd: 1 });
    const srv = await connect(engine);
    env.input(env.editor, '😀 Hi. Next');
    await settle(env, 50);
    srv.accepted(10);
    srv.response('r1', '😀 Hi.', 0, { start: 0, end: 5 });   // 5 code points, 6 UTF-16 units
    srv.delta('r1', 0);
    srv.done('r1');
    await settle(env);
    env.input(env.editor, '😀 Hi! Next');           // the last character spoken
    await settle(env, 50);
    assert.equal(sent(srv.ws, 'response.cancel').length, 0, 'it was spoken: nothing to take back');
    env.input(env.editor, '😀 Hi! Text');           // the text not spoken yet
    await settle(env);
    assert.equal(sent(srv.ws, 'response.cancel').length, 1);
    srv.cancelled(0);
    await settle(env);
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['😀 Hi. Next', ' Text']);
    engine.stop();
  });

  it('[LR-10] an edit while the sentence playing is still arriving waits for the rest of its audio, then cancels; the corrected text follows it', async () => {
    const { engine } = await setup({ text: '', caret: 0 });
    const srv = await connect(engine);
    env.input(env.editor, 'One. Two three. ');
    await settle(env, 50);
    srv.accepted(16);
    srv.response('r1', 'One.', 0, { start: 0, end: 4 });
    srv.delta('r1', 0, { samples: 12000 });       // 0.5 s, and more of it is still to come
    await settle(env);
    env.input(env.editor, 'One. Two thre. ');     // Backspace in "three", which the server holds unspoken
    await settle(env);
    assert.equal(engine.isActive(), true);
    assert.equal(sent(srv.ws, 'response.cancel').length, 0, 'a cancel now would cut off the sentence playing');
    srv.delta('r1', 1, { samples: 12000 });
    srv.done('r1');
    await settle(env);
    assert.equal(sent(srv.ws, 'response.cancel').length, 1, 'once all of it is here, the rest is cancelled');
    srv.response('r2', 'Two three.', 0, { start: 5, end: 15 });   // already on its way: it goes with the cancel
    srv.delta('r2', 0);
    srv.cancelled(0);
    await settle(env);
    const sources = FakeAudioContext.instances[0].__sources;
    assert.equal(sources.length, 2, 'the cancelled sentence never plays');
    assert.ok(sources.every((x) => !x.stopped), 'all of the sentence playing plays');
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['One. Two three. ', ' Two thre. ']);
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    engine.stop();
  });

  it('[LR-10] an edit across spoken text reads the new text from the edit; the initial text keeps its commit when it goes out again', async () => {
    const { engine } = await setup({ text: 'Old one. Old two.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(17);
    await settle(env);
    assert.deepEqual(sent(srv.ws).filter((e) => e.type.startsWith('input_text')).map((e) => e.type), ['input_text.append', 'input_text.commit']);
    srv.response('r1', 'Old one.', 0, { start: 0, end: 8 });
    srv.delta('r1', 0);
    srv.done('r1');
    await settle(env);

    env.input(env.editor, 'Old one. New two.');   // the sentence not spoken yet
    await settle(env);
    srv.cancelled(0);
    await settle(env);
    srv.accepted(9, 1);
    await settle(env);
    const flowOf = () => sent(srv.ws).filter((e) => e.type.startsWith('input_text') || e.type === 'response.cancel')
      .map((e) => (e.type === 'input_text.append' ? e.text : e.type));
    // The text sent again starts mid-line, after "Old one.".
    assert.deepEqual(flowOf(), ['Old one. Old two.', 'input_text.commit', 'response.cancel', 'input_text.context', ' New two.', 'input_text.commit'], 'what was committed is committed again');

    env.input(env.editor, 'Fresh start.');        // select all and paste: it begins inside what was spoken
    await settle(env);
    srv.cancelled(1);
    await settle(env);
    srv.accepted(12, 2);
    await settle(env);
    assert.deepEqual(flowOf().slice(6), ['response.cancel', 'Fresh start.', 'input_text.commit'], 'the pasted text is read, from where it begins (a line start: no context)');
    assert.equal(engine.isActive(), true);
    assert.ok(!toasts().some((t) => t.startsWith('Live stopped')), toasts().join(' | '));
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  // After response.cancel the server reads the next text with
  // a fresh segmenter, as a line start, so a resend from mid-line could lose a leading "12.", "3)", ">" or "-", or
  // open a code fence. input_text.context tells it what comes before: after response.cancelled and
  // before the first append, the text before the resume point from the last line break (nothing when a line starts
  // there). It is not spoken and counts in no offset.
  it('[LR-10] the text sent again after an edit\'s cancel goes on from mid-line: input_text.context first gives the server the text before it on its line', async () => {
    const { live, engine } = await setup({ text: '', caret: 0 });
    const srv = await connect(engine);
    env.input(env.editor, 'Intro.\nPrices rose. Then 12 more.');
    await settle(env, 50);
    srv.accepted(32);
    srv.response('r1', 'Intro.', 0, { start: 0, end: 6 });
    srv.delta('r1', 0);                           // 0.1 s
    srv.done('r1');
    srv.response('r2', 'Prices rose.', 0, { start: 7, end: 19 });
    srv.delta('r2', 0, { samples: 24000 });       // 1 s: it plays while the number is corrected
    srv.done('r2');
    await settle(env, 200);
    const flow = () => sent(srv.ws).filter((e) => e.type.startsWith('input_text') || e.type === 'response.cancel');
    const mark = flow().length;
    env.input(env.editor, 'Intro.\nPrices rose. Then 13 more.');   // the number the server holds unspoken
    await settle(env);
    srv.cancelled(0);
    await settle(env);
    assert.deepEqual(flow().slice(mark), [
      { type: 'response.cancel' },
      { type: 'input_text.context', text: 'Prices rose.' },
      { type: 'input_text.append', text: ' Then 13 more.' },
    ], 'the line so far, then the text again from the end of what was spoken: "13." continues the sentence');
    srv.accepted(14, 1);
    await settle(env);

    // An edit that begins just after the line break: the text sent again starts a line, so no context goes out.
    const second = flow().length;
    env.input(env.editor, 'Intro.\nNew line.');
    await settle(env);
    srv.cancelled(1);
    await settle(env);
    assert.deepEqual(flow().slice(second), [
      { type: 'response.cancel' },
      { type: 'input_text.append', text: 'New line.' },
    ]);
    assert.equal(engine.isActive(), true);
    assert.deepEqual(env.errors, []);
    engine.stop();

    // PURE: what the context holds.
    assert.equal(live.lineContext('Intro.\nPrices rose. Then 12 more.', 19), 'Prices rose.', 'from the last line break');
    assert.equal(live.lineContext('Intro.\nNext', 7), '', 'just after a line break a line starts: nothing to say');
    assert.equal(live.lineContext('Intro.', 0), '');
    assert.equal(live.lineContext('\nNext', 0), '');
    assert.equal(live.lineContext('One line', 99), 'One line', 'an offset past the end reads to the end');
    // At most `max` code points (within one append's limit), the end of the line, never half a surrogate pair.
    assert.equal(live.lineContext('ab😀cd', 5, 2), '😀c');
    assert.equal(live.lineContext('ab😀cd', 4, 1), '😀');
    assert.equal(live.lineContext('ab😀cd', 5, 1), 'c');
  });

  // Start reads from the cursor, which can sit in the middle of a
  // line, and the server's segmenter is fresh after session.updated, so it read the initial text as a line start: a
  // leading "12.", "3)", ">" or "-" there was dropped as a marker. As resend() does after a cancel, Start sends
  // input_text.context (lineContext: the text before the start offset on its line) after session.updated and before
  // the first append; nothing when a line starts there, or for Read all.
  const opening = (ws) => sent(ws).filter((e) => e.type === 'session.update' || e.type.startsWith('input_text'))
    .map((e) => (e.type === 'input_text.append' || e.type === 'input_text.context' ? { type: e.type, text: e.text } : { type: e.type }));

  it('[LR-03] Start from the cursor in the middle of a line sends input_text.context first, the text before it on its line; from a line start or for Read all, nothing', async () => {
    const text = 'Intro.\nSales rose to 12. Then more.';
    const { engine } = await setup({ text, caret: text.indexOf('12.') });
    const srv = await connect(engine);
    assert.deepEqual(opening(srv.ws), [
      { type: 'session.update' },
      { type: 'input_text.context', text: 'Sales rose to ' },
      { type: 'input_text.append', text: '12. Then more.' },
    ], 'after session.updated and before the first append: "12." continues the line');
    srv.accepted(14);
    await settle(env);
    assert.equal(sent(srv.ws).at(-1).type, 'input_text.commit', 'the initial text is still committed once accepted');
    engine.stop();
    await settle(env);

    env.editor.selectionStart = text.indexOf('Sales');   // just after the line break: a line starts there
    const srv2 = await connect(engine);
    assert.deepEqual(opening(srv2.ws), [{ type: 'session.update' }, { type: 'input_text.append', text: 'Sales rose to 12. Then more.' }]);
    engine.stop();
    await settle(env);

    env.editor.selectionStart = text.indexOf('12.');
    const srv3 = await connect(engine, { readAll: true });
    assert.deepEqual(opening(srv3.ws), [{ type: 'session.update' }, { type: 'input_text.append', text }], 'Read all starts at the beginning');
    assert.equal(engine.isActive(), true);
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-03] the context of a Start mid-line is the line as it reads once the session is configured, within max_buffer_chars, and also goes out when nothing follows the cursor yet', async () => {
    // An edit earlier in the line while the session connects: the start offset follows the text (LR-10).
    const { engine } = await setup({ text: 'Prices rose. Then 12 more.', caret: 13 });
    assert.equal(env.withGesture(() => engine.start()), true);
    await settle(env);
    const srv = server(env.sockets.at(-1));
    srv.open();
    srv.created();
    await settle(env);
    env.input(env.editor, 'Prices fell. Then 12 more.');
    await settle(env);
    assert.deepEqual(opening(srv.ws), [{ type: 'session.update' }], 'nothing before session.updated (it would be dropped)');
    srv.updated();
    await settle(env);
    assert.deepEqual(opening(srv.ws).slice(1), [
      { type: 'input_text.context', text: 'Prices fell. ' },
      { type: 'input_text.append', text: 'Then 12 more.' },
    ]);
    engine.stop();
    await settle(env);

    // The cursor at the end of a line: the text typed next continues it.
    env.editor.value = 'Intro.\nTotal:';
    env.editor.selectionStart = env.editor.value.length;
    const srv2 = await connect(engine);
    assert.deepEqual(opening(srv2.ws), [{ type: 'session.update' }, { type: 'input_text.context', text: 'Total:' }]);
    env.input(env.editor, 'Intro.\nTotal: 12. Apples.');
    await settle(env, 50);
    assert.deepEqual(opening(srv2.ws).slice(2), [{ type: 'input_text.append', text: ' 12. Apples.' }], '"12." reads as the line\'s continuation');
    engine.stop();
    await settle(env);

    // A very long line keeps its last max_buffer_chars code points (the session's limit, LR-08).
    env.editor.value = 'Sales rose to 12.';
    env.editor.selectionStart = 14;
    assert.equal(env.withGesture(() => engine.start()), true);
    await settle(env);
    const srv3 = server(env.sockets.at(-1));
    srv3.open();
    srv3.created({ max_buffer_chars: 5 });
    srv3.updated();
    await settle(env);
    assert.deepEqual(sent(srv3.ws, 'input_text.context'), [{ type: 'input_text.context', text: 'e to ' }]);
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-11] audio deltas play back to back on Web Audio, and playback.ack goes out per chunk only after it played, in order', async () => {
    const { engine } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(6);
    srv.response('r1', 'Hello.');
    srv.delta('r1', 1, { samples: 2400 });    // 0.1 s at 24 kHz
    srv.delta('r1', 2, { samples: 2400 });
    await settle(env);
    const ctx = FakeAudioContext.instances[0];
    const [a, b] = ctx.__sources;
    assert.ok(a && b, 'two buffer sources');
    assert.ok(a.startedAt >= 0.025 - 1e-9, 'the first starts at least 25 ms ahead');
    assert.ok(Math.abs(b.startedAt - (a.startedAt + 0.1)) < 1e-6, 'the second starts when the first ends');
    assert.equal(a.buffer.sampleRate, 24000);
    assert.ok(Math.abs(a.buffer.getChannelData(0)[0] - 1000 / 32768) < 1e-9, 'PCM16 decoded to floats');
    assert.equal(sent(srv.ws, 'playback.ack').length, 0, 'no ack before playback');
    await settle(env, 126);
    assert.deepEqual(sent(srv.ws, 'playback.ack'), [{ type: 'playback.ack', response_id: 'r1', sequence: 1 }]);
    await settle(env, 100);
    assert.deepEqual(sent(srv.ws, 'playback.ack').map((e) => e.sequence), [1, 2], 'FIFO');
    engine.stop();
  });

  it('[LR-11] [LR-13] a chunk of a cancelled generation is never acked, not even when it ends after the cancel', async () => {
    const { engine } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(6);
    srv.response('r1', 'Hello.');
    srv.delta('r1', 1, { samples: 2400 });    // 0.1 s
    srv.delta('r1', 2, { samples: 2400 });
    await settle(env);
    const [a, b] = FakeAudioContext.instances[0].__sources;
    assert.ok(a && b, 'two chunks scheduled');
    srv.cancelled(0);
    await settle(env);
    // The browser still fires 'ended' for a source stopped by the cancel (or one that was ending anyway).
    for (const source of [a, b]) if (typeof source.onended === 'function') source.onended();
    await settle(env, 400);
    assert.deepEqual(sent(srv.ws, 'playback.ack'), [], 'no ack for generation 0 after response.cancelled');
    engine.stop();
  });

  it('[LR-11] the ack rule of the old UI: a chunk whose generation moved on while it played (session.status) is not acked; the next generation is acked', async () => {
    const { engine } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(6);
    srv.response('r1', 'Hello.');
    srv.delta('r1', 1, { samples: 2400, generation: 0 });   // plays 0.1 s
    await settle(env);
    // The server's generation moves on (it cancelled; its status arrives before response.cancelled).
    srv.status(0, 1);
    srv.response('r2', 'Again.', 1);
    srv.delta('r2', 1, { samples: 2400, generation: 1 });
    await settle(env);
    assert.equal(FakeAudioContext.instances[0].__sources.length, 2, 'the generation 1 chunk is queued behind the old one');
    await settle(env, 126);
    assert.deepEqual(sent(srv.ws, 'playback.ack'), [], 'the generation 0 chunk ended: no ack for a stale generation');
    await settle(env, 100);
    assert.deepEqual(sent(srv.ws, 'playback.ack'), [{ type: 'playback.ack', response_id: 'r2', sequence: 1 }], 'generation 1 is acked after it played');
    engine.stop();
  });

  it('[LR-11] a chunk the browser cannot schedule still counts as played: the acks behind it flow, and three in a row stop Live', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    const createBuffer = ctx.createBuffer.bind(ctx);
    let failing = 1;
    ctx.createBuffer = (...args) => {
      if (failing > 0) {
        failing -= 1;
        throw new DOMException('The sample rate is not supported', 'NotSupportedError');
      }
      return createBuffer(...args);
    };
    const logged = [];
    const consoleError = console.error;
    console.error = (error) => logged.push(error);
    try {
      srv.response('r1', 'Hi.');
      srv.delta('r1', 1);                        // can't be scheduled
      srv.delta('r1', 2);
      srv.delta('r1', 3);
      srv.done('r1');
      await settle(env, 400);
      assert.deepEqual(sent(srv.ws, 'playback.ack').map((e) => e.sequence), [1, 2, 3], 'in order, the failed chunk first');
      assert.ok(toasts().includes("This browser couldn't play part of the Live audio."), toasts().join(' | '));
      assert.ok(!toasts().includes('Live received invalid data from the server.'), 'a local problem, not the server\'s');
      assert.equal(engine.isActive(), true);
      await settle(env, 800);
      assert.equal(engine.getState().status, 'Listening');

      failing = 3;                                 // it keeps failing: stop rather than read on in silence
      srv.response('r2', 'Again.');
      srv.delta('r2', 1);
      srv.delta('r2', 2);
      srv.delta('r2', 3);
      await settle(env);
      const state = engine.getState();
      assert.equal(state.active, false);
      assert.equal(state.status, 'Error');
      assert.equal(state.error.message, "This browser can't play Live audio.");
      assert.match(state.error.detail, /sample rate/);
      assert.equal(ctx.state, 'closed');
      assert.equal(logged.length, 4, 'each failure is logged for developers');
    } finally {
      console.error = consoleError;
    }
  });

  it('[LR-13] response.cancelled moves to the next generation, drops the queue, the chunk in flight and the batch, and ignores stale audio', async () => {
    const { engine } = await setup({ text: 'x'.repeat(3000), caret: 0 });
    const srv = await connect(engine);
    assert.equal(sent(srv.ws, 'input_text.append').length, 1, 'one chunk in flight; the rest queued');
    env.input(env.editor, `${'x'.repeat(3000)} more`);
    srv.cancelled(0);
    await settle(env, 50);
    assert.equal(sent(srv.ws, 'input_text.append').length, 1, 'the batch was dropped');
    srv.accepted(0, 1);
    await settle(env);
    assert.equal(sent(srv.ws, 'input_text.append').length, 1, 'the queue was dropped');
    srv.response('old', 'stale', 0);
    srv.delta('old', 1, { generation: 0 });
    await settle(env);
    assert.equal(FakeAudioContext.instances[0].__sources.length, 0, 'generation 0 audio is ignored after the cancel');
    srv.response('new', 'fresh', 1);
    srv.delta('new', 1, { generation: 1 });
    await settle(env);
    assert.equal(FakeAudioContext.instances[0].__sources.length, 1, 'the next generation plays');
    engine.stop();
  });
});

describe('Live: errors and closes', () => {
  it('[LR-14] a failed response toasts and shows the reason, and reading continues', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    srv.error('tts_error', 'Live TTS synthesis failed', { response_id: 'r1' });
    await settle(env);
    const state = engine.getState();
    assert.equal(state.active, true);
    assert.equal(state.status, 'Listening');
    assert.deepEqual(state.now, { kind: 'reason', text: 'Live TTS synthesis failed' });
    assert.ok(toasts().includes('Live TTS synthesis failed'));
    srv.response('r2', 'Next one.');
    await settle(env);
    assert.deepEqual(engine.getState().now, { kind: 'spoken', text: 'Next one.' }, 'the next response replaces the reason');
    engine.stop();
  });

  it('[LR-14] [BUG-20] a failed response in the server\'s real order (created, error, done) shows the reason at once, never "Speaking"', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(3);
    srv.response('r1', 'Hi.');                     // the server announces every response before synthesising it
    await settle(env);
    assert.equal(engine.getState().status, 'Speaking');
    srv.error('tts_error', 'Live TTS synthesis failed', { response_id: 'r1' });
    await settle(env);
    assert.equal(engine.getState().status, 'Listening', 'at once, not after the quiet delay');
    assert.deepEqual(engine.getState().now, { kind: 'reason', text: 'Live TTS synthesis failed' }, 'the text that will not be spoken is gone');
    srv.done('r1', 0, 'failed');
    await settle(env, 800);
    assert.equal(engine.getState().status, 'Listening');
    assert.deepEqual(engine.getState().now, { kind: 'reason', text: 'Live TTS synthesis failed' }, 'the reason stays until the next response');

    // One response fails while another one's audio plays: the Now line keeps the text being heard.
    srv.response('r2', 'Two.');
    srv.delta('r2', 1, { samples: 2400 });
    srv.response('r3', 'Three.');
    srv.error('tts_error', 'Live TTS synthesis failed', { response_id: 'r3' });
    await settle(env);
    assert.deepEqual(engine.getState().now, { kind: 'spoken', text: 'Two.' });
    assert.equal(engine.getState().status, 'Speaking');
    engine.stop();
  });

  it('[LR-14] an error before session.updated (rejected settings) closes the socket and releases with the reason, never staying Connecting', async () => {
    const { engine } = await setup();
    engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created();
    srv.error('invalid_event', "Voice 'gone' is not available");
    await settle(env);
    assert.equal(srv.ws.closeCode, 1000);
    assert.equal(srv.ws.closeReason, 'Session settings rejected');
    const state = engine.getState();
    assert.equal(state.active, false);
    assert.equal(state.status, 'Error');
    assert.equal(state.error.message, "Voice 'gone' is not available");
    assert.equal(FakeAudioContext.instances[0].state, 'closed');
  });

  it('[LR-14] idle_timeout, pause_timeout and playback_stalled keep their status and reason for the server close', async () => {
    for (const [code, copy] of [
      ['idle_timeout', 'Live stopped after a long time without new text. Start again to continue.'],
      ['pause_timeout', 'Live stopped after being paused for too long. Start again to continue.'],
      ['playback_stalled', 'Live stopped because this browser stopped playing the audio. Start again to continue.'],
    ]) {
      const { engine } = await setup({ text: 'Hi.', caret: 0 });
      const srv = await connect(engine);
      srv.error(code, 'Server words');
      await settle(env);
      assert.equal(srv.ws.readyState, 1, 'the client leaves the close to the server');
      assert.equal(engine.getState().status, 'Error');
      srv.close(4008, 'Server words');
      await settle(env);
      const state = engine.getState();
      assert.equal(state.active, false);
      assert.equal(state.status, 'Error');
      assert.equal(state.error.message, copy);
      assert.match(state.error.detail, /Server words/);
      env.restore();
    }
  });

  it('[LR-18] close codes 4004, 4001, 1008 and 1013 get plain copy; a drop reads Disconnected', async () => {
    const cases = [
      [4004, 'Live TTS is disabled', 'Error', 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).', ''],
      [4001, 'Invalid or missing API key', 'Error', 'This server needs an API key. Enter it in Settings › Connection.', 'open-settings'],
      [1008, 'Origin not allowed', 'Error', "This page's address isn't allowed to connect. Open Open Speech from its own address, or set OS_WS_ALLOWED_ORIGINS.", ''],
      [1013, 'Live Reader busy', 'Busy', 'Another Live session is running. Stop it first or try again shortly.', 'retry'],
      [1006, '', 'Disconnected', 'The connection to the server dropped.', 'retry'],
    ];
    for (const [code, reason, status, message, action] of cases) {
      const { engine } = await setup();
      engine.start();
      await settle(env);
      const srv = server(env.sockets[0]);
      srv.open();
      srv.close(code, reason);
      await settle(env);
      const state = engine.getState();
      assert.equal(state.status, status, `code ${code}`);
      assert.equal(state.error.title, 'Live stopped');
      assert.equal(state.error.message, message, `code ${code}`);
      assert.equal(state.error.action, action, `code ${code}`);
      if (reason) assert.match(state.error.detail, new RegExp(reason));
      assert.equal(FakeAudioContext.instances[0].state, 'closed');
      env.restore();
    }
    // With a key set, a 4001 means the ticket or key was refused.
    const { engine } = await setup({ key: 'k' });
    engine.start();
    await settle(env);
    server(env.sockets[0]).close(4001, 'Invalid or missing API key');
    await settle(env);
    assert.equal(engine.getState().error.message, 'The server refused the API key or ticket. Check Settings › Connection.');
  });

  it(`[LR-18] a socket the server refused before it opened (an HTTP 403 reaches the page as 1006 without open) "didn't open": an error with Start again, never "dropped"`, async () => {
    const { engine } = await setup();
    engine.start();
    await settle(env);
    server(env.sockets[0]).close(1006, '');
    await settle(env);
    let state = engine.getState();
    assert.equal(state.active, false);
    assert.equal(state.phase, 'error', 'refused, not disconnected');
    assert.equal(state.status, 'Error');
    assert.equal(state.error.title, 'Live stopped');
    assert.equal(state.error.message, "The Live speech stream didn't open. Check that Open Speech is running and that this page comes from its own address.");
    assert.equal(state.error.action, 'retry');
    assert.equal(FakeAudioContext.instances[0].state, 'closed');
    env.restore();

    // The same code once the socket had opened is a drop.
    const second = await setup();
    second.engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.close(1006, '');
    await settle(env);
    state = second.engine.getState();
    assert.equal(state.phase, 'disconnected');
    assert.equal(state.error.message, 'The connection to the server dropped.');
  });

  it(`[LR-06] a refused ticket request shows the server's message, and Details read the message and its code, never the JSON envelope`, async () => {
    const { engine } = await setup({
      key: 'k',
      routes: { 'POST /api/auth/ws-ticket': () => json(503, { error: { message: 'Live is warming up', code: 'live_unavailable' } }) },
    });
    engine.start();
    await settle(env);
    const state = engine.getState();
    assert.equal(env.sockets.length, 0, 'no socket without a ticket');
    assert.equal(state.phase, 'error');
    assert.equal(state.error.title, 'Live stopped');
    assert.equal(state.error.message, 'Live is warming up');
    assert.equal(state.error.detail, 'Live is warming up (live_unavailable)');
    assert.equal(state.error.action, 'retry');
  });

  it('[LR-18] an admission session_busy keeps Busy and the server reason through the 1013 close', async () => {
    const { engine } = await setup();
    engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.error('session_busy', 'Live Reader is already in use', { reason: 'Another Live session is active' });
    await settle(env);
    assert.equal(engine.getState().status, 'Busy');
    assert.equal(srv.ws.readyState, 1, 'the server closes it');
    srv.close(1013, 'Live Reader busy');
    await settle(env);
    const state = engine.getState();
    assert.equal(state.status, 'Busy');
    assert.equal(state.error.message, 'Another Live session is running. Stop it first or try again shortly.');
    assert.match(state.error.detail, /already in use/);
    assert.match(state.error.detail, /Another Live session is active/);
  });

  it('[LR-18] a Start refused right after Live\'s own stop (the server still shutting it down) is retried with a fresh ticket; a server that stays busy says so', async () => {
    const { engine, live } = await setup({ key: 'k', text: 'Hi.', caret: 0 });
    await connect(engine);                         // ticket t1
    engine.stop('Text changed');
    await settle(env);
    const refuse = (ws, reason) => {
      ws.open();
      ws.serverSend({ type: 'error', error: { code: 'session_busy', message: 'Live Reader is already in use', reason } });
      ws.serverClose(1013, 'Live Reader busy');
    };

    // Start again at once: the server has not let go of the last session yet.
    env.withGesture(() => engine.start());
    await settle(env);
    refuse(env.sockets.at(-1), 'active');          // ticket t2
    await settle(env);
    assert.equal(engine.getState().status, 'Connecting…', 'no Busy banner');
    assert.equal(engine.getState().error, null);
    assert.equal(env.sockets.length, 2, 'it waits a moment first');
    await settle(env, live.ADMISSION_RETRY_MS[0]);
    assert.equal(env.sockets.length, 3);
    const ws = env.sockets.at(-1);
    assert.equal(new URL(ws.url).searchParams.get('ticket'), 't3', 'each attempt gets a fresh ticket');
    const srv = server(ws);
    srv.open();
    srv.created();
    srv.updated();
    await settle(env);
    assert.equal(engine.getState().status, 'Listening');
    assert.deepEqual(sent(ws, 'input_text.append').map((e) => e.text), ['Hi.']);
    engine.stop();
    await settle(env);

    // The server keeps draining: after the retries the banner says the last session is still finishing.
    env.withGesture(() => engine.start());
    for (let i = 0; i <= live.ADMISSION_RETRY_MS.length; i += 1) {
      await settle(env, i ? live.ADMISSION_RETRY_MS[i - 1] : 0);
      refuse(env.sockets.at(-1), 'draining');
    }
    await settle(env);
    let state = engine.getState();
    assert.equal(state.status, 'Busy');
    assert.equal(state.error.message, 'The last Live session is still finishing. Start again in a moment.');
    assert.match(state.error.detail, /already in use/);
    assert.equal(env.sockets.length, 4 + live.ADMISSION_RETRY_MS.length);

    // Long after Live's own session ended, an active one elsewhere is reported at once.
    await settle(env, live.OWN_CLOSE_WINDOW_MS);
    const before = env.sockets.length;
    env.withGesture(() => engine.start());
    await settle(env);
    refuse(env.sockets.at(-1), 'active');
    await settle(env, 5000);
    state = engine.getState();
    assert.equal(state.status, 'Busy');
    assert.equal(state.error.message, 'Another Live session is running. Stop it first or try again shortly.');
    assert.equal(env.sockets.length, before + 1, 'no retry');
  });

  it('[LR-18] Stop stops the scheduled audio, cancels, closes 1000 "Stopped by user" and releases', async () => {
    const { engine, states } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    srv.response('r1', 'Hello.');
    srv.delta('r1', 1);
    srv.delta('r1', 2);
    await settle(env);
    const ctx = FakeAudioContext.instances[0];
    assert.equal(env.app.get().player.kind, 'live');
    engine.stop();
    await settle(env);
    assert.ok(ctx.__sources.every((s) => s.stopped), 'scheduled sources stopped');
    assert.equal(sent(srv.ws).at(-1).type, 'response.cancel');
    assert.equal(srv.ws.closeCode, 1000);
    assert.equal(srv.ws.closeReason, 'Stopped by user');
    assert.equal(sent(srv.ws, 'playback.ack').length, 0, 'stopped chunks are never acked');
    assert.equal(ctx.state, 'closed');
    const state = engine.getState();
    assert.equal(state.active, false);
    assert.equal(state.status, 'Stopped');
    assert.equal(state.now.text, 'Start, then type or paste. Live reads from the cursor as you write.');
    assert.equal(env.app.get().player, null, 'the audio session is released');
    assert.equal(states.at(-1).status, 'Stopped');
  });

  it('[LR-22] a malformed server frame toasts "Live received invalid data" and the session carries on', async () => {
    const { engine } = await setup();
    const srv = await connect(engine);
    srv.ws.serverSend('{not json');
    srv.ws.serverSend('{"no":"type"}');
    srv.ws.serverSend(new ArrayBuffer(4));
    await settle(env);
    assert.ok(toasts().includes('Live received invalid data from the server.'));
    assert.equal(engine.isActive(), true);
    assert.equal(engine.getState().status, 'Listening');
    engine.stop();
  });

  it('[LR-23] the client never sends session.update once text is accepted (the server would refuse it with session_busy)', async () => {
    const { engine } = await setup({ text: 'Words here.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(11);
    srv.created();                 // a stray repeat after text was accepted
    srv.cancelled(0);
    srv.created();
    await settle(env);
    const types = sent(srv.ws).map((e) => e.type);
    assert.equal(types.filter((t) => t === 'session.update').length, 1, 'exactly one session.update');
    assert.ok(types.indexOf('session.update') < types.indexOf('input_text.append'), 'and it went before any text');
    engine.stop();
  });

  it('[BUG-20] a mid-session session_busy is not fatal, and the Now line shows the text whose audio is playing', async () => {
    const { engine } = await setup({ text: 'One. Two.', caret: 0 });
    const srv = await connect(engine);
    srv.accepted(9);
    srv.error('session_busy', 'Session voice/model settings can only change before text is accepted or after cancel');
    await settle(env);
    assert.equal(engine.isActive(), true, 'the socket stays open and so does the session');
    assert.equal(engine.getState().status, 'Listening');
    assert.equal(srv.ws.readyState, 1);

    srv.response('r1', 'One.');
    srv.delta('r1', 1, { samples: 2400 });
    srv.response('r2', 'Two.');                 // created while r1 still plays
    srv.delta('r2', 1, { samples: 2400 });
    await settle(env);
    assert.deepEqual(engine.getState().now, { kind: 'spoken', text: 'One.' }, 'r1 plays first');
    assert.equal(engine.getState().status, 'Speaking');
    await settle(env, 126);
    assert.deepEqual(engine.getState().now, { kind: 'spoken', text: 'Two.' }, 'the line moves when r2 starts playing');
    srv.done('r1');
    srv.done('r2');
    await settle(env, 100);
    assert.equal(engine.getState().status, 'Speaking', 'no flap to Listening right away');
    await settle(env, 700);
    assert.equal(engine.getState().status, 'Listening');
    assert.deepEqual(engine.getState().now, { kind: 'hint', text: 'Listening for text…' });
    engine.stop();
  });
});

describe('Live: pause, background, keepalive, unload', () => {
  it('[LR-15] Pause and Resume suspend and resume the AudioContext and tell the server', async () => {
    const { engine } = await setup();
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    env.withGesture(() => engine.togglePause());
    assert.equal(ctx.state, 'suspended');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.pause');
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(engine.getState().paused, true);
    env.withGesture(() => engine.togglePause());
    assert.equal(ctx.state, 'running');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');
    assert.equal(engine.getState().status, 'Listening');
    engine.stop();
  });

  it('[LR-15] Resume waits while the microphone records: the toggle, the mini-player and the lock screen say why, and nothing resumes by itself', async () => {
    const { engine } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    const audio = await load('core/audio.js');
    env.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    audio.pauseAll();                              // what mic.open() does first
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(ctx.state, 'suspended');
    const before = sent(srv.ws).length;
    env.withGesture(() => engine.togglePause());   // the toggle's Resume
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(engine.getState().refusal, 'Stop recording to listen.');
    assert.equal(ctx.state, 'suspended');
    env.withGesture(() => audio.current().controls.play());   // the mini-player or the lock screen
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(env.app.get().player.playing, false);
    assert.equal(env.app.get().player.statusText, 'Stop recording to listen', 'the words a clip uses');
    assert.equal(sent(srv.ws).length, before, 'no playback.resume');

    env.app.set({ recording: null });
    await settle(env, 1000);
    assert.equal(engine.getState().status, 'Paused', 'never resumes by itself');
    assert.equal(engine.getState().refusal, '');
    assert.equal(env.app.get().player.statusText, 'Paused');
    env.withGesture(() => engine.togglePause());
    assert.equal(engine.getState().status, 'Listening');
    assert.equal(ctx.state, 'running');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');
    engine.stop();
  });

  it('[LR-15] a context the system suspends (an iOS interruption) or a resume the browser holds back counts as the listener\'s pause; Resume brings it back', async () => {
    const { engine, live } = await setup({ text: 'Hello.', caret: 0 });
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    await settle(env, live.CTX_RESUME_WAIT_MS + live.CTX_SETTLE_MS);
    assert.equal(engine.getState().status, 'Listening', 'its own resume and suspend are not system pauses');

    ctx.state = 'interrupted';                     // a phone call: the system stops the audio, not Live
    ctx.dispatchEvent(new Event('statechange'));
    await settle(env, live.CTX_SETTLE_MS);
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(engine.getState().paused, true);
    assert.equal(sent(srv.ws).at(-1).type, 'playback.pause', 'the server holds the audio meanwhile');
    await settle(env, 5000);
    assert.equal(engine.getState().status, 'Paused', 'never resumed by itself');
    env.withGesture(() => engine.togglePause());   // the listener's Resume tap (a gesture)
    assert.equal(ctx.state, 'running');
    assert.equal(engine.getState().status, 'Listening');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');

    // Back from the background, the browser holds the resume back (no gesture): a pause, not silent "Listening".
    env.setHidden(true);
    await settle(env);
    assert.equal(engine.getState().status, 'Paused in background');
    ctx.resume = () => new Promise(() => {});      // never settles; the context stays suspended
    env.setHidden(false);
    await settle(env);
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');
    await settle(env, live.CTX_RESUME_WAIT_MS + live.CTX_SETTLE_MS);
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.pause');
    assert.deepEqual(env.errors, []);
    engine.stop();
  });

  it('[LR-15] a slow output device never turns Start or the toggle\'s Resume into a pause; a resume that never comes still counts as one', async () => {
    const { engine, live } = await setup({ text: 'Hello.', caret: 0 });
    const proto = FakeAudioContext.prototype;
    const realResume = proto.resume;
    let wake = 3000;                               // Bluetooth headphones waking up
    proto.resume = function slowResume() {
      // The context stays suspended until the device is up; then it runs and the promise resolves.
      if (!Number.isFinite(wake)) return new Promise(() => {});
      return new Promise((resolve) => setTimeout(() => realResume.call(this).then(resolve), wake));
    };
    FakeAudioContext.defaults.state = 'suspended';
    try {
      const srv = await connect(engine);            // Start: resume() inside the click
      const ctx = FakeAudioContext.instances[0];
      assert.equal(ctx.state, 'suspended');
      await settle(env, live.CTX_RESUME_WAIT_MS + live.CTX_SETTLE_MS);
      assert.equal(engine.getState().status, 'Listening', 'a device still waking up is not a pause');
      assert.deepEqual(sent(srv.ws, 'playback.pause'), []);
      await settle(env, wake + live.CTX_SETTLE_MS);
      assert.equal(ctx.state, 'running');
      assert.equal(engine.getState().status, 'Listening');
      assert.deepEqual(sent(srv.ws, 'playback.pause'), []);

      env.withGesture(() => engine.togglePause());  // Pause, then the toggle's Resume (also inside a click)
      env.withGesture(() => engine.togglePause());
      await settle(env, live.CTX_RESUME_WAIT_MS + live.CTX_SETTLE_MS);
      assert.equal(engine.getState().status, 'Listening');
      await settle(env, wake + live.CTX_SETTLE_MS);
      assert.equal(ctx.state, 'running');
      assert.equal(engine.getState().status, 'Listening');
      assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');

      // A resume that never comes (the context never runs): Live says Paused rather than reading in silence.
      env.withGesture(() => engine.togglePause());
      wake = Infinity;
      env.withGesture(() => engine.togglePause());
      await settle(env, live.CTX_RESUME_WAIT_MS + live.CTX_SETTLE_MS);
      assert.equal(engine.getState().status, 'Listening');
      await settle(env, live.CTX_GESTURE_RESUME_WAIT_MS);
      assert.equal(engine.getState().status, 'Paused');
      assert.equal(sent(srv.ws).at(-1).type, 'playback.pause');
      assert.deepEqual(env.errors, []);
      engine.stop();
    } finally {
      proto.resume = realResume;
    }
  });

  it('[LR-16] a hidden page pauses ("Paused in background") and showing it resumes, unless the listener paused', async () => {
    const { engine } = await setup();
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    env.setHidden(true);
    await settle(env);
    assert.equal(engine.getState().status, 'Paused in background');
    assert.equal(ctx.state, 'suspended');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.pause');
    env.setHidden(false);
    await settle(env);
    assert.equal(engine.getState().status, 'Listening');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');

    engine.togglePause();
    env.setHidden(true);
    env.setHidden(false);
    await settle(env);
    assert.equal(engine.getState().status, 'Paused', 'a user pause survives the page coming back');
    assert.equal(ctx.state, 'suspended');
    assert.equal(sent(srv.ws, 'playback.pause').length, 2, 'no duplicate pause for the hidden page');
    engine.stop();
  });

  it('[LR-17] session.keepalive goes out every 30 s while connected, and stops with the session', async () => {
    const { engine } = await setup();
    const srv = await connect(engine);
    await settle(env, 30000);
    assert.equal(sent(srv.ws, 'session.keepalive').length, 1);
    await settle(env, 30000);
    assert.equal(sent(srv.ws, 'session.keepalive').length, 2);
    engine.stop();
    await settle(env, 60000);
    assert.equal(sent(srv.ws, 'session.keepalive').length, 2);
  });

  it('[LR-18] leaving the page (beforeunload) closes the socket (GS-20)', async () => {
    const { engine } = await setup();
    const srv = await connect(engine);
    window.dispatchEvent(new Event('beforeunload'));
    await settle(env);
    assert.equal(srv.ws.closeCode, 1000);
    assert.equal(srv.ws.closeReason, 'Page closed');
    engine.stop();
  });

  /** A page lifecycle event as the browser sends it ({persisted} is always a boolean there). */
  function pageEvent(type, persisted) {
    const event = new Event(type);
    Object.defineProperty(event, 'persisted', { value: persisted });
    return event;
  }

  it('[LR-18] [GS-20] a pagehide that discards the page closes the socket and lets the session go (iOS Safari never fires beforeunload)', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    const ctx = FakeAudioContext.instances[0];
    assert.equal(env.app.get().player?.id, 'live');
    window.dispatchEvent(pageEvent('pagehide', false));
    await settle(env);
    assert.equal(srv.ws.closeCode, 1000);
    assert.equal(srv.ws.closeReason, 'Page closed', 'the same close as beforeunload');
    assert.equal(engine.isActive(), false);
    assert.equal(engine.getState().status, 'Stopped: left the page');
    assert.equal(engine.getState().error, null, 'leaving is not an error');
    assert.equal(ctx.state, 'closed', 'the AudioContext is closed');
    assert.notEqual(env.app.get().player?.id, 'live', 'the audio session is let go');
  });

  it('[LR-18] [LR-16] a page kept in the back/forward cache keeps its session; back with a socket the browser closed, it shows "Stopped: left the page", never a stale "Listening"', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    window.dispatchEvent(pageEvent('pagehide', true));
    await settle(env);
    assert.equal(engine.isActive(), true, 'a page that may come back is not torn down here');
    assert.equal(srv.ws.readyState, 1, 'the socket stays open');
    window.dispatchEvent(pageEvent('pageshow', true));
    await settle(env);
    assert.equal(engine.isActive(), true, 'back with its socket open: Live goes on');
    assert.equal(engine.getState().status, 'Listening');

    window.dispatchEvent(pageEvent('pagehide', true));
    srv.ws.readyState = 3;   // the browser closed the socket while the page was frozen; no close event yet
    window.dispatchEvent(pageEvent('pageshow', true));
    await settle(env);
    assert.equal(engine.isActive(), false, 'a session without its socket is released');
    assert.equal(engine.getState().status, 'Stopped: left the page');
    assert.equal(FakeAudioContext.instances[0].state, 'closed');
  });
});

describe('Live: the app audio session', () => {
  it('[LR-18] Live claims a live audio session; another player starting stops it with "Stopped: another player started"', async () => {
    const { engine } = await setup({ text: 'Hi.', caret: 0 });
    const srv = await connect(engine);
    const player = env.app.get().player;
    assert.equal(player.id, 'live');
    assert.equal(player.kind, 'live');
    assert.equal(player.homeRoute, 'speak/live');
    assert.equal(player.title, 'Live');
    assert.equal(player.statusText, 'Listening');
    assert.equal(player.playing, true);
    const audio = await load('core/audio.js');
    audio.claim({ id: 'take:1', owner: 'speak', kind: 'clip', title: 'Take 1', controls: { play() {}, pause() {}, toggle() {}, stop() {} }, getState: () => ({ playing: true }) });
    await settle(env);
    assert.equal(engine.isActive(), false);
    assert.equal(engine.getState().status, 'Stopped: another player started');
    assert.equal(engine.getState().reason, 'Another player started');
    assert.equal(srv.ws.closeCode, 1000);
    assert.equal(env.app.get().player.id, 'take:1');
  });

  it('[LR-15] the mini-player and lock-screen controls pause, resume and stop Live', async () => {
    const { engine } = await setup();
    await connect(engine);
    const audio = await load('core/audio.js');
    const controls = audio.current().controls;
    controls.toggle();
    assert.equal(engine.getState().status, 'Paused');
    assert.equal(env.app.get().player.playing, false);
    controls.play();
    assert.equal(engine.getState().status, 'Listening');
    controls.pause();
    assert.equal(engine.getState().status, 'Paused');
    audio.closeCurrent();
    await settle(env);
    assert.equal(engine.isActive(), false);
    assert.equal(engine.getState().status, 'Stopped');
    assert.equal(env.app.get().player, null);
  });

  it('[LR-11] the voice latency is the rolling median of the last 5 responses, from response.created to its first audio', async () => {
    let now = 0;
    const { live, engine } = await setup({ clock: () => now });
    const srv = await connect(engine);
    const delays = [400, 100, 200, 300, 250, 900];
    delays.forEach((ms, i) => {
      now += 1000;
      srv.response(`r${i}`, `Sentence ${i}.`);
      now += ms;
      srv.delta(`r${i}`, 1, { samples: 0 });
    });
    await settle(env);
    assert.equal(engine.getState().latencyMs, 250, 'median of 100, 200, 300, 250, 900');
    assert.equal(live.latencyLabel(engine.getState().latencyMs), 'Voice latency ~250 ms');
    assert.equal(live.latencyLabel(1830), 'Voice latency ~1.8 s');
    assert.equal(live.latencyLabel(null), '');
    engine.stop();
  });

  it('[LR-11] the voice latency leaves out a response whose first audio waited on a pause', async () => {
    let now = 0;
    const { live, engine } = await setup({ clock: () => now });
    const srv = await connect(engine);
    now += 1000;
    srv.response('r1', 'One.');
    now += 200;
    srv.delta('r1', 1, { samples: 0 });
    await settle(env);
    assert.equal(engine.getState().latencyMs, 200);

    now += 1000;
    srv.response('r2', 'Two.');
    env.withGesture(() => engine.togglePause());   // paused before r2's first audio
    now += 10000;                                  // the server holds that audio until playback resumes
    env.withGesture(() => engine.togglePause());
    now += 150;
    srv.delta('r2', 1, { samples: 0 });
    await settle(env);
    assert.equal(engine.getState().latencyMs, 200, 'the paused wait is not voice latency');

    env.setHidden(true);                           // created while the page is in the background
    srv.response('r3', 'Three.');
    now += 5000;
    env.setHidden(false);
    srv.delta('r3', 1, { samples: 0 });
    await settle(env);
    assert.equal(engine.getState().latencyMs, 200);
    assert.equal(live.latencyLabel(engine.getState().latencyMs), 'Voice latency ~200 ms');

    now += 1000;
    srv.response('r4', 'Four.');
    now += 300;
    srv.delta('r4', 1, { samples: 0 });
    await settle(env);
    assert.equal(engine.getState().latencyMs, 250, 'the next unpaused response counts again (median of 200 and 300)');
    engine.stop();
  });
});
