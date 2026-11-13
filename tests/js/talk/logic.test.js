// Talk's pure parts: the microphone's resampling and encoding, the GA events Talk
// sends (src/realtime/session.py's shapes, a typed question too), the presets it offers, the voice it
// speaks in, and the status and error copy.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function logic() {
  env = installBrowserEnv();
  return importFresh('views/talk/logic.js');
}

describe('Talk logic', () => {
  it('the resampler turns 48 kHz frames into 24 kHz by averaging, keeping its place across frames', async () => {
    const L = await logic();
    const push = L.createResampler(48000, 24000);
    const a = push(Float32Array.from([0, 1, 1, 0, 0.5]));
    assert.deepEqual(Array.from(a), [0.5, 0.5]);
    const b = push(Float32Array.from([0.5, -1, -1]));
    assert.deepEqual(Array.from(b), [0.5, -1], 'the sample held back joins the next frame');
    let total = 0;
    const steady = L.createResampler(48000, 24000);
    for (let i = 0; i < 10; i += 1) total += steady(new Float32Array(2048).fill(0.2)).length;
    assert.equal(total, 10240);
  });

  it('the resampler handles 44.1 kHz, 16 kHz (upsampling) and 24 kHz (a copy)', async () => {
    const L = await logic();
    const down = L.createResampler(44100, 24000);
    let n = 0;
    for (let i = 0; i < 100; i += 1) {
      const out = down(new Float32Array(2048).fill(0.5));
      n += out.length;
      for (const v of out) assert.ok(Math.abs(v - 0.5) < 1e-6, 'a constant stays constant');
    }
    assert.ok(Math.abs(n - (100 * 2048 * 24000) / 44100) <= 2, `about 24000/44100 of the input (${n})`);
    const up = L.createResampler(16000, 24000);
    const ramp = up(Float32Array.from([0, 0.3, 0.6, 0.9, 1.2]));
    assert.ok(ramp.length >= 5 && ramp.length <= 7, `${ramp.length}`);
    assert.ok(Math.abs(ramp[1] - 0.2) < 1e-6, 'linear between samples');
    const same = L.createResampler(24000, 24000);
    const src = Float32Array.from([0.1, 0.2]);
    const copy = same(src);
    assert.deepEqual(Array.from(copy), Array.from(src));
    assert.notEqual(copy, src, 'a copy, never the frame the worklet reuses');
  });

  it('an append carries little-endian PCM16, clamped, as base64', async () => {
    const L = await logic();
    const { decodePcm16 } = await importFresh('views/speak/live-session.js');
    const ev = L.appendEvent(Float32Array.from([0, 0.5, -0.5, 1.5, -2]));
    assert.equal(ev.type, 'input_audio_buffer.append');
    const back = Array.from(decodePcm16(ev.audio));
    assert.deepEqual(back.map((v) => Math.round(v * 32768)), [0, 16384, -16384, 32767, -32768]);
    assert.equal(L.int16ToBase64(Int16Array.from([1, -1])), btoa(String.fromCharCode(1, 0, 255, 255)));
  });

  it('session.update: 24 kHz PCM both ways, server turn detection that replies and can be interrupted, the voice, the model and the AI preset', async () => {
    const L = await logic();
    const ev = L.sessionUpdate({ model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', preset: 'short', interrupt: true }, { eventId: 'evt_update_1' });
    assert.deepEqual(ev, {
      type: 'session.update',
      event_id: 'evt_update_1',
      session: {
        type: 'realtime',
        model: 'kokoro',
        audio: {
          input: { format: { type: 'audio/pcm', rate: 24000 }, turn_detection: { type: 'server_vad', create_response: true, interrupt_response: true } },
          output: { format: { type: 'audio/pcm', rate: 24000 }, voice: 'af_bella(2)+af_sky(1)' },
        },
        assistant: { preset: 'short' },
      },
    });
    const quiet = L.sessionUpdate({ model: '', voice: '', preset: '', interrupt: false }, { turnDetection: false });
    assert.equal('event_id' in quiet, false);
    assert.equal('model' in quiet.session, false, 'no model: the server keeps TTS_MODEL');
    assert.equal('voice' in quiet.session.audio.output, false);
    assert.equal('turn_detection' in quiet.session.audio.input, false, 'left alone once the server said it has none');
    assert.deepEqual(quiet.session.assistant, { preset: 'voice' });
    assert.equal(L.sessionUpdate({ interrupt: false }).session.audio.input.turn_detection.interrupt_response, false);
  });

  it('the keep-alive is a session.update that changes nothing', async () => {
    const L = await logic();
    assert.deepEqual(L.keepalive(), { type: 'session.update', session: {} });
    assert.ok(L.KEEPALIVE_MS < 120000 / 2, 'well inside the default OS_REALTIME_IDLE_TIMEOUT_S');
  });

  it('a typed question is conversation.item.create with a user input_text message, its own item and event ids', async () => {
    const L = await logic();
    const q = L.typedQuestion("What's the weather tomorrow?", 7);
    assert.deepEqual(q, {
      itemId: 'typed_7',
      eventId: 'evt_typed_7',
      event: {
        type: 'conversation.item.create',
        event_id: 'evt_typed_7',
        item: { id: 'typed_7', type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's the weather tomorrow?" }] },
      },
    });
    assert.ok(q.itemId.length <= 32);
    assert.equal(L.TEXT_MAX, 8000);
  });

  it('the presets offered are the voice presets: everything but the book preset', async () => {
    const L = await logic();
    assert.deepEqual(L.voicePresets([{ id: 'voice', name: 'Voice' }, { id: 'book', name: 'Book' }, { id: 'short', name: 'Short answers' }, null, { name: 'no id' }]),
      [{ id: 'voice', name: 'Voice' }, { id: 'short', name: 'Short answers' }]);
    assert.deepEqual(L.voicePresets([]), [{ id: 'voice', name: 'Voice' }]);
    assert.deepEqual(L.voicePresets(undefined), [{ id: 'voice', name: 'Voice' }]);
  });

  // A remembered preset that no longer exists falls back to the default, quietly.
  it('the preset used: the remembered one while the list offers it, else the default (voice), else the first offered', async () => {
    const L = await logic();
    const list = [{ id: 'voice', name: 'Voice' }, { id: 'book', name: 'Book' }, { id: 'short', name: 'Short answers' }];
    assert.equal(L.pickPreset(list, 'short'), 'short');
    assert.equal(L.pickPreset(list, 'deleted'), 'voice', 'gone: the default');
    assert.equal(L.pickPreset([{ id: 'short' }, { id: 'voice' }], 'deleted'), 'voice', 'the default, wherever the list has it (not its first)');
    assert.equal(L.pickPreset(list, 'book'), 'voice', 'the book preset is never offered for talking');
    assert.equal(L.pickPreset(list, ''), 'voice');
    assert.equal(L.pickPreset([{ id: 'calm' }, { id: 'short' }], 'deleted'), 'calm', 'no voice preset: the first one');
    assert.equal(L.pickPreset([], 'short'), 'voice');
  });

  it("the reply voice: the one chosen in Talk, else Speak's (never a raw recording), else the default model's", async () => {
    const L = await logic();
    const chosen = { model: 'piper/en_US-amy-medium', voice: 'en_US-amy-medium', label: 'Amy' };
    assert.deepEqual(L.talkVoice({ chosen, speak: { model: 'kokoro', voice: 'af_bella', label: 'Bella' }, defaultModel: 'kokoro' }), { ...chosen, source: 'chosen' });
    assert.deepEqual(L.talkVoice({ speak: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator', kind: 'named' }, defaultModel: 'kokoro' }),
      { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator', source: 'speak' });
    const fallback = L.talkVoice({ speak: { model: 'kokoro', voice: 'demo_reference', label: 'demo', kind: 'recording' }, defaultModel: 'kokoro' });
    assert.equal(fallback.source, 'default');
    assert.equal(fallback.model, 'kokoro');
    assert.equal(fallback.voice, 'af_heart');
    const none = L.talkVoice({ speak: null, defaultModel: '' });
    assert.deepEqual(none, { model: '', voice: '', label: 'Server default', source: 'default' });
  });

  it("an AI error's entry is the preset's spoken sentence, with the server's words as detail; without speak_errors, the server's words", async () => {
    const L = await logic();
    const preset = { speak_errors: true, error_sentences: { assistant_unreachable: "I couldn't reach the AI server." } };
    assert.deepEqual(L.errorEntry({ type: 'server_error', code: 'assistant_unreachable', message: "Can't reach the AI server at 127.0.0.1:9" }, preset),
      { text: "I couldn't reach the AI server.", detail: "Can't reach the AI server at 127.0.0.1:9" });
    assert.deepEqual(L.errorEntry({ type: 'server_error', code: 'assistant_unreachable', message: 'down' }, { ...preset, speak_errors: false }), { text: 'down', detail: '' });
    assert.deepEqual(L.errorEntry({ type: 'server_error', code: 'assistant_timeout', message: 'slow' }, preset), { text: 'slow', detail: '' });
    assert.deepEqual(L.errorEntry({ code: 'tts_error', message: 'kokoro fell over' }, preset), { text: "Couldn't speak the answer.", detail: 'kokoro fell over' });
    assert.deepEqual(L.errorEntry({ code: 'other', message: '' }, null), { text: 'Something went wrong on the server.', detail: '' });
  });

  // A streamed answer that an error or the token limit cut
  // off after some of it was spoken ends `incomplete` with the code in status_details.error.code, and
  // Talk puts a short note under it, in plain words, from the code.
  it('the note under a cut-off answer: the token limit in plain words; another code the preset\'s error sentence, or the plain message Talk uses for it', async () => {
    const L = await logic();
    assert.equal(typeof L.cutOffNote, 'function', 'a note for a cut-off answer');
    const preset = { speak_errors: true, error_sentences: { assistant_timeout: 'The AI took too long to answer.', assistant_reply_cut_off: 'The answer was cut off before it started.' } };
    assert.equal(L.cutOffNote('assistant_reply_cut_off', preset), 'The answer was cut off: the AI reached its token limit.', 'never the sentence for an answer cut off before it started');
    assert.equal(L.cutOffNote('assistant_reply_cut_off', null), 'The answer was cut off: the AI reached its token limit.');
    assert.equal(L.cutOffNote('assistant_timeout', preset), 'The AI took too long to answer.');
    assert.equal(L.cutOffNote('assistant_timeout', { ...preset, speak_errors: false }), 'Something went wrong on the server.', 'as the error entry reads without speak_errors');
    assert.equal(L.cutOffNote('assistant_http_error', preset), 'Something went wrong on the server.', 'a code the preset has no sentence for');
    assert.equal(L.cutOffNote('', preset), 'The answer was cut off.', 'no code: still says it was cut off');
  });

  it('microphone problems say what happened and that typing still works', async () => {
    const L = await logic();
    const denied = L.micProblem(Object.assign(new Error("Microphone access is blocked. Allow it in your browser's site settings, then try again."), { name: 'MicError', reason: 'denied' }));
    assert.equal(denied.title, 'Microphone problem');
    assert.match(denied.message, /^Microphone access is blocked\..*You can still type your questions\.$/);
    const busy = L.micProblem(Object.assign(new Error('Stop the Transcribe recording first'), { name: 'MicBusyError' }));
    assert.equal(busy.message, 'Stop the Transcribe recording first. You can still type your questions.');
  });

  it('the status: one word per phase; the spoken status changes with the phase only, never per turn', async () => {
    const L = await logic();
    const base = { phase: 'ready', mic: 'on', reply: 'idle', hearing: false, transcribing: false, paused: false, stopReason: '', error: null };
    const label = (patch) => L.statusOf({ ...base, ...patch }).label;
    const spoken = (patch) => L.statusOf({ ...base, ...patch }).spoken;
    assert.equal(label({}), 'Listening');
    assert.equal(label({ hearing: true }), 'Hearing you…');
    assert.equal(label({ transcribing: true }), 'Writing down your words…');
    assert.equal(label({ reply: 'thinking' }), 'Thinking…');
    assert.equal(label({ reply: 'speaking' }), 'Speaking');
    assert.equal(label({ mic: 'off' }), 'Connected');
    assert.equal(label({ paused: true }), 'Paused');
    assert.equal(label({ phase: 'connecting' }), 'Connecting…');
    assert.equal(label({ phase: 'reconnecting' }), 'Reconnecting…');
    assert.equal(label({ phase: 'stopped' }), 'Stopped');
    assert.equal(label({ phase: 'stopped', stopReason: 'Another player started' }), 'Stopped: another player started');
    for (const patch of [{}, { hearing: true }, { transcribing: true }, { reply: 'thinking' }, { reply: 'speaking' }]) {
      assert.equal(spoken(patch), 'Listening', JSON.stringify(patch));
    }
    assert.equal(spoken({ mic: 'off', reply: 'speaking' }), 'Connected');
    assert.equal(L.statusOf({ ...base, phase: 'stopped', error: { title: 'Talk stopped' } }).tone, 'error');
  });
});
