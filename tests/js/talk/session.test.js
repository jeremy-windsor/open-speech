// Talk's session engine against a fake WebSocket that plays the GA /v1/realtime
// events (src/realtime/server.py: a spoken turn, a typed question), the real microphone
// module on fake devices, and the real audio coordinator. No real server or AI is reached.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  engineSetup, startConnected, accept, answer, settle, lastSocket, sent, sentTypes, micFrame, micContext,
  playbackContext, samplesIn, ga, logOf, pendingTimers, PRESETS, load,
} from './helpers.js';

let t;
afterEach(() => {
  try { t?.engine.destroy(); } catch { /* already gone */ }
  t?.env.restore();
  t = null;
});

const appends = (ws) => sent(ws, { appends: true }).filter((e) => e.type === 'input_audio_buffer.append');

describe('Talk session: starting', () => {
  it('Start makes the AudioContext and opens the microphone inside the click, claims a live audio session, then opens a GA /v1/realtime socket', async () => {
    t = await engineSetup();
    const ok = t.env.withGesture(() => t.engine.start({ mic: true }));
    assert.equal(ok, true);
    assert.equal(t.env.audioContexts.length >= 2, true, 'the playback context and the microphone context, both inside the click');
    assert.equal(t.env.mediaDevices.calls.length, 1, 'the microphone is asked for inside the click');
    const cur = t.audio.current();
    assert.ok(cur, 'an audio session is claimed');
    assert.deepEqual([cur.id, cur.owner, cur.kind, cur.homeRoute], ['talk', 'talk', 'live', 'talk']);
    for (const name of ['play', 'pause', 'toggle', 'stop']) assert.equal(typeof cur.controls[name], 'function', name);
    await settle(t.env);
    assert.deepEqual(t.readyCalls, [['kokoro', { kind: 'tts' }]], 'the voice model is made ready first');
    const audioReq = t.env.mediaDevices.calls[0].audio;
    assert.equal(audioReq.echoCancellation, true, 'the browser cancels the reply\'s echo');
    assert.equal(t.mic.owner(), 'talk');
    const ws = lastSocket(t.env);
    assert.ok(ws, 'a socket');
    assert.equal(new URL(ws.url).pathname, '/v1/realtime');
    assert.equal(ws.protocols, undefined, 'no subprotocol: a GA session');
    assert.equal(t.state().phase, 'connecting');
    ws.open();
    ws.serverSend(ga.created());
    await settle(t.env);
    const update = sent(ws)[0];
    assert.equal(update.type, 'session.update');
    assert.equal(update.session.model, 'kokoro');
    assert.equal(update.session.audio.output.voice, 'af_bella');
    assert.deepEqual(update.session.audio.input.format, { type: 'audio/pcm', rate: 24000 });
    assert.deepEqual(update.session.audio.input.turn_detection, { type: 'server_vad', create_response: true, interrupt_response: true });
    assert.deepEqual(update.session.assistant, { preset: 'voice' });
    ws.serverSend(ga.updated());
    await settle(t.env);
    assert.equal(t.state().phase, 'ready');
    assert.equal(t.state().mic, 'on');
    assert.equal(t.state().status.label, 'Listening');
  });

  it('microphone frames before the session is ready are dropped; then 48 kHz frames go out as 100 ms PCM16 appends at 24 kHz', async () => {
    t = await engineSetup();
    t.env.withGesture(() => t.engine.start({ mic: true }));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    micFrame(t.env);
    micFrame(t.env);
    await settle(t.env);
    assert.equal(appends(ws).length, 0, 'nothing before session.updated');
    ws.serverSend(ga.created());
    ws.serverSend(ga.updated());
    await settle(t.env);
    for (let i = 0; i < 5; i += 1) micFrame(t.env);   // 5 × 2048 at 48 kHz = 5120 samples at 24 kHz
    await settle(t.env);
    const out = appends(ws);
    assert.equal(out.length, 2, 'two whole 100 ms appends (2400 samples each); the rest waits');
    for (const ev of out) assert.equal(samplesIn(ev.audio), 2400);
  });

  it('a start the browser can\'t play refuses; one that is already running is a no-op', async () => {
    t = await engineSetup({ engine: { AudioContextCtor: function Broken() { throw new Error('no audio'); } } });
    assert.equal(t.env.withGesture(() => t.engine.start({ mic: true })), false);
    assert.match(t.state().refusal, /can't play/);
    assert.equal(t.audio.current(), null);
    t.engine.destroy();
    t.env.restore();
    t = await engineSetup();
    assert.equal(t.env.withGesture(() => t.engine.start({ mic: false })), true);
    assert.equal(t.env.withGesture(() => t.engine.start({ mic: false })), false);
    await settle(t.env);
    assert.equal(t.env.sockets.length, 1);
  });

  it('Talk refuses to start while another view records, naming it', async () => {
    t = await engineSetup();
    t.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    assert.equal(t.env.withGesture(() => t.engine.start({ mic: true })), false);
    assert.equal(t.state().refusal, 'Stop the Transcribe recording first.');
    assert.equal(t.env.sockets.length, 0);
  });
});

describe('Talk session: a spoken turn', () => {
  it('your words, the waiting phrases and the thinking (dimmed kinds), then the answer\'s text before its audio, played back to back', async () => {
    t = await engineSetup();
    const ws = await startConnected(t);
    ws.serverSend(ga.speechStarted());
    await settle(t.env);
    assert.equal(t.state().hearing, true);
    assert.equal(t.state().status.label, 'Hearing you…');
    ws.serverSend(ga.speechStopped());
    ws.serverSend(ga.committed());
    ws.serverSend(ga.audioItem());
    await settle(t.env);
    assert.equal(t.state().hearing, false);
    assert.equal(t.state().transcribing, true);
    ws.serverSend(ga.transcript('item_1', 'What is the weather tomorrow?'));
    await settle(t.env);
    assert.equal(t.state().transcribing, false);
    assert.deepEqual(logOf(t.state()), [['you', 'What is the weather tomorrow?']]);

    ws.serverSend(ga.responseCreated('resp_1'));
    ws.serverSend(ga.itemAdded('resp_1'));
    ws.serverSend(ga.partAdded('resp_1'));
    ws.serverSend(ga.filler('resp_1', 'Hmm.'));
    ws.serverSend(ga.audio('resp_1', 1200));
    await settle(t.env);
    assert.equal(t.state().reply, 'thinking', 'a waiting phrase plays while the AI thinks');
    assert.equal(t.state().status.label, 'Thinking…');
    ws.serverSend(ga.thinking('resp_1', 'They want the weather.'));
    ws.serverSend(ga.textDelta('resp_1', 'Sunny, with a light wind.'));
    await settle(t.env);
    assert.deepEqual(logOf(t.state()), [
      ['you', 'What is the weather tomorrow?'], ['filler', 'Hmm.'], ['thinking', 'They want the weather.'], ['ai', 'Sunny, with a light wind.'],
    ], 'the text shows before its audio');
    ws.serverSend(ga.audio('resp_1'));
    ws.serverSend(ga.audio('resp_1'));
    ws.serverSend(ga.audioDone('resp_1'));
    ws.serverSend(ga.textDone('resp_1', 'Sunny, with a light wind.'));
    ws.serverSend(ga.done('resp_1'));
    await settle(t.env);
    assert.equal(t.state().reply, 'speaking', 'the audio is still playing');
    const ctx = playbackContext(t.env);
    const sources = ctx.__sources;
    assert.equal(sources.length, 3, 'the phrase and two answer chunks');
    for (let i = 1; i < sources.length; i += 1) {
      const prev = sources[i - 1];
      assert.ok(Math.abs(sources[i].startedAt - (prev.startedAt + prev.buffer.duration)) < 1e-9, 'back to back, no gap');
    }
    assert.equal(sources[1].buffer.sampleRate, 24000);
    await settle(t.env, 300);
    assert.equal(t.state().reply, 'idle');
    assert.equal(t.state().status.label, 'Listening');
  });

  it('an AI error: the preset\'s spoken sentence in the log with the server\'s words, its audio plays, the session carries on', async () => {
    t = await engineSetup();
    const ws = await startConnected(t);
    ws.serverSend(ga.responseCreated('resp_2'));
    ws.serverSend(ga.error('assistant_unreachable', "Can't reach the AI server at 127.0.0.1:9", { type: 'server_error' }));
    ws.serverSend(ga.audio('resp_2'));
    ws.serverSend(ga.audioDone('resp_2'));
    ws.serverSend(ga.done('resp_2', 'failed', { type: 'failed', error: { type: 'server_error', code: 'assistant_unreachable' } }));
    await settle(t.env);
    const entry = t.state().log.at(-1);
    assert.deepEqual([entry.kind, entry.text, entry.detail], ['error', "I couldn't reach the AI server.", "Can't reach the AI server at 127.0.0.1:9"]);
    assert.equal(playbackContext(t.env).__sources.length, 1, 'the sentence is heard');
    assert.equal(t.state().error, null, 'not a stop');
    assert.equal(t.state().phase, 'ready');
    assert.equal(t.state().log.filter((e) => e.kind === 'error').length, 1, 'said once');
  });

  it('a failed transcription and a failed synthesis each leave one plain entry', async () => {
    t = await engineSetup();
    const ws = await startConnected(t);
    ws.serverSend(ga.speechStarted());
    ws.serverSend(ga.speechStopped());
    ws.serverSend(ga.transcriptFailed());
    ws.serverSend(ga.error('transcription_error', 'whisper fell over', { type: 'invalid_request_error' }));
    await settle(t.env);
    ws.serverSend(ga.responseCreated('resp_3'));
    ws.serverSend(ga.error('tts_error', 'kokoro fell over', { type: 'invalid_request_error' }));
    ws.serverSend(ga.done('resp_3', 'failed'));
    await settle(t.env);
    assert.deepEqual(t.state().log.map((e) => [e.kind, e.text, e.detail]), [
      ['error', "Couldn't write down what you said.", 'whisper fell over'],
      ['error', "Couldn't speak the answer.", 'kokoro fell over'],
    ]);
    assert.equal(t.state().transcribing, false);
    assert.equal(t.state().reply, 'idle');
  });

  it('talking over a reply (the server cancels it, turn_detected): its audio stops at once, later chunks of it are dropped, the answer is marked interrupted', async () => {
    t = await engineSetup();
    const ws = await startConnected(t);
    ws.serverSend(ga.responseCreated('resp_4'));
    ws.serverSend(ga.textDelta('resp_4', 'A long answer.'));
    ws.serverSend(ga.audio('resp_4', 24000));
    ws.serverSend(ga.audio('resp_4', 24000));
    await settle(t.env);
    const ctx = playbackContext(t.env);
    assert.equal(ctx.__sources.length, 2);
    ws.serverSend(ga.speechStarted('item_9'));
    ws.serverSend(ga.done('resp_4', 'cancelled', { type: 'cancelled', reason: 'turn_detected' }));
    await settle(t.env);
    assert.ok(ctx.__sources.every((s) => s.stopped), 'flushed');
    ws.serverSend(ga.audio('resp_4', 2400));
    await settle(t.env);
    assert.equal(ctx.__sources.length, 2, 'a late chunk of the cancelled reply is not played');
    const ai = t.state().log.find((e) => e.kind === 'ai');
    assert.equal(ai.state, 'interrupted');
    assert.equal(t.state().reply, 'idle');
  });
});

// With streaming, an error or the token limit
// after some of the answer was spoken ends the response `incomplete`, with the code in status_details.error.code and
// no error event. Talk keeps what was said and puts a short note under it, announced politely.
describe('Talk session: a cut-off answer', () => {
  const CUT_OFF = 'The answer was cut off: the AI reached its token limit.';
  const incomplete = (id, code) => ga.done(id, 'incomplete', { type: 'incomplete', error: { type: 'server_error', code } });
  const said = (t) => t.env.document.getElementById('app-sr-status').textContent;

  /** A streamed answer: each sentence's text just before its audio (src/realtime/server.py). */
  function streamed(ws, id, sentences) {
    ws.serverSend(ga.responseCreated(id));
    ws.serverSend(ga.itemAdded(id));
    ws.serverSend(ga.partAdded(id));
    for (const sentence of sentences) {
      ws.serverSend(ga.textDelta(id, sentence));
      ws.serverSend(ga.audio(id, 2400));
    }
  }

  it('an answer that ends incomplete keeps what was said and gets a note under it, in plain words from the code; the session goes on', async () => {
    t = await engineSetup();
    (await load('core/a11y.js')).__resetForTests();
    const ws = await startConnected(t);
    ws.serverSend(ga.transcript('item_1', 'Tell me about the Moon.'));
    streamed(ws, 'resp_1', ['The Moon is about 384,000 km away.', ' It has no air.']);
    ws.serverSend(ga.audioDone('resp_1'));
    ws.serverSend(ga.textDone('resp_1', 'The Moon is about 384,000 km away. It has no air.'));
    ws.serverSend(incomplete('resp_1', 'assistant_reply_cut_off'));
    await settle(t.env);
    assert.deepEqual(logOf(t.state()), [
      ['you', 'Tell me about the Moon.'],
      ['ai', 'The Moon is about 384,000 km away. It has no air.'],
      ['note', CUT_OFF],
    ], 'what was said stays, and a note says it was cut off');
    const ai = t.state().log.find((e) => e.kind === 'ai');
    assert.equal(ai.state, '', 'not marked interrupted or stopped');
    assert.equal(t.state().log.filter((e) => e.kind === 'error').length, 0, 'not an error entry');
    assert.deepEqual([t.state().phase, t.state().error], ['ready', null], 'the session goes on');
    assert.notEqual(said(t), CUT_OFF, 'not announced over the answer while its sound still plays');
    await settle(t.env, 300);
    assert.equal(t.state().reply, 'idle', 'the sound has played');
    await settle(t.env, 600);
    assert.equal(said(t), CUT_OFF, 'announced politely once the answer\'s sound has played');
    const region = t.env.document.getElementById('app-sr-status');
    assert.equal(region.getAttribute('aria-live'), 'polite');

    // Another code: the preset's error sentence (the voice preset speaks errors).
    streamed(ws, 'resp_2', ['First, the weather.']);
    ws.serverSend(ga.audioDone('resp_2'));
    ws.serverSend(incomplete('resp_2', 'assistant_timeout'));
    await settle(t.env, 1000);
    assert.deepEqual(logOf(t.state()).slice(-2), [['ai', 'First, the weather.'], ['note', 'The AI took too long to answer.']]);
    assert.equal(said(t), 'The AI took too long to answer.');
    // A completed answer gets no note.
    await answer(t.env, ws, 'resp_3', { text: 'All done.' });
    await settle(t.env, 1000);
    assert.deepEqual(logOf(t.state()).slice(-1), [['ai', 'All done.']]);
  });

  it('the note goes right under the answer it is about, even when your next words came first; New conversation before its sound ends says nothing', async () => {
    t = await engineSetup({ choices: { model: 'kokoro', voice: 'af_bella', label: 'Bella', preset: 'voice', interrupt: false } });
    (await load('core/a11y.js')).__resetForTests();
    const ws = await startConnected(t);
    streamed(ws, 'resp_1', ['A long answer, read slowly.']);
    ws.serverSend(ga.audio('resp_1', 24000));
    // With Interrupt by talking off, your next turn can be written down while the answer still plays.
    ws.serverSend(ga.transcript('item_2', 'And the Sun?'));
    ws.serverSend(ga.audioDone('resp_1'));
    ws.serverSend(incomplete('resp_1', 'assistant_reply_cut_off'));
    await settle(t.env);
    assert.deepEqual(logOf(t.state()), [['ai', 'A long answer, read slowly.'], ['note', CUT_OFF], ['you', 'And the Sun?']]);
    t.engine.newConversation();
    await settle(t.env, 2000);
    assert.notEqual(said(t), CUT_OFF, 'the chat it was about is gone: nothing is announced');
    assert.deepEqual(t.state().log, []);
  });
});

/* ───────────── Interrupt by talking off: the microphone waits while it answers ───────────── */

const NO_INTERRUPT = Object.freeze({ model: 'kokoro', voice: 'af_bella', label: 'Bella', preset: 'voice', interrupt: false });

/** Whether microphone frames reach the socket now: five frames make two appends when they do. */
function micSends(ws) {
  const before = appends(ws).length;
  for (let i = 0; i < 5; i += 1) micFrame(t.env);
  return appends(ws).length > before;
}

/** A spoken turn the server's voice detection ended and committed. */
async function committedTurn(ws, item = 'item_1') {
  ws.serverSend(ga.speechStarted(item));
  ws.serverSend(ga.speechStopped(item));
  ws.serverSend(ga.committed(item));
  ws.serverSend(ga.transcript(item, 'What about tomorrow?'));
  await settle(t.env);
}

/** An answer whose sound (two 100 ms chunks) is still to play: response.done has come. */
function spokenAnswer(ws, id = 'resp_1') {
  ws.serverSend(ga.responseCreated(id));
  ws.serverSend(ga.textDelta(id, 'Sunny.'));
  ws.serverSend(ga.audio(id));
  ws.serverSend(ga.audio(id));
  ws.serverSend(ga.audioDone(id));
  ws.serverSend(ga.done(id));
}

describe('Talk session: Interrupt by talking off', () => {
  it('your committed turn holds the microphone through the answer (open, nothing sent); once it is done and its sound has played, 300 ms later it listens again', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    const track = t.env.mediaDevices.streams[0].getTracks()[0];
    assert.equal(micSends(ws), true, 'listening');
    await committedTurn(ws);
    assert.equal(t.state().micHeld, true);
    assert.equal(t.state().status.label, 'Paused while it answers');
    assert.equal(micSends(ws), false, 'held: what it hears is dropped');
    spokenAnswer(ws);
    await settle(t.env);
    assert.equal(t.state().status.label, 'Speaking');
    assert.equal(micSends(ws), false, 'the answer is done on the server, but still playing');
    await settle(t.env, 250);
    assert.equal(t.state().reply, 'idle', 'its sound has played');
    assert.equal(t.state().micHeld, true);
    assert.equal(micSends(ws), false, 'not yet: 300 ms more');
    await settle(t.env, 320);
    assert.equal(t.state().micHeld, false);
    assert.equal(t.state().status.label, 'Listening');
    assert.equal(micSends(ws), true, 'listening again');
    assert.equal(track.readyState, 'live', 'the microphone stayed open all along');
    assert.equal(t.env.mediaDevices.calls.length, 1, 'never asked for again');
  });

  it('with Interrupt by talking on, nothing is held', async () => {
    t = await engineSetup();
    const ws = await startConnected(t);
    await committedTurn(ws);
    spokenAnswer(ws);
    await settle(t.env);
    assert.equal(t.state().micHeld, false);
    assert.equal(micSends(ws), true);
  });

  it('Stop reply stops the sound and listens again at once', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    await committedTurn(ws);
    ws.serverSend(ga.responseCreated('resp_1'));
    ws.serverSend(ga.audio('resp_1', 24000));
    await settle(t.env);
    assert.equal(t.state().micHeld, true);
    t.engine.cancelReply();
    await settle(t.env);
    assert.equal(t.state().micHeld, false);
    assert.equal(t.state().reply, 'idle');
    assert.equal(micSends(ws), true);
  });

  it('an answer that fails: with its error sentence spoken, it listens again once that has played; with nothing spoken, at once', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    await committedTurn(ws);
    ws.serverSend(ga.responseCreated('resp_1'));
    ws.serverSend(ga.error('assistant_unreachable', "Can't reach the AI server", { type: 'server_error' }));
    ws.serverSend(ga.audio('resp_1'));
    ws.serverSend(ga.done('resp_1', 'failed'));
    await settle(t.env);
    assert.equal(t.state().micHeld, true, 'the spoken error sentence still plays');
    await settle(t.env, 150);
    await settle(t.env, 320);
    assert.equal(t.state().micHeld, false, 'it has played');
    await committedTurn(ws, 'item_2');
    ws.serverSend(ga.responseCreated('resp_2'));
    ws.serverSend(ga.error('assistant_unreachable', "Can't reach the AI server", { type: 'server_error' }));
    ws.serverSend(ga.done('resp_2', 'failed'));
    await settle(t.env);
    assert.equal(t.state().micHeld, false, 'nothing spoken: at once');
    assert.equal(micSends(ws), true);
  });

  it('Interrupt by talking turned on mid-answer listens at once; turned off mid-answer, it holds from the next turn', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    await committedTurn(ws);
    ws.serverSend(ga.responseCreated('resp_1'));
    ws.serverSend(ga.audio('resp_1', 24000));
    await settle(t.env);
    assert.equal(t.state().micHeld, true);
    t.choices = { ...t.choices, interrupt: true };
    t.engine.update();
    await settle(t.env);
    assert.equal(t.state().micHeld, false, 'on: listening at once');
    assert.equal(micSends(ws), true);
    t.choices = { ...t.choices, interrupt: false };
    t.engine.update();
    await settle(t.env);
    assert.equal(t.state().micHeld, false, 'off mid-answer: not this one');
    assert.equal(micSends(ws), true);
    ws.serverSend(ga.done('resp_1'));
    await settle(t.env, 2000);
    await committedTurn(ws, 'item_2');
    assert.equal(t.state().micHeld, true, 'the next turn');
  });

  it('a typed question holds it too, a turn being spoken then ending with silence; a refused question listens again at once', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    ws.serverSend(ga.speechStarted());
    await settle(t.env);
    const before = appends(ws).length;
    t.engine.ask('And tomorrow?');
    await settle(t.env);
    assert.equal(t.state().micHeld, true);
    const silence = appends(ws).slice(before);
    assert.equal(silence.length, 1, 'one silence, so the voice detection is not left mid-speech');
    assert.equal(samplesIn(silence[0].audio), 24000 * 0.8, '800 ms of it');
    assert.equal(micSends(ws), false);
    const create = sent(ws).filter((e) => e.type === 'conversation.item.create').at(-1);
    ws.serverSend(ga.error('invalid_request', 'Text too long', { event_id: create.event_id }));
    await settle(t.env);
    assert.equal(t.state().micHeld, false);
    assert.equal(micSends(ws), true);
  });

  it('a turn that gets no answer (no words, or its transcription failed) listens again at once', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    ws.serverSend(ga.committed('item_1'));
    ws.serverSend(ga.transcript('item_1', '  '));
    await settle(t.env);
    assert.equal(t.state().micHeld, false, 'no words');
    assert.equal(micSends(ws), true);
    ws.serverSend(ga.committed('item_2'));
    ws.serverSend(ga.transcriptFailed('item_2'));
    ws.serverSend(ga.error('transcription_error', 'whisper fell over'));
    await settle(t.env);
    assert.equal(t.state().micHeld, false, 'transcription failed');
    assert.equal(micSends(ws), true);
  });

  it('a turn with words stays held until its answer starts', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    await committedTurn(ws);
    ws.serverSend(ga.responseCreated('resp_1'));
    await settle(t.env, 400);
    assert.equal(t.state().micHeld, true);
  });

  it('a typed question that could not be sent does not hold it', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    const ws = await startConnected(t);
    ws.readyState = 2;   // closing
    t.engine.ask('And tomorrow?');
    await settle(t.env);
    assert.equal(t.state().micHeld, false);
    ws.readyState = 1;
    assert.equal(micSends(ws), true);
  });

  it('a dropped connection, New conversation or Stop never leave it held', async () => {
    t = await engineSetup({ choices: NO_INTERRUPT });
    let ws = await startConnected(t);
    await committedTurn(ws);
    ws.serverSend(ga.responseCreated('resp_1'));
    await settle(t.env);
    assert.equal(t.state().micHeld, true);
    ws.serverClose(1006, '');
    await settle(t.env, 1000);
    ws = lastSocket(t.env);
    await accept(t.env, ws);
    assert.equal(t.state().phase, 'ready');
    assert.equal(t.state().micHeld, false, 'reconnected: listening');
    assert.equal(micSends(ws), true);
    await committedTurn(ws);
    assert.equal(t.state().micHeld, true);
    t.engine.newConversation();
    await settle(t.env);
    ws = lastSocket(t.env);
    await accept(t.env, ws);
    assert.equal(t.state().micHeld, false, 'a new conversation: listening');
    await committedTurn(ws);
    t.engine.stop();
    await settle(t.env);
    assert.equal(t.state().micHeld, false);
    await settle(t.env, 400);
    assert.equal(t.state().micHeld, false);
    assert.deepEqual(t.env.errors, []);
  });
});

describe('Talk session: typed questions', () => {
  it('a typed question: conversation.item.create, then response.create only once the item is acknowledged; your words join the log', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    assert.equal(t.engine.ask("What's the weather tomorrow?"), true);
    await settle(t.env);
    const create = sent(ws).at(-1);
    assert.equal(create.type, 'conversation.item.create');
    assert.equal(create.event_id, 'evt_typed_1');
    assert.deepEqual(create.item, { id: 'typed_1', type: 'message', role: 'user', content: [{ type: 'input_text', text: "What's the weather tomorrow?" }] });
    assert.equal(sentTypes(ws).includes('response.create'), false, 'never sent blind');
    assert.equal(t.state().typed.busy, true);
    ws.serverSend(ga.typedAdded('typed_1', "What's the weather tomorrow?"));
    ws.serverSend(ga.typedDone('typed_1', "What's the weather tomorrow?"));
    await settle(t.env);
    assert.deepEqual(sent(ws).at(-1), { type: 'response.create' }, 'no text, no instructions: the AI answers');
    assert.equal(sentTypes(ws).filter((x) => x === 'response.create').length, 1);
    assert.deepEqual(t.state().typed, { busy: false, error: '', sent: 1 });
    const you = t.state().log.at(-1);
    assert.deepEqual([you.kind, you.text, you.typed], ['you', "What's the weather tomorrow?", true]);
    await answer(t.env, ws, 'resp_t1');
    assert.equal(t.state().log.at(-1).kind, 'ai');
  });

  it('a question typed during a reply cancels it first (response.cancel and a flush), then sends the item', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    ws.serverSend(ga.responseCreated('resp_5'));
    ws.serverSend(ga.textDelta('resp_5', 'Talking…'));
    ws.serverSend(ga.audio('resp_5', 24000));
    await settle(t.env);
    const ctx = playbackContext(t.env);
    t.engine.ask('Stop, a new question');
    await settle(t.env);
    assert.deepEqual(sentTypes(ws).slice(-2), ['response.cancel', 'conversation.item.create']);
    assert.ok(ctx.__sources.every((s) => s.stopped), 'the reply\'s audio stops at once');
    ws.serverSend(ga.done('resp_5', 'cancelled', { type: 'cancelled', reason: 'client_cancelled' }));
    ws.serverSend(ga.typedAdded('typed_1', 'Stop, a new question'));
    await settle(t.env);
    assert.equal(sentTypes(ws).at(-1), 'response.create');
    assert.equal(t.state().log.find((e) => e.kind === 'ai').state, 'stopped');
  });

  it('a refused question (error with its event_id): its message, no response.create, the text is not taken', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    t.engine.ask('x'.repeat(10));
    await settle(t.env);
    ws.serverSend(ga.error('response_in_progress', 'A response is in progress; cancel it with response.cancel before adding a message', { event_id: 'evt_typed_1' }));
    await settle(t.env);
    assert.equal(sentTypes(ws).includes('response.create'), false);
    assert.deepEqual(t.state().typed, { busy: false, error: 'A response is in progress; cancel it with response.cancel before adding a message', sent: 0 });
    assert.equal(t.state().log.length, 0);
    t.engine.clearTyped();
    assert.equal(t.state().typed.error, '');
    assert.equal(t.engine.ask('   '), false, 'blank text is never sent');
    assert.equal(t.engine.ask('y'.repeat(8001)), false, 'over 8,000 characters is never sent');
  });

  it('typing while Talk is stopped starts a session with the microphone off and sends the question once it is ready', async () => {
    t = await engineSetup();
    assert.equal(t.env.withGesture(() => t.engine.ask('Hello?')), true);
    await settle(t.env);
    assert.equal(t.env.mediaDevices.calls.length, 0, 'the microphone stays off');
    assert.equal(t.state().mic, 'off');
    const ws = lastSocket(t.env);
    assert.equal(t.state().typed.busy, true);
    await accept(t.env, ws);
    assert.equal(sentTypes(ws).at(-1), 'conversation.item.create');
    assert.equal(sent(ws).at(-1).item.content[0].text, 'Hello?');
  });
});

describe('Talk session: keeping the conversation', () => {
  it('with the microphone off, a no-op session.update goes out after 30 s without any event, so the idle timeout never ends the conversation', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    const before = sent(ws).length;
    await settle(t.env, 25000);
    assert.equal(sent(ws).length, before, 'not yet');
    await settle(t.env, 15000);
    const keep = sent(ws).slice(before);
    assert.deepEqual(keep, [{ type: 'session.update', session: {} }]);
    for (let i = 0; i < 4; i += 1) await settle(t.env, 30000);
    const later = sent(ws).slice(before);
    assert.ok(later.length >= 4 && later.every((e) => e.type === 'session.update' && JSON.stringify(e.session) === '{}'), JSON.stringify(later));
    assert.ok(later.length <= 6, 'about one every 30 to 40 s');
  });

  it('while the microphone streams, no keep-alive is needed', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    for (let i = 0; i < 40; i += 1) {
      micFrame(t.env);
      await settle(t.env, 1000);
    }
    assert.equal(sent(ws).filter((e) => e.type === 'session.update' && JSON.stringify(e.session) === '{}').length, 0);
  });

  it('a dropped socket reconnects quietly: a fresh socket, the same microphone, and "Reconnected: the conversation started over." in the log', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    const track = t.env.mediaDevices.streams[0].getTracks()[0];
    ws.serverSend(ga.transcript('item_1', 'Hi'));
    await settle(t.env);
    ws.serverClose(1006, '');
    await settle(t.env);
    assert.equal(t.state().phase, 'reconnecting');
    assert.equal(t.state().status.label, 'Reconnecting…');
    assert.equal(t.state().error, null);
    assert.equal(t.env.sockets.length, 1, 'waits before trying again');
    await settle(t.env, 600);
    const ws2 = lastSocket(t.env);
    assert.notEqual(ws2, ws);
    await accept(t.env, ws2);
    assert.equal(t.state().phase, 'ready');
    assert.equal(track.readyState, 'live', 'the microphone stayed open');
    assert.equal(t.env.mediaDevices.calls.length, 1);
    assert.deepEqual(logOf(t.state()), [['you', 'Hi'], ['note', 'Reconnected: the conversation started over.']]);
    micFrame(t.env);
    micFrame(t.env);
    micFrame(t.env);
    await settle(t.env);
    assert.ok(appends(ws2).length >= 1, 'the microphone streams to the new socket');
  });

  it('the server\'s idle timeout (error, then close 4008) reconnects quietly too, with no error entry', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    ws.serverSend(ga.error('idle_timeout', 'Session idle timeout'));
    ws.serverClose(4008, 'Session idle timeout');
    await settle(t.env, 600);
    await accept(t.env, lastSocket(t.env));
    assert.deepEqual(logOf(t.state()), [['note', 'Reconnected: the conversation started over.']]);
  });

  it('when every reconnect fails, Talk stops and says the connection dropped, with Retry', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    ws.serverClose(1006, '');
    for (let i = 0; i < 6; i += 1) {
      await settle(t.env, 6000);
      const next = lastSocket(t.env);
      if (next.readyState !== 3) next.error();
    }
    await settle(t.env, 6000);
    const st = t.state();
    assert.equal(st.active, false);
    assert.equal(st.error.title, 'Talk stopped');
    assert.equal(st.error.message, 'The connection to the server dropped.');
    assert.equal(st.error.action, 'retry');
    assert.equal(t.env.sockets.length, 4, 'the first socket and three tries');
    assert.equal(t.mic.currentSession(), null, 'the microphone is released');
    assert.equal(t.audio.current(), null);
  });

  it('closes that a retry can\'t fix stop at once: a refused key (Settings), Realtime turned off, a socket that never opened', async () => {
    for (const [code, opened, message, action] of [
      [4001, true, /API key/, 'open-settings'],
      [4004, true, /turned off on this server \(OS_REALTIME_ENABLED\)/, ''],
      [1006, false, /didn't open/, 'retry'],
    ]) {
      t = await engineSetup();
      t.env.withGesture(() => t.engine.start({ mic: false }));
      await settle(t.env);
      const ws = lastSocket(t.env);
      if (opened) ws.open();
      ws.serverClose(code, code === 4004 ? 'Realtime API is disabled' : '');
      await settle(t.env, 10000);
      assert.equal(t.env.sockets.length, 1, `no retry for ${code}`);
      assert.equal(t.state().active, false);
      assert.match(t.state().error.message, message);
      assert.equal(t.state().error.action, action, String(code));
      t.engine.destroy();
      t.env.restore();
      t = null;
    }
  });

  // /v1/realtime accepts a refused socket, then closes it with 4001, 1008
  // or 4004 and the reason (src/routes/realtime.py), so it opens first. A reconnect refused that way stops
  // at once with why, never tries again (NO_RETRY), and offers what helps.
  it('a reconnect the server refuses (opened, then closed with its code and reason) stops at once with why: no more tries', async () => {
    for (const [code, reason, message, action] of [
      [4001, 'Invalid or missing API key', /API key/, 'open-settings'],
      [1008, 'Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it.', /^This page's address isn't allowed to connect\./, ''],
      [4004, 'Realtime API is disabled', /turned off on this server \(OS_REALTIME_ENABLED\)/, ''],
    ]) {
      t = await engineSetup();
      const ws = await startConnected(t, { mic: false });
      ws.serverClose(1006, '');
      await settle(t.env, 600);
      const again = lastSocket(t.env);
      assert.notEqual(again, ws, 'a drop reconnects');
      again.open();
      again.serverClose(code, reason);
      await settle(t.env, 10000);
      assert.equal(t.env.sockets.length, 2, `no retry after ${code}`);
      const st = t.state();
      assert.equal(st.active, false, String(code));
      assert.match(st.error.message, message, String(code));
      assert.equal(st.error.action, action, String(code));
      assert.match(st.error.detail, new RegExp(`\\(code ${code}\\)$`));
      t.engine.destroy();
      t.env.restore();
      t = null;
    }
  });

  it('New conversation: a fresh socket and an empty log, no reconnect note; the microphone stays on', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    ws.serverSend(ga.transcript('item_1', 'Remember the number 7.'));
    await settle(t.env);
    t.engine.newConversation();
    await settle(t.env);
    assert.equal(ws.readyState, 3, 'the old socket closed: the server forgets the conversation');
    assert.equal(t.state().log.length, 0);
    const ws2 = lastSocket(t.env);
    assert.notEqual(ws2, ws);
    await accept(t.env, ws2);
    assert.equal(t.state().log.length, 0, 'no "Reconnected" note');
    assert.equal(t.state().mic, 'on');
  });

  it('starting again after Stop keeps the old log and says the AI does not remember it', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    ws.serverSend(ga.typedAdded('x', 'ignored'));
    ws.serverSend(ga.transcript('item_1', 'Old question'));
    await settle(t.env);
    t.engine.stop();
    await settle(t.env);
    await startConnected(t, { mic: false });
    assert.deepEqual(logOf(t.state()), [['you', 'Old question'], ['note', "A new conversation started. The AI doesn't remember the one above."]]);
  });
});

// Delete chat. While Talk runs it is New conversation (above);
// while Talk is stopped, clearLog() empties the chat on screen and restoreLog() is its Undo. Nothing is sent or
// stored.
describe('Talk session: deleting the chat', () => {
  it('clearLog() empties the chat only while Talk is stopped and hands back what it held; restoreLog() puts it back while Talk is still stopped and the chat empty', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    ws.serverSend(ga.transcript('item_1', 'Remember the number 7.'));
    await answer(t.env, ws, 'resp_1', { text: 'I will remember 7.' });
    const held = logOf(t.state());
    assert.deepEqual(held, [['you', 'Remember the number 7.'], ['ai', 'I will remember 7.']]);
    assert.equal(t.engine.clearLog(), null, 'not while a conversation runs: Delete chat is New conversation then');
    assert.deepEqual(logOf(t.state()), held);
    t.engine.stop();
    await settle(t.env);
    const sockets = t.env.sockets.length;
    const calls = t.env.fetch.calls.length;
    const published = t.states.length;
    const removed = t.engine.clearLog();
    assert.deepEqual(removed.map((e) => [e.kind, e.text]), held, 'what was deleted, for Undo');
    assert.deepEqual(t.state().log, []);
    assert.equal(t.state().active, false, 'Talk stays stopped');
    assert.ok(t.states.length > published, 'the view is told');
    assert.deepEqual(t.states.at(-1).log, []);
    assert.equal(t.engine.clearLog(), null, 'an empty chat: nothing to delete');
    assert.equal(t.engine.restoreLog(removed), true);
    assert.deepEqual(logOf(t.state()), held, 'Undo');
    assert.equal(t.env.sockets.length, sockets, 'nothing connects');
    assert.equal(t.env.fetch.calls.length, calls, 'nothing is sent or stored');
    // Deleted, then started again: an empty chat with no "doesn't remember" note, and Undo no longer applies.
    const again = t.engine.clearLog();
    const ws2 = await startConnected(t, { mic: false });
    assert.deepEqual(t.state().log, []);
    assert.equal(t.engine.restoreLog(again), false, 'not into a running conversation');
    ws2.serverSend(ga.transcript('item_2', 'A new question'));
    await settle(t.env);
    t.engine.stop();
    await settle(t.env);
    assert.equal(t.engine.restoreLog(again), false, 'not over a chat that has entries again');
    assert.deepEqual(logOf(t.state()), [['you', 'A new question']]);
  });
});

describe('Talk session: choices, the microphone and the audio session', () => {
  it('changing the preset, the voice or the interrupt switch sends one session.update; a refused one reports its message and keeps the accepted choices', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    assert.deepEqual(t.state().accepted, { model: 'kokoro', voice: 'af_bella', preset: 'voice', assistant: '', interrupt: true, textOnly: false });
    assert.equal('assistant' in sent(ws).at(-1).session.assistant, false, 'no assistant chosen: none is sent');
    t.choices = { ...t.choices, preset: 'short', interrupt: false };
    t.engine.update();
    await settle(t.env);
    const up = sent(ws).at(-1);
    assert.equal(up.type, 'session.update');
    assert.deepEqual(up.session.assistant, { preset: 'short' });
    assert.equal(up.session.audio.input.turn_detection.interrupt_response, false);
    ws.serverSend(ga.updated({ preset: 'short' }));
    await settle(t.env);
    assert.deepEqual(t.state().accepted, { model: 'kokoro', voice: 'af_bella', preset: 'short', assistant: '', interrupt: false, textOnly: false });
    t.choices = { ...t.choices, voice: 'voice:cloned' };
    t.engine.update();
    await settle(t.env);
    ws.serverSend(ga.error('invalid_request', "Named voice 'cloned' needs a reference recording", { event_id: sent(ws).at(-1).event_id }));
    await settle(t.env);
    assert.equal(t.state().choiceError, "Named voice 'cloned' needs a reference recording");
    assert.equal(t.state().accepted.voice, 'af_bella');
    assert.equal(t.state().phase, 'ready', 'the session carries on');
  });

  it('an assistant chosen in Talk goes in session.update with the preset; one the server refuses is reported and the accepted one stays', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    t.choices = { ...t.choices, assistant: 'a-strata' };
    t.engine.update();
    await settle(t.env);
    const up = sent(ws).at(-1);
    assert.equal(up.type, 'session.update');
    assert.deepEqual(up.session.assistant, { preset: 'voice', assistant: 'a-strata' });
    ws.serverSend(ga.updated({ preset: 'voice' }));
    await settle(t.env);
    assert.equal(t.state().accepted.assistant, 'a-strata');
    t.choices = { ...t.choices, assistant: 'a-gone' };
    t.engine.update();
    await settle(t.env);
    ws.serverSend(ga.error('invalid_request', "Invalid AI assistant setting: assistant 'a-gone' not found", { event_id: sent(ws).at(-1).event_id }));
    await settle(t.env);
    assert.match(t.state().choiceError, /assistant 'a-gone' not found/);
    assert.equal(t.state().accepted.assistant, 'a-strata');
    assert.equal(t.state().phase, 'ready', 'the conversation carries on');
  });

  it('settings the server refuses before the session is ready stop it with the reason', async () => {
    t = await engineSetup({ choices: { model: 'kokoro', voice: 'voice:nope', preset: 'gone', interrupt: true } });
    t.env.withGesture(() => t.engine.start({ mic: false }));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    ws.serverSend(ga.created());
    await settle(t.env);
    ws.serverSend(ga.error('invalid_request', "Invalid AI assistant setting: preset 'gone' not found", { event_id: sent(ws)[0].event_id }));
    await settle(t.env);
    assert.equal(t.state().active, false);
    assert.equal(t.state().error.message, "The session's settings weren't accepted.");
    assert.match(t.state().error.detail, /preset 'gone' not found/);
    assert.equal(ws.readyState, 3);
  });

  it('no voice detection on the server: the microphone turns off and says why; typing still works', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true, updated: { turnDetection: null } });
    ws.serverSend(ga.error('vad_unavailable', 'Server VAD could not be loaded, so turn detection is off; commit audio manually', { type: 'server_error' }));
    await settle(t.env);
    assert.equal(t.state().mic, 'unavailable');
    assert.match(t.state().micProblem.message, /can't tell when you stop talking/);
    assert.equal(t.mic.currentSession(), null, 'the capture is released');
    assert.equal(t.engine.setMic(true), false);
    assert.equal(t.engine.ask('Typed instead'), true);
    await settle(t.env);
    assert.equal(sentTypes(ws).at(-1), 'conversation.item.create');
    t.choices = { ...t.choices, preset: 'short' };
    t.engine.update();
    await settle(t.env);
    assert.equal('turn_detection' in sent(ws).at(-1).session.audio.input, false, 'no new attempt at voice detection on each change');
  });

  it('a refused microphone says so in plain words; the session still connects for typing', async () => {
    t = await engineSetup({ env: { mediaDevices: { policy: 'NotAllowedError' } } });
    const ws = await startConnected(t, { mic: true });
    assert.equal(t.state().mic, 'off');
    assert.equal(t.state().micProblem.title, 'Microphone problem');
    assert.match(t.state().micProblem.message, /blocked.*You can still type your questions\./);
    assert.equal(t.state().phase, 'ready');
    t.engine.ask('Still here?');
    await settle(t.env);
    assert.equal(sentTypes(ws).at(-1), 'conversation.item.create');
  });

  it('turning the microphone off mid-turn ends the turn with silence, then nothing more is sent; on again opens it inside the click', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    ws.serverSend(ga.speechStarted());
    await settle(t.env);
    assert.equal(t.engine.setMic(false), true);
    await settle(t.env);
    const last = appends(ws).at(-1);
    assert.equal(samplesIn(last.audio), 24000 * 0.8, '800 ms of silence, longer than the server\'s 500 ms');
    assert.ok(atob(last.audio).split('').every((c) => c.charCodeAt(0) === 0));
    assert.equal(t.mic.currentSession(), null);
    assert.equal(t.env.mediaDevices.streams[0].getTracks()[0].readyState, 'ended');
    assert.equal(t.state().mic, 'off');
    assert.equal(t.state().status.label, 'Connected');
    const count = appends(ws).length;
    await settle(t.env, 1000);
    assert.equal(appends(ws).length, count);
    assert.equal(t.env.withGesture(() => t.engine.setMic(true)), true);
    assert.equal(t.state().mic, 'starting');
    await settle(t.env);
    assert.equal(t.state().mic, 'on');
    assert.equal(t.audio.current().id, 'talk', 'its own microphone never pauses or displaces Talk');
    assert.equal(t.state().paused, false);
  });

  it('another player starting stops Talk: "Stopped: another player started", the microphone and socket released', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    t.audio.claim({ id: 'live', owner: 'live', kind: 'live', title: 'Live', homeRoute: 'speak/live', controls: { play() {}, pause() {}, toggle() {}, stop() {} }, getState: () => ({}) });
    await settle(t.env);
    assert.equal(t.state().active, false);
    assert.equal(t.state().status.label, 'Stopped: another player started');
    assert.equal(ws.readyState, 3);
    assert.equal(t.mic.currentSession(), null);
    assert.equal(t.app.get().recording, null);
  });

  it('Pause from the mini-player or the lock screen holds the reply\'s sound; Resume brings it back', async () => {
    t = await engineSetup();
    await startConnected(t, { mic: false });
    const ctx = playbackContext(t.env);
    t.audio.current().controls.pause();
    await settle(t.env, 300);
    assert.equal(t.state().paused, true);
    assert.equal(ctx.state, 'suspended');
    assert.equal(t.state().status.label, 'Paused');
    t.env.withGesture(() => t.engine.resume());
    await settle(t.env, 300);
    assert.equal(t.state().paused, false);
    assert.equal(ctx.state, 'running');
  });

  it('Stop reply cancels the running answer and flushes its audio; the session goes on listening', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    ws.serverSend(ga.responseCreated('resp_6'));
    ws.serverSend(ga.textDelta('resp_6', 'Blah blah.'));
    ws.serverSend(ga.audio('resp_6', 24000));
    await settle(t.env);
    t.engine.cancelReply();
    await settle(t.env);
    assert.equal(sentTypes(ws).at(-1), 'response.cancel');
    assert.ok(playbackContext(t.env).__sources.every((s) => s.stopped));
    assert.equal(t.state().phase, 'ready');
  });
});

describe('Talk session: nothing is left behind', () => {
  it('five starts and stops, with a turn each: no timer, listener, socket, microphone track, audio context or object URL is left', async () => {
    t = await engineSetup();
    const win = t.env.window;
    const doc = t.env.document;
    const before = {
      pagehide: win.__listenerCount('pagehide'), beforeunload: win.__listenerCount('beforeunload'),
      pageshow: win.__listenerCount('pageshow'), visibility: doc.__listenerCount('visibilitychange'),
      devicechange: t.env.mediaDevices.__listenerCount('devicechange'),
    };
    const baseTimers = pendingTimers(t.env).length;
    for (let i = 0; i < 5; i += 1) {
      const ws = await startConnected(t, { mic: i % 2 === 0 });
      if (i % 2 === 0) micFrame(t.env);
      await answer(t.env, ws, `resp_${i}`, { fillers: ['Hmm.'] });
      t.engine.ask(`Question ${i}`);
      await settle(t.env);
      t.env.withGesture(() => t.engine.stop());
      await settle(t.env, 1000);
      assert.equal(ws.readyState, 3, `socket ${i} closed`);
    }
    await settle(t.env, 15000);
    assert.equal(t.state().active, false);
    assert.equal(pendingTimers(t.env).length, baseTimers, JSON.stringify(pendingTimers(t.env)));
    assert.ok(t.env.sockets.every((ws) => ws.readyState === 3), 'every socket closed');
    for (const stream of t.env.mediaDevices.streams) for (const track of stream.getTracks()) assert.equal(track.readyState, 'ended');
    assert.ok(t.env.audioContexts.every((ctx) => ctx.state === 'closed'), 'every AudioContext closed');
    assert.equal(t.env.createdUrls.length, 0, 'no object URLs');
    assert.deepEqual({
      pagehide: win.__listenerCount('pagehide'), beforeunload: win.__listenerCount('beforeunload'),
      pageshow: win.__listenerCount('pageshow'), visibility: doc.__listenerCount('visibilitychange'),
      devicechange: t.env.mediaDevices.__listenerCount('devicechange'),
    }, before);
    assert.equal(t.audio.current(), null);
    assert.equal(t.app.get().recording, null);
    assert.equal(t.mic.owner(), null);
    assert.ok(t.state().log.length <= 200, 'the log is bounded');
  });

  it('the log keeps the newest 200 entries', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: false });
    for (let i = 0; i < 230; i += 1) ws.serverSend(ga.transcript(`item_${i}`, `Turn ${i}`));
    await settle(t.env);
    const log = t.state().log;
    assert.equal(log.length, 200);
    assert.equal(log[0].text, 'Turn 30');
    assert.equal(log.at(-1).text, 'Turn 229');
  });

  it('leaving the page closes the socket and lets everything go', async () => {
    t = await engineSetup();
    const ws = await startConnected(t, { mic: true });
    t.env.pageHide({ persisted: false });
    await settle(t.env);
    assert.equal(ws.readyState, 3);
    assert.equal(t.state().active, false);
    assert.equal(t.mic.currentSession(), null);
    assert.ok(PRESETS.length);
  });
});
