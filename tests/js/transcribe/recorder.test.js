// Transcribe: live transcription from the microphone, and the plain copy for its close codes.
// The real recorder runs against the fake browser: FakeMediaDevices, FakeAudioContext with its AudioWorklet,
// FakeSocket for /v1/audio/stream and the fake fetch for the model check and the ticket.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountView, settle, startRecording, begin, said, lastSocket, micFrame, announced, answerConfirm, load, json, STT, MODELS, FakeEvent,
  trackSpoken, countWrites, chooseFile, audioFile, verbose, sameNode, noNode,
} from './helpers.js';
import { FakeSocket, delay } from '../harness/fakes.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let t;
afterEach(() => t?.env?.restore());

const phase = () => t.$('tr-phase').textContent;
const recordName = () => t.$('tr-record').getAttribute('aria-label');
const recError = () => (t.$('tr-rec-error').hidden ? '' : t.$('tr-rec-error').textContent.replace(/\s+/g, ' ').trim());
const help = () => (t.$('tr-mic-help').hidden ? '' : t.$('tr-mic-help').textContent.replace(/\s+/g, ' ').trim());
const tracks = () => t.env.mediaDevices.streams.flatMap((s) => s.getTracks());

describe('transcribe recorder: starting a session', () => {
  it('[TR-06] the socket carries the real AudioContext rate, interim_results and the model; frames are PCM16', async () => {
    t = await mountView({ env: { audio: { sampleRate: 44100 } } });
    const mic = await load('core/mic.js');
    t.env.change(t.$('tr-mic'), 'mic-b');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    // The saved device is asked for (an opaque id), for this owner.
    assert.deepEqual(t.env.mediaDevices.calls[0].audio.deviceId, { exact: 'mic-b' });
    assert.equal(t.env.mediaDevices.calls[0].audio.echoCancellation, true);
    assert.equal(mic.owner(), 'transcribe');
    const ws = lastSocket(t.env);
    const url = new URL(ws.url);
    assert.equal(url.protocol, 'ws:');
    assert.equal(url.pathname, '/v1/audio/stream');
    assert.equal(url.searchParams.get('sample_rate'), '44100');
    assert.equal(url.searchParams.get('interim_results'), 'true');
    assert.equal(url.searchParams.get('model'), STT);
    assert.equal(url.searchParams.has('api_key'), false);
    assert.equal(url.searchParams.has('ticket'), false, 'no key: no ticket is requested');
    // Audio captured while the socket is still opening is kept and sent first, so no words are lost.
    micFrame(t.env, new Float32Array(2048).fill(0.5));
    assert.equal(ws.sent.length, 0);
    ws.open();
    await settle(t.env);
    micFrame(t.env, new Float32Array(2048).fill(-0.25));
    assert.equal(ws.sent.length, 2);
    for (const frame of ws.sent) assert.ok(frame instanceof ArrayBuffer && frame.byteLength === 4096, 'binary PCM16 LE frames');
    assert.equal(new Int16Array(ws.sent[0])[0], Math.round(0.5 * 32767));
    assert.ok(new Int16Array(ws.sent[1])[0] < -8000);
    assert.equal(ws.binaryType, 'arraybuffer');
  });

  it('[TR-06] with an API key every session gets its own one-time ticket; the key never goes in the URL', async () => {
    t = await mountView({ env: { seed: { 'open-speech-api-key': 'sk-secret-key' } } });
    const auth = await load('core/auth.js');
    auth.init();
    const ws1 = await startRecording(t);
    t.env.click(t.$('tr-record'));   // Stop
    ws1.serverSend({ type: 'session.end' });
    await settle(t.env);
    const ws2 = await startRecording(t);
    const tickets = t.env.fetch.callsTo('POST /api/auth/ws-ticket');
    assert.equal(tickets.length, 2);
    assert.equal(tickets[0].headers.authorization, 'Bearer sk-secret-key');
    const q1 = new URL(ws1.url).searchParams;
    const q2 = new URL(ws2.url).searchParams;
    assert.ok(q1.get('ticket') && q2.get('ticket') && q1.get('ticket') !== q2.get('ticket'));
    assert.ok(!ws1.url.includes('sk-secret-key') && !ws2.url.includes('sk-secret-key'));
    assert.equal(q1.has('api_key'), false);
  });

  it('[TR-06] the model is made ready (downloaded, then loaded) before the socket opens, and the chip-job shows', async () => {
    const models = MODELS.map((m) => (m.id === STT ? { ...m, state: 'available' } : m));
    let state = 'available';
    t = await mountView({
      models,
      routes: {
        'GET /api/models/:id/status': () => json(200, { state }),
        'POST /api/models/:id/download': () => delay(3000, () => { state = 'downloaded'; return json(200, { status: 'downloaded' }); }),
        'POST /api/models/:id/load': () => { state = 'loaded'; return json(200, { status: 'loaded' }); },
        'GET /api/models': () => json(200, { models: models.map((m) => (m.id === STT ? { ...m, state } : m)), default_stt_model: STT }),
      },
    });
    let callsAtSocket = -1;
    FakeSocket.onCreate = () => { callsAtSocket = t.env.fetch.calls.length; };
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    // The microphone is already open (inside the click); the model downloads first.
    assert.equal(t.env.sockets.length, 0);
    assert.equal(phase(), 'Downloading whisper base…');
    assert.equal(t.$('tr-record').getAttribute('aria-busy'), 'true');
    await settle(t.env, 2999);
    assert.equal(t.env.sockets.length, 0, 'no socket while the model downloads');
    await settle(t.env, 1);
    await settle(t.env);
    const ws = lastSocket(t.env);
    assert.ok(ws, 'the socket opened once the model was ready');
    // What ensureReady did before the socket (the order of core's own checks is core's business): the model's
    // status was checked, and it was downloaded, then loaded.
    const before = t.env.fetch.calls.slice(0, callsAtSocket).map((c) => `${c.method} ${c.path.split('/').pop()}`);
    assert.ok(before.includes(`GET status`), JSON.stringify(before));
    assert.ok(before.includes('POST download') && before.includes('POST load'), JSON.stringify(before));
    assert.ok(before.indexOf('POST download') < before.indexOf('POST load'), JSON.stringify(before));
    assert.equal(new URL(ws.url).searchParams.get('model'), STT);
  });

  it('[TR-06] record, stop, record again: the model just worked, so the second start checks nothing', async () => {
    t = await mountView();
    t.app.set({ models: { ...t.app.get().models, fetchedAt: 0 } });   // an old inventory: no fast path from it
    const status = () => t.env.fetch.callsTo('GET /api/models/:id/status').length;
    const ws1 = await startRecording(t);
    assert.equal(status(), 1, 'the first start checks the status once');
    await settle(t.env, 45000);   // a long session: the check is now far behind
    t.env.click(t.$('tr-record'));
    ws1.serverSend({ type: 'session.end' });
    await settle(t.env);
    await startRecording(t);
    assert.equal(status(), 1, 'no second check: the session that just ended proved the model works');
  });

  it('[TR-06] a model that failed before session.begin is checked again on the next start (no stale fast path)', async () => {
    t = await mountView();
    t.app.set({ models: { ...t.app.get().models, fetchedAt: 0 } });
    const status = () => t.env.fetch.callsTo('GET /api/models/:id/status').length;
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.equal(status(), 1);
    const ws = lastSocket(t.env);
    ws.open();
    ws.serverSend({ type: 'error', message: 'Failed to load model: out of memory' });
    await settle(t.env);
    assert.equal(recordName(), 'Record');
    // Seconds later (the status check above would still count as fresh) Record asks the server again.
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.equal(status(), 2);
  });

  it('[TR-06] refused while Voices records: "Stop the Voices recording first", and no socket', async () => {
    t = await mountView();
    const mic = await load('core/mic.js');
    const voices = await mic.open({ owner: 'voices' });
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.equal(t.env.sockets.length, 0);
    assert.match(recError(), /Microphone problem/);
    assert.match(recError(), /Stop the Voices recording first/);
    assert.equal(mic.owner(), 'voices', "Voices keeps the microphone and its own state");
    assert.equal(recordName(), 'Record');
    voices.stop();
  });

  it('[BUG-08] a double click never opens two microphones or two sockets: Record is busy from the first await', async () => {
    t = await mountView();
    const button = t.$('tr-record');
    button.focus();
    t.env.click(button);
    // Synchronously after the first click: busy and disabled.
    assert.equal(button.getAttribute('aria-busy'), 'true');
    assert.equal(button.disabled, true);
    assert.equal(phase(), 'Starting…');
    t.env.click(button);
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    assert.equal(t.env.mediaDevices.calls.length, 1);
    assert.equal(t.env.sockets.length, 1);
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env);
    // Accepted, but the server is still loading: busy until it begins the session, and still one of each.
    assert.equal(button.getAttribute('aria-busy'), 'true');
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    assert.equal(t.env.mediaDevices.calls.length, 1);
    assert.equal(t.env.sockets.length, 1);
    begin(ws);
    await settle(t.env);
    assert.equal(button.disabled, false);
    assert.equal(button.hasAttribute('aria-busy'), false);
    sameNode(t.doc.activeElement, button, 'focus comes back to the button it was on');
    // The recorder itself refuses a second start.
    const env2 = t.env;
    const { createRecorder } = await load('views/transcribe/recorder.js');
    const r = createRecorder();
    const first = r.start({ model: STT });
    const second = await r.start({ model: STT });
    assert.equal(second, false);
    r.cancel();
    await first;
    await settle(env2);
  });

  it('[TR-12] recording starts when the server begins the session: until session.begin Record is busy and nothing is said', async () => {
    t = await mountView();
    const spoken = trackSpoken(t);
    const { app } = await load('core/store.js');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    micFrame(t.env);
    ws.open();
    await settle(t.env, 600);   // anything announced at the open would be out by now
    // Accepted, while the server loads its model and VAD: the audio already goes out, but no recording yet.
    assert.equal(ws.sent.length, 1, 'the kept audio is sent on open');
    assert.equal(phase(), 'Starting…');
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-record').getAttribute('aria-busy'), 'true');
    assert.equal(t.$('tr-timer').hidden, true);
    assert.equal(t.$('tr-rec-dot').hidden, true);
    assert.deepEqual(spoken, []);
    assert.equal(t.ctl.state().transcript.source, '', 'the transcript on screen is not replaced yet');
    assert.equal(app.get().recording.owner, 'transcribe', 'the microphone is open: the chip says so');
    begin(ws);
    await settle(t.env);
    assert.equal(phase(), 'Recording');
    assert.equal(recordName(), 'Stop recording');
    assert.equal(t.$('tr-record').hasAttribute('aria-busy'), false);
    assert.equal(t.$('tr-timer').hidden, false);
    assert.equal(t.ctl.state().transcript.source, 'mic');
    assert.deepEqual(spoken, ['Recording started']);
    assert.equal(app.get().recording.startedAt, t.ctl.state().rec.startedAt, 'the chip counts from the same moment as the timer');
  });
});

describe('transcribe recorder: recording', () => {
  it('[TR-12] Record turns into Stop: the name swaps, the dot and the aria-hidden timer show, start and stop are announced', async () => {
    t = await mountView();
    const toasts = () => t.$('app-toasts-polite').textContent;
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-rec-dot').hidden, true);
    const ws = await startRecording(t);
    assert.equal(recordName(), 'Stop recording');
    assert.equal(phase(), 'Recording');
    assert.ok(t.$('tr-record').classList.contains('is-recording'));
    assert.equal(t.$('tr-rec-dot').hidden, false);
    assert.equal(t.$('tr-timer').hidden, false);
    assert.equal(t.$('tr-timer').getAttribute('aria-hidden'), 'true', 'the counter never ticks in a live region');
    assert.equal(t.$('tr-rec-actions').hidden, false, 'Cancel shows');
    assert.equal(t.$('tr-mic-field').hidden, true);
    assert.equal(announced(t), 'Recording started');
    assert.equal(toasts(), '', 'no toast: the button, the dot and the chip say it');
    const { app } = await load('core/store.js');
    assert.equal(app.get().recording.owner, 'transcribe', 'the status chip shows Recording');
    await settle(t.env, 3000);
    assert.equal(t.$('tr-timer').textContent, '0:03');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    ws.serverSend({ type: 'session.end' });
    await settle(t.env, 600);
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-rec-dot').hidden, true);
    assert.equal(announced(t), 'Recording stopped');
  });

  it('[REV-12] the record button never carries aria-pressed: only its name changes', async () => {
    t = await mountView();
    const button = t.$('tr-record');
    const seen = [];
    const look = () => seen.push([button.getAttribute('aria-label'), button.hasAttribute('aria-pressed')]);
    look();
    t.env.click(button);
    look();
    await settle(t.env);
    look();
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env);
    look();
    begin(ws);
    await settle(t.env);
    look();
    t.env.click(button);
    look();
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    look();
    assert.ok(seen.every(([, pressed]) => pressed === false), JSON.stringify(seen));
    assert.deepEqual([...new Set(seen.map(([name]) => name))].sort(), ['Record', 'Stop recording']);
  });

  it('[TR-09] the VAD pill says Speech or Silence in words, and only when the server runs VAD', async () => {
    t = await mountView();
    const ws = await startRecording(t, { begin: { vad_enabled: true } });
    const pill = t.$('tr-vad');
    assert.equal(pill.hidden, false);
    assert.equal(t.$('tr-vad-label').textContent, 'Silence');
    ws.serverSend({ type: 'vad', state: 'speech_start' });
    await settle(t.env);
    assert.equal(t.$('tr-vad-label').textContent, 'Speech');
    assert.ok(pill.classList.contains('pill--ok'));
    ws.serverSend({ type: 'vad', state: 'speech_end' });
    await settle(t.env);
    assert.equal(t.$('tr-vad-label').textContent, 'Silence');
    t.ctl.destroy();
    t.env.restore();
    t = await mountView();
    await startRecording(t, { begin: { vad_enabled: false } });
    assert.equal(t.$('tr-vad').hidden, true, 'no VAD on the server: no pill that would always say Silence');
  });

  it('[TR-09] the level meter follows the microphone and says the level in words', async () => {
    t = await mountView();
    await startRecording(t);
    micFrame(t.env, new Float32Array(2048).fill(0.1));
    await settle(t.env, 300);
    assert.equal(t.$('tr-meter-label').textContent, 'Good level');
    assert.equal(t.$('tr-meter').getAttribute('aria-valuetext'), 'Good level');
    micFrame(t.env, new Float32Array(2048).fill(1));
    await settle(t.env, 300);
    assert.equal(t.$('tr-meter-label').textContent, 'Clipping');
  });
});

describe('transcribe recorder: stopping', () => {
  it('[TR-10] Stop is graceful: {type:"stop"}, capture ends at once, the socket stays up to 4 s for the last segments', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    micFrame(t.env);
    const { app } = await load('core/store.js');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.deepEqual(ws.sentJson, [{ type: 'stop' }]);
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'), 'the microphone is released at once');
    assert.equal(app.get().recording, null);
    assert.equal(phase(), 'Finishing…');
    assert.equal(t.$('tr-record').disabled, true);
    assert.equal(ws.readyState, FakeSocket.OPEN);
    // The last words still arrive and are kept.
    said(ws, 'the last words', { start: 0.2, end: 1.4 });
    await settle(t.env, 3999);
    assert.equal(ws.readyState, FakeSocket.OPEN);
    assert.ok(t.$('tr-log').textContent.includes('the last words'));
    await settle(t.env, 1);
    await settle(t.env);
    assert.equal(ws.readyState, FakeSocket.CLOSED);
    assert.equal(ws.closeCode, 1000);
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-record').disabled, false);
  });

  it('[TR-10] session.end finishes the stop at once, and one timer per socket: an old timer never closes the next session', async () => {
    t = await mountView();
    const ws1 = await startRecording(t);
    t.env.click(t.$('tr-record'));
    await settle(t.env, 500);
    ws1.serverSend({ type: 'session.end', reason: 'client_stop' });
    await settle(t.env);
    assert.equal(ws1.readyState, FakeSocket.CLOSED);
    assert.equal(recordName(), 'Record');
    const ws2 = await startRecording(t);
    await settle(t.env, 5000);
    assert.equal(ws2.readyState, FakeSocket.OPEN, 'the first session left no timer behind');
    assert.equal(recordName(), 'Stop recording');
  });
});

describe('transcribe recorder: the clock starts at the first sound the server gets', () => {
  // Record with the model not downloaded: a one-minute download runs while the microphone is open (frames of
  // 2048 samples at 48 kHz, about 24 a second), then the socket opens and the server begins.
  async function recordAfterDownload(seconds = 60) {
    const models = MODELS.map((m) => (m.id === STT ? { ...m, state: 'available' } : m));
    let state = 'available';
    t = await mountView({
      models,
      routes: {
        'GET /api/models/:id/status': () => json(200, { state }),
        'POST /api/models/:id/download': () => delay(seconds * 1000, () => { state = 'downloaded'; return json(200, { status: 'downloaded' }); }),
        'POST /api/models/:id/load': () => { state = 'loaded'; return json(200, { status: 'loaded' }); },
        'GET /api/models': () => json(200, { models: models.map((m) => (m.id === STT ? { ...m, state } : m)), default_stt_model: STT }),
      },
    });
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    for (let s = 0; s < seconds + 10 && !t.env.sockets.length; s += 1) {
      for (let i = 0; i < 24; i += 1) micFrame(t.env);
      await settle(t.env, 1000);
    }
    const ws = lastSocket(t.env);
    assert.ok(ws, 'the socket opens once the model is ready');
    ws.open();
    begin(ws);
    await settle(t.env);
    return ws;
  }

  it('[TR-12] a model download before the stream never counts: the timer and the 30 s discard confirm start with the kept audio', async () => {
    const ws = await recordAfterDownload(60);
    // The server got the kept audio, 20 s at most (whole 2048-sample frames: 19.968 s), not the minute.
    const sent = ws.sent.filter((d) => d instanceof ArrayBuffer).reduce((n, b) => n + b.byteLength / 2, 0) / 48000;
    assert.ok(sent > 19.9 && sent <= 20, String(sent));
    assert.equal(t.$('tr-timer').textContent, '0:19');
    await settle(t.env, 5000);
    assert.equal(t.$('tr-timer').textContent, '0:24', 'not 1:05');
    const { app } = await load('core/store.js');
    assert.equal(app.get().recording.startedAt, t.ctl.state().rec.startedAt, 'the status chip counts the same');
    // 25 s of recording: Cancel needs no confirm (it would have asked after the download alone).
    t.env.click(t.$('tr-cancel'));
    await settle(t.env);
    assert.equal(t.$('app-confirm').open, false);
    assert.equal(ws.readyState, FakeSocket.CLOSED);
    assert.equal(recordName(), 'Record');
  });

  it('[TR-12] "Recorded m:ss" and the segment times count the same audio as the timer', async () => {
    const ws = await recordAfterDownload(60);
    await settle(t.env, 5000);
    // The server's times start at its first sample (the kept audio): a word 21 s in reads 0:21 on the timer too.
    said(ws, 'Words after the download.', { start: 21, end: 23.5 });
    await settle(t.env);
    assert.equal(t.$('tr-log').querySelector('.tr-ts').textContent, '0:21');
    t.env.click(t.$('tr-record'));
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.equal(t.$('tr-meta').textContent, 'Recorded 0:24 · whisper base');
  });
});

describe('transcribe recorder: errors and close codes', () => {
  it('[TR-11] the socket closing while recording: "The microphone stream disconnected", capture stops, back to Record', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    const { app } = await load('core/store.js');
    ws.serverClose(1008, 'Invalid sample_rate: must be 8000-192000');
    await settle(t.env);
    assert.match(recError(), /The microphone stream disconnected/);
    assert.match(recError(), /The microphone's sample rate wasn't accepted\./);
    assert.match(t.$('tr-rec-error').querySelector('pre').textContent, /Invalid sample_rate/);
    assert.equal(t.$('tr-rec-error').getAttribute('role'), 'alert');
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'));
    assert.equal(app.get().recording, null);
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-record').disabled, false);
  });

  // /v1/audio/stream sends session.end without a stop only after a failure (reason
  // "disconnect"), then closes 1011 Streaming session failed. Both orders, and a bare 1011, are a disconnect; only
  // the user's Stop, or the server's own 1000 (or no close at all within the grace), ends a recording quietly.
  it('[TR-11] the server ends a recording nobody stopped (session.end, then 1011): "The microphone stream disconnected", never a quiet stop', async () => {
    const { STOP_GRACE_MS } = await load('views/transcribe/recorder.js');
    const failure = { type: 'session.end', reason: 'disconnect', transcriptions: 1, errors: 0 };
    // In one burst.
    t = await mountView();
    let ws = await startRecording(t);
    said(ws, 'Words before the failure.', { start: 0, end: 1 });
    await settle(t.env);
    ws.serverSend(failure);
    ws.serverClose(1011, 'Streaming session failed');
    await settle(t.env);
    assert.match(recError(), /The microphone stream disconnected/, 'session.end then 1011 is a failure, not a Stop');
    assert.match(t.$('tr-rec-error').querySelector('pre').textContent, /Streaming session failed/);
    assert.equal(t.$('tr-rec-error').getAttribute('role'), 'alert');
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'), 'the microphone is released');
    assert.equal(recordName(), 'Record');
    assert.ok(t.$('tr-log').textContent.includes('Words before the failure.'), 'the words so far are kept');
    t.ctl.destroy();
    t.env.restore();
    // The Close frame a moment after session.end (separate network reads): the microphone is let go at once, the
    // view says it is finishing, and the 1011 still decides.
    t = await mountView();
    ws = await startRecording(t);
    ws.serverSend(failure);
    await settle(t.env, 20);
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'), 'the server stopped reading: capture stops');
    assert.equal(phase(), 'Finishing…');
    assert.equal(recError(), '');
    ws.serverClose(1011, 'Streaming session failed');
    await settle(t.env);
    assert.match(recError(), /The microphone stream disconnected/);
    assert.equal(recordName(), 'Record');
    t.ctl.destroy();
    t.env.restore();
    // A bare 1011 (no session.end) stays a disconnect.
    t = await mountView();
    ws = await startRecording(t);
    ws.serverClose(1011, 'Streaming session failed');
    await settle(t.env);
    assert.match(recError(), /The microphone stream disconnected/);
    t.ctl.destroy();
    t.env.restore();
    // session.end then the server's own 1000: a quiet end, "Recording stopped".
    t = await mountView();
    ws = await startRecording(t);
    said(ws, 'Kept words.', { start: 0, end: 1 });
    ws.serverSend({ type: 'session.end', transcriptions: 1, errors: 0 });
    ws.serverClose(1000, 'Stream ended');
    await settle(t.env, 600);
    assert.equal(recError(), '');
    assert.equal(recordName(), 'Record');
    assert.equal(announced(t), 'Recording stopped');
    t.ctl.destroy();
    t.env.restore();
    // session.end and no close at all: it finishes quietly once the grace has passed, and closes the socket itself.
    t = await mountView();
    ws = await startRecording(t);
    ws.serverSend({ type: 'session.end', transcriptions: 0, errors: 0 });
    await settle(t.env, STOP_GRACE_MS - 1);
    assert.equal(phase(), 'Finishing…');
    assert.equal(ws.readyState, FakeSocket.OPEN);
    await settle(t.env, 1);
    await settle(t.env);
    assert.equal(ws.readyState, FakeSocket.CLOSED);
    assert.equal(ws.closeCode, 1000);
    assert.equal(recError(), '');
    assert.equal(recordName(), 'Record');
  });

  it('[TR-10] Stop answered with session.end {client_stop} and 1000 still ends quietly: no banner', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'Words before Stop.', { start: 0, end: 1 });
    await settle(t.env);
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.deepEqual(ws.sentJson, [{ type: 'stop' }]);
    ws.serverSend({ type: 'session.end', reason: 'client_stop', transcriptions: 1, errors: 0 });
    ws.serverClose(1000, 'Stream ended');
    await settle(t.env);
    assert.equal(recError(), '');
    assert.equal(recordName(), 'Record');
    assert.ok(t.$('tr-log').textContent.includes('Words before Stop.'));
  });

  it('[TR-11] a socket error (no close frame) says the connection dropped; Retry starts a new session', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.error();
    await settle(t.env);
    assert.match(recError(), /The connection to the server dropped\./);
    const retry = t.$('tr-rec-error').querySelector('[data-action="record-retry"]');
    assert.ok(retry, 'Retry');
    assert.equal(retry.textContent.trim(), 'Retry');
    t.env.click(retry);
    await settle(t.env);
    assert.equal(t.env.sockets.length, 2);
    assert.equal(t.$('tr-rec-error').hidden, true);
  });

  it('[TR-11] events from an earlier session\'s socket are ignored', async () => {
    t = await mountView();
    const ws1 = await startRecording(t);
    t.env.click(t.$('tr-cancel'));
    await settle(t.env);
    const ws2 = await startRecording(t);
    said(ws1, 'a ghost from the old socket');
    ws1.serverSend({ type: 'vad', state: 'speech_start' });
    ws1.serverClose(1013, 'Too many concurrent streams');
    await settle(t.env);
    assert.equal(t.$('tr-log').textContent.includes('ghost'), false);
    assert.equal(t.$('tr-rec-error').hidden, true);
    assert.equal(recordName(), 'Stop recording');
    said(ws2, 'the real words');
    await settle(t.env);
    assert.ok(t.$('tr-log').textContent.includes('the real words'));
  });

  it('[BUG-02] a server error before session.begin ends the session with its message (model or VAD failed)', async () => {
    t = await mountView();
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    ws.serverSend({ type: 'error', message: 'Failed to load VAD model: no network' });
    await settle(t.env);
    // It says what happened, in the same words as any other start or file that failed, and offers Retry.
    const banner = t.$('tr-rec-error');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Failed to load VAD model: no network');
    assert.equal(banner.querySelector('[data-action="record-retry"]').textContent.trim(), 'Retry');
    assert.equal(recordName(), 'Record');
    assert.ok(ws.readyState >= FakeSocket.CLOSING, 'the client closed it');
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'));
  });

  it('[BUG-02] an error after session.begin is shown, and recording goes on', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.serverSend({ type: 'error', message: 'Transcription failed: CUDA out of memory' });
    await settle(t.env);
    // Not a generic "Transcription error": the title says what was lost; the server's message says why.
    const banner = t.$('tr-rec-error');
    assert.equal(banner.querySelector('.banner__title').textContent, "Part of the recording wasn't transcribed");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Transcription failed: CUDA out of memory');
    assert.equal(recordName(), 'Stop recording');
    said(ws, 'still listening');
    await settle(t.env);
    assert.ok(t.$('tr-log').textContent.includes('still listening'));
  });

  it('[BUG-02] a notice (a server error while the recording goes on) offers only Dismiss: no banner action stops the recording', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.serverSend({ type: 'error', message: 'Transcription failed: CUDA out of memory' });
    await settle(t.env);
    const banner = t.$('tr-rec-error');
    assert.match(recError(), /Transcription failed: CUDA out of memory/);
    // Retry beside a running recording could only stop it, so it is not offered.
    assert.deepEqual(banner.querySelectorAll('[data-action]').map((b) => b.getAttribute('data-action')), ['dismiss-rec-error']);
    const dismiss = banner.querySelector('[data-action="dismiss-rec-error"]');
    dismiss.focus();
    t.env.key('Enter', {}, dismiss);
    await settle(t.env);
    assert.equal(banner.hidden, true);
    assert.deepEqual(ws.sentJson, [], 'no {type:"stop"}: the recording goes on');
    assert.equal(ws.readyState, FakeSocket.OPEN);
    assert.equal(recordName(), 'Stop recording');
    sameNode(t.doc.activeElement, t.$('tr-record'), 'focus goes to Stop, never to <body>');
    said(ws, 'still going');
    await settle(t.env);
    assert.ok(t.$('tr-log').textContent.includes('still going'));
    // Stopped later, the dismissed notice does not come back.
    t.env.click(t.$('tr-record'));
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.equal(banner.hidden, true);
  });

  it('[BUG-02] noticeKey: the same failure with other numbers or addresses is the same notice, another failure is not', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { noticeKey } = await importFresh('views/transcribe/recorder.js');
    assert.equal(noticeKey('CUDA out of memory. Tried to allocate 20.00 MiB'), noticeKey('CUDA out of memory. Tried to allocate 512.50 MiB'));
    assert.equal(noticeKey('Illegal memory access at 0x7f3a2c00'), noticeKey('illegal memory access at 0x7fffb000'));
    assert.equal(noticeKey('cuDNN error  7'), noticeKey('cuDNN error 12'));
    assert.notEqual(noticeKey('Transcription failed: CUDA out of memory'), noticeKey('Unknown language xx. The stream continues with auto-detect.'));
  });

  it('[BUG-02] the titles say what happened: a start that failed reads like a file that failed; a failed chunk says part was lost', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { noticeTitle, START_FAILED_TITLE } = await importFresh('views/transcribe/recorder.js');
    assert.equal(START_FAILED_TITLE, "Couldn't transcribe", 'the title api.explain() gives a file that failed');
    assert.equal(noticeTitle('Transcription failed: CUDA out of memory'), "Part of the recording wasn't transcribed");
    assert.equal(noticeTitle('Unknown language xx. The stream continues with auto-detect.'), 'The server reported a problem');
    assert.equal(noticeTitle(undefined), 'The server reported a problem');
  });

  it('[BUG-02] a failure the server repeats on every chunk is shown once: one alert, and on another view one toast saying what failed', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    const banner = t.$('tr-rec-error');
    const writes = countWrites(banner);
    // A persistent failure: one error per 100 ms chunk while speech is active, its numbers changing.
    for (let i = 0; i < 6; i += 1) {
      ws.serverSend({ type: 'error', message: `Transcription failed: CUDA out of memory. Tried to allocate ${20 + i}.00 MiB at 0x7f3a${i}c00` });
      await settle(t.env, 100);
    }
    assert.equal(writes.n, 1, 'the role=alert banner is written once, so it is read out once');
    assert.match(recError(), /Tried to allocate 20\.00 MiB/);
    // A different failure is news: it replaces the banner.
    ws.serverSend({ type: 'error', message: 'Unknown language xx. The stream continues with auto-detect.' });
    await settle(t.env);
    assert.equal(writes.n, 2);
    assert.match(recError(), /Unknown language xx/);
    // Nothing was lost this time, so the title doesn't say so.
    assert.equal(banner.querySelector('.banner__title').textContent, 'The server reported a problem');
    // While another view shows: one toast per different failure, with its message, never one per chunk.
    t.ctl.hide();
    const region = t.$('app-toasts-alert');
    let inserted = 0;
    const append = region.appendChild.bind(region);
    region.appendChild = (node) => { inserted += 1; return append(node); };
    for (let i = 0; i < 6; i += 1) {
      ws.serverSend({ type: 'error', message: `Transcription failed: cuDNN error ${i}` });
      await settle(t.env, 100);
    }
    assert.equal(inserted, 1);
    assert.match(region.textContent, /Part of the recording wasn't transcribed\. Transcription failed: cuDNN error 0/);
    assert.equal(recordName(), 'Stop recording', 'the recording goes on');
  });

  it('[BUG-02] a stream refused right after it was accepted (1013 before session.begin) is a start that failed, not a disconnect', async () => {
    t = await mountView();
    const spoken = trackSpoken(t);
    const { app } = await load('core/store.js');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env, 600);
    ws.serverClose(1013, 'Too many concurrent streams');
    await settle(t.env, 600);
    const banner = t.$('tr-rec-error');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'Too many live transcriptions at once. Try again shortly.');
    assert.equal(/disconnected/i.test(recError()), false, recError());
    assert.equal(banner.querySelector('[data-action="record-retry"]').textContent.trim(), 'Retry');
    assert.equal(spoken.includes('Recording started'), false, JSON.stringify(spoken));
    assert.equal(app.get().recording, null);
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'));
    assert.equal(recordName(), 'Record');
    assert.equal(t.ctl.state().transcript.source, '', 'nothing was recorded, so nothing replaced the transcript');
    assert.equal(t.$('tr-meta').textContent, '', 'no "Recorded 0:00" line');
  });

  it('[BUG-02] a start that fails before session.begin (the VAD didn\'t load) leaves a file\'s transcript and its audio as they were', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const url = t.ctl.state().fileUrl;
    const meta = t.$('tr-meta').textContent;
    assert.match(meta, /^Duration 0:05 · Processed in/);
    await settle(t.env, 600);
    const spoken = trackSpoken(t);
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env, 600);
    assert.equal(t.$('tr-log').querySelectorAll('button.tr-ts').length, 2, 'on screen while the server loads');
    // What the empty-scenario preview server answers when the Silero download is refused.
    ws.serverSend({ type: 'error', message: 'Failed to load VAD model: preview server: model downloads are disabled' });
    await settle(t.env, 600);
    assert.equal(t.$('tr-rec-error').querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.match(recError(), /Failed to load VAD model/);
    assert.equal(t.$('tr-log').querySelectorAll('.tr-seg').length, 2);
    assert.equal(t.$('tr-log').querySelectorAll('button.tr-ts').length, 2, 'its times still play the file');
    assert.equal(t.$('tr-empty').hidden, true);
    assert.equal(t.$('tr-meta').textContent, meta, 'no "Recorded 0:00 · whisper base" line');
    assert.equal(t.$('tr-play').hidden, false);
    assert.equal(t.ctl.state().fileUrl, url);
    assert.equal(t.env.revokedUrls.includes(url), false, 'its audio is kept');
    assert.equal(spoken.includes('Recording started'), false, JSON.stringify(spoken));
  });

  it('[TR-11] a recording after a file that ends without a word gives the file back with its audio; one with words replaces it', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const url = t.ctl.state().fileUrl;
    // The microphone goes right after the session began: the recording stops before a word came back.
    const ws1 = await startRecording(t);
    assert.equal(t.$('tr-log').querySelectorAll('.tr-seg').length, 0, 'a recording shows its own transcript');
    for (const track of tracks()) {
      track.readyState = 'ended';
      track.dispatchEvent(new FakeEvent('ended'));
    }
    await settle(t.env);
    assert.deepEqual(ws1.sentJson, [{ type: 'stop' }]);
    ws1.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.match(recError(), /The microphone stopped/);
    assert.equal(t.$('tr-log').querySelectorAll('button.tr-ts').length, 2, "the file's transcript is back, its times play");
    assert.match(t.$('tr-meta').textContent, /^Duration 0:05/);
    assert.equal(t.ctl.state().fileUrl, url);
    assert.equal(t.env.revokedUrls.includes(url), false);
    // A recording that brings words back replaces the file for good: its audio is released.
    const ws2 = await startRecording(t);
    said(ws2, 'New words.', { start: 0.5, end: 1.5 });
    t.env.click(t.$('tr-record'));
    ws2.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.deepEqual(t.$('tr-log').querySelectorAll('.tr-seg').map((p) => p.querySelector('.tr-seg__text').textContent), ['New words.']);
    assert.ok(t.env.revokedUrls.includes(url));
    assert.equal(t.ctl.state().fileUrl, '');
    assert.equal(t.$('tr-play').hidden, true);
  });

  it('[BUG-02] close codes give their own copy (1008, 1003, 1013, 4001), never only "disconnected"', async () => {
    // A close code reaches the page only from a socket the server accepted. The server accepts a socket it refuses,
    // then closes it with the code and reason; an HTTP refusal of the handshake (an
    // unknown Host, a rate limit) still arrives as 1006: "a stream that never opens" below. Before session.begin it
    // is a start that failed.
    const cases = [
      [1008, 'bad rate', "The microphone's sample rate wasn't accepted."],
      [1003, 'Unsupported encoding', 'Unsupported audio encoding.'],
      [1013, 'Too many concurrent streams', 'Too many live transcriptions at once. Try again shortly.'],
      [4001, 'Invalid ticket', 'This server needs an API key. Enter it in Settings › Connection.'],
      [4999, 'odd', 'Disconnected (code 4999).'],
    ];
    for (const [code, reason, copy] of cases) {
      t = await mountView();
      t.env.click(t.$('tr-record'));
      await settle(t.env);
      const ws = lastSocket(t.env);
      ws.open();
      await settle(t.env);
      ws.serverClose(code, reason);
      await settle(t.env);
      const banner = t.$('tr-rec-error');
      assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe", `${code}`);
      assert.equal(banner.querySelector('.banner__msg').textContent, copy, `${code}: ${recError()}`);
      if (code === 4001) assert.ok(banner.querySelector('[data-action="open-settings"]'), '4001 offers Open Settings');
      else assert.equal(banner.querySelector('[data-action="record-retry"]').textContent.trim(), 'Retry', `${code}`);
      t.env.restore();
    }
    t = null;
  });

  it('[BUG-02] a stream that never opens (a refused handshake reaches the browser as 1006) is not called "dropped"', async () => {
    t = await mountView();
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    lastSocket(t.env).error();   // HTTP 403 on the upgrade, or no server: an error, then close 1006 before open
    await settle(t.env);
    const banner = t.$('tr-rec-error');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe", 'a start that failed, not a disconnect');
    assert.equal(banner.querySelector('.banner__msg').textContent,
      "The live transcription stream didn't open. Check that Open Speech is running and that this page comes from its own address.");
    assert.equal(recError().includes('dropped'), false);
    noNode(banner.querySelector('details'), 'no reason came with it');
    assert.equal(t.$('tr-rec-error').querySelector('[data-action="record-retry"]').textContent.trim(), 'Retry');
    assert.ok(tracks().every((tr) => tr.readyState === 'ended'), 'the microphone is released');
    assert.equal(recordName(), 'Record');
  });

  it('[BUG-02] 4001 with a key set says the ticket was refused', async () => {
    t = await mountView({ env: { seed: { 'open-speech-api-key': 'k1' } } });
    (await load('core/auth.js')).init();
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    lastSocket(t.env).serverClose(4001, 'Invalid or expired ticket');
    await settle(t.env);
    assert.match(recError(), /The server refused the API key or ticket\. Check Settings › Connection\./);
  });

  // The server's refusals as a browser sees them (src/routes/streaming.py
  // and src/streaming.py: accepted, then closed with the code and reason at once). Each says why, and offers what
  // helps: Open Settings for a key, no Retry for an address that isn't allowed (it would be refused again).
  it('a refused stream (opened, then closed with the server\'s code and reason) says why: the key with Open Settings, the address with no Retry, the sample rate with Retry', async () => {
    const cases = [
      [4001, 'Invalid or missing API key', 'This server needs an API key. Enter it in Settings › Connection.', 'open-settings'],
      [1008, 'Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it.', "This page's address isn't allowed to connect. Open Open Speech from its own address, or set OS_WS_ALLOWED_ORIGINS.", ''],
      [1008, 'Invalid sample_rate: must be 8000-192000', "The microphone's sample rate wasn't accepted.", 'record-retry'],
    ];
    for (const [code, reason, copy, action] of cases) {
      t = await mountView();
      t.env.click(t.$('tr-record'));
      await settle(t.env);
      const ws = lastSocket(t.env);
      ws.open();
      ws.serverClose(code, reason);
      await settle(t.env);
      const banner = t.$('tr-rec-error');
      assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe", `${code}`);
      assert.equal(banner.querySelector('.banner__msg').textContent, copy, `${code}: ${recError()}`);
      assert.match(banner.querySelector('pre').textContent, new RegExp(reason.slice(0, 12)), 'the reason in Details');
      const actions = [...banner.querySelectorAll('[data-action]')].map((b) => b.getAttribute('data-action')).filter((a) => !a.startsWith('dismiss'));
      assert.deepEqual(actions, action ? [action] : [], `${code} ${reason}`);
      assert.ok(tracks().every((tr) => tr.readyState === 'ended'), 'the microphone is released');
      assert.equal(recordName(), 'Record');
      t.env.restore();
    }
    t = null;
  });

  it('[TR-13] a start failure cleans up everything: tracks, context, the mic owner and the chip', async () => {
    let fail = true;
    t = await mountView({ routes: { 'GET /api/models/:id/status': () => (fail ? json(500, { error: { message: 'Model store unreadable' } }) : json(200, { state: 'loaded' })) } });
    const mic = await load('core/mic.js');
    const { app } = await load('core/store.js');
    t.app.set({ models: { ...t.app.get().models, fetchedAt: 0 } });   // no fast path: the status is checked
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.env.sockets.length, 0);
    assert.match(recError(), /Couldn't transcribe/);
    assert.match(recError(), /Model store unreadable/);
    noNode(t.$('tr-rec-error').querySelector('details'), 'no code of its own: Details would only repeat the message');
    assert.ok(tracks().length === 1 && tracks().every((tr) => tr.readyState === 'ended'));
    assert.equal(t.env.audioContexts[0].state, 'closed');
    assert.equal(mic.owner(), null);
    assert.equal(app.get().recording, null);
    assert.equal(recordName(), 'Record');
    assert.equal(t.$('tr-record').disabled, false);
    fail = false;
  });

  it('[TR-13] a start failure says the server\'s message; its Details give the message and code, never the JSON envelope', async () => {
    t = await mountView({ routes: { 'GET /api/models/:id/status': () => json(500, { error: { message: 'CUDA out of memory', code: 'model_load_failed' } }) } });
    t.app.set({ models: { ...t.app.get().models, fetchedAt: 0 } });   // no fast path: the status is checked
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('tr-rec-error');
    assert.equal(banner.querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.equal(banner.querySelector('.banner__msg').textContent, 'CUDA out of memory');
    assert.equal(banner.querySelector('details pre').textContent, 'CUDA out of memory (model_load_failed)');
    assert.equal(banner.textContent.includes('{'), false, 'no JSON envelope');
    assert.ok(banner.querySelector('[data-action="record-retry"]'), 'a 500 is worth another try');
  });

  it('[TR-13] permission denied: the microphone help says how to allow it; nothing stays open', async () => {
    t = await mountView({ env: { mediaDevices: { policy: 'NotAllowedError' } } });
    const mic = await load('core/mic.js');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.match(help(), /Microphone access is blocked\. Allow it in your browser's site settings, then try again\./);
    assert.equal(recError(), '', 'the help banner says it once');
    assert.equal(mic.owner(), null);
    assert.equal(t.env.sockets.length, 0);
    assert.equal(t.$('tr-record').disabled, false, 'the listener can try again after allowing it');
  });
});

describe('transcribe recorder: cancel and shortcuts', () => {
  it('[TR-12] Cancel and Esc close the session without keeping its words; the earlier transcript comes back', async () => {
    t = await mountView();
    const ws1 = await startRecording(t);
    said(ws1, 'keep me');
    t.env.click(t.$('tr-record'));
    ws1.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.ok(t.$('tr-log').textContent.includes('keep me'));
    const ws2 = await startRecording(t);
    said(ws2, 'throw me away');
    await settle(t.env);
    assert.ok(t.$('tr-log').textContent.includes('throw me away'));
    t.env.key('Escape', {}, t.doc.body);
    await settle(t.env);
    assert.equal(ws2.readyState, FakeSocket.CLOSED);
    assert.equal(ws2.sentJson.length, 0, 'no graceful stop: nothing more is wanted');
    assert.equal(t.$('tr-log').textContent.includes('throw me away'), false);
    assert.ok(t.$('tr-log').textContent.includes('keep me'));
    await settle(t.env, 600);   // announcements within 500 ms are coalesced; the last one wins
    assert.equal(announced(t), 'Recording discarded');
    assert.equal(recordName(), 'Record');
  });

  it('[TR-12] after 30 s Cancel asks "Discard this recording?" first', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    await settle(t.env, 31000);
    t.env.click(t.$('tr-cancel'));
    await settle(t.env);
    assert.equal(t.$('app-confirm').open, true);
    assert.equal(t.$('app-confirm-title').textContent, 'Discard this recording?');
    await answerConfirm(t, false);
    assert.equal(ws.readyState, FakeSocket.OPEN, 'kept recording');
    assert.equal(recordName(), 'Stop recording');
    t.env.key('Escape', {}, t.doc.body);
    await settle(t.env);
    assert.equal(t.$('app-confirm').open, true);
    await answerConfirm(t, true);
    assert.equal(ws.readyState, FakeSocket.CLOSED);
    assert.equal(recordName(), 'Record');
  });

  it('[TR-12] "r" starts and stops recording while Transcribe shows (not while typing in a field)', async () => {
    t = await mountView();
    t.app.set({ route: { view: 'transcribe', sub: '', id: '', path: 'transcribe', shot: '' } });
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    const ws = lastSocket(t.env);
    assert.ok(ws, 'started');
    ws.open();
    begin(ws);
    await settle(t.env);
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    assert.deepEqual(ws.sentJson, [{ type: 'stop' }]);
    // Typing an "r" in a field never toggles recording.
    const before = t.env.sockets.length;
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    t.$('tr-model').focus();
    t.env.key('r');
    await settle(t.env);
    assert.equal(t.env.sockets.length, before);
  });
});

describe('transcribe: the "r" shortcut and the "g" sequences', () => {
  it('[TR-12] "g" then "r" goes to the Reader and never starts a recording; "r" alone still does', async () => {
    t = await mountView();
    const shortcuts = await load('core/shortcuts.js');
    const went = [];
    // The shell registers the "g <key>" sequences; this one stands in for it.
    shortcuts.register({ id: 'test-go-reader', keys: ['g r'], singleKey: true, description: 'Go to Reader', handler: () => went.push('reader') });
    t.env.key('g', {}, t.doc.body);
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    assert.deepEqual(went, ['reader']);
    assert.equal(t.env.mediaDevices.calls.length, 0, 'no microphone');
    assert.equal(t.env.sockets.length, 0);
    await settle(t.env, 1500);   // the sequence window is over
    t.env.key('r', {}, t.doc.body);
    await settle(t.env);
    assert.equal(t.env.mediaDevices.calls.length, 1);
  });
});

describe('transcribe: microphone help states and devices', () => {
  it('[TR-13] an insecure page: the HTTPS help, and Record is off (uploads still work)', async () => {
    t = await mountView({ env: { secure: false } });
    assert.match(help(), /needs a secure connection\. Open this page over HTTPS \(Open Speech can make a self-signed certificate\)\./);
    assert.equal(t.$('tr-record').disabled, true);
    assert.equal(t.$('tr-file').disabled, false);
    noNode(t.$('tr-mic-help').querySelector('[data-action]'), 'nothing to dismiss: it is how the page is');
  });

  it('[TR-13] no capture support, no device, and a permission already denied each have their copy', async () => {
    t = await mountView({ env: { mediaDevices: null } });
    assert.match(help(), /This browser can't capture audio\./);
    assert.equal(t.$('tr-record').disabled, true);
    t.env.restore();
    t = await mountView({ env: { mediaDevices: { devices: [] } } });
    await settle(t.env);
    assert.match(help(), /No microphone found\./);
    assert.equal(t.$('tr-mic').disabled, true);
    t.env.restore();
    t = await mountView({ env: { permissions: { microphone: 'denied' } } });
    await settle(t.env);
    assert.match(help(), /Microphone access is blocked/);
    const dismiss = t.$('tr-mic-help').querySelector('[data-action="dismiss-help"]');
    t.env.click(dismiss);
    await settle(t.env);
    assert.equal(t.$('tr-mic-help').hidden, true);
  });

  it('[TR-13] an old "blocked" answer goes once the microphone opens, not only when the server begins the session', async () => {
    t = await mountView({ env: { permissions: { microphone: 'denied' } } });
    await settle(t.env);
    assert.match(help(), /Microphone access is blocked/);
    // The listener allowed it in the site settings meanwhile; this page still has the old answer.
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env);
    assert.equal(phase(), 'Starting…', 'the server is still loading');
    assert.equal(help(), '', 'audio is coming in: it is plainly not blocked');
    begin(ws);
    await settle(t.env);
    assert.equal(help(), '');
    assert.equal(recordName(), 'Stop recording');
  });

  it('[TR-11] an iPhone Home Screen app that loses the microphone in the background says so', async () => {
    t = await mountView();
    await startRecording(t);
    t.env.setHidden(true);
    for (const track of tracks()) {
      track.readyState = 'ended';
      track.dispatchEvent(new FakeEvent('ended'));
    }
    t.env.setHidden(false);
    await settle(t.env, 4100);
    assert.match(help(), /Recording stopped when the app went to the background\./);
    assert.equal(recordName(), 'Record');
  });

  it('[TR-06] the microphone select lists the inputs, keeps the saved one, saves a choice and follows devicechange', async () => {
    t = await mountView();
    const mic = await load('core/mic.js');
    const select = t.$('tr-mic');
    assert.deepEqual(select.querySelectorAll('option').map((o) => o.textContent), ['Default microphone', 'Built-in Microphone', 'USB Microphone']);
    t.env.change(select, 'mic-b');
    assert.equal(mic.savedDevice(), 'mic-b');
    t.env.mediaDevices.devices.push({ deviceId: 'mic-c', kind: 'audioinput', label: 'Headset', groupId: 'g3' });
    t.env.mediaDevices.fireDeviceChange();
    await settle(t.env);
    assert.deepEqual(select.querySelectorAll('option').map((o) => o.value), ['', 'mic-a', 'mic-b', 'mic-c']);
    assert.equal(select.value, 'mic-b');
    const ws = await startRecording(t);
    assert.equal(select.disabled, true, 'no device change mid-recording');
    assert.deepEqual(t.env.mediaDevices.calls[0].audio.deviceId, { exact: 'mic-b' });
    assert.ok(ws);
  });
});

describe('transcribe recorder: the pure state machine', () => {
  it('[BUG-08] start() is refused while a session is active, and stop() before the socket opens cancels', async () => {
    const env = installBrowserEnv({ routes: { 'GET /api/models/:id/status': () => delay(10000, json(200, { state: 'loaded' })) } });
    t = { env };
    const { createRecorder } = await importFresh('views/transcribe/recorder.js');
    const ends = [];
    const r = createRecorder({ on: { end: (d) => ends.push(d.reason) } });
    const p1 = r.start({ model: STT });
    assert.equal(r.state.phase, 'requesting');
    assert.equal(r.isActive(), true);
    assert.equal(await r.start({ model: STT }), false);
    await settle(env);
    assert.equal(r.state.phase, 'starting');
    assert.equal(r.stop(), true);
    assert.equal(r.state.phase, 'idle');
    assert.deepEqual(ends, ['cancelled']);
    // The model check answers after the cancel: the old start bails out and opens no socket.
    await settle(env, 10000);
    assert.equal(await p1, false);
    assert.equal(env.sockets.length, 0);
    assert.equal(env.mediaDevices.calls.length, 1);
    assert.equal(r.state.phase, 'idle');
  });
});

// Live sessions could end with no transcript, and nothing on the page said why.
// The server got audio its VAD never scored as speech (no speech_start, transcriptions=0), which is what a microphone
// sending silence gives; the view showed "Recording", "Silence" and "Listening…" for as long as it ran, so a device
// problem looked like a broken feature. A few seconds of recording with no speech now says so, and says when no sound
// comes in at all. "A few seconds": nothing at 3 s, the message by 5 s.
describe('transcribe recorder: nothing heard', () => {
  // Its own banner, under the level meter: never #tr-mic-help above Record, so Stop never moves under the pointer.
  const heard = () => (t.$('tr-heard').hidden ? '' : t.$('tr-heard').textContent.replace(/\s+/g, ' ').trim());
  const bannerTitle = () => (t.$('tr-heard').hidden ? '' : (t.$('tr-heard').querySelector('.banner__title') || {}).textContent || '');
  const bannerMsg = () => (t.$('tr-heard').hidden ? '' : (t.$('tr-heard').querySelector('.banner__msg') || {}).textContent || '');

  it('[TR-09] a few seconds with a level but no speech_start: "No speech heard yet", check the microphone and its level; speech takes it away', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    micFrame(t.env, new Float32Array(2048).fill(0.02));   // a level (-34 dBFS) the server's VAD doesn't call speech
    await settle(t.env, 3000);
    assert.equal(heard(), '', 'not at once: the listener may not have started talking');
    micFrame(t.env, new Float32Array(2048).fill(0.02));
    await settle(t.env, 2000);
    assert.equal(bannerTitle(), 'No speech heard yet');
    assert.equal(bannerMsg(), 'Check the microphone and its level.');
    assert.equal(t.$('tr-heard').getAttribute('role'), 'status', 'said politely, never an alert');
    assert.equal(help(), '', 'not in the microphone help above Record');
    assert.equal(phase(), 'Recording', 'the recording goes on');
    ws.serverSend({ type: 'vad', state: 'speech_start' });
    await settle(t.env);
    assert.equal(heard(), '', 'speech was heard');
    // Silence after speech is a pause, not a problem: nothing more is said.
    ws.serverSend({ type: 'vad', state: 'speech_end' });
    await settle(t.env, 10000);
    assert.equal(heard(), '');
  });

  it('[TR-09] with no level at all it says the microphone sends silence, also when no frame comes; a level later says "No speech heard yet" instead', async () => {
    t = await mountView();
    await startRecording(t);
    micFrame(t.env, new Float32Array(2048));                       // digital silence
    micFrame(t.env, new Float32Array(2048).fill(1 / 65536));       // under one 16-bit step: the server gets zeros
    await settle(t.env, 5000);
    assert.equal(bannerTitle(), 'The microphone sends silence');
    assert.match(bannerMsg(), /isn't muted/);
    assert.match(bannerMsg(), /choose another microphone/);
    micFrame(t.env, new Float32Array(2048).fill(0.01));
    await settle(t.env);
    assert.equal(bannerTitle(), 'No speech heard yet', 'sound comes in now: the level is the question');
    t.ctl.destroy();
    t.env.restore();
    // The capture runs but no frame ever comes (no audio reaches the page at all).
    t = await mountView();
    await startRecording(t);
    await settle(t.env, 5000);
    assert.equal(bannerTitle(), 'The microphone sends silence');
  });

  it('[TR-09] the message waits for the server: nothing is said while the model loads, only a few seconds after session.begin', async () => {
    t = await mountView();
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    ws.open();
    await settle(t.env, 8000);   // the server is still loading the model
    assert.equal(phase(), 'Starting…');
    assert.equal(heard(), '');
    begin(ws);
    await settle(t.env, 3000);
    assert.equal(heard(), '');
    await settle(t.env, 2000);
    assert.equal(bannerTitle(), 'The microphone sends silence');
  });

  it('[TR-09] Dismiss hides it for the session; a session stopped with nothing heard says so where the transcript would be; with the VAD off a transcript counts as heard', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    micFrame(t.env, new Float32Array(2048).fill(0.02));
    await settle(t.env, 5000);
    assert.equal(bannerTitle(), 'No speech heard yet');
    t.env.click(t.$('tr-heard').querySelector('[data-action="dismiss-heard"]'));
    await settle(t.env);
    assert.equal(heard(), '');
    sameNode(t.doc.activeElement, t.$('tr-record'), 'focus goes to Stop, never to <body>');
    await settle(t.env, 5000);
    assert.equal(heard(), '', 'dismissed stays dismissed while this session runs');
    t.env.click(t.$('tr-record'));   // Stop
    await settle(t.env);
    ws.serverSend({ type: 'session.end', reason: 'client_stop', transcriptions: 0, errors: 0 });
    await settle(t.env);
    assert.equal(phase(), 'Record');
    assert.equal(t.$('tr-empty').hidden, false);
    assert.equal(t.$('tr-empty-title').textContent, 'No speech was heard.');
    assert.equal(t.$('tr-empty-text').textContent, 'Check the microphone and its level, then record again.');
    // The next session starts afresh, and with the VAD off (no vad events) words are what count as heard.
    const ws2 = await startRecording(t, { begin: { vad_enabled: false } });
    micFrame(t.env, new Float32Array(2048).fill(0.02));
    said(ws2, 'hello there', { isFinal: false, speechFinal: false });
    await settle(t.env, 10000);
    assert.equal(heard(), '');
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    ws2.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.equal(t.$('tr-empty-title').textContent, 'Your transcript appears here.');
  });
});
