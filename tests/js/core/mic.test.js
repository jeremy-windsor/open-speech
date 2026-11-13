import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<main id="main"></main><div id="app-toasts-polite" role="status"></div><div id="app-toasts-alert" role="alert"></div><div id="app-audio-host" hidden><audio id="app-audio"></audio></div>';

async function setup(options = {}) {
  env = installBrowserEnv({ html: SHELL, ...options });
  const mic = await importFresh('core/mic.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  const audio = await import('../../../src/static/js/core/audio.js');
  return { mic, app, audio };
}

describe('core/mic: support and devices', () => {
  it('[GS-27] capture needs a secure context; without getUserMedia it is unsupported', async () => {
    let { mic } = await setup({ secure: false });
    assert.deepEqual(mic.support(), { ok: false, reason: 'insecure' });
    await assert.rejects(mic.open({ owner: 'transcribe' }), (e) => e.name === 'MicError' && e.reason === 'insecure' && /HTTPS/.test(e.message));
    env.restore();
    ({ mic } = await setup({ mediaDevices: null }));
    assert.deepEqual(mic.support(), { ok: false, reason: 'unsupported' });
    env.restore();
    ({ mic } = await setup());
    assert.deepEqual(mic.support(), { ok: true });
  });

  it('[TR-14] the device list: audio inputs, label or "Microphone N", "Default microphone" first', async () => {
    const { mic } = await setup({ mediaDevices: { devices: [
      { deviceId: 'a1', kind: 'audioinput', label: 'Built-in' },
      { deviceId: 'a2', kind: 'audioinput', label: '' },
      { deviceId: 'o1', kind: 'audiooutput', label: 'Speakers' },
      { deviceId: 'default', kind: 'audioinput', label: 'Default - Built-in' },
    ] } });
    assert.deepEqual(await mic.listDevices(), [
      { id: '', label: 'Default microphone' }, { id: 'a1', label: 'Built-in' }, { id: 'a2', label: 'Microphone 2' },
    ]);
  });

  it('[TR-14] listing that is unsupported, fails or finds no input gives [] (the select is disabled)', async () => {
    let { mic } = await setup({ mediaDevices: { devices: [{ deviceId: 'o', kind: 'audiooutput', label: 'x' }] } });
    assert.deepEqual(await mic.listDevices(), []);
    env.mediaDevices.enumerateDevices = async () => { throw new Error('nope'); };
    assert.deepEqual(await mic.listDevices(), []);
    env.restore();
    ({ mic } = await setup({ mediaDevices: null }));
    assert.deepEqual(await mic.listDevices(), []);
  });

  it('[TR-14] the list refreshes on devicechange and after a permission grant (one shared list)', async () => {
    const { mic } = await setup();
    let calls = 0;
    const off = mic.onDeviceChange(() => { calls += 1; });
    env.mediaDevices.fireDeviceChange();
    assert.equal(calls, 1);
    const s = await mic.open({ owner: 'voices' });
    assert.equal(calls, 2, 'labels become visible after the grant');
    s.stop();
    off();
    env.mediaDevices.fireDeviceChange();
    assert.equal(calls, 2);
  });

  it('[TR-15] the choice persists as the opaque deviceId only; a vanished saved device is cleared', async () => {
    const { mic } = await setup();
    mic.saveDevice('mic-b');
    assert.equal(env.localStorage.__data['open-speech-mic-device'], 'mic-b');
    assert.equal(mic.savedDevice(), 'mic-b');
    await mic.listDevices();
    assert.equal(mic.savedDevice(), 'mic-b', 'still there');
    mic.saveDevice('gone');
    await mic.listDevices();
    assert.equal(mic.savedDevice(), '', 'cleared');
    mic.saveDevice('');
    assert.equal(env.localStorage.__data['open-speech-mic-device'], undefined);
  });

  it('[TR-15] a chosen device that fails falls back to the default device with a toast', async () => {
    const { mic } = await setup();
    mic.saveDevice('unplugged');
    const s = await mic.open({ owner: 'transcribe' });
    assert.deepEqual(env.mediaDevices.calls.map((c) => c.audio.deviceId?.exact ?? 'default'), ['unplugged', 'default']);
    assert.equal(mic.savedDevice(), '');
    assert.match(document.getElementById('app-toasts-polite').textContent, /Selected microphone unavailable; using default microphone/);
    s.stop();
  });

  it('a denied permission explains how to allow it (and keeps the saved device)', async () => {
    const { mic } = await setup({ mediaDevices: { policy: 'NotAllowedError' } });
    mic.saveDevice('mic-a');
    await assert.rejects(mic.open({ owner: 'transcribe' }), (e) => e.reason === 'denied' && /Allow it in your browser's site settings/.test(e.message));
    assert.equal(mic.savedDevice(), 'mic-a');
    assert.equal(mic.owner(), null);
  });

  it('permission() reads the Permissions API', async () => {
    const { mic } = await setup({ permissions: { microphone: 'granted' } });
    assert.equal(await mic.permission(), 'granted');
  });
});

describe('core/mic: capture', () => {
  it('[BUG-29] capture uses the static AudioWorklet module (not ScriptProcessor) and delivers its frames', async () => {
    const { mic, app } = await setup({ audio: { sampleRate: 44100 } });
    const s = await mic.open({ owner: 'transcribe', deviceId: 'mic-a', echoCancellation: false });
    const ctx = env.audioContexts[0];
    assert.deepEqual(ctx.__modules, ['/static/js/core/mic-worklet.js']);
    assert.equal(ctx.__processors.length, 0, 'no ScriptProcessorNode');
    assert.equal(s.mode, 'worklet');
    assert.equal(s.sampleRate, 44100);
    assert.deepEqual(env.mediaDevices.calls[0].audio, { echoCancellation: false, noiseSuppression: true, deviceId: { exact: 'mic-a' } });
    const frames = [];
    s.onFrame((f) => frames.push(f.length));
    ctx.__worklets[0].__emit(new Float32Array(2048).fill(0.5));
    assert.deepEqual(frames, [2048]);
    assert.ok(Math.abs(s.level().rms - 0.5) < 1e-6);
    assert.ok(Math.abs(s.level().db - (20 * Math.log10(0.5))) < 1e-6);
    assert.deepEqual(app.get().recording.owner, 'transcribe');
    s.stop();
  });

  it('[BUG-29] falls back to ScriptProcessor(4096) when AudioWorklet is missing or refused', async () => {
    for (const worklet of [false, 'reject']) {
      const { mic } = await setup({ audio: { worklet } });
      const s = await mic.open({ owner: 'voices' });
      const ctx = env.audioContexts[0];
      assert.equal(s.mode, 'script-processor');
      assert.equal(ctx.__processors[0].bufferSize, 4096);
      const frames = [];
      s.onFrame((f) => frames.push(f.length));
      ctx.__processors[0].__process(new Float32Array(4096));
      assert.deepEqual(frames, [4096]);
      s.stop();
      env.restore();
    }
  });

  it('[TR-19] Transcribe and Voices recording are mutually exclusive', async () => {
    const { mic } = await setup();
    const s = await mic.open({ owner: 'transcribe' });
    assert.equal(mic.owner(), 'transcribe');
    await assert.rejects(mic.open({ owner: 'voices' }), (e) => e.name === 'MicBusyError' && e.message === 'Stop the Transcribe recording first');
    await assert.rejects(mic.open({ owner: 'transcribe' }), { name: 'MicBusyError' });
    s.stop();
    const v = await mic.open({ owner: 'voices' });
    assert.equal(mic.owner(), 'voices');
    v.stop();
  });

  it('[TR-19] a double start never opens two captures (the owner is reserved before the first await)', async () => {
    const { mic } = await setup();
    const first = mic.open({ owner: 'transcribe' });
    const second = mic.open({ owner: 'transcribe' }).catch((e) => e);
    const s = await first;
    assert.equal((await second).name, 'MicBusyError');
    assert.equal(env.mediaDevices.calls.length, 1);
    s.stop();
  });

  it('[TR-19] Cancel while the permission prompt is up, then Record again: the new open() waits for the old one, whose caller stops it, and records', async () => {
    // The stale open() used to keep the microphone reserved until the prompt was answered, so the
    // second Record failed with "A recording is already running" while nothing recorded.
    const { mic, app } = await setup();
    let answer = null;
    const real = env.mediaDevices.getUserMedia.bind(env.mediaDevices);
    env.mediaDevices.getUserMedia = (c) => (answer ? real(c) : new Promise((resolve, reject) => { answer = { allow: () => resolve(real(c)), deny: reject }; }));
    let cancelled = true;
    const first = mic.open({ owner: 'transcribe' }).then((s) => { if (cancelled) s.stop(); return s; });   // the view: "if (!current(my)) session.stop()"
    await env.flush();
    const second = mic.open({ owner: 'transcribe' });          // Record again, the prompt still up
    await env.flush();
    assert.equal(env.audioContexts.length, 2, 'the second context is still made inside the gesture');
    answer.allow();
    const stale = await first;
    const s = await second;
    assert.equal(stale.active, false, 'the stale capture was stopped by its caller');
    assert.ok(env.mediaDevices.streams[0].getTracks().every((t) => t.readyState === 'ended'));
    assert.equal(s.active, true, 'the new Record records');
    assert.equal(mic.owner(), 'transcribe');
    assert.equal(mic.currentSession(), s);
    assert.equal(app.get().recording.owner, 'transcribe');
    s.stop();
    assert.equal(mic.owner(), null);
    cancelled = false;
  });

  it('[TR-19] after Transcribe gave up during the prompt, Voices can record; if Transcribe kept its capture, Voices is refused as before', async () => {
    const { mic } = await setup();
    let answer = null;
    const real = env.mediaDevices.getUserMedia.bind(env.mediaDevices);
    const hold = () => { answer = null; env.mediaDevices.getUserMedia = (c) => (answer ? real(c) : new Promise((resolve) => { answer = { allow: () => resolve(real(c)) }; })); };
    hold();
    const gaveUp = mic.open({ owner: 'transcribe' }).then((s) => { s.stop(); return s; });
    await env.flush();
    const voices = mic.open({ owner: 'voices' });
    await env.flush();
    answer.allow();
    await gaveUp;
    const v = await voices;
    assert.equal(mic.owner(), 'voices');
    v.stop();
    hold();
    const kept = mic.open({ owner: 'transcribe' });
    await env.flush();
    const refused = mic.open({ owner: 'voices' }).catch((e) => e);
    await env.flush();
    answer.allow();
    const t = await kept;
    const error = await refused;
    assert.equal(error.name, 'MicBusyError');
    assert.equal(error.message, 'Stop the Transcribe recording first');
    assert.equal(mic.owner(), 'transcribe', 'the refused open() leaves the reservation alone');
    assert.equal(t.active, true);
    assert.equal(env.audioContexts.at(-1).state, 'closed', "the refused open's context is closed");
    t.stop();
  });

  it('[TR-19] a stale open() that is denied frees the microphone for the one waiting behind it', async () => {
    const { mic } = await setup();
    let answer = null;
    env.mediaDevices.getUserMedia = () => new Promise((resolve, reject) => { answer = { deny: () => reject(new DOMException('Permission denied', 'NotAllowedError')) }; });
    const first = mic.open({ owner: 'transcribe' }).catch((e) => e);
    await env.flush();
    const second = mic.open({ owner: 'transcribe' }).catch((e) => e);
    await env.flush();
    answer.deny();
    assert.equal((await first).reason, 'denied');
    await env.flush();
    answer.deny();
    const error = await second;
    assert.equal(error.name, 'MicError', 'the second asked the browser itself (not "already running")');
    assert.equal(error.reason, 'denied');
    assert.equal(mic.owner(), null);
  });

  it('[REV-40] opening the mic pauses playback first; stop() releases every track and the context', async () => {
    const { mic, app, audio } = await setup();
    const log = [];
    audio.claim({ id: 'take:1', owner: 'speak', kind: 'clip', title: 'T', controls: { play() {}, pause: () => log.push('pause'), toggle() {}, stop() {} }, getState: () => ({ playing: true }) });
    const s = await mic.open({ owner: 'transcribe' });
    assert.deepEqual(log, ['pause']);
    const tracks = s.stream.getTracks();
    s.stop();
    assert.ok(tracks.every((t) => t.readyState === 'ended'));
    assert.equal(env.audioContexts[0].state, 'closed');
    assert.equal(app.get().recording, null);
    assert.equal(mic.owner(), null);
    s.stop();
  });

  it("Safari's audio session: a capture runs as 'play-and-record' (set before getUserMedia) and the earlier type comes back when it stops or fails", async () => {
    const { mic, audio } = await setup({ audioSession: true });
    const seen = [];
    const gum = env.mediaDevices.getUserMedia.bind(env.mediaDevices);
    env.mediaDevices.getUserMedia = (constraints) => { seen.push(navigator.audioSession.type); return gum(constraints); };
    audio.claim({ id: 'live', owner: 'live', kind: 'live', title: 'Live', controls: { play() {}, pause() {}, toggle() {}, stop() {} }, getState: () => ({ playing: true }) });
    assert.equal(env.audioSession.type, 'playback', 'Live made the page a playback page');
    const s = await mic.open({ owner: 'transcribe' });
    assert.deepEqual(seen, ['play-and-record'], 'the capture starts in play-and-record');
    s.stop();
    assert.equal(env.audioSession.type, 'playback', 'and gives the earlier type back');
    s.stop();
    assert.deepEqual(env.audioSession.history, ['playback', 'play-and-record', 'playback'], 'once');
    env.mediaDevices.policy = 'NotAllowedError';
    await assert.rejects(mic.open({ owner: 'voices' }), (e) => e.reason === 'denied');
    assert.equal(env.audioSession.type, 'playback', 'a capture that never started gives it back too');
  });

  it('[GS-20] pagehide or beforeunload stops an open capture (tracks and context)', async () => {
    const { mic } = await setup();
    const s = await mic.open({ owner: 'voices' });
    env.window.dispatchEvent(new Event('beforeunload'));
    assert.equal(s.active, false);
    assert.ok(s.stream.getTracks().every((t) => t.readyState === 'ended'));
    assert.equal(env.audioContexts[0].state, 'closed');
    const again = await mic.open({ owner: 'voices' });
    env.window.dispatchEvent(new Event('pagehide'));
    assert.equal(again.active, false);
  });

  it('the worklet frame buffer packs render quanta into 2048-sample frames with their RMS', async () => {
    await setup();
    const { createFrameBuffer, FRAME_SIZE } = await import('../../../src/static/js/core/mic-worklet.js');
    assert.equal(FRAME_SIZE, 2048);
    const frames = [];
    const push = createFrameBuffer(4, (f, rms) => frames.push([Array.from(f), rms]));
    push(new Float32Array([1, 1, 1]));
    push(new Float32Array([1, 0, 0, 0, 0, 0.5]));
    assert.deepEqual(frames.map((f) => f[0]), [[1, 1, 1, 1], [0, 0, 0, 0]]);
    assert.equal(frames[0][1], 1);
  });
});
