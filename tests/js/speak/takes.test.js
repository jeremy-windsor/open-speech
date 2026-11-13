// Takes (inventory SP-20, SP-21, SP-24; REV-33).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { peaks, excerpt, metaLine, perfLine, timeText, rtfText, MAX_TAKES } from '../../../src/static/js/views/speak/takes.js';
import { FakeOfflineAudioContext, FakeAudioContext, FakeAudioBuffer } from '../harness/fakes.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import { mountSpeak, ready, type, settle, callsTo, captureDownloads, bytes, json, speechBytes, delay, blobBytes, modelsPayload } from './helpers.js';

let t;
afterEach(() => {
  FakeOfflineAudioContext.decodeResult = null;
  t?.env.restore();
});

/** Generate one take (WAV: no streaming) and let it settle. */
async function generate(text = 'A take.') {
  type(t, text);
  t.env.click(t.$('sp-generate'));
  await settle(t.env);
  await settle(t.env);
  await settle(t.env);
}

const takeEls = () => [...t.$('sp-takes').querySelectorAll('.sp-take')];

describe('views/speak/takes (pure)', () => {
  it('[SP-24] peaks: the loudest sample per bucket, scaled to 1; silence is flat', () => {
    assert.deepEqual(peaks(new Float32Array([0, 0.5, -1, 0.25]), 2), [0.5, 1]);
    assert.deepEqual(peaks(new Float32Array(10), 4), [0, 0, 0, 0]);
    assert.deepEqual(peaks(null, 3), [0, 0, 0]);
    assert.equal(peaks(new Float32Array(1000).fill(0.2)).length, 96);
  });

  it('take lines: title, meta and timing', () => {
    assert.equal(excerpt('  In a hole   in the ground  '), '“In a hole in the ground”');
    assert.equal(excerpt('x'.repeat(200), 10), `“${'x'.repeat(9)}…”`);
    assert.equal(metaLine({ voiceLabel: 'Heart', modelLabel: 'Kokoro', speed: 1, format: 'mp3', chars: 1204 }), 'Heart · Kokoro · 1× · MP3 · 1,204 chars');
    assert.equal(perfLine({ ttfaMs: 139.6, rtf: 0.0412, device: 'GPU' }), 'First audio 140 ms · RTF 0.04 · GPU');
    assert.equal(rtfText(0.001), 'RTF < 0.01');
    assert.equal(timeText(0, 12), '0:12');
    assert.equal(timeText(3.2, 12), '0:03 / 0:12');
  });
});

describe('views/speak takes (view)', () => {
  it('[SP-20] a non-streamed take becomes an object URL that plays on the shared element; a refused autoplay is no error', async () => {
    // The browser refuses autoplay; the env's policy reaches the shell's #app-audio too.
    t = await mountSpeak({ env: { playPolicy: 'NotAllowedError' } });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    await generate('Play me.');
    const take = t.parts.takes.all()[0];
    assert.ok(take.url.startsWith('blob:'), 'an object URL of the blob');
    assert.equal(t.$('app-audio').getAttribute('src'), take.url, 'the shared element gets the take');
    assert.equal(t.$('sp-gen-error').hidden, true, 'autoplay refusal is not an error');
    assert.equal(t.app.get().player.id, take.id);
    assert.equal(t.app.get().player.statusText, 'Tap play to start listening');
    assert.equal(t.app.get().player.homeRoute, 'speak');
    assert.equal(t.app.get().player.owner, 'speak');
    assert.equal(takeEls()[0].querySelector('[data-take-act="play"]').disabled, false);
  });

  it('[SP-21] Download saves open-speech-<timestamp>.<format generated>, not the format picked since', async () => {
    t = await mountSpeak();
    await ready(t);
    const downloads = captureDownloads(t.env);
    t.env.change(t.$('sp-format'), 'wav');
    await generate('Saved as WAV.');
    t.env.change(t.$('sp-format'), 'flac');
    t.env.click(takeEls()[0].querySelector('[data-take-act="download"]'));
    assert.equal(downloads.length, 1);
    assert.match(downloads[0].download, /^open-speech-\d+\.wav$/);
    assert.deepEqual(await blobBytes(downloads[0].blob), Array.from(speechBytes()));
  });

  it('[SP-24] the take plays and pauses from its own button, shows its time, and draws a static waveform decoded after playback started', async () => {
    t = await mountSpeak({ env: { playPolicy: 'pending' } });   // playback starts when the test says so
    await ready(t);
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ length: 48000 * 2, sampleRate: 48000 });
    t.env.change(t.$('sp-format'), 'wav');
    await generate('Waveform.');
    assert.equal(FakeOfflineAudioContext.instances.length, 0, 'nothing is decoded before playback starts');
    const audioEl = t.$('app-audio');
    audioEl.__resolvePlay();
    await settle(t.env);
    await settle(t.env);
    assert.equal(FakeOfflineAudioContext.instances.length, 1, 'decoded with an OfflineAudioContext');
    assert.equal(FakeAudioContext.instances.length, 0, 'never a realtime context: the element is never routed through Web Audio');
    const take = t.parts.takes.all()[0];
    assert.equal(take.peaks.length, 96);
    assert.equal(take.durationS, 2);
    const el = takeEls()[0];
    assert.equal(el.querySelector('canvas.sp-wave').getAttribute('aria-hidden'), 'true');
    const play = el.querySelector('[data-take-act="play"]');
    assert.equal(play.getAttribute('aria-label'), 'Pause take 1');
    t.env.click(play);
    await settle(t.env);
    assert.equal(audioEl.paused, true);
    assert.equal(play.getAttribute('aria-label'), 'Play take 1');
    audioEl.__loadMetadata(2);
    await t.env.clock.tickAsync(300);
    audioEl.__timeupdate(1.2);
    await settle(t.env);
    const time = el.querySelector('.sp-take__time');
    assert.equal(time.textContent, '0:01 / 0:02');
    assert.equal(time.hasAttribute('aria-hidden'), false, 'the take\'s length can be read');
    assert.equal(time.closest('[role="status"], [role="alert"], [aria-live]:not([aria-live="off"])'), null, 'and never ticks in a live region');
    t.env.click(play);
    audioEl.__resolvePlay();
    await settle(t.env);
    assert.equal(audioEl.paused, false, 'the same take resumes from its button');
    assert.equal(play.getAttribute('aria-label'), 'Pause take 1');
    assertAccessibleNames(t.$('sp-takes'));
    // Each tooltip equals its button's name, so it is never read twice as a description.
    for (const button of el.querySelectorAll('button[data-tooltip]')) {
      assert.equal(button.getAttribute('data-tooltip'), button.getAttribute('aria-label'), button.getAttribute('aria-label'));
    }
  });

  it('[SP-24] a take that couldn\'t start because a recording ran shows no first-audio time when it is played later', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    t.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    await generate('Played much later.');
    const take = t.parts.takes.all()[0];
    assert.equal(take.hint, 'Stop recording to listen.');
    t.app.set({ recording: null });
    await t.env.clock.tickAsync(90000);
    t.env.click(takeEls()[0].querySelector('[data-take-act="play"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('app-audio').paused, false, 'it plays now');
    assert.equal(take.ttfaMs, null, 'a tap 90 s later is not "first audio"');
    assert.equal(takeEls()[0].querySelector('.sp-take__perf').textContent.includes('First audio'), false);
  });

  it('[SP-24] first audio, real-time factor, device and Cached are measured and shown; slow models are tagged', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => delay(100, bytes(200, speechBytes(), 'audio/mpeg')) } });
    await ready(t);
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ length: 48000 * 2, sampleRate: 48000 });
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Timed.');
    t.env.click(t.$('sp-generate'));
    await t.env.clock.tickAsync(100);
    await settle(t.env);
    await settle(t.env);
    const take = t.parts.takes.all()[0];
    assert.equal(Math.round(take.ttfaMs), 100);
    assert.equal(Math.round(take.renderMs), 100);
    assert.equal(take.rtf.toFixed(2), '0.05');
    const perf = takeEls()[0].querySelector('.sp-take__perf').textContent;
    assert.ok(perf.includes('First audio 100 ms · RTF 0.05 · GPU'), perf);
    assert.equal(t.$('sp-gen-status-text').textContent, 'Done', 'announced once');
    assert.equal(t.$('sp-gen-status').textContent, 'Done · RTF 0.05', 'the factor joins the line on screen');
    const stored = JSON.parse(localStorage.getItem('open-speech-perf')).kokoro;
    assert.equal(stored.ttfaMs, 100);
    assert.equal(stored.durationS, 2);
    assert.match(takeEls()[0].querySelector('.sp-take__meta').textContent, /^Heart · Kokoro · 1\.1?× · WAV · 6 chars$|· WAV · 6 chars$/);
    // Slower than real time
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ length: 4800, sampleRate: 48000 });
    type(t, 'Slow.');
    t.env.click(t.$('sp-generate'));
    await t.env.clock.tickAsync(100);
    await settle(t.env);
    await settle(t.env);
    assert.ok(takeEls()[0].querySelector('.badge--warn').textContent.includes('Slower than real time'));
  });

  it('[SP-24] a take whose model loaded for it shows the device it rendered on, though the model had none at the click', async () => {
    const models = modelsPayload({ kokoroState: 'downloaded' });
    models.models[0] = { ...models.models[0], device: null };   // not loaded yet: the server reports no device
    let loaded = false;
    const kokoro = () => ({ ...models.models[0], state: loaded ? 'loaded' : 'downloaded', device: loaded ? 'cuda:0' : null });
    t = await mountSpeak({
      models,
      routes: {
        'GET /api/models': () => json(200, { ...models, models: [kokoro(), ...models.models.slice(1)] }),
        'GET /api/models/:id/status': (req) => json(200, req.params.id === 'kokoro' ? { id: 'kokoro', state: kokoro().state, device: kokoro().device } : { id: req.params.id, state: 'downloaded' }),
        'POST /api/models/:id/load': () => { loaded = true; return json(200, { status: 'loaded' }); },
      },
    });
    await ready(t);
    assert.equal(t.$('sp-model-hint').textContent.trim(), 'Loads when you generate');
    t.env.change(t.$('sp-format'), 'wav');
    await generate('Load, then speak.');
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /api/models/:id/load').length, 1, 'the model loaded for this take');
    const take = t.parts.takes.all()[0];
    assert.equal(take.device, 'GPU');
    assert.match(takeEls()[0].querySelector('.sp-take__perf').textContent, /· GPU$/);
  });

  it('[SP-24] a take\'s length and RTF come from its own audio, never from the 0.1 s clip the click primed', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => delay(100, bytes(200, speechBytes(), 'audio/mpeg')) } });
    await ready(t);
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ length: 48000 * 2, sampleRate: 48000 });
    t.env.change(t.$('sp-format'), 'wav');
    type(t, 'Timed.');
    const audioEl = t.$('app-audio');
    t.env.click(t.$('sp-generate'));
    audioEl.__loadMetadata(0.1);   // the primed silent clip (a data: URL) has its metadata at once
    await t.env.clock.tickAsync(100);
    await settle(t.env);
    await settle(t.env);
    const take = t.parts.takes.all()[0];
    assert.equal(take.durationS, 2, 'the decoded length, not the primed clip\'s');
    assert.equal(takeEls()[0].querySelector('.sp-take__time').textContent, '0:02');
    assert.equal(take.rtf.toFixed(2), '0.05');
    assert.equal(JSON.parse(localStorage.getItem('open-speech-perf')).kokoro.durationS, 2, 'and so is the speed record');
  });

  it('[SP-24] a cached answer (X-Cache: HIT) is tagged Cached and left out of the speed record', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => bytes(200, speechBytes(), 'audio/mpeg', { 'X-Cache': 'HIT' }) } });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    await generate('Cached.');
    assert.ok(takeEls()[0].querySelector('.sp-take__perf').textContent.includes('Cached'));
    assert.equal(localStorage.getItem('open-speech-perf'), null);
    assert.match(t.$('sp-gen-status-text').textContent, /^Done · cached/);
  });

  it('[REV-33] a take in a format this browser can\'t play is downloaded with a hint; PCM is never played', async () => {
    t = await mountSpeak();
    await ready(t);
    window.__canPlayTypes = { 'audio/mpeg': 'probably', 'audio/wav': 'probably' };
    const downloads = captureDownloads(t.env);
    t.env.change(t.$('sp-format'), 'opus');
    await settle(t.env);
    assert.equal(t.$('sp-format-hint').hidden, false);
    assert.equal(t.$('sp-format-hint').textContent, "This browser can't play Opus, so takes download instead.");
    await generate('Opus here.');
    assert.equal(downloads.length, 1, 'downloaded instead');
    assert.match(downloads[0].download, /\.opus$/);
    const el = takeEls()[0];
    assert.equal(el.querySelector('.sp-take__hint').textContent, "This browser can't play Opus, so it was downloaded.");
    assert.equal(el.querySelector('[data-take-act="play"]').hidden, true);
    assert.equal(t.app.get().player, null, 'nothing played');
    t.env.change(t.$('sp-format'), 'pcm');
    await generate('Raw PCM.');
    assert.equal(downloads.length, 2);
    assert.match(downloads[1].download, /\.pcm$/);
    assert.equal(takeEls()[0].querySelector('.sp-take__hint').textContent, 'PCM is raw audio, so it was downloaded.');
    assert.equal(t.app.get().player, null);
    delete window.__canPlayTypes;
  });

  it('takes stack newest first, at most 10; a removed take revokes its object URL and focus moves on', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    for (let i = 1; i <= MAX_TAKES + 1; i += 1) await generate(`Take number ${i}.`);
    const all = t.parts.takes.all();
    assert.equal(all.length, MAX_TAKES);
    assert.equal(all[0].n, 11, 'newest first');
    assert.equal(takeEls()[0].querySelector('.sp-take__title').textContent, '“Take number 11.”');
    assert.equal(t.$('sp-takes-empty').hidden, true);
    const second = all[1];
    const url = second.url;
    const el = takeEls()[1];
    t.env.click(el.querySelector('[aria-haspopup="menu"]'));
    await settle(t.env);
    t.env.click(el.parentElement.querySelector('[data-take-menu="remove"]'));
    await settle(t.env);
    assert.ok(t.env.revokedUrls.includes(url));
    assert.equal(t.parts.takes.all().length, MAX_TAKES - 1);
    assert.equal(document.activeElement, takeEls()[1].querySelector('[data-take-act="play"]'), 'focus moves to the next take');
  });

  it('Use these settings brings a take\'s model, voice and delivery back', async () => {
    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    t.env.input(t.$('sp-speed'), '1.5');
    await generate('Remember me.');
    t.env.input(t.$('sp-speed'), '0.8');
    t.env.change(t.$('sp-format'), 'flac');
    await settle(t.env);
    const el = takeEls()[0];
    t.env.click(el.querySelector('[aria-haspopup="menu"]'));
    await settle(t.env);
    t.env.click(el.parentElement.querySelector('[data-take-menu="use"]'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.inspector.state().speed, 1.5);
    assert.equal(t.$('sp-format').value, 'wav');
  });
});
