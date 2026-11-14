// Voices: the 3-step clone wizard at #voices/clone (inventory VL-01, VL-02, VL-04…VL-10,
// defect REV-05, and the record start guard of BUG-08).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, accessibleName } from '../harness/a11y.js';
import { FakeOfflineAudioContext, FakeAudioBuffer } from '../harness/fakes.js';
import { voicesEnv, MODELS, recordings, json, bytes, load, delay } from './helpers.js';

let ctx;
afterEach(() => {
  FakeOfflineAudioContext.decodeResult = null;
  ctx?.env.restore();
});

/** The pre-save test speaks of the new recording: nothing is saved yet (the Recordings cards say "saved"). */
const WIZARD_SENTENCE = 'This is an Open Speech voice-cloning test using your new recording.';
const SCRIPT = 'Today I am recording a clear reference for Open Speech. My voice should sound natural, steady, and easy to understand. I will speak at my normal pace, with a short pause between sentences. This sample includes numbers like seven and forty-two, plus questions, statements, and a little expression.';
const $ = (id) => document.getElementById(id);
const text = (el) => (el ? el.textContent.replace(/\s+/g, ' ').trim() : '');
const status = () => text($('vo-status'));

async function wizard(options = {}) {
  ctx = await voicesEnv({ sub: 'clone', ...options });
  return ctx;
}

/** Click Record and let the microphone open. */
async function record() {
  ctx.env.click($('vo-record'));
  await ctx.settle();
  await ctx.settle();
}

/** Deliver `seconds` of frames at `value` through the AudioWorklet. */
function frames(seconds, value = 0.1, rate = 48000) {
  const node = ctx.env.audioContexts.at(-1).__worklets[0];
  const total = Math.round(seconds * rate);
  for (let sent = 0; sent < total; sent += 2048) {
    const n = Math.min(2048, total - sent);
    const frame = new Float32Array(n);
    for (let i = 0; i < n; i += 1) frame[i] = value * Math.sin((sent + i) / 7);
    node.__emit(frame);
  }
}

/** A recorded draft of `seconds`. */
async function recordDraft(seconds = 4) {
  await record();
  frames(seconds);
  ctx.env.click($('vo-record'));
  await ctx.settle();
}

/** Upload a file through the hidden input (the decoder answers with `buffer`). */
async function upload(buffer, file = new File([new Uint8Array([1, 2, 3, 4])], 'clip.mp3', { type: 'audio/mpeg' })) {
  FakeOfflineAudioContext.decodeResult = buffer;
  const input = $('vo-file');
  input.files = [file];
  input.dispatchEvent(new Event('change', { bubbles: true }));
  await ctx.settle();
  await ctx.settle();
}

/** A WAV file whose fmt chunk declares `rate` (a LIST chunk first, as some recorders write). */
function wavFile(rate) {
  const bytes = new Uint8Array(80);
  const view = new DataView(bytes.buffer);
  const tag = (offset, s) => { for (let i = 0; i < 4; i += 1) bytes[offset + i] = s.charCodeAt(i); };
  tag(0, 'RIFF'); view.setUint32(4, 72, true); tag(8, 'WAVE');
  tag(12, 'LIST'); view.setUint32(16, 4, true); tag(20, 'INFO');
  tag(24, 'fmt '); view.setUint32(28, 16, true); view.setUint16(32, 1, true); view.setUint16(34, 1, true);
  view.setUint32(36, rate, true); view.setUint32(40, rate * 2, true); view.setUint16(44, 2, true); view.setUint16(46, 16, true);
  tag(48, 'data'); view.setUint32(52, 24, true);
  return new File([bytes], 'clip.wav', { type: 'audio/wav' });
}

function stereo(seconds, rate = 48000) {
  const length = Math.round(seconds * rate);
  const left = new Float32Array(length).map((_, i) => 0.2 * Math.sin(i / 9));
  const right = new Float32Array(length).map((_, i) => 0.1 * Math.sin(i / 5));
  return new FakeAudioBuffer({ numberOfChannels: 2, length, sampleRate: rate, channels: [left, right] });
}

async function toStep(n) {
  ctx.env.click(ctx.$(`.vo-steps [data-step="${n}"]`));
  await ctx.settle();
}

describe('views/voices clone wizard: capture', () => {
  it('[VL-01] Record opens the microphone (AudioWorklet), shows the timer and level, and Stop makes the clip the draft', async () => {
    await wizard();
    const rec = $('vo-record');
    assert.equal(text(rec), 'Record');
    assert.equal(text($('vo-script')), SCRIPT, 'the script card shows the reference paragraph');
    await record();
    assert.equal(text(rec), 'Stop');
    assert.equal(ctx.app.get().recording.owner, 'voices', 'the status chip shows Recording');
    assert.equal(status(), 'Recording… 60-second maximum.');
    assert.equal(ctx.env.audioContexts.at(-1).__modules[0], '/static/js/core/mic-worklet.js');
    assert.equal($('vo-rec-dot').hidden, false);
    frames(2.2);
    ctx.env.clock.tick(250);
    assert.equal(text($('vo-timer-text')), '0:02');
    assert.equal($('vo-timer').getAttribute('aria-hidden'), 'true', 'the ticking timer is never announced');
    ctx.env.clock.tick(100);
    assert.equal($('vo-meter').getAttribute('data-zone'), 'good');
    ctx.env.click(rec);
    await ctx.settle();
    assert.equal(ctx.app.get().recording, null);
    assert.ok(ctx.env.mediaDevices.streams[0].getTracks().every((t) => t.readyState === 'ended'), 'the tracks are released');
    assert.equal($('vo-draft').hidden, false);
    assert.match(text($('vo-draft-meta')), /^2\.2 s · 48 kHz · mono · /);
    assert.match($('vo-draft-audio').src, /^blob:/);
    assert.equal(status(), 'Recording stopped. 2.2 s recorded. Check the quality below.');
    assert.equal(ctx.$$('#vo-quality .vo-quality__item').length, 5, 'quality checks follow the capture');
    assert.equal(text(rec), 'Record again');
    assert.equal($('vo-next-1').getAttribute('aria-disabled'), null, 'Next is available once there is a draft');
    assert.ok($('vo-next-1').classList.contains('btn--primary'), 'the primary action moves to Next');
    assertAccessibleNames(ctx.section);
  });

  it('[VL-01] recording stops by itself at the server maximum, trimmed to the cap', async () => {
    await wizard({ routes: { 'GET /api/voices/library-config': () => json(200, { max_seconds: 1 }) } });
    await record();
    assert.equal(status(), 'Recording… 1-second maximum.');
    frames(1.5);
    await ctx.settle();
    assert.equal(ctx.app.get().recording, null, 'auto-stop');
    assert.match(text($('vo-draft-meta')), /^1\.0 s · /, 'the clip is trimmed to exactly the cap');
    assert.match(status(), /^Recording stopped at the 1-second maximum\./);
  });

  it('[VL-01] the optional 3-2-1 countdown is announced once, shows the level, and can be cancelled', async () => {
    await wizard();
    ctx.env.change($('vo-countdown'), true);
    await record();
    assert.equal(status(), 'Recording starts in 3 seconds.');
    assert.equal(text($('vo-countdown-num')), '3');
    assert.equal($('vo-countdown-num').getAttribute('aria-hidden'), 'true');
    assert.equal(text($('vo-record')), 'Cancel');
    frames(0.5, 0.3);
    ctx.env.clock.tick(100);
    assert.notEqual($('vo-meter').getAttribute('data-zone'), 'idle', 'the meter works during the countdown');
    ctx.env.clock.tick(1000);
    assert.equal(text($('vo-countdown-num')), '2');
    assert.equal(status(), 'Recording starts in 3 seconds.', 'the status does not tick');
    ctx.env.clock.tick(2000);
    assert.equal(status(), 'Recording… 60-second maximum.');
    frames(1);
    ctx.env.click($('vo-record'));
    await ctx.settle();
    assert.match(text($('vo-draft-meta')), /^1\.0 s/, 'frames before the count reached zero are not in the clip');
    await record();
    ctx.env.click($('vo-record'));
    await ctx.settle();
    assert.equal(status(), 'Recording canceled.');
    assert.equal(ctx.app.get().recording, null);
  });

  it('[VL-01] a refused microphone shows the reason inline; without HTTPS recording is unavailable but uploads work', async () => {
    await wizard({ mediaDevices: { policy: 'NotAllowedError' } });
    await record();
    const banner = $('vo-capture-error').querySelector('.banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(text(banner.querySelector('.banner__title')), 'Microphone problem');
    assert.match(text(banner.querySelector('.banner__msg')), /blocked/);
    assert.equal(text($('vo-record')), 'Record');
    ctx.env.restore();
    await wizard({ secure: false });
    assert.equal($('vo-mic-help').hidden, false);
    assert.match(text($('vo-mic-help')), /secure connection \(HTTPS\)/);
    assert.equal($('vo-record').getAttribute('aria-disabled'), 'true');
    assert.equal($('vo-mic').disabled, true);
    await upload(stereo(4));
    assert.match(text($('vo-draft-meta')), /^4\.0 s/, 'an upload still makes a draft');
  });

  it('[VL-02] recording is refused while Transcribe holds the microphone', async () => {
    await wizard();
    const mic = await load('core/mic.js');
    const other = await mic.open({ owner: 'transcribe' });
    await record();
    assert.equal(text($('vo-capture-error').querySelector('.banner__msg')), 'Stop the Transcribe recording first');
    assert.equal(mic.owner(), 'transcribe', "Voices never touches Transcribe's capture");
    other.stop();
  });

  it('[VL-01][BUG-08] a second press while the microphone starts is ignored (the start guard: no second stream)', async () => {
    await wizard();
    ctx.env.click($('vo-record'));
    assert.equal($('vo-record').getAttribute('aria-busy'), 'true');
    assert.equal(text($('vo-record')), 'Starting…');
    ctx.env.click($('vo-record'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.env.mediaDevices.calls.length, 1, 'one getUserMedia, no orphaned stream');
    assert.equal(ctx.env.audioContexts.length, 1);
    ctx.env.click($('vo-record'));
    await ctx.settle();
  });

  it('[VL-06] the maximum comes from /api/voices/library-config: 0 means no limit, 60 when the route is missing', async () => {
    await wizard({ routes: { 'GET /api/voices/library-config': () => json(200, { max_seconds: 0 }) } });
    assert.match(text($('vo-capture-hint')), /^No duration limit\./);
    await record();
    assert.equal(status(), 'Recording… No duration limit.');
    assert.equal(text($('vo-timer-max')), '');
    ctx.env.click($('vo-record'));
    await ctx.settle();
    ctx.env.restore();
    await wizard({ routes: { 'GET /api/voices/library-config': () => json(404, { detail: 'Not Found' }) } });
    assert.match(text($('vo-capture-hint')), /^60-second maximum\./);
    assert.equal(text($('vo-timer-max')), '/ 1:00');
  });

  it('[VL-06] the maximum is asked for each time the wizard opens, so the 60 s fallback of a failed request never sticks', async () => {
    let answer = () => json(503, { error: { message: 'Starting up' } });
    await wizard({ routes: { 'GET /api/voices/library-config': () => answer() } });
    assert.match(text($('vo-capture-hint')), /^60-second maximum\./, 'a failed request (offline, no API key yet) falls back to 60 s');
    answer = () => json(200, { max_seconds: 120 });
    await ctx.go('voices/recordings');
    await ctx.go('voices/clone');
    assert.equal(ctx.calls('GET /api/voices/library-config').length, 2, 'asked again on the next visit');
    assert.match(text($('vo-capture-hint')), /^120-second maximum\./);
    assert.equal(text($('vo-timer-max')), '/ 2:00');
    await record();
    assert.equal(status(), 'Recording… 120-second maximum.');
    frames(61);
    await ctx.settle();
    assert.equal(ctx.app.get().recording.owner, 'voices', 'past 60 s the recording goes on: the server allows 120');
    ctx.env.click($('vo-record'));
    await ctx.settle();
    assert.match(text($('vo-draft-meta')), /^61\.0 s/, 'and the clip is not refused as too long');
    assert.equal($('vo-capture-error').hidden, true);
  });

  it('[VL-01] Clone a voice opens the wizard with focus on its heading (the header button hides there)', async () => {
    ctx = await voicesEnv({ sub: 'recordings' });
    const clone = $('vo-clone');
    clone.focus();
    ctx.env.click(clone);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'voices/clone');
    assert.equal(clone.hidden, true);
    assert.equal(document.activeElement?.id, 'vo-wizard-title', 'the wizard\'s own heading, not the view\'s h1 above the back button');
    assert.equal(text(document.activeElement), 'Clone a voice');
  });

  it('[VL-04] an upload is decoded to a mono WAV draft; an undecodable or empty file says so; it stops a running recording', async () => {
    await wizard();
    await record();
    frames(1);
    await upload(stereo(4));
    assert.equal(ctx.app.get().recording, null, 'the upload stopped the recording');
    assert.match(text($('vo-draft-meta')), /^4\.0 s · 48 kHz · mono · /);
    assert.equal(status(), 'Uploaded 4.0 s. Check the quality below, then type its exact words.');
    await upload(new Error('EncodingError'));
    const banner = $('vo-capture-error').querySelector('.banner');
    assert.equal(text(banner.querySelector('.banner__title')), "Couldn't use that file");
    assert.equal(text(banner.querySelector('.banner__msg')), 'This browser could not decode that file. Upload WAV, MP3, M4A, FLAC, OGG, or WebM.');
    await upload(stereo(1), new File([], 'empty.wav', { type: 'audio/wav' }));
    assert.equal(text($('vo-capture-error').querySelector('.banner__msg')), 'Choose a non-empty audio file');
    assert.match(text($('vo-draft-meta')), /^4\.0 s/, 'a failed upload keeps the previous draft');
  });

  it('[VL-04] a WAV decodes at the rate it declares, so the sample-rate check tells the truth; other files at 48 kHz', async () => {
    await wizard();
    const atContextRate = () => {
      const rate = FakeOfflineAudioContext.instances.at(-1).sampleRate;
      return new FakeAudioBuffer({ length: rate * 4, sampleRate: rate, channels: [new Float32Array(rate * 4).map((_, i) => 0.1 * Math.sin(i / 3))] });
    };
    await upload(atContextRate, wavFile(8000));
    assert.equal(FakeOfflineAudioContext.instances.at(-1).sampleRate, 8000, 'decoded at the file\'s own 8 kHz, not upsampled');
    assert.match(text($('vo-draft-meta')), /^4\.0 s · 8 kHz · mono/);
    const rate = ctx.$$('#vo-quality .vo-quality__item').find((li) => /Sample rate/.test(text(li)));
    assert.equal(rate.getAttribute('data-status'), 'warn', 'a low-rate clip is flagged');
    assert.match(text(rate), /8 kHz/);
    await upload(atContextRate);   // an MP3: its rate is not read, the standard decode applies
    assert.equal(FakeOfflineAudioContext.instances.at(-1).sampleRate, 48000);
    // A rate this browser refuses falls back to the standard decode instead of failing.
    await upload((data) => {
      if (FakeOfflineAudioContext.instances.at(-1).sampleRate !== 48000) return new Error('NotSupportedError');
      return atContextRate(data);
    }, wavFile(22050));
    assert.match(text($('vo-draft-meta')), /^4\.0 s · 48 kHz/);
    assert.equal($('vo-capture-error').hidden, true, 'no error for a file that decodes the second way');
    const { wavSampleRate } = await load('views/voices/clone.js');
    assert.equal(wavSampleRate(new ArrayBuffer(10)), 0);
    assert.equal(wavSampleRate(new TextEncoder().encode('ID3\u0003 not a wav file at all, just bytes').buffer), 0);
  });

  it('[VL-01] Record before the device list has loaded uses the saved microphone, not the default one', async () => {
    ctx = await voicesEnv({ sub: 'named', seed: { 'open-speech-mic-device': 'mic-b' } });
    ctx.env.mediaDevices.enumerateDevices = () => new Promise(() => {});   // the list never arrives
    await ctx.go('voices/clone');
    await record();
    const audio = ctx.env.mediaDevices.calls[0].audio;
    assert.deepEqual(audio.deviceId, { exact: 'mic-b' }, 'the saved device is asked for');
    ctx.env.click($('vo-record'));
    await ctx.settle();
  });

  it('[VL-04] a file dropped on the dropzone is used like a chosen one', async () => {
    await wizard();
    FakeOfflineAudioContext.decodeResult = stereo(3);
    const drop = new Event('drop', { bubbles: true, cancelable: true });
    drop.dataTransfer = { files: [new File([new Uint8Array([9, 9])], 'take.wav', { type: 'audio/wav' })] };
    $('vo-dropzone').dispatchEvent(drop);
    await ctx.settle();
    await ctx.settle();
    assert.equal(drop.defaultPrevented, true);
    assert.match(text($('vo-draft-meta')), /^3\.0 s/);
  });

  it('[VL-05] a clip longer than the maximum is refused with the server\'s wording; a new draft resets the transcript checks', async () => {
    await wizard();
    await upload(stereo(61));
    assert.equal(text($('vo-capture-error').querySelector('.banner__msg')), 'Reference audio is too long (61.00s). Max: 60s');
    assert.equal($('vo-draft').hidden, true);
    await recordDraft(4);
    await toStep(2);
    assert.equal($('vo-transcript').value, SCRIPT, 'a recording keeps the read-aloud script');
    ctx.env.change($('vo-confirm'), true);
    ctx.env.change($('vo-skip'), true);
    await toStep(1);
    await upload(stereo(5));
    assert.equal($('vo-confirm').checked, false, 'a new draft un-confirms');
    assert.equal($('vo-skip').checked, false, 'and un-skips');
    assert.equal($('vo-transcript').value, '', 'an upload clears the words');
  });
});

describe('views/voices clone wizard: verify and save', () => {
  it('[VL-07] the name is sanitized to a-z, 0-9 and _ (spaces and hyphens become _), at most 64, shown as "Saves as"', async () => {
    const { sanitizeVoiceName } = await load('views/voices/clone.js');
    assert.equal(sanitizeVoiceName("  Alex's Voice-2! "), 'alexs_voice_2');
    assert.equal(sanitizeVoiceName('Ünïcode only'), 'ncode_only');
    assert.equal(sanitizeVoiceName('x'.repeat(80)).length, 64);
    assert.equal(sanitizeVoiceName('!!!'), '');
    await wizard();
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Reference-2');
    await ctx.settle();
    assert.equal(text($('vo-name-preview')), 'Saves as: alex_reference_2');
    assert.match($('vo-name').getAttribute('aria-describedby'), /vo-name-preview/);
    ctx.env.input($('vo-name'), 'demo reference');
    await ctx.settle();
    assert.equal(text($('vo-name-preview')), 'Saves as: demo_reference · replaces the recording with this name');
  });

  it('[VL-08] Save needs a draft, a name and a confirmed transcript (or Save without a transcript); editing the words un-confirms', async () => {
    const { saveGate } = await load('views/voices/clone.js');
    assert.equal(saveGate({ draft: null, name: 'a', transcript: 'x', confirmed: true }).ok, false);
    assert.equal(saveGate({ draft: {}, name: '!!', transcript: 'x', confirmed: true }).ok, false);
    assert.equal(saveGate({ draft: {}, name: 'a', transcript: 'x', confirmed: false }).ok, false);
    assert.equal(saveGate({ draft: {}, name: 'a', transcript: ' ', confirmed: true }).ok, false);
    assert.equal(saveGate({ draft: {}, name: 'a', transcript: '', skipped: true }).ok, true);
    assert.equal(saveGate({ draft: {}, name: 'a', transcript: 'x', confirmed: true }).ok, true);
    await wizard();
    assert.equal(ctx.$('.vo-steps [data-step="2"]').getAttribute('aria-disabled'), 'true', 'no draft: later steps are locked');
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex');
    ctx.env.click($('vo-next-2'));
    await ctx.settle();
    assert.equal(ctx.$('#vo-step-3').hidden, true, 'an unconfirmed transcript stops at step 2');
    assert.equal($('vo-confirm').getAttribute('aria-invalid'), 'true');
    assert.match(text($('vo-confirm-error')), /Confirm that the transcript matches/);
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    ctx.env.input($('vo-transcript'), `${SCRIPT} Extra.`);
    await ctx.settle();
    assert.equal($('vo-confirm').checked, false, 'editing the words un-confirms them');
    ctx.env.change($('vo-skip'), true);
    await ctx.settle();
    assert.equal($('vo-transcript').disabled, true, 'skipping disables the transcript and its confirmation');
    assert.equal($('vo-confirm').disabled, true);
    ctx.env.click($('vo-next-2'));
    await ctx.settle();
    assert.equal(ctx.$('#vo-step-3').hidden, false);
    assert.equal($('vo-save').getAttribute('aria-disabled'), null);
    assert.equal(ctx.$('.vo-steps [data-step="3"]').getAttribute('aria-current'), 'step');
    assert.equal(document.activeElement?.id, 'vo-step-3-title', 'focus moves to the new step');
  });

  it('[VL-08] a completed step shows a check for its number and its name says "done" (never colour alone)', async () => {
    await wizard();
    const step = (n) => ctx.$(`.vo-steps [data-step="${n}"]`);
    const names = () => [1, 2, 3].map((n) => accessibleName(step(n)));
    const done = () => [1, 2, 3].filter((n) => step(n).closest('li').classList.contains('is-done'));
    assert.deepEqual(names(), ['Capture', 'Verify words', 'Test & save']);
    assert.ok([1, 2, 3].every((n) => step(n).querySelector('.vo-steps__check')), 'each step has a check to show when it is done');
    await recordDraft(4);
    await toStep(2);
    assert.deepEqual(names(), ['Capture, done', 'Verify words', 'Test & save']);
    assert.deepEqual(done(), [1]);
    ctx.env.input($('vo-name'), 'n');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    assert.deepEqual(names(), ['Capture, done', 'Verify words, done', 'Test & save']);
    assert.deepEqual(done(), [1, 2]);
    assert.equal(step(3).getAttribute('aria-current'), 'step');
    await toStep(1);
    assert.deepEqual(names(), ['Capture', 'Verify words', 'Test & save'], 'done means before the current step');
    assert.deepEqual(done(), []);
    assertAccessibleNames(ctx.section);
  });

  it('[VL-09][TE-09] Save checks the name, POSTs name, <name>.wav and the transcript, then shows the new card in Recordings', async () => {
    let library = recordings();
    await wizard({
      routes: {
        'GET /api/voices/library': () => json(200, library),
        'GET /api/voices/library/:name': () => json(404, { detail: "Voice 'x' not found" }),
        'POST /api/voices/library': (req) => {
          library = [...library, { name: req.body.get('name'), duration_s: 4, sample_rate: 48000, channels: 1, transcript: req.body.get('transcript') }];
          return json(201, { name: req.body.get('name') });
        },
      },
    });
    const events = [];
    ctx.store.on('voices:changed', (d) => events.push(d.what));
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Ref');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-save'));
    await ctx.settle();
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('GET /api/voices/library/:name')[0].path, '/api/voices/library/alex_ref', 'the existing name is checked first');
    const post = ctx.calls('POST /api/voices/library')[0];
    assert.equal(post.body.get('name'), 'alex_ref');
    assert.equal(post.body.get('audio').name, 'alex_ref.wav');
    assert.equal(post.body.get('audio').type, 'audio/wav');
    assert.equal(post.body.get('transcript'), SCRIPT);
    assert.ok(ctx.toasts().includes('Saved recording alex_ref'));
    assert.deepEqual(events, ['recordings']);
    assert.equal(ctx.app.get().route.path, 'voices/recordings');
    assert.ok(document.getElementById('vo-rec-alex_ref'), 'the new card is listed');
    assert.equal(document.activeElement?.id, 'vo-rec-alex_ref-title', 'and focused');
    // A deliberate change from the old Voice Lab: saving no longer selects the recording in Speak.
    // Speak's reference list refreshes on the event above; the new card's Use in Speak hands it over.
    assert.equal(ctx.store.takeIntent('speak'), null, "Speak's selection is left alone");
    assert.equal(text(document.getElementById('vo-rec-alex_ref-use')), 'Use in Speak');
  });

  it('[VL-09] an existing name asks before replacing it; Save without a transcript sends none', async () => {
    await wizard({
      routes: {
        'GET /api/voices/library/:name': (req) => json(200, { name: req.params.name }),
        'POST /api/voices/library': () => json(201, {}),
      },
    });
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'demo_reference');
    ctx.env.change($('vo-skip'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-save'));
    await ctx.settle();
    await ctx.settle();
    assert.deepEqual([ctx.confirmText().title, ctx.confirmText().message], ['Replace the existing recording “demo_reference”?', 'Its audio and transcript will be overwritten.']);
    await ctx.answerConfirm(false);
    assert.equal(ctx.calls('POST /api/voices/library').length, 0);
    assert.equal(status(), 'Not saved. Choose another name to keep both.');
    ctx.env.click($('vo-save'));
    await ctx.settle();
    await ctx.settle();
    await ctx.answerConfirm(true);
    const post = ctx.calls('POST /api/voices/library')[0];
    assert.equal(post.body.get('transcript'), null, 'skipped: no transcript field');
  });

  it('[VL-09] a save that finishes after the listener left the wizard keeps them where they went (no route change, no focus move)', async () => {
    let library = recordings();
    await wizard({
      routes: {
        'GET /api/voices/library': () => json(200, library),
        'GET /api/voices/library/:name': () => json(404, { detail: 'Not Found' }),
        'POST /api/voices/library': (req) => {
          library = [...library, { name: req.body.get('name'), duration_s: 4, sample_rate: 48000, channels: 1, transcript: req.body.get('transcript') }];
          return delay(3000, json(201, { name: req.body.get('name') }));   // a long upload over phone Wi-Fi
        },
      },
    });
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Slow upload');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-save'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /api/voices/library').length, 1, 'the upload is running');
    await ctx.go('speak');
    const focused = document.activeElement;
    assert.equal(focused?.id, 'sp-title');
    await ctx.settle(3100);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'speak', 'never pulled back to Voices › Recordings');
    assert.equal(document.activeElement, focused, 'focus stays where the listener is');
    assert.ok(ctx.toasts().includes('Saved recording slow_upload'), 'the toast still says it was saved');
    assert.equal(ctx.calls('GET /api/voices/library').length, 2, 'and the recordings reload');
    await ctx.go('voices/recordings');
    assert.ok(document.getElementById('vo-rec-slow_upload'), 'the new card is there on the next visit');
    await ctx.go('voices/clone');
    assert.equal($('vo-draft').hidden, true, 'the wizard starts afresh');
    assert.equal(ctx.$('#vo-step-1').hidden, false);
  });

  it('[VL-10] Suggest transcript loads the STT model, posts the draft as JSON transcription and marks the words unverified', async () => {
    let loaded = false;
    await wizard({
      routes: {
        'GET /api/models/:id/status': () => json(200, { state: loaded ? 'loaded' : 'downloaded' }),
        'POST /api/models/:id/load': () => { loaded = true; return json(200, { status: 'loaded' }); },
        'POST /v1/audio/transcriptions': () => json(200, { text: '  Today I am recording a clear reference.  ' }),
      },
    });
    await recordDraft(4);
    await toStep(2);
    ctx.env.change($('vo-confirm'), true);
    ctx.env.click($('vo-suggest'));
    await ctx.settle();
    await ctx.settle();
    await ctx.settle();
    const load = ctx.calls('POST /api/models/:id/load');
    assert.equal(load[0].path, `/api/models/${encodeURIComponent('Systran/faster-whisper-base')}/load`, 'ensureReady loads the STT model');
    const post = ctx.calls('POST /v1/audio/transcriptions')[0];
    assert.equal(post.body.get('model'), 'Systran/faster-whisper-base');
    assert.equal(post.body.get('response_format'), 'json');
    assert.equal(post.body.get('file').name, 'voice-reference.wav');
    assert.equal($('vo-transcript').value, 'Today I am recording a clear reference.');
    assert.ok($('vo-transcript').classList.contains('is-unverified'));
    assert.equal(text($('vo-transcript-status')), 'Unverified draft. Check every word, then confirm.');
    assert.equal($('vo-confirm').checked, false, 'a suggestion is never confirmed for you');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    assert.equal($('vo-transcript').classList.contains('is-unverified'), false);
  });

  // A suggestion belongs to the clip it was made from (VL-05, VL-10). One that
  // answers after another clip became the draft leaves that clip's words, its status and the status line alone.
  const slowSuggestion = {
    'GET /api/models/:id/status': () => json(200, { state: 'loaded' }),
    'POST /v1/audio/transcriptions': () => delay(3000, json(200, { text: 'Words of the first clip.' })),
  };
  async function suggestThenBackToCapture() {
    await upload(stereo(5));
    await toStep(2);
    ctx.env.click($('vo-suggest'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/transcriptions').length, 1, 'the first clip is being transcribed');
    await toStep(1);
    assert.equal(ctx.$('#vo-step-1').hidden, false);
  }
  async function lateAnswer() {
    await ctx.settle(3100);
    await ctx.settle();
    await ctx.settle();
  }

  it('[VL-05][VL-10] a suggestion that answers after a new upload leaves the new clip\'s empty words alone', async () => {
    await wizard({ routes: slowSuggestion });
    await suggestThenBackToCapture();
    await upload(stereo(7));
    assert.match(text($('vo-draft-meta')), /^7\.0 s/, 'the second clip is the draft');
    assert.equal($('vo-transcript').value, '');
    await lateAnswer();
    assert.equal($('vo-transcript').value, '', 'the first clip\'s words never land on the second');
    assert.equal($('vo-transcript').classList.contains('is-unverified'), false);
    assert.equal(text($('vo-transcript-status')), 'Type the exact words spoken in the clip, or ask for a suggested draft.');
    assert.equal(status(), 'Uploaded 7.0 s. Check the quality below, then type its exact words.');
    assert.equal($('vo-suggest').hasAttribute('aria-busy'), false, 'Suggest is free again');
  });

  it('[VL-05][VL-10] a suggestion that answers after a new recording leaves the read-aloud script alone', async () => {
    await wizard({ routes: slowSuggestion });
    await suggestThenBackToCapture();
    await recordDraft(4);
    assert.match(text($('vo-draft-meta')), /^4\.0 s/, 'the recording is the draft');
    assert.equal($('vo-transcript').value, SCRIPT);
    await lateAnswer();
    assert.equal($('vo-transcript').value, SCRIPT, 'the script that was read aloud stays');
    assert.equal($('vo-transcript').classList.contains('is-unverified'), false);
    assert.notEqual(status(), 'A draft transcript is in. Check every word, then confirm.');
  });

  it('[REV-05] Test before saving POSTs /v1/audio/speech/clone with voice set to the sanitized name, the draft WAV and no speed', async () => {
    await wizard({ routes: { 'POST /v1/audio/speech/clone': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const audio = await load('core/audio.js');
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Ref');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    assert.equal($('vo-test-model').value, 'chatterbox');
    ctx.env.click($('vo-test'));
    await ctx.settle();
    await ctx.settle();
    const post = ctx.calls('POST /v1/audio/speech/clone')[0];
    const form = post.body;
    assert.equal(form.get('voice'), 'alex_ref', 'always sent: the server default "Ryan" may be refused by a clone model');
    assert.equal(form.get('input'), WIZARD_SENTENCE);
    assert.equal(form.get('model'), 'chatterbox');
    assert.equal(form.get('response_format'), 'wav');
    assert.equal(form.get('transcript'), SCRIPT);
    assert.equal(form.get('reference_audio').type, 'audio/wav');
    assert.equal(form.get('speed'), null, 'no speed: the server default 1.0 passes models without speed control');
    assert.equal(audio.current().id, 'clone-test');
    assert.equal(status(), 'Clone test complete with Chatterbox.');
    assert.equal(text($('vo-test')), 'Stop test');
  });

  it('[REV-05] a Test before saving that finishes after a new upload neither plays nor is announced as the new clip\'s test', async () => {
    await wizard({ routes: { 'POST /v1/audio/speech/clone': () => delay(3000, bytes(200, [1, 2, 3], 'audio/wav')) } });
    const audio = await load('core/audio.js');
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Ref');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech/clone').length, 1, 'the first clip\'s test is generating');
    await toStep(1);
    await upload(stereo(7));
    assert.match(text($('vo-draft-meta')), /^7\.0 s/, 'the second clip is the draft');
    await ctx.settle(3100);
    await ctx.settle();
    await ctx.settle();
    assert.equal(audio.current(), null, 'the first clip\'s clone does not play as the second clip\'s test');
    assert.equal(status(), 'Uploaded 7.0 s. Check the quality below, then type its exact words.');
    assert.equal(text($('vo-test')), 'Test before saving', 'the button is free again');
    assert.equal($('vo-test').hasAttribute('aria-busy'), false);
  });

  it('[REV-05] the test sends voice even without a name, and a transcript-required model refuses a skipped transcript', async () => {
    const { cloneTestForm } = await load('views/voices/clone.js');
    const form = cloneTestForm({ model: 'm', name: '', blob: new Blob([new Uint8Array([1])], { type: 'audio/wav' }), skipped: true, transcript: 'words' });
    assert.equal(form.get('voice'), 'reference');
    assert.equal(form.get('transcript'), null);
    await wizard();
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'n');
    ctx.env.change($('vo-skip'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-test'));
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech/clone').length, 0);
    assert.equal(text($('vo-test-error').querySelector('.banner__msg')), 'Chatterbox requires an exact reference transcript');
  });

  it('the test clip stops from its button, and a clip the browser left paused offers Play test (no second synthesis)', async () => {
    await wizard({ routes: { 'POST /v1/audio/speech/clone': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const audio = await load('core/audio.js');
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Ref');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text($('vo-test')), 'Stop test');
    assert.equal(audio.current().homeRoute, 'voices/clone', 'the mini-player stays out of the wizard, which has its own button');
    ctx.env.click($('vo-test'));
    await ctx.settle();
    assert.equal(audio.current(), null, 'Stop test released the clip');
    assert.equal(text($('vo-test')), 'Test before saving');
    audio.element().__playPolicy = 'NotAllowedError';
    ctx.env.click($('vo-test'));
    await ctx.settle();
    await ctx.settle();
    assert.equal(text($('vo-test')), 'Play test', 'autoplay refused: the clip waits, and the button says so');
    audio.element().__playPolicy = 'resolve';
    ctx.env.click($('vo-test'));
    await ctx.settle();
    assert.equal(text($('vo-test')), 'Stop test');
    assert.equal(ctx.calls('POST /v1/audio/speech/clone').length, 2, 'Play test plays the clip it already has');
  });

  it('[VL-13] "Test with" lists the clone models and keeps the listener\'s choice when the models refresh', async () => {
    const withFish = {
      ...MODELS,
      models: [...MODELS.models, { id: 'fish', type: 'tts', provider: 'fish', state: 'downloaded', provider_available: true, capabilities: { voice_clone: true, speed: true } }],
    };
    await wizard({ models: withFish });
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'n');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    const select = $('vo-test-model');
    assert.deepEqual([...select.querySelectorAll('option')].map((o) => o.value), ['chatterbox', 'fish']);
    ctx.env.change(select, 'fish');
    await ctx.settle();
    ctx.app.set({ models: { ...ctx.app.get().models, list: withFish.models.slice().reverse() } });
    await ctx.settle();
    assert.equal(select.value, 'fish', 'a models refresh never resets the pick');
  });

  it('[VL-02] while Transcribe records, Test before saving and the clip player say to stop recording first, and a press sends nothing', async () => {
    await wizard({ routes: { 'POST /v1/audio/speech/clone': () => bytes(200, [1, 2, 3], 'audio/wav') } });
    const audio = await load('core/audio.js');
    const mic = await load('core/mic.js');
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'Alex Ref');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    // A Speak take plays, then Transcribe starts recording in the background (opening the microphone pauses it).
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:take1' });
    const capture = await mic.open({ owner: 'transcribe' });
    await ctx.settle();
    const test = $('vo-test');
    const note = $('vo-test-note');
    assert.equal(test.getAttribute('aria-disabled'), 'true');
    assert.equal(test.getAttribute('aria-describedby'), 'vo-test-note', 'the visible note says why');
    assert.equal(note.hidden, false);
    assert.equal(text(note), 'Stop recording to listen.');
    ctx.env.click(test);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech/clone').length, 0, 'no synthesis that could not be heard');
    assert.equal(audio.current()?.id, 'take:1', 'the paused take keeps its place in the mini-player (prime() never ran)');
    await toStep(1);
    assert.equal($('vo-draft-note').hidden, false, 'the clip player says the same');
    assert.equal($('vo-draft-audio').getAttribute('aria-describedby'), 'vo-draft-note');
    capture.stop();
    await ctx.settle();
    assert.equal($('vo-draft-note').hidden, true);
    assert.equal($('vo-draft-audio').getAttribute('aria-describedby'), null);
    await toStep(3);
    assert.equal(test.getAttribute('aria-disabled'), null, 'available again once the recording stops');
    assert.equal(test.getAttribute('aria-describedby'), null);
    assert.equal(note.hidden, true);
    ctx.env.click(test);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.calls('POST /v1/audio/speech/clone').length, 1);
    assert.equal(audio.current()?.id, 'clone-test');
  });

  it('[REV-05] a failed test shows the server\'s reason inline', async () => {
    await wizard({ routes: { 'POST /v1/audio/speech/clone': () => json(400, { error: { message: "Voice 'Ryan' is not available for this model" } }) } });
    await recordDraft(4);
    await toStep(2);
    ctx.env.input($('vo-name'), 'n');
    ctx.env.change($('vo-confirm'), true);
    await ctx.settle();
    await toStep(3);
    ctx.env.click($('vo-test'));
    await ctx.settle();
    await ctx.settle();
    const banner = $('vo-test-error').querySelector('.banner');
    assert.equal(text(banner.querySelector('.banner__title')), 'Clone test failed');
    assert.equal(text(banner.querySelector('.banner__msg')), "Voice 'Ryan' is not available for this model");
    assert.equal(text($('vo-test')), 'Test before saving');
  });

  it('[VL-10] without a speech-to-text model, Suggest says so inline; the banner\'s Open Models button is the action itself (data-action) and opens the STT models', async () => {
    const noStt = { ...MODELS, models: MODELS.models.filter((m) => m.type !== 'stt'), default_stt_model: null };
    await wizard({ models: noStt });
    await recordDraft(4);
    await toStep(2);
    ctx.env.click($('vo-suggest'));
    await ctx.settle();
    const banner = $('vo-suggest-error').querySelector('.banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(text(banner.querySelector('.banner__title')), "Couldn't suggest a transcript");
    const open = banner.querySelector('.banner__actions button');
    assert.equal(open.getAttribute('data-action'), 'open-models-stt', 'the button names its action; no dispatcher in between');
    assert.equal(accessibleName(open), 'Open Models');
    assert.equal(ctx.calls('POST /v1/audio/transcriptions').length, 0, 'nothing is sent without a model');
    ctx.env.click(open);
    await ctx.settle();
    await ctx.settle();
    assert.equal(ctx.app.get().route.path, 'models/stt', 'the speech-to-text models, where one can be downloaded');
  });

  it('[VL-01] leaving the wizard while recording keeps what was recorded', async () => {
    await wizard();
    await record();
    frames(3);
    await ctx.go('voices/recordings');
    assert.equal(ctx.app.get().recording, null, 'the microphone is released');
    await ctx.go('voices/clone');
    assert.equal($('vo-draft').hidden, false);
    assert.match(text($('vo-draft-meta')), /^3\.0 s/);
  });
});

describe('views/voices clone wizard: a microphone that goes away', () => {
  const MIC_ENDED = 'The microphone stopped. Check that it is still connected, then record again.';
  const MIC_BACKGROUND = 'Recording stopped when the app went to the background.';
  const banner = () => $('vo-capture-error').querySelector('.banner');

  /** The track of the newest capture ends under the page (unplugged, a phone call, another app took it). */
  function endTrack() {
    const track = ctx.env.mediaDevices.streams.at(-1).getTracks()[0];
    track.readyState = 'ended';
    track.dispatchEvent(new Event('ended'));
  }

  /** A tab switch on a desktop: the page hides and shows again, the capture keeps running (no pagehide). */
  function tabAwayAndBack() {
    for (const hidden of [true, false]) {
      document.hidden = hidden;
      document.visibilityState = hidden ? 'hidden' : 'visible';
      document.dispatchEvent(new Event('visibilitychange'));
    }
  }

  it('[VL-01] a track that ends mid-recording stops it at once: what was recorded is kept, the banner says why, and the app plays again', async () => {
    await wizard();
    await record();
    frames(2);
    assert.equal(ctx.app.get().recording.owner, 'voices');
    endTrack();
    await ctx.settle();
    assert.equal(ctx.app.get().recording, null, 'the dead capture no longer holds app.recording');
    assert.equal(text($('vo-record')), 'Record again', 'never a Stop for a microphone that is gone');
    assert.equal(banner().getAttribute('role'), 'alert');
    assert.equal(text(banner().querySelector('.banner__title')), 'Recording stopped');
    assert.equal(text(banner().querySelector('.banner__msg')), MIC_ENDED);
    assert.equal($('vo-draft').hidden, false);
    assert.match(text($('vo-draft-meta')), /^2\.0 s · /, 'the two seconds before it stopped are the draft');
    assert.equal(status(), '2.0 s recorded before the microphone stopped. Check the quality below.');
    ctx.env.clock.tick(5000);
    assert.equal(text($('vo-timer-text')), '0:02', 'the timer shows the draft, it does not tick on');
    const audio = await load('core/audio.js');
    let refused = '';
    try {
      await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:take' });
    } catch (error) {
      refused = error?.name || String(error);
    }
    assert.equal(refused, '', 'other players work again (no "Stop recording to listen")');
    assertAccessibleNames(ctx.section);
  });

  it('[VL-01] an iPhone Home Screen app that loses the microphone in the background says so; a tab switch alone changes nothing', async () => {
    await wizard();
    await record();
    frames(1);
    tabAwayAndBack();
    await ctx.settle();
    assert.equal(text($('vo-record')), 'Stop', 'a desktop tab switch keeps recording');
    assert.equal(ctx.app.get().recording?.owner, 'voices');
    assert.equal($('vo-capture-error').hidden, true);
    // iOS: in the background the system takes the microphone. The app was not left (no pagehide) but suspended,
    // so its track's end is only seen when the page shows again.
    ctx.env.setHidden(true);
    ctx.env.mediaDevices.streams.at(-1).getTracks()[0].readyState = 'ended';
    ctx.env.setHidden(false);
    await ctx.settle();
    assert.equal(ctx.app.get().recording, null);
    assert.equal(text($('vo-record')), 'Record again');
    assert.equal(text(banner().querySelector('.banner__title')), 'Recording stopped');
    assert.equal(text(banner().querySelector('.banner__msg')), MIC_BACKGROUND);
    assert.match(text($('vo-draft-meta')), /^1\.0 s · /);
  });

  it('[VL-01] a track that ends during the countdown cancels it and says why; after a normal Stop nothing is watched any more', async () => {
    await wizard();
    ctx.env.change($('vo-countdown'), true);
    await record();
    assert.equal(text($('vo-record')), 'Cancel');
    endTrack();
    await ctx.settle();
    assert.equal(ctx.app.get().recording, null);
    assert.equal(text($('vo-record')), 'Record');
    assert.equal(text(banner().querySelector('.banner__msg')), MIC_ENDED);
    assert.equal(status(), '', 'the banner says it; no "Recording canceled." the listener did not ask for');
    assert.equal($('vo-draft').hidden, true, 'nothing was recorded');
    ctx.env.clock.tick(4000);
    assert.equal(text($('vo-record')), 'Record', 'the countdown never starts a recording on a dead microphone');
    // A new recording clears the banner; once it is stopped normally, its old track and a page that hides
    // and shows again change nothing.
    ctx.env.change($('vo-countdown'), false);
    const watching = document.__listenerCount('visibilitychange');
    await record();
    assert.equal($('vo-capture-error').hidden, true, 'a new start clears the old banner');
    assert.equal(document.__listenerCount('visibilitychange'), watching + 1, 'the capture is watched');
    const track = ctx.env.mediaDevices.streams.at(-1).getTracks()[0];
    assert.equal(track.__listenerCount('ended'), 1);
    frames(2);
    ctx.env.click($('vo-record'));
    await ctx.settle();
    assert.equal(status(), 'Recording stopped. 2.0 s recorded. Check the quality below.');
    assert.equal(document.__listenerCount('visibilitychange'), watching, 'Stop removes the listeners');
    assert.equal(track.__listenerCount('ended'), 0);
    endTrack();
    ctx.env.setHidden(true);
    ctx.env.setHidden(false);
    await ctx.settle();
    assert.equal($('vo-capture-error').hidden, true, 'no banner for a capture that already ended');
    assert.equal(status(), 'Recording stopped. 2.0 s recorded. Check the quality below.');
    assert.match(text($('vo-draft-meta')), /^2\.0 s · /);
  });
});

describe('views/voices clone wizard: the draft player and the app\'s audio', () => {
  /** A fake owner session (Live, the Reader) that records its control calls. */
  function fakeSession(id, kind, extra = {}) {
    const log = [];
    const state = { playing: extra.playing ?? true, waiting: false, position: 0, duration: 0, rate: 1 };
    const session = {
      id, owner: extra.owner || id, kind, title: id, subtitle: '', homeRoute: extra.homeRoute || '',
      controls: {
        play: () => { log.push('play'); state.playing = true; },
        pause: () => { log.push('pause'); state.playing = false; },
        toggle: () => log.push('toggle'),
        stop: (reason) => { log.push(reason ? `stop:${reason}` : 'stop'); state.playing = false; },
      },
      getState: () => ({ ...state }),
    };
    return { session, log };
  }

  /** The listener presses play on "Your clip" (#vo-draft-audio, a native <audio controls>). */
  async function playDraft() {
    $('vo-draft-audio').dispatchEvent(new Event('play'));
    await ctx.settle();
  }

  it('playing the draft stops a Live session (never pauses it) and releases it', async () => {
    await wizard();
    const audio = await load('core/audio.js');
    const live = fakeSession('live', 'live', { homeRoute: 'speak/live' });
    audio.claim(live.session);
    await playDraft();
    assert.deepEqual(live.log, ['stop:Another player started'], 'a paused Live session would keep the server\'s only Live slot');
    assert.equal(ctx.app.get().player, null, 'Live no longer holds the app session');
    assert.equal(ctx.app.get().resumable, null, 'Live is never resumable');
  });

  it('playing the draft pauses the Reader and keeps it resumable; a take is paused and released', async () => {
    await wizard();
    const audio = await load('core/audio.js');
    const reader = fakeSession('reader:d1', 'long', { owner: 'reader', homeRoute: 'reader/d1' });
    audio.claim(reader.session);
    await playDraft();
    assert.deepEqual(reader.log, ['pause']);
    assert.equal(ctx.app.get().player, null);
    assert.equal(ctx.app.get().resumable?.id, 'reader:d1', 'the mini-player offers the Reader again, paused');
    const take = await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take 1', homeRoute: 'speak', src: 'blob:take1' });
    await playDraft();
    assert.equal(audio.element().paused, true, 'the take paused');
    assert.equal(take.active, false, 'and was released');
    assert.equal(ctx.app.get().player, null);
    assert.equal(ctx.app.get().resumable?.id, 'reader:d1', 'the Reader is still offered');
    assert.deepEqual(reader.log, ['pause'], 'and never resumed');
  });

  it('Space on the focused draft player is the player\'s own: the Reader paused in the mini-player stays paused', async () => {
    await wizard();
    const audio = await load('core/audio.js');
    (await load('shell/miniplayer.js')).initMiniplayer();
    await recordDraft(4);
    const reader = fakeSession('reader:d1', 'long', { owner: 'reader', homeRoute: 'reader/d1', playing: false });
    audio.claim(reader.session);
    await ctx.settle();
    const draft = $('vo-draft-audio');
    assert.ok(draft.hasAttribute('controls'), 'a native player');
    draft.focus();
    assert.equal(document.activeElement, draft);
    const space = ctx.env.key(' ');
    assert.deepEqual(reader.log, [], 'the player shortcut left the Reader alone');
    assert.equal(space.defaultPrevented, false, "the draft player's own Space handling is not cancelled");
    $('vo-title').focus();
    ctx.env.key(' ');
    assert.deepEqual(reader.log, ['toggle'], 'away from the player, Space still plays the Reader');
  });

  it('with the real Live engine, playing the draft ends Live ("Stopped: another player started") and closes its socket', async () => {
    const { server, voiceConfig } = await import('../speak-live/helpers.js');
    await wizard({
      routes: {
        'POST /api/auth/ws-ticket': () => json(200, { ticket: 't1', expires_in: 60 }),
        'GET /api/models/:id/status': () => json(200, { id: 'kokoro', state: 'loaded' }),
      },
    });
    const live = await load('views/speak/live-session.js');
    const editor = document.createElement('textarea');
    document.body.appendChild(editor);
    editor.value = 'Hi.';
    editor.selectionStart = 0;
    const engine = live.createLiveEngine({ editor, getVoiceConfig: () => voiceConfig(), getMode: () => 'natural', onChange: () => {} });
    assert.equal(ctx.env.withGesture(() => engine.start({ readAll: false })), true, 'Live started');
    await ctx.settle();
    const ws = ctx.env.sockets.at(-1);
    assert.ok(ws, 'a Live socket was opened');
    const srv = server(ws);
    srv.open();
    srv.created();
    srv.updated();
    await ctx.settle();
    assert.equal(ctx.app.get().player?.id, 'live');
    assert.equal(engine.getState().status, 'Listening');

    await playDraft();
    assert.equal(engine.getState().status, 'Stopped: another player started', 'stopped, not paused');
    assert.equal(engine.isActive(), false);
    assert.equal(ws.closeCode, 1000, 'the socket closes, freeing the Live slot the Reader renderer waits for');
    assert.equal(ctx.app.get().player, null);
  });
});
