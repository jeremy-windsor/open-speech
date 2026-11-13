// Speak request building and streaming (inventory SP-17, SP-18, SP-19, SP-37; REV-01).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildSpeechRequest, voiceSpec, effectsPayload, effectsState, emptyEffects, streamBlocker, mseSupported, clampSpeed,
  playbackPlan, takeFilename, FORMATS,
} from '../../../src/static/js/views/speak/request.js';
import { receive } from '../../../src/static/js/views/speak/takes.js';
import { mountSpeak, ready, type, settle, callsTo, captureDownloads, mediaSources, blobBytes, speechBytes, bytes, load } from './helpers.js';

/** A streamed speech answer whose chunks the test sends: {route, send(bytes), end()}. */
function streamedAnswer(first) {
  let controller = null;
  return {
    route: () => new Response(new ReadableStream({
      start(c) {
        controller = c;
        if (first) c.enqueue(first);
      },
    }), { status: 200, headers: { 'content-type': 'audio/mpeg' } }),
    send: (chunk) => controller.enqueue(chunk),
    end: () => controller.close(),
  };
}

let t;
afterEach(() => t?.env.restore());

const KOKORO = { voice_blend: true, streaming: true, speed_control: true, voice_clone: false };

describe('views/speak/request (pure)', () => {
  it('[SP-17] the body is {model, voice, input (trimmed), speed, response_format, effects} with X-History, instructions only when present, and a reference as voice_library_ref', () => {
    const plain = buildSpeechRequest({ model: 'kokoro', voice: 'af_heart', input: '  Hello there.  ', speed: 1.25, format: 'wav', effects: null, caps: KOKORO, stream: true, mse: true });
    assert.equal(plain.url, '/v1/audio/speech');
    assert.deepEqual(plain.body, { model: 'kokoro', voice: 'af_heart', input: 'Hello there.', speed: 1.25, response_format: 'wav', effects: null });
    assert.equal(plain.headers['X-History'], 'true');
    assert.equal(plain.headers['Content-Type'], 'application/json');
    assert.equal('instructions' in plain.body, false, 'no instructions key when none were typed');
    assert.equal('voice_library_ref' in plain.body, false);

    const blank = buildSpeechRequest({ model: 'chatterbox', voice: '', input: 'Hi', instructions: '   ', reference: 'demo_reference', caps: { voice_clone: true, instructions: true } });
    assert.equal('instructions' in blank.body, false, 'whitespace-only instructions are not sent');
    assert.equal(blank.body.voice_library_ref, 'demo_reference');
    assert.equal(blank.body.voice, 'demo_reference', 'a clone model with no stock voice sends the reference name as the voice');

    const styled = buildSpeechRequest({ model: 'chatterbox', voice: 'Ryan', input: 'Hi', instructions: ' calm ', reference: 'demo_reference', caps: { voice_clone: true, instructions: true } });
    assert.equal(styled.body.instructions, 'calm');
    assert.equal(styled.body.voice, 'Ryan', 'a stock voice stays the voice next to its reference');

    const blended = buildSpeechRequest({ model: 'kokoro', voice: voiceSpec({ blendSpec: 'af_bella(2)+af_sky(1)', voice: 'af_bella' }), input: 'Hi', caps: KOKORO });
    assert.equal(blended.body.voice, 'af_bella(2)+af_sky(1)', 'a blend travels in the voice field');
    assert.equal('voice_blend' in blended.body, false, 'never a voice_blend field');
  });

  it('[SP-18] a named voice is sent as voice:<identity id>, even over a blend', () => {
    assert.equal(voiceSpec({ identity: 'n1', blendSpec: 'af_bella(2)+af_sky(1)', voice: 'af_bella' }), 'voice:n1');
    assert.equal(voiceSpec({ voice: 'af_heart' }), 'af_heart');
    assert.equal(voiceSpec({ reference: 'demo_reference' }), 'demo_reference');
    const req = buildSpeechRequest({ model: 'kokoro', voice: voiceSpec({ identity: 'n1' }), input: 'Hi', caps: KOKORO });
    assert.equal(req.body.voice, 'voice:n1');
  });

  it('[SP-19] it streams only with the switch on, a streaming model, MP3, no effect and MediaSource support', () => {
    const base = { model: 'kokoro', voice: 'af_heart', input: 'Hi', caps: KOKORO, format: 'mp3', effects: null, stream: true, mse: true };
    const yes = buildSpeechRequest(base);
    assert.equal(yes.stream, true);
    assert.equal(yes.url, '/v1/audio/speech?stream=true');
    assert.equal(buildSpeechRequest({ ...base, stream: false }).stream, false, 'switch off');
    assert.equal(buildSpeechRequest({ ...base, caps: { ...KOKORO, streaming: false } }).stream, false, 'model without streaming');
    assert.equal(buildSpeechRequest({ ...base, format: 'wav' }).stream, false, 'not MP3');
    assert.equal(buildSpeechRequest({ ...base, mse: false }).stream, false, 'no MediaSource (iPhone Safari)');
    assert.equal(buildSpeechRequest({ ...base, mse: false }).url, '/v1/audio/speech', 'it quietly falls back to a normal request');
    assert.equal(streamBlocker({ caps: KOKORO, format: 'opus', mse: true }), 'format');
    assert.equal(streamBlocker({ caps: KOKORO, format: 'mp3', effects: [{ type: 'robot' }], mse: true }), 'effects');
    assert.equal(streamBlocker({ caps: KOKORO, format: 'mp3', mse: false }), 'browser');
    assert.equal(mseSupported({ isTypeSupported: (t2) => t2 === 'audio/mpeg' }), true);
    assert.equal(mseSupported(undefined), false);
    assert.equal(mseSupported({ isTypeSupported: () => { throw new Error('no'); } }), false);
  });

  it('[REV-01] a request with an effect is never streamed (the server answers 400 to stream=true with effects)', () => {
    for (const type of ['normalize', 'podcast_eq', 'reverb', 'pitch', 'robot']) {
      const fx = emptyEffects();
      if (type === 'reverb' || type === 'pitch') fx[type] = { ...fx[type], on: true };
      else fx[type] = true;
      const req = buildSpeechRequest({ model: 'kokoro', voice: 'af_heart', input: 'Hi', caps: KOKORO, format: 'mp3', effects: effectsPayload(fx), stream: true, mse: true });
      assert.equal(req.stream, false, type);
      assert.equal(req.url.includes('stream=true'), false, type);
      assert.deepEqual(req.body.effects.map((e) => e.type), [type]);
    }
  });

  it('[SP-37] effects build today\'s payload: an array in order with reverb room and integer pitch semitones (−12…12), or null', () => {
    assert.equal(effectsPayload(emptyEffects()), null);
    const fx = { normalize: true, podcast_eq: true, reverb: { on: true, room: 'large' }, pitch: { on: true, semitones: 19.6 }, robot: true };
    assert.deepEqual(effectsPayload(fx), [
      { type: 'normalize' }, { type: 'podcast_eq' }, { type: 'reverb', room: 'large' }, { type: 'pitch', semitones: 12 }, { type: 'robot' },
    ]);
    assert.deepEqual(effectsPayload({ ...emptyEffects(), pitch: { on: true, semitones: -40 } }), [{ type: 'pitch', semitones: -12 }]);
    assert.deepEqual(effectsPayload({ ...emptyEffects(), reverb: { on: true, room: 'cathedral' } }), [{ type: 'reverb', room: 'small' }]);
    // A preset's or history entry's list restores the chips and their parameters (strings or objects).
    const state = effectsState(['normalize', { type: 'reverb', room: 'medium' }, { type: 'pitch', semitones: -3 }, { type: 'bogus' }]);
    assert.equal(state.normalize, true);
    assert.deepEqual(state.reverb, { on: true, room: 'medium' });
    assert.deepEqual(state.pitch, { on: true, semitones: -3 });
    assert.deepEqual(effectsPayload(effectsState(effectsPayload(fx))), effectsPayload(fx), 'round trip');
  });

  it('speeds snap to the 0.05 grid within 0.5–2.0 and the formats match the server', () => {
    assert.equal(clampSpeed(1.234), 1.25);
    assert.equal(clampSpeed(0.1), 0.5);
    assert.equal(clampSpeed(9), 2);
    assert.equal(clampSpeed('x'), 1);
    assert.deepEqual(FORMATS.map((f) => f.value), ['mp3', 'wav', 'flac', 'opus', 'aac', 'm4a', 'pcm']);
    assert.equal(playbackPlan('pcm', () => true), 'download-pcm');
    assert.equal(playbackPlan('opus', () => false), 'download');
    assert.equal(playbackPlan('mp3', () => true), 'play');
    assert.equal(takeFilename('wav', 1760000000000), 'open-speech-1760000000000.wav');
  });
});

describe('views/speak streaming (view)', () => {
  it('[SP-19] a streamed take plays through MediaSource and Download saves a Blob of exactly the streamed bytes; the MediaSource URL is revoked once attached', async () => {
    const all = speechBytes();
    const answer = streamedAnswer(all.slice(0, 32));
    // Like a browser: 'playing' comes once the element has audio, not at play() (the env's policy reaches the
    // shell's #app-audio too).
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': answer.route }, env: { playPolicy: 'pending' } });
    await ready(t);
    const audioEl = t.$('app-audio');
    const { on } = await load('core/store.js');
    let logged = 0;
    on('history:changed', () => { logged += 1; });
    const downloads = captureDownloads(t.env);
    type(t, 'Stream this sentence, please.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.search, '?stream=true', 'streams by default');
    assert.equal(post.headers['x-history'], 'true');
    assert.equal(logged, 1, '[SP-22] the server logged the streamed take before it answered: History hears of it at once');
    const [ms] = await mediaSources();
    assert.ok(ms, 'a MediaSource was created');
    const msUrl = t.env.createdUrls[t.env.createdUrls.length - 1];
    assert.equal(audioEl.getAttribute('src'), msUrl, 'the shared element plays the MediaSource');
    assert.equal(t.$('sp-gen-status-text').textContent, 'Generating…', 'no audio yet: nothing to time');
    await t.env.clock.tickAsync(140);
    ms.__open();
    await settle(t.env);
    assert.ok(t.env.revokedUrls.includes(msUrl), 'the MediaSource URL is revoked once attached');
    assert.equal(ms.sourceBuffers[0].appended.length, 1, 'the first chunk is in');
    audioEl.__resolvePlay();   // the element starts on the first chunk
    await settle(t.env);
    assert.equal(t.$('sp-gen-status-text').textContent, 'Streaming · first audio 140 ms', 'timed from the request to the first playing');
    answer.send(all.slice(32));
    answer.end();
    await settle(t.env);
    await settle(t.env);
    assert.equal(ms.sourceBuffers[0].appended.length, 2, 'every chunk was appended');
    assert.equal(ms.readyState, 'ended');
    const take = t.parts.takes.all()[0];
    assert.equal(Math.round(take.ttfaMs), 140);
    assert.ok(take.blob, 'the take keeps a Blob');
    assert.equal(take.blob.type, 'audio/mpeg');
    assert.deepEqual(await blobBytes(take.blob), Array.from(speechBytes()));
    const download = t.$('sp-takes').querySelector('[data-take-act="download"]');
    assert.equal(download.disabled, false);
    t.env.click(download);
    assert.equal(downloads.length, 1);
    assert.match(downloads[0].download, /^open-speech-\d+\.mp3$/);
    assert.ok(downloads[0].href.startsWith('blob:'), 'never the MediaSource URL');
    assert.notEqual(downloads[0].href, msUrl);
    assert.deepEqual(await blobBytes(downloads[0].blob), Array.from(speechBytes()));
  });

  it('[REV-01] with an effect selected the view sends no stream=true and the Stream switch explains why', async () => {
    t = await mountSpeak();
    await ready(t);
    assert.equal(t.$('sp-stream-wrap').hidden, false, 'Kokoro streams: the switch shows');
    assert.equal(t.$('sp-stream').checked, true, 'on by default');
    t.env.click(t.$('sp-effects').querySelector('[data-effect="robot"]'));
    await settle(t.env);
    assert.equal(t.$('sp-stream').disabled, true);
    assert.equal(t.$('sp-stream-hint').hidden, false);
    assert.equal(t.$('sp-stream-hint').textContent, 'Effects are applied after the full render');
    type(t, 'Robot voice.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.search, '', 'never ?stream=true with effects');
    assert.deepEqual(post.json.effects, [{ type: 'robot' }]);
  });

  it('[SP-19] the switch is hidden for a model without streaming and disabled with a hint for formats other than MP3 or without MediaSource', async () => {
    t = await mountSpeak({ env: { mediaSource: false } });
    await ready(t);
    assert.equal(t.$('sp-stream').disabled, true);
    assert.equal(t.$('sp-stream-hint').textContent, "This browser can't stream; the take plays when it's ready");
    type(t, 'No MediaSource here.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(callsTo(t, 'POST /v1/audio/speech')[0].search, '', 'iPhone Safari: a normal request');
    t.env.restore();

    t = await mountSpeak();
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    await settle(t.env);
    assert.equal(t.$('sp-stream').disabled, true);
    assert.equal(t.$('sp-stream-hint').textContent, 'Streaming plays MP3 only');
    t.env.change(t.$('sp-model'), 'chatterbox');
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-stream-wrap').hidden, true, 'a model without streaming hides the switch');
  });

  it('[SP-19] a stream that fails mid-way keeps no half take and shows the error', async () => {
    let controllerRef = null;
    t = await mountSpeak({
      routes: {
        'POST /v1/audio/speech': () => new Response(new ReadableStream({
          start(controller) {
            controllerRef = controller;
            controller.enqueue(new Uint8Array([1, 2, 3]));
          },
        }), { status: 200, headers: { 'content-type': 'audio/mpeg' } }),
      },
    });
    await ready(t);
    type(t, 'This stream breaks.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    const [ms] = await mediaSources();
    ms.__open();
    await settle(t.env);
    controllerRef.error(new TypeError('network error'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.parts.takes.all().length, 0, 'no half take');
    assert.equal(t.$('sp-gen-error').hidden, false);
    assert.equal(t.$('sp-gen-error').getAttribute('role'), 'alert');
  });

  it('[SP-19] when another clip takes the shared element mid-stream, the take keeps every byte and no error shows', async () => {
    const answer = streamedAnswer(new Uint8Array([1, 2, 3]));
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': answer.route } });
    await ready(t);
    type(t, 'A long streamed take.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    const [ms] = await mediaSources();
    ms.__open();
    await settle(t.env);
    // A voice preview (or an older take, a Studio render) plays on the shared element. A browser then
    // detaches the MediaSource: it closes and its SourceBuffer is removed, so appending throws.
    const audio = await load('core/audio.js');
    await audio.playClip({ id: 'preview', owner: 'preview', title: 'Preview', src: 'blob:http://127.0.0.1:8100/other', ephemeral: true });
    ms.readyState = 'closed';
    ms.sourceBuffers[0].appendBuffer = () => { throw new DOMException('This SourceBuffer has been removed from the parent media source.', 'InvalidStateError'); };
    answer.send(new Uint8Array([4, 5, 6]));
    answer.end();
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-gen-error').hidden, true, 'no "Couldn\'t generate speech"');
    const takes = t.parts.takes.all();
    assert.equal(takes.length, 1, 'the take is kept');
    assert.deepEqual(await blobBytes(takes[0].blob), [1, 2, 3, 4, 5, 6], 'with all of its audio');
    assert.match(t.$('sp-gen-status-text').textContent, /^Done/);
    assert.equal(audio.current().id, 'preview', 'the preview keeps the element');
    assert.equal(t.$('sp-takes').querySelector('[data-take-act="play"]').disabled, false, 'the take plays from its button');
  });

  it('[SP-19] a full buffer (QuotaExceededError) stops feeding the element, ends what it has, and keeps every byte', async () => {
    const answer = streamedAnswer(new Uint8Array([1, 2]));
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': answer.route } });
    await ready(t);
    type(t, 'A very long take.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    const [ms] = await mediaSources();
    ms.__open();
    await settle(t.env);
    ms.sourceBuffers[0].appendBuffer = () => { throw new DOMException('The SourceBuffer is full.', 'QuotaExceededError'); };
    answer.send(new Uint8Array([3, 4]));
    answer.send(new Uint8Array([5]));
    answer.end();
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-gen-error').hidden, true);
    assert.equal(ms.readyState, 'ended', 'what was fed plays to its end');
    assert.deepEqual(await blobBytes(t.parts.takes.all()[0].blob), [1, 2, 3, 4, 5]);
  });

  it('[SP-19] a stream the element never takes (playback could not start) is read whole: no wait for a source that never opens', async () => {
    t = await mountSpeak();
    await ready(t);
    const res = new Response(new ReadableStream({
      start(c) {
        c.enqueue(new Uint8Array([7, 8]));
        c.enqueue(new Uint8Array([9]));
        c.close();
      },
    }), { status: 200, headers: { 'content-type': 'audio/mpeg' } });
    let opened = 0;
    const blob = await receive(res, { stream: true, onSource: (url, source) => { opened += 1; source.release(); } });
    assert.equal(opened, 1);
    assert.deepEqual(await blobBytes(blob), [7, 8, 9]);
    const [ms] = await mediaSources();
    assert.equal(ms.sourceBuffers.length, 0, 'nothing was fed to a source nobody plays');
  });

  it('[SP-17] the view sends the request it snapshotted at the click, with bytes() from the server', async () => {
    t = await mountSpeak({ routes: { 'POST /v1/audio/speech': () => bytes(200, speechBytes(8), 'audio/wav') } });
    await ready(t);
    t.env.change(t.$('sp-format'), 'wav');
    type(t, '  Spaces around.  ');
    t.env.click(t.$('sp-generate'));
    t.env.change(t.$('sp-format'), 'flac');   // a change after the click doesn't reach this take
    await settle(t.env);
    await settle(t.env);
    const [post] = callsTo(t, 'POST /v1/audio/speech');
    assert.equal(post.json.input, 'Spaces around.');
    assert.equal(post.json.response_format, 'wav');
    assert.equal(post.json.model, 'kokoro');
    assert.equal(typeof post.json.speed, 'number');
    assert.equal(post.json.effects, null);
  });
});
