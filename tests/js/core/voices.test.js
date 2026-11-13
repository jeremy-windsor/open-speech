import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { json, bytes, networkError } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

async function setup(routes = {}, options = {}) {
  env = installBrowserEnv({ routes, ...options });
  return importFresh('core/voices.js');
}

describe('core/voices: blends', () => {
  it('[SP-27] the recipe grammar: name(weight)+name(weight), names [a-zA-Z0-9_], decimal weights, default 1', async () => {
    const v = await setup();
    assert.deepEqual(v.parseBlend('af_bella(2)+af_sky'), [{ voice: 'af_bella', weight: 2 }, { voice: 'af_sky', weight: 1 }]);
    assert.deepEqual(v.parseBlend(' af_heart(0.75) + bf_emma(1.25) '), [{ voice: 'af_heart', weight: 0.75 }, { voice: 'bf_emma', weight: 1.25 }]);
    assert.deepEqual(v.parseBlend('af_heart'), [{ voice: 'af_heart', weight: 1 }]);
    for (const bad of ['', 'af-bella', 'af_bella()', 'af_bella(-1)', 'af_bella(1.)', 'a+', 'af_bella(x)']) {
      assert.throws(() => v.parseBlend(bad), v.BlendError, bad);
    }
    assert.throws(() => v.parseBlend('af_bella(0)'), /must be more than 0/);
    assert.throws(() => v.parseBlend('af_bella+af_bella(2)'), /more than once/);
  });

  it('[SP-27] blends round-trip exactly to the canonical string the server receives', async () => {
    const v = await setup();
    assert.equal(v.formatBlend(v.parseBlend('af_bella(2)+af_sky')), 'af_bella(2)+af_sky(1)');
    assert.equal(v.formatBlend(v.parseBlend('af_bella(2)+af_sky(1)')), 'af_bella(2)+af_sky(1)');
    assert.equal(v.formatBlend([{ voice: 'af_heart', weight: 0.333333 }, { voice: 'am_adam', weight: 1.10 }]), 'af_heart(0.333333)+am_adam(1.1)', 'no rounding: the server gets the weight given');
    assert.equal(v.formatBlend([{ voice: 'af_heart', weight: 2 }]), 'af_heart', 'one voice is not a blend');
    assert.equal(v.formatBlend([]), '');
    for (const spec of ['af_bella(2)+af_sky(1)', 'af_heart(0.5)+bf_emma(1.25)+am_adam(3)']) {
      assert.equal(v.formatBlend(v.parseBlend(spec)), spec);
    }
    // Finer weights (Paste blend, History, an API-made preset) used to be rounded to 2
    // decimals, and a tiny weight became "(0)", which the server mutes and parseBlend then refuses.
    for (const spec of ['af_heart(0.725)+af_bella(0.275)', 'af_heart(0.333)+af_sky(0.667)', 'af_heart(0.004)+af_bella(1)', 'af_heart(0.0005)+af_sky(2.125)+am_adam(0.1)']) {
      assert.equal(v.formatBlend(v.parseBlend(spec)), spec, spec);
    }
    const tiny = v.formatBlend([{ voice: 'af_heart', weight: 1e-7 }, { voice: 'af_sky', weight: 1 }]);
    assert.equal(tiny, 'af_heart(0.0000001)+af_sky(1)', 'never exponent form, which the grammar refuses');
    assert.deepEqual(v.parseBlend(tiny).map((i) => i.weight), [1e-7, 1], 'and it parses back to the same weights');
    assert.doesNotMatch(v.formatBlend([{ voice: 'af_heart', weight: 1e-25 }, { voice: 'af_sky', weight: 1 }]), /\(0\)/, 'a positive weight is never written as 0');
    assert.equal(v.formatBlend([{ voice: 'af_heart', weight: 0.1 + 0.2 }, { voice: 'af_sky', weight: Math.round(2.94 * 10) / 10 }]), 'af_heart(0.30000000000000004)+af_sky(2.9)', 'the exact number, and a stepped weight stays short');
    assert.equal(v.isBlend('af_bella(2)+af_sky(1)'), true);
    assert.equal(v.isBlend('af_heart'), false);
    assert.equal(v.isBlend('voice:abc'), false);
  });

  it('blendPercents: integers that sum to 100 (largest remainder)', async () => {
    const v = await setup();
    assert.deepEqual(v.blendPercents([{ weight: 2 }, { weight: 1 }]), [67, 33]);
    assert.deepEqual(v.blendPercents([{ weight: 1 }, { weight: 1 }, { weight: 1 }]), [34, 33, 33]);
    assert.deepEqual(v.blendPercents([{ weight: 0 }, { weight: 0 }]), [50, 50]);
    assert.deepEqual(v.blendPercents([]), []);
    const p = v.blendPercents([{ weight: 0.3 }, { weight: 1.7 }, { weight: 0.9 }, { weight: 2.2 }]);
    assert.equal(p.reduce((a, b) => a + b, 0), 100);
  });
});

describe('core/voices: descriptions', () => {
  it('describeVoice: Kokoro ids give name, language and gender; Recommended for the top two', async () => {
    const v = await setup();
    assert.deepEqual(v.describeVoice({ id: 'af_heart', name: 'Heart', gender: 'female' }), {
      id: 'af_heart', label: 'Heart', detail: 'American English · female', group: 'Recommended', lang: 'a', language: 'American English', gender: 'female', tags: ['recommended'],
    });
    const george = v.describeVoice('bm_george');
    assert.deepEqual([george.label, george.detail, george.group, george.lang], ['George', 'British English · male', 'British English', 'b']);
    const piper = v.describeVoice({ id: 'en_US-amy-medium', name: 'en_US-amy-medium', gender: 'unknown', language: 'en-us' });
    assert.deepEqual([piper.label, piper.detail, piper.group, piper.language], ['en_US-amy-medium', '', 'Voices', 'en-us']);
  });

  it('[BUG-27] when two Kokoro voices share a display name, their ids tell them apart', async () => {
    const v = await setup();
    const rows = v.describeVoices([{ id: 'af_nicole', name: 'Nicole' }, { id: 'zf_nicole', name: 'Nicole' }, { id: 'af_sky', name: 'Sky' }]);
    assert.deepEqual(rows.map((r) => r.detail), ['American English · female · af_nicole', 'Mandarin · female · zf_nicole', 'American English · female']);
    assert.equal(new Set(rows.map((r) => `${r.label}|${r.detail}`)).size, 3);
  });
});

describe('core/voices: server lists', () => {
  it('listVoices returns the array in either shape, and [] on error', async () => {
    const v = await setup({
      'GET /v1/audio/voices': (req) => (req.query.get('model') === 'kokoro' ? json(200, { voices: [{ id: 'af_heart' }] }) : req.query.get('model') === 'piper/x' ? json(200, [{ id: 'p' }]) : networkError()),
    });
    assert.deepEqual(await v.listVoices('kokoro'), [{ id: 'af_heart' }]);
    assert.deepEqual(await v.listVoices('piper/x'), [{ id: 'p' }]);
    assert.deepEqual(await v.listVoices('broken'), []);
    assert.equal(env.fetch.calls[1].search, '?model=piper%2Fx');
  });

  it('listNamed, listPresets, listRecordings and recordingConfig', async () => {
    let config = json(200, { max_seconds: 45 });
    const v = await setup({
      'GET /api/voices/identities': () => json(200, { voices: [{ id: 'n1', name: 'Narrator' }] }),
      'GET /api/profiles': () => json(200, { profiles: [{ id: 'p1', name: 'Audiobook' }], default_profile_id: 'p1' }),
      'GET /api/voices/library': () => json(200, [{ name: 'demo_reference', transcript: 'hi' }]),
      'GET /api/voices/library-config': () => config,
    });
    assert.deepEqual(await v.listNamed({ model: 'kokoro' }), [{ id: 'n1', name: 'Narrator' }]);
    await v.listNamed({ check: false });
    assert.deepEqual(env.fetch.calls.map((c) => c.search), ['?model=kokoro', '?check_availability=false']);
    assert.deepEqual(await v.listPresets(), { presets: [{ id: 'p1', name: 'Audiobook' }], defaultId: 'p1' });
    assert.deepEqual(await v.listRecordings(), [{ name: 'demo_reference', transcript: 'hi' }]);
    assert.deepEqual(await v.recordingConfig(), { maxSeconds: 45 });
    config = json(200, { max_seconds: 0 });
    assert.deepEqual(await v.recordingConfig(), { maxSeconds: 0 }, '0 means no limit');
    config = json(404, { error: { message: 'Not Found' } });
    assert.deepEqual(await v.recordingConfig(), { maxSeconds: 60 }, 'an older server: 60 s');
  });
});

describe('core/voices: favourites and previews', () => {
  it('favourites persist as "model|voiceId"', async () => {
    const v = await setup();
    assert.deepEqual(v.favorites(), []);
    assert.equal(v.toggleFavorite('kokoro', 'af_heart'), true);
    assert.equal(v.isFavorite('kokoro', 'af_heart'), true);
    assert.equal(v.isFavorite('piper/x', 'af_heart'), false);
    assert.deepEqual(JSON.parse(env.localStorage.__data['open-speech-voice-favorites']), ['kokoro|af_heart']);
    assert.equal(v.toggleFavorite('kokoro', 'af_heart'), false);
    assert.deepEqual(v.favorites(), []);
  });

  it('preview: MP3 in the voice language, without X-History, cached (LRU 20) with evicted URLs revoked', async () => {
    const v = await setup({ 'POST /v1/audio/speech': () => bytes(200, [1, 2, 3], 'audio/mpeg') });
    const first = await v.preview('kokoro', 'ef_dora');
    assert.match(first.url, /^blob:/);
    const call = env.fetch.calls[0];
    assert.deepEqual(call.json, { model: 'kokoro', voice: 'ef_dora', input: '¡Hola! Así suena mi voz.', response_format: 'mp3' });
    assert.equal(call.headers['x-history'], undefined, 'previews stay out of History');
    assert.equal((await v.preview('kokoro', 'ef_dora')).url, first.url);
    assert.equal(env.fetch.calls.length, 1, 'cached');
    for (let i = 0; i < 20; i += 1) await v.preview('kokoro', `af_v${i}`, { text: 'x' });
    assert.ok(env.revokedUrls.includes(first.url), 'the oldest preview was evicted and revoked');
    await v.preview('piper/x', 'en_US-amy-medium');
    assert.equal(env.fetch.calls.at(-1).json.input, 'Hello! This is how I sound.');
  });
});

describe('core/voices: shared previews and strict lists (foundation repair)', () => {
  const turn = async (n = 6) => { for (let i = 0; i < n; i += 1) await new Promise((r) => setImmediate(r)); };

  it('two callers share one preview request; one aborting does not cancel it for the other; all aborting cancels it', async () => {
    let answer;
    const v = await setup({ 'POST /v1/audio/speech': () => new Promise((r) => { answer = () => r(bytes(200, [1], 'audio/mpeg')); }) });
    const a = new AbortController();
    const b = new AbortController();
    const first = v.preview('kokoro', 'af_heart', { signal: a.signal }).catch((e) => e);
    const second = v.preview('kokoro', 'af_heart', { signal: b.signal });
    await turn();
    assert.equal(env.fetch.calls.length, 1);
    a.abort();
    assert.equal((await first).kind, 'aborted', 'the first caller stops waiting');
    assert.equal(env.fetch.calls[0].signal.aborted, false, 'the request goes on for the second caller');
    answer();
    assert.match((await second).url, /^blob:/);
    const c = new AbortController();
    const third = v.preview('kokoro', 'af_bella', { signal: c.signal }).catch((e) => e);
    await turn();
    c.abort();
    assert.equal((await third).kind, 'aborted');
    assert.equal(env.fetch.calls.at(-1).signal.aborted, true, 'nobody waits any more: the synthesis is cancelled');
  });

  it('cached preview URLs are revoked on a pagehide that unloads the page, not when it goes into the back/forward cache', async () => {
    const v = await setup({ 'POST /v1/audio/speech': () => bytes(200, [1], 'audio/mpeg') });
    const { url } = await v.preview('kokoro', 'af_heart');
    const kept = new Event('pagehide');
    kept.persisted = true;
    window.dispatchEvent(kept);
    assert.equal(env.revokedUrls.includes(url), false, 'a page restored for Back keeps its previews');
    window.dispatchEvent(new Event('pagehide'));
    assert.ok(env.revokedUrls.includes(url));
  });

  it('listVoices({strict: true}) throws the ApiError, so a picker can tell "no voices" from "did not load"', async () => {
    const v = await setup({ 'GET /v1/audio/voices': () => json(500, { error: { message: 'boom' } }) });
    assert.deepEqual(await v.listVoices('kokoro'), []);
    await assert.rejects(v.listVoices('kokoro', { strict: true }), (e) => e.status === 500);
  });
});
