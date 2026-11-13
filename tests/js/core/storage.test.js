import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  return importFresh('core/storage.js');
}

describe('core/storage', () => {
  it('KEYS lists exactly the allowed keys', async () => {
    const { KEYS } = await setup();
    assert.deepEqual(Object.values(KEYS).sort(), [
      'open-speech-api-key', 'open-speech-mic-device', 'open-speech-perf', 'open-speech-prefs', 'open-speech-reader-speed',
      'open-speech-reader-voice', 'open-speech-reading', 'open-speech-speak', 'open-speech-speak-draft', 'open-speech-theme',
      'open-speech-voice-favorites',
    ]);
    assert.ok(Object.isFrozen(KEYS));
  });

  it('reads, writes and removes string values', async () => {
    const s = await setup();
    assert.equal(s.read(s.KEYS.theme), null);
    assert.equal(s.read(s.KEYS.theme, 'system'), 'system');
    assert.equal(s.write(s.KEYS.theme, 'dark'), true);
    assert.equal(s.read(s.KEYS.theme), 'dark');
    assert.equal(env.localStorage.__data['open-speech-theme'], 'dark');
    s.write(s.KEYS.readerSpeed, 1.25);
    assert.equal(s.read(s.KEYS.readerSpeed), '1.25');
    s.remove(s.KEYS.theme);
    assert.equal(s.read(s.KEYS.theme), null);
  });

  it('[TH-02] storage that throws never breaks a caller: read falls back, write is false, remove is silent', async () => {
    const s = await setup({ storage: 'throwing' });
    assert.throws(() => globalThis.localStorage, /insecure|denied/i);
    assert.equal(s.read(s.KEYS.theme, 'system'), 'system');
    assert.equal(s.write(s.KEYS.theme, 'dark'), false);
    assert.doesNotThrow(() => s.remove(s.KEYS.theme));
    assert.deepEqual(s.readJson(s.KEYS.prefs, { v: 1 }), { v: 1 });
    assert.equal(s.writeJson(s.KEYS.prefs, { v: 1 }), false);
  });

  it('[TH-02] a full quota makes write return false without throwing', async () => {
    const s = await setup({ storage: 'quota', seed: { 'open-speech-theme': 'light' } });
    assert.equal(s.read(s.KEYS.theme), 'light');
    assert.equal(s.write(s.KEYS.theme, 'dark'), false);
    assert.equal(s.read(s.KEYS.theme), 'light');
  });

  it('[SP-23] corrupt or invalid JSON is removed and the fallback returned (the recent-list recovery, kept in readJson)', async () => {
    const s = await setup({ seed: { 'open-speech-voice-favorites': '{not json', 'open-speech-perf': '[1,2]' } });
    assert.deepEqual(s.readJson(s.KEYS.favorites, []), []);
    assert.equal(env.localStorage.__data['open-speech-voice-favorites'], undefined, 'corrupt value removed');
    const isObject = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
    assert.deepEqual(s.readJson(s.KEYS.perf, {}, isObject), {});
    assert.equal(env.localStorage.__data['open-speech-perf'], undefined, 'invalid value removed');
    assert.equal(s.writeJson(s.KEYS.favorites, ['kokoro|af_heart']), true);
    assert.deepEqual(s.readJson(s.KEYS.favorites, [], Array.isArray), ['kokoro|af_heart']);
  });

  it('[BUG-26] there is one history: the retired local Recent-list keys are neither read nor written', async () => {
    const s = await setup({ seed: { 'open-speech-tts-history': '[{"text":"old"}]', 'open-speech-more-tab': 'models' } });
    const keys = Object.values(s.KEYS);
    for (const retired of ['open-speech-tts-history', 'open-speech-stt-history', 'open-speech-more-tab']) {
      assert.ok(!keys.includes(retired), `${retired} is retired`);
      assert.equal(s.read(retired, 'fallback'), 'fallback', 'unknown keys are not read');
      assert.equal(s.write(retired, 'x'), false, 'unknown keys are not written');
    }
    // nothing deletes the old keys (the old UI may still run on this origin)
    s.remove('open-speech-tts-history');
    assert.equal(env.localStorage.__data['open-speech-tts-history'], '[{"text":"old"}]');
  });

  it('writeJson refuses values JSON cannot represent', async () => {
    const s = await setup();
    const cyclic = {};
    cyclic.self = cyclic;
    assert.equal(s.writeJson(s.KEYS.speak, cyclic), false);
    assert.equal(s.writeJson(s.KEYS.speak, undefined), false);
  });
});
