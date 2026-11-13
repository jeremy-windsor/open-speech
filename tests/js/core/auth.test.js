import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { json } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  const auth = await importFresh('core/auth.js');
  const store = await import('../../../src/static/js/core/store.js');
  return { auth, store };
}

describe('core/auth', () => {
  it('[AU-01] the key lives in localStorage open-speech-api-key, trimmed, and init() reads it', async () => {
    const { auth, store } = await setup({ seed: { 'open-speech-api-key': '  secret-key \n' } });
    auth.init();
    assert.equal(auth.getKey(), 'secret-key');
    assert.equal(auth.hasKey(), true);
    assert.deepEqual(store.app.get().auth, { hasKey: true, refused: false });
    assert.deepEqual(auth.authHeaders(), { Authorization: 'Bearer secret-key' });
  });

  it('[AU-01] setKey stores the trimmed key, an empty one removes it, and auth:changed fires', async () => {
    const { auth, store } = await setup();
    auth.init();
    const events = [];
    store.on('auth:changed', (d) => events.push(d));
    assert.deepEqual(auth.setKey('  abc  '), { ok: true });
    assert.equal(env.localStorage.__data['open-speech-api-key'], 'abc');
    assert.deepEqual(auth.setKey(''), { ok: true });
    assert.equal(env.localStorage.__data['open-speech-api-key'], undefined);
    assert.equal(auth.hasKey(), false);
    assert.deepEqual(events, [{ hasKey: true }, { hasKey: false }]);
  });

  it('[AU-07] a key with a character a header cannot carry is stored but never sent, and says why', async () => {
    const { auth } = await setup();
    auth.init();
    for (const bad of ['key“quoted”', 'ключ', 'line\nbreak']) {
      const result = auth.setKey(bad);
      if (bad.includes('\n')) {
        // trimmed inner newlines stay; Headers refuses them
        assert.equal(auth.isSendable(auth.getKey()), false);
      }
      assert.deepEqual(result, { ok: false, reason: 'unsendable' });
      assert.deepEqual(auth.authHeaders(), {});
      assert.equal(auth.refusalMessage(), "The API key contains a character that can't be sent, such as a curly quote. Check it in Settings › Connection.");
    }
    assert.equal(auth.isSendable('plain-ascii_key.123'), true);
  });

  it('markRefused stops sending the key until a key is entered again', async () => {
    const { auth, store } = await setup({ seed: { 'open-speech-api-key': 'wrong' } });
    auth.init();
    auth.markRefused();
    assert.equal(auth.isRefused(), true);
    assert.deepEqual(auth.authHeaders(), {});
    assert.equal(store.app.get().auth.refused, true);
    assert.equal(auth.refusalMessage(), "The server didn't accept the API key. Check it in Settings › Connection.");
    auth.setKey('wrong');
    assert.equal(auth.isRefused(), false, 'entering a key (even the same one) tries it again');
    assert.deepEqual(auth.authHeaders(), { Authorization: 'Bearer wrong' });
  });

  it('refusalMessage without a key points to Settings › Connection', async () => {
    const { auth } = await setup();
    auth.init();
    assert.equal(auth.refusalMessage(), 'This server needs an API key. Enter it in Settings › Connection.');
  });

  it('wsTicket: null without a key, a fresh POST /api/auth/ws-ticket per call with one', async () => {
    let n = 0;
    const { auth } = await setup({ routes: { 'POST /api/auth/ws-ticket': () => json(200, { ticket: `t${(n += 1)}`, expires_in: 60 }) } });
    auth.init();
    assert.equal(await auth.wsTicket(), null);
    assert.equal(env.fetch.calls.length, 0);
    auth.setKey('k');
    assert.equal(await auth.wsTicket(), 't1');
    assert.equal(await auth.wsTicket(), 't2');
    const calls = env.fetch.callsTo('POST /api/auth/ws-ticket');
    assert.equal(calls.length, 2);
    assert.equal(calls[0].headers.authorization, 'Bearer k');
  });
});
