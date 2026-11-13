import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { bytes } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

async function setup({ key = '', ...options } = {}) {
  env = installBrowserEnv({ seed: key ? { 'open-speech-api-key': key } : {}, ...options });
  const download = await importFresh('core/download.js');
  const auth = await import('../../../src/static/js/core/auth.js');
  auth.init();
  return download;
}

function captureClicks() {
  const clicks = [];
  const original = document.createElement.bind(document);
  document.createElement = (tag) => {
    const el = original(tag);
    if (tag === 'a') el.addEventListener('click', () => clicks.push({ href: el.getAttribute('href'), download: el.getAttribute('download') }));
    return el;
  };
  return clicks;
}

describe('core/download', () => {
  it('downloadBlob clicks a download link and revokes the object URL a minute later (WebKit reads it asynchronously)', async () => {
    const d = await setup();
    const clicks = captureClicks();
    d.downloadBlob(new Blob(['x']), 'transcript-1.txt');
    assert.equal(clicks.length, 1);
    assert.match(clicks[0].href, /^blob:/);
    assert.equal(clicks[0].download, 'transcript-1.txt');
    assert.equal(document.querySelector('a'), null, 'the link is removed');
    assert.deepEqual(env.revokedUrls, []);
    env.clock.tick(1000);
    assert.deepEqual(env.revokedUrls, [], 'not after 1 s: iOS Safari would fail the download (WebKitBlobResource error 1)');
    env.clock.tick(59000);
    assert.deepEqual(env.revokedUrls, [clicks[0].href]);
  });

  it('[AU-14] with a key a server file is fetched with the key into a Blob', async () => {
    const d = await setup({ key: 'k1', routes: { 'GET /api/conversations/:id/audio': () => bytes(200, [1, 2], 'audio/wav') } });
    const clicks = captureClicks();
    await d.downloadFromApi('/api/conversations/c1/audio?format=wav', 'conversation.wav');
    assert.equal(env.fetch.calls[0].headers.authorization, 'Bearer k1');
    assert.equal(env.fetch.calls[0].search, '?format=wav');
    assert.match(clicks[0].href, /^blob:/);
  });

  it('[AU-14] without a key it is a plain link and nothing is fetched', async () => {
    const d = await setup();
    const clicks = captureClicks();
    await d.downloadFromApi('/api/conversations/c1/audio', 'conversation.wav');
    assert.deepEqual(clicks, [{ href: '/api/conversations/c1/audio', download: 'conversation.wav' }]);
    assert.equal(env.fetch.calls.length, 0);
  });

  it('refuses another origin', async () => {
    const d = await setup({ key: 'k1' });
    await assert.rejects(d.downloadFromApi('https://evil.example/x', 'x'), TypeError);
  });
});
