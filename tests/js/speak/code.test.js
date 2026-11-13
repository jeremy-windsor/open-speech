// Copy as code: curl, Python and JavaScript for the current request, never with the key.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { toCurl, toPython, toJs, shellQuote, pyLiteral } from '../../../src/static/js/views/speak/code.js';
import { buildSpeechRequest } from '../../../src/static/js/views/speak/request.js';
import { mountSpeak, ready, type, settle, load } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const SECRET = 'sk-very-secret-key';
const req = buildSpeechRequest({
  model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', input: "It's a 'quoted' line.", speed: 1.25, format: 'wav',
  effects: [{ type: 'reverb', room: 'small' }], instructions: '', caps: { streaming: true, voice_blend: true }, stream: true, mse: true,
});

describe('views/speak/code (pure)', () => {
  it('curl posts the JSON body to /v1/audio/speech with single quotes escaped for POSIX shells', () => {
    const curl = toCurl(req, { origin: 'http://127.0.0.1:8100', hasKey: false });
    assert.match(curl, /^curl -sS -X POST "http:\/\/127\.0\.0\.1:8100\/v1\/audio\/speech"/);
    assert.ok(curl.includes('-H "Content-Type: application/json"'));
    assert.ok(!curl.includes('Authorization'), 'no key, no Authorization header');
    assert.ok(curl.includes(`It'\\''s a '\\''quoted'\\'' line.`), 'single quotes closed, escaped and reopened');
    assert.ok(curl.endsWith('-o speech.wav'));
    assert.ok(!curl.includes('stream=true'), 'a copied request writes a file');
    assert.equal(shellQuote("a'b"), "'a'\\''b'");
  });

  it('with a key set, every variant reads it from the environment and never embeds it', () => {
    const curl = toCurl(req, { origin: 'https://speech.lan', hasKey: true });
    const py = toPython(req, { origin: 'https://speech.lan', hasKey: true });
    const js = toJs(req, { origin: 'https://speech.lan', hasKey: true });
    assert.ok(curl.includes('-H "Authorization: Bearer $OPEN_SPEECH_API_KEY"'));
    assert.ok(py.includes('os.environ["OPEN_SPEECH_API_KEY"]'));
    assert.ok(py.startsWith('import os'));
    assert.ok(js.includes('process.env.OPEN_SPEECH_API_KEY'));
    for (const code of [curl, py, js]) assert.ok(!code.includes(SECRET));
  });

  it('Python uses requests with a Python literal of the body; JavaScript uses fetch', () => {
    const py = toPython(req, { origin: 'http://h' });
    assert.ok(py.includes('requests.post('));
    assert.ok(py.includes('"effects": [\n'));
    assert.ok(py.includes('with open("speech.wav", "wb")'));
    assert.equal(pyLiteral(null), 'None');
    assert.equal(pyLiteral(true), 'True');
    assert.equal(pyLiteral({ a: [1, false] }), '{\n    "a": [\n        1,\n        False,\n    ],\n}');
    const js = toJs(req, { origin: 'http://h' });
    assert.ok(js.startsWith('const response = await fetch("http://h/v1/audio/speech"'));
    assert.ok(js.includes('"voice": "af_bella(2)+af_sky(1)"'));
  });
});

describe('views/speak copy as code (view)', () => {
  it('the More menu copies the current setup as curl, Python or JavaScript, without the key', async () => {
    t = await mountSpeak({
      seed: { 'open-speech-api-key': SECRET },
      beforeShow: async () => { (await load('core/auth.js')).init(); },
    });
    await ready(t);
    type(t, 'Copy me.');
    for (const [act, needle] of [['curl', '$OPEN_SPEECH_API_KEY'], ['python', 'os.environ'], ['js', 'process.env']]) {
      t.env.click(t.$('sp-more'));
      await settle(t.env);
      t.env.click(t.$('sp-more-menu').querySelector(`[data-more="${act}"]`));
      await settle(t.env);
      const copied = t.env.clipboard.written[t.env.clipboard.written.length - 1];
      assert.ok(copied.includes(needle), act);
      assert.ok(copied.includes('"input": "Copy me."') || copied.includes('Copy me.'), act);
      assert.ok(!copied.includes(SECRET), `${act} never embeds the key`);
    }
    assert.ok(document.getElementById('app-toasts').textContent.includes('Copied the request as'));
    const text = document.body.textContent;
    assert.ok(!text.includes(SECRET), 'the key is nowhere in the DOM');
  });

  it("a take's Copy request (curl) copies the request that take was made with", async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'First take.');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    await settle(t.env);
    type(t, 'Changed afterwards.');
    const take = t.$('sp-takes').querySelector('.sp-take');
    t.env.click(take.querySelector('[aria-haspopup="menu"]'));
    await settle(t.env);
    t.env.click(take.parentElement.querySelector('[data-take-menu="copy"]'));
    await settle(t.env);
    const copied = t.env.clipboard.written[t.env.clipboard.written.length - 1];
    assert.ok(copied.startsWith('curl -sS -X POST'));
    assert.ok(copied.includes('First take.'));
    assert.ok(!copied.includes('Changed afterwards.'));
  });
});
