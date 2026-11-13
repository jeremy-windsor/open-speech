// The fake browser behaves like a real one where the difference would let a view test pass on code that
// fails in a browser (foundation repair): aborts after the headers, WebSocket close codes, media src and
// module singletons between tests.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh, loadHtml } from '../harness/env.js';
import { json, FakeSocket, FakeAudioSession } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

describe('harness: fetch', () => {
  it('aborting the request signal after the headers errors the body, as in a browser', async () => {
    env = installBrowserEnv({ routes: { 'GET /x': () => new Response('hello world', { headers: { 'content-type': 'text/plain' } }) } });
    const controller = new AbortController();
    const res = await fetch('/x', { signal: controller.signal });
    controller.abort();
    await assert.rejects(res.text(), { name: 'AbortError' });
    const plain = await fetch('/x');
    assert.equal(await plain.text(), 'hello world', 'without a signal the body reads normally');
    const kept = await fetch('/x', { signal: new AbortController().signal });
    assert.equal(await kept.text(), 'hello world');
    const empty = await fetch('/x', { signal: controller.signal }).catch((e) => e);
    assert.equal(empty.name, 'AbortError', 'an already aborted signal rejects the fetch');
  });

  it('plain-object and JSON answers keep their status and headers through the wrapper', async () => {
    env = installBrowserEnv({ routes: { 'GET /j': () => json(201, { a: 1 }, { 'X-Cache': 'HIT' }) } });
    const res = await fetch('/j', { signal: new AbortController().signal });
    assert.equal(res.status, 201);
    assert.equal(res.headers.get('x-cache'), 'HIT');
    assert.deepEqual(await res.json(), { a: 1 });
  });
});

describe('harness: WebSocket and media', () => {
  it('FakeSocket.close() accepts only 1000 or 3000–4999 and a reason of at most 123 bytes, like a browser', async () => {
    env = installBrowserEnv();
    const socket = new WebSocket('ws://127.0.0.1:8100/v1/audio/speech/stream');
    assert.ok(socket instanceof FakeSocket);
    assert.throws(() => socket.close(1001), { name: 'InvalidAccessError' }, 'a page cannot send 1001 (going away)');
    assert.throws(() => socket.close(1006), { name: 'InvalidAccessError' });
    assert.throws(() => socket.close(4000, 'x'.repeat(124)), { name: 'SyntaxError' });
    socket.close(4000, 'navigating away');
    assert.equal(socket.closeCode, 4000);
    const other = new WebSocket('ws://127.0.0.1:8100/x');
    other.close();
    assert.equal(other.closeCode, 1000);
  });

  it('a media element reports src (and currentSrc) as an absolute URL, as browsers do', async () => {
    env = installBrowserEnv({ html: '<audio id="a"></audio>' });
    const el = document.getElementById('a');
    assert.equal(el.src, '');
    el.src = '/v1/reader/media/d1/p3.mp3?sig=abc';
    assert.equal(el.src, 'http://127.0.0.1:8100/v1/reader/media/d1/p3.mp3?sig=abc');
    assert.equal(el.currentSrc, el.src);
    assert.equal(el.getAttribute('src'), '/v1/reader/media/d1/p3.mp3?sig=abc');
    el.src = 'blob:http://127.0.0.1:8100/abc';
    assert.equal(el.src, 'blob:http://127.0.0.1:8100/abc');
  });
});

describe('harness: module state between tests', () => {
  it('importFresh resets the shell and component singletons too (an open voice picker, the tooltip controller)', async () => {
    env = installBrowserEnv({ html: '<dialog id="app-voice-picker"><h2 id="vp-title">Choose</h2><input id="vp-search"><div id="vp-list"></div><p id="vp-empty"></p><button id="vp-lang"></button><span id="vp-lang-label"></span><div id="vp-lang-menu" role="menu" hidden></div><button id="vp-female"></button><button id="vp-male"></button><button id="vp-favs"></button><p id="vp-summary-label"></p><p id="vp-summary-detail"></p><button id="vp-preview"></button><button id="vp-fav"></button><button id="vp-use"><span class="btn__label">Use</span></button><p id="vp-error"></p><p id="vp-note"></p></dialog>', routes: { 'GET /v1/audio/voices': () => json(200, { voices: [] }), 'GET /api/voices/identities': () => json(200, []) } });
    const picker = await importFresh('components/voice-picker.js');
    const pending = picker.openVoicePicker({ model: 'kokoro' });
    assert.equal(picker.isOpen(), true);
    await importFresh('core/api.js');   // any later importFresh, as the next test would do
    assert.equal(picker.isOpen(), false, 'the picker singleton was reset');
    assert.equal(await pending, null);
  });

  it("importFresh calls the imported module's own __resetForTests (a view module's singleton)", async () => {
    env = installBrowserEnv();
    const mod = await importFresh('../../../tests/js/harness/fixtures/stateful.js');
    mod.bump();
    mod.bump();
    assert.equal(mod.value(), 2);
    const again = await importFresh('../../../tests/js/harness/fixtures/stateful.js');
    assert.equal(again, mod, 'the same module instance');
    assert.equal(again.value(), 0, 'reset');
  });
});

describe('harness: playPolicy in a supplied document, navigator.audioSession', () => {
  it("[REV-20] playPolicy reaches the media elements already in a supplied document (the shell's #app-audio), not only the ones created later", async () => {
    env = installBrowserEnv({ document: loadHtml('src/static/index.html'), playPolicy: 'gesture' });
    const shared = document.getElementById('app-audio');
    assert.ok(shared, 'the shell has the shared element');
    assert.equal(shared.__playPolicy, 'gesture');
    await assert.rejects(shared.play(), { name: 'NotAllowedError' }, 'a play() outside a gesture is refused, as on iOS');
    const audio = await importFresh('core/audio.js');
    const { app } = await import('../../../src/static/js/core/store.js');
    await audio.playClip({ id: 'take:1', owner: 'speak', title: 'Take', src: 'blob:t/1' });
    assert.equal(app.get().player.statusText, 'Tap play to start listening', 'without prime() the result would not autoplay: a view test now sees it');
    env.restore();
    env = installBrowserEnv({ document: loadHtml('src/static/index.html'), playPolicy: 'pending' });
    assert.equal(document.getElementById('app-audio').__playPolicy, 'pending');
    assert.equal(document.createElement('audio').__playPolicy, 'pending');
    env.restore();
    env = installBrowserEnv({ document: loadHtml('src/static/index.html') });
    assert.equal(document.getElementById('app-audio').__playPolicy, 'resolve', 'the default is unchanged');
  });

  it('navigator.audioSession exists only when asked for (Safari 17+); its type ignores unknown values and records what was set', async () => {
    env = installBrowserEnv();
    assert.equal(navigator.audioSession, undefined);
    assert.equal(env.audioSession, null);
    env.restore();
    env = installBrowserEnv({ audioSession: true });
    assert.ok(navigator.audioSession instanceof FakeAudioSession);
    assert.equal(navigator.audioSession, env.audioSession);
    assert.equal(navigator.audioSession.type, 'auto');
    navigator.audioSession.type = 'playback';
    navigator.audioSession.type = 'loud';
    assert.equal(navigator.audioSession.type, 'playback');
    assert.deepEqual(env.audioSession.history, ['playback']);
    env.restore();
    env = installBrowserEnv({ audioSession: 'ambient' });
    assert.equal(navigator.audioSession.type, 'ambient');
  });
});

describe('harness: failing assertions on fake elements', () => {
  // node:assert formats a failure with util.inspect (depth 1000, getters on). Back-links that inspect could walk
  // (a node's document and parent, a token list's element) once made one element expand the whole document again
  // from every descendant: a failing assert.equal(elementA, elementB) grew until the run hit its memory cap, and
  // the run died without the assertion's message.
  it('a failing assert.equal between two elements of the real shell throws at once, with a message, in bounded memory', () => {
    env = installBrowserEnv({ document: loadHtml('src/static/index.html') });
    const main = document.querySelector('main');
    const header = document.querySelector('header');
    main.classList.add('probe');
    main.focus();
    const before = process.memoryUsage().rss;
    const started = Date.now();
    let error = null;
    try { assert.equal(main, header); } catch (e) { error = e; }
    try { assert.equal(document.activeElement, null); } catch (e) { error = error && e; }
    try { assert.deepEqual([main], [header]); } catch (e) { error = error && e; }
    const grewMb = (process.memoryUsage().rss - before) / 2 ** 20;
    assert.ok(error instanceof assert.AssertionError, 'each assertion failed with an AssertionError');
    assert.ok(error.message.length > 0 && error.message.length < 2_000_000, `the message is bounded (${error.message.length} chars)`);
    assert.ok(Date.now() - started < 5000, `the three failures took ${Date.now() - started} ms`);
    assert.ok(grewMb < 200, `formatting the failures grew the process by ${grewMb.toFixed(0)} MB`);
  });

  it('back-links stay usable but are not enumerable: ownerDocument, parentNode, classList and defaultView', () => {
    env = installBrowserEnv({ document: loadHtml('src/static/index.html') });
    const main = document.querySelector('main');
    assert.equal(main.ownerDocument, document);
    assert.equal(main.parentNode, document.body.querySelector('main').parentNode);
    assert.ok(main.parentNode.contains(main));
    assert.equal(document.defaultView, globalThis.window);
    for (const key of ['ownerDocument', 'parentNode']) assert.ok(!Object.keys(main).includes(key), `${key} is not enumerable`);
    assert.ok(!Object.keys(main.classList).includes('el'), "a token list's element is not enumerable");
    assert.ok(!Object.keys(document).includes('defaultView'), "the document's window is not enumerable");
    const moved = document.createElement('p');
    document.body.append(moved);
    assert.equal(moved.parentNode, document.body, 'assigning a back-link keeps working');
    moved.remove();
    assert.equal(moved.parentNode, null);
  });
});

describe('harness: page visibility and page lifecycle', () => {
  // setHidden(true) used to fire pagehide too, which a browser fires only when the page is left. A test of a
  // locked screen or a tab switch then ran the code that tears a page down (closing the Live socket, stopping
  // the microphone, revoking object URLs) and passed on behaviour a phone never shows.
  it('setHidden() is a tab switch or a locked screen: visibilitychange only, never pagehide or pageshow', () => {
    env = installBrowserEnv();
    const seen = [];
    document.addEventListener('visibilitychange', () => seen.push(`visibility:${document.visibilityState}`));
    window.addEventListener('pagehide', () => seen.push('pagehide'));
    window.addEventListener('pageshow', () => seen.push('pageshow'));
    env.setHidden(true);
    env.setHidden(false);
    assert.deepEqual(seen, ['visibility:hidden', 'visibility:visible']);
  });

  it('pageHide() and pageShow() are leaving the page and coming back from the back/forward cache, with persisted', () => {
    env = installBrowserEnv();
    const seen = [];
    document.addEventListener('visibilitychange', () => seen.push(`visibility:${document.visibilityState}`));
    window.addEventListener('pagehide', (event) => seen.push(`pagehide:${event.persisted}`));
    window.addEventListener('pageshow', (event) => seen.push(`pageshow:${event.persisted}`));
    env.pageHide({ persisted: true });
    env.pageShow({ persisted: true });
    env.pageHide();
    assert.deepEqual(seen, ['visibility:hidden', 'pagehide:true', 'pageshow:true', 'visibility:visible', 'visibility:hidden', 'pagehide:false']);
  });
});
