// Settings › Connection: the API key card (SE-01). Saving and removing the key
// (AU-08), what happens after a key change (AU-09), Show/Hide (AU-11), Clear (AU-12), and Test.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountSettings, bootShell, settle, load, json, delay, networkError, keyedModels, textOf, toasts, callsSince, KEY_STORAGE, INVENTORY,
} from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

const UNSENDABLE = "The API key contains a character that can't be sent, such as a curly quote.";
const stored = (t) => t.env.localStorage.getItem(KEY_STORAGE);
const statusText = (t) => textOf(t.$('se-key-status'));
const modelCalls = (t, mark = 0) => callsSince(t.env, mark).filter((c) => c.path === '/api/models');
const REFUSED = { error: { message: 'Invalid or missing API key', code: 'unauthorized' } };
/** A keyed /api/models that takes `ms` to answer. */
const slowKeyed = (key, ms) => (req) => delay(ms, (req.headers.authorization || '') === `Bearer ${key}` ? json(200, INVENTORY) : json(401, REFUSED));

describe('settings/connection: pure parts', () => {
  it('[SE-01] keyCheckMessage: accepted, no key needed, refused, locked out, unreachable, unsendable', async () => {
    installBrowserEnv();
    const { keyCheckMessage, KEY_COPY } = await importFresh('views/settings/logic.js');
    assert.deepEqual(keyCheckMessage({ ok: true, hasKey: true }), { text: 'Connected. The server accepted the key.', tone: 'ok', kind: 'accepted' });
    assert.deepEqual(keyCheckMessage({ ok: true, hasKey: false }), { text: "This server doesn't require a key.", tone: 'ok', kind: 'open' });
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'auth', status: 401 } }).text, "The server didn't accept the key.");
    assert.equal(keyCheckMessage({ ok: false, hasKey: false, error: { kind: 'auth', status: 401 } }).text, 'This server needs an API key.');
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'locked', retryAfter: 52 } }).text, 'Locked out for 52 s after too many wrong keys.');
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'locked', retryAfter: null } }).text, 'Locked out for a while after too many wrong keys.');
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'network' } }).text, "Can't reach the server.");
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'timeout' } }).text, "Can't reach the server.");
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, sendable: false, error: { kind: 'auth' } }).text, KEY_COPY.unsendable);
    assert.equal(keyCheckMessage({ ok: false, hasKey: true, error: { kind: 'http', status: 500, message: 'database is locked' } }).text, 'The server answered with an error: database is locked.');
    assert.deepEqual(
      [[true, {}], [false, { kind: 'auth' }], [false, { kind: 'locked' }], [false, { kind: 'network' }], [false, { status: 500 }]]
        .map(([ok, error]) => keyCheckMessage({ ok, error, hasKey: true }).kind),
      ['accepted', 'refused', 'locked', 'offline', 'error'],
    );
    assert.equal(keyCheckMessage({ ok: false, hasKey: false, error: { kind: 'auth' } }).kind, 'needed');
  });
});

describe('settings/connection: the API key card', () => {
  it('[SE-01] a password field (autocomplete off, no spellcheck, paste allowed) with its label and hint, prefilled from this browser; the key never reaches markup or a URL', async () => {
    t = await mountSettings({ seed: { [KEY_STORAGE]: '  s3cret-key  ' }, routes: { 'GET /api/models': keyedModels('s3cret-key') } });
    const input = t.$('se-api-key');
    assert.equal(input.getAttribute('type'), 'password');
    assert.equal(input.getAttribute('autocomplete'), 'off');
    assert.equal(input.getAttribute('spellcheck'), 'false');
    assert.equal(input.hasAttribute('onpaste'), false);
    assert.equal(t.doc.querySelector('label[for="se-api-key"]').textContent, 'API key');
    assert.match(textOf(t.$('se-api-key-hint')), /^Needed when the server sets OS_API_KEY\. Kept in this browser only\.$/);
    assert.deepEqual(input.getAttribute('aria-describedby').split(' '), ['se-api-key-hint', 'se-api-key-error']);
    assert.equal(input.value, 's3cret-key', 'prefilled with the stored key, trimmed');
    assert.equal(input.getAttribute('value'), null, 'the key is a property, never an attribute');
    assert.doesNotMatch(t.root.innerHTML, /s3cret-key/);
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    const calls = modelCalls(t);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].headers.authorization, 'Bearer s3cret-key', 'the key travels as a header');
    assert.ok(t.env.fetch.calls.every((c) => !c.url.includes('s3cret-key')), 'never in a URL');
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    assert.equal(t.$('se-key-status').getAttribute('role'), 'status');
    assert.match(textOf(t.root.querySelector('.se-note')), /one-time ticket made from this key\. The key itself is never put in a URL\./);
    assertAccessibleNames(t.root);
  });

  it('[AU-08] the field\'s change event saves the key ("API key saved in this browser"); an empty field removes it ("API key removed")', async () => {
    t = await mountSettings();
    const input = t.$('se-api-key');
    const auth = t.auth;
    t.env.input(input, '  my-key ', { change: true });
    await settle(t.env);
    assert.equal(stored(t), 'my-key');
    assert.equal(auth.getKey(), 'my-key');
    assert.equal(input.value, 'my-key', 'the field shows the trimmed key that was stored');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'API key saved in this browser' }]);
    t.env.input(input, '', { change: true });
    await settle(t.env);
    assert.equal(stored(t), null);
    assert.equal(auth.hasKey(), false);
    assert.deepEqual(toasts(t).map((x) => x.text), ['API key saved in this browser', 'API key removed']);
  });

  it('[AU-08] Enter in the field and the Save button save too, once (Save after the field\'s change does not save twice)', async () => {
    t = await mountSettings();
    const input = t.$('se-api-key');
    input.focus();
    t.env.input(input, 'enter-key');
    t.env.key('Enter');
    await settle(t.env);
    assert.equal(stored(t), 'enter-key');
    assert.equal(toasts(t).length, 1);
    // Typing then clicking Save: the field's change fires first (blur), then the click.
    t.env.input(input, 'save-key', { change: true });
    t.env.click(t.$('se-key-save'));
    await settle(t.env);
    assert.equal(stored(t), 'save-key');
    assert.deepEqual(toasts(t).map((x) => x.text), ['API key saved in this browser', 'API key saved in this browser']);
  });

  it('[AU-08] a key with a character a header cannot carry is refused at the field and never sent', async () => {
    t = await mountSettings();
    const input = t.$('se-api-key');
    const mark = t.env.fetch.calls.length;
    t.env.input(input, '“curly-key”', { change: true });
    await settle(t.env);
    assert.equal(t.$('se-api-key-error').hidden, false);
    assert.equal(textOf(t.$('se-api-key-error')), UNSENDABLE);
    assert.equal(input.getAttribute('aria-invalid'), 'true');
    assert.deepEqual(toasts(t), [], 'no success toast for a key that cannot be used');
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.ok(callsSince(t.env, mark).every((c) => !c.headers.authorization), 'it is never sent');
    // Editing the field clears the error until the next save.
    t.env.input(input, 'plain-key');
    assert.equal(t.$('se-api-key-error').hidden, true);
    assert.equal(input.hasAttribute('aria-invalid'), false);
  });

  it('[AU-11] Show/Hide toggles the field between password and text, its label Show|Hide and aria-pressed; leaving Settings hides it again', async () => {
    t = await mountSettings({ seed: { [KEY_STORAGE]: 'k' } });
    const input = t.$('se-api-key');
    const show = t.$('se-key-show');
    assert.equal(show.getAttribute('aria-pressed'), 'false');
    assert.equal(textOf(show), 'Show');
    t.env.click(show);
    assert.equal(input.type, 'text');
    assert.equal(textOf(show), 'Hide');
    assert.equal(show.getAttribute('aria-pressed'), 'true');
    t.env.click(show);
    assert.equal(input.type, 'password');
    assert.equal(textOf(show), 'Show');
    assert.equal(show.getAttribute('aria-pressed'), 'false');
    t.env.click(show);
    t.ctl.hide();
    assert.equal(input.type, 'password', 'leaving Settings masks the key');
    assert.equal(show.getAttribute('aria-pressed'), 'false');
  });

  it('[AU-12] Clear empties the field, removes the stored key, says so, and reloads no data: with the shell, only the status checks run again', async () => {
    t = await bootShell({ seed: { [KEY_STORAGE]: 'old-key' }, routes: { 'GET /api/models': keyedModels(null) } });
    const input = t.$('se-api-key');
    assert.equal(input.value, 'old-key');
    const mark = t.env.fetch.calls.length;
    t.env.click(t.$('se-key-clear'));
    await settle(t.env, 10);
    assert.equal(input.value, '');
    assert.equal(stored(t), null);
    assert.equal(t.auth.hasKey(), false);
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'API key removed' }]);
    assert.equal(t.doc.activeElement, input);
    // auth:changed re-runs the shell's status checks (GET /health and one GET /api/models, which the status line
    // shares) and re-shows Settings; nothing else is fetched again.
    const calls = callsSince(t.env, mark).map((c) => `${c.method} ${c.path}`).sort();
    assert.deepEqual(calls, ['GET /api/models', 'GET /health']);
    assert.equal(statusText(t), "This server doesn't require a key.");
    assert.equal(t.root.hidden, false, 'Settings stays on screen');
    // With nothing stored, Clear only empties the field.
    const mark2 = t.env.fetch.calls.length;
    t.env.input(input, 'typed-not-saved');
    t.env.click(t.$('se-key-clear'));
    await settle(t.env, 10);
    assert.equal(input.value, '');
    assert.equal(toasts(t).length, 1);
    assert.deepEqual(callsSince(t.env, mark2), []);
  });

  it('[AU-08] Enter or a change on the unchanged key the server refused sends nothing and says it was refused; Save tries it once more, without a "saved" toast', async () => {
    t = await mountSettings({ routes: { 'GET /api/models': keyedModels('right-key') } });
    const input = t.$('se-api-key');
    input.focus();
    t.env.input(input, 'wrong-key');
    t.env.key('Enter');
    await settle(t.env);
    assert.equal(statusText(t), "The server didn't accept the key.");
    assert.equal(modelCalls(t).length, 1);
    assert.equal(toasts(t).length, 1);
    t.$('se-key-status').textContent = '';   // forget the line, to see that Enter says it again
    t.env.key('Enter');
    t.env.key('Enter');
    t.env.input(input, 'wrong-key', { change: true });
    await settle(t.env);
    assert.equal(modelCalls(t).length, 1, 'the refused key is not sent again (every wrong key counts toward the lockout)');
    assert.equal(toasts(t).length, 1, 'and nothing was saved');
    assert.equal(statusText(t), "The server didn't accept the key.");
    // Save is the one way to try the same key again (the server's key may have changed since).
    t.env.click(t.$('se-key-save'));
    assert.equal(statusText(t), 'Trying the key again…');
    await settle(t.env);
    assert.deepEqual(modelCalls(t).slice(1).map((c) => c.headers.authorization), ['Bearer wrong-key']);
    assert.equal(toasts(t).length, 1, 'no "saved" toast for a retry');
    assert.equal(statusText(t), "The server didn't accept the key.");
  });

  it('[AU-08] pressing Save after typing saves and sends the key once, even when the answer is back before the button is released', async () => {
    t = await mountSettings({ routes: { 'GET /api/models': keyedModels('right-key') } });
    const input = t.$('se-api-key');
    const save = t.$('se-key-save');
    input.focus();
    t.env.input(input, 'wrong-key');
    save.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));   // the press blurs the field
    await settle(t.env);
    assert.equal(statusText(t), "The server didn't accept the key.");
    t.env.click(save);
    await settle(t.env);
    assert.equal(modelCalls(t).length, 1, 'one press, one request');
    assert.deepEqual(toasts(t).map((x) => x.text), ['API key saved in this browser']);
    assert.equal(statusText(t), "The server didn't accept the key.");
  });

  it('[SE-01] Test reports what the server answered: refused, locked out, unreachable; a refused key is never re-sent by Test', async () => {
    t = await mountSettings({ seed: { [KEY_STORAGE]: 'wrong-key' }, routes: { 'GET /api/models': keyedModels('right-key') } });
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.equal(statusText(t), "The server didn't accept the key.");
    assert.equal(t.$('se-key-status').classList.contains('status-text--error'), true);
    const sent = modelCalls(t).length;
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.equal(modelCalls(t).length, sent, 'the refused key is not sent again (every wrong key counts toward the lockout)');
    assert.equal(statusText(t), "The server didn't accept the key.");
    t.env.restore();
    t = await mountSettings({
      seed: { [KEY_STORAGE]: 'some-key' },
      routes: { 'GET /api/models': () => json(429, { error: { message: 'Too many failed API key attempts', code: 'auth_locked' } }, { 'Retry-After': '52' }) },
    });
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.equal(statusText(t), 'Locked out for 52 s after too many wrong keys.');
    t.env.restore();
    t = await mountSettings({ seed: { [KEY_STORAGE]: 'some-key' }, routes: { 'GET /api/models': () => networkError() } });
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.equal(statusText(t), "Can't reach the server.");
  });

  it('[SE-01] Test with an edited, unsaved key saves it first, then checks it; while checking, the button is busy', async () => {
    t = await mountSettings({ routes: { 'GET /api/models': keyedModels('fresh') } });
    const input = t.$('se-api-key');
    t.env.input(input, 'fresh');
    const test = t.$('se-key-test');
    t.env.click(test);
    assert.equal(test.getAttribute('aria-busy'), 'true');
    assert.equal(statusText(t), 'Checking…');
    await settle(t.env);
    assert.equal(stored(t), 'fresh');
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    assert.equal(test.hasAttribute('aria-busy'), false);
  });
});

describe('settings/connection: after a key change (with the shell)', () => {
  it('[AU-09] after a refused page load, saving a key re-runs what the server refused, once, with the key, and keeps the listener\'s picks', async () => {
    t = await bootShell({ routes: { 'GET /api/models': keyedModels('right-key') } });
    assert.equal(t.app.get().connection.state, 'unauthorized', 'the page load was refused');
    assert.equal(t.app.get().models.list.length, 0);
    assert.equal(t.root.hidden, false, 'Settings is showing');
    // A pick the listener made elsewhere (Speak's voice) must survive the key change.
    t.app.set({ speakVoice: { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella and Sky' } });
    const mark = t.env.fetch.calls.length;
    const input = t.$('se-api-key');
    t.env.input(input, 'right-key', { change: true });
    await settle(t.env, 10);
    const calls = callsSince(t.env, mark).filter((c) => c.path === '/api/models');
    assert.equal(calls.length, 1, 'one request: the status line shares the shell\'s probe');
    assert.equal(calls[0].headers.authorization, 'Bearer right-key');
    assert.equal(t.app.get().connection.state, 'online');
    assert.equal(t.app.get().models.list.length, 3, 'what the server refused is loaded now');
    assert.deepEqual(t.app.get().speakVoice, { model: 'kokoro', voice: 'af_bella(2)+af_sky(1)', label: 'Bella and Sky' });
    assert.equal(textOf(t.$('se-key-status')), 'Connected. The server accepted the key.');
    assert.deepEqual(toasts(t), [{ kind: 'success', text: 'API key saved in this browser' }]);
    assert.ok(callsSince(t.env, mark).some((c) => c.path === '/health'), 'the status checks ran again');
  });

  it('[AU-09] on a page that loaded fine, a new key sends one probe; a refused key is reported in the status line and the connection banner, not as an error toast', async () => {
    t = await bootShell({ routes: { 'GET /api/models': keyedModels(null) } });
    assert.equal(t.app.get().connection.state, 'online');
    assert.equal(t.app.get().models.list.length, 3);
    const mark = t.env.fetch.calls.length;
    t.env.input(t.$('se-api-key'), 'any-key', { change: true });
    await settle(t.env, 10);
    assert.equal(callsSince(t.env, mark).filter((c) => c.path === '/api/models').length, 1);
    assert.equal(textOf(t.$('se-key-status')), 'Connected. The server accepted the key.');
    t.env.restore();
    t = await bootShell({ routes: { 'GET /api/models': keyedModels('the-real-key') }, seed: { [KEY_STORAGE]: 'the-real-key' } });
    assert.equal(t.app.get().connection.state, 'online');
    t.env.input(t.$('se-api-key'), 'a-wrong-key', { change: true });
    await settle(t.env, 10);
    assert.equal(textOf(t.$('se-key-status')), "The server didn't accept the key.");
    assert.equal(t.app.get().connection.state, 'unauthorized');
    assert.equal(toasts(t).filter((x) => x.kind === 'error').length, 0, 'the banner carries it; never a toast too');
    // The banner says the key was refused; its title never claims that no key was given.
    const banner = t.$('app-conn-banner');
    assert.equal(textOf(banner.querySelector('.banner__title')), "The server didn't accept the API key.");
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'Check it in Settings › Connection.');
    assert.doesNotMatch(textOf(banner), /This server needs an API key/);
  });

  it('[AU-09] a key saved while a /api/models request without a key is still out (the shell\'s background poll on a keyed server) is sent: one probe with the new key, reported as accepted', async () => {
    t = await bootShell({ routes: { 'GET /api/models': slowKeyed('right-key', 3000) } });
    await settle(t.env, 3100);
    assert.equal(t.app.get().connection.state, 'unauthorized', 'the page load was refused');
    // The status poll's next GET /api/models goes out without a key (none is stored).
    const models = await load('core/models.js');
    const mark0 = t.env.fetch.calls.length;
    models.refresh({ silent: true });
    await settle(t.env, 100);
    assert.deepEqual(modelCalls(t, mark0).map((c) => c.headers.authorization || ''), ['']);
    const mark = t.env.fetch.calls.length;
    t.env.input(t.$('se-api-key'), 'right-key', { change: true });
    for (let i = 0; i < 8; i += 1) await settle(t.env, 1000);
    assert.deepEqual(modelCalls(t, mark).map((c) => c.headers.authorization), ['Bearer right-key'], 'the new key went out once');
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    assert.equal(t.app.get().connection.state, 'online');
    assert.equal(t.app.get().models.list.length, 3, 'what the server refused is loaded now');
    assert.equal(t.app.get().models.error, '');
  });

  it('[AU-09] a wrong key saved while a request with the old, valid key is still out gets its own probe: the old key\'s answer never reads as accepted', async () => {
    t = await bootShell({ routes: { 'GET /api/models': slowKeyed('right-key', 3000) }, seed: { [KEY_STORAGE]: 'right-key' } });
    await settle(t.env, 3100);
    assert.equal(t.app.get().connection.state, 'online', 'the page loaded with the stored key');
    // A request with the old key (the shell's background poll) is still out when the key changes.
    const models = await load('core/models.js');
    const mark0 = t.env.fetch.calls.length;
    models.refresh({ silent: true });
    await settle(t.env, 100);
    assert.deepEqual(modelCalls(t, mark0).map((c) => c.headers.authorization), ['Bearer right-key']);
    const mark = t.env.fetch.calls.length;
    t.env.input(t.$('se-api-key'), 'wrong-key', { change: true });
    await settle(t.env, 2950);
    // The old key's request has answered 200 by now; it says nothing about the new key.
    assert.equal(statusText(t), 'Checking…');
    for (let i = 0; i < 4; i += 1) await settle(t.env, 1000);
    assert.deepEqual(modelCalls(t, mark).map((c) => c.headers.authorization), ['Bearer wrong-key'], 'one probe with the new key');
    assert.equal(statusText(t), "The server didn't accept the key.");
    assert.equal(t.app.get().connection.state, 'unauthorized');
    assert.equal(textOf(t.$('app-conn-banner').querySelector('.banner__title')), "The server didn't accept the API key.");
  });

  it('[AU-09] two keys saved within one round trip (a wrong one, then the right one): the second key is sent too, and the status line reports it', async () => {
    t = await bootShell({ routes: { 'GET /api/models': slowKeyed('right-key', 500) } });
    await settle(t.env, 600);
    assert.equal(t.app.get().connection.state, 'unauthorized');
    const input = t.$('se-api-key');
    const mark = t.env.fetch.calls.length;
    t.env.input(input, 'wrong-key', { change: true });
    await settle(t.env, 100);
    t.env.input(input, 'right-key', { change: true });
    for (let i = 0; i < 4; i += 1) await settle(t.env, 500);
    assert.deepEqual(modelCalls(t, mark).map((c) => c.headers.authorization), ['Bearer wrong-key', 'Bearer right-key'], 'each key once');
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    assert.equal(t.app.get().connection.state, 'online');
    assert.equal(t.app.get().models.list.length, 3);
    assert.deepEqual(toasts(t).map((x) => x.text), ['API key saved in this browser', 'API key saved in this browser']);
  });

  it('[SE-01] the status line follows the connection afterwards: a later refusal or lockout of the key, and the server accepting it again', async () => {
    let answer = 'ok';
    t = await mountSettings({
      seed: { [KEY_STORAGE]: 'the-key' },
      routes: {
        'GET /api/models': () => {
          if (answer === 'locked') return json(429, { error: { message: 'Too many failed API key attempts', code: 'auth_locked' } }, { 'Retry-After': '30' });
          if (answer === 'refused') return json(401, REFUSED);
          return json(200, INVENTORY);
        },
      },
    });
    const models = await load('core/models.js');
    t.env.click(t.$('se-key-test'));
    await settle(t.env);
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    // Another view's request meets a lockout: the line says so, rather than "accepted" under the banner.
    answer = 'locked';
    await models.refresh({ silent: true });
    await settle(t.env);
    assert.equal(t.app.get().connection.state, 'locked');
    assert.equal(statusText(t), 'Locked out for 30 s after too many wrong keys.');
    // The lockout ends and a request with the key passes.
    answer = 'ok';
    await models.refresh({ silent: true });
    await settle(t.env);
    assert.equal(t.app.get().connection.state, 'online');
    assert.equal(statusText(t), 'Connected. The server accepted the key.');
    // The server's key changed: the next request is refused.
    answer = 'refused';
    await models.refresh({ silent: true });
    await settle(t.env);
    assert.equal(t.app.get().connection.state, 'unauthorized');
    assert.equal(statusText(t), "The server didn't accept the key.");
  });
});
