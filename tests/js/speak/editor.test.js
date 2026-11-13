// The Speak editor (inventory SP-14, SP-15, SP-25).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { counterText, checkInput, limitMessage, estimate, EMPTY_MESSAGE } from '../../../src/static/js/views/speak/editor.js';
import { mountSpeak, ready, type, settle, callsTo, CAPS, json } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

/** Settings › Developer shows Speak › Live; off, Speak has Generate only. */
const LIVE_ON = Object.freeze({ 'open-speech-prefs': JSON.stringify({ v: 1, showLive: true }) });

describe('views/speak/editor (pure)', () => {
  it('[SP-15] the counter counts code points, locale-formatted, with the length at the current speed', () => {
    assert.equal(counterText('', { limit: 4096 }), '0 / 4,096');
    assert.equal(counterText('héllo 👋', { limit: 4096 }), '7 / 4,096 · ~1 s at 1×', 'an emoji is one character, as on the server');
    assert.equal(counterText('x'.repeat(1204), { limit: 4096, speed: 1.25 }), '1,204 / 4,096 · ~1 min at 1.25×');
    assert.equal(estimate(15 * 30, 1), '~30 s');
    assert.equal(estimate(15 * 60 * 5, 1), '~5 min');
  });

  it('[SP-14] empty text and text over the limit are refused, counting code points', () => {
    assert.equal(checkInput('   '), 'empty');
    assert.equal(checkInput('👋'.repeat(10), 10), '', 'ten emoji are ten characters');
    assert.equal(checkInput('👋'.repeat(11), 10), 'over');
    // Live is a developer tool, hidden unless Settings shows it; the Reader takes long text.
    assert.equal(limitMessage(4096), 'Generate accepts up to 4,096 characters. For longer text, use the Reader.');
    assert.equal(limitMessage(4096, { live: true }), 'Generate accepts up to 4,096 characters. Use Live for longer text.');
  });
});

describe('views/speak editor (view)', () => {
  it('[SP-14] Generate with no text shows "Enter some text to speak first." on the field and never touches the model', async () => {
    t = await mountSpeak({ models: undefined });
    await ready(t);
    t.env.fetch.reset();
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(t.$('sp-limit-hint').hidden, false);
    assert.equal(t.$('sp-limit-text').textContent, EMPTY_MESSAGE);
    assert.equal(t.$('sp-input').getAttribute('aria-invalid'), 'true');
    assert.ok(t.$('sp-input').getAttribute('aria-describedby').split(' ').includes('sp-limit-hint'));
    assert.equal(document.activeElement, t.$('sp-input'));
    assert.equal(t.env.fetch.calls.length, 0, 'no status check, no request');
    type(t, 'Now there is text.');
    assert.equal(t.$('sp-limit-hint').hidden, true, 'typing clears it');
    assert.equal(t.$('sp-input').hasAttribute('aria-invalid'), false);
  });

  it('[SP-14] over the limit: the field error names the limit, Generate is aria-disabled and described by it, and the model is never touched', async () => {
    t = await mountSpeak({ routes: { 'GET /api/tts/capabilities': () => json(200, { capabilities: CAPS.kokoro.capabilities, max_input_length: 4096, max_input_chars: 20 }) } });
    await ready(t);
    assert.equal(t.parts.inspector.limit(), 20, 'min(max_input_length, max_input_chars)');
    t.env.fetch.reset();
    type(t, '👋'.repeat(21));
    assert.equal(t.$('sp-limit-hint').hidden, false);
    assert.equal(t.$('sp-limit-text').textContent, 'Generate accepts up to 20 characters. For longer text, use the Reader.');
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'true');
    assert.equal(t.$('sp-generate').getAttribute('aria-describedby'), 'sp-limit-hint');
    t.env.click(t.$('sp-generate'));
    await settle(t.env);
    assert.equal(t.env.fetch.calls.length, 0);
    assert.equal(t.$('sp-counter').textContent.startsWith('21 / 20'), true);
    type(t, '👋'.repeat(20));
    assert.equal(t.$('sp-limit-hint').hidden, true, 'exactly the limit is fine (an emoji counts once)');
    assert.equal(t.$('sp-generate').getAttribute('aria-disabled'), 'false');
  });

  it('[SP-14] the limit applies to Generate only: Live mode (shown by Settings) reads longer text', async () => {
    t = await mountSpeak({ seed: LIVE_ON, routes: { 'GET /api/tts/capabilities': () => json(200, { capabilities: CAPS.kokoro.capabilities, max_input_length: 10 }) } });
    await ready(t);
    type(t, 'x'.repeat(30));
    assert.equal(t.$('sp-limit-hint').hidden, false);
    t.ctl.show({ view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.equal(t.$('sp-limit-hint').hidden, true);
  });

  it('[SP-15] in Live mode the counter shows the count alone: Generate\'s limit would read like an error there', async () => {
    assert.equal(counterText('', { limit: 0 }), '0 characters');
    assert.equal(counterText('x', { limit: 0 }), '1 character · ~1 s at 1×');
    assert.equal(counterText('x'.repeat(5000), { limit: 0 }), '5,000 characters · ~6 min at 1×');
    t = await mountSpeak({ seed: LIVE_ON });   // Settings shows Speak › Live
    await ready(t);
    type(t, 'x'.repeat(5000));
    assert.match(t.$('sp-counter').textContent, /^5,000 \/ 4,096/);
    t.ctl.show({ view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.match(t.$('sp-counter').textContent, /^5,000 characters · ~6 min at /);
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    assert.match(t.$('sp-counter').textContent, /^5,000 \/ 4,096/, 'back in Generate the limit shows again');
  });

  it('[SP-15] the counter updates on typing, upload, a History hand-off and Clear text', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Hello');
    assert.match(t.$('sp-counter').textContent, /^5 \/ 4,096 · ~1 s at /);
    assert.equal(t.$('sp-counter').getAttribute('aria-live'), 'off', 'the counter never announces itself');
    // Upload
    const upload = t.$('sp-upload');
    Object.defineProperty(upload, 'files', { value: [new File(['Uploaded text, twenty-nine.'], 'note.txt', { type: 'text/plain' })], configurable: true });
    upload.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(t.env);
    assert.match(t.$('sp-counter').textContent, /^27 \/ 4,096/);
    // History hand-off
    t.setIntent('speak', { type: 'apply-history', entry: { id: 'h1', type: 'tts', model: 'kokoro', voice: 'af_sky', full_text: 'From history', speed: 1, format: 'mp3', meta: {} } });
    t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' }, { focus: 'keep' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('sp-input').value, 'From history');
    assert.match(t.$('sp-counter').textContent, /^12 \/ 4,096/);
    // Clear text (More menu)
    t.env.click(t.$('sp-more'));
    await settle(t.env);
    t.env.click(t.$('sp-more-menu').querySelector('[data-more="clear"]'));
    await settle(t.env);
    assert.equal(t.$('sp-input').value, '');
    assert.equal(t.$('sp-counter').textContent, '0 / 4,096');
  });

  it('[SP-25] Open .txt replaces the text, stops Live with "Text changed" and updates the counter', async () => {
    t = await mountSpeak();
    await ready(t);
    const stops = [];
    t.ctx.live = { isActive: () => true, stop: (reason) => stops.push(reason), destroy() {} };
    type(t, 'Old text');
    const upload = t.$('sp-upload');
    assert.ok(upload.classList.contains('visually-hidden'), 'reachable by keyboard, not display:none');
    assert.equal(document.querySelector('label[for="sp-upload"]').textContent.trim(), 'Open .txt');
    assert.equal(upload.getAttribute('accept'), '.txt,text/plain');
    Object.defineProperty(upload, 'files', { value: [new File(['Brand new text.'], 'a.txt', { type: 'text/plain' })], configurable: true });
    upload.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(t.env);
    assert.equal(t.$('sp-input').value, 'Brand new text.');
    assert.deepEqual(stops, ['Text changed']);
    assert.match(t.$('sp-counter').textContent, /^15 \/ 4,096/);
    assert.equal(upload.value, '', 'the same file can be opened again');
  });

  it('the unsent draft is saved 500 ms after typing stops and restored on the next visit', async () => {
    t = await mountSpeak();
    await ready(t);
    type(t, 'Draft words');
    await t.env.clock.tickAsync(499);
    assert.equal(localStorage.getItem('open-speech-speak-draft'), null);
    await t.env.clock.tickAsync(1);
    assert.equal(localStorage.getItem('open-speech-speak-draft'), 'Draft words');
    t.env.restore();
    t = await mountSpeak({ seed: { 'open-speech-speak-draft': 'Draft words' } });
    await ready(t);
    assert.equal(t.$('sp-input').value, 'Draft words');
    assert.match(t.$('sp-counter').textContent, /^11 \//);
    type(t, '');
    await t.env.clock.tickAsync(500);
    assert.equal(localStorage.getItem('open-speech-speak-draft'), null, 'an empty editor removes the draft');
  });
});
