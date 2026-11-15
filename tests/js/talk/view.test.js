// The Talk view: the off state, the controls, the conversation log, typing, the
// errors, the screenshot states, on the real partial inside the real shell markup. The socket is the fake one;
// nothing reaches a real server or AI.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';
import {
  talkEnv, load, settle, json, text, PARTIAL, MODELS, lastSocket, accept, answer, ga, sent, sentTypes,
} from './helpers.js';

let t;
afterEach(() => {
  try { t?.ctl.destroy(); } catch { /* gone */ }
  t?.env.restore();
  t = null;
});

const ROUTE = Object.freeze({ view: 'talk', sub: '', id: '', path: 'talk', shot: '' });

async function mountView({ routes = {}, shot = '', speakVoice = { model: 'kokoro', voice: 'af_bella', label: 'Bella', kind: 'voice' }, models = null, env: envOptions = {} } = {}) {
  const env = talkEnv({ routes, ...envOptions });
  const mod = await importFresh('views/talk/index.js');
  const { app } = await load('core/store.js');
  const dom = await load('core/dom.js');
  app.set({
    models: models || { list: MODELS, defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' },
    connection: { ...app.get().connection, state: 'online' },
    route: { ...ROUTE, shot },
    speakVoice,
  });
  const root = env.document.getElementById('view-talk');
  assert.ok(root, 'the shell has a Talk section');
  dom.render(root, dom.raw(PARTIAL));
  root.hidden = false;
  const ctl = mod.default.init(root);
  ctl.show({ ...ROUTE, shot }, { focus: 'none', scrolled: false });
  await settle(env);
  const $ = (id) => env.document.getElementById(id);
  return { env, mod, ctl, app, root, $, doc: env.document };
}

const visible = (el) => {
  for (let n = el; n && n.nodeType === 1; n = n.parentElement) if (n.hidden) return false;
  return true;
};
const label = (el) => (el.querySelector('.btn__label') || el).textContent.trim();

describe('Talk view: off and loading', () => {
  it('with the AI assistant off, the view says so, links to Settings › AI assistant, and shows nothing else', async () => {
    t = await mountView({ routes: { 'GET /api/assistant': () => json(200, { enabled: false, host: null, model: '' }) } });
    assert.equal(visible(t.$('tk-off')), true);
    assert.match(t.$('tk-off').textContent, /The AI assistant is off/);
    assert.match(t.$('tk-off').textContent, /Settings › AI assistant › Connection/);
    assert.equal(visible(t.$('tk-layout')), false);
    assert.equal(visible(t.$('tk-loading')), false);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/presets').length, 0);
    t.env.click(t.$('tk-open-settings'));
    await settle(t.env);
    assert.equal(t.env.location.hash, '#settings');
  });

  it('a load that fails shows why with Retry; a refused key says "API key needed"', async () => {
    let fail = true;
    t = await mountView({ routes: { 'GET /api/assistant': () => (fail ? json(500, { error: { message: 'boom' } }) : json(200, { enabled: true, host: 'x', model: 'm' })) } });
    const banner = t.$('tk-load-error');
    assert.equal(visible(banner), true);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.match(banner.textContent, /Couldn't load the AI assistant/);
    fail = false;
    t.env.click(banner.querySelector('[data-action="load-retry"]'));
    await settle(t.env);
    assert.equal(visible(banner), false);
    assert.equal(visible(t.$('tk-layout')), true);
    t.ctl.destroy();
    t.env.restore();
    t = await mountView({ routes: { 'GET /api/assistant': () => json(401, { error: { message: 'Invalid API key' } }) } });
    assert.match(t.$('tk-load-error').textContent, /API key needed/);
  });
});

describe('Talk view: talking', () => {
  it('on: the voice presets (no book preset), Speak\'s voice by default, the empty state with its link to Transcribe', async () => {
    t = await mountView();
    assert.equal(visible(t.$('tk-layout')), true);
    assert.equal(visible(t.$('tk-off')), false);
    const options = Array.from(t.$('tk-preset').querySelectorAll('option')).map((o) => [o.value, o.textContent]);
    assert.deepEqual(options, [['voice', 'Voice'], ['short', 'Short answers']]);
    assert.equal(t.$('tk-preset').value, 'voice');
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
    assert.equal(visible(t.$('tk-empty')), true);
    const link = t.$('tk-empty').querySelector('a[href="#transcribe"]');
    assert.ok(link, 'a link to Transcribe');
    assert.match(t.$('tk-empty').textContent, /Transcribe/);
    assert.equal(label(t.$('tk-start')), 'Start talking');
    assert.equal(visible(t.$('tk-mic')), false, 'the microphone toggle shows while a conversation runs');
    assertAccessibleNames(t.root);
  });

  it('Start talking connects with the chosen preset and voice; the log shows your words, the phrases, the thinking (collapsed) and the answer', async () => {
    t = await mountView();
    t.env.change(t.$('tk-preset'), 'short');
    assert.equal(visible(t.$('tk-state')), false, 'no "Stopped" pill before the first start');
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    assert.equal(label(t.$('tk-start')), 'Stop');
    // The glyphs are SVG elements: in a browser they have no `hidden` property, so the attribute must change.
    assert.equal(t.root.querySelector('.tk-start__go').hasAttribute('hidden'), true, 'the microphone glyph goes');
    assert.equal(t.root.querySelector('.tk-start__stop').hasAttribute('hidden'), false, 'the stop glyph shows');
    assert.equal(visible(t.$('tk-state')), true);
    assert.equal(visible(t.$('tk-mic')), true);
    assert.equal(t.$('tk-mic').getAttribute('aria-pressed'), 'true');
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    const update = sent(ws)[0];
    assert.deepEqual(update.session.assistant, { preset: 'short' });
    assert.equal(update.session.audio.output.voice, 'af_bella');
    assert.equal(t.$('tk-pill').textContent.trim(), 'Listening');
    ws.serverSend(ga.speechStarted());
    await settle(t.env);
    assert.equal(visible(t.$('tk-interim')), true);
    assert.equal(t.$('tk-log').contains(t.$('tk-interim')), false, 'your words as you speak sit outside the log');
    ws.serverSend(ga.speechStopped());
    ws.serverSend(ga.transcript('item_1', 'What is the weather tomorrow?'));
    await settle(t.env);
    await answer(t.env, ws, 'resp_1', { fillers: ['Hmm.'] });
    ws.serverSend(ga.thinking('resp_2', 'Hidden steps.'));
    await settle(t.env);
    const log = t.$('tk-log');
    assert.equal(log.getAttribute('role'), 'log');
    assert.equal(log.getAttribute('aria-live'), 'off', 'the answer is heard; a screen reader reading it at once would talk over it');
    const entries = Array.from(log.querySelectorAll('.tk-entry'));
    assert.deepEqual(entries.map((e) => e.getAttribute('data-kind')), ['you', 'filler', 'ai', 'thinking']);
    assert.match(entries[0].textContent, /You.*What is the weather tomorrow\?/s);
    assert.match(entries[2].textContent, /Sunny, with a light wind\./);
    const details = entries[3].querySelector('details');
    assert.ok(details && !details.open, 'the thinking is collapsed');
    assert.equal(visible(t.$('tk-empty')), false);
    assertAccessibleNames(t.root);
  });

  it('the status region speaks only phase changes, never each turn', async () => {
    t = await mountView();
    const region = t.$('tk-status');
    assert.equal(region.getAttribute('role'), 'status');
    const said = [];
    let proto = Object.getPrototypeOf(region);
    while (proto && !Object.getOwnPropertyDescriptor(proto, 'textContent')) proto = Object.getPrototypeOf(proto);
    const inner = Object.getOwnPropertyDescriptor(proto, 'textContent');
    Object.defineProperty(region, 'textContent', { configurable: true, get() { return inner.get.call(this); }, set(v) { if (String(v)) said.push(String(v)); inner.set.call(this, v); } });
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    ws.serverSend(ga.speechStarted());
    ws.serverSend(ga.speechStopped());
    ws.serverSend(ga.transcript('item_1', 'Hi'));
    await settle(t.env);
    await answer(t.env, ws, 'resp_1');
    await settle(t.env, 1000);
    assert.deepEqual(said, ['Connecting…', 'Listening']);
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    assert.equal(said.at(-1), 'Stopped');
  });

  it('typing: Enter sends, Shift+Enter does not; the box empties once the question is taken; a refusal shows next to it and keeps the text', async () => {
    t = await mountView();
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    const box = t.$('tk-text');
    assert.equal(box.getAttribute('maxlength'), '8000');
    box.focus();
    t.env.input(box, 'First line');
    t.env.key('Enter', { shiftKey: true }, box);
    await settle(t.env);
    assert.equal(sentTypes(ws).includes('conversation.item.create'), false);
    t.env.input(box, 'What time is it?');
    t.env.key('Enter', {}, box);
    await settle(t.env);
    const create = sent(ws).at(-1);
    assert.equal(create.type, 'conversation.item.create');
    assert.equal(create.item.content[0].text, 'What time is it?');
    assert.equal(box.value, 'What time is it?', 'kept until the server takes it');
    assert.equal(t.$('tk-send').getAttribute('aria-busy'), 'true');
    ws.serverSend(ga.typedAdded(create.item.id, 'What time is it?'));
    await settle(t.env);
    assert.equal(box.value, '');
    assert.equal(sentTypes(ws).at(-1), 'response.create');
    const you = t.$('tk-log').querySelector('.tk-entry[data-kind="you"]');
    assert.match(you.textContent, /What time is it\?/);
    t.env.input(box, 'Another');
    t.env.click(t.$('tk-send'));
    await settle(t.env);
    const second = sent(ws).at(-1);
    ws.serverSend(ga.error('assistant_disabled', 'The AI assistant is off, so nothing would answer a typed message', { event_id: second.event_id }));
    await settle(t.env);
    const err = t.$('tk-text-error');
    assert.equal(visible(err), true);
    assert.match(err.textContent, /The AI assistant is off/);
    assert.equal(box.getAttribute('aria-invalid'), 'true');
    assert.equal(box.value, 'Another');
    t.env.input(box, 'Another, edited');
    await settle(t.env);
    assert.equal(visible(err), false, 'editing clears the refusal');
  });

  it('typing with Talk stopped starts it with the microphone off', async () => {
    t = await mountView();
    t.env.input(t.$('tk-text'), 'Hello?');
    t.env.click(t.$('tk-send'));
    await settle(t.env);
    assert.equal(t.env.mediaDevices.calls.length, 0);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    assert.equal(sentTypes(ws).at(-1), 'conversation.item.create');
    assert.equal(t.$('tk-mic').getAttribute('aria-pressed'), 'false');
  });

  it('the preset, the voice and the interrupt switch change the running session', async () => {
    t = await mountView();
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    t.env.change(t.$('tk-interrupt'), false);
    await settle(t.env);
    assert.equal(sent(ws).at(-1).session.audio.input.turn_detection.interrupt_response, false);
    t.env.change(t.$('tk-preset'), 'short');
    await settle(t.env);
    assert.deepEqual(sent(ws).at(-1).session.assistant, { preset: 'short' });
    t.app.set({ speakVoice: { model: 'kokoro', voice: 'af_sky', label: 'Sky', kind: 'voice' } });
    await settle(t.env);
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Sky/);
    assert.equal(sent(ws).at(-1).session.audio.output.voice, 'af_sky', 'following Speak, a new Speak voice is used at once');
  });

  it('a refused microphone shows a plain banner and Talk goes on for typing; a stop shows its reason with Retry', async () => {
    t = await mountView({ env: { mediaDevices: { policy: 'NotAllowedError' } } });
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    const help = t.$('tk-mic-help');
    assert.equal(visible(help), true);
    assert.match(help.textContent, /Microphone problem/);
    assert.match(help.textContent, /You can still type your questions\./);
    ws.serverClose(4001, 'Invalid or missing API key');
    await settle(t.env);
    const banner = t.$('tk-error');
    assert.equal(visible(banner), true);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.match(banner.textContent, /Talk stopped/);
    assert.match(banner.textContent, /API key/);
    assert.ok(banner.querySelector('[data-action="open-settings"]'));
    assert.equal(label(t.$('tk-start')), 'Start talking');
  });

  it('New conversation empties the log; Stop reply shows only while the AI answers', async () => {
    t = await mountView();
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    ws.serverSend(ga.transcript('item_1', 'Remember 7'));
    ws.serverSend(ga.responseCreated('resp_1'));
    await settle(t.env);
    assert.equal(visible(t.$('tk-cancel')), true);
    t.env.click(t.$('tk-cancel'));
    await settle(t.env);
    assert.equal(sentTypes(ws).at(-1), 'response.cancel');
    assert.equal(visible(t.$('tk-cancel')), false);
    assert.equal(t.$('tk-log').querySelectorAll('.tk-entry').length, 1);
    t.env.click(t.$('tk-new'));
    await settle(t.env);
    assert.equal(t.$('tk-log').querySelectorAll('.tk-entry').length, 0);
    assert.equal(visible(t.$('tk-empty')), true);
    assert.notEqual(lastSocket(t.env), ws);
  });

  it('leaving the view keeps the conversation; the mini-player can bring you back', async () => {
    t = await mountView();
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    await accept(t.env, lastSocket(t.env));
    t.ctl.hide();
    t.app.set({ route: { view: 'reader', sub: '', id: '', path: 'reader', shot: '' } });
    await settle(t.env);
    const player = t.app.get().player;
    assert.ok(player, 'the audio session is mirrored for the mini-player');
    assert.equal(player.homeRoute, 'talk');
    assert.equal(lastSocket(t.env).readyState, 1, 'still connected');
  });
});

// The AI preset, the reply voice and "Interrupt by talking" are remembered per browser,
// in prefs.talk (core/prefs.js); a remembered preset or voice that no longer exists falls back to the default
// without an error.
describe('Talk view: remembered choices', () => {
  const VOICES = Object.freeze({
    'GET /v1/audio/voices': () => json(200, { voices: [{ id: 'af_bella', name: 'Bella' }, { id: 'af_sky', name: 'Sky' }] }),
    'GET /api/voices/identities': () => json(200, { voices: [{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', voice: 'af_sky', available: true }] }] }),
  });
  const storedTalk = (env) => JSON.parse(env.localStorage.__data['open-speech-prefs'] || '{}').talk;
  const seedTalk = (talk) => ({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, talk: { v: 1, ...talk } }) } });
  const quiet = () => {
    for (const id of ['tk-error', 'tk-choice-error', 'tk-load-error', 'tk-refusal']) assert.equal(visible(t.$(id)), false, `${id} stays hidden`);
    assert.deepEqual(t.env.errors, []);
  };

  async function pick(optionLabel) {
    t.env.click(t.$('tk-voice'));
    await settle(t.env);
    await settle(t.env);
    const option = [...t.doc.querySelectorAll('#vp-list [role=option]')].find((o) => o.querySelector('.option__label').textContent === optionLabel);
    assert.ok(option, `option ${optionLabel}`);
    t.env.click(option);
    await settle(t.env);
    await settle(t.env);
  }

  async function firstUpdate() {
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    return { ws, update: sent(ws)[0] };
  }

  it('the preset, the reply voice and Interrupt by talking are remembered in this browser, and the next page talks with them', async () => {
    t = await mountView({ routes: VOICES });
    t.env.change(t.$('tk-preset'), 'short');
    t.env.change(t.$('tk-interrupt'), false);
    await pick('Sky');
    assert.match(t.$('tk-voice').textContent, /Sky/);
    assert.deepEqual(storedTalk(t.env), { v: 1, preset: 'short', voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky' }, interrupt: false },
      'kept with the other preferences (open-speech-prefs)');
    const seed = { 'open-speech-prefs': t.env.localStorage.__data['open-speech-prefs'] };
    t.ctl.destroy();
    t.env.restore();

    // The next page load.
    t = await mountView({ routes: VOICES, env: { seed } });
    assert.equal(t.$('tk-preset').value, 'short', 'the remembered preset waits for the list, then shows');
    assert.equal(t.$('tk-interrupt').checked, false);
    assert.match(t.$('tk-voice').textContent, /Sky/);
    assert.doesNotMatch(t.$('tk-voice').textContent, /Same as Speak/);
    const { ws, update } = await firstUpdate();
    assert.deepEqual(update.session.assistant, { preset: 'short' });
    assert.equal(update.session.audio.output.voice, 'af_sky');
    assert.equal(update.session.audio.input.turn_detection.interrupt_response, false);

    // "Same as Speak" follows Speak again, and that is remembered too.
    await pick('Same as Speak · Bella');
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
    assert.equal(storedTalk(t.env).voice, null);
    assert.equal(sent(ws).at(-1).session.audio.output.voice, 'af_bella');
    quiet();
  });

  it('a change the session refuses goes back in the remembered choices too', async () => {
    t = await mountView();
    const { ws } = await firstUpdate();
    t.env.change(t.$('tk-preset'), 'short');
    await settle(t.env);
    assert.equal(storedTalk(t.env).preset, 'short');
    const update = sent(ws).at(-1);
    ws.serverSend(ga.error('invalid_request', 'Unknown AI preset: short', { event_id: update.event_id }));
    await settle(t.env);
    assert.match(t.$('tk-choice-error').textContent, /Unknown AI preset: short/);
    assert.equal(t.$('tk-preset').value, 'voice');
    assert.equal(storedTalk(t.env).preset, 'voice', 'the next page uses what the session took');
  });

  it("a remembered preset that no longer exists, or a voice the model doesn't have any more, falls back to the default without an error", async () => {
    const gone = { model: 'kokoro', voice: 'af_gone', label: 'Gone' };
    t = await mountView({ routes: VOICES, env: seedTalk({ preset: 'deleted', voice: gone }) });
    assert.equal(t.$('tk-preset').value, 'voice');
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
    quiet();
    const { update } = await firstUpdate();
    assert.deepEqual(update.session.assistant, { preset: 'voice' });
    assert.equal(update.session.audio.output.voice, 'af_bella');
    quiet();
    assert.deepEqual(storedTalk(t.env), { v: 1, preset: 'deleted', voice: gone },
      'a fallback writes nothing: the remembered choices stay until the listener chooses (a worker that is down comes back)');
    t.ctl.destroy();
    t.env.restore();

    // A model this server can't speak with now: no voice list is asked for.
    t = await mountView({ routes: VOICES, env: seedTalk({ voice: { model: 'qwen3-tts', voice: 'Ryan', label: 'Ryan' } }) });
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
    assert.equal(t.env.fetch.callsTo('GET /v1/audio/voices').length, 0);
    quiet();
  });

  // Talk's reply-voice picker shows the cloning-model note too, as Speak's
  // does. Talk offers no raw recordings, so the note is its first sentence, without Clone a voice….
  it('the reply-voice picker for a model with no voices of its own says so and lists the named voices that have no version for it last', async () => {
    const CLONING = {
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'chatterbox' ? [] : [{ id: 'af_bella', name: 'Bella' }] }),
      'GET /api/voices/identities': () => json(200, { voices: [
        { id: 'narrator', name: 'Narrator', available: false, realizations: [{ model: 'kokoro', voice: 'af_sky', available: false, reason: "Select this realization's model" }] },
        { id: 'butler', name: 'Butler', available: true, realizations: [{ model: 'chatterbox', voice: 'butler_take', available: true }] },
      ] }),
    };
    const models = { list: [...MODELS, { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded' }], defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' };
    t = await mountView({ routes: CLONING, models, speakVoice: { model: 'chatterbox', voice: 'voice:butler', label: 'Butler', kind: 'named' } });
    t.env.click(t.$('tk-voice'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('app-voice-picker').open, true);
    assert.equal(t.$('vp-model-note').hidden, false, 'the note shows');
    assert.equal(t.$('vp-model-note-text').textContent, 'Chatterbox has no built-in voices.');
    assert.equal(t.$('vp-clone').hidden, true, 'no raw recordings here: no Clone a voice…');
    const groupNames = [...t.doc.querySelectorAll('#vp-list [role=group] .listbox__label')].map((g) => g.textContent);
    assert.deepEqual(groupNames, ['Suggested', 'Named voices', 'Named voices with no Chatterbox version']);
  });

  // Start pressed before the check of a remembered voice ended sent a deleted voice,
  // and the server refused the session ("The session's settings weren't accepted.").
  it('Start pressed while the remembered voice is still being checked waits for the check: a deleted voice is never sent, a kept one is', async () => {
    for (const [named, want] of [[[], 'af_bella'], [null, 'voice:narrator']]) {
      let release;
      const held = new Promise((resolve) => { release = resolve; });
      const list = named || [{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', voice: 'af_sky', available: true }] }];
      t = await mountView({
        routes: { ...VOICES, 'GET /api/voices/identities': () => held.then(() => json(200, { voices: list })) },
        env: seedTalk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }),
      });
      assert.equal(t.env.fetch.callsTo('GET /api/voices/identities').length, 1, 'the check waits for the list');
      t.env.click(t.$('tk-start'));
      await settle(t.env);
      assert.equal(t.env.sockets.length, 0, 'nothing connects before the check ends');
      release();
      await settle(t.env);
      const ws = lastSocket(t.env);
      assert.ok(ws, 'then Talk connects');
      await accept(t.env, ws);
      assert.equal(sent(ws)[0].session.audio.output.voice, want, named ? "deleted: Speak's voice, quietly" : 'still offered: kept');
      assert.equal(label(t.$('tk-start')), 'Stop');
      assert.equal(t.env.fetch.callsTo('GET /api/voices/identities').length, 1, 'checked once');
      quiet();
      t.ctl.destroy();
      t.env.restore();
      t = null;
    }
  });

  // Talk can't speak in a cloned voice yet (/v1/realtime refuses a named
  // voice whose version holds a recording: "The session's settings weren't accepted."). In Talk's picker such rows
  // show but are disabled with "Can't speak in Talk yet"; a remembered or "Same as Speak" cloned voice falls back
  // quietly to the server default (Kokoro's recommended voice here).
  describe('a cloned voice', () => {
    const CHATTERBOX_MODELS = { list: [...MODELS, { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded' }], defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' };
    const BUTLER = { model: 'chatterbox', voice: 'voice:butler', label: 'Butler', kind: 'named' };
    // As services/tts.named_voice_catalog answers ?model=: a version for another model is unavailable, and so is the
    // voice; Butler's Chatterbox version is a recording (reference_audio_id), which Chatterbox speaks elsewhere.
    const identities = (req) => {
      const model = req.query.get('model');
      const realizations = [{ id: 'r1', voice_identity_id: 'butler', model: 'chatterbox', voice: 'butler_take', reference_audio_id: 'butler_take' }]
        .map((r) => (r.model === model ? { ...r, available: true, reason: null } : { ...r, available: false, reason: "Select this realization's model" }));
      return json(200, { voices: [{ id: 'butler', name: 'Butler', available: realizations.some((r) => r.available), preset_count: 0, realizations }] });
    };
    const CLONED = Object.freeze({
      'GET /v1/audio/voices': (req) => json(200, { voices: req.query.get('model') === 'chatterbox' ? [] : [{ id: 'af_heart', name: 'Heart' }, { id: 'af_bella', name: 'Bella' }] }),
      'GET /api/voices/identities': identities,
    });
    async function pickerRows() {
      t.env.click(t.$('tk-voice'));
      await settle(t.env);
      await settle(t.env);
      assert.equal(t.$('app-voice-picker').open, true, 'the picker opened');
      return [...t.doc.querySelectorAll('#vp-list [role=option]')].map((o) => ({
        el: o,
        label: o.querySelector('.option__label').textContent,
        detail: (o.querySelector('.option__detail') || { textContent: '' }).textContent,
        group: o.closest('[role=group]').querySelector('.listbox__label').textContent,
        disabled: o.getAttribute('aria-disabled') === 'true',
      }));
    }

    it("Speak's voice is a cloned voice: Talk falls back to the server default quietly, the Same as Speak row says why it can't be chosen, and Start talks", async () => {
      t = await mountView({ routes: CLONED, models: CHATTERBOX_MODELS, speakVoice: BUTLER });
      assert.doesNotMatch(t.$('tk-voice').textContent, /Same as Speak/, 'not following a voice Talk cannot speak in');
      assert.match(t.$('tk-voice').textContent, /Heart/, "Kokoro's recommended voice, as with no Speak voice");
      quiet();
      const rows = await pickerRows();
      const same = rows.find((r) => r.label === 'Same as Speak · Butler');
      assert.ok(same, 'the row still shows');
      assert.deepEqual([same.group, same.detail, same.disabled], ['Suggested', "Can't speak in Talk yet", true]);
      t.env.click(same.el);
      await settle(t.env);
      assert.equal(t.$('app-voice-picker').open, true, 'never chosen');
      t.env.key('Escape', {}, t.$('vp-search'));
      await settle(t.env);
      const { update } = await firstUpdate();
      assert.equal(update.session.model, 'kokoro');
      assert.equal(update.session.audio.output.voice, 'af_heart', 'never voice:butler, which /v1/realtime refuses');
      assert.equal(label(t.$('tk-start')), 'Stop');
      quiet();
    });

    it('for a cloning model, the named voices whose version is a recording show disabled with the reason; a remembered one falls back quietly', async () => {
      // The server's default model is Chatterbox, and Speak speaks a raw recording (never followed): Talk's picker opens for Chatterbox.
      const models = { ...CHATTERBOX_MODELS, defaults: { ...CHATTERBOX_MODELS.defaults, tts: 'chatterbox' } };
      t = await mountView({ routes: CLONED, models, speakVoice: { model: 'chatterbox', voice: 'butler_take', label: 'butler_take', kind: 'recording' } });
      const rows = await pickerRows();
      const butler = rows.find((r) => r.label === 'Butler');
      assert.deepEqual([butler.group, butler.detail, butler.disabled], ['Named voices', "Can't speak in Talk yet", true]);
      t.env.click(butler.el);
      await settle(t.env);
      assert.equal(t.$('app-voice-picker').open, true, 'never chosen');
      t.env.key('Escape', {}, t.$('vp-search'));
      await settle(t.env);
      t.ctl.destroy();
      t.env.restore();

      // Remembered from before: Talk follows Speak again, and nothing is written (a fallback is silent).
      t = await mountView({ routes: CLONED, models: CHATTERBOX_MODELS, env: seedTalk({ voice: { model: 'chatterbox', voice: 'voice:butler', label: 'Butler' } }) });
      assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
      const { update } = await firstUpdate();
      assert.equal(update.session.audio.output.voice, 'af_bella');
      assert.deepEqual(storedTalk(t.env).voice, { model: 'chatterbox', voice: 'voice:butler', label: 'Butler' });
      quiet();
    });

    it("Start pressed while Speak's voice is still being checked waits for the check: a cloned voice is never sent", async () => {
      let release;
      const held = new Promise((resolve) => { release = resolve; });
      t = await mountView({ routes: { ...CLONED, 'GET /api/voices/identities': (req) => held.then(() => identities(req)) }, models: CHATTERBOX_MODELS, speakVoice: BUTLER });
      t.env.click(t.$('tk-start'));
      await settle(t.env);
      assert.equal(t.env.sockets.length, 0, 'nothing connects before the check ends');
      release();
      await settle(t.env);
      const ws = lastSocket(t.env);
      assert.ok(ws, 'then Talk connects');
      await accept(t.env, ws);
      assert.equal(sent(ws)[0].session.audio.output.voice, 'af_heart');
      quiet();
    });
  });

  it('a remembered named voice is kept while the server offers it for the model; the voice check waits for the model list, and a list that fails to load keeps the voice', async () => {
    t = await mountView({ routes: VOICES, env: seedTalk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }) });
    assert.match(t.$('tk-voice').textContent, /Narrator/);
    assert.equal((await firstUpdate()).update.session.audio.output.voice, 'voice:narrator');
    quiet();
    t.ctl.destroy();
    t.env.restore();

    for (const named of [[], [{ id: 'narrator', name: 'Narrator', available: true, realizations: [{ model: 'kokoro', available: false, reason: 'No Kokoro version' }] }]]) {
      t = await mountView({ routes: { ...VOICES, 'GET /api/voices/identities': () => json(200, { voices: named }) }, env: seedTalk({ voice: { model: 'kokoro', voice: 'voice:narrator', label: 'Narrator' } }) });
      assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/, named.length ? 'unavailable for the model' : 'deleted');
      quiet();
      t.ctl.destroy();
      t.env.restore();
    }

    // The voice list fails to load: nothing says the voice is gone, so it stays.
    t = await mountView({ routes: { ...VOICES, 'GET /v1/audio/voices': () => json(503, { error: { message: 'busy' } }) }, env: seedTalk({ voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky' } }) });
    assert.match(t.$('tk-voice').textContent, /Sky/);
    quiet();
    t.ctl.destroy();
    t.env.restore();

    // A page that opens on Talk before the model list is known: the check waits for it.
    t = await mountView({
      routes: VOICES, env: seedTalk({ voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky' } }),
      models: { list: [], defaults: { tts: '', stt: '' }, fetchedAt: 0, error: '' },
    });
    assert.match(t.$('tk-voice').textContent, /Sky/);
    assert.equal(t.env.fetch.callsTo('GET /v1/audio/voices').length, 0, 'not before the model list');
    t.app.set({ models: { list: MODELS.filter((m) => m.type !== 'tts'), defaults: { tts: '', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' } });
    await settle(t.env);
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/, 'Kokoro is not among the models');
    quiet();
  });
});

// A Delete chat control in the conversation panel's header shows
// whenever the chat has entries, running or not. While Talk runs it does what New conversation does (the AI forgets
// the conversation, the microphone stays); while Talk is stopped it empties the chat on screen, with no confirm and
// an Undo toast for 5 s. Nothing is stored, so nothing else is deleted.
describe('Talk view: Delete chat', () => {
  const entries = () => Array.from(t.$('tk-log').querySelectorAll('.tk-entry'));
  const chatToast = () => Array.from(t.doc.querySelectorAll('#app-toasts .toast')).find((el) => /Chat deleted/.test(el.textContent)) || null;
  const toastActions = (el) => Array.from(el.querySelectorAll('button')).filter((b) => b.getAttribute('aria-label') !== 'Dismiss notification');
  const openDialogs = () => Array.from(t.doc.querySelectorAll('dialog')).filter((d) => d.open);

  /** A conversation with your words and an answer, then Stop: the chat stays on screen. */
  async function chatThenStop() {
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    ws.serverSend(ga.transcript('item_1', 'Remember the number 7.'));
    await settle(t.env);
    await answer(t.env, ws, 'resp_1', { text: 'I will remember 7.' });
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    return ws;
  }

  it('with Talk stopped, Delete chat empties the chat on screen, with no confirm, and Undo brings it back for 5 s', async () => {
    t = await mountView();
    const del = t.$('tk-delete');
    assert.ok(del, 'a Delete chat control');
    assert.equal(visible(del), false, 'an empty chat has nothing to delete');
    await chatThenStop();
    assert.equal(label(t.$('tk-start')), 'Start talking', 'Talk is stopped');
    assert.equal(visible(t.$('tk-new')), false, 'New conversation shows only while Talk runs');
    const before = entries().map((e) => e.textContent);
    assert.equal(before.length, 2);
    assert.equal(visible(del), true, 'shown while the stopped chat has entries');
    assert.equal(del.disabled, false);
    assert.equal(label(del), 'Delete chat');
    assert.ok(del.querySelector('svg use[href$="#i-trash"]'), 'an icon with the text label');
    assert.equal(t.$('tk-panel').contains(del), true, 'in the conversation panel');
    assert.equal(t.$('tk-log').contains(del), false, 'in its header, not among the entries');
    const sockets = t.env.sockets.length;
    const calls = t.env.fetch.calls.length;
    t.env.click(del);
    await settle(t.env);
    assert.deepEqual(openDialogs(), [], 'no confirm');
    assert.equal(entries().length, 0);
    assert.equal(visible(t.$('tk-log')), false);
    assert.equal(visible(t.$('tk-empty')), true, 'the empty state again');
    assert.equal(visible(del), false);
    assert.equal(t.doc.activeElement, t.$('tk-log-title'), 'focus moves to the conversation heading, not lost with the button');
    assert.equal(label(t.$('tk-start')), 'Start talking', 'Talk stays stopped');
    assert.equal(t.env.sockets.length, sockets, 'nothing connects');
    assert.equal(t.env.fetch.calls.length, calls, 'nothing is asked of the server: nothing is stored');
    const note = chatToast();
    assert.ok(note, 'a toast says so');
    assert.equal(note.parentElement.getAttribute('role'), 'status');
    const undo = toastActions(note).find((b) => b.textContent.trim() === 'Undo');
    assert.ok(undo, 'with Undo');
    t.env.click(undo);
    await settle(t.env);
    assert.deepEqual(entries().map((e) => e.textContent), before, 'Undo brings the chat back');
    assert.equal(visible(del), true);
    assert.equal(chatToast(), null, 'the toast goes with Undo');
    // Undo is offered for 5 s.
    t.env.click(del);
    await settle(t.env, 4900);
    assert.ok(chatToast(), 'still offered at 4.9 s');
    await settle(t.env, 200);
    assert.equal(chatToast(), null, 'gone after 5 s');
    assert.equal(entries().length, 0, 'and the chat stays deleted');
    assert.deepEqual(t.env.errors, []);
    assertAccessibleNames(t.root);
  });

  it("while Talk runs, Delete chat does what New conversation does: the AI forgets, the microphone stays, and the toast has no Undo", async () => {
    t = await mountView();
    const del = t.$('tk-delete');
    assert.ok(del, 'a Delete chat control');
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    assert.equal(visible(del), false, 'nothing to delete yet');
    ws.serverSend(ga.transcript('item_1', 'Remember the number 7.'));
    await settle(t.env);
    assert.equal(visible(del), true, 'shown once the chat has an entry');
    assert.equal(del.disabled, false);
    t.env.click(del);
    await settle(t.env);
    assert.deepEqual(openDialogs(), [], 'no confirm');
    assert.equal(ws.readyState, 3, 'the old socket closed: the AI forgets the conversation');
    const ws2 = lastSocket(t.env);
    assert.notEqual(ws2, ws, 'a fresh conversation connects');
    assert.equal(entries().length, 0);
    assert.equal(visible(t.$('tk-empty')), true);
    assert.equal(visible(del), false);
    assert.equal(t.doc.activeElement, t.$('tk-log-title'), 'focus moves to the conversation heading');
    assert.equal(label(t.$('tk-start')), 'Stop', 'Talk keeps running');
    assert.equal(t.$('tk-mic').getAttribute('aria-pressed'), 'true', 'the microphone stays on');
    assert.equal(t.env.mediaDevices.calls.length, 1, 'and is not asked for again');
    assert.equal(visible(t.$('tk-new')), true, 'New conversation stays where it is');
    const note = chatToast();
    assert.ok(note, 'a toast says the conversation was cleared');
    assert.match(note.textContent, /The AI doesn't remember it\./);
    assert.deepEqual(toastActions(note), [], "no Undo: the AI's memory can't come back");
    await accept(t.env, ws2);
    assert.equal(entries().length, 0, 'no "Reconnected" note');
    assert.deepEqual(t.env.errors, []);
  });

  it('over an old chat, Delete chat waits while the new conversation connects, as New conversation does; starting again ends an Undo still offered', async () => {
    t = await mountView();
    await chatThenStop();
    const del = t.$('tk-delete');
    assert.ok(del, 'a Delete chat control');
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    assert.equal(visible(del), true, 'the old chat is still on screen');
    assert.equal(del.disabled, true, 'not before the session is ready');
    assert.equal(t.$('tk-new').disabled, true, 'like New conversation');
    await accept(t.env, lastSocket(t.env));
    assert.equal(del.disabled, false);
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    t.env.click(del);
    await settle(t.env);
    assert.ok(chatToast(), 'Undo is offered');
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    assert.equal(chatToast(), null, 'starting again ends the Undo: it is for a stopped chat');
    await accept(t.env, lastSocket(t.env));
    assert.equal(entries().length, 0, "the new conversation starts with an empty chat, with no \"doesn't remember\" note");
  });

  it('?shot=talk-stopped shows the sample chat with Talk stopped and Delete chat in the header; nothing connects', async () => {
    t = await mountView({ shot: 'talk-stopped' });
    const del = t.$('tk-delete');
    assert.ok(del, 'a Delete chat control');
    assert.equal(visible(del), true);
    assert.equal(visible(t.$('tk-layout')), true);
    assert.equal(label(t.$('tk-start')), 'Start talking');
    assert.equal(visible(t.$('tk-actions')), false);
    assert.equal(t.$('tk-pill').textContent.trim(), 'Stopped');
    assert.ok(entries().length > 0);
    assert.equal(t.env.sockets.length, 0);
    assert.equal(t.env.mediaDevices.calls.length, 0);
    assertAccessibleNames(t.root);
    t.ctl.destroy();
    t.env.restore();
    t = await mountView({ shot: 'talk-conversation' });
    assert.equal(visible(t.$('tk-delete')), true, 'and in the running sample');
  });
});

// A streamed answer cut off by an error or the token limit
// (response.done `incomplete`, the code in status_details.error.code) gets a short note under it.
describe('Talk view: a cut-off answer', () => {
  const CUT_OFF = 'The answer was cut off: the AI reached its token limit.';
  const entries = () => Array.from(t.$('tk-log').querySelectorAll('.tk-entry'));

  it('the note shows under the cut-off answer, styled like the other notes, and is announced politely once the answer has played', async () => {
    t = await mountView();
    (await load('core/a11y.js')).__resetForTests();
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    ws.serverSend(ga.transcript('item_1', 'Tell me about the Moon.'));
    ws.serverSend(ga.responseCreated('resp_1'));
    ws.serverSend(ga.textDelta('resp_1', 'The Moon is about 384,000 km away.'));
    ws.serverSend(ga.audio('resp_1', 2400));
    ws.serverSend(ga.textDelta('resp_1', ' It has no air.'));
    ws.serverSend(ga.audio('resp_1', 2400));
    ws.serverSend(ga.audioDone('resp_1'));
    // Your next words, written down while the answer still plays (Interrupt by talking off), come in first.
    ws.serverSend(ga.transcript('item_2', 'And the Sun?'));
    ws.serverSend(ga.done('resp_1', 'incomplete', { type: 'incomplete', error: { type: 'server_error', code: 'assistant_reply_cut_off' } }));
    await settle(t.env);
    assert.deepEqual(entries().map((e) => [e.getAttribute('data-kind'), e.querySelector('.tk-entry__text').textContent]), [
      ['you', 'Tell me about the Moon.'],
      ['ai', 'The Moon is about 384,000 km away. It has no air.'],
      ['note', CUT_OFF],
      ['you', 'And the Sun?'],
    ], 'the note sits right under the answer it is about');
    const note = entries()[2];
    assert.equal(note.querySelector('.tk-entry__text').textContent, CUT_OFF, 'a plain note, like "Reconnected"');
    assert.equal(note.querySelector('.tk-entry__who'), null);
    assert.equal(t.$('tk-log').getAttribute('aria-live'), 'off', 'the log stays quiet');
    assert.equal(visible(t.$('tk-error')), false, 'not a stop');
    assert.equal(label(t.$('tk-start')), 'Stop');
    const said = () => t.doc.getElementById('app-sr-status').textContent;
    assert.notEqual(said(), CUT_OFF, 'not over the answer\'s sound');
    await settle(t.env, 1000);
    assert.equal(said(), CUT_OFF, 'announced politely');
    assert.deepEqual(t.env.errors, []);
    assertAccessibleNames(t.root);
  });

  it('?shot=talk-cut-off shows a running conversation whose last answer was cut off, with its note; nothing connects', async () => {
    t = await mountView({ shot: 'talk-cut-off' });
    assert.ok(t.mod.SHOTS.includes('talk-cut-off'), 'a screenshot state');
    assert.equal(visible(t.$('tk-layout')), true);
    const kinds = entries().map((e) => e.getAttribute('data-kind'));
    assert.deepEqual(kinds.slice(-2), ['ai', 'note'], 'the cut-off answer, then its note');
    assert.equal(entries().at(-1).textContent.trim(), CUT_OFF);
    assert.equal(label(t.$('tk-start')), 'Stop');
    assert.equal(t.env.sockets.length, 0);
    assert.equal(t.env.mediaDevices.calls.length, 0);
    assertAccessibleNames(t.root);
  });
});

describe('Talk view: screenshot states', () => {
  it('?shot=talk-conversation shows a sample conversation without connecting; ?shot=talk-off shows the off state', async () => {
    t = await mountView({ shot: 'talk-conversation' });
    const kinds = Array.from(t.$('tk-log').querySelectorAll('.tk-entry')).map((e) => e.getAttribute('data-kind'));
    for (const kind of ['you', 'filler', 'thinking', 'ai', 'error', 'note']) assert.ok(kinds.includes(kind), kind);
    assert.equal(t.env.sockets.length, 0);
    assert.equal(t.env.mediaDevices.calls.length, 0);
    assertAccessibleNames(t.root);
    t.ctl.destroy();
    t.env.restore();
    t = await mountView({ shot: 'talk-off' });
    assert.equal(visible(t.$('tk-off')), true);
    assert.equal(visible(t.$('tk-layout')), false);
  });

  // With the server's assistant off, a staged conversation showed the off state too.
  it('with the AI assistant off, a staged conversation shows without "The AI assistant is off"; ?shot=talk-off still shows it', async () => {
    const off = { 'GET /api/assistant': () => json(200, { enabled: false, host: null, model: '' }) };
    for (const shot of ['talk-conversation', 'talk-stopped', 'talk-cut-off']) {
      t = await mountView({ shot, routes: off });
      assert.equal(visible(t.$('tk-layout')), true, `${shot}: the staged conversation`);
      assert.equal(visible(t.$('tk-off')), false, `${shot}: one state at a time, not the off state as well`);
      assert.equal(visible(t.$('tk-loading')), false, shot);
      t.ctl.destroy();
      t.env.restore();
      t = null;
    }
    t = await mountView({ shot: 'talk-off', routes: off });
    assert.equal(visible(t.$('tk-off')), true);
    assert.equal(visible(t.$('tk-layout')), false);
  });
});

/* ───────────── who answers ───────────── */

const ASSISTANTS = Object.freeze([
  { id: 'a-hermes', name: 'Hermes', url: 'http://hermes.lan:8642/v1', model: 'hermes-agent', api_key_set: true, sources: {}, enabled: true },
  { id: 'a-strata', name: 'Strata', url: 'http://strata.lan:9000/v1', model: 'strata-1', api_key_set: false, sources: {}, enabled: true },
]);

/** The assistant routes: the status names the one in use; activate puts another in use. */
function assistantRoutes({ active = 'a-hermes', presets = null, refuse = false } = {}) {
  const state = { active, activates: [] };
  const list = () => ({ active: state.active, locked: false, assistants: ASSISTANTS.map((a) => ({ ...a, active: a.id === state.active })) });
  const routes = {
    'GET /api/assistant': () => {
      const a = ASSISTANTS.find((x) => x.id === state.active);
      return json(200, { enabled: true, host: new URL(a.url).host, model: a.model, server_settings: {}, locked: false, sources: {}, assistant: { id: a.id, name: a.name } });
    },
    'GET /api/assistant/assistants': () => json(200, list()),
    'POST /api/assistant/assistants/:id/activate': (req) => {
      state.activates.push(req.params.id);
      if (refuse) return json(500, { error: { message: 'The list could not be saved', code: 'internal_error' } });
      state.active = req.params.id;
      return json(200, { ...ASSISTANTS.find((a) => a.id === req.params.id), active: true });
    },
  };
  if (presets) routes['GET /api/assistant/presets'] = () => json(200, { presets });
  return { routes, state };
}

describe('Talk view: who answers', () => {
  it('with saved assistants, the Assistant select shows the one in use and lists them all; choosing another puts it in use (activate) and the next session.update of the open conversation names it', async () => {
    const fake = assistantRoutes();
    t = await mountView({ routes: fake.routes });
    const pick = t.$('tk-assistant');
    assert.equal(visible(t.$('tk-assistant-field')), true);
    assert.deepEqual(Array.from(pick.querySelectorAll('option')).map((o) => [o.value, o.textContent]), [['a-hermes', 'Hermes (in use)'], ['a-strata', 'Strata']]);
    assert.equal(pick.value, 'a-hermes');
    assertAccessibleNames(t.root);
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    assert.deepEqual(sent(ws)[0].session.assistant, { preset: 'voice' }, 'nothing chosen here: the preset decides');
    const store = await load('core/store.js');
    let told = 0;
    const off = store.on('assistants:changed', () => { told += 1; });
    t.env.change(pick, 'a-strata');
    await settle(t.env);
    await settle(t.env);
    off();
    assert.deepEqual(fake.state.activates, ['a-strata']);
    assert.equal(told, 1, 'the other views hear of it');
    const update = sent(ws).filter((e) => e.type === 'session.update').at(-1);
    assert.deepEqual(update.session.assistant, { preset: 'voice', assistant: 'a-strata' });
    assert.equal(ws.readyState, 1, 'the conversation goes on');
    assert.deepEqual(Array.from(pick.querySelectorAll('option')).map((o) => o.textContent), ['Hermes', 'Strata (in use)']);
    assert.equal(pick.value, 'a-strata');
    // Another preset follows its own assistant again.
    t.env.change(t.$('tk-preset'), 'short');
    await settle(t.env);
    const next = sent(ws).filter((e) => e.type === 'session.update').at(-1);
    assert.deepEqual(next.session.assistant, { preset: 'short', assistant: null }, 'the choice made here is taken back');
  });

  it('with one chosen here, another put in use elsewhere while the conversation runs is followed in session.update', async () => {
    const fake = assistantRoutes();
    t = await mountView({ routes: fake.routes });
    t.env.click(t.$('tk-start'));
    await settle(t.env);
    const ws = lastSocket(t.env);
    await accept(t.env, ws);
    t.env.change(t.$('tk-assistant'), 'a-strata');
    await settle(t.env);
    await settle(t.env);
    fake.state.active = 'a-hermes';
    const store = await load('core/store.js');
    store.emit('assistants:changed');
    await settle(t.env);
    await settle(t.env);
    const update = sent(ws).filter((e) => e.type === 'session.update').at(-1);
    assert.deepEqual(update.session.assistant, { preset: 'voice', assistant: 'a-hermes' });
    assert.equal(t.$('tk-assistant').value, 'a-hermes');
  });

  it('a preset that names a saved assistant shows it as the one that answers; one that names a gone one shows the one in use', async () => {
    const presets = [{ id: 'voice', name: 'Voice', builtin: true, assistant: 'a-strata' }, { id: 'short', name: 'Short answers', builtin: false, assistant: 'a-gone' }];
    t = await mountView({ routes: assistantRoutes({ presets }).routes });
    assert.equal(t.$('tk-assistant').value, 'a-strata');
    t.env.change(t.$('tk-preset'), 'short');
    await settle(t.env);
    assert.equal(t.$('tk-assistant').value, 'a-hermes');
  });

  it('a switch the server refuses says so in a toast and the select shows who answers still; nothing is sent', async () => {
    const fake = assistantRoutes({ refuse: true });
    t = await mountView({ routes: fake.routes });
    t.env.change(t.$('tk-assistant'), 'a-strata');
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('tk-assistant').value, 'a-hermes');
    const toast = t.doc.querySelector('.toast');
    assert.ok(toast, 'a toast');
    assert.match(toast.textContent, /Couldn't switch to “Strata”/);
  });

  it('with no saved assistants (an older server, or none saved) there is no Assistant select and the list is never asked for', async () => {
    t = await mountView();
    assert.equal(visible(t.$('tk-assistant-field')), false);
    assert.equal(t.env.fetch.callsTo('GET /api/assistant/assistants').length, 0);
  });
});

describe('Talk in the shell', () => {
  it('#talk is a route, Talk is in the rail and in the bottom bar, and More stays plain while Talk is current', async () => {
    t = { env: talkEnv(), ctl: { destroy() {} } };
    const router = await importFresh('core/router.js');
    assert.deepEqual(router.parseHash('#talk'), { view: 'talk', sub: '', id: '', path: 'talk', shot: '' });
    assert.equal(router.parseHash('#talk/whatever').path, 'talk');
    assert.ok(router.VIEWS.includes('talk'));
    const host = await load('shell/view-host.js');
    assert.equal(typeof host.VIEW_MODULES.talk, 'function');
    const def = (await host.VIEW_MODULES.talk()).default;
    assert.deepEqual([def.id, def.title, def.partial], ['talk', 'Talk', '/static/partials/talk.html']);
    const doc = t.env.document;
    assert.ok(doc.querySelector('#app-rail a.nav-link[data-view="talk"][href="#talk"]'));
    assert.ok(doc.querySelector('#app-bottombar a.nav-link--tab[data-view="talk"][href="#talk"]'));
    assert.equal(doc.querySelector('#app-more-sheet a.more-link[data-view="talk"]'), null);
    assert.ok(doc.querySelector('main > section#view-talk[data-view="talk"]'));
    const nav = await load('shell/nav.js');
    assert.equal(nav.moreLabel('talk'), 'More');
  });

  it('g then a goes to Talk, and the shortcuts sheet lists it', async () => {
    const env = talkEnv({ routes: { 'GET /static/partials/:name': () => text(200, '<header class="view-head"><h1 id="x-title" class="view-head__title" data-view-heading tabindex="-1">x</h1></header>') } });
    t = { env, ctl: { destroy() {} } };
    const shell = await importFresh('shell/index.js');
    const stub = (id) => async () => ({ default: { id, title: id, partial: `/static/partials/${id}.html`, init: () => ({ show() {}, hide() {} }) } });
    shell.init({ registry: { reader: stub('reader'), talk: stub('talk') } });
    const shortcuts = await load('core/shortcuts.js');
    const go = shortcuts.list().find((s) => s.description === 'Go to Talk');
    assert.ok(go, 'registered');
    assert.deepEqual(go.keys, ['g a']);
    env.document.getElementById('main').focus();
    env.key('g');
    env.key('a');
    await settle(env);
    assert.equal(env.location.hash, '#talk');
    shell.destroy();
  });
});

describe('Talk view: a model that only designs voices', () => {
  it("Speak's voice on a design-only model is not followed: the server default speaks, and a hint points to Design a voice", async () => {
    const DESIGN = { id: 'qwen3/1.7b-voice-design', type: 'tts', provider: 'qwen3', state: 'loaded', capabilities: { voice_design: true, voice_clone: false, speakers: [] } };
    const models = { list: [...MODELS, DESIGN], defaults: { tts: 'kokoro', stt: 'Systran/faster-whisper-base' }, fetchedAt: Date.now(), error: '' };
    t = await mountView({ models, speakVoice: { model: DESIGN.id, voice: 'voice:designed', label: 'Designed', kind: 'named' } });
    assert.doesNotMatch(t.$('tk-voice').textContent, /Same as Speak/);
    assert.equal(t.$('tk-voice-hint').hidden, false);
    assert.match(t.$('tk-voice-hint').textContent, /Voices › Design a voice/);
    t.app.set({ speakVoice: { model: 'kokoro', voice: 'af_bella', label: 'Bella', kind: 'voice' } });
    await settle(t.env);
    assert.match(t.$('tk-voice').textContent, /Same as Speak · Bella/);
    assert.equal(t.$('tk-voice-hint').hidden, true);
  });
});
