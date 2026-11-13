// Speak's screenshot states are display only: a link to one must never write to the server.
// The page's requests carry the stored key, so the server's cross-site guard can't tell a link from the listener.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountSpeak, ready, settle, modelsPayload, load } from './helpers.js';
import { shotSamples } from '../../../src/static/js/views/speak/index.js';

let t;
afterEach(() => t?.env.restore());

const KEY = 'sk-test-shot';
const SPEAK_SHOTS = ['take', 'limit', 'blend', 'voice-picker'];
const LIVE_SHOTS = ['live-speaking', 'live-paused', 'live-recording', 'live-connecting', 'live-error', 'live-unavailable', 'live-menu', 'live-prompt', 'live-idle'];

function routeFor(shot) {
  return shot.startsWith('live-')
    ? { sub: 'live', path: 'speak/live', shot }
    : { shot };
}

/**
 * Mount Speak on a screenshot route with a stored key, and let everything the state starts run out. A Live state
 * needs Speak › Live shown (Settings › Developer); off, the link shows Generate.
 */
async function mountShot(shot, { kokoroState = 'loaded', seed = {} } = {}) {
  const live = shot.startsWith('live-') ? { 'open-speech-prefs': JSON.stringify({ v: 1, showLive: true }) } : {};
  t = await mountSpeak({
    route: routeFor(shot),
    models: modelsPayload({ kokoroState }),
    seed: { 'open-speech-api-key': KEY, ...live, ...seed },
  });
  await ready(t);
  for (let i = 0; i < 8; i += 1) await settle(t.env, 250);
  return t;
}

/** Every request that is not a read. */
function writes() {
  return t.env.fetch.calls.filter((call) => !['GET', 'HEAD'].includes(call.method)).map((call) => `${call.method} ${call.path}`);
}

describe('Speak screenshot states never write', () => {
  for (const kokoroState of ['loaded', 'downloaded']) {
    for (const shot of [...SPEAK_SHOTS, ...LIVE_SHOTS]) {
      it(`#speak${shot.startsWith('live-') ? '/live' : ''}?shot=${shot} with Kokoro ${kokoroState} sends no POST, PUT, PATCH or DELETE, opens no socket and plays nothing`, async () => {
        await mountShot(shot, { kokoroState });
        if (shot.startsWith('live-')) assert.equal(t.ctx.mode, 'live', 'the Live state shows (Speak › Live is on)');
        assert.deepEqual(writes(), [], 'no write: no speech request, no History row, no model load');
        assert.equal(t.env.sockets.length, 0, 'no Live socket');
        const audio = await load('core/audio.js');
        assert.equal(audio.current(), null, 'no audio session claimed');
        assert.ok(t.env.fetch.calls.length > 0, 'the view did load its data (the check is not vacuous)');
      });
    }
  }

  it('the take state shows a finished take made locally: a WAV of the sample text, timed, playable and downloadable, with no History event', async () => {
    const { on } = await load('core/store.js');
    let historyEvents = 0;
    const off = on('history:changed', () => { historyEvents += 1; });
    try {
      await mountShot('take', { kokoroState: 'downloaded' });
      const takes = t.parts.takes.all();
      assert.equal(takes.length, 1, 'one take');
      const take = takes[0];
      assert.equal(take.state, 'done');
      assert.equal(take.blob.type, 'audio/wav', 'the audio was made here');
      assert.ok(take.url.startsWith('blob:'));
      assert.ok(take.durationS > 6 && take.durationS < 7, String(take.durationS));
      assert.ok(take.peaks.some((v) => v > 0), 'a waveform that looks like speech, not a flat line');
      const card = t.$('sp-takes').querySelector('.sp-take');
      assert.match(card.querySelector('.sp-take__title').textContent, /^“Alice was beginning/);
      assert.equal(card.querySelector('.sp-take__perf').textContent.includes('First audio 140 ms · RTF 0.04'), true);
      assert.equal(card.querySelector('[data-take-act="play"]').disabled, false);
      assert.equal(card.querySelector('[data-take-act="download"]').disabled, false);
      assert.equal(t.$('sp-gen-status-text').textContent, 'Done');
      assert.equal(t.$('sp-gen-status').getAttribute('data-phase'), 'done');
      assert.equal(t.$('sp-generate').querySelector('.btn__label').textContent, 'Generate', 'nothing is generating');
      assert.equal(historyEvents, 0, 'History is not told about a take the server never made');
      // Showing the view again (Back onto the address) adds no second take.
      t.ctl.hide();
      t.ctl.show({ view: 'speak', sub: '', id: '', path: 'speak', shot: 'take' }, { focus: 'keep' });
      await settle(t.env, 250);
      assert.equal(t.parts.takes.all().length, 1);
      assert.deepEqual(writes(), []);
    } finally {
      off();
    }
  });

  it('the take and limit states show their sample text without replacing the saved draft', async () => {
    const { KEYS, read } = await load('core/storage.js');
    for (const shot of ['take', 'limit']) {
      await mountShot(shot, { seed: { [KEYS.speakDraft]: 'My own unsent draft.' } });
      assert.notEqual(t.$('sp-input').value, 'My own unsent draft.', `${shot}: the sample text shows`);
      await settle(t.env, 1000);
      t.ctl.hide();
      assert.equal(read(KEYS.speakDraft, ''), 'My own unsent draft.', `${shot}: the draft is kept`);
      t.env.restore();
      t = null;
    }
  });

  it('shotSamples is deterministic, bounded and in range', () => {
    const a = shotSamples(1, 8000);
    const b = shotSamples(1, 8000);
    assert.equal(a.length, 8000);
    assert.deepEqual(Array.from(a.slice(0, 64)), Array.from(b.slice(0, 64)));
    assert.ok(a.every((v) => v >= -1 && v <= 1));
  });
});
