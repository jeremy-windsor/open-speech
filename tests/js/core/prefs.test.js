import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  const prefs = await importFresh('core/prefs.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  return { prefs, app };
}

describe('core/prefs', () => {
  it('load() fills app.prefs and app.reading with the defaults when nothing is stored', async () => {
    const { prefs, app } = await setup();
    prefs.load();
    assert.deepEqual(app.get().prefs, prefs.PREF_DEFAULTS);
    assert.deepEqual(app.get().reading, prefs.READING_DEFAULTS);
    assert.equal(prefs.READING_DEFAULTS.highlight, 'both', 'the spoken sentence is never marked by colour alone');
    assert.equal(prefs.PREF_DEFAULTS.skipBack, 15);
    assert.equal(prefs.PREF_DEFAULTS.skipForward, 30);
  });

  it('validates field by field: a bad field falls back alone', async () => {
    const stored = { v: 1, skipBack: 7, skipForward: 60, shortcuts: 'yes', trackUnit: 'chapter', lastRoute: 'speak/live', extra: 1 };
    const { prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify(stored), 'open-speech-reading': '{"size":9,"paper":"sepia","justify":true}' } });
    const p = prefs.load();
    assert.equal(p.skipBack, 15, 'not one of 5/10/15/30/60');
    assert.equal(p.skipForward, 60);
    assert.equal(p.shortcuts, true);
    assert.equal(p.trackUnit, 'chapter');
    assert.equal(p.lastRoute, 'speak/live');
    assert.ok(!('extra' in p));
    const r = prefs.getReading();
    assert.equal(r.size, null);
    assert.equal(r.paper, 'sepia');
    assert.equal(r.justify, true);
  });

  it('corrupt JSON falls back to the defaults', async () => {
    const { prefs } = await setup({ seed: { 'open-speech-prefs': '{oops' } });
    assert.deepEqual(prefs.load(), prefs.PREF_DEFAULTS);
  });

  it('set() keeps valid fields, ignores invalid ones and persists', async () => {
    const { prefs, app } = await setup();
    prefs.load();
    const next = prefs.set({ skipBack: 30, stepUnit: 'word', shortcuts: false, v: 2 });
    assert.equal(next.skipBack, 30);
    assert.equal(next.stepUnit, 'paragraph');
    assert.equal(next.shortcuts, false);
    assert.equal(next.v, 1);
    assert.equal(app.get().prefs.skipBack, 30);
    assert.equal(JSON.parse(env.localStorage.__data['open-speech-prefs']).skipBack, 30);
  });

  it('setReading() and resetReading() persist the reading appearance', async () => {
    const { prefs } = await setup();
    prefs.setReading({ size: 4, face: 'legible', highlight: 'off', measure: 'huge' });
    const r = prefs.getReading();
    assert.equal(r.size, 4);
    assert.equal(r.face, 'legible');
    assert.equal(r.highlight, 'off');
    assert.equal(r.measure, 'medium');
    assert.equal(JSON.parse(env.localStorage.__data['open-speech-reading']).face, 'legible');
    assert.deepEqual(prefs.resetReading(), prefs.READING_DEFAULTS);
  });

  it('[LR-01] the Live mode (latency_mode) is remembered across page loads inside open-speech-prefs; anything else falls back to natural', async () => {
    let { prefs } = await setup();
    assert.equal(prefs.load().liveMode, 'natural', 'Book (full sentences) by default');
    for (const mode of ['responsive', 'instant_word', 'natural']) assert.equal(prefs.set({ liveMode: mode }).liveMode, mode);
    assert.equal(prefs.set({ liveMode: 'word' }).liveMode, 'natural', 'not a mode the server accepts: ignored');
    prefs.set({ liveMode: 'instant_word' });
    const stored = env.localStorage.__data['open-speech-prefs'];
    assert.equal(JSON.parse(stored).liveMode, 'instant_word');
    assert.deepEqual(Object.keys(env.localStorage.__data).sort(), ['open-speech-prefs'], 'no new storage key');
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': stored } }));
    assert.equal(prefs.load().liveMode, 'instant_word', 'a new page load keeps it');
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, liveMode: 'fast', skipBack: 30 }) } }));
    const p = prefs.load();
    assert.deepEqual([p.liveMode, p.skipBack], ['natural', 30], 'a bad value falls back alone');
  });

  // Talk's AI preset, reply voice and "Interrupt by talking" are remembered per browser,
  // in prefs like the other UI preferences: one `talk` object, versioned and validated field by field like them.
  it('Talk\'s choices are remembered inside open-speech-prefs as prefs.talk, versioned and validated field by field; a patch merges into them', async () => {
    let { prefs } = await setup();
    const DEFAULT = { v: 1, preset: 'voice', voice: null, interrupt: true, textOnly: false };
    assert.deepEqual(prefs.load().talk, DEFAULT, 'the voice preset, Speak\'s voice (null), interrupting on');
    assert.deepEqual(prefs.TALK_DEFAULTS, DEFAULT);
    assert.equal(prefs.PREF_DEFAULTS.talk, prefs.TALK_DEFAULTS);
    assert.equal(prefs.set({ talk: { preset: 'short' } }).talk.preset, 'short');
    let talk = prefs.set({ talk: { interrupt: false } }).talk;
    assert.deepEqual([talk.preset, talk.interrupt], ['short', false], 'a patch keeps the other choices');
    talk = prefs.set({ talk: { voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky', kind: 'voice' } } }).talk;
    assert.deepEqual(talk.voice, { model: 'kokoro', voice: 'af_sky', label: 'Sky' }, 'only the model, the voice and its label');
    assert.ok(Object.isFrozen(talk) && Object.isFrozen(talk.voice));
    for (const bad of [{ preset: '' }, { preset: 7 }, { preset: 'x'.repeat(65) }, { voice: { model: '', voice: 'af_sky' } }, { voice: { model: 'kokoro' } }, { voice: 'af_sky' }, { interrupt: 'no' }, { textOnly: 'yes' }, { v: 2 }]) {
      assert.deepEqual(prefs.set({ talk: bad }).talk, talk, `ignored: ${JSON.stringify(bad)}`);
    }
    assert.deepEqual(prefs.set({ talk: 'short' }).talk, talk, 'not an object: ignored');
    assert.equal(prefs.set({ talk: { voice: { model: 'piper/en_US-amy', voice: 'en_US-amy' } } }).talk.voice.label, 'en_US-amy', 'no label: the voice');
    assert.equal(prefs.set({ talk: { voice: null } }).talk.voice, null, 'back to following Speak');
    prefs.set({ talk: { voice: { model: 'kokoro', voice: 'voice:3f2a', label: 'Narrator' } } });
    const stored = env.localStorage.__data['open-speech-prefs'];
    assert.deepEqual(JSON.parse(stored).talk, { v: 1, preset: 'short', voice: { model: 'kokoro', voice: 'voice:3f2a', label: 'Narrator' }, interrupt: false, textOnly: false });
    assert.deepEqual(Object.keys(env.localStorage.__data), ['open-speech-prefs'], 'no new storage key');

    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': stored } }));
    assert.deepEqual(prefs.load().talk, JSON.parse(stored).talk, 'a new page load keeps them');

    // A bad field falls back alone, inside talk as outside it.
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, skipBack: 30, talk: { v: 3, preset: 7, voice: { model: 'kokoro' }, interrupt: false, extra: 1 } }) } }));
    const p = prefs.load();
    assert.deepEqual(p.talk, { v: 1, preset: 'voice', voice: null, interrupt: false, textOnly: false });
    assert.equal(p.skipBack, 30);
    for (const talkValue of ['short', null, [1, 2]]) {
      env.restore();
      ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, talk: talkValue }) } }));
      assert.deepEqual(prefs.load().talk, DEFAULT, `not an object (${JSON.stringify(talkValue)}): the defaults`);
    }
  });

  // The Ask AI sheet's choices (the preset per input kind, the outputs last sent and the
  // answer voice) are remembered per browser, in prefs like Talk's: one `askAi` object, validated field by field.
  it('Ask AI\'s choices are remembered inside open-speech-prefs as prefs.askAi, validated field by field; a patch merges into them', async () => {
    let { prefs } = await setup();
    // `questions` (the recent questions) is the newest field; its own test is below.
    const DEFAULT = { v: 1, presets: {}, outputs: null, voice: null, questions: [] };
    assert.deepEqual(prefs.load().askAi, DEFAULT, 'no preset per kind yet, the preset\'s own outputs, Speak\'s voice');
    assert.deepEqual(prefs.ASK_AI_DEFAULTS, DEFAULT);
    assert.equal(prefs.PREF_DEFAULTS.askAi, prefs.ASK_AI_DEFAULTS);
    let ask = prefs.set({ askAi: { presets: { text: 'summarize', document: 'book' } } }).askAi;
    assert.deepEqual(ask.presets, { text: 'summarize', document: 'book' });
    ask = prefs.set({ askAi: { outputs: { kind: 'text', preset: 'summarize', show: true, speak: false, save: true } } }).askAi;
    assert.deepEqual(ask.presets, { text: 'summarize', document: 'book' }, 'a patch keeps the other choices');
    assert.deepEqual(ask.outputs, { kind: 'text', preset: 'summarize', show: true, speak: false, save: true });
    ask = prefs.set({ askAi: { voice: { model: 'kokoro', voice: 'af_sky', label: 'Sky', kind: 'voice' } } }).askAi;
    assert.deepEqual(ask.voice, { model: 'kokoro', voice: 'af_sky', label: 'Sky' }, 'only the model, the voice and its label');
    assert.ok(Object.isFrozen(ask) && Object.isFrozen(ask.presets) && Object.isFrozen(ask.outputs) && Object.isFrozen(ask.voice));
    for (const bad of [
      { presets: 'summarize' }, { presets: ['book'] }, { outputs: { kind: 'book', preset: 'x', show: true, speak: false, save: false } },
      { outputs: { kind: 'text', preset: '', show: true, speak: false, save: false } }, { outputs: { kind: 'text', preset: 'x', show: 'yes', speak: false, save: false } },
      { outputs: 'all' }, { voice: { model: '', voice: 'af_sky' } }, { voice: 'af_sky' }, { v: 2 },
    ]) {
      assert.deepEqual(prefs.set({ askAi: bad }).askAi, ask, `ignored: ${JSON.stringify(bad)}`);
    }
    assert.deepEqual(prefs.set({ askAi: { presets: { text: 'voice', transcript: 7, poem: 'x', document: 'y'.repeat(65) } } }).askAi.presets, { text: 'voice' },
      'a preset map keeps only its valid kinds and ids (it is sent whole: the sheet merges it)');
    assert.equal(prefs.set({ askAi: { outputs: null, voice: null } }).askAi.voice, null, 'back to the preset\'s outputs and Speak\'s voice');
    prefs.set({ askAi: { presets: { transcript: 'voice', document: 'book' }, outputs: { kind: 'document', preset: 'book', show: false, speak: false, save: true }, voice: { model: 'kokoro', voice: 'voice:3f2a', label: 'Narrator' } } });
    const stored = env.localStorage.__data['open-speech-prefs'];
    assert.deepEqual(JSON.parse(stored).askAi, {
      v: 1, presets: { transcript: 'voice', document: 'book' }, outputs: { kind: 'document', preset: 'book', show: false, speak: false, save: true },
      voice: { model: 'kokoro', voice: 'voice:3f2a', label: 'Narrator' }, questions: [],
    });
    assert.deepEqual(Object.keys(env.localStorage.__data), ['open-speech-prefs'], 'no new storage key');

    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': stored } }));
    assert.deepEqual(prefs.load().askAi, JSON.parse(stored).askAi, 'a new page load keeps them');
    assert.deepEqual(prefs.get().talk, prefs.TALK_DEFAULTS, "Talk's choices are their own");

    // A bad field falls back alone, inside askAi as outside it.
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, skipBack: 30, askAi: { v: 3, presets: { text: 'voice', bad: 1 }, outputs: { kind: 'text' }, voice: { model: 'kokoro' }, extra: 1 } }) } }));
    const p = prefs.load();
    assert.deepEqual(p.askAi, { v: 1, presets: { text: 'voice' }, outputs: null, voice: null, questions: [] });
    assert.equal(p.skipBack, 30);
    for (const value of ['summarize', null, [1, 2]]) {
      env.restore();
      ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, askAi: value }) } }));
      assert.deepEqual(prefs.load().askAi, DEFAULT, `not an object (${JSON.stringify(value)}): the defaults`);
    }
  });

  // The Ask AI sheet's question box offers the last few questions again, kept in this
  // browser like the sheet's other choices: prefs.askAi.questions, newest first.
  it('the recent questions are kept in prefs.askAi.questions: newest first, at most three, each 1 to 2,000 characters, no repeats', async () => {
    let { prefs } = await setup();
    assert.deepEqual(prefs.load().askAi.questions, [], 'none yet');
    assert.deepEqual(prefs.ASK_AI_DEFAULTS.questions, []);
    let ask = prefs.set({ askAi: { questions: ['Explain this', 'What does “curious” mean here?'] } }).askAi;
    assert.deepEqual(ask.questions, ['Explain this', 'What does “curious” mean here?']);
    assert.ok(Object.isFrozen(ask.questions));
    ask = prefs.set({ askAi: { presets: { document: 'book' } } }).askAi;
    assert.deepEqual(ask.questions, ['Explain this', 'What does “curious” mean here?'], 'a patch of another field keeps them');
    const long = 'x'.repeat(2000);
    ask = prefs.set({ askAi: { questions: ['Four', 'Three', 'Four', '  ', '', 7, null, 'y'.repeat(2001), long, 'Two', 'One'] } }).askAi;
    assert.deepEqual(ask.questions, ['Four', 'Three', long], 'only text of 1 to 2,000 characters, the first of a repeat, the newest three');
    for (const bad of ['Explain this', { 0: 'Explain this' }, null]) {
      assert.deepEqual(prefs.set({ askAi: { questions: bad } }).askAi.questions, ask.questions, `ignored: ${JSON.stringify(bad)}`);
    }
    assert.deepEqual(prefs.set({ askAi: { questions: [] } }).askAi.questions, [], 'an empty list forgets them');
    prefs.set({ askAi: { questions: ['Summarize', 'Define the hard words'] } });
    const stored = env.localStorage.__data['open-speech-prefs'];
    assert.deepEqual(JSON.parse(stored).askAi.questions, ['Summarize', 'Define the hard words']);
    assert.deepEqual(Object.keys(env.localStorage.__data), ['open-speech-prefs'], 'no new storage key');
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': stored } }));
    assert.deepEqual(prefs.load().askAi.questions, ['Summarize', 'Define the hard words'], 'a new page load keeps them');
    env.restore();
    ({ prefs } = await setup({ seed: { 'open-speech-prefs': JSON.stringify({ v: 1, askAi: { v: 1, presets: { text: 'voice' }, questions: 'Explain' } }) } }));
    assert.deepEqual([prefs.load().askAi.questions, prefs.get().askAi.presets.text], [[], 'voice'], 'a bad list falls back alone');
  });

  it('works when storage is unavailable', async () => {
    const { prefs } = await setup({ storage: 'throwing' });
    assert.deepEqual(prefs.load(), prefs.PREF_DEFAULTS);
    assert.equal(prefs.set({ wakeLock: true }).wakeLock, true, 'kept in memory for this page');
  });
});
