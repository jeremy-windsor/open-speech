import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames, liveRegions } from '../harness/a11y.js';
import {
  setup, settle, playingEl, startPlaying, loadMeta, makeDoc, mountPlayer, showReading, load, $, json, text, delay,
  spyWrites, ReaderServer, MODELS,
} from './helpers.js';

let ctx = null;
let speakVoiceModule = null;
afterEach(() => {
  if (speakVoiceModule) speakVoiceModule.__resetForTests();   // stop Speak's boot publisher
  speakVoiceModule = null;
  ctx?.env.restore();
  ctx = null;
});

/** Start Speak's boot publisher, as main.js does once the first view mounted. */
async function startBootVoice() {
  speakVoiceModule = await load('views/speak/speak-voice.js');
  speakVoiceModule.__resetForTests();
  return speakVoiceModule.startBootVoice();
}

const useHref = (button) => button.querySelector('use').getAttribute('href');
const tl = await load('views/reader/timeline.js');
const CHATTERBOX = { id: 'chatterbox', type: 'tts', provider: 'chatterbox', state: 'loaded' };
const SPEAK_RECORDING = { 'open-speech-speak': JSON.stringify({ model: 'chatterbox', voice: 'demo_reference', reference: 'demo_reference' }) };

/** Run animation frames while playing: each flush runs the tracking ticker, the next one paints. */
function frames(env, n, step = null) {
  for (let i = 0; i < n; i += 1) {
    if (step) step(i);
    env.raf.flush();
  }
}

describe('reader-player/player', () => {
  it('mounts its own partial into the slot: one transport, every control named, the chips open their sheets', async () => {
    ctx = await setup();
    const chapters = [];
    await mountPlayer(ctx, { onOpenChapters: (button) => chapters.push(button.id) });
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    assert.ok($('rdp-transport'));
    assert.equal($('rdp-transport').getAttribute('data-chrome'), 'bottom', 'it counts in the chrome metrics');
    assertAccessibleNames($('rd-player-slot'));
    assert.equal($('rdp-chapter').textContent, 'Down the Rabbit-Hole');
    assert.equal($('rdp-speed').getAttribute('aria-label'), 'Playback speed 1 times. Change speed');
    assert.equal($('rdp-speed-label').textContent, '1×');
    assert.equal($('rdp-sleep-label').textContent, 'Sleep');
    assert.equal($('rdp-sleep').getAttribute('aria-label'), 'Sleep timer: off. Set sleep timer');
    ctx.env.click($('rdp-chapters'));
    assert.deepEqual(chapters, ['rdp-chapters']);
    ctx.env.click($('rdp-speed'));
    assert.equal($('rdp-speed-sheet').open, true);
    ctx.env.key('Escape');
    ctx.env.click($('rdp-sleep'));
    assert.equal($('rdp-sleep-sheet').open, true);
    ctx.env.key('Escape');
    assert.equal(document.activeElement, $('rdp-sleep'), 'focus returns to the chip');
    ctx.env.restore();
    // A document with one chapter has no Chapters chip.
    ctx = await setup({ doc: makeDoc({ chapters: [['Only', 3]] }) });
    await mountPlayer(ctx, { onOpenChapters() {} });
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    assert.equal($('rdp-chapters').hidden, true);
    assert.equal($('rdp-chapter').textContent, "Alice's Adventures in Wonderland");
  });

  it('[RE-29] #rdp-status is the one polite status line of the player: waiting, retry, error and the plain notices, never a counter', async () => {
    ctx = await setup({ routes: { 'POST /v1/reader/documents/:id/render': (req) => delay(1500, () => json(200, { parts: ctx.server.partsFrom(req.json.from_block) })) } });
    await mountPlayer(ctx);
    const regions = liveRegions($('rdp-transport'));
    assert.deepEqual(regions.map((r) => [r.el.id, r.role]), [['rdp-status', 'status']]);
    const status = $('rdp-status');
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(ctx.env, 1100);
    assert.equal(status.textContent, 'Preparing audio…');
    assert.equal(status.getAttribute('data-kind'), 'preparing');
    await settle(ctx.env, 600);
    assert.equal(status.textContent, '');
    ctx.server.errors.set(3, 'The model crashed');
    ctx.engine.playFrom(3);
    await settle(ctx.env, 1600);
    assert.equal(status.textContent, "Couldn't play: The model crashed");
    assert.equal(status.classList.contains('is-error'), true);
    ctx.engine.setSleep({ minutes: 5 });
    assert.doesNotMatch(status.textContent, /\d+:\d\d/);
  });

  it('[RE-63] the voice chip names the voice (Kokoro label, named voice name), says "Reading voice: <name>. Change voice", shows the saved voice before the models load and follows Speak and the models', async () => {
    ctx = await setup({ seed: {}, modelsKnown: false, doc: makeDoc({ progress: { block_index: 0, offset_ms: 0, model: 'kokoro', voice: 'af_heart', speed: 1 } }) });
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    assert.equal($('rdp-voice-label').textContent, 'Kokoro · Heart', 'the saved voice before the models load');
    assert.equal($('rdp-voice').getAttribute('aria-label'), 'Reading voice: Kokoro · Heart. Change voice');
    ctx.env.restore();
    // No voice the Reader can use (Speak reads with a raw recording, published as kind 'recording'): "Voice".
    ctx = await setup({ seed: SPEAK_RECORDING, models: [...MODELS, CHATTERBOX] });
    await startBootVoice();
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    assert.equal($('rdp-voice-label').textContent, 'Voice');
    assert.equal($('rdp-voice').getAttribute('aria-label'), 'Reading voice: Voice. Change voice');
    ctx.env.restore();
    // Nothing saved and Speak's view not run yet: the chip names what Play will use, the voice Speak publishes
    // at boot (its default here), and follows it when it arrives.
    ctx = await setup({ seed: {}, modelsKnown: false });
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    assert.equal($('rdp-voice-label').textContent, 'Voice', 'nothing known yet');
    await startBootVoice();
    ctx.app.set({ models: { list: MODELS, defaults: { tts: 'kokoro', stt: '' }, fetchedAt: Date.now(), error: '' } });
    await settle(ctx.env);
    assert.equal($('rdp-voice-label').textContent, 'Kokoro · Heart');
    ctx.app.set({ speakVoice: { model: 'kokoro', voice: 'am_adam', label: 'Adam', kind: 'voice' } });
    await settle(ctx.env);
    assert.equal($('rdp-voice-label').textContent, 'Kokoro · Adam', "Speak's voice, as the Reader would use it");
    ctx.engine.setVoice({ model: 'kokoro', voice: 'voice:n1' });
    await settle(ctx.env, 50);
    assert.equal($('rdp-voice-label').textContent, 'Narrator', 'a named voice shows its name');
    assert.equal($('rdp-voice').getAttribute('aria-label'), 'Reading voice: Narrator. Change voice');
  });

  it('[RE-69] the play button swaps its name and icon between Play and Pause (no aria-pressed) and shows a busy ring while a part is prepared', async () => {
    ctx = await setup();
    ctx.server.pending.set(0, 3);
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await settle(ctx.env);
    const play = $('rdp-play');
    assert.equal(play.getAttribute('aria-label'), 'Play');
    assert.match(useHref(play), /#i-play$/);
    assert.equal(play.hasAttribute('aria-pressed'), false);
    ctx.env.click(play);
    await settle(ctx.env, 1100);
    assert.equal(play.getAttribute('aria-label'), 'Pause');
    assert.match(useHref(play), /#i-pause$/);
    assert.equal(play.getAttribute('aria-busy'), 'true');
    await settle(ctx.env, 3000);
    assert.equal(play.hasAttribute('aria-busy'), false);
    ctx.env.click(play);
    await settle(ctx.env);
    assert.equal(play.getAttribute('aria-label'), 'Play');
    assert.equal(play.hasAttribute('aria-pressed'), false);
  });

  it('[REV-13] the scrubber is a native range over the chapter (VoiceOver can adjust it): the thumb follows the exact position, the keyboard steps by the skip interval without re-announcing every second, spoken times with "about" while estimated, input previews, change seeks, PageUp/PageDown 60 s, Home and End', async () => {
    ctx = await setup({ prefs: { snapToSentence: false } });
    const marks = Array.from({ length: 60 }, (_, i) => ({ block: 2, index: i, start_ms: i * 2000 }));
    ctx.server.override.set(2, { duration_ms: 120000, sentences: marks });
    await mountPlayer(ctx);
    await showReading(ctx);
    const el = await startPlaying(ctx, { block: 2 });
    const scrub = $('rdp-scrub');
    assert.equal(scrub.localName, 'input');
    assert.equal(scrub.getAttribute('type'), 'range');
    assert.equal(scrub.getAttribute('role'), null, 'no custom role=slider');
    assert.equal(scrub.getAttribute('aria-label'), 'Chapter position');
    el.__timeupdate(30.5);
    await settle(ctx.env);
    let t = ctx.engine.state.timeline;
    assert.equal(Number(scrub.max), Math.round(t.durationMs));
    // Nothing on it: no step, so a browser keeps the exact value and the thumb and fill follow playback.
    assert.equal(scrub.hasAttribute('step'), false);
    assert.ok(Math.abs(Number(scrub.value) - t.elapsedMs) < 1000, `the position, not a 15 s step (${scrub.value} vs ${t.elapsedMs})`);
    assert.equal(scrub.style.getPropertyValue('--fill'), tl.percentOf(Number(scrub.value), Number(scrub.max)));
    assert.match(scrub.getAttribute('aria-valuetext'), / of about /);
    assert.match($('rdp-elapsed').textContent, /^~\d+:\d\d$/);
    assert.equal($('rdp-elapsed').getAttribute('aria-hidden'), 'true');
    assert.match(scrub.style.getPropertyValue('--buffer'), /%$/);
    // A pointer drag: still no step (the exact place); input previews without seeking, the release seeks.
    scrub.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    scrub.focus();
    assert.equal(scrub.hasAttribute('step'), false, 'a pointer drags to the exact position');
    const before = el.currentTime;
    ctx.env.input(scrub, String(Math.round(t.elapsedMs + 20000)));
    assert.equal(el.currentTime, before, 'input only previews');
    assert.notEqual($('rdp-elapsed').textContent, '');
    ctx.env.change(scrub);
    assert.equal(Math.round(el.currentTime * 10) / 10, 50.5);
    scrub.blur();
    // The keyboard (or a screen reader moving focus): the step is the skip interval and the value sits on that grid.
    scrub.focus();
    assert.equal(scrub.getAttribute('step'), '15000', 'the skip interval');
    assert.equal(Number(scrub.value) % 15000, 0);
    // While it has focus, playback moves the value (and its spoken text) only when the stepped value moves.
    const seen = [];
    for (let s = 51; s <= 80; s += 1) {
      el.__timeupdate(s);
      await settle(ctx.env);
      seen.push([scrub.value, scrub.getAttribute('aria-valuetext')]);
    }
    const values = new Set(seen.map(([v]) => v));
    const texts = new Set(seen.map(([, v]) => v));
    assert.ok(values.size >= 2 && values.size <= 3, `30 s of playback crosses one or two 15 s steps (${[...values]})`);
    assert.equal(texts.size, values.size, 'the spoken value changes only with the stepped value: no ticking');
    // An arrow: the browser steps the value by the step attribute (the harness plays that part), change seeks.
    const at = Number(scrub.value);
    ctx.env.key('ArrowRight', {}, scrub);
    ctx.env.change(scrub, String(at + Number(scrub.getAttribute('step'))));
    await settle(ctx.env);
    t = ctx.engine.state.timeline;
    assert.ok(Math.abs(t.elapsedMs - (at + 15000)) < 5, 'the arrow moved 15 s');
    el.__timeupdate(40);
    await settle(ctx.env);
    ctx.env.key('PageUp', {}, scrub);
    assert.equal(Math.round(el.currentTime), 100, 'PageUp: 60 s on');
    ctx.env.key('PageDown', {}, scrub);
    assert.equal(Math.round(el.currentTime), 40, 'PageDown: 60 s back');
    ctx.env.key('Home', {}, scrub);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 0, 'Home: the start of the chapter');
    ctx.env.key('End', {}, scrub);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 4, "End: the start of the chapter's last paragraph");
    scrub.blur();
    await settle(ctx.env);
    assert.equal(scrub.hasAttribute('step'), false, 'leaving the slider gives the thumb back its exact position');
    loadMeta(ctx, playingEl());
  });

  it('[REV-13] a chapter shorter than one skip step: the thumb and fill follow playback to the end, and the arrows still reach several places', async () => {
    ctx = await setup({ doc: makeDoc({ chapters: [['A short note', 1]] }), prefs: { snapToSentence: false } });
    await mountPlayer(ctx);
    await showReading(ctx);
    const el = await startPlaying(ctx, { block: 1 });
    const scrub = $('rdp-scrub');
    const max = Number(scrub.max);
    assert.ok(max > 0 && max < 15000, `a chapter shorter than one 15 s step (${max} ms)`);
    const values = [];
    for (const s of [0.5, 1.5, 2.5, 3.5]) {
      el.__timeupdate(s);
      await settle(ctx.env);
      values.push(Number(scrub.value));
      assert.equal(scrub.style.getPropertyValue('--fill'), tl.percentOf(Number(scrub.value), max));
    }
    assert.ok(values[0] > 0, 'never stuck at 0');
    assert.deepEqual(values, [...values].sort((a, b) => a - b));
    assert.equal(new Set(values).size, 4, 'it moves every second');
    assert.ok(Math.abs(values[3] - ctx.engine.state.timeline.elapsedMs) <= tl.scrubQuantumMs(max));
    // The keyboard: a quarter of the chapter in whole seconds (at least 1 s), never 15 s.
    scrub.focus();
    const step = Number(scrub.getAttribute('step'));
    assert.equal(step, tl.keyStepMs(15000, max));
    assert.ok(step >= 1000 && step < max, `the arrows can move (${step} of ${max})`);
    scrub.blur();
    // [RE-38] Heard to the end: "Finished", and the transport shows the end of the chapter, nothing left.
    el.__end();
    await settle(ctx.env);
    assert.equal(ctx.engine.state.finished, true);
    assert.equal($('rdp-status').textContent, 'Finished');
    assert.equal(Number(scrub.value), Number(scrub.max));
    assert.equal(scrub.style.getPropertyValue('--fill'), '100%');
    assert.equal($('rdp-remaining').textContent, '−0:00', 'the minus only: "~" marks the elapsed time');
  });

  it('[REV-13] assistive technology that adjusts the scrubber without focusing it (TalkBack keeps its own cursor) moves one skip step, not the browser\'s 1 ms default step; a pointer and the keyboard are taken at their word', async () => {
    ctx = await setup({ prefs: { snapToSentence: false } });
    const marks = Array.from({ length: 60 }, (_, i) => ({ block: 2, index: i, start_ms: i * 2000 }));
    ctx.server.override.set(2, { duration_ms: 120000, sentences: marks });
    await mountPlayer(ctx);
    await showReading(ctx);
    const el = await startPlaying(ctx, { block: 2 });
    const scrub = $('rdp-scrub');
    el.__timeupdate(30.5);
    await settle(ctx.env);
    assert.notEqual(document.activeElement, scrub);
    assert.equal(scrub.hasAttribute('step'), false, 'unfocused, the default step (1 ms) applies');
    const elapsed = () => ctx.engine.state.timeline.elapsedMs;
    // An increment: the value moves by the default step and the browser fires input and change; no pointer, no key.
    let at = Number(scrub.value);
    ctx.env.change(scrub, String(at + 1));
    await settle(ctx.env);
    assert.ok(Math.abs(elapsed() - (at + 15000)) < 5, `one skip step on (${elapsed()} vs ${at + 15000})`);
    assert.ok(Math.abs(Number(scrub.value) - elapsed()) < tl.scrubQuantumMs(Number(scrub.max)), 'and the thumb is there');
    // A decrement, from where the thumb now is.
    at = Number(scrub.value);
    ctx.env.change(scrub, String(at - 1));
    await settle(ctx.env);
    assert.ok(Math.abs(elapsed() - (at - 15000)) < 5, 'one skip step back');
    // A pointer that moves the thumb half a second is taken at its word.
    at = Number(scrub.value);
    scrub.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    ctx.env.input(scrub, String(at + 500));
    ctx.env.change(scrub);
    await settle(ctx.env);
    assert.ok(Math.abs(elapsed() - (at + 500)) < 5, 'a small drag stays small');
    // So is a key that changes the value inside its keydown (the browser's own step), before its keyup.
    at = Number(scrub.value);
    const nudge = (event) => { if (event.key === 'ArrowUp') ctx.env.change(scrub, String(at + 700)); };
    scrub.addEventListener('keydown', nudge);
    ctx.env.key('ArrowUp', {}, scrub);
    scrub.removeEventListener('keydown', nudge);
    await settle(ctx.env);
    assert.ok(Math.abs(elapsed() - (at + 700)) < 5, 'a key is taken at its word');
    scrub.blur();
  });

  it('[REV-13] a step always moves: inside a long paragraph whose audio is not rendered yet, an arrow, PageUp and an assistive increment go on to the next paragraph instead of back to the start of this one; back and a pointer still land where they did', async () => {
    const doc = makeDoc();
    const long = 'Suddenly she came upon a little three-legged table, all made of solid glass. '.repeat(20);
    doc.blocks[2] = { ...doc.blocks[2], text: long, sentences: [[0, long.length]] };
    ctx = await setup({ doc });
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await showReading(ctx);
    ctx.engine.jumpTo(2);   // paused there, as right after opening: no part is rendered
    await settle(ctx.env);
    assert.equal(ctx.server.renders.length, 0, 'nothing rendered: every position is an estimate');
    const scrub = $('rdp-scrub');
    const t0 = ctx.engine.state.timeline;
    assert.equal(t0.estimated, true);
    const step = 15000;
    const startOf = (block) => tl.timeOfBlock(tl.segments({ blocks: doc.blocks, start: 0, end: 5 }), block);
    assert.ok(startOf(3) - startOf(2) > 4 * step, `paragraph 3 spans several 15 s steps (${startOf(3) - startOf(2)} ms)`);
    // The browser's own arrow step: it moves the value by the step attribute and fires change inside the keydown.
    const arrow = (key) => {
      const at = Number(scrub.value);
      const delta = Number(scrub.getAttribute('step')) * (key === 'ArrowRight' ? 1 : -1);
      const native = (event) => { if (event.key === key) ctx.env.change(scrub, String(at + delta)); };
      scrub.addEventListener('keydown', native);
      ctx.env.key(key, {}, scrub);
      scrub.removeEventListener('keydown', native);
    };
    scrub.focus();
    assert.equal(scrub.getAttribute('step'), String(step));
    const before = { value: Number(scrub.value), text: scrub.getAttribute('aria-valuetext') };
    arrow('ArrowRight');
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 3, 'ArrowRight: on to the next paragraph, not back to the start of this one');
    assert.ok(Number(scrub.value) > before.value, `the thumb moved forward (${scrub.value} > ${before.value})`);
    assert.notEqual(scrub.getAttribute('aria-valuetext'), before.text, 'and a screen reader hears the new place');
    // Back still goes to the start of the paragraph it reaches (it always lands behind).
    arrow('ArrowLeft');
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 2, 'ArrowLeft: back into the long paragraph');
    assert.ok(Math.abs(ctx.engine.state.timeline.elapsedMs - startOf(2)) < 1, 'at its start');
    // PageUp: 60 s on is still inside it, so it moves on to the next paragraph too.
    assert.ok(startOf(2) + 60000 < startOf(3));
    ctx.env.key('PageUp', {}, scrub);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 3, 'PageUp: on to the next paragraph');
    scrub.blur();
    await settle(ctx.env);
    // Assistive technology adjusting the unfocused slider (the browser's 1 ms step, TalkBack, VoiceOver).
    ctx.engine.jumpTo(2);
    await settle(ctx.env);
    assert.notEqual(document.activeElement, scrub);
    ctx.env.change(scrub, String(Number(scrub.value) + 1));
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 3, 'an assistive increment moves on as well');
    // A pointer is not a step: let go in the middle of the long paragraph, it starts that paragraph.
    scrub.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    ctx.env.input(scrub, String(Math.round(startOf(2) + 3 * step)));
    ctx.env.change(scrub);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 2, 'a drag lands on the paragraph it was let go in');
    scrub.blur();
  });

  it('[REV-13] each arrow changes the value a screen reader hears: a short paragraph that starts inside the same 15 s step is passed over', async () => {
    const doc = makeDoc({ chapters: [['Down the Rabbit-Hole', 4]] });
    const long = 'Suddenly she came upon a little three-legged table, all made of solid glass. '.repeat(20);   // ~104 s
    const short = 'Alice opened the door and found a passage.';   // ~3 s
    for (const [block, text] of [[1, long], [2, short], [3, long]]) doc.blocks[block] = { ...doc.blocks[block], text, sentences: [[0, text.length]] };
    ctx = await setup({ doc });
    await mountPlayer(ctx);
    ctx.engine.load(ctx.doc);
    await showReading(ctx);
    ctx.engine.jumpTo(2);
    await settle(ctx.env);
    const scrub = $('rdp-scrub');
    scrub.focus();
    const grid = Number(scrub.getAttribute('step'));
    const segs = tl.segments({ blocks: doc.blocks, start: 0, end: 5 });
    const shown = Number(scrub.value);
    assert.equal(tl.snapToStep(segs[3].startMs, grid, Number(scrub.max)), shown, 'paragraph 3 starts inside the step paragraph 2 shows');
    const at = Number(scrub.value);
    const native = (event) => { if (event.key === 'ArrowRight') ctx.env.change(scrub, String(at + grid)); };
    scrub.addEventListener('keydown', native);
    ctx.env.key('ArrowRight', {}, scrub);
    scrub.removeEventListener('keydown', native);
    await settle(ctx.env);
    assert.equal(ctx.engine.state.cursor, 4, 'on to the paragraph after it');
    assert.ok(Number(scrub.value) > shown, `the value moved (${scrub.value} > ${shown})`);
    scrub.blur();
  });

  it('paints every frame while playing but writes only what changed: a quiet frame writes nothing (no same-value hidden, disabled, attribute or style writes), a moving one only the position', async () => {
    // One chapter: the Chapters chip stays hidden, which used to be re-set on every frame.
    ctx = await setup({ doc: makeDoc({ chapters: [['Only', 6]] }) });
    await mountPlayer(ctx, { onOpenChapters() {} });
    await showReading(ctx);
    const el = await startPlaying(ctx, { block: 1 });
    el.__timeupdate(1);
    await settle(ctx.env);
    assert.equal($('rdp-chapters').hidden, true);
    let paints = 0;
    ctx.engine.on('state', () => { paints += 1; });
    const spy = spyWrites($('rd-player-slot'));
    try {
      frames(ctx.env, 12);
      assert.ok(paints >= 5, `it painted (${paints})`);
      assert.deepEqual(spy.list, [], 'a frame in which nothing changed writes nothing');
      // Playback moving: the value, the fill, the times and the spoken value, and nothing else.
      frames(ctx.env, 30, (i) => { el.__currentTime = 1 + i * 0.05; });
      const kinds = new Set(spy.list);
      for (const write of kinds) {
        assert.ok(['value rdp-scrub', 'style rdp-scrub --fill', 'textContent rdp-elapsed', 'textContent rdp-remaining', 'setAttribute rdp-scrub aria-valuetext'].includes(write), write);
      }
      assert.ok(kinds.has('value rdp-scrub') && kinds.has('style rdp-scrub --fill'), 'the thumb moved');
      // With the sheets open (they repaint every frame too), a quiet frame writes nothing either.
      spy.list.length = 0;
      ctx.env.click($('rdp-speed'));
      await settle(ctx.env);
      spy.list.length = 0;
      frames(ctx.env, 8);
      assert.deepEqual(spy.list, [], 'the speed sheet');
      ctx.env.key('Escape');
      ctx.env.click($('rdp-sleep'));
      await settle(ctx.env);
      spy.list.length = 0;
      frames(ctx.env, 8);
      assert.deepEqual(spy.list, [], 'the sleep sheet');
      ctx.env.key('Escape');
    } finally {
      spy.restore();
    }
  });

  it("a player partial that doesn't load shows \"The player didn't load\" with Retry (no unhandled rejection, no empty slot); Retry mounts the transport and moves focus to Play", async () => {
    const server = new ReaderServer(makeDoc());
    const real = server.routes()['GET /static/partials/:name'];
    let fail = true;
    let requests = 0;
    const routes = {
      'GET /static/partials/:name': (req) => {
        requests += 1;
        return fail ? text(503, 'Service Unavailable', { 'content-type': 'text/plain' }) : real(req);
      },
    };
    const unhandled = [];
    const onRejection = (reason) => { unhandled.push(String(reason?.message || reason)); };
    const warned = [];
    const warn = console.warn;
    console.warn = (...args) => { warned.push(args.map(String).join(' ')); };
    process.on('unhandledRejection', onRejection);
    try {
      ctx = await setup({ server, routes });
      const { mountPlayer: mount } = await load('views/reader/player.js');
      const player = mount($('rd-player-slot'), ctx.engine, {});
      let ready = false;
      player.ready.then(() => { ready = true; });
      await settle(ctx.env, 10);
      await new Promise((resolve) => setImmediate(resolve));
      assert.equal($('rdp-transport'), null);
      const bar = $('rdp-failed');
      assert.ok(bar, 'the slot says so instead of staying empty');
      assert.equal(bar.getAttribute('data-chrome'), 'bottom', 'it counts as bottom chrome, like the transport');
      assert.equal($('rdp-failed-msg').textContent, "The player didn't load. Check your connection, then try again.");
      assertAccessibleNames($('rd-player-slot'));
      assert.equal(ready, false, 'ready waits for the transport');
      await settle(ctx.env, 600);
      assert.match($('app-sr-status').textContent, /The player didn't load/);
      assert.ok(warned.some((w) => w.includes('did not load')));
      // Retry: the transport arrives, focus goes from Retry to Play, ready resolves, the player works.
      fail = false;
      $('rdp-retry').focus();
      ctx.env.click($('rdp-retry'));
      await settle(ctx.env, 10);
      await new Promise((resolve) => setImmediate(resolve));
      assert.ok($('rdp-transport'));
      assert.equal($('rdp-failed'), null);
      assert.equal(document.activeElement.id, 'rdp-play');
      assert.equal(ready, true);
      assert.equal(requests, 2);
      ctx.engine.load(ctx.doc);
      await settle(ctx.env);
      assert.equal($('rdp-chapter').textContent, 'Down the Rabbit-Hole');
      // A Retry that fails again keeps the bar (and focus on its Retry).
      player.destroy();
      ctx.env.restore();
      fail = true;
      ctx = await setup({ server: new ReaderServer(makeDoc()), routes });
      const again = mount($('rd-player-slot'), ctx.engine, {});
      void again;
      await settle(ctx.env, 10);
      await new Promise((resolve) => setImmediate(resolve));
      $('rdp-retry').focus();
      ctx.env.click($('rdp-retry'));
      await settle(ctx.env, 10);
      await new Promise((resolve) => setImmediate(resolve));
      assert.ok($('rdp-failed'));
      assert.equal(document.activeElement.id, 'rdp-retry');
      assert.deepEqual(unhandled, []);
    } finally {
      process.off('unhandledRejection', onRejection);
      console.warn = warn;
    }
  });

  it('screenshot states open the sheets once the routed document is in the engine', async () => {
    ctx = await setup();
    await mountPlayer(ctx);
    const router = await load('core/router.js');
    await router.navigate('reader/doc1?shot=speed');
    ctx.engine.load(ctx.doc);
    await settle(ctx.env, 10);
    assert.equal($('rdp-speed-sheet').open, true);
    ctx.env.key('Escape');
    await router.navigate('reader/doc1?shot=sleep');
    await settle(ctx.env, 10);
    assert.equal($('rdp-sleep-sheet').open, true);
    assert.equal(ctx.engine.state.sleep.mode, 'minutes', 'the sleep state shows a running timer');
    ctx.env.key('Escape');
    await router.navigate('reader/doc1?shot=miniplayer');
    await settle(ctx.env, 10);
    assert.equal(ctx.app.get().route.view, 'speak', 'the mini-player state leaves the Reader');
    // It shows a display-only session: the engine never plays, so nothing is rendered or saved
    // (tests/js/reader-player/shots.test.js covers every state against the whole Reader).
    assert.equal(ctx.app.get().player.id, 'shot:reader:doc1');
    assert.equal(ctx.engine.state.playing, false);
    assert.deepEqual(ctx.server.renders, [], 'no render request');
    assert.deepEqual(ctx.server.saved, [], 'no progress save');
  });
});
