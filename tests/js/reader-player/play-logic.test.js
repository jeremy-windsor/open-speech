import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv } from '../harness/env.js';
import { FakeMediaElement } from '../harness/fakes.js';
import { load, setup, settle, startPlaying, playingEl, loadMeta, makeDoc } from './helpers.js';

const logic = await load('views/reader/play-logic.js');

let env = null;
afterEach(() => { env?.restore(); env = null; });

function media() {
  env = installBrowserEnv();
  return new FakeMediaElement(document, 'audio');
}

const parts = [
  { id: 'a', first_block: 3, last_block: 5, sentences: [{ block: 3, index: 0, start_ms: 0 }, { block: 4, index: 0, start_ms: 2100 }, { block: 5, index: 0, start_ms: 4800 }] },
  { id: 'b', first_block: 6, last_block: 6, sentences: [] },
  { id: 'c', first_block: 7, last_block: 9, sentences: [{ block: 7, index: 0, start_ms: 0 }] },
];

describe('reader-player/play-logic', () => {
  it('[RE-09] locate() finds a part that starts at the block, or a sentence mark inside one, and never guesses an offset', () => {
    assert.deepEqual(logic.locate(6, parts), { part: parts[1], offsetMs: 0 });
    assert.deepEqual(logic.locate(4, parts), { part: parts[0], offsetMs: 2100 });
    assert.deepEqual(logic.locate(5, parts), { part: parts[0], offsetMs: 4800 });
    assert.equal(logic.locate(8, parts), null, 'block 8 is inside part c but has no mark: no guessed offset');
    assert.equal(logic.locate(12, parts), null);
    assert.equal(logic.locate(1, null), null);
  });

  it('[RE-10] network errors, 5xx, 408 and 429 are retried with 500 → 1000 → 2000 → 4000 ms backoff; other 4xx and a refused key are final', () => {
    assert.equal(logic.isRetryable({ kind: 'network', status: 0 }), true);
    assert.equal(logic.isRetryable({ kind: 'timeout', status: 0 }), true);
    for (const status of [500, 502, 503, 408, 429]) assert.equal(logic.isRetryable({ kind: 'http', status }), true, status);
    for (const status of [400, 404, 409, 413, 422]) assert.equal(logic.isRetryable({ kind: 'http', status }), false, status);
    assert.equal(logic.isRetryable({ kind: 'auth', status: 401 }), false, 'a refused key is never retried (each wrong key counts)');
    assert.equal(logic.isRetryable({ kind: 'locked', status: 429 }), false, 'a lockout is final');
    assert.deepEqual([0, 1, 2, 3, 4, 5, 9].map(logic.backoffMs), [500, 500, 1000, 2000, 4000, 4000, 4000]);
    assert.equal(logic.MAX_FAILURES, 10);
  });

  it('[RE-14] a seek before metadata is parked on the element, applied once on loadedmetadata, the latest wins, and a new src drops it', async () => {
    const el = media();
    logic.parkSeek(el, 3);
    logic.parkSeek(el, 7.5);
    assert.equal(logic.positionOf(el), 7.5, 'the latest parked seek wins');
    assert.equal(logic.takeParkedSeek(el), 7.5);
    assert.equal(logic.takeParkedSeek(el), null, 'applied once');
    logic.parkSeek(el, 2);
    logic.clearParkedSeek(el);
    assert.equal(logic.positionOf(el), 0);
    env.restore();
    env = null;
    // In the engine: a resume parks its offset, metadata applies it, and the next part's src never inherits it.
    const ctx = await setup({ doc: makeDoc({ progress: { block_index: 2, offset_ms: 1500, model: 'kokoro', voice: 'af_heart', speed: 1 } }) });
    env = ctx.env;
    ctx.engine.load(ctx.doc);
    ctx.engine.play();
    await settle(env);
    const el2 = playingEl();
    assert.equal(el2.dataset.seek, '1.5', 'parked until metadata');
    loadMeta(ctx, el2);
    await settle(env);
    assert.equal(el2.currentTime, 1.5);
    assert.equal(el2.dataset.seek, undefined);
    ctx.engine.seekBy(0.5);   // a normal seek on a loaded element is not parked
    assert.equal(el2.dataset.seek, undefined);
  });

  it('[RE-45] previous track restarts the paragraph after 3 s of listening (rate-adjusted), else goes back one, never before block 0', async () => {
    assert.equal(logic.previousTarget(5, 3001), 5);
    assert.equal(logic.previousTarget(5, 3000), 4);
    assert.equal(logic.previousTarget(0, 0), 0);
    const ctx = await setup();
    env = ctx.env;
    const el = await startPlaying(ctx, { block: 2 });
    ctx.engine.setSpeed(2);
    el.__timeupdate(3.5);   // 3.5 s of audio at 2× is 1.75 s of listening
    ctx.engine.prevTrack();
    await settle(env);
    assert.equal(ctx.engine.state.cursor, 1, 'under 3 s of listening: back one paragraph');
    const el2 = playingEl() || el;
    loadMeta(ctx, el2);
    ctx.engine.setSpeed(1);
    el2.__timeupdate(3.5);
    ctx.engine.prevTrack();
    await settle(env);
    assert.equal(ctx.engine.state.cursor, 1, 'past 3 s: the same paragraph restarts');
  });

  it('speeds clamp to 0.5–3.5 in 0.05 steps; a stored default outside the range reads as 1; a document keeps its own', () => {
    assert.equal(logic.clampSpeed(1.26), 1.25);
    assert.equal(logic.clampSpeed(9), 3.5);
    assert.equal(logic.clampSpeed(0.1), 0.5);
    assert.equal(logic.clampSpeed('x'), null);
    assert.equal(logic.storedSpeed('1.5'), 1.5);
    assert.equal(logic.storedSpeed('7'), 1);
    assert.equal(logic.storedSpeed(null), 1);
    assert.equal(logic.documentSpeed({ speed: 1.25 }, 1), 1.25);
    assert.equal(logic.documentSpeed({ speed: null }, 1.5), 1.5);
    assert.equal(logic.documentSpeed(null, 2), 2);
    assert.deepEqual([...logic.SPEED_PRESETS], [0.75, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3]);
  });

  it('chapters: front matter counts as the first chapter, ranges end where the next chapter starts', () => {
    const doc = { blocks: new Array(10).fill({ text: 'x' }), chapters: [{ first_block: 2 }, { first_block: 5 }, { first_block: 8 }] };
    assert.deepEqual(logic.chapterMap(10, doc.chapters), [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]);
    assert.deepEqual(logic.chapterRange(doc, 0), [0, 5]);
    assert.deepEqual(logic.chapterRange(doc, 2), [8, 10]);
    assert.equal(logic.chapterStart(doc, 1), 5);
    assert.equal(logic.chapterStart(doc, 3), 10, 'past the last chapter: the block count');
    assert.deepEqual(logic.chapterRange({ blocks: [1, 2, 3], chapters: [] }, 0), [0, 3]);
  });

  it('tracking positions: sentence marks, or a character-share estimate for a multi-block part without marks', () => {
    assert.deepEqual(logic.trackPosition(parts[0], 2500), { block: 4, sentence: 0 });
    assert.deepEqual(logic.trackPosition(parts[0], 0), { block: 3, sentence: 0 });
    const blocks = [];
    blocks[7] = { text: 'aaaa' };
    blocks[8] = { text: 'bbbbbbbbbbbb' };
    blocks[9] = { text: 'cccc' };
    const noMarks = { first_block: 7, last_block: 9, sentences: [] };
    assert.deepEqual(logic.trackPosition(noMarks, 1000, { blocks, durationMs: 4000 }), { block: 8, sentence: -1 });
    assert.deepEqual(logic.trackPosition(noMarks, 3900, { blocks, durationMs: 4000 }), { block: 9, sentence: -1 });
    assert.equal(logic.sentenceStartMs(parts[0], 3000), 2100);
    assert.equal(logic.sentenceStartMs(parts[1], 3000), 3000, 'no marks: no snap');
    assert.deepEqual(logic.sentenceStep(parts[0], 4, 0, 1), parts[0].sentences[2]);
    assert.deepEqual(logic.sentenceStep(parts[0], 4, 0, -1), parts[0].sentences[0]);
    assert.equal(logic.sentenceStep(parts[0], 5, 0, 1), null, 'past the part: the caller falls back to the paragraph');
  });

  it('voice labels: Kokoro names, blends joined, named voices by name (or "Named voice" until the names load)', () => {
    assert.equal(logic.voiceLabel({ voice: 'af_heart' }), 'Heart');
    assert.equal(logic.voiceLabel({ voice: 'af_bella(2)+af_sky(1)' }), 'Bella + Sky');
    assert.equal(logic.voiceLabel({ model: 'kokoro', voice: 'af_heart' }), 'Kokoro · Heart', 'with its model, the voice carries its name');
    assert.equal(logic.voiceLabel({ model: 'supertonic', voice: 'M1(0.6)+F2(0.4)' }), 'Supertonic · M1 + F2');
    assert.equal(logic.voiceLabel({ model: 'piper/en_US-amy-medium', voice: 'piper/en_US-amy-medium' }), 'Piper · Amy (US)');
    assert.equal(logic.voiceLabel({ voice: 'voice:n1' }, new Map([['n1', 'Narrator']])), 'Narrator');
    assert.equal(logic.voiceLabel({ voice: 'voice:n1' }), 'Named voice');
    assert.equal(logic.voiceLabel(null), '');
    assert.equal(logic.voiceDetail({ voice: 'af_bella(2)+af_sky(1)' }), 'af_bella(2)+af_sky(1)');
    assert.equal(logic.voiceDetail({ voice: 'af_heart' }), 'American English · female');
  });

  it("[REV-09] Speak's voice is a raw recording only when Speak says so (kind 'recording')", () => {
    assert.equal(logic.isRecordingChoice({ model: 'chatterbox', voice: 'demo', kind: 'recording' }), true);
    for (const kind of ['voice', 'blend', 'named', '']) {
      assert.equal(logic.isRecordingChoice({ model: 'chatterbox', voice: 'demo', kind }), false, kind || 'no kind');
    }
    assert.equal(logic.isRecordingChoice({ model: 'chatterbox', voice: '', kind: 'recording' }), false, 'no voice: nothing to leave out');
    assert.equal(logic.isRecordingChoice(null), false);
  });

  it('the silent clip is a 0.1 s data: WAV (allowed by media-src data:)', () => {
    const url = logic.silentWavUrl();
    assert.match(url, /^data:audio\/wav;base64,/);
    const bytes = Buffer.from(url.split(',')[1], 'base64');
    assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
    assert.equal(bytes.length, 44 + 800);
    assert.equal(logic.silentWavUrl(), url, 'cached');
  });
});
