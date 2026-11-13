import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { load } from './helpers.js';

const tl = await load('views/reader/timeline.js');

// 6 blocks of 150 characters: 10 s each at the default 15 characters a second.
const blocks = Array.from({ length: 6 }, () => ({ text: 'x'.repeat(150) }));
const ready = (first, last, ms, extra = {}) => ({ id: `p${first}`, first_block: first, last_block: last, status: 'ready', duration_ms: ms, url: '/m', sentences: [], ...extra });

describe('reader-player/timeline', () => {
  it('calibrates ms per character from ready parts, else 15 characters a second', () => {
    assert.equal(tl.msPerChar(new Map(), blocks), 1000 / 15);
    const known = tl.remember(new Map(), [ready(0, 0, 12000), ready(1, 1, 6000), { id: 'x', first_block: 2, last_block: 2, status: 'rendering' }]);
    assert.equal(known.size, 2, 'only ready parts with a duration count');
    assert.equal(tl.msPerChar(known, blocks), 18000 / 300);
  });

  it('a newer part replaces a known part it overlaps (render windows from different starts)', () => {
    const known = tl.remember(new Map(), [ready(0, 1, 20000)]);
    tl.remember(known, [ready(1, 1, 9000, { id: 'other' })]);
    assert.deepEqual([...known.keys()], [1]);
  });

  it('chapter duration is the estimate replaced by real durations where parts are ready; estimated until all are', () => {
    const known = tl.remember(new Map(), [ready(0, 0, 8000), ready(1, 2, 16000)]);
    const t = tl.chapterTimeline({ blocks, start: 0, end: 4, known, perChar: 1000 / 15, active: known.get(1), activeMs: 4000 });
    assert.equal(t.durationMs, 8000 + 16000 + 10000);
    assert.equal(t.elapsedMs, 8000 + 4000);
    assert.equal(t.bufferedMs, 24000, 'buffered runs to the end of the last ready part in a row');
    assert.equal(t.estimated, true);
    const all = tl.remember(new Map(known), [ready(3, 3, 9000)]);
    assert.equal(tl.chapterTimeline({ blocks, start: 0, end: 4, known: all, cursor: 3 }).estimated, false);
  });

  it('without an active part, elapsed is the cursor block plus a saved offset', () => {
    const t = tl.chapterTimeline({ blocks, start: 2, end: 6, cursor: 4, offsetMs: 2500 });
    assert.equal(t.elapsedMs, 20000 + 2500);
    assert.equal(t.durationMs, 40000);
  });

  it('a position maps to a part and an offset inside it, or to an unrendered block start', () => {
    const marks = [{ block: 1, index: 0, start_ms: 0 }, { block: 2, index: 0, start_ms: 7000 }];
    const known = tl.remember(new Map(), [ready(1, 2, 16000, { sentences: marks })]);
    const segs = tl.segments({ blocks, start: 0, end: 4, known });
    assert.deepEqual(tl.seekTarget(segs, 5000), { block: 0, part: null, partOffsetMs: 0, blockOffsetMs: 0 });
    const inside = tl.seekTarget(segs, 10000 + 9000);
    assert.equal(inside.block, 2);
    assert.equal(inside.partOffsetMs, 9000);
    assert.equal(inside.blockOffsetMs, 2000);
    assert.equal(tl.seekTarget(segs, 99e9).block, 3, 'past the end: the last block');
    assert.equal(tl.timeOfBlock(segs, 2), 10000 + 7000);
  });

  it('shows times at the playback speed, with "~" on the elapsed time and "about" in the spoken text while estimated (the remaining time has its minus only: "−~8:12" read as "--8:12" at small sizes)', () => {
    const shown = tl.display({ elapsedMs: 243000, durationMs: 735000, estimated: true }, 1);
    assert.deepEqual(shown, { elapsed: '~4:03', remaining: '−8:12', valuetext: '4 minutes 3 seconds of about 12 minutes 15 seconds' });
    const fast = tl.display({ elapsedMs: 60000, durationMs: 120000, estimated: false }, 2);
    assert.deepEqual(fast, { elapsed: '0:30', remaining: '−0:30', valuetext: '30 seconds of 1 minute' });
    assert.equal(tl.spokenAt(30000, { durationMs: 60000 }, 1), '30 seconds of 1 minute');
  });

  it('[REV-13] the scrubber value: no step and the exact position (to ~1/600 of the chapter) unless the keyboard is on it; then the skip interval, capped for short chapters, on the browser\'s step grid', () => {
    // A 9.2 s note: the thumb follows the position, it is never stuck at 0 by a 15 s step.
    const note = { elapsedMs: 4321, durationMs: 9152 };
    assert.deepEqual(tl.scrubPosition(note), { value: 4320, step: null }, 'a 16 ms quantum: one frame');
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 9152, durationMs: 9152 }), { value: 9152, step: null }, 'the end is reachable');
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 9246.7, durationMs: 9246.7 }), { value: 9247, step: null }, 'a fractional end too');
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 0, durationMs: 0 }), { value: 0, step: null });
    // A 20-minute chapter moves in whole seconds at most (1/600 of it would be 2 s).
    assert.equal(tl.scrubQuantumMs(1200000), 1000);
    assert.equal(tl.scrubQuantumMs(60000), 100);
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 754321, durationMs: 1200000 }), { value: 754000, step: null });
    // The keyboard step: the skip interval, at most a quarter of a short chapter in whole seconds, at least 1 s.
    assert.equal(tl.keyStepMs(15000, 1200000), 15000);
    assert.equal(tl.keyStepMs(15000, 9152), 2000);
    assert.equal(tl.keyStepMs(15000, 40000), 10000);
    assert.equal(tl.keyStepMs(15000, 1500), 1000);
    assert.equal(tl.keyStepMs(10000, 0), 10000);
    assert.deepEqual(tl.scrubPosition(note, { keyboard: true, skipMs: 15000 }), { value: 4000, step: 2000 });
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 9152, durationMs: 9152 }, { keyboard: true }), { value: 8000, step: 2000 }, 'never past max');
    assert.deepEqual(tl.scrubPosition({ elapsedMs: 754321, durationMs: 1200000 }, { keyboard: true, skipMs: 15000 }), { value: 750000, step: 15000 });
    assert.equal(tl.snapToStep(7600, 15000, 1200000), 15000, 'the nearest step, as a browser rounds a range value');
    assert.equal(tl.snapToStep(-5, 1000, 9000), 0);
    assert.equal(tl.percentOf(4320, 9152), '47.2%');
    assert.equal(tl.percentOf(5, 0), '0%');
    assert.equal(tl.percentOf(20000, 10000), '100%');
  });

  it('[REV-13] a scrubber change that no pointer or key made (assistive technology adjusting it without focus) moves at least one key step that way, within the chapter', () => {
    const long = { skipMs: 15000, durationMs: 1200000 };
    assert.equal(tl.assistedTarget(600001, 600000, long), 615000, "the browser's 1 ms default step becomes the skip interval");
    assert.equal(tl.assistedTarget(599999, 600000, long), 585000);
    assert.equal(tl.assistedTarget(660000, 600000, long), 660000, '5% of the chapter is already more than a step');
    assert.equal(tl.assistedTarget(600000, 600000, long), 600000, 'no move, no seek elsewhere');
    assert.equal(tl.assistedTarget(1199999, 1199998, long), 1200000, 'never past the end');
    assert.equal(tl.assistedTarget(0, 1, long), 0, 'never before the start');
    // A short note: one key step is a quarter of it (2 s of 9.2 s), so the adjustments still reach several places.
    assert.equal(tl.assistedTarget(4321, 4320, { skipMs: 15000, durationMs: 9152 }), 6320);
    assert.equal(tl.assistedTarget(4778, 4320, { skipMs: 15000, durationMs: 9152 }), 6320, 'a 5% move too');
  });

  it('[REV-13] a scrubber step always moves: forward inside an unrendered block it goes to the start of the next block, never back to the start of the block it is in; back, a ready part, a drag and the last block are left alone', () => {
    // Blocks 2..5 of the fixture; block 3 is unrendered, the others too unless a part is known.
    const segs = tl.segments({ blocks, start: 2, end: 6 });
    const [b2, b3, b4] = segs;
    const inside3 = b3.startMs + b3.ms / 2;
    assert.equal(tl.stepSeekMs(segs, b3.startMs, inside3, 1), b4.startMs, 'forward inside the block it starts in: the next block');
    assert.equal(tl.seekTarget(segs, tl.stepSeekMs(segs, b3.startMs, inside3, 1)).block, b4.from);
    assert.equal(tl.stepSeekMs(segs, b3.startMs + 10, inside3, 1), b4.startMs, 'also from a saved offset inside it');
    assert.equal(tl.stepSeekMs(segs, b2.startMs, inside3, 1), inside3, 'a step that already reaches a later block stands');
    assert.equal(tl.stepSeekMs(segs, inside3, b2.startMs + 1, -1), b2.startMs + 1, 'back always lands behind: it stands');
    assert.equal(tl.stepSeekMs(segs, b3.startMs, inside3, 0), inside3, 'a drag is not a step');
    const last = segs[segs.length - 1];
    assert.equal(tl.stepSeekMs(segs, last.startMs, last.startMs + 1, 1), last.startMs + 1, 'the last block has no next start (End lands there too)');
    // Inside a ready part the step is exact.
    const known = tl.remember(new Map(), [{ id: 'p3', first_block: 3, last_block: 3, status: 'ready', duration_ms: 40000, sentences: [] }]);
    const ready = tl.segments({ blocks, start: 2, end: 6, known });
    assert.equal(tl.stepSeekMs(ready, ready[1].startMs, ready[1].startMs + 15000, 1), ready[1].startMs + 15000);
    assert.equal(tl.stepSeekMs([], 0, 500, 1), 500);
    // The keyboard's grid (15 s): from a paragraph at 9:26 the thumb shows 9:30; the next paragraph starts at 9:35,
    // which the grid also shows as 9:30. With the floor (half a grid step past the shown value) the step goes on to
    // the next start the grid shows as a move, so a screen reader hears a new value.
    const est = (from, startMs, ms) => ({ from, to: from, startMs, ms, part: null });
    const grid = [est(0, 0, 566000), est(1, 566000, 9000), est(2, 575000, 40000), est(3, 615000, 30000)];
    assert.equal(tl.stepSeekMs(grid, 566000, 585000, 1), 585000, 'without a floor the paragraph at 9:35 stands');
    assert.equal(tl.stepSeekMs(grid, 566000, 585000, 1, { atLeastMs: 570000 + 7500 }), 615000, 'with it, the next one the grid can show');
    assert.equal(tl.stepSeekMs(grid, 615000, 645000, 1, { atLeastMs: 622500 }), 645000, 'nowhere further: it stands');
  });

  it('[RE-38] a finished document sits at the end of its last chapter: nothing left to play', () => {
    const t = tl.chapterTimeline({ blocks, start: 2, end: 6, cursor: 5, finished: true });
    assert.equal(t.elapsedMs, t.durationMs);
    assert.equal(t.bufferedMs, t.durationMs);
    assert.equal(tl.display(t, 1).remaining, '−0:00');
  });

  it('the time left in the document: this chapter on its timeline plus later blocks estimated, at the speed', () => {
    const t = { durationMs: 40000, elapsedMs: 10000 };
    const ms = tl.documentRemainingMs({ blocks, chapterEnd: 4, perChar: 1000 / 15, timeline: t });
    assert.equal(ms, 30000 + 20000);
    assert.equal(tl.leftLabel(ms, 1), '~1 min left at 1×');
    assert.equal(tl.leftLabel(ms, 2), 'Less than a minute left at 2×');
    assert.equal(tl.leftLabel(7800000 * 1.25, 1.25), '~2 h 10 min left at 1.25×');
    assert.equal(tl.leftLabel(18 * 60000, 1), '~18 min left at 1×');
  });
});
