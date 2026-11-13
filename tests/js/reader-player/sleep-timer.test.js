import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { load } from './helpers.js';

const sleep = await load('views/reader/sleep-timer.js');

describe('reader-player/sleep-timer', () => {
  it('counts down in wall-clock time only while playing; a pause or a wait for a part holds it', () => {
    let s = sleep.start({ minutes: 15 });
    assert.equal(s.mode, 'minutes');
    assert.equal(s.remainingMs, 900000);
    s = sleep.tick(s, { playing: true, now: 1000 });   // the first tick only notes the time
    assert.equal(s.remainingMs, 900000);
    s = sleep.tick(s, { playing: true, now: 61000 });
    assert.equal(s.remainingMs, 840000);
    s = sleep.tick(s, { playing: false, now: 121000 });   // the minute before the pause counted
    assert.equal(s.remainingMs, 780000);
    s = sleep.tick(s, { playing: false, now: 3600000 });   // paused for an hour: nothing
    assert.equal(s.remainingMs, 780000);
    s = sleep.tick(s, { playing: true, now: 3601000 });
    s = sleep.tick(s, { playing: true, now: 3602000 });
    assert.equal(s.remainingMs, 779000);
  });

  it('fades the volume over the last 10 s and expires at 0', () => {
    let s = sleep.start({ minutes: 1 });
    s = sleep.tick(s, { playing: true, now: 0 });
    s = sleep.tick(s, { playing: true, now: 55000 });
    assert.equal(s.fading, true);
    assert.equal(sleep.volumeFor(s), 0.5);
    assert.equal(sleep.volumeFor(s, { fade: false }), null, 'fade off: the volume is never touched');
    s = sleep.tick(s, { playing: true, now: 61000 });
    assert.equal(sleep.expired(s), true);
    assert.equal(s.remainingMs, 0);
  });

  it('chapters mode stops before the part that starts the boundary block', () => {
    const s = sleep.start({ chapters: 2 }, { boundaryBlock: 40 });
    assert.equal(s.mode, 'chapters');
    assert.equal(sleep.stopsBefore(s, 39), false);
    assert.equal(sleep.stopsBefore(s, 40), true);
    assert.equal(sleep.stopsBefore(sleep.start({ minutes: 5 }), 40), false);
    assert.equal(sleep.start({ chapters: 50 }).chapters, sleep.MAX_CHAPTERS);
  });

  it('labels: the chip, its name (never ticking each second), the sheet line and the radio value', () => {
    const off = sleep.OFF;
    assert.equal(sleep.chipLabel(off), 'Sleep');
    assert.equal(sleep.chipName(off), 'Sleep timer: off. Set sleep timer');
    let s = sleep.start({ minutes: 15 });
    s = { ...s, remainingMs: 760000 };
    assert.equal(sleep.chipLabel(s), '12:40');
    assert.equal(sleep.chipName(s), 'Sleep timer: 13 minutes left. Change sleep timer');
    assert.equal(sleep.chipName({ ...s, remainingMs: 755000 }), sleep.chipName(s), 'the name holds within a minute');
    assert.equal(sleep.remainingText(s), 'Stops in 12:40');
    assert.equal(sleep.optionValue(s), '15');
    const ch = sleep.start({ chapters: 1 }, { boundaryBlock: 9 });
    assert.equal(sleep.chipLabel(ch), 'End of ch.');
    assert.equal(sleep.optionValue(ch), 'chapter');
    assert.equal(sleep.remainingText(ch, 250000), 'Stops at the end of this chapter, in about 4:10');
    assert.equal(sleep.chipLabel(sleep.start({ chapters: 3 })), 'End of 3 ch.');
    assert.equal(sleep.optionValue(sleep.start({ chapters: 3 })), 'chapters');
    assert.deepEqual(sleep.specFor('chapters', 4), { chapters: 4 });
    assert.deepEqual(sleep.specFor('chapter'), { chapters: 1 });
    assert.deepEqual(sleep.specFor('45'), { minutes: 45 });
    assert.equal(sleep.specFor('off'), null);
    assert.deepEqual([...sleep.MINUTE_OPTIONS], [5, 10, 15, 30, 45, 60]);
  });
});
