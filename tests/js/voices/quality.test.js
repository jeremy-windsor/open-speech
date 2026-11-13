// Voices: the clip quality checks of the clone wizard, quality.js (step 1). Pure: no browser.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { load } from './helpers.js';

const RATE = 16000;

/** A 220 Hz sine at `amp` for `seconds`, with `gapEvery` s of silence between `talk` s bursts when given. */
function clip({ seconds = 5, amp = 0.1, rate = RATE, talk = 0, gap = 0, noise = 0 } = {}) {
  const out = new Float32Array(Math.round(seconds * rate));
  let seed = 3;
  for (let i = 0; i < out.length; i += 1) {
    const t = i / rate;
    const inBurst = !talk || (t % (talk + gap)) < talk;
    seed = (seed * 16807) % 2147483647;
    const n = noise ? noise * ((seed / 2147483647) * 2 - 1) : 0;
    out[i] = (inBurst ? amp * Math.sin(2 * Math.PI * 220 * t) : 0) + n;
  }
  return out;
}

const check = (result, id) => result.checks.find((c) => c.id === id);

describe('views/voices quality checks', () => {
  it('a clean 5 s clip with pauses passes every check', async () => {
    const { analyzeClip, summaryText } = await load('views/voices/quality.js');
    const r = analyzeClip(clip({ seconds: 5, amp: 0.2, talk: 0.8, gap: 0.3 }), RATE);
    assert.deepEqual(r.checks.map((c) => [c.id, c.status]), [['duration', 'ok'], ['clipping', 'ok'], ['loudness', 'ok'], ['noise', 'ok'], ['rate', 'ok']]);
    assert.equal(r.warnings, 0);
    assert.equal(summaryText(r), 'All checks look good.');
    assert.equal(check(r, 'duration').value, '5.0 s');
    assert.equal(check(r, 'rate').value, '16 kHz');
  });

  it('duration: under 3 s or over 30 s is a warning that says what to do', async () => {
    const { analyzeClip } = await load('views/voices/quality.js');
    const short = check(analyzeClip(clip({ seconds: 2, amp: 0.2, talk: 0.5, gap: 0.2 }), RATE), 'duration');
    assert.equal(short.status, 'warn');
    assert.match(short.text, /3 to 30 seconds/);
    const long = check(analyzeClip(clip({ seconds: 31, amp: 0.2, talk: 0.8, gap: 0.3, rate: 8000 }), 8000), 'duration');
    assert.equal(long.status, 'warn');
    assert.equal(check(analyzeClip(clip({ seconds: 3, amp: 0.2 }), RATE), 'duration').status, 'ok', '3 s is enough');
  });

  it('clipping: a few peaks at the limit are fine; more than 0.1 % of the samples is a warning', async () => {
    const { analyzeClip } = await load('views/voices/quality.js');
    const few = clip({ seconds: 4, amp: 0.2, talk: 0.8, gap: 0.3 });
    few[100] = 1;
    few[200] = -1;
    const fine = check(analyzeClip(few, RATE), 'clipping');
    assert.equal(fine.status, 'ok');
    const many = clip({ seconds: 4, amp: 1.4, talk: 0.8, gap: 0.3 }).map((v) => Math.max(-1, Math.min(1, v)));
    const bad = check(analyzeClip(many, RATE), 'clipping');
    assert.equal(bad.status, 'warn');
    assert.match(bad.text, /cut off/);
  });

  it('loudness: RMS between -30 and -12 dBFS is good; quieter or louder warns', async () => {
    const { analyzeClip } = await load('views/voices/quality.js');
    assert.equal(check(analyzeClip(clip({ amp: 0.005, talk: 0.8, gap: 0.3 }), RATE), 'loudness').status, 'warn');
    assert.match(check(analyzeClip(clip({ amp: 0.005, talk: 0.8, gap: 0.3 }), RATE), 'loudness').text, /Too quiet/);
    assert.equal(check(analyzeClip(clip({ amp: 0.1, talk: 0.8, gap: 0.3 }), RATE), 'loudness').status, 'ok');
    assert.match(check(analyzeClip(clip({ amp: 0.9 }), RATE), 'loudness').text, /Too loud/);
  });

  it('noise floor: the quietest 50 ms frames must be below -50 dBFS', async () => {
    const { analyzeClip } = await load('views/voices/quality.js');
    const noisy = check(analyzeClip(clip({ amp: 0.2, talk: 0.8, gap: 0.3, noise: 0.05 }), RATE), 'noise');
    assert.equal(noisy.status, 'warn');
    assert.match(noisy.text, /noisy/);
    assert.equal(check(analyzeClip(clip({ amp: 0.2, talk: 0.8, gap: 0.3, noise: 0.0005 }), RATE), 'noise').status, 'ok');
  });

  it('sample rate: below 16 kHz warns; the summary counts the warnings and never blocks', async () => {
    const { analyzeClip, summaryText } = await load('views/voices/quality.js');
    const r = analyzeClip(clip({ seconds: 4, amp: 0.2, talk: 0.8, gap: 0.3, rate: 8000 }), 8000);
    assert.equal(check(r, 'rate').status, 'warn');
    assert.equal(check(r, 'rate').value, '8 kHz');
    assert.equal(r.warnings, 1);
    assert.equal(summaryText(r), '1 thing to check. You can still use this clip.');
    assert.equal(summaryText({ warnings: 3 }), '3 things to check. You can still use this clip.');
  });

  it('frameRms and percentile handle short and empty input', async () => {
    const { frameRms, percentile, analyzeClip } = await load('views/voices/quality.js');
    assert.deepEqual(frameRms(new Float32Array(0), 10), []);
    assert.equal(frameRms(new Float32Array([0.5, -0.5]), 10).length, 1);
    assert.equal(percentile([5, 1, 3, 2, 4], 0.1), 1);
    assert.equal(percentile([], 0.5), 0);
    const empty = analyzeClip(new Float32Array(0), 0);
    assert.equal(empty.durationS, 0);
    assert.equal(check(empty, 'loudness').value, 'silent');
  });
});
