/**
 * Quality checks for a reference clip (step 1): duration, clipping, loudness, noise floor and
 * sample rate. Pure: no DOM, no browser APIs. Warnings never block saving; they say what to change.
 * @module views/voices/quality
 */

/** The quality-check thresholds (one place, so the copy and the tests agree). */
export const LIMITS = Object.freeze({
  minSeconds: 3,
  maxSeconds: 30,
  clipLevel: 0.999,        // a sample at or above this is clipped
  clipShare: 0.001,        // more than 0.1 % clipped samples is a warning
  loudMinDb: -30,          // RMS of the whole clip, dBFS
  loudMaxDb: -12,
  noiseMaxDb: -50,         // 10th-percentile 50 ms frame RMS, dBFS
  noiseFrameMs: 50,
  noisePercentile: 0.1,
  minRate: 16000,
});

const FLOOR_DB = -100;

function db(value) {
  return value > 0 ? Math.max(FLOOR_DB, 20 * Math.log10(value)) : FLOOR_DB;
}

function dbText(value) {
  return value <= FLOOR_DB ? 'silent' : `${Math.round(value)} dBFS`;
}

function secondsText(seconds) {
  return `${(Math.round(seconds * 10) / 10).toFixed(1)} s`;
}

function percentText(share) {
  const pct = share * 100;
  return `${pct < 0.1 ? pct.toFixed(2) : pct.toFixed(1)}%`;
}

/**
 * The RMS of every whole `frame`-sample window (a clip shorter than one frame gives one window).
 * @param {Float32Array} samples
 * @param {number} frame
 * @returns {number[]}
 */
export function frameRms(samples, frame) {
  const n = samples ? samples.length : 0;
  const size = Math.max(1, Math.floor(frame));
  if (!n) return [];
  if (n < size) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += samples[i] * samples[i];
    return [Math.sqrt(sum / n)];
  }
  const out = [];
  for (let start = 0; start + size <= n; start += size) {
    let sum = 0;
    for (let i = start; i < start + size; i += 1) sum += samples[i] * samples[i];
    out.push(Math.sqrt(sum / size));
  }
  return out;
}

/**
 * The value at `p` (0..1) of a list, nearest-rank on a sorted copy.
 * @param {number[]} values
 * @param {number} p
 */
export function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[rank];
}

/**
 * Measure a mono clip and turn the numbers into checks.
 * @param {Float32Array} samples mono samples in -1..1
 * @param {number} sampleRate Hz
 * @returns {{durationS: number, sampleRate: number, peak: number, clipped: number, clippedShare: number,
 *   rmsDb: number, noiseDb: number, warnings: number,
 *   checks: {id: string, label: string, value: string, status: 'ok'|'warn', text: string}[]}}
 */
export function analyzeClip(samples, sampleRate) {
  const data = samples || new Float32Array(0);
  const rate = Number(sampleRate) > 0 ? Number(sampleRate) : 0;
  const n = data.length;
  const durationS = rate ? n / rate : 0;
  let sum = 0;
  let peak = 0;
  let clipped = 0;
  for (let i = 0; i < n; i += 1) {
    const v = data[i];
    const a = v < 0 ? -v : v;
    sum += v * v;
    if (a > peak) peak = a;
    if (a >= LIMITS.clipLevel) clipped += 1;
  }
  const rmsDb = db(n ? Math.sqrt(sum / n) : 0);
  const clippedShare = n ? clipped / n : 0;
  const frames = frameRms(data, rate ? (rate * LIMITS.noiseFrameMs) / 1000 : n || 1);
  const noiseDb = db(percentile(frames, LIMITS.noisePercentile));

  const checks = [];
  // Duration: 3–30 s.
  if (durationS < LIMITS.minSeconds) {
    checks.push({ id: 'duration', label: 'Length', value: secondsText(durationS), status: 'warn', text: 'Too short for a good clone. Aim for 3 to 30 seconds.' });
  } else if (durationS > LIMITS.maxSeconds) {
    checks.push({ id: 'duration', label: 'Length', value: secondsText(durationS), status: 'warn', text: 'Longer than needed. 3 to 30 seconds works best.' });
  } else {
    checks.push({ id: 'duration', label: 'Length', value: secondsText(durationS), status: 'ok', text: durationS > 15 ? 'A good length for cloning. Breeze does best with 5 to 15 seconds.' : 'A good length for cloning.' });
  }
  // Clipping: none is ideal; more than 0.1 % of the samples is a warning.
  if (clippedShare > LIMITS.clipShare) {
    checks.push({ id: 'clipping', label: 'Clipping', value: percentText(clippedShare), status: 'warn', text: 'The loudest parts are cut off. Record a little quieter or further from the microphone.' });
  } else if (clipped > 0) {
    checks.push({ id: 'clipping', label: 'Clipping', value: percentText(clippedShare), status: 'ok', text: 'A few peaks touch the limit. That is fine.' });
  } else {
    checks.push({ id: 'clipping', label: 'Clipping', value: 'None', status: 'ok', text: 'Nothing is cut off.' });
  }
  // Loudness: RMS between -30 and -12 dBFS.
  if (rmsDb < LIMITS.loudMinDb) {
    checks.push({ id: 'loudness', label: 'Loudness', value: dbText(rmsDb), status: 'warn', text: 'Too quiet. Move closer to the microphone or raise its input level.' });
  } else if (rmsDb > LIMITS.loudMaxDb) {
    checks.push({ id: 'loudness', label: 'Loudness', value: dbText(rmsDb), status: 'warn', text: 'Too loud. Move back a little or lower the input level.' });
  } else {
    checks.push({ id: 'loudness', label: 'Loudness', value: dbText(rmsDb), status: 'ok', text: 'A good speaking level.' });
  }
  // Noise floor: the quietest 50 ms frames (10th percentile) below -50 dBFS.
  if (noiseDb < LIMITS.noiseMaxDb) {
    checks.push({ id: 'noise', label: 'Background noise', value: dbText(noiseDb), status: 'ok', text: 'A quiet background.' });
  } else {
    checks.push({ id: 'noise', label: 'Background noise', value: dbText(noiseDb), status: 'warn', text: 'The background is noisy. A clone copies the noise too, so record somewhere quieter.' });
  }
  // Sample rate: 16 kHz or more.
  const kHz = rate ? `${Math.round(rate / 100) / 10} kHz` : 'unknown';
  if (rate >= LIMITS.minRate) {
    checks.push({ id: 'rate', label: 'Sample rate', value: kHz, status: 'ok', text: 'Enough detail for cloning.' });
  } else {
    checks.push({ id: 'rate', label: 'Sample rate', value: kHz, status: 'warn', text: 'A low sample rate loses detail. Use a better microphone or file.' });
  }
  const warnings = checks.filter((c) => c.status === 'warn').length;
  return { durationS, sampleRate: rate, peak, clipped, clippedShare, rmsDb, noiseDb, warnings, checks };
}

/**
 * One sentence for the whole set (shown above the list and announced once).
 * @param {{warnings: number}} result
 */
export function summaryText(result) {
  const n = result ? result.warnings : 0;
  if (!n) return 'All checks look good.';
  return `${n === 1 ? '1 thing' : `${n} things`} to check. You can still use this clip.`;
}
