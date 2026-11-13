/**
 * PCM helpers (pure): conversion, WAV encoding, levels, and decoding a file to mono.
 * @module core/pcm
 */

/**
 * Float samples -> 16-bit PCM, clamped (today's conversion).
 * @param {Float32Array|number[]} float32
 * @returns {Int16Array}
 */
export function floatToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) {
    out[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
  }
  return out;
}

/**
 * Join captured chunks, keeping at most `maxSamples` (the server's duration cap × the sample rate).
 * @param {Float32Array[]} chunks
 * @param {{maxSamples?: number}} [options] 0 or missing = no cap
 * @returns {Float32Array}
 */
export function concat(chunks, { maxSamples = 0 } = {}) {
  const total = (chunks || []).reduce((n, c) => n + c.length, 0);
  const length = maxSamples > 0 ? Math.min(total, Math.floor(maxSamples)) : total;
  const out = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks || []) {
    if (offset >= length) break;
    const take = Math.min(chunk.length, length - offset);
    out.set(take === chunk.length ? chunk : chunk.subarray(0, take), offset);
    offset += take;
  }
  return out;
}

/**
 * Mono 16-bit PCM WAV (RIFF/WAVE), today's encodeWavPcm16.
 * @param {Float32Array} float32
 * @param {number} sampleRate
 * @returns {Blob} type audio/wav
 */
export function encodeWav(float32, sampleRate) {
  const samples = float32 || new Float32Array(0);
  const rate = Math.round(sampleRate) || 16000;
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, s) => { for (let i = 0; i < s.length; i += 1) view.setUint8(offset + i, s.charCodeAt(i)); };
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);   // byte rate
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, v < 0 ? v * 32768 : v * 32767, true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

export const DECODE_ERROR = 'This browser could not decode that file. Upload WAV, MP3, M4A, FLAC, OGG, or WebM.';
export const EMPTY_FILE_ERROR = 'Choose a non-empty audio file';

export const DECODE_RATE = 48000;

/**
 * Decode an audio file and mix it down to mono. Decoding uses an OfflineAudioContext:
 * never a realtime context, which iOS limits per page and which may not start outside a gesture. An
 * OfflineAudioContext resamples to its own rate, so the samples come back at DECODE_RATE (48 kHz) unless a
 * caller passes its own constructor (it is then called as `new Ctor(1, 1, DECODE_RATE)` for an offline
 * context, or `new Ctor()` for anything else).
 * @param {Blob} file
 * @param {{AudioContextCtor?: Function}} [options]
 * @returns {Promise<{samples: Float32Array, sampleRate: number, durationS: number}>}
 * @throws {Error} EMPTY_FILE_ERROR or DECODE_ERROR
 */
export async function decodeToMono(file, { AudioContextCtor } = {}) {
  if (!file || !file.size) throw new Error(EMPTY_FILE_ERROR);
  const Offline = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  const Ctor = AudioContextCtor || Offline;
  if (!Ctor) throw new Error(DECODE_ERROR);
  let ctx;
  try {
    ctx = Ctor === Offline || /Offline/.test(Ctor.name || '') ? new Ctor(1, 1, DECODE_RATE) : new Ctor();
  } catch {
    throw new Error(DECODE_ERROR);
  }
  try {
    const data = await file.arrayBuffer();
    const decoded = await ctx.decodeAudioData(data.slice(0));
    const mono = new Float32Array(decoded.length);
    const channels = decoded.numberOfChannels || 1;
    for (let c = 0; c < channels; c += 1) {
      const source = decoded.getChannelData(c);
      for (let i = 0; i < source.length; i += 1) mono[i] += source[i] / channels;
    }
    return { samples: mono, sampleRate: decoded.sampleRate, durationS: decoded.duration || mono.length / decoded.sampleRate };
  } catch {
    throw new Error(DECODE_ERROR);
  } finally {
    // A realtime context passed in by a caller is closed; an offline one has nothing to release.
    if (ctx && typeof ctx.close === 'function' && !(Offline && ctx instanceof Offline)) {
      try { await ctx.close(); } catch { /* already closed */ }
    }
  }
}

/** Root mean square of the samples (0 for none). */
export function rms(float32) {
  if (!float32 || !float32.length) return 0;
  let sum = 0;
  for (let i = 0; i < float32.length; i += 1) sum += float32[i] * float32[i];
  return Math.sqrt(sum / float32.length);
}

/** Level in dBFS; silence is -100. */
export function toDb(value) {
  return value > 0 ? Math.max(-100, 20 * Math.log10(value)) : -100;
}

/** The largest absolute sample. */
export function peak(float32) {
  let max = 0;
  for (let i = 0; i < (float32 ? float32.length : 0); i += 1) {
    const a = Math.abs(float32[i]);
    if (a > max) max = a;
  }
  return max;
}
