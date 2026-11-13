import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { FakeAudioBuffer } from '../harness/fakes.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv(options);
  return importFresh('core/pcm.js');
}

const text = (view, offset, n) => String.fromCharCode(...new Uint8Array(view.buffer, offset, n));

describe('core/pcm', () => {
  it('[VL-03] a recording is joined, trimmed to maxSeconds × rate and encoded as mono 16-bit PCM WAV', async () => {
    const pcm = await setup();
    const rate = 8;
    const chunks = [new Float32Array([0.1, 0.2, 0.3, 0.4]), new Float32Array([0.5, 0.6, 0.7, 0.8]), new Float32Array([0.9, 1, 1, 1])];
    const samples = pcm.concat(chunks, { maxSamples: 1 * rate });
    assert.equal(samples.length, 8);
    assert.ok(Math.abs(samples[7] - 0.8) < 1e-6);
    assert.equal(pcm.concat(chunks).length, 12, 'no cap');
    const blob = pcm.encodeWav(samples, rate);
    assert.equal(blob.type, 'audio/wav');
    const view = new DataView(await blob.arrayBuffer());
    assert.equal(text(view, 0, 4), 'RIFF');
    assert.equal(text(view, 8, 4), 'WAVE');
    assert.equal(text(view, 12, 4), 'fmt ');
    assert.equal(view.getUint16(20, true), 1, 'PCM');
    assert.equal(view.getUint16(22, true), 1, 'mono');
    assert.equal(view.getUint32(24, true), rate);
    assert.equal(view.getUint32(28, true), rate * 2);
    assert.equal(view.getUint16(34, true), 16);
    assert.equal(text(view, 36, 4), 'data');
    assert.equal(view.getUint32(40, true), 16);
    assert.equal(view.getUint32(4, true), 36 + 16);
    assert.equal(view.byteLength, 44 + 16);
    assert.equal(view.getInt16(44 + 7 * 2, true), Math.trunc(0.8 * 32767));
  });

  it('encodeWav clamps and uses the asymmetric 16-bit range', async () => {
    const pcm = await setup();
    const view = new DataView(await pcm.encodeWav(new Float32Array([2, -2, -1, 1]), 16000).arrayBuffer());
    assert.deepEqual([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true)), [32767, -32768, -32768, 32767]);
  });

  it('floatToInt16 clamps like today', async () => {
    const pcm = await setup();
    assert.deepEqual(Array.from(pcm.floatToInt16(new Float32Array([0, 0.5, -0.5, 1.5, -1.5]))), [0, 16384, -16384, 32767, -32768]);
  });

  it('decodeToMono averages the channels; empty and undecodable files say what to do', async () => {
    const pcm = await setup();
    const left = new Float32Array([1, 0]);
    const right = new Float32Array([0, 1]);
    class Ctx {
      constructor() { this.closed = false; }
      async decodeAudioData() { return new FakeAudioBuffer({ numberOfChannels: 2, length: 2, sampleRate: 48000, channels: [left, right] }); }
      async close() { this.closed = true; }
    }
    const file = new Blob([new Uint8Array([1, 2, 3])]);
    const out = await pcm.decodeToMono(file, { AudioContextCtor: Ctx });
    assert.deepEqual(Array.from(out.samples), [0.5, 0.5]);
    assert.equal(out.sampleRate, 48000);
    await assert.rejects(pcm.decodeToMono(new Blob([]), { AudioContextCtor: Ctx }), { message: 'Choose a non-empty audio file' });
    class Broken extends Ctx { async decodeAudioData() { throw new Error('EncodingError'); } }
    await assert.rejects(pcm.decodeToMono(file, { AudioContextCtor: Broken }), { message: 'This browser could not decode that file. Upload WAV, MP3, M4A, FLAC, OGG, or WebM.' });
  });

  it('rms, peak and toDb', async () => {
    const pcm = await setup();
    assert.equal(pcm.rms(new Float32Array([0.5, -0.5])), 0.5);
    assert.equal(pcm.rms(new Float32Array([])), 0);
    assert.equal(pcm.peak(new Float32Array([0.2, -0.9, 0.5])), Math.fround(0.9));
    assert.equal(pcm.toDb(0), -100);
    assert.equal(pcm.toDb(1), 0);
  });
});

describe('core/pcm.decodeToMono default (foundation repair)', () => {
  it('decodes with an OfflineAudioContext by default, never a realtime AudioContext', async () => {
    const pcm = await setup();
    const { FakeAudioContext, FakeOfflineAudioContext } = await import('../harness/fakes.js');
    const realtime = FakeAudioContext.instances.length;
    FakeOfflineAudioContext.decodeResult = new FakeAudioBuffer({ numberOfChannels: 1, length: 4, sampleRate: 48000, channels: [new Float32Array([0.25, 0.25, 0.25, 0.25])] });
    try {
      const out = await pcm.decodeToMono(new Blob([new Uint8Array([1, 2, 3])]));
      assert.equal(FakeAudioContext.instances.length, realtime, 'no realtime context');
      const ctx = FakeOfflineAudioContext.instances.at(-1);
      assert.equal(ctx.sampleRate, pcm.DECODE_RATE);
      assert.equal(out.sampleRate, 48000);
      assert.deepEqual(Array.from(out.samples), [0.25, 0.25, 0.25, 0.25]);
    } finally {
      FakeOfflineAudioContext.decodeResult = null;
    }
  });
});
