/**
 * AudioWorklet processor for microphone capture: posts frames of 2048 samples with their RMS to the main
 * thread. Loaded as the static file /static/js/core/mic-worklet.js (the CSP has no worker-src, so blob:
 * worklets are refused). Importing it outside an AudioWorkletGlobalScope only defines the pure helper.
 * @module core/mic-worklet
 */

export const FRAME_SIZE = 2048;
export const PROCESSOR_NAME = 'open-speech-mic';

/**
 * Collect 128-sample render quanta into fixed-size frames (pure, for tests).
 * @param {number} size
 * @param {(frame: Float32Array, rms: number) => void} onFrame
 * @returns {(input: Float32Array) => void} push
 */
export function createFrameBuffer(size, onFrame) {
  let frame = new Float32Array(size);
  let filled = 0;
  return (input) => {
    let offset = 0;
    while (offset < input.length) {
      const take = Math.min(size - filled, input.length - offset);
      frame.set(input.subarray(offset, offset + take), filled);
      filled += take;
      offset += take;
      if (filled === size) {
        let sum = 0;
        for (let i = 0; i < size; i += 1) sum += frame[i] * frame[i];
        onFrame(frame, Math.sqrt(sum / size));
        frame = new Float32Array(size);
        filled = 0;
      }
    }
  };
}

if (typeof AudioWorkletProcessor === 'function' && typeof registerProcessor === 'function') {
  class MicCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.push = createFrameBuffer(FRAME_SIZE, (samples, level) => {
        this.port.postMessage({ samples, rms: level }, [samples.buffer]);
      });
    }

    process(inputs) {
      const channel = inputs[0] && inputs[0][0];
      if (channel && channel.length) this.push(channel);
      return true;
    }
  }
  registerProcessor(PROCESSOR_NAME, MicCaptureProcessor);
}
