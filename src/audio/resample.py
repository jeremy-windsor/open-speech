"""Anti-aliased sample-rate conversion, for whole clips or streams of chunks.

Both paths use the polyphase low-pass FIR that scipy.signal.resample_poly
designs, so a stream converted chunk by chunk with StreamingResampler matches
converting it in one go: there are no seams at chunk boundaries, and the
output length does not drift with the chunk size.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import firwin, resample_poly

# A rate coprime with the other (e.g. 191999 Hz to 16 kHz) would need a
# polyphase filter with millions of taps. Approximating the ratio keeps every
# common rate exact and is off by at most about 0.05% for the rest.
MAX_RATIO_TERM = 1000

# Outputs computed per block, which bounds the scratch memory for long chunks
_BLOCK_OUTPUTS = 4096


def conversion_ratio(src_rate: int, dst_rate: int) -> tuple[int, int]:
    """Return the (up, down) factors for src_rate to dst_rate, reduced by their gcd."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("Sample rates must be positive")
    ratio = Fraction(dst_rate, src_rate)
    if max(ratio.numerator, ratio.denominator) > MAX_RATIO_TERM:
        if ratio < 1:
            ratio = ratio.limit_denominator(MAX_RATIO_TERM)
        else:
            ratio = 1 / (1 / ratio).limit_denominator(MAX_RATIO_TERM)
    if ratio == 0:
        raise ValueError(f"Cannot resample {src_rate} Hz to {dst_rate} Hz")
    return ratio.numerator, ratio.denominator


def resample(audio: np.ndarray, src_rate: int, dst_rate: int, *, padtype: str = "constant") -> np.ndarray:
    """Resample mono audio, returning float32.

    ``padtype`` is passed to resample_poly; the default treats audio outside
    the clip as silence, which is what StreamingResampler does too.
    """
    samples = np.asarray(audio, dtype=np.float64)
    up, down = conversion_ratio(src_rate, dst_rate)
    if up == down or samples.size == 0:
        return samples.astype(np.float32)
    return resample_poly(samples, up, down, padtype=padtype).astype(np.float32)


@lru_cache(maxsize=32)
def _polyphase_filter(up: int, down: int) -> tuple[np.ndarray, int]:
    """Return resample_poly's filter split into phases, and its half length.

    Row p holds taps h[p], h[p + up], ... in reverse order, so output sample
    n is the dot product of row (n*down + half_len) % up with the input
    window that ends at sample (n*down + half_len) // up.
    """
    max_rate = max(up, down)
    half_len = 10 * max_rate
    h = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", 5.0)) * up
    taps = -(-len(h) // up)
    padded = np.zeros(taps * up)
    padded[:len(h)] = h
    phases = padded.reshape(taps, up).T[:, ::-1]
    return np.ascontiguousarray(phases), half_len


class StreamingResampler:
    """Resample a mono stream chunk by chunk, keeping filter state between chunks.

    process() returns every output sample whose input has arrived, which
    lags the input by half the filter (about 10 samples at the lower rate).
    flush() ends the stream as if silence followed it and returns the rest,
    so the concatenated output equals resample() of the whole stream.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        self._up, self._down = conversion_ratio(src_rate, dst_rate)
        self._passthrough = self._up == self._down
        if not self._passthrough:
            self._phases, self._half_len = _polyphase_filter(self._up, self._down)
        self._reset()

    def _reset(self) -> None:
        taps = 0 if self._passthrough else self._phases.shape[1]
        # Input history, starting at absolute sample _history_start. The
        # zeros stand for the silence before the stream starts.
        self._history = np.zeros(max(taps - 1, 0))
        self._history_start = -len(self._history)
        self._input_samples = 0
        self._next_output = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed the next chunk and return the output samples it completes (float32)."""
        samples = np.asarray(chunk, dtype=np.float64)
        if samples.ndim != 1:
            raise ValueError("StreamingResampler takes mono 1-D audio")
        if self._passthrough:
            return samples.astype(np.float32)
        self._history = np.concatenate((self._history, samples))
        self._input_samples += len(samples)
        # Output n needs input up to sample (n*down + half_len) // up.
        ready = self._input_samples * self._up - self._half_len
        return self._emit(max(0, -(-ready // self._down)))

    def flush(self) -> np.ndarray:
        """End the stream: return the remaining output and start over (float32)."""
        if self._passthrough:
            return np.zeros(0, dtype=np.float32)
        total = -(-self._input_samples * self._up // self._down)
        if total > self._next_output:
            last_input = ((total - 1) * self._down + self._half_len) // self._up
            missing = last_input + 1 - (self._history_start + len(self._history))
            if missing > 0:
                self._history = np.concatenate((self._history, np.zeros(missing)))
        tail = self._emit(total)
        self._reset()
        return tail

    def _emit(self, end: int) -> np.ndarray:
        if end <= self._next_output:
            return np.zeros(0, dtype=np.float32)
        taps = self._phases.shape[1]
        windows = np.lib.stride_tricks.sliding_window_view(self._history, taps)
        out = np.empty(end - self._next_output, dtype=np.float32)
        for block in range(self._next_output, end, _BLOCK_OUTPUTS):
            n = np.arange(block, min(block + _BLOCK_OUTPUTS, end))
            position = n * self._down + self._half_len
            starts = position // self._up - (taps - 1) - self._history_start
            out[block - self._next_output:block - self._next_output + len(n)] = np.einsum(
                "ij,ij->i", windows[starts], self._phases[position % self._up],
            )
        self._next_output = end
        # Keep only the history the next output sample still needs.
        next_start = (end * self._down + self._half_len) // self._up - (taps - 1)
        drop = min(max(0, next_start - self._history_start), len(self._history))
        self._history = self._history[drop:]
        self._history_start += drop
        return out
