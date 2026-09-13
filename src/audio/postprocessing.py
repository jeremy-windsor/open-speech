from __future__ import annotations

from typing import Iterator

import numpy as np


class StreamingEdgeTrimmer:
    """Trim only stream edges while preserving voiced chunks and interior pauses."""

    def __init__(
        self,
        sample_rate: int,
        *,
        threshold: float = 0.01,
        lead_padding_ms: int = 50,
        tail_padding_ms: int = 150,
    ) -> None:
        self.threshold = float(threshold)
        self.lead_padding_samples = max(0, int(sample_rate * lead_padding_ms / 1000))
        self.tail_padding_samples = max(0, int(sample_rate * tail_padding_ms / 1000))
        self._started = False
        self._leading = np.empty(0, dtype=np.float32)
        self._trailing = np.empty(0, dtype=np.float32)

    def push(self, chunk: np.ndarray) -> np.ndarray:
        samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if not samples.size:
            return samples
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)

        if not self._started:
            voiced = np.flatnonzero(np.abs(samples) > self.threshold)
            if not voiced.size:
                self._leading = self._keep_last(
                    self._join(self._leading, samples),
                    self.lead_padding_samples,
                )
                return np.empty(0, dtype=np.float32)
            first_voiced = int(voiced[0])
            leading = self._keep_last(
                self._join(self._leading, samples[:first_voiced]),
                self.lead_padding_samples,
            )
            samples = self._join(leading, samples[first_voiced:])
            self._leading = np.empty(0, dtype=np.float32)
            self._started = True

        combined = self._join(self._trailing, samples)
        voiced = np.flatnonzero(np.abs(combined) > self.threshold)
        if not voiced.size:
            self._trailing = combined
            return np.empty(0, dtype=np.float32)
        last_voiced = int(voiced[-1]) + 1
        output = combined[:last_voiced]
        self._trailing = combined[last_voiced:]
        return output

    def finish(self) -> np.ndarray:
        if not self._started:
            return np.empty(0, dtype=np.float32)
        output = self._trailing[: self.tail_padding_samples]
        self._trailing = np.empty(0, dtype=np.float32)
        return output

    @staticmethod
    def _join(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        if not first.size:
            return second.copy()
        if not second.size:
            return first.copy()
        return np.concatenate((first, second)).astype(np.float32, copy=False)

    @staticmethod
    def _keep_last(samples: np.ndarray, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        return samples[-count:].copy()


def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if len(audio) == 0:
        return audio
    idx = np.where(np.abs(audio) > threshold)[0]
    if len(idx) == 0:
        return audio
    return audio[idx[0]: idx[-1] + 1]


def normalize_output(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if len(audio) == 0:
        return audio
    max_val = float(np.max(np.abs(audio)))
    if max_val <= 1e-8:
        return audio
    return np.clip(audio * (peak / max_val), -1.0, 1.0)


def process_tts_chunks(
    chunks: Iterator[np.ndarray],
    *,
    trim: bool = True,
    normalize: bool = True,
) -> Iterator[np.ndarray]:
    all_chunks = list(chunks)
    if not all_chunks:
        return iter(())
    audio = np.concatenate(all_chunks)
    if trim:
        audio = trim_silence(audio)
    if normalize:
        audio = normalize_output(audio)
    return iter([audio.astype(np.float32)])
