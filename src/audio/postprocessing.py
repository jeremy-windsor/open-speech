from __future__ import annotations

from typing import Iterator

import numpy as np


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
