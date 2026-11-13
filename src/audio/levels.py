"""A live session's input audio in a few numbers, for the line it logs when it ends.

The "ended" lines of /v1/audio/stream and /v1/realtime say how much audio arrived, how loud it got
and how speech-like it scored. So a client that sends silence can be told apart, at INFO, from one
whose speech the VAD rejects or whose speech is transcribed to nothing.
"""

from __future__ import annotations

import math

import numpy as np

# PCM16's full scale: -32768 is the loudest sample, at 0 dBFS
PCM16_FULL_SCALE = 32768


class InputLevels:
    """How much PCM16 audio a session received, its peak, the VAD's top score and its speech starts.

    It keeps four numbers and never the audio, so each frame costs O(1) memory. Nothing the user
    said reaches the log through it, and no text.
    """

    def __init__(self) -> None:
        self.seconds = 0.0
        self.peak = 0  # the largest |sample|, of PCM16_FULL_SCALE
        self.vad_max: float | None = None  # None until the VAD scores a window
        self.speech_starts = 0

    def add_pcm16(self, pcm16: bytes, sample_rate: int) -> None:
        """Count a frame of PCM16 LE mono audio at ``sample_rate``."""
        samples = np.frombuffer(pcm16, dtype="<i2", count=len(pcm16) // 2)
        if samples.size == 0:
            return
        self.seconds += samples.size / sample_rate
        # int() first: -32768 has no int16 absolute value.
        self.peak = max(self.peak, int(samples.max()), -int(samples.min()))

    def add_vad(self, probability: float) -> None:
        """Count one window's speech probability."""
        if self.vad_max is None or probability > self.vad_max:
            self.vad_max = probability

    def speech_started(self) -> None:
        """Count a speech start (the session's speech_start or speech_started event)."""
        self.speech_starts += 1

    def summary(self) -> str:
        """For example ``audio=6.9 s, peak=-23.1 dBFS, vad_max=0.97, speech=2``.

        The peak is "silent" when every sample was 0 (or no audio came), and vad_max is "none" when
        the VAD scored no window: it was off, or less than one 32 ms window of audio arrived.
        """
        if self.peak:
            peak = f"{20 * math.log10(self.peak / PCM16_FULL_SCALE):.1f} dBFS"
        else:
            peak = "silent"
        vad_max = "none" if self.vad_max is None else f"{self.vad_max:.2f}"
        return f"audio={self.seconds:.1f} s, peak={peak}, vad_max={vad_max}, speech={self.speech_starts}"
