from __future__ import annotations

import io
import wave

import numpy as np


def wav_bytes_to_float32_mono(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError("Only 16-bit WAV is supported for preprocessing")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def float32_mono_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def normalize_gain(audio: np.ndarray, target_dbfs: float = -18.0) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms <= 1e-8:
        return audio
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0)


def reduce_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    try:
        import noisereduce as nr  # type: ignore
    except ImportError as e:
        raise RuntimeError("Noise reduction requires optional dependency: pip install 'open-speech[noise]'") from e
    return nr.reduce_noise(y=audio, sr=sample_rate)


def preprocess_stt_audio(wav_bytes: bytes, *, noise_reduce: bool, normalize: bool) -> bytes:
    try:
        audio, sr = wav_bytes_to_float32_mono(wav_bytes)
    except Exception:
        # Keep backward compatibility for tests/inputs that provide non-WAV bytes
        return wav_bytes
    if noise_reduce:
        audio = reduce_noise(audio, sr)
    if normalize:
        audio = normalize_gain(audio)
    return float32_mono_to_wav_bytes(audio, sr)
