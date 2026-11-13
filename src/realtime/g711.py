"""G.711 mu-law and A-law codecs for PCM16 mono, using lookup tables.

These give the same bytes as the stdlib audioop module (removed in Python
3.13) for every 16-bit sample and every code.
"""

from __future__ import annotations

import numpy as np

_BIAS = 0x84
_ULAW_CLIP = 8159
_ULAW_SEGMENT_ENDS = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF])
_ALAW_SEGMENT_ENDS = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF])


def _ulaw_encode_table() -> np.ndarray:
    # Index i is the int16 sample whose bit pattern is i.
    value = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.int32) >> 2
    mask = np.where(value < 0, 0x7F, 0xFF)
    magnitude = np.minimum(np.abs(value), _ULAW_CLIP) + (_BIAS >> 2)
    segment = np.searchsorted(_ULAW_SEGMENT_ENDS, magnitude)
    code = (segment << 4) | ((magnitude >> (segment + 1)) & 0x0F)
    code = np.where(segment >= 8, 0x7F, code)
    return (code ^ mask).astype(np.uint8)


def _alaw_encode_table() -> np.ndarray:
    value = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.int32) >> 3
    negative = value < 0
    mask = np.where(negative, 0x55, 0xD5)
    magnitude = np.where(negative, -value - 1, value)
    segment = np.searchsorted(_ALAW_SEGMENT_ENDS, magnitude)
    shift = np.where(segment < 2, 1, segment)
    code = (segment << 4) | ((magnitude >> shift) & 0x0F)
    return (code ^ mask).astype(np.uint8)


def _ulaw_decode_table() -> np.ndarray:
    code = ~np.arange(256, dtype=np.int32) & 0xFF
    magnitude = (((code & 0x0F) << 3) + _BIAS) << ((code & 0x70) >> 4)
    return np.where(code & 0x80, _BIAS - magnitude, magnitude - _BIAS).astype("<i2")


def _alaw_decode_table() -> np.ndarray:
    code = np.arange(256, dtype=np.int32) ^ 0x55
    segment = (code & 0x70) >> 4
    magnitude = (code & 0x0F) << 4
    magnitude = np.where(
        segment == 0,
        magnitude + 8,
        (magnitude + 0x108) << np.maximum(segment - 1, 0),
    )
    return np.where(code & 0x80, magnitude, -magnitude).astype("<i2")


_ULAW_ENCODE = _ulaw_encode_table()
_ALAW_ENCODE = _alaw_encode_table()
_ULAW_DECODE = _ulaw_decode_table()
_ALAW_DECODE = _alaw_decode_table()


def pcm16_to_ulaw(pcm16: bytes) -> bytes:
    """Encode PCM16 LE mono to G.711 mu-law, one byte per sample."""
    return _ULAW_ENCODE[np.frombuffer(pcm16, dtype="<u2")].tobytes()


def pcm16_to_alaw(pcm16: bytes) -> bytes:
    """Encode PCM16 LE mono to G.711 A-law, one byte per sample."""
    return _ALAW_ENCODE[np.frombuffer(pcm16, dtype="<u2")].tobytes()


def ulaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 mu-law to PCM16 LE mono."""
    return _ULAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()


def alaw_to_pcm16(data: bytes) -> bytes:
    """Decode G.711 A-law to PCM16 LE mono."""
    return _ALAW_DECODE[np.frombuffer(data, dtype=np.uint8)].tobytes()
