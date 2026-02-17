from __future__ import annotations

import tempfile
from dataclasses import dataclass


@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float


class PyannoteDiarizer:
    def __init__(self) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Speaker diarization requires optional dependency: pip install 'open-speech[diarize]'"
            ) from e
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

    def diarize(self, wav_bytes: bytes) -> list[DiarizationSegment]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(wav_bytes)
            f.flush()
            diarization = self._pipeline(f.name)
        out: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.append(DiarizationSegment(speaker=speaker, start=float(turn.start), end=float(turn.end)))
        return out


def attach_text_to_speakers(text: str, segments: list[DiarizationSegment]) -> list[dict]:
    """Simple text allocation across diarization segments."""
    if not segments:
        return []
    words = text.split()
    if not words:
        return [{"speaker": s.speaker, "start": s.start, "end": s.end, "text": ""} for s in segments]

    total_dur = sum(max(0.0, s.end - s.start) for s in segments) or 1.0
    cursor = 0
    out: list[dict] = []
    for i, seg in enumerate(segments):
        dur = max(0.0, seg.end - seg.start)
        share = dur / total_dur
        take = max(1, int(round(share * len(words)))) if i < len(segments) - 1 else len(words) - cursor
        seg_words = words[cursor: cursor + take]
        cursor += take
        out.append({"speaker": seg.speaker, "start": seg.start, "end": seg.end, "text": " ".join(seg_words)})
    if cursor < len(words):
        out[-1]["text"] = (out[-1]["text"] + " " + " ".join(words[cursor:])).strip()
    return out
