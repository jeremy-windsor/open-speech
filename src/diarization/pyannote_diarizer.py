from __future__ import annotations

import bisect
import logging
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("open-speech")

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# Gated models a pipeline loads on top of its own, each with its own user conditions.
GATED_DEPENDENCIES = {DIARIZATION_MODEL: ("pyannote/segmentation-3.0",)}

# Loading a pipeline takes seconds or more, so each (model, device) is loaded once per
# process. Requests share it, one diarization at a time.
_pipelines: dict[tuple[str, str], Any] = {}
_load_lock = threading.Lock()
_diarize_lock = threading.Lock()
_cpu_fallback_logged: set[str] = set()


class DiarizationSetupError(RuntimeError):
    """Diarization cannot run until the operator installs or grants something."""


@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float


def _resolve_device(device: str | None) -> str:
    """Map STT_DEVICE to a torch device, falling back to CPU when CUDA is unavailable."""
    if device is None:
        from src.config import settings

        device = settings.stt_device
    device = (device or "cpu").lower()
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        # Resolved on every request; say it once.
        if device not in _cpu_fallback_logged:
            _cpu_fallback_logged.add(device)
            logger.warning("STT_DEVICE=%s but CUDA is unavailable; diarizing on CPU", device)
        return "cpu"
    return "cuda" if device == "auto" else device


def _gated_error(model_id: str) -> DiarizationSetupError:
    repos = " and ".join((model_id, *GATED_DEPENDENCIES.get(model_id, ())))
    return DiarizationSetupError(
        f"Could not load {model_id}: Hugging Face refused a gated model it needs. Accept the "
        f"user conditions of {repos} on Hugging Face, then set HF_TOKEN to a token with "
        "access to them."
    )


def _load_pipeline(pipeline_cls, model_id: str, device: str):
    # huggingface_hub reads HF_TOKEN from the environment for the gated download.
    try:
        pipeline = pipeline_cls.from_pretrained(model_id)
    except (AttributeError, TypeError) as exc:
        # pyannote.audio 3.x also returns None for a refused sub-model download, then
        # calls it ("'NoneType' object has no attribute 'eval'").
        if "NoneType" not in str(exc):
            raise
        logger.warning("Loading %s got no model back", model_id, exc_info=True)
        raise _gated_error(model_id) from exc
    if pipeline is None:
        # pyannote.audio 3.x prints a hint and returns None when the download is refused.
        raise _gated_error(model_id)
    if device != "cpu":
        import torch

        pipeline.to(torch.device(device))
    return pipeline


def unload_pipelines() -> None:
    """Drop the cached diarization pipelines and release their GPU memory."""
    with _load_lock:
        released = bool(_pipelines)
        _pipelines.clear()
    if not released:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PyannoteDiarizer:
    _device = "cpu"

    def __init__(self, model_id: str = DIARIZATION_MODEL, device: str | None = None) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except ImportError as e:
            raise DiarizationSetupError(
                "Speaker diarization requires optional dependency: pip install 'open-speech[diarize]'"
            ) from e
        key = (model_id, _resolve_device(device))
        with _load_lock:
            pipeline = _pipelines.get(key)
            if pipeline is None:
                pipeline = _load_pipeline(Pipeline, *key)
                _pipelines[key] = pipeline
        self._pipeline = pipeline
        self._device = key[1]

    def diarize(self, wav_bytes: bytes) -> list[DiarizationSegment]:
        with tempfile.TemporaryDirectory(prefix="open-speech-diarize-") as temp_dir:
            audio_path = Path(temp_dir) / "input.wav"
            audio_path.write_bytes(wav_bytes)
            with _diarize_lock:
                try:
                    diarization = self._pipeline(str(audio_path))
                finally:
                    if self._device != "cpu":
                        # The pipeline stays cached; its working memory for a long file
                        # should not stay reserved on a GPU shared with Whisper and TTS.
                        import torch

                        torch.cuda.empty_cache()
        out: list[DiarizationSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.append(DiarizationSegment(speaker=speaker, start=float(turn.start), end=float(turn.end)))
        return out


def _timed_segments(transcript_segments: list[dict] | None) -> list[dict]:
    return [
        seg for seg in transcript_segments or []
        if isinstance(seg.get("start"), (int, float)) and isinstance(seg.get("end"), (int, float))
    ]


def _speaker_turn_indices(turns: list[DiarizationSegment], spans: list[tuple[float, float]]) -> list[int]:
    """For each (start, end), the turn overlapping it most, or the nearest if none does.

    Ties go to the earlier turn. Spans are swept in start order against the turns sorted
    by start, so only the turns still open are compared: a long meeting has thousands of
    both, and comparing every pair took seconds.
    """
    by_start = sorted(range(len(turns)), key=lambda i: (turns[i].start, i))
    starts = [turns[i].start for i in by_start]
    by_end = sorted(range(len(turns)), key=lambda i: (turns[i].end, -i))
    ends = [turns[i].end for i in by_end]

    def rank(index: int, start: float, end: float) -> tuple[float, float, int]:
        turn = turns[index]
        overlap = min(end, turn.end) - max(start, turn.start)
        gap = max(turn.start - end, start - turn.end, 0.0)
        return (-max(overlap, 0.0), gap, index)

    best = [0] * len(spans)
    open_turns: list[int] = []
    added = 0
    for k in sorted(range(len(spans)), key=lambda k: spans[k][0]):
        start, end = spans[k]
        while added < len(by_start) and starts[added] < end:
            open_turns.append(by_start[added])
            added += 1
        # A turn that ended by this start overlaps none of the later spans either.
        open_turns = [i for i in open_turns if turns[i].end > start]
        candidates = list(open_turns)
        # When nothing overlaps: the last turn to end before, the first to start after.
        before = bisect.bisect_right(ends, start)
        if before:
            candidates.append(by_end[before - 1])
        after = bisect.bisect_left(starts, end)
        if after < len(starts):
            candidates.append(by_start[after])
        best[k] = min(candidates, key=lambda index: rank(index, start, end))
    return best


def label_transcript_segments(transcript_segments: list[dict] | None, turns: list[DiarizationSegment]) -> list[dict]:
    """Copy each timed transcript segment with the speaker of the turn it overlaps most."""
    if not turns:
        return []
    timed = _timed_segments(transcript_segments)
    indices = _speaker_turn_indices(turns, [(seg["start"], seg["end"]) for seg in timed])
    return [{**seg, "speaker": turns[index].speaker} for seg, index in zip(timed, indices)]


def attach_text_to_speakers(
    text: str,
    segments: list[DiarizationSegment],
    transcript_segments: list[dict] | None = None,
) -> list[dict]:
    """Give each speaker turn the transcript text spoken during it.

    With timestamped transcript segments, each segment goes to the turn it overlaps
    most. Without them, the words are split by each turn's share of the total duration.
    """
    if not segments:
        return []
    timed = _timed_segments(transcript_segments)
    if timed:
        parts: list[list[str]] = [[] for _ in segments]
        indices = _speaker_turn_indices(segments, [(seg["start"], seg["end"]) for seg in timed])
        for seg, index in zip(timed, indices):
            seg_text = str(seg.get("text", "")).strip()
            if seg_text:
                parts[index].append(seg_text)
        return [
            {"speaker": s.speaker, "start": s.start, "end": s.end, "text": " ".join(words)}
            for s, words in zip(segments, parts)
        ]

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
