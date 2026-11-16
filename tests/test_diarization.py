from __future__ import annotations

import logging
import random
import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

from src.config import settings
from src.diarization import pyannote_diarizer
from src.diarization.pyannote_diarizer import (
    DIARIZATION_MODEL,
    DiarizationSegment,
    DiarizationSetupError,
    PyannoteDiarizer,
    attach_text_to_speakers,
    label_transcript_segments,
    unload_pipelines,
)


def test_attach_text_no_segments():
    assert attach_text_to_speakers("hello", []) == []


def test_attach_text_empty_text():
    segs = [DiarizationSegment("SPEAKER_00", 0.0, 1.0)]
    out = attach_text_to_speakers("", segs)
    assert out[0]["text"] == ""


def test_attach_text_single_segment():
    segs = [DiarizationSegment("SPEAKER_00", 0.0, 2.0)]
    out = attach_text_to_speakers("hello world", segs)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[0]["text"] == "hello world"


def test_attach_text_multiple_segments():
    segs = [
        DiarizationSegment("SPEAKER_00", 0.0, 1.0),
        DiarizationSegment("SPEAKER_01", 1.0, 3.0),
    ]
    out = attach_text_to_speakers("one two three four", segs)
    assert len(out) == 2
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["speaker"] == "SPEAKER_01"


def test_attach_text_preserves_times():
    segs = [DiarizationSegment("S", 1.25, 2.5)]
    out = attach_text_to_speakers("x", segs)
    assert out[0]["start"] == 1.25
    assert out[0]["end"] == 2.5


def test_attach_text_assigns_remaining_words_to_last():
    segs = [
        DiarizationSegment("A", 0.0, 0.1),
        DiarizationSegment("B", 0.1, 10.0),
    ]
    out = attach_text_to_speakers("a b c d e f", segs)
    assert "f" in out[-1]["text"]


def test_diarizer_import_error_message(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pyannote"):
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from src.diarization.pyannote_diarizer import PyannoteDiarizer
    try:
        PyannoteDiarizer()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "open-speech[diarize]" in str(e)


# ── Speakers placed by the transcript's timestamps ──────────────────────────


def test_attach_text_follows_transcript_timestamps_not_turn_length():
    # Each turn's text is what was said between its start and end, however long the
    # turn and however fast the speaker talks.
    turns = [DiarizationSegment("A", 0.0, 10.0), DiarizationSegment("B", 10.0, 12.0)]
    transcript = [
        {"start": 0.0, "end": 10.0, "text": " Yes."},
        {"start": 10.0, "end": 12.0, "text": " one two three four five six seven eight nine ten"},
    ]

    out = attach_text_to_speakers("Yes. one two three four five six seven eight nine ten", turns, transcript)

    assert [(o["speaker"], o["start"], o["end"], o["text"]) for o in out] == [
        ("A", 0.0, 10.0, "Yes."),
        ("B", 10.0, 12.0, "one two three four five six seven eight nine ten"),
    ]


def test_attach_text_gives_a_segment_to_the_turn_it_overlaps_most():
    turns = [DiarizationSegment("A", 0.0, 2.0), DiarizationSegment("B", 2.0, 4.0)]
    transcript = [
        {"start": 0.0, "end": 1.4, "text": "first"},
        {"start": 1.5, "end": 3.0, "text": "second"},
    ]

    out = attach_text_to_speakers("first second", turns, transcript)

    assert [o["text"] for o in out] == ["first", "second"]


def test_attach_text_gives_a_segment_outside_every_turn_to_the_nearest_one():
    turns = [DiarizationSegment("A", 0.0, 2.0), DiarizationSegment("B", 6.0, 8.0)]

    out = attach_text_to_speakers("late", turns, [{"start": 5.0, "end": 5.5, "text": "late"}])

    assert [o["text"] for o in out] == ["", "late"]


def test_attach_text_without_timestamps_splits_by_turn_duration():
    turns = [DiarizationSegment("A", 0.0, 1.0), DiarizationSegment("B", 1.0, 2.0)]

    out = attach_text_to_speakers("one two three four", turns, [{"text": "one two three four"}])

    assert [o["text"] for o in out] == ["one two", "three four"]


def test_label_transcript_segments_copies_each_segment_with_its_speaker():
    turns = [DiarizationSegment("A", 0.0, 2.0), DiarizationSegment("B", 2.0, 4.0)]
    transcript = [
        {"id": 0, "start": 0.0, "end": 1.9, "text": " hi", "avg_logprob": -0.1},
        {"id": 1, "start": 2.1, "end": 3.5, "text": " hello"},
    ]

    labelled = label_transcript_segments(transcript, turns)

    assert labelled == [{**transcript[0], "speaker": "A"}, {**transcript[1], "speaker": "B"}]
    assert "speaker" not in transcript[0]
    assert label_transcript_segments(transcript, []) == []


def _turn_by_the_rule(turns, start, end) -> int:
    """The turn overlapping [start, end] most, else the nearest; ties to the earlier turn."""
    def rank(index):
        turn = turns[index]
        overlap = max(min(end, turn.end) - max(start, turn.start), 0.0)
        gap = max(turn.start - end, start - turn.end, 0.0)
        return (-overlap, gap, index)

    return min(range(len(turns)), key=rank)


def test_speakers_follow_the_overlap_rule_in_random_meetings():
    # pyannote turns overlap, leave gaps and touch at their ends; Whisper segments can
    # fall in a gap or have no length. Times on a half-second grid make ties common.
    rng = random.Random(7)
    for _ in range(300):
        turns = []
        for index in range(rng.randint(1, 8)):
            start = rng.randint(0, 40) / 2
            turns.append(DiarizationSegment(f"S{index}", start, start + rng.randint(0, 10) / 2))
        transcript = []
        for index in range(rng.randint(1, 12)):
            start = rng.randint(0, 44) / 2
            transcript.append({"start": start, "end": start + rng.randint(0, 6) / 2, "text": f"w{index}"})

        labelled = label_transcript_segments(transcript, turns)
        attached = attach_text_to_speakers("", turns, transcript)

        expected = [_turn_by_the_rule(turns, seg["start"], seg["end"]) for seg in transcript]
        assert [seg["speaker"] for seg in labelled] == [turns[index].speaker for index in expected]
        assert [turn["text"] for turn in attached] == [
            " ".join(seg["text"] for seg, best in zip(transcript, expected) if best == index)
            for index in range(len(turns))
        ]


def test_speaker_matching_stays_fast_for_long_meetings():
    # Comparing every Whisper segment with every turn took seconds for an hour-long
    # meeting, all of it on the request's event loop.
    turns = [DiarizationSegment(f"SPEAKER_{i % 3:02d}", i * 2.0, i * 2.0 + 2.5) for i in range(3000)]
    transcript = [{"start": i * 2.0 + 0.3, "end": i * 2.0 + 1.9, "text": f"w{i}"} for i in range(3000)]

    started = time.perf_counter()
    labelled = label_transcript_segments(transcript, turns)
    attached = attach_text_to_speakers("", turns, transcript)
    elapsed = time.perf_counter() - started

    assert [seg["speaker"] for seg in labelled] == [turn.speaker for turn in turns]
    assert [turn["text"] for turn in attached] == [seg["text"] for seg in transcript]
    assert elapsed < 2.0


# ── Pipeline lifecycle ──────────────────────────────────────────────────────


class _FakePipeline:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.devices: list[str] = []

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, path):
        turn = SimpleNamespace(start=0.0, end=1.0)
        return SimpleNamespace(itertracks=lambda **kwargs: [(turn, None, "SPEAKER_00")])


@pytest.fixture
def fake_pyannote(monkeypatch):
    """pyannote.audio stand-in recording each Pipeline.from_pretrained download."""
    loads: list[str] = []
    refusals: list[str] = []
    errors: list[Exception] = []

    class Pipeline:
        @staticmethod
        def from_pretrained(checkpoint, *args, **kwargs):
            loads.append(checkpoint)
            if refusals:
                refusals.pop()
                return None  # pyannote.audio 3.x when the gated download is refused
            if errors:
                raise errors.pop()
            return _FakePipeline(checkpoint)

    pyannote = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = Pipeline
    pyannote.audio = pyannote_audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setattr(pyannote_diarizer, "_pipelines", {})
    return SimpleNamespace(loads=loads, refusals=refusals, errors=errors)


@pytest.fixture
def fake_torch(monkeypatch):
    state = SimpleNamespace(cuda_available=False, emptied=0)

    def empty_cache():
        state.emptied += 1

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: state.cuda_available, empty_cache=empty_cache)
    torch.device = lambda name: name
    monkeypatch.setitem(sys.modules, "torch", torch)
    return state


def test_pipeline_is_loaded_once_and_shared_by_later_requests(fake_pyannote):
    # Loading the pipeline takes seconds or more; it must not happen on every request.
    first = PyannoteDiarizer(device="cpu")
    second = PyannoteDiarizer(device="cpu")

    assert fake_pyannote.loads == [DIARIZATION_MODEL]
    assert second._pipeline is first._pipeline
    assert second.diarize(b"RIFF") == [DiarizationSegment("SPEAKER_00", 0.0, 1.0)]


def test_each_model_gets_its_own_pipeline(fake_pyannote):
    a = PyannoteDiarizer("org/a", device="cpu")
    b = PyannoteDiarizer("org/b", device="cpu")

    assert PyannoteDiarizer("org/a", device="cpu")._pipeline is a._pipeline
    assert (a._pipeline.checkpoint, b._pipeline.checkpoint) == ("org/a", "org/b")
    assert fake_pyannote.loads == ["org/a", "org/b"]


def test_unload_pipelines_releases_the_cached_pipeline(fake_pyannote, fake_torch):
    PyannoteDiarizer(device="cpu")

    unload_pipelines()
    PyannoteDiarizer(device="cpu")

    assert fake_pyannote.loads == [DIARIZATION_MODEL, DIARIZATION_MODEL]
    assert fake_torch.emptied == 0  # CUDA unavailable: nothing to release there


def test_unload_pipelines_frees_gpu_memory(fake_pyannote, fake_torch):
    fake_torch.cuda_available = True
    PyannoteDiarizer(device="cuda")

    unload_pipelines()

    assert fake_torch.emptied == 1


@pytest.mark.parametrize(("stt_device", "cuda_available", "expected"), [
    ("cuda", True, ["cuda"]),
    ("auto", True, ["cuda"]),
    ("cuda", False, []),
    ("cpu", True, []),
])
def test_pipeline_runs_on_the_configured_stt_device(
    fake_pyannote, fake_torch, monkeypatch, stt_device, cuda_available, expected
):
    fake_torch.cuda_available = cuda_available
    monkeypatch.setattr(settings, "stt_device", stt_device)

    diarizer = PyannoteDiarizer()

    assert diarizer._pipeline.devices == expected


def test_cuda_fallback_is_logged_once_not_on_every_request(fake_pyannote, fake_torch, monkeypatch, caplog):
    monkeypatch.setattr(pyannote_diarizer, "_cpu_fallback_logged", set())
    monkeypatch.setattr(settings, "stt_device", "cuda")

    with caplog.at_level(logging.WARNING, logger="open-speech"):
        for _ in range(3):
            PyannoteDiarizer()

    assert [record.message for record in caplog.records if "CUDA is unavailable" in record.message] == [
        "STT_DEVICE=cuda but CUDA is unavailable; diarizing on CPU"
    ]


@pytest.mark.parametrize(("device", "emptied"), [("cuda", 1), ("cpu", 0)])
def test_gpu_diarization_returns_its_working_memory(fake_pyannote, fake_torch, device, emptied):
    # The pipeline stays cached, but the activations of a long file should not stay
    # reserved on a GPU that Whisper and TTS share.
    fake_torch.cuda_available = True
    diarizer = PyannoteDiarizer(device=device)

    diarizer.diarize(b"RIFF")

    assert fake_torch.emptied == emptied


def test_refused_gated_download_is_an_actionable_error_and_is_retried(fake_pyannote):
    # pyannote/speaker-diarization-3.1 is gated: without accepted conditions and HF_TOKEN,
    # pyannote.audio 3.x returns None, which used to fail later as "'NoneType' ... callable".
    fake_pyannote.refusals.append(DIARIZATION_MODEL)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        PyannoteDiarizer(device="cpu")

    # Once the operator fixes access, the next request loads it without a restart.
    assert PyannoteDiarizer(device="cpu")._pipeline.checkpoint == DIARIZATION_MODEL
    assert fake_pyannote.loads == [DIARIZATION_MODEL, DIARIZATION_MODEL]


def test_refused_gated_segmentation_download_is_an_actionable_error_and_is_retried(fake_pyannote):
    # speaker-diarization-3.1 also loads pyannote/segmentation-3.0, gated separately.
    # pyannote.audio 3.x returns None for that refusal and then calls .eval() on it.
    fake_pyannote.errors.append(AttributeError("'NoneType' object has no attribute 'eval'"))

    with pytest.raises(DiarizationSetupError) as exc_info:
        PyannoteDiarizer(device="cpu")

    message = str(exc_info.value)
    assert "HF_TOKEN" in message
    assert "pyannote/segmentation-3.0" in message
    assert "NoneType" not in message
    assert PyannoteDiarizer(device="cpu")._pipeline.checkpoint == DIARIZATION_MODEL


def test_unrelated_pipeline_load_errors_are_not_reported_as_gated(fake_pyannote):
    fake_pyannote.errors.append(AttributeError("'SpeakerDiarization' object has no attribute 'klustering'"))

    with pytest.raises(AttributeError, match="klustering"):
        PyannoteDiarizer(device="cpu")
