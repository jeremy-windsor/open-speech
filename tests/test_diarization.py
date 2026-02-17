from __future__ import annotations

from src.diarization.pyannote_diarizer import DiarizationSegment, attach_text_to_speakers


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
        assert False
    except RuntimeError as e:
        assert "open-speech[diarize]" in str(e)
