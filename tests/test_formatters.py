"""Tests for SRT/VTT/text formatters."""

import pytest

from src.formatters import format_as_text, format_as_srt, format_as_vtt, format_transcription


def _diarized_result():
    return {
        "text": "Hello there. Hi. Bye.",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": " there.", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 3.0, "text": " Hi.", "speaker": "SPEAKER_01"},
            {"start": 3.0, "end": 4.0, "text": " Bye.", "speaker": "SPEAKER_00"},
        ],
    }


class TestFormatAsText:
    def test_extracts_text(self):
        assert format_as_text({"text": "Hello world"}) == "Hello world"

    def test_strips_whitespace(self):
        assert format_as_text({"text": "  hello  "}) == "hello"

    def test_empty(self):
        assert format_as_text({}) == ""

    def test_diarized_segments_give_one_line_per_speaker_turn(self):
        assert format_as_text(_diarized_result()) == (
            "SPEAKER_00: Hello there.\nSPEAKER_01: Hi.\nSPEAKER_00: Bye."
        )


class TestFormatAsSrt:
    def test_with_segments(self):
        result = {
            "text": "Hello world",
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "Hello"},
                {"start": 1.5, "end": 3.0, "text": "world"},
            ],
        }
        srt = format_as_srt(result)
        assert "1\n00:00:00,000 --> 00:00:01,500\nHello" in srt
        assert "2\n00:00:01,500 --> 00:00:03,000\nworld" in srt

    def test_without_segments(self):
        result = {"text": "Hello world", "duration": 2.5}
        srt = format_as_srt(result)
        assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in srt

    def test_empty(self):
        assert format_as_srt({"text": ""}) == ""

    def test_hours(self):
        result = {"text": "test", "segments": [{"start": 3661.5, "end": 3663.0, "text": "test"}]}
        srt = format_as_srt(result)
        assert "01:01:01,500" in srt

    @pytest.mark.parametrize(("end", "expected"), [(2.3, "00:00:02,300"), (59.9996, "00:01:00,000")])
    def test_rounds_to_nearest_millisecond(self, end, expected):
        # whisper.utils.format_timestamp rounds; truncating gives 02,299 and 00:59,999.
        result = {"text": "Hello", "segments": [{"start": 1.0, "end": end, "text": "Hello"}]}
        assert f"00:00:01,000 --> {expected}" in format_as_srt(result)

    def test_cue_numbers_stay_sequential_when_empty_segment_skipped(self):
        result = {
            "text": "one three",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "one"},
                {"start": 1.0, "end": 2.0, "text": "   "},
                {"start": 2.0, "end": 3.0, "text": "three"},
            ],
        }
        lines = format_as_srt(result).splitlines()
        counters = [lines[i - 1] for i, line in enumerate(lines) if "-->" in line]
        assert counters == ["1", "2"]

    def test_diarized_cues_name_their_speaker(self):
        # SRT has no speaker markup, so the label prefixes the cue text.
        srt = format_as_srt(_diarized_result())
        assert "2\n00:00:01,000 --> 00:00:02,000\nSPEAKER_00: there.\n" in srt
        assert "3\n00:00:02,000 --> 00:00:03,000\nSPEAKER_01: Hi.\n" in srt

    @pytest.mark.parametrize("segments", [True, False], ids=["segments", "no_segments"])
    def test_an_arrow_in_the_text_cannot_end_the_cue(self, segments):
        # A cue line holding "-->" reads as a timing line; whisper.utils writes "->" instead.
        # SRT has no escapes, so < and & stay as they are.
        result = {"text": "go --> there <b> & back", "duration": 1.0}
        if segments:
            result["segments"] = [{"start": 0.0, "end": 1.0, "text": " go --> there <b> & back"}]

        cue_lines = format_as_srt(result).splitlines()[2:]

        assert cue_lines[0] == "go -> there <b> & back"
        assert not any("-->" in line for line in cue_lines)

    def test_a_blank_line_in_the_text_does_not_end_the_cue(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.0, "text": "one\n\ntwo"}]}
        assert format_as_srt(result) == "1\n00:00:00,000 --> 00:00:01,000\none\ntwo\n"


class TestFormatAsVtt:
    def test_starts_with_header(self):
        result = {"text": "Hello", "segments": [{"start": 0, "end": 1, "text": "Hello"}]}
        vtt = format_as_vtt(result)
        assert vtt.startswith("WEBVTT")

    def test_uses_dot_separator(self):
        result = {"text": "Hello", "segments": [{"start": 0, "end": 1.5, "text": "Hello"}]}
        vtt = format_as_vtt(result)
        assert "00:00:00.000 --> 00:00:01.500" in vtt

    def test_rounds_to_nearest_millisecond(self):
        result = {"text": "Hello", "segments": [{"start": 1.0, "end": 2.3, "text": "Hello"}]}
        assert "00:00:01.000 --> 00:00:02.300" in format_as_vtt(result)

    def test_diarized_cues_use_voice_spans(self):
        # WebVTT names a cue's speaker with a <v> voice span.
        vtt = format_as_vtt(_diarized_result())
        assert "00:00:00.000 --> 00:00:01.000\n<v SPEAKER_00>Hello\n" in vtt
        assert "00:00:02.000 --> 00:00:03.000\n<v SPEAKER_01>Hi.\n" in vtt

    @pytest.mark.parametrize("segments", [True, False], ids=["segments", "no_segments"])
    def test_cue_text_escapes_markup_and_arrows(self, segments):
        # WebVTT: "<" opens a tag (the text after it is lost), "&" an entity, and a cue line
        # may not hold "-->" (it reads as a timing line and the cue is dropped).
        result = {"text": "Scores <5 points & a --> b", "duration": 1.0}
        if segments:
            result["segments"] = [{"start": 0.0, "end": 1.0, "text": " Scores <5 points & a --> b"}]

        cue_lines = format_as_vtt(result).splitlines()[3:]

        assert cue_lines[0] == "Scores &lt;5 points &amp; a --&gt; b"
        assert not any("-->" in line for line in cue_lines)

    def test_diarized_voice_span_keeps_the_text_after_a_less_than(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.0, "text": " x <y and z", "speaker": "SPEAKER_00"}]}
        assert "\n<v SPEAKER_00>x &lt;y and z\n" in format_as_vtt(result)

    def test_a_blank_line_in_the_text_does_not_end_the_cue(self):
        result = {"text": "", "segments": [{"start": 0.0, "end": 1.0, "text": "one\n\ntwo"}]}
        assert format_as_vtt(result) == "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\none\ntwo\n"

    def test_without_segments(self):
        result = {"text": "Hello", "duration": 2.0}
        vtt = format_as_vtt(result)
        assert "WEBVTT" in vtt
        assert "Hello" in vtt

    def test_empty(self):
        assert format_as_vtt({"text": ""}) == "WEBVTT\n"


class TestFormatTranscription:
    def test_text_format(self):
        content, ct = format_transcription({"text": "hello"}, "text")
        assert content == "hello"
        assert ct == "text/plain"

    def test_srt_format(self):
        content, ct = format_transcription({"text": "hello", "duration": 1.0}, "srt")
        assert "00:00:00,000" in content
        assert ct == "text/plain"

    def test_vtt_format(self):
        content, ct = format_transcription({"text": "hello", "duration": 1.0}, "vtt")
        assert "WEBVTT" in content
        assert ct == "text/vtt"

    def test_json_format(self):
        content, ct = format_transcription({"text": "hello"}, "json")
        assert content == ""
        assert ct == "application/json"
