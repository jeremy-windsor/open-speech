"""Unit tests for incremental Live Reader text segmentation."""

from src.live_tts.segmenter import LiveTextSegmenter, clean_markdown


def test_natural_mode_waits_for_speakable_boundary():
    segmenter = LiveTextSegmenter(mode="natural")

    assert segmenter.append("Hello unfinished") == []
    segments = segmenter.append(" sentence. ")

    assert [segment.text for segment in segments] == ["Hello unfinished sentence."]
    assert segments[0].source_start == 0
    assert segments[0].source_end <= len("Hello unfinished sentence. ")


def test_idle_flush_keeps_an_unfinished_trailing_word():
    segmenter = LiveTextSegmenter(mode="natural")
    segmenter.append("Read these words then wai")

    segments = segmenter.flush(idle=True)

    assert [segment.text for segment in segments] == ["Read these words then"]
    assert [segment.text for segment in segmenter.flush()] == ["wai"]


def test_instant_word_mode_emits_completed_words():
    segmenter = LiveTextSegmenter(mode="instant_word")

    segments = segmenter.append("one two ")

    assert [segment.text for segment in segments] == ["one", "two"]
    assert segmenter.flush() == []


def test_segment_size_is_bounded_without_punctuation():
    segmenter = LiveTextSegmenter(mode="natural", max_chars=40, max_words=4)

    segments = segmenter.append("one two three four five six ")

    assert segments[0].text == "one two three four"
    assert all(len(segment.text) <= 40 for segment in segments)


def test_markdown_prose_is_cleaned_and_fenced_code_is_skipped():
    segmenter = LiveTextSegmenter(mode="natural")
    text = "# Read **this** [label](https://example.com).\n```python\nprint('secret')\n``` Done."

    segments = segmenter.append(text) + segmenter.flush()
    spoken = " ".join(segment.text for segment in segments)

    assert "Read this label." in spoken
    assert "Code block skipped." in spoken
    assert "print" not in spoken
    assert "Done." in spoken


def test_clean_markdown_keeps_inline_code_and_link_labels():
    assert clean_markdown("Use `open_speech` from [the project](https://example.com).") == (
        "Use open_speech from the project."
    )


def test_markdown_cleanup_preserves_identifiers_and_math_operators():
    assert clean_markdown("TTS_LIVE_ENABLED means 2 * 3, with *emphasis*.") == (
        "TTS_LIVE_ENABLED means 2 * 3, with emphasis."
    )


def test_abbreviations_and_numbered_markers_do_not_end_a_segment():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append("Dr. Smith wrote:\n1. First item\n")

    assert [segment.text for segment in segments] == ["Dr. Smith wrote:", "First item"]


def test_hard_cap_does_not_split_combining_character_cluster():
    segmenter = LiveTextSegmenter(mode="natural", max_chars=20, max_words=50)
    text = "x" * 19 + "e\u0301" + "remaining"

    first = segmenter.append(text)[0]

    assert first.text.endswith("e\u0301")


def test_idle_flush_preserves_a_split_closing_code_fence():
    segmenter = LiveTextSegmenter(mode="natural")
    segmenter.append("Before. ```python\nhidden\n``")

    segmenter.flush(idle=True)
    segments = segmenter.append("` After.")

    assert segmenter.in_code_fence is False
    assert [segment.text for segment in segments] == ["After."]


def test_commit_ends_an_unclosed_code_fence_for_future_input():
    segmenter = LiveTextSegmenter(mode="natural")
    segmenter.append("```python\nhidden")

    segmenter.flush()
    segments = segmenter.append("Future prose.")

    assert segmenter.in_code_fence is False
    assert [segment.text for segment in segments] == ["Future prose."]
