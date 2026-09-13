"""Unit tests for incremental Live Reader text segmentation."""

from src.live_tts.segmenter import LiveTextSegmenter, clean_markdown


def test_natural_mode_waits_for_speakable_boundary():
    segmenter = LiveTextSegmenter(mode="natural")

    assert segmenter.append("Hello unfinished") == []
    segments = segmenter.append(" sentence. ")

    assert [segment.text for segment in segments] == ["Hello unfinished sentence."]
    assert segments[0].source_start == 0
    assert segments[0].source_end <= len("Hello unfinished sentence. ")


def test_natural_mode_keeps_commas_and_clauses_in_the_sentence():
    segmenter = LiveTextSegmenter(mode="natural", max_chars=600, max_words=100)
    sentence = (
        "Inasmuch as many have undertaken to compile a narrative among us, "
        "just as eyewitnesses delivered it to us; it seemed good to me also, "
        "having followed all things closely, to write an orderly account."
    )

    segments = segmenter.append(sentence + " ")

    assert [segment.text for segment in segments] == [sentence]


def test_responsive_mode_releases_short_comma_clause():
    segmenter = LiveTextSegmenter(mode="responsive")

    segments = segmenter.append("In the days of Herod, king of Judea, there was a priest.")

    assert segments[0].text == "In the days of Herod,"


def test_streamed_token_punctuation_waits_for_the_next_delta():
    segmenter = LiveTextSegmenter(mode="natural")

    assert segmenter.append("We use Node.") == []
    segments = segmenter.append("js for the build today. ")

    assert [segment.text for segment in segments] == ["We use Node.js for the build today."]


def test_natural_mode_recognizes_curly_closing_quote_across_deltas():
    segmenter = LiveTextSegmenter(mode="natural")

    assert segmenter.append("“Fear not, Zacharias.") == []
    segments = segmenter.append("” And the angel said more. ")

    assert [segment.text for segment in segments] == [
        "“Fear not, Zacharias.”",
        "And the angel said more.",
    ]


def test_natural_mode_recognizes_encoded_closing_quote():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append("He said &ldquo;Stop.&rdquo; Then he left. ")

    assert [segment.text for segment in segments] == ['He said “Stop.”', "Then he left."]


def test_responsive_mode_does_not_split_html_entity_or_number():
    segmenter = LiveTextSegmenter(mode="responsive")

    assert segmenter.append("Peter &amp; ") == []
    assert segmenter.append("John counted 1,") == []
    segments = segmenter.append("000 people. ")

    assert [segment.text for segment in segments] == ["Peter & John counted 1,000 people."]


def test_link_split_across_deltas_is_cleaned_as_one_link():
    segmenter = LiveTextSegmenter(mode="natural")

    assert segmenter.append("See [the docs](https://example.") == []
    segments = segmenter.append("com/guide) for details. ")

    assert [segment.text for segment in segments] == ["See the docs for details."]


def test_natural_hard_cap_prefers_a_late_clause_boundary():
    segmenter = LiveTextSegmenter(mode="natural", max_chars=50, max_words=100)

    segments = segmenter.append(
        "This opening phrase continues for a while, then more unpunctuated prose follows "
    )

    assert segments[0].text == "This opening phrase continues for a while,"


def test_idle_flush_releases_terminal_punctuation_at_buffer_end():
    segmenter = LiveTextSegmenter(mode="natural")
    segmenter.append("Hello.")

    segments = segmenter.flush(idle=True)

    assert [segment.text for segment in segments] == ["Hello."]


def test_natural_mode_keeps_hard_wrapped_prose_in_one_sentence():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append(
        "It was the best of times, it was the worst\nof times, it was the age of wisdom. "
    )

    assert [segment.text for segment in segments] == [
        "It was the best of times, it was the worst of times, it was the age of wisdom."
    ]


def test_natural_mode_keeps_bold_heading_as_its_own_segment():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append("**Chapter One**\nThe story begins here. ")

    assert [segment.text for segment in segments] == ["Chapter One", "The story begins here."]


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


def test_markdown_cleanup_decodes_html_entities_after_removing_markup():
    source = "[**LUKE**](chapter.xhtml)**&#x20;1&#x20;**&#x49;nasmuch &amp; more."

    assert clean_markdown(source) == "LUKE 1 Inasmuch & more."


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
    segments = segmenter.append("` After. ")

    assert segmenter.in_code_fence is False
    assert [segment.text for segment in segments] == ["After."]


def test_commit_ends_an_unclosed_code_fence_for_future_input():
    segmenter = LiveTextSegmenter(mode="natural")
    segmenter.append("```python\nhidden")

    segmenter.flush()
    segments = segmenter.append("Future prose. ")

    assert segmenter.in_code_fence is False
    assert [segment.text for segment in segments] == ["Future prose."]
