"""Unit tests for incremental Live Reader text segmentation."""

import itertools
import time

import pytest

from src.live_tts.segmenter import LiveTextSegmenter, clean_markdown, split_sentences


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
    segmenter.append("Before.\n```python\nhidden\n``")

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


@pytest.mark.parametrize(
    ("mode", "deltas", "spoken"),
    [
        ("instant_word", ["See [the docs](https://example.com/guide) now. "], ["See", "the docs", "now."]),
        ("instant_word", ["See [the ", "docs](https://exa", "mple.com/guide) now. "], ["See", "the docs", "now."]),
        (
            "responsive",
            ["Read [Herod, king of Judea, and the priest](https://example.com/luke) now. "],
            ["Read Herod, king of Judea, and the priest now."],
        ),
    ],
)
def test_word_and_comma_boundaries_never_split_a_link(mode, deltas, spoken):
    # Link labels are spoken without URLs, which needs the whole link in one segment.
    segmenter = LiveTextSegmenter(mode=mode)

    segments = [segment for delta in deltas for segment in segmenter.append(delta)] + segmenter.flush()

    assert [segment.text for segment in segments] == spoken


def test_tilde_fenced_code_is_skipped_like_a_backtick_fence():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append("Before.\n~~") + segmenter.append("~python\nprint('secret')\n~")
    segments += segmenter.append("~~\nAfter. ") + segmenter.flush()

    assert [segment.text for segment in segments] == ["Before.", "Code block skipped.", "After."]


def test_comparison_signs_are_prose_not_html_tags():
    segmenter = LiveTextSegmenter(mode="natural")

    segments = segmenter.append("Stop when x < 5 and y > 3, then rest. ")

    assert [segment.text for segment in segments] == ["Stop when x < 5 and y > 3, then rest."]
    assert clean_markdown("Use <b>bold</b> text.<br/>") == "Use bold text."


@pytest.mark.parametrize(
    "text",
    [
        "Somewhere " * 90 + "tonight. ",
        # A 378-character sentence followed by more text in the same delta, as the web UI sends it.
        "Somewhere " * 37 + "tonight. " + "Another short sentence follows here. " * 10,
    ],
    ids=["unpunctuated-run", "long-sentence-then-more"],
)
def test_max_chars_bounds_a_segment_even_when_the_word_cap_is_reached(text):
    # The cap is the model's max_input_chars for Chatterbox (350); 80 short words pass it.
    segmenter = LiveTextSegmenter(mode="natural", max_chars=350, max_words=80)

    segments = segmenter.append(text) + segmenter.flush()

    assert max(len(segment.text) for segment in segments) <= 350
    assert " ".join(segment.text for segment in segments).split() == text.split()


@pytest.mark.parametrize(
    "text",
    [
        "That was so much fun~~~ See you tomorrow. Bye for now. ",
        "Wrap code in triple backticks (```) to format it. See you tomorrow. ",
        "Run this:\n```pip install x```\nSee you tomorrow. ",
    ],
    ids=["mid-line-tildes", "mid-line-backticks", "backticks-in-the-info-string"],
)
def test_fence_markers_that_do_not_start_a_fence_line_are_prose(text):
    # CommonMark 4.5: a fence starts a line, and a backtick fence's info string has no backticks.
    segmenter = LiveTextSegmenter(mode="natural")

    spoken = " ".join(segment.text for segment in segmenter.append(text) + segmenter.flush())

    assert "See you tomorrow." in spoken
    assert "Code block skipped" not in spoken


@pytest.mark.parametrize(
    "fenced",
    [
        "````\n```python\nprint(1)\n```\n````",
        "~~~markdown\n```python\nprint(1)\n```\n~~~",
        "  ```\n  ~~~\n  print(1)\n  ~~~\n  ```",
    ],
    ids=["four-backticks", "tildes-around-backticks", "indented-backticks-around-tildes"],
)
def test_a_fence_closes_only_on_its_own_character_at_least_as_long(fenced):
    # CommonMark 4.5: the closing fence uses the opening character, at least as many times.
    segmenter = LiveTextSegmenter(mode="natural")
    text = f"Intro.\n\n{fenced}\n\nAfter the example. "

    segments = [segment for delta in (text[:12], text[12:31], text[31:]) for segment in segmenter.append(delta)]
    segments += segmenter.flush()

    assert [segment.text for segment in segments] == ["Intro.", "Code block skipped.", "After the example."]


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("instant_word", "This is **very important** now. "),
        ("instant_word", "Run `pip install x` now. "),
        ("natural", "**Warning. Do not touch the wire.** Next sentence. "),
        ("responsive", "Use the `alpha, beta, gamma, delta, epsilon` values here. "),
    ],
    ids=["instant-bold", "instant-inline-code", "natural-bold-sentences", "responsive-code-commas"],
)
def test_a_marker_pair_split_between_segments_is_not_spoken(mode, text):
    # README: Markdown prose markers are removed and inline code is spoken.
    segmenter = LiveTextSegmenter(mode=mode)

    spoken = [segment.text for segment in segmenter.append(text) + segmenter.flush()]

    assert len(spoken) > 1
    assert not [piece for piece in spoken if "*" in piece or "`" in piece], spoken
    assert clean_markdown("Keep 2 * 3, snake_case and `x * y`.", fragment=True) == "Keep 2 * 3, snake_case and x * y."


@pytest.mark.parametrize(
    ("max_chars", "max_words", "text", "step"),
    [
        (200, 20, "See [the docs](https://example.com/reference/" + "a1b2c3d4" * 30 + ") for details. ", None),
        (
            200,
            20,
            "We went to the store and then we walked home along the river and we saw a lot of "
            "[the old stone bridge](https://example.com/bridge) birds today. ",
            None,
        ),
        (60, 20, "Read [the reference](https://example.com/" + "a1b2c3d4" * 30 + ") first. ", 7),
    ],
    ids=["url-longer-than-the-char-cap", "word-cap-inside-a-label", "streamed-past-the-char-cap"],
)
def test_the_hard_cap_never_exposes_a_link_url(max_chars, max_words, text, step):
    # README: link labels are spoken without URLs.
    segmenter = LiveTextSegmenter(mode="responsive", max_chars=max_chars, max_words=max_words)
    deltas = [text] if step is None else [text[index : index + step] for index in range(0, len(text), step)]

    segments = [segment for delta in deltas for segment in segmenter.append(delta)] + segmenter.flush()

    spoken = [segment.text for segment in segments]
    assert not [piece for piece in spoken if "example.com" in piece or "a1b2" in piece or "](" in piece], spoken
    assert max(len(piece) for piece in spoken) <= max_chars


@pytest.mark.parametrize("mode", ["natural", "responsive"])
def test_a_link_at_the_end_of_a_sentence_still_ends_it(mode):
    # Citation links end sentences in AI web-search answers; a one-character label is not an initial.
    segmenter = LiveTextSegmenter(mode=mode)
    text = "The sky is blue [1](https://ex.com/a). It rains a lot here. Plan A is [A](https://ex.com/b). Plan B follows. "

    segments = [segment for index in range(0, len(text), 4) for segment in segmenter.append(text[index : index + 4])]

    assert [segment.text for segment in segments] == [
        "The sky is blue 1.",
        "It rains a lot here.",
        "Plan A is A.",
        "Plan B follows.",
    ]


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
def test_a_link_label_holding_brackets_is_spoken_without_its_url(mode):
    # README: link labels are spoken without URLs; CommonMark lets a label hold balanced brackets.
    segmenter = LiveTextSegmenter(mode=mode)

    segments = segmenter.append("Sky is blue [[1]](https://ex.com/a). Read [the [new] docs](https://ex.com/b) now. ")

    spoken = " ".join(segment.text for segment in segments + segmenter.flush())
    assert "ex.com" not in spoken
    assert spoken.split() == "Sky is blue [1]. Read the [new] docs now.".split()


@pytest.mark.parametrize(
    ("mode", "url"),
    [("responsive", "https://example.com/bridge"), ("natural", "https://example.com/" + "a1b2c3d4" * 30)],
    ids=["split-at-the-line-break", "url-longer-than-the-char-cap"],
)
def test_a_link_label_wrapped_onto_the_next_line_never_exposes_its_url(mode, url):
    # A line break ends the hold on an unfinished link: responsive mode splits the link there, and
    # natural mode must still find the link's "[" on the earlier line before the cap cuts the URL.
    segmenter = LiveTextSegmenter(mode=mode)

    # The sentence is still arriving, so only the caps can end a segment.
    segments = segmenter.append("See [the old\n") + segmenter.append(f"stone bridge]({url}) and then")

    spoken = [segment.text for segment in segments + segmenter.flush()]
    assert not [piece for piece in spoken if "example.com" in piece or "](" in piece], spoken
    assert " ".join(spoken).replace("[", "").split() == "See the old stone bridge and then".split()


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
def test_a_stray_bracket_does_not_hold_back_later_lines(mode):
    # A "[" that never closes holds at most its own line; typed text is not committed.
    segmenter = LiveTextSegmenter(mode=mode)
    text = "Note [draft\nThe meeting is at noon.\nBring snacks please.\n"

    segments = [segment for index in range(0, len(text), 3) for segment in segmenter.append(text[index : index + 3])]

    spoken = " ".join(segment.text for segment in segments)
    assert "The meeting is at noon." in spoken
    assert spoken.endswith("Bring snacks please.")


def test_the_idle_flush_keeps_a_word_with_an_unfinished_link_whole():
    segmenter = LiveTextSegmenter(mode="responsive")
    segmenter.append("I am sad :-[ today.")

    assert [segment.text for segment in segmenter.flush(idle=True)] == ["I am sad"]
    assert [segment.text for segment in segmenter.append("\nBut tomorrow will be better.\n")] == [
        ":-[ today.",
        "But tomorrow will be better.",
    ]


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
def test_a_link_inside_inline_code_is_handled_the_same_however_the_text_is_split(mode):
    # Whether a link is inside a code span is decided from its line, including words already
    # spoken, not from where a delta began.
    text = "Use `a [b c](https://x.y/z) d` now and `[label](https://example.com/a)` then. Stop here. "

    def spoken(deltas):
        segmenter = LiveTextSegmenter(mode=mode)
        return [segment.text for delta in deltas for segment in segmenter.append(delta)] + [
            segment.text for segment in segmenter.flush()
        ]

    whole = spoken([text])
    assert spoken(list(text)) == whole
    assert spoken([text[:5], text[5:40], text[40:]]) == whole


def test_the_hard_cap_keeps_flag_pairs_together():
    # UAX #29 GB12/GB13: a regional-indicator pair is one flag; an odd cut re-pairs every later flag.
    flags = "\U0001f1fa\U0001f1f8" * 30
    segmenter = LiveTextSegmenter(mode="natural", max_chars=21, max_words=20)

    segments = segmenter.append(flags + " rest") + segmenter.flush()

    def regional_indicators(text):
        return sum("🇦" <= char <= "🇿" for char in text)

    assert [regional_indicators(segment.text) % 2 for segment in segments] == [0] * len(segments)
    assert [(end - start) % 2 for start, end in split_sentences(flags, 21)] == [0, 0, 0]


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
@pytest.mark.parametrize(
    "text",
    [
        "[ " * 4000,
        "[" * 8000,
        "x" * 8000,
        "[a](b) " * 1100,
        "\n" * 8000 + "Done. ",
        "Ok.\n\n" * 1600,
        "1. Yes\n" * 1150,
    ],
    ids=["open-brackets", "bracket-run", "no-whitespace", "many-links", "blank-lines", "short-paragraphs", "numbered-list"],
)
def test_a_large_delta_does_not_stall_the_event_loop(mode, text):
    # The segmenter runs on the shared event loop, and one append may carry
    # TTS_LIVE_MAX_BUFFER_CHARS (8,192) characters. Each of these took up to 9 seconds.
    segmenter = LiveTextSegmenter(mode=mode)

    started = time.perf_counter()
    segments = segmenter.append(text) + segmenter.flush()
    elapsed = time.perf_counter() - started

    assert segments
    assert elapsed < 1.0, f"segmenting one {len(text):,}-character delta took {elapsed:.1f} s"


def test_the_time_to_cut_one_large_delta_grows_linearly_with_its_length():
    """A whole AI answer of a few hundred thousand characters comes as one delta.
    Each cut deleted its sentence from the front of the whole buffer, so 8 times the text took about 45 times as
    long (400,000 characters 4.3 s). Linear, it takes about 8 times as long; the best of two runs of each."""
    paragraph = (
        "The committee met on Tuesday to discuss the budget. Several members raised concerns about the cost of the "
        "new building, which had risen by almost a third since the spring. After a long debate, they agreed to ask "
        "the architects for a cheaper design, and to meet again in two weeks.\n\n"
    )

    def cut(size: int) -> tuple[float, list]:
        text = (paragraph * (size // len(paragraph) + 1))[:size]
        best, segments = float("inf"), []
        for _ in range(2):
            segmenter = LiveTextSegmenter(mode="natural", max_chars=200, max_words=40)
            started = time.perf_counter()
            segments = segmenter.append(text) + segmenter.flush()
            best = min(best, time.perf_counter() - started)
        return best, segments

    small, small_segments = cut(50_000)
    large, large_segments = cut(400_000)

    assert [segment.text for segment in large_segments[:len(small_segments) - 1]] == [
        segment.text for segment in small_segments[:-1]]
    assert large_segments[-1].source_end == 400_000
    assert all(a.source_end <= b.source_start for a, b in itertools.pairwise(large_segments))
    assert large / small < 20, (
        f"50,000 characters took {small:.3f} s and 400,000 took {large:.3f} s: {large / small:.0f} times as long")


# A few tens of KB in one call, as a Reader paragraph arrives. Matching each span with a lazy
# regex read the rest of the text again from every opener without a closer, holding the GIL.
@pytest.mark.parametrize(
    "text",
    [
        "Hello " + "[" * 32_000,
        "Hello " + "[a](" * 30_000,
        "Hello " + "**x " * 6_000,
        "Hello" + " *x" * 10_000,
        "Hello" + " _x" * 10_000,
        "Hello " + "<!--" * 16_000,
        "Hello " + "<?" * 32_000,
        "Hello " + "<!a" * 36_000,
        "Hello " + "`a` " * 30_000,
    ],
    ids=["brackets", "links", "strong", "asterisks", "underscores", "comments", "instructions", "declarations", "code"],
)
def test_clean_markdown_time_grows_linearly_with_unclosed_markup(text):
    started = time.perf_counter()
    clean_markdown(text)
    assert time.perf_counter() - started < 1.5


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("**a **b** c**", "a *b c*"),
        ("*one\ntwo* and _three\nfour_", "*one two* and _three four_"),
        ("a*b*c and snake_case_name", "a*b*c and snake_case_name"),
        ("~~a~~ __b__ **c** ***x***", "a b c x"),
        ("[a](b) [c](d", "a [c](d"),
        ("![x](y) [](z) [w]", "x [](z) [w]"),
        ("<!-- x --> y <!--", "y <!--"),
        ("<?php x ?> <!DOCTYPE html> <!-->", "<!-->"),
        ("`a` and ￰5￱", "a and ￰5￱"),   # only placeholders of its own are restored
    ],
)
def test_clean_markdown_pairs_each_opener_with_the_first_closer_on_its_line(text, spoken):
    assert clean_markdown(text) == spoken


# CommonMark 0.31.2 (5.1, 5.2) reads a quote or list marker only where a
# line starts. A number such as "2024." or "3)", or a ">", that only starts a segment mid-line is
# prose, and docs/API.md (Live Reader) removes Markdown markers, not prose.
_MID_LINE_MARKER_SHAPED_PROSE = [
    "The meeting moved to 2024. Bring snacks. ",
    "Sales rose 5% to 12. Next quarter is better. ",
    "We won the game 3) to everyone's surprise. ",
    "Total: 42. ",
    "There are three rules. 1) Be kind. 2) Be fair. ",
    "Stop when x < 5 and y > 3, then rest. ",
]


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
@pytest.mark.parametrize("word_by_word", [False, True], ids=["one-append", "token-stream"])
@pytest.mark.parametrize(
    "text", _MID_LINE_MARKER_SHAPED_PROSE, ids=["year", "total", "paren", "after-colon", "numbered-rules", "greater-than"]
)
def test_a_marker_shaped_word_that_starts_a_segment_mid_line_is_spoken(mode, word_by_word, text):
    # instant_word dropped "2024.", "12.", "3)" and ">", responsive "42."
    # after "Total:", and natural "1)" and "2)" after a sentence.
    segmenter = LiveTextSegmenter(mode=mode)
    deltas = [word + " " for word in text.split()] if word_by_word else [text]

    segments = [segment for delta in deltas for segment in segmenter.append(delta)] + segmenter.flush()

    spoken = [segment.text for segment in segments]
    assert " ".join(spoken).split() == text.split(), spoken


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
def test_an_idle_flush_does_not_start_a_new_line(mode):
    # An idle flush releases text mid-line; the next text continues that line.
    segmenter = LiveTextSegmenter(mode=mode)

    segments = segmenter.append("Prices rose to ") + segmenter.flush(idle=True)
    segments += segmenter.append("12. Then they fell. ") + segmenter.flush()

    spoken = [segment.text for segment in segments]
    assert " ".join(spoken).split() == "Prices rose to 12. Then they fell.".split(), spoken


@pytest.mark.parametrize("mode", ["instant_word", "responsive", "natural"])
@pytest.mark.parametrize("step", [None, 1], ids=["one-append", "char-by-char"])
def test_markers_that_start_a_line_are_still_not_spoken(mode, step):
    # A control: list, quote and bullet markers at a line start are Markdown,
    # also after a segment that ended mid-line, and in text that starts a line after a commit.
    text = "Rules: there are two.\n1. Be kind.\n2) Be fair.\n> Quoted 3. here.\n  - Last 4) item.\n"
    deltas = [text] if step is None else list(text)
    segmenter = LiveTextSegmenter(mode=mode)

    segments = [segment for delta in deltas for segment in segmenter.append(delta)] + segmenter.flush()
    segments += segmenter.append("3. After the commit.\n") + segmenter.flush()

    spoken = [segment.text for segment in segments]
    assert " ".join(spoken).split() == (
        "Rules: there are two. Be kind. Be fair. Quoted 3. here. Last 4) item. After the commit.".split()
    ), spoken


# Like the harness, a number is a list number only where it starts its line
# or its sentence ("Add the flour. 2. Mix well.", docs/TTS-BACKENDS.md). After an idle release
# (TTS_LIVE_SENTENCE_IDLE_MS) or a cap cut, the next segment starts mid-sentence, so "12." still ends
# "Prices rose to 12." and "1) mix" is no list item; natural mode "prefers full sentences" (docs/API.md, Live Reader).


def _released(mode, steps, **caps):
    """The segment texts each step releases: ("append", text), ("idle", None) or ("commit", None)."""
    segmenter = LiveTextSegmenter(mode=mode, **caps)
    released = []
    for action, text in steps:
        segments = segmenter.append(text) if action == "append" else segmenter.flush(idle=action == "idle")
        released.append([segment.text for segment in segments])
    return released


@pytest.mark.parametrize("word", ["12.", "twelve."], ids=["number", "control-word"])
def test_a_sentence_end_after_an_idle_release_is_released_at_once(word):
    # Natural mode held "12. " like unfinished prose, until the next sentence
    # ended or another idle window passed.
    released = _released("natural", [("append", "Prices rose to "), ("idle", None), ("append", f"{word} ")])

    assert released == [[], ["Prices rose to"], [word]]


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
@pytest.mark.parametrize("word", ["12.", "twelve."], ids=["number", "control-word"])
@pytest.mark.parametrize("pause_before", [True, False], ids=["idle-release-before", "one-append"])
def test_an_idle_flush_releases_a_final_sentence_end_mid_line(mode, word, pause_before):
    # After an idle release, or after instant_word's word cuts, the idle flush
    # kept a final "12." back ("An idle flush preserves a trailing unfinished word or link"), and only a
    # commit released it. Speak > Live sends typed text without a commit.
    if pause_before:
        steps = [("append", "Prices rose to "), ("idle", None), ("append", word)]
    else:
        steps = [("append", f"Prices rose to {word}")]

    released = _released(mode, steps + [("idle", None), ("commit", None)])

    assert " ".join(text for step in released[:-1] for text in step).split() == ["Prices", "rose", "to", word], released
    assert released[-1] == [], released


@pytest.mark.parametrize("word", ["12.", "twelve."], ids=["number", "control-word"])
def test_a_sentence_that_fits_the_caps_is_not_cut_after_an_idle_release(word):
    # "12." joined the next sentence, and the cap then cut that 80-word sentence
    # one word before its end ("... Then the" | "today.").
    sentence = " ".join(("Then the prices fell again and again " * 12).split()[:79] + ["today."])
    assert len(sentence.split()) == 80 and len(sentence) < 500  # the sentence-mode caps (README)

    released = _released(
        "natural",
        [("append", "Prices rose to "), ("idle", None), ("append", f"{word} {sentence} ")],
        max_chars=500,
        max_words=80,
    )

    assert released[2] == [word, sentence]


@pytest.mark.parametrize("word", ["12.", "twelve."], ids=["number", "control-word"])
def test_a_sentence_end_after_a_cap_cut_is_not_joined_to_the_next_sentence(word):
    # After the word cap cut "one two three four", "12." was read as a list
    # number and "12. Next." was spoken as one sentence.
    released = _released("natural", [("append", f"one two three four {word} Next. ")], max_words=4)

    assert released == [["one two three four", word, "Next."]]


@pytest.mark.parametrize("word", ["1)", "to"], ids=["number", "control-word"])
@pytest.mark.parametrize("word_by_word", [False, True], ids=["one-append", "token-stream"])
def test_a_soft_line_break_after_an_idle_release_does_not_cut_the_sentence(word, word_by_word):
    # The first line after an idle release ("1) mix") was taken for a list item,
    # so natural mode cut "The steps are 1) mix\nand bake." at its soft line break.
    rest = f"{word} mix\nand bake. Done. "
    deltas = [piece + " " for piece in rest.split(" ") if piece] if word_by_word else [rest]

    released = _released("natural", [("append", "The steps are "), ("idle", None)] + [("append", d) for d in deltas])

    assert released[1] == ["The steps are"]
    assert [text for step in released[2:] for text in step] == [f"{word} mix and bake.", "Done."]


@pytest.mark.parametrize("mode", ["natural", "responsive"])
@pytest.mark.parametrize("pause", [None, "before-the-space", "after-the-space"])
@pytest.mark.parametrize(
    ("first", "rest", "spoken"),
    [
        ("Add the flour.", "2. Mix well. ", ["Add the flour.", "2. Mix well."]),
        ("I was born in 1990.", "2001. Was a good year. ", ["I was born in 1990.", "2001. Was a good year."]),
    ],
    ids=["documented-flour", "year-after-a-sentence"],
)
def test_a_number_that_starts_its_sentence_stays_a_list_number(mode, pause, first, rest, spoken):
    # A control: docs/TTS-BACKENDS.md, "Add the flour. 2. Mix well." keeps
    # "2. Mix well." together, also when an idle release came between the sentences.
    if pause is None:
        steps = [("append", f"{first} {rest}")]
    elif pause == "before-the-space":
        steps = [("append", first), ("idle", None), ("append", f" {rest}")]
    else:
        steps = [("append", f"{first} "), ("idle", None), ("append", rest)]

    released = _released(mode, steps + [("commit", None)])

    assert [text for step in released for text in step] == spoken


@pytest.mark.parametrize("mode", ["natural", "responsive"])
def test_text_after_a_commit_continues_the_line_it_cut(mode):
    # A control, formerly test_text_after_a_commit_starts_a_line_again. Its
    # expectation ["Mix well."] was incorrect: a commit doesn't end the line, and
    # "Prices rose to 2. Mix well." is one CommonMark paragraph, so "2." is prose, mid-sentence.
    released = _released(
        mode, [("append", "Prices rose to "), ("idle", None), ("commit", None), ("append", "2. Mix well. ")]
    )

    assert released[-1] == ["2.", "Mix well."]


# Speak > Live sends the text from the caret with a commit (LR-08), then what the
# listener types on with no commit (LR-10). A commit doesn't end the line it cuts (docs/API.md, Live Reader: a commit
# only marks what input_text.done reports), so typed text that continues the line is prose: CommonMark 0.31.2
# reads a list item (5.2), a quote (5.1) or a fence (4.5) only where a line starts.
_TYPED_ON_MID_LINE = {
    "number-period": ("Sales rose 5% to", " 12. Next quarter is better. "),
    "number-paren": ("We won the game", " 3) to everyone's surprise. "),
    "greater-than": ("It is", " > 5 degrees today. "),
    "dash": ("The list is", " - not long. "),
    "backtick-run": ("The code is", " ``` and then it fell.\nThen more. "),
    "tilde-run": ("The code is", " ~~~ and then it fell. Then more. "),
}


def _speakable_words(text):
    # A bare run of backticks or tildes is no word to say; the words around it are.
    return [word for word in text.split() if word.strip("`~")]


def _commit_then_type(mode, before, typed):
    segmenter = LiveTextSegmenter(mode=mode)
    segments = segmenter.append(before) + segmenter.flush()  # Start: the text from the caret, and a commit
    for batch in typed:  # typing on, with no commit
        segments += segmenter.append(batch)
    return [segment.text for segment in segments + segmenter.flush(idle=True)]


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
@pytest.mark.parametrize("feed", ["one-batch", "keystrokes"])
@pytest.mark.parametrize("case", list(_TYPED_ON_MID_LINE))
def test_text_typed_on_after_a_commit_mid_line_keeps_its_first_word(mode, feed, case):
    # A commit forced a line start, so "12.", "3)", ">" or "-" typed on after it
    # was dropped as a marker, and "```" or "~~~" opened a fence that swallowed the rest.
    before, after = _TYPED_ON_MID_LINE[case]

    spoken = _commit_then_type(mode, before, [after] if feed == "one-batch" else list(after))

    assert LiveTextSegmenter.CODE_BLOCK_CUE.strip() not in " ".join(spoken), spoken
    assert _speakable_words(" ".join(spoken)) == _speakable_words(before + after), spoken


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
@pytest.mark.parametrize("feed", ["one-batch", "keystrokes"])
@pytest.mark.parametrize(
    ("before", "after", "spoken_words"),
    [
        ("Rules:\n", "1. Be kind.\n2) Be fair.\n", "Rules: Be kind. Be fair."),
        ("Intro.\n", "> Quoted line.\n- Item.\n", "Intro. Quoted line. Item."),
        ("Intro:\n", "```\nprint(1)\n```\nAfter.\n", f"Intro: {LiveTextSegmenter.CODE_BLOCK_CUE.strip()} After."),
        ("Intro:\n  ", "~~~\nprint(1)\n~~~\nAfter.\n", f"Intro: {LiveTextSegmenter.CODE_BLOCK_CUE.strip()} After."),
    ],
    ids=["list", "quote-and-bullet", "fence", "fence-after-indent"],
)
def test_markers_and_fences_that_start_a_line_after_a_commit_are_still_removed(mode, feed, before, after, spoken_words):
    # A control: a commit after a line break, or after a line break and spaces,
    # still leaves the next text at a line start.
    spoken = _commit_then_type(mode, before, [after] if feed == "one-batch" else list(after))

    assert " ".join(spoken).split() == spoken_words.split(), spoken


@pytest.mark.parametrize("mode", ["natural", "responsive", "instant_word"])
@pytest.mark.parametrize(
    ("context", "typed", "spoken_words"),
    [
        ("Sales rose 5% to", " 12. Next quarter is better. ", "12. Next quarter is better."),
        ("The code is", " ~~~ and then it fell. ", "and then it fell."),
        ("Rules:\n", "1. Be kind. ", "Be kind."),
        ("Rules:\n  ", "> Quoted. ", "Quoted."),
        ("", "- Item. ", "Item."),
    ],
    ids=["mid-line-number", "mid-line-tildes", "after-a-line-break", "after-an-indent", "no-context"],
)
def test_context_tells_a_fresh_segmenter_where_the_next_text_continues(mode, context, typed, spoken_words):
    # After a reset (response.cancel), a client that resends from mid-line gives
    # the text before the resume point; none of it is spoken, and the next text is cleaned as its continuation.
    segmenter = LiveTextSegmenter(mode=mode)
    segmenter.continue_from(context)

    segments = segmenter.append(typed) + segmenter.flush(idle=True)

    spoken = " ".join(segment.text for segment in segments)
    assert LiveTextSegmenter.CODE_BLOCK_CUE.strip() not in spoken
    assert _speakable_words(spoken) == spoken_words.split()
    assert segments[0].source_start >= 0 and segments[-1].source_end <= len(typed)
    assert segmenter.accepted_chars == len(typed)


@pytest.mark.parametrize("mode", ["natural", "responsive"])
@pytest.mark.parametrize(
    ("context", "spoken"),
    [("Add the flour. ", ["2. Mix well."]), ("Prices rose to ", ["2.", "Mix well."])],
    ids=["after-a-sentence", "mid-sentence"],
)
def test_context_tells_whether_the_next_text_starts_a_sentence(mode, context, spoken):
    # A number that starts a sentence is a list number
    # ("Add the flour. 2. Mix well."); one mid-sentence ends "Prices rose to 2.".
    segmenter = LiveTextSegmenter(mode=mode)
    segmenter.continue_from(context)

    segments = segmenter.append("2. Mix well. ") + segmenter.flush()

    assert [segment.text for segment in segments] == spoken


def test_context_is_refused_once_text_has_been_appended():
    # The context describes the text before the first append; later, the
    # segmenter already knows where it stands.
    segmenter = LiveTextSegmenter()
    segmenter.append("Hello")

    with pytest.raises(ValueError):
        segmenter.continue_from("Earlier text")

# ---------------------------------------------------------------------------
# A sound tag such as "[laugh]" or "[clear throat]" (Chatterbox Turbo reads each as
# one token) reaches the voice whole from Talk. A cut by a cap must
# not fall inside one: the AI's answer goes through this segmenter (speakable_segments, a streamed turn).
# ---------------------------------------------------------------------------

SOUND_TAGS = [
    "[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]", "[clear throat]", "[sniff]", "[groan]", "[shush]",
    "[crying]", "[angry]", "[fear]", "[happy]", "[surprised]", "[sarcastic]", "[dramatic]", "[whispering]",
    "[narration]", "[advertisement]",
]


def _broken_tags(texts, tag):
    """The texts holding a piece of ``tag`` without all of it."""
    words = tag[1:-1].split()
    return [text for text in texts if any(word in text for word in words) and tag not in text]


def _natural(text, *, max_chars=500, max_words=80, pieces=None):
    segmenter = LiveTextSegmenter(mode="natural", max_chars=max_chars, max_words=max_words)
    segments = []
    for piece in pieces or [text]:
        segments += segmenter.append(piece)
    return [segment.text for segment in segments + segmenter.flush()]


@pytest.mark.parametrize("tag", SOUND_TAGS)
@pytest.mark.parametrize(
    "template",
    ["{tag} That is a good one.", "That is {tag} a good one.", "That is a good one {tag}",
     "That is a good one. {tag} Anyway, on we go."],
    ids=["start", "middle", "end", "own-sentence"],
)
def test_a_sound_tag_is_spoken_whole(tag, template):
    """A sound tag is prose, not Markdown, and stays as it was written."""
    spoken = _natural(template.format(tag=tag))

    assert sum(text.count(tag) for text in spoken) == 1, spoken
    assert not _broken_tags(spoken, tag), spoken


@pytest.mark.parametrize("tag", ["[laugh]", "[clear throat]"])
@pytest.mark.parametrize("by", ["words", "chars"])
@pytest.mark.parametrize("shift", range(4))
def test_a_cut_by_a_cap_never_falls_inside_a_sound_tag(tag, by, shift):
    """The word cap (TTS_LIVE_SENTENCE_MAX_WORDS) and the character cap (the model's
    input limit) cut a long sentence; "[clear throat]" was cut after "[clear". The tag goes whole into the next
    segment, and every segment stays within the caps."""
    if by == "words":
        text = " ".join(["word"] * (19 + shift)) + f" {tag} and then some more words to finish it."
        caps = {"max_words": 20 + shift, "max_chars": 500}  # the cap ends on the tag's first word
    else:
        text = "a" * 10 + " " + " ".join(["word"] * 10) + f" {tag} and then some more words to finish it."
        caps = {"max_words": 80, "max_chars": 67 + shift}  # the cap ends in the tag (it starts at 61)

    spoken = _natural(text, **caps)

    assert not _broken_tags(spoken, tag), spoken
    assert sum(text.count(tag) for text in spoken) == 1, spoken
    assert all(len(text) <= caps["max_chars"] and len(text.split()) <= caps["max_words"] for text in spoken), spoken
    assert " ".join(spoken).split() == text.split()


@pytest.mark.parametrize("pieces", [
    ["That is funny. [", "laugh", "] Anyway."],
    ["That is funny. [laugh", "] Anyway."],
    ["That is funny.", " [", "clear", " throat", "]", " Anyway."],
], ids=["split-3", "split-2", "clear-throat"])
def test_a_sound_tag_that_arrives_in_pieces_is_spoken_whole(pieces):
    """A streamed answer's tag may come in several deltas."""
    tag = "[clear throat]" if "clear" in pieces else "[laugh]"
    spoken = _natural("", pieces=pieces)

    assert sum(text.count(tag) for text in spoken) == 1, spoken
    assert not _broken_tags(spoken, tag), spoken


def test_a_sound_tag_longer_than_the_word_cap_is_still_cut():
    """A guard: a cap smaller than the tag still bounds the cut, so the segmenter
    never stalls on it."""
    spoken = _natural("[clear throat] and more", max_words=1)

    assert all(len(text.split()) <= 1 for text in spoken), spoken
    assert " ".join(spoken).split() == ["[clear", "throat]", "and", "more"]


# Japanese and Chinese end a sentence at 。！？ with no space after it, and Hindi at । and ॥
# (UAX #29 Sentence_Terminal); 、，；： end a clause. The Live Reader's caps are its defaults: TTS_LIVE_SENTENCE_MAX_CHARS
# 500 and 80 words for natural, 200 and 20 for responsive and instant_word.
JA_1 = "今朝は早く起きて、窓を開けると、冷たい空気が部屋の中に入ってきました。"
JA_2 = "空はまだ少し暗く、遠くの山の上に、薄い雲がゆっくりと流れていました。"
ZH_1 = "春天来了，小镇上的人们纷纷走出家门。"
ZH_2 = "河边的柳树长出了嫩绿的新芽，田野里开满了金黄色的油菜花。"
HI_SENTENCES = ["मैं आज सुबह बाजार गया।", "मैंने ताज़ी सब्ज़ियाँ खरीदीं।", "फिर मैं घर लौट आया।"]
WIDE_MARKS = ("。", "！", "？", "、", "，", "；", "：")
LIVE_CAPS = {"natural": (500, 80), "responsive": (200, 20), "instant_word": (200, 20)}


def _live(mode: str) -> LiveTextSegmenter:
    max_chars, max_words = LIVE_CAPS[mode]
    return LiveTextSegmenter(mode=mode, max_chars=max_chars, max_words=max_words)


@pytest.mark.parametrize("first, second", [(JA_1, JA_2), (ZH_1, ZH_2)], ids=["ja", "zh"])
def test_natural_mode_releases_a_cjk_sentence_at_its_full_stop(first, second):
    """A finished sentence goes out on the append, the last one on the idle window."""
    segmenter = _live("natural")

    assert [segment.text for segment in segmenter.append(first + second)] == [first]
    assert [segment.text for segment in segmenter.flush(idle=True)] == [second]


def test_natural_mode_releases_a_hindi_sentence_at_its_danda():
    """। ends a sentence as "." does."""
    segmenter = _live("natural")

    released = segmenter.append(" ".join(HI_SENTENCES))
    released += segmenter.flush(idle=True)

    assert [segment.text for segment in released] == HI_SENTENCES


def test_a_closing_bracket_after_a_full_stop_stays_with_its_sentence():
    """A 」 may still follow a streamed 。, so the sentence waits for the next character."""
    segmenter = _live("natural")

    assert segmenter.append("「おはよう。") == []
    released = segmenter.append("」今日は晴れです。明")

    assert [segment.text for segment in released] == ["「おはよう。」", "今日は晴れです。"]


@pytest.mark.parametrize("first, second", [(JA_1, JA_2), (ZH_1, ZH_2)], ids=["ja", "zh"])
def test_responsive_mode_releases_cjk_clauses_and_sentences(first, second):
    """Responsive cuts at 、 and ， (after a few words' worth of text) as at ", "."""
    segmenter = _live("responsive")

    released = segmenter.append(first + second)
    released += segmenter.flush(idle=True)
    texts = [segment.text for segment in released]

    assert len(texts) > 2, texts
    assert all(text.endswith(WIDE_MARKS) for text in texts), texts
    assert "".join(texts) == first + second
    if first == JA_1:
        assert texts[0] == "今朝は早く起きて、"


def test_instant_word_releases_each_japanese_clause_as_it_ends():
    """With no spaces, "Each word" speaks up to each clause or sentence mark."""
    segmenter = _live("instant_word")

    on_append = [segment.text for segment in segmenter.append(JA_1 + JA_2)]
    on_idle = [segment.text for segment in segmenter.flush(idle=True)]

    assert on_append == [
        "今朝は早く起きて、", "窓を開けると、", "冷たい空気が部屋の中に入ってきました。", "空はまだ少し暗く、", "遠くの山の上に、",
    ]
    assert on_idle == ["薄い雲がゆっくりと流れていました。"]


@pytest.mark.parametrize("mode", ["natural", "responsive"])
def test_the_idle_window_releases_cjk_text_up_to_its_last_mark(mode):
    """An idle flush keeps the unfinished clause, as it keeps an unfinished word."""
    segmenter = _live(mode)

    assert segmenter.append("はい、窓を開け") == []
    assert [segment.text for segment in segmenter.flush(idle=True)] == ["はい、"]
    assert segmenter.append("ると、冷たい空気が入ってきました。") == []

    assert [segment.text for segment in segmenter.flush(idle=True)] == ["窓を開けると、冷たい空気が入ってきました。"]
    assert segmenter.flush() == []


@pytest.mark.parametrize("mode", list(LIVE_CAPS))
@pytest.mark.parametrize("text", [(JA_1 + JA_2) * 8, (ZH_1 + ZH_2) * 12], ids=["ja", "zh"])
def test_a_committed_cjk_text_is_cut_at_its_marks_not_mid_word(mode, text):
    """No segment ends at the cap in the middle of a word."""
    segmenter = _live(mode)

    texts = [segment.text for segment in segmenter.append(text) + segmenter.flush()]

    assert "".join(texts) == text
    assert all(text.endswith(WIDE_MARKS) for text in texts), texts


@pytest.mark.parametrize("mode", list(LIVE_CAPS))
def test_a_cjk_sentence_over_the_cap_is_cut_at_a_clause_mark(mode):
    """A sentence longer than max_chars is cut at its last 、 before the cap."""
    sentence = "今朝は早く起きて窓を開けると、" * 20 + "終わりました。"
    segmenter = LiveTextSegmenter(mode=mode, max_chars=100, max_words=80)

    texts = [segment.text for segment in segmenter.append(sentence) + segmenter.flush()]

    assert "".join(texts) == sentence
    assert all(len(text) <= 100 and text.endswith(("、", "。")) for text in texts), texts


@pytest.mark.parametrize(
    "text, expected",
    [
        ((JA_1 + JA_2) * 3, [JA_1, JA_2] * 3),
        ((ZH_1 + ZH_2) * 3, [ZH_1, ZH_2] * 3),
        (" ".join(HI_SENTENCES * 3), HI_SENTENCES * 3),
        ("धर्म की जय हो॥ सब सुखी हों॥", ["धर्म की जय हो॥", "सब सुखी हों॥"]),
        ("本当？うそ！「はい。」そうです。", ["本当？", "うそ！", "「はい。」", "そうです。"]),
    ],
    ids=["ja", "zh", "hi", "hi-double-danda", "ja-marks-and-brackets"],
)
def test_split_sentences_ends_cjk_and_hindi_sentences_at_their_marks(text, expected):
    """The Reader stores (and highlights) each sentence, at its SENTENCE_MAX_CHARS."""
    assert [text[start:end] for start, end in split_sentences(text, 400)] == expected


def test_split_sentences_cuts_a_long_cjk_sentence_at_a_clause_mark():
    """A sentence over max_chars is cut at a clause mark, not mid-word."""
    text = "今朝は早く起きて窓を開けると、" * 30 + "終わりました。"
    pieces = [text[start:end] for start, end in split_sentences(text, 400)]

    assert "".join(pieces) == text
    assert all(len(piece) <= 400 and piece.endswith(("、", "。")) for piece in pieces), [len(p) for p in pieces]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "Pi is 3.14. Really?! Yes… e.g. this, 1,000 times; Mr. Smith: done.",
            ["Pi is 3.14.", "Really?!", "Yes… e.g. this, 1,000 times; Mr. Smith: done."],
        ),
        (
            "“Fear not, Zacharias.” And the angel said more. 2. Mix well.",
            ["“Fear not, Zacharias.”", "And the angel said more.", "2. Mix well."],
        ),
        ("¿Dónde está? ¡Aquí! Él dijo: «ya voy». Fin.", ["¿Dónde está?", "¡Aquí!", "Él dijo: «ya voy».", "Fin."]),
    ],
    ids=["en", "en-quotes-and-list", "es"],
)
def test_text_without_the_new_marks_is_cut_as_before(text, expected):
    """A guard: the ASCII rules, which need a space after the mark, are unchanged."""
    assert [text[start:end] for start, end in split_sentences(text, 400)] == expected
