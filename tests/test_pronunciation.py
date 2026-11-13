from __future__ import annotations

import json
import logging
import time

import pytest

from src.pronunciation.dictionary import PronunciationDictionary, parse_ssml


def test_dict_load_json(tmp_path):
    p = tmp_path / "dict.json"
    p.write_text(json.dumps({"AWS": "A W S"}))
    d = PronunciationDictionary(str(p))
    assert d.entries["AWS"] == "A W S"


def test_dict_load_yaml(tmp_path):
    p = tmp_path / "dict.yaml"
    p.write_text("AWS: A W S\n")
    d = PronunciationDictionary(str(p))
    assert d.entries["AWS"] == "A W S"


# A YAML dictionary maps text as written to text to speak, as a JSON one does.
# Read with YAML 1.1 typing, "NO" was the boolean False, "12:30" the integer 750, "007" the integer 7 and an empty
# value null, so those entries were never found and "False", "750" or "None" were replaced instead.
_TEXT_THAT_LOOKS_TYPED_YAML = (
    "NO: Norway\nON: Ontario\n12:30: half past twelve\n007: double oh seven\nNein: no\num:\n"
)
_TEXT_THAT_LOOKS_TYPED = {
    "NO": "Norway", "ON": "Ontario", "12:30": "half past twelve", "007": "double oh seven", "Nein": "no", "um": "",
}


@pytest.mark.parametrize("name", ["dict.json", "dict.yaml", "dict.yml"])
def test_dict_entries_are_the_text_written_in_json_and_yaml(tmp_path, name):
    p = tmp_path / name
    text = json.dumps(_TEXT_THAT_LOOKS_TYPED) if name.endswith(".json") else _TEXT_THAT_LOOKS_TYPED_YAML
    p.write_text(text, encoding="utf-8")
    d = PronunciationDictionary(str(p))

    assert d.entries == _TEXT_THAT_LOOKS_TYPED
    assert list(d.entries) == list(_TEXT_THAT_LOOKS_TYPED)  # the file's order is part of the fingerprint
    assert d.apply("Flights to NO and ON") == "Flights to Norway and Ontario"
    assert d.apply("Meet at 12:30") == "Meet at half past twelve"
    assert d.apply("Agent 007") == "Agent double oh seven"
    assert d.apply("Nein, danke") == "no, danke"
    assert d.apply("um, okay") == ", okay"
    assert d.apply("The flag is False, True or None") == "The flag is False, True or None"
    assert d.apply("Bus 750 to 7") == "Bus 750 to 7"


# A JSON value that isn't a string is the text written too, as in YAML. json.loads
# typed it, so 0.50 was spoken "0.5", 1e3 "1000.0", -0 "0", Infinity "inf", true "True" and null "None". A null value
# has no text, so its entry is left out, with a warning that names it.
_JSON_VALUES_THAT_ARE_NOT_STRINGS = (
    '{"half": 0.50, "kilo": 1e3, "minus zero": -0, "dozen": 12, "forever": Infinity, "yes": true, "nope": false, '
    '"um": null, "AWS": "A W S"}'
)
_JSON_VALUES_AS_WRITTEN = {
    "half": "0.50", "kilo": "1e3", "minus zero": "-0", "dozen": "12", "forever": "Infinity", "yes": "true",
    "nope": "false", "AWS": "A W S",
}


def test_dict_json_values_that_are_not_strings_are_the_text_written(tmp_path, caplog):
    p = tmp_path / "dict.json"
    p.write_text(_JSON_VALUES_THAT_ARE_NOT_STRINGS, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="src.pronunciation.dictionary"):
        d = PronunciationDictionary(str(p))

    assert d.entries == _JSON_VALUES_AS_WRITTEN
    assert list(d.entries) == list(_JSON_VALUES_AS_WRITTEN)  # the file's order is part of the fingerprint
    assert d.apply("half a kilo") == "0.50 a 1e3"
    assert d.apply("yes or nope") == "true or false"
    assert d.apply("um, okay") == "um, okay"  # not "None, okay"
    warnings = [r for r in caplog.records if r.name == "src.pronunciation.dictionary"]
    assert [r.levelno for r in warnings] == [logging.WARNING]
    assert "'um'" in warnings[0].getMessage() and "null" in warnings[0].getMessage()

    # The same entries written as strings give the same fingerprint, so the Reader keeps its part ids for them.
    as_strings = tmp_path / "strings.json"
    as_strings.write_text(json.dumps(_JSON_VALUES_AS_WRITTEN), encoding="utf-8")
    assert d.fingerprint() == PronunciationDictionary(str(as_strings)).fingerprint()


@pytest.mark.parametrize("name", ["dict.json", "dict.yaml"])
def test_dict_an_emoji_escaped_as_a_utf16_pair_is_the_emoji_in_json_and_yaml(tmp_path, name):
    # TTS_PRONUNCIATION_DICT: json.dumps writes an emoji as
    # a UTF-16 escape pair (backslash-u d83d, backslash-u de00), one character (RFC 8259 section 7), and json.loads
    # reads it so. PyYAML read the pair as two lone surrogates, so in a YAML dictionary the entry for the emoji never
    # matched the emoji in a text, and a value with one put half-characters into the spoken text, which UTF-8 can't
    # encode.
    p = tmp_path / name
    p.write_text(json.dumps({"\U0001F600": "smiley", "grin": "big \U0001F642"}) + "\n", encoding="utf-8")
    assert "\\ud83d\\ude00" in p.read_text(encoding="utf-8")
    d = PronunciationDictionary(str(p))

    assert d.entries == {"\U0001F600": "smiley", "grin": "big \U0001F642"}
    assert d.apply("Hi \U0001F600") == "Hi smiley"
    assert d.apply("a grin").encode("utf-8") == "a big \U0001F642".encode("utf-8")


def test_dict_apply_replaces():
    d = PronunciationDictionary()
    d._entries = {"AWS": "A W S"}
    assert d.apply("Use AWS") == "Use A W S"


def test_dict_apply_longest_first():
    d = PronunciationDictionary()
    d._entries = {"AWS": "A W S", "AWS Lambda": "A W S Lambda"}
    assert "A W S Lambda" in d.apply("AWS Lambda")


def test_dict_apply_inserts_values_literally(tmp_path):
    # A value is text to speak, not an re.sub template: a backslash in one entry must not
    # fail every request (re.error "bad escape") or turn into a newline.
    p = tmp_path / "dict.json"
    p.write_text(json.dumps({"winpath": r"C:\Windows", "nl": r"a\nb"}), encoding="utf-8")
    d = PronunciationDictionary(str(p))

    assert d.apply("Hello world") == "Hello world"
    assert d.apply("open winpath") == r"open C:\Windows"
    assert d.apply("nl") == r"a\nb"


def test_dict_missing_path_safe(tmp_path):
    d = PronunciationDictionary(str(tmp_path / "missing.json"))
    assert d.entries == {}


def test_dict_fingerprint_changes_exactly_when_what_is_spoken_can(tmp_path):
    # The Reader keys its cached parts on this, so they are spoken
    # again after the dictionary changes, and keep their ids where there is none.
    p = tmp_path / "dict.json"
    assert PronunciationDictionary().fingerprint() == ""
    assert PronunciationDictionary(str(tmp_path / "missing.json")).fingerprint() == ""

    p.write_text(json.dumps({"SQL": "S Q L"}), encoding="utf-8")
    first = PronunciationDictionary(str(p)).fingerprint()
    assert first and PronunciationDictionary(str(p)).fingerprint() == first
    p.write_text(json.dumps({"SQL": "sequel"}), encoding="utf-8")
    assert PronunciationDictionary(str(p)).fingerprint() not in ("", first)

    # Entries of one length apply in the file's order, so that order is part of it.
    ab_first, bc_first = PronunciationDictionary(), PronunciationDictionary()
    ab_first._entries = {"ab": "x", "bc": "y"}
    bc_first._entries = {"bc": "y", "ab": "x"}
    assert ab_first.apply("abc") != bc_first.apply("abc")
    assert ab_first.fingerprint() != bc_first.fingerprint()


def test_reader_and_conversations_read_the_server_dictionary_when_they_speak(tmp_path, monkeypatch):
    # TTS_PRONUNCIATION_DICT is server-wide. Conversations were built without it, and the
    # Reader kept the dictionary from import time, so swapping src.main's did not reach it.
    from src import main as main_module

    p = tmp_path / "dict.json"
    p.write_text(json.dumps({"Nguyen": "Win"}), encoding="utf-8")
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(p)))

    assert main_module.conversation_manager.pronounce_fn("Hello Nguyen") == "Hello Win"
    assert main_module.reader_renderer.pronounce_fn("Hello Nguyen") == "Hello Win"


def test_ssml_break_to_pause():
    out = parse_ssml('<speak>Hello <break time="500ms"/> world</speak>')
    assert "Hello" in out and "world" in out


def test_ssml_break_in_seconds_matches_milliseconds():
    # W3C SSML 1.1 section 3.2.3: break time is a CSS2 time ("3s" or "250ms").
    milliseconds = parse_ssml('<speak>Hello <break time="1000ms"/> world</speak>')

    assert parse_ssml('<speak>Hello <break time="1s"/> world</speak>') == milliseconds
    assert parse_ssml("<speak>Hello <break time='1s'/> world</speak>") == milliseconds


def test_ssml_break_is_capped_at_ten_seconds():
    assert parse_ssml('<speak>Hi<break time="100000000ms"/></speak>') == "Hi " + "." * 40


def test_ssml_parsing_is_linear_on_unclosed_markup():
    # parse_ssml runs on request text, so its patterns must not backtrack quadratically
    # (CWE-1333). An earlier break pattern took about 2.3 s on the digit run.
    hostile = [
        '<speak><break time="' + "9" * 16000 + '"/></speak>',
        "<" * 16000,
        "<break " + 'a="' * 5000 + ">",
        "<break " + "a" * 16000,
        "<!--" * 4000,
        "<![CDATA[" * 1800,
        "<!-" * 5000,
    ]
    for markup in hostile:
        best = min(_parse_seconds(markup) for _ in range(3))
        assert best < 0.25, f"parse_ssml took {best:.2f} s for {markup[:20]!r}..."


def _parse_seconds(markup: str) -> float:
    start = time.perf_counter()
    parse_ssml(markup)
    return time.perf_counter() - start


@pytest.mark.parametrize(
    "markup",
    [
        'Wait<break strength="medium" time="1s"/>now',
        'Wait<break time="1s"></break>now',
        'Wait<break time = "1s" />now',
    ],
    ids=["time-after-strength", "start-and-end-tag", "spaced-attribute"],
)
def test_ssml_break_time_is_honored_in_any_well_formed_spelling(markup):
    # XML 1.0 section 3.1: attribute order is not significant, and <x></x> is the same
    # empty element as <x/>.
    assert parse_ssml(f"<speak>{markup}</speak>") == "Wait .... now"


def test_ssml_break_without_time_pauses_by_strength():
    # SSML 1.1 section 3.2.3: a break without time takes its strength, "medium" by default.
    medium = parse_ssml("<speak>Hello<break/>world</speak>")

    assert medium == "Hello ... world"
    assert parse_ssml('<speak>Hello<break strength="medium"/>world</speak>') == medium
    assert parse_ssml('<speak>Hello<break strength="x-strong"/>world</speak>') == "Hello ..... world"
    assert parse_ssml('<speak>Hello<break strength="none"/>world</speak>') == "Hello world"


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ("<s>First one</s><s>Second one</s>", "First one Second one"),
        ("<p>First one</p><p>Second one</p>", "First one Second one"),
        ('Hello<voice name="x">world</voice>', "Hello world"),
        ('Very <emphasis>im</emphasis>portant', "Very important"),
    ],
    ids=["sentences", "paragraphs", "voice", "inline-emphasis"],
)
def test_ssml_structure_tags_separate_words_and_inline_tags_do_not(markup, expected):
    # SSML 1.1 section 3.1.8.1: p and s are paragraph and sentence boundaries. Inline markup
    # such as emphasis sits inside the text it marks.
    assert parse_ssml(f"<speak>{markup}</speak>") == expected


@pytest.mark.parametrize(
    ("markup", "expected"),
    [
        ('A <!-- note <break time="10s"/> --> B', "A B"),
        ("A <!-- <s>old</s> -->B", "A B"),
        ("<![CDATA[ a < b & c ]]>", "a < b & c"),
        ("<![CDATA[ <!-- kept --> &lt; ]]>", "<!-- kept --> &lt;"),
    ],
    ids=["comment-around-break", "comment-around-markup", "cdata", "cdata-is-literal"],
)
def test_ssml_comments_are_silent_and_cdata_is_spoken_literally(markup, expected):
    # XML 1.0 sections 2.5 and 2.7: a comment may contain "<" and is not text, so a
    # commented-out break is not a pause. CDATA content is text, taken without parsing.
    assert parse_ssml(f"<speak>{markup}</speak>") == expected


def test_ssml_xml_entities_are_decoded():
    # SSML is XML, so a literal "&" must be written "&amp;".
    assert parse_ssml("<speak>Tom &amp; Jerry don&apos;t say &lt;hi&gt;</speak>") == (
        "Tom & Jerry don't say <hi>"
    )


def test_ssml_emphasis_removed():
    out = parse_ssml("<speak><emphasis>Important</emphasis></speak>")
    assert out == "Important"


def test_ssml_phoneme_keeps_text():
    out = parse_ssml('<phoneme ph="t eh s t">test</phoneme>')
    assert out == "test"


def test_ssml_strips_unknown_tags():
    out = parse_ssml("<speak><foo>bar</foo></speak>")
    assert out == "bar"


def test_ssml_whitespace_collapse():
    out = parse_ssml("<speak> a   b </speak>")
    assert out == "a b"
