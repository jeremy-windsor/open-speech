from __future__ import annotations

import json

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


def test_dict_apply_replaces():
    d = PronunciationDictionary()
    d._entries = {"AWS": "A W S"}
    assert d.apply("Use AWS") == "Use A W S"


def test_dict_apply_longest_first():
    d = PronunciationDictionary()
    d._entries = {"AWS": "A W S", "AWS Lambda": "A W S Lambda"}
    assert "A W S Lambda" in d.apply("AWS Lambda")


def test_dict_missing_path_safe(tmp_path):
    d = PronunciationDictionary(str(tmp_path / "missing.json"))
    assert d.entries == {}


def test_ssml_break_to_pause():
    out = parse_ssml('<speak>Hello <break time="500ms"/> world</speak>')
    assert "Hello" in out and "world" in out


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
