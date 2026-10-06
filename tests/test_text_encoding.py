"""UTF-8 application assets must not depend on the host's default encoding."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.pronunciation.dictionary import PronunciationDictionary


@pytest.fixture
def windows_default_encoding(monkeypatch):
    read_text = Path.read_text

    def read_using_windows_default(path, encoding=None, errors=None):
        return read_text(path, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_using_windows_default)


def test_web_page_is_utf8_on_non_utf8_hosts(windows_default_encoding):
    response = TestClient(app, raise_server_exceptions=False).get("/web")

    assert response.status_code == 200
    assert "Open Speech" in response.text
    assert "utf-8" in response.headers["content-type"]


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_pronunciation_dictionary_preserves_unicode(tmp_path, suffix, windows_default_encoding):
    path = tmp_path / f"pronunciation{suffix}"
    entries = {"bonjour": "你好"}
    text = json.dumps(entries, ensure_ascii=False) if suffix == ".json" else "bonjour: 你好\n"
    path.write_text(text, encoding="utf-8")

    assert PronunciationDictionary(str(path)).apply("bonjour") == "你好"
