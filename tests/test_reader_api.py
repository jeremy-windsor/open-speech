from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage as storage_module
from src.config import Settings
from src.main import app
from src.reader.renderer import ReaderRenderer
from tests.test_reader_sources import _epub

PASTED = "# Notes\n\nFirst paragraph here.\n\n## Part two\n\nSecond paragraph. With two sentences."


@pytest.fixture
def client(tmp_path, monkeypatch):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()
    renderer = ReaderRenderer(
        store=main_module.reader_store,
        synthesize_fn=lambda **_kwargs: np.ones(2400, dtype=np.float32),
        encode_fn=lambda audio, sample_rate: b"ID3" + bytes(64),
        pronounce_fn=lambda text: text,
        busy_fn=lambda: False,
        signer=main_module.reader_signer,
        cache_dir=str(tmp_path / "reader"),
    )
    monkeypatch.setattr(main_module, "reader_renderer", renderer)
    yield TestClient(app)
    renderer.stop()


def _paste(client: TestClient, text: str = PASTED, **extra):
    return client.post("/v1/reader/documents", json={"type": "text", "text": text, **extra})


def _render_ready(client: TestClient, document_id: str, from_block: int = 0) -> list[dict]:
    query = {"from_block": from_block, "model": "kokoro", "voice": "af_heart"}
    started = client.post(f"/v1/reader/documents/{document_id}/render", json=query)
    assert started.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        parts = client.get(f"/v1/reader/documents/{document_id}/parts", params=query).json()["parts"]
        if all(part["status"] == "ready" for part in parts):
            return parts
        time.sleep(0.02)
    raise AssertionError(f"parts never became ready: {parts}")


def test_pasted_text_import_dedupes_and_lists(client):
    created = _paste(client, title="My notes")
    assert created.status_code == 201
    body = created.json()
    assert body["title"] == "My notes"
    assert body["source_type"] == "markdown"
    assert [chapter["title"] for chapter in body["chapters"]] == ["Notes", "Part two"]
    assert "blocks" not in body

    again = _paste(client, title="Another title")
    assert again.status_code == 200
    assert again.json()["id"] == body["id"]

    listed = client.get("/v1/reader/documents").json()["documents"]
    assert [document["id"] for document in listed] == [body["id"]]

    full = client.get(f"/v1/reader/documents/{body['id']}").json()
    assert full["blocks"][-1]["text"] == "Second paragraph. With two sentences."
    assert len(full["blocks"][-1]["sentences"]) == 2


def test_epub_upload(client):
    data = _epub([("Chapter One", "<p>Hello from the book.</p>")])
    response = client.post(
        "/v1/reader/documents",
        files={"file": ("tiny.epub", data, "application/epub+zip")},
    )
    assert response.status_code == 201
    assert response.json()["title"] == "Tiny Book"
    assert response.json()["author"] == "A. Writer"
    assert response.json()["source_type"] == "epub"


def test_import_errors(client, monkeypatch):
    corrupt = bytearray(_epub([("Chapter One", "<p>" + "Hello from the book. " * 50 + "</p>")]))
    corrupt[200:260] = b"\x00" * 60
    assert client.post(
        "/v1/reader/documents", files={"file": ("bad.epub", bytes(corrupt), "application/epub+zip")}
    ).status_code == 422
    assert client.post("/v1/reader/documents", files={"file": ("song.mp3", b"ID3", "audio/mpeg")}).status_code == 422
    assert client.post("/v1/reader/documents", files={"title": (None, "no file")}).status_code == 422
    assert _paste(client, text="   ").status_code == 422
    assert client.post("/v1/reader/documents", json={"type": "url", "url": "https://example.com"}).status_code == 422
    assert client.post("/v1/reader/documents", content=b"hello", headers={"content-type": "text/plain"}).status_code == 415

    monkeypatch.setattr(main_module.settings, "reader_max_document_chars", 10)
    too_long = _paste(client)
    assert too_long.status_code == 413
    assert "limit" in too_long.json()["error"]["message"]


def test_progress_and_delete(client):
    document_id = _paste(client).json()["id"]
    saved = client.put(
        f"/v1/reader/documents/{document_id}/progress",
        json={"block_index": 2, "offset_ms": 1500, "model": "kokoro", "voice": "af_heart", "speed": 1.25},
    )
    assert saved.status_code == 200
    assert saved.json()["progress"]["block_index"] == 2

    listed = client.get("/v1/reader/documents").json()["documents"][0]
    assert listed["progress"]["speed"] == 1.25
    assert client.put(f"/v1/reader/documents/{document_id}/progress", json={"block_index": -1}).status_code == 422
    assert client.put(f"/v1/reader/documents/{document_id}/progress", json={"block_index": 99}).status_code == 422

    parts = _render_ready(client, document_id)
    part_path = main_module.reader_store.get_part(parts[0]["id"])["path"]

    assert client.delete(f"/v1/reader/documents/{document_id}").status_code == 204
    assert client.get(f"/v1/reader/documents/{document_id}").status_code == 404
    assert client.delete(f"/v1/reader/documents/{document_id}").status_code == 404
    assert client.put(f"/v1/reader/documents/{document_id}/progress", json={"block_index": 0}).status_code == 404
    assert not (main_module.reader_renderer.cache_dir / document_id).exists()
    assert main_module.reader_store.get_part(parts[0]["id"]) is None
    assert not Path(part_path).exists()


def test_render_parts_and_signed_media(client):
    document_id = _paste(client).json()["id"]

    assert client.post(
        f"/v1/reader/documents/{document_id}/render", json={"from_block": 99, "model": "kokoro", "voice": "af_heart"}
    ).status_code == 422
    assert client.post(
        "/v1/reader/documents/missing/render", json={"from_block": 0, "model": "kokoro", "voice": "af_heart"}
    ).status_code == 404

    parts = _render_ready(client, document_id)
    assert (parts[0]["first_block"], parts[0]["last_block"]) == (0, 1)  # heading plus its paragraph

    url = parts[0]["url"]
    audio = client.get(url)
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/mpeg"
    assert audio.content.startswith(b"ID3")

    ranged = client.get(url, headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert len(ranged.content) == 10

    path, query = url.split("?")
    params = dict(item.split("=") for item in query.split("&"))
    flipped = ("1" if params["sig"][0] == "0" else "0") + params["sig"][1:]
    assert client.get(f"{path}?exp={params['exp']}&sig={flipped}").status_code == 403
    assert client.get(f"{path}?exp={int(params['exp']) + 1}&sig={params['sig']}").status_code == 403
    assert client.get(f"{path}?exp={int(time.time()) - 5}&sig={'0' * 32}").status_code == 403
    other = path.replace(parts[0]["id"], parts[1]["id"])
    assert client.get(f"{other}?{query}").status_code == 403
    assert client.get(f"{path}?exp={params['exp']}&sig={'é' * 8}").status_code == 422


def test_missing_audio_file_is_rendered_again(client):
    document_id = _paste(client).json()["id"]
    part = _render_ready(client, document_id)[0]
    Path(main_module.reader_store.get_part(part["id"])["path"]).unlink()

    assert client.get(part["url"]).status_code == 404
    again = _render_ready(client, document_id)[0]
    assert again["id"] == part["id"]
    assert client.get(again["url"]).status_code == 200


def test_media_links_skip_api_key_but_library_needs_it(client):
    document_id = _paste(client).json()["id"]
    url = _render_ready(client, document_id)[0]["url"]
    secured = Settings(os_api_key="secret123", stt_model="test-model", stt_device="cpu")

    with patch("src.middleware.settings", secured):
        assert client.get("/v1/reader/documents").status_code == 401
        assert client.get("/v1/reader/documents", headers={"Authorization": "Bearer secret123"}).status_code == 200
        assert client.get(url).status_code == 200
        assert client.get(url.split("?")[0]).status_code == 422
