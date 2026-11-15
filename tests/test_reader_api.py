from __future__ import annotations

import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage as storage_module
from src.config import Settings
from src.main import app
from src.reader.renderer import ReaderRenderer
from src.routes import reader as reader_routes
from tests.test_reader_sources import _deep_contents_epub, _epub, _rewrite_member

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


def test_a_japanese_document_stored_before_the_cjk_sentence_fix_keeps_its_spans(client):
    """A new Japanese import is stored one span per sentence, and a document
    already in the library, stored with the one span per paragraph the old splitter wrote, keeps its id and spans when
    the same text is imported again (no migration of sentences_json, so its rendered marks still match)."""
    from src.reader import store as reader_store_module

    sentence = "今日は朝から雨が降っていたので、私は家でゆっくりと本を読みながら温かいお茶を飲んで過ごしました。"
    fresh = _paste(client, text=sentence * 6, title="new")
    assert fresh.status_code == 201
    fresh_blocks = client.get(f"/v1/reader/documents/{fresh.json()['id']}").json()["blocks"]
    assert [len(block["sentences"]) for block in fresh_blocks] == [6]

    old_text = sentence * 5
    stored = _paste(client, text=old_text, title="old")
    assert stored.status_code == 201
    old_id = stored.json()["id"]
    with reader_store_module._writing() as db:   # what the splitter before the fix stored
        db.execute(
            "UPDATE reader_blocks SET sentences_json = ? WHERE document_id = ?", (f"[[0, {len(old_text)}]]", old_id)
        )

    again = _paste(client, text=old_text, title="old again")
    assert again.status_code == 200
    assert again.json()["id"] == old_id
    blocks = client.get(f"/v1/reader/documents/{old_id}").json()["blocks"]
    assert [block["sentences"] for block in blocks] == [[[0, len(old_text)]]]


def test_import_reads_the_saved_document_back_off_the_event_loop(client, monkeypatch):
    # The renderer thread can hold the Reader's store while SQLite makes it wait for a write
    # lock; the import must not wait for it on the event loop.
    import asyncio

    get_document = main_module.reader_store.get_document
    on_event_loop: list[bool] = []

    def get_document_noting_the_loop(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_event_loop.append(True)
        except RuntimeError:
            on_event_loop.append(False)
        return get_document(*args, **kwargs)

    monkeypatch.setattr(main_module.reader_store, "get_document", get_document_noting_the_loop)
    assert _paste(client).status_code == 201
    assert on_event_loop == [False]


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


@pytest.mark.parametrize("kind", ["nav", "ncx"])
def test_epub_upload_with_contents_nested_past_the_recursion_limit_imports(client, kind):
    # The upload answered 500 (an unhandled RecursionError from the contents
    # walk). Server errors come back as responses here, because the default client re-raises them.
    server = TestClient(app, raise_server_exceptions=False)
    response = server.post(
        "/v1/reader/documents", files={"file": ("deep.epub", _deep_contents_epub(kind), "application/epub+zip")}
    )
    assert response.status_code == 201, (kind, response.status_code, response.text)
    assert [chapter["title"] for chapter in response.json()["chapters"]] == ["Level 0"]


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

    monkeypatch.setattr(main_module.settings, "reader_max_upload_mb", 1)
    huge = b'{"type": "text", "text": "' + b"x" * (3 * 1024 * 1024) + b'"}'
    assert client.post(
        "/v1/reader/documents", content=huge, headers={"content-type": "application/json"}
    ).status_code == 413
    assert client.post(
        "/v1/reader/documents", content=b"{not json", headers={"content-type": "application/json"}
    ).status_code == 400

    monkeypatch.setattr(main_module.settings, "reader_max_document_chars", 10)
    too_long = _paste(client)
    assert too_long.status_code == 413
    assert "limit" in too_long.json()["error"]["message"]


def test_a_new_document_opens_with_no_saved_progress(client):
    # Ported from the old web UI's flow tests at the swap: a document the Reader has never played opens at
    # its start; progress stays null until the first PUT.
    created = _paste(client, text="A long first paragraph.\n\nThe second paragraph.\n\nThe third paragraph.")
    assert created.status_code == 201
    document = client.get(f"/v1/reader/documents/{created.json()['id']}")

    assert document.status_code == 200
    assert document.json()["progress"] is None and len(document.json()["blocks"]) == 3
    assert client.get("/v1/reader/documents").json()["documents"][0]["progress"] is None


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


def test_progress_offset_is_bounded_by_what_sqlite_stores(client):
    document_id = _paste(client).json()["id"]
    progress = f"/v1/reader/documents/{document_id}/progress"

    assert client.put(progress, json={"block_index": 0, "offset_ms": 2**63 - 1}).status_code == 200
    assert client.put(progress, json={"block_index": 0, "offset_ms": 2**63}).status_code == 422


def test_progress_can_mark_a_document_listened_to_the_end(client):
    # The Reader saves the position after the last block when playback finishes; the library shows it as 100%.
    document_id = _paste(client).json()["id"]
    blocks = client.get("/v1/reader/documents").json()["documents"][0]["block_count"]
    progress = f"/v1/reader/documents/{document_id}/progress"

    finished = client.put(progress, json={"block_index": blocks, "model": "kokoro", "voice": "af_heart"})
    assert finished.status_code == 200
    assert client.get("/v1/reader/documents").json()["documents"][0]["progress"]["block_index"] == blocks
    assert client.put(progress, json={"block_index": blocks + 1}).status_code == 422


def test_import_with_an_unencodable_title_is_rejected(client):
    # Like a lone surrogate in the text, one in the title cannot be stored.
    body = b'{"type": "text", "text": "One paragraph.", "title": "\\ud800"}'

    response = client.post("/v1/reader/documents", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422
    assert client.get("/v1/reader/documents").json()["documents"] == []


def test_media_suffix_range_longer_than_the_part_returns_the_whole_part(client):
    # RFC 9110 14.1.2: a suffix longer than the representation selects all of it.
    document_id = _paste(client).json()["id"]
    url = _render_ready(client, document_id)[0]["url"]
    whole = client.get(url).content

    longer = client.get(url, headers={"Range": f"bytes=-{len(whole) + 100}"})
    tail = client.get(url, headers={"Range": "bytes=-10"})

    assert longer.status_code == 206 and longer.content == whole
    assert longer.headers["content-range"] == f"bytes 0-{len(whole) - 1}/{len(whole)}"
    assert tail.status_code == 206 and tail.content == whole[-10:]


def test_app_renderer_splits_sentences_by_the_routers_model_limit(monkeypatch):
    # Reader parts for Chatterbox (350 characters) fail unless the app's renderer asks the router.
    limits = {"chatterbox/turbo": 350}
    monkeypatch.setattr(main_module.tts_router, "max_input_chars_for", limits.get)

    assert main_module.reader_renderer.max_input_chars_fn is not None
    assert main_module.reader_renderer.max_input_chars_fn("chatterbox/turbo") == 350
    assert main_module.reader_renderer.max_input_chars_fn("kokoro") is None


def test_media_ignores_ranges_it_does_not_serve(client):
    # RFC 9110 14.2: a range unit the server does not understand MUST be ignored (Starlette answered 400),
    # and a server MAY ignore any Range: Starlette sent several ranges as a 206 labelled audio/mpeg.
    document_id = _paste(client).json()["id"]
    url = _render_ready(client, document_id)[0]["url"]
    whole = client.get(url).content

    for range_ in ("items=0-5", "bytes=0-1,5-6"):
        response = client.get(url, headers={"Range": range_})

        assert response.status_code == 200, range_
        assert response.headers["content-type"] == "audio/mpeg"
        assert "content-range" not in response.headers
        assert response.content == whole


def test_media_supports_head(client):
    # RFC 9110 9.1: "All general-purpose servers MUST support the methods GET and HEAD."
    document_id = _paste(client).json()["id"]
    url = _render_ready(client, document_id)[0]["url"]
    whole = client.get(url).content
    secured = Settings(os_api_key="secret123", stt_model="test-model", stt_device="cpu")

    with patch("src.middleware.settings", secured):
        response = client.head(url)

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-length"] == str(len(whole))
    assert response.content == b""


def test_media_head_shares_no_operation_id_with_get():
    # OpenAPI 3.1: operationId "MUST be unique among all operations described in the API". FastAPI
    # names a GET+HEAD route's operations after whichever method its set yields first.
    routes = FastAPI()
    routes.include_router(
        reader_routes.create_router(get_settings=None, get_store=None, get_renderer=None, get_signer=None)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schema = routes.openapi()

    assert not [str(warning.message) for warning in caught if "Duplicate Operation ID" in str(warning.message)]
    operations = schema["paths"]["/v1/reader/media/{name}"]
    assert {method: operation["operationId"] for method, operation in operations.items()} == {
        "get": "media_v1_reader_media__name__get"
    }


def test_media_part_pruned_as_it_is_requested_is_not_found(client, monkeypatch):
    # The renderer forgets pruned parts under the store lock and unlinks them after it.
    document_id = _paste(client).json()["id"]
    part = _render_ready(client, document_id)[0]
    store = main_module.reader_store
    path = Path(store.get_part(part["id"])["path"])
    touch = store.touch_part

    def pruned_first(part_id: str) -> bool:
        store.forget_part(part_id)
        path.unlink()
        return touch(part_id)

    monkeypatch.setattr(store, "touch_part", pruned_first)

    assert client.get(part["url"]).status_code == 404


def test_media_part_removed_after_the_route_found_it_is_not_found(client, monkeypatch):
    # The file can still vanish between the route's check and the response's stat (a document
    # deleted meanwhile); the player gets 404, not a 500 from Starlette.
    document_id = _paste(client).json()["id"]
    part = _render_ready(client, document_id)[0]

    class _RemovedAfterTheCheck(Path):
        def is_file(self) -> bool:
            found = super().is_file()
            self.unlink()
            return found

    monkeypatch.setattr(reader_routes, "Path", _RemovedAfterTheCheck)
    response = client.get(part["url"])

    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Audio part not found"


def test_parts_poll_during_delete_does_not_keep_a_deleted_document(client, monkeypatch):
    # The player polls /parts every half second; a poll just before the rows go must not cache the outline again.
    document_id = _paste(client).json()["id"]
    store = main_module.reader_store
    delete = store.delete_document
    query = {"from_block": 0, "model": "kokoro", "voice": "af_heart"}

    def delete_after_a_poll(doc_id: str):
        assert main_module.reader_renderer.status(doc_id, 0, "kokoro", "af_heart") is not None
        return delete(doc_id)

    monkeypatch.setattr(store, "delete_document", delete_after_a_poll)
    assert client.delete(f"/v1/reader/documents/{document_id}").status_code == 204

    assert client.get(f"/v1/reader/documents/{document_id}/parts", params=query).status_code == 404
    assert client.post(f"/v1/reader/documents/{document_id}/render", json=query).status_code == 404


@pytest.mark.parametrize("label", ["base64", "utf-7"])
def test_upload_declaring_a_charset_that_is_no_web_encoding_imports(client, label):
    # base64.decode raised LookupError and UTF-7 made a lone surrogate SQLite could not store: both were 500s.
    page = f'<html><head><meta charset="{label}"></head><body><p>Hello +2AA- there.</p></body></html>'.encode("ascii")
    book = _rewrite_member(
        _epub([("Chapter One", "<p>Hello +2AA- there.</p>")]),
        "OEBPS/text/ch0.xhtml",
        lambda payload: f'<?xml version="1.0" encoding="{label}"?>'.encode("ascii") + payload,
    )

    html = client.post("/v1/reader/documents", files={"file": ("page.html", page, "text/html")})
    epub = client.post("/v1/reader/documents", files={"file": ("book.epub", book, "application/epub+zip")})

    assert html.status_code == 201, html.text
    assert epub.status_code == 201, epub.text
    blocks = client.get(f"/v1/reader/documents/{html.json()['id']}").json()["blocks"]
    assert [block["text"] for block in blocks] == ["Hello +2AA- there."]


@pytest.mark.parametrize(
    ("filename", "data", "status"),
    [
        ("book.txt", b"A sentence for the reader. " * 60_000, 201),   # past Starlette's 1 MB spool: a temp file
        ("note.txt", b"Hello reader.", 201),
        ("song.mp3", b"ID3" + bytes(64), 422),
    ],
    ids=["temp-file", "in-memory", "refused"],
)
def test_import_closes_the_uploaded_file_it_parsed(client, monkeypatch, filename, data, status):
    # Starlette leaves closing a form to whoever parsed it; an open temp file waited for the GC.
    from starlette import formparsers

    forms = []
    parse = formparsers.MultiPartParser.parse

    async def recording_parse(self):
        forms.append(await parse(self))
        return forms[-1]

    monkeypatch.setattr(formparsers.MultiPartParser, "parse", recording_parse)
    response = client.post("/v1/reader/documents", files={"file": (filename, data, "text/plain")})

    assert response.status_code == status, response.text
    uploads = [value for form in forms for _key, value in form.multi_items() if not isinstance(value, str)]
    assert [upload.file.closed for upload in uploads] == [True]


def test_markdown_import_does_not_stall_the_event_loop(client):
    # A regex holds the GIL while it matches, so a quadratic one stalled every request and stream
    # although the parse runs in a worker thread.
    import asyncio

    import httpx

    async def import_while_ticking() -> tuple[int, float]:
        gaps: list[float] = []
        done = asyncio.Event()

        async def tick() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        ticker = asyncio.create_task(tick())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as async_client:
            response = await async_client.post(
                "/v1/reader/documents", json={"type": "text", "text": "# a" + " " * 12_000 + "b"}
            )
        done.set()
        await ticker
        return response.status_code, max(gaps)

    status, longest_gap = asyncio.run(import_while_ticking())

    assert status == 201
    assert longest_gap < 0.5


class _KokoroRouter:
    """Two Kokoro voices; records the provider voice each synthesis used."""

    VOICES = ("af_heart", "af_bella")

    def __init__(self):
        self.voices: list[str] = []

    def get_backend(self, model):
        if model != "kokoro":
            raise ValueError("Unknown model")
        from types import SimpleNamespace

        return SimpleNamespace(name="kokoro", capabilities={"speed_control": True})

    def get_capabilities(self, model):
        return self.get_backend(model).capabilities

    def validate_voice(self, model, voice):
        if voice not in self.VOICES:
            raise ValueError("Unknown Kokoro voice")

    def list_voices(self, model):
        return list(self.VOICES)

    def sample_rate_for(self, model):
        return 24000

    def max_input_chars_for(self, model):
        return None

    def synthesize(self, **kwargs):
        self.voices.append(kwargs["voice"])
        yield np.full(2400, 0.2, dtype=np.float32)


@pytest.fixture
def named_voice_client(tmp_path, monkeypatch):
    from src.voice_library import VoiceLibraryManager

    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()
    monkeypatch.setattr(main_module.settings, "tts_cache_enabled", False)
    router = _KokoroRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices"))
    renderer = ReaderRenderer(
        store=main_module.reader_store,
        synthesize_fn=main_module._synthesize_array,   # resolves voice:<id> as it speaks
        voice_key_fn=main_module.reader_renderer.voice_key_fn,   # the app's own wiring
        encode_fn=lambda audio, sample_rate: b"ID3" + bytes(64),
        pronounce_fn=lambda text: text,
        busy_fn=lambda: False,
        signer=main_module.reader_signer,
        cache_dir=str(tmp_path / "reader"),
        lookahead_parts=1,
    )
    monkeypatch.setattr(main_module, "reader_renderer", renderer)
    yield TestClient(app), router
    renderer.stop()


def _render_named(client: TestClient, document_id: str, voice: str) -> list[dict]:
    query = {"from_block": 0, "model": "kokoro", "voice": voice}
    assert client.post(f"/v1/reader/documents/{document_id}/render", json=query).status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        parts = client.get(f"/v1/reader/documents/{document_id}/parts", params=query).json()["parts"]
        if all(part["status"] in {"ready", "error"} for part in parts):
            return parts
        time.sleep(0.02)
    raise AssertionError(f"parts never finished: {parts}")


def test_reader_parts_follow_a_named_voice_and_end_with_it(named_voice_client):
    # The Reader keyed parts on "voice:<id>", so a new version kept playing the old voice
    # wherever parts were cached, and a deleted named voice kept playing at all.
    client, router = named_voice_client
    identity = client.post("/api/voices/identities", json={"name": "Narrator"}).json()
    added = client.post(
        f"/api/voices/identities/{identity['id']}/realizations", json={"model": "kokoro", "voice": "af_heart"}
    )
    assert added.status_code == 201, added.text
    document_id = _paste(client, "Hello there.").json()["id"]
    query = {"from_block": 0, "model": "kokoro", "voice": identity["voice"]}
    before = _render_named(client, document_id, identity["voice"])[0]
    assert (before["status"], router.voices) == ("ready", ["af_heart"])

    updated = client.put(
        f"/api/voices/identities/{identity['id']}/realizations/kokoro", json={"model": "kokoro", "voice": "af_bella"}
    )
    assert updated.status_code == 200, updated.text
    stale = client.get(f"/v1/reader/documents/{document_id}/parts", params=query).json()["parts"][0]
    assert stale["status"] == "pending" and stale["id"] != before["id"]
    after = _render_named(client, document_id, identity["voice"])[0]
    assert (after["status"], router.voices) == ("ready", ["af_heart", "af_bella"])

    assert client.delete(f"/api/voices/identities/{identity['id']}").status_code == 200
    gone = client.get(f"/v1/reader/documents/{document_id}/parts", params=query)
    assert (gone.status_code, gone.json()["error"]["message"]) == (404, "Named voice not found")
    assert client.post(f"/v1/reader/documents/{document_id}/render", json=query).status_code == 404


def test_reader_parts_of_a_named_voice_follow_its_recording(named_voice_client, tmp_path):
    import io
    import wave

    def recording(level: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(level.to_bytes(2, "little") * 2400)
        return buffer.getvalue()

    client, _router = named_voice_client
    main_module.voice_library.save("narrator", recording(1))
    identity = client.post("/api/voices/identities", json={"name": "Clone"}).json()
    from src.voice_identities import VoiceIdentityManager

    VoiceIdentityManager().add_realization(identity["id"], model="kokoro", voice="af_heart", reference_audio_id="narrator")
    document_id = _paste(client, "Hello again.").json()["id"]
    query = {"from_block": 0, "model": "kokoro", "voice": identity["voice"]}
    first = client.get(f"/v1/reader/documents/{document_id}/parts", params=query).json()["parts"][0]

    main_module.voice_library.save("narrator", recording(2))   # "Replace the existing recording"
    replaced = client.get(f"/v1/reader/documents/{document_id}/parts", params=query).json()["parts"][0]
    assert replaced["id"] != first["id"]

    main_module.voice_library.delete("narrator")
    gone = client.get(f"/v1/reader/documents/{document_id}/parts", params=query)
    assert (gone.status_code, gone.json()["error"]["message"]) == (404, "Voice library entry 'narrator' not found")


def test_reader_media_is_spoken_with_the_current_pronunciation_dictionary(tmp_path, monkeypatch):
    # After TTS_PRONUNCIATION_DICT was edited and the server restarted,
    # GET /v1/reader/media kept serving the part cached with the old pronunciation.
    import json
    from urllib.parse import urlsplit

    from src.pronunciation.dictionary import PronunciationDictionary

    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()
    app_pronounce = main_module.reader_renderer.pronounce_fn   # the app's own wiring
    app_speech_key = main_module.reader_renderer.speech_key_fn
    dictionary = tmp_path / "pronunciations.json"
    spoken: list[str] = []
    running: list[ReaderRenderer] = []

    def speak(*, text, **_kwargs):
        spoken.append(text)
        return np.ones(2400, dtype=np.float32)

    def encode(audio, sample_rate) -> bytes:
        # The "MP3" says what was spoken into it.
        data = ("MP3|" + "|".join(spoken)).encode("utf-8")
        spoken.clear()
        return data

    def restart(entries: dict[str, str]) -> None:
        """A server start: the dictionary file is read again and a new renderer runs on the same cache."""
        if running:
            running.pop().stop()
        dictionary.write_text(json.dumps(entries), encoding="utf-8")
        monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(dictionary)))
        renderer = ReaderRenderer(
            store=main_module.reader_store,
            synthesize_fn=speak,
            encode_fn=encode,
            pronounce_fn=app_pronounce,
            speech_key_fn=app_speech_key,
            busy_fn=lambda: False,
            signer=main_module.reader_signer,
            cache_dir=str(tmp_path / "reader"),
            lookahead_parts=1,
        )
        running.append(renderer)
        monkeypatch.setattr(main_module, "reader_renderer", renderer)

    client = TestClient(app)

    def served(document_id: str) -> str:
        url = urlsplit(_render_ready(client, document_id)[0]["url"])
        response = client.get(f"{url.path}?{url.query}")
        assert response.status_code == 200
        return response.content.decode("utf-8")

    try:
        restart({"SQL": "S Q L"})
        document_id = _paste(client, "I like SQL.").json()["id"]
        assert served(document_id) == "MP3|I like S Q L."

        restart({"SQL": "sequel"})
        assert served(document_id) == "MP3|I like sequel.", (
            "GET /v1/reader/media serves the part spoken with the dictionary the server no longer has"
        )
    finally:
        for renderer in running:
            renderer.stop()


def test_reader_media_follows_tts_normalize_output_after_a_restart(tmp_path, monkeypatch):
    # After TTS_NORMALIZE_OUTPUT (or TTS_TRIM_SILENCE) was turned off and the
    # server restarted, GET /v1/reader/media kept serving the part normalized before, next to new parts that
    # were not. The renderer gets the app's own wiring: main's synthesis (over a fake tts_router) and its
    # speech_key_fn, which read src.main.settings.
    from urllib.parse import urlsplit

    from src.pronunciation.dictionary import PronunciationDictionary

    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()
    monkeypatch.setattr(main_module, "tts_router", _KokoroRouter())   # speaks every text at 0.2
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(None))
    monkeypatch.setattr(main_module.settings, "tts_trim_silence", True)   # the defaults
    monkeypatch.setattr(main_module.settings, "tts_normalize_output", True)
    app_renderer = main_module.reader_renderer
    running: list[ReaderRenderer] = []

    def peak(audio, sample_rate) -> bytes:
        # The "MP3" says how loud its audio was made.
        return f"MP3|peak={float(np.max(np.abs(audio))):.2f}".encode("utf-8")

    def restart() -> None:
        """A server start: a new renderer with the app's hooks runs on the same store and cache."""
        if running:
            running.pop().stop()
        renderer = ReaderRenderer(
            store=main_module.reader_store,
            synthesize_fn=app_renderer.synthesize_fn,
            encode_fn=peak,
            pronounce_fn=app_renderer.pronounce_fn,
            speech_key_fn=app_renderer.speech_key_fn,
            busy_fn=lambda: False,
            signer=main_module.reader_signer,
            cache_dir=str(tmp_path / "reader"),
            lookahead_parts=1,
        )
        running.append(renderer)
        monkeypatch.setattr(main_module, "reader_renderer", renderer)

    client = TestClient(app)

    def served(document_id: str) -> str:
        url = urlsplit(_render_ready(client, document_id)[0]["url"])
        response = client.get(f"{url.path}?{url.query}")
        assert response.status_code == 200
        return response.content.decode("utf-8")

    try:
        restart()
        document_id = _paste(client, "Hello there.").json()["id"]
        assert served(document_id) == "MP3|peak=0.95"   # normalized, as on the defaults

        monkeypatch.setattr(main_module.settings, "tts_normalize_output", False)
        restart()
        assert served(document_id) == "MP3|peak=0.20", (
            "GET /v1/reader/media serves the part normalized before TTS_NORMALIZE_OUTPUT was turned off"
        )
    finally:
        for renderer in running:
            renderer.stop()
