"""Tests for API endpoints (mocked backend)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import threading
import time
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src import router as router_module


@pytest.fixture
def client():
    """Create test client with mocked backend router."""
    mock_backend = MagicMock()
    mock_backend.name = "faster-whisper"
    mock_backend.loaded_models.return_value = []
    mock_backend.is_model_loaded.return_value = False
    mock_backend.transcribe.return_value = {"text": "hello world"}
    mock_backend.translate.return_value = {"text": "hello world"}

    with patch.object(router_module.router, "_default_backend", mock_backend):
        with patch.object(router_module.router, "_backends", {"faster-whisper": mock_backend}):
            yield TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == main_module.get_runtime_version()


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1


def test_get_model(client):
    resp = client.get("/v1/models/some-model")
    assert resp.status_code == 404


class _Worker:
    """Isolated provider whose manifest serves one model."""

    def __init__(self, served: str):
        self.served = served

    def advertises_model(self, model_id):
        return model_id == self.served


@pytest.mark.parametrize(("workers", "model_id", "status_code"), [
    # docs/TTS-BACKENDS.md: "External models appear only if their exact ID is advertised by the
    # provider", as /v1/models and /api/models show them. No worker configured:
    ({}, "chatterbox/regular", 404),
    ({}, "qwen3/0.6b-base", 404),
    ({}, "cosyvoice/2-0.5b", 404),
    # A configured worker whose manifest leaves the model out, and one that serves it:
    ({"qwen3": _Worker("qwen3/0.6b-custom-voice")}, "qwen3/0.6b-base", 404),
    ({"qwen3": _Worker("qwen3/0.6b-custom-voice")}, "qwen3/0.6b-custom-voice", 200),
    # An in-process provider that is not installed (/api/models: provider_missing), and one that is:
    ({}, "piper/en_US-lessac-medium", 404),
    ({"piper": object()}, "piper/en_US-lessac-medium", 200),
    # Registry STT models load on first use, listed or not.
    ({}, "Systran/faster-whisper-small", 200),
])
def test_model_detail_answers_only_for_models_a_client_can_use(client, workers, model_id, status_code):
    tts_router = MagicMock(_backends=workers)
    tts_router.loaded_models.return_value = []

    with patch.object(main_module, "tts_router", tts_router):
        resp = client.get(f"/v1/models/{model_id}")

    assert resp.status_code == status_code, resp.text


def test_transcribe(client):
    audio = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"model": "test-model"},
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


def test_translate(client):
    audio = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/v1/audio/translations",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"model": "test-model"},
    )
    assert resp.status_code == 200
    assert "text" in resp.json()


# Whisper large-v3-turbo answers in the speaker's own
# language even with task=translate (OpenAI's Whisper README), and an English-only checkpoint (`*.en`,
# distil-large-v3) only transcribes (faster-whisper sets no task for it). A translation on such a model is a 400
# naming the models that translate, with nothing decoded; STT_TRANSLATE_MODEL (empty by default) takes the
# translations sent with whisper-1 or no model.
_TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
_CANNOT_TRANSLATE = [
    _TURBO,
    "Systran/faster-whisper-tiny.en",
    "Systran/faster-whisper-small.en",
    "Systran/faster-whisper-medium.en",
    "Systran/faster-distil-whisper-small.en",
    "Systran/faster-distil-whisper-large-v3",
]


def _translate(client, model=None, *, stt_model=_TURBO, translate_model=None):
    """POST a translation; `translate_model=None` leaves STT_TRANSLATE_MODEL at its default (empty)."""
    backend = router_module.router._default_backend
    data = {} if model is None else {"model": model}
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(main_module.settings, "stt_model", stt_model))
        if translate_model is not None:
            stack.enter_context(patch.object(main_module.settings, "stt_translate_model", translate_model))
        resp = client.post(
            "/v1/audio/translations",
            files={"file": ("es.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data=data,
        )
    return resp, [call.args[1] for call in backend.translate.call_args_list]


@pytest.mark.parametrize("model", [None, "whisper-1", *_CANNOT_TRANSLATE])
def test_a_translation_on_a_model_that_cannot_translate_is_refused_before_decoding(client, model):
    resp, decoded_on = _translate(client, model)

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "translation_unsupported"
    assert "Systran/faster-whisper-medium" in error["message"]
    assert "Systran/faster-whisper-large-v3" in error["message"]
    assert "STT_TRANSLATE_MODEL" in error["message"]
    assert decoded_on == []


@pytest.mark.parametrize("model", ["Systran/faster-whisper-medium", "Systran/faster-whisper-large-v3"])
def test_a_translation_on_a_model_that_translates_is_decoded_on_it(client, model):
    resp, decoded_on = _translate(client, model)

    assert resp.status_code == 200, resp.text
    assert decoded_on == [model]


@pytest.mark.parametrize("model", [None, "whisper-1"])
def test_the_translation_model_setting_takes_whisper_1_and_no_model(client, model):
    resp, decoded_on = _translate(client, model, translate_model="Systran/faster-whisper-medium")

    assert resp.status_code == 200, resp.text
    assert decoded_on == ["Systran/faster-whisper-medium"]


def test_the_translation_model_setting_leaves_a_named_model_and_transcriptions_alone(client):
    # A model the client names is used as named (and refused when it can't translate); transcriptions keep STT_MODEL.
    resp, decoded_on = _translate(client, _TURBO, translate_model="Systran/faster-whisper-medium")
    assert resp.status_code == 400, resp.text
    assert decoded_on == []

    backend = router_module.router._default_backend
    with (
        patch.object(main_module.settings, "stt_model", _TURBO),
        patch.object(main_module.settings, "stt_translate_model", "Systran/faster-whisper-medium"),
    ):
        resp = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("es.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"model": "whisper-1"},
        )
    assert resp.status_code == 200, resp.text
    assert backend.transcribe.call_args.args[1] == _TURBO


def test_a_translation_model_setting_that_cannot_translate_is_refused_by_name(client):
    resp, decoded_on = _translate(client, "whisper-1", translate_model="Systran/faster-whisper-small.en")

    assert resp.status_code == 400, resp.text
    assert "Systran/faster-whisper-small.en" in resp.json()["error"]["message"]
    assert decoded_on == []


# OpenAI's translation object (Verbose JSON) gives `language` as "the language of the output
# translation (always `english`)"; faster-whisper's info.language is the language detected in the audio. A Spanish clip
# translated to English answered "language": "es" for English text. This server spells languages as codes: "en".
@pytest.mark.parametrize(("route", "language"), [
    ("/v1/audio/translations", "en"),
    ("/v1/audio/transcriptions", "es"),  # control: a transcription keeps the language heard
])
def test_verbose_json_language_is_the_language_of_the_text(route, language):
    from types import SimpleNamespace

    from src.backends.faster_whisper import FasterWhisperBackend

    model = "Systran/faster-whisper-medium"
    segment = SimpleNamespace(seek=0, start=0.0, end=1.0, text=" My name is Maria.", tokens=[1, 2],
                              temperature=0.0, avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0)
    tasks = []

    def transcribe(path, **kwargs):  # WhisperModel.transcribe on a Spanish clip
        tasks.append(kwargs["task"])
        return iter([segment]), SimpleNamespace(language="es", duration=1.0)

    backend = FasterWhisperBackend()
    backend._models[model] = SimpleNamespace(transcribe=transcribe)
    backend._loaded_at[model] = time.time()
    with patch.object(router_module.router, "_default_backend", backend), \
         patch.object(router_module.router, "_backends", {"faster-whisper": backend}):
        resp = TestClient(app).post(
            route,
            files={"file": ("es.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
            data={"model": model, "response_format": "verbose_json"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    task = "translate" if route.endswith("translations") else "transcribe"
    assert tasks == [task] and body["task"] == task
    assert body["text"] == "My name is Maria."
    assert body["language"] == language


@pytest.mark.parametrize("route", ["/v1/audio/transcriptions", "/v1/audio/translations"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_a_non_finite_temperature_is_refused_before_decoding(client, route, value):
    # NaN or Infinity was decoded "as given" (README), then verbose_json answered 500
    # because JSON has no NaN. A finite value is still used as given; one no decoder can use is a client error.
    backend = router_module.router._default_backend
    resp = client.post(
        route,
        files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
        data={"model": "test-model", "response_format": "verbose_json", "temperature": value},
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["error"]["code"] == "validation_error"
    assert "temperature" in resp.json()["error"]["message"]
    backend.transcribe.assert_not_called()
    backend.translate.assert_not_called()


@pytest.mark.parametrize("value", ["1.5", "-0.5"])
def test_a_finite_temperature_outside_zero_to_one_is_still_used_as_given(client, value):
    # Only non-finite values are refused; README: "other values are used as given".
    backend = router_module.router._default_backend
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("test.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
        data={"model": "test-model", "temperature": value},
    )
    assert resp.status_code == 200, resp.text
    assert backend.transcribe.call_args.kwargs["temperature"] == float(value)


def test_loaded_models(client):
    resp = client.get("/api/ps")
    assert resp.status_code == 200
    assert "models" in resp.json()


def test_web_page_and_static_files_are_revalidated(client):
    # Without this, Safari can keep running an old module after a deploy (heuristic caching).
    page = client.get("/web")
    assert page.status_code == 200
    assert page.headers["cache-control"] == "no-cache"
    assert 'type="module"' in page.text
    script = client.get("/static/js/main.js")
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-cache"
    assert script.headers["content-type"].split(";")[0] == "text/javascript"
    unchanged = client.get("/static/js/main.js", headers={"If-None-Match": script.headers["etag"]})
    assert unchanged.status_code == 304
    assert unchanged.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("voice", ["ash", "ballad", "coral", "sage", "verse", "marin", "cedar"])
def test_speech_accepts_the_openai_ga_voices_with_kokoro(monkeypatch, voice):
    # The OpenAI SDK offers these voices; with Kokoro they were a 400 "Unknown Kokoro voice".
    from src.tts.backends import kokoro as kokoro_module
    from src.tts.router import TTSRouter

    spoken = []

    class SilentKokoro(kokoro_module.KokoroBackend):
        def synthesize(self, text, voice, speed=1.0, lang_code=None, **_kwargs):
            spoken.append(voice)
            yield np.zeros(2400, dtype=np.float32)

    # An unknown voice falls back to scanning the local Hugging Face cache.
    monkeypatch.setattr(kokoro_module, "_discover_voices_from_package", lambda: None)
    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("kokoro", SilentKokoro(device="cpu"))
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module.settings, "tts_cache_enabled", False)

    resp = TestClient(app).post("/v1/audio/speech", json={
        "model": "kokoro", "input": "Hello", "voice": voice, "response_format": "wav",
    })

    assert resp.status_code == 200, resp.text
    assert spoken == [voice]


async def _start_and_stop_the_app() -> None:
    async def run():
        async with main_module.lifespan(app):
            pass

    await asyncio.wait_for(run(), timeout=30)


async def test_shutdown_releases_the_cached_diarization_pipelines(monkeypatch):
    # A loaded pyannote pipeline stays cached, with its GPU memory, for the life of the process.
    released = threading.Event()
    monkeypatch.setattr(main_module, "unload_pipelines", released.set)
    monkeypatch.setattr(main_module.settings, "tts_preload_default", False)

    await _start_and_stop_the_app()

    assert released.is_set()


async def test_shutdown_goes_on_when_releasing_the_diarization_pipelines_fails(monkeypatch):
    def fail():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    stopped = []
    stop_renderer = main_module.reader_renderer.stop
    monkeypatch.setattr(main_module, "unload_pipelines", fail)
    monkeypatch.setattr(main_module.reader_renderer, "stop", lambda: (stopped.append(True), stop_renderer()))
    monkeypatch.setattr(main_module.settings, "tts_preload_default", False)

    await _start_and_stop_the_app()

    assert stopped == [True]


def test_405_names_the_allowed_methods_in_the_error_envelope(client):
    # RFC 9110 15.5.6: a 405 MUST carry Allow; the JSON error handler keeps Starlette's headers.
    resp = client.put("/v1/audio/speech", json={})

    assert resp.status_code == 405
    assert "POST" in resp.headers["allow"]
    assert resp.json()["error"]["code"] == "http_error"


def test_error_handler_keeps_the_headers_an_http_exception_carries():
    from fastapi import FastAPI, HTTPException

    app_with_handlers = FastAPI()
    main_module._register_exception_handlers(app_with_handlers)

    @app_with_handlers.get("/busy")
    def busy():
        raise HTTPException(status_code=429, detail="busy", headers={"Retry-After": "5"})

    resp = TestClient(app_with_handlers).get("/busy")

    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "5"
    assert resp.json() == {"error": {"message": "busy", "code": "http_error"}}


# OS_LOG_LEVEL (DEBUG, INFO, WARNING or ERROR; INFO by default) is the level of the app's
# logs and of uvicorn's. At DEBUG, the third-party loggers that print wire detail stay at INFO.
_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
_APP_LOGGERS = ("open-speech", "src.streaming", "src.realtime.server", "src.assistant.client")
# httpx, httpcore, h11, urllib3, numba, python-multipart (and its old "multipart" name) and websockets, by the names
# their modules log under.
_CHATTY_LOGGERS = ("httpx", "httpcore.http11", "httpcore.connection", "h11", "urllib3.connectionpool",
                   "numba.core.byteflow", "multipart.multipart", "python_multipart.multipart", "websockets.client")
# uvicorn's loggers, as a fresh process finds them: no level, handler or filter, propagating to the root. Other tests
# leave them changed (uvicorn.Config with access_log=False stops uvicorn.access propagating).
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")


def _logger_state(logger: logging.Logger) -> tuple:
    return logger.level, logger.handlers[:], logger.filters[:], logger.propagate, logger.disabled


def _set_logger_state(logger: logging.Logger, state: tuple) -> None:
    level, handlers, filters, propagate, disabled = state
    logger.setLevel(level)
    logger.handlers[:] = handlers
    logger.filters[:] = filters
    logger.propagate, logger.disabled = propagate, disabled


@contextlib.contextmanager
def _root_logger_as_at_startup():
    """The root logger with no handler yet, as `python -m src.main` finds it; everything is put back afterwards.

    uvicorn's loggers start as a fresh process has them too, without the filter src.main's import installed.
    Used inside the test body: pytest attaches its capture handlers to the root logger for each phase.
    """
    root = logging.getLogger()
    names = {name.split(".")[0] for name in _CHATTY_LOGGERS} | set(_CHATTY_LOGGERS)
    saved_handlers, saved_root_level = root.handlers[:], root.level
    saved_levels = {name: logging.getLogger(name).level for name in names}
    saved_uvicorn = {name: _logger_state(logging.getLogger(name)) for name in _UVICORN_LOGGERS}
    root.handlers[:] = []
    for name in _UVICORN_LOGGERS:
        _set_logger_state(logging.getLogger(name), (logging.NOTSET, [], [], True, False))
    try:
        yield root
    finally:
        for handler in root.handlers:
            if handler not in saved_handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_root_level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
        for name, state in saved_uvicorn.items():
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if handler not in state[1]:
                    handler.close()
            _set_logger_state(logger, state)


_LEVELS = (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)


@pytest.mark.parametrize(
    ("level", "lowest_shown"),
    [("DEBUG", logging.DEBUG), ("INFO", logging.INFO), ("WARNING", logging.WARNING), ("ERROR", logging.ERROR)],
)
def test_the_log_level_decides_which_app_lines_are_logged(caplog, level, lowest_shown):
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging(level)
        [handler] = root.handlers  # the one handler basicConfig adds, in the existing line format
        assert handler.formatter._fmt == _LOG_FORMAT
        root.addHandler(caplog.handler)
        for name in _APP_LOGGERS:
            for line_level in _LEVELS:
                logging.getLogger(name).log(line_level, "a %s line", logging.getLevelName(line_level))

    logged = {(record.name, record.levelno) for record in caplog.records if record.name in _APP_LOGGERS}
    assert logged == {(name, line_level) for name in _APP_LOGGERS for line_level in _LEVELS
                      if line_level >= lowest_shown}


def test_at_debug_the_libraries_that_log_wire_detail_stay_at_info(caplog):
    # Their DEBUG lines are connection traces, URLs with query strings, request headers (websockets prints
    # Authorization) and JIT compiler steps by the thousand; their INFO lines (httpx's one line per request) stay.
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging("DEBUG")
        root.addHandler(caplog.handler)
        logging.getLogger("src.streaming").debug("the app's own DEBUG line")
        for name in _CHATTY_LOGGERS:
            logging.getLogger(name).debug("wire detail")
            logging.getLogger(name).info("one line per request")

    names = (*_CHATTY_LOGGERS, "src.streaming")
    logged = sorted((record.name, record.levelname) for record in caplog.records if record.name in names)
    assert logged == sorted([("src.streaming", "DEBUG"), *((name, "INFO") for name in _CHATTY_LOGGERS)])


@pytest.mark.parametrize(("level", "shown"), [("INFO", True), ("WARNING", False), ("ERROR", False)])
def test_above_debug_the_libraries_follow_the_level_as_before(caplog, level, shown):
    with _root_logger_as_at_startup() as root:
        before = {name: logging.getLogger(name).level for name in _CHATTY_LOGGERS}
        main_module.configure_logging(level)
        assert {name: logging.getLogger(name).level for name in _CHATTY_LOGGERS} == before
        root.addHandler(caplog.handler)
        logging.getLogger("httpx").info("HTTP Request: POST http://ai-server.lan:8080/v1/chat/completions")

    assert any(record.name == "httpx" for record in caplog.records) is shown


@pytest.mark.parametrize(
    ("level", "uvicorn_level"),
    # uvicorn has no DEBUG lines of its own; at "debug" its websockets protocol prints every handshake's request
    # headers, Authorization: Bearer <OS_API_KEY> among them, so DEBUG gives uvicorn "info".
    [("DEBUG", "info"), ("INFO", "info"), ("WARNING", "warning"), ("ERROR", "error")],
)
def test_the_entrypoint_gives_uvicorn_the_log_level_so_its_access_log_follows(monkeypatch, caplog, level,
                                                                                  uvicorn_level):
    import uvicorn

    run = MagicMock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(main_module.settings, "os_log_level", level)
    monkeypatch.setattr(main_module.settings, "os_ssl_enabled", False)

    with caplog.at_level(logging.INFO, logger="open-speech"):
        main_module.main()

    run.assert_called_once()
    assert run.call_args.args == (main_module.app,)
    assert run.call_args.kwargs["log_level"] == uvicorn_level
    assert (run.call_args.kwargs["host"], run.call_args.kwargs["port"]) == (main_module.settings.os_host,
                                                                            main_module.settings.os_port)
    assert "Listening on http://" in caplog.text


# uvicorn's own INFO lines print each request's path with its query string (the access
# line on uvicorn.access; the WebSocket "[accepted]" and "403" lines on uvicorn.error). So docker logs kept the
# deprecated ?api_key=, every WebSocket ?ticket= and the Reader's signed media links (?exp= and &sig=). The values of
# those four parameters are logged as *** instead, at every OS_LOG_LEVEL; every other parameter and the path stay.
_FAKE_KEY = "fake-key-for-log-redaction"  # never a real key
_ACCESS_LINE = '%s - "%s %s HTTP/%s" %d'  # uvicorn's access line, five arguments


async def _stub_asgi_app(scope, receive, send):
    """200 for any HTTP request; a WebSocket to /v1/realtime is closed before accept (uvicorn's 403), others open."""
    if scope["type"] == "http":
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"0")]})
        await send({"type": "http.response.body", "body": b""})
        return
    if scope["type"] != "websocket":
        return
    await receive()  # websocket.connect
    if scope["path"] == "/v1/realtime":
        await send({"type": "websocket.close", "code": 1008})
        return
    await send({"type": "websocket.accept"})
    while (await receive())["type"] != "websocket.disconnect":
        pass


@contextlib.asynccontextmanager
async def _uvicorn_serving(http: str, ws: str):
    """A real uvicorn server for _stub_asgi_app on 127.0.0.1; its port. log_config=None: logging stays as set up."""
    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(_stub_asgi_app, http=http, ws=ws, lifespan="off", log_config=None))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(10):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("uvicorn exited before startup")
                    await asyncio.sleep(0.01)
            yield port
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=10)


@pytest.mark.parametrize(("http", "ws"), [("h11", "auto"), ("httptools", "auto"), ("h11", "websockets-sansio")])
async def test_uvicorns_request_lines_mask_the_keys_tickets_and_media_signatures(caplog, http, ws):
    import httpx
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus

    from src.reader.media import MediaSigner

    ticket, refused_ticket = "fake-ticket-opened-for-log-redaction", "fake-ticket-refused-for-log-redaction"
    media_link = MediaSigner("").url("part-1", 600)
    media_path, media_query = media_link.split("?")
    expires, signature = (pair.split("=")[1] for pair in media_query.split("&"))
    normal = "/v1/audio/voices?model=kokoro&lang=en-us"

    with _root_logger_as_at_startup() as root:
        main_module.configure_logging("INFO")
        root.addHandler(caplog.handler)
        async with _uvicorn_serving(http, ws) as port:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=10) as client:
                for target in (f"/v1/models?api_key={_FAKE_KEY}", media_link, normal):
                    assert (await client.get(target)).status_code == 200
            for target in (f"/v1/audio/stream?ticket={ticket}", f"/v1/audio/speech/stream?api_key={_FAKE_KEY}"):
                async with connect(f"ws://127.0.0.1:{port}{target}", proxy=None, open_timeout=10):
                    pass
            with pytest.raises(InvalidStatus) as refused:
                async with connect(f"ws://127.0.0.1:{port}/v1/realtime?ticket={refused_ticket}", proxy=None,
                                   open_timeout=10):
                    pass
            assert refused.value.response.status_code == 403

    lines = [record for record in caplog.records if record.name in ("uvicorn.access", "uvicorn.error")]
    logged = "\n".join(record.getMessage() for record in lines)
    for secret in (_FAKE_KEY, ticket, refused_ticket, expires, signature):
        assert secret not in logged
    access = [record for record in lines if record.name == "uvicorn.access"]
    assert [len(record.args) for record in access] == [5, 5, 5]  # the shape AccessFormatter unpacks
    assert [record.args[2] for record in access] == [
        "/v1/models?api_key=***", f"{media_path}?exp=***&sig=***", normal,
    ]
    assert access[2].getMessage() == f'{access[2].args[0]} - "GET {normal} HTTP/1.1" 200'  # as it always was
    websocket = [record.getMessage().split(" - ", 1)[1] for record in lines if "WebSocket" in str(record.msg)]
    assert websocket == [
        '"WebSocket /v1/audio/stream?ticket=***" [accepted]',
        '"WebSocket /v1/audio/speech/stream?api_key=***" [accepted]',
        '"WebSocket /v1/realtime?ticket=***" 403',
    ]


@contextlib.contextmanager
def _every_logger_kept():
    """Every existing logger's level, handlers, filters, propagate and disabled flag are put back afterwards."""
    loggers = [logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger)]
    saved = [(logger, _logger_state(logger)) for logger in loggers]
    try:
        yield
    finally:
        for logger, state in saved:
            for handler in logger.handlers:
                if handler not in state[1]:
                    handler.close()
            _set_logger_state(logger, state)


def test_with_uvicorns_own_logging_config_the_printed_lines_are_masked(monkeypatch, capsys):
    # main() runs uvicorn.run(app, log_level=...): its Config applies uvicorn's LOGGING_CONFIG with dictConfig, which
    # replaces the uvicorn loggers' handlers (the access line goes to stdout through AccessFormatter) but keeps their
    # filters.
    import logging.config

    import uvicorn

    monkeypatch.setattr(logging.config, "_clearExistingHandlers", lambda: None)  # it would close pytest's handlers
    with _every_logger_kept(), _root_logger_as_at_startup():
        main_module.configure_logging("INFO")
        uvicorn.Config(_stub_asgi_app, log_level=main_module.uvicorn_log_level("INFO"))
        logging.getLogger("uvicorn.access").info(_ACCESS_LINE, "127.0.0.1:50000", "GET",
                                                 f"/v1/models?api_key={_FAKE_KEY}", "1.1", 200)
        logging.getLogger("uvicorn.error").info('%s - "WebSocket %s" 403', "127.0.0.1:50001",
                                                "/v1/realtime?ticket=fake-ticket-for-log-redaction")
    printed = capsys.readouterr()

    assert _FAKE_KEY not in printed.out + printed.err and "fake-ticket-for-log-redaction" not in printed.out + printed.err
    assert '127.0.0.1:50000 - "GET /v1/models?api_key=*** HTTP/1.1" 200 OK' in printed.out
    assert '127.0.0.1:50001 - "WebSocket /v1/realtime?ticket=***" 403' in printed.err


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_every_log_level_masks_uvicorns_lines(caplog, level):
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging(level)
        root.addHandler(caplog.handler)
        logging.getLogger("uvicorn.error").error('%s - "WebSocket %s" 500', "127.0.0.1:50000",
                                                 "/v1/realtime?ticket=fake-ticket-for-log-redaction")
        logging.getLogger("uvicorn.access").error(_ACCESS_LINE, "127.0.0.1:50001", "GET",
                                                  f"/v1/models?api_key={_FAKE_KEY}", "1.1", 500)

    logged = sorted(record.getMessage() for record in caplog.records if record.name.startswith("uvicorn."))
    assert logged == ['127.0.0.1:50000 - "WebSocket /v1/realtime?ticket=***" 500',
                      '127.0.0.1:50001 - "GET /v1/models?api_key=*** HTTP/1.1" 500']


def _logged_paths(caplog, *paths) -> list:
    """The path argument of uvicorn.access's line for each of `paths`, logged after configure_logging()."""
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging("INFO")
        root.addHandler(caplog.handler)
        for path in paths:
            logging.getLogger("uvicorn.access").info(_ACCESS_LINE, "127.0.0.1:50000", "GET", path, "1.1", 200)
    return [record.args[2] for record in caplog.records if record.name == "uvicorn.access"]


@pytest.mark.parametrize(("path", "logged"), [
    pytest.param("/v1/models?api_key=k3y", "/v1/models?api_key=***", id="api-key"),
    pytest.param("/v1/audio/stream?model=x&ticket=t1&lang=en", "/v1/audio/stream?model=x&ticket=***&lang=en",
                 id="other-parameters-kept"),
    pytest.param("/v1/reader/media/p.mp3?exp=1762600000&sig=0123456789abcdef0123456789abcdef",
                 "/v1/reader/media/p.mp3?exp=***&sig=***", id="signed-media-link"),
    pytest.param("/x?api%5Fkey=k3y&%74icket=t1&si%67=s", "/x?api%5Fkey=***&%74icket=***&si%67=***",
                 id="percent-encoded-names"),
    pytest.param("/x?API_KEY=k3y&Ticket=t1&SIG=s&Exp=1", "/x?API_KEY=***&Ticket=***&SIG=***&Exp=***", id="any-case"),
    pytest.param("/x?ticket=t1&ticket=t2&api_key=a&api_key=b", "/x?ticket=***&ticket=***&api_key=***&api_key=***",
                 id="repeated"),
    pytest.param("/x?ticket=t1?t2=t3&model=x", "/x?ticket=***&model=x", id="question-mark-in-a-value"),
    pytest.param("/x?api_key=a=b%26c+d;e", "/x?api_key=***", id="value-runs-to-the-next-ampersand"),
    pytest.param("/x?next=/y?ticket=t1", "/x?next=/y?ticket=***", id="a-link-inside-another-value"),
    pytest.param("/x?api_keys=a&my_ticket=b&expires=c&sigma=d&x_sig=e&key=f&api+key=g",
                 "/x?api_keys=a&my_ticket=b&expires=c&sigma=d&x_sig=e&key=f&api+key=g", id="near-names-kept"),
    pytest.param("/x?ticket=&api_key=&model=x", "/x?ticket=&api_key=&model=x", id="empty-values-kept"),
    pytest.param("/v1/api_key=1/ticket=2", "/v1/api_key=1/ticket=2", id="the-path-kept"),
    pytest.param("/v1/models", "/v1/models", id="no-query-string"),
])
def test_only_the_secret_parameters_values_are_masked(caplog, path, logged):
    assert _logged_paths(caplog, path) == [logged]


class _CountingPattern:
    """A compiled pattern whose search() adds up the characters it walks: to the match's end, or to the text's end."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.scanned = 0

    def search(self, text, pos=0, *rest):
        match = self.pattern.search(text, pos, *rest)
        self.scanned += (match.end() if match else len(text)) - pos
        return match


def _nested_query_path(characters: int = 65_000) -> str:
    return "/?x=" + "?b=" * (characters // 3)  # "?" is legal in a query (RFC 3986 3.4); httptools takes 65535 bytes


def test_masking_reads_a_long_query_a_bounded_number_of_times(monkeypatch):
    # Every "?b=" restarted the scan inside the value and searched for its end ("&" or
    # whitespace) to the end of the text, so a 65 KB URL cost about n^2/6 steps, 5 s on the event loop per request.
    import src.log_redaction as redaction

    parameter = _CountingPattern(redaction._PARAMETER)
    value_end = _CountingPattern(redaction._VALUE_END)
    monkeypatch.setattr(redaction, "_PARAMETER", parameter)
    monkeypatch.setattr(redaction, "_VALUE_END", value_end)
    text = _nested_query_path()
    assert redaction.mask_query_secrets(text) is text
    secret = text + "?ticket=t1" + "?b=" * 1000 + "&api_key=k&z=1"
    assert redaction.mask_query_secrets(secret) == text + "?ticket=***&api_key=***&z=1"
    scanned = parameter.scanned + value_end.scanned
    assert scanned <= 10 * (len(text) + len(secret)), f"{scanned:,} characters scanned"


def test_one_access_line_with_a_64_kb_url_is_masked_quickly():
    # The filter runs in uvicorn's send(), on the event loop.
    from src.log_redaction import QuerySecretFilter

    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, _ACCESS_LINE,
                               ("127.0.0.1:50000", "GET", _nested_query_path() + "&sig=s", "1.1", 404), None)
    started = time.perf_counter()
    QuerySecretFilter().filter(record)
    elapsed = time.perf_counter() - started
    assert record.args[2].endswith("&sig=***")
    assert elapsed < 0.5, f"one access line took {elapsed:.2f} s in the filter"


def test_a_line_without_arguments_and_one_with_a_mapping_are_masked_too(caplog):
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging("INFO")
        root.addHandler(caplog.handler)
        uvicorn_error = logging.getLogger("uvicorn.error")
        uvicorn_error.warning(f'"GET /v1/models?api_key={_FAKE_KEY} HTTP/1.1" 401')  # no arguments: msg is the line
        uvicorn_error.info("%s %s", "GET", "/x?ticket=t1&model=x")
        uvicorn_error.info("%(method)s %(path)s", {"method": "GET", "path": "/x?sig=s1"})

    assert [record.getMessage() for record in caplog.records if record.name == "uvicorn.error"] == [
        '"GET /v1/models?api_key=*** HTTP/1.1" 401', "GET /x?ticket=***&model=x", "GET /x?sig=***",
    ]


class _HostileStr(str):
    def __contains__(self, item):
        raise RuntimeError("a str subclass whose __contains__ raises")


class _HostileMapping(Mapping):
    """Formats like a dict, but can't be iterated."""

    def __getitem__(self, key):
        return "/x?ticket=t1"

    def __iter__(self):
        raise RuntimeError("a mapping that can't be iterated")

    def __len__(self):
        return 1


def test_the_masking_never_raises_and_never_drops_a_line(caplog):
    with _root_logger_as_at_startup() as root:
        main_module.configure_logging("INFO")
        root.addHandler(caplog.handler)
        access, error = logging.getLogger("uvicorn.access"), logging.getLogger("uvicorn.error")
        access.info(_ACCESS_LINE, ("127.0.0.1", 50000), "GET", b"/x?ticket=t1", "1.1", 200)  # args that aren't str
        error.info('%s - "WebSocket %s" 403', "127.0.0.1:50000", _HostileStr("/v1/realtime?ticket=t1"))
        error.info(_HostileStr("/v1/realtime?ticket=t2"))
        error.info(RuntimeError("an exception as the message"))
        error.info("%(path)s", _HostileMapping())

    messages = [record.getMessage() for record in caplog.records if record.name.startswith("uvicorn.")]
    assert messages == [
        "('127.0.0.1', 50000) - \"GET b'/x?ticket=t1' HTTP/1.1\" 200",  # bytes are not masked: uvicorn logs str
        '127.0.0.1:50000 - "WebSocket /v1/realtime?ticket=***" 403',
        "/v1/realtime?ticket=***",
        "an exception as the message",
        "/x?ticket=t1",  # a mapping it can't read is left as it is
    ]
