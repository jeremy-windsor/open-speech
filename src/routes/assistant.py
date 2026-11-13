"""AI assistant routes: its status, the AI presets, one test call to the AI server, the connection, and "send
anything to the AI" (``POST /api/assistant/ask``, ``src/assistant/ask.py``).

Behind the API key check like the rest (``src/middleware.py``). Every route but ``GET /api/assistant`` and the
``/api/assistant/connection`` routes answers ``404`` (code ``assistant_disabled``) while the assistant is off (no
server URL in ``.env`` or saved). An invalid field gets ``422`` with code ``validation_error`` and the field's name
in ``param``.

The connection routes work whether or not the assistant is on, because they are how you turn it on. While
``OS_ASSISTANT_CONNECTION_LOCKED`` is true, a change answers ``403`` (``connection_locked``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from src import assistant
from src.assistant import ask as ask_module
from src.assistant import connection, presets, prompts
from src.assistant.client import display_host
from src.assistant.errors import ASSISTANT_TIMEOUT, AssistantError
from src.config import ASSISTANT_PROMPT_MAX_CHARS

DEFAULT_TEST_TEXT = "Say hello in one sentence."
_TEST_FIELDS = ("preset", "text", "overrides")


def _invalid(field: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": {"message": message, "code": "validation_error", "param": field}})


def _not_found(preset_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail={"message": f"AI preset {preset_id!r} not found", "code": "preset_not_found"})


def _locked(exc: connection.ConnectionLockedError) -> HTTPException:
    return HTTPException(status_code=403, detail={"message": exc.message, "code": "connection_locked"})


def create_router(
    *,
    get_settings: Callable,
    get_tts_router: Callable | None = None,
    get_voice_library: Callable | None = None,
    get_pronunciation_dict: Callable | None = None,
) -> APIRouter:
    """``get_tts_router`` (with the voice library and the pronunciation dictionary, as the TTS routes take them)
    lets ``POST /api/assistant/ask`` speak its replies; without it, speech is ``503`` ``speech_unavailable``."""
    router = APIRouter()

    def require_enabled() -> None:
        if not get_settings().assistant_enabled:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL",
                    "code": "assistant_disabled",
                },
            )

    enabled = [Depends(require_enabled)]

    @router.get("/api/assistant")
    def assistant_status():
        """Whether the assistant is on, the server's host and the default model, from the effective connection.
        Of each connection setting, only whether it is set (differs from its default) and where it comes from
        (``env``, ``saved`` or ``default``), never its value; and whether the connection is locked."""
        current = connection.effective(get_settings())
        return {
            "enabled": current.enabled,
            "host": display_host(current.url) if current.enabled else None,
            "model": current.model,
            "server_settings": {
                "OS_ASSISTANT_URL": bool(current.url),
                "OS_ASSISTANT_API_KEY": bool(current.api_key),
                "OS_ASSISTANT_EXTRA_BODY": bool(current.extra_body),
                "OS_ASSISTANT_VERIFY_TLS": current.verify_tls is not True,
                "OS_ASSISTANT_MAX_TOKENS_FIELD": current.max_tokens_field != "max_tokens",
                "OS_ASSISTANT_TEMPLATE_KWARGS": bool(current.template_kwargs),
                # Stream replies is on by default: true means it is off.
                "OS_ASSISTANT_STREAM": current.stream is not True,
            },
            "locked": current.locked,
            "sources": {connection.ENV_NAMES[name]: current.sources[name] for name in connection.FIELDS},
        }

    # ── The connection ───────────────────────────────────────────────────────

    @router.get("/api/assistant/connection")
    def get_connection():
        """Each connection field's effective value (the key only as ``api_key_set``), its ``sources`` entry
        (``env``, ``saved`` or ``default``), ``enabled`` and ``locked``."""
        return connection.effective(get_settings()).view()

    @router.put("/api/assistant/connection")
    async def save_connection(payload: dict[str, Any] = Body(...)):
        """Save the given fields over the stored ones, rebuild the shared client, and answer the new view. A field
        left out keeps its value; ``null`` makes it follow ``.env`` again; ``api_key`` ``""`` clears the key."""
        settings = get_settings()
        try:
            await run_in_threadpool(connection.save, payload, settings)
        except connection.ConnectionLockedError as exc:
            raise _locked(exc) from None
        except connection.ConnectionFieldError as exc:
            return _invalid(exc.field, exc.message)
        await assistant.rebuild_client(settings)
        return connection.effective(settings).view()

    @router.delete("/api/assistant/connection")
    async def forget_connection():
        """Forget the saved connection, so every field follows ``.env`` again; rebuild the client."""
        settings = get_settings()
        try:
            await run_in_threadpool(connection.forget, settings)
        except connection.ConnectionLockedError as exc:
            raise _locked(exc) from None
        await assistant.rebuild_client(settings)
        return connection.effective(settings).view()

    @router.post("/api/assistant/connection/check")
    async def check_connection(payload: dict[str, Any] = Body(default_factory=dict)):
        """One ``GET {url}/models`` for a candidate connection (the fields of a save, applied without saving; an
        empty body checks the connection in use). Answers ``{reachable, status, models, listed, elapsed_ms}``:
        ``models`` only from an OpenAI-style list (``listed``), nothing else of the body. When the server can't be
        reached: ``502``, or ``504`` for a timeout, with the error code and ``reachable: false``."""
        settings = get_settings()
        try:
            trial = await run_in_threadpool(connection.candidate, payload, settings)
        except connection.ConnectionLockedError as exc:
            raise _locked(exc) from None
        except connection.ConnectionFieldError as exc:
            return _invalid(exc.field, exc.message)
        if not trial.url:
            return _invalid("url", "url is needed: the AI server's base URL, such as http://ai-server.lan:8080/v1")
        client = assistant.build_client(trial)
        started = time.perf_counter()
        try:
            listed = await client.list_models()
        except AssistantError as exc:
            return JSONResponse(
                status_code=504 if exc.code == ASSISTANT_TIMEOUT else 502,
                content={"error": {"message": exc.message, "code": exc.code}, "reachable": False,
                         "elapsed_ms": _elapsed_ms(started)},
            )
        finally:
            await client.aclose()
        return {
            "reachable": True,
            "status": listed.status,
            "models": listed.models,
            "listed": listed.listed,
            "elapsed_ms": _elapsed_ms(started),
        }

    @router.get("/api/assistant/presets", dependencies=enabled)
    def list_presets():
        return {"presets": presets.list_presets()}

    @router.post("/api/assistant/presets", status_code=201, dependencies=enabled)
    def create_preset(payload: dict[str, Any] = Body(...)):
        try:
            return presets.create_preset(payload)
        except presets.PresetError as exc:
            return _invalid(exc.field, exc.message)
        except presets.PresetConflictError as exc:
            raise HTTPException(status_code=409, detail={"message": exc.message, "code": exc.code}) from None

    @router.get("/api/assistant/presets/{preset_id}", dependencies=enabled)
    def get_preset(preset_id: str):
        view = presets.get_preset(preset_id)
        if view is None:
            raise _not_found(preset_id)
        return view

    @router.put("/api/assistant/presets/{preset_id}", dependencies=enabled)
    def replace_preset(preset_id: str, payload: dict[str, Any] = Body(...)):
        try:
            return presets.replace_preset(preset_id, payload)
        except presets.PresetNotFoundError:
            raise _not_found(preset_id) from None
        except presets.PresetError as exc:
            return _invalid(exc.field, exc.message)

    @router.delete("/api/assistant/presets/{preset_id}", dependencies=enabled)
    def delete_preset(preset_id: str):
        """Delete a preset you made (204), or restore a built-in's defaults (200, the restored preset)."""
        try:
            restored = presets.delete_preset(preset_id)
        except presets.PresetNotFoundError:
            raise _not_found(preset_id) from None
        return Response(status_code=204) if restored is None else restored

    @router.post("/api/assistant/test", dependencies=enabled)
    async def test_preset(payload: dict[str, Any] = Body(default_factory=dict)):
        """One real call: ``{preset, text?, overrides?}`` answers ``{reply, thinking, finish_reason, elapsed_ms}``,
        or the AssistantError's code (``504`` for a timeout, else ``502``). ``overrides`` takes the preset fields,
        so a preset can be tried as it is on screen, unsaved: a field sent as ``null`` or empty is tried as an
        emptied field (``.env`` or the code's default), as it would be once saved, and a field left out keeps the
        saved value."""
        for name in payload:
            if name not in _TEST_FIELDS:
                return _invalid(str(name), f"{name} is not a test field; send {', '.join(_TEST_FIELDS)}")
        text = payload.get("text")
        if text is None or (isinstance(text, str) and not text.strip()):
            text = DEFAULT_TEST_TEXT
        elif not isinstance(text, str) or len(text) > ASSISTANT_PROMPT_MAX_CHARS:
            return _invalid("text", f"text must be text of up to {ASSISTANT_PROMPT_MAX_CHARS} characters")
        elif presets.has_lone_surrogate(text):  # it can't be sent to the AI
            return _invalid("text", presets.lone_surrogate_message("text"))
        try:
            settings = await run_in_threadpool(
                presets.effective, payload.get("preset"), payload.get("overrides"), unsaved=True,
            )
        except presets.PresetError as exc:
            return _invalid(exc.field, exc.message)
        client = assistant.get_client()
        if client is None:
            raise HTTPException(
                status_code=503,
                detail={"message": "The AI assistant client hasn't started", "code": "assistant_not_started"},
            )
        started = time.perf_counter()
        try:
            reply = await client.reply(prompts.messages_for(settings, [], text), settings)
        except AssistantError as exc:
            return JSONResponse(
                status_code=504 if exc.code == ASSISTANT_TIMEOUT else 502,
                content={"error": {"message": exc.message, "code": exc.code}, "elapsed_ms": _elapsed_ms(started)},
            )
        return {
            "reply": reply.text,
            "thinking": reply.thinking if settings.show_thinking else None,
            "finish_reason": reply.finish_reason,
            "elapsed_ms": _elapsed_ms(started),
        }

    # ── Send anything to the AI ──────────────────────────────────────────────

    @router.post("/api/assistant/ask", dependencies=enabled)
    async def ask(payload: Any = Body(...)):
        """Send a text, a transcript or a Reader document (or part of one) to the AI through a preset, and get the
        reply back: shown, spoken or saved (``src/assistant/ask.py``).

        Body: ``{preset?, input: {kind, ...}, output?: {text?, speech?: {model?, voice?, format?, speed?},
        document?: {title?}}, overrides?, stream?}``. ``input`` is ``{kind: "text", text}``, ``{kind:
        "transcript", text}`` or ``{kind: "transcript", history_id}``, or ``{kind: "document", document_id}`` with
        ``chapters: [first, last]``, ``blocks: [first, last]`` or ``block_start`` and ``block_end``. ``preset``
        defaults to ``book`` for a document and ``voice`` otherwise; ``overrides`` takes the preset's tunable
        fields; ``stream`` takes only ``false`` for now.

        An input that fits in one piece of ``chunk_chars`` answers ``200 {reply, thinking, finish_reason,
        elapsed_ms, preset, input: {kind, chars}}``, with ``audio: {format, content_type, base64, bytes}`` (or
        ``audio_error: {code, message}``) for speech and ``document_id`` (or ``document_error``) for a document. A
        longer one answers ``202 {job, preset, input}``: a book job, read and cancelled through
        ``/v1/reader/assistant-jobs/{id}``. An AI error is ``502`` (``504`` for a timeout) ``{error, elapsed_ms}``,
        with ``sentence`` and its ``audio`` when speech was asked and the preset speaks errors.
        """
        settings = get_settings()
        speaker = None
        if get_tts_router is not None:
            speaker = ask_module.Speaker(
                get_tts_router(),
                settings=settings,
                voice_library=get_voice_library() if get_voice_library is not None else None,
                pronunciation_dict=get_pronunciation_dict() if get_pronunciation_dict is not None else None,
            )
        try:
            status, content = await ask_module.handle(payload, settings=settings, speaker=speaker)
        except ask_module.AskError as exc:
            return JSONResponse(status_code=exc.status, content=exc.content())
        return JSONResponse(status_code=status, content=content)

    return router


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))
