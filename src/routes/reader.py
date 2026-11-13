"""Reader routes: import documents, render them to audio parts, save position, and send them to the AI."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Annotated, Any, Callable

import anyio
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.datastructures import Headers, UploadFile
from starlette.types import Receive, Scope, Send

from src import assistant
from src.assistant import presets as assistant_presets
from src.reader.assistant_jobs import TITLE_MAX_CHARS, AssistantJobs, ChaptersError, set_runner
from src.reader.renderer import ReaderVoiceError
from src.reader.sources import parse_pasted, parse_upload
from src.services.tts import _read_upload_limited

MB = 1024 * 1024
# The fields of a book job's body (POST /v1/reader/documents/{id}/assistant).
_JOB_FIELDS = ("preset", "overrides", "chapters", "title")


def _invalid(field: str, message: str) -> JSONResponse:
    """422 naming the field, as the AI assistant's own routes answer (src/routes/assistant.py)."""
    return JSONResponse(status_code=422, content={"error": {"message": message, "code": "validation_error", "param": field}})


class ProgressPayload(BaseModel):
    block_index: int = Field(ge=0)
    offset_ms: int = Field(default=0, ge=0, le=2**63 - 1)   # SQLite's INTEGER range
    model: str | None = Field(default=None, max_length=200)
    voice: str | None = Field(default=None, max_length=200)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class RenderPayload(BaseModel):
    from_block: int = Field(ge=0)
    model: str = Field(min_length=1, max_length=200)
    voice: str = Field(min_length=1, max_length=200)


class _MediaResponse(FileResponse):
    """A part's audio file, with the Range handling Starlette gets wrong.

    A suffix range longer than the file is the whole file (RFC 9110 14.1.2); Starlette computes
    a negative start for it and answers 416. A range unit other than bytes MUST be ignored
    (RFC 9110 14.2); Starlette answers 400. Several ranges come back from Starlette as a 206
    labelled audio/mpeg with the multipart type in Content-Range, so they are ignored too, as
    a server MAY: players only ask for one.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Stat before anything is sent, so a part unlinked after the route found it is a 404. (Starlette
        # opens the file only after sending the headers: a delete in that moment still cuts the response.)
        try:
            stat_result = await anyio.to_thread.run_sync(os.stat, self.path)
        except FileNotFoundError:
            stat_result = None
        if stat_result is None or not stat.S_ISREG(stat_result.st_mode):
            raise HTTPException(status_code=404, detail="Audio part not found")
        self.stat_result = stat_result
        self.set_stat_headers(stat_result)
        range_ = Headers(scope=scope).get("range")
        if range_ is not None:
            unit, _, specs = range_.partition("=")
            if unit.strip().lower() != "bytes" or sum(1 for spec in specs.split(",") if spec.strip()) > 1:
                scope = {**scope, "headers": [(name, value) for name, value in scope["headers"] if name != b"range"]}
        await super().__call__(scope, receive, send)

    @classmethod
    def _parse_ranges(cls, range_: str, file_size: int) -> list[tuple[int, int]]:
        return [(max(0, start), end) for start, end in super()._parse_ranges(range_, file_size)]


async def _read_body_limited(request: Request, limit: int, limit_mb: int) -> bytes:
    """Read a request body, refusing it as soon as it passes ``limit`` bytes."""
    too_large = HTTPException(status_code=413, detail=f"Text is larger than {limit_mb} MB")
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise too_large
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)


def create_router(*, get_settings: Callable, get_store: Callable, get_renderer: Callable, get_signer: Callable) -> APIRouter:
    router = APIRouter()
    # Book jobs, one at a time; the task that runs them starts with the first one. POST /api/assistant/ask queues
    # its long inputs on the same runner, so they share the one slot and these job routes.
    jobs = AssistantJobs(get_store)
    set_runner(jobs)

    def _check_size(document, settings) -> None:
        if document.char_count > settings.reader_max_document_chars:
            raise HTTPException(
                status_code=413,
                detail=f"Document has {document.char_count:,} characters; the limit is {settings.reader_max_document_chars:,}",
            )

    @router.post("/v1/reader/documents")
    async def import_document(request: Request):
        settings = get_settings()
        max_bytes = settings.reader_max_upload_mb * MB
        content_type = request.headers.get("content-type", "")
        try:
            if content_type.startswith("application/json"):
                raw = await _read_body_limited(request, max_bytes + MB, settings.reader_max_upload_mb)
                try:
                    body = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise HTTPException(status_code=400, detail="Request body is not valid JSON") from exc
                if not isinstance(body, dict) or body.get("type") != "text":
                    raise HTTPException(status_code=422, detail='JSON imports need {"type": "text", "text": "..."}')
                text = body.get("text")
                title = body.get("title") or None
                if not isinstance(text, str) or not text.strip():
                    raise HTTPException(status_code=422, detail="text is required")
                if len(text.encode("utf-8")) > max_bytes:
                    raise HTTPException(status_code=413, detail=f"Text is larger than {settings.reader_max_upload_mb} MB")
                if isinstance(title, str):
                    title.encode("utf-8")   # a lone surrogate cannot be stored: refuse it here, as in the text
                document = await run_in_threadpool(parse_pasted, text, title=title if isinstance(title, str) else None)
            elif content_type.startswith("multipart/form-data"):
                form = await request.form()
                try:
                    upload = form.get("file")
                    if not isinstance(upload, UploadFile) or not upload.filename:
                        raise HTTPException(status_code=422, detail="Upload a file in the 'file' field")
                    title = form.get("title")
                    data = await _read_upload_limited(
                        upload, max_bytes, too_large_detail=f"File is larger than {settings.reader_max_upload_mb} MB"
                    )
                finally:
                    # Starlette leaves an upload's temporary file to whoever parsed the form.
                    await form.close()
                document = await run_in_threadpool(
                    parse_upload,
                    upload.filename,
                    data,
                    title=title if isinstance(title, str) and title.strip() else None,
                    max_uncompressed=max_bytes * 8,
                )
            else:
                raise HTTPException(status_code=415, detail="Send a file as multipart/form-data or text as JSON")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _check_size(document, settings)
        store = get_store()
        document_id, created = await run_in_threadpool(store.save_document, document)
        saved = await run_in_threadpool(store.get_document, document_id, include_blocks=False)
        return JSONResponse(saved, status_code=201 if created else 200)

    # The handlers below are plain functions so FastAPI runs them in its threadpool:
    # they wait on SQLite and the file system, and a big document takes a moment.
    @router.get("/v1/reader/documents")
    def list_documents():
        return {"documents": get_store().list_documents()}

    @router.get("/v1/reader/documents/{document_id}")
    def get_document(document_id: str):
        document = get_store().get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return document

    @router.delete("/v1/reader/documents/{document_id}")
    def delete_document(document_id: str):
        paths = get_store().delete_document(document_id)
        if paths is None:
            raise HTTPException(status_code=404, detail="Document not found")
        # After the rows are gone: a /parts poll before that would cache the outline again.
        get_renderer().forget_document(document_id)
        jobs.forget_document(document_id)  # its jobs went with it; a running one stops
        for path in paths:
            with contextlib.suppress(OSError):
                Path(path).unlink(missing_ok=True)
        shutil.rmtree(get_renderer().cache_dir / document_id, ignore_errors=True)
        return Response(status_code=204)

    @router.put("/v1/reader/documents/{document_id}/progress")
    def save_progress(document_id: str, payload: ProgressPayload):
        try:
            progress = get_store().set_progress(document_id, **payload.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if progress is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"progress": progress}

    @router.post("/v1/reader/documents/{document_id}/render")
    def render(document_id: str, payload: RenderPayload):
        if not get_settings().tts_enabled:  # refused before any part is queued
            raise HTTPException(status_code=404, detail="TTS is disabled")
        try:
            parts = get_renderer().request(document_id, payload.from_block, payload.model, payload.voice)
        except ReaderVoiceError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if parts is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"parts": parts}

    # ── Book jobs: a document sent to the AI in pieces (src/reader/assistant_jobs.py) ──

    def require_assistant() -> None:
        if not get_settings().assistant_enabled:  # as render answers while TTS is off
            # The words of src/routes/assistant.py: a URL saved in Settings turns it on without a restart.
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL",
                    "code": "assistant_disabled",
                },
            )

    assistant_on = [Depends(require_assistant)]

    @router.post("/v1/reader/documents/{document_id}/assistant", status_code=202, dependencies=assistant_on)
    async def start_assistant_job(document_id: str, payload: dict[str, Any] = Body(default_factory=dict)):
        """Send the document to the AI in pieces; its answers become a new Reader document.

        Body: ``preset`` (an AI preset id, ``book`` by default), ``overrides`` (the preset fields), ``chapters``
        (``[first, last]``, Reader chapter indexes from 0, or null for all) and ``title`` (the result is named
        "<document title>: <title, or the preset's name>"). Answers ``202`` with the job.
        """
        for name in payload:
            if name not in _JOB_FIELDS:
                return _invalid(str(name), f"{name} is not a job field; send {', '.join(_JOB_FIELDS)}")
        preset_id = payload.get("preset")
        if preset_id is None:
            preset_id = "book"
        elif not isinstance(preset_id, str):
            return _invalid("preset", "preset must be an AI preset id")
        title = payload.get("title")
        if title is not None:
            if not isinstance(title, str) or len(title) > TITLE_MAX_CHARS:
                return _invalid("title", f"title must be text of up to {TITLE_MAX_CHARS} characters")
            try:
                title.encode("utf-8")  # a lone surrogate can't be stored
            except UnicodeEncodeError:
                return _invalid("title", "title must be valid text")
            title = title.strip() or None
        try:
            settings = await run_in_threadpool(
                assistant_presets.effective, preset_id, payload.get("overrides"), purpose="book"
            )
        except assistant_presets.PresetError as exc:
            return _invalid(exc.field, exc.message)
        if assistant.get_client() is None:
            raise HTTPException(
                status_code=503,
                detail={"message": "The AI assistant client hasn't started", "code": "assistant_not_started"},
            )
        try:
            job = await jobs.start(document_id, settings, chapters=payload.get("chapters"), title=title)
        except ChaptersError as exc:
            return _invalid("chapters", str(exc))
        if job is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return JSONResponse(job, status_code=202)

    @router.get("/v1/reader/assistant-jobs", dependencies=assistant_on)
    def list_assistant_jobs():
        """Every book job, newest first, without its chapters."""
        return {"jobs": jobs.list()}

    @router.get("/v1/reader/assistant-jobs/{job_id}", dependencies=assistant_on)
    def get_assistant_job(job_id: str):
        """The job, with each chapter's status, its text once done, and its thinking if the preset shows it."""
        job = jobs.view(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="AI job not found")
        return job

    @router.delete("/v1/reader/assistant-jobs/{job_id}", dependencies=assistant_on)
    async def cancel_assistant_job(job_id: str):
        """Cancel a queued or running job; its finished pieces stay saved. An ended job is answered as it is."""
        job = await jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="AI job not found")
        return job

    @router.get("/v1/reader/documents/{document_id}/parts")
    def parts(
        document_id: str,
        from_block: Annotated[int, Query(ge=0)],
        model: Annotated[str, Query(min_length=1, max_length=200)],
        voice: Annotated[str, Query(min_length=1, max_length=200)],
    ):
        try:
            planned = get_renderer().status(document_id, from_block, model, voice)
        except ReaderVoiceError as exc:   # a named voice that is gone does not keep playing its old parts
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if planned is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"parts": planned}

    # Exempt from API-key auth in the middleware; the signed link is the credential.
    @router.get("/v1/reader/media/{name}")
    def media(name: str, exp: int, sig: Annotated[str, Query(pattern=r"^[0-9a-f]{32}$")]):
        part_id = name.removesuffix(".mp3")
        if not get_signer().verify(part_id, exp, sig):
            raise HTTPException(status_code=403, detail="Audio link is invalid or has expired")
        store = get_store()
        # Touched first: the renderer never prunes a part played in the last few minutes.
        part = store.get_part(part_id) if store.touch_part(part_id) else None
        if part is not None and not Path(part["path"]).is_file():
            # Forget it so the next render request makes the part again.
            store.forget_part(part_id)
            part = None
        if part is None:
            raise HTTPException(status_code=404, detail="Audio part not found")
        return _MediaResponse(part["path"], media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})

    # HEAD as well (RFC 9110 9.1). In the schema it would share GET's operationId, which must be unique.
    router.add_api_route("/v1/reader/media/{name}", media, methods=["HEAD"], include_in_schema=False)

    return router
