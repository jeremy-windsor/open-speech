"""Reader routes: import documents, render them to audio parts, and save position."""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path
from typing import Annotated, Callable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from src.reader.sources import parse_pasted, parse_upload
from src.services.tts import _read_upload_limited

MB = 1024 * 1024


class ProgressPayload(BaseModel):
    block_index: int = Field(ge=0)
    offset_ms: int = Field(default=0, ge=0)
    model: str | None = Field(default=None, max_length=200)
    voice: str | None = Field(default=None, max_length=200)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class RenderPayload(BaseModel):
    from_block: int = Field(ge=0)
    model: str = Field(min_length=1, max_length=200)
    voice: str = Field(min_length=1, max_length=200)


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
                document = await run_in_threadpool(parse_pasted, text, title=title if isinstance(title, str) else None)
            elif content_type.startswith("multipart/form-data"):
                form = await request.form()
                upload = form.get("file")
                if not isinstance(upload, UploadFile) or not upload.filename:
                    raise HTTPException(status_code=422, detail="Upload a file in the 'file' field")
                title = form.get("title")
                data = await _read_upload_limited(
                    upload, max_bytes, too_large_detail=f"File is larger than {settings.reader_max_upload_mb} MB"
                )
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
        return JSONResponse(store.get_document(document_id, include_blocks=False), status_code=201 if created else 200)

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
        get_renderer().forget_document(document_id)
        paths = get_store().delete_document(document_id)
        if paths is None:
            raise HTTPException(status_code=404, detail="Document not found")
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
        try:
            parts = get_renderer().request(document_id, payload.from_block, payload.model, payload.voice)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if parts is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"parts": parts}

    @router.get("/v1/reader/documents/{document_id}/parts")
    def parts(
        document_id: str,
        from_block: Annotated[int, Query(ge=0)],
        model: Annotated[str, Query(min_length=1, max_length=200)],
        voice: Annotated[str, Query(min_length=1, max_length=200)],
    ):
        try:
            planned = get_renderer().status(document_id, from_block, model, voice)
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
        part = store.get_part(part_id)
        if part is not None and not Path(part["path"]).is_file():
            # Forget it so the next render request makes the part again.
            store.forget_part(part_id)
            part = None
        if part is None:
            raise HTTPException(status_code=404, detail="Audio part not found")
        store.touch_part(part_id)
        return FileResponse(part["path"], media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})

    return router
