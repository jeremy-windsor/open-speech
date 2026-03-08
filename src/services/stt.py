"""Speech-to-text service helpers."""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.audio.preprocessing import preprocess_stt_audio
from src.diarization.pyannote_diarizer import PyannoteDiarizer, attach_text_to_speakers
from src.formatters import format_transcription
from src.utils.audio import convert_to_wav, get_suffix_from_content_type

logger = logging.getLogger("open-speech")


EXTENSION_SUFFIXES = {
    ".wav": ".wav",
    ".mp3": ".mp3",
    ".ogg": ".ogg",
    ".flac": ".flac",
    ".m4a": ".m4a",
    ".webm": ".webm",
    ".opus": ".ogg",
    ".aac": ".m4a",
}


def suffix_from_filename(filename: str) -> str | None:
    """Extract audio suffix from filename."""
    for ext, suffix in EXTENSION_SUFFIXES.items():
        if filename.lower().endswith(ext):
            return suffix
    return None


async def read_and_prepare_upload(
    *,
    file: UploadFile,
    settings,
    allow_filename_override: bool = False,
) -> bytes:
    """Read uploaded audio, validate it, convert to WAV, and preprocess it."""
    audio_bytes = await file.read()
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = get_suffix_from_content_type(file.content_type)
    if allow_filename_override and suffix == ".ogg" and file.filename:
        ext_suffix = suffix_from_filename(file.filename)
        if ext_suffix:
            suffix = ext_suffix

    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)
    return preprocess_stt_audio(
        audio_wav,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )


async def transcribe_request(
    *,
    file: UploadFile,
    model: str,
    language: str | None,
    prompt: str | None,
    response_format: str,
    temperature: float,
    diarize: bool,
    raw_request,
    settings,
    backend_router,
    history_manager,
    diarizer_cls=PyannoteDiarizer,
    attach_speakers_fn=attach_text_to_speakers,
):
    """Handle an OpenAI-compatible transcription request."""
    if diarize and not settings.stt_diarize_enabled:
        raise HTTPException(status_code=400, detail="Diarization is disabled. Set STT_DIARIZE_ENABLED=true")

    audio_wav = await read_and_prepare_upload(
        file=file,
        settings=settings,
        allow_filename_override=True,
    )

    backend_format = "verbose_json" if response_format in ("srt", "vtt", "json", "verbose_json") else response_format
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: backend_router.transcribe(
                audio=audio_wav,
                model=model,
                language=language,
                response_format=backend_format,
                temperature=temperature,
                prompt=prompt,
            ),
        )
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
        try:
            history_manager.log_stt(model=model, input_filename=file.filename or "", result_text=result.get("text", ""))
        except Exception:
            logger.exception("Failed to log STT history entry")

    if diarize:
        try:
            diarizer = diarizer_cls()
            diarized_segments = await loop.run_in_executor(None, lambda: diarizer.diarize(audio_wav))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Diarization failed: {exc}")
        text = result.get("text", "")
        return JSONResponse({"text": text, "segments": attach_speakers_fn(text, diarized_segments)})

    if response_format in ("text", "srt", "vtt"):
        content, content_type = format_transcription(result, response_format)
        return PlainTextResponse(content, media_type=content_type)

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)


async def translate_request(
    *,
    file: UploadFile,
    model: str,
    prompt: str | None,
    response_format: str,
    temperature: float,
    settings,
    backend_router,
):
    """Handle an OpenAI-compatible translation request."""
    audio_wav = await read_and_prepare_upload(file=file, settings=settings)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: backend_router.translate(
                audio=audio_wav,
                model=model,
                response_format=response_format,
                temperature=temperature,
                prompt=prompt,
            ),
        )
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(exc))

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)
