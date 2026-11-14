"""Speech-to-text service helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from src.audio.preprocessing import preprocess_stt_audio
from src.backends.faster_whisper import NetworkModelPathError, refuse_network_model_path
from src.backends.parakeet import LanguageUnsupportedError, check_language
from src.diarization.pyannote_diarizer import (
    DiarizationSetupError,
    PyannoteDiarizer,
    attach_text_to_speakers,
    label_transcript_segments,
)
from src.formatters import format_transcription
from src.model_registry import TRANSLATING_STT_MODELS, WHISPER_LANGUAGES, stt_model_can_translate
from src.router import await_fetch
from src.utils.audio import (
    UnsupportedAudioError,
    check_input_sample_rate,
    convert_to_wav,
    get_suffix_from_content_type,
)
from src.utils.redact import redact_paths

logger = logging.getLogger("open-speech")

# faster-whisper decodes one file at a time (its inference lock). Requests queued behind a long
# decode wait in this pool, not in asyncio's default executor: there they would hold every
# thread it has, which every other request and to_thread call in the process shares. One
# thread decodes and one waits for the lock, so the next decode starts at once; the rest
# queue. Live streams and realtime have their own pools.
_stt_inference_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="stt-inference"
)


async def await_stt_fetch(backend_router, model: str) -> None:
    """Wait, outside the STT pool, for the files of a model that isn't loaded, fetching them on its first use.

    The backend fetches in its own pool, one fetch per model, so requests for a model that is downloading wait here
    holding no decode thread, and transcriptions on the loaded models go on. The decode's own fetch then finds the
    files.

    A request's model id that is a network (UNC) or device path raises NetworkModelPathError (a ValueError, code
    ``invalid_model``) before anything stats it, and the check of whether the id is a local folder runs off the
    event loop.
    """
    refuse_network_model_path(model)
    start_fetch = getattr(backend_router, "start_fetch", None)
    is_model_loaded = getattr(backend_router, "is_model_loaded", None)
    if not callable(start_fetch) or not callable(is_model_loaded) or is_model_loaded(model):
        return
    # start_fetch stats the id (a local folder is used as is): a slow drive must not hold every request.
    future = await asyncio.to_thread(start_fetch, model)
    if isinstance(future, concurrent.futures.Future):
        await await_fetch(future)


async def run_stt_inference(fn, /, *args, **kwargs):
    """Run a decode, which waits for the STT backend's inference lock, in the STT pool.

    A decode by a router's method (``router.transcribe``) on a model that isn't loaded waits for the model's files
    first, outside the pool (await_stt_fetch).
    """
    router = getattr(fn, "__self__", None)
    model = kwargs.get("model")
    if router is not None and isinstance(model, str):
        await await_stt_fetch(router, model)
    call = functools.partial(contextvars.copy_context().run, fn, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_stt_inference_executor, call)


# Diarization runs one pyannote pipeline at a time (_diarize_lock), and building a diarizer waits for the
# pipeline's load (_load_lock). Requests queued behind either wait in this pool, as decodes do in theirs: one
# thread runs and one waits, so the next starts at once.
_diarization_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="diarization")


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
VALID_RESPONSE_FORMATS = frozenset({"json", "verbose_json", "text", "srt", "vtt"})
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
# The registry's copy of faster_whisper.tokenizer._LANGUAGE_CODES, so that validating a
# request on the event loop does not import faster_whisper (and av, ctranslate2).
WHISPER_LANGUAGE_CODES = frozenset(WHISPER_LANGUAGES)
# ISO-639-1 codes Whisper spells differently: Norwegian Bokmal, Javanese, and the
# withdrawn codes Java locales still send for Hebrew, Indonesian and Yiddish.
WHISPER_LANGUAGE_ALIASES = {"nb": "no", "jv": "jw", "iw": "he", "in": "id", "ji": "yi"}


def _is_invalid_audio_error(exc: Exception) -> bool:
    """Identify decoder rejection separately from inference/runtime failures."""
    try:
        from av.error import InvalidDataError
    except ImportError:
        return False
    return isinstance(exc, InvalidDataError)


def suffix_from_filename(filename: str) -> str | None:
    """Extract audio suffix from filename."""
    for ext, suffix in EXTENSION_SUFFIXES.items():
        if filename.lower().endswith(ext):
            return suffix
    return None


def whisper_language_code(language: str | None) -> str | None:
    """Map a requested language to a code Whisper knows, or raise ValueError.

    Case and a region suffix are tolerated ("EN", "en-US" -> "en"); an empty value or
    "auto" means auto-detect. faster-whisper only checks the code after decoding.
    """
    if language is None or not language.strip():
        return None
    code = language.strip().lower().replace("_", "-")
    if code == "auto":
        return None
    if code not in WHISPER_LANGUAGE_CODES:
        code = code.split("-", 1)[0]
        code = WHISPER_LANGUAGE_ALIASES.get(code, code)
    if code not in WHISPER_LANGUAGE_CODES:
        raise ValueError(f"Invalid language {language!r}. Use an ISO-639-1 code such as 'en'")
    return code


def normalize_language(language: str | None) -> str | None:
    """Validate a request's language before any audio is read or decoded.

    An unknown code is the client's mistake: 400, as OpenAI answers it, rather than
    the 500 faster-whisper's ValueError would surface as.
    """
    try:
        return whisper_language_code(language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def refuse_unsupported_language(model: str, language: str | None) -> None:
    """A language the model doesn't transcribe (Parakeet is English-only) is the client's mistake: 400
    language_unsupported, naming a Whisper model to use, before the upload is read."""
    try:
        check_language(model, language)
    except LanguageUnsupportedError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "code": exc.code}) from exc


def _client_error(exc: Exception) -> HTTPException | None:
    """The 400 for a backend's refusal of a request (a language or a translation its model can't do), and the 503
    for a model whose files aren't downloaded while Hugging Face can't be reached, or that doesn't fit on the GPU."""
    code = getattr(exc, "code", None)
    if isinstance(exc, ValueError) and code in ("language_unsupported", "translation_unsupported"):
        return HTTPException(status_code=400, detail={"message": str(exc), "code": code})
    if code == "model_files_unavailable":
        return HTTPException(status_code=503, detail={"message": redact_paths(str(exc)), "code": code})
    if code == "insufficient_gpu_memory":
        # The GPU memory guard refused the model's first load (src/gpu_memory.py): its message says what to unload.
        return HTTPException(status_code=503, detail={"message": str(exc), "code": code})
    return None


def _server_error(exc: Exception) -> HTTPException:
    """The 500 for a failed decode: its message without the server's filesystem paths (the log has them)."""
    return HTTPException(status_code=500, detail=redact_paths(str(exc)))


def prepare_audio(audio_bytes: bytes, *, suffix: str, settings) -> bytes:
    """Convert audio to 16 kHz mono WAV and apply the configured STT preprocessing.

    Raises UnsupportedAudioError for a sample rate or a length that would decode to a
    huge buffer.
    """
    check_input_sample_rate(audio_bytes)
    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)
    return preprocess_stt_audio(
        audio_wav,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )


async def read_and_prepare_upload(
    *,
    file: UploadFile,
    settings,
    allow_filename_override: bool = False,
) -> bytes:
    """Read uploaded audio, validate it, convert to WAV, and preprocess it."""
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        read_size = min(UPLOAD_READ_CHUNK_BYTES, max_bytes - total_bytes + 1)
        chunk = await file.read(max(1, read_size))
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB",
            )
        chunks.append(chunk)

    audio_bytes = b"".join(chunks)
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = get_suffix_from_content_type(file.content_type)
    if allow_filename_override and suffix == ".ogg" and file.filename:
        ext_suffix = suffix_from_filename(file.filename)
        if ext_suffix:
            suffix = ext_suffix

    # ffmpeg can run for up to 30 s and denoising is CPU-bound; keep both off the loop.
    try:
        return await asyncio.to_thread(prepare_audio, audio_bytes, suffix=suffix, settings=settings)
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _diarize(audio_wav: bytes, diarizer_cls):
    """Run speaker diarization off the loop, mapping setup failures to 400."""
    loop = asyncio.get_running_loop()
    # The constructor may load the pyannote pipeline, so build it off the loop as well.
    # Only a setup error (missing dependency, refused gated download) is a 400; CUDA OOM
    # while moving the pipeline to the GPU is a RuntimeError too, but the server's.
    try:
        diarizer = await loop.run_in_executor(_diarization_executor, diarizer_cls)
    except DiarizationSetupError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {exc}")
    try:
        return await loop.run_in_executor(_diarization_executor, lambda: diarizer.diarize(audio_wav))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {exc}")


def _diarized_response(result: dict, turns, response_format: str, attach_speakers_fn):
    """Render a verbose_json backend result with speakers in the requested format.

    A long meeting has thousands of segments and turns, so callers run this off the loop.
    """
    text = result.get("text", "")
    transcript_segments = result.get("segments")
    if response_format == "json":
        speaker_turns = attach_speakers_fn(text, turns, transcript_segments)
        return JSONResponse({"text": text, "segments": speaker_turns})
    # verbose_json, text and subtitles keep Whisper's segments, each labelled with its
    # speaker; without timestamps the speaker turns stand in for them.
    segments = (
        label_transcript_segments(transcript_segments, turns)
        or attach_speakers_fn(text, turns, transcript_segments)
        or transcript_segments
        or []
    )
    if response_format == "verbose_json":
        return JSONResponse({**result, "segments": segments})
    content, content_type = format_transcription({**result, "segments": segments}, response_format)
    return PlainTextResponse(content, media_type=content_type)


def refuse_network_model(model: str) -> None:
    """A network (UNC) or device path as the model is the client's mistake: 400 invalid_model, before the upload is
    read or anything stats the id."""
    try:
        refuse_network_model_path(model)
    except NetworkModelPathError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "code": exc.code}) from exc


def validate_response_format(response_format: str) -> None:
    """Reject unsupported OpenAI-compatible transcription formats."""
    if response_format not in VALID_RESPONSE_FORMATS:
        allowed = ", ".join(sorted(VALID_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {allowed}",
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
    validate_response_format(response_format)
    refuse_network_model(model)

    if diarize and not settings.stt_diarize_enabled:
        raise HTTPException(status_code=400, detail="Diarization is disabled. Set STT_DIARIZE_ENABLED=true")
    language = normalize_language(language)
    refuse_unsupported_language(model, language)

    audio_wav = await read_and_prepare_upload(
        file=file,
        settings=settings,
        allow_filename_override=True,
    )

    # Diarization places speakers by Whisper's segment timestamps, whatever the format, and
    # srt/vtt are built from them. A plain json request gets the backend's {"text"} answer.
    if diarize or response_format in ("srt", "vtt", "verbose_json"):
        backend_format = "verbose_json"
    else:
        backend_format = response_format
    try:
        result = await run_stt_inference(
            backend_router.transcribe,
            audio=audio_wav,
            model=model,
            language=language,
            response_format=backend_format,
            temperature=temperature,
            prompt=prompt,
        )
    except Exception as exc:
        if _is_invalid_audio_error(exc):
            raise HTTPException(status_code=400, detail="Invalid or unsupported audio file") from exc
        if (refusal := _client_error(exc)) is not None:
            raise refusal from exc
        logger.exception("Transcription failed")
        raise _server_error(exc) from exc

    # Diarize before logging, so a transcription whose diarization fails (400 or 500) leaves no History entry.
    turns = await _diarize(audio_wav, diarizer_cls) if diarize else None

    if settings.os_history_enabled and raw_request.headers.get("x-history", "").lower() == "true":
        try:
            # Off the event loop: a write to the studio.db connection every thread shares.
            await asyncio.to_thread(
                history_manager.log_stt,
                model=model,
                input_filename=file.filename or "",
                result_text=result.get("text", ""),
            )
        except Exception:
            logger.exception("Failed to log STT history entry")

    if diarize:
        return await asyncio.to_thread(
            _diarized_response, result, turns, response_format, attach_speakers_fn
        )

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
    validate_response_format(response_format)
    refuse_network_model(model)
    # Turbo answers in the speaker's language and an English-only model only transcribes, so either would answer 200
    # with no translation. Refuse before reading or decoding.
    if not stt_model_can_translate(model):
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"{model} can't translate to English. Use {' or '.join(TRANSLATING_STT_MODELS)}, "
                    "or set STT_TRANSLATE_MODEL to one of them for whisper-1 and requests with no model."
                ),
                "code": "translation_unsupported",
            },
        )

    audio_wav = await read_and_prepare_upload(file=file, settings=settings)

    backend_format = "verbose_json" if response_format in ("srt", "vtt") else response_format
    try:
        result = await run_stt_inference(
            backend_router.translate,
            audio=audio_wav,
            model=model,
            response_format=backend_format,
            temperature=temperature,
            prompt=prompt,
        )
    except Exception as exc:
        if _is_invalid_audio_error(exc):
            raise HTTPException(status_code=400, detail="Invalid or unsupported audio file") from exc
        if (refusal := _client_error(exc)) is not None:
            raise refusal from exc
        logger.exception("Translation failed")
        raise _server_error(exc) from exc

    if response_format in ("text", "srt", "vtt"):
        content, content_type = format_transcription(result, response_format)
        return PlainTextResponse(content, media_type=content_type)

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)
