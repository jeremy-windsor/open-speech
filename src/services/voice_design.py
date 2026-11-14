"""Voice design: a short sample spoken in a voice made from a description.

The web app saves a sample it likes as a recording with its words as the transcript, so it then clones like any
other recording, with any cloning model.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from src.audio.postprocessing import process_tts_chunks
from src.services.tts import (
    _concatenated,
    _provider_missing,
    _sample_rate_for_model,
    provider_http_error,
    run_on_provider,
    validate_tts_feature_support,
    validate_tts_input_limit,
)
from src.tts.external import ExternalProviderError
from src.tts.pipeline import encode_audio
from src.tts.router import NoTTSBackendsError

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_TEXT = (
    "Hello! This is the voice you described. I can read your stories, your notes and your messages aloud."
)
MAX_DESCRIPTION_CHARS = 1000
MAX_SAMPLE_CHARS = 500


class VoiceDesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)
    text: str = Field(default=DEFAULT_SAMPLE_TEXT, min_length=1, max_length=MAX_SAMPLE_CHARS)
    language: str | None = None


async def design_voice_sample_response(*, request: VoiceDesignRequest, settings, tts_router) -> Response:
    """Speak ``request.text`` in a voice made from ``request.description``, as WAV at the model's own rate."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")
    description = request.description.strip()
    text = request.text.strip()
    if not description:
        raise HTTPException(status_code=400, detail="Describe the voice")
    if not text:
        raise HTTPException(status_code=400, detail="Sample text is empty")
    try:
        feature_error = await asyncio.to_thread(
            validate_tts_feature_support,
            tts_router=tts_router,
            model_id=request.model,
            voice_design=description,
            language=request.language,
        )
        if feature_error is None:
            feature_error = await asyncio.to_thread(
                validate_tts_input_limit, tts_router=tts_router, model_id=request.model, text=text
            )
        sample_rate = await asyncio.to_thread(_sample_rate_for_model, tts_router=tts_router, model_id=request.model)
    except ExternalProviderError as exc:
        raise provider_http_error(exc) from exc
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    if feature_error:
        raise HTTPException(status_code=400, detail=feature_error)

    def synthesize():
        return _concatenated(process_tts_chunks(
            tts_router.synthesize(
                text=text, model=request.model, voice="designed", speed=1.0, lang_code=request.language,
                voice_design=description,
            ),
            trim=settings.tts_trim_silence,
            normalize=settings.tts_normalize_output,
        ))

    try:
        samples = await run_on_provider(tts_router, request.model, synthesize)
        audio = await asyncio.to_thread(encode_audio, iter([samples]), fmt="wav", sample_rate=sample_rate)
    except ExternalProviderError as exc:
        raise provider_http_error(exc) from exc
    except NoTTSBackendsError as exc:
        raise _provider_missing(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Voice design failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not samples.size:
        raise HTTPException(status_code=502, detail="The model returned no audio")
    return Response(content=audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})
