"""Let clients written for OpenAI's audio API work without changes.

OpenAI model names map onto this server's configured default models, and
fields a client leaves out fall back to the configured TTS defaults.
"""

from __future__ import annotations

OPENAI_TTS_MODELS = frozenset({"tts-1", "tts-1-hd", "gpt-4o-mini-tts"})
OPENAI_STT_MODELS = frozenset({"whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"})
# Realtime model families; each name may carry a suffix such as "-mini" or a date
OPENAI_REALTIME_MODEL_FAMILIES = ("gpt-realtime", "gpt-4o-realtime-preview", "gpt-4o-mini-realtime-preview")


def is_openai_realtime_model(model: str | None) -> bool:
    """Whether ``model`` names an OpenAI realtime model (gpt-realtime, gpt-4o-realtime-preview-...)."""
    if not isinstance(model, str):
        return False
    return any(model == family or model.startswith(family + "-") for family in OPENAI_REALTIME_MODEL_FAMILIES)


def resolve_stt_model(model: str | None, settings) -> str:
    """Return the STT model to use, mapping OpenAI names to ``STT_MODEL``."""
    if not model or model in OPENAI_STT_MODELS:
        return settings.stt_model
    return model


def resolve_translation_model(model: str | None, settings) -> str:
    """Return the model for ``/v1/audio/translations``: OpenAI names and no model use ``STT_TRANSLATE_MODEL``,
    or ``STT_MODEL`` while that is empty."""
    if not model or model in OPENAI_STT_MODELS:
        return settings.stt_translate_model or settings.stt_model
    return model


def resolve_tts_model(model: str | None, settings) -> str:
    """Return the TTS model to use, mapping OpenAI speech and realtime names to ``TTS_MODEL``."""
    if not model or model in OPENAI_TTS_MODELS or is_openai_realtime_model(model):
        return settings.tts_model
    return model


def apply_speech_defaults(request, settings):
    """Fill omitted speech fields from ``TTS_*`` settings and map OpenAI model names.

    The configured voice is only applied when the model is also the configured
    one, since a voice ID belongs to a specific model.
    """
    sent = request.model_fields_set
    updates = {}
    uses_default_model = "model" not in sent or request.model in OPENAI_TTS_MODELS
    if uses_default_model:
        updates["model"] = settings.tts_model
        if "voice" not in sent:
            updates["voice"] = settings.tts_voice
    if "speed" not in sent:
        updates["speed"] = settings.tts_speed
    if "response_format" not in sent:
        updates["response_format"] = settings.tts_default_format
    return request.model_copy(update=updates) if updates else request
