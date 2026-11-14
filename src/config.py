"""Application configuration via environment variables.

Naming convention:
  OS_*   — Harness-level / shared settings
  STT_*  — Speech-to-text specific
  TTS_*  — Text-to-speech specific

Old STT_* names for server settings still work (backwards compat) but log
deprecation warnings on startup.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# OS_LOG_LEVEL's choices, as logging names them.
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

# AI assistant field limits, shared by the OS_ASSISTANT_* defaults below and the AI presets (src/assistant/presets.py).
ASSISTANT_THINKING_MODES = ("default", "on", "off")
ASSISTANT_REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")
ASSISTANT_MAX_TOKENS_FIELDS = ("max_tokens", "max_completion_tokens")
ASSISTANT_PROMPT_MAX_CHARS = 8000
ASSISTANT_LIMITS: dict[str, tuple[float, float]] = {
    "thinking_budget": (0, 1_000_000),
    "max_tokens": (0, 32768),
    "temperature": (0, 2),
    "top_p": (0, 1),
    "timeout_s": (5, 3600),
    "history_turns": (0, 100),
    "chunk_chars": (1000, 200000),
    "filler_after_ms": (0, 30000),
    "filler_every_ms": (0, 30000),
    "filler_max": (0, 10),
}
# Fields the extra body may not replace: Open Speech sets them itself (a voice turn streams).
ASSISTANT_RESERVED_BODY_FIELDS = ("messages", "stream")
ASSISTANT_URL_MAX_CHARS = 2000
ASSISTANT_API_KEY_MAX_CHARS = 4096
# Objects and lists an extra body may nest, the object itself included (chat_template_kwargs needs 2). The saved
# connection is deep-copied, which recurses about two frames per level.
ASSISTANT_EXTRA_BODY_MAX_DEPTH = 32


def check_assistant_api_key(value: str, name: str = "OS_ASSISTANT_API_KEY") -> str:
    """The AI server's API key, stripped, or ValueError naming ``name`` that never quotes the key.

    One rule for OS_ASSISTANT_API_KEY and a key saved through /api/assistant/connection: it is sent as
    ``Authorization: Bearer <key>``, so it must be printable ASCII of up to ASSISTANT_API_KEY_MAX_CHARS characters. A
    key read with a trailing line break or tab (a secret file, a YAML block scalar in Compose) is stripped, not sent:
    h11 would refuse the header and quote it in its error.
    """
    key = value.strip()
    if len(key) > ASSISTANT_API_KEY_MAX_CHARS or not (key.isascii() and key.isprintable()):
        raise ValueError(f"{name} must be printable ASCII of up to {ASSISTANT_API_KEY_MAX_CHARS} characters")
    return key


def check_assistant_url(value: str, name: str = "OS_ASSISTANT_URL") -> str:
    """The AI server's base URL, normalised (no trailing slash; "" when empty), or ValueError naming ``name``.

    The connection guards apply to every URL, from .env or saved through /api/assistant/connection: http or https only,
    with a host and a valid port; no user name or password in it (the key has its own setting); no query or fragment,
    because /chat/completions and /models are appended to it; and a URL httpx can parse, since it parses it again for
    each request (an IP literal such as 192.168.1.300 or an IDNA name that urlsplit takes).
    """
    value = value.strip()
    if not value:
        return ""
    example = "such as http://ai-server.lan:8080/v1"
    if len(value) > ASSISTANT_URL_MAX_CHARS or not value.isprintable() or any(ch.isspace() for ch in value):
        raise ValueError(f"{name} must be one base URL of up to {ASSISTANT_URL_MAX_CHARS} characters, {example}")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{name} must be an http:// or https:// base URL {example}")
    if "@" in parts.netloc:
        raise ValueError(f"{name} can't hold a user name or password; the API key has its own setting")
    if parts.query or parts.fragment or "?" in value or "#" in value:
        raise ValueError(f"{name} can't have a query or a fragment: /chat/completions is appended to it")
    try:
        _ = parts.port  # raises ValueError for a port out of range
    except ValueError:
        raise ValueError(f"{name} has a port that isn't valid") from None
    value = value.rstrip("/")
    try:
        httpx.URL(value)
    except httpx.InvalidURL as exc:  # not a ValueError: it would escape every request as a plain exception
        raise ValueError(f"{name} isn't a valid URL ({exc}): give a base URL {example}") from None
    return value


def _nests_deeper_than(value, limit: int) -> bool:
    """Whether dicts and lists nest more than ``limit`` levels in ``value``; checked without recursion."""
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, (list, tuple)):
            children = item
        else:
            continue
        if depth > limit:
            return True
        pending.extend((child, depth + 1) for child in children)
    return False


def parse_assistant_extra_body(value, name: str = "OS_ASSISTANT_EXTRA_BODY") -> dict:
    """The extra body as a dict, from JSON text ("" is none) or an object, or ValueError naming ``name``.

    It may nest at most ASSISTANT_EXTRA_BODY_MAX_DEPTH levels, and it must be JSON that can be written strictly: no NaN,
    Infinity or -Infinity, which Python's json module reads but RFC 8259 (section 6) doesn't allow, and which the API's
    answers and httpx's request bodies refuse.
    """
    too_deep = f"{name} can nest objects and lists at most {ASSISTANT_EXTRA_BODY_MAX_DEPTH} levels deep"
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            value = json.loads(value)
        except RecursionError:  # deeper than the json module itself takes
            raise ValueError(too_deep) from None
        except ValueError as exc:
            raise ValueError(f"{name} must be a JSON object such as {{\"top_k\": 20}}: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object such as {{\"top_k\": 20}}, not {type(value).__name__}")
    if _nests_deeper_than(value, ASSISTANT_EXTRA_BODY_MAX_DEPTH):
        raise ValueError(too_deep)
    try:
        # Written as the store and httpx write it: UTF-8 with no escapes, which a lone surrogate (half of a UTF-16
        # pair, such as half an emoji) can't be.
        json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:  # a ValueError: before the NaN case
        raise ValueError(
            f"{name} has a lone surrogate (half of a UTF-16 pair, such as half an emoji), which can't be stored or "
            "sent to the AI; send whole characters"
        ) from None
    except ValueError:
        raise ValueError(f"{name} can't hold NaN or Infinity: JSON has no such numbers") from None
    except TypeError:
        raise ValueError(f"{name} must hold only JSON values: objects, lists, text, numbers, booleans, null") from None
    reserved = sorted(set(value) & set(ASSISTANT_RESERVED_BODY_FIELDS))
    if reserved:
        raise ValueError(f"{name} can't set {', '.join(reserved)}: Open Speech sets those fields itself")
    return value


def _one_of(env_name: str, value: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        shown = ", ".join(repr(choice) for choice in allowed)
        raise ValueError(f"{env_name} must be one of {shown}, not {value!r}")
    return value

# Map: new_env_name -> old_env_name
_DEPRECATED_ENV_MAP: dict[str, str] = {
    # Harness/shared transport settings
    "OS_PORT": "STT_PORT",
    "OS_HOST": "STT_HOST",
    "OS_API_KEY": "STT_API_KEY",
    "OS_CORS_ORIGINS": "STT_CORS_ORIGINS",
    "OS_TRUST_PROXY": "STT_TRUST_PROXY",
    "OS_MAX_UPLOAD_MB": "STT_MAX_UPLOAD_MB",
    "OS_RATE_LIMIT": "STT_RATE_LIMIT",
    "OS_RATE_LIMIT_BURST": "STT_RATE_LIMIT_BURST",
    "OS_SSL_ENABLED": "STT_SSL_ENABLED",
    "OS_SSL_CERTFILE": "STT_SSL_CERTFILE",
    "OS_SSL_KEYFILE": "STT_SSL_KEYFILE",
    # Lifecycle
    "OS_MODEL_TTL": "STT_MODEL_TTL",
    "OS_MAX_LOADED_MODELS": "STT_MAX_LOADED_MODELS",
    # Streaming
    "OS_STREAM_CHUNK_MS": "STT_STREAM_CHUNK_MS",
    "OS_STREAM_ENDPOINTING_MS": "STT_STREAM_ENDPOINTING_MS",
    "OS_STREAM_MAX_CONNECTIONS": "STT_STREAM_MAX_CONNECTIONS",
    # STT renames
    "STT_MODEL": "STT_DEFAULT_MODEL",
    # TTS renames
    "TTS_MODEL": "TTS_DEFAULT_MODEL",
    "TTS_VOICE": "TTS_DEFAULT_VOICE",
    "TTS_SPEED": "TTS_DEFAULT_SPEED",
}


def _check_deprecated_env_vars() -> dict[str, str]:
    """Check for deprecated env var names and return warnings.

    If old name is set and new name isn't, copies old -> new in os.environ
    so pydantic picks it up. Returns list of warning messages.
    """
    warnings: dict[str, str] = {}
    for new_name, old_name in _DEPRECATED_ENV_MAP.items():
        old_val = os.environ.get(old_name)
        if old_val is not None:
            new_val = os.environ.get(new_name)
            # docker-compose.yml passes an unset new name through as "" (OS_CORS_ORIGINS=${OS_CORS_ORIGINS:-}).
            if not new_val:
                # Use old value
                os.environ[new_name] = old_val
            warnings[old_name] = new_name
    return warnings


def log_deprecation_warnings(warnings: dict[str, str]) -> None:
    """Log deprecation warnings for old env var names."""
    for old_name, new_name in sorted(warnings.items()):
        logger.warning(
            "Deprecated env var '%s' — use '%s' instead. "
            "Old names will be removed in a future release.",
            old_name,
            new_name,
        )


# Map: removed env name -> why setting it does nothing
_OBSOLETE_ENV_VARS: dict[str, str] = {
    "OS_HISTORY_RETAIN_AUDIO": "history entries never store audio files",
    "OS_HISTORY_MAX_MB": "history entries never store audio files; OS_HISTORY_MAX_ENTRIES bounds history",
}


def _check_obsolete_env_vars() -> list[str]:
    """Return the removed env var names that are set; an empty value counts as unset."""
    return sorted(name for name in _OBSOLETE_ENV_VARS if os.environ.get(name))


def log_obsolete_warnings(names: list[str]) -> None:
    """Log that removed env vars are ignored."""
    for name in names:
        logger.warning("Env var '%s' has no effect and is ignored: %s.", name, _OBSOLETE_ENV_VARS[name])


# Check before Settings is instantiated so values are in os.environ
_deprecation_warnings = _check_deprecated_env_vars()
_obsolete_env_vars = _check_obsolete_env_vars()


class Settings(BaseSettings):
    """Open Speech settings — unified harness, STT, and TTS configuration."""

    # ── Harness / transports (OS_ prefix) ────────────────────────────────────
    os_port: int = 8100
    os_host: str = "0.0.0.0"
    os_api_key: str = ""
    os_auth_required: bool = False
    os_auth_failure_limit: int = 10  # Wrong API keys per IP per minute before a 60 s lockout; 0 = off
    os_cors_origins: str = ""  # Empty = same-origin only; "*" or a comma list reopens CORS
    os_ws_allowed_origins: str = ""  # Browser Origins besides this server's own; "*" = any
    os_allowed_hosts: str = ""  # Host names accepted besides IPs and LAN names; "*" = any
    os_security_headers: bool = True
    os_api_docs: bool = False  # Swagger UI at /docs and ReDoc at /redoc; they load scripts from a CDN
    # DEBUG, INFO, WARNING or ERROR, in any case: the app's log level, and uvicorn's (src/main.py)
    os_log_level: str = "INFO"
    os_trust_proxy: bool = False
    os_trusted_proxy_hops: int = 1  # Proxies that append to X-Forwarded-For
    os_max_upload_mb: int = 100
    os_rate_limit: int = 0
    os_rate_limit_burst: int = 0
    os_ssl_enabled: bool = True
    os_ssl_certfile: str = ""
    os_ssl_keyfile: str = ""
    os_tls_extra_sans: str = ""
    os_voice_library_path: str = "/home/openspeech/data/voices"
    os_voice_library_max_count: int = 100  # Max stored voices; 0 = unlimited
    os_voice_library_max_seconds: int = 60  # Max reference duration; 0 = unlimited
    os_studio_db_path: str = "/home/openspeech/data/studio.db"
    os_history_enabled: bool = True
    os_history_max_entries: int = 1000
    os_effects_enabled: bool = True
    os_conversations_dir: str = "/home/openspeech/data/conversations"
    os_composer_dir: str = "/home/openspeech/data/composer"
    os_batch_workers: int = 2
    os_batch_max_pending: int = 10   # max queued+running jobs before returning 429
    os_batch_max_total_mb: int = 500  # max aggregate upload size per request in MB

    # ── Wyoming Protocol ───────────────────────────────────────────────────────
    os_wyoming_enabled: bool = False
    os_wyoming_host: str = "127.0.0.1"
    os_wyoming_port: int = 10400
    # The port has no authentication. Home Assistant opens one short-lived connection per request.
    # Close a connection whose next event takes longer than this to arrive in full, or that
    # stops reading replies this long; 0 = never
    os_wyoming_idle_timeout_s: int = 120
    # At this many open connections, a new one replaces the longest idle, or is closed at once
    # while all are mid-request; 0 = unlimited
    os_wyoming_max_connections: int = 32

    # ── Realtime API ─────────────────────────────────────────────────────────
    os_realtime_enabled: bool = True
    os_realtime_max_buffer_mb: int = 50
    os_realtime_idle_timeout_s: int = 120
    os_realtime_barge_in_max_ms: int = 1500
    os_realtime_audio_lead_ms: int = 1000

    # ── AI assistant (a local OpenAI-compatible server) ──────────────────────
    # Connection: the URL, key, extra body, TLS, token-field name, template switch and default model start here,
    # and Settings > AI assistant (/api/assistant/connection) can change them; no preset can. The key is never
    # returned or logged. src/assistant/connection.py merges the saved ones over these.
    # true: .env only, as before; the saved connection is ignored and can't be changed.
    os_assistant_connection_locked: bool = False
    os_assistant_url: str = ""  # Base URL ending in /v1; /chat/completions is appended. Empty = off
    os_assistant_api_key: str = ""  # Sent as Authorization: Bearer; never logged or returned
    os_assistant_extra_body: str = ""  # A JSON object merged into every request
    os_assistant_verify_tls: bool = True
    os_assistant_max_tokens_field: str = "max_tokens"  # or max_completion_tokens
    # Also send the thinking controls as chat-template variables (chat_template_kwargs), for llama.cpp-style
    # llama.cpp-style servers whose template reads enable_thinking and reasoning_effort. Off: OpenAI refuses the field.
    os_assistant_template_kwargs: bool = False
    # Stream replies: a voice turn asks with stream: true and speaks each sentence as it arrives. false asks for the
    # whole reply, for a server that can't stream; Ask AI and book jobs never stream.
    os_assistant_stream: bool = True
    # Defaults for the AI presets: a preset field left empty falls back to these.
    os_assistant_model: str = ""
    os_assistant_system_prompt: str = Field("", max_length=ASSISTANT_PROMPT_MAX_CHARS)  # Empty = the built-in
    os_assistant_thinking: str = "default"  # default sends nothing; on/off send chat_template_kwargs
    os_assistant_reasoning_effort: str = ""  # Empty sends nothing; low keeps voice replies quick
    # Sent as reasoning_budget_tokens (the llama.cpp-style name); 0 = none sent
    os_assistant_thinking_budget: int = Field(0, ge=0, le=ASSISTANT_LIMITS["thinking_budget"][1])
    os_assistant_max_tokens: int = Field(0, ge=0, le=ASSISTANT_LIMITS["max_tokens"][1])  # 0 = no limit sent
    os_assistant_temperature: float | None = Field(None, ge=0, le=ASSISTANT_LIMITS["temperature"][1])
    os_assistant_top_p: float | None = Field(None, ge=0, le=ASSISTANT_LIMITS["top_p"][1])
    os_assistant_timeout_s: float = Field(120, ge=ASSISTANT_LIMITS["timeout_s"][0], le=ASSISTANT_LIMITS["timeout_s"][1])
    os_assistant_job_timeout_s: float = Field(
        600, ge=ASSISTANT_LIMITS["timeout_s"][0], le=ASSISTANT_LIMITS["timeout_s"][1]
    )
    os_assistant_history_turns: int = Field(20, ge=0, le=ASSISTANT_LIMITS["history_turns"][1])
    os_assistant_chunk_chars: int = Field(
        12000, ge=ASSISTANT_LIMITS["chunk_chars"][0], le=ASSISTANT_LIMITS["chunk_chars"][1]
    )
    os_assistant_show_thinking: bool = True
    os_assistant_speak_errors: bool = True

    @field_validator("os_log_level")
    @classmethod
    def _log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in LOG_LEVELS:
            # The value isn't quoted: a value pasted on the wrong line can be a key.
            raise ValueError("OS_LOG_LEVEL must be one of DEBUG, INFO, WARNING or ERROR, in any case")
        return level

    @field_validator("os_gpu_max_big_models")
    @classmethod
    def _gpu_max_big_models(cls, value: str) -> str:
        value = str(value).strip().lower()
        if value != "auto" and not value.isdigit():
            raise ValueError(f"OS_GPU_MAX_BIG_MODELS must be 'auto' or a whole number (0 = no cap), not {value!r}")
        return value

    @field_validator("os_assistant_url")
    @classmethod
    def _assistant_url(cls, value: str) -> str:
        return check_assistant_url(value)

    @field_validator("os_assistant_api_key")
    @classmethod
    def _assistant_api_key(cls, value: str) -> str:
        return check_assistant_api_key(value)

    @field_validator("os_assistant_extra_body")
    @classmethod
    def _assistant_extra_body(cls, value: str) -> str:
        if not value.strip():
            return ""
        parse_assistant_extra_body(value)
        return value

    @field_validator("os_assistant_max_tokens_field")
    @classmethod
    def _assistant_max_tokens_field(cls, value: str) -> str:
        return _one_of("OS_ASSISTANT_MAX_TOKENS_FIELD", value, ASSISTANT_MAX_TOKENS_FIELDS)

    @field_validator("os_assistant_thinking")
    @classmethod
    def _assistant_thinking(cls, value: str) -> str:
        return _one_of("OS_ASSISTANT_THINKING", value, ASSISTANT_THINKING_MODES)

    @field_validator("os_assistant_reasoning_effort")
    @classmethod
    def _assistant_reasoning_effort(cls, value: str) -> str:
        return _one_of("OS_ASSISTANT_REASONING_EFFORT", value, ASSISTANT_REASONING_EFFORTS)

    # ── Model Lifecycle (OS_ prefix) ─────────────────────────────────────────
    os_model_ttl: int = 300
    os_max_loaded_models: int = 0
    # TTL and LRU eviction never unload STT_MODEL or TTS_MODEL; an explicit unload, or the
    # explicit load of another model of the same type, still does. Pinned defaults are loaded on
    # top of OS_MAX_LOADED_MODELS, which then caps the other models.
    os_pin_default_models: bool = True
    # The GPU memory guard (src/gpu_memory.py): before a model loads onto the GPU, the card's free memory is checked
    # against the model's size; idle models that aren't pinned are unloaded to make room, else the load is refused.
    os_gpu_memory_guard: bool = True
    # GPU memory kept free on top of a model's size, in MiB: what a desktop and the CUDA libraries take as they run.
    os_gpu_headroom_mb: int = Field(600, ge=0)
    # Big models (2 GiB or more of GPU memory) loaded at once besides the pinned defaults: "auto" takes it from the
    # card (under 11 GiB: 1, under 22 GiB: 2, else no cap), a number sets it, and 0 means no cap.
    os_gpu_max_big_models: str = "auto"

    # ── Streaming (OS_ prefix) ───────────────────────────────────────────────
    os_stream_chunk_ms: int = 100
    os_stream_endpointing_ms: int = 300
    os_stream_max_connections: int = 10

    # ── VAD (Voice Activity Detection) ───────────────────────────────────────
    stt_vad_enabled: bool = True
    stt_vad_threshold: float = 0.5
    stt_vad_min_speech_ms: int = 250
    stt_vad_silence_ms: int = 800

    # ── STT ──────────────────────────────────────────────────────────────────
    stt_model: str = "Systran/faster-whisper-base"
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_model_dir: str | None = None
    stt_preload_models: str = ""
    # The model /v1/audio/translations uses for whisper-1 or no model; empty = STT_MODEL.
    stt_translate_model: str = ""
    # The most GPU memory Parakeet's ONNX Runtime arena may take, in MB (0 = no cap). Past the card's memory,
    # Windows (WDDM) pages the other loaded models out instead of failing, and their next request is seconds slower.
    stt_parakeet_gpu_mem_limit_mb: int = Field(2304, ge=0)

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_enabled: bool = True
    tts_model: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_device: str | None = None
    tts_max_input_length: int = 4096
    tts_default_format: str = "mp3"
    tts_speed: float = 1.0
    tts_preload_models: str = ""
    # Load TTS_MODEL in the background at startup when its provider is installed.
    tts_preload_default: bool = True
    tts_voices_config: str = ""
    tts_external_providers: str = ""
    tts_cache_enabled: bool = False
    tts_cache_max_mb: int = 500
    tts_cache_dir: str = "/var/lib/open-speech/cache"
    tts_trim_silence: bool = True
    tts_normalize_output: bool = True
    tts_pronunciation_dict: str = ""
    tts_live_enabled: bool = True
    tts_live_max_connections: int = 1
    tts_live_idle_timeout_s: int = 300
    tts_live_max_buffer_chars: int = 8192
    tts_live_max_pending_segments: int = 3
    tts_live_max_unacked_seconds: float = 15.0
    tts_live_audio_frame_ms: int = 50
    tts_live_segment_idle_ms: int = 250
    tts_live_max_segment_chars: int = 200
    tts_live_max_segment_words: int = 20
    tts_live_sentence_idle_ms: int = 1500
    tts_live_sentence_max_chars: int = 500
    tts_live_sentence_max_words: int = 80
    tts_live_max_pause_s: int = 900
    tts_live_playback_stall_timeout_s: int = 30
    tts_live_shutdown_timeout_s: int = 5

    # ── Reader (books and documents read aloud) ──────────────────────────────
    reader_enabled: bool = True
    reader_cache_dir: str = "/home/openspeech/data/reader"
    reader_cache_max_mb: int = 2000
    reader_max_upload_mb: int = 50
    reader_max_document_chars: int = 5_000_000
    reader_lookahead_parts: int = 3
    reader_part_max_chars: int = 450
    reader_media_url_ttl_s: int = 21600

    # ── Diarization / Audio Processing ───────────────────────────────────────
    stt_diarize_enabled: bool = False
    stt_noise_reduce: bool = False
    stt_normalize: bool = True

    @property
    def tts_effective_device(self) -> str:
        return self.tts_device or self.stt_device

    @property
    def assistant_enabled(self) -> bool:
        """The AI assistant is on when its effective server URL is set: OS_ASSISTANT_URL, or a URL saved through
        /api/assistant/connection (Settings > AI assistant) unless OS_ASSISTANT_CONNECTION_LOCKED is true."""
        from src.assistant.connection import effective_url  # the package imports this module

        return bool(effective_url(self))

    @property
    def assistant_extra_body(self) -> dict:
        """OS_ASSISTANT_EXTRA_BODY parsed (validated as a JSON object at startup); {} when unset.

        Parsed with the startup check's rules, so a value set some other way that breaks them raises ValueError;
        the connection then uses no extra body, with a warning."""
        return parse_assistant_extra_body(self.os_assistant_extra_body)

    # ── Backwards-compat aliases (read-only properties) ──────────────────────
    # These let code that references settings.stt_port etc. still work.
    @property
    def stt_port(self) -> int:
        return self.os_port

    @property
    def stt_host(self) -> str:
        return self.os_host

    @property
    def stt_api_key(self) -> str:
        return self.os_api_key

    @property
    def stt_cors_origins(self) -> str:
        return self.os_cors_origins

    @property
    def stt_trust_proxy(self) -> bool:
        return self.os_trust_proxy

    @property
    def stt_ws_allowed_origins(self) -> str:
        return self.os_ws_allowed_origins

    @property
    def voice_library_path(self) -> str:
        return self.os_voice_library_path

    @property
    def voice_library_max_count(self) -> int:
        return self.os_voice_library_max_count

    @property
    def voice_library_max_seconds(self) -> int:
        return self.os_voice_library_max_seconds

    @property
    def stt_max_upload_mb(self) -> int:
        return self.os_max_upload_mb

    @property
    def stt_rate_limit(self) -> int:
        return self.os_rate_limit

    @property
    def stt_rate_limit_burst(self) -> int:
        return self.os_rate_limit_burst

    @property
    def stt_ssl_enabled(self) -> bool:
        return self.os_ssl_enabled

    @property
    def stt_ssl_certfile(self) -> str:
        return self.os_ssl_certfile

    @property
    def stt_ssl_keyfile(self) -> str:
        return self.os_ssl_keyfile

    @property
    def stt_model_ttl(self) -> int:
        return self.os_model_ttl

    @property
    def stt_max_loaded_models(self) -> int:
        return self.os_max_loaded_models

    @property
    def stt_stream_chunk_ms(self) -> int:
        return self.os_stream_chunk_ms

    @property
    def stt_stream_endpointing_ms(self) -> int:
        return self.os_stream_endpointing_ms

    @property
    def stt_stream_max_connections(self) -> int:
        return self.os_stream_max_connections

    # Old STT_DEFAULT_MODEL / TTS_DEFAULT_MODEL / TTS_DEFAULT_VOICE / TTS_DEFAULT_SPEED
    @property
    def stt_default_model(self) -> str:
        return self.stt_model

    @property
    def tts_default_model(self) -> str:
        return self.tts_model

    @property
    def tts_default_voice(self) -> str:
        return self.tts_voice

    @property
    def tts_default_speed(self) -> float:
        return self.tts_speed

    # An empty variable counts as unset, as in _check_deprecated_env_vars: docker-compose.yml passes
    # OS_RATE_LIMIT=${OS_RATE_LIMIT:-} through empty so the default here (or the legacy name) applies.
    # A refused value is never printed in the startup error, only its setting and the reason: it can be a secret,
    # such as OS_ASSISTANT_API_KEY.
    model_config = {
        "env_prefix": "", "case_sensitive": False, "extra": "ignore", "env_ignore_empty": True,
        "hide_input_in_errors": True,
    }


settings = Settings()


def hf_hub_cache_dir() -> Path:
    """The Hugging Face hub cache that downloads without an explicit directory go to.

    Resolved from the environment as huggingface_hub.constants does at import (HF_HUB_CACHE,
    else $HF_HOME/hub, with HF_HOME defaulting to $XDG_CACHE_HOME/huggingface, '~' and $VARs
    expanded), but on every call, so the cache scans see the directory a download would use.
    """
    def expand(path: str) -> str:
        return os.path.expandvars(os.path.expanduser(path))

    cache_home = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    hf_home = expand(os.getenv("HF_HOME", os.path.join(cache_home, "huggingface")))
    legacy_cache = os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
    return Path(expand(os.getenv("HF_HUB_CACHE", legacy_cache)))


# Log deprecation warnings after settings are created
if _deprecation_warnings:
    log_deprecation_warnings(_deprecation_warnings)
if _obsolete_env_vars:
    log_obsolete_warnings(_obsolete_env_vars)
