"""Application configuration via environment variables.

Naming convention (Phase 3a):
  OS_*   — Server-level / shared settings
  STT_*  — Speech-to-text specific
  TTS_*  — Text-to-speech specific

Old STT_* names for server settings still work (backwards compat) but log
deprecation warnings on startup.
"""

from __future__ import annotations

import logging
import os

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Map: new_env_name -> old_env_name
_DEPRECATED_ENV_MAP: dict[str, str] = {
    # Server
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
    "OS_STREAM_VAD_THRESHOLD": "STT_STREAM_VAD_THRESHOLD",
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
            if new_val is None:
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


# Check before Settings is instantiated so values are in os.environ
_deprecation_warnings = _check_deprecated_env_vars()


class Settings(BaseSettings):
    """Open Speech settings — unified server, STT, and TTS configuration."""

    # ── Server (OS_ prefix) ──────────────────────────────────────────────────
    os_port: int = 8100
    os_host: str = "0.0.0.0"
    os_api_key: str = ""
    os_auth_required: bool = False
    os_cors_origins: str = "*"
    os_ws_allowed_origins: str = ""
    os_trust_proxy: bool = False
    os_max_upload_mb: int = 100
    os_rate_limit: int = 0
    os_rate_limit_burst: int = 0
    os_ssl_enabled: bool = True
    os_ssl_certfile: str = ""
    os_ssl_keyfile: str = ""

    # ── Wyoming Protocol ───────────────────────────────────────────────────────
    os_wyoming_enabled: bool = False
    os_wyoming_host: str = "127.0.0.1"
    os_wyoming_port: int = 10400

    # ── Realtime API ─────────────────────────────────────────────────────────
    os_realtime_enabled: bool = True
    os_realtime_max_buffer_mb: int = 50
    os_realtime_idle_timeout_s: int = 120

    # ── Model Lifecycle (OS_ prefix) ─────────────────────────────────────────
    os_model_ttl: int = 300
    os_max_loaded_models: int = 0

    # ── Streaming (OS_ prefix) ───────────────────────────────────────────────
    os_stream_chunk_ms: int = 2000
    os_stream_vad_threshold: float = 0.5
    os_stream_endpointing_ms: int = 300
    os_stream_max_connections: int = 10

    # ── VAD (Voice Activity Detection) ───────────────────────────────────────
    stt_vad_enabled: bool = True
    stt_vad_threshold: float = 0.5
    stt_vad_min_speech_ms: int = 250
    stt_vad_silence_ms: int = 800

    # ── STT ──────────────────────────────────────────────────────────────────
    stt_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_model_dir: str | None = None
    stt_preload_models: str = ""

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_enabled: bool = True
    tts_model: str = "kokoro"
    tts_voice: str = "af_heart"
    tts_device: str | None = None
    tts_max_input_length: int = 4096
    tts_default_format: str = "mp3"
    tts_speed: float = 1.0
    tts_preload_models: str = ""
    tts_voices_config: str = ""
    tts_cache_enabled: bool = False
    tts_cache_max_mb: int = 500
    tts_cache_dir: str = "/var/lib/open-speech/cache"
    tts_trim_silence: bool = True
    tts_normalize_output: bool = True
    tts_pronunciation_dict: str = ""
    tts_qwen3_size: str = "1.7B"
    tts_qwen3_flash_attn: bool = False
    tts_qwen3_device: str = "cuda:0"

    # ── Diarization / Audio Processing ───────────────────────────────────────
    stt_diarize_enabled: bool = False
    stt_noise_reduce: bool = False
    stt_normalize: bool = True

    @property
    def tts_effective_device(self) -> str:
        return self.tts_device or self.stt_device

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
    def stt_stream_vad_threshold(self) -> float:
        return self.os_stream_vad_threshold

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

    model_config = {"env_prefix": "", "case_sensitive": False, "extra": "ignore"}


settings = Settings()

# Log deprecation warnings after settings are created
if _deprecation_warnings:
    log_deprecation_warnings(_deprecation_warnings)
