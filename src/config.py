"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """STT server settings, all configurable via environment variables."""

    stt_default_model: str = "deepdml/faster-whisper-large-v3-turbo-ct2"
    stt_device: str = "cuda"
    stt_compute_type: str = "float16"
    stt_host: str = "0.0.0.0"
    stt_port: int = 8100
    stt_model_dir: str | None = None  # None = use default HF cache
    stt_preload_models: str = ""  # Comma-separated list of models to preload on startup
    stt_model_ttl: int = 300  # Seconds idle before auto-unload (0 = never). Default model exempt.
    stt_max_loaded_models: int = 0  # Max models in memory (0 = unlimited). LRU eviction, default exempt.

    # Streaming settings
    stt_stream_chunk_ms: int = 2000          # Chunk size in ms
    stt_stream_vad_threshold: float = 0.5    # VAD confidence threshold
    stt_stream_endpointing_ms: int = 300     # Silence before finalizing utterance
    stt_stream_max_connections: int = 10     # Max concurrent streaming sessions

    # Security settings
    stt_api_key: str = ""                    # API key for auth (empty = auth disabled)
    stt_rate_limit: int = 0                  # Requests per minute per IP (0 = disabled)
    stt_rate_limit_burst: int = 0            # Burst allowance (0 = same as rate_limit)
    stt_max_upload_mb: int = 100             # Max upload size in MB
    stt_cors_origins: str = "*"              # Comma-separated CORS origins (* = allow all)
    stt_trust_proxy: bool = False            # Trust X-Forwarded-For for rate limiting

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
