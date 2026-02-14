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

    # Streaming settings
    stt_stream_chunk_ms: int = 2000          # Chunk size in ms
    stt_stream_vad_threshold: float = 0.5    # VAD confidence threshold
    stt_stream_endpointing_ms: int = 300     # Silence before finalizing utterance
    stt_stream_max_connections: int = 10     # Max concurrent streaming sessions

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
