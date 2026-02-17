FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg espeak-ng openssl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

# Install with TTS support (GPU torch from default index)
RUN pip install --no-cache-dir ".[tts]"

# Config
ENV STT_HOST=0.0.0.0
ENV STT_PORT=8100
ENV STT_DEVICE=cuda
ENV STT_COMPUTE_TYPE=float16
ENV STT_DEFAULT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
ENV TTS_ENABLED=true
ENV TTS_DEVICE=cuda

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

CMD ["python", "-m", "src.main"]
