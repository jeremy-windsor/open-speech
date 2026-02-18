###############################################################################
# Open Speech — Unified Dockerfile (CPU + GPU)
#
# Uses NVIDIA CUDA base image. Falls back to CPU automatically when no GPU.
# Installs all provider packages; models download at runtime.
#
# Build:  docker build -t jwindsor1/open-speech:latest .
# Run:    docker run -d -p 8100:8100 jwindsor1/open-speech:latest
###############################################################################

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps + Python 3.12 from deadsnakes PPA (Ubuntu 22.04 ships 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip ffmpeg espeak-ng openssl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python3.12 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip

RUN useradd -m -s /bin/bash openspeech
RUN mkdir -p /home/openspeech/.cache/huggingface /home/openspeech/.cache/silero-vad /var/lib/open-speech/certs \
    && chown -R openspeech:openspeech /home/openspeech /var/lib/open-speech

WORKDIR /app

COPY pyproject.toml README.md requirements.lock ./
COPY src/ src/

# Reproducible install option:
# - prefer requirements.lock when present
# - fallback to extras install for full provider matrix
RUN pip install --no-cache-dir -r requirements.lock || pip install --no-cache-dir ".[all]"

# Config — uses new OS_ naming convention
ENV HF_HOME=/home/openspeech/.cache/huggingface
ENV STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub
ENV OS_HOST=0.0.0.0
ENV OS_PORT=8100
ENV STT_DEVICE=cuda
ENV STT_COMPUTE_TYPE=float16
ENV STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
ENV TTS_ENABLED=true
ENV TTS_DEVICE=cuda
ENV TTS_MODEL=kokoro

EXPOSE 8100
EXPOSE 10400

VOLUME ["/home/openspeech/.cache/huggingface", "/home/openspeech/.cache/silero-vad", "/var/lib/open-speech/certs"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
