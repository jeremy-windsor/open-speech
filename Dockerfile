###############################################################################
# Open Speech — GPU Dockerfile
#
# Uses python:slim base. torch bundles its own CUDA runtime, so no need
# for nvidia/cuda base image (~20GB savings).
# Pre-bakes torch + kokoro for zero-wait TTS. Other providers install at
# runtime via the Models tab (persisted to data/providers/ volume).
#
# Build:  docker build -t jwindsor1/open-speech:latest .
# Run:    docker run -d -p 8100:8100 jwindsor1/open-speech:latest
###############################################################################

FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libssl-dev libffi-dev \
        ffmpeg espeak-ng openssl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# ── User + dirs ──────────────────────────────────────────────────────────────
RUN useradd -m -s /bin/bash openspeech && \
    mkdir -p /home/openspeech/.cache/huggingface \
             /home/openspeech/.cache/silero-vad \
             /home/openspeech/data/conversations \
             /home/openspeech/data/composer \
             /home/openspeech/data/providers \
             /var/lib/open-speech/certs \
             /var/lib/open-speech/cache \
             /opt/venv && \
    chown -R openspeech:openspeech /home/openspeech /var/lib/open-speech /opt/venv

WORKDIR /app

# ── Virtualenv ───────────────────────────────────────────────────────────────
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python3 -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

# ── Heavy deps (cached layer — changes rarely) ──────────────────────────────
# torch bundles CUDA 12.x runtime (~2.5GB). Kokoro + spacy add ~400MB.
RUN pip install --no-cache-dir torch "kokoro>=0.9.4" && \
    python -m spacy download en_core_web_sm

# ── App deps ─────────────────────────────────────────────────────────────────
COPY pyproject.toml README.md requirements.lock ./

RUN (pip install --no-cache-dir -r requirements.lock || pip install --no-cache-dir ".[all]") && \
    chown -R openspeech:openspeech "$VIRTUAL_ENV"

# ── App source (changes most often — last layer) ────────────────────────────
COPY src/ src/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# ── Config ───────────────────────────────────────────────────────────────────
ENV HOME=/home/openspeech \
    XDG_CACHE_HOME=/home/openspeech/.cache \
    HF_HOME=/home/openspeech/.cache/huggingface \
    STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub \
    OS_HOST=0.0.0.0 \
    OS_PORT=8100 \
    STT_DEVICE=cuda \
    STT_COMPUTE_TYPE=float16 \
    STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2 \
    TTS_ENABLED=true \
    TTS_DEVICE=cuda \
    TTS_MODEL=kokoro

EXPOSE 8100 10400

VOLUME ["/home/openspeech/.cache/huggingface", \
        "/home/openspeech/.cache/silero-vad", \
        "/var/lib/open-speech/certs", \
        "/var/lib/open-speech/cache"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8100/health')" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
