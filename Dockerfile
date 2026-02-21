###############################################################################
# Open Speech — GPU Dockerfile
#
# Uses NVIDIA CUDA base image. Falls back to CPU automatically when no GPU.
# Pre-bakes torch + kokoro for zero-wait TTS. Other providers install at
# runtime via the Models tab (persisted to data/providers/ volume).
#
# Build:  docker build -t jwindsor1/open-speech:latest .
# Run:    docker run -d -p 8100:8100 jwindsor1/open-speech:latest
###############################################################################

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System deps + Python 3.12 ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        ffmpeg espeak-ng openssl && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    python3.12 -m ensurepip --upgrade && \
    python3 -m pip install --upgrade pip

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
# torch is ~2.5GB with CUDA. Kokoro + spacy add ~400MB. Keep this layer early
# so Docker caches it when only app code or requirements change.
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
