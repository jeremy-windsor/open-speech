###############################################################################
# Open Speech — GPU Dockerfile
#
# Uses python:slim base. torch bundles its own CUDA runtime, so no need
# for nvidia/cuda base image (~20GB savings).
# Pre-bakes torch + provider runtimes for zero-wait setup. Providers can be
# customized at build time:
#   --build-arg BAKED_PROVIDERS=kokoro,pocket-tts,piper
#   --build-arg BAKED_TTS_MODELS=kokoro,pocket-tts,piper/en_US-ryan-medium
#
# Build:  docker build -t jwindsor1/open-speech:latest .
# Run:    docker run -d -p 8100:8100 jwindsor1/open-speech:latest
###############################################################################

FROM python:3.12-slim-bookworm

ARG BAKED_PROVIDERS="kokoro,qwen3"
ARG BAKED_TTS_MODELS="kokoro"

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
# torch bundles CUDA 12.x runtime (~2.5GB). Additional providers are optional.
ENV OS_BAKED_PROVIDERS=${BAKED_PROVIDERS} \
    OS_BAKED_TTS_MODELS=${BAKED_TTS_MODELS}
RUN python - <<'PY'
import os
import subprocess
import sys

providers = [p.strip() for p in os.environ.get("OS_BAKED_PROVIDERS", "kokoro").split(",") if p.strip()]
specs = {
    "kokoro": ["kokoro>=0.9.4"],
    "pocket-tts": ["pocket-tts"],
    "piper": ["piper-tts"],
    "qwen3": ["accelerate>=0.26.0", "soundfile>=0.12.0", "librosa>=0.10", "qwen-tts>=0.1.0"],
    "faster-whisper": ["faster-whisper"],
}

# qwen-tts hard-pins transformers==4.57.3 — must be installed FIRST, alone,
# so pip resolves it without fighting kokoro's looser transformers dep.
if "qwen3" in providers:
    # Step 1: torchaudio from CUDA index (avoids CPU build from PyPI)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-cache-dir",
        "--index-url", "https://download.pytorch.org/whl/cu121",
        "torchaudio"
    ])
    # Step 2: qwen-tts + its deps (this pins transformers==4.57.3)
    qwen_pkgs = specs["qwen3"]  # no transformers override
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir"] + qwen_pkgs)
    # Remove qwen3 from the combined install below to avoid re-triggering conflict
    specs["qwen3"] = []

# Combined install: torch + remaining providers (kokoro etc.)
packages = ["torch"]
for provider in providers:
    packages.extend(specs.get(provider, []))

# dedupe + deterministic order
seen = set()
ordered = []
for p in packages:
    if p not in seen:
        seen.add(p)
        ordered.append(p)

if ordered:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade"] + ordered)
if "kokoro" in providers:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
PY

# ── App deps ─────────────────────────────────────────────────────────────────
COPY pyproject.toml README.md requirements.lock ./

RUN (pip install --no-cache-dir -r requirements.lock || pip install --no-cache-dir ".[all]") && \
    chown -R openspeech:openspeech "$VIRTUAL_ENV"

# ── App source (changes most often — last layer) ────────────────────────────
COPY src/ src/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# Optional weight prefetch into image layer (best-effort by selected model IDs)
RUN python - <<'PY'
import os

models = [m.strip() for m in os.environ.get("OS_BAKED_TTS_MODELS", "kokoro").split(",") if m.strip()]
if not models:
    raise SystemExit(0)

os.environ.setdefault("HOME", "/home/openspeech")
os.environ.setdefault("HF_HOME", "/home/openspeech/.cache/huggingface")
os.environ.setdefault("HF_HUB_CACHE", "/home/openspeech/.cache/huggingface/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/home/openspeech/.cache/huggingface/hub")
os.environ.setdefault("STT_MODEL_DIR", "/home/openspeech/.cache/huggingface/hub")

try:
    from src.main import model_manager
except Exception as e:
    print(f"Skipping prefetch; app imports unavailable: {e}")
    raise SystemExit(0)

for model_id in models:
    try:
        info = model_manager.download(model_id)
        print(f"Pre-cached {model_id}: {info.state.value}")
    except Exception as e:
        print(f"WARNING: failed to pre-cache {model_id}: {e}")
PY

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
