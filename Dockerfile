# syntax=docker/dockerfile:1
###############################################################################
# Open Speech — GPU Dockerfile
#
# Uses python:slim base plus the CUDA userspace libraries that CTranslate2
# dynamically loads for faster-whisper GPU inference.
# Pre-bakes torch + provider runtimes for zero-wait setup. Providers can be
# customized at build time:
#   --build-arg BAKED_PROVIDERS=kokoro,pocket-tts,piper
#   --build-arg BAKED_TTS_MODELS=kokoro,pocket-tts,piper/en_US-ryan-medium
#
# Build:  docker build -t jwindsor1/open-speech:latest .
# Run:    docker run -d -p 8100:8100 jwindsor1/open-speech:latest
###############################################################################

FROM python:3.12-slim-bookworm

ARG BAKED_PROVIDERS="kokoro,piper,pocket-tts"
ARG BAKED_TTS_MODELS="kokoro"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libssl-dev libffi-dev \
        ffmpeg espeak-ng openssl && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip

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

RUN --mount=type=cache,target=/root/.cache/pip python3 -m venv "$VIRTUAL_ENV" && \
    pip install --upgrade pip

# ── Heavy deps (cached layer — changes rarely) ──────────────────────────────
# CTranslate2 dynamically loads CUDA libraries at inference time. The Python
# wheel does not vendor them, so bake cuBLAS/runtime into the GPU image.
ENV OS_BAKED_PROVIDERS=${BAKED_PROVIDERS} \
    OS_BAKED_TTS_MODELS=${BAKED_TTS_MODELS}
RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
import os
import subprocess
import sys

providers = [p.strip() for p in os.environ.get("OS_BAKED_PROVIDERS", "kokoro").split(",") if p.strip()]
specs = {
    "kokoro": ["kokoro>=0.9.4"],
    "pocket-tts": ["pocket-tts"],
    "piper": ["piper-tts"],
    "faster-whisper": ["faster-whisper"],
}

packages = []
for provider in providers:
    packages.extend(specs.get(provider, []))

seen = set()
ordered = []
for pkg in packages:
    if pkg not in seen:
        seen.add(pkg)
        ordered.append(pkg)

if ordered:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + ordered)

if "kokoro" in providers:
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
PY

# ── App deps ─────────────────────────────────────────────────────────────────
COPY pyproject.toml README.md requirements.lock ./

RUN --mount=type=cache,target=/root/.cache/pip (pip install -r requirements.lock || pip install ".[all]") && \
    chown -R openspeech:openspeech "$VIRTUAL_ENV"

# CTranslate2 dynamically dlopens CUDA userspace libraries for GPU inference.
# Install cuBLAS/runtime only when another dependency (for example torch) did
# not already install compatible nvidia-* wheels, then expose the wheel library
# directories on a stable LD_LIBRARY_PATH entry.
RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
import importlib.metadata as metadata
import site
import subprocess
import sys
from pathlib import Path

fallback_cuda_packages = {
    "nvidia-cublas-cu12": "12.4.5.8",
    "nvidia-cuda-runtime-cu12": "12.4.127",
}

missing = []
for package, version in fallback_cuda_packages.items():
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        missing.append(f"{package}=={version}")

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", *missing])

link_dir = Path("/opt/venv/cuda-libs")
link_dir.mkdir(parents=True, exist_ok=True)

linked = 0
for site_dir in map(Path, site.getsitepackages()):
    nvidia_dir = site_dir / "nvidia"
    if not nvidia_dir.exists():
        continue
    for lib in nvidia_dir.glob("*/lib/*.so*"):
        link = link_dir / lib.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(lib)
        linked += 1

required = {"libcublas.so.12", "libcudart.so.12"}
linked_names = {path.name for path in link_dir.iterdir()}
missing_libs = required - linked_names
if missing_libs:
    raise RuntimeError(f"Missing CUDA libraries for CTranslate2: {sorted(missing_libs)}")

print(f"Linked {linked} NVIDIA CUDA libraries into {link_dir}")
PY

# ── App source (changes most often — last layer) ────────────────────────────
COPY src/ src/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh

# Optional weight prefetch into image layer (best-effort by selected model IDs)
RUN --mount=type=cache,target=/root/.cache/pip python - <<'PY'
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
    LD_LIBRARY_PATH=/opt/venv/cuda-libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    HF_TOKEN="" \
    HUGGINGFACE_HUB_TOKEN="" \
    OS_HOST=0.0.0.0 \
    OS_PORT=8100 \
    STT_DEVICE=cuda \
    STT_COMPUTE_TYPE=float16 \
    STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2 \
    TTS_ENABLED=true \
    TTS_DEVICE=cuda \
    TTS_MODEL=kokoro \
    OS_WYOMING_ENABLED=true \
    OS_WYOMING_HOST=0.0.0.0 \
    OS_MAX_LOADED_MODELS=2

EXPOSE 8100 10400

VOLUME ["/home/openspeech/.cache/huggingface", \
        "/home/openspeech/.cache/silero-vad", \
        "/var/lib/open-speech/certs", \
        "/var/lib/open-speech/cache"]

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import os, ssl, urllib.request; scheme='https' if os.getenv('OS_SSL_ENABLED','true').lower() in ('1','true','yes','on') else 'http'; ctx=ssl._create_unverified_context() if scheme == 'https' else None; urllib.request.urlopen(f'{scheme}://localhost:{os.getenv(\"OS_PORT\", \"8100\")}/health', context=ctx, timeout=3)" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
