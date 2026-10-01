# syntax=docker/dockerfile:1
ARG DEVICE=cuda
FROM ghcr.io/astral-sh/uv:0.12.21 AS uv
FROM python:3.12.14-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg espeak-ng openssl && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN useradd -m -s /bin/bash openspeech && mkdir -p \
    /home/openspeech/.cache/huggingface /home/openspeech/.cache/silero-vad \
    /home/openspeech/data/conversations /home/openspeech/data/composer \
    /home/openspeech/data/providers /var/lib/open-speech/certs /var/lib/open-speech/cache
WORKDIR /app
ARG DEVICE
ARG BAKED_PROVIDERS="kokoro,piper,pocket-tts"
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:${PATH}" OS_BAKED_PROVIDERS=${BAKED_PROVIDERS} \
    DEVICE=${DEVICE}
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv python - <<'PY'
import os
import subprocess
from pathlib import Path

extras = {"kokoro": "tts", "piper": "piper", "pocket-tts": "pocket"}
providers = {p.strip() for p in os.environ["OS_BAKED_PROVIDERS"].split(",") if p.strip()}
unknown = providers - extras.keys()
if unknown:
    raise SystemExit(f"Unknown baked providers: {sorted(unknown)}")
command = ["uv", "sync", "--frozen", "--no-dev", "--no-install-project", "--extra", os.environ["DEVICE"]]
for provider in sorted(providers):
    command.extend(["--extra", extras[provider]])
subprocess.check_call(command)
if "kokoro" in providers:
    subprocess.check_call(["/opt/venv/bin/python", "-c", "import unidic,unidic_lite; from pathlib import Path; Path(unidic.DICDIR).symlink_to(unidic_lite.DICDIR, target_is_directory=True)"])
if os.environ["DEVICE"] == "cuda":
    link_dir = Path("/opt/venv/cuda-libs")
    link_dir.mkdir()
    for library in Path("/opt/venv/lib/python3.12/site-packages/nvidia").glob("*/lib/*.so*"):
        link = link_dir / library.name
        if not link.exists():
            link.symlink_to(library)
    missing = {"libcublas.so.12", "libcudart.so.12", "libcudnn.so.9"} - {p.name for p in link_dir.iterdir()}
    if missing:
        raise SystemExit(f"Missing CUDA libraries: {sorted(missing)}")
PY
ARG REVISION=unknown
LABEL org.opencontainers.image.source="https://github.com/jeremy-windsor/open-speech" \
      org.opencontainers.image.revision=${REVISION}
ENV OS_BUILD_REVISION=${REVISION}
COPY README.md ./
COPY src/ src/
COPY scripts/tts_conformance.py scripts/tts_conformance.py
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && chmod +x /usr/local/bin/docker-entrypoint.sh
ENV HOME=/home/openspeech XDG_CACHE_HOME=/home/openspeech/.cache \
    HF_HOME=/home/openspeech/.cache/huggingface STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub \
    LD_LIBRARY_PATH=/opt/venv/cuda-libs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    OS_HOST=0.0.0.0 OS_PORT=8100 TTS_ENABLED=true TTS_MODEL=kokoro TTS_VOICE=af_heart \
    OS_WYOMING_ENABLED=true OS_WYOMING_HOST=0.0.0.0 OS_MAX_LOADED_MODELS=2
ARG BAKED_TTS_MODELS=""
ENV OS_BAKED_TTS_MODELS=${BAKED_TTS_MODELS}
RUN python - <<'PY'
import os
models = [m.strip() for m in os.environ["OS_BAKED_TTS_MODELS"].split(",") if m.strip()]
if models:
    from src.main import model_manager
    for model in models:
        model_manager.download(model)
PY
RUN chown -R openspeech:openspeech /home/openspeech /var/lib/open-speech
EXPOSE 8100 10400
VOLUME ["/home/openspeech/.cache/huggingface", "/home/openspeech/.cache/silero-vad", "/var/lib/open-speech/certs", "/var/lib/open-speech/cache"]
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import os,ssl,urllib.request; scheme='https' if os.getenv('OS_SSL_ENABLED','true').lower() in ('1','true','yes','on') else 'http'; urllib.request.urlopen('%s://localhost:%s/health' % (scheme,os.getenv('OS_PORT','8100')),context=ssl._create_unverified_context(),timeout=3)" || exit 1
ENTRYPOINT ["docker-entrypoint.sh"]

FROM runtime AS cpu
ENV STT_DEVICE=cpu STT_COMPUTE_TYPE=int8 TTS_DEVICE=cpu STT_MODEL=Systran/faster-whisper-base
FROM runtime AS cuda
ENV STT_DEVICE=cuda STT_COMPUTE_TYPE=float16 TTS_DEVICE=cuda STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
FROM ${DEVICE} AS final
