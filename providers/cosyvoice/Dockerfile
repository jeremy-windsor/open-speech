# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG COSYVOICE_COMMIT=074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git git-lfs libsndfile1 sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install --system

RUN git clone --filter=blob:none --no-checkout https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice && \
    cd /opt/CosyVoice && \
    git checkout "${COSYVOICE_COMMIT}" && \
    git submodule update --init --recursive

WORKDIR /worker
COPY cosyvoice/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && \
    apt-get install -y --no-install-recommends g++ && \
    pip install "setuptools<81" wheel && \
    pip install --no-build-isolation openai-whisper==20231117 && \
    pip install --no-deps --index-url \
      https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      onnxruntime-gpu==1.18.0 && \
    pip install --index-url https://pypi.org/simple -r requirements.txt && \
    pip install --index-url https://pypi.org/simple --no-deps --force-reinstall \
      typing-extensions==4.14.1 && \
    pip check && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.3.1'; assert torch.version.cuda == '12.1'" && \
    apt-get purge -y --auto-remove g++ && \
    rm -rf /var/lib/apt/lists/*

# The worker always passes a pinned local Hugging Face snapshot, so CosyVoice's own ModelScope
# fallback is unused. wetext's ModelScope fetch is replaced by the files baked in below.
RUN sed -i 's/from modelscope import snapshot_download/from huggingface_hub import snapshot_download/' \
      /opt/CosyVoice/cosyvoice/cli/cosyvoice.py

# CosyVoice's text normalizer, wetext 0.0.4, reads four FST files of the ModelScope repository
# pengzhendong/wetext, and on its own would fetch the whole repository from its default branch at
# every model load in a new container. They are fetched here once, at a pinned commit and checked
# against their SHA-256 (cosyvoice/fetch_wetext.py), and the worker points wetext at /opt/wetext
# (WETEXT_DIR in cosyvoice/app.py).
COPY cosyvoice/fetch_wetext.py /tmp/
RUN python /tmp/fetch_wetext.py /opt/wetext && rm /tmp/fetch_wetext.py

ARG REVISION=unknown
LABEL org.opencontainers.image.revision=${REVISION}
# The build context is providers/, so the worker keeps the repository layout and imports
# the shared scaffolding as providers._common.
COPY __init__.py providers/
COPY _common/ providers/_common/
COPY cosyvoice/app.py providers/cosyvoice/
RUN python -c "import providers.cosyvoice.app"
# The baked normalizer must load and work with no network, as a model load will use it.
RUN --network=none python -c "from providers.cosyvoice.app import check_text_normalizer; print(check_text_normalizer())"

RUN useradd -m -s /bin/bash cosyvoice && \
    mkdir -p /home/cosyvoice/.cache/huggingface && \
    chown -R cosyvoice:cosyvoice /worker /home/cosyvoice

USER cosyvoice
ENV HOME=/home/cosyvoice \
    HF_HOME=/home/cosyvoice/.cache/huggingface \
    PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

EXPOSE 8220
CMD ["uvicorn", "providers.cosyvoice.app:app", "--host", "0.0.0.0", "--port", "8220"]
