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
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "setuptools<81" wheel && \
    pip install --no-build-isolation openai-whisper==20231117 && \
    pip install --no-deps --index-url \
      https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ \
      onnxruntime-gpu==1.18.0 && \
    pip install --index-url https://pypi.org/simple -r requirements.txt && \
    pip install --index-url https://pypi.org/simple --no-deps --force-reinstall \
      typing-extensions==4.14.1 && \
    pip check && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.3.1'; assert torch.version.cuda == '12.1'"

# The worker always passes a pinned local Hugging Face snapshot, so the ModelScope fallback is unused.
RUN sed -i 's/from modelscope import snapshot_download/from huggingface_hub import snapshot_download/' \
      /opt/CosyVoice/cosyvoice/cli/cosyvoice.py

COPY app.py ./

RUN useradd -m -s /bin/bash cosyvoice && \
    mkdir -p /home/cosyvoice/.cache/huggingface && \
    chown -R cosyvoice:cosyvoice /worker /home/cosyvoice

USER cosyvoice
ENV HOME=/home/cosyvoice \
    HF_HOME=/home/cosyvoice/.cache/huggingface \
    PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

EXPOSE 8220
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8220"]
