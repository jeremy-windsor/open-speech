# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TQDM_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements.txt ./
# The voxcpm package also lists its web demo, training and denoiser stacks (gradio, funasr,
# modelscope, datasets). The worker only runs inference, so it installs without them.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://pypi.org/simple -r requirements.txt && \
    pip check && \
    pip install --index-url https://pypi.org/simple --no-deps voxcpm==2.0.3 && \
    python -c "import torch, torchaudio; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'" && \
    python -c "import voxcpm.model.voxcpm2, voxcpm.modules.audiovae"

ARG REVISION=unknown
LABEL org.opencontainers.image.revision=${REVISION}
COPY app.py ./

RUN useradd -m -s /bin/bash voxcpm && \
    mkdir -p /home/voxcpm/.cache/huggingface && \
    chown -R voxcpm:voxcpm /worker /home/voxcpm

USER voxcpm
ENV HOME=/home/voxcpm \
    HF_HOME=/home/voxcpm/.cache/huggingface

EXPOSE 8240
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8240"]
