# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG PERTH_COMMIT=ff1c8ac55a976971245cdd53c18d6131ca00d993

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git libsndfile1 sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /worker

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchaudio==2.6.0

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt && \
    pip install --no-deps --force-reinstall \
      "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@${PERTH_COMMIT}" && \
    pip check && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'"

COPY app.py ./

RUN useradd -m -s /bin/bash chatterbox && \
    mkdir -p /home/chatterbox/.cache/huggingface && \
    chown -R chatterbox:chatterbox /worker /home/chatterbox

USER chatterbox
ENV HOME=/home/chatterbox \
    HF_HOME=/home/chatterbox/.cache/huggingface

EXPOSE 8210
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8210"]
