# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /worker

# Keep CUDA/Torch deterministic and isolated from the Open Speech core image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchaudio==2.6.0

COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c constraints.txt -r requirements.txt && \
    pip check && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'"

RUN python - <<'PY'
import inspect

from qwen_tts import Qwen3TTSModel

for method_name in ("generate_custom_voice", "generate_voice_clone"):
    parameters = inspect.signature(getattr(Qwen3TTSModel, method_name)).parameters
    assert "non_streaming_mode" in parameters, f"{method_name} lacks non_streaming_mode"
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    assert "max_new_tokens" in parameters or accepts_kwargs, (
        f"{method_name} lacks max_new_tokens forwarding"
    )
PY

COPY app.py ./

RUN useradd -m -s /bin/bash qwenworker && \
    mkdir -p /home/qwenworker/.cache/huggingface && \
    chown -R qwenworker:qwenworker /worker /home/qwenworker

USER qwenworker
ENV HOME=/home/qwenworker \
    HF_HOME=/home/qwenworker/.cache/huggingface

EXPOSE 8200
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8200"]
