# syntax=docker/dockerfile:1
# Fish Audio S2 Pro through the bitsandbytes NF4 fork of fish-speech, so it fits next to the core's models on an
# 8 GB card. The weights are not in the image: the worker fetches them at its first load, at a pinned revision,
# into its Hugging Face cache volume (about 5.0 GB).
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG FISH_COMMIT=3fd7da6a65dfeb9f8ad1975813301ed16c4da47a

# gcc and libc6-dev: Triton compiles its launcher at run time (torch.compile), and pyaudio, a hard
# dependency of the fork, builds against portaudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg gcc git libc6-dev libsndfile1 portaudio19-dev sox libsox-fmt-all && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --no-checkout https://github.com/scarxity/fish-speech-int4-patch.git /opt/fish-speech && \
    cd /opt/fish-speech && \
    git checkout "${FISH_COMMIT}" && \
    test "$(git rev-parse HEAD)" = "${FISH_COMMIT}"

WORKDIR /worker

# The cu126 wheels carry sm_75 (Turing).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu126 \
      torch==2.8.0 torchaudio==2.8.0

COPY fish/requirements.txt fish/constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c constraints.txt -r requirements.txt -e "/opt/fish-speech[bnb]" && \
    pip check && \
    python -c "import torch; assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__; assert torch.version.cuda == '12.6', torch.version.cuda; assert 'sm_75' in torch._C._cuda_getArchFlags().split(), torch._C._cuda_getArchFlags()"

ENV PYTHONPATH=/opt/fish-speech

# The worker calls these names of the fork at FISH_COMMIT (providers/fish/app.py); a fork that moved them fails
# here instead of at the first load.
RUN python - <<'PY'
import inspect
from pathlib import Path

import bitsandbytes
from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.model_manager import ModelManager

parameters = inspect.signature(ModelManager.__init__).parameters
for name in ("mode", "device", "half", "compile", "bnb4", "llama_checkpoint_path", "decoder_checkpoint_path",
             "decoder_config_name", "max_seq_len", "lazy_load", "codec_decode_only"):
    assert name in parameters, f"ModelManager takes no {name}"
assert callable(getattr(TTSInferenceEngine, "inference", None)), "TTSInferenceEngine.inference is missing"
assert callable(getattr(TTSInferenceEngine, "load_by_id", None)), "TTSInferenceEngine.load_by_id is missing"
for name in ("text", "reference_id", "use_memory_cache", "max_new_tokens", "chunk_length", "top_p",
             "temperature", "seed", "format", "streaming"):
    assert name in ServeTTSRequest.model_fields, f"ServeTTSRequest has no {name}"
for name in ("config_name", "checkpoint_path", "device"):
    assert name in inspect.signature(load_model).parameters, f"load_model takes no {name}"
libraries = sorted(path.name for path in Path(bitsandbytes.__file__).parent.glob("libbitsandbytes_cuda*"))
assert any("cuda126" in name for name in libraries), libraries
print("fish fork API ok; bitsandbytes", bitsandbytes.__version__, libraries)
PY

ARG REVISION=unknown
LABEL org.opencontainers.image.revision=${REVISION}
# The build context is providers/, so the worker keeps the repository layout and imports
# the shared scaffolding as providers._common.
COPY __init__.py providers/
COPY _common/ providers/_common/
COPY fish/app.py providers/fish/
RUN python -c "import providers.fish.app"

RUN useradd -m -s /bin/bash fishworker && \
    mkdir -p /worker/references \
      /home/fishworker/.cache/huggingface \
      /home/fishworker/.cache/torch/inductor \
      /home/fishworker/.cache/torch/triton && \
    chown -R fishworker:fishworker /worker /home/fishworker

USER fishworker
ENV HOME=/home/fishworker \
    HF_HOME=/home/fishworker/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/home/fishworker/.cache/torch/inductor \
    TRITON_CACHE_DIR=/home/fishworker/.cache/torch/triton

EXPOSE 8230
CMD ["uvicorn", "providers.fish.app:app", "--host", "0.0.0.0", "--port", "8230"]
