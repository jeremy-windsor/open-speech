# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TQDM_DISABLE=1

ARG MING_COMMIT=e999c39fa676722aaa9ea29b5aa19a90525b271f

RUN apt-get update && apt-get install -y --no-install-recommends git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --filter=blob:none --no-checkout https://github.com/inclusionAI/Ming-3.0-tts.git /opt/ming && \
    cd /opt/ming && \
    git checkout "${MING_COMMIT}"

WORKDIR /worker
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://pypi.org/simple -r requirements.txt && \
    pip check && \
    python -c "import torch, torchaudio; assert torch.__version__.split('+')[0] == '2.6.0'; assert torchaudio.__version__.split('+')[0] == '2.6.0'; assert torch.version.cuda == '12.4'"

# Ming hard-codes bf16, which Turing cards such as the RTX 2070 SUPER lack. Point every use at
# ming_dtype.MING_DTYPE so the worker can pick the dtype (MING_DTYPE) before it loads the model.
COPY ming_dtype.py /opt/ming/
RUN python - <<'PATCH'
from pathlib import Path

for name, count in (("ming_tts.py", 2), ("modeling_bailingmm.py", 5)):
    path = Path("/opt/ming") / name
    source = path.read_text()
    assert source.count("torch.bfloat16") == count, (name, source.count("torch.bfloat16"))
    source = source.replace("torch.bfloat16", "ming_dtype.MING_DTYPE")
    source = source.replace("import torch\n", "import torch\nimport ming_dtype\n", 1)
    path.write_text(source)
PATCH

# Fail the build if the text normalizer's FST files do not load with this pynini.
RUN cd /opt/ming && python -c "from frontend.talker_tn.talker_tn import TalkerTN; print(TalkerTN().normalize('Open Speech 2.0 costs \$5.'))"

ARG REVISION=unknown
LABEL org.opencontainers.image.revision=${REVISION}
COPY app.py ./

RUN useradd -m -s /bin/bash ming && \
    mkdir -p /home/ming/.cache/huggingface && \
    chown -R ming:ming /worker /home/ming

USER ming
ENV HOME=/home/ming \
    HF_HOME=/home/ming/.cache/huggingface \
    PYTHONPATH=/opt/ming

EXPOSE 8230
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8230"]
