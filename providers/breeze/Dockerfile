# syntax=docker/dockerfile:1
# Breeze TTS 2 through the community C++/ggml port (HoppouAI/Breeze-TTS-2.cpp), built with its CUDA backend
# (Vulkan is not available in WSL2 containers). The Q8_0 weights are not in the image: the worker fetches them at
# its first load, at a pinned revision, into its Hugging Face cache volume (about 3.6 GB).
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake git ninja-build && \
    rm -rf /var/lib/apt/lists/*

ARG BREEZE_REF=v0.1.7
ARG BREEZE_COMMIT=4110a9fff4e47d4d1ba850d92d81f3ba86bf39c6
RUN git clone --recursive --depth 1 --branch "${BREEZE_REF}" https://github.com/HoppouAI/Breeze-TTS-2.cpp /src && \
    cd /src && \
    test "$(git rev-parse HEAD)" = "${BREEZE_COMMIT}" && \
    git submodule update --init --recursive --depth 1

WORKDIR /src
# 75 is Turing (RTX 20). A bare number also embeds PTX that newer GPUs compile at their first load; list your
# card's own (86 RTX 30, 89 RTX 40, 120 RTX 50, separated by ";") to skip that. Each one adds minutes to the build.
ARG BREEZE_CUDA_ARCHITECTURES=75
ARG JOBS=4
# libggml-cuda calls the CUDA driver API, so it links against the toolkit's libcuda stub; the driver's own
# library is mounted into the container when it runs.
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBREEZE_VULKAN=OFF -DBREEZE_CUDA=ON \
      "-DCMAKE_CUDA_ARCHITECTURES=${BREEZE_CUDA_ARCHITECTURES}" \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,/opt/breeze/lib -Wl,-rpath-link,/usr/local/cuda/lib64/stubs"
# The CUDA kernels take the longest to compile: their own layer, cached apart from the rest.
RUN cmake --build build -j "${JOBS}" --target ggml-cuda
RUN cmake --build build -j "${JOBS}"
RUN mkdir -p /opt/breeze/bin /opt/breeze/lib && \
    find build -maxdepth 3 -type f -name 'breeze-server' -perm -u+x -exec cp {} /opt/breeze/bin/ \; && \
    find build -name '*.so*' -exec cp -P {} /opt/breeze/lib/ \; && \
    test -x /opt/breeze/bin/breeze-server

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: the ggml CPU backend uses OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libgomp1 python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/breeze /opt/breeze
ARG BREEZE_CUDA_ARCHITECTURES=75
ENV PATH=/opt/venv/bin:/opt/breeze/bin:$PATH \
    LD_LIBRARY_PATH=/opt/breeze/lib \
    BREEZE_CUDA_ARCHITECTURES=${BREEZE_CUDA_ARCHITECTURES}
# The binary must find every library it links, the driver's aside.
RUN ! ldd /opt/breeze/bin/breeze-server | grep "not found" | grep -v libcuda.so

WORKDIR /worker
COPY breeze/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/venv && \
    pip install -r requirements.txt && \
    pip check

ARG REVISION=unknown
LABEL org.opencontainers.image.revision=${REVISION}
# The build context is providers/, so the worker keeps the repository layout and imports
# the shared scaffolding as providers._common.
COPY __init__.py providers/
COPY _common/ providers/_common/
COPY breeze/app.py providers/breeze/
RUN python -c "import providers.breeze.app"

RUN useradd -m -s /bin/bash breezeworker && \
    mkdir -p /home/breezeworker/.cache/huggingface && \
    chown -R breezeworker:breezeworker /worker /home/breezeworker

USER breezeworker
ENV HOME=/home/breezeworker \
    HF_HOME=/home/breezeworker/.cache/huggingface

EXPOSE 8240
CMD ["uvicorn", "providers.breeze.app:app", "--host", "0.0.0.0", "--port", "8240"]
