FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg openssl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Copy everything needed for install
COPY pyproject.toml README.md ./
COPY src/ src/

# Install
RUN pip install --no-cache-dir .

# Config
ENV STT_HOST=0.0.0.0
ENV STT_PORT=8100
ENV STT_DEVICE=cuda
ENV STT_COMPUTE_TYPE=float16
ENV STT_DEFAULT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2

EXPOSE 8100

CMD ["sh", "-c", "openssl req -x509 -newkey rsa:2048 -keyout /tmp/key.pem -out /tmp/cert.pem -days 3650 -nodes -subj '/CN=open-speech' 2>/dev/null && python -m uvicorn src.main:app --host 0.0.0.0 --port 8100 --ssl-keyfile /tmp/key.pem --ssl-certfile /tmp/cert.pem"]
