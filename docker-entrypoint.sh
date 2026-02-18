#!/bin/sh
# Fix volume ownership — Docker volumes persist perms from prior runs.
chown -R openspeech:openspeech \
    /home/openspeech/.cache/huggingface \
    /home/openspeech/.cache/silero-vad \
    /var/lib/open-speech/certs \
    /var/lib/open-speech/cache \
    /opt/venv \
    2>/dev/null || true

# Ensure runtime cache paths resolve to openspeech home (Path.home/XDG-aware libs)
export HOME=/home/openspeech
export XDG_CACHE_HOME=/home/openspeech/.cache
export HF_HOME=/home/openspeech/.cache/huggingface
export STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub

# Preserve all Docker-provided env vars (OS_*, STT_*, TTS_*, etc.) while forcing
# the cache/home paths above for predictable non-root runtime behavior.
exec su -p -s /bin/sh openspeech -c 'export HOME=/home/openspeech XDG_CACHE_HOME=/home/openspeech/.cache HF_HOME=/home/openspeech/.cache/huggingface STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub; exec python -m src.main'
