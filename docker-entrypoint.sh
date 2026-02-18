#!/bin/sh
# Fix volume ownership — Docker volumes persist perms from prior runs.
# Runs as root (Dockerfile USER not set before ENTRYPOINT), then drops to openspeech.
chown -R openspeech:openspeech \
    /home/openspeech/.cache/huggingface \
    /home/openspeech/.cache/silero-vad \
    /var/lib/open-speech/certs \
    2>/dev/null || true

exec su -s /bin/sh openspeech -c "python -m src.main"
