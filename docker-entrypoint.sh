#!/bin/sh
# Fix volume ownership — Docker volumes persist perms from prior runs.
chown -R openspeech:openspeech \
    /home/openspeech/.cache/huggingface \
    /home/openspeech/.cache/silero-vad \
    /var/lib/open-speech/certs \
    2>/dev/null || true

# Export env vars that su would otherwise drop
export HF_HOME=/home/openspeech/.cache/huggingface
export STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub

exec su -p -s /bin/sh openspeech -c 'HF_HOME=/home/openspeech/.cache/huggingface STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub python -m src.main'
