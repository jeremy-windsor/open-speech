#!/bin/sh
# Fix volume ownership only if needed — avoids slow chown on large model caches.
_fix_owner() {
    dir="$1"
    if [ -d "$dir" ]; then
        owner=$(stat -c '%U' "$dir" 2>/dev/null || stat -f '%Su' "$dir" 2>/dev/null)
        if [ "$owner" != "openspeech" ]; then
            echo "[entrypoint] Fixing ownership: $dir (was $owner)"
            chown -R openspeech:openspeech "$dir" 2>/dev/null || true
        fi
    fi
}

_fix_owner /home/openspeech/.cache/huggingface
_fix_owner /home/openspeech/.cache/silero-vad
_fix_owner /var/lib/open-speech/certs
_fix_owner /var/lib/open-speech/cache
_fix_owner /opt/venv

# Use explicit venv python — su -p doesn't reliably preserve PATH on all systems
exec su -p -s /bin/sh openspeech -c 'export HOME=/home/openspeech XDG_CACHE_HOME=/home/openspeech/.cache HF_HOME=/home/openspeech/.cache/huggingface STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub PATH=/opt/venv/bin:$PATH; exec /opt/venv/bin/python -m src.main'
