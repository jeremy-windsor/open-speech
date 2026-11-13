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
# Compose binds the checkout's ./data here, and on Linux a bind mount keeps the host's owner: a checkout made with
# sudo, or by an account other than uid 1000, can't be written by the server. Give it to openspeech only when that
# user can't write it, so a working library isn't walked on every start.
mkdir -p /home/openspeech/data
if ! su -s /bin/sh openspeech -c 'test -w /home/openspeech/data' 2>/dev/null; then
    echo "[entrypoint] Fixing ownership: /home/openspeech/data (not writable by openspeech)"
    chown -R openspeech:openspeech /home/openspeech/data 2>/dev/null || true
fi
mkdir -p /home/openspeech/data/providers
_fix_owner /home/openspeech/data/providers

# Use explicit venv python — su -p doesn't reliably preserve PATH on all systems
exec su -p -s /bin/sh openspeech -c 'export HOME=/home/openspeech XDG_CACHE_HOME=/home/openspeech/.cache HF_HOME=/home/openspeech/.cache/huggingface STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub PATH=/opt/venv/bin:$PATH; exec /opt/venv/bin/python -m src.main'
