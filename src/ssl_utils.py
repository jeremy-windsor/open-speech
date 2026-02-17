"""Auto-generate self-signed SSL certificates for HTTPS."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("open-speech")

DEFAULT_CERT_DIR = "/tmp/open-speech-certs"
DEFAULT_CERT_FILE = f"{DEFAULT_CERT_DIR}/cert.pem"
DEFAULT_KEY_FILE = f"{DEFAULT_CERT_DIR}/key.pem"


def ensure_ssl_certs(cert_path: str, key_path: str) -> None:
    """Generate a self-signed certificate if it doesn't already exist."""
    cert = Path(cert_path)
    key = Path(key_path)

    if cert.exists() and key.exists():
        logger.info("SSL certs already exist: %s, %s", cert_path, key_path)
        return

    # Ensure parent dirs exist
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating self-signed SSL certificate...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:0.0.0.0",
            ],
            check=True,
            capture_output=True,
        )
        logger.info("SSL certificate generated: %s", cert_path)
    except FileNotFoundError:
        raise RuntimeError(
            "openssl not found. Install openssl or set STT_SSL_ENABLED=false"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate SSL cert: {e.stderr.decode()}")
