"""Auto-generate self-signed SSL certificates for HTTPS."""

from __future__ import annotations

import logging
import ipaddress
import subprocess
from pathlib import Path

logger = logging.getLogger("open-speech")

DEFAULT_CERT_DIR = "/var/lib/open-speech/certs"
DEFAULT_CERT_FILE = f"{DEFAULT_CERT_DIR}/cert.pem"
DEFAULT_KEY_FILE = f"{DEFAULT_CERT_DIR}/key.pem"


def _format_san_entry(value: str) -> str | None:
    """Return an OpenSSL SAN entry for a DNS name or IP address."""
    item = value.strip()
    if not item:
        return None
    if item.startswith(("DNS:", "IP:")):
        return item
    try:
        ipaddress.ip_address(item)
    except ValueError:
        return f"DNS:{item}"
    return f"IP:{item}"


def _subject_alt_names(extra_sans: str = "") -> str:
    entries = ["DNS:localhost", "IP:127.0.0.1", "IP:0.0.0.0"]
    for raw in extra_sans.split(","):
        entry = _format_san_entry(raw)
        if entry and entry not in entries:
            entries.append(entry)
    return "subjectAltName=" + ",".join(entries)


def ensure_ssl_certs(cert_path: str, key_path: str, extra_sans: str = "") -> None:
    """Generate a self-signed certificate if it doesn't already exist."""
    cert = Path(cert_path)
    key = Path(key_path)

    if cert.exists() and key.exists():
        cert.parent.chmod(0o700)
        key.parent.chmod(0o700)
        if key.exists():
            key.chmod(0o600)
        logger.info("SSL certs already exist: %s, %s", cert_path, key_path)
        return

    # Ensure parent dirs exist
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    cert.parent.chmod(0o700)
    key.parent.chmod(0o700)

    logger.info("Generating self-signed SSL certificate...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
                "-addext", _subject_alt_names(extra_sans),
            ],
            check=True,
            capture_output=True,
        )
        if key.exists():
            key.chmod(0o600)
        if cert.exists():
            cert.chmod(0o644)
        logger.info("SSL certificate generated: %s", cert_path)
    except FileNotFoundError:
        raise RuntimeError(
            "openssl not found. Install openssl or set OS_SSL_ENABLED=false"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate SSL cert: {e.stderr.decode()}")
