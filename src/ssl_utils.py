"""Auto-generate self-signed SSL certificates for HTTPS."""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger("open-speech")

DEFAULT_CERT_DIR = "/var/lib/open-speech/certs"
DEFAULT_CERT_FILE = f"{DEFAULT_CERT_DIR}/cert.pem"
DEFAULT_KEY_FILE = f"{DEFAULT_CERT_DIR}/key.pem"

CERT_VALID_DAYS = 365
# A certificate generated here is generated again at startup once it expires within this many days.
RENEW_WITHIN_DAYS = 30
# Written next to a generated cert ("cert.pem.generated"); holds the SHA-256 of the cert, then of
# its key, so a file someone puts in place of either later counts as theirs and is left alone.
GENERATED_MARKER_SUFFIX = ".generated"
GENERATED_SUBJECT = "CN=localhost"


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


def _restrict_mode(path: Path, mode: int) -> None:
    """Clear the permission bits ``mode`` does not allow; never add any."""
    try:
        current = path.stat().st_mode & 0o777
        if current & ~mode:
            path.chmod(current & mode)
    except OSError as exc:
        logger.warning("Could not set mode %o on %s: %s", mode, path, exc)


def _is_managed_dir(directory: Path) -> bool:
    """The default cert directory, which belongs to this service (the image creates it)."""
    try:
        return directory.resolve() == Path(DEFAULT_CERT_DIR).resolve()
    except OSError:
        return False


def _marker_path(cert: Path) -> Path:
    return cert.with_name(cert.name + GENERATED_MARKER_SUFFIX)


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded_fingerprints(cert: Path) -> list[str]:
    """The fingerprints in ``cert``'s marker: the generated cert's, then its key's."""
    try:
        return _marker_path(cert).read_text(encoding="utf-8").split()
    except OSError:
        return []


def _record_generated(cert: Path, key: Path) -> None:
    try:
        _marker_path(cert).write_text(f"{_fingerprint(cert)}\n{_fingerprint(key)}\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not record %s as generated, so it will not be renewed: %s", cert, exc)


def _openssl_x509(cert: Path, *args: str) -> str | None:
    """Output of ``openssl x509 -noout -in cert args``, or None when openssl cannot run."""
    try:
        result = subprocess.run(
            ["openssl", "x509", "-noout", "-in", str(cert), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,  # -checkend reports "will expire" with exit status 1
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not inspect SSL certificate %s: %s", cert, exc)
        return None
    return result.stdout


def _is_generated(cert: Path) -> bool:
    """True for a certificate this module generated, which it may therefore replace."""
    recorded = _recorded_fingerprints(cert)
    try:
        if recorded and recorded[0] == _fingerprint(cert):
            return True
    except OSError:
        pass
    # Certificates generated before the marker existed: the self-signed default pair.
    if cert.name != Path(DEFAULT_CERT_FILE).name or not _is_managed_dir(cert.parent):
        return False
    names = _openssl_x509(cert, "-subject", "-issuer", "-nameopt", "RFC2253") or ""
    fields = dict(line.replace(" ", "").split("=", 1) for line in names.splitlines() if "=" in line)
    return fields.get("subject") == GENERATED_SUBJECT and fields.get("issuer") == GENERATED_SUBJECT


def _is_generated_key(key: Path, cert: Path) -> bool:
    """True for a key generated here together with ``cert``, which it may therefore replace."""
    recorded = _recorded_fingerprints(cert)
    try:
        if len(recorded) > 1 and recorded[1] == _fingerprint(key):
            return True
    except OSError:
        pass
    # Keys generated before the marker existed, or left by an interrupted generation.
    return key.name == Path(DEFAULT_KEY_FILE).name and _is_managed_dir(key.parent)


def _expires_within(cert: Path, days: int) -> bool:
    output = _openssl_x509(cert, "-checkend", str(days * 24 * 3600))
    return output is not None and "will expire" in output and "will not expire" not in output


def ensure_ssl_certs(cert_path: str, key_path: str, extra_sans: str = "") -> None:
    """Use an existing cert/key pair, or generate a self-signed pair when both files are missing.

    A pair generated here is generated again once it expires within RENEW_WITHIN_DAYS days, or
    when one of its files is gone. Other existing files are only read: never replaced, and
    neither they nor their directories are chmodded. Half of a pair that is not generated here
    raises RuntimeError, because generating would overwrite it.
    """
    cert = Path(cert_path)
    key = Path(key_path)
    cert_exists, key_exists = cert.exists(), key.exists()

    if cert_exists and key_exists:
        if not (_is_generated(cert) and _is_generated_key(key, cert)):
            logger.info("SSL certs already exist: %s, %s", cert_path, key_path)
            return
        if not _expires_within(cert, RENEW_WITHIN_DAYS):
            if _is_managed_dir(cert.parent):
                _restrict_mode(cert.parent, 0o700)
                _restrict_mode(key, 0o600)
            logger.info("SSL certs already exist: %s, %s", cert_path, key_path)
            return
        logger.warning(
            "The generated SSL certificate %s expires within %d days; generating a new one. "
            "Clients that trusted the old certificate need to trust the new one.",
            cert_path,
            RENEW_WITHIN_DAYS,
        )
    elif cert_exists or key_exists:
        # openssl writes both paths, so only a file generated here may be replaced.
        present, missing = (cert_path, key_path) if cert_exists else (key_path, cert_path)
        if not (_is_generated(cert) if cert_exists else _is_generated_key(key, cert)):
            raise RuntimeError(
                f"SSL file {present} exists but {missing} does not, and generating a certificate would "
                f"overwrite {present}. Set OS_SSL_CERTFILE and OS_SSL_KEYFILE to an existing cert and key, "
                "or leave both unset to use a generated self-signed certificate."
            )
        logger.warning("Part of the generated SSL pair is missing (%s); generating a new pair.", missing)

    # Only directories created here, or the service's own default one, get restricted.
    own_dirs = [d for d in dict.fromkeys((cert.parent, key.parent)) if not d.exists() or _is_managed_dir(d)]
    cert.parent.mkdir(parents=True, exist_ok=True)
    key.parent.mkdir(parents=True, exist_ok=True)
    for directory in own_dirs:
        _restrict_mode(directory, 0o700)

    logger.info("Generating self-signed SSL certificate...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", str(CERT_VALID_DAYS), "-nodes",
                "-subj", f"/{GENERATED_SUBJECT}",
                "-addext", _subject_alt_names(extra_sans),
            ],
            check=True,
            capture_output=True,
        )
        if key.exists():
            _restrict_mode(key, 0o600)
        if cert.exists():
            cert.chmod(0o644)
            _record_generated(cert, key)
        logger.info("SSL certificate generated: %s", cert_path)
    except FileNotFoundError:
        raise RuntimeError(
            "openssl not found. Install openssl or set OS_SSL_ENABLED=false"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to generate SSL cert: {e.stderr.decode()}")
