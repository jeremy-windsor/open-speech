"""Tests for SSL certificate auto-generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ssl_utils import ensure_ssl_certs, DEFAULT_CERT_DIR


class TestEnsureSSLCerts:
    """Tests for ensure_ssl_certs()."""

    def test_skips_when_both_exist(self, tmp_path):
        """No generation when cert + key already exist."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text("CERT")
        key.write_text("KEY")

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            mock_run.assert_not_called()

    def test_generates_when_missing(self, tmp_path):
        """Calls openssl when certs don't exist."""
        cert = tmp_path / "sub" / "cert.pem"
        key = tmp_path / "sub" / "key.pem"

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "openssl"
            assert "-x509" in args
            assert str(key) in args
            assert str(cert) in args

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories for cert and key."""
        cert = tmp_path / "deep" / "nested" / "cert.pem"
        key = tmp_path / "deep" / "nested" / "key.pem"

        with patch("src.ssl_utils.subprocess.run"):
            ensure_ssl_certs(str(cert), str(key))
            assert cert.parent.exists()

    def test_openssl_not_found_raises(self, tmp_path):
        """Raises RuntimeError if openssl isn't installed."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"

        with patch("src.ssl_utils.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="openssl not found"):
                ensure_ssl_certs(str(cert), str(key))

    def test_openssl_failure_raises(self, tmp_path):
        """Raises RuntimeError on openssl error."""
        import subprocess
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"

        error = subprocess.CalledProcessError(1, "openssl", stderr=b"bad stuff")
        with patch("src.ssl_utils.subprocess.run", side_effect=error):
            with pytest.raises(RuntimeError, match="Failed to generate SSL cert"):
                ensure_ssl_certs(str(cert), str(key))

    def test_generates_when_cert_exists_key_missing(self, tmp_path):
        """Generates if cert exists but key doesn't."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        cert.write_text("CERT")

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            mock_run.assert_called_once()

    def test_generates_when_key_exists_cert_missing(self, tmp_path):
        """Generates if key exists but cert doesn't."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        key.write_text("KEY")

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            mock_run.assert_called_once()

    def test_openssl_called_with_check_true(self, tmp_path):
        """Verifies subprocess.run is called with check=True."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            _, kwargs = mock_run.call_args
            assert kwargs.get("check") is True
            assert kwargs.get("capture_output") is True
