"""Tests for SSL certificate auto-generation."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import ssl_utils
from src.ssl_utils import ensure_ssl_certs


def _fake_openssl(*, expiring: bool = False, subject: str = "CN=localhost"):
    """subprocess.run stand-in for the openssl commands ensure_ssl_certs runs."""
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "req":
            Path(args[args.index("-keyout") + 1]).write_text(f"KEY {len(calls)}")
            Path(args[args.index("-out") + 1]).write_text(f"CERT {len(calls)}")
            return subprocess.CompletedProcess(args, 0, b"", b"")
        if "-checkend" in args:
            verdict = "will expire" if expiring else "will not expire"
            return subprocess.CompletedProcess(args, int(expiring), f"Certificate {verdict}\n", "")
        if "-subject" in args:
            return subprocess.CompletedProcess(args, 0, f"subject={subject}\nissuer={subject}\n", "")
        raise AssertionError(f"unexpected command: {args}")

    run.calls = calls
    return run


def _generations(fake) -> int:
    return sum(1 for args in fake.calls if args[1] == "req")


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

    def test_existing_cert_in_unowned_directory_does_not_abort_startup(self, tmp_path, caplog):
        """A mounted custom cert dir may refuse chmod (EPERM, or EROFS on a :ro mount)."""
        cert = tmp_path / "certs" / "cert.pem"
        key = tmp_path / "certs" / "key.pem"
        cert.parent.mkdir()
        cert.write_text("CERT")
        key.write_text("KEY")

        def refuse_chmod(self, mode, *, follow_symlinks=True):
            raise PermissionError(1, "Operation not permitted", str(self))

        with patch.object(Path, "chmod", refuse_chmod), patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            mock_run.assert_not_called()
        # Using user-supplied files needs no chmod at all, so there is nothing to warn about.
        assert "Could not set mode" not in caplog.text

    def test_existing_user_supplied_files_are_used_without_changing_permissions(self, tmp_path):
        """A shared /etc/ssl/private (0710 root:ssl-cert, keys 0640) must keep its group access."""
        certs = tmp_path / "certs"
        certs.mkdir()
        cert, key = certs / "cert.pem", certs / "key.pem"
        cert.write_text("CERT")
        key.write_text("KEY")
        chmods = []

        with patch.object(Path, "chmod", lambda self, mode, **kw: chmods.append((self.name, oct(mode)))), \
             patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
        assert chmods == []
        mock_run.assert_not_called()

    def test_generation_restricts_only_directories_it_creates(self, tmp_path):
        shared = tmp_path / "shared"
        shared.mkdir()
        cases = [(shared / "cert.pem", shared / "key.pem"), (tmp_path / "new" / "cert.pem", tmp_path / "new" / "key.pem")]
        chmods = []

        with patch.object(Path, "chmod", lambda self, mode, **kw: chmods.append((self.name, oct(mode)))), \
             patch("src.ssl_utils.subprocess.run", side_effect=_fake_openssl()):
            for cert, key in cases:
                ensure_ssl_certs(str(cert), str(key))
        assert ("shared", "0o700") not in chmods
        # The new directory may already be 0700 under a strict umask; it is never widened.
        assert {name for name, _ in chmods} <= {"new", "key.pem", "cert.pem"}
        assert ("cert.pem", "0o644") in chmods

    def test_restrict_mode_only_clears_bits(self):
        from src.ssl_utils import _restrict_mode

        class FakePath:
            def __init__(self, mode):
                self.mode = mode
                self.chmods = []

            def stat(self):
                return SimpleNamespace(st_mode=0o100000 | self.mode)

            def chmod(self, mode):
                self.chmods.append(mode)

        strict, loose = FakePath(0o400), FakePath(0o644)
        _restrict_mode(strict, 0o600)
        _restrict_mode(loose, 0o600)
        assert strict.chmods == []
        assert loose.chmods == [0o600]

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

    def test_extra_sans_added_to_generated_cert(self, tmp_path):
        """Extra DNS and IP SANs are included for new generated certs."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key), "192.0.2.24,openspeech.local")
            args = mock_run.call_args[0][0]
            addext = args[args.index("-addext") + 1]
            assert "IP:192.0.2.24" in addext
            assert "DNS:openspeech.local" in addext

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

    @pytest.mark.parametrize("present", ["cert", "key"])
    def test_half_of_a_user_supplied_pair_is_never_overwritten(self, tmp_path, present):
        """Setting only OS_SSL_CERTFILE (or a typo in OS_SSL_KEYFILE) must not destroy the file that exists."""
        cert, key = tmp_path / "fullchain.pem", tmp_path / "privkey.pem"
        existing, missing = (cert, key) if present == "cert" else (key, cert)
        existing.write_text("USER FILE")
        fake = _fake_openssl()

        with patch("src.ssl_utils.subprocess.run", side_effect=fake), \
             pytest.raises(RuntimeError, match="OS_SSL_CERTFILE and OS_SSL_KEYFILE") as refused:
            ensure_ssl_certs(str(cert), str(key))

        assert str(existing) in str(refused.value) and str(missing) in str(refused.value)
        assert _generations(fake) == 0
        assert existing.read_text() == "USER FILE"
        assert not missing.exists()
        assert not (tmp_path / "fullchain.pem.generated").exists()

    @pytest.mark.parametrize("lost", ["cert", "key"])
    def test_a_generated_pair_missing_one_file_is_generated_again(self, tmp_path, lost):
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        with patch("src.ssl_utils.subprocess.run", side_effect=_fake_openssl()):
            ensure_ssl_certs(str(cert), str(key))
        (cert if lost == "cert" else key).unlink()
        fake = _fake_openssl()

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert _generations(fake) == 1
        assert cert.exists() and key.exists()

    def test_a_lone_key_left_in_the_managed_dir_is_generated_again(self, tmp_path, monkeypatch):
        """A generation interrupted after openssl wrote the key, before markers existed."""
        monkeypatch.setattr(ssl_utils, "DEFAULT_CERT_DIR", str(tmp_path))
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        key.write_text("HALF-GENERATED KEY")
        fake = _fake_openssl()

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert _generations(fake) == 1

    def test_openssl_called_with_check_true(self, tmp_path):
        """Verifies subprocess.run is called with check=True."""
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"

        with patch("src.ssl_utils.subprocess.run") as mock_run:
            ensure_ssl_certs(str(cert), str(key))
            _, kwargs = mock_run.call_args
            assert kwargs.get("check") is True
            assert kwargs.get("capture_output") is True


class TestGeneratedCertRenewal:
    """Only a pair ensure_ssl_certs generated is renewed; other certificates are never touched."""

    def _generate(self, tmp_path):
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        with patch("src.ssl_utils.subprocess.run", side_effect=_fake_openssl()):
            ensure_ssl_certs(str(cert), str(key))
        return cert, key

    def test_generation_records_the_cert_and_key_as_generated(self, tmp_path):
        cert, key = self._generate(tmp_path)
        marker = tmp_path / "cert.pem.generated"
        assert marker.read_text().split() == [
            hashlib.sha256(cert.read_bytes()).hexdigest(),
            hashlib.sha256(key.read_bytes()).hexdigest(),
        ]

    def test_generated_cert_expiring_within_30_days_is_regenerated(self, tmp_path):
        cert, key = self._generate(tmp_path)
        old = cert.read_text()
        fake = _fake_openssl(expiring=True)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        checkend = next(args for args in fake.calls if "-checkend" in args)
        assert checkend[checkend.index("-checkend") + 1] == str(30 * 24 * 3600)
        assert _generations(fake) == 1
        assert cert.read_text() != old
        # The new certificate is recorded as well, so it is renewed in turn.
        assert (tmp_path / "cert.pem.generated").read_text().split()[0] == hashlib.sha256(cert.read_bytes()).hexdigest()

    def test_generated_cert_that_stays_valid_is_kept(self, tmp_path):
        cert, key = self._generate(tmp_path)
        old = cert.read_text()
        fake = _fake_openssl(expiring=False)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert _generations(fake) == 0
        assert cert.read_text() == old

    def test_user_supplied_cert_is_never_renewed(self, tmp_path):
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        cert.write_text("USER CERT")
        key.write_text("USER KEY")
        fake = _fake_openssl(expiring=True)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert fake.calls == []
        assert (cert.read_text(), key.read_text()) == ("USER CERT", "USER KEY")

    def test_a_cert_put_in_place_of_a_generated_one_is_left_alone(self, tmp_path):
        cert, key = self._generate(tmp_path)
        cert.write_text("USER CERT")
        fake = _fake_openssl(expiring=True)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert _generations(fake) == 0
        assert cert.read_text() == "USER CERT"

    def test_renewal_never_overwrites_a_user_key_paired_with_the_generated_cert(self, tmp_path):
        """OS_SSL_KEYFILE set later while the cert path still points at the generated cert."""
        cert, _ = self._generate(tmp_path)
        user_key = tmp_path / "private" / "privkey.pem"
        user_key.parent.mkdir()
        user_key.write_text("USER KEY")
        fake = _fake_openssl(expiring=True)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(user_key))

        assert _generations(fake) == 0
        assert user_key.read_text() == "USER KEY"

    @pytest.mark.parametrize(("subject", "renewed"), [("CN=localhost", True), ("CN=speech.example.com", False)])
    def test_default_pair_from_before_markers_is_renewed_only_if_self_generated(
        self, tmp_path, monkeypatch, subject, renewed
    ):
        monkeypatch.setattr(ssl_utils, "DEFAULT_CERT_DIR", str(tmp_path))
        monkeypatch.setattr(ssl_utils, "DEFAULT_CERT_FILE", str(tmp_path / "cert.pem"))
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        cert.write_text("OLD CERT")
        key.write_text("OLD KEY")
        fake = _fake_openssl(expiring=True, subject=subject)

        with patch("src.ssl_utils.subprocess.run", side_effect=fake):
            ensure_ssl_certs(str(cert), str(key))

        assert _generations(fake) == int(renewed)
        assert (cert.read_text() != "OLD CERT") is renewed

    @pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is not installed")
    def test_a_real_generated_cert_is_recognised_and_not_yet_due(self, tmp_path):
        cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
        ensure_ssl_certs(str(cert), str(key))

        assert ssl_utils._is_generated(cert)
        assert not ssl_utils._expires_within(cert, ssl_utils.RENEW_WITHIN_DAYS)
        assert ssl_utils._expires_within(cert, ssl_utils.CERT_VALID_DAYS + 1)
