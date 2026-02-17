"""Tests for environment variable backwards compatibility."""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest


class TestEnvVarBackwardsCompat:
    """Old env var names should still work."""

    def test_old_stt_port_works(self):
        """STT_PORT should set os_port via deprecation compat."""
        env = {"STT_PORT": "9999", "STT_SSL_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=False):
            # Re-run the deprecation check
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_PORT" in warnings
            # OS_PORT should be set in environ
            assert os.environ.get("OS_PORT") == "9999"
            # Clean up
            if "OS_PORT" in os.environ:
                del os.environ["OS_PORT"]

    def test_old_stt_host_works(self):
        env = {"STT_HOST": "127.0.0.1"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_HOST" in warnings
            assert os.environ.get("OS_HOST") == "127.0.0.1"
            if "OS_HOST" in os.environ:
                del os.environ["OS_HOST"]

    def test_new_name_takes_precedence(self):
        """If both old and new are set, new wins (old is ignored)."""
        env = {"STT_PORT": "9999", "OS_PORT": "7777"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            _check_deprecated_env_vars()
            # OS_PORT should stay as 7777
            assert os.environ.get("OS_PORT") == "7777"

    def test_old_stt_default_model_works(self):
        env = {"STT_DEFAULT_MODEL": "my-model"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_DEFAULT_MODEL" in warnings
            assert os.environ.get("STT_MODEL") == "my-model"
            if "STT_MODEL" in os.environ:
                del os.environ["STT_MODEL"]

    def test_old_tts_default_model_works(self):
        env = {"TTS_DEFAULT_MODEL": "piper"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "TTS_DEFAULT_MODEL" in warnings
            assert os.environ.get("TTS_MODEL") == "piper"
            if "TTS_MODEL" in os.environ:
                del os.environ["TTS_MODEL"]

    def test_old_tts_default_voice_works(self):
        env = {"TTS_DEFAULT_VOICE": "echo"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "TTS_DEFAULT_VOICE" in warnings
            assert os.environ.get("TTS_VOICE") == "echo"
            if "TTS_VOICE" in os.environ:
                del os.environ["TTS_VOICE"]

    def test_old_stt_api_key_works(self):
        env = {"STT_API_KEY": "my-secret"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_API_KEY" in warnings
            assert os.environ.get("OS_API_KEY") == "my-secret"
            if "OS_API_KEY" in os.environ:
                del os.environ["OS_API_KEY"]


class TestDeprecationWarnings:
    def test_deprecation_warning_logged(self, caplog):
        """Deprecated env var names should produce log warnings."""
        from src.config import log_deprecation_warnings
        with caplog.at_level(logging.WARNING):
            log_deprecation_warnings({"STT_PORT": "OS_PORT"})
        assert "Deprecated env var 'STT_PORT'" in caplog.text
        assert "use 'OS_PORT'" in caplog.text

    def test_no_warnings_for_new_names(self):
        """Using new names should not produce deprecation warnings."""
        env = {"OS_PORT": "8100"}
        with patch.dict(os.environ, env, clear=False):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "OS_PORT" not in warnings


class TestSettingsProperties:
    """Test that old property names still work as read accessors."""

    def test_stt_port_property(self):
        from src.config import Settings
        s = Settings(os_port=9999)
        assert s.stt_port == 9999

    def test_stt_host_property(self):
        from src.config import Settings
        s = Settings(os_host="127.0.0.1")
        assert s.stt_host == "127.0.0.1"

    def test_stt_default_model_property(self):
        from src.config import Settings
        s = Settings(stt_model="my-model")
        assert s.stt_default_model == "my-model"

    def test_tts_default_model_property(self):
        from src.config import Settings
        s = Settings(tts_model="piper")
        assert s.tts_default_model == "piper"

    def test_tts_default_voice_property(self):
        from src.config import Settings
        s = Settings(tts_voice="echo")
        assert s.tts_default_voice == "echo"

    def test_tts_default_speed_property(self):
        from src.config import Settings
        s = Settings(tts_speed=1.5)
        assert s.tts_default_speed == 1.5

    def test_stt_model_ttl_property(self):
        from src.config import Settings
        s = Settings(os_model_ttl=600)
        assert s.stt_model_ttl == 600

    def test_stt_max_loaded_models_property(self):
        from src.config import Settings
        s = Settings(os_max_loaded_models=5)
        assert s.stt_max_loaded_models == 5
