"""Global test configuration."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

# Disable SSL in tests so TestClient works over plain HTTP
os.environ["STT_SSL_ENABLED"] = "false"

# Import-time services need writable paths before individual test fixtures run.
# Keep their persistent data separate from the checkout and the running service.
_test_data = TemporaryDirectory(prefix="open-speech-tests-")
_test_root = Path(_test_data.name)
os.environ["OS_VOICE_LIBRARY_PATH"] = str(_test_root / "voices")
os.environ["OS_STUDIO_DB_PATH"] = str(_test_root / "studio.db")
os.environ["OS_CONVERSATIONS_DIR"] = str(_test_root / "conversations")
os.environ["OS_COMPOSER_DIR"] = str(_test_root / "composer")
os.environ["OS_PROVIDERS_DIR"] = str(_test_root / "providers")
os.environ["TTS_CACHE_DIR"] = str(_test_root / "tts-cache")
