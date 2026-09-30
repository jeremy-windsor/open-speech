"""Global test configuration."""

import gc
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

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


@pytest.fixture
def create_symlink():
    def create(link, target, *, target_is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows account lacks symbolic-link privilege")
            raise

    return create


def pytest_sessionfinish(session, exitstatus):
    # Windows cannot remove SQLite files while connections still own them.
    # Close import-time stores before TemporaryDirectory's exit finalizer runs.
    storage = sys.modules.get("src.storage")
    if storage is not None and storage._conn is not None:
        storage._conn.close()
        storage._conn = None
    main = sys.modules.get("src.main")
    if main is not None and main.batch_store._conn is not None:
        main.batch_store._conn.close()
        main.batch_store._conn = None
    gc.collect()
    _test_data.cleanup()
