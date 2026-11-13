"""The suite never writes outside its temporary folder.

src.config builds its `settings` once, at first import. tests/conftest.py points every data path at a temporary
folder and then imports src.config, so the settings always hold those paths, whatever a test does to os.environ
later. Without that import, tests/test_env_compat.py run on its own first imported src.config inside a
cleared environment, and a later `from src import main` created the app's data folders at the default
/home/openspeech (D:\\home\\openspeech on Windows).
"""

import os
import sys
from pathlib import Path


def test_settings_were_built_by_conftest_before_any_test_ran():
    assert "src.config" in sys.modules, "tests/conftest.py must import src.config after it sets the test paths"


def test_every_data_path_in_the_settings_is_the_test_environments(tmp_path_factory):
    """Each path is conftest's, or a test's own tmp_path.

    Many test modules point a path at their own tmp_path for the rest of the session (test_history_header.py,
    test_batch.py, test_reader_api.py: `main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")`), and
    this file runs after some of them, so a path may hold one of those instead of conftest's. Both are inside the
    suite's temporary folders; the default /home/openspeech is in neither.
    """
    from src.config import settings

    basetemp = tmp_path_factory.getbasetemp().resolve()
    for setting, variable in (
        ("os_voice_library_path", "OS_VOICE_LIBRARY_PATH"),
        ("os_studio_db_path", "OS_STUDIO_DB_PATH"),
        ("os_conversations_dir", "OS_CONVERSATIONS_DIR"),
        ("os_composer_dir", "OS_COMPOSER_DIR"),
        ("tts_cache_dir", "TTS_CACHE_DIR"),
        ("reader_cache_dir", "READER_CACHE_DIR"),
    ):
        value = Path(getattr(settings, setting))
        from_a_test = value.resolve().is_relative_to(basetemp)
        assert value == Path(os.environ[variable]) or from_a_test, (setting, value)
        assert "openspeech" not in value.parts[:3], (setting, value)
