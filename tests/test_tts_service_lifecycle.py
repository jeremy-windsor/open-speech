"""Safety checks for direct TTS model loading."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.services.tts import load_tts_model


def test_failed_unload_prevents_replacement_tts_load():
    router = MagicMock()
    router.loaded_models.return_value = [SimpleNamespace(model="kokoro")]
    router.unload_model.side_effect = RuntimeError("Kokoro is still in memory")

    with pytest.raises(HTTPException) as error:
        load_tts_model(
            settings=SimpleNamespace(tts_enabled=True),
            tts_router=router,
            model_id="qwen3/0.6b-custom-voice",
        )

    assert error.value.status_code == 500
    router.load_model.assert_not_called()
