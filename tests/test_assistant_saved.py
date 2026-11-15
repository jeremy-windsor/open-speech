"""Saved AI assistants: the store, /api/assistant/assistants, presets' ``assistant``, one client per assistant, and
``assistant`` in a realtime session.update (docs/AI-ASSISTANT.md, "Saved assistants").

No network: every AI server is an httpx.MockTransport by host (the ``servers`` fixture).
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import httpx
import pytest

from tests.test_assistant import (  # noqa: F401 - fixtures
    AI_SERVER,
    API_KEY,
    MESSAGES,
    NEW_KEY,
    _answer,
    _connection_file,
    _error,
    _settings,
    api,
    assistant_on,
    connection_api,
    connection_on,
    no_shared_client,
    servers,
)

FIRST = "http://first.local:8001/v1"
SECOND = "http://second.local:8002/v1"
ASSISTANTS = "/api/assistant/assistants"
CONNECTION = "/api/assistant/connection"


def _file(settings) -> dict:
    return json.loads(_connection_file(settings).read_text(encoding="utf-8"))


def _write_v1(settings, connection: dict) -> None:
    path = _connection_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "connection": connection}), encoding="utf-8")


def _two(connection_api):
    """Two saved assistants: First (in use) and Second."""
    first = connection_api.post(ASSISTANTS, json={"name": "First", "url": FIRST, "model": "first-model"})
    second = connection_api.post(ASSISTANTS, json={"name": "Second", "url": SECOND, "model": "second-model"})
    assert first.status_code == 201 and second.status_code == 201, (first.text, second.text)
    return first.json(), second.json()


# ── The store ────────────────────────────────────────────────────────────────


class TestStore:
    def test_a_version_1_file_reads_as_one_active_assistant_named_after_its_model(self, connection_on, monkeypatch):
        from src.assistant import connection

        _write_v1(connection_on, {"url": FIRST, "model": "hermes-agent", "api_key": NEW_KEY})
        before = _connection_file(connection_on).read_bytes()
        listed = connection.list_assistants()
        assert listed["active"] and listed["locked"] is False
        (only,) = listed["assistants"]
        assert (only["id"], only["name"], only["active"], only["url"], only["api_key_set"]) == (
            listed["active"], "hermes-agent", True, FIRST, True,
        )
        assert len(only["id"]) == 32 and int(only["id"], 16) >= 0
        assert connection.effective().api_key == NEW_KEY and connection.effective().id == only["id"]
        assert _connection_file(connection_on).read_bytes() == before, "reading never writes"
        monkeypatch.setattr(connection, "_store", connection._Store())  # a new process reads the same id
        assert connection.active_id() == only["id"]

    def test_it_is_written_back_as_version_2_on_the_next_save_with_its_fields(self, connection_on):
        from src.assistant import connection

        _write_v1(connection_on, {"url": FIRST, "api_key": NEW_KEY})
        migrated = connection.active_id()
        connection.save({"model": "your-model"})
        stored = _file(connection_on)
        assert stored["version"] == 2 and stored["active"] == migrated
        (entry,) = stored["assistants"]
        assert entry == {
            "id": migrated, "name": "qwen3-30b", "url": FIRST, "api_key": NEW_KEY,
            "api_key_origin": "http://first.local:8001", "api_key_tls_checked": True, "model": "your-model",
        }
        assert connection.effective().api_key == NEW_KEY

    @pytest.mark.parametrize(
        ("env_model", "connection", "name"),
        [("", {"url": FIRST}, "first.local"), ("", {"verify_tls": False}, "ai-server.local"),
         ("from-env", {"url": FIRST}, "from-env")],
        ids=["the-urls-host", "the-env-urls-host", "the-env-model"],
    )
    def test_without_a_model_the_name_is_the_urls_host(self, connection_on, monkeypatch, env_model, connection, name):
        from src.assistant import connection as connection_module

        monkeypatch.setattr(connection_on, "os_assistant_model", env_model)
        _write_v1(connection_on, connection)
        assert connection_module.list_assistants()["assistants"][0]["name"] == name

    def test_with_no_model_and_no_url_it_is_named_assistant(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_model", "")
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        _write_v1(connection_on, {"verify_tls": False})
        assert connection.list_assistants()["assistants"][0]["name"] == "Assistant"

    def test_a_missing_field_follows_env_per_field(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        saved = connection.create({"name": "Path only", "url": AI_SERVER + "/extra"})
        assert (saved.url, saved.model, saved.api_key) == (AI_SERVER + "/extra", "qwen3-30b", API_KEY)
        assert {name: saved.sources[name] for name in ("url", "model", "api_key", "verify_tls")} == {
            "url": "saved", "model": "env", "api_key": "env", "verify_tls": "default",
        }
        monkeypatch.setattr(connection_on, "os_assistant_model", "edited-env-model")
        assert connection.get(saved.id).model == "edited-env-model"
        assert connection.update(saved.id, {"model": "own"}).model == "own"
        assert connection.update(saved.id, {"model": None}).sources["model"] == "env", "null follows .env again"

    def test_with_nothing_saved_the_env_connection_is_used_and_listed_as_nothing(self, connection_on):
        from src.assistant import connection

        assert connection.list_assistants() == {"active": None, "locked": False, "assistants": []}
        current = connection.effective()
        assert (current.url, current.id, current.name) == (AI_SERVER, "", "")
        assert connection.active_id() is None

    def test_an_assistant_in_the_file_that_is_not_valid_is_skipped(self, connection_on, caplog):
        from src.assistant import connection

        good, clash = "a" * 32, "b" * 32
        path = _connection_file(connection_on)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 2, "active": "gone", "assistants": [
            {"id": good, "name": "Good", "url": FIRST}, {"id": "not-hex", "name": "Bad id"},
            {"id": clash, "name": "GOOD"}, {"id": "c" * 32, "name": ""}, "not an object",
        ]}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            listed = connection.list_assistants()
        assert [(a["id"], a["name"]) for a in listed["assistants"]] == [(good, "Good")]
        assert listed["active"] == good, "an active id that is gone: the first one"
        assert any("assistant-connection.json" in record.getMessage() for record in caplog.records)

    def test_the_file_is_owner_only_and_removed_with_the_last_assistant(self, connection_on):
        import os
        import stat

        from src.assistant import connection

        saved = connection.create({"name": "Only", "url": FIRST})
        path = _connection_file(connection_on)
        if os.name != "nt":
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
        connection.delete(saved.id)
        assert not path.exists() and connection.effective().url == AI_SERVER


# ── The API ──────────────────────────────────────────────────────────────────


class TestAssistantsApi:
    def test_the_list_holds_each_assistants_view_with_its_id_name_and_whether_it_is_in_use(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        first, second = _two(connection_api)
        response = connection_api.get(ASSISTANTS)
        assert response.status_code == 200
        listed = response.json()
        assert listed == {"active": first["id"], "locked": False, "assistants": [first, second]}
        assert first["active"] is True and second["active"] is False
        assert set(first) == {"id", "name", "active", "enabled", "locked", "url", "model", "api_key_set",
                              "api_key_tls_checked", "verify_tls", "template_kwargs", "max_tokens_field",
                              "extra_body", "stream", "sources"}
        assert API_KEY not in response.text and "api_key" not in first

    def test_post_answers_201_the_first_one_is_in_use_and_the_shared_client_follows_it(self, connection_api, servers, no_shared_client):
        first_server = servers.add(FIRST, lambda _r: _answer(content="From first."))
        response = connection_api.post(ASSISTANTS, json={"name": "  First  ", "url": FIRST, "api_key": NEW_KEY})
        assert response.status_code == 201, response.text
        created = response.json()
        assert (created["name"], created["active"], created["api_key_set"]) == ("First", True, True)
        assert NEW_KEY not in response.text
        assert no_shared_client.get_client().host == "first.local:8001"
        status = connection_api.get("/api/assistant").json()
        assert status["assistant"] == {"id": created["id"], "name": "First"} and status["host"] == "first.local:8001"
        tested = connection_api.post("/api/assistant/test", json={})
        assert tested.json()["reply"] == "From first." and len(first_server.requests) == 1

    @pytest.mark.parametrize(
        ("body", "param"),
        [({"url": FIRST}, "name"), ({"name": ""}, "name"), ({"name": "   "}, "name"), ({"name": "n" * 61}, "name"),
         ({"name": 5}, "name"), ({"name": "X", "url": "ftp://x/v1"}, "url"), ({"name": "X", "surprise": 1}, "surprise")],
    )
    def test_a_bad_name_or_field_gets_422_and_saves_nothing(self, connection_api, connection_on, body, param):
        response = connection_api.post(ASSISTANTS, json=body)
        assert response.status_code == 422, response.text
        assert (_error(response)["code"], _error(response)["param"]) == ("validation_error", param)
        assert not _connection_file(connection_on).exists()

    def test_names_are_unique_ignoring_case(self, connection_api):
        first, second = _two(connection_api)
        clash = connection_api.post(ASSISTANTS, json={"name": "fIrSt", "url": SECOND})
        assert clash.status_code == 409 and _error(clash)["code"] == "assistant_exists"
        renamed = connection_api.put(f"{ASSISTANTS}/{second['id']}", json={"name": "FIRST"})
        assert renamed.status_code == 409 and _error(renamed)["code"] == "assistant_exists"
        own = connection_api.put(f"{ASSISTANTS}/{first['id']}", json={"name": "FIRST"})
        assert own.status_code == 200 and own.json()["name"] == "FIRST", "its own name in another case"
        assert len(connection_api.get(ASSISTANTS).json()["assistants"]) == 2

    def test_put_saves_the_name_and_fields_by_todays_rules(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        first, _second = _two(connection_api)
        path = f"{ASSISTANTS}/{first['id']}"
        saved = connection_api.put(path, json={"name": "Renamed", "api_key": NEW_KEY, "stream": False}).json()
        assert (saved["name"], saved["api_key_set"], saved["stream"], saved["url"], saved["active"]) == (
            "Renamed", True, False, FIRST, True,
        )
        cleared = connection_api.put(path, json={"api_key": ""}).json()
        assert (cleared["api_key_set"], cleared["sources"]["api_key"]) == (False, "saved")
        back = connection_api.put(path, json={"url": None, "api_key": None}).json()
        assert (back["url"], back["sources"]["url"], back["api_key_set"]) == (AI_SERVER, "env", True)
        assert connection_api.put(path, json={"url": "nope"}).status_code == 422

    def test_an_unknown_id_gets_404(self, connection_api):
        _two(connection_api)
        for method, path, body in (("PUT", f"{ASSISTANTS}/{'f' * 32}", {"name": "X"}),
                                   ("DELETE", f"{ASSISTANTS}/{'f' * 32}", None),
                                   ("POST", f"{ASSISTANTS}/{'f' * 32}/activate", None),
                                   ("POST", f"{ASSISTANTS}/{'f' * 32}/check", {})):
            response = connection_api.request(method, path, json=body)
            assert response.status_code == 404, (method, path)
            assert _error(response)["code"] == "assistant_not_found"

    def test_activate_puts_one_in_use_and_rebuilds_the_shared_client(self, connection_api, servers, no_shared_client):
        first, second = _two(connection_api)
        servers.add(FIRST)
        second_server = servers.add(SECOND, lambda _r: _answer(content="From second."))
        response = connection_api.post(f"{ASSISTANTS}/{second['id']}/activate")
        assert response.status_code == 200
        assert (response.json()["id"], response.json()["active"]) == (second["id"], True)
        assert no_shared_client.get_client().host == "second.local:8002"
        listed = connection_api.get(ASSISTANTS).json()
        assert listed["active"] == second["id"] and [a["active"] for a in listed["assistants"]] == [False, True]
        assert connection_api.get(CONNECTION).json()["url"] == SECOND
        assert connection_api.get("/api/assistant").json()["assistant"] == {"id": second["id"], "name": "Second"}
        assert connection_api.post("/api/assistant/test", json={}).json()["reply"] == "From second."
        assert second_server.body["model"] == "second-model", "the presets' default model is the active one's"

    def test_deleting_the_active_one_puts_the_first_left_in_use_and_the_last_leaves_env(self, connection_api, connection_on, no_shared_client):
        first, second = _two(connection_api)
        third = connection_api.post(ASSISTANTS, json={"name": "Third", "url": FIRST}).json()
        assert connection_api.post(f"{ASSISTANTS}/{third['id']}/activate").status_code == 200
        deleted = connection_api.delete(f"{ASSISTANTS}/{third['id']}")
        assert deleted.status_code == 204 and deleted.content == b""
        assert connection_api.get(ASSISTANTS).json()["active"] == first["id"]
        assert no_shared_client.get_client().host == "first.local:8001"
        assert connection_api.delete(f"{ASSISTANTS}/{second['id']}").status_code == 204
        assert connection_api.get(ASSISTANTS).json()["active"] == first["id"], "not the active one: no change"
        assert connection_api.delete(f"{ASSISTANTS}/{first['id']}").status_code == 204
        assert connection_api.get(ASSISTANTS).json() == {"active": None, "locked": False, "assistants": []}
        assert not _connection_file(connection_on).exists()
        assert connection_api.get(CONNECTION).json()["sources"]["url"] == "env"
        assert no_shared_client.get_client().host == "ai-server.local:8000"
        assert connection_api.get("/api/assistant").json()["assistant"] is None

    def test_check_tries_that_assistants_connection_with_the_bodys_fields(self, connection_api, servers):
        first, second = _two(connection_api)
        connection_api.put(f"{ASSISTANTS}/{second['id']}", json={"api_key": NEW_KEY})
        second_server = servers.add(SECOND, lambda _r: httpx.Response(200, json={"data": [{"id": "second-model"}]}))
        response = connection_api.post(f"{ASSISTANTS}/{second['id']}/check", json={})
        assert response.status_code == 200, response.text
        assert response.json()["models"] == ["second-model"]
        assert second_server.requests[0].headers["authorization"] == f"Bearer {NEW_KEY}"
        trial = connection_api.post(f"{ASSISTANTS}/{second['id']}/check", json={"url": "http://second.local:8002/api/v1"})
        assert trial.status_code == 200 and str(second_server.requests[-1].url) == "http://second.local:8002/api/v1/models"
        assert connection_api.get(f"{ASSISTANTS}").json()["assistants"][1]["url"] == SECOND, "nothing saved"

    def test_when_locked_the_list_is_empty_and_every_change_answers_403(self, connection_api, connection_on, monkeypatch, no_shared_client):
        first, _second = _two(connection_api)
        before = _connection_file(connection_on).read_bytes()
        monkeypatch.setattr(connection_on, "os_assistant_connection_locked", True)
        assert connection_api.get(ASSISTANTS).json() == {"active": None, "locked": True, "assistants": []}
        assert connection_api.get(CONNECTION).json()["url"] == AI_SERVER
        assert connection_api.get("/api/assistant").json()["assistant"] is None
        for method, path, body in (("POST", ASSISTANTS, {"name": "New"}),
                                   ("PUT", f"{ASSISTANTS}/{first['id']}", {"name": "X"}),
                                   ("DELETE", f"{ASSISTANTS}/{first['id']}", None),
                                   ("POST", f"{ASSISTANTS}/{first['id']}/activate", None),
                                   ("POST", f"{ASSISTANTS}/{first['id']}/check", {})):
            response = connection_api.request(method, path, json=body)
            assert response.status_code == 403, (method, path)
            assert _error(response)["code"] == "connection_locked"
        assert _connection_file(connection_on).read_bytes() == before

    def test_the_routes_work_while_the_assistant_is_off(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        assert connection_api.get(ASSISTANTS).status_code == 200
        created = connection_api.post(ASSISTANTS, json={"name": "First", "url": FIRST})
        assert created.status_code == 201 and connection_api.get("/api/assistant").json()["enabled"] is True

    def test_the_routes_need_the_api_key_when_one_is_set(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_api_key", "open-speech-key")
        for method, path in (("GET", ASSISTANTS), ("POST", ASSISTANTS), ("PUT", f"{ASSISTANTS}/x"),
                             ("DELETE", f"{ASSISTANTS}/x"), ("POST", f"{ASSISTANTS}/x/activate"),
                             ("POST", f"{ASSISTANTS}/x/check")):
            assert connection_api.request(method, path, json={"name": "X"}).status_code == 401, (method, path)


class TestConnectionCompat:
    def test_put_with_nothing_saved_saves_the_first_assistant(self, connection_api, connection_on):
        response = connection_api.put(CONNECTION, json={"url": FIRST, "model": "hermes-agent"})
        assert response.status_code == 200
        assert "id" not in response.json() and "name" not in response.json(), "today's view"
        (only,) = connection_api.get(ASSISTANTS).json()["assistants"]
        assert (only["name"], only["active"], only["url"]) == ("hermes-agent", True, FIRST)
        view = connection_api.get(CONNECTION).json()
        assert view == {key: value for key, value in only.items() if key not in ("id", "name", "active")}

    def test_the_connection_routes_act_on_the_active_assistant(self, connection_api, servers):
        first, second = _two(connection_api)
        connection_api.post(f"{ASSISTANTS}/{second['id']}/activate")
        assert connection_api.get(CONNECTION).json()["model"] == "second-model"
        assert connection_api.put(CONNECTION, json={"model": "changed"}).json()["model"] == "changed"
        listed = {a["name"]: a["model"] for a in connection_api.get(ASSISTANTS).json()["assistants"]}
        assert listed == {"First": "first-model", "Second": "changed"}
        servers.add(SECOND, lambda _r: httpx.Response(200, json={"data": [{"id": "changed"}]}))
        assert connection_api.post("/api/assistant/connection/check", json={}).json()["models"] == ["changed"]
        forgotten = connection_api.delete(CONNECTION)
        assert forgotten.status_code == 200 and forgotten.json()["url"] == FIRST, "the first one left is in use"
        assert [a["name"] for a in connection_api.get(ASSISTANTS).json()["assistants"]] == ["First"]


# ── Presets' assistant ───────────────────────────────────────────────────────


class TestPresetAssistant:
    def test_a_preset_names_an_assistant_by_id_or_name_and_its_model_is_the_default(self, connection_on):
        from src.assistant import connection, presets

        first = connection.create({"name": "First", "url": FIRST, "model": "first-model"})
        second = connection.create({"name": "Second", "url": SECOND, "model": "second-model"})
        assert presets.get_preset("voice")["assistant"] == ""
        assert presets.effective("voice").assistant == first.id, "\"\" is the active one"
        view = presets.replace_preset("voice", {"assistant": "second"})
        assert view["assistant"] == second.id and view["model"] == "second-model"
        assert presets.effective("voice").model == "second-model"
        assert presets.effective("book").model == "first-model"
        made = presets.create_preset({"name": "Mine", "assistant": first.id})
        assert made["assistant"] == first.id
        stored = json.loads((_connection_file(connection_on).parent / "assistant-presets.json").read_text("utf-8"))
        assert {entry["id"]: entry.get("assistant") for entry in stored["presets"]} == {
            "voice": second.id, "mine": first.id,
        }

    def test_a_put_that_leaves_it_out_keeps_it_and_empty_or_null_is_the_active_one(self, connection_on):
        from src.assistant import connection, presets

        connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        presets.replace_preset("voice", {"assistant": second.id})
        assert presets.replace_preset("voice", {"temperature": 0.5})["assistant"] == second.id
        assert presets.replace_preset("voice", {"assistant": ""})["assistant"] == ""
        presets.replace_preset("voice", {"assistant": second.id})
        assert presets.replace_preset("voice", {"assistant": None})["assistant"] == ""

    def test_an_unknown_assistant_is_refused(self, connection_api, connection_on):
        from src.assistant import connection, presets

        connection.create({"name": "First", "url": FIRST})
        response = connection_api.put("/api/assistant/presets/voice", json={"assistant": "nobody"})
        assert response.status_code == 422 and _error(response)["param"] == "assistant"
        error = pytest.raises(presets.PresetError, presets.effective, "voice", {"assistant": "nobody"}).value
        assert error.field == "assistant" and "Unknown AI assistant 'nobody'" == error.message

    def test_a_preset_whose_assistant_is_gone_uses_the_active_one(self, connection_on):
        from src.assistant import connection, presets

        first = connection.create({"name": "First", "url": FIRST, "model": "first-model"})
        second = connection.create({"name": "Second", "url": SECOND, "model": "second-model"})
        presets.replace_preset("voice", {"assistant": second.id})
        connection.delete(second.id)
        assert presets.get_preset("voice")["assistant"] == ""
        settings = presets.effective("voice")
        assert (settings.assistant, settings.model) == (first.id, "first-model")

    def test_an_override_picks_an_assistant_for_one_request(self, connection_on):
        from src.assistant import connection, presets

        connection.create({"name": "First", "url": FIRST, "model": "first-model"})
        second = connection.create({"name": "Second", "url": SECOND, "model": "second-model"})
        settings = presets.effective("voice", {"assistant": "SECOND"})
        assert (settings.assistant, settings.model) == (second.id, "second-model")
        assert presets.get_preset("voice")["assistant"] == "", "nothing saved"

    def test_ask_ai_uses_the_presets_assistant(self, connection_api, servers):
        first, second = _two(connection_api)
        first_server = servers.add(FIRST)
        second_server = servers.add(SECOND, lambda _r: _answer(content="Second answers."))
        connection_api.put("/api/assistant/presets/voice", json={"assistant": second["id"]})
        response = connection_api.post("/api/assistant/ask", json={"input": {"kind": "text", "text": "Hi"}})
        assert response.status_code == 200, response.text
        assert response.json()["reply"] == "Second answers."
        assert second_server.body["model"] == "second-model" and first_server.requests == []


# ── One client per assistant ─────────────────────────────────────────────────


class TestClients:
    async def test_each_assistant_has_its_own_client_built_on_first_use(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        first = connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        shared = await assistant.rebuild_client()
        assert shared.host == "first.local:8001" and assistant.get_client(first.id) is shared
        assert second.id not in assistant._others, "not built before it is used"
        other = assistant.get_client(second.id)
        assert other.host == "second.local:8002" and assistant.get_client(second.id) is other
        assert assistant.get_client("f" * 32) is shared, "one that is gone: the active one's"
        assert assistant.get_client("second") is other, "by name too"
        await assistant.close_client()
        assert shared.is_closed and other.is_closed and assistant._others == {}

    async def test_a_change_rebuilds_that_assistants_client_and_a_delete_closes_it(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        shared = await assistant.rebuild_client()
        old = assistant.get_client(second.id)
        connection.update(second.id, {"url": "http://second.local:9999/v1"})
        assert await assistant.rebuild_client(None, second.id) is shared, "the shared client stays"
        assert old.is_closed and old.retired and not shared.is_closed
        new = assistant.get_client(second.id)
        assert new is not old and new.host == "second.local:9999"
        connection.delete(second.id)
        await assistant.rebuild_client(None, second.id)
        assert new.is_closed and second.id not in assistant._others and assistant.get_client() is shared

    async def test_a_retired_client_sends_new_requests_to_its_own_assistants_current_client(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        await assistant.rebuild_client()
        first_server = servers.add(FIRST)
        moved = servers.add("http://moved.local/v1", lambda _r: _answer(content="Moved."))
        old = assistant.get_client(second.id)
        connection.update(second.id, {"url": "http://moved.local/v1"})
        await assistant.rebuild_client(None, second.id)
        assert (await old.reply(MESSAGES, _settings())).text == "Moved."
        assert len(moved.requests) == 1 and first_server.requests == []

    async def test_a_job_finishes_on_the_assistant_it_started_with(self, connection_on, servers, no_shared_client):
        from src.assistant import connection, presets

        assistant = no_shared_client
        first = connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        first_server = servers.add(FIRST, lambda _r: _answer(content="First."))
        second_server = servers.add(SECOND, lambda _r: _answer(content="Second."))
        shared = await assistant.rebuild_client()
        started = presets.effective("book")  # what a book job keeps
        assert started.assistant == first.id
        connection.activate(second.id)
        await assistant.rebuild_client()
        assert assistant.get_client().host == "second.local:8002"
        kept = assistant.get_client(started.assistant)
        assert kept is shared and not shared.retired and not shared.is_closed
        assert (await kept.reply(MESSAGES, started)).text == "First."
        assert presets.effective("book").assistant == second.id, "a new job uses the one in use"
        assert len(first_server.requests) == 1 and second_server.requests == []
        connection.activate(first.id)
        assert await assistant.rebuild_client() is shared, "its client comes back as the shared one"

    async def test_the_reader_runner_asks_the_jobs_assistant(self, connection_on, servers, no_shared_client):
        from src.assistant import connection, presets
        from src.reader.assistant_jobs import AssistantJobs

        assistant = no_shared_client
        connection.create({"name": "First", "url": FIRST})
        second = connection.create({"name": "Second", "url": SECOND})
        servers.add(FIRST)
        second_server = servers.add(SECOND, lambda _r: _answer(content="From second."))
        await assistant.rebuild_client()
        presets.replace_preset("book", {"assistant": second.id})
        settings = presets.effective("book")
        reply = await AssistantJobs(lambda: None)._ask(MESSAGES, settings)
        assert reply.text == "From second." and len(second_server.requests) == 1


# ── Talk: /v1/realtime ───────────────────────────────────────────────────────


@pytest.fixture
def two_assistants(connection_on):
    from src.assistant import connection

    return (connection.create({"name": "First", "url": FIRST, "model": "first-model"}),
            connection.create({"name": "Second", "url": SECOND, "model": "second-model"}))


class TestRealtime:
    @pytest.mark.parametrize("block", [{"assistant": "second"}, "SECOND", {"preset": "voice", "assistant": "Second"}],
                             ids=["in-the-block", "as-text", "with-the-preset"])
    def test_session_update_picks_an_assistant_by_id_or_name_for_the_session(self, two_assistants, block):
        from src.assistant import presets
        from src.realtime.session import SessionConfig

        _first, second = two_assistants
        config = SessionConfig(assistant_on=True)
        config.update_from({"session": {"assistant": block}})
        settings = config.assistant_settings()
        assert (settings.assistant, settings.model) == (second.id, "second-model")
        assert presets.get_preset("voice")["assistant"] == "", "for this session only"
        config.update_from({"session": {"assistant": {"assistant": None}}})
        assert config.assistant_settings().assistant == two_assistants[0].id

    def test_an_unknown_assistant_refuses_the_update_as_an_unknown_preset_does(self, two_assistants):
        from src.realtime.session import SessionConfig

        config = SessionConfig(assistant_on=True)
        with pytest.raises(ValueError) as unknown_preset:
            config.update_from({"session": {"assistant": {"preset": "nobody"}}})
        with pytest.raises(ValueError) as unknown_assistant:
            config.update_from({"session": {"assistant": {"assistant": "nobody"}}})
        assert str(unknown_preset.value) == "Invalid AI assistant setting: Unknown AI preset 'nobody'"
        assert str(unknown_assistant.value) == "Invalid AI assistant setting: Unknown AI assistant 'nobody'"
        assert config.assistant_overrides == {}, "the session stays as it was"

    async def test_each_turn_uses_its_assistants_client(self, two_assistants, servers, no_shared_client):
        from src.assistant import connection
        from src.realtime.server import RealtimeSession, _NoServer

        assistant = no_shared_client
        first, second = two_assistants
        shared = await assistant.rebuild_client()
        session = RealtimeSession(MagicMock(), MagicMock(), assistant=shared)
        assert session._client_for(session.config.assistant_settings()) is shared
        session.config.update_from({"session": {"assistant": "second"}})
        turn_client = session._client_for(session.config.assistant_settings())
        assert turn_client is assistant.get_client(second.id) and turn_client.host == "second.local:8002"
        session.config.update_from({"session": {"assistant": {"assistant": None}}})
        connection.activate(second.id)
        await assistant.rebuild_client()
        assert session._client_for(session.config.assistant_settings()) is assistant.get_client(), (
            "the next turn uses the one now in use"
        )
        connection.update(second.id, {"url": ""})
        await assistant.rebuild_client(None, second.id)
        assert isinstance(session._client_for(session.config.assistant_settings()), _NoServer)

    def test_with_nothing_saved_a_turn_uses_the_sessions_client(self, connection_on):
        from src.realtime.server import RealtimeSession

        fake = MagicMock()
        session = RealtimeSession(MagicMock(), MagicMock(), assistant=fake)
        assert session._client_for(session.config.assistant_settings()) is fake
