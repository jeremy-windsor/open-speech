"""Tests for environment variable backwards compatibility."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
ENV_EXAMPLE = COMPOSE_FILE.parent / ".env.example"
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:?-)(.*))?\}\Z", re.DOTALL)


def _interpolate(value: str, env: dict[str, str]) -> str:
    """Compose interpolation of $$, ${NAME}, ${NAME:-default} and ${NAME-default}, nested defaults included."""
    out, i = [], 0
    while i < len(value):
        if value.startswith("$$", i):
            out.append("$")
            i += 2
        elif value.startswith("${", i):
            depth, j = 1, i + 2
            while depth:
                if value.startswith("${", j):
                    depth, j = depth + 1, j + 2
                else:
                    depth -= value[j] == "}"
                    j += 1
            name, op, default = _VAR.match(value[i:j]).groups()
            current = env.get(name)
            use_default = op == ":-" and not current or op == "-" and current is None
            out.append(_interpolate(default, env) if use_default else current or "")
            i = j
        else:
            out.append(value[i])
            i += 1
    return "".join(out)


def _settings_under_compose(dotenv: dict[str, str]):
    """Settings as the shipped Compose file builds them from a project .env."""
    from src.config import Settings, _check_deprecated_env_vars

    service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
    container_env = dict(dotenv)  # env_file: .env
    for entry in service["environment"]:
        name, _, raw = entry.partition("=")
        container_env[name] = _interpolate(raw, dotenv)  # environment: wins over env_file
    with patch.dict(os.environ, container_env, clear=True):
        _check_deprecated_env_vars()
        return Settings(), container_env


class TestEnvVarBackwardsCompat:
    """Old env var names should still work."""

    def test_old_stt_port_works(self):
        """STT_PORT should set os_port via deprecation compat."""
        env = {"STT_PORT": "9999", "STT_SSL_ENABLED": "false"}
        with patch.dict(os.environ, env, clear=True):
            # Re-run the deprecation check
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_PORT" in warnings
            # OS_PORT should be set in environ
            assert os.environ.get("OS_PORT") == "9999"

    def test_old_stt_host_works(self):
        env = {"STT_HOST": "127.0.0.1"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_HOST" in warnings
            assert os.environ.get("OS_HOST") == "127.0.0.1"

    def test_new_name_takes_precedence(self):
        """If both old and new are set, new wins (old is ignored)."""
        env = {"STT_PORT": "9999", "OS_PORT": "7777"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            _check_deprecated_env_vars()
            # OS_PORT should stay as 7777
            assert os.environ.get("OS_PORT") == "7777"

    def test_old_stt_default_model_works(self):
        env = {"STT_DEFAULT_MODEL": "my-model"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_DEFAULT_MODEL" in warnings
            assert os.environ.get("STT_MODEL") == "my-model"

    def test_old_tts_default_model_works(self):
        env = {"TTS_DEFAULT_MODEL": "piper"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "TTS_DEFAULT_MODEL" in warnings
            assert os.environ.get("TTS_MODEL") == "piper"

    def test_old_tts_default_voice_works(self):
        env = {"TTS_DEFAULT_VOICE": "echo"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "TTS_DEFAULT_VOICE" in warnings
            assert os.environ.get("TTS_VOICE") == "echo"

    def test_old_stt_api_key_works(self):
        env = {"STT_API_KEY": "my-secret"}
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "STT_API_KEY" in warnings
            assert os.environ.get("OS_API_KEY") == "my-secret"

    def test_legacy_api_key_is_honored_when_compose_sets_new_name_empty(self):
        """docker-compose.yml always passes OS_API_KEY=${OS_API_KEY:-}, empty when unset."""
        env = {"STT_API_KEY": "my-secret", "OS_API_KEY": ""}
        with patch.dict(os.environ, env, clear=True):
            from src.config import Settings, _check_deprecated_env_vars
            _check_deprecated_env_vars()
            configured = Settings()
        assert configured.os_api_key == "my-secret"

    def test_empty_variable_counts_as_unset(self):
        """Compose passes ${OS_RATE_LIMIT:-} through as "": the default applies, not a parse error."""
        env = {"OS_RATE_LIMIT": "", "OS_TRUST_PROXY": "", "OS_AUTH_FAILURE_LIMIT": "", "STT_MODEL_DIR": ""}
        with patch.dict(os.environ, env, clear=True):
            from src.config import Settings
            configured = Settings()
        assert (configured.os_rate_limit, configured.os_trust_proxy) == (0, False)
        assert configured.os_auth_failure_limit == 10
        assert configured.stt_model_dir is None


# Every legacy name with a value that differs from both defaults. OS_HOST and OS_PORT stay
# fixed under Compose: the container must listen on 0.0.0.0:8100 for the port mapping.
_LEGACY_VALUES = {
    "STT_API_KEY": "legacy-key",
    "STT_CORS_ORIGINS": "https://speech.home.lan",
    "STT_TRUST_PROXY": "true",
    "STT_MAX_UPLOAD_MB": "7",
    "STT_RATE_LIMIT": "30",
    "STT_RATE_LIMIT_BURST": "5",
    "STT_SSL_ENABLED": "false",
    "STT_SSL_CERTFILE": "/certs/legacy.pem",
    "STT_SSL_KEYFILE": "/certs/legacy.key",
    "STT_MODEL_TTL": "60",
    "STT_MAX_LOADED_MODELS": "4",
    "STT_STREAM_CHUNK_MS": "250",
    "STT_STREAM_ENDPOINTING_MS": "500",
    "STT_STREAM_MAX_CONNECTIONS": "3",
    "STT_DEFAULT_MODEL": "legacy/stt-model",
    "TTS_DEFAULT_MODEL": "piper",
    "TTS_DEFAULT_VOICE": "echo",
    "TTS_DEFAULT_SPEED": "1.25",
}


class TestShippedComposeFile:
    """.env.example: old STT_* names still work, and "Compose reads .env automatically"."""

    def test_every_legacy_name_is_covered(self):
        from src.config import _DEPRECATED_ENV_MAP

        assert set(_LEGACY_VALUES) == set(_DEPRECATED_ENV_MAP.values()) - {"STT_HOST", "STT_PORT"}

    @pytest.mark.parametrize("old_name", sorted(_LEGACY_VALUES))
    def test_legacy_name_in_dotenv_is_not_shadowed_by_a_compose_default(self, old_name):
        from src.config import _DEPRECATED_ENV_MAP, Settings

        new_name = {old: new for new, old in _DEPRECATED_ENV_MAP.items()}[old_name]
        configured, _ = _settings_under_compose({old_name: _LEGACY_VALUES[old_name]})
        with patch.dict(os.environ, {}, clear=True):
            expected = Settings(**{new_name.lower(): _LEGACY_VALUES[old_name]})
        assert getattr(configured, new_name.lower()) == getattr(expected, new_name.lower())

    def test_without_dotenv_the_app_and_container_defaults_apply(self):
        configured, container_env = _settings_under_compose({})
        # The app's secure defaults are not reopened by Compose.
        assert configured.os_cors_origins == ""
        assert configured.os_allowed_hosts == ""
        assert configured.os_security_headers is True
        assert configured.os_api_docs is False
        assert (configured.os_rate_limit, configured.os_trust_proxy, configured.os_max_upload_mb) == (0, False, 100)
        # Container-specific defaults stay.
        assert (configured.os_max_loaded_models, configured.os_stream_chunk_ms) == (2, 2000)
        assert configured.stt_model == "nvidia/parakeet-tdt-0.6b-v2"
        assert (configured.os_host, configured.os_port, configured.os_ssl_enabled) == ("0.0.0.0", 8100, True)
        assert container_env["OS_SSL_ENABLED"] == "true"  # the healthcheck reads it

    def test_healthcheck_scheme_follows_a_legacy_ssl_setting(self):
        configured, container_env = _settings_under_compose({"STT_SSL_ENABLED": "false"})
        assert configured.os_ssl_enabled is False
        assert container_env["OS_SSL_ENABLED"] == "false"

    @pytest.mark.parametrize(
        "dotenv",
        [{"OS_HOST": "127.0.0.1"}, {"OS_PORT": "9000"}, {"OS_WYOMING_PORT": "10500"}, {"STT_PORT": "9000"},
         {"OS_HOST": "127.0.0.1", "OS_PORT": "9000", "OS_WYOMING_PORT": "10500"}],
        ids=repr,
    )
    def test_a_dotenv_listen_address_does_not_move_the_server_off_the_published_ports(self, dotenv):
        """OS_HOST and OS_PORT stay fixed: the container must listen on 0.0.0.0:8100 for `ports:`."""
        service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
        published = {int(_interpolate(str(entry), dotenv).rsplit(":", 1)[-1].split("/")[0])
                     for entry in service["ports"]}
        configured, container_env = _settings_under_compose(dotenv)
        assert configured.os_host == "0.0.0.0"
        assert configured.os_port in published
        assert configured.os_wyoming_port in published
        assert int(container_env["OS_PORT"]) in published  # the port the healthcheck probes

    def test_new_security_settings_pass_through(self):
        dotenv = {
            "OS_ALLOWED_HOSTS": "speech.example.com",
            "OS_SECURITY_HEADERS": "false",
            "OS_AUTH_FAILURE_LIMIT": "0",
            "OS_TRUSTED_PROXY_HOPS": "2",
            "OS_API_DOCS": "true",
        }
        configured, container_env = _settings_under_compose(dotenv)
        assert configured.os_allowed_hosts == "speech.example.com"
        assert configured.os_security_headers is False
        assert configured.os_api_docs is True and container_env["OS_API_DOCS"] == "true"
        assert (configured.os_auth_failure_limit, configured.os_trusted_proxy_hops) == (0, 2)


_EXAMPLE_SETTING = re.compile(r"^# ([A-Z][A-Z0-9_]*)=(.*)$")


def _example_setting_lines() -> list[tuple[int, str, str]]:
    """(line number, name, the line uncommented) for each `# NAME=value` line of .env.example."""
    lines = []
    for number, raw in enumerate(ENV_EXAMPLE.read_text(encoding="utf-8").splitlines(), start=1):
        match = _EXAMPLE_SETTING.match(raw)
        if match:
            lines.append((number, match.group(1), raw[2:]))
    return lines


def _docker_cli_env_file(text: str) -> dict[str, str]:
    """`docker run --env-file`, as docker/cli opts/file.go parses it: leading blanks of a line are
    dropped, a line starting with '#' is skipped, and the value is everything after the first '='
    with no trimming. A '#' later in the line is part of the value."""
    env = {}
    for line in text.splitlines():
        line = line.lstrip()
        if not line or line.startswith("#"):
            continue
        name, has_value, value = line.partition("=")
        name = name.lstrip(" \t")
        assert name and not any(c in name for c in " \t"), f"docker: bad variable name {name!r}"
        if has_value:
            env[name] = value
    return env


def _shown_value(line: str) -> str:
    """The value a reader of the line takes it to hold, as Compose and uv's --env-file read an
    unquoted value: blanks around it and a ` # comment` after it are not part of it."""
    value = line.partition("=")[2].strip(" \t")
    if value.startswith("#"):
        return ""
    value = re.split(r"[ \t]+#", value, maxsplit=1)[0].rstrip(" \t")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    return value


class TestEnvExampleUnderDockerEnvFile:
    """.env.example: "Copy to .env and edit as needed"; a plain `docker run --env-file .env` is one
    of the documented ways to use it, and that parser keeps a trailing `# comment` in the value."""

    def test_every_setting_line_has_its_shown_value_under_docker_env_file(self):
        lines = _example_setting_lines()
        assert len(lines) > 50  # the parser still finds the settings
        wrong = [
            f".env.example:{number} {name}: docker run --env-file gives {got!r}, the line shows {want!r}"
            for number, name, line in lines
            if (got := _docker_cli_env_file(line)[name]) != (want := _shown_value(line))
        ]
        assert not wrong, f"{len(wrong)} of {len(lines)} lines change meaning:\n" + "\n".join(wrong)

    def test_settings_accept_every_uncommented_line_under_docker_env_file(self):
        from pydantic import ValidationError

        from src.config import Settings

        checked, wrong = 0, []
        for number, name, line in _example_setting_lines():
            if name.lower() not in Settings.model_fields:
                continue  # a Compose or provider-worker variable, not an app setting
            checked += 1
            shown = _shown_value(line)
            with patch.dict(os.environ, {name: shown} if shown else {}, clear=True):
                expected = getattr(Settings(), name.lower())
            with patch.dict(os.environ, {name: _docker_cli_env_file(line)[name]}, clear=True):
                try:
                    got = getattr(Settings(), name.lower())
                except ValidationError as exc:
                    wrong.append(f".env.example:{number} {name}: startup fails: {exc.errors()[0]['msg']}")
                    continue
            if got != expected:
                wrong.append(f".env.example:{number} {name}: Settings has {got!r}, the line shows {expected!r}")
        assert checked > 50
        assert not wrong, f"{len(wrong)} of {checked} app settings:\n" + "\n".join(wrong)

    def test_blank_api_key_line_keeps_auth_off_under_docker_env_file(self):
        from src.config import Settings

        line = next(line for _, name, line in _example_setting_lines() if name == "OS_API_KEY")
        with patch.dict(os.environ, _docker_cli_env_file(line), clear=True):
            assert Settings().os_api_key == ""


# The README is short; the detail is in docs/*.md. Each doc that the checks below read, by what it holds.
README = COMPOSE_FILE.parent / "README.md"
DOCS = README.parent / "docs"
API_DOC = DOCS / "API.md"  # every endpoint, error and limit
CONFIG_DOC = DOCS / "CONFIGURATION.md"  # every setting, .env rules, Docker Compose, image defaults, volumes
MODELS_DOC = DOCS / "MODELS.md"  # the model tables, Parakeet, worker models, residency, the GPU memory guard
WEB_DOC = DOCS / "WEB-APP.md"
ASSISTANT_DOC = DOCS / "AI-ASSISTANT.md"  # the AI assistant's guide and reference
SECURITY_DOC = DOCS / "SECURITY.md"
DEV_DOC = DOCS / "DEVELOPMENT.md"  # the source install, the tests, building images
TROUBLE_DOC = DOCS / "TROUBLESHOOTING.md"
_PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def _assistant_setting_names() -> list[str]:
    from src.config import Settings

    return sorted(name.upper() for name in Settings.model_fields if name.startswith("os_assistant_"))


def _all_docs() -> dict[str, str]:
    """README.md and every docs/*.md, by their path from the repository root."""
    return {path.relative_to(README.parent).as_posix(): path.read_text(encoding="utf-8")
            for path in [README, *sorted(DOCS.glob("*.md"))]}


def _table_rows(doc: Path = CONFIG_DOC) -> list[list[str]]:
    """The cells of every Markdown table row in a doc (by default docs/CONFIGURATION.md, the settings tables)."""
    rows = []
    for line in doc.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if len(line) > 1 and line.startswith("|") and line.endswith("|"):
            rows.append([cell.strip() for cell in line[1:-1].split("|")])
    return rows


class TestAssistantSettingsAreDocumented:
    """Every OS_ASSISTANT_* setting is in .env.example, commented out with its source
    default ("Values shown are source defaults"); docker-compose.yml passes it through empty, in the file's style,
    so the source default applies; docs/CONFIGURATION.md's settings table lists it; and docs/API.md's endpoint tables
    list every route of the AI assistant and of the Reader's book jobs."""

    def test_the_assistant_has_its_settings(self):
        names = _assistant_setting_names()
        assert len(names) >= 20
        assert {"OS_ASSISTANT_URL", "OS_ASSISTANT_API_KEY", "OS_ASSISTANT_CONNECTION_LOCKED"} <= set(names)

    def test_env_example_shows_every_assistant_setting_commented_out_with_its_default(self):
        from src.config import Settings

        shown: dict[str, list[tuple[int, str]]] = {}
        for number, name, line in _example_setting_lines():
            shown.setdefault(name, []).append((number, _shown_value(line)))
        wrong = []
        for name in _assistant_setting_names():
            entries = shown.get(name, [])
            if len(entries) != 1:
                wrong.append(f"{name}: {len(entries)} '# {name}=...' lines in .env.example, expected one")
                continue
            number, value = entries[0]
            with patch.dict(os.environ, {}, clear=True):
                default = getattr(Settings(), name.lower())
            with patch.dict(os.environ, {name: value} if value else {}, clear=True):
                got = getattr(Settings(), name.lower())
            if got != default:
                wrong.append(f".env.example:{number} {name}={value} gives {got!r}; the source default is {default!r}")
        assert not wrong, f"{len(wrong)} AI assistant settings:\n" + "\n".join(wrong)

    def test_compose_passes_every_assistant_setting_through_empty(self):
        from src.config import Settings

        names = _assistant_setting_names()
        service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
        listed = {str(entry).partition("=")[0]: str(entry) for entry in service["environment"]}
        wrong = [f"{name}: {listed.get(name)!r}, expected {name}=${{{name}:-}}"
                 for name in names if listed.get(name) != f"{name}=${{{name}:-}}"]
        assert not wrong, "docker-compose.yml environment:\n" + "\n".join(wrong)

        configured, _ = _settings_under_compose({})
        with patch.dict(os.environ, {}, clear=True):
            source = Settings()
        assert {n: getattr(configured, n.lower()) for n in names} == {n: getattr(source, n.lower()) for n in names}

        dotenv = {
            "OS_ASSISTANT_URL": "http://ai-server.lan:8080/v1",
            "OS_ASSISTANT_MODEL": "your-model",
            "OS_ASSISTANT_TEMPLATE_KWARGS": "true",
            "OS_ASSISTANT_REASONING_EFFORT": "low",
            "OS_ASSISTANT_CONNECTION_LOCKED": "true",
            "OS_ASSISTANT_EXTRA_BODY": '{"top_k": 20}',
            "OS_ASSISTANT_TIMEOUT_S": "60",
        }
        configured, _ = _settings_under_compose(dotenv)
        assert configured.os_assistant_url == "http://ai-server.lan:8080/v1"
        assert configured.os_assistant_model == "your-model"
        assert configured.os_assistant_template_kwargs is True
        assert configured.os_assistant_reasoning_effort == "low"
        assert configured.os_assistant_connection_locked is True
        assert configured.assistant_extra_body == {"top_k": 20}
        assert configured.os_assistant_timeout_s == 60

    def test_the_settings_tables_list_every_assistant_setting(self):
        first_cells = [row[0] for row in _table_rows() if row]
        missing = [name for name in _assistant_setting_names()
                   if not any(f"`{name}`" in cell for cell in first_cells)]
        assert not missing, "docs/CONFIGURATION.md settings tables are missing: " + ", ".join(missing)

    def test_the_settings_tables_list_every_setting(self):
        """Every server setting has a row of its own in docs/CONFIGURATION.md, with its source default."""
        from src.config import Settings

        rows = {row[0]: row for row in _table_rows() if len(row) == 3}
        names = sorted(name.upper() for name in Settings.model_fields)
        assert len(names) > 100
        missing = [name for name in names if f"`{name}`" not in rows]
        assert not missing, "docs/CONFIGURATION.md settings tables are missing: " + ", ".join(missing)

    def test_readme_endpoint_tables_list_every_assistant_route(self):
        from src.routes import assistant as assistant_routes
        from src.routes import reader as reader_routes

        unused = lambda: None  # noqa: E731 - the factories only keep these
        routers = [
            assistant_routes.create_router(get_settings=unused),
            reader_routes.create_router(get_settings=unused, get_store=unused, get_renderer=unused,
                                        get_signer=unused),
        ]
        wanted = sorted({
            (method, _PLACEHOLDER.sub("{}", route.path))
            for router in routers for route in router.routes
            if "assistant" in route.path for method in route.methods if method != "HEAD"
        })
        rows = [row for row in _table_rows(API_DOC) if len(row) > 1]
        missing = [f"{method} {path}" for method, path in wanted
                   if not any(_PLACEHOLDER.sub("{}", row[1]) == f"`{path}`" and f"`{method}`" in row[0]
                              for row in rows)]
        assert len(wanted) >= 15
        assert not missing, "docs/API.md endpoint tables are missing:\n" + "\n".join(missing)


def _doc_part(doc: Path, start: str, end: str) -> str:
    """A doc from the line that starts with ``start`` up to the next line that starts with ``end``."""
    text = doc.read_text(encoding="utf-8")
    at = text.index("\n" + start) + 1
    stop = text.find("\n" + end, at + len(start))
    return text[at:] if stop == -1 else text[at:stop]


def _readme_part(start: str, end: str) -> str:
    return _doc_part(README, start, end)


def _setting_row(name: str) -> str:
    """What docs/CONFIGURATION.md's settings tables say about a setting: the last cell of its row."""
    return next(row[-1] for row in _table_rows() if row and row[0] == f"`{name}`")


def _example_comment(name: str) -> str:
    """The comment lines right above `# NAME=value` in .env.example, as one line."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    at = next(number for number, found, _ in _example_setting_lines() if found == name) - 1
    first = at
    while first > 0 and lines[first - 1].startswith("#") and not _EXAMPLE_SETTING.match(lines[first - 1]):
        first -= 1
    return " ".join(" ".join(lines[first:at]).split())


class TestDocsSayWhatTheCodeDoes:
    """Places where the docs list or limit something the code
    decides, checked against the code."""

    def test_readme_lists_every_field_of_a_book_job(self):
        """The "The job" bullet names a job's fields one by one, so it names every field the
        server returns in a job; `blocks_requested` was missing."""
        from collections import defaultdict

        from src.reader import assistant_jobs

        fields = [*assistant_jobs._summary(defaultdict(lambda: None)), "settings", "chapters"]
        bullet = _doc_part(ASSISTANT_DOC, "- **The job** (`GET /v1/reader/assistant-jobs/{job_id}`)", "- **")
        missing = [name for name in fields if f"`{name}`" not in bullet]
        assert len(fields) >= 14 and "blocks_requested" in fields
        assert not missing, "docs/AI-ASSISTANT.md's list of a book job's fields is missing: " + ", ".join(missing)

    def test_readme_says_which_fields_the_book_job_list_leaves_out(self):
        """`GET /v1/reader/assistant-jobs` returns each job's summary (`_summary`), so it leaves out
        every field of a job that the summary has not: `settings` as well as `chapters`. The README said "without
        their chapters" only."""
        import ast
        import inspect
        import textwrap

        from src.reader import assistant_jobs

        # The code: a job (`view`) is `{**_summary(job), <field>: …}`; the list (`list`) is the summaries alone.
        assert "_summary(row)" in inspect.getsource(assistant_jobs.AssistantJobs.list)
        view = ast.parse(textwrap.dedent(inspect.getsource(assistant_jobs.AssistantJobs.view)))
        returned = [node.value for node in ast.walk(view) if isinstance(node, ast.Return) and node.value is not None]
        job = next(value for value in returned if isinstance(value, ast.Dict))
        left_out = [key.value for key in job.keys if isinstance(key, ast.Constant)]
        assert None in job.keys and left_out == ["settings", "chapters"], left_out

        bullet = _doc_part(ASSISTANT_DOC, "- **The job** (`GET /v1/reader/assistant-jobs/{job_id}`)", "- **")
        sentence = " ".join(bullet.split("`GET /v1/reader/assistant-jobs` lists", 1)[1].split(".")[0].split())
        unsaid = [name for name in left_out if name not in sentence]
        assert not unsaid, ("docs/AI-ASSISTANT.md's line on the list of book jobs doesn't say it leaves out: "
                            + ", ".join(unsaid))

    def test_the_input_limit_row_names_ask_ais_spoken_reply(self):
        """Ask AI refuses to speak a reply longer than `TTS_MAX_INPUT_LENGTH` (`audio_error`
        `speech_too_long`, `src/assistant/ask.py`), so the setting's row and its `.env.example` comment name it."""
        import inspect

        from src.assistant import ask

        assert "tts_max_input_length" in inspect.getsource(ask.Speaker._speak)
        places = {
            "docs/CONFIGURATION.md, the TTS_MAX_INPUT_LENGTH row": _setting_row("TTS_MAX_INPUT_LENGTH"),
            ".env.example, the comment above TTS_MAX_INPUT_LENGTH": _example_comment("TTS_MAX_INPUT_LENGTH"),
        }
        silent = [where for where, text in places.items() if "Ask AI" not in text]
        assert not silent, "These don't say the limit caps Ask AI's spoken reply: " + "; ".join(silent)

    def test_docs_say_which_realtime_reply_the_input_limit_caps(self, monkeypatch):
        """`TTS_MAX_INPUT_LENGTH` caps the text a realtime client sends to be spoken (the text
        of a `response.create`). An AI reply is never refused for its length: it is cut into segments, each within
        `TTS_LIVE_SENTENCE_MAX_CHARS` and `TTS_LIVE_SENTENCE_MAX_WORDS`. Three places said the limit caps "realtime
        replies", and the two caps' lines did not say that they cut an AI reply."""
        from src.assistant.speech import speakable_segments
        from src.config import settings

        # The code: a reply far over the input limit is spoken whole, in segments within the two caps.
        reply = "word " * 300
        monkeypatch.setattr(settings, "tts_max_input_length", 50)
        monkeypatch.setattr(settings, "tts_live_sentence_max_chars", 60)
        monkeypatch.setattr(settings, "tts_live_sentence_max_words", 1000)
        by_chars = speakable_segments(reply)
        monkeypatch.setattr(settings, "tts_live_sentence_max_chars", 5000)
        monkeypatch.setattr(settings, "tts_live_sentence_max_words", 12)
        by_words = speakable_segments(reply)
        assert len(" ".join(by_chars).split()) == len(" ".join(by_words).split()) == 300
        assert max(len(segment) for segment in by_chars) <= 60
        assert max(len(segment.split()) for segment in by_words) <= 12

        # The docs: the limit's three places name what it caps, and none says "realtime replies".
        backends = (DOCS / "TTS-BACKENDS.md").read_text(encoding="utf-8")
        over_limit = next(part for part in backends.split("\n\n") if "Input too long for <model>" in part)
        limit_places = {
            "docs/CONFIGURATION.md, the TTS_MAX_INPUT_LENGTH row": _setting_row("TTS_MAX_INPUT_LENGTH"),
            ".env.example, the comment above TTS_MAX_INPUT_LENGTH": _example_comment("TTS_MAX_INPUT_LENGTH"),
            "docs/TTS-BACKENDS.md, the paragraph on over-limit text": " ".join(over_limit.split()),
        }
        wrong = [where for where, text in limit_places.items()
                 if "realtime replies" in text or "response.create" not in text or "AI reply" not in text]
        assert not wrong, (
            "TTS_MAX_INPUT_LENGTH caps the text of a realtime `response.create`, not an AI reply; say so in: "
            + "; ".join(wrong)
        )

        # The two caps cut an AI reply as well as the Live Reader's text, and their lines say so.
        silent = [
            f"{where} {name}"
            for name in ("TTS_LIVE_SENTENCE_MAX_CHARS", "TTS_LIVE_SENTENCE_MAX_WORDS")
            for where, text in (("docs/CONFIGURATION.md, the row of", _setting_row(name)),
                                (".env.example, the comment above", _example_comment(name)))
            if "AI reply" not in text
        ]
        assert not silent, "These don't say the cap also cuts an AI reply: " + "; ".join(silent)

    def test_the_assistant_doc_names_every_output_of_ask_ai(self):
        """The Ask AI section says "every field outside these" is refused, so it names every
        field `output` takes. `output.text` was missing, and so were its two refusals."""
        from src.assistant import ask

        section = " ".join(_doc_part(ASSISTANT_DOC, "## Ask AI: send anything to the AI", "## ").split())
        unnamed = [f"output.{name}" for name in ask.OUTPUT_FIELDS
                   if f'"output": {{"{name}"' not in section and f"`output.{name}`" not in section]
        assert {"text", "speech", "document"} <= set(ask.OUTPUT_FIELDS)
        assert not unnamed, "docs/AI-ASSISTANT.md's Ask AI section doesn't name: " + ", ".join(unnamed)

        # What the code refuses for `output.text`, and the `param` it answers with, is in the section too.
        params = []
        for output in ({"text": "yes"}, {"text": False}):
            with pytest.raises(ask.AskError) as refused:
                ask._output(output, None)
            assert (refused.value.status, refused.value.code) == (422, "validation_error")
            params.append(refused.value.param)
        assert params == ["output.text", "output"]
        untold = [param for param in params if f"`param` `{param}`" not in section]
        assert not untold, ("docs/AI-ASSISTANT.md's Ask AI section doesn't give the refusal with `param`: "
                            + ", ".join(untold))


# ── Refused WebSockets, xhigh, empty uploads and the key's TLS flag, documented ─────────────────────────────────


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _changelog_release() -> str:
    """CHANGELOG.md's [0.9.0] section."""
    text = (README.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    at = text.index("## [0.9.0]")
    return text[at:text.index("\n## [", at)]


def _feature_off_reason(module) -> str:
    """The reason a WebSocket route closes with when its feature is turned off, read from the route's code."""
    import inspect

    found = re.findall(r'refuse_websocket\(\s*websocket,\s*WS_CLOSE_FEATURE_OFF,\s*"([^"]+)"', inspect.getsource(module))
    assert len(found) == 1, found
    return found[0]


class TestRefusalsEffortUploadAndTlsChangesAreDocumented:
    """A refused WebSocket opens and then closes with a code and reason (it used to
    fail the handshake), `xhigh` is a reasoning effort, an upload that decodes to nothing answers 400, and the AI
    connection's view shows `api_key_tls_checked`. The docs, `.env.example` and the CHANGELOG say so."""

    def test_the_api_reference_gives_each_websocket_refusal_its_close_code_and_reason(self):
        import inspect

        from src import middleware, streaming
        from src.routes import realtime, tts

        refusals = [
            (middleware.WS_CLOSE_KEY, middleware.WS_KEY_REFUSED),
            (middleware.WS_CLOSE_ORIGIN, middleware.WS_ORIGIN_REFUSED),
            (middleware.WS_CLOSE_FEATURE_OFF, _feature_off_reason(tts)),
            (middleware.WS_CLOSE_FEATURE_OFF, _feature_off_reason(realtime)),
        ]
        assert [code for code, _ in refusals] == [4001, 1008, 4004, 4004]
        part = _doc_part(API_DOC, "## Refused WebSockets", "## ")
        table = [line for line in part.splitlines() if line.startswith("|")]
        missing = [f"{code} {reason!r}" for code, reason in refusals
                   if not any(f"`{code}`" in row and reason in row for row in table)]
        assert not missing, "docs/API.md's Refused WebSockets table doesn't give: " + "; ".join(missing)
        section = _one_line(part)
        unsaid = [word for word in ("`400`", "`429`", "OS_ALLOWED_HOSTS", "StreamRefusedError", "1006")
                  if word not in section]
        assert not unsaid, "docs/API.md's Refused WebSockets section doesn't say: " + ", ".join(unsaid)

        # /v1/audio/stream refuses an invalid sample_rate the same way, with 1008.
        source = inspect.getsource(streaming.streaming_endpoint)
        assert "refuse_websocket(" in source and "WS_CLOSE_ORIGIN" in source
        reason = f"Invalid sample_rate: must be {streaming.MIN_SAMPLE_RATE}-{streaming.MAX_SAMPLE_RATE}"
        stream = _one_line(_doc_part(API_DOC, "### `WS /v1/audio/stream`", "## "))
        assert f"`1008 {reason}`" in stream, "WS /v1/audio/stream doesn't give the sample_rate refusal"

        # Each WebSocket route points at the table, and nothing says a refusal fails the handshake any more.
        routes = {
            "WS /v1/audio/stream": stream,
            "Live Reader": _doc_part(API_DOC, "## Live Reader: `WS /v1/audio/speech/stream`", "## "),
            "Realtime audio": _doc_part(API_DOC, "## Realtime audio", "## "),
        }
        unlinked = [name for name, text in routes.items() if "(#refused-websockets)" not in text]
        assert not unlinked, "These don't link to Refused WebSockets: " + ", ".join(unlinked)
        stale = [f"{name}: {old}" for name, text in _all_docs().items() for old in (
            "is refused with `403` (the route closes it", "A refused handshake gets `403`",
            "otherwise the handshake is refused") if old in _one_line(text)]
        assert not stale, "These still say: " + "; ".join(stale)
        for doc in ("LOCAL-LISTENER.md", "AI-ASSISTANT.md"):
            text = _one_line((DOCS / doc).read_text(encoding="utf-8"))
            assert "HTTP 403" not in text and "4001" in text, f"docs/{doc} still gives HTTP 403 for a wrong key"

    def test_env_example_gives_the_close_codes_of_a_refused_websocket(self):
        text = _one_line(ENV_EXAMPLE.read_text(encoding="utf-8"))
        assert all(code in text for code in ("4001", "1008", "4004"))
        for name in ("OS_REALTIME_ENABLED", "TTS_LIVE_ENABLED"):
            assert "4004" in _example_comment(name), f"the comment above {name} doesn't give the 4004 close"
        assert "1008" in _example_comment("OS_WS_ALLOWED_ORIGINS")

    def test_every_list_of_reasoning_efforts_names_each_one_xhigh_included(self):
        from src.config import ASSISTANT_REASONING_EFFORTS

        efforts = [effort for effort in ASSISTANT_REASONING_EFFORTS if effort]
        assert "xhigh" in efforts
        preset_row = next(row for row in _table_rows(ASSISTANT_DOC) if row[:2] == ["`reasoning_effort`", "both"])
        places = {
            "docs/AI-ASSISTANT.md, the AI presets' reasoning_effort row": preset_row[-1],
            "docs/CONFIGURATION.md, the OS_ASSISTANT_REASONING_EFFORT row": _setting_row("OS_ASSISTANT_REASONING_EFFORT"),
        }
        unnamed = [f"{where}: {effort}" for where, text in places.items() for effort in efforts
                   if f"`{effort}`" not in text]
        comment = _example_comment("OS_ASSISTANT_REASONING_EFFORT")
        unnamed += [f".env.example: {effort}" for effort in efforts if not re.search(rf"\b{effort}\b", comment)]
        assert not unnamed, "These lists of reasoning efforts leave out: " + "; ".join(unnamed)

    def test_the_api_reference_gives_the_400_for_an_upload_that_decodes_to_nothing(self):
        from src.utils.audio import EMPTY_AUDIO_MESSAGE

        upload = _one_line(_doc_part(API_DOC, "### `POST /v1/audio/transcriptions`", "### "))
        batch = _one_line(_doc_part(API_DOC, "### `POST /v1/audio/transcriptions/batch`", "### "))
        assert f"`400 {EMPTY_AUDIO_MESSAGE}`" in upload and "no audio samples" in upload
        assert f'"{EMPTY_AUDIO_MESSAGE}"' in batch and "no audio samples" in batch

    def test_the_docs_list_every_field_of_the_connection_view(self):
        from src.assistant.connection import READ_ONLY_FIELDS, Connection

        fields = list(Connection().view())
        assert "api_key_tls_checked" in fields and "api_key_tls_checked" in READ_ONLY_FIELDS
        row = next(" ".join(cells) for cells in _table_rows(API_DOC)
                   if cells[:2] == ["`GET`", "`/api/assistant/connection`"])
        section = _one_line(_doc_part(ASSISTANT_DOC, "## The connection", "## "))
        answer = re.split(r"[,\s]+", section.split("answers `{", 1)[1].split("}`", 1)[0])
        missing = [f"the API table: {name}" for name in fields if f"`{name}`" not in row]
        missing += [f"The connection: {name}" for name in fields if name not in answer]
        assert not missing, "The docs don't list these fields of GET /api/assistant/connection: " + "; ".join(missing)
        said = section.split("`api_key_tls_checked`", 1)[-1][:400]
        assert "read-only" in said and "`null`" in said

    def test_the_changelog_has_the_upgrade_note_and_says_old_reader_documents_keep_their_sentences(self):
        release = _changelog_release()
        notes = _one_line(release.split("### Upgrade notes", 1)[1].split("\n### ", 1)[0])
        note = next((bullet for bullet in notes.split("- **") if "4001" in bullet), "")
        assert all(code in note for code in ("4001", "1008", "4004")) and "/v1/realtime" in note, note
        reader = next(bullet for bullet in _one_line(release).split("- **") if "sound tags" in bullet)
        assert "imported again" in reader, reader


# ── A page on cloning models ────────────────────────────────────────────────────────────────────────────────

CLONING_DOC = README.parent / "docs" / "CLONING-MODELS.md"
VOICE_PICKER = README.parent / "src" / "static" / "js" / "components" / "voice-picker.js"


def _worker_models(monkeypatch) -> list[tuple[object, dict]]:
    """(worker module, manifest model) for each model the three GPU workers advertise, Qwen3 Base included as the
    voice-models Compose file turns it on. torch is stubbed out, as in tests/test_voice_model_workers.py."""
    import importlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    cuda = SimpleNamespace(OutOfMemoryError=type("FakeOutOfMemoryError", (Exception,), {}),
                           is_available=lambda: False, empty_cache=lambda: None)
    torch_stub = SimpleNamespace(cuda=cuda, inference_mode=lambda: nullcontext(),
                                 float16=object(), float32=object(), bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    for name in ("CHATTERBOX_MAX_INPUT_CHARS", "FISH_MAX_INPUT_CHARS", "BREEZE_MAX_INPUT_CHARS"):
        monkeypatch.delenv(name, raising=False)
    models = []
    # fish is the Fish Audio S2 worker, breeze the Breeze TTS 2 one.
    for provider in ("chatterbox", "qwen3", "fish", "breeze"):
        name = f"providers.{provider}.app"
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        models += [(module, model) for model in module.MANIFEST["models"]]
    return models


def _language_codes(module, languages: list[str]) -> list[str]:
    """ISO 639-1 codes of a manifest's languages; Qwen3 lists names ("English"), and "Auto", which isn't one."""
    by_name = {name: code for code, name in getattr(module, "LANGUAGE_NAMES", {}).items() if len(code) == 2}
    return sorted(language if len(language) == 2 else by_name[language] for language in languages
                  if language != "Auto")


def _cloning_table() -> dict[str, dict[str, str]]:
    """docs/CLONING-MODELS.md's table of models: model id -> {column header: cell}."""
    lines = [line.strip() for line in CLONING_DOC.read_text(encoding="utf-8").splitlines()]
    at = next(number for number, line in enumerate(lines) if line.startswith("| Model |"))
    header = [cell.strip() for cell in lines[at].strip("|").split("|")]
    rows = {}
    for line in lines[at + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = dict(zip(header, cells))
    return rows


class TestCloningModelsAreDocumented:
    """docs/CLONING-MODELS.md, linked from the README, says
    what a cloning model is, how it gets a voice (a recording, or a named voice with a version for the model), what
    the voice pickers show for it, the languages each one takes, and how to start the workers. The facts are the
    workers' manifests (`providers/*/app.py`) and `docker-compose.voice-models.yml`."""

    def test_the_readme_and_the_docs_link_the_page(self):
        assert CLONING_DOC.is_file(), "docs/CLONING-MODELS.md is missing"
        for start, end in (("## Models", "## "), ("## Optional voice workers", "## ")):
            assert "(docs/CLONING-MODELS.md)" in _readme_part(start, end), f"README.md's {start} doesn't link it"
        for doc, start, end in ((WEB_DOC, "## Speech models, blends and presets", "## "),
                                (CONFIG_DOC, "### GPU voice workers", "### "),
                                (MODELS_DOC, "## Text-to-speech models", "## ")):
            assert "(CLONING-MODELS.md)" in _doc_part(doc, start, end), f"docs/{doc.name}'s {start} doesn't link it"

    def test_each_cloning_model_has_a_row_with_its_languages_and_what_it_needs(self, monkeypatch):
        table = _cloning_table()
        cloning = [(module, model) for module, model in _worker_models(monkeypatch)
                   if model["capabilities"].get("voice_clone")]
        assert sorted(model["id"] for _, model in cloning) == [
            "breeze/tts-2", "chatterbox/regular", "chatterbox/turbo", "fish/s2-pro", "qwen3/0.6b-base",
            "qwen3/1.7b-base"]
        wrong = []
        for module, model in cloning:
            row = table.get(model["id"])
            if row is None:
                wrong.append(f"{model['id']}: no row")
                continue
            capabilities = model["capabilities"]
            assert model["voices"] == [], model["id"]  # no built-in voices: what makes it a cloning model here
            languages = _language_codes(module, capabilities["languages"])
            if sorted(re.findall(r"`([a-z]{2})`", row["Languages"])) != languages:
                wrong.append(f"{model['id']}: languages {row['Languages']!r}, the manifest {languages}")
            if capabilities.get("clone_transcript_required", False) != row["Transcript"].startswith("Required"):
                wrong.append(f"{model['id']}: transcript {row['Transcript']!r}")
            if capabilities.get("live_reader", False) != row["Live Reader"].startswith("Yes"):
                wrong.append(f"{model['id']}: Live Reader {row['Live Reader']!r}")
            if f"{model['max_input_chars']:,}" not in row["Text per request"]:
                wrong.append(f"{model['id']}: text per request {row['Text per request']!r}")
        extra = sorted(set(table) - {model["id"] for _, model in cloning})
        assert not wrong and not extra, "docs/CLONING-MODELS.md's table: " + "; ".join(wrong + extra)

    def test_the_page_says_how_to_give_a_voice_what_the_pickers_show_and_how_to_start_the_workers(self):
        page = CLONING_DOC.read_text(encoding="utf-8")
        flat = _one_line(page.replace("\\\r\n", " ").replace("\\\n", " "))
        compose = (README.parent / "docker-compose.voice-models.yml").read_text(encoding="utf-8")
        command = _one_line(next(line.lstrip("# ") for line in compose.splitlines() if "docker compose -f" in line))
        assert command in flat, "the page doesn't give the voice-models Compose command"
        note = re.search(r"\$\{state\.noBuiltIn\} (has no built-in voices\.)", VOICE_PICKER.read_text(encoding="utf-8"))
        assert note and note.group(1) in flat, "the page doesn't quote the pickers' note"
        unsaid = [word for word in ("voice_library_ref", "clone_transcript", "voice:<", "/api/voices/library",
                                    "realizations", "Clone a voice", "QWEN3_ENABLE_BASE", "TTS_EXTERNAL_PROVIDERS",
                                    "Worker offline", "/v1/realtime")
                  if word not in flat]
        assert not unsaid, "docs/CLONING-MODELS.md doesn't say: " + ", ".join(unsaid)


DOCKERFILE = COMPOSE_FILE.parent / "Dockerfile"
_IMAGE_ENV = {"OS_HOST": "0.0.0.0", "OS_PORT": "8100"}  # set by the Dockerfile's runtime stage


def _healthcheck_code(command: str) -> str:
    match = re.search(r'python -c "(.*)" \|\| exit 1\s*$', command)
    assert match, command
    assert "$" not in match.group(1)  # the shell (and Compose) would expand it
    return match.group(1)


def _image_healthcheck() -> str:
    lines = DOCKERFILE.read_text(encoding="utf-8").splitlines()
    return _healthcheck_code(next(line for line in lines if line.startswith("HEALTHCHECK")))


def _compose_healthcheck() -> str:
    service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
    assert service["healthcheck"]["test"][0] == "CMD-SHELL"
    return _healthcheck_code(service["healthcheck"]["test"][1])


def _probed_url(code: str, container_env: dict[str, str]) -> str:
    """The URL the healthcheck opens inside a container with this environment."""
    opened = []
    with patch.dict(os.environ, container_env, clear=True), \
            patch("urllib.request.urlopen", lambda url, **_kwargs: opened.append(url)), \
            patch("ssl._create_unverified_context", lambda: None):
        exec(compile(code, "<healthcheck>", "exec"), {})
    assert len(opened) == 1
    return opened[0]


def _served_url(container_env: dict[str, str]) -> str:
    """The /health URL src.main serves in that container, legacy names included."""
    from src.config import Settings, _check_deprecated_env_vars

    with patch.dict(os.environ, container_env, clear=True):
        _check_deprecated_env_vars()
        configured = Settings()
    scheme = "https" if configured.os_ssl_enabled else "http"
    return f"{scheme}://localhost:{configured.os_port}/health"


# `docker run -e ...` on top of the image ENV. An empty value counts as unset, an old name applies
# when its new name is unset or empty, and a boolean takes pydantic's spellings (t, y, on, 1, ...).
_DOCKER_RUN_ENV = {
    "no SSL setting": {},
    "empty OS_SSL_ENABLED": {"OS_SSL_ENABLED": ""},
    **{f"OS_SSL_ENABLED={v}": {"OS_SSL_ENABLED": v} for v in ("true", "t", "T", "y", "Y", "on", "1")},
    **{f"OS_SSL_ENABLED={v}": {"OS_SSL_ENABLED": v} for v in ("false", "f", "N", "no", "off", "0")},
    "legacy STT_SSL_ENABLED=false": {"STT_SSL_ENABLED": "false"},
    "empty OS_SSL_ENABLED, legacy false": {"OS_SSL_ENABLED": "", "STT_SSL_ENABLED": "false"},
    "new name wins over legacy": {"OS_SSL_ENABLED": "true", "STT_SSL_ENABLED": "false"},
    "OS_PORT=9000": {"OS_PORT": "9000", "OS_SSL_ENABLED": "false"},
    "empty OS_PORT": {"OS_PORT": ""},
    "empty OS_PORT, legacy STT_PORT": {"OS_PORT": "", "STT_PORT": "9000"},
}


class TestHealthcheck:
    """The Dockerfile and Compose healthchecks must probe the scheme and port the app serves, or a
    healthy container is reported unhealthy and supervisors restart it in a loop."""

    @pytest.mark.parametrize("case", list(_DOCKER_RUN_ENV))
    def test_image_healthcheck_probes_what_the_app_serves(self, case):
        container_env = {**_IMAGE_ENV, **_DOCKER_RUN_ENV[case]}
        assert _probed_url(_image_healthcheck(), container_env) == _served_url(container_env)

    @pytest.mark.parametrize(
        "dotenv",
        [{}, {"OS_SSL_ENABLED": ""}, {"STT_SSL_ENABLED": "false"}, {"OS_SSL_ENABLED": "false"},
         *({"OS_SSL_ENABLED": v} for v in ("t", "T", "y", "Y", "on", "1", "f", "n", "0"))],
        ids=repr,
    )
    def test_compose_healthcheck_probes_what_the_app_serves(self, dotenv):
        _, compose_env = _settings_under_compose(dotenv)
        container_env = {**_IMAGE_ENV, **compose_env}  # environment: wins over the image ENV
        assert _probed_url(_compose_healthcheck(), container_env) == _served_url(container_env)

    def test_a_core_still_starting_is_not_reported_unhealthy(self):
        """The core's healthcheck had no start period, so Docker counted every
        failed probe from the first and reported a core whose lifespan was still running (a model preload, a
        cold host: about 2 minutes on a test machine) unhealthy after 90 s. Both healthchecks must carry the
        same start period, and a core whose /health first answers 120 s after start must not be reported
        unhealthy (Docker: probe failures in the start period don't count towards retries)."""

        def seconds(value: str) -> float:
            parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value)
            assert parts and "".join(n + u for n, u in parts) == value, value
            return sum(float(n) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[u] for n, u in parts)

        line = next(line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
                    if line.startswith("HEALTHCHECK"))
        image = dict(re.findall(r"--([a-z-]+)=(\S+)", line.split(" CMD ", 1)[0]))
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]["healthcheck"]
        settings = {
            "Dockerfile": {key: image.get(key) for key in ("interval", "timeout", "retries", "start-period")},
            "docker-compose.yml": {key.replace("_", "-"): None if compose.get(key) is None else str(compose[key])
                                   for key in ("interval", "timeout", "retries", "start_period")},
        }
        for where, setting in settings.items():
            assert setting["start-period"], f"{where}: the core's healthcheck has no start period"
            unhealthy_at = seconds(setting["start-period"]) + int(setting["retries"]) * seconds(setting["interval"])
            assert unhealthy_at > 120, f"{where}: a core still starting is reported unhealthy at {unhealthy_at} s"
        assert settings["Dockerfile"] == settings["docker-compose.yml"], settings


class TestWorkerImageTags:
    """A release raised the voice worker tags in docker-compose.voice-models.yml, but
    the CHANGELOG's upgrade note went on saying they are tagged `0.2.0`, so an upgrader pinned or looked for an image
    without that release's worker fixes. The note, docs/CONFIGURATION.md's worker settings table and `.env.example` must
    give each worker the tag Compose builds and runs (read from the Compose file, not written here)."""

    WORKERS = ("chatterbox", "qwen3", "fish", "breeze")
    # The workers whose tag the [0.9.0] upgrade note gives; a later worker or tag has its own release's entry.
    NOTED_WORKERS = ("chatterbox", "fish")

    def _compose_images(self) -> dict[str, str]:
        services = yaml.safe_load((COMPOSE_FILE.parent / "docker-compose.voice-models.yml").read_text(
            encoding="utf-8"))["services"]
        images = {}
        for worker in self.WORKERS:
            match = re.fullmatch(r"\$\{OPEN_SPEECH_([A-Z0-9]+)_IMAGE:-(open-speech-[a-z0-9]+:[^}]+)\}",
                                 services[worker]["image"])
            assert match and match.group(1) == worker.upper(), services[worker]["image"]
            images[worker] = match.group(2)
        return images

    @staticmethod
    def _named(text: str, worker: str) -> set[str]:
        return set(re.findall(rf"open-speech-{worker}:([0-9][0-9A-Za-z.\-]*[0-9A-Za-z])", text))

    def test_the_upgrade_note_gives_each_worker_the_tag_compose_runs(self):
        # The 0.9.0 release's notes, where the worker tags changed.
        notes = _changelog_release().split("### Upgrade notes", 1)[1].split("\n### ", 1)[0]
        note = " ".join(next(item for item in notes.split("\n- ") if "Voice worker images" in item).split())
        changelog = (README.parent / "CHANGELOG.md").read_text(encoding="utf-8")
        for worker, image in self._compose_images().items():
            if worker not in self.NOTED_WORKERS:
                assert f"`{image}`" in changelog, f"no CHANGELOG entry gives {image}"
                continue
            tag = image.split(":", 1)[1]
            assert self._named(note, worker) == {tag}, (
                f"the upgrade note gives open-speech-{worker} {sorted(self._named(note, worker))}; Compose runs {image}"
            )

    def test_the_configuration_doc_and_env_example_give_the_tags_compose_runs(self):
        docs = {"docs/CONFIGURATION.md": CONFIG_DOC.read_text(encoding="utf-8"),
                ".env.example": ENV_EXAMPLE.read_text(encoding="utf-8")}
        for worker, image in self._compose_images().items():
            tag = image.split(":", 1)[1]
            for name, text in docs.items():
                lines = [line for line in text.splitlines() if f"OPEN_SPEECH_{worker.upper()}_IMAGE" in line
                         and f"open-speech-{worker}:" in line]
                assert lines and all(self._named(line, worker) == {tag} for line in lines), (
                    f"{name} gives OPEN_SPEECH_{worker.upper()}_IMAGE as {lines}; Compose runs {image}"
                )


# ── Each GPU voice worker is an opt-in Compose profile ───────────────────────────────────────────────────────

VOICE_MODELS_FILE = COMPOSE_FILE.parent / "docker-compose.voice-models.yml"
GPU_WORKERS = ("chatterbox", "qwen3", "fish", "breeze")
# Docker's Compose profiles page (docs.docker.com/compose/how-tos/profiles/): "Valid profiles names follow the regex
# format of [a-zA-Z0-9][a-zA-Z0-9_.-]+".
_PROFILE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+")


def _enabled_services(services: dict, active: set[str]) -> set[str]:
    """The services `up` starts for the active profiles, by Compose's rule (the profiles page): "Services without a
    `profiles` attribute are always enabled", and the others only start when one of their profiles is active."""
    return {name for name, service in services.items()
            if not service.get("profiles") or active & set(service["profiles"])}


def _voice_models_commands(text: str) -> list[str]:
    """Each `docker compose` command line that layers the voice-models file and runs `up`, on one line (Markdown
    `\\` continuations joined, a Compose file's `#` comment marks dropped)."""
    joined = text.replace("\\\r\n", " ").replace("\\\n", " ")
    commands = []
    for line in joined.splitlines():
        line = line.strip().lstrip("#").strip().strip("`")
        at = line.find("docker compose")
        if at != -1 and "docker-compose.voice-models.yml" in line and re.search(r"\sup(\s|`|$)", line):
            commands.append(_one_line(line[at:]))
    return commands


class TestVoiceWorkersAreOptInProfiles:
    """The GPU workers are optional providers, off by default and easy to turn on one at a time. Each worker service in
    docker-compose.voice-models.yml has a Compose profile of its own, so layering the file starts (and builds) none,
    and `--profile qwen3` (or COMPOSE_PROFILES=qwen3) starts only that one. The docs say so, give the profile in
    every command that starts a worker, and say that only Qwen3 Custom Voice has built-in voices."""

    @staticmethod
    def _services() -> dict:
        return yaml.safe_load(VOICE_MODELS_FILE.read_text(encoding="utf-8"))["services"]

    def test_each_worker_has_a_profile_of_its_own_and_none_starts_without_one(self):
        services = self._services()
        assert sorted(name for name in services if name != "open-speech") == sorted(GPU_WORKERS), sorted(services)
        for worker in GPU_WORKERS:
            assert services[worker].get("profiles") == [worker], (
                f"{worker}: profiles {services[worker].get('profiles')!r}, not [{worker!r}] (it starts by default)"
            )
            assert _PROFILE_NAME.fullmatch(worker), worker
        assert "profiles" not in services["open-speech"], "the core must start without a profile"

    def test_no_profile_starts_only_the_core_and_one_profile_starts_one_worker(self):
        services = self._services()
        assert _enabled_services(services, set()) == {"open-speech"}
        for worker in GPU_WORKERS:
            assert _enabled_services(services, {worker}) == {"open-speech", worker}, worker
        assert _enabled_services(services, set(GPU_WORKERS)) == {"open-speech", *GPU_WORKERS}
        # A service that is always enabled can't depend on one whose profile is off (the profiles page): the core,
        # in any of the files it is layered from, depends on no worker.
        for name in ("docker-compose.yml", "docker-compose.gpu.yml", "docker-compose.voice-models.yml"):
            core = yaml.safe_load((COMPOSE_FILE.parent / name).read_text(encoding="utf-8"))["services"]["open-speech"]
            assert "depends_on" not in core and "links" not in core, f"{name}: the core depends on a worker"

    def test_the_core_still_knows_every_worker(self):
        """A worker that is turned on is found without editing anything; one that isn't shows as offline."""
        environment = self._services()["open-speech"]["environment"]
        raw = next(item.split("=", 1)[1] for item in environment if item.startswith("TTS_EXTERNAL_PROVIDERS="))
        import json
        import urllib.parse

        providers = json.loads(raw)
        assert sorted(providers) == sorted(GPU_WORKERS)
        # Each by its worker's own network alias (see TestOffWorkerNamesStayOnTheNetwork).
        assert all(urllib.parse.urlsplit(url).hostname == _worker_host(name) for name, url in providers.items()), \
            providers

    def test_every_command_that_starts_a_worker_names_its_profile(self):
        texts = {
            **_all_docs(),  # README.md and every docs/*.md
            "docker-compose.voice-models.yml": VOICE_MODELS_FILE.read_text(encoding="utf-8"),
            ".env.example": ENV_EXAMPLE.read_text(encoding="utf-8"),
            "CHANGELOG.md [0.9.0]": _changelog_release(),
        }
        found, unnamed = 0, []
        for name, text in texts.items():
            for command in _voice_models_commands(text):
                found += 1
                if not re.search(r"--profile\s+(\"\*\"|'\*'|[a-z0-9])|COMPOSE_PROFILES=", command):
                    unnamed.append(f"{name}: {command}")
        assert found, "no command layers docker-compose.voice-models.yml"
        assert not unnamed, "these start no worker now (they name no profile): " + " | ".join(unnamed)

    def test_the_docs_say_the_workers_are_off_by_default_and_how_to_turn_one_on(self):
        parts = {
            "README.md, Optional voice workers": _readme_part("## Optional voice workers", "## "),
            "docs/CONFIGURATION.md, GPU voice workers": _doc_part(CONFIG_DOC, "### GPU voice workers", "### "),
            "docs/API.md, voice cloning (POST /v1/audio/speech/clone)": _doc_part(
                API_DOC, "### `POST /v1/audio/speech/clone`", "### "),
            "docs/CLONING-MODELS.md": CLONING_DOC.read_text(encoding="utf-8"),
            "docs/TTS-BACKENDS.md": (DOCS / "TTS-BACKENDS.md").read_text(encoding="utf-8"),
            "docker-compose.voice-models.yml": VOICE_MODELS_FILE.read_text(encoding="utf-8"),
            ".env.example": ENV_EXAMPLE.read_text(encoding="utf-8"),
        }
        wrong = []
        for name, text in parts.items():
            flat = _one_line(re.sub(r"(?m)^[ \t]*#", "", text))  # a Compose or .env comment read as prose
            for needed in ("off by default", "--profile", "COMPOSE_PROFILES"):
                if needed not in flat:
                    wrong.append(f"{name}: no {needed!r}")
            if name.startswith("docs/API.md, voice"):
                continue  # the endpoint's paragraph points at GPU voice workers for the commands
            if not all(f"--profile {worker}" in flat or f"COMPOSE_PROFILES={worker}" in flat for worker in GPU_WORKERS):
                wrong.append(f"{name}: doesn't name each worker's profile")
            if not re.search(r"[Oo]nly Qwen3 Custom Voice( \([^)]*\))? has built-in voices", flat):
                wrong.append(f"{name}: doesn't say only Qwen3 Custom Voice has built-in voices")
        row = next((row[-1] for row in _table_rows() if row and row[0] == "`COMPOSE_PROFILES`"), "")
        if not all(f"`{worker}`" in row for worker in GPU_WORKERS):
            wrong.append(f"docs/CONFIGURATION.md's settings table: COMPOSE_PROFILES says {row!r}")
        assert not wrong, "; ".join(wrong)

    def test_the_upgrade_note_says_to_name_the_profiles(self):
        notes = _changelog_release().split("### Upgrade notes", 1)[1].split("\n### ", 1)[0]
        note = next((_one_line(item) for item in notes.split("\n- ") if "COMPOSE_PROFILES" in item), "")
        assert "docker-compose.voice-models.yml" in note and "--profile" in note, (
            "no upgrade note says the voice-models file now starts no worker unless its profile is named"
        )
        assert all(f"`{worker}`" in note for worker in TestWorkerImageTags.NOTED_WORKERS), note


# With a worker's profile off no container on the project network holds its name, and
# Docker's embedded DNS forwards a name it has no record of to the host's DNS servers (docs.docker.com/engine/network/,
# "DNS services"; moby libnetwork resolver.go forwards single-label names too unless ndots is set). A home router's
# DNS answers each DHCP client's hostname bare and under the LAN's domain (`chatterbox`, `chatterbox.lan`), so a device
# called `chatterbox` would be taken for the worker. The core names each worker by a network alias under a reserved
# private domain (ICANN keeps `.internal` out of the public DNS) with a trailing dot, so the container's resolver
# appends no search domain; moby trims that dot before it matches the alias.
WORKER_ALIAS_DOMAIN = "voice-workers.internal"
# The names a LAN's DNS gives a DHCP client called NAME, and the search domains a container inherits from the host.
_LAN_DOMAINS = ("lan", "home", "fritz.box", "local", "home.arpa", "internal", "localdomain")


def _worker_host(worker: str) -> str:
    return f"{worker}.{WORKER_ALIAS_DOMAIN}."


def _compose_worker_urls() -> dict[str, str]:
    import json

    environment = yaml.safe_load(VOICE_MODELS_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]["environment"]
    return json.loads(next(item.split("=", 1)[1] for item in environment if item.startswith("TTS_EXTERNAL_PROVIDERS=")))


class TestOffWorkerNamesStayOnTheNetwork:
    """docs/TTS-BACKENDS.md (Voice provider deployment): a worker whose profile is off reads "Worker offline" and its
    models can't be loaded, and the workers are reachable only on the private Compose network."""

    def test_each_worker_url_is_an_alias_only_that_worker_holds(self):
        import urllib.parse

        services = yaml.safe_load(VOICE_MODELS_FILE.read_text(encoding="utf-8"))["services"]
        wrong = []
        for worker, url in _compose_worker_urls().items():
            parts = urllib.parse.urlsplit(url)
            host = parts.hostname or ""
            if host != _worker_host(worker):
                wrong.append(f"{worker}: the core asks for {host!r}, not {_worker_host(worker)!r}")
            holders = sorted(name for name, service in services.items()
                             if host.rstrip(".") in ((service.get("networks") or {}).get("default") or {})
                             .get("aliases", []))
            if holders != [worker]:
                wrong.append(f"{worker}: the alias {host.rstrip('.')!r} is held by {holders}")
            if f"localhost:{parts.port}/health" not in str(services[worker]["healthcheck"]):
                wrong.append(f"{worker}: port {parts.port} is not the one the worker serves")
        assert not wrong, "; ".join(wrong)

    def test_a_profile_off_worker_reads_offline_when_a_lan_device_has_its_name(self, monkeypatch):
        """No worker container runs (its profile is off); a LAN device is called `chatterbox`, so the host's DNS answers
        `chatterbox` and `chatterbox.<domain>` (a stand-in `getaddrinfo`, no real lookup). Expected: the core reads the
        worker as offline and sends that device nothing, not even the clone's text and recording."""
        import json
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from src.tts.external import ExternalTTSBackend, parse_external_provider_urls

        received: list[str] = []

        class Device(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _answer(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                received.append(f"GET {self.path}")
                if self.path == "/health":
                    self._answer(json.dumps({"status": "ok", "provider": "chatterbox"}).encode())
                else:
                    self._answer(json.dumps({"schema_version": 1, "provider": "chatterbox", "models": [{
                        "id": "chatterbox/turbo", "provider": "chatterbox", "sample_rate": 24000,
                        "capabilities": {"voice_clone": True}, "voices": []}]}).encode())

            def do_POST(self):
                received.append(f"POST {self.path}")
                self._answer(b"{}")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Device)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        lan_names = {"chatterbox", *(f"chatterbox.{domain}" for domain in _LAN_DOMAINS)}
        real_getaddrinfo = socket.getaddrinfo

        def lan_getaddrinfo(host, port, *args, **kwargs):
            if host == "127.0.0.1":
                return real_getaddrinfo(host, port, *args, **kwargs)
            name = str(host)
            # The resolver tries a name without a trailing dot as given, then with each search domain appended.
            tried = [name.rstrip(".")] if name.endswith(".") else [name, *(f"{name}.{d}" for d in _LAN_DOMAINS)]
            if any(candidate in lan_names for candidate in tried):
                return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", server.server_port))]
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        monkeypatch.setattr(socket, "getaddrinfo", lan_getaddrinfo)
        try:
            url = parse_external_provider_urls(json.dumps(_compose_worker_urls()))["chatterbox"]
            backend = ExternalTTSBackend(provider="chatterbox", base_url=url, metadata_timeout_s=3.0,
                                         synthesis_timeout_s=10.0)
            available = backend.is_provider_available()
            try:
                list(backend.synthesize("A private sentence.", "clone", model_id="chatterbox/turbo",
                                        reference_audio=b"RIFF-the-users-own-voice", clone_transcript="mine"))
            except Exception:  # the expected refusal: the worker is offline
                pass
        finally:
            server.shutdown()
            server.server_close()
        assert (available, received) == (False, []), (
            f"the core asked for {url}: the LAN device answered it (is_provider_available={available}) and got {received}"
        )


def _code_spans(text: str) -> list[str]:
    """Each fenced code block and inline code span of a Markdown text, on one line (`\\` continuations joined)."""
    fenced = re.findall(r"```[a-z]*\r?\n(.*?)```", text, re.DOTALL)
    inline = re.findall(r"`([^`]+)`", re.sub(r"```[a-z]*\r?\n.*?```", "", text, flags=re.DOTALL))
    return [_one_line(span.replace("\\\r\n", " ").replace("\\\n", " ")) for span in fenced + inline]


# A command that stops or removes worker containers: `... -f docker-compose.voice-models.yml --profile "*" down`, or
# `... stop chatterbox` / `... rm -s -f fish` naming the service.
_STOP_WORKERS = re.compile(
    r"docker-compose\.voice-models\.yml.*?(--profile\s+[\"']\*[\"']\s+(down|stop)\b"
    rf"|\b(stop|rm(\s+-[sf])*)\s+({'|'.join(GPU_WORKERS)})\b)"
)


class TestWorkersStartedBeforeAreStopped:
    """The docs of the worker profiles promise that a worker that isn't turned on
    reads "Worker offline". Compose counts a service whose profile is off as part of the project, not as an orphan
    (docker/compose pkg/compose/containers.go `isOrphaned`), so `up` leaves the worker containers an earlier `up`
    started running on their old images, and a `down` without their profile doesn't stop them (Docker's profiles
    page). A `--profile` flag replaces COMPOSE_PROFILES rather than adding to it (compose-go `WithDefaultProfiles`).
    The docs that make the promise say so and give the command that stops them."""

    @staticmethod
    def _parts() -> dict[str, str]:
        notes = _changelog_release().split("### Upgrade notes", 1)[1].split("\n### ", 1)[0]
        return {
            "CHANGELOG.md, the profiles upgrade note": next(
                (item for item in notes.split("\n- ") if "COMPOSE_PROFILES" in item), ""),
            "docs/CONFIGURATION.md, GPU voice workers": _doc_part(CONFIG_DOC, "### GPU voice workers", "### "),
            "docs/CLONING-MODELS.md": CLONING_DOC.read_text(encoding="utf-8"),
            "docs/TTS-BACKENDS.md": (DOCS / "TTS-BACKENDS.md").read_text(encoding="utf-8"),
        }

    def test_the_docs_say_workers_started_before_keep_running_and_how_to_stop_them(self):
        wrong = []
        for name, text in self._parts().items():
            flat = _one_line(text)
            assert "Worker offline" in flat and "--profile" in flat, name
            if not re.search(r"keeps? running", flat):
                wrong.append(f"{name}: doesn't say a worker an earlier `up` started keeps running until stopped")
            if not any(_STOP_WORKERS.search(span) for span in _code_spans(text)):
                wrong.append(f"{name}: gives no command that stops or removes the worker containers")
            if not name.startswith("CHANGELOG") and not re.search(r"--profile`?( flag)? replaces `?COMPOSE_PROFILES",
                                                                   flat):
                wrong.append(f"{name}: doesn't say a --profile flag replaces COMPOSE_PROFILES")
        assert not wrong, "; ".join(wrong)


# Compose interpolates every .env value that isn't single-quoted ($NAME, ${NAME}, `$$`
# a literal `$`) and takes a single-quoted one as written (Docker's "Interpolation" page, ".env file syntax");
# `docker run --env-file` does neither. Seen on Docker Compose v5.5.1.
_BARE_VAR = re.compile(r"\$\$|\$([A-Za-z_][A-Za-z0-9_]*)")


def _compose_dotenv_value(raw: str, env: dict[str, str]) -> str:
    """A `.env` value as Docker Compose reads it."""
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    return _interpolate(_BARE_VAR.sub(lambda m: m.group(1) and "${%s}" % m.group(1) or "$$", raw), env)


class TestDollarInDotenvUnderCompose:
    """.env.example says to set a value by uncommenting its line and not to quote it, and that Compose reads .env. Under
    Compose `OS_API_KEY=$ecretKey123` reaches the server empty (auth off) and `Xk9$mP2q` as `Xk9`. The docs that say
    not to quote say how to write a `$` for Compose, and that `docker run --env-file` keeps the escape as written."""

    def test_compose_cuts_a_dollar_value_and_the_two_escapes_keep_it(self):
        # What `docker compose config` gave for these lines (Docker Compose v5.5.1).
        observed = {"$ecretKey123": "", "Xk9$mP2q": "Xk9", "hf_${abc}def": "hf_def",
                    "$$ecretKey123": "$ecretKey123", "'Xk9$mP2q'": "Xk9$mP2q", "hf_$${abc}def": "hf_${abc}def"}
        assert {raw: _compose_dotenv_value(raw, {}) for raw in observed} == observed

    @staticmethod
    def _parts() -> dict[str, str]:
        return {
            ".env.example, the header": ENV_EXAMPLE.read_text(encoding="utf-8").split("\n\n", 1)[0],
            "docs/CONFIGURATION.md, Writing values in .env": _doc_part(CONFIG_DOC, "## Writing values in .env",
                                                                         "## "),
        }

    def test_the_docs_say_how_to_write_a_dollar_for_compose_and_that_docker_run_keeps_it(self):
        wrong = []
        for name, text in self._parts().items():
            flat = _one_line(re.sub(r"(?m)^[ \t]*#", "", text))
            assert "not to quote" in flat or "do not quote" in flat or "quotes included" in flat, name
            if "$$" not in flat or not re.search(r"single-quot", flat):
                wrong.append(f"{name}: doesn't say to write a `$` as `$$`, or single-quote the value, under Compose")
            if not re.search(r"docker run --env-file`?.{0,80}(keeps|expands nothing).{0,80}\$\$", flat):
                wrong.append(f"{name}: doesn't say `docker run --env-file` keeps a `$$` (and quotes) as written")
            examples = re.findall(r"[A-Z][A-Z0-9_]*=('[^'\s]*\$[^'\s]*'|[^\s`)'\"]*\$\$[^\s`)'\"]*)", flat)
            if not examples:
                wrong.append(f"{name}: shows no NAME=value example with the escape")
            for value in examples:
                if "$" not in _compose_dotenv_value(value, {}):
                    wrong.append(f"{name}: the example {value!r} reaches the server without its `$` under Compose")
        assert not wrong, "; ".join(wrong)

    def test_every_doc_that_sets_a_key_in_dotenv_says_how_to_write_a_dollar(self):
        texts = {
            "docs/SECURITY.md, API key": _doc_part(SECURITY_DOC, "## API key\n", "## "),
            "docs/AI-ASSISTANT.md": ASSISTANT_DOC.read_text(encoding="utf-8"),
            **{f".env.example, {name}": _example_comment(name)
               for name in ("OS_API_KEY", "OS_ASSISTANT_API_KEY", "HF_TOKEN")},
        }
        missing = [name for name, text in texts.items() if "$$" not in text]
        assert not missing, "these set a key or token and don't say a `$` in it is written `$$` under Compose: " + \
            ", ".join(missing)


# docs/CONFIGURATION.md GPU launch, "plain `docker compose up -d` is not a GPU launch"; Compose's
# `up` reference: a service whose configuration changed since its container was created is stopped and recreated. A
# step that restarts with a bare `docker compose up` recreates a GPU core from docker-compose.yml alone (the `:cpu`
# image, no GPU, no voice workers).
_BARE_UP = re.compile(r"(?<![\w-])docker compose up\b")
_GPU_UP = re.compile(r"docker compose -f docker-compose\.yml -f docker-compose\.gpu\.yml\b[^`]*?\bup\b")


# The sections that are the launches themselves: README's Compose quick start, and docs/CONFIGURATION.md's CPU
# launch and its GPU launch, which says plain `up -d` isn't one.
_LAUNCH_SECTIONS = {("README.md", "### Docker Compose"), ("docs/CONFIGURATION.md", "### CPU launch"),
                    ("docs/CONFIGURATION.md", "### GPU launch")}


def _doc_sections() -> list[tuple[str, str, str]]:
    """(doc, heading, section) for each `##` to `####` section of README.md and docs/*.md."""
    found = []
    for name, text in _all_docs().items():
        for section in re.split(r"(?m)^(?=#{2,4} )", text):
            heading = section.split("\n", 1)[0].strip()
            if (name, heading) not in _LAUNCH_SECTIONS:
                found.append((name, heading, section))
    return found


class TestRestartsKeepTheLaunchFiles:
    def test_the_launch_sections_are_there(self):
        headings = {(name, section.split("\n", 1)[0].strip()) for name, text in _all_docs().items()
                    for section in re.split(r"(?m)^(?=#{2,4} )", text)}
        assert _LAUNCH_SECTIONS <= headings, sorted(_LAUNCH_SECTIONS - headings)

    def test_every_section_that_restarts_with_a_bare_up_also_gives_the_gpu_files(self):
        wrong = []
        for name, heading, section in _doc_sections():
            spans = _code_spans(section)
            if any(_BARE_UP.search(span) for span in spans) and not any(_GPU_UP.search(span) for span in spans):
                wrong.append(f"{name}: {heading}")
        assert not wrong, (
            "these sections restart with a bare `docker compose up`, which loads only docker-compose.yml and recreates "
            "a GPU core as the CPU image without the workers, and don't give the GPU launch's -f files: "
            + ", ".join(wrong))

    def test_every_section_that_restarts_with_a_bare_up_also_gives_the_cpu_file(self):
        """docker-compose.cpu.yml is the same project, service and container as
        docker-compose.yml (docs/CONFIGURATION.md CPU launch: it "extends the base file"), so a bare
        `docker compose up -d` recreates a cpu.yml server from the base file: Wyoming on, on 0.0.0.0, with no
        authentication (docs/API.md Wyoming), the Turbo STT model and a two-model cap. A section that gives the bare
        restart also gives the cpu.yml launch's own and says what the bare one does to it."""
        wrong = []
        for name, heading, section in _doc_sections():
            spans = _code_spans(section)
            if not any(_BARE_UP.search(span) for span in spans):
                continue
            if not any(_CPU_FILE_UP.search(span) for span in spans):
                wrong.append(f"{name}: {heading}: gives no `docker compose -f docker-compose.cpu.yml up -d`")
            elif "0.0.0.0" not in section:
                wrong.append(f"{name}: {heading}: doesn't say a bare `up -d` opens Wyoming on 0.0.0.0 for a cpu.yml "
                             "server")
        assert not wrong, "; ".join(wrong)


_CPU_FILE_UP = re.compile(r"docker compose -f docker-compose\.cpu\.yml\b[^`]*?\bup\b")


# A source install loads .env through `uv run --env-file`, which reads it with dotenvy
# 0.15.7 (astral-sh/uv 0.9.28 Cargo.lock; crates/uv/src/commands/project/run.rs calls dotenvy::from_path per file and
# only warns on a LineParse error, and dotenvy's Iter::load stops at the first bad line). A port of dotenvy's
# dotenv/src/parse.rs `parse_value` (tag v0.15.7), for a value with no variables set: a `$` outside single quotes
# starts a substitution and `$` after `$` starts another (so `$$` is no escape), a `"` is dropped, an unquoted blank
# followed by anything but `#` and a backslash other than `\\ \' \" \$ \n` and space are errors, and single quotes
# keep everything as written. Checked against dotenvy's own tests in test_uv_reads_a_value_as_dotenvy_does.
class _UvLineParse(ValueError):
    pass


def _uv_dotenv_value(value: str) -> str:
    strong = weak = escaped = expecting_end = False
    out: list[str] = []
    mode, name = "none", []  # dotenvy's SubstitutionMode: none, block ($NAME) or escaped_block (${NAME})
    for c in value:
        if expecting_end:
            if c in " \t":
                continue
            if c == "#":
                break
            raise _UvLineParse(value)
        if escaped:
            if c in "\\'\"$ ":
                out.append(c)
            elif c == "n":
                out.append("\n")
            else:
                raise _UvLineParse(value)
            escaped = False
        elif strong:
            if c == "'":
                strong = False
            else:
                out.append(c)
        elif mode != "none":
            if c.isalnum():
                name.append(c)
            elif mode == "block":
                if c == "{" and not name:
                    mode = "escaped_block"
                else:
                    name.clear()  # an unset NAME becomes ""
                    if c == "$":
                        mode = "block"
                    else:
                        mode = "none"
                        out.append(c)
            elif c == "}":
                mode = "none"
                name.clear()
            else:
                name.append(c)
        elif c == "$":
            mode = "block"
        elif weak:
            if c == '"':
                weak = False
            elif c == "\\":
                escaped = True
            else:
                out.append(c)
        elif c == "'":
            strong = True
        elif c == '"':
            weak = True
        elif c == "\\":
            escaped = True
        elif c in " \t":
            expecting_end = True
        else:
            out.append(c)
    if mode == "escaped_block" or strong or weak:
        raise _UvLineParse(value)
    return "".join(out)


def _uv_dotenv_load(text: str) -> dict[str, str]:
    """The NAME=value lines `uv run --env-file` loads from a .env of single-line values: it stops at the first bad
    line, so that setting and every later one stay unset."""
    loaded = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, raw = line.partition("=")
        raw = raw.strip()
        try:
            loaded[name.strip()] = "" if not raw or raw.startswith("#") else _uv_dotenv_value(raw)
        except _UvLineParse:
            break
    return loaded


_QUOTED_EXAMPLE = re.compile(r"\b([A-Z][A-Z0-9_]*)=('[^']*')")


class TestDotenvUnderUvRun:
    """.env.example says to copy it to .env and set a value by uncommenting its line, "do not quote values", and that a
    source install needs `uv run --env-file .env`; docs/DEVELOPMENT.md's source install starts the server that way. A
    .env written as the docs say must reach a source install as written: the docs say how uv reads a value, and that single quotes
    keep a `$`, a `"`, a blank and a backslash under both uv and Compose."""

    def test_uv_reads_a_value_as_dotenvy_does(self):
        def read(raw):
            try:
                return _uv_dotenv_value(raw)
            except _UvLineParse:
                return "LineParse"

        # dotenvy's own tests (parse.rs test_parse_line_env, test_parse_value_escapes, the invalid-line tests).
        assert [read(v) for v in ('"2"', "'fo ur'", '"fi ve"', "s\\ ix", "my\\ cool\\ value", "\\$sweet",
                                  '"lol" #well you see', "my uncool value", "h\\8u", "'>${KEY1}<>$KEY2<'")] == [
            "2", "fo ur", "fi ve", "s ix", "my cool value", "$sweet", "lol", "LineParse", "LineParse",
            ">${KEY1}<>$KEY2<"]
        # What a source install gets (from the full port of dotenvy).
        assert {v: read(v) for v in ("$ecretKey123", "Xk9$$mP2q", "'Xk9$mP2q'", '{"top_k": 20}', '{"top_k":20}',
                                     "C:\\Users\\me\\dict.yaml", "'C:\\Users\\me\\dict.yaml'", "'{\"top_k\": 20}'")} \
            == {"$ecretKey123": "", "Xk9$$mP2q": "Xk9", "'Xk9$mP2q'": "Xk9$mP2q", '{"top_k": 20}': "LineParse",
                '{"top_k":20}': "{top_k:20}", "C:\\Users\\me\\dict.yaml": "LineParse",
                "'C:\\Users\\me\\dict.yaml'": "C:\\Users\\me\\dict.yaml", "'{\"top_k\": 20}'": '{"top_k": 20}'}

    @staticmethod
    def _parts() -> dict[str, str]:
        listener = (DOCS / "LOCAL-LISTENER.md").read_text(encoding="utf-8")
        return {
            ".env.example, the header": ENV_EXAMPLE.read_text(encoding="utf-8").split("\n\n", 1)[0],
            "docs/DEVELOPMENT.md, Install from source": _doc_part(DEV_DOC, "## Install from source", "## "),
            "docs/CONFIGURATION.md, Writing values in .env": _doc_part(CONFIG_DOC, "## Writing values in .env", "## "),
            "docs/SECURITY.md, API key": _doc_part(SECURITY_DOC, "## API key\n", "## "),
            "docs/AI-ASSISTANT.md": ASSISTANT_DOC.read_text(encoding="utf-8"),
            "docs/LOCAL-LISTENER.md, Start the harness": listener[listener.index("## Start the harness"):
                                                                  listener.index("## Prewarm")],
        }

    def test_the_docs_say_how_uv_run_reads_a_value_and_to_single_quote_it(self):
        wrong = []
        for name, text in self._parts().items():
            flat = _one_line(re.sub(r"(?m)^[ \t]*#", "", text))
            if "uv run --env-file" not in flat:
                wrong.append(f"{name}: doesn't name `uv run --env-file`")
            if not _QUOTED_EXAMPLE.search(flat):
                wrong.append(f"{name}: gives no single-quoted NAME='value' example")
            if "$$" in flat and not re.search(r"\$\$.{0,160}?(no escape|Compose only|Compose alone)"
                                              r"|(Compose alone|Compose only).{0,80}?\$\$", flat):
                wrong.append(f"{name}: gives `$$` without saying uv doesn't take it")
            if name.startswith((".env.example", "docs/DEVELOPMENT.md", "docs/CONFIGURATION.md")):
                for needed in ("double quote", "backslash", "blank", "single-quote"):
                    if needed not in flat:
                        wrong.append(f"{name}: doesn't say what uv does with a {needed}")
        assert not wrong, "; ".join(wrong)

    def test_every_single_quoted_example_reaches_the_server_as_written(self):
        examples = {}
        for name, text in self._parts().items():
            for setting, raw in _QUOTED_EXAMPLE.findall(_one_line(re.sub(r"(?m)^[ \t]*#", "", text))):
                examples[f"{name}: {setting}={raw}"] = raw
        examples["docs/CONFIGURATION.md, OS_ASSISTANT_EXTRA_BODY row"] = next(
            iter(re.findall(r"OS_ASSISTANT_EXTRA_BODY=('[^']*')", _setting_row("OS_ASSISTANT_EXTRA_BODY"))), "")
        wrong = []
        for where, raw in examples.items():
            meant = raw[1:-1]
            try:
                uv = _uv_dotenv_value(raw)
            except _UvLineParse:
                uv = "(uv stops reading .env here)"
            compose = _compose_dotenv_value(raw, {})
            if not raw or (uv, compose) != (meant, meant):
                wrong.append(f"{where}: uv reads {uv!r}, Compose {compose!r}")
        assert not wrong, "; ".join(wrong)

    def test_the_extra_body_example_copied_into_env_example_loads_every_line(self):
        """.env.example copied to .env, OS_ASSISTANT_EXTRA_BODY set to its own comment's example, and the two later
        local paths docs/DEVELOPMENT.md's source install needs set."""
        import json

        comment = _example_comment("OS_ASSISTANT_EXTRA_BODY")
        shown = re.search(r"OS_ASSISTANT_EXTRA_BODY=('[^']*'|\S+(?: \S+)*?\})", comment) or re.search(
            r"such as (\{[^}]*\})", comment)
        assert shown, comment
        text = ENV_EXAMPLE.read_text(encoding="utf-8").replace("\r\n", "\n")
        edits = {
            "# OS_ASSISTANT_EXTRA_BODY=\n": f"OS_ASSISTANT_EXTRA_BODY={shown.group(1)}\n",
            "# TTS_CACHE_DIR=/var/lib/open-speech/cache\n": "TTS_CACHE_DIR=data/cache\n",
            "# READER_CACHE_DIR=/home/openspeech/data/reader\n": "READER_CACHE_DIR=data/reader\n",
        }
        for old, new in edits.items():
            assert text.count(old) == 1, old
            text = text.replace(old, new)
        loaded = _uv_dotenv_load(text)
        assert {key: loaded.get(key) for key in ("TTS_CACHE_DIR", "READER_CACHE_DIR")} == {
            "TTS_CACHE_DIR": "data/cache", "READER_CACHE_DIR": "data/reader"}, (
            f"uv stops reading .env at OS_ASSISTANT_EXTRA_BODY={shown.group(1)}: it and every later line are unset")
        assert json.loads(loaded["OS_ASSISTANT_EXTRA_BODY"]) == {"top_k": 20}, loaded["OS_ASSISTANT_EXTRA_BODY"]


class TestPiperVoiceCounts:
    def test_the_arctic_row_says_it_has_18_speakers(self):
        """rhasspy/piper-voices en_US-arctic-medium.onnx.json has num_speakers 18,
        and PiperBackend.list_voices lists one `piper/en_US-arctic-medium/<speaker>` voice per speaker_id_map entry
        (tests/test_piper_backend.py covers the listing), so docs/MODELS.md's TTS table can't call it one voice."""
        rows = {row[0]: row for row in _table_rows(MODELS_DOC) if len(row) == 4}
        note = rows["`piper/en_US-arctic-medium`"][-1]
        assert "one voice" not in note and "18 speakers" in note, note
        assert "/v1/audio/voices?model=piper/en_US-arctic-medium" in note, note


_NUMBER_WORDS = {"no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10}


def _test_cases(function) -> int:
    """How many tests a test function collects as: the length of its parametrize list, when the list is a literal or a
    `NAME[:N]` slice, else one."""
    import ast

    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func) == "pytest.mark.parametrize":
            values = decorator.args[1]
            if isinstance(values, (ast.List, ast.Tuple)):
                return len(values.elts)
            if isinstance(values, ast.Subscript) and isinstance(values.slice, ast.Slice) and \
                    values.slice.lower is None and isinstance(values.slice.upper, ast.Constant):
                return values.slice.upper.value
            raise AssertionError(f"can't count the cases of {function.name}")
    return 1


def _tests_that(path: Path, uses) -> int:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(_test_cases(node) for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
               and uses(node))


class TestDevelopmentSkipList:
    def test_the_tests_section_counts_the_tests_that_may_skip(self):
        """docs/DEVELOPMENT.md's Tests section says which tests skip and why;
        its counts are the suite's. Torch: the Kokoro tests marked `needs_torch`, and the other tests that
        `pytest.importorskip("torch")`. Symbolic links: the tests that use conftest's `create_symlink` fixture or skip
        themselves when the host can't make one."""
        import ast

        tests = COMPOSE_FILE.parent / "tests"
        kokoro_torch = _tests_that(tests / "test_tts_kokoro.py", lambda node: any(
            ast.unparse(d) == "needs_torch" for d in node.decorator_list))
        other_torch = sum(_tests_that(path, lambda node: any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "pytest.importorskip" and call.args
            and isinstance(call.args[0], ast.Constant) and call.args[0].value == "torch" for call in ast.walk(node)))
            for path in sorted(tests.glob("test_*.py")))
        symlinks = sum(_tests_that(path, lambda node: any(a.arg == "create_symlink" for a in node.args.args) or any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "pytest.skip" and call.args
            and isinstance(call.args[0], ast.Constant) and re.search(r"symbolic.link", str(call.args[0].value))
            for call in ast.walk(node)))
            for path in sorted(tests.glob("test_*.py")))
        section = _one_line(_doc_part(DEV_DOC, "## Tests", "## "))
        torch = re.search(r"(\w+) Kokoro unit tests?(?: and (\w+) other tests?)? .{0,80}?require the optional "
                          r"Torch runtime", section)
        symlink = re.search(r"skips the (\w+) symlink tests", section)
        assert torch and symlink, section
        said = (_NUMBER_WORDS.get(torch[1].lower()), _NUMBER_WORDS.get((torch[2] or "no").lower()),
                _NUMBER_WORDS.get(symlink[1].lower()))
        assert said == (kokoro_torch, other_torch, symlinks), (
            f"docs/DEVELOPMENT.md says {torch[0]!r} and {symlink[0]!r}; the suite has {kokoro_torch} Kokoro tests "
            f"that need Torch, {other_torch} other, and {symlinks} symlink tests")


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
        with patch.dict(os.environ, env, clear=True):
            from src.config import _check_deprecated_env_vars
            warnings = _check_deprecated_env_vars()
            assert "OS_PORT" not in warnings


class TestRemovedSettings:
    """OS_HISTORY_RETAIN_AUDIO and OS_HISTORY_MAX_MB governed audio files history never stores."""

    @pytest.mark.parametrize("name", ["OS_HISTORY_RETAIN_AUDIO", "OS_HISTORY_MAX_MB"])
    def test_removed_history_setting_is_reported_as_ignored(self, name, caplog):
        from src.config import Settings, _check_obsolete_env_vars, log_obsolete_warnings
        with patch.dict(os.environ, {name: "1", "OS_HISTORY_MAX_ENTRIES": "50"}, clear=True):
            obsolete = _check_obsolete_env_vars()
            configured = Settings()
        assert obsolete == [name]
        assert not hasattr(configured, name.lower())
        assert configured.os_history_max_entries == 50
        with caplog.at_level(logging.WARNING):
            log_obsolete_warnings(obsolete)
        assert f"Env var '{name}' has no effect" in caplog.text

    def test_empty_removed_setting_counts_as_unset(self):
        # Compose passes an unset variable through as "".
        from src.config import _check_obsolete_env_vars
        with patch.dict(os.environ, {"OS_HISTORY_RETAIN_AUDIO": "", "OS_HISTORY_MAX_MB": ""}, clear=True):
            assert _check_obsolete_env_vars() == []

    def test_startup_warns_once_that_a_removed_setting_is_ignored(self):
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("OS_HISTORY_")}
        env.update({"OS_HISTORY_MAX_MB": "500", "PYTHONIOENCODING": "utf-8"})
        started = subprocess.run(
            [sys.executable, "-c", "import src.config"],  # only the settings module, not the app
            cwd=COMPOSE_FILE.parent, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert started.returncode == 0, started.stderr
        assert started.stderr.count("Env var 'OS_HISTORY_MAX_MB' has no effect") == 1
        assert "OS_HISTORY_RETAIN_AUDIO" not in started.stderr


class TestLogLevelSetting:
    """OS_LOG_LEVEL is DEBUG, INFO, WARNING or ERROR, in any case, and INFO by default, so
    nothing changes unless it is set. A bad value stops startup with a message that names the setting but never
    quotes the value. .env.example shows it commented out with its default; docker-compose.yml passes it through empty,
    like the other OS_* settings. tests/test_api.py checks what the level does at startup."""

    @pytest.mark.parametrize(
        ("value", "level"),
        [(None, "INFO"), ("", "INFO"), ("debug", "DEBUG"), ("Info", "INFO"), (" WARNING ", "WARNING"),
         ("error", "ERROR")],
        ids=repr,
    )
    def test_the_level_parses_in_any_case_and_defaults_to_info(self, value, level):
        from src.config import Settings

        with patch.dict(os.environ, {} if value is None else {"OS_LOG_LEVEL": value}, clear=True):
            assert Settings().os_log_level == level

    @pytest.mark.parametrize("value", ["verbose", "trace", "critical", "warn", "0"])
    def test_a_bad_level_is_refused_naming_the_setting(self, value):
        from pydantic import ValidationError

        from src.config import Settings

        with patch.dict(os.environ, {"OS_LOG_LEVEL": value}, clear=True):
            with pytest.raises(ValidationError) as refused:
                Settings()
        assert "OS_LOG_LEVEL must be one of DEBUG, INFO, WARNING or ERROR" in str(refused.value)

    def test_a_bad_level_stops_startup_without_printing_the_value(self):
        secret_looking = "sk-live-4f9b2c7e"  # a value pasted on the wrong line must not end up in the log
        env = {key: value for key, value in os.environ.items() if key.upper() != "OS_LOG_LEVEL"}
        env.update({"OS_LOG_LEVEL": secret_looking, "PYTHONIOENCODING": "utf-8"})
        started = subprocess.run(
            [sys.executable, "-c", "import src.config"],  # only the settings module, not the app
            cwd=COMPOSE_FILE.parent, env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
        assert started.returncode != 0
        assert "OS_LOG_LEVEL must be one of DEBUG, INFO, WARNING or ERROR" in started.stderr
        assert secret_looking not in started.stderr

    def test_env_example_shows_the_level_commented_out_with_its_default(self):
        from src.config import Settings

        shown = [(number, _shown_value(line)) for number, name, line in _example_setting_lines()
                 if name == "OS_LOG_LEVEL"]
        assert len(shown) == 1, f"expected one '# OS_LOG_LEVEL=...' line in .env.example, found {shown}"
        assert shown[0][1] == "INFO"
        assert "OS_LOG_LEVEL" not in _docker_cli_env_file(ENV_EXAMPLE.read_text(encoding="utf-8"))
        with patch.dict(os.environ, {}, clear=True):
            assert Settings().os_log_level == "INFO"

    def test_compose_passes_the_level_through_empty(self):
        service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
        assert "OS_LOG_LEVEL=${OS_LOG_LEVEL:-}" in [str(entry) for entry in service["environment"]]
        configured, container_env = _settings_under_compose({})
        assert (configured.os_log_level, container_env["OS_LOG_LEVEL"]) == ("INFO", "")
        configured, _ = _settings_under_compose({"OS_LOG_LEVEL": "debug"})
        assert configured.os_log_level == "DEBUG"


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

    def test_bare_metal_stt_defaults_are_cpu_safe(self):
        from src.config import Settings
        with patch.dict(os.environ, {}, clear=True):
            s = Settings()
            assert s.stt_model == "Systran/faster-whisper-base"
            assert s.stt_device == "cpu"
            assert s.stt_compute_type == "int8"


class TestTranslationModelSetting:
    """STT_TRANSLATE_MODEL is empty by default, so nothing is
    downloaded until it is set; .env.example shows it commented out and empty, docker-compose.yml passes it through
    empty, docs/CONFIGURATION.md's settings table lists it, and the translation docs say which models translate."""

    def test_the_setting_is_empty_by_default_and_read_from_the_environment(self):
        from src.config import Settings

        with patch.dict(os.environ, {}, clear=True):
            assert Settings().stt_translate_model == ""
        with patch.dict(os.environ, {"STT_TRANSLATE_MODEL": "Systran/faster-whisper-medium"}, clear=True):
            assert Settings().stt_translate_model == "Systran/faster-whisper-medium"

    def test_env_example_shows_it_commented_out_and_empty(self):
        lines = [(number, _shown_value(line)) for number, name, line in _example_setting_lines()
                 if name == "STT_TRANSLATE_MODEL"]
        assert len(lines) == 1, lines
        assert lines[0][1] == "", lines

    def test_compose_passes_it_through_empty(self):
        service = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))["services"]["open-speech"]
        assert "STT_TRANSLATE_MODEL=${STT_TRANSLATE_MODEL:-}" in service["environment"]
        configured, _ = _settings_under_compose({})
        assert configured.stt_translate_model == ""
        configured, _ = _settings_under_compose({"STT_TRANSLATE_MODEL": "Systran/faster-whisper-medium"})
        assert configured.stt_translate_model == "Systran/faster-whisper-medium"

    def test_the_api_reference_and_the_agents_guide_say_which_models_translate(self):
        rows = [row for row in _table_rows() if row and "`STT_TRANSLATE_MODEL`" in row[0]]
        assert len(rows) == 1, rows
        assert rows[0][1] == '`""`', rows
        api = _doc_part(API_DOC, "### `POST /v1/audio/translations`", "### ")
        agents = (DOCS / "AI-AGENTS.md").read_text(encoding="utf-8")
        for name, text in (("docs/API.md, POST /v1/audio/translations", api), ("docs/AI-AGENTS.md", agents)):
            for needle in ("translation_unsupported", "STT_TRANSLATE_MODEL", "Systran/faster-whisper-medium"):
                assert needle in text, f"{name} does not mention {needle}"


# ── Docs and config ──────────────────────────────────────────────────────────────────────────────────────────────────


def _doc_section(doc: Path, heading: str) -> str:
    """A doc's `## heading` section, up to the next `## ` heading."""
    text = doc.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## )", text, re.M | re.S)
    assert match, heading
    return match.group(1)


def _code_lines(text: str, starts: str) -> list[str]:
    """The lines of the fenced code blocks in ``text`` that start with ``starts``, continuation lines joined."""
    lines = []
    for block in re.findall(r"```\w*\n(.*?)```", text, re.S):
        for line in re.sub(r"[\\`][ \t]*\n", " ", block).splitlines():
            if line.strip().startswith(starts):
                lines.append(" ".join(line.split()))
    return lines


class TestApostropheInDotenvUnderUvRun:
    """`uv run --env-file` (dotenvy 0.15.7, see TestDotenvUnderUvRun above) takes an
    unquoted `'` as the start of a single-quoted part (parse.rs `parse_value`: `} else if c == '\\'' { strong_quote =
    true;`), and iter.rs's `QuotedLines` carries it over the next lines, so one apostrophe in a key is a `LineParse`
    error: uv stops reading .env there, and the key and `OS_AUTH_REQUIRED=true` below it stay unset (auth off on
    0.0.0.0). Two are dropped. Single quotes can't hold a `'`; dotenvy's `\\'` escape does. Every place the docs list
    the characters to take care with must name the apostrophe and say how to write it."""

    def test_uv_mishandles_these_six_characters_in_an_unquoted_value(self):
        import string

        bad = []
        for c in sorted(set(string.punctuation) | {" ", "\t"}):
            value = f"Xk9{c}mP2q"
            if _uv_dotenv_load(f"OS_API_KEY={value}\nOS_AUTH_REQUIRED=true\n") != {
                    "OS_API_KEY": value, "OS_AUTH_REQUIRED": "true"}:
                bad.append(c)
        # The same six as the full dotenvy port.
        assert bad == ["\t", " ", '"', "$", "'", "\\"], bad

    def test_an_apostrophe_key_loses_the_auth_guard_and_the_escape_keeps_it(self):
        import string

        block = "OS_API_KEY={}\n\n# Stop at startup when OS_API_KEY is empty\nOS_AUTH_REQUIRED=true\n"
        assert _uv_dotenv_load(block.format("Xk9'mP2q")) == {}
        assert _uv_dotenv_load(block.format("it's-Bob's"))["OS_API_KEY"] == "its-Bobs"
        assert _uv_dotenv_load(block.format("Xk9\\'mP2q")) == {"OS_API_KEY": "Xk9'mP2q", "OS_AUTH_REQUIRED": "true"}
        assert _compose_dotenv_value("Xk9'mP2q", {}) == "Xk9'mP2q"  # Compose: a quote only as the first character
        # secrets.token_urlsafe's alphabet reads the same under uv, Compose and `docker run --env-file`.
        safe = string.ascii_letters + string.digits + "-_"
        assert _uv_dotenv_value(safe) == _compose_dotenv_value(safe, {}) == _docker_cli_env_file(f"K={safe}")["K"]

    def test_every_list_of_the_characters_names_the_apostrophe_and_how_to_write_it(self):
        wrong = []
        listed = set()
        for name, text in TestDotenvUnderUvRun._parts().items():
            flat = _one_line(re.sub(r"(?m)^[ \t]*#", "", text))
            if "backslash" not in flat and not name.endswith("API key"):
                continue  # this part lists no characters
            listed.add(name)
            if not re.search(r"(?i)apostrophe", flat):
                wrong.append(f"{name}: lists what uv mishandles but not the apostrophe")
            if "\\'" not in flat:
                wrong.append(f"{name}: doesn't say to write an apostrophe as \\' for uv")
            for setting, raw in re.findall(r"\b([A-Z][A-Z0-9_]*)=([^\s`'\"]*\\'[^\s`]*?)[`),.;]*(?:\s|$)", flat):
                if "'" not in _uv_dotenv_value(raw) or "\\" in _uv_dotenv_value(raw):
                    wrong.append(f"{name}: {setting}={raw} reaches a source install as {_uv_dotenv_value(raw)!r}")
        # Every place that tells you how to write a value in .env lists them; docs/AI-ASSISTANT.md only sets a key
        # and points at them.
        assert listed == {".env.example, the header", "docs/DEVELOPMENT.md, Install from source",
                          "docs/CONFIGURATION.md, Writing values in .env", "docs/SECURITY.md, API key",
                          "docs/LOCAL-LISTENER.md, Start the harness"}, listed
        for name, text in (("docs/SECURITY.md, API key", _doc_part(SECURITY_DOC, "## API key\n", "## ")),
                           (".env.example, the header", ENV_EXAMPLE.read_text(encoding="utf-8").split("\n\n", 1)[0])):
            if "secrets.token_urlsafe" not in text:
                wrong.append(f"{name}: doesn't offer a key with none of these characters")
        assert not wrong, "; ".join(wrong)


class TestSourceInstallSystemPrograms:
    """The server runs the `ffmpeg` program for every mp3, opus, aac and flac answer
    (src/tts/pipeline.py; mp3 is the OpenAI default, the web app's Speak and previews, every Reader part) and `openssl`
    for a generated certificate (src/ssl_utils.py); `uv sync` installs neither, only the image does. Without ffmpeg
    each of those answers `500 ffmpeg not found`, so docs/DEVELOPMENT.md's source install must say both are needed."""

    def test_the_source_install_says_ffmpeg_and_openssl_must_be_on_path(self):
        root = README.parent
        assert '"ffmpeg", "-y"' in (root / "src" / "tts" / "pipeline.py").read_text(encoding="utf-8")
        assert '"openssl", "req"' in (root / "src" / "ssl_utils.py").read_text(encoding="utf-8")
        flat = _one_line(_doc_section(DEV_DOC, "Install from source"))
        wrong = []
        if not re.search(r"`ffmpeg`[^.]*`PATH`", flat):
            wrong.append("doesn't say the ffmpeg program must be on PATH")
        for command in ("apt install ffmpeg", "brew install ffmpeg", "winget install"):
            if command not in flat:
                wrong.append(f"doesn't give `{command}`")
        if not re.search(r"(?i)mp3", flat) or "ffmpeg not found" not in flat:
            wrong.append("doesn't say what fails without ffmpeg")
        if not re.search(r"`openssl`[^.]*`PATH`", flat):
            wrong.append("offers a generated certificate without saying it needs the openssl program on PATH")
        assert not wrong, "docs/DEVELOPMENT.md Install from source: " + "; ".join(wrong)


class TestQuickStartKeepsTheLibrary:
    """The image declares no volume for /home/openspeech/data (Dockerfile VOLUME),
    where every voice, recording, Reader document, history entry and AI setting is written, and its model cache is an
    anonymous volume a new container doesn't reuse. A container can't change its image, so upgrading replaces it: a
    `docker run` in the docs that is meant to stay (not `--rm`) must mount both paths, and the README's Quick start
    must say how to upgrade."""

    @staticmethod
    def _targets(argv: list[str]) -> dict[str, str]:
        """Container path -> source of each `-v`/`--volume` mount."""
        mounts = {}
        for flag, value in zip(argv, argv[1:]):
            if flag in ("-v", "--volume"):
                source, target = value.split(":")[:2]
                mounts[target] = source
        return mounts

    def test_every_lasting_docker_run_mounts_the_data_and_the_model_cache(self):
        dockerfile = (README.parent / "Dockerfile").read_text(encoding="utf-8")
        assert "/home/openspeech/data" not in re.search(r"^VOLUME .*$", dockerfile, re.M).group(0)
        quick = _doc_section(README, "Quick start")
        runs = _code_lines(quick, "docker run")
        assert len(runs) >= 2, runs
        wrong = []
        for run in [run for text in _all_docs().values() for run in _code_lines(text.replace("\r\n", "\n"),
                                                                                 "docker run")]:
            argv = run.split()
            if "--rm" in argv:
                continue  # a throw-away run, such as the canary
            mounts = self._targets(argv)
            for path in ("/home/openspeech/data", "/home/openspeech/.cache/huggingface"):
                if path not in mounts:
                    wrong.append(f"{run}: nothing mounted at {path}")
        flat = _one_line(quick)
        if "docker pull" not in flat or "docker rm" not in flat:
            wrong.append("Quick Start doesn't say how to upgrade (pull, remove, run again)")
        if "/home/openspeech/data" not in flat:
            wrong.append("Quick Start doesn't say what is lost without the volumes")
        assert not wrong, "; ".join(wrong)


class TestEntrypointDataOwner:
    """Every Compose launch binds the checkout's ./data at /home/openspeech/data,
    and a Linux bind mount keeps the host's owner, so a checkout made with sudo (or by an account other than uid 1000)
    left the server user unable to create studio.db and the container restart-looped. The entrypoint, which runs as
    root, gives the folder to `openspeech` when that user can't write it, and only then (so a working library isn't
    walked on every start). Run with stub `su`, `chown`, `mkdir` and `stat` on PATH, so nothing real changes."""

    @staticmethod
    def _run(tmp_path: Path, writable: bool) -> list[str]:
        import shutil

        sh = shutil.which("sh")
        if sh is None:
            pytest.skip("no POSIX sh on PATH")
        stubs = tmp_path / "bin"
        stubs.mkdir()
        log = tmp_path / "calls.log"
        log.write_bytes(b"")
        for name in ("su", "chown", "mkdir", "stat"):
            body = f'#!/bin/sh\necho "{name} $*" >> "$STUB_LOG"\n'
            if name == "su":
                body += 'case "$*" in *"test -w"*) exit "$STUB_WRITE_RC";; esac\n'
            (stubs / name).write_bytes((body + "exit 0\n").encode())
            (stubs / name).chmod(0o755)
        script = tmp_path / "entrypoint.sh"
        # The image's build strips CRs (Dockerfile: sed -i 's/\r$//').
        script.write_bytes((README.parent / "docker-entrypoint.sh").read_text(encoding="utf-8")
                           .replace("\r\n", "\n").encode())
        env = {"PATH": str(stubs), "STUB_LOG": str(log), "STUB_WRITE_RC": "0" if writable else "1"}
        for keep in ("SYSTEMROOT", "TEMP", "TMP"):  # Windows' sh needs these
            if keep in os.environ:
                env[keep] = os.environ[keep]
        subprocess.run([sh, str(script)], env=env, capture_output=True, timeout=60, check=True)
        return log.read_text(encoding="utf-8").splitlines()

    def test_a_data_folder_the_server_user_cant_write_is_given_to_it_before_the_server_starts(self, tmp_path):
        calls = self._run(tmp_path, writable=False)
        start = next(i for i, call in enumerate(calls) if call.startswith("su -p "))
        fixed = [i for i, call in enumerate(calls) if call == "chown -R openspeech:openspeech /home/openspeech/data"]
        assert fixed and fixed[0] < start, calls

    def test_a_writable_data_folder_is_left_alone(self, tmp_path):
        calls = self._run(tmp_path, writable=True)
        assert any("test -w /home/openspeech/data" in call for call in calls), calls
        assert not [call for call in calls if call.startswith("chown") and call.endswith(" /home/openspeech/data")]
        assert calls[-1].startswith("su -p "), calls

    def test_the_volumes_section_says_who_writes_the_data_folder(self):
        flat = _one_line(_doc_part(CONFIG_DOC, "### Volumes", "## "))
        assert "uid 1000" in flat and "chown" in flat, flat


class TestSourceInstallOnCuda:
    """docs/DEVELOPMENT.md's source install offers `--extra cuda`, but STT_DEVICE defaults to cpu
    and TTS_DEVICE falls back to it, so the CUDA install ran Whisper and every TTS model on the CPU. The section must
    give the three device settings, as the GPU Compose override and the canary run do."""

    def test_the_cuda_install_gives_the_device_settings(self):
        from src.config import Settings

        section = _doc_section(DEV_DOC, "Install from source")
        assert "--extra cuda" in section
        lines = {"STT_DEVICE": "cuda", "STT_COMPUTE_TYPE": "float16", "TTS_DEVICE": "cuda"}
        missing = [f"{k}={v}" for k, v in lines.items() if not re.search(rf"^{k}={v}$", section, re.M)]
        assert not missing, f"docs/DEVELOPMENT.md Install from source doesn't give {missing} for a CUDA install"
        with patch.dict(os.environ, {}, clear=True):
            assert (Settings().stt_device, Settings().tts_effective_device) == ("cpu", "cpu")
        with patch.dict(os.environ, lines, clear=True):
            configured = Settings()
            assert (configured.stt_device, configured.stt_compute_type, configured.tts_effective_device) == (
                "cuda", "float16", "cuda")


class TestDevelopmentSyncKeepsTheProviders:
    """`uv sync` is exact by default (uv 0.9.28 `uv help sync`: "uv removes
    packages that are not declared as dependencies of the project"), so a Development sync without the Installation's
    `--extra all` uninstalled Kokoro, Piper and Pocket-TTS. Every later `uv sync`/`uv run` with extras keeps
    the Installation's extras, or syncs `--inexact`."""

    def test_development_commands_keep_the_installation_extras(self):
        install = _code_lines(_doc_section(DEV_DOC, "Install from source"), "uv sync")[0]
        wanted = set(re.findall(r"--extra[ =](\S+)", install)) - {"cpu", "cuda"}
        assert wanted == {"all"}, install
        # The test commands of docs/DEVELOPMENT.md and the README's Development section.
        commands = []
        for section in (_doc_section(DEV_DOC, "Tests"), _doc_section(README, "Development")):
            commands += _code_lines(section, "uv sync") + [c for c in _code_lines(section, "uv run") if "--extra" in c]
        assert len(commands) >= 4, commands
        bad = [c for c in commands if "--inexact" not in c and not wanted <= set(re.findall(r"--extra[ =](\S+)", c))]
        assert not bad, f"The development commands run {bad}: uv's exact sync uninstalls the TTS providers"

    def test_guides_and_makefile_keep_the_installation_extras(self):
        """A fix to the installation extras reached only the source install; docs/AI-ASSISTANT.md,
        docs/LOCAL-LISTENER.md and docs/AI-AGENTS.md still gave `uv sync --frozen --extra cpu --extra dev --extra
        examples`, and the Makefile `--extra cpu --extra dev`, which uninstall Kokoro, Piper and Pocket-TTS from a
        source install. Every `uv sync` in docs/*.md and the Makefile keeps `all` (or tts, piper and pocket), or
        syncs `--inexact`."""
        root = README.parent
        commands = []
        for path in sorted((root / "docs").glob("*.md")) + [root / "Makefile"]:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                for command in re.findall(r"uv sync(?:[ \t]+--?[\w=-]+(?:[ \t]+(?!-)[\w.-]+)?)*", line):
                    commands.append((f"{path.relative_to(root).as_posix()}:{number}", command.strip()))
        assert len(commands) >= 4, commands
        bad = [f"{where}: {command}" for where, command in commands
               if "--inexact" not in command
               and not ({"all"} <= set(re.findall(r"--extra[ =](\S+)", command))
                        or {"tts", "piper", "pocket"} <= set(re.findall(r"--extra[ =](\S+)", command)))]
        assert not bad, f"these guides run {bad}: uv's exact sync uninstalls Kokoro, Piper and Pocket-TTS"


class TestCudaSourceInstallHasOneOnnxRuntime:
    """faster-whisper (a base dependency) and Piper require the CPU `onnxruntime`, and the cuda extra adds
    `onnxruntime-gpu`; both install the same `onnxruntime` package, so `uv sync --extra cuda` without
    `--no-install-package onnxruntime` installed both and one overwrote the other's files (`uv sync --frozen
    --dry-run --extra cuda --extra all` lists both). No extra can drop the CPU build while faster-whisper requires it,
    so the source install's CUDA command leaves it out, as the Dockerfile does, and later `uv run`s don't sync it back."""

    def test_the_source_install_cuda_sync_leaves_out_the_cpu_build(self):
        section = _doc_section(DEV_DOC, "Install from source")
        cuda = [c for c in _code_lines(section, "uv sync") if "--extra cuda" in c]
        assert cuda == ["uv sync --frozen --extra cuda --extra all --no-install-package onnxruntime"], cuda
        assert "uv run --no-sync" in section and "UV_NO_SYNC=1" in section
        assert any(c.startswith("# A CUDA install: uv run --no-sync") for c in _code_lines(section, "# A CUDA"))

    def test_every_cuda_install_instruction_names_it(self):
        root = README.parent
        bad = []
        for path in [README, *sorted((root / "docs").glob("*.md"))]:
            text = " ".join(path.read_text(encoding="utf-8").split())
            for match in re.finditer(r"--extra cuda", text):
                around = text[match.start():match.start() + 200]
                if text[max(0, match.start() - 2):match.start()] == "(`" and around.startswith("--extra cuda`)"):
                    continue  # "For a CUDA install (`--extra cuda`), add these lines": not an install command
                if "--no-install-package onnxruntime" not in around:
                    bad.append(f"{path.relative_to(root).as_posix()}: ...{around[:120]}...")
        assert not bad, "a CUDA install instruction without --no-install-package onnxruntime:\n" + "\n".join(bad)

    def test_the_dockerfile_and_the_lock_still_need_it(self):
        dockerfile = (README.parent / "Dockerfile").read_text(encoding="utf-8")
        assert '["--no-install-package", "onnxruntime"]' in dockerfile
        lock = (README.parent / "uv.lock").read_text(encoding="utf-8")
        whisper = lock[lock.index('name = "faster-whisper"'):]
        whisper = whisper[:whisper.index("[[package]]")]
        assert '{ name = "onnxruntime" }' in whisper  # so the base install can't do without the CPU build


class TestParakeetFirstLoadDocs:
    """The docs said the first GPU load takes "about a minute and 4 GB of RAM" (measured: about 7 minutes, most of
    it the 2.5 GB download, and up to about 6 GB), gave the cache as 1.3 GB on a GPU (2.5 GB fp32 kept + 1.2 GB
    fp16), and advised STT_PRELOAD_MODELS, which keeps the server from answering until the load finishes."""

    def test_the_models_doc_gives_the_measured_time_memory_and_size(self):
        parakeet = " ".join(_doc_part(MODELS_DOC, "## Parakeet", "## ").split())
        assert "about a minute" not in parakeet and "4 GB of RAM" not in parakeet
        assert "about 7 minutes" in parakeet and "about 6 GB" in parakeet and "about 3.7 GB" in parakeet
        row = next(r for r in _table_rows(MODELS_DOC) if r and r[0] == "`nvidia/parakeet-tdt-0.6b-v2`")
        assert row[1] == "~3.7GB (GPU), ~0.7GB (CPU)"

    def test_preloading_is_documented_as_running_in_the_background(self):
        # 0.10.1 runs STT_PRELOAD_MODELS after the server has started (src/main.py); it delayed startup before.
        parakeet = " ".join(_doc_part(MODELS_DOC, "## Parakeet", "## ").split())
        assert "Download on it in Models" in parakeet
        assert "the preload runs in the background after the server starts" in parakeet
        assert "answers nothing" not in parakeet
        row = _setting_row("STT_PRELOAD_MODELS")
        assert "in the background once the server has started" in row and "delays startup" not in row
        comment = _example_comment("STT_PRELOAD_MODELS")
        assert "in the background once the server is up" in comment and "answers only" not in comment
        compose = " ".join((README.parent / "docker-compose.yml").read_text(encoding="utf-8").replace("#", "").split())
        assert "STT_PRELOAD_MODELS loads at startup, in the background, while the server answers" in compose
        changelog = (README.parent / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "to\n  do this while the server starts" not in changelog

    def test_the_residency_section_and_the_model_api_say_what_0_10_1_changed(self):
        residency = " ".join(_doc_part(MODELS_DOC, "## Model residency", "## ").split())
        assert "from Transcribe and from Models › Load" in residency
        assert "row's `pinned`" in residency
        assert "`POST /api/ps/{model}` whose load fails loads back the STT models it unloaded" in residency
        assert "`language_unsupported`" in residency and "transcribed in English" in residency
        assert "`STT_PRELOAD_MODELS` loads its models in the background" in residency
        api = " ".join(_doc_part(API_DOC, "## Model management", "## Studio").split())
        assert "has `pinned`: `true` for `STT_MODEL` and `TTS_MODEL`" in api
        assert "| `409` | `model_fetching` |" in api

    def test_the_stream_caps_match_the_backend(self):
        from src.backends.parakeet import STREAM_MAX_UTTERANCE_S

        assert STREAM_MAX_UTTERANCE_S == 24
        stream = " ".join(_doc_part(API_DOC, "### `WS /v1/audio/stream`", "## ").split())
        parakeet = " ".join(_doc_part(MODELS_DOC, "## Parakeet", "## ").split())
        assert "finalized after 30 seconds at most (about 24 s for Parakeet)" in stream
        assert "streaming STT ends an utterance at about 24 s" in parakeet
        assert "reports any other language the same way and continues in English" in stream

    def test_the_agents_guide_names_the_default_model_and_its_first_load(self):
        guide = " ".join((DOCS / "AI-AGENTS.md").read_text(encoding="utf-8").split())
        assert '"model": "Systran/faster-whisper-base"' not in guide
        assert "about 7 minutes the first time" in guide and "timeout" in guide
        assert 'Add `-F "language=en"` to skip language detection' not in guide
        assert "about 24 s for Parakeet" in guide


def test_the_version_is_the_same_everywhere():
    import json
    import tomllib

    root = README.parent
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert version == "0.16.4"
    assert f'__version__ = "{version}"' in (root / "src" / "main.py").read_text(encoding="utf-8")
    assert json.loads((root / "client-js" / "package.json").read_text(encoding="utf-8"))["version"] == version
    assert f'name = "open-speech"\nversion = "{version}"' in (root / "uv.lock").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    assert f"badge/version-{version}-blue" in readme and f"Current release: **v{version}**" in readme
    assert f"## [{version}] - " in (root / "CHANGELOG.md").read_text(encoding="utf-8")


# ── Links between the docs ───────────────────────────────────────────────────────────────────────────────────────


def _heading_anchors(text: str) -> set[str]:
    """The anchors GitHub gives a Markdown file's headings: lower case, punctuation but `-` and `_` dropped, spaces as
    `-`, and `-1`, `-2` ... for a repeated heading. Fenced code is skipped (a `# comment` there is no heading)."""
    text = re.sub(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", "", text.replace("\r\n", "\n"), flags=re.S | re.M)
    anchors, seen = set(), {}
    for heading in re.findall(r"^#{1,6} +(.+?) *#*$", text, re.M):
        anchor = re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


def test_every_relative_link_in_the_docs_reaches_its_file_and_heading():
    """The README's detail lives in docs/*.md, and the docs link each other's sections. Every relative link in
    README.md, docs/*.md, CHANGELOG.md and the two other READMEs names a file that exists and, after `#`, a heading
    of that file."""
    import posixpath

    root = README.parent
    files = [README, *sorted(DOCS.glob("*.md")), root / "CHANGELOG.md", root / "client-js" / "README.md",
             root / "src" / "static" / "README.md"]
    checked, broken = 0, []
    for path in files:
        text = path.read_text(encoding="utf-8")
        prose = re.sub(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", "", text, flags=re.S | re.M)
        here = path.relative_to(root).as_posix()
        for target in re.findall(r"\]\(([^)\s]+)\)", prose):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue  # https:, mailto:
            checked += 1
            file_part, _, anchor = target.partition("#")
            dest = posixpath.normpath(posixpath.join(posixpath.dirname(here), file_part)) if file_part else here
            if not (root / dest).exists():
                broken.append(f"{here}: {target} (no file {dest})")
            elif anchor and dest.endswith(".md") and anchor not in _heading_anchors(
                    (root / dest).read_text(encoding="utf-8")):
                broken.append(f"{here}: {target} (no heading #{anchor} in {dest})")
    assert checked > 150, checked
    assert not broken, "broken links:\n" + "\n".join(broken)

