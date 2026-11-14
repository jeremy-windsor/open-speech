"""The AI assistant core: settings, client, prompts, presets, conversation, speech and /api/assistant/*.

The rules are in README.md, "AI assistant": "Settings, presets and the three layers", "Prompts" and "Thinking
controls", and the OS_ASSISTANT_* settings.

No network: every call to the AI server goes to an httpx.MockTransport. Nothing here can hang: the slow handlers
sleep at most a second against a deadline of a fraction of one, and every await is bounded by the client's own
deadline (pyproject's faulthandler_timeout bounds the run).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.config import Settings

AI_SERVER = "http://ai-server.local:8000/v1"


@pytest.fixture
def clean_assistant_env(monkeypatch):
    """No OS_ASSISTANT_* variable from the shell reaches a Settings() built in a test."""
    import os

    for name in list(os.environ):
        if name.upper().startswith("OS_ASSISTANT_"):
            monkeypatch.delenv(name)


# ── Settings (.env) ──────────────────────────────────────────────────────────


class TestSettings:
    def test_the_assistant_is_off_by_default_with_the_plans_defaults(self, clean_assistant_env):
        s = Settings(_env_file=None)
        assert s.assistant_enabled is False
        assert {
            "url": s.os_assistant_url,
            "api_key": s.os_assistant_api_key,
            "extra_body": s.os_assistant_extra_body,
            "verify_tls": s.os_assistant_verify_tls,
            "max_tokens_field": s.os_assistant_max_tokens_field,
            "model": s.os_assistant_model,
            "system_prompt": s.os_assistant_system_prompt,
            "thinking": s.os_assistant_thinking,
            "reasoning_effort": s.os_assistant_reasoning_effort,
            "thinking_budget": s.os_assistant_thinking_budget,
            "max_tokens": s.os_assistant_max_tokens,
            "temperature": s.os_assistant_temperature,
            "top_p": s.os_assistant_top_p,
            "timeout_s": s.os_assistant_timeout_s,
            "job_timeout_s": s.os_assistant_job_timeout_s,
            "history_turns": s.os_assistant_history_turns,
            "chunk_chars": s.os_assistant_chunk_chars,
            "show_thinking": s.os_assistant_show_thinking,
            "speak_errors": s.os_assistant_speak_errors,
        } == {
            "url": "",
            "api_key": "",
            "extra_body": "",
            "verify_tls": True,
            "max_tokens_field": "max_tokens",
            "model": "",
            "system_prompt": "",
            "thinking": "default",
            "reasoning_effort": "",
            "thinking_budget": 0,
            "max_tokens": 0,
            "temperature": None,
            "top_p": None,
            "timeout_s": 120,
            "job_timeout_s": 600,
            "history_turns": 20,
            "chunk_chars": 12000,
            "show_thinking": True,
            "speak_errors": True,
        }
        assert s.assistant_extra_body == {}
        assert s.os_assistant_template_kwargs is False

    def test_a_url_turns_it_on(self, clean_assistant_env, monkeypatch):
        monkeypatch.setenv("OS_ASSISTANT_URL", AI_SERVER)
        monkeypatch.setenv("OS_ASSISTANT_REASONING_EFFORT", "low")
        s = Settings(_env_file=None)
        assert s.assistant_enabled is True
        assert s.os_assistant_reasoning_effort == "low"
        assert Settings(_env_file=None, os_assistant_url="  ").assistant_enabled is False

    def test_the_extra_body_is_parsed_as_a_json_object(self, clean_assistant_env, monkeypatch):
        monkeypatch.setenv("OS_ASSISTANT_EXTRA_BODY", '{"think": false, "top_k": 20}')
        assert Settings(_env_file=None).assistant_extra_body == {"think": False, "top_k": 20}

    @pytest.mark.parametrize(
        "value",
        ["[1, 2]", "not json", '"text"', "42", "null", '{"stream": true}', '{"messages": []}'],
    )
    def test_an_extra_body_that_is_not_a_json_object_is_refused_at_startup(self, clean_assistant_env, monkeypatch, value):
        monkeypatch.setenv("OS_ASSISTANT_EXTRA_BODY", value)
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert "OS_ASSISTANT_EXTRA_BODY" in str(caught.value)

    def test_the_token_limit_field_name_is_one_of_two(self, clean_assistant_env, monkeypatch):
        monkeypatch.setenv("OS_ASSISTANT_MAX_TOKENS_FIELD", "max_completion_tokens")
        assert Settings(_env_file=None).os_assistant_max_tokens_field == "max_completion_tokens"
        monkeypatch.setenv("OS_ASSISTANT_MAX_TOKENS_FIELD", "max_new_tokens")
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert "os_assistant_max_tokens_field" in str(caught.value).lower()

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("OS_ASSISTANT_URL", "ai-server.local:8000/v1"),
            ("OS_ASSISTANT_URL", "ftp://ai-server.local/v1"),
            ("OS_ASSISTANT_THINKING", "maybe"),
            ("OS_ASSISTANT_REASONING_EFFORT", "extreme"),
            ("OS_ASSISTANT_MAX_TOKENS", "-1"),
            ("OS_ASSISTANT_MAX_TOKENS", "32769"),
            ("OS_ASSISTANT_TEMPERATURE", "2.5"),
            ("OS_ASSISTANT_TOP_P", "1.5"),
            ("OS_ASSISTANT_TIMEOUT_S", "4"),
            ("OS_ASSISTANT_TIMEOUT_S", "3601"),
            ("OS_ASSISTANT_JOB_TIMEOUT_S", "3601"),
            ("OS_ASSISTANT_HISTORY_TURNS", "101"),
            ("OS_ASSISTANT_CHUNK_CHARS", "999"),
            ("OS_ASSISTANT_CHUNK_CHARS", "200001"),
            ("OS_ASSISTANT_SYSTEM_PROMPT", "x" * 8001),
        ],
    )
    def test_enums_and_ranges_are_refused_at_startup(self, clean_assistant_env, monkeypatch, name, value):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert name.lower() in str(caught.value).lower()

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("OS_ASSISTANT_URL", "https://ai-server.local/v1"),
            ("OS_ASSISTANT_THINKING", "off"),
            ("OS_ASSISTANT_REASONING_EFFORT", "minimal"),
            ("OS_ASSISTANT_MAX_TOKENS", "32768"),
            ("OS_ASSISTANT_TEMPERATURE", "0"),
            ("OS_ASSISTANT_TOP_P", "1"),
            ("OS_ASSISTANT_TIMEOUT_S", "5"),
            ("OS_ASSISTANT_JOB_TIMEOUT_S", "3600"),
            ("OS_ASSISTANT_HISTORY_TURNS", "0"),
            ("OS_ASSISTANT_CHUNK_CHARS", "1000"),
        ],
    )
    def test_the_bounds_themselves_are_accepted(self, clean_assistant_env, monkeypatch, name, value):
        monkeypatch.setenv(name, value)
        Settings(_env_file=None)


# ── Errors and their sentences ───────────────────────────────────────────────


def test_every_error_code_has_the_plans_default_sentence():
    from src.assistant.errors import ERROR_SENTENCES, AssistantError

    assert ERROR_SENTENCES == {
        "assistant_unreachable": "I couldn't reach the AI server.",
        "assistant_timeout": "The AI took too long to answer.",
        "assistant_http_error": "The AI server returned an error.",
        "assistant_bad_response": "The AI didn't send back an answer.",
        "assistant_empty_reply": "The AI didn't send back an answer.",
        "assistant_reply_cut_off": "The answer was cut off before it started.",
    }
    error = AssistantError("The AI server answered HTTP 500", "assistant_http_error")
    assert (error.message, error.code, str(error)) == (
        "The AI server answered HTTP 500", "assistant_http_error", "The AI server answered HTTP 500",
    )


def test_a_presets_sentence_replaces_the_default_and_a_code_left_out_keeps_it():
    from src.assistant.presets import EffectiveSettings

    settings = EffectiveSettings(error_sentences={"assistant_timeout": "Ada is slow today."})
    assert settings.error_sentence("assistant_timeout") == "Ada is slow today."
    assert settings.error_sentence("assistant_unreachable") == "I couldn't reach the AI server."
    assert settings.error_sentence("something_new") == "The AI server returned an error."


# ── The client (POST {OS_ASSISTANT_URL}/chat/completions, stream: false) ─────

API_KEY = "sk-ai-server-secret-123"
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def _answer(content="Hello there.", finish_reason="stop", status=200, **message):
    return httpx.Response(
        status,
        json={"choices": [{"index": 0, "message": {"role": "assistant", "content": content, **message},
                           "finish_reason": finish_reason}]},
    )


def _fail(error_type, text):
    """A handler whose transport fails with an httpx error, as a refused connection or a TLS failure does."""

    def handler(request):
        raise error_type(text, request=request)

    return handler


class _Server:
    """A MockTransport handler that records each request and answers from a function."""

    def __init__(self, answer=None):
        self.requests: list[httpx.Request] = []
        self.answer = answer or (lambda _request: _answer())

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        return self.answer(request)

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)


class _RecordingStream(httpx.AsyncByteStream):
    """A response body that counts the chunks read from it (the guards read a body in a
    stream, stop at the cap, and never read an error's body)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.chunks_read = 0

    async def __aiter__(self):
        for chunk in self._chunks:
            self.chunks_read += 1
            yield chunk


def _client(server, **kwargs):
    from src.assistant.client import AssistantClient

    return AssistantClient(AI_SERVER, transport=httpx.MockTransport(server), **kwargs)


def _settings(**changes):
    from src.assistant.presets import EffectiveSettings

    return replace(EffectiveSettings(model="qwen3-30b"), **changes)


async def _reply(server, settings=None, messages=MESSAGES, **client_kwargs):
    client = _client(server, **client_kwargs)
    try:
        return await client.reply(messages, settings or _settings())
    finally:
        await client.aclose()


class TestAssistantClient:
    async def test_a_good_reply_posts_the_messages_without_streaming_and_nothing_unset(self):
        server = _Server()
        reply = await _reply(server)
        request = server.requests[0]
        assert (request.method, str(request.url)) == ("POST", f"{AI_SERVER}/chat/completions")
        assert server.body == {"model": "qwen3-30b", "messages": MESSAGES, "stream": False}
        assert "authorization" not in request.headers
        assert (reply.text, reply.thinking, reply.finish_reason) == ("Hello there.", None, "stop")

    async def test_an_empty_model_is_not_sent(self):
        server = _Server()
        await _reply(server, _settings(model=""))
        assert server.body == {"messages": MESSAGES, "stream": False}

    async def test_content_given_as_a_list_of_parts_is_joined(self):
        parts = [
            {"type": "text", "text": "Hello "}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "there."},
        ]
        reply = await _reply(_Server(lambda _r: _answer(content=parts)))
        assert reply.text == "Hello there."

    @pytest.mark.parametrize(
        ("message", "thinking"),
        [
            ({"content": "The answer.", "reasoning_content": "Let me see."}, "Let me see."),
            ({"content": "The answer.", "reasoning": "Let me see."}, "Let me see."),
            ({"content": "<think>\nLet me see.\n</think>\n\nThe answer."}, "Let me see."),
            # A chat template that opens <think> in the prompt: the reply starts inside the thinking.
            ({"content": "Let me see.\n</think>\n\nThe answer."}, "Let me see."),
        ],
        ids=["reasoning_content", "reasoning", "think-tags", "closing-tag-only"],
    )
    async def test_the_thinking_is_taken_from_each_place_and_kept_out_of_the_answer(self, message, thinking):
        reply = await _reply(_Server(lambda _r: _answer(**message)))
        assert (reply.text, reply.thinking) == ("The answer.", thinking)

    @pytest.mark.parametrize(
        ("changes", "client_kwargs", "sent"),
        [
            ({"thinking": "on"}, {}, {"chat_template_kwargs": {"enable_thinking": True}}),
            ({"thinking": "off"}, {}, {"chat_template_kwargs": {"enable_thinking": False}}),
            ({"thinking": "default"}, {}, {}),
            ({"reasoning_effort": "low"}, {}, {"reasoning_effort": "low"}),
            ({"reasoning_effort": ""}, {}, {}),
            ({"max_tokens": 256}, {}, {"max_tokens": 256}),
            ({"max_tokens": 256}, {"max_tokens_field": "max_completion_tokens"}, {"max_completion_tokens": 256}),
            ({"max_tokens": 0}, {"max_tokens_field": "max_completion_tokens"}, {}),
            ({"temperature": 0.7, "top_p": 0.9}, {}, {"temperature": 0.7, "top_p": 0.9}),
            ({"temperature": 0.0}, {}, {"temperature": 0.0}),
        ],
        ids=["thinking-on", "thinking-off", "thinking-default", "effort-low", "effort-unset", "max_tokens",
             "max_completion_tokens", "no-limit", "sampling", "temperature-zero"],
    )
    async def test_each_control_sends_exactly_its_field_and_nothing_when_unset(self, changes, client_kwargs, sent):
        server = _Server()
        await _reply(server, _settings(**changes), **client_kwargs)
        assert server.body == {"model": "qwen3-30b", "messages": MESSAGES, "stream": False, **sent}

    async def test_the_extra_body_is_merged_into_every_request(self):
        server = _Server()
        await _reply(server, _settings(reasoning_effort="low"), extra_body={"think": False, "top_k": 20})
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_effort": "low",
            "think": False, "top_k": 20,
        }

    # Some servers' chat templates read enable_thinking and reasoning_effort as template variables
    # (chat_template_kwargs), and think at its heaviest effort when it gets
    # none, so a top-level reasoning_effort alone may not reach it.
    @pytest.mark.parametrize(
        ("changes", "sent"),
        [
            ({"reasoning_effort": "low"},
             {"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "low"}}),
            ({"thinking": "off", "reasoning_effort": "low"},
             {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}),
            ({"thinking": "on"}, {"chat_template_kwargs": {"enable_thinking": True}}),
            ({}, {}),
        ],
        ids=["effort", "off-and-effort", "on", "nothing-set"],
    )
    async def test_template_kwargs_also_sends_the_thinking_controls_as_template_variables(self, changes, sent):
        server = _Server()
        await _reply(server, _settings(**changes), template_kwargs=True)
        assert server.body == {"model": "qwen3-30b", "messages": MESSAGES, "stream": False, **sent}

    async def test_the_extra_bodys_template_variables_merge_key_by_key_and_the_presets_switch_wins(self):
        server = _Server()
        extra = {"chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": True}, "top_k": 20}
        await _reply(server, _settings(thinking="off"), extra_body=extra)
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "top_k": 20,
            "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": False},
        }
        assert extra["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": True}  # not mutated

    async def test_the_extra_bodys_template_variables_are_sent_when_no_control_is_set(self):
        server = _Server()
        await _reply(server, _settings(), extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}})
        assert server.body["chat_template_kwargs"] == {"reasoning_effort": "low"}

    # A preset's reasoning_effort wins over a top-level
    # reasoning_effort in the connection's extra body, as its other thinking controls do; when the preset sends none,
    # the extra body's still goes. Nothing else in the extra body changes.
    @pytest.mark.parametrize("client_kwargs", [{}, {"template_kwargs": True}], ids=["plain", "with-template-kwargs"])
    async def test_a_presets_effort_wins_over_the_extra_bodys_and_the_extra_bodys_goes_when_the_preset_sets_none(
        self, client_kwargs,
    ):
        extra = {"reasoning_effort": "high", "top_k": 20}
        server = _Server()
        await _reply(server, _settings(reasoning_effort="low"), extra_body=extra, **client_kwargs)
        template = {"chat_template_kwargs": {"reasoning_effort": "low"}} if client_kwargs else {}
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_effort": "low", "top_k": 20,
            **template,
        }, "the preset's effort, not the extra body's"
        await _reply(server, _settings(reasoning_effort=""), extra_body=extra, **client_kwargs)
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_effort": "high", "top_k": 20,
        }, "the preset sets none: the extra body's goes, and only at the top level, as it gave it"
        assert extra == {"reasoning_effort": "high", "top_k": 20}  # not mutated

    async def test_the_client_reads_template_kwargs_from_the_settings(self, clean_assistant_env, monkeypatch):
        from src.assistant.client import AssistantClient

        monkeypatch.setenv("OS_ASSISTANT_URL", AI_SERVER)
        monkeypatch.setenv("OS_ASSISTANT_TEMPLATE_KWARGS", "true")
        server = _Server()
        client = AssistantClient.from_settings(Settings(_env_file=None), transport=httpx.MockTransport(server))
        try:
            await client.reply(MESSAGES, _settings(reasoning_effort="low"))
        finally:
            await client.aclose()
        assert server.body["chat_template_kwargs"] == {"reasoning_effort": "low"}

    async def test_the_client_is_built_from_the_server_only_settings(self, clean_assistant_env, monkeypatch):
        from src.assistant.client import AssistantClient

        monkeypatch.setenv("OS_ASSISTANT_URL", AI_SERVER + "/")
        monkeypatch.setenv("OS_ASSISTANT_API_KEY", API_KEY)
        monkeypatch.setenv("OS_ASSISTANT_EXTRA_BODY", '{"top_k": 20}')
        monkeypatch.setenv("OS_ASSISTANT_MAX_TOKENS_FIELD", "max_completion_tokens")
        server = _Server()
        client = AssistantClient.from_settings(Settings(_env_file=None), transport=httpx.MockTransport(server))
        try:
            await client.reply(MESSAGES, _settings(max_tokens=64))
        finally:
            await client.aclose()
        assert str(server.requests[0].url) == f"{AI_SERVER}/chat/completions"
        assert server.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
        assert server.body["top_k"] == 20 and server.body["max_completion_tokens"] == 64
        assert "max_tokens" not in server.body
        assert API_KEY not in repr(client)

    @pytest.mark.parametrize(
        ("answer", "code"),
        [
            (_fail(httpx.ConnectError, "[Errno 11001] getaddrinfo failed"), "assistant_unreachable"),
            (_fail(httpx.ConnectError, "[SSL: CERTIFICATE_VERIFY_FAILED]"), "assistant_unreachable"),
            (_fail(httpx.RemoteProtocolError, "Server disconnected"), "assistant_unreachable"),
            (_fail(httpx.ReadTimeout, "timed out"), "assistant_timeout"),
            (lambda _r: httpx.Response(500, text="model crashed"), "assistant_http_error"),
            (lambda _r: httpx.Response(302, headers={"location": "http://elsewhere/"}), "assistant_http_error"),
            (lambda _r: httpx.Response(200, text="<html>not json</html>"), "assistant_bad_response"),
            (lambda _r: httpx.Response(200, json={"choices": []}), "assistant_bad_response"),
            (lambda _r: httpx.Response(200, json={"choices": [{"text": "legacy"}]}), "assistant_bad_response"),
            (lambda _r: httpx.Response(200, json=["not", "an", "object"]), "assistant_bad_response"),
            (lambda _r: _answer(content="   "), "assistant_empty_reply"),
            (lambda _r: _answer(content=None), "assistant_empty_reply"),
            (lambda _r: _answer(content="<think>All thinking, no answer.</think>"), "assistant_empty_reply"),
            (lambda _r: _answer(content="", finish_reason="length"), "assistant_reply_cut_off"),
            (lambda _r: _answer(content="<think>Still thinking when the limit", finish_reason="length"),
             "assistant_reply_cut_off"),
        ],
        ids=["dns", "tls", "disconnect", "read-timeout", "http-500", "redirect-not-followed", "not-json",
             "no-choices", "no-message", "not-an-object", "blank", "null-content", "only-thinking",
             "cut-off", "cut-off-while-thinking"],
    )
    async def test_each_error_code_from_its_cause(self, answer, code):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(answer))
        assert caught.value.code == code

    async def test_no_whole_reply_within_timeout_s_is_a_timeout(self):
        from src.assistant.errors import AssistantError

        async def slow(_request):
            await asyncio.sleep(1.0)
            return _answer()

        started = asyncio.get_running_loop().time()
        with pytest.raises(AssistantError) as caught:
            await _reply(slow, _settings(timeout_s=0.05))
        assert caught.value.code == "assistant_timeout"
        assert asyncio.get_running_loop().time() - started < 0.9

    async def test_an_http_error_names_only_its_status_and_never_reads_or_echoes_the_body(self):
        # The connection guards replace an earlier 200-character excerpt
        # with the status only, so a URL pointed at another service never has its answer echoed back.
        from src.assistant.errors import AssistantError

        stream = _RecordingStream([b"upstream says no ", b"internal secret " * 50])
        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(lambda _r: httpx.Response(503, stream=stream)))
        assert caught.value.code == "assistant_http_error"
        assert caught.value.message == "The AI server answered HTTP 503"
        assert stream.chunks_read == 0, "an error's body is never read"

    async def test_a_reply_larger_than_the_cap_is_a_bad_response(self, monkeypatch):
        from src.assistant import client as client_module
        from src.assistant.errors import AssistantError

        monkeypatch.setattr(client_module, "MAX_REPLY_BYTES", 1024)
        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(lambda _r: _answer(content="y" * 5000)))
        assert caught.value.code == "assistant_bad_response"

    @pytest.mark.parametrize(
        "answer",
        [
            lambda _r: httpx.Response(401, text=f"invalid api key {API_KEY}"),
            _fail(httpx.ConnectError, f"refused for Bearer {API_KEY}"),
            lambda _r: httpx.Response(200, text=f"not json {API_KEY}"),
        ],
        ids=["http-error-body", "transport-error", "bad-response"],
    )
    async def test_the_api_key_never_appears_in_an_error(self, answer):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(answer), api_key=API_KEY)
        assert API_KEY not in caught.value.message and API_KEY not in str(caught.value)


# ── Prompts: one pass, literal replacement ───────────────────────────────────


class TestPrompts:
    def test_rendering_is_one_pass_so_inserted_text_is_never_searched_for_placeholders(self):
        from src.assistant.prompts import render

        template = 'Summarise this part of "{chapter}":\n\n{text}'
        rendered = render(template, text="The page said {title} and {chapter}.", chapter="One", title="Dune")
        assert rendered == 'Summarise this part of "One":\n\nThe page said {title} and {chapter}.'

    def test_braces_elsewhere_in_the_template_survive(self):
        from src.assistant.prompts import render

        template = 'Reply as JSON like {"answer": 1}, {{text}} or { text } about {text}'
        assert render(template, text="cats") == 'Reply as JSON like {"answer": 1}, {cats} or { text } about cats'

    def test_inserted_text_is_literal_even_with_backslashes_and_group_references(self):
        from src.assistant.prompts import render

        text = r"C:\new\1 \g<0> $1"
        assert render("Q: {text}", text=text) == "Q: " + text

    def test_a_placeholder_without_a_value_is_empty_and_an_unknown_one_is_kept(self):
        from src.assistant.prompts import render

        assert render("{title}: {text} ({lang})", text="hi") == ": hi ({lang})"

    def test_the_placeholders_of_a_template_are_listed_in_order(self):
        from src.assistant.prompts import placeholders_in

        assert placeholders_in('{chapter} of {title}: {text} {"a": 1} {lang} {text}') == [
            "chapter", "title", "text", "lang", "text",
        ]

    def test_messages_are_the_system_prompt_the_history_and_the_rendered_users_words(self):
        from src.assistant.prompts import messages_for

        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."}]
        settings = _settings(system_prompt="Be brief.", user_template="Question: {text}")
        messages = messages_for(settings, history, "Why is the sky blue?")
        assert messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Question: Why is the sky blue?"},
        ]
        messages[1]["content"] = "changed"
        assert history[0]["content"] == "Hi", "the caller's history is not shared"

    def test_a_book_piece_gets_its_title_author_and_chapter(self):
        from src.assistant.prompts import messages_for

        settings = _settings(system_prompt="", user_template="{title} by {author}, {chapter}:\n{text}")
        messages = messages_for(settings, [], "It was dark.", title="Dune", author="Herbert", chapter="One")
        assert messages == [{"role": "user", "content": "Dune by Herbert, One:\nIt was dark."}]

    def test_without_text_the_history_already_ends_with_the_users_words(self):
        from src.assistant.prompts import messages_for

        history = [{"role": "user", "content": "Hi"}]
        assert messages_for(_settings(system_prompt="Be brief."), history) == [
            {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"},
        ]


# ── AI presets: the three layers, validation and the file store ──────────────

VOICE_PROMPT = (
    "You are a helpful voice assistant. Your replies are read aloud. Answer in short, plain spoken sentences. "
    "Don't use Markdown, lists, tables, code, links or emoji."
)
BOOK_PROMPT = (
    "You rework book text for listening, following the user's instructions. Reply only with plain prose "
    "paragraphs, because your reply is read aloud. Don't use Markdown, lists or headings, and don't comment on "
    "what you did."
)
DOCUMENTED_DEFAULTS = {
    "os_assistant_url": AI_SERVER,
    "os_assistant_api_key": "",
    "os_assistant_extra_body": "",
    "os_assistant_verify_tls": True,
    "os_assistant_max_tokens_field": "max_tokens",
    "os_assistant_model": "qwen3-30b",
    "os_assistant_system_prompt": "",
    "os_assistant_thinking": "default",
    "os_assistant_reasoning_effort": "",
    "os_assistant_thinking_budget": 0,
    "os_assistant_max_tokens": 0,
    "os_assistant_temperature": None,
    "os_assistant_top_p": None,
    "os_assistant_timeout_s": 120.0,
    "os_assistant_job_timeout_s": 600.0,
    "os_assistant_history_turns": 20,
    "os_assistant_chunk_chars": 12000,
    "os_assistant_show_thinking": True,
    "os_assistant_speak_errors": True,
}


@pytest.fixture
def assistant_on(monkeypatch, tmp_path):
    """The assistant on with the documented defaults (a model set), and a data folder of its own."""
    from src.config import settings

    for name, value in DOCUMENTED_DEFAULTS.items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(settings, "os_studio_db_path", str(tmp_path / "data" / "studio.db"))
    return settings


def _presets_file(settings) -> Path:
    return Path(settings.os_studio_db_path).parent / "assistant-presets.json"


def _preset_error(call, *args, **kwargs):
    from src.assistant.presets import PresetError

    with pytest.raises(PresetError) as caught:
        call(*args, **kwargs)
    return caught.value


class TestPresets:
    def test_the_built_ins_come_first_with_the_plans_defaults_and_no_file(self, assistant_on):
        from src.assistant import presets

        listed = presets.list_presets()
        assert [view["id"] for view in listed] == ["voice", "book"]
        voice, book = listed
        assert voice == {
            "id": "voice",
            "name": "Voice",
            # The action's name, the inputs it suits and its default output.
            "label": "Answer",
            "inputs": ["text", "transcript"],
            "output": "speech",
            "builtin": True,
            "customized": False,
            "inherited": list(presets.TUNABLE_FIELDS),
            "model": "qwen3-30b",
            "system_prompt": VOICE_PROMPT,
            "user_template": "{text}",
            "thinking": "default",
            "reasoning_effort": "",
            "thinking_budget": 0,
            "show_thinking": True,
            "max_tokens": 0,
            "temperature": None,
            "top_p": None,
            "timeout_s": 120.0,
            "history_turns": 20,
            "fillers": ["Hmm.", "OK, just a second.", "Let me think about that."],
            "filler_after_ms": 1200,
            "filler_every_ms": 4000,
            "filler_max": 3,
            "speak_errors": True,
            "error_sentences": {
                "assistant_unreachable": "I couldn't reach the AI server.",
                "assistant_timeout": "The AI took too long to answer.",
                "assistant_http_error": "The AI server returned an error.",
                "assistant_bad_response": "The AI didn't send back an answer.",
                "assistant_empty_reply": "The AI didn't send back an answer.",
                "assistant_reply_cut_off": "The answer was cut off before it started.",
            },
            "chunk_chars": 12000,
        }
        assert (book["name"], book["system_prompt"], book["timeout_s"]) == ("Book", BOOK_PROMPT, 600.0)
        assert presets.get_preset("voice") == voice
        assert presets.effective("voice").to_dict() == {
            "preset_id": "voice", **{k: v for k, v in voice.items()
                                     if k not in ("id", "builtin", "customized", "inherited", *presets.ACTION_FIELDS)},
        }
        assert not _presets_file(assistant_on).exists(), "the file is written only on save"

    def test_the_env_system_prompt_is_the_voice_default_and_not_the_books(self, assistant_on, monkeypatch):
        from src.assistant import presets

        monkeypatch.setattr(assistant_on, "os_assistant_system_prompt", "You are Ada. Be brief.")
        assert presets.effective("voice").system_prompt == "You are Ada. Be brief."
        assert presets.effective("book").system_prompt == BOOK_PROMPT

    def test_the_three_layers_merge_in_order(self, assistant_on, monkeypatch):
        from src.assistant import presets

        assert presets.effective("voice").max_tokens == 0  # the code's default
        monkeypatch.setattr(assistant_on, "os_assistant_max_tokens", 100)
        monkeypatch.setattr(assistant_on, "os_assistant_reasoning_effort", "low")
        assert (presets.effective("voice").max_tokens, presets.effective("voice").reasoning_effort) == (100, "low")

        presets.replace_preset("voice", {"max_tokens": 200, "error_sentences": {"assistant_timeout": "Slow today."}})
        saved = presets.effective("voice")
        assert (saved.max_tokens, saved.reasoning_effort) == (200, "low")

        request = presets.effective("voice", {
            "max_tokens": 300, "reasoning_effort": "high", "error_sentences": {"assistant_unreachable": "No AI server."},
        })
        assert (request.max_tokens, request.reasoning_effort, request.user_template) == (300, "high", "{text}")
        assert request.error_sentence("assistant_timeout") == "Slow today."
        assert request.error_sentence("assistant_unreachable") == "No AI server."
        assert request.error_sentence("assistant_http_error") == "The AI server returned an error."
        assert presets.effective("voice", {"max_tokens": None}).max_tokens == 200, "None in an override inherits"
        assert presets.get_preset("voice")["inherited"] == [
            field for field in presets.TUNABLE_FIELDS if field not in ("max_tokens", "error_sentences")
        ]

    def test_the_effective_settings_are_a_snapshot(self, assistant_on):
        from src.assistant import presets

        before = presets.effective("book")
        presets.replace_preset("book", {"chunk_chars": 5000})
        assert (before.chunk_chars, presets.effective("book").chunk_chars) == (12000, 5000)
        with pytest.raises(AttributeError):
            before.chunk_chars = 1  # frozen: a running job keeps what it started with

    def test_a_custom_preset_falls_back_by_purpose(self, assistant_on):
        from src.assistant import presets

        presets.create_preset({"name": "Short answers"})
        assert presets.effective("short-answers").timeout_s == 120.0
        assert presets.effective("short-answers").system_prompt == VOICE_PROMPT
        as_book = presets.effective("short-answers", purpose="book")
        assert (as_book.timeout_s, as_book.system_prompt) == (600.0, BOOK_PROMPT)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("name", ""), ("name", "x" * 81), ("name", 5),
            ("model", 5), ("model", "m" * 201),
            ("system_prompt", "x" * 8001), ("system_prompt", 3),
            ("user_template", "no placeholder"), ("user_template", "{text} in {lang}"),
            ("user_template", "{text}" + "x" * 8000), ("user_template", 7),
            ("thinking", "maybe"), ("thinking", True),
            ("reasoning_effort", "extreme"),
            ("show_thinking", "yes"), ("show_thinking", 1),
            ("max_tokens", -1), ("max_tokens", 32769), ("max_tokens", 1.5), ("max_tokens", True), ("max_tokens", "256"),
            ("temperature", -0.1), ("temperature", 2.1), ("temperature", math.nan), ("temperature", "hot"),
            ("top_p", 1.1), ("top_p", math.inf),
            ("timeout_s", 4), ("timeout_s", 3601),
            ("history_turns", -1), ("history_turns", 101),
            ("fillers", "Hmm."), ("fillers", ["x" * 81]), ("fillers", [""]), ("fillers", ["a"] * 11), ("fillers", [5]),
            ("filler_after_ms", -1), ("filler_after_ms", 30001), ("filler_every_ms", 30001), ("filler_max", 11),
            ("speak_errors", "no"),
            ("error_sentences", {"assistant_unknown": "Hm."}), ("error_sentences", {"assistant_timeout": "x" * 301}),
            ("error_sentences", ["Hm."]), ("error_sentences", {"assistant_timeout": 5}),
            ("chunk_chars", 999), ("chunk_chars", 200001),
            ("unknown_field", 1),
        ],
    )
    def test_every_fields_validation(self, assistant_on, field, value):
        from src.assistant import presets

        body = {"name": "Mine", field: value}
        assert _preset_error(presets.create_preset, body).field == field
        if field != "name":
            assert _preset_error(presets.effective, "voice", {field: value}).field == field
        assert not _presets_file(assistant_on).exists()

    def test_the_bounds_themselves_are_accepted(self, assistant_on):
        from src.assistant import presets

        low = {
            "name": "Low", "model": "m" * 200, "system_prompt": "x" * 8000, "user_template": "{text}" + "x" * 7994,
            "thinking": "off", "reasoning_effort": "none", "show_thinking": False, "max_tokens": 0,
            "temperature": 0, "top_p": 0, "timeout_s": 5, "history_turns": 0, "fillers": [],
            "filler_after_ms": 0, "filler_every_ms": 0, "filler_max": 0, "speak_errors": False,
            "error_sentences": {"assistant_timeout": "x" * 300}, "chunk_chars": 1000,
        }
        high = {
            "name": "x" * 80, "max_tokens": 32768, "temperature": 2, "top_p": 1, "timeout_s": 3600,
            "history_turns": 100, "fillers": ["y" * 80] * 10, "filler_after_ms": 30000, "filler_every_ms": 30000,
            "filler_max": 10, "chunk_chars": 200000,
        }
        assert presets.create_preset(low)["fillers"] == []
        assert presets.create_preset(high)["max_tokens"] == 32768
        assert presets.create_preset({"name": "Float", "max_tokens": 256.0})["max_tokens"] == 256

    def test_a_template_needs_text_and_only_its_presets_placeholders(self, assistant_on):
        from src.assistant import presets

        assert _preset_error(presets.replace_preset, "voice", {"user_template": "{title}: {text}"}).field == "user_template"
        assert _preset_error(presets.replace_preset, "book", {"user_template": "Summarise {chapter}"}).field == "user_template"
        book = presets.replace_preset("book", {"user_template": 'Summarise this part of "{chapter}" of {title} by {author}:\n\n{text}'})
        assert book["user_template"].startswith('Summarise this part of "{chapter}"')
        mine = presets.create_preset({"name": "Mine", "user_template": "{chapter}: {text}"})
        assert mine["user_template"] == "{chapter}: {text}"

    def test_an_empty_text_field_inherits(self, assistant_on):
        from src.assistant import presets

        view = presets.replace_preset("voice", {"model": "", "system_prompt": "  ", "user_template": "", "reasoning_effort": ""})
        assert (view["model"], view["system_prompt"], view["user_template"]) == ("qwen3-30b", VOICE_PROMPT, "{text}")
        assert view["customized"] is False

    def test_overrides_take_only_tunable_fields_and_a_known_preset(self, assistant_on):
        from src.assistant import presets

        assert _preset_error(presets.effective, "voice", {"name": "Other"}).field == "name"
        assert _preset_error(presets.effective, "nope").field == "preset"
        assert _preset_error(presets.effective, "voice", ["max_tokens"]).field == "overrides"

    def test_ids_are_slugs_of_the_name_and_never_taken_twice(self, assistant_on):
        from src.assistant import presets
        from src.assistant.presets import PresetConflictError

        made = presets.create_preset({"name": "Translate to Spanish!", "id": "ignored", "builtin": True})
        assert (made["id"], made["name"], made["builtin"], made["customized"]) == (
            "translate-to-spanish", "Translate to Spanish!", False, True,
        )
        for name in ("translate to spanish", "Voice", "BOOK"):
            with pytest.raises(PresetConflictError) as caught:
                presets.create_preset({"name": name})
            assert caught.value.code == "preset_exists"
        assert _preset_error(presets.create_preset, {"name": "!!!"}).field == "name"
        assert _preset_error(presets.create_preset, {"model": "m"}).field == "name"

    def test_the_file_is_written_only_on_save_and_holds_only_what_was_changed(self, assistant_on):
        from src.assistant import presets

        path = _presets_file(assistant_on)
        presets.list_presets(), presets.get_preset("book"), presets.effective("voice")
        assert not path.exists()
        presets.create_preset({"name": "Short answers", "max_tokens": 64})
        presets.replace_preset("voice", {"reasoning_effort": "low"})
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["presets"] == [
            {"id": "short-answers", "name": "Short answers", "max_tokens": 64},
            {"id": "voice", "reasoning_effort": "low"},
        ] or stored["presets"] == [
            {"id": "voice", "reasoning_effort": "low"},
            {"id": "short-answers", "name": "Short answers", "max_tokens": 64},
        ]
        assert [view["id"] for view in presets.list_presets()] == ["voice", "book", "short-answers"]

    def test_a_save_is_atomic_through_a_temporary_file_in_the_same_folder(self, assistant_on, monkeypatch):
        from src.assistant import presets

        path = _presets_file(assistant_on)
        presets.create_preset({"name": "First"})
        before = path.read_bytes()
        replaced = []
        real_replace = presets.os.replace

        def failing_replace(source, target):
            replaced.append((Path(source), Path(target)))
            raise OSError("disk full")

        monkeypatch.setattr(presets.os, "replace", failing_replace)
        with pytest.raises(OSError):
            presets.create_preset({"name": "Second"})
        assert replaced and replaced[0][0].parent == path.parent and replaced[0][1] == path
        assert path.read_bytes() == before, "the old file is intact"
        assert sorted(p.name for p in path.parent.iterdir() if p.name != "studio.db") == ["assistant-presets.json"]
        monkeypatch.setattr(presets.os, "replace", real_replace)
        assert [view["id"] for view in presets.list_presets()] == ["voice", "book", "first"]

    @pytest.mark.parametrize(
        "content",
        ["not json", '["a list"]', '{"presets": "nope"}', b"\xff\xfe\x00broken"],
        ids=["not-json", "a-list", "presets-not-a-list", "not-utf8"],
    )
    def test_an_unreadable_file_falls_back_to_the_built_ins_with_a_warning(self, assistant_on, caplog, content):
        from src.assistant import presets

        path = _presets_file(assistant_on)
        path.parent.mkdir(parents=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        with caplog.at_level(logging.WARNING):
            listed = presets.list_presets()
        assert [view["id"] for view in listed] == ["voice", "book"]
        assert listed[0]["customized"] is False
        assert any("assistant-presets.json" in record.getMessage() for record in caplog.records)

    def test_one_bad_preset_in_the_file_is_skipped_and_the_rest_kept(self, assistant_on, caplog):
        from src.assistant import presets

        path = _presets_file(assistant_on)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "presets": [
            {"id": "voice", "max_tokens": -5},
            {"id": "good", "name": "Good", "max_tokens": 10},
            {"id": "Bad Id", "name": "Bad"},
            {"name": "No id"},
        ]}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            listed = presets.list_presets()
        assert [view["id"] for view in listed] == ["voice", "book", "good"]
        assert listed[0]["max_tokens"] == 0 and listed[2]["max_tokens"] == 10
        assert caplog.records

    def test_delete_restores_a_built_in_and_removes_a_preset_you_made(self, assistant_on):
        from src.assistant import presets
        from src.assistant.presets import PresetNotFoundError

        presets.replace_preset("voice", {"name": "My voice", "max_tokens": 50})
        assert presets.get_preset("voice")["customized"] is True
        restored = presets.delete_preset("voice")
        assert (restored["name"], restored["max_tokens"], restored["customized"]) == ("Voice", 0, False)
        assert json.loads(_presets_file(assistant_on).read_text(encoding="utf-8"))["presets"] == []

        presets.create_preset({"name": "Mine"})
        assert presets.delete_preset("mine") is None
        assert presets.get_preset("mine") is None
        with pytest.raises(PresetNotFoundError):
            presets.delete_preset("mine")
        with pytest.raises(PresetNotFoundError):
            presets.replace_preset("mine", {"name": "Mine"})

    def test_a_preset_you_made_needs_its_name_when_replaced(self, assistant_on):
        from src.assistant import presets

        presets.create_preset({"name": "Mine", "max_tokens": 10})
        assert _preset_error(presets.replace_preset, "mine", {"max_tokens": 20}).field == "name"
        replaced = presets.replace_preset("mine", {"name": "Mine again", "top_p": 0.5})
        assert (replaced["id"], replaced["name"], replaced["max_tokens"], replaced["top_p"]) == ("mine", "Mine again", 0, 0.5)

    def test_at_most_50_presets(self, assistant_on):
        from src.assistant import presets
        from src.assistant.presets import PresetConflictError

        assert presets.MAX_PRESETS == 50
        for number in range(48):
            presets.create_preset({"name": f"Preset {number}"})
        assert len(presets.list_presets()) == 50
        with pytest.raises(PresetConflictError) as caught:
            presets.create_preset({"name": "One more"})
        assert caught.value.code == "too_many_presets"
        presets.replace_preset("voice", {"max_tokens": 5})  # changing one still works at the cap

    def test_the_api_key_never_reaches_the_preset_file_or_a_preset(self, assistant_on, monkeypatch):
        from src.assistant import presets

        monkeypatch.setattr(assistant_on, "os_assistant_api_key", API_KEY)
        monkeypatch.setattr(assistant_on, "os_assistant_extra_body", '{"top_k": 20}')
        presets.create_preset({"name": "Mine", "model": "other"})
        assert API_KEY not in _presets_file(assistant_on).read_text(encoding="utf-8")
        assert API_KEY not in json.dumps(presets.list_presets())
        assert "top_k" not in json.dumps(presets.effective("mine").to_dict())


# ── The thinking budget ──────────────────────────────────────────────────────
# A preset's thinking_budget (tokens; 0, the default, sends nothing; up to 1,000,000) goes as reasoning_budget_tokens
# (the llama.cpp-style name): at the budget the server closes the thinking and the model still writes a whole answer,
# where max_tokens cuts the answer off. It layers like the other thinking controls (OS_ASSISTANT_THINKING_BUDGET, the
# preset, a request's override), and a connection's extra body that sets the same key loses to the preset.


class TestThinkingBudget:
    def test_the_env_default_sends_none_and_its_range_is_checked_at_startup(self, clean_assistant_env, monkeypatch):
        assert "os_assistant_thinking_budget" in Settings.model_fields, "OS_ASSISTANT_THINKING_BUDGET is a setting"
        assert Settings(_env_file=None).os_assistant_thinking_budget == 0
        for value in ("0", "1000000"):
            monkeypatch.setenv("OS_ASSISTANT_THINKING_BUDGET", value)
            assert Settings(_env_file=None).os_assistant_thinking_budget == int(value)
        for value in ("-1", "1000001", "lots"):
            monkeypatch.setenv("OS_ASSISTANT_THINKING_BUDGET", value)
            with pytest.raises(ValidationError) as caught:
                Settings(_env_file=None)
            assert "os_assistant_thinking_budget" in str(caught.value).lower(), value

    def test_it_is_a_preset_field_after_the_effort_and_its_three_layers_merge_in_order(self, assistant_on, monkeypatch):
        from src.assistant import presets

        tunable = presets.TUNABLE_FIELDS
        assert "thinking_budget" in tunable, "a preset field"
        assert tunable.index("thinking_budget") == tunable.index("reasoning_effort") + 1, "shown with the thinking"
        assert presets.effective("voice").thinking_budget == 0  # the code's default: nothing sent
        assert presets.get_preset("voice")["thinking_budget"] == 0
        monkeypatch.setattr(assistant_on, "os_assistant_thinking_budget", 2048)
        assert (presets.effective("voice").thinking_budget, presets.effective("book").thinking_budget) == (2048, 2048)

        view = presets.replace_preset("voice", {"thinking_budget": 512})
        assert view["thinking_budget"] == 512 and "thinking_budget" not in view["inherited"]
        assert presets.effective("voice").thinking_budget == 512
        assert presets.effective("voice", {"thinking_budget": 256}).thinking_budget == 256, "a request's override"
        assert presets.effective("voice", {"thinking_budget": None}).thinking_budget == 512, "None in an override inherits"
        # Settings' Test: a budget emptied on screen falls to .env, as it would once saved.
        assert presets.effective("voice", {"thinking_budget": None}, unsaved=True).thinking_budget == 2048
        # 0 in a preset is its own value, as for max_tokens: no budget, whatever .env says.
        assert presets.replace_preset("voice", {"thinking_budget": 0})["thinking_budget"] == 0
        assert presets.effective("voice").thinking_budget == 0
        stored = json.loads(_presets_file(assistant_on).read_text(encoding="utf-8"))["presets"]
        assert stored == [{"id": "voice", "thinking_budget": 0}]
        assert presets.replace_preset("voice", {})["thinking_budget"] == 2048, "left out, it follows .env again"

    @pytest.mark.parametrize("value", [-1, 1_000_001, 1.5, True, "512", math.inf, None])
    def test_a_budget_that_is_not_a_whole_number_from_0_to_a_million_is_refused(self, assistant_on, value):
        from src.assistant import presets

        assert "thinking_budget" in presets.TUNABLE_FIELDS
        if value is None:  # empty: the layer below, never an error
            assert presets.create_preset({"name": "Mine", "thinking_budget": None})["inherited"].count("thinking_budget") == 1
            return
        message = "thinking_budget must be a whole number from 0 to 1000000"
        refused = _preset_error(presets.create_preset, {"name": "Mine", "thinking_budget": value})
        assert (refused.field, refused.message) == ("thinking_budget", message)
        overridden = _preset_error(presets.effective, "voice", {"thinking_budget": value})
        assert (overridden.field, overridden.message) == ("thinking_budget", message)
        assert not _presets_file(assistant_on).exists()

    def test_the_bounds_themselves_are_accepted(self, assistant_on):
        from src.assistant import presets

        assert "thinking_budget" in presets.TUNABLE_FIELDS
        assert presets.create_preset({"name": "Low", "thinking_budget": 0})["thinking_budget"] == 0
        assert presets.create_preset({"name": "High", "thinking_budget": 1_000_000})["thinking_budget"] == 1_000_000
        assert presets.create_preset({"name": "Float", "thinking_budget": 512.0})["thinking_budget"] == 512

    @pytest.mark.parametrize(
        ("budget", "client_kwargs", "sent"),
        [
            (1024, {}, {"reasoning_budget_tokens": 1024}),
            (0, {}, {}),
            # Only the top-level field (the llama.cpp-style name): no template variable for it.
            (1024, {"template_kwargs": True}, {"reasoning_budget_tokens": 1024}),
        ],
        ids=["set", "none", "with-template-kwargs"],
    )
    async def test_the_body_sends_it_as_reasoning_budget_tokens_only_when_set(self, budget, client_kwargs, sent):
        from dataclasses import fields

        from src.assistant.presets import EffectiveSettings

        assert "thinking_budget" in {item.name for item in fields(EffectiveSettings)}
        server = _Server()
        await _reply(server, _settings(thinking_budget=budget), **client_kwargs)
        assert server.body == {"model": "qwen3-30b", "messages": MESSAGES, "stream": False, **sent}

    async def test_a_connections_extra_body_that_sets_it_loses_to_the_preset_and_is_sent_when_the_preset_sets_none(self):
        from dataclasses import fields

        from src.assistant.presets import EffectiveSettings

        assert "thinking_budget" in {item.name for item in fields(EffectiveSettings)}
        extra = {"reasoning_budget_tokens": 9000, "top_k": 20}
        server = _Server()
        await _reply(server, _settings(thinking_budget=512), extra_body=extra)
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_budget_tokens": 512, "top_k": 20,
        }
        await _reply(server, _settings(thinking_budget=0), extra_body=extra)
        assert server.body == {
            "model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_budget_tokens": 9000, "top_k": 20,
        }
        assert extra == {"reasoning_budget_tokens": 9000, "top_k": 20}  # not mutated

    def test_a_saved_preset_and_settings_test_send_it_to_the_ai_server(self, api, no_shared_client):
        server = _Server()
        no_shared_client.set_client(_client(server))
        saved = api.put("/api/assistant/presets/voice", json={"name": "Voice", "thinking_budget": 700})
        assert saved.status_code == 200, saved.text
        assert saved.json()["thinking_budget"] == 700
        assert api.post("/api/assistant/test", json={"preset": "voice"}).status_code == 200
        assert server.body["reasoning_budget_tokens"] == 700
        tried = api.post("/api/assistant/test", json={"preset": "voice", "overrides": {"thinking_budget": 300}})
        assert tried.status_code == 200 and server.body["reasoning_budget_tokens"] == 300
        api.post("/api/assistant/test", json={"preset": "voice", "overrides": {"thinking_budget": 0}})
        assert "reasoning_budget_tokens" not in server.body, "0 on screen sends none"
        refused = api.put("/api/assistant/presets/voice", json={"name": "Voice", "thinking_budget": 1_000_001})
        assert refused.status_code == 422 and _error(refused)["param"] == "thinking_budget"
        assert api.get("/api/assistant/presets/voice").json()["thinking_budget"] == 700


class TestReasoningEffortXhigh:
    """`xhigh` is a reasoning effort, as some AI server templates take low, medium and xhigh.
    OS_ASSISTANT_REASONING_EFFORT, the AI presets and the preset and Settings-test APIs accept it, and it goes to the
    AI server as reasoning_effort."""

    def test_the_env_setting_accepts_it_and_a_preset_inherits_and_sends_it(
        self, clean_assistant_env, assistant_on, monkeypatch,
    ):
        from src.assistant import presets

        monkeypatch.setenv("OS_ASSISTANT_REASONING_EFFORT", "xhigh")
        assert Settings(_env_file=None).os_assistant_reasoning_effort == "xhigh"
        monkeypatch.setattr(assistant_on, "os_assistant_reasoning_effort", "xhigh")
        assert presets.effective("voice").reasoning_effort == "xhigh"

    async def test_the_client_sends_it_at_the_top_level_and_as_a_template_variable(self):
        server = _Server()
        await _reply(server, _settings(reasoning_effort="xhigh"))
        assert server.body == {"model": "qwen3-30b", "messages": MESSAGES, "stream": False, "reasoning_effort": "xhigh"}
        await _reply(server, _settings(reasoning_effort="xhigh"), template_kwargs=True)
        assert server.body["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}

    def test_a_saved_preset_and_settings_test_send_it_to_the_ai_server(self, api, no_shared_client):
        server = _Server()
        no_shared_client.set_client(_client(server))
        saved = api.put("/api/assistant/presets/voice", json={"name": "Voice", "reasoning_effort": "xhigh"})
        assert saved.status_code == 200, saved.text
        assert saved.json()["reasoning_effort"] == "xhigh"
        created = api.post("/api/assistant/presets", json={"name": "Deep", "reasoning_effort": "xhigh"})
        assert created.status_code == 201 and created.json()["reasoning_effort"] == "xhigh", created.text
        assert api.post("/api/assistant/test", json={"preset": "voice"}).status_code == 200
        assert server.body["reasoning_effort"] == "xhigh"
        tried = api.post("/api/assistant/test", json={"preset": "book", "overrides": {"reasoning_effort": "xhigh"}})
        assert tried.status_code == 200 and server.body["reasoning_effort"] == "xhigh"
        refused = api.put("/api/assistant/presets/voice", json={"name": "Voice", "reasoning_effort": "max"})
        assert refused.status_code == 422 and _error(refused)["param"] == "reasoning_effort"


# ── The conversation (in memory, per realtime connection) ────────────────────


def _user(text):
    return {"role": "user", "content": text}


def _ai(text):
    return {"role": "assistant", "content": text}


class TestConversation:
    async def test_the_thinking_is_kept_out_of_the_history(self):
        from src.assistant.conversation import Conversation

        reply = await _reply(_Server(lambda _r: _answer(content="<think>Secret plan.</think>The answer.")))
        conversation = Conversation()
        conversation.add_user("Question?")
        conversation.add_assistant(reply.text, "item_1")
        assert conversation.for_request(20) == [_user("Question?"), _ai("The answer.")]
        assert "Secret plan" not in json.dumps(conversation.for_request(20))

    def test_history_turns_counts_exchanges_and_the_users_pending_words_always_go(self):
        from src.assistant.conversation import Conversation

        conversation = Conversation()
        for number in (1, 2, 3):
            conversation.add_user(f"q{number}")
            conversation.add_assistant(f"a{number}", f"item_{number}")
        assert conversation.for_request(2) == [_user("q2"), _ai("a2"), _user("q3"), _ai("a3")]
        assert conversation.for_request(0) == []
        conversation.add_user("q4")
        assert conversation.for_request(0) == [_user("q4")]
        assert conversation.for_request(1) == [_user("q3"), _ai("a3"), _user("q4")]
        assert len(conversation.for_request(100)) == 7

    def test_an_unanswered_question_stays_and_two_user_messages_in_a_row_are_one_exchange(self):
        from src.assistant.conversation import Conversation

        conversation = Conversation()
        conversation.add_user("first, interrupted")
        conversation.add_user("second")
        assert conversation.for_request(20) == [_user("first, interrupted"), _user("second")]
        conversation.add_assistant("answer", "item_1")
        assert conversation.for_request(1) == [_user("first, interrupted"), _user("second"), _ai("answer")]

    def test_the_history_handed_out_is_a_copy_and_reset_starts_fresh(self):
        from src.assistant.conversation import Conversation

        conversation = Conversation()
        conversation.add_user("q")
        conversation.for_request(20)[0]["content"] = "changed"
        assert conversation.for_request(20) == [_user("q")]
        conversation.reset()
        assert conversation.for_request(20) == []

    def test_the_stored_history_is_bounded(self):
        from src.assistant.conversation import MAX_EXCHANGES, Conversation

        conversation = Conversation()
        for number in range(MAX_EXCHANGES + 50):
            conversation.add_user(f"q{number}")
            conversation.add_assistant(f"a{number}", None)
        kept = conversation.for_request(10_000)
        assert len(kept) == 2 * MAX_EXCHANGES and kept[0] == _user("q50")
        assert MAX_EXCHANGES >= 100, "history_turns goes up to 100"

    def test_truncating_an_answer_keeps_what_was_heard(self):
        from src.assistant.conversation import Conversation

        conversation = Conversation()
        conversation.add_user("Tell me a story.")
        conversation.add_assistant("It was a dark and stormy night.", "item_7")
        assert conversation.truncate_assistant("item_unknown", 0.5) is False
        assert conversation.truncate_assistant("item_7", math.nan) is False
        assert conversation.truncate_assistant("item_7", 2.0) is True
        assert conversation.for_request(5)[-1] == _ai("It was a dark and stormy night.")
        assert conversation.truncate_assistant("item_7", 0.5) is True
        assert conversation.for_request(5)[-1] == _ai("It was a dark")
        assert conversation.truncate_assistant("item_7", 0.0) is True
        assert conversation.for_request(5) == [_user("Tell me a story.")], "nothing heard: the answer goes"


# ── A reply cleaned for speech ───────────────────────────────────────────────


class TestSpeech:
    def test_markdown_links_and_code_are_cleaned_and_no_segment_is_empty(self):
        from src.assistant.speech import speakable_segments

        reply = (
            "Hello **there**. See [the docs](http://example.com/secret).\n\n"
            "```python\nprint('hidden')\n```\n\n- Done.\n"
        )
        segments = speakable_segments(reply, 500)
        spoken = " ".join(segments)
        assert segments and all(segment.strip() for segment in segments)
        assert "Hello there." in spoken and "the docs" in spoken and "Code block skipped." in spoken
        for gone in ("**", "http", "hidden", "```"):
            assert gone not in spoken

    def test_a_long_reply_is_split_within_the_limit_and_spoken_in_full(self):
        from src.assistant.speech import speakable_segments

        sentences = " ".join(f"This is sentence number {n} of a long answer about the sea." for n in range(80))
        segments = speakable_segments(sentences, 200)
        assert len(segments) > 1 and max(len(segment) for segment in segments) <= 200
        assert " ".join(segments).split() == sentences.split()

    def test_the_limit_defaults_to_the_live_readers_sentence_limit(self, monkeypatch):
        from src.assistant.speech import speakable_segments
        from src.config import settings

        monkeypatch.setattr(settings, "tts_live_sentence_max_chars", 300)
        monkeypatch.setattr(settings, "tts_live_sentence_max_words", 1000)
        long_sentence = "word " * 400
        assert max(len(segment) for segment in speakable_segments(long_sentence)) <= 300
        assert max(len(segment) for segment in speakable_segments(long_sentence, 4096)) <= 300
        assert max(len(segment) for segment in speakable_segments(long_sentence, 120)) <= 120

    def test_nothing_to_say_gives_no_segments(self):
        from src.assistant.speech import speakable_segments

        assert speakable_segments("", 500) == [] and speakable_segments("  \n\n ", 500) == []


# ── The shared client: set_client, get_client, on_close, close_client ───────


@pytest.fixture
def no_shared_client():
    """Each test starts and ends without a shared client or close callbacks."""
    from src import assistant

    yield assistant
    assistant.set_client(None)
    assistant._close_callbacks.clear()
    getattr(assistant, "_retiring", set()).clear()  # replaced clients still closing


class TestSharedClient:
    async def test_close_client_runs_the_on_close_callbacks_then_closes_the_client(self, no_shared_client, caplog):
        assistant = no_shared_client
        client = _client(_Server())
        assistant.set_client(client)
        assert assistant.get_client() is client
        ran = []

        async def first():
            ran.append(("first", assistant.get_client() is client, client.is_closed))

        async def broken():
            raise RuntimeError("a job runner that fails to stop")

        async def last():
            ran.append(("last", assistant.get_client() is client, client.is_closed))

        assert assistant.on_close(first) is first
        assistant.on_close(broken)
        assistant.on_close(last)
        with caplog.at_level(logging.WARNING):
            await assistant.close_client()
        assert ran == [("first", True, False), ("last", True, False)], "callbacks run before the client closes"
        assert client.is_closed and assistant.get_client() is None
        assert any("a job runner that fails to stop" in record.getMessage() or record.exc_info for record in caplog.records)
        await assistant.close_client()
        assert len(ran) == 2, "a second close runs nothing again"

    def test_the_package_names_the_later_phases_use(self):
        from src import assistant
        from src.assistant import presets, prompts

        for name in ("get_client", "set_client", "on_close", "close_client", "AssistantClient", "AssistantError",
                     "AssistantReply", "Conversation", "EffectiveSettings", "speakable_segments", "ERROR_SENTENCES"):
            assert hasattr(assistant, name), name
        assert callable(presets.effective) and callable(prompts.render) and callable(prompts.messages_for)

    def test_the_app_builds_the_client_at_startup_when_a_url_is_set_and_closes_it_at_shutdown(self, monkeypatch, no_shared_client):
        import src.main as main_module

        assistant = no_shared_client
        monkeypatch.setattr(main_module.settings, "os_assistant_url", AI_SERVER)
        monkeypatch.setattr(main_module.settings, "os_assistant_api_key", API_KEY)
        closed = []

        async def stop_jobs():
            closed.append(True)

        with TestClient(main_module.app):
            client = assistant.get_client()
            assert client is not None and repr(client) == "<AssistantClient ai-server.local:8000>"
            assistant.on_close(stop_jobs)
        assert closed == [True] and assistant.get_client() is None and client.is_closed

    def test_without_a_url_the_app_starts_with_no_client(self, monkeypatch, no_shared_client):
        import src.main as main_module

        monkeypatch.setattr(main_module.settings, "os_assistant_url", "")
        with TestClient(main_module.app):
            assert no_shared_client.get_client() is None


# ── The API: /api/assistant and /api/assistant/presets, /api/assistant/test ──


@pytest.fixture
def api(assistant_on, no_shared_client, monkeypatch):
    """The configured app (no lifespan: no models) with the assistant on and no API key."""
    import src.main as main_module

    monkeypatch.setattr(assistant_on, "os_api_key", "")
    monkeypatch.setattr(assistant_on, "os_security_headers", True)
    return TestClient(main_module.create_app())


def _error(response) -> dict:
    return response.json()["error"]


class _FailingClient:
    """A stand-in for the shared client whose reply raises one AssistantError."""

    def __init__(self, code):
        self.code = code

    async def reply(self, messages, settings):
        from src.assistant.errors import AssistantError

        raise AssistantError(f"failed with {self.code}", self.code)


class TestApi:
    def test_the_status_shows_the_host_and_model_but_never_the_key_the_urls_path_or_the_extra_body(self, api, assistant_on, monkeypatch):
        monkeypatch.setattr(assistant_on, "os_assistant_url", "http://someone:pw123@ai-server.local:8000/v1/hidden-path?token=abc")
        monkeypatch.setattr(assistant_on, "os_assistant_api_key", API_KEY)
        monkeypatch.setattr(assistant_on, "os_assistant_extra_body", '{"top_k": 20}')
        monkeypatch.setattr(assistant_on, "os_assistant_template_kwargs", True)
        response = api.get("/api/assistant")
        assert response.status_code == 200
        assert response.json() == {
            "enabled": True,
            "host": "ai-server.local:8000",
            "model": "qwen3-30b",
            "server_settings": {
                "OS_ASSISTANT_URL": True,
                "OS_ASSISTANT_API_KEY": True,
                "OS_ASSISTANT_EXTRA_BODY": True,
                "OS_ASSISTANT_VERIFY_TLS": False,
                "OS_ASSISTANT_MAX_TOKENS_FIELD": False,
                "OS_ASSISTANT_TEMPLATE_KWARGS": True,
                "OS_ASSISTANT_STREAM": False,
            },
            # Where each comes from, and the lock.
            "locked": False,
            "sources": {
                "OS_ASSISTANT_URL": "env",
                "OS_ASSISTANT_API_KEY": "env",
                "OS_ASSISTANT_MODEL": "env",
                "OS_ASSISTANT_VERIFY_TLS": "default",
                "OS_ASSISTANT_TEMPLATE_KWARGS": "env",
                "OS_ASSISTANT_MAX_TOKENS_FIELD": "default",
                "OS_ASSISTANT_EXTRA_BODY": "env",
                "OS_ASSISTANT_STREAM": "default",
            },
        }
        for secret in (API_KEY, "hidden-path", "token", "abc", "pw123", "someone", "top_k", "/v1"):
            assert secret not in response.text

    def test_when_off_the_status_says_so_and_every_other_route_is_not_found(self, api, assistant_on, monkeypatch):
        monkeypatch.setattr(assistant_on, "os_assistant_url", "")
        assert api.get("/api/assistant").json() == {
            "enabled": False,
            "host": None,
            "model": "qwen3-30b",
            "server_settings": {
                "OS_ASSISTANT_URL": False,
                "OS_ASSISTANT_API_KEY": False,
                "OS_ASSISTANT_EXTRA_BODY": False,
                "OS_ASSISTANT_VERIFY_TLS": False,
                "OS_ASSISTANT_MAX_TOKENS_FIELD": False,
                "OS_ASSISTANT_TEMPLATE_KWARGS": False,
                "OS_ASSISTANT_STREAM": False,
            },
            "locked": False,
            "sources": {
                "OS_ASSISTANT_URL": "default",
                "OS_ASSISTANT_API_KEY": "default",
                "OS_ASSISTANT_MODEL": "env",
                "OS_ASSISTANT_VERIFY_TLS": "default",
                "OS_ASSISTANT_TEMPLATE_KWARGS": "default",
                "OS_ASSISTANT_MAX_TOKENS_FIELD": "default",
                "OS_ASSISTANT_EXTRA_BODY": "default",
                "OS_ASSISTANT_STREAM": "default",
            },
        }
        for method, path, body in [
            ("GET", "/api/assistant/presets", None),
            ("POST", "/api/assistant/presets", {"name": "Mine"}),
            ("GET", "/api/assistant/presets/voice", None),
            ("PUT", "/api/assistant/presets/voice", {"max_tokens": 5}),
            ("DELETE", "/api/assistant/presets/voice", None),
            ("POST", "/api/assistant/test", {}),
        ]:
            response = api.request(method, path, json=body)
            assert response.status_code == 404, (method, path)
            assert _error(response)["code"] == "assistant_disabled", (method, path)
        assert not _presets_file(assistant_on).exists()

    def test_the_status_says_whether_stream_replies_differs_from_its_default(self, api, assistant_on, monkeypatch):
        """server_settings has OS_ASSISTANT_STREAM like
        the other connection settings, true only while Stream replies is off (it is on by default), from .env or
        saved in Settings, and its source in sources."""
        from src.assistant import connection

        monkeypatch.setattr(assistant_on, "os_assistant_connection_locked", False)
        monkeypatch.setattr(assistant_on, "os_assistant_stream", True)

        def stream_status():
            status = api.get("/api/assistant").json()
            return status["server_settings"].get("OS_ASSISTANT_STREAM"), status["sources"]["OS_ASSISTANT_STREAM"]

        assert stream_status() == (False, "default")
        monkeypatch.setattr(assistant_on, "os_assistant_stream", False)
        assert stream_status() == (True, "env")
        connection.save({"stream": True})
        assert stream_status() == (False, "saved")
        monkeypatch.setattr(assistant_on, "os_assistant_stream", True)
        connection.save({"stream": False})
        assert stream_status() == (True, "saved")
        connection.save({"stream": None})
        assert stream_status() == (False, "default")

    def test_the_routes_need_the_api_key_when_one_is_set(self, api, assistant_on, monkeypatch):
        monkeypatch.setattr(assistant_on, "os_api_key", "open-speech-key")
        assert api.get("/api/assistant").status_code == 401
        assert api.get("/api/assistant/presets").status_code == 401
        assert api.get("/api/assistant", headers={"Authorization": "Bearer open-speech-key"}).status_code == 200

    def test_every_preset_route(self, api):
        listed = api.get("/api/assistant/presets")
        assert listed.status_code == 200 and [p["id"] for p in listed.json()["presets"]] == ["voice", "book"]

        created = api.post("/api/assistant/presets", json={"name": "Short answers", "max_tokens": 64})
        assert created.status_code == 201
        assert (created.json()["id"], created.json()["max_tokens"], created.json()["builtin"]) == ("short-answers", 64, False)
        assert api.get("/api/assistant/presets/short-answers").json() == created.json()
        assert [p["id"] for p in api.get("/api/assistant/presets").json()["presets"]] == ["voice", "book", "short-answers"]

        replaced = api.put("/api/assistant/presets/short-answers", json={**created.json(), "name": "Short", "max_tokens": 32})
        assert replaced.status_code == 200 and (replaced.json()["name"], replaced.json()["max_tokens"]) == ("Short", 32)

        assert api.delete("/api/assistant/presets/short-answers").status_code == 204
        missing = api.get("/api/assistant/presets/short-answers")
        assert missing.status_code == 404 and _error(missing)["code"] == "preset_not_found"
        assert api.put("/api/assistant/presets/short-answers", json={"name": "x"}).status_code == 404
        assert api.delete("/api/assistant/presets/short-answers").status_code == 404

        voice = api.put("/api/assistant/presets/voice", json={"reasoning_effort": "low"})
        assert voice.status_code == 200 and (voice.json()["reasoning_effort"], voice.json()["customized"]) == ("low", True)
        restored = api.delete("/api/assistant/presets/voice")
        assert restored.status_code == 200
        assert (restored.json()["reasoning_effort"], restored.json()["customized"]) == ("", False)

    @pytest.mark.parametrize(
        ("method", "path", "body", "param"),
        [
            ("POST", "/api/assistant/presets", {"name": "X", "max_tokens": -1}, "max_tokens"),
            ("POST", "/api/assistant/presets", {"name": "X", "fillers": ["x" * 81]}, "fillers"),
            ("POST", "/api/assistant/presets", {"max_tokens": 5}, "name"),
            ("PUT", "/api/assistant/presets/voice", {"user_template": "no text here"}, "user_template"),
            ("PUT", "/api/assistant/presets/book", {"surprise": 1}, "surprise"),
            ("POST", "/api/assistant/test", {"preset": "nope"}, "preset"),
            ("POST", "/api/assistant/test", {"overrides": {"top_p": 3}}, "top_p"),
            ("POST", "/api/assistant/test", {"overrides": ["top_p"]}, "overrides"),
            ("POST", "/api/assistant/test", {"text": 5}, "text"),
            ("POST", "/api/assistant/test", {"text": "x" * 8001}, "text"),
            ("POST", "/api/assistant/test", {"what": 1}, "what"),
        ],
    )
    def test_an_invalid_field_gets_422_with_its_name(self, api, method, path, body, param):
        response = api.request(method, path, json=body)
        assert response.status_code == 422, response.text
        error = _error(response)
        assert (error["code"], error["param"]) == ("validation_error", param)
        assert param in error["message"]

    def test_a_body_that_is_not_an_object_gets_422(self, api):
        response = api.post("/api/assistant/presets", json=["name"])
        assert response.status_code == 422 and _error(response)["code"] == "validation_error"

    def test_a_taken_id_gets_409(self, api):
        response = api.post("/api/assistant/presets", json={"name": "Voice"})
        assert response.status_code == 409 and _error(response)["code"] == "preset_exists"

    def test_the_test_call_makes_one_request_with_the_preset_and_overrides(self, api, no_shared_client):
        server = _Server(lambda _r: _answer(content="Hello there.", reasoning_content="Let me see."))
        no_shared_client.set_client(_client(server))
        response = api.post("/api/assistant/test", json={
            "preset": "voice", "overrides": {"reasoning_effort": "low", "max_tokens": 128},
        })
        assert response.status_code == 200, response.text
        answer = response.json()
        assert {key: answer[key] for key in ("reply", "thinking", "finish_reason")} == {
            "reply": "Hello there.", "thinking": "Let me see.", "finish_reason": "stop",
        }
        assert isinstance(answer["elapsed_ms"], int) and answer["elapsed_ms"] >= 0
        assert server.body == {
            "model": "qwen3-30b",
            "messages": [{"role": "system", "content": VOICE_PROMPT}, {"role": "user", "content": "Say hello in one sentence."}],
            "stream": False,
            "reasoning_effort": "low",
            "max_tokens": 128,
        }

    def test_the_test_call_renders_the_template_and_hides_the_thinking_when_not_shown(self, api, no_shared_client):
        server = _Server(lambda _r: _answer(content="Hola.", reasoning_content="Pensando."))
        no_shared_client.set_client(_client(server))
        response = api.post("/api/assistant/test", json={
            "preset": "book", "text": "Hello {title}.",
            "overrides": {"user_template": "Translate: {text}", "show_thinking": False},
        })
        assert response.status_code == 200
        assert (response.json()["reply"], response.json()["thinking"]) == ("Hola.", None)
        assert server.body["messages"][-1] == {"role": "user", "content": "Translate: Hello {title}."}

    @pytest.mark.parametrize(("code", "status"), [
        ("assistant_unreachable", 502), ("assistant_timeout", 504), ("assistant_http_error", 502),
        ("assistant_bad_response", 502), ("assistant_empty_reply", 502), ("assistant_reply_cut_off", 502),
    ])
    def test_the_test_call_answers_the_error_code(self, api, no_shared_client, code, status):
        no_shared_client.set_client(_FailingClient(code))
        response = api.post("/api/assistant/test", json={})
        assert response.status_code == status
        assert _error(response)["code"] == code and isinstance(response.json()["elapsed_ms"], int)

    def test_the_test_calls_error_never_shows_the_key(self, api, no_shared_client):
        server = _Server(lambda _r: httpx.Response(401, text=f"bad key {API_KEY}"))
        no_shared_client.set_client(_client(server, api_key=API_KEY))
        response = api.post("/api/assistant/test", json={"text": "hi"})
        assert response.status_code == 502 and _error(response)["code"] == "assistant_http_error"
        assert API_KEY not in response.text and "401" in _error(response)["message"]

    def test_the_test_call_before_the_client_has_started_is_unavailable(self, api):
        response = api.post("/api/assistant/test", json={})
        assert response.status_code == 503 and _error(response)["code"] == "assistant_not_started"

    # Test tries the preset as it is on screen, even unsaved (the route's
    # docstring). A field emptied on screen is sent as null, or as a value its check empties (""), and falls to .env
    # or the code, as it does once saved; a field left out keeps the saved value. .env's reasoning_effort is "low".

    @staticmethod
    def _tried(api, server, overrides=None) -> dict:
        payload = {"preset": "voice"} if overrides is None else {"preset": "voice", "overrides": overrides}
        response = api.post("/api/assistant/test", json=payload)
        assert response.status_code == 200, response.text
        return server.body

    @pytest.mark.parametrize(("field", "saved", "emptied", "expected"), [
        pytest.param("reasoning_effort", "medium", None, {"reasoning_effort": "low"}, id="reasoning_effort-null"),
        pytest.param("reasoning_effort", "medium", "", {"reasoning_effort": "low"}, id="reasoning_effort-empty"),
        pytest.param("temperature", 0.7, None, {"temperature": "not sent"}, id="temperature-null"),
        pytest.param("system_prompt", "Talk like a pirate.", "", {"system": VOICE_PROMPT}, id="system_prompt-empty"),
    ])
    def test_the_test_call_tries_a_field_emptied_on_screen_as_saving_it_would(
        self, api, assistant_on, no_shared_client, monkeypatch, field, saved, emptied, expected,
    ):
        monkeypatch.setattr(assistant_on, "os_assistant_reasoning_effort", "low")
        server = _Server()
        no_shared_client.set_client(_client(server))
        assert api.put("/api/assistant/presets/voice", json={"name": "Voice", field: saved}).status_code == 200

        unsaved = self._tried(api, server, {field: emptied})
        assert api.put("/api/assistant/presets/voice", json={"name": "Voice", field: emptied}).status_code == 200
        saved_then_tried = self._tried(api, server)

        assert unsaved == saved_then_tried, f"{field} emptied on screen was tried with its saved value {saved!r}"
        shown = {
            "reasoning_effort": unsaved.get("reasoning_effort", "not sent"),
            "temperature": unsaved.get("temperature", "not sent"),
            "system": unsaved["messages"][0]["content"],
        }
        assert {key: shown[key] for key in expected} == expected

    def test_the_test_call_keeps_the_saved_value_of_a_field_left_out(self, api, assistant_on, no_shared_client, monkeypatch):
        monkeypatch.setattr(assistant_on, "os_assistant_reasoning_effort", "low")
        server = _Server()
        no_shared_client.set_client(_client(server))
        body = {"name": "Voice", "reasoning_effort": "medium", "temperature": 0.7, "system_prompt": "Talk like a pirate."}
        assert api.put("/api/assistant/presets/voice", json=body).status_code == 200

        tried = self._tried(api, server, {"temperature": None})
        assert (tried["reasoning_effort"], "temperature" in tried) == ("medium", False)
        assert tried["messages"][0] == {"role": "system", "content": "Talk like a pirate."}
        tried = self._tried(api, server, {})
        assert (tried["reasoning_effort"], tried["temperature"]) == ("medium", 0.7)

    def test_only_the_test_call_empties_a_field_other_requests_overrides_keep_the_preset_value(self, assistant_on):
        # A realtime session.update and a book job's overrides still skip an empty override (presets.py docstring).
        from src.assistant import presets

        presets.replace_preset("voice", {"reasoning_effort": "medium", "temperature": 0.7})
        kept = presets.effective("voice", {"reasoning_effort": None, "temperature": None})
        assert (kept.reasoning_effort, kept.temperature) == ("medium", 0.7)
        kept = presets.effective("voice", {"reasoning_effort": ""}, purpose="voice")
        assert kept.reasoning_effort == "medium"


# ── The connection: stored over .env, guarded and lockable ─────
#
# README.md, "The connection": a URL, key and model saved in Settings win over .env unless
# OS_ASSISTANT_CONNECTION_LOCKED is set, and every URL is checked before use. No network: the `servers` fixture
# replaces src.assistant.build_client, which every rebuilt client and every /check client comes from, with an
# httpx.MockTransport that answers only the hosts a test adds, so a request anywhere else fails the test.

OTHER = "http://other.local:9000/v1"
NEW_KEY = "sk-new-key-456"
AI_SERVER_PATH = "http://ai-server.local:8000/api/v1"  # AI_SERVER's origin with another path
# URLs urllib's urlsplit takes but httpx refuses (httpx.InvalidURL, not an HTTPError)
# when it builds each request. The guards refuse them, so .env stops at startup and Settings answers 422 naming url.
HTTPX_REFUSED_URLS = [
    "http://192.168.1.300:8080/v1",  # an octet over 255: a typo
    "http://192.168.001.010:8080/v1",  # leading zeros, which Python's ipaddress refuses
    "http://ex_ämple.lan:8080/v1",  # a non-ASCII name IDNA refuses (the underscore)
]
HTTPX_REFUSED_IDS = ["octet-over-255", "leading-zeros", "bad-idna"]


def _connection_file(settings) -> Path:
    return Path(settings.os_studio_db_path).parent / "assistant-connection.json"


def _connection_error(call, *args, **kwargs):
    from src.assistant.connection import ConnectionFieldError

    with pytest.raises(ConnectionFieldError) as caught:
        call(*args, **kwargs)
    return caught.value


@pytest.fixture
def connection_on(assistant_on, monkeypatch):
    """The assistant on from .env (OS_ASSISTANT_URL is AI_SERVER), unlocked, with its own data folder."""
    monkeypatch.setattr(assistant_on, "os_assistant_connection_locked", False)
    monkeypatch.setattr(assistant_on, "os_assistant_template_kwargs", False)
    return assistant_on


class _Servers:
    """MockTransport servers by host. A request to a host no test added fails the test."""

    def __init__(self):
        self.by_host: dict[str, _Server] = {}

    def add(self, url, answer=None) -> _Server:
        server = _Server(answer)
        self.by_host[httpx.URL(url).host] = server
        return server

    def __call__(self, request: httpx.Request):
        server = self.by_host.get(request.url.host)
        if server is None:
            raise AssertionError(f"a request to {request.url.host}, which no test server answers")
        return server(request)


@pytest.fixture
def servers(monkeypatch, no_shared_client):
    from src.assistant.client import AssistantClient

    routes = _Servers()
    monkeypatch.setattr(
        no_shared_client, "build_client",
        lambda connection: AssistantClient.from_connection(connection, transport=httpx.MockTransport(routes)),
    )
    return routes


async def _until(condition, timeout=2.0):
    async def wait():
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout)


class TestConnectionSettings:
    def test_the_connection_is_unlocked_by_default_and_the_lock_is_a_setting(self, clean_assistant_env, monkeypatch):
        assert Settings(_env_file=None).os_assistant_connection_locked is False
        monkeypatch.setenv("OS_ASSISTANT_CONNECTION_LOCKED", "true")
        assert Settings(_env_file=None).os_assistant_connection_locked is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://user:pw@ai-server.local:8000/v1", "http://user@ai-server.local/v1", "http://:pw@ai-server.local/v1",
            "http://@ai-server.local/v1", "http://ai-server.local/v1?key=abc", "http://ai-server.local/v1#part",
            "http://ai-server.local:99999/v1", "http://ai-server .local/v1", "file:///etc/passwd", "http:///v1",
            *HTTPX_REFUSED_URLS,
        ],
        ids=["user-and-password", "user", "password", "empty-userinfo", "query", "fragment", "bad-port", "space",
             "file", "no-host", *HTTPX_REFUSED_IDS],
    )
    def test_a_url_the_guards_refuse_is_refused_at_startup(self, clean_assistant_env, monkeypatch, url):
        monkeypatch.setenv("OS_ASSISTANT_URL", url)
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert "os_assistant_url" in str(caught.value).lower()

    @pytest.mark.parametrize("url", ["http://192.168.1.50:8080/v1", "https://[::1]:8000/v1", "http://ai-server/v1/"])
    def test_a_plain_base_url_is_accepted(self, clean_assistant_env, monkeypatch, url):
        monkeypatch.setenv("OS_ASSISTANT_URL", url)
        assert Settings(_env_file=None).os_assistant_url == url.rstrip("/")


class TestConnectionStore:
    def test_without_a_saved_connection_it_is_the_env_with_each_fields_source(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        current = connection.effective()
        assert (current.url, current.api_key, current.model, current.verify_tls, current.template_kwargs,
                current.max_tokens_field, dict(current.extra_body), current.locked, current.enabled) == (
            AI_SERVER, API_KEY, "qwen3-30b", True, False, "max_tokens", {}, False, True,
        )
        assert dict(current.sources) == {
            "url": "env", "api_key": "env", "model": "env", "verify_tls": "default", "template_kwargs": "default",
            "max_tokens_field": "default", "extra_body": "default", "stream": "default",
        }
        assert API_KEY not in repr(current)
        assert not _connection_file(connection_on).exists(), "reading never writes"

    def test_a_saved_field_overrides_its_env_value_and_an_unsaved_one_follows_the_env(self, connection_on, monkeypatch):
        from src.assistant import connection

        saved = connection.save({"url": OTHER, "verify_tls": False, "extra_body": {"top_k": 20}})
        assert (saved.url, saved.verify_tls, dict(saved.extra_body), saved.model) == (OTHER, False, {"top_k": 20}, "qwen3-30b")
        assert [saved.sources[name] for name in ("url", "verify_tls", "extra_body", "model", "api_key")] == [
            "saved", "saved", "saved", "env", "default",
        ]
        monkeypatch.setattr(connection_on, "os_assistant_model", "from-env")
        monkeypatch.setattr(connection_on, "os_assistant_url", "http://ignored.local/v1")
        current = connection.effective()
        assert (current.url, current.model) == (OTHER, "from-env")
        assert json.loads(_connection_file(connection_on).read_text(encoding="utf-8")) == {
            "version": 1, "connection": {"url": OTHER, "verify_tls": False, "extra_body": {"top_k": 20}},
        }

    def test_a_save_merges_and_null_returns_a_field_to_the_env(self, connection_on):
        from src.assistant import connection

        connection.save({"url": OTHER})
        both = connection.save({"model": "your-model"})
        assert (both.url, both.model) == (OTHER, "your-model")
        back = connection.save({"url": None})
        assert (back.url, back.sources["url"], back.model, back.sources["model"]) == (AI_SERVER, "env", "your-model", "saved")

    def test_the_api_key_is_write_only(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        view = connection.save({"api_key": NEW_KEY}).view()
        assert "api_key" not in view and view["api_key_set"] is True and view["sources"]["api_key"] == "saved"
        assert NEW_KEY not in json.dumps(view) and API_KEY not in json.dumps(view)
        assert connection.effective().api_key == NEW_KEY and NEW_KEY not in repr(connection.effective())
        # For the same server only; TestConnectionKeyOrigin has the rest.
        assert connection.save({"url": AI_SERVER_PATH}).api_key == NEW_KEY, "leaving it out keeps the stored key"
        cleared = connection.save({"api_key": ""})
        assert (cleared.api_key, cleared.view()["api_key_set"], cleared.sources["api_key"]) == ("", False, "saved")
        assert connection.save({"api_key": None}).api_key == API_KEY, "null goes back to OS_ASSISTANT_API_KEY"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("url", "ai-server.local/v1"), ("url", "ftp://ai-server.local/v1"), ("url", "http:///v1"),
            ("url", "http://user:pw@ai-server.local/v1"), ("url", "http://ai-server.local/v1?x=1"),
            ("url", "http://ai-server.local/v1#x"), ("url", "http://ai-server.local:99999/v1"),
            ("url", "http://ai-server local/v1"), ("url", "http://ai-server.local/" + "v" * 2000), ("url", 5),
            ("api_key", 5), ("api_key", "sk-a\nb"), ("api_key", "sk-é"), ("api_key", "k" * 4097),
            ("model", 5), ("model", "m" * 201),
            ("verify_tls", "false"), ("verify_tls", 0), ("template_kwargs", "true"),
            ("max_tokens_field", "max_new_tokens"), ("max_tokens_field", 5),
            ("extra_body", "[1]"), ("extra_body", "not json"), ("extra_body", ["top_k"]), ("extra_body", 5),
            ("extra_body", {"stream": True}), ("extra_body", '{"messages": []}'), ("extra_body", {"pad": "x" * 20000}),
            ("surprise", 1),
        ],
    )
    def test_every_fields_validation(self, connection_on, field, value):
        from src.assistant import connection

        error = _connection_error(connection.save, {field: value})
        assert error.field == field and field in error.message
        assert _connection_error(connection.candidate, {field: value}).field == field
        assert not _connection_file(connection_on).exists()

    def test_a_body_that_is_not_an_object_is_refused(self, connection_on):
        from src.assistant import connection

        assert _connection_error(connection.save, ["url"]).field == "body"

    def test_values_are_normalised_and_an_empty_url_turns_the_assistant_off(self, connection_on):
        from src.assistant import connection

        saved = connection.save({
            "url": "  http://other.local:9000/v1/  ", "model": "  your-model  ", "extra_body": '{"top_k": 20}',
            "api_key": "  sk-x  ", "enabled": False, "locked": True, "api_key_set": False, "sources": {},
        })
        assert (saved.url, saved.model, dict(saved.extra_body), saved.api_key, saved.locked) == (
            OTHER, "your-model", {"top_k": 20}, "sk-x", False,
        ), "the read-only fields of an answer are ignored"
        assert dict(connection.save({"extra_body": ""}).extra_body) == {}
        off = connection.save({"url": ""})
        assert (off.url, off.enabled, off.sources["url"]) == ("", False, "saved")
        assert connection_on.assistant_enabled is False

    def test_the_file_is_written_only_on_save_atomically_and_forget_removes_it(self, connection_on, monkeypatch):
        from src.assistant import connection

        path = _connection_file(connection_on)
        connection.effective().view()
        assert connection_on.assistant_enabled is True
        assert not path.exists()
        connection.save({"url": OTHER})
        assert path.parent == _presets_file(connection_on).parent, "next to assistant-presets.json"
        before = path.read_bytes()
        real_replace = connection.os.replace

        def failing_replace(source, target):
            raise OSError("disk full")

        monkeypatch.setattr(connection.os, "replace", failing_replace)
        with pytest.raises(OSError):
            connection.save({"model": "m2"})
        monkeypatch.setattr(connection.os, "replace", real_replace)
        assert path.read_bytes() == before, "the old file is intact"
        assert sorted(p.name for p in path.parent.iterdir() if p.name != "studio.db") == ["assistant-connection.json"]
        assert connection.effective().model == "qwen3-30b"
        forgotten = connection.forget()
        assert not path.exists() and (forgotten.url, forgotten.sources["url"]) == (AI_SERVER, "env")
        assert connection.forget().url == AI_SERVER, "forgetting nothing is fine"

    @pytest.mark.parametrize(
        "content",
        ["not json", '["a list"]', '{"connection": "nope"}', b"\xff\xfe\x00broken"],
        ids=["not-json", "a-list", "connection-not-an-object", "not-utf8"],
    )
    def test_an_unreadable_file_means_env_only_with_a_warning(self, connection_on, caplog, content):
        from src.assistant import connection

        path = _connection_file(connection_on)
        path.parent.mkdir(parents=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        with caplog.at_level(logging.WARNING):
            current = connection.effective()
        assert (current.url, current.sources["url"], current.enabled) == (AI_SERVER, "env", True)
        assert any("assistant-connection.json" in record.getMessage() for record in caplog.records)

    def test_a_bad_field_in_the_file_is_skipped_and_the_rest_kept(self, connection_on, caplog):
        from src.assistant import connection

        path = _connection_file(connection_on)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "connection": {
            "url": "http://user:pw@evil.local/v1", "model": "your-model", "surprise": 1, "verify_tls": "no",
        }}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            current = connection.effective()
        assert (current.url, current.model, current.verify_tls) == (AI_SERVER, "your-model", True)
        assert caplog.records and "pw" not in caplog.text

    def test_the_lock_ignores_the_saved_file_and_refuses_every_change(self, connection_on, monkeypatch):
        from src.assistant import connection
        from src.assistant.connection import ConnectionLockedError

        connection.save({"url": OTHER, "model": "your-model"})
        path = _connection_file(connection_on)
        before = path.read_bytes()
        monkeypatch.setattr(connection_on, "os_assistant_connection_locked", True)
        current = connection.effective()
        assert (current.url, current.model, current.locked, current.view()["locked"]) == (AI_SERVER, "qwen3-30b", True, True)
        assert "saved" not in current.sources.values()
        for call, args in ((connection.save, ({"model": "x"},)), (connection.forget, ()),
                           (connection.candidate, ({"url": OTHER},))):
            with pytest.raises(ConnectionLockedError):
                call(*args)
        assert connection.candidate({}).url == AI_SERVER
        assert connection.candidate({"locked": False, "sources": {}, "api_key_set": True}).url == AI_SERVER
        assert path.read_bytes() == before
        assert connection_on.assistant_enabled is True
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        assert connection_on.assistant_enabled is False, "while locked, a saved URL doesn't turn it on"

    def test_a_candidate_applies_the_save_rules_without_saving(self, connection_on):
        from src.assistant import connection

        connection.save({"api_key": "sk-saved"})
        path = _connection_file(connection_on)
        before = path.read_bytes()
        trial = connection.candidate({"url": AI_SERVER_PATH, "verify_tls": False, "api_key_set": False})
        assert (trial.url, trial.api_key, trial.verify_tls) == (AI_SERVER_PATH, "sk-saved", False)
        assert connection.candidate({"url": OTHER}).api_key == "", "another server gets no stored key"
        assert connection.candidate({"api_key": ""}).api_key == ""
        assert connection.candidate({"api_key": None}).api_key == "", "null: OS_ASSISTANT_API_KEY, empty here"
        assert path.read_bytes() == before and connection.effective().url == AI_SERVER

    def test_a_saved_url_turns_the_assistant_on_with_nothing_in_env(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        assert connection_on.assistant_enabled is False
        connection.save({"url": OTHER})
        assert connection_on.assistant_enabled is True
        connection.forget()
        assert connection_on.assistant_enabled is False

    def test_the_connections_model_is_the_presets_default_model(self, connection_on):
        from src.assistant import connection, presets

        connection.save({"model": "your-model"})
        assert presets.effective("voice").model == "your-model"
        book = presets.get_preset("book")
        assert book["model"] == "your-model" and "model" in book["inherited"]
        presets.replace_preset("voice", {"model": "own-model"})
        assert presets.effective("voice").model == "own-model", "a preset's own model still wins"


class TestConnectionGuards:
    def test_the_body_cap_is_8_mib(self):
        from src.assistant import client as client_module

        assert client_module.MAX_REPLY_BYTES == 8 * 1024 * 1024

    async def test_a_reply_is_read_in_a_stream_and_refused_past_the_cap(self, monkeypatch):
        from src.assistant import client as client_module
        from src.assistant.errors import AssistantError

        monkeypatch.setattr(client_module, "MAX_REPLY_BYTES", 4096)
        stream = _RecordingStream([b"x" * 1024] * 100)
        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(lambda _r: httpx.Response(200, stream=stream)))
        assert caught.value.code == "assistant_bad_response"
        assert stream.chunks_read <= 5, "reading stops at the cap"

    async def test_a_reply_that_is_not_json_is_never_echoed(self):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(lambda _r: httpx.Response(200, text="<html>internal admin page</html>")))
        assert caught.value.code == "assistant_bad_response" and "internal admin" not in caught.value.message

    async def test_a_redirect_is_never_followed_or_shown(self):
        from src.assistant.errors import AssistantError

        server = _Server(lambda _r: httpx.Response(307, headers={"location": "http://elsewhere.local/v1/chat/completions"}))
        with pytest.raises(AssistantError) as caught:
            await _reply(server)
        assert caught.value.code == "assistant_http_error" and len(server.requests) == 1
        assert "elsewhere" not in caught.value.message

    @pytest.mark.parametrize(
        "url",
        ["http://user:pw@ai-server.local/v1", "ftp://ai-server.local/v1", "file:///etc/passwd", "ai-server.local/v1",
         "http://ai-server.local/v1?x=1", *HTTPX_REFUSED_URLS],
    )
    def test_the_client_refuses_a_url_the_guards_refuse(self, url):
        from src.assistant.client import AssistantClient

        with pytest.raises(ValueError):
            AssistantClient(url, transport=httpx.MockTransport(_Server()))

    @pytest.mark.parametrize("url", HTTPX_REFUSED_URLS, ids=HTTPX_REFUSED_IDS)
    @pytest.mark.parametrize("call", ["reply", "list_models"])
    async def test_a_url_httpx_refuses_past_the_guard_is_assistant_unreachable(self, monkeypatch, call, url):
        # A client built past the guard (patched here) gets httpx.InvalidURL, which
        # isn't an HTTPError, while httpx builds the request; reply() and list_models() still raise the code their
        # docstrings promise. The transport, which fails the test if reached, never is.
        from src.assistant import client as client_module
        from src.assistant.errors import AssistantError

        monkeypatch.setattr(client_module, "check_assistant_url", lambda value, _name: value.strip().rstrip("/"))
        client = client_module.AssistantClient(url, transport=httpx.MockTransport(_Servers()))
        try:
            with pytest.raises(AssistantError) as caught:
                if call == "reply":
                    await client.reply(MESSAGES, _settings(timeout_s=5))
                else:
                    await client.list_models(timeout=5)
        finally:
            await client.aclose()
        assert caught.value.code == "assistant_unreachable", caught.value.message

    async def test_the_client_built_at_startup_uses_the_saved_connection(self, connection_on, monkeypatch):
        from src.assistant import connection
        from src.assistant.client import AssistantClient

        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        connection.save({"url": OTHER, "api_key": NEW_KEY, "max_tokens_field": "max_completion_tokens"})
        assert connection_on.assistant_enabled is True, "so lifespan() builds a client"
        server = _Server()
        client = AssistantClient.from_settings(connection_on, transport=httpx.MockTransport(server))
        try:
            await client.reply(MESSAGES, _settings(max_tokens=64))
        finally:
            await client.aclose()
        assert str(server.requests[0].url) == f"{OTHER}/chat/completions"
        assert server.requests[0].headers["authorization"] == f"Bearer {NEW_KEY}"
        assert server.body["max_completion_tokens"] == 64 and NEW_KEY not in repr(client)


class TestClientRebuild:
    async def test_a_rebuild_replaces_the_shared_client_and_closes_an_idle_old_one(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        old = _client(_Server())
        assistant.set_client(old)
        connection.save({"url": OTHER})
        new = await assistant.rebuild_client()
        assert assistant.get_client() is new and new is not old and new.host == "other.local:9000"
        assert old.is_closed and old.retired and not new.is_closed
        other = servers.add(OTHER)
        await assistant.get_client().reply(MESSAGES, _settings())
        assert len(other.requests) == 1

    async def test_with_no_url_a_rebuild_leaves_no_client(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        old = _client(_Server())
        assistant.set_client(old)
        connection.save({"url": ""})
        assert await assistant.rebuild_client() is None
        assert assistant.get_client() is None and old.is_closed

    async def test_a_request_in_flight_finishes_on_the_old_client_before_it_closes(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        release = asyncio.Event()

        async def slow(_request):
            await release.wait()
            return _answer(content="Finished on the old client.")

        old = _client(_Server(slow))
        assistant.set_client(old)
        turn = asyncio.create_task(old.reply(MESSAGES, _settings(timeout_s=5)))
        await _until(lambda: old.busy)
        connection.save({"url": OTHER})
        new = await assistant.rebuild_client()
        closers = set(assistant._retiring)
        assert assistant.get_client() is new and len(closers) == 1
        assert not old.is_closed, "a save during a voice turn doesn't cut it off"
        release.set()
        assert (await asyncio.wait_for(turn, 2)).text == "Finished on the old client."
        await asyncio.wait_for(asyncio.gather(*closers), 2)
        assert old.is_closed and not new.is_closed

    async def test_a_hanging_request_ends_at_its_timeout_and_then_the_old_client_closes(self, connection_on, servers, no_shared_client):
        from src.assistant import connection
        from src.assistant.errors import AssistantError

        assistant = no_shared_client
        loop = asyncio.get_running_loop()

        async def hang(_request):
            await asyncio.sleep(5)
            return _answer()

        old = _client(_Server(hang))
        assistant.set_client(old)
        started = loop.time()
        turn = asyncio.create_task(old.reply(MESSAGES, _settings(timeout_s=0.2)))
        await _until(lambda: old.busy)
        connection.save({"url": OTHER})
        await assistant.rebuild_client()
        closers = set(assistant._retiring)
        with pytest.raises(AssistantError) as caught:
            await asyncio.wait_for(turn, 2)
        assert caught.value.code == "assistant_timeout"
        await asyncio.wait_for(asyncio.gather(*closers), 2)
        assert old.is_closed and loop.time() - started < 1.5

    async def test_the_closer_is_bounded_by_the_requests_deadline_even_if_one_never_ends(self, monkeypatch):
        from src.assistant import client as client_module

        monkeypatch.setattr(client_module, "CLOSE_GRACE_S", 0.05)
        old = _client(_Server())
        loop = asyncio.get_running_loop()
        old._deadlines[object()] = loop.time() + 0.1  # a request whose own timeout never landed
        started = loop.time()
        await asyncio.wait_for(old.aclose_when_idle(), 2)
        assert old.is_closed and 0.1 <= loop.time() - started < 1.0

    async def test_a_replaced_client_sends_a_new_request_to_the_current_one(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client
        old_server = _Server()
        old = _client(old_server)
        assistant.set_client(old)
        connection.save({"url": OTHER})
        other = servers.add(OTHER, lambda _r: _answer(content="From the new server."))
        await assistant.rebuild_client()
        # A realtime session opened before the save still holds the old client.
        reply = await old.reply(MESSAGES, _settings())
        assert reply.text == "From the new server." and len(other.requests) == 1 and old_server.requests == []

    async def test_a_replaced_client_answers_unreachable_once_the_connection_is_turned_off(self, connection_on, servers, no_shared_client):
        from src.assistant import connection
        from src.assistant.errors import AssistantError

        assistant = no_shared_client
        old_server = _Server()
        old = _client(old_server)
        assistant.set_client(old)
        connection.save({"url": ""})
        await assistant.rebuild_client()
        with pytest.raises(AssistantError) as caught:
            await old.reply(MESSAGES, _settings())
        assert caught.value.code == "assistant_unreachable" and old_server.requests == []

    async def test_shutdown_also_closes_a_replaced_client_still_finishing_a_request(self, connection_on, servers, no_shared_client):
        from src.assistant import connection

        assistant = no_shared_client

        async def hang(_request):
            await asyncio.sleep(5)
            return _answer()

        old = _client(_Server(hang))
        assistant.set_client(old)
        turn = asyncio.create_task(old.reply(MESSAGES, _settings(timeout_s=10)))
        await _until(lambda: old.busy)
        connection.save({"url": OTHER})
        new = await assistant.rebuild_client()
        await asyncio.wait_for(assistant.close_client(), 2)
        assert old.is_closed and new.is_closed and assistant.get_client() is None
        turn.cancel()
        await asyncio.gather(turn, return_exceptions=True)


CONNECTION = "/api/assistant/connection"
CHECK = "/api/assistant/connection/check"


@pytest.fixture
def connection_api(api, connection_on, servers):
    """The configured app (no lifespan) with the assistant on from .env and the MockTransport servers."""
    return api


class TestConnectionApi:
    def test_get_shows_each_effective_value_and_its_source_but_never_the_key(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        monkeypatch.setattr(connection_on, "os_assistant_extra_body", '{"top_k": 20}')
        response = connection_api.get(CONNECTION)
        assert response.status_code == 200
        assert response.json() == {
            "enabled": True, "locked": False, "url": AI_SERVER, "model": "qwen3-30b", "api_key_set": True,
            "api_key_tls_checked": True,  # read-only, never the key
            "verify_tls": True, "template_kwargs": False, "max_tokens_field": "max_tokens", "extra_body": {"top_k": 20},
            "stream": True,
            "sources": {
                "url": "env", "api_key": "env", "model": "env", "verify_tls": "default", "template_kwargs": "default",
                "max_tokens_field": "default", "extra_body": "env", "stream": "default",
            },
        }
        assert API_KEY not in response.text
        assert not _connection_file(connection_on).exists()

    def test_put_saves_the_given_fields_and_rebuilds_the_shared_client(self, connection_api, servers, no_shared_client):
        old = _client(_Server())
        no_shared_client.set_client(old)
        other = servers.add(OTHER, lambda _r: _answer(content="From the other server."))
        response = connection_api.put(CONNECTION, json={
            "url": OTHER, "api_key": NEW_KEY, "model": "your-model", "template_kwargs": True,
        })
        assert response.status_code == 200, response.text
        view = response.json()
        assert (view["url"], view["model"], view["api_key_set"], view["template_kwargs"], view["enabled"]) == (
            OTHER, "your-model", True, True, True,
        )
        assert {name: view["sources"][name] for name in ("url", "api_key", "model", "template_kwargs", "verify_tls")} == {
            "url": "saved", "api_key": "saved", "model": "saved", "template_kwargs": "saved", "verify_tls": "default",
        }
        assert NEW_KEY not in response.text
        new = no_shared_client.get_client()
        assert new is not old and old.is_closed and new.host == "other.local:9000"
        tested = connection_api.post("/api/assistant/test", json={"overrides": {"reasoning_effort": "low"}})
        assert tested.status_code == 200 and tested.json()["reply"] == "From the other server."
        assert other.requests[0].headers["authorization"] == f"Bearer {NEW_KEY}"
        assert other.body["model"] == "your-model" and other.body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert connection_api.get(CONNECTION).json() == view

    @pytest.mark.parametrize(
        ("field", "value"),
        [("url", "http://user:pw@other.local/v1"), ("url", "gopher://other.local/v1"), ("api_key", 5),
         ("extra_body", "[1]"), ("max_tokens_field", "tokens"), ("verify_tls", "no"), ("surprise", 1),
         *[("url", url) for url in HTTPX_REFUSED_URLS]],
    )
    def test_put_answers_422_with_the_fields_name_and_saves_nothing(self, connection_api, connection_on, no_shared_client, field, value):
        response = connection_api.put(CONNECTION, json={field: value})
        assert response.status_code == 422, response.text
        assert (_error(response)["code"], _error(response)["param"]) == ("validation_error", field)
        assert "pw" not in response.text
        assert not _connection_file(connection_on).exists() and no_shared_client.get_client() is None

    def test_a_put_body_that_is_not_an_object_gets_422(self, connection_api):
        response = connection_api.put(CONNECTION, json=["url"])
        assert response.status_code == 422 and _error(response)["code"] == "validation_error"

    def test_delete_forgets_the_saved_connection_and_rebuilds_the_client(self, connection_api, connection_on, no_shared_client):
        assert connection_api.put(CONNECTION, json={"url": OTHER}).status_code == 200
        after_put = no_shared_client.get_client()
        response = connection_api.delete(CONNECTION)
        assert response.status_code == 200
        assert (response.json()["url"], response.json()["sources"]["url"]) == (AI_SERVER, "env")
        assert not _connection_file(connection_on).exists()
        client = no_shared_client.get_client()
        assert client is not after_put and after_put.is_closed and client.host == "ai-server.local:8000"

    def test_the_routes_work_while_the_assistant_is_off_and_a_saved_url_turns_it_on(self, connection_api, connection_on, monkeypatch, no_shared_client):
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        assert connection_api.get("/api/assistant").json()["enabled"] is False
        assert connection_api.get("/api/assistant/presets").status_code == 404
        assert connection_api.get(CONNECTION).json()["enabled"] is False
        response = connection_api.put(CONNECTION, json={"url": OTHER})
        assert response.status_code == 200 and response.json()["enabled"] is True
        status = connection_api.get("/api/assistant").json()
        assert (status["enabled"], status["host"], status["sources"]["OS_ASSISTANT_URL"]) == (True, "other.local:9000", "saved")
        assert status["server_settings"]["OS_ASSISTANT_URL"] is True and "/v1" not in json.dumps(status)
        assert connection_api.get("/api/assistant/presets").status_code == 200
        assert no_shared_client.get_client().host == "other.local:9000"
        assert connection_api.put(CONNECTION, json={"url": ""}).json()["enabled"] is False
        assert connection_api.get("/api/assistant/presets").status_code == 404
        assert no_shared_client.get_client() is None

    def test_when_locked_get_says_so_and_changes_answer_403(self, connection_api, connection_on, servers, monkeypatch, no_shared_client):
        assert connection_api.put(CONNECTION, json={"url": OTHER}).status_code == 200
        before = _connection_file(connection_on).read_bytes()
        client_before = no_shared_client.get_client()
        monkeypatch.setattr(connection_on, "os_assistant_connection_locked", True)
        view = connection_api.get(CONNECTION).json()
        assert (view["locked"], view["url"], view["sources"]["url"]) == (True, AI_SERVER, "env")
        assert connection_api.get("/api/assistant").json()["locked"] is True
        for method, path, body in (("PUT", CONNECTION, {"model": "x"}), ("DELETE", CONNECTION, None),
                                   ("POST", CHECK, {"url": OTHER})):
            response = connection_api.request(method, path, json=body)
            assert response.status_code == 403, (method, path)
            assert _error(response)["code"] == "connection_locked"
        assert _connection_file(connection_on).read_bytes() == before
        assert no_shared_client.get_client() is client_before
        servers.add(AI_SERVER, lambda _r: httpx.Response(200, json={"data": [{"id": "qwen3-30b"}]}))
        checked = connection_api.post(CHECK, json={})
        assert checked.status_code == 200 and checked.json()["models"] == ["qwen3-30b"], "the .env connection can be checked"

    def test_the_connection_routes_need_the_api_key_when_one_is_set(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_api_key", "open-speech-key")
        for method, path, body in (("GET", CONNECTION, None), ("PUT", CONNECTION, {"url": OTHER}),
                                   ("DELETE", CONNECTION, None), ("POST", CHECK, {"url": OTHER})):
            assert connection_api.request(method, path, json=body).status_code == 401, (method, path)
        assert not _connection_file(connection_on).exists()
        authorized = connection_api.get(CONNECTION, headers={"Authorization": "Bearer open-speech-key"})
        assert authorized.status_code == 200


class TestConnectionCheck:
    def test_check_lists_the_models_with_one_get_and_saves_nothing(self, connection_api, connection_on, servers, no_shared_client):
        other = servers.add(OTHER, lambda _r: httpx.Response(200, json={"object": "list", "data": [
            {"id": "your-model", "object": "model", "owned_by": "secret-owner"}, {"id": "qwen3-30b"},
        ]}))
        response = connection_api.post(CHECK, json={"url": OTHER, "api_key": "sk-trial"})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert {key: answer[key] for key in ("reachable", "status", "models", "listed")} == {
            "reachable": True, "status": 200, "models": ["your-model", "qwen3-30b"], "listed": True,
        }
        assert isinstance(answer["elapsed_ms"], int) and answer["elapsed_ms"] >= 0
        assert [(request.method, str(request.url)) for request in other.requests] == [("GET", f"{OTHER}/models")]
        assert other.requests[0].headers["authorization"] == "Bearer sk-trial" and other.requests[0].content == b""
        assert "secret-owner" not in response.text and "sk-trial" not in response.text
        assert not _connection_file(connection_on).exists() and no_shared_client.get_client() is None

    def test_check_keeps_the_stored_key_unless_the_candidate_gives_one(self, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"api_key": "sk-saved"}).status_code == 200
        ai_server = servers.add(AI_SERVER, lambda _r: httpx.Response(200, json={"data": []}))
        answer = connection_api.post(CHECK, json={}).json()
        assert (answer["models"], answer["listed"]) == ([], True)
        assert ai_server.requests[-1].headers["authorization"] == "Bearer sk-saved"
        assert connection_api.post(CHECK, json={"api_key": ""}).status_code == 200
        assert "authorization" not in ai_server.requests[-1].headers

    @pytest.mark.parametrize(
        ("answer", "models", "listed"),
        [
            (lambda _r: httpx.Response(200, json={"data": [
                {"id": "ok"}, {"name": "secret-a"}, 5, {"id": 7}, {"id": "x" * 201}, {"id": "ok"}, {"id": " "},
            ]}), ["ok"], True),
            (lambda _r: httpx.Response(200, json={"data": "secret-b"}), [], False),
            (lambda _r: httpx.Response(200, json=["secret-c"]), [], False),
            (lambda _r: httpx.Response(200, json={"error": "secret-d"}), [], False),
            (lambda _r: httpx.Response(200, text="<html>secret-e</html>"), [], False),
            (lambda _r: httpx.Response(200, content=b"\xff\xfesecret-f"), [], False),
        ],
        ids=["mixed-list", "data-not-a-list", "a-bare-list", "an-error-object", "html", "not-utf8"],
    )
    def test_check_returns_only_model_ids_from_an_openai_style_list_never_other_body_text(self, connection_api, servers, answer, models, listed):
        servers.add(OTHER, answer)
        response = connection_api.post(CHECK, json={"url": OTHER})
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["reachable"], body["status"], body["models"], body["listed"]) == (True, 200, models, listed)
        assert "secret" not in response.text

    @pytest.mark.parametrize(
        ("status", "headers"),
        [(401, {}), (404, {}), (500, {}), (302, {"location": "http://elsewhere.local/v1/models"})],
    )
    def test_check_reports_another_status_without_its_body_and_follows_no_redirect(self, connection_api, servers, status, headers):
        other = servers.add(OTHER, lambda _r: httpx.Response(
            status, headers=headers, json={"data": [{"id": "secret-model"}], "message": "secret"},
        ))
        response = connection_api.post(CHECK, json={"url": OTHER})
        assert response.status_code == 200
        body = response.json()
        assert (body["reachable"], body["status"], body["models"], body["listed"]) == (True, status, [], False)
        assert len(other.requests) == 1 and "secret" not in response.text and "elsewhere" not in response.text

    @pytest.mark.parametrize(
        ("answer", "code", "status"),
        [
            (_fail(httpx.ConnectError, "[Errno 111] Connection refused"), "assistant_unreachable", 502),
            (_fail(httpx.ConnectError, "[SSL: CERTIFICATE_VERIFY_FAILED]"), "assistant_unreachable", 502),
            (_fail(httpx.ConnectTimeout, "timed out"), "assistant_timeout", 504),
        ],
        ids=["refused", "tls", "connect-timeout"],
    )
    def test_check_answers_the_error_code_when_the_server_cant_be_reached(self, connection_api, servers, answer, code, status):
        servers.add(OTHER, answer)
        response = connection_api.post(CHECK, json={"url": OTHER})
        assert response.status_code == status
        body = response.json()
        assert (body["error"]["code"], body["reachable"]) == (code, False) and isinstance(body["elapsed_ms"], int)

    def test_check_gives_up_after_its_timeout(self, connection_api, servers, monkeypatch):
        from src.assistant import client as client_module

        monkeypatch.setattr(client_module, "CHECK_TIMEOUT_S", 0.05)

        async def slow(_request):
            await asyncio.sleep(1.0)
            return httpx.Response(200, json={"data": []})

        servers.add(OTHER, slow)
        response = connection_api.post(CHECK, json={"url": OTHER})
        assert response.status_code == 504 and _error(response)["code"] == "assistant_timeout"

    def test_check_reads_the_list_in_a_stream_and_stops_at_the_cap(self, connection_api, servers, monkeypatch):
        from src.assistant import client as client_module

        monkeypatch.setattr(client_module, "MAX_REPLY_BYTES", 4096)
        stream = _RecordingStream([b'{"data": [' + b'{"id": "m"},' * 100] + [b" " * 1024] * 50)
        servers.add(OTHER, lambda _r: httpx.Response(200, stream=stream))
        response = connection_api.post(CHECK, json={"url": OTHER})
        assert response.status_code == 200
        assert (response.json()["models"], response.json()["listed"]) == ([], False)
        assert stream.chunks_read <= 6

    def test_check_needs_a_url_and_valid_fields(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        for body, param in (({}, "url"), ({"url": "http://user:pw@other.local/v1"}, "url"),
                            ({"url": OTHER, "verify_tls": "no"}, "verify_tls"), ({"what": 1}, "what")):
            response = connection_api.post(CHECK, json=body)
            assert response.status_code == 422, (body, response.text)
            assert (_error(response)["code"], _error(response)["param"]) == ("validation_error", param)

    @pytest.mark.parametrize("url", HTTPX_REFUSED_URLS, ids=HTTPX_REFUSED_IDS)
    def test_check_refuses_a_url_httpx_cant_parse_with_422_naming_url(self, connection_api, url):
        # Refused by the guard, never the catch-all 500 internal_error.
        response = connection_api.post(CHECK, json={"url": url})
        assert response.status_code == 422, response.text
        assert (_error(response)["code"], _error(response)["param"]) == ("validation_error", "url")

    def test_check_with_an_empty_body_checks_the_connection_in_use(self, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"url": OTHER}).status_code == 200
        other = servers.add(OTHER, lambda _r: httpx.Response(200, json={"data": [{"id": "your-model"}]}))
        assert connection_api.post(CHECK, json={}).json()["models"] == ["your-model"]
        assert len(other.requests) == 1


# ── The example URL ──────────────────────────────────────────────────────────
# One example AI server URL everywhere, with a literal host. The Settings hints and
# tests/js/settings/assistant-connection.test.js copy these messages.
EXAMPLE_URL = "http://ai-server.lan:8080/v1"


class TestTheExampleUrl:
    def test_the_settings_and_the_url_check_give_the_one_example(self, clean_assistant_env, monkeypatch):
        from src.config import ASSISTANT_URL_MAX_CHARS, check_assistant_url

        monkeypatch.setenv("OS_ASSISTANT_URL", "ftp://ai-server.local/v1")
        with pytest.raises(ValidationError) as refused:
            Settings(_env_file=None)
        assert f"OS_ASSISTANT_URL must be an http:// or https:// base URL such as {EXAMPLE_URL}" in str(refused.value)
        with pytest.raises(ValueError) as caught:
            check_assistant_url("http://ai-server.local/" + "v" * ASSISTANT_URL_MAX_CHARS, "url")
        assert str(caught.value) == (
            f"url must be one base URL of up to {ASSISTANT_URL_MAX_CHARS} characters, such as {EXAMPLE_URL}"
        )

    def test_the_connection_routes_give_the_one_example(self, connection_api, connection_on, monkeypatch):
        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        answers = {
            "save, not http": connection_api.put(CONNECTION, json={"url": "ftp://other.local/v1"}),
            "save, not text": connection_api.put(CONNECTION, json={"url": 5}),
            "check, no url": connection_api.post(CHECK, json={}),
        }
        assert {name: (r.status_code, _error(r)["param"], _error(r)["message"]) for name, r in answers.items()} == {
            "save, not http": (422, "url", f"url must be an http:// or https:// base URL such as {EXAMPLE_URL}"),
            "save, not text": (422, "url", f"url must be text: the AI server's base URL, such as {EXAMPLE_URL}"),
            "check, no url": (422, "url", f"url is needed: the AI server's base URL, such as {EXAMPLE_URL}"),
        }


# ── The key stays with its server ────────────────────────────────────────────
# The API key is tied to the origin (scheme, host and port)
# of the URL it was set with. A saved URL or a check with another origin gets no key unless the same request gives
# one, and a PUT that moves to another origin without api_key clears the stored key. The .env key goes only to
# OS_ASSISTANT_URL's origin. A change of path keeps the key.


def _authorization(server: _Server) -> str | None:
    return server.requests[-1].headers.get("authorization")


class TestConnectionKeyOrigin:
    def test_a_put_to_another_origin_without_the_key_clears_the_stored_key(self, connection_api, connection_on, servers):
        assert connection_api.put(CONNECTION, json={"api_key": "sk-saved"}).json()["api_key_set"] is True
        view = connection_api.put(CONNECTION, json={"url": OTHER}).json()
        assert (view["url"], view["api_key_set"], view["sources"]["api_key"]) == (OTHER, False, "default")
        other = servers.add(OTHER)
        assert connection_api.post("/api/assistant/test", json={}).status_code == 200
        assert _authorization(other) is None, "the rebuilt client sends no key to the new server"
        assert "sk-saved" not in _connection_file(connection_on).read_text(encoding="utf-8")
        back = connection_api.put(CONNECTION, json={"url": AI_SERVER}).json()
        assert back["api_key_set"] is False, "cleared, not only withheld: the old server needs the key again"

    def test_a_put_that_gives_a_key_with_the_new_url_keeps_that_key(self, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"api_key": "sk-saved"}).status_code == 200
        view = connection_api.put(CONNECTION, json={"url": OTHER, "api_key": NEW_KEY}).json()
        assert (view["api_key_set"], view["sources"]["api_key"]) == (True, "saved")
        other = servers.add(OTHER)
        assert connection_api.post("/api/assistant/test", json={}).status_code == 200
        assert _authorization(other) == f"Bearer {NEW_KEY}"

    @pytest.mark.parametrize(
        ("first", "second", "kept"),
        [
            (AI_SERVER, AI_SERVER_PATH, True),
            (AI_SERVER, "HTTP://AI-Server.Local:8000/v2", True),
            ("http://ai-server.local/v1", "http://ai-server.local:80/api", True),
            ("https://ai-server.local/v1", "https://ai-server.local:443/v1", True),
            (AI_SERVER, "https://ai-server.local:8000/v1", False),
            (AI_SERVER, "http://ai-server.local:8001/v1", False),
            (AI_SERVER, "http://ai-server.local.example:8000/v1", False),
            (AI_SERVER, OTHER, False),
        ],
        ids=["path", "case", "http-default-port", "https-default-port", "scheme", "port", "host", "other"],
    )
    def test_the_key_stays_with_the_origin_it_was_set_for(self, connection_on, first, second, kept):
        from src.assistant import connection

        connection.save({"url": first, "api_key": "sk-saved"})
        moved = connection.save({"url": second})
        assert (moved.api_key, moved.view()["api_key_set"]) == (("sk-saved", True) if kept else ("", False))

    def test_the_env_key_goes_only_to_the_env_urls_origin(self, connection_api, connection_on, monkeypatch, servers):
        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        assert connection_api.get(CONNECTION).json()["api_key_set"] is True
        view = connection_api.put(CONNECTION, json={"url": OTHER}).json()
        assert (view["api_key_set"], view["sources"]["api_key"]) == (False, "default")
        status = connection_api.get("/api/assistant").json()
        assert (status["server_settings"]["OS_ASSISTANT_API_KEY"], status["sources"]["OS_ASSISTANT_API_KEY"]) == (
            False, "default",
        )
        other = servers.add(OTHER)
        assert connection_api.post("/api/assistant/test", json={}).status_code == 200
        assert _authorization(other) is None
        same_origin = connection_api.put(CONNECTION, json={"url": AI_SERVER_PATH}).json()
        assert (same_origin["api_key_set"], same_origin["sources"]["api_key"]) == (True, "env")
        ai_server = servers.add(AI_SERVER)
        assert connection_api.post("/api/assistant/test", json={}).status_code == 200
        assert _authorization(ai_server) == f"Bearer {API_KEY}"
        assert connection_api.put(CONNECTION, json={"url": None}).json()["api_key_set"] is True

    @pytest.mark.parametrize("stored", [None, "sk-saved"], ids=["env-key", "saved-key"])
    def test_a_check_with_another_origin_sends_no_key_unless_the_body_gives_one(
        self, connection_api, connection_on, monkeypatch, servers, stored
    ):
        monkeypatch.setattr(connection_on, "os_assistant_api_key", API_KEY)
        if stored:
            assert connection_api.put(CONNECTION, json={"api_key": stored}).status_code == 200
        view_before = connection_api.get(CONNECTION).json()
        other = servers.add(OTHER, lambda _r: httpx.Response(200, json={"data": []}))
        ai_server = servers.add(AI_SERVER, lambda _r: httpx.Response(200, json={"data": []}))
        assert connection_api.post(CHECK, json={"url": OTHER}).status_code == 200
        assert _authorization(other) is None
        assert connection_api.post(CHECK, json={"url": OTHER, "api_key": "sk-trial"}).status_code == 200
        assert _authorization(other) == "Bearer sk-trial"
        assert connection_api.post(CHECK, json={"url": AI_SERVER_PATH}).status_code == 200
        assert _authorization(ai_server) == f"Bearer {stored or API_KEY}"
        assert connection_api.get(CONNECTION).json() == view_before, "a check saves nothing"

    def test_the_keys_origin_is_saved_beside_it_and_a_changed_env_url_does_not_get_the_key(self, connection_on, monkeypatch):
        from src.assistant import connection

        connection.save({"api_key": "sk-saved"})  # with .env's URL, AI_SERVER
        stored = json.loads(_connection_file(connection_on).read_text(encoding="utf-8"))["connection"]
        # api_key_tls_checked: whether the certificate check was on.
        assert stored == {"api_key": "sk-saved", "api_key_origin": "http://ai-server.local:8000", "api_key_tls_checked": True}
        monkeypatch.setattr(connection_on, "os_assistant_url", OTHER)  # .env edited and the server restarted
        current = connection.effective()
        assert (current.url, current.api_key, current.view()["api_key_set"], current.sources["api_key"]) == (
            OTHER, "", False, "default",
        )
        monkeypatch.setattr(connection_on, "os_assistant_url", AI_SERVER_PATH)
        assert connection.effective().api_key == "sk-saved"

    def test_a_key_saved_before_origins_were_kept_stays_with_the_url_it_was_saved_with(self, connection_on):
        from src.assistant import connection

        path = _connection_file(connection_on)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "connection": {"url": OTHER, "api_key": "sk-old"}}), encoding="utf-8")
        assert connection.effective().api_key == "sk-old"
        assert connection.save({"model": "your-model"}).api_key == "sk-old"
        assert connection.save({"url": "http://third.local/v1"}).api_key == ""

    def test_a_key_saved_with_no_url_goes_to_no_server(self, connection_on, monkeypatch):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_url", "")
        connection.save({"api_key": "sk-early"})
        assert connection.save({"url": OTHER}).api_key == "", "send the key with the URL, in one save"
        assert connection.save({"url": OTHER, "api_key": "sk-early"}).api_key == "sk-early"

    def test_turning_the_assistant_off_and_on_again_keeps_the_key_for_its_server(self, connection_on):
        from src.assistant import connection

        connection.save({"url": AI_SERVER, "api_key": "sk-saved"})
        off = connection.save({"url": ""})
        assert (off.enabled, off.api_key, off.view()["api_key_set"]) == (False, "", False), "no URL: no key sent"
        assert connection.save({"url": AI_SERVER_PATH}).api_key == "sk-saved"
        connection.save({"url": ""})
        assert connection.save({"url": OTHER}).api_key == "", "off, then another server: the key is cleared"


# ── The key stays with its certificate check ─────────────────────────────────
# Turning the certificate check off on an https URL is a
# downgrade like moving to http, so a key set while the check was on is not sent with it off. A PUT that turns it off
# without api_key clears the saved key, and a /check with verify_tls false sends no stored key, unless the same request
# gives one. The .env key counts as set with OS_ASSISTANT_VERIFY_TLS. Turning the check back on keeps a key, and an
# http URL (no certificate) isn't affected. Every request goes to httpx.MockTransport at the made-up ai.cloud.test.
TLS_URL = "https://ai.cloud.test/v1"
TLS_URL_PATH = "https://ai.cloud.test/api/v1"  # TLS_URL's origin with another path
TLS_HTTP_URL = "http://ai.cloud.test/v1"  # the same host over http
TLS_ENV_KEY = "sk-env-r3s02-key-0001"
TLS_SAVED_KEY = "sk-saved-r3s02-key-0002"
TLS_GIVEN_KEY = "sk-given-r3s02-key-0003"


@pytest.fixture
def https_env(connection_on, monkeypatch):
    """.env: OS_ASSISTANT_URL is an https URL, with OS_ASSISTANT_API_KEY, and the certificate checked."""
    monkeypatch.setattr(connection_on, "os_assistant_url", TLS_URL)
    monkeypatch.setattr(connection_on, "os_assistant_api_key", TLS_ENV_KEY)
    monkeypatch.setattr(connection_on, "os_assistant_verify_tls", True)
    return connection_on


@pytest.fixture
def built(https_env, no_shared_client, servers, monkeypatch):
    """Every Connection given to build_client: the shared client a save rebuilds, and each /check client."""
    connections = []
    inner = no_shared_client.build_client

    def record(conn):
        connections.append(conn)
        return inner(conn)

    monkeypatch.setattr(no_shared_client, "build_client", record)
    return connections


def _sent_key(server: _Server) -> str | None:
    return server.requests[-1].headers.get("authorization") if server.requests else None


def _one_test_call(connection_api, servers, url=TLS_URL) -> _Server:
    """One Settings Test through the shared client, as a voice turn or a book job uses it."""
    server = servers.add(url)
    assert connection_api.post("/api/assistant/test", json={}).status_code == 200
    return server


def _one_check(connection_api, servers, body, url=TLS_URL) -> _Server:
    server = servers.add(url, lambda _r: httpx.Response(200, json={"data": [{"id": "m"}]}))
    response = connection_api.post(CHECK, json=body)
    assert response.status_code == 200, response.text
    return server


class TestConnectionKeyCertificateCheck:
    def test_a_put_turning_the_check_off_withholds_the_env_key(self, https_env, built, connection_api, servers):
        assert connection_api.get(CONNECTION).json()["api_key_set"] is True
        view = connection_api.put(CONNECTION, json={"verify_tls": False}).json()
        server = _one_test_call(connection_api, servers)
        assert (view["api_key_set"], view["sources"]["api_key"], view["verify_tls"]) == (False, "default", False)
        assert (built[-1].verify_tls, _sent_key(server)) == (False, None)

    def test_a_put_turning_the_check_off_clears_the_saved_key(self, https_env, built, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"api_key": TLS_SAVED_KEY}).json()["api_key_set"] is True
        view = connection_api.put(CONNECTION, json={"verify_tls": False}).json()
        server = _one_test_call(connection_api, servers)
        assert (view["api_key_set"], built[-1].verify_tls, _sent_key(server)) == (False, False, None)
        assert TLS_SAVED_KEY not in _connection_file(https_env).read_text(encoding="utf-8"), "cleared, not withheld"
        back_on = connection_api.put(CONNECTION, json={"verify_tls": True}).json()
        assert (back_on["api_key_set"], back_on["sources"]["api_key"]) == (True, "env"), "the .env key is back"

    def test_a_path_change_with_the_check_off_clears_the_saved_key(self, https_env, built, connection_api, servers):
        """The same origin with another path keeps the key, but not with the check going off."""
        assert connection_api.put(CONNECTION, json={"api_key": TLS_SAVED_KEY}).json()["api_key_set"] is True
        view = connection_api.put(CONNECTION, json={"url": TLS_URL_PATH, "verify_tls": False}).json()
        server = _one_test_call(connection_api, servers, TLS_URL_PATH)
        assert (view["api_key_set"], _sent_key(server)) == (False, None)

    @pytest.mark.parametrize("stored", [None, TLS_SAVED_KEY], ids=["env-key", "saved-key"])
    def test_a_check_with_the_check_off_sends_no_stored_key(self, https_env, built, connection_api, servers, stored):
        if stored:
            assert connection_api.put(CONNECTION, json={"api_key": stored}).json()["api_key_set"] is True
        before = connection_api.get(CONNECTION).json()
        server = _one_check(connection_api, servers, {"verify_tls": False})
        assert connection_api.get(CONNECTION).json() == before, "a check saves nothing"
        assert (built[-1].verify_tls, _sent_key(server)) == (False, None)
        checked_on = _one_check(connection_api, servers, {})
        assert _sent_key(checked_on) == f"Bearer {stored or TLS_ENV_KEY}", "with the check on, the key still goes"

    def test_a_scheme_downgrade_on_the_same_host_still_clears_the_env_key(self, https_env, built, connection_api):
        assert connection_api.put(CONNECTION, json={"url": TLS_HTTP_URL}).json()["api_key_set"] is False

    def test_a_key_given_with_the_check_off_is_saved_and_sent(self, https_env, built, connection_api, servers):
        view = connection_api.put(CONNECTION, json={"verify_tls": False, "api_key": TLS_GIVEN_KEY}).json()
        assert view["api_key_set"] is True
        server = _one_test_call(connection_api, servers)
        assert (built[-1].verify_tls, _sent_key(server)) == (False, f"Bearer {TLS_GIVEN_KEY}")
        check = _one_check(connection_api, servers, {"verify_tls": False, "api_key": TLS_GIVEN_KEY})
        assert _sent_key(check) == f"Bearer {TLS_GIVEN_KEY}"

    def test_the_env_pair_with_the_check_off_sends_the_env_key(self, https_env, built, connection_api, servers, monkeypatch):
        """OS_ASSISTANT_VERIFY_TLS=false beside OS_ASSISTANT_API_KEY is the operator's own choice, made in .env."""
        monkeypatch.setattr(https_env, "os_assistant_verify_tls", False)
        view = connection_api.get(CONNECTION).json()
        assert (view["api_key_set"], view["verify_tls"]) == (True, False)
        server = _one_check(connection_api, servers, {})
        assert (built[-1].verify_tls, _sent_key(server)) == (False, f"Bearer {TLS_ENV_KEY}")

    def test_turning_the_check_back_on_keeps_a_key_set_with_it_off(self, https_env, built, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"verify_tls": False, "api_key": TLS_GIVEN_KEY}).json()["api_key_set"]
        view = connection_api.put(CONNECTION, json={"verify_tls": True}).json()
        assert (view["api_key_set"], view["verify_tls"]) == (True, True)
        server = _one_test_call(connection_api, servers)
        assert (built[-1].verify_tls, _sent_key(server)) == (True, f"Bearer {TLS_GIVEN_KEY}")
        off_again = connection_api.put(CONNECTION, json={"verify_tls": False}).json()
        assert off_again["api_key_set"] is True, "it was set with the check off: turning it off again keeps it"

    def test_an_http_url_has_no_certificate_so_the_setting_keeps_the_key(self, https_env, built, connection_api, servers):
        assert connection_api.put(CONNECTION, json={"url": TLS_HTTP_URL, "api_key": TLS_SAVED_KEY}).json()["api_key_set"]
        assert connection_api.put(CONNECTION, json={"verify_tls": False}).json()["api_key_set"] is True
        server = _one_test_call(connection_api, servers, TLS_HTTP_URL)
        assert _sent_key(server) == f"Bearer {TLS_SAVED_KEY}"

    def test_the_certificate_check_setting_reaches_the_ai_client(self, https_env, monkeypatch):
        """Every other test gives the client an httpx.MockTransport, and httpx then
        ignores verify=, so dropping verify_tls from the AI client left the suite green while a self-signed AI server
        was refused. Built as the server builds it, with no transport (.env, then a saved connection through
        build_client); nothing is sent."""
        import ssl

        import src.assistant as assistant
        from src.assistant import connection
        from src.assistant.client import AssistantClient

        def context(client) -> ssl.SSLContext:
            return client._http._transport._pool._ssl_context  # httpx 0.28: the connection pool keeps it

        clients = [AssistantClient.from_settings(https_env)]
        try:
            assert (context(clients[-1]).verify_mode, context(clients[-1]).check_hostname) == (ssl.CERT_REQUIRED, True)
            monkeypatch.setattr(https_env, "os_assistant_verify_tls", False)
            clients.append(AssistantClient.from_settings(https_env))
            assert (context(clients[-1]).verify_mode, context(clients[-1]).check_hostname) == (ssl.CERT_NONE, False), (
                "OS_ASSISTANT_VERIFY_TLS=false didn't reach httpx: a self-signed AI server is refused")
            monkeypatch.setattr(https_env, "os_assistant_verify_tls", True)
            saved = connection.save({"verify_tls": False})  # Settings › AI assistant
            clients.append(assistant.build_client(saved))
            assert (context(clients[-1]).verify_mode, context(clients[-1]).check_hostname) == (ssl.CERT_NONE, False), (
                "the saved verify_tls false didn't reach httpx")
        finally:
            for client in clients:
                asyncio.run(client._http.aclose())

    def test_whether_the_check_was_on_is_saved_beside_the_key_shown_read_only_and_the_key_never(
        self, https_env, connection_api,
    ):
        # The view shows the flag, read-only, so the Settings form can tell whether turning the check off will clear
        # the key. A boolean about how the key was saved may be shown; the key stays write-only, and no body sets the
        # flag: sent back with a view it is ignored, as api_key_set is.
        from src.assistant import connection

        connection.save({"api_key": TLS_SAVED_KEY})
        stored = json.loads(_connection_file(https_env).read_text(encoding="utf-8"))["connection"]
        assert stored == {"api_key": TLS_SAVED_KEY, "api_key_origin": "https://ai.cloud.test:443",
                          "api_key_tls_checked": True}
        response = connection_api.get(CONNECTION)
        view = response.json()
        assert view["api_key_tls_checked"] is True and "api_key_tls_checked" not in view["sources"]
        assert TLS_SAVED_KEY not in response.text and "api_key" not in view
        ignored = connection_api.put(CONNECTION, json={**view, "api_key_tls_checked": False})
        assert ignored.status_code == 200, ignored.text
        assert ignored.json()["api_key_tls_checked"] is True
        assert json.loads(_connection_file(https_env).read_text(encoding="utf-8"))["connection"][
            "api_key_tls_checked"] is True, "a body never sets it"

    def test_the_view_tells_a_key_set_with_the_check_off_from_one_set_with_it_on(self, https_env, connection_api):
        # A key saved with the check off, then the check turned on: the next "off" save keeps it, so the view must
        # differ from a key set with the check on.
        set_off = connection_api.put(CONNECTION, json={"verify_tls": False, "api_key": TLS_SAVED_KEY}).json()
        assert (set_off["api_key_set"], set_off["api_key_tls_checked"]) == (True, False)
        turned_on = connection_api.put(CONNECTION, json={"verify_tls": True}).json()
        assert (turned_on["api_key_set"], turned_on["api_key_tls_checked"]) == (True, False)
        assert connection_api.put(CONNECTION, json={"verify_tls": False}).json()["api_key_set"] is True, "kept"

        set_on = connection_api.put(CONNECTION, json={"verify_tls": True, "api_key": TLS_SAVED_KEY}).json()
        assert (set_on["api_key_set"], set_on["api_key_tls_checked"]) == (True, True)
        dropped = connection_api.put(CONNECTION, json={"verify_tls": False}).json()
        assert (dropped["api_key_set"], dropped["api_key_tls_checked"]) == (False, None), "no key: null"

    @pytest.mark.parametrize("verify_tls", [True, False])
    def test_the_env_key_counts_as_set_with_os_assistant_verify_tls(
        self, https_env, connection_api, monkeypatch, verify_tls,
    ):
        # As _resolve treats it.
        monkeypatch.setattr(https_env, "os_assistant_verify_tls", verify_tls)
        view = connection_api.get(CONNECTION).json()
        assert view["sources"]["api_key"] == "env"
        assert (view["api_key_set"], view["api_key_tls_checked"]) == (True, verify_tls)

    @pytest.mark.parametrize("verify_tls", [True, False], ids=["saved-with-the-check-on", "saved-with-it-off"])
    def test_a_key_saved_before_the_flag_was_kept_counts_as_set_with_the_check_then_in_effect(
        self, https_env, verify_tls
    ):
        from src.assistant import connection

        path = _connection_file(https_env)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "connection": {
            "verify_tls": verify_tls, "api_key": "sk-old", "api_key_origin": "https://ai.cloud.test:443",
        }}), encoding="utf-8")
        assert connection.effective().api_key == "sk-old", "a key in use before the fix keeps working"
        assert connection.save({"model": "your-model"}).api_key == "sk-old"
        turned = connection.save({"verify_tls": not verify_tls})
        assert turned.api_key == ("" if verify_tls else "sk-old"), "only turning a check that was on off costs it"

    def test_a_flag_that_isnt_true_or_false_in_the_file_is_skipped_with_a_warning(self, https_env, caplog):
        from src.assistant import connection

        path = _connection_file(https_env)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 1, "connection": {
            "api_key": "sk-old", "api_key_origin": "https://ai.cloud.test:443", "api_key_tls_checked": "yes",
        }}), encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="src.assistant.connection"):
            assert connection.effective().api_key == "sk-old"
        assert "Skipping api_key_tls_checked" in caplog.text and "sk-old" not in caplog.text


# ── The key is never in error text, whatever it ends with ────────────────────
# An OS_ASSISTANT_API_KEY read with a trailing line break or tab (a secret file, a
# YAML block scalar in Compose) made h11 refuse the Authorization header and quote it with {!r}, which escapes the
# line break, so _redact's plain match missed the key and Test, Check, voice turns and book jobs answered it. The
# .env key is now stripped and checked as a saved key is, and the client never puts a LocalProtocolError's text in
# a message and redacts the key's escaped and stripped forms too. See README.md,
# "Security and privacy", and src/assistant/errors.py ("message never holds the API key").
#
# These tests need a real listener on 127.0.0.1: httpx.MockTransport skips httpcore and h11, where the header is
# refused. The listener closes each connection at once, so a request that goes out fails fast; tests/conftest.py
# refuses every other address. The key is made up.

MADE_UP_KEY = "sk-made-up-r3p02-0123456789"
KEY_ENDINGS = {"LF": "\n", "CRLF": "\r\n", "TAB": "\t"}


@pytest.fixture
def hang_up_url():
    """A base URL on 127.0.0.1 whose listener accepts each connection and closes it at once."""
    import socket
    import threading

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.1)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                accepted, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            accepted.close()

    thread = threading.Thread(target=serve, name="r3p02-hang-up", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        stop.set()
        thread.join(timeout=5)
        listener.close()


class TestTheKeyInErrorText:
    @pytest.mark.parametrize("ending", list(KEY_ENDINGS.values()), ids=list(KEY_ENDINGS))
    @pytest.mark.parametrize("call", ["reply", "list_models"])
    async def test_the_client_never_puts_a_key_h11_refuses_in_its_error(self, hang_up_url, call, ending):
        from src.assistant.client import AssistantClient
        from src.assistant.errors import AssistantError

        client = AssistantClient(hang_up_url, api_key=MADE_UP_KEY + ending)
        try:
            with pytest.raises(AssistantError) as caught:
                if call == "reply":
                    await client.reply(MESSAGES, _settings(timeout_s=5))
                else:
                    await client.list_models(timeout=5)
        finally:
            await client.aclose()
        assert caught.value.code == "assistant_unreachable"
        assert MADE_UP_KEY not in caught.value.message and MADE_UP_KEY not in str(caught.value), caught.value.message

    @pytest.mark.parametrize("ending", list(KEY_ENDINGS.values()), ids=list(KEY_ENDINGS))
    async def test_an_error_quoting_the_key_escaped_or_stripped_never_shows_it(self, ending):
        from src.assistant.errors import AssistantError

        key = MADE_UP_KEY + ending
        text = f"as text {key!r}, as bytes {key.encode()!r}, stripped {key.strip()}."
        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(_fail(httpx.ConnectError, text)), api_key=key)
        assert caught.value.code == "assistant_unreachable"
        assert MADE_UP_KEY not in caught.value.message, caught.value.message

    @pytest.mark.parametrize("ending", list(KEY_ENDINGS.values()), ids=list(KEY_ENDINGS))
    def test_the_env_key_is_stripped_as_a_saved_key_is(self, clean_assistant_env, monkeypatch, ending):
        monkeypatch.setenv("OS_ASSISTANT_API_KEY", " " + MADE_UP_KEY + ending)
        assert Settings(_env_file=None).os_assistant_api_key == MADE_UP_KEY

    @pytest.mark.parametrize(
        "key",
        [MADE_UP_KEY + "é", MADE_UP_KEY[:8] + "\x01" + MADE_UP_KEY[8:], MADE_UP_KEY[:8] + "\n" + MADE_UP_KEY[8:],
         MADE_UP_KEY * 200],
        ids=["non-ascii", "control-character", "inner-line-break", "too-long"],
    )
    def test_an_env_key_that_isnt_printable_ascii_is_refused_at_startup_without_printing_it(
        self, clean_assistant_env, monkeypatch, key
    ):
        monkeypatch.setenv("OS_ASSISTANT_API_KEY", key)
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        text = f"{caught.value}\n{caught.value!r}"
        assert "os_assistant_api_key" in text and "OS_ASSISTANT_API_KEY must be printable ASCII" in text, text
        assert MADE_UP_KEY[8:] not in text, f"the startup error prints the key: {text}"

    def test_a_key_set_some_other_way_is_stripped_or_counts_as_unset(self, connection_on, monkeypatch, caplog):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_api_key", MADE_UP_KEY + "\r\n")
        assert connection.effective().api_key == MADE_UP_KEY
        monkeypatch.setattr(connection_on, "os_assistant_api_key", MADE_UP_KEY + "é")
        with caplog.at_level(logging.WARNING):
            current = connection.effective()
        assert (current.api_key, current.view()["api_key_set"], current.sources["api_key"]) == ("", False, "default")
        assert "OS_ASSISTANT_API_KEY" in caplog.text and MADE_UP_KEY not in caplog.text

    @pytest.mark.parametrize("ending", list(KEY_ENDINGS.values()), ids=list(KEY_ENDINGS))
    @pytest.mark.parametrize("route", ["/api/assistant/test", CHECK], ids=["test", "check"])
    def test_test_and_check_never_answer_the_env_key(
        self, clean_assistant_env, connection_on, no_shared_client, hang_up_url, monkeypatch, route, ending
    ):
        import src.main as main_module
        from src.assistant.client import AssistantClient

        monkeypatch.setenv("OS_ASSISTANT_URL", hang_up_url)
        monkeypatch.setenv("OS_ASSISTANT_API_KEY", MADE_UP_KEY + ending)
        started = Settings(_env_file=None)  # what the server reads from its environment at startup
        monkeypatch.setattr(connection_on, "os_api_key", "")
        monkeypatch.setattr(connection_on, "os_assistant_url", started.os_assistant_url)
        monkeypatch.setattr(connection_on, "os_assistant_api_key", started.os_assistant_api_key)
        no_shared_client.set_client(AssistantClient.from_settings(connection_on))  # what lifespan() does
        response = TestClient(main_module.create_app(), raise_server_exceptions=False).post(route, json={})
        assert MADE_UP_KEY not in response.text, f"POST {route} answered {response.status_code} {response.text}"
        assert response.status_code == 502 and _error(response)["code"] == "assistant_unreachable", response.text


# ── The server's answer is never in error text, even one h11 refuses ─────────
# An answer whose head h11 can't parse raised httpx.RemoteProtocolError, whose text
# quotes the refused line ("illegal status line: bytearray(b'...')", "illegal header line: ...", "illegal chunk
# header: ..."), and the client put that text in its assistant_unreachable message. So Check, with any candidate
# URL, Test, voice turns and book jobs answered the first line or a header line of whatever listened at the URL.
# Now every httpx.ProtocolError is named by its type with a fixed sentence (a LocalProtocolError's sentence is
# the one above), and a DecodingError by its type; a transport error (connect, read, TLS, DNS) keeps its text,
# which comes from this machine: the target's response is never echoed back (src/assistant/client.py's module
# docstring).
#
# The listener tests need a real listener on 127.0.0.1: httpx.MockTransport skips httpcore and h11, where the answer
# is refused. It reads each whole request, answers fixed bytes, and waits for the client to hang up before closing,
# so no reset races the answer. tests/conftest.py refuses every other address. The marker is made up.

ANSWER_MARKER = "made-up-marker-r3s01-0123456789"
REFUSED_ANSWERS = {
    "status-line": ("+OK " + ANSWER_MARKER + "\r\n\r\n").encode(),
    "header-line-200": ("HTTP/1.1 200 OK\r\nX Bad: " + ANSWER_MARKER + "\r\nContent-Length: 0\r\n\r\n").encode(),
    "header-line-401": (
        "HTTP/1.1 401 Unauthorized\r\nX Bad: " + ANSWER_MARKER + "\r\nContent-Length: 0\r\n\r\n"
    ).encode(),
    "chunk-header": ("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + ANSWER_MARKER + "\r\n").encode(),
}


@pytest.fixture(params=list(REFUSED_ANSWERS))
def refused_answer_url(request):
    """A base URL on 127.0.0.1 whose listener answers each request with REFUSED_ANSWERS[param], which h11 refuses."""
    import socket
    import threading

    answer = REFUSED_ANSWERS[request.param]
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.1)
    stop = threading.Event()

    def answer_one(accepted) -> None:
        with accepted:
            accepted.settimeout(5)
            try:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = accepted.recv(65536)
                    if not chunk:
                        return
                    data += chunk
                head, _, body = data.partition(b"\r\n\r\n")
                length = 0
                for line in head.split(b"\r\n")[1:]:
                    name, _, value = line.partition(b":")
                    if name.strip().lower() == b"content-length":
                        length = int(value.strip())
                while len(body) < length:
                    chunk = accepted.recv(65536)
                    if not chunk:
                        return
                    body += chunk
                accepted.sendall(answer)
                accepted.shutdown(socket.SHUT_WR)
                while accepted.recv(65536):  # until the client hangs up
                    pass
            except OSError:
                pass

    def serve() -> None:
        while not stop.is_set():
            try:
                accepted, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            answer_one(accepted)

    thread = threading.Thread(target=serve, name="r3s01-refused-answer", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/v1"
    finally:
        stop.set()
        thread.join(timeout=10)
        listener.close()


class TestTheAnswerInErrorText:
    @pytest.mark.parametrize("call", ["reply", "list_models"])
    async def test_the_client_never_puts_an_answer_h11_refuses_in_its_error(self, refused_answer_url, call):
        from src.assistant.client import AssistantClient
        from src.assistant.errors import AssistantError

        client = AssistantClient(refused_answer_url)
        try:
            with pytest.raises(AssistantError) as caught:
                if call == "reply":
                    await client.reply(MESSAGES, _settings(timeout_s=5))
                else:
                    await client.list_models(timeout=5)
        finally:
            await client.aclose()
        message = caught.value.message
        assert caught.value.code == "assistant_unreachable", message
        assert ANSWER_MARKER not in message and ANSWER_MARKER not in str(caught.value), message
        assert "RemoteProtocolError" in message, message

    @pytest.mark.parametrize("route", ["/api/assistant/test", CHECK], ids=["test", "check"])
    def test_test_and_check_never_answer_an_answer_h11_refuses(
        self, connection_on, no_shared_client, refused_answer_url, monkeypatch, route
    ):
        """No OS_API_KEY, as a home server may run. Check is sent the listener as a candidate URL, which it doesn't save;
        Test uses the connection in use."""
        import src.main as main_module
        from src.assistant.client import AssistantClient

        monkeypatch.setattr(connection_on, "os_api_key", "")
        if route == CHECK:
            payload = {"url": refused_answer_url}
        else:
            payload = {}
            monkeypatch.setattr(connection_on, "os_assistant_url", refused_answer_url)
            no_shared_client.set_client(AssistantClient.from_settings(connection_on))  # what lifespan() does
        response = TestClient(main_module.create_app(), raise_server_exceptions=False).post(route, json=payload)
        assert ANSWER_MARKER not in response.text, f"POST {route} answered {response.status_code} {response.text}"
        assert response.status_code == 502 and _error(response)["code"] == "assistant_unreachable", response.text

    @pytest.mark.parametrize("error_type", [httpx.RemoteProtocolError, httpx.ProtocolError])
    async def test_a_protocol_error_is_named_by_its_type_never_by_its_text(self, error_type):
        from src.assistant.errors import AssistantError

        text = f"illegal status line: bytearray(b'+OK {ANSWER_MARKER}')"
        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(_fail(error_type, text)))
        message = caught.value.message
        assert caught.value.code == "assistant_unreachable", message
        assert ANSWER_MARKER not in message and error_type.__name__ in message, message

    @pytest.mark.parametrize(
        ("error_type", "text"),
        [
            (httpx.ConnectError, "[Errno 11001] getaddrinfo failed"),
            (httpx.ConnectError, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
            (httpx.ReadError, "[WinError 10054] An existing connection was forcibly closed by the remote host"),
        ],
        ids=["dns", "tls", "read"],
    )
    async def test_a_transport_error_keeps_its_text_which_comes_from_this_machine(self, error_type, text):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(_fail(error_type, text)))
        assert caught.value.code == "assistant_unreachable"
        assert f"{error_type.__name__}: {text}" in caught.value.message, caught.value.message

    async def test_a_reply_that_cant_be_decoded_is_named_by_its_type_never_by_its_text(self):
        """httpx gives a DecodingError the decoder's own text; only its type is named."""
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(_fail(httpx.DecodingError, f"made-up decoder text {ANSWER_MARKER}")))
        message = caught.value.message
        assert caught.value.code == "assistant_bad_response", message
        assert ANSWER_MARKER not in message and "DecodingError" in message, message


# An extra body may nest at most 32 levels (the object itself is one; chat_template_kwargs
# needs two). A deeper one, as an object or as JSON text, is refused before anything is written: 422 naming
# extra_body from PUT and /check, a startup error naming OS_ASSISTANT_EXTRA_BODY. An assistant-connection.json from
# before the bound that holds one has that field skipped, with a warning, so the server still starts.
JSON_HEADERS = {"content-type": "application/json"}


def _nested_text(depth: int) -> str:
    """JSON text of an extra body nesting `depth` containers: the object, then depth - 1 lists inside it."""
    return '{"x": ' + "[" * (depth - 1) + "]" * (depth - 1) + "}"


def _deep_file(settings, depth: int, **fields) -> Path:
    """An assistant-connection.json as a save before the bound wrote it, with an extra body `depth` levels deep."""
    path = _connection_file(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = json.dumps({"version": 1, "connection": {**fields, "extra_body": "@"}}, indent=2)
    path.write_text(stored.replace('"@"', _nested_text(depth)) + "\n", encoding="utf-8")
    return path


class TestExtraBodyDepth:
    @pytest.mark.parametrize(
        ("form", "depth"),
        [("object", 33), ("object", 600), ("text", 33), ("text", 600), ("text", 5000)],
        ids=lambda value: str(value),
    )
    @pytest.mark.parametrize("route", ["put", "check"])
    def test_a_body_nested_too_deep_is_refused_with_422_and_nothing_is_written(
        self, connection_api, connection_on, form, depth, route
    ):
        assert connection_api.put(CONNECTION, json={"model": "before"}).status_code == 200
        path = _connection_file(connection_on)
        before = path.read_bytes()
        inner = _nested_text(depth)
        raw = '{"model": "after", "extra_body": ' + (inner if form == "object" else json.dumps(inner)) + "}"
        if route == "put":
            response = connection_api.put(CONNECTION, content=raw, headers=JSON_HEADERS)
        else:
            response = connection_api.post(CHECK, content=raw, headers=JSON_HEADERS)  # no server: nothing is sent
        assert response.status_code == 422, response.text[:300]
        assert _error(response)["param"] == "extra_body"
        assert "32 levels" in _error(response)["message"]
        assert path.read_bytes() == before
        view = connection_api.get(CONNECTION).json()
        assert (view["model"], view["extra_body"]) == ("before", {})

    def test_a_body_32_levels_deep_is_saved_and_read_back_after_a_restart(self, connection_api, connection_on, monkeypatch):
        from src.assistant import connection

        body = json.loads(_nested_text(32))
        response = connection_api.put(CONNECTION, json={"extra_body": body})
        assert response.status_code == 200 and response.json()["extra_body"] == body
        monkeypatch.setattr(connection, "_store", connection._Store())  # a new process
        assert dict(connection.effective(connection_on).extra_body) == body

    @pytest.mark.parametrize("depth", [33, 600, 5000])
    def test_an_env_extra_body_nested_too_deep_stops_startup_naming_the_setting(self, clean_assistant_env, monkeypatch, depth):
        monkeypatch.setenv("OS_ASSISTANT_EXTRA_BODY", _nested_text(depth))
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert "OS_ASSISTANT_EXTRA_BODY can nest" in str(caught.value) and "32 levels" in str(caught.value)

    @pytest.mark.parametrize("depth", [33, 600])
    def test_a_saved_file_from_before_the_bound_is_read_without_its_extra_body(self, connection_on, monkeypatch, caplog, depth):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_extra_body", '{"top_k": 20}')
        _deep_file(connection_on, depth, model="after")
        monkeypatch.setattr(connection, "_store", connection._Store())  # a new process
        with caplog.at_level(logging.WARNING, logger="src.assistant.connection"):
            current = connection.effective(connection_on)
            view = current.view()
        assert (view["model"], view["extra_body"], view["sources"]["extra_body"]) == ("after", {"top_k": 20}, "env")
        assert any("extra_body" in record.getMessage() and "assistant-connection.json" in record.getMessage()
                   for record in caplog.records), caplog.text

    def test_the_server_starts_with_such_a_file(self, connection_on, no_shared_client, monkeypatch, caplog):
        """The real lifespan, which reads settings.assistant_enabled and so the saved file (src/main.py)."""
        import src.main as main_module
        from src.assistant import connection

        _deep_file(connection_on, 600, model="after")
        monkeypatch.setattr(connection, "_store", connection._Store())  # a new process
        with caplog.at_level(logging.WARNING, logger="src.assistant.connection"):
            with TestClient(main_module.app) as client:
                answer = client.get(CONNECTION)
                assert no_shared_client.get_client() is not None
        assert answer.status_code == 200, answer.text
        assert (answer.json()["model"], answer.json()["extra_body"]) == ("after", {})
        assert "Skipping extra_body" in caplog.text

    def test_a_write_that_fails_leaves_the_file_and_the_store_as_they_were(self, connection_on, monkeypatch):
        """The store's copy is made before the file is replaced, so the server never answers from other fields
        than its file holds."""
        import copy as copy_module

        from src.assistant import connection

        connection.save({"model": "before"})
        path = _connection_file(connection_on)
        before = path.read_bytes()

        class _FailingCopy:
            """copy.deepcopy, except that copying the new fields fails, as a 600-level extra body did."""

            @staticmethod
            def deepcopy(value):
                if isinstance(value, dict) and value.get("model") == "after":
                    raise RecursionError("maximum recursion depth exceeded")
                return copy_module.deepcopy(value)

        with monkeypatch.context() as patched:
            patched.setattr(connection, "copy", _FailingCopy)
            with pytest.raises(RecursionError):
                connection.save({"model": "after"})
        assert connection.copy is copy_module
        assert path.read_bytes() == before
        assert connection.effective(connection_on).model == "before"
        monkeypatch.setattr(connection, "_store", connection._Store())  # and a new process agrees
        assert connection.effective(connection_on).model == "before"


# NaN, Infinity and -Infinity aren't JSON numbers (RFC 8259, section 6), though Python's
# json module reads them. An extra body holding one, as text or in the request's own JSON, is refused before it is
# saved: 422 naming extra_body. One already in the saved file, or in OS_ASSISTANT_EXTRA_BODY set past the startup check,
# is unusable, with a warning, and .env (or no extra body) is used.
NOT_JSON_NUMBERS = ["NaN", "Infinity", "-Infinity"]


class TestExtraBodyNumbers:
    @pytest.mark.parametrize("number", NOT_JSON_NUMBERS)
    @pytest.mark.parametrize("form", ["text", "object"])
    @pytest.mark.parametrize("route", ["put", "check"])
    def test_a_number_json_has_not_is_refused_with_422_and_nothing_is_saved(
        self, connection_api, connection_on, route, form, number
    ):
        assert connection_api.put(CONNECTION, json={"model": "before"}).status_code == 200
        path = _connection_file(connection_on)
        before = path.read_bytes()
        inner = '{"top_k": 20, "sampler": {"weights": [1, %s]}}' % number
        raw = '{"model": "after", "extra_body": ' + (inner if form == "object" else json.dumps(inner)) + "}"
        if route == "put":
            response = connection_api.put(CONNECTION, content=raw, headers=JSON_HEADERS)
        else:
            response = connection_api.post(CHECK, content=raw, headers=JSON_HEADERS)  # no server: nothing is sent
        assert response.status_code == 422, response.text[:300]
        assert _error(response)["param"] == "extra_body"
        assert "NaN or Infinity" in _error(response)["message"]
        assert path.read_bytes() == before
        after = connection_api.get(CONNECTION)
        assert after.status_code == 200 and (after.json()["model"], after.json()["extra_body"]) == ("before", {})

    @pytest.mark.parametrize("number", NOT_JSON_NUMBERS)
    def test_an_env_extra_body_holding_one_stops_startup_naming_the_setting(self, clean_assistant_env, monkeypatch, number):
        monkeypatch.setenv("OS_ASSISTANT_EXTRA_BODY", '{"top_k": %s}' % number)
        with pytest.raises(ValidationError) as caught:
            Settings(_env_file=None)
        assert "OS_ASSISTANT_EXTRA_BODY can't hold NaN or Infinity" in str(caught.value)

    @pytest.mark.parametrize("number", NOT_JSON_NUMBERS)
    def test_a_saved_file_holding_one_is_read_without_its_extra_body(self, connection_api, connection_on, monkeypatch, caplog, number):
        monkeypatch.setattr(connection_on, "os_assistant_extra_body", '{"top_k": 20}')
        path = _connection_file(connection_on)
        path.parent.mkdir(parents=True)
        path.write_text('{"version": 1, "connection": {"model": "after", "extra_body": {"seed": %s}}}' % number,
                        encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="src.assistant.connection"):
            response = connection_api.get(CONNECTION)
        assert response.status_code == 200, response.text
        view = response.json()
        assert (view["model"], view["extra_body"], view["sources"]["extra_body"]) == ("after", {"top_k": 20}, "env")
        assert "Skipping extra_body" in caplog.text

    def test_an_env_extra_body_set_past_the_startup_check_is_ignored_with_a_warning(
        self, connection_api, connection_on, servers, no_shared_client, monkeypatch, caplog
    ):
        from src.assistant import connection

        monkeypatch.setattr(connection_on, "os_assistant_extra_body", '{"top_k": NaN}')
        with caplog.at_level(logging.WARNING, logger="src.assistant.connection"):
            view = connection_api.get(CONNECTION)
        assert view.status_code == 200, view.text
        assert view.json()["extra_body"] == {} and "Ignoring OS_ASSISTANT_EXTRA_BODY" in caplog.text
        ai_server = servers.add(AI_SERVER)
        no_shared_client.set_client(no_shared_client.build_client(connection.effective(connection_on)))
        tested = connection_api.post("/api/assistant/test", json={"preset": "voice"})
        assert tested.status_code == 200, tested.text
        assert "top_k" not in ai_server.body


# Text with a lone UTF-16 surrogate (the JSON escape "\ud83d", half an emoji, as a
# JavaScript slice can leave it) can't be encoded as UTF-8, so it can be neither stored nor sent to the AI server. It
# is refused with 422 naming the field: in every preset text field, in the Test text and overrides, and in the
# connection's model and extra body. A preset file holding one has that preset skipped, with a warning.
HALF_AN_EMOJI = "\ud83d"  # json.dumps writes it as the escape \ud83d


def _raw(body) -> bytes:
    """The body as the JSON a client sends: ASCII, with each lone surrogate as its \\u escape."""
    return json.dumps(body).encode("ascii")


PRESET_TEXT_FIELDS = [
    ("name", "Mine " + HALF_AN_EMOJI),
    ("model", "your-model" + HALF_AN_EMOJI),
    ("system_prompt", "Be brief " + HALF_AN_EMOJI),
    ("user_template", "{text} " + HALF_AN_EMOJI),
    ("fillers", ["Hmm.", "Let me see " + HALF_AN_EMOJI]),
    ("error_sentences", {"assistant_timeout": "Too slow " + HALF_AN_EMOJI}),
]


class TestLoneSurrogates:
    @pytest.mark.parametrize(("field", "value"), PRESET_TEXT_FIELDS, ids=[name for name, _ in PRESET_TEXT_FIELDS])
    @pytest.mark.parametrize("route", ["create", "replace"])
    def test_a_preset_text_field_with_one_is_refused_with_422_naming_it(self, api, assistant_on, route, field, value):
        body = {"name": "Mine", field: value}
        if route == "create":
            response = api.post("/api/assistant/presets", content=_raw(body), headers=JSON_HEADERS)
        else:
            assert api.post("/api/assistant/presets", json={"name": "Mine"}).status_code == 201
            response = api.put("/api/assistant/presets/mine", content=_raw(body), headers=JSON_HEADERS)
        assert response.status_code == 422, response.text
        assert _error(response)["param"] == field
        assert "lone surrogate" in _error(response)["message"]
        listed = api.get("/api/assistant/presets")
        assert listed.status_code == 200
        assert [preset["id"] for preset in listed.json()["presets"]] == (
            ["voice", "book"] if route == "create" else ["voice", "book", "mine"])

    @pytest.mark.parametrize(
        ("body", "param"),
        [({"preset": "voice", "text": "Hi " + HALF_AN_EMOJI}, "text"),
         ({"preset": "voice", "overrides": {"system_prompt": "Be brief " + HALF_AN_EMOJI}}, "system_prompt")],
        ids=["text", "override"],
    )
    def test_the_test_call_refuses_one_with_422_and_sends_nothing(self, api, assistant_on, servers, no_shared_client, body, param):
        from src.assistant import connection

        ai_server = servers.add(AI_SERVER)
        no_shared_client.set_client(no_shared_client.build_client(connection.effective(assistant_on)))
        response = api.post("/api/assistant/test", content=_raw(body), headers=JSON_HEADERS)
        assert response.status_code == 422, response.text
        assert _error(response)["param"] == param and "lone surrogate" in _error(response)["message"]
        assert ai_server.requests == []

    def test_a_book_job_or_voice_turns_overrides_are_refused_naming_the_field(self, assistant_on):
        from src.assistant import presets

        for purpose in ("book", "voice"):
            error = _preset_error(presets.effective, None, {"system_prompt": "Be brief " + HALF_AN_EMOJI}, purpose=purpose)
            assert error.field == "system_prompt" and "lone surrogate" in error.message

    @pytest.mark.parametrize(
        ("field", "value"),
        [("model", "your-model" + HALF_AN_EMOJI),
         ("extra_body", {"stop": ["\n", HALF_AN_EMOJI]}),
         ("extra_body", json.dumps({"stop": [HALF_AN_EMOJI]}))],
        ids=["model", "extra-body-object", "extra-body-text"],
    )
    @pytest.mark.parametrize("route", ["put", "check"])
    def test_a_connection_field_with_one_is_refused_with_422_and_nothing_is_written(
        self, connection_api, connection_on, route, field, value
    ):
        body = {field: value}
        if route == "put":
            response = connection_api.put(CONNECTION, content=_raw(body), headers=JSON_HEADERS)
        else:
            response = connection_api.post(CHECK, content=_raw(body), headers=JSON_HEADERS)  # no server: nothing sent
        assert response.status_code == 422, response.text
        assert _error(response)["param"] == field and "lone surrogate" in _error(response)["message"]
        assert not _connection_file(connection_on).exists()
        assert connection_api.get(CONNECTION).status_code == 200

    def test_a_preset_file_holding_one_has_that_preset_skipped_with_a_warning(self, api, assistant_on, caplog):
        path = _presets_file(assistant_on)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "presets": [
            {"id": "mine", "name": "Mine", "system_prompt": "Be brief " + HALF_AN_EMOJI},
            {"id": "yours", "name": "Yours"},
        ]}), encoding="ascii")
        with caplog.at_level(logging.WARNING, logger="src.assistant.presets"):
            listed = api.get("/api/assistant/presets")
        assert listed.status_code == 200, listed.text
        assert [preset["id"] for preset in listed.json()["presets"]] == ["voice", "book", "yours"]
        assert "Skipping AI preset 'mine'" in caplog.text


# An answer of JSON nested deeper than Python's json module reads (a few thousand levels,
# far under the 8 MiB cap) makes json.loads raise RecursionError, not ValueError. It is a bad response like any
# other undecodable answer: the reply raises AssistantError assistant_bad_response (spoken in a voice turn, retried
# once by a book job), Test answers 502 with that code, and /check answers 200 with listed: false.
DEEP_ANSWERS = {
    "lists": b"[" * 100_000 + b"]" * 100_000,  # 200 KB
    "objects": b'{"a": ' * 100_000 + b"1" + b"}" * 100_000,  # 700 KB
    "in-choices": b'{"choices": [{"message": {"content": ' + b"[" * 100_000 + b"]" * 100_000 + b"}}]}",
}


class TestDeepJsonAnswers:
    @pytest.mark.parametrize("answer", list(DEEP_ANSWERS))
    async def test_a_reply_nested_too_deep_to_read_is_a_bad_response(self, answer):
        from src.assistant.errors import AssistantError

        with pytest.raises(AssistantError) as caught:
            await _reply(_Server(lambda _request: httpx.Response(200, content=DEEP_ANSWERS[answer])))
        assert caught.value.code == "assistant_bad_response", caught.value.message

    @pytest.mark.parametrize("answer", list(DEEP_ANSWERS))
    async def test_a_model_list_nested_too_deep_to_read_is_not_a_list(self, answer):
        client = _client(_Server(lambda _request: httpx.Response(200, content=DEEP_ANSWERS[answer])))
        try:
            listed = await client.list_models(timeout=5)
        finally:
            await client.aclose()
        assert (listed.status, listed.models, listed.listed) == (200, [], False)

    def test_check_answers_200_with_listed_false(self, connection_api, servers):
        servers.add(AI_SERVER, lambda _request: httpx.Response(200, content=DEEP_ANSWERS["lists"]))
        response = connection_api.post(CHECK, json={})
        assert response.status_code == 200, response.text
        assert {key: response.json()[key] for key in ("reachable", "status", "models", "listed")} == {
            "reachable": True, "status": 200, "models": [], "listed": False,
        }

    def test_test_answers_502_assistant_bad_response(self, connection_api, connection_on, servers, no_shared_client):
        from src.assistant import connection

        servers.add(AI_SERVER, lambda _request: httpx.Response(200, content=DEEP_ANSWERS["in-choices"]))
        no_shared_client.set_client(no_shared_client.build_client(connection.effective(connection_on)))
        response = connection_api.post("/api/assistant/test", json={"preset": "voice"})
        assert response.status_code == 502, response.text
        assert _error(response)["code"] == "assistant_bad_response"


# ── Preset actions: label, inputs and output ────
#
# Presets have three optional fields, the action's name (label), the input kinds it suits (inputs)
# and its default output (output). The built-ins get sensible values, and old preset files still load.

ACTIONS = ("label", "inputs", "output")
EVERY_INPUT = ["text", "transcript", "document"]


def _action(view) -> dict:
    return {name: view[name] for name in ACTIONS}


class TestPresetActions:
    def test_the_built_ins_name_their_action_the_inputs_they_suit_and_their_output(self, assistant_on):
        from src.assistant import presets

        assert presets.ACTION_FIELDS == ACTIONS
        assert _action(presets.get_preset("voice")) == {"label": "Answer", "inputs": ["text", "transcript"], "output": "speech"}
        assert _action(presets.get_preset("book")) == {
            "label": "Rewrite for listening", "inputs": ["document"], "output": "document",
        }

    def test_a_preset_you_make_is_labelled_by_its_name_suits_every_input_and_shows_text(self, assistant_on):
        from src.assistant import presets

        made = presets.create_preset({"name": "Summarize"})
        assert _action(made) == {"label": "Summarize", "inputs": EVERY_INPUT, "output": "text"}
        assert _action(presets.replace_preset("summarize", {"name": "Sum up"})) == {
            "label": "Sum up", "inputs": EVERY_INPUT, "output": "text",
        }

    def test_they_are_saved_trimmed_and_in_the_kinds_order(self, assistant_on):
        from src.assistant import presets

        made = presets.create_preset({
            "name": "Translate", "label": "  Translate to French ", "inputs": ["document", "text", "text"],
            "output": "document",
        })
        assert _action(made) == {"label": "Translate to French", "inputs": ["text", "document"], "output": "document"}
        stored = json.loads(_presets_file(assistant_on).read_text(encoding="utf-8"))["presets"]
        assert stored == [{"id": "translate", "name": "Translate", "label": "Translate to French",
                           "inputs": ["text", "document"], "output": "document"}]
        assert presets.get_preset("translate") == made

    @pytest.mark.parametrize(("field", "value"), [
        ("label", 5), ("label", "x" * 81), ("label", "half \ud83d"), ("inputs", "text"), ("inputs", ["file"]),
        ("inputs", [1]), ("output", "audio"), ("output", 1), ("output", ["text"]),
    ])
    def test_a_bad_value_is_refused_naming_the_field(self, assistant_on, field, value):
        from src.assistant import presets

        error = _preset_error(presets.create_preset, {"name": "Mine", field: value})
        assert error.field == field and field in error.message
        assert not _presets_file(assistant_on).exists()

    def test_an_emptied_field_is_the_default_again(self, assistant_on):
        from src.assistant import presets

        presets.create_preset({"name": "Mine", "label": "Mine to you", "inputs": ["text"], "output": "speech"})
        emptied = presets.replace_preset("mine", {"name": "Mine", "label": " ", "inputs": [], "output": ""})
        assert _action(emptied) == {"label": "Mine", "inputs": EVERY_INPUT, "output": "text"}
        presets.replace_preset("voice", {"label": "Ask", "output": "text"})
        reset = presets.replace_preset("voice", {"label": None, "output": None})
        assert _action(reset) == {"label": "Answer", "inputs": ["text", "transcript"], "output": "speech"}
        assert reset["customized"] is False

    def test_a_save_that_leaves_them_out_keeps_them(self, assistant_on):
        # Settings › AI assistant saves the name and the tunable fields only (saveBody in assistant-logic.js), so a
        # save there keeps the label, inputs and output set through the API.
        from src.assistant import presets

        presets.replace_preset("voice", {"label": "Ask", "inputs": ["text"], "output": "text"})
        kept = presets.replace_preset("voice", {"name": "Voice", "max_tokens": 64})
        assert (_action(kept), kept["max_tokens"]) == ({"label": "Ask", "inputs": ["text"], "output": "text"}, 64)
        assert presets.get_preset("voice") == kept

    def test_a_built_ins_own_values_sent_back_are_not_a_customization(self, assistant_on):
        from src.assistant import presets

        view = presets.replace_preset("book", {"name": "Book", **_action(presets.get_preset("book"))})
        assert view["customized"] is False and not _presets_file(assistant_on).read_text(encoding="utf-8").count("label")

    def test_a_request_cannot_override_them(self, assistant_on):
        from src.assistant import presets

        for field, value in (("label", "Ask"), ("inputs", ["text"]), ("output", "text")):
            error = _preset_error(presets.effective, "voice", {field: value})
            assert error.field == field and "can't override" in error.message

    def test_they_never_reach_the_ai_or_a_jobs_settings(self, assistant_on):
        from src.assistant import presets

        presets.create_preset({"name": "Explain", "label": "Explain this", "inputs": ["document"], "output": "speech"})
        assert not set(ACTIONS) & set(presets.effective("explain").to_dict())

    def test_an_older_preset_file_without_them_still_loads(self, assistant_on):
        from src.assistant import presets

        path = _presets_file(assistant_on)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "presets": [
            {"id": "voice", "max_tokens": 5}, {"id": "mine", "name": "Mine", "reasoning_effort": "low"},
        ]}), encoding="utf-8")
        listed = {view["id"]: view for view in presets.list_presets()}
        assert list(listed) == ["voice", "book", "mine"]
        assert (listed["voice"]["max_tokens"], listed["voice"]["label"]) == (5, "Answer")
        assert _action(listed["mine"]) == {"label": "Mine", "inputs": EVERY_INPUT, "output": "text"}
        assert presets.effective("mine").reasoning_effort == "low"

    def test_the_preset_routes_take_them(self, api):
        created = api.post("/api/assistant/presets", json={
            "name": "Explain", "label": "Explain this", "inputs": ["text"], "output": "speech",
        })
        assert created.status_code == 201, created.text
        assert _action(created.json()) == {"label": "Explain this", "inputs": ["text"], "output": "speech"}
        refused = api.put("/api/assistant/presets/explain", json={"name": "Explain", "inputs": ["video"]})
        assert refused.status_code == 422 and _error(refused)["param"] == "inputs"

    def test_a_job_falls_back_to_the_job_timeout_and_keeps_its_presets_own_prompt(self, assistant_on):
        # An ask that becomes a job (a long input) waits on no one, as a book job doesn't.
        from src.assistant import presets

        as_job = presets.effective("voice", job=True)
        assert (as_job.timeout_s, as_job.system_prompt) == (600.0, VOICE_PROMPT)
        assert presets.effective("voice", {"timeout_s": 30}, job=True).timeout_s == 30.0
        assert presets.effective("voice").timeout_s == 120.0


# ── Send anything to the AI: POST /api/assistant/ask ──
#
# README.md, "Ask AI: send anything to the AI", and the route's docstring. No network: the AI is an
# httpx.MockTransport (or a _FailingClient), and the TTS router is _FakeTTS, wired into the route as src/main.py wires
# the real one. Document inputs, document outputs and jobs are in tests/test_reader_assistant.py, which gives each
# test a studio.db of its own.

ASK = "/api/assistant/ask"
TEXT = {"kind": "text", "text": "Why is the sky blue?"}
WAV = {"model": "kokoro", "voice": "af_heart", "format": "wav"}


class _FakeTTS:
    """Stands in for the TTS router: one model, ``kokoro``, whose synthesize() records each request and gives a
    tenth of a second of tone. ``fail`` makes it raise."""

    def __init__(self) -> None:
        self.spoken: list[dict] = []
        self.fail: Exception | None = None

    def list_backends(self):
        return ["kokoro"]

    def get_backend(self, model):
        if model != "kokoro":
            raise ValueError(f"Model '{model}' not found")
        return SimpleNamespace(name="kokoro")

    def get_capabilities(self, model):
        self.get_backend(model)
        return {"speed_control": True}

    def validate_voice(self, model, voice):
        if voice == "nobody":
            raise ValueError(f"Voice '{voice}' is not available for {model}")

    def max_input_chars_for(self, _model):
        return None

    def sample_rate_for(self, _model):
        return 24000

    def synthesize(self, **request):
        if self.fail is not None:
            raise self.fail
        self.spoken.append(request)
        return iter([np.full(2400, 0.2, dtype=np.float32)])

    @property
    def texts(self) -> list[str]:
        return [request["text"] for request in self.spoken]


class _Pronounce:
    """Stands in for TTS_PRONUNCIATION_DICT."""

    def apply(self, text: str) -> str:
        return text.replace("Ada", "Ayda")


class _Expand:
    """A dictionary entry that lengthens text."""

    def apply(self, text: str) -> str:
        return text.replace("AI", "artificial intelligence")


class _PausingModel(_FakeTTS):
    """One 24 kHz model that speaks as a real one does: 0.1 s of silence, 0.5 s of tone, then 0.3 s of pause.
    ``limit`` is its input limit (``max_input_chars_for``)."""

    def __init__(self, limit: int | None = None) -> None:
        super().__init__()
        self.limit = limit

    def max_input_chars_for(self, _model):
        return self.limit

    def synthesize(self, **request):
        if self.limit is not None and len(request["text"]) > self.limit:
            raise AssertionError(f"{len(request['text'])} characters for a model that takes {self.limit}")
        self.spoken.append(request)
        tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(12000) / 24000)).astype(np.float32)
        return iter([np.zeros(2400, np.float32), tone, np.zeros(7200, np.float32)])


_SPEECH_SETTINGS = SimpleNamespace(tts_enabled=True, tts_max_input_length=4096, tts_trim_silence=True,
                                   tts_normalize_output=True)  # the shipped defaults


def _speaker(router, dictionary=None):
    from src.assistant.ask import Speaker

    return Speaker(router, settings=_SPEECH_SETTINGS, pronunciation_dict=dictionary)


def _speech():
    from src.assistant.ask import Speech

    return Speech(model="kokoro", voice="af_heart", format="wav", speed=1.0)


def _speak_reply(router, reply: str, dictionary=None) -> tuple[np.ndarray, dict]:
    """The reply spoken through Speaker.speak as a WAV: its samples, and the answer."""
    import io
    import wave

    answer = asyncio.run(asyncio.wait_for(_speaker(router, dictionary).speak(reply, _speech()), timeout=30))
    if "audio" not in answer:
        return np.zeros(0, np.float32), answer
    with wave.open(io.BytesIO(base64.b64decode(answer["audio"]["base64"]))) as wav:
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768, answer


def _pauses_inside(samples: np.ndarray) -> list[float]:
    """Each quiet run of 20 ms or more between the first and last loud sample, in seconds."""
    loud = np.flatnonzero(np.abs(samples) > 0.01)
    return [float(gap) / 24000 for gap in np.diff(loud) - 1 if gap >= 480]


@pytest.fixture
def tts():
    return _FakeTTS()


@pytest.fixture
def ask_api(assistant_on, no_shared_client, monkeypatch, tts):
    """The configured app (no lifespan), the assistant on, no API key, and the fake TTS router and dictionary wired
    into the assistant routes. The fakes replace what `src/main.py` passes itself (its own router, which has no
    backends without the lifespan)."""
    import src.main as main_module
    from src.routes import assistant as assistant_routes

    monkeypatch.setattr(assistant_on, "os_api_key", "")
    monkeypatch.setattr(assistant_on, "tts_enabled", True)
    monkeypatch.setattr(assistant_on, "tts_model", "kokoro")
    monkeypatch.setattr(assistant_on, "tts_voice", "af_heart")
    real_create_router = assistant_routes.create_router

    def wired(*args, **kwargs):
        kwargs.update(
            get_tts_router=lambda: tts, get_voice_library=lambda: None, get_pronunciation_dict=lambda: _Pronounce(),
        )
        return real_create_router(*args, **kwargs)

    monkeypatch.setattr(assistant_routes, "create_router", wired)
    return TestClient(main_module.create_app())


def _ai_server(assistant, answer=None) -> _Server:
    """The shared client, answering through a MockTransport."""
    server = _Server(answer)
    assistant.set_client(_client(server))
    return server


ASK_REFUSED = [
    pytest.param({}, "input", id="no-input"),
    pytest.param({"input": "Why?"}, "input", id="input-not-an-object"),
    pytest.param({"input": {"text": "Why?"}}, "input.kind", id="no-kind"),
    pytest.param({"input": {"kind": "file"}}, "input.kind", id="unknown-kind"),
    pytest.param({"input": {"kind": "text"}}, "input.text", id="no-text"),
    pytest.param({"input": {"kind": "text", "text": "  \n"}}, "input.text", id="blank-text"),
    pytest.param({"input": {"kind": "text", "text": 5}}, "input.text", id="text-not-text"),
    pytest.param({"input": {"kind": "text", "text": "Why?", "history_id": "h"}}, "input.history_id", id="text-with-a-history-id"),
    pytest.param({"input": {"kind": "transcript"}}, "input.text", id="transcript-with-neither"),
    pytest.param({"input": {"kind": "transcript", "text": "Hi", "history_id": "h"}}, "input.history_id", id="transcript-with-both"),
    pytest.param({"input": {"kind": "transcript", "history_id": 7}}, "input.history_id", id="history-id-not-text"),
    pytest.param({"input": {"kind": "document"}}, "input.document_id", id="no-document-id"),
    pytest.param({"input": {"kind": "document", "document_id": "x" * 65}}, "input.document_id", id="long-document-id"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "chapters": [0, 0], "blocks": [0, 0]}}, "input.blocks", id="chapters-and-blocks"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "blocks": [2, 1]}}, "input.blocks", id="blocks-backwards"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "blocks": [0]}}, "input.blocks", id="blocks-one-number"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "chapters": "all"}}, "input.chapters", id="chapters-not-a-pair"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block_start": 1}}, "input.block_end", id="selection-without-an-end"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block_start": True, "block_end": 1}}, "input.block_start", id="selection-of-a-bool"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "blocks": [0, 1], "block_start": 0, "block_end": 1}}, "input.block_start", id="blocks-and-a-selection"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "text": "x"}}, "input.text", id="document-with-text"),
    # A sentence is its block and its index in that block, both whole numbers from 0, and
    # names the part alone.
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0}}, "input.sentence", id="block-without-a-sentence"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "sentence": 0}}, "input.block", id="sentence-without-a-block"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": -1, "sentence": 0}}, "input.block", id="block-below-0"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0, "sentence": True}}, "input.sentence", id="sentence-of-a-bool"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0, "sentence": 1.5}}, "input.sentence", id="sentence-not-whole"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0, "sentence": 0, "blocks": [0, 0]}}, "input.block", id="sentence-and-blocks"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0, "sentence": 0, "chapters": [0, 0]}}, "input.block", id="sentence-and-chapters"),
    pytest.param({"input": {"kind": "document", "document_id": "d", "block": 0, "sentence": 0, "block_start": 0, "block_end": 0}}, "input.block", id="sentence-and-a-selection"),
    pytest.param({"input": {"kind": "text", "text": "Why?", "sentence": 0}}, "input.sentence", id="text-with-a-sentence"),
    # The question follows the input's text rules, up to 2,000 characters.
    pytest.param({"input": TEXT, "question": 5}, "question", id="question-not-text"),
    pytest.param({"input": TEXT, "question": " \n\t"}, "question", id="blank-question"),
    pytest.param({"input": TEXT, "question": "x" * 2001}, "question", id="question-over-2000"),
    pytest.param({"input": TEXT, "question": ["Explain this"]}, "question", id="question-a-list"),
    pytest.param({"input": TEXT, "output": "speech"}, "output", id="output-not-an-object"),
    pytest.param({"input": TEXT, "output": {"text": False}}, "output", id="no-output"),
    pytest.param({"input": TEXT, "output": {"text": "yes"}}, "output.text", id="text-not-a-bool"),
    pytest.param({"input": TEXT, "output": {"audio": {}}}, "output.audio", id="unknown-output"),
    pytest.param({"input": TEXT, "output": {"speech": "af_heart"}}, "output.speech", id="speech-not-an-object"),
    pytest.param({"input": TEXT, "output": {"speech": {"pitch": 2}}}, "output.speech.pitch", id="unknown-speech-field"),
    pytest.param({"input": TEXT, "output": {"speech": {"voice": ""}}}, "output.speech.voice", id="empty-voice"),
    pytest.param({"input": TEXT, "output": {"speech": {"model": "m" * 201}}}, "output.speech.model", id="long-model"),
    pytest.param({"input": TEXT, "output": {"speech": {"format": "ogg"}}}, "output.speech.format", id="unknown-format"),
    pytest.param({"input": TEXT, "output": {"speech": {"speed": 5}}}, "output.speech.speed", id="speed-over-4"),
    pytest.param({"input": TEXT, "output": {"speech": {"speed": "fast"}}}, "output.speech.speed", id="speed-not-a-number"),
    pytest.param({"input": TEXT, "output": {"document": "Notes"}}, "output.document", id="document-not-an-object"),
    pytest.param({"input": TEXT, "output": {"document": {"title": 5}}}, "output.document.title", id="title-not-text"),
    pytest.param({"input": TEXT, "output": {"document": {"title": "x" * 201}}}, "output.document.title", id="long-title"),
    pytest.param({"input": TEXT, "output": {"document": {"name": "x"}}}, "output.document.name", id="unknown-document-field"),
    pytest.param({"input": TEXT, "preset": "nope"}, "preset", id="unknown-preset"),
    pytest.param({"input": TEXT, "preset": 5}, "preset", id="preset-not-an-id"),
    pytest.param({"input": TEXT, "overrides": {"top_p": 3}}, "top_p", id="bad-override"),
    pytest.param({"input": TEXT, "overrides": {"label": "Ask"}}, "label", id="override-of-a-presets-own-field"),
    pytest.param({"input": TEXT, "overrides": ["top_p"]}, "overrides", id="overrides-not-an-object"),
    pytest.param({"input": TEXT, "stream": True}, "stream", id="stream-true"),
    pytest.param({"input": TEXT, "stream": "no"}, "stream", id="stream-not-a-bool"),
    pytest.param({"input": TEXT, "what": 1}, "what", id="unknown-field"),
]


class TestAsk:
    def test_the_app_wires_its_tts_router_into_ask(self, api, no_shared_client):
        """The running app passes its TTS router to /api/assistant/ask, so speech output is checked against it
        (the src/main.py wiring). This app has no lifespan, so the router has no
        backends and refuses the speech (`provider_missing`, or `validation_error` for the model) before the AI is
        asked; without the wiring the answer is `503 speech_unavailable`."""
        server = _ai_server(no_shared_client)
        speech = {"model": "no-such-model", "voice": "x", "format": "wav"}
        response = api.post(ASK, json={"input": TEXT, "output": {"speech": speech}})
        error = _error(response)
        assert error["code"] != "speech_unavailable", response.text
        assert error["code"] in ("provider_missing", "validation_error") and error["param"] == "output.speech"
        assert server.requests == []

    def test_text_in_gives_the_reply_at_once_through_the_preset(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client, lambda _r: _answer(content="Light scatters.", reasoning_content="Rayleigh."))
        response = ask_api.post(ASK, json={"input": TEXT})
        assert response.status_code == 200, response.text
        answer = response.json()
        elapsed = answer.pop("elapsed_ms")
        assert isinstance(elapsed, int) and elapsed >= 0
        assert answer == {
            "reply": "Light scatters.", "thinking": "Rayleigh.", "finish_reason": "stop",
            "preset": "voice", "input": {"kind": "text", "chars": 20},
        }
        assert server.body == {
            "model": "qwen3-30b",
            "messages": [{"role": "system", "content": VOICE_PROMPT}, {"role": "user", "content": "Why is the sky blue?"}],
            "stream": False,
        }

    def test_a_preset_and_overrides_shape_the_request(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client, lambda _r: _answer(content="Short.", reasoning_content="Hidden."))
        response = ask_api.post(ASK, json={
            "preset": "book", "input": {"kind": "transcript", "text": "We met at noon."},
            "overrides": {"user_template": "Summarize: {text}", "reasoning_effort": "low", "show_thinking": False},
        })
        assert response.status_code == 200, response.text
        answer = response.json()
        assert (answer["reply"], answer["thinking"], answer["preset"]) == ("Short.", None, "book")
        assert answer["input"] == {"kind": "transcript", "chars": 15}
        assert server.body["messages"] == [
            {"role": "system", "content": BOOK_PROMPT}, {"role": "user", "content": "Summarize: We met at noon."},
        ]
        assert server.body["reasoning_effort"] == "low"

    def test_speech_is_the_reply_cleaned_spoken_and_sent_back_in_base64(self, ask_api, no_shared_client, tts):
        written = "**Ada** says hi. See [the guide](http://docs.example/guide)."
        _ai_server(no_shared_client, lambda _r: _answer(content=written))
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": {**WAV, "speed": 1.25}}})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["reply"] == written, "shown as the AI wrote it"
        audio = answer["audio"]
        data = base64.b64decode(audio["base64"], validate=True)
        assert (audio["format"], audio["content_type"], audio["bytes"]) == ("wav", "audio/wav", len(data))
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        # Spoken as the Live Reader speaks: the Markdown and the link's address gone, the dictionary applied.
        assert " ".join(tts.texts) == "Ayda says hi. See the guide."
        assert {(r["model"], r["voice"], r["speed"]) for r in tts.spoken} == {("kokoro", "af_heart", 1.25)}
        assert "audio_error" not in answer and "document_id" not in answer

    def test_speech_takes_the_servers_model_and_voice_and_mp3_by_default(self, ask_api, no_shared_client, tts, monkeypatch):
        from src.tts import pipeline

        monkeypatch.setattr(pipeline, "encode_with_ffmpeg", lambda _audio, fmt, _rate: b"ID3-" + fmt.encode())
        _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": {}}})
        assert response.status_code == 200, response.text
        audio = response.json()["audio"]
        assert (audio["format"], audio["content_type"], base64.b64decode(audio["base64"])) == ("mp3", "audio/mpeg", b"ID3-mp3")
        assert {(r["model"], r["voice"], r["speed"]) for r in tts.spoken} == {("kokoro", "af_heart", 1.0)}

    @pytest.mark.parametrize("speech", [{**WAV, "voice": "nobody"}, {**WAV, "model": "nope"}], ids=["voice", "model"])
    def test_a_voice_or_model_the_tts_router_refuses_is_422_before_the_ai_is_asked(self, ask_api, no_shared_client, tts, speech):
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": speech}})
        assert response.status_code == 422, response.text
        assert (_error(response)["code"], _error(response)["param"]) == ("validation_error", "output.speech")
        assert server.requests == [] and tts.spoken == []

    def test_with_tts_off_speech_is_refused_before_the_ai_is_asked(self, ask_api, assistant_on, no_shared_client, monkeypatch):
        monkeypatch.setattr(assistant_on, "tts_enabled", False)
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": WAV}})
        assert response.status_code == 422 and _error(response)["param"] == "output.speech"
        assert "TTS" in _error(response)["message"] and server.requests == []

    def test_routes_built_without_a_tts_router_answer_speech_unavailable(self, assistant_on, no_shared_client):
        from fastapi import FastAPI

        from src.routes import assistant as assistant_routes

        app = FastAPI()
        app.include_router(assistant_routes.create_router(get_settings=lambda: assistant_on))
        server = _ai_server(no_shared_client)
        response = TestClient(app).post(ASK, json={"input": TEXT, "output": {"speech": WAV}})
        assert response.status_code == 503 and _error(response)["code"] == "speech_unavailable"
        assert server.requests == []

    def test_a_speech_failure_keeps_the_reply_and_says_why(self, ask_api, no_shared_client, tts):
        _ai_server(no_shared_client)
        tts.fail = RuntimeError("the GPU is busy")
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": WAV}})
        assert response.status_code == 200, response.text
        answer = response.json()
        assert answer["reply"] == "Hello there." and "audio" not in answer
        assert answer["audio_error"]["code"] == "speech_failed" and "RuntimeError" in answer["audio_error"]["message"]

    def test_a_reply_longer_than_the_tts_input_limit_is_not_spoken(self, ask_api, assistant_on, no_shared_client, tts, monkeypatch):
        monkeypatch.setattr(assistant_on, "tts_max_input_length", 20)
        _ai_server(no_shared_client, lambda _r: _answer(content="A reply of more than twenty characters."))
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": WAV}})
        assert response.status_code == 200, response.text
        error = response.json()["audio_error"]
        assert error["code"] == "speech_too_long" and "TTS_MAX_INPUT_LENGTH" in error["message"]
        assert "audio" not in response.json() and tts.spoken == []

    def test_spoken_sentences_keep_a_pause_between_them(self):
        """With TTS_TRIM_SILENCE on (the default), each sentence was trimmed to its first
        and last loud sample and joined to the next with nothing between, so the sentences ran together."""
        router = _PausingModel()
        samples, answer = _speak_reply(router, "The sky scatters blue light. Red light passes through. That is all.")
        assert router.texts == ["The sky scatters blue light.", "Red light passes through.", "That is all."]
        pauses = _pauses_inside(samples)
        assert len(pauses) == 2 and min(pauses) >= 0.1, f"pauses between the sentences: {pauses} s"

    def test_a_segment_the_dictionary_lengthens_past_the_models_limit_is_cut_again_and_spoken(self):
        """The dictionary made a segment longer than the model's input limit, the model
        refused it, and the whole spoken reply came back as audio_error speech_invalid. It is cut again, as
        /v1/realtime cuts it, and every piece is spoken, with a pause between pieces as between sentences."""
        router = _PausingModel(limit=60)
        reply = "An AI helps you think. AI tools save time for AI users today."
        samples, answer = _speak_reply(router, reply, dictionary=_Expand())
        assert "audio_error" not in answer, answer
        assert len(router.texts) >= 3 and all(len(text) <= 60 for text in router.texts), router.texts
        assert " ".join(router.texts).split() == _Expand().apply(reply).split()
        assert len(_pauses_inside(samples)) == len(router.texts) - 1

    def test_a_long_reply_is_refused_without_holding_the_event_loop(self, monkeypatch):
        """The whole reply was segmented on the event loop before its length was
        checked, so a 400,000-character reply froze the server for about 3.5 s and was refused anyway. Segmenting
        now runs on a thread and stops once the spoken text passes TTS_MAX_INPUT_LENGTH."""
        import threading
        import time

        from src.assistant import speech as speech_module

        fed: list[tuple[int, int]] = []  # (characters appended, thread)
        real_segmenter = speech_module.LiveTextSegmenter

        class _Watched(real_segmenter):
            def append(self, text):
                fed.append((len(text), threading.get_ident()))
                return super().append(text)

        monkeypatch.setattr(speech_module, "LiveTextSegmenter", _Watched)
        paragraph = ("The committee met on Tuesday to discuss the budget. Several members raised concerns about the "
                     "cost of the new building, which had risen by almost a third since the spring.\n\n")
        reply = (paragraph * (400_000 // len(paragraph) + 1))[:400_000]
        router = _PausingModel()
        gaps: list[float] = []
        # Built before the ticker starts: run alone, the first import of src.assistant.ask (over a second) would
        # otherwise be timed as the stall.
        speaker, speech = _speaker(router), _speech()

        async def run():
            loop_thread = threading.get_ident()
            stop = asyncio.Event()

            async def ticker():
                last = time.perf_counter()
                while not stop.is_set():
                    await asyncio.sleep(0.01)
                    now = time.perf_counter()
                    gaps.append(now - last)
                    last = now

            task = asyncio.create_task(ticker())
            await asyncio.sleep(0.05)
            try:
                return loop_thread, await asyncio.wait_for(speaker.speak(reply, speech), timeout=60)
            finally:
                stop.set()
                await task

        loop_thread, answer = asyncio.run(run())
        assert answer.get("audio_error", {}).get("code") == "speech_too_long", answer
        assert "TTS_MAX_INPUT_LENGTH" in answer["audio_error"]["message"] and router.texts == []
        assert fed and all(thread != loop_thread for _chars, thread in fed), "segmented on the event loop"
        assert sum(chars for chars, _thread in fed) <= 40_000, f"{sum(c for c, _t in fed):,} characters segmented"
        assert max(gaps) < 0.5, f"the event loop stalled {max(gaps):.2f} s"

    def test_the_models_input_limit_is_read_off_the_event_loop(self, monkeypatch):
        """Speaker._speak read the model's input limit on the event loop. For an
        isolated worker (Qwen3, Chatterbox, Fish) that is a blocking GET of its /v1/manifest once the 30 s cache
        is stale, as it always is after a slow AI reply, so a busy or hung worker froze the whole server."""
        import io
        import threading
        import time
        import urllib.error

        from src.tts.external import ExternalTTSBackend
        from src.tts.router import TTSRouter

        model = "qwen3/0.6b-custom-voice"
        manifest = {"schema_version": 1, "provider": "qwen3", "models": [{
            "id": model, "provider": "qwen3", "sample_rate": 24000, "revision": "test-1", "max_input_chars": 500,
            "capabilities": {"voice_clone": False, "instructions": False, "speed_control": False,
                             "live_reader": True, "languages": ["Auto", "English"]},
            "voices": [{"id": "Ryan", "name": "Ryan", "language": "en", "gender": "male"}],
        }]}
        reads: list[int] = []  # the thread of each manifest read

        class _Response(io.BytesIO):
            headers: dict = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        def worker(request, **_kwargs):
            if request.full_url.endswith("/v1/manifest"):
                reads.append(threading.get_ident())
                time.sleep(0.05)  # a busy worker
                return _Response(json.dumps(manifest).encode())
            raise urllib.error.URLError("this worker speaks nothing")  # fake: no speech needed

        monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
        router = TTSRouter(device="cpu", external_providers="")
        backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3:8200")
        monkeypatch.setattr(backend._opener, "open", worker)
        router.register_backend("qwen3", backend)
        speaker = _speaker(router)
        from src.assistant.ask import Speech

        speech = Speech(model=model, voice="Ryan", format="wav", speed=1.0)

        async def run():
            await asyncio.wait_for(asyncio.to_thread(speaker.check, speech), timeout=10)  # as handle() does
            checked = len(reads)
            backend._manifest_checked_at -= 31.0  # the AI took its time: the manifest cache is stale
            answer = await asyncio.wait_for(speaker.speak("Hello there. How are you?", speech), timeout=30)
            return threading.get_ident(), checked, answer

        loop_thread, checked, answer = asyncio.run(run())
        assert "audio_error" in answer, answer  # the fake worker speaks nothing
        assert len(reads) > checked, "speaking read no fresh manifest: the test no longer covers the stale cache"
        on_loop = sum(1 for thread in reads[checked:] if thread == loop_thread)
        assert on_loop == 0, f"{on_loop} manifest read(s) of the worker ran on the event loop"

    @pytest.mark.parametrize(("code", "status"), [
        ("assistant_unreachable", 502), ("assistant_timeout", 504), ("assistant_reply_cut_off", 502),
    ])
    def test_an_ai_error_answers_its_code(self, ask_api, no_shared_client, tts, code, status):
        no_shared_client.set_client(_FailingClient(code))
        response = ask_api.post(ASK, json={"input": TEXT})
        assert response.status_code == status
        assert _error(response)["code"] == code and isinstance(response.json()["elapsed_ms"], int)
        assert "sentence" not in response.json() and tts.spoken == []

    def test_with_speech_an_ai_error_is_spoken_in_the_presets_sentence(self, ask_api, no_shared_client, tts):
        no_shared_client.set_client(_FailingClient("assistant_timeout"))
        response = ask_api.post(ASK, json={
            "input": TEXT, "output": {"speech": WAV},
            "overrides": {"error_sentences": {"assistant_timeout": "Ada is slow today."}},
        })
        assert response.status_code == 504, response.text
        body = response.json()
        assert body["error"]["code"] == "assistant_timeout" and body["sentence"] == "Ada is slow today."
        assert base64.b64decode(body["audio"]["base64"])[:4] == b"RIFF"
        assert tts.texts == ["Ayda is slow today."]

    def test_with_speak_errors_off_an_ai_error_is_not_spoken(self, ask_api, no_shared_client, tts):
        no_shared_client.set_client(_FailingClient("assistant_unreachable"))
        response = ask_api.post(ASK, json={"input": TEXT, "output": {"speech": WAV}, "overrides": {"speak_errors": False}})
        assert response.status_code == 502
        assert not {"sentence", "audio"} & set(response.json()) and tts.spoken == []

    def test_an_error_never_shows_the_key_or_the_ai_servers_answer(self, ask_api, no_shared_client):
        server = _Server(lambda _r: httpx.Response(401, text=f"bad key {API_KEY} from the AI server"))
        no_shared_client.set_client(_client(server, api_key=API_KEY))
        response = ask_api.post(ASK, json={"input": TEXT})
        assert response.status_code == 502 and _error(response)["code"] == "assistant_http_error"
        assert API_KEY not in response.text and "from the AI server" not in response.text

    def test_while_the_assistant_is_off_it_is_not_found(self, ask_api, assistant_on, no_shared_client, monkeypatch):
        monkeypatch.setattr(assistant_on, "os_assistant_url", "")
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": TEXT})
        assert response.status_code == 404 and _error(response)["code"] == "assistant_disabled"
        assert server.requests == []

    def test_it_needs_the_api_key_when_one_is_set(self, ask_api, assistant_on, no_shared_client, monkeypatch):
        server = _ai_server(no_shared_client)
        monkeypatch.setattr(assistant_on, "os_api_key", "open-speech-key")
        assert ask_api.post(ASK, json={"input": TEXT}).status_code == 401
        allowed = ask_api.post(ASK, json={"input": TEXT}, headers={"Authorization": "Bearer open-speech-key"})
        assert allowed.status_code == 200 and len(server.requests) == 1

    def test_before_the_client_has_started_it_is_unavailable(self, ask_api):
        response = ask_api.post(ASK, json={"input": TEXT})
        assert response.status_code == 503 and _error(response)["code"] == "assistant_not_started"

    @pytest.mark.parametrize(("body", "param"), ASK_REFUSED)
    def test_an_invalid_field_gets_422_with_its_name_and_the_ai_is_not_asked(self, ask_api, no_shared_client, body, param):
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json=body)
        assert response.status_code == 422, response.text
        error = _error(response)
        assert (error["code"], error["param"]) == ("validation_error", param), error
        assert server.requests == []

    def test_a_body_that_is_not_an_object_gets_422(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json=["input"])
        assert response.status_code == 422 and _error(response)["param"] == "body"
        assert server.requests == []

    def test_a_lone_surrogate_in_the_text_gets_422(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        raw = b'{"input": {"kind": "text", "text": "half an emoji \\ud83d"}}'
        response = ask_api.post(ASK, content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422 and _error(response)["param"] == "input.text"
        assert server.requests == []

    # A question of your own goes with any input, before
    # its text, and the answer counts it but never shows it. Documents and jobs are in tests/test_reader_assistant.py.

    @pytest.mark.parametrize("kind", ["text", "transcript"])
    def test_a_question_goes_before_the_text_and_the_answer_counts_it(self, ask_api, no_shared_client, kind):
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": {**TEXT, "kind": kind}, "question": "Explain this to me."})
        assert response.status_code == 200, response.text
        assert server.body["messages"] == [
            {"role": "system", "content": VOICE_PROMPT},
            {"role": "user", "content": "Explain this to me.\n\nWhy is the sky blue?"},
        ]
        assert response.json()["input"] == {"kind": kind, "chars": 20, "question_chars": 19}
        assert "Explain this" not in json.dumps(response.json()["input"])

    def test_a_question_goes_before_the_rendered_template_as_typed(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        question = "  Keep it short,\nplease. 🙂 "
        response = ask_api.post(ASK, json={
            "preset": "book", "input": {"kind": "transcript", "text": "We met at noon."}, "question": question,
            "overrides": {"user_template": "Summarize: {text}"},
        })
        assert response.status_code == 200, response.text
        assert server.body["messages"][-1] == {"role": "user", "content": f"{question}\n\nSummarize: We met at noon."}
        assert response.json()["input"]["question_chars"] == len(question) == 27

    def test_a_question_of_2000_characters_is_sent_and_without_one_nothing_changes(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        longest = "🙂" * 2000  # characters, not UTF-16 units or bytes
        sent = ask_api.post(ASK, json={"input": TEXT, "question": longest})
        assert sent.status_code == 200, sent.text
        assert server.body["messages"][-1]["content"] == f"{longest}\n\nWhy is the sky blue?"
        for body in ({"input": TEXT}, {"input": TEXT, "question": None}):
            plain = ask_api.post(ASK, json=body)
            assert plain.status_code == 200, plain.text
            assert server.body["messages"][-1]["content"] == "Why is the sky blue?"
            assert plain.json()["input"] == {"kind": "text", "chars": 20}

    def test_a_lone_surrogate_in_the_question_gets_422(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        raw = b'{"input": {"kind": "text", "text": "Why?"}, "question": "half an emoji \\ud83d"}'
        response = ask_api.post(ASK, content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422 and _error(response)["param"] == "question"
        assert server.requests == []

    def test_text_longer_than_the_reader_takes_gets_422(self, ask_api, assistant_on, no_shared_client, monkeypatch):
        monkeypatch.setattr(assistant_on, "reader_max_document_chars", 100)
        server = _ai_server(no_shared_client)
        response = ask_api.post(ASK, json={"input": {"kind": "text", "text": "word " * 25}})
        assert response.status_code == 422 and _error(response)["param"] == "input.text"
        assert "100" in _error(response)["message"] and server.requests == []

    def test_only_stream_false_for_now_and_the_body_has_room_for_true(self, ask_api, no_shared_client):
        server = _ai_server(no_shared_client)
        assert ask_api.post(ASK, json={"input": TEXT, "stream": False}).status_code == 200
        refused = ask_api.post(ASK, json={"input": TEXT, "stream": True})
        assert refused.status_code == 422 and _error(refused)["param"] == "stream"
        assert len(server.requests) == 1

    def test_with_the_reader_off_a_document_or_a_long_text_is_refused(self, ask_api, assistant_on, no_shared_client, monkeypatch):
        monkeypatch.setattr(assistant_on, "reader_enabled", False)
        server = _ai_server(no_shared_client)
        for body, param in (
            ({"input": {"kind": "document", "document_id": "abc"}}, "input.kind"),
            ({"input": TEXT, "output": {"document": {}}}, "output.document"),
            ({"input": {"kind": "text", "text": "word " * 300}, "overrides": {"chunk_chars": 1000}}, "input.text"),
        ):
            response = ask_api.post(ASK, json=body)
            assert response.status_code == 422, (body, response.text)
            assert _error(response)["param"] == param, body
        assert server.requests == []
