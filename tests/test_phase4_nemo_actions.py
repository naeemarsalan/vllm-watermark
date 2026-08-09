"""Local static tests for the embedded Phase 4 managed-NeMo action.

These tests execute no cluster, GPU, or network operation.  They extract the
ConfigMap's actions.py and replace urlopen with an in-process fake.
"""

import asyncio
from contextvars import ContextVar
import json
import sys
import types
from pathlib import Path
from urllib.error import URLError

import pytest


_ACTIONS_PATH = Path("deploy/phase4/30-nemo-watermark-config.yaml")
_VALIDATION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _headers(**overrides):
    result = {
        "x-watermark-validation-id": _VALIDATION_ID,
        "x-watermark-response-id": "chatcmpl-test-123",
        "x-watermark-content-sha256": "a" * 64,
        "x-watermark-scheme": "kgw",
        "x-watermark-key-id": "test-key-1",
    }
    result.update(overrides)
    return result


_API_REQUEST_HEADERS = ContextVar(
    "api_request_headers", default=_headers()
)


def _load_actions_namespace() -> dict:
    source = _ACTIONS_PATH.read_text().split("  actions.py: |\n", 1)[1]
    actions_source = "\n".join(
        line[4:] if line.startswith("    ") else line
        for line in source.splitlines()
    )
    actions_module = types.ModuleType("nemoguardrails.actions")
    actions_module.action = lambda **_kwargs: lambda func: func
    package = types.ModuleType("nemoguardrails")
    server_package = types.ModuleType("nemoguardrails.server")
    server_api_module = types.ModuleType("nemoguardrails.server.api")
    server_api_module.api_request_headers = _API_REQUEST_HEADERS
    server_package.api = server_api_module
    previous = {
        name: sys.modules.get(name)
        for name in (
            "nemoguardrails",
            "nemoguardrails.actions",
            "nemoguardrails.server",
            "nemoguardrails.server.api",
        )
    }
    sys.modules["nemoguardrails"] = package
    sys.modules["nemoguardrails.actions"] = actions_module
    sys.modules["nemoguardrails.server"] = server_package
    sys.modules["nemoguardrails.server.api"] = server_api_module
    namespace = {"__name__": "phase4_embedded_actions"}
    try:
        exec(actions_source, namespace)
    finally:
        for name, module in previous.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module
    return namespace


@pytest.fixture
def actions(monkeypatch):
    monkeypatch.setenv(
        "WATERMARK_VALIDATION_BROKER_URL",
        "http://broker.internal/internal/v1/guardrail-action",
    )
    monkeypatch.setenv("WATERMARK_VALIDATION_BROKER_TOKEN", "dummy-test-token")
    monkeypatch.setenv("WATERMARK_VALIDATION_FAILURE_POLICY", "closed")
    token = _API_REQUEST_HEADERS.set(_headers())
    try:
        yield _load_actions_namespace()
    finally:
        _API_REQUEST_HEADERS.reset(token)


def _context(**overrides):
    result = {
        "bot_message": "redacted test response",
        "watermark_validation_id": _VALIDATION_ID,
        "watermark_response_id": "chatcmpl-test-123",
        "watermark_content_sha256": "a" * 64,
        "watermark_scheme": "kgw",
        "watermark_key_id": "test-key-1",
    }
    result.update(overrides)
    return result


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _run(actions, context):
    return asyncio.run(actions["watermark_check"](context))


@pytest.mark.parametrize("verdict", [True, False])
def test_broker_positive_and_clean_echo_metadata(actions, verdict):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        payload = dict(captured["body"])
        payload["verdict"] = verdict
        return _Response(payload)

    actions["urlopen"] = fake_urlopen
    assert _run(actions, _context()) is verdict
    assert captured["timeout"] == 30
    assert captured["authorization"] == "Bearer dummy-test-token"
    assert captured["body"] == {
        "validation_id": "123e4567-e89b-12d3-a456-426614174000",
        "response_id": "chatcmpl-test-123",
        "content_sha256": "a" * 64,
        "scheme": "kgw",
        "key_id": "test-key-1",
    }
    assert "text" not in captured["body"]


@pytest.mark.parametrize("bot_message", ["", " \t\n ", "水印 text — 🚀"])
def test_bot_message_is_not_sent_to_broker(actions, bot_message):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({**captured["body"], "verdict": False})

    actions["urlopen"] = fake_urlopen
    assert _run(actions, _context(bot_message=bot_message)) is False
    assert captured["body"] == {
        "validation_id": _VALIDATION_ID,
        "response_id": "chatcmpl-test-123",
        "content_sha256": "a" * 64,
        "scheme": "kgw",
        "key_id": "test-key-1",
    }
    assert "text" not in captured["body"]


@pytest.mark.parametrize("context", [{}, {"bot_message": None}, {"bot_message": 123}, {"bot_message": []}])
def test_missing_or_non_string_bot_message_fails_closed_without_broker_call(
    actions, context
):
    called = False

    def unexpected_urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("broker must not be called for an invalid bot message")

    actions["urlopen"] = unexpected_urlopen
    assert _run(actions, context) is True
    assert called is False


@pytest.mark.parametrize("payload", [
    {"verdict": True},
    {"verdict": "true", "validation_id": "123e4567-e89b-12d3-a456-426614174000", "response_id": "chatcmpl-test-123",
     "content_sha256": "a" * 64, "scheme": "kgw", "key_id": "test-key-1"},
    {"verdict": True, "validation_id": "123e4567-e89b-12d3-a456-426614174001", "response_id": "chatcmpl-test-123",
     "content_sha256": "a" * 64, "scheme": "kgw", "key_id": "test-key-1"},
    {"verdict": True, "validation_id": "123e4567-e89b-12d3-a456-426614174000", "response_id": "chatcmpl-other",
     "content_sha256": "a" * 64, "scheme": "kgw", "key_id": "test-key-1"},
    {"verdict": False, "validation_id": "123e4567-e89b-12d3-a456-426614174000", "response_id": "chatcmpl-test-123",
     "content_sha256": "a" * 64, "scheme": "kgw", "key_id": "test-key-1", "extra": "reject-me"},
])
def test_malformed_or_mismatched_broker_reply_uses_closed_policy(actions, payload):
    actions["urlopen"] = lambda *_args, **_kwargs: _Response(payload)
    assert _run(actions, _context()) is True


def test_transport_error_uses_open_policy(actions, monkeypatch):
    monkeypatch.setenv("WATERMARK_VALIDATION_FAILURE_POLICY", "open")
    actions["urlopen"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("test"))
    assert _run(actions, _context()) is False


def test_missing_context_uses_closed_policy(actions):
    assert _run(actions, None) is True


@pytest.mark.parametrize("field, value", [
    ("x-watermark-validation-id", "123E4567-e89b-12d3-a456-426614174000"),
    ("x-watermark-response-id", "bad response id"),
    ("x-watermark-content-sha256", "A" * 64),
    ("x-watermark-scheme", "KGW"),
    ("x-watermark-key-id", "bad key"),
])
def test_invalid_correlation_header_uses_closed_policy(actions, field, value):
    token = _API_REQUEST_HEADERS.set(_headers(**{field: value}))
    try:
        assert _run(actions, _context()) is True
    finally:
        _API_REQUEST_HEADERS.reset(token)


def test_missing_broker_url_uses_open_policy(actions, monkeypatch):
    monkeypatch.delenv("WATERMARK_VALIDATION_BROKER_URL")
    monkeypatch.setenv("WATERMARK_VALIDATION_FAILURE_POLICY", "open")
    assert _run(actions, _context()) is False


def test_missing_broker_token_uses_closed_policy(actions, monkeypatch):
    monkeypatch.delenv("WATERMARK_VALIDATION_BROKER_TOKEN")
    assert _run(actions, _context()) is True


@pytest.mark.parametrize("header_mutation", [
    lambda headers: headers.pop("x-watermark-key-id"),
    lambda headers: headers.update({"X-Watermark-Key-Id": "test-key-1"}),
    lambda headers: headers.update({"x-watermark-key-id": "test-key-1,other"}),
])
def test_missing_or_ambiguous_header_fails_closed_without_broker_call(actions, header_mutation):
    headers = _headers()
    header_mutation(headers)
    token = _API_REQUEST_HEADERS.set(headers)
    called = False

    def unexpected_urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("broker must not be called for invalid request headers")

    actions["urlopen"] = unexpected_urlopen
    try:
        assert _run(actions, _context()) is True
    finally:
        _API_REQUEST_HEADERS.reset(token)
    assert called is False


def test_unavailable_header_contextvar_fails_closed_without_broker_call(actions):
    actions["_api_request_headers"] = None
    called = False

    def unexpected_urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("broker must not be called without NeMo request headers")

    actions["urlopen"] = unexpected_urlopen
    assert _run(actions, _context()) is True
    assert called is False


def test_request_headers_are_source_not_action_context(actions):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        payload = dict(captured["body"])
        payload["verdict"] = False
        return _Response(payload)

    actions["urlopen"] = fake_urlopen
    conflicting_context = _context(
        watermark_validation_id="123e4567-e89b-12d3-a456-426614174001",
        watermark_response_id="wrong-response",
        watermark_content_sha256="b" * 64,
        watermark_scheme="synthid",
        watermark_key_id="wrong-key",
    )
    assert _run(actions, conflicting_context) is False
    assert captured["body"]["validation_id"] == _VALIDATION_ID
    assert captured["body"]["response_id"] == "chatcmpl-test-123"
    assert captured["body"]["content_sha256"] == "a" * 64
    assert captured["body"]["scheme"] == "kgw"
    assert captured["body"]["key_id"] == "test-key-1"


def test_broker_token_whitespace_is_stripped_before_authorization(actions, monkeypatch):
    monkeypatch.setenv("WATERMARK_VALIDATION_BROKER_TOKEN", "\n  dummy-test-token  \t")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        body = json.loads(request.data.decode("utf-8"))
        payload = dict(body)
        payload["verdict"] = False
        return _Response(payload)

    actions["urlopen"] = fake_urlopen
    assert _run(actions, _context()) is False
    assert captured["authorization"] == "Bearer dummy-test-token"


def test_whitespace_only_broker_token_fails_closed_without_request(actions, monkeypatch):
    monkeypatch.setenv("WATERMARK_VALIDATION_BROKER_TOKEN", " \n\t ")
    called = False

    def unexpected_urlopen(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("broker must not be called with a blank token")

    actions["urlopen"] = unexpected_urlopen
    assert _run(actions, _context()) is True
    assert called is False
