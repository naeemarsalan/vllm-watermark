"""Health-safe fake-transport tests for D10 concrete HTTP adapters."""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from validation.gateway import ConfigurationError, RetryableValidationError, TerminalValidationError, ValidationRequest
from validation.http_clients import DirectDetectorClient, HttpClientSettings, ManagedNeMo021Client, OpenAIUpstreamClient
from validation.main import RuntimeConfig, create_runtime_app


VALIDATION_ID = "123e4567-e89b-12d3-a456-426614174000"
TEXT = "redacted local test content"
DIGEST = hashlib.sha256(TEXT.encode()).hexdigest()


def run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def request() -> ValidationRequest:
    return ValidationRequest(VALIDATION_ID, "cmpl-local", TEXT, DIGEST, True, "kgw", "test-key")


def transport(handler):
    return httpx.MockTransport(handler)


def test_detector_sends_canonical_correlation_and_validates_full_echo() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://detector.example/v1/watermark/detect"
        assert req.headers["authorization"] == "Bearer detector-token"
        assert req.headers["x-watermark-validation-id"] == VALIDATION_ID
        body = __import__("json").loads(req.content)
        assert body == {"text": TEXT, "validation_id": VALIDATION_ID, "response_id": "cmpl-local", "scheme": "kgw", "key_id": "test-key"}
        return httpx.Response(200, json={"validation_id": VALIDATION_ID, "response_id": "cmpl-local", "content_sha256": DIGEST, "scheme": "kgw", "key_id": "test-key", "verdict": True})

    client = DirectDetectorClient("https://detector.example/v1/watermark/detect", "detector-token", transport=transport(handler))
    assert run(client.detect(request())) == "watermarked"
    assert "detector-token" not in repr(client)


def test_internal_clients_omit_authorization_when_token_is_absent() -> None:
    def upstream_handler(req: httpx.Request) -> httpx.Response:
        assert "authorization" not in req.headers
        return httpx.Response(200, json={"choices": []})

    upstream = OpenAIUpstreamClient("https://upstream.example", transport=transport(upstream_handler))
    assert run(upstream.complete("/v1/completions", {"model": "test"})) == {"choices": []}
    assert "Authorization" not in repr(upstream)

    def detector_handler(req: httpx.Request) -> httpx.Response:
        assert "authorization" not in req.headers
        return httpx.Response(200, json={
            "validation_id": VALIDATION_ID,
            "response_id": "cmpl-local",
            "content_sha256": DIGEST,
            "scheme": "kgw",
            "key_id": "test-key",
            "verdict": False,
        })

    detector = DirectDetectorClient(
        "https://detector.example/v1/watermark/detect", transport=transport(detector_handler)
    )
    assert run(detector.detect(request())) == "clean"
    assert "Authorization" not in repr(detector)


@pytest.mark.parametrize("client_factory", [
    lambda: OpenAIUpstreamClient("https://upstream.example", ""),
    lambda: DirectDetectorClient("https://detector.example/v1/watermark/detect", ""),
])
def test_internal_client_rejects_explicit_empty_token(client_factory) -> None:
    with pytest.raises(ConfigurationError, match="TOKEN"):
        client_factory()


@pytest.mark.parametrize("payload", [
    {"validation_id": VALIDATION_ID, "response_id": "cmpl-local", "content_sha256": "0" * 64, "scheme": "kgw", "key_id": "test-key", "verdict": False},
    {"validation_id": VALIDATION_ID, "response_id": "cmpl-local", "content_sha256": DIGEST, "scheme": "kgw", "key_id": "test-key", "verdict": "false"},
    {"validation_id": VALIDATION_ID, "response_id": "cmpl-wrong", "content_sha256": DIGEST, "scheme": "kgw", "key_id": "test-key", "verdict": False},
])
def test_detector_rejects_mismatched_or_non_boolean_result(payload: dict[str, object]) -> None:
    client = DirectDetectorClient(
        "https://detector.example/v1/watermark/detect",
        "detector-token",
        transport=transport(lambda _request: httpx.Response(200, json=payload)),
    )
    with pytest.raises(TerminalValidationError):
        run(client.detect(request()))


def test_nemo_021_uses_exact_checks_path_assistant_and_context() -> None:
    context = {
        "watermark_validation_id": VALIDATION_ID,
        "watermark_response_id": "cmpl-local",
        "watermark_content_sha256": DIGEST,
        "watermark_scheme": "kgw",
        "watermark_key_id": "test-key",
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert str(req.url) == "https://nemo.example/v1/guardrail/checks"
        assert req.headers["authorization"] == "Bearer nemo-token"
        assert req.headers["x-watermark-validation-id"] == VALIDATION_ID
        assert req.headers["x-watermark-response-id"] == "cmpl-local"
        assert req.headers["x-watermark-content-sha256"] == DIGEST
        assert req.headers["x-watermark-scheme"] == "kgw"
        assert req.headers["x-watermark-key-id"] == "test-key"
        assert __import__("json").loads(req.content) == {
            "model": "watermark-vllm",
            "messages": [{"role": "assistant", "content": TEXT}],
            "guardrails": {"config_id": "watermark-validation", "context": context},
        }
        return httpx.Response(200, json={
            "status": "blocked",
            "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": "blocked"}}}],
            "rails_status": {"watermark check": {"status": "blocked"}},
        })

    client = ManagedNeMo021Client("https://nemo.example", "nemo-token", "watermark-validation", "watermark-vllm", transport=transport(handler))
    assert run(client.validate(TEXT, context)) == "blocked"
    assert "nemo-token" not in repr(client)


@pytest.mark.parametrize("payload", [
    {"status": "passed", "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": "passed"}}}], "rails_status": {"watermark check": {"status": "passed"}}},
    {},
    {"status": True, "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": True}}}], "rails_status": {"watermark check": {"status": True}}},
    {"status": "success", "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": "blocked"}}}], "rails_status": {"watermark check": {"status": "success"}}},
    {"status": "success", "messages": [], "rails_status": {"watermark check": {"status": "success"}}},
    {"status": "success", "messages": [{"index": 1, "role": "assistant", "rails": {"watermark check": {"status": "success"}}}], "rails_status": {"watermark check": {"status": "success"}}},
    {"status": "success", "messages": [{"index": 0, "role": "user", "rails": {"watermark check": {"status": "success"}}}], "rails_status": {"watermark check": {"status": "success"}}},
    {"status": "success", "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": "success", "extra": "invalid"}}}], "rails_status": {"watermark check": {"status": "success"}}},
    {"status": "success", "messages": [{"index": 0, "role": "assistant", "rails": {"watermark check": {"status": "success"}}}], "rails_status": {"watermark check": {"status": "blocked"}}},
])
def test_nemo_rejects_invalid_live_response_shapes(payload: dict[str, object]) -> None:
    client = ManagedNeMo021Client(
        "https://nemo.example", "nemo-token", "cfg", "model", transport=transport(lambda _request: httpx.Response(200, json=payload))
    )
    with pytest.raises(TerminalValidationError):
        run(client.validate(TEXT, {
            "watermark_validation_id": VALIDATION_ID,
            "watermark_response_id": "cmpl-local",
            "watermark_content_sha256": DIGEST,
            "watermark_scheme": "kgw",
            "watermark_key_id": "test-key",
        }))


@pytest.mark.parametrize(("field", "value"), [
    ("watermark_response_id", "invalid response id"),
    ("watermark_content_sha256", "not-a-digest"),
    ("watermark_scheme", "unknown"),
    ("watermark_key_id", "unsafe/key"),
])
def test_nemo_rejects_unbounded_or_malformed_header_correlation_before_transport(field: str, value: str) -> None:
    context = {
        "watermark_validation_id": VALIDATION_ID,
        "watermark_response_id": "cmpl-local",
        "watermark_content_sha256": DIGEST,
        "watermark_scheme": "kgw",
        "watermark_key_id": "test-key",
    }
    context[field] = value
    client = ManagedNeMo021Client(
        "https://nemo.example", "nemo-token", "cfg", "model",
        transport=transport(lambda _request: pytest.fail("malformed context reached transport")),
    )
    with pytest.raises(TerminalValidationError):
        run(client.validate(TEXT, context))


def test_retryable_and_bounded_responses_do_not_echo_remote_body() -> None:
    client = OpenAIUpstreamClient("https://upstream.example", "upstream-token", transport=transport(lambda _request: httpx.Response(503, content="untrusted body")))
    with pytest.raises(RetryableValidationError, match="retryable HTTP status"):
        run(client.complete("/v1/completions", {"model": "test"}))

    limited = OpenAIUpstreamClient(
        "https://upstream.example", "upstream-token", HttpClientSettings(max_response_bytes=8),
        transport=transport(lambda _request: httpx.Response(200, content=b'{"large": true}')),
    )
    with pytest.raises(TerminalValidationError, match="size limit"):
        run(limited.complete("/v1/completions", {"model": "test"}))


def complete_env(tmp_path) -> dict[str, str]:
    return {
        "VALIDATION_SAMPLE_EVERY": "2",
        "VALIDATION_SAMPLER_DB_PATH": str(tmp_path / "sampler.sqlite"),
        "VALIDATION_REPLICA_COUNT": "1",
        "VALIDATION_BROKER_TOKEN": "broker-token",
        "VALIDATION_ADMIN_TOKEN": "admin-token",
        "VALIDATION_UPSTREAM_URL": "https://upstream.example",
        "VALIDATION_UPSTREAM_TOKEN": "upstream-token",
        "VALIDATION_DETECTOR_URL": "https://detector.example/v1/watermark/detect",
        "VALIDATION_DETECTOR_TOKEN": "detector-token",
        "VALIDATION_NEMO_URL": "https://nemo.example",
        "VALIDATION_NEMO_TOKEN": "nemo-token",
        "VALIDATION_NEMO_CONFIG_ID": "watermark-validation",
        "VALIDATION_NEMO_MODEL": "watermark-vllm",
    }


@pytest.mark.parametrize("field", ["VALIDATION_UPSTREAM_URL", "VALIDATION_DETECTOR_URL", "VALIDATION_NEMO_URL", "VALIDATION_NEMO_TOKEN", "VALIDATION_NEMO_CONFIG_ID", "VALIDATION_NEMO_MODEL"])
def test_runtime_configuration_requires_every_remote_url_token_and_config(tmp_path, field: str) -> None:
    env = complete_env(tmp_path)
    del env[field]
    with pytest.raises(ConfigurationError, match=field):
        RuntimeConfig.from_environment(env)


@pytest.mark.parametrize("field", ["VALIDATION_UPSTREAM_TOKEN", "VALIDATION_DETECTOR_TOKEN"])
def test_runtime_configuration_rejects_explicit_empty_internal_token(tmp_path, field: str) -> None:
    env = complete_env(tmp_path)
    env[field] = "  "
    with pytest.raises(ConfigurationError, match=field):
        RuntimeConfig.from_environment(env)


def test_runtime_configuration_allows_absent_internal_tokens(tmp_path) -> None:
    env = complete_env(tmp_path)
    del env["VALIDATION_UPSTREAM_TOKEN"]
    del env["VALIDATION_DETECTOR_TOKEN"]
    runtime = RuntimeConfig.from_environment(env)
    assert runtime.upstream_token is None
    assert runtime.detector_token is None


def test_tls_verification_cannot_be_disabled_and_runtime_assembles_app(tmp_path) -> None:
    env = complete_env(tmp_path)
    assert RuntimeConfig.from_environment(env).http_settings.verify_tls is True
    env["VALIDATION_TLS_VERIFY"] = "false"
    with pytest.raises(ConfigurationError, match="VALIDATION_TLS_VERIFY"):
        RuntimeConfig.from_environment(env)
    env["VALIDATION_TLS_VERIFY"] = "perhaps"
    with pytest.raises(ConfigurationError, match="VALIDATION_TLS_VERIFY"):
        RuntimeConfig.from_environment(env)
    env["VALIDATION_TLS_VERIFY"] = "true"
    app = create_runtime_app(env, transport=transport(lambda _request: httpx.Response(500)))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200


def test_runtime_pins_one_uvicorn_process(monkeypatch) -> None:
    import validation.main as runtime_main

    app = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_main.RuntimeConfig,
        "from_environment",
        classmethod(lambda cls: SimpleNamespace(host="127.0.0.1", port=8080)),
    )
    monkeypatch.setattr(runtime_main, "create_runtime_app", lambda: app)
    monkeypatch.setattr(
        runtime_main.uvicorn,
        "run",
        lambda passed_app, **kwargs: captured.update(app=passed_app, **kwargs),
    )

    runtime_main.main()

    assert captured == {
        "app": app,
        "host": "127.0.0.1",
        "port": 8080,
        "log_level": "info",
        "workers": 1,
    }
