"""Fake-transport tests for the D10 synchronous validation gateway."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import stat
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from validation.gateway import (
    ConfigurationError,
    GatewayConfig,
    GatewayService,
    ManagedNeMoAdapter,
    PendingValidationBroker,
    PositiveValidationBlock,
    PrometheusMetrics,
    RetryableValidationError,
    TerminalValidationError,
    UnsafeRequest,
    ValidationOutcome,
    ValidationDeliveryBlocked,
    ValidationRequest,
    create_app,
)


class FakeUpstream:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def complete(self, endpoint: str, request: dict[str, object]) -> dict[str, object]:
        self.calls.append((endpoint, request))
        number = len(self.calls)
        if endpoint == "/v1/completions":
            return {"id": f"cmpl-{number}", "choices": [{"text": f"generated-{number}"}]}
        return {"id": f"chat-{number}", "choices": [{"message": {"content": f"generated-{number}"}}]}


class FakeValidator:
    def __init__(self, results: list[object] | None = None) -> None:
        self.results = results or [ValidationOutcome("watermarked", "block")]
        self.calls: list[ValidationRequest] = []

    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        self.calls.append(request)
        result = self.results.pop(0) if self.results else ValidationOutcome("clean", "pass")
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]


class FakeDetector:
    def __init__(self, verdict: str = "clean") -> None:
        self.requests: list[ValidationRequest] = []
        self.verdict = verdict

    async def detect(self, request: ValidationRequest) -> str:
        self.requests.append(request)
        return self.verdict


class BrokerCallingNeMo:
    def __init__(self, broker: PendingValidationBroker, *, force_action: str | None = None) -> None:
        self.broker = broker
        self.force_action = force_action

    async def validate(self, text: str, context: dict[str, str]) -> str:
        result = await self.broker.resolve({
            "validation_id": context["watermark_validation_id"],
            "response_id": context["watermark_response_id"],
            "content_sha256": context["watermark_content_sha256"],
            "scheme": context["watermark_scheme"],
            "key_id": context["watermark_key_id"],
        })
        return self.force_action or ("blocked" if result["verdict"] else "success")


class BlockingValidator:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[ValidationRequest] = []

    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        self.calls.append(request)
        self.entered.set()
        await self.release.wait()
        return ValidationOutcome("clean", "pass")


class SlowValidator:
    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        await asyncio.sleep(1)
        return ValidationOutcome("clean", "pass")


class MatrixValidator:
    """Deterministic detector/guardrails fake for the fixed D10 matrices."""

    def __init__(self) -> None:
        self.calls: list[ValidationRequest] = []

    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        self.calls.append(request)
        return ValidationOutcome("watermarked" if request.expected_enabled else "clean", "blocked" if request.expected_enabled else "success")


def body(*, watermark: object = "off") -> dict[str, object]:
    return {
        "model": "test",
        "n": 1,
        "vllm_xargs": {
            "watermark": watermark,
            "watermark_scheme": "kgw",
            "watermark_key_id": "control-key",
        },
    }


def matrix_body(enabled: bool, scheme: str, *, request_id: str | None = None) -> dict[str, object]:
    result = body(watermark="on" if enabled else "off")
    result["vllm_xargs"] = {
        "watermark": "on" if enabled else "off",
        "watermark_scheme": scheme,
        "watermark_key_id": "control-key",
    }
    if request_id is not None:
        result["metadata"] = {"watermark_validation_request_id": request_id}
    return result


def run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def config(tmp_path: Path, *, every: int = 1, policy: str = "closed") -> GatewayConfig:
    return GatewayConfig(sample_every=every, sqlite_path=tmp_path / "pvc" / "sampler.sqlite", failure_policy=policy)  # type: ignore[arg-type]


def test_completed_response_ordinals_are_persistent_and_sampled(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream, validator = FakeUpstream(), FakeValidator()
        service = GatewayService(config(tmp_path, every=2), upstream, validator)
        await service.start()
        try:
            await service.proxy("/v1/completions", body())
            await service.proxy("/v1/chat/completions", body(watermark=True))
            assert len(validator.calls) == 1
            selected = validator.calls[0]
            assert selected.expected_enabled is True
            assert selected.content_sha256 == hashlib.sha256(b"generated-2").hexdigest()
            assert selected.content not in service.metrics.render()
        finally:
            await service.stop()

        restarted = GatewayService(config(tmp_path, every=2), upstream, validator)
        await restarted.start()
        try:
            await restarted.proxy("/v1/completions", body())
            await restarted.proxy("/v1/completions", body())
            assert len(validator.calls) == 2, "ordinal 4, not ordinal 3, is selected after restart"
            metrics = restarted.metrics.render()
            assert "validation_requests_total{outcome=\"started\"} 2" in metrics
            assert "validation_requests_total{outcome=\"completed\"} 2" in metrics
            assert "validation_selected_total 1" in metrics
            assert "validation_unsampled_total 1" in metrics
            assert "validation_terminal_total{verdict=\"clean\"} 1" in metrics
        finally:
            await restarted.stop()

        db = sqlite3.connect(tmp_path / "pvc" / "sampler.sqlite")
        rows = db.execute("SELECT content_sha256,expected_enabled,scheme,key_id FROM validation_records ORDER BY ordinal").fetchall()
        assert len(rows) == 2
        assert all("generated" not in repr(row) for row in rows)

    run(scenario())


def test_restart_terminalizes_pending_record_and_harness_can_reset(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = config(tmp_path)
        first = GatewayService(settings, FakeUpstream(), FakeValidator())
        await first.start()
        try:
            request = ValidationRequest(
                validation_id=str(uuid.uuid4()), response_id="cmpl-pending", content="transient-only",
                content_sha256=hashlib.sha256(b"transient-only").hexdigest(), expected_enabled=False,
                scheme="kgw", key_id="control-key",
            )
            ordinal, selected = first.sampler.claim(request, 1)
            assert (ordinal, selected) == (1, True)
        finally:
            await first.stop()

        restarted = GatewayService(settings, FakeUpstream(), FakeValidator())
        await restarted.start()
        try:
            records = restarted.records()
            assert records[0]["status"] == "restart_interrupted"
            assert records[0]["verdict"] == "error"
            assert records[0]["delivery_outcome"] == "restart_interrupted"
            assert "transient-only" not in repr(records)
        finally:
            await restarted.stop()
        restarted.reset_sampler_for_harness()
        restarted.sampler.start()
        try:
            assert restarted.records() == []
        finally:
            restarted.sampler.close()

    run(scenario())


def test_retry_is_one_selected_record_and_fail_open_is_explicit(tmp_path: Path) -> None:
    async def scenario() -> None:
        validator = FakeValidator([RetryableValidationError("temporary"), ValidationOutcome("clean", "pass")])
        service = GatewayService(config(tmp_path, policy="open"), FakeUpstream(), validator)
        await service.start()
        try:
            response = await service.proxy("/v1/completions", body())
            assert response["id"] == "cmpl-1"
            assert len(validator.calls) == 2
            metrics = service.metrics.render()
            assert "validation_retries_total 1" in metrics
            assert "validation_terminal_total{verdict=\"clean\"} 1" in metrics
        finally:
            await service.stop()

    run(scenario())


def test_fail_closed_withholds_selected_response(tmp_path: Path) -> None:
    async def scenario() -> None:
        validator = FakeValidator([TerminalValidationError("bad schema")])
        service = GatewayService(config(tmp_path, policy="closed"), FakeUpstream(), validator)
        await service.start()
        try:
            with pytest.raises(TerminalValidationError, match="blocked"):
                await service.proxy("/v1/completions", body())
            assert "validation_fail_closed_blocks_total 1" in service.metrics.render()
        finally:
            await service.stop()

    run(scenario())


def test_queue_capacity_has_an_explicit_fail_open_terminal_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        validator = BlockingValidator()
        settings = GatewayConfig(sample_every=1, sqlite_path=tmp_path / "pvc" / "sampler.sqlite", failure_policy="open", queue_capacity=1, max_inflight=1)
        service = GatewayService(settings, FakeUpstream(), validator)
        await service.start()
        try:
            first = asyncio.create_task(service.proxy("/v1/completions", body()))
            await validator.entered.wait()
            third = await service.proxy("/v1/completions", body())
            assert third["id"] == "cmpl-2"
            assert "validation_dropped_items_total{reason=\"queue_full\"} 1" in service.metrics.render()
            validator.release.set()
            await first
        finally:
            await service.stop()

    run(scenario())


def test_each_validation_attempt_has_a_finite_timeout(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = GatewayConfig(
            sample_every=1,
            sqlite_path=tmp_path / "pvc" / "sampler.sqlite",
            max_attempts=1,
            attempt_timeout_seconds=0.001,
        )
        service = GatewayService(settings, FakeUpstream(), SlowValidator())
        await service.start()
        try:
            with pytest.raises(TerminalValidationError, match="blocked"):
                await service.proxy("/v1/completions", body())
            assert service.records()[0]["attempts"] == 1
            assert service.records()[0]["status"] == "error"
        finally:
            await service.stop()

    run(scenario())


def test_selected_client_cancellation_is_explicit_and_validation_terminalizes_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        validator = BlockingValidator()
        service = GatewayService(GatewayConfig(sample_every=1, sqlite_path=tmp_path / "cancel.sqlite"), FakeUpstream(), validator)
        await service.start()
        try:
            pending = asyncio.create_task(service.proxy("/v1/completions", body()))
            await validator.entered.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert service.records()[0]["delivery_outcome"] == "client_cancelled"
            validator.release.set()
            await service._queue.join()
            record = service.records()[0]
            assert record["status"] == "terminal" and record["delivery_outcome"] == "client_cancelled"
            assert service.status(None)["counters"]["cancelled"] == service.status(None)["counters"]["terminal"] == 1
        finally:
            await service.stop()

    run(scenario())


def test_positive_flag_and_block_follow_detector_broker_not_outer_neMo_status(tmp_path: Path) -> None:
    async def scenario() -> None:
        async def service_for(policy: str, *, action: str | None = None) -> GatewayService:
            broker = PendingValidationBroker(FakeDetector("watermarked"))
            adapter = ManagedNeMoAdapter(BrokerCallingNeMo(broker, force_action=action), broker)
            settings = GatewayConfig(
                sample_every=1,
                sqlite_path=tmp_path / f"{policy}-{action or 'normal'}.sqlite",
                positive_policy=policy,  # type: ignore[arg-type]
            )
            service = GatewayService(settings, FakeUpstream(), adapter, broker=broker)
            await service.start()
            return service

        flagged = await service_for("flag")
        try:
            response = await flagged.proxy("/v1/completions", body())
            assert response["id"] == "cmpl-1"
            assert flagged.records()[0]["delivery_outcome"] == "delivered"
        finally:
            await flagged.stop()

        blocked = await service_for("block")
        try:
            with pytest.raises(PositiveValidationBlock):
                await blocked.proxy("/v1/completions", body())
            assert blocked.records()[0]["delivery_outcome"] == "positive_blocked"
        finally:
            await blocked.stop()

        mismatched = await service_for("flag", action="success")
        try:
            with pytest.raises(TerminalValidationError, match="blocked"):
                await mismatched.proxy("/v1/completions", body())
            record = mismatched.records()[0]
            assert record["status"] == "error"
            assert record["verdict"] is None
            assert record["delivery_outcome"] == "fail_closed"
        finally:
            await mismatched.stop()

    run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "", "word"])
def test_sample_setting_rejects_non_positive_or_non_integer_values(value: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        GatewayConfig.from_environment({
            "VALIDATION_SAMPLE_EVERY": value,
            "VALIDATION_SAMPLER_DB_PATH": str(tmp_path / "sampler.sqlite"),
            "VALIDATION_REPLICA_COUNT": "1",
            "VALIDATION_BROKER_TOKEN": "unit-test-token",
            "VALIDATION_ADMIN_TOKEN": "unit-test-admin-token",
        })


@pytest.mark.parametrize("value", ["1", "5"])
def test_sample_setting_accepts_positive_integer_values(value: str, tmp_path: Path) -> None:
    settings = GatewayConfig.from_environment({
        "VALIDATION_SAMPLE_EVERY": value,
        "VALIDATION_SAMPLER_DB_PATH": str(tmp_path / "sampler.sqlite"),
        "VALIDATION_REPLICA_COUNT": "1",
        "VALIDATION_BROKER_TOKEN": "unit-test-token",
        "VALIDATION_ADMIN_TOKEN": "unit-test-admin-token",
    })
    assert settings.sample_every == int(value)


def test_token_settings_strip_surrounding_whitespace(tmp_path: Path) -> None:
    settings = GatewayConfig.from_environment({
        "VALIDATION_SAMPLE_EVERY": "1",
        "VALIDATION_SAMPLER_DB_PATH": str(tmp_path / "sampler.sqlite"),
        "VALIDATION_REPLICA_COUNT": "1",
        "VALIDATION_BROKER_TOKEN": "\n broker-token\t",
        "VALIDATION_ADMIN_TOKEN": "  admin-token\r\n",
    })
    assert settings.broker_token == "broker-token"
    assert settings.admin_token == "admin-token"

    direct = GatewayConfig(
        sample_every=1,
        sqlite_path=tmp_path / "direct.sqlite",
        broker_token="\n broker-token\t",
        admin_token="  admin-token\r\n",
    )
    assert direct.broker_token == "broker-token"
    assert direct.admin_token == "admin-token"


@pytest.mark.parametrize("field", ["VALIDATION_BROKER_TOKEN", "VALIDATION_ADMIN_TOKEN"])
def test_token_settings_reject_whitespace_only(field: str, tmp_path: Path) -> None:
    environment = {
        "VALIDATION_SAMPLE_EVERY": "1",
        "VALIDATION_SAMPLER_DB_PATH": str(tmp_path / "sampler.sqlite"),
        "VALIDATION_REPLICA_COUNT": "1",
        "VALIDATION_BROKER_TOKEN": "broker-token",
        "VALIDATION_ADMIN_TOKEN": "admin-token",
    }
    environment[field] = " \t\r\n"
    with pytest.raises(ConfigurationError, match="must be non-empty"):
        GatewayConfig.from_environment(environment)


def test_strict_request_shape_and_single_replica_enforcement(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = GatewayService(config(tmp_path), FakeUpstream(), FakeValidator())
        await service.start()
        try:
            missing_metadata = {"model": "test", "n": 1, "vllm_xargs": {"watermark": "off"}}
            with pytest.raises(UnsafeRequest, match="watermark_scheme"):
                await service.proxy("/v1/completions", missing_metadata)
            with pytest.raises(UnsafeRequest, match="streaming"):
                await service.proxy("/v1/completions", {**body(), "stream": True})
            with pytest.raises(UnsafeRequest, match="n=1"):
                await service.proxy("/v1/completions", {**body(), "n": 2})
        finally:
            await service.stop()
    run(scenario())
    with pytest.raises(ConfigurationError, match="exactly 1"):
        GatewayConfig(sample_every=1, sqlite_path=tmp_path / "sampler.sqlite", replica_count=2)


def test_guardrail_action_broker_accepts_exact_pending_metadata_only() -> None:
    async def scenario() -> None:
        detector = FakeDetector()
        broker = PendingValidationBroker(detector)
        text = "not persisted"
        validation_id = str(uuid.uuid4())
        request = ValidationRequest(
            validation_id=validation_id,
            response_id="cmpl-1",
            content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            expected_enabled=False,
            scheme="synthid",
            key_id="control-key",
        )
        broker.register(request)
        request_body = {
            "validation_id": validation_id,
            "response_id": request.response_id,
            "content_sha256": request.content_sha256,
            "scheme": "synthid",
            "key_id": "control-key",
        }
        result, repeated = await asyncio.gather(broker.resolve(request_body), broker.resolve(request_body))
        expected = {"validation_id": validation_id, "response_id": request.response_id, "content_sha256": request.content_sha256, "scheme": "synthid", "key_id": "control-key", "verdict": False}
        assert result == repeated == expected
        assert detector.requests == [request]
        with pytest.raises(UnsafeRequest, match="exactly"):
            await broker.resolve({
                "validation_id": validation_id,
                "response_id": request.response_id,
                "content_sha256": request.content_sha256,
                "scheme": "synthid",
                "key_id": "control-key",
                "text": text,
            })
        with pytest.raises(UnsafeRequest, match="digest"):
            await broker.resolve({
                "validation_id": validation_id,
                "response_id": request.response_id,
                "content_sha256": "0" * 64,
                "scheme": "synthid",
                "key_id": "control-key",
            })
        with pytest.raises(UnsafeRequest, match="metadata"):
            await broker.resolve({
                "validation_id": validation_id,
                "response_id": "cmpl-different",
                "content_sha256": request.content_sha256,
                "scheme": "synthid",
                "key_id": "control-key",
            })
        await broker.unregister(validation_id)

    run(scenario())


def test_broker_rejects_active_id_replacement_and_preserves_original_verdict() -> None:
    class ActiveDetector:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.requests: list[ValidationRequest] = []

        async def detect(self, request: ValidationRequest) -> str:
            self.requests.append(request)
            self.started.set()
            await self.release.wait()
            return "watermarked"

    async def scenario() -> None:
        detector = ActiveDetector()
        broker = PendingValidationBroker(detector)
        validation_id = str(uuid.uuid4())
        original_text = "original"
        original = ValidationRequest(
            validation_id=validation_id, response_id="cmpl-original", content=original_text,
            content_sha256=hashlib.sha256(original_text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        replacement_text = "replacement"
        replacement = ValidationRequest(
            validation_id=validation_id, response_id="cmpl-replacement", content=replacement_text,
            content_sha256=hashlib.sha256(replacement_text.encode()).hexdigest(), expected_enabled=True,
            scheme="synthid", key_id="other-key",
        )
        broker.register(original)
        resolving = asyncio.create_task(broker.resolve({
            "validation_id": validation_id, "response_id": original.response_id,
            "content_sha256": original.content_sha256, "scheme": original.scheme, "key_id": original.key_id,
        }))
        await detector.started.wait()
        with pytest.raises(TerminalValidationError, match="already active"):
            broker.register(replacement)
        detector.release.set()
        result = await resolving
        assert result["response_id"] == original.response_id
        assert result["scheme"] == original.scheme
        assert result["verdict"] is True
        assert detector.requests == [original]
        await broker.unregister(validation_id)

    run(scenario())


def test_broker_retryable_detector_failure_is_consumed_before_fresh_resolve() -> None:
    class RetryThenSuccessDetector:
        def __init__(self) -> None:
            self.requests: list[ValidationRequest] = []

        async def detect(self, request: ValidationRequest) -> str:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise RetryableValidationError("temporary detector failure")
            return "clean"

    async def scenario() -> None:
        detector = RetryThenSuccessDetector()
        broker = PendingValidationBroker(detector)
        text = "retry me"
        request = ValidationRequest(
            validation_id=str(uuid.uuid4()), response_id="cmpl-retry", content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        broker.register(request)
        payload = {
            "validation_id": request.validation_id, "response_id": request.response_id,
            "content_sha256": request.content_sha256, "scheme": request.scheme, "key_id": request.key_id,
        }
        with pytest.raises(RetryableValidationError):
            await broker.resolve(payload)
        assert request.validation_id in broker._detector_tasks
        with pytest.raises(RetryableValidationError):
            await broker.detector_result(request.validation_id)
        assert request.validation_id not in broker._detector_tasks
        result = await broker.resolve(payload)
        assert result["verdict"] is False
        assert detector.requests == [request, request]
        await broker.unregister(request.validation_id)

    run(scenario())


def test_managed_adapter_consumes_retryable_broker_failure_before_retry() -> None:
    class RetryThenSuccessDetector:
        def __init__(self) -> None:
            self.requests: list[ValidationRequest] = []

        async def detect(self, request: ValidationRequest) -> str:
            self.requests.append(request)
            if len(self.requests) == 1:
                raise RetryableValidationError("temporary detector failure")
            return "clean"

    class ManagedActionClient:
        def __init__(self, broker: PendingValidationBroker) -> None:
            self.broker = broker
            self.calls = 0

        async def validate(self, text: str, context: dict[str, str]) -> str:
            del text
            self.calls += 1
            payload = {
                "validation_id": context["watermark_validation_id"],
                "response_id": context["watermark_response_id"],
                "content_sha256": context["watermark_content_sha256"],
                "scheme": context["watermark_scheme"],
                "key_id": context["watermark_key_id"],
            }
            try:
                result = await self.broker.resolve(payload)
            except RetryableValidationError:
                # Managed action maps a detector transport failure to its
                # outer action state; detector_result owns retry-task cleanup.
                return "blocked"
            return "blocked" if result["verdict"] else "success"

    async def scenario() -> None:
        detector = RetryThenSuccessDetector()
        broker = PendingValidationBroker(detector)
        client = ManagedActionClient(broker)
        adapter = ManagedNeMoAdapter(client, broker)
        text = "managed retry"
        request = ValidationRequest(
            validation_id=str(uuid.uuid4()), response_id="cmpl-managed-retry", content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        broker.register(request)
        with pytest.raises(RetryableValidationError):
            await adapter.validate(request)
        assert request.validation_id not in broker._detector_tasks
        outcome = await adapter.validate(request)
        assert outcome == ValidationOutcome("clean", "success")
        assert client.calls == 2
        assert detector.requests == [request, request]
        await broker.unregister(request.validation_id)

    run(scenario())


def test_missing_managed_broker_call_is_retryable_and_exhausts_gateway_attempts(tmp_path: Path) -> None:
    class MissingBrokerClient:
        def __init__(self) -> None:
            self.calls = 0

        async def validate(self, text: str, context: dict[str, str]) -> str:
            del text, context
            self.calls += 1
            return "blocked"

    async def scenario() -> None:
        detector = FakeDetector()
        broker = PendingValidationBroker(detector)
        client = MissingBrokerClient()
        adapter = ManagedNeMoAdapter(client, broker)
        settings = GatewayConfig(
            sample_every=1,
            sqlite_path=tmp_path / "missing-broker.sqlite",
            max_attempts=3,
            retry_backoff_seconds=0.0,
        )
        service = GatewayService(settings, FakeUpstream(), adapter, broker=broker)
        await service.start()
        try:
            with pytest.raises(ValidationDeliveryBlocked):
                await service.proxy("/v1/completions", body())
            record = service.records()[0]
            assert record["terminal_state"] == "retry_exhausted"
            assert record["attempts"] == 3
            assert client.calls == 3
            assert "validation_retries_total 2" in service.metrics.render()
            assert detector.requests == []
        finally:
            await service.stop()

    run(scenario())


@pytest.mark.parametrize("scheme", ["KGW", " kgw", "synthid "])
def test_broker_requires_exact_lowercase_scheme(scheme: str) -> None:
    async def scenario() -> None:
        detector = FakeDetector()
        broker = PendingValidationBroker(detector)
        text = "scheme test"
        request = ValidationRequest(
            validation_id=str(uuid.uuid4()), response_id="cmpl-scheme", content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        broker.register(request)
        with pytest.raises(UnsafeRequest, match="exact lowercase"):
            await broker.resolve({
                "validation_id": request.validation_id, "response_id": request.response_id,
                "content_sha256": request.content_sha256, "scheme": scheme, "key_id": request.key_id,
            })
        await broker.unregister(request.validation_id)

    run(scenario())


def test_gateway_rejects_unsafe_upstream_response_id_before_sqlite(tmp_path: Path) -> None:
    class UnsafeIdUpstream(FakeUpstream):
        async def complete(self, endpoint: str, request: dict[str, object]) -> dict[str, object]:
            return {"id": "unsafe response id", "choices": [{"text": "generated"}]}

    async def scenario() -> None:
        service = GatewayService(config(tmp_path), UnsafeIdUpstream(), FakeValidator())
        await service.start()
        try:
            with pytest.raises(TerminalValidationError, match="response id"):
                await service.proxy("/v1/completions", body())
            assert service.sampler.records() == []
        finally:
            await service.stop()

    run(scenario())


def test_broker_route_requires_bearer_token(tmp_path: Path) -> None:
    async def make_service() -> tuple[GatewayService, dict[str, object]]:
        detector = FakeDetector()
        broker = PendingValidationBroker(detector)
        text = "temporary content"
        request = ValidationRequest(
            validation_id=str(uuid.uuid4()), response_id="cmpl-route", content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        broker.register(request)
        settings = GatewayConfig(sample_every=1, sqlite_path=tmp_path / "sampler.sqlite", broker_token="test-broker-token", admin_token="test-admin-token")
        return GatewayService(settings, FakeUpstream(), FakeValidator(), broker=broker), {
            "validation_id": request.validation_id, "content_sha256": request.content_sha256,
            "response_id": request.response_id, "scheme": "kgw", "key_id": "control-key",
        }

    service, payload = run(make_service())  # type: ignore[misc]
    with TestClient(create_app(service)) as client:
        assert client.post("/internal/v1/guardrail-action", json=payload).status_code == 401
        assert client.post("/internal/v1/guardrail-action", json=payload, headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.post("/internal/v1/guardrail-action", json=payload, headers={"Authorization": "Bearer test-broker-token"})
        assert response.status_code == 200
        assert response.json()["verdict"] is False


def test_broker_unregister_cancels_and_awaits_plaintext_bearing_detector_task() -> None:
    class BlockingDetector:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def detect(self, request: ValidationRequest) -> str:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return "clean"

    async def scenario() -> None:
        detector = BlockingDetector()
        broker = PendingValidationBroker(detector)
        text = "only-in-memory"
        request = ValidationRequest(
            validation_id=str(uuid.uuid4()), response_id="cmpl-broker", content=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(), expected_enabled=False,
            scheme="kgw", key_id="control-key",
        )
        broker.register(request)
        call = asyncio.create_task(broker.resolve({
            "validation_id": request.validation_id, "content_sha256": request.content_sha256,
            "response_id": request.response_id, "scheme": "kgw", "key_id": "control-key",
        }))
        await detector.started.wait()
        await broker.unregister(request.validation_id)
        with pytest.raises(asyncio.CancelledError):
            await call
        assert detector.cancelled is True
        assert broker._pending == {} and broker._detector_tasks == {}

    run(scenario())


def test_fixed_n1_and_n5_runs_reconcile_hash_only_harness_contract(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream, validator = FakeUpstream(), MatrixValidator()
        settings = GatewayConfig(sample_every=1, sqlite_path=tmp_path / "pvc" / "sampler.sqlite", positive_policy="flag")
        service = GatewayService(settings, upstream, validator)
        await service.start()
        try:
            for sample_every, total in ((1, 20), (5, 100)):
                run_id = f"run-{sample_every}"
                await service.reset_for_run(run_id, sample_every)
                selected_cases = [(True, "kgw")] * 5 + [(True, "synthid")] * 5 + [(False, "kgw")] * 5 + [(False, "synthid")] * 5
                selected_index = 0
                generated: list[dict[str, object]] = []
                for ordinal in range(1, total + 1):
                    enabled, scheme = selected_cases[selected_index] if ordinal % sample_every == 0 else (False, "kgw")
                    if ordinal % sample_every == 0:
                        selected_index += 1
                    response = await service.proxy("/v1/chat/completions", matrix_body(enabled, scheme, request_id=f"case-{ordinal}"))
                    generated.append(dict(response["watermark_validation"]))  # type: ignore[arg-type]
                assert [item["ordinal"] for item in generated] == list(range(1, total + 1))
                selected = [item for item in generated if item["selected"]]
                assert [item["ordinal"] for item in selected] == list(range(sample_every, total + 1, sample_every))
                status = service.status(run_id)
                assert status["counters"] == {
                    "started": total, "completed": total, "selected": 20, "unsampled": total - 20,
                    "terminal": 20, "watermarked": 10, "clean": 10, "errors": 0, "failed": 0,
                    "cancelled": 0, "detector_attempts": 20, "guardrails_attempts": 20,
                    "retries": 0, "queue_overflow": 0, "dropped": 0,
                }
                assert status["queue"] == {"depth": 0, "peak_depth": 1, "capacity": 32, "overflow_policy": "non_blocking", "consumer": "running"}
                assert status["latency_samples"] == {"generation_completion": total, "client_delivery": total, "validation": 20, "validation_lag": 20}
                records = service.harness_records(run_id)
                assert len(records) == 20
                assert {record["mode"] for record in records} == {"synchronous"}
                assert {record["guardrails_action"] for record in records} == {"block", "pass"}
                assert {record["managed_action"] for record in records} == {"blocked", "success"}
                assert len({record["detector_call_id"] for record in records}) == 20
                assert len({record["guardrails_action_id"] for record in records}) == 20
                assert all(record["detector_call_id"] == record["guardrails_action_id"] == record["validation_id"] for record in records)
                assert all("generated-" not in repr(record) for record in records)
        finally:
            await service.stop()

    run(scenario())


def test_admin_contract_faults_queue_and_safe_error_metadata(tmp_path: Path) -> None:
    settings = GatewayConfig(
        sample_every=1,
        sqlite_path=tmp_path / "sampler.sqlite",
        broker_token="broker-test-token",
        admin_token="admin-test-token",
        test_controls=True,
        positive_policy="flag",
    )
    service = GatewayService(settings, FakeUpstream(), MatrixValidator())
    with TestClient(create_app(service)) as client:
        admin = {"Authorization": "Bearer admin-test-token"}
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/v1/continuous-validation/admin/reset", json={"validation_sample_every": 1, "run_id": "run-a"}).status_code == 401
        assert client.post("/v1/continuous-validation/admin/reset", json={"validation_sample_every": 1, "run_id": "run-a"}, headers=admin).json() == {"accepted": True}
        assert client.post("/v1/continuous-validation/config/validate", json={"validation_sample_every": 5}, headers=admin).json() == {"valid": True}
        assert client.post("/v1/continuous-validation/config/validate", json={"validation_sample_every": 1.5}, headers=admin).json() == {"valid": False}
        assert client.post("/v1/continuous-validation/admin/consumer", json={"run_id": "run-a", "state": "paused", "capacity": 2}, headers=admin).json() == {"accepted": True}
        assert client.post("/v1/continuous-validation/admin/consumer", json={"run_id": "run-a", "state": "running"}, headers=admin).json() == {"accepted": True}

        assert client.post("/v1/continuous-validation/admin/faults", json={"run_id": "run-a", "scenario": "retry_then_success", "max_attempts": 3}, headers=admin).json() == {"accepted": True}
        generated = client.post("/v1/chat/completions", json=matrix_body(True, "kgw", request_id="fault-a")).json()
        assert generated["watermark_validation"]["selected"] is True
        record = client.get("/v1/continuous-validation/records", params={"run_id": "run-a"}, headers=admin).json()["records"][0]
        assert record["attempts"] == 2 and record["terminal_state"] == "success"
        assert client.get("/v1/continuous-validation/status", params={"run_id": "run-a"}, headers=admin).json()["counters"]["retries"] == 1

        assert client.post("/v1/continuous-validation/admin/reset", json={"validation_sample_every": 1, "run_id": "run-b"}, headers=admin).status_code == 200
        assert client.post("/v1/continuous-validation/admin/faults", json={"run_id": "run-b", "scenario": "retry_exhausted", "max_attempts": 3}, headers=admin).status_code == 200
        failed = client.post("/v1/chat/completions", json=matrix_body(True, "kgw", request_id="fault-b"))
        assert failed.status_code == 503
        assert set(failed.json()) == {"error", "watermark_validation"}
        assert "generated-" not in repr(failed.json())
        failed_record = client.get("/v1/continuous-validation/records", params={"run_id": "run-b"}, headers=admin).json()["records"][0]
        assert failed_record["terminal_state"] == "retry_exhausted"
        assert failed_record["delivery_outcome"] == "fail_closed"

        assert client.post("/v1/continuous-validation/admin/reset", json={"validation_sample_every": 1, "run_id": "run-c"}, headers=admin).status_code == 200
        assert client.post("/v1/continuous-validation/admin/faults", json={"run_id": "run-c", "scenario": "malformed_success", "max_attempts": 3}, headers=admin).status_code == 200
        malformed = client.post("/v1/chat/completions", json=matrix_body(True, "kgw", request_id="fault-c"))
        assert malformed.status_code == 503 and "generated-" not in repr(malformed.json())
        malformed_record = client.get("/v1/continuous-validation/records", params={"run_id": "run-c"}, headers=admin).json()["records"][0]
        assert malformed_record["attempts"] == 1 and malformed_record["terminal_state"] == "malformed_response"

        events = client.get("/v1/continuous-validation/admin/redacted-events", headers=admin).json()
        assert "generated-" not in repr(events)


def test_paused_consumer_counts_queued_plus_inflight_slots_and_honors_closed_overflow(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = GatewayService(
            GatewayConfig(sample_every=1, sqlite_path=tmp_path / "queue.sqlite", test_controls=True, positive_policy="flag"),
            FakeUpstream(),
            MatrixValidator(),
        )
        await service.start()
        try:
            await service.reset_for_run("queue-run", 1)
            await service.set_consumer_state("queue-run", "paused", 2)
            first = asyncio.create_task(service.proxy("/v1/chat/completions", matrix_body(True, "kgw", request_id="queue-1")))
            second = asyncio.create_task(service.proxy("/v1/chat/completions", matrix_body(True, "kgw", request_id="queue-2")))
            for _ in range(20):
                if service.status("queue-run")["queue"]["depth"] == 2:
                    break
                await asyncio.sleep(0)
            paused = service.status("queue-run")
            assert paused["queue"]["depth"] == paused["queue"]["peak_depth"] == paused["queue"]["capacity"] == 2
            with pytest.raises(ValidationDeliveryBlocked) as blocked:
                await service.proxy("/v1/chat/completions", matrix_body(True, "kgw", request_id="queue-3"))
            assert "generated-" not in repr(blocked.value.metadata)
            assert service.status("queue-run")["counters"]["queue_overflow"] == 1
            await service.set_consumer_state("queue-run", "running")
            await asyncio.gather(first, second)
            final = service.status("queue-run")
            assert final["queue"]["depth"] == 0
            records = service.harness_records("queue-run")
            assert len(records) == 3
            assert sum(record["terminal_state"] == "success" for record in records) == 2
            overflow = [record for record in records if record["terminal_state"] == "queue_overflow"]
            assert len(overflow) == 1 and overflow[0]["attempts"] == 0 and "verdict" not in overflow[0]
            assert overflow[0]["delivery_outcome"] == "fail_closed"
        finally:
            await service.stop()

    run(scenario())


def test_gateway_metadata_is_stripped_key_id_is_intersection_and_pvc_files_are_owner_only(tmp_path: Path) -> None:
    async def scenario() -> None:
        upstream = FakeUpstream()
        service = GatewayService(GatewayConfig(sample_every=5, sqlite_path=tmp_path / "pvc" / "sampler.sqlite"), upstream, MatrixValidator())
        await service.start()
        try:
            response = await service.proxy("/v1/completions", matrix_body(False, "kgw", request_id="gateway-only-id"))
            assert response["watermark_validation"]["selected"] is False
            forwarded = upstream.calls[0][1]
            assert "metadata" not in forwarded
            for path in (tmp_path / "pvc" / "sampler.sqlite", tmp_path / "pvc" / "sampler.sqlite.singleton.lock"):
                assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
            for forbidden in ("unsafe/key", "unsafe@key"):
                bad = matrix_body(False, "kgw")
                bad["vllm_xargs"] = {**bad["vllm_xargs"], "watermark_key_id": forbidden}  # type: ignore[index]
                with pytest.raises(UnsafeRequest):
                    await service.proxy("/v1/completions", bad)
        finally:
            await service.stop()

    run(scenario())


def test_test_controls_and_metric_labels_are_strict_and_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        service = GatewayService(
            GatewayConfig(sample_every=1, sqlite_path=tmp_path / "controls.sqlite", test_controls=False),
            FakeUpstream(),
            MatrixValidator(),
        )
        await service.start()
        try:
            await service.reset_for_run("controls-run", 1)
            with pytest.raises(UnsafeRequest, match="disabled"):
                await service.configure_fault("controls-run", "retry_exhausted", 3)
            with pytest.raises(UnsafeRequest, match="disabled"):
                await service.set_consumer_state("controls-run", "paused", 2)
        finally:
            await service.stop()

    run(scenario())
    metrics = PrometheusMetrics()
    with pytest.raises(ValueError, match="allowlist"):
        metrics.inc("validation_attempts", key_id="not-an-allowed-label")
    for _ in range(1000):
        metrics.observe("validation_latency_seconds", 0.01)
    histogram = metrics._histograms["validation_latency_seconds"]
    assert histogram["count"] == 1000
    assert len(histogram["buckets"]) == len(metrics._HISTOGRAM_BUCKETS)
    rendered = metrics.render()
    assert "watermark_validation_latency_seconds_count 1000" in rendered
    assert sum(line.startswith('validation_latency_seconds_bucket{le="') for line in rendered.splitlines()) == len(metrics._HISTOGRAM_BUCKETS) + 1
    assert "key_id=" not in rendered and "content_digest=" not in rendered and "generated-" not in rendered
