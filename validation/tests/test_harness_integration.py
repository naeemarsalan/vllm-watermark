"""End-to-end local HTTP acceptance coverage for the D10 harness.

The fake transports deliberately include a marker in transient generated text.
The marker must never appear in the harness report, events, or metrics.  This
test exercises the real ASGI routes over a localhost socket; it does not claim
cluster, vLLM, or managed-NeMo execution.
"""

from __future__ import annotations

import json
import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import uvicorn
from fastapi import FastAPI

from validation.gateway import GatewayConfig, GatewayService, ValidationOutcome, create_app


_HARNESS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "continuous_validation.py"
_HARNESS_SPEC = importlib.util.spec_from_file_location("_d10_continuous_validation_test", _HARNESS_PATH)
assert _HARNESS_SPEC is not None and _HARNESS_SPEC.loader is not None
_HARNESS_MODULE = importlib.util.module_from_spec(_HARNESS_SPEC)
sys.modules[_HARNESS_SPEC.name] = _HARNESS_MODULE
_HARNESS_SPEC.loader.exec_module(_HARNESS_MODULE)
CheckFailure = _HARNESS_MODULE.CheckFailure
ContractError = _HARNESS_MODULE.ContractError
normalise_record = _HARNESS_MODULE.normalise_record
reconcile_run = _HARNESS_MODULE.reconcile_run
synthetic_run_evidence = _HARNESS_MODULE.synthetic_run_evidence
queue_run = _HARNESS_MODULE.queue_run
Generated = _HARNESS_MODULE.Generated


MARKER = "integration-secret-marker"
VALIDATION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _harness_record() -> dict[str, Any]:
    return {
        "validation_id": VALIDATION_ID,
        "response_id": "chat-integration-1",
        "content_digest": "0" * 64,
        "scheme": "kgw",
        "key_id": "integration-key",
        "verdict": True,
        "mode": "synchronous",
        "attempts": 1,
        "timing": {"validation_latency_seconds": 0.1, "validation_lag_seconds": 0.01},
        "detector_call_id": VALIDATION_ID,
        "guardrails_action_id": VALIDATION_ID,
        "managed_action": "blocked",
        "guardrails_action": "block",
        "delivery_outcome": "delivered",
    }


def test_harness_requires_raw_managed_action_and_exact_canonical_correlation() -> None:
    record = normalise_record(_harness_record())
    assert record["managed_action"] == "blocked"
    with pytest.raises(ContractError):
        normalise_record({**record, "validation_id": VALIDATION_ID.upper()})
    with pytest.raises(ContractError):
        normalise_record({**record, "managed_action": "block"})

    evidence, cases = synthetic_run_evidence("integration-key", n=1, total=20)
    evidence.records[0]["detector_call_id"] = "not-the-validation-id"
    with pytest.raises(CheckFailure, match="exactly correlated"):
        reconcile_run(evidence, cases, n=1, total=20, positive_action="block", clean_action="pass", expected_mode="synchronous")


class SlowQueueGateway:
    """CPU-only queue-run double with generation slower than scheduling grace."""

    def __init__(self, generation_delay: float | tuple[float, float, float], *, completed_overshoot: bool = False) -> None:
        if isinstance(generation_delay, tuple):
            assert len(generation_delay) == 3
            self.generation_delays = generation_delay
        else:
            self.generation_delays = (generation_delay, generation_delay, generation_delay)
        self.completed_overshoot = completed_overshoot
        self._lock = threading.Lock()
        self._resume = threading.Event()
        self._completed = 0
        self._third_completed = False
        self._depth = 0
        self._overflow = 0
        self._records: list[dict[str, Any]] = []
        self.consumer_state = "running"
        self.resume_calls = 0

    def reset(self, sample_every: int, run_id: str) -> None:
        assert sample_every == 1
        with self._lock:
            self._completed = 0
            self._third_completed = False
            self._depth = 0
            self._overflow = 0
            self._records = []
            self.consumer_state = "running"
            self.resume_calls = 0
        self._resume.clear()

    def request(
        self,
        name: str,
        body: Mapping[str, Any],
        *,
        admin: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        assert admin and name == "consumer"
        assert body["state"] in {"paused", "running"}
        self.consumer_state = body["state"]
        if body["state"] == "running":
            self.resume_calls += 1
            self._resume.set()
        return {"accepted": True}

    def generate(
        self,
        case: Any,
        request_id: str,
        *,
        expected_error_statuses: tuple[int, ...] = (),
        timeout: float | None = None,
    ) -> Any:
        ordinal = int(request_id.rsplit("-", 1)[-1])
        time.sleep(self.generation_delays[ordinal - 1])
        validation_id = f"00000000-0000-4000-8000-{ordinal:012d}"
        with self._lock:
            self._completed += 1
            if ordinal == 3:
                self._third_completed = True
                self._overflow += 1
                self._records.append({
                    "validation_id": validation_id,
                    "response_id": f"slow-response-{ordinal}",
                    "content_digest": "f" * 64,
                    "mode": "synchronous",
                    "attempts": 0,
                    "terminal_state": "queue_overflow",
                    "delivery_outcome": "fail_closed",
                })
                return None
            self._depth += 1
        self._resume.wait(2)
        with self._lock:
            self._depth -= 1
            self._records.append({
                "validation_id": validation_id,
                "response_id": f"slow-response-{ordinal}",
                "content_digest": "a" * 64,
                "scheme": "kgw",
                "key_id": "slow-key",
                "verdict": False,
                "mode": "synchronous",
                "attempts": 1,
                "terminal_state": "success",
                "delivery_outcome": "delivered",
                "timing": {"validation_latency_seconds": 0.01, "validation_lag_seconds": 0.01},
                "detector_call_id": validation_id,
                "guardrails_action_id": validation_id,
                "managed_action": "success",
                "guardrails_action": "pass",
            })
        return Generated(
            response_id=f"slow-response-{ordinal}",
            digest="a" * 64,
            ordinal=ordinal,
            selected=True,
            generation_latency=self.generation_delays[ordinal - 1],
            delivery_latency=self.generation_delays[ordinal - 1],
        )

    def status(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            completed = self._completed + (1 if self.completed_overshoot and self._third_completed else 0)
            return {
                "counters": {
                    "completed": completed,
                    "queue_overflow": self._overflow,
                    "dropped": self._overflow,
                },
                "queue": {
                    "depth": self._depth,
                    "peak_depth": 2,
                    "capacity": 2,
                    "overflow_policy": "non_blocking",
                },
            }

    def records(self, run_id: str, *, timeout: float | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)


class TimeoutBeforeCompletedGateway(SlowQueueGateway):
    """Fail the third completion poll and release all workers during cleanup."""

    def __init__(self) -> None:
        super().__init__(0.0)
        self.third_started = threading.Event()
        self.active_workers = 0
        self.worker_threads: list[threading.Thread] = []

    def generate(
        self,
        case: Any,
        request_id: str,
        *,
        expected_error_statuses: tuple[int, ...] = (),
        timeout: float | None = None,
    ) -> Any:
        ordinal = int(request_id.rsplit("-", 1)[-1])
        worker = threading.current_thread()
        self.worker_threads.append(worker)
        with self._lock:
            self.active_workers += 1
        try:
            if ordinal == 3:
                self.third_started.set()
                self._resume.wait(2)
                return None
            return super().generate(
                case,
                request_id,
                expected_error_statuses=expected_error_statuses,
                timeout=timeout,
            )
        finally:
            with self._lock:
                self.active_workers -= 1

    def status(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        if self.third_started.is_set():
            raise CheckFailure("third generation status timed out")
        return super().status(run_id)


class MalformedPauseAcknowledgementGateway(SlowQueueGateway):
    """Model a pause that took effect before its acknowledgement was invalid."""

    def request(
        self,
        name: str,
        body: Mapping[str, Any],
        *,
        admin: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = super().request(name, body, admin=admin, timeout=timeout)
        if body["state"] == "paused":
            return {"accepted": "malformed"}
        return response


class AmbiguousPauseFailureGateway(SlowQueueGateway):
    """Model a pause applied server-side before its client call fails."""

    def request(
        self,
        name: str,
        body: Mapping[str, Any],
        *,
        admin: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        response = super().request(name, body, admin=admin, timeout=timeout)
        if body["state"] == "paused":
            raise CheckFailure("pause transport failed after mutation")
        return response


class DeadlineBeforeCompletedGateway(TimeoutBeforeCompletedGateway):
    """Keep the third generation incomplete until cleanup resumes it."""

    def status(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return SlowQueueGateway.status(self, run_id, timeout=timeout)


def test_queue_run_waits_for_slow_generation_before_scheduling_grace() -> None:
    gateway = SlowQueueGateway(generation_delay=0.05)
    args = SimpleNamespace(
        key_id="slow-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=1.0,
        queue_pending_check_seconds=0.01,
    )
    result = queue_run(gateway, args)
    assert result == {
        "overflow_policy": "non_blocking",
        "terminal_records": 3,
        "peak_depth": 2,
        "queue_depth": 0,
        "queue_overflow": 1,
        "validated_records": 2,
    }


def test_queue_run_uses_one_deadline_for_delayed_generations_and_accepts_counter_overshoot() -> None:
    gateway = SlowQueueGateway((0.02, 0.04, 0.06), completed_overshoot=True)
    args = SimpleNamespace(
        key_id="slow-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=1.0,
        queue_pending_check_seconds=0.001,
    )
    result = queue_run(gateway, args)
    assert result == {
        "overflow_policy": "non_blocking",
        "terminal_records": 3,
        "peak_depth": 2,
        "queue_depth": 0,
        "queue_overflow": 1,
        "validated_records": 2,
    }
    assert gateway.consumer_state == "running"
    assert gateway.resume_calls == 1


def test_queue_run_resumes_and_cleans_workers_when_completion_times_out() -> None:
    gateway = TimeoutBeforeCompletedGateway()
    args = SimpleNamespace(
        key_id="timeout-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=0.5,
        queue_pending_check_seconds=0.001,
    )
    with pytest.raises(CheckFailure, match="third generation status timed out"):
        queue_run(gateway, args)
    assert gateway.consumer_state == "running"
    assert gateway.resume_calls == 1
    assert gateway.active_workers == 0
    assert gateway.worker_threads
    assert all(not worker.is_alive() for worker in gateway.worker_threads)


def test_queue_run_resumes_after_malformed_pause_acknowledgement() -> None:
    gateway = MalformedPauseAcknowledgementGateway(0.0)
    args = SimpleNamespace(
        key_id="pause-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=0.5,
        queue_pending_check_seconds=0.001,
    )
    with pytest.raises(CheckFailure, match="consumer did not pause"):
        queue_run(gateway, args)
    assert gateway.consumer_state == "running"
    assert gateway.resume_calls == 1


def test_queue_run_resumes_after_ambiguous_pause_transport_failure() -> None:
    gateway = AmbiguousPauseFailureGateway(0.0)
    args = SimpleNamespace(
        key_id="pause-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=0.5,
        queue_pending_check_seconds=0.001,
    )
    with pytest.raises(CheckFailure, match="pause transport failed after mutation"):
        queue_run(gateway, args)
    assert gateway.consumer_state == "running"
    assert gateway.resume_calls == 1


def test_queue_run_uses_bounded_cleanup_grace_after_operation_deadline() -> None:
    gateway = DeadlineBeforeCompletedGateway()
    args = SimpleNamespace(
        key_id="deadline-key",
        expected_failure_policy="closed",
        expected_mode="synchronous",
        timeout_seconds=0.05,
        queue_pending_check_seconds=0.001,
    )
    started = time.monotonic()
    with pytest.raises(CheckFailure, match="third generation did not complete"):
        queue_run(gateway, args)
    assert time.monotonic() - started < 0.5
    assert gateway.consumer_state == "running"
    assert gateway.resume_calls == 1
    assert gateway.active_workers == 0
    assert gateway.worker_threads
    assert all(not worker.is_alive() for worker in gateway.worker_threads)


class FakeUpstream:
    """Deterministic OpenAI-compatible upstream with transient-only content."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, endpoint: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        assert endpoint == "/v1/chat/completions"
        # Gateway-only correlation metadata must not cross the upstream
        # boundary.  The harness does not send it, but this assertion protects
        # the real proxy contract if that changes.
        assert "metadata" not in request
        self.calls += 1
        return {
            "id": f"chat-integration-{self.calls}",
            "choices": [{"message": {"content": f"generated-{self.calls}-{MARKER}"}}],
        }


class FakeManagedValidator:
    """Return the managed action expected for each pre-registered case."""

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.calls = 0

    async def validate(self, request: Any) -> ValidationOutcome:
        self.calls += 1
        action = "blocked" if request.expected_enabled else "success"
        self.actions.append(action)
        return ValidationOutcome(
            verdict="watermarked" if request.expected_enabled else "clean",
            managed_action=action,
        )


@pytest.fixture
def running_gateway(tmp_path: Path):
    """Serve the real FastAPI app on an ephemeral localhost socket."""

    upstream = FakeUpstream()
    validator = FakeManagedValidator()
    config = GatewayConfig(
        sample_every=1,
        sqlite_path=tmp_path / "sampler.sqlite",
        failure_policy="closed",
        positive_policy="flag",
        max_attempts=3,
        retry_backoff_seconds=0.001,
        max_retry_elapsed_seconds=1.0,
        attempt_timeout_seconds=1.0,
        queue_capacity=32,
        max_inflight=4,
        broker_token="integration-broker-token",
        admin_token="integration-admin-token",
        test_controls=True,
    )
    service = GatewayService(config, upstream, validator)
    app: FastAPI = create_app(service)

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            log_level="critical",
            access_log=False,
            log_config=None,
            lifespan="on",
        )
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(5)
        raise RuntimeError("local gateway did not start")
    try:
        yield f"http://127.0.0.1:{port}", service, upstream, validator
    finally:
        server.should_exit = True
        thread.join(10)
        if thread.is_alive():
            server.force_exit = True
            thread.join(5)
        listener.close()


def test_continuous_validation_harness_over_real_http(running_gateway: tuple[str, GatewayService, FakeUpstream, FakeManagedValidator]) -> None:
    base_url, service, upstream, validator = running_gateway
    repo = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["INTEGRATION_ADMIN_TOKEN"] = "integration-admin-token"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/continuous_validation.py",
            "--gateway-url",
            base_url,
            "--model",
            "integration-model",
            "--key-id",
            "integration-key",
            "--admin-token-env",
            "INTEGRATION_ADMIN_TOKEN",
            "--forbidden-marker",
            MARKER,
            "--timeout-seconds",
            "10",
            "--queue-pending-check-seconds",
            "0.15",
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    # Do not include either captured stream in assertion messages: an upstream
    # or proxy regression must not turn this test into a body/content sink.
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["passed"] is True
    assert report["content_logged"] is False
    assert report["unsampled_baseline"]["responses"] == 4
    assert report["unsampled_baseline"]["sample_every"] == 5
    assert report["unsampled_baseline"]["selected"] == 0
    assert report["unsampled_baseline"]["counters"]["unsampled"] == 4
    assert report["unsampled_baseline"]["counters"]["detector_attempts"] == 0
    assert report["unsampled_baseline"]["counters"]["guardrails_attempts"] == 0
    assert report["unsampled_baseline"]["counters"]["retries"] == 0
    assert report["unsampled_baseline"]["counters"]["queue_overflow"] == 0
    assert report["unsampled_baseline"]["generation_completion_latency"]["count"] == 4
    assert report["unsampled_baseline"]["client_delivery_latency"]["count"] == 4
    for run_name in ("unsampled_baseline", "n1", "n5"):
        for latency_name in ("generation_completion_latency", "client_delivery_latency"):
            assert set(report[run_name][latency_name]) == {"count", "p50_seconds", "p95_seconds", "p99_seconds"}
            assert all(isinstance(report[run_name][latency_name][field], (int, float)) for field in report[run_name][latency_name])
    for run_name in ("n1", "n5"):
        for latency_name in ("validation_latency", "validation_lag"):
            assert set(report[run_name][latency_name]) == {"count", "p50_seconds", "p95_seconds", "p99_seconds"}
    assert report["n1"]["responses"] == 20
    assert report["n1"]["selected"] == 20
    assert report["n1"]["counters"]["watermarked"] == 10
    assert report["n1"]["counters"]["clean"] == 10
    assert report["n1"]["counters"]["detector_attempts"] == 20
    assert report["n1"]["counters"]["guardrails_attempts"] == 20
    assert len(report["n1"]["record_evidence"]) == 20
    assert all(record["ids_correlated"] is True for record in report["n1"]["record_evidence"])
    assert {record["delivery_outcome"] for record in report["n1"]["record_evidence"]} == {"delivered"}
    assert report["n5"]["responses"] == 100
    assert report["n5"]["selected"] == 20
    assert report["n5"]["counters"]["unsampled"] == 80
    assert len(report["n5"]["record_evidence"]) == 20
    assert report["latency_semantics"]["validation_lag"] == "validation_queue_wait_to_attempt_start"
    assert report["policy_semantics"] == {
        "mode": "synchronous",
        "validation_failure": "closed",
        "managed_guardrails_positive_action": "block",
        "gateway_positive_delivery": "flag",
    }
    assert report["faults"]["retry_then_success"]["attempts"] == 2
    assert report["faults"]["retry_exhausted"]["attempts"] == 3
    assert report["faults"]["malformed_success"]["attempts"] == 1
    assert report["queue"]["terminal_records"] == 3
    assert report["queue"]["peak_depth"] == 2
    assert report["queue"]["queue_overflow"] == 1
    assert report["queue"]["validated_records"] == 2
    assert report["observability"]["marker_count"] == 1
    assert MARKER not in result.stdout
    assert "generated-" not in result.stdout
    assert MARKER not in result.stderr
    assert "generated-" not in result.stderr

    # The positive policy is flag: both managed action states are delivered to
    # the gateway, while the detector/managed outcome remains in the records.
    assert service.config.positive_policy == "flag"
    assert service.config.failure_policy == "closed"
    assert service.config.test_controls is True
    assert set(validator.actions) == {"blocked", "success"}
    assert upstream.calls >= 124
