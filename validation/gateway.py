"""Synchronous, non-streaming OpenAI gateway with sampled validation.

The gateway accepts only one generated choice (``n=1``) and buffers the
upstream response before delivery.  This is intentional: it makes the
completed-response ordinal and fail-closed policy unambiguous.  Generated
content is sent to the configured validator in memory, but is neither logged
nor written to SQLite; only its SHA-256 digest is retained.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

try:  # Kept optional so the core/adapters are importable in minimal test jobs.
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
except ImportError:  # pragma: no cover - exercised only in a deliberately minimal install.
    FastAPI = Any  # type: ignore[misc,assignment]


class ConfigurationError(ValueError):
    """A safety-critical gateway setting was malformed."""


class UnsafeRequest(ValueError):
    """The OpenAI request cannot produce one safely correlated response."""


class RetryableValidationError(RuntimeError):
    """A transport/temporary validator failure that may be retried."""


class TerminalValidationError(RuntimeError):
    """A malformed or terminal validation response which must not be retried."""


class PositiveValidationBlock(TerminalValidationError):
    """A watermarked response was intentionally withheld by positive policy."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__("watermarked response blocked")
        self.metadata = dict(metadata)


class ValidationDeliveryBlocked(TerminalValidationError):
    """A selected response was withheld with safe correlation metadata only."""

    def __init__(self, metadata: Mapping[str, Any]) -> None:
        super().__init__("selected response blocked because validation did not complete")
        self.metadata = dict(metadata)


FailurePolicy = Literal["open", "closed"]
PositivePolicy = Literal["flag", "block"]
Verdict = Literal["watermarked", "clean"]
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
# OpenAI-compatible servers commonly emit IDs such as ``cmpl-...`` and
# ``chatcmpl-...``.  Keep the propagated ID deliberately narrow: it crosses
# several HTTP and persistence boundaries and is correlation metadata, not an
# arbitrary user string.
_RESPONSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _strict_positive_int(value: str | int, name: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[1-9][0-9]*", value):
        parsed = int(value)
    else:
        raise ConfigurationError(f"{name} must be a positive integer")
    if parsed < 1:
        raise ConfigurationError(f"{name} must be a positive integer")
    return parsed


def _watermark_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"on", "true", "1", "yes"}:
            return True
        if normalized in {"off", "false", "0", "no"}:
            return False
    raise UnsafeRequest("vllm_xargs.watermark must be a documented watermark flag")


def _scheme(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in {"kgw", "synthid"}:
        return value.strip().lower()
    raise UnsafeRequest("vllm_xargs.watermark_scheme must be kgw or synthid")


def _key_id(value: Any) -> str:
    if isinstance(value, str) and _KEY_ID.fullmatch(value):
        return value
    raise UnsafeRequest("vllm_xargs.watermark_key_id is invalid")


def _normalize_token(value: Any, name: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ConfigurationError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ConfigurationError(f"{name} must be non-empty")
    return normalized


def _response_id(value: Any) -> str:
    if isinstance(value, str) and _RESPONSE_ID.fullmatch(value):
        return value
    raise TerminalValidationError("upstream response id is not a bounded safe identifier")


@dataclass(frozen=True)
class GatewayConfig:
    sample_every: int
    sqlite_path: Path
    replica_count: int = 1
    failure_policy: FailurePolicy = "closed"
    positive_policy: PositivePolicy = "flag"
    max_attempts: int = 3
    retry_backoff_seconds: float = 0.05
    max_retry_elapsed_seconds: float = 5.0
    attempt_timeout_seconds: float = 10.0
    queue_capacity: int = 32
    max_inflight: int = 4
    broker_token: str | None = field(default=None, repr=False)
    admin_token: str | None = field(default=None, repr=False)
    test_controls: bool = False
    default_watermark_enabled: bool = False
    default_watermark_scheme: str = "kgw"
    default_watermark_key_id: str = "default"

    def __post_init__(self) -> None:
        _strict_positive_int(self.sample_every, "VALIDATION_SAMPLE_EVERY")
        if self.replica_count != 1:
            raise ConfigurationError("VALIDATION_REPLICA_COUNT must be exactly 1 for SQLite sampling")
        if self.failure_policy not in ("open", "closed"):
            raise ConfigurationError("VALIDATION_FAILURE_POLICY must be open or closed")
        if self.positive_policy not in ("flag", "block"):
            raise ConfigurationError("VALIDATION_POSITIVE_POLICY must be flag or block")
        _strict_positive_int(self.max_attempts, "VALIDATION_MAX_ATTEMPTS")
        _strict_positive_int(self.queue_capacity, "VALIDATION_QUEUE_CAPACITY")
        _strict_positive_int(self.max_inflight, "VALIDATION_MAX_INFLIGHT")
        for name, value in (
            ("VALIDATION_RETRY_BACKOFF_SECONDS", self.retry_backoff_seconds),
            ("VALIDATION_MAX_RETRY_ELAPSED_SECONDS", self.max_retry_elapsed_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ConfigurationError(f"{name} must be a finite non-negative number")
        if not math.isfinite(self.attempt_timeout_seconds) or self.attempt_timeout_seconds <= 0:
            raise ConfigurationError("VALIDATION_ATTEMPT_TIMEOUT_SECONDS must be a finite positive number")
        try:
            _scheme(self.default_watermark_scheme)
            _key_id(self.default_watermark_key_id)
        except UnsafeRequest as exc:
            raise ConfigurationError("invalid default watermark metadata") from exc
        if not self.sqlite_path.is_absolute():
            raise ConfigurationError("VALIDATION_SAMPLER_DB_PATH must be an absolute PVC-mounted path")
        for name, field_name, value in (
            ("VALIDATION_BROKER_TOKEN", "broker_token", self.broker_token),
            ("VALIDATION_ADMIN_TOKEN", "admin_token", self.admin_token),
        ):
            normalized = _normalize_token(value, name)
            if normalized is not None:
                object.__setattr__(self, field_name, normalized)

    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None) -> "GatewayConfig":
        env = os.environ if env is None else env
        required = "VALIDATION_SAMPLE_EVERY"
        if required not in env:
            raise ConfigurationError(f"{required} is required")
        path = env.get("VALIDATION_SAMPLER_DB_PATH")
        if not path:
            raise ConfigurationError("VALIDATION_SAMPLER_DB_PATH is required")
        if "VALIDATION_REPLICA_COUNT" not in env:
            raise ConfigurationError("VALIDATION_REPLICA_COUNT=1 must be explicitly configured for SQLite sampling")
        broker_token = _normalize_token(env.get("VALIDATION_BROKER_TOKEN"), "VALIDATION_BROKER_TOKEN", required=True)
        admin_token = _normalize_token(env.get("VALIDATION_ADMIN_TOKEN"), "VALIDATION_ADMIN_TOKEN", required=True)
        test_controls_value = env.get("VALIDATION_TEST_CONTROLS", "off").strip().lower()
        if test_controls_value not in ("on", "off"):
            raise ConfigurationError("VALIDATION_TEST_CONTROLS must be on or off")
        policy = env.get("VALIDATION_FAILURE_POLICY", "closed").strip().lower()
        positive_policy = env.get("VALIDATION_POSITIVE_POLICY", "flag").strip().lower()
        try:
            return cls(
                sample_every=_strict_positive_int(env[required], required),
                sqlite_path=Path(path),
                replica_count=_strict_positive_int(env["VALIDATION_REPLICA_COUNT"], "VALIDATION_REPLICA_COUNT"),
                failure_policy=policy,  # type: ignore[arg-type]
                positive_policy=positive_policy,  # type: ignore[arg-type]
                max_attempts=_strict_positive_int(env.get("VALIDATION_MAX_ATTEMPTS", "3"), "VALIDATION_MAX_ATTEMPTS"),
                retry_backoff_seconds=float(env.get("VALIDATION_RETRY_BACKOFF_SECONDS", "0.05")),
                max_retry_elapsed_seconds=float(env.get("VALIDATION_MAX_RETRY_ELAPSED_SECONDS", "5")),
                attempt_timeout_seconds=float(env.get("VALIDATION_ATTEMPT_TIMEOUT_SECONDS", "10")),
                queue_capacity=_strict_positive_int(env.get("VALIDATION_QUEUE_CAPACITY", "32"), "VALIDATION_QUEUE_CAPACITY"),
                max_inflight=_strict_positive_int(env.get("VALIDATION_MAX_INFLIGHT", "4"), "VALIDATION_MAX_INFLIGHT"),
                broker_token=broker_token,
                admin_token=admin_token,
                test_controls=test_controls_value == "on",
                default_watermark_enabled=_watermark_flag(env.get("VLLM_WATERMARK_DEFAULT", "off")),
                default_watermark_scheme=_scheme(env.get("VLLM_WATERMARK_SCHEME", "kgw")),
                default_watermark_key_id=_key_id(env.get("WATERMARK_KEY_ID", "default")),
            )
        except ConfigurationError:
            raise
        except (UnsafeRequest, ValueError) as exc:
            raise ConfigurationError("invalid gateway configuration") from exc


@dataclass(frozen=True)
class ResponseMetadata:
    expected_enabled: bool
    scheme: str
    key_id: str


@dataclass(frozen=True)
class ValidationOutcome:
    verdict: Verdict
    managed_action: str


@dataclass(frozen=True)
class ValidationRequest:
    validation_id: str
    response_id: str
    content: str
    content_sha256: str
    expected_enabled: bool
    scheme: str
    key_id: str
    detector_call_id: str = ""
    guardrails_action_id: str = ""
    generation_completion_seconds: float = 0.0


class UpstreamTransport(Protocol):
    async def complete(self, endpoint: str, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ValidationAdapter(Protocol):
    async def validate(self, request: ValidationRequest) -> ValidationOutcome: ...


class DetectorAdapter(Protocol):
    async def detect(self, request: ValidationRequest) -> Verdict: ...


class GuardrailsAdapter(Protocol):
    async def apply(self, request: ValidationRequest, verdict: Verdict) -> str: ...


class ManagedValidationAdapter:
    """Composition point for the managed-NeMo contract.

    The future NeMo adapter owns its own response-schema parsing.  It receives
    exactly the correlation data needed for detector/guardrails calls, and no
    watermark key material.
    """

    def __init__(self, detector: DetectorAdapter, guardrails: GuardrailsAdapter) -> None:
        self._detector = detector
        self._guardrails = guardrails

    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        verdict = await self._detector.detect(request)
        if verdict not in ("watermarked", "clean"):
            raise TerminalValidationError("detector returned an invalid verdict")
        action = await self._guardrails.apply(request, verdict)
        if not isinstance(action, str) or not action:
            raise TerminalValidationError("guardrails returned an invalid action")
        return ValidationOutcome(verdict=verdict, managed_action=action)


class ManagedNeMoClient(Protocol):
    """Transport boundary for the version-pinned managed-NeMo endpoint."""

    # The concrete client parses its outer response to exactly ``blocked`` or
    # ``success``.  It must not derive a watermark verdict from that response.
    async def validate(self, text: str, context: Mapping[str, str]) -> str: ...


class ManagedNeMoAdapter:
    """Adapter that sends correlation metadata in NeMo's ``guardrails.context``.

    The concrete client is deliberately injected because the managed endpoint's
    URL and outer request/response envelope are deployment-specific.  Its custom
    action calls the gateway's internal broker, which performs the detector call.
    """

    def __init__(self, client: ManagedNeMoClient, broker: "PendingValidationBroker") -> None:
        self._client = client
        self._broker = broker

    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        context = {
            "watermark_validation_id": request.validation_id,
            "watermark_response_id": request.response_id,
            "watermark_content_sha256": request.content_sha256,
            "watermark_scheme": request.scheme,
            "watermark_key_id": request.key_id,
        }
        action = await self._client.validate(request.content, context)
        if action not in ("blocked", "success"):
            raise TerminalValidationError("managed NeMo returned an invalid action state")
        verdict = await self._broker.detector_result(request.validation_id)
        expected_action = "blocked" if verdict == "watermarked" else "success"
        if action != expected_action:
            raise TerminalValidationError("managed NeMo action does not match the detector verdict")
        return ValidationOutcome(verdict=verdict, managed_action=action)


class PendingValidationBroker:
    """In-memory bridge for an idempotent managed-NeMo active operation.

    It retains plaintext only while the selected response is actively being
    validated.  SQLite records never receive plaintext and the broker has no
    logging path.
    """

    def __init__(self, detector: DetectorAdapter) -> None:
        self._detector = detector
        self._pending: dict[str, ValidationRequest] = {}
        self._detector_tasks: dict[str, asyncio.Task[Verdict]] = {}

    def register(self, request: ValidationRequest) -> None:
        # The operation ID is generated by this gateway, and response IDs have
        # already been validated at the upstream boundary.  Recheck before
        # retaining either in the plaintext-bearing pending map.
        try:
            parsed = uuid.UUID(request.validation_id)
        except ValueError as exc:
            raise TerminalValidationError("validation id is not a canonical UUID") from exc
        if str(parsed) != request.validation_id:
            raise TerminalValidationError("validation id is not a canonical UUID")
        _response_id(request.response_id)
        if request.validation_id in self._pending or request.validation_id in self._detector_tasks:
            raise TerminalValidationError("validation id is already active")
        self._pending[request.validation_id] = request

    async def unregister(self, validation_id: str) -> None:
        self._pending.pop(validation_id, None)
        task = self._detector_tasks.pop(validation_id, None)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def resolve(self, body: Mapping[str, Any]) -> dict[str, str | bool]:
        expected_keys = {"validation_id", "response_id", "content_sha256", "scheme", "key_id"}
        if set(body) != expected_keys:
            raise UnsafeRequest("guardrail-action body must contain exactly validation_id, response_id, content_sha256, scheme, key_id")
        raw_id, response_id, digest, scheme, key_id = (body[key] for key in ("validation_id", "response_id", "content_sha256", "scheme", "key_id"))
        try:
            parsed_validation_id = uuid.UUID(raw_id) if isinstance(raw_id, str) else None
        except ValueError as exc:
            raise UnsafeRequest("validation_id must be a canonical lowercase UUID") from exc
        if parsed_validation_id is None or str(parsed_validation_id) != raw_id:
            raise UnsafeRequest("validation_id must be a canonical lowercase UUID")
        try:
            _response_id(response_id)
        except TerminalValidationError as exc:
            raise UnsafeRequest("response_id must be a bounded safe identifier") from exc
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise UnsafeRequest("content_sha256 must be a canonical SHA-256 digest")
        if not isinstance(scheme, str) or scheme not in {"kgw", "synthid"}:
            raise UnsafeRequest("scheme must be exact lowercase kgw or synthid")
        try:
            normalized_key_id = _key_id(key_id)
        except UnsafeRequest as exc:
            raise UnsafeRequest("key_id is invalid") from exc
        request = self._pending.get(raw_id)
        if request is None:
            raise UnsafeRequest("unknown or expired validation_id")
        if not hmac.compare_digest(request.content_sha256, digest):
            raise UnsafeRequest("guardrail-action digest does not match pending validation")
        if response_id != request.response_id or scheme != request.scheme or normalized_key_id != request.key_id:
            raise UnsafeRequest("guardrail-action metadata does not match pending validation")
        detector_task = self._detector_tasks.get(raw_id)
        if detector_task is None:
            detector_task = asyncio.create_task(self._detector.detect(request))
            self._detector_tasks[raw_id] = detector_task
        # Keep a retryable task available for detector_result().  The managed
        # action can map this broker failure to its outer action state; the
        # follow-up detector_result call owns retry-task eviction.
        verdict = await self._await_detector(detector_task)
        if verdict not in ("watermarked", "clean"):
            raise TerminalValidationError("detector returned an invalid verdict")
        return {
            "validation_id": request.validation_id,
            "response_id": request.response_id,
            "content_sha256": request.content_sha256,
            "scheme": request.scheme,
            "key_id": request.key_id,
            "verdict": verdict == "watermarked",
        }

    async def detector_result(self, validation_id: str) -> Verdict:
        """Return the exact broker result; it is never inferred from NeMo."""
        request = self._pending.get(validation_id)
        if request is None:
            raise TerminalValidationError("missing or expired broker validation")
        task = self._detector_tasks.get(validation_id)
        if task is None:
            raise RetryableValidationError("managed NeMo did not deliver a detector broker call")
        try:
            verdict = await self._await_detector(task)
        except RetryableValidationError:
            # Evict only the exact failed task.  A concurrent/new operation
            # must never be removed by a stale waiter.
            if self._detector_tasks.get(validation_id) is task:
                self._detector_tasks.pop(validation_id, None)
            raise
        if verdict not in ("watermarked", "clean"):
            raise TerminalValidationError("detector returned an invalid verdict")
        return verdict

    async def _await_detector(self, task: asyncio.Task[Verdict]) -> Verdict:
        return await asyncio.shield(task)


class SQLiteSampler:
    """A persistent ordinal counter protected by an OS lifetime singleton lock."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection: sqlite3.Connection | None = None
        self._lock_handle: Any | None = None

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(self._path.suffix + ".singleton.lock")
        # SQLite records contain hashes/key IDs.  Create and re-harden both
        # durable files instead of depending on the process umask/PVC default.
        for protected_path in (self._path, lock_path):
            descriptor = os.open(protected_path, os.O_RDWR | os.O_CREAT, 0o600)
            os.close(descriptor)
            os.chmod(protected_path, 0o600)
        self._lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            self._lock_handle = None
            raise ConfigurationError("another validation gateway owns the SQLite sampler") from exc
        self._connection = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS sampler_state (id INTEGER PRIMARY KEY CHECK(id=1), ordinal INTEGER NOT NULL)")
        self._connection.execute("INSERT OR IGNORE INTO sampler_state(id, ordinal) VALUES (1, 0)")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS validation_records ("
            "validation_id TEXT PRIMARY KEY, response_id TEXT NOT NULL, ordinal INTEGER NOT NULL, run_id TEXT, "
            "content_sha256 TEXT NOT NULL, expected_enabled INTEGER NOT NULL, scheme TEXT NOT NULL, key_id TEXT NOT NULL, "
            "status TEXT NOT NULL, terminal_state TEXT, verdict TEXT, managed_action TEXT, attempts INTEGER NOT NULL DEFAULT 0, "
            "delivery_outcome TEXT, detector_call_id TEXT, guardrails_action_id TEXT, "
            "generation_completion_seconds REAL, validation_latency_seconds REAL, validation_lag_seconds REAL, "
            "client_delivery_seconds REAL, created_at REAL NOT NULL, completed_at REAL)"
        )
        existing = {row[1] for row in self._connection.execute("PRAGMA table_info(validation_records)")}
        migrations = {
            "run_id": "TEXT",
            "terminal_state": "TEXT",
            "delivery_outcome": "TEXT",
            "detector_call_id": "TEXT",
            "guardrails_action_id": "TEXT",
            "generation_completion_seconds": "REAL",
            "validation_latency_seconds": "REAL",
            "validation_lag_seconds": "REAL",
            "client_delivery_seconds": "REAL",
        }
        for column, column_type in migrations.items():
            if column not in existing:
                self._connection.execute(f"ALTER TABLE validation_records ADD COLUMN {column} {column_type}")
        self._harden_permissions()

    def _harden_permissions(self) -> None:
        for protected_path in (
            self._path,
            self._path.with_suffix(self._path.suffix + "-wal"),
            self._path.with_suffix(self._path.suffix + "-shm"),
            self._path.with_suffix(self._path.suffix + ".singleton.lock"),
        ):
            if protected_path.exists():
                os.chmod(protected_path, 0o600)

    @staticmethod
    def _require_one(cursor: sqlite3.Cursor, operation: str) -> None:
        if cursor.rowcount != 1:
            raise RuntimeError(f"sampler {operation} did not affect exactly one record")

    def claim(self, request: ValidationRequest, sample_every: int, run_id: str | None = None) -> tuple[int, bool]:
        """Atomically advance the completed ordinal and create selected state."""
        if self._connection is None:
            raise RuntimeError("sampler is not started")
        # Enforce this again at the persistence boundary.  A caller cannot
        # smuggle arbitrary upstream text into the hash-only SQLite record.
        _response_id(request.response_id)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            ordinal = int(self._connection.execute("SELECT ordinal FROM sampler_state WHERE id=1").fetchone()[0]) + 1
            self._connection.execute("UPDATE sampler_state SET ordinal=? WHERE id=1", (ordinal,))
            selected = ordinal % sample_every == 0
            if selected:
                self._connection.execute(
                    "INSERT INTO validation_records("
                    "validation_id,response_id,ordinal,run_id,content_sha256,expected_enabled,scheme,key_id,status,"
                    "detector_call_id,guardrails_action_id,generation_completion_seconds,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        request.validation_id, request.response_id, ordinal, run_id, request.content_sha256,
                        int(request.expected_enabled), request.scheme, request.key_id, "pending",
                        request.detector_call_id, request.guardrails_action_id,
                        request.generation_completion_seconds, time.time(),
                    ),
                )
            self._connection.execute("COMMIT")
            self._harden_permissions()
            return ordinal, selected
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def update_attempts(self, validation_id: str, attempts: int) -> None:
        assert self._connection is not None
        cursor = self._connection.execute(
            "UPDATE validation_records SET attempts=? WHERE validation_id=?",
            (attempts, validation_id),
        )
        self._require_one(cursor, "attempt update")

    def finish_record(
        self,
        validation_id: str,
        *,
        status: str,
        terminal_state: str,
        attempts: int,
        outcome: ValidationOutcome | None = None,
        validation_latency_seconds: float,
        validation_lag_seconds: float,
    ) -> None:
        assert self._connection is not None
        cursor = self._connection.execute(
            "UPDATE validation_records SET status=?, terminal_state=?, verdict=?, managed_action=?, attempts=?, "
            "validation_latency_seconds=?, validation_lag_seconds=?, completed_at=? WHERE validation_id=? AND status='pending'",
            (
                status, terminal_state, outcome.verdict if outcome else None, outcome.managed_action if outcome else None,
                attempts, validation_latency_seconds, validation_lag_seconds, time.time(), validation_id,
            ),
        )
        self._require_one(cursor, "terminal transition")
        self._harden_permissions()

    def set_delivery_outcome(self, validation_id: str, delivery_outcome: str, client_delivery_seconds: float) -> None:
        assert self._connection is not None
        cursor = self._connection.execute(
            "UPDATE validation_records SET delivery_outcome=?, client_delivery_seconds=? WHERE validation_id=? AND delivery_outcome IS NULL",
            (delivery_outcome, client_delivery_seconds, validation_id),
        )
        self._require_one(cursor, "delivery transition")
        self._harden_permissions()

    def terminalize_stale_pending(self) -> int:
        assert self._connection is not None
        cursor = self._connection.execute(
            "UPDATE validation_records SET status='restart_interrupted', terminal_state='restart_interrupted', verdict='error', "
            "delivery_outcome='restart_interrupted', completed_at=? WHERE status='pending'",
            (time.time(),),
        )
        self._harden_permissions()
        return cursor.rowcount

    def records(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return hash-only records for the deterministic acceptance harness."""
        assert self._connection is not None
        columns = (
            "validation_id", "response_id", "ordinal", "run_id", "content_sha256", "expected_enabled", "scheme", "key_id",
            "status", "terminal_state", "verdict", "managed_action", "attempts", "delivery_outcome",
            "detector_call_id", "guardrails_action_id", "generation_completion_seconds", "validation_latency_seconds",
            "validation_lag_seconds", "client_delivery_seconds", "created_at", "completed_at",
        )
        query = f"SELECT {','.join(columns)} FROM validation_records"
        args: tuple[str, ...] = ()
        if run_id is not None:
            query += " WHERE run_id=?"
            args = (run_id,)
        query += " ORDER BY ordinal"
        return [dict(zip(columns, row, strict=True)) for row in self._connection.execute(query, args)]

    def reset(self) -> None:
        """Reset only the sampler state and hash-only records for a fixed harness run."""
        assert self._connection is not None
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute("DELETE FROM validation_records")
            self._connection.execute("UPDATE sampler_state SET ordinal=0 WHERE id=1")
            self._connection.execute("COMMIT")
            self._harden_permissions()
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None


class PrometheusMetrics:
    """Small bounded-label Prometheus exposition; never accepts response data."""

    _ALLOWED_LABEL_VALUES = {
        "outcome": frozenset({"started", "completed", "failed", "cancelled"}),
        "reason": frozenset({"queue_full", "test_queue_full"}),
        "component": frozenset({"managed", "detector", "guardrails"}),
        "verdict": frozenset({"watermarked", "clean", "error"}),
        "mode": frozenset({"synchronous"}),
    }
    _HISTOGRAM_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0)

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._gauges: dict[str, float] = {"validation_queue_depth": 0.0}
        self._histograms: dict[str, dict[str, float | int | list[int]]] = {}

    def reset(self) -> None:
        self._counters.clear()
        self._gauges = {"validation_queue_depth": 0.0}
        self._histograms.clear()

    def inc(self, name: str, **labels: str) -> None:
        self._validate_metric(name, labels)
        key = (name, tuple(sorted(labels.items())))
        self._counters[key] = self._counters.get(key, 0) + 1

    def gauge(self, name: str, value: float) -> None:
        self._validate_metric(name, {})
        self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        self._validate_metric(name, {})
        if not math.isfinite(value) or value < 0:
            raise ValueError("metric observation must be finite and non-negative")
        histogram = self._histograms.setdefault(name, {"count": 0, "sum": 0.0, "buckets": [0] * len(self._HISTOGRAM_BUCKETS)})
        histogram["count"] = int(histogram["count"]) + 1
        histogram["sum"] = float(histogram["sum"]) + value
        buckets = histogram["buckets"]
        assert isinstance(buckets, list)
        for index, upper_bound in enumerate(self._HISTOGRAM_BUCKETS):
            if value <= upper_bound:
                buckets[index] += 1

    @classmethod
    def _validate_metric(cls, name: str, labels: Mapping[str, str]) -> None:
        if not name.startswith("validation_"):
            raise ValueError("metric names must use the validation_ namespace")
        for key, value in labels.items():
            allowed_values = cls._ALLOWED_LABEL_VALUES.get(key)
            if allowed_values is None or value not in allowed_values:
                raise ValueError("metric labels must use the bounded D10 allowlist")

    def histogram_count(self, name: str) -> int:
        histogram = self._histograms.get(name)
        return int(histogram["count"]) if histogram else 0

    @staticmethod
    def _names(name: str) -> tuple[str, ...]:
        return (name, f"watermark_{name}") if name.startswith("validation_") else (name,)

    def render(self) -> str:
        lines: list[str] = []
        for (name, labels), value in sorted(self._counters.items()):
            rendered = "" if not labels else "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}"
            for exposed_name in self._names(name):
                lines.append(f"{exposed_name}_total{rendered} {value}")
        for name, value in sorted(self._gauges.items()):
            for exposed_name in self._names(name):
                lines.append(f"{exposed_name} {value}")
        for name, histogram in sorted(self._histograms.items()):
            buckets = histogram["buckets"]
            assert isinstance(buckets, list)
            for exposed_name in self._names(name):
                for upper_bound, count in zip(self._HISTOGRAM_BUCKETS, buckets, strict=True):
                    lines.append(f'{exposed_name}_bucket{{le="{upper_bound}"}} {count}')
                lines.append(f'{exposed_name}_bucket{{le="+Inf"}} {histogram["count"]}')
                lines.append(f"{exposed_name}_count {histogram['count']}")
                lines.append(f"{exposed_name}_sum {histogram['sum']}")
        return "\n".join(lines) + "\n"


@dataclass
class _Job:
    request: ValidationRequest
    future: asyncio.Future[ValidationOutcome]
    enqueued_at: float


class GatewayService:
    def __init__(self, config: GatewayConfig, upstream: UpstreamTransport, validator: ValidationAdapter, *, broker: PendingValidationBroker | None = None) -> None:
        self.config, self.upstream, self.validator = config, upstream, validator
        self.sampler = SQLiteSampler(config.sqlite_path)
        self.broker = broker
        self.metrics = PrometheusMetrics()
        self._queue: asyncio.Queue[_Job | None] = asyncio.Queue(maxsize=config.queue_capacity)
        self._workers: list[asyncio.Task[None]] = []
        self._started = False
        self._sample_every = config.sample_every
        self._run_id: str | None = None
        self._counters = self._empty_counters()
        self._slots_in_use = 0
        self._peak_depth = 0
        self._active_jobs = 0
        self._consumer_paused = False
        self._test_capacity: int | None = None
        self._fault: tuple[str, int] | None = None
        self._events: list[dict[str, str]] = []

    @staticmethod
    def _empty_counters() -> dict[str, int]:
        return {
            "started": 0, "completed": 0, "selected": 0, "unsampled": 0,
            "terminal": 0, "watermarked": 0, "clean": 0, "errors": 0,
            "failed": 0, "cancelled": 0, "detector_attempts": 0,
            "guardrails_attempts": 0, "retries": 0, "queue_overflow": 0,
            "dropped": 0,
        }

    def _counter(self, name: str, amount: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + amount

    def _event(self, event: str, state: str) -> None:
        # Explicit fixed fields only: no IDs, digests, text, headers, or errors.
        if len(self._events) >= 256:
            self._events.pop(0)
        self._events.append({"event": event, "state": state})

    def _start_workers(self) -> None:
        self._workers = [asyncio.create_task(self._worker(), name=f"validation-worker-{i}") for i in range(self.config.max_inflight)]

    def _capacity(self) -> int:
        return self._test_capacity if self._test_capacity is not None else self.config.queue_capacity

    async def start(self) -> None:
        if self._started:
            return
        self.sampler.start()
        stale = self.sampler.terminalize_stale_pending()
        for _ in range(stale):
            self.metrics.inc("validation_terminal", verdict="error")
            self._counter("terminal")
            self._counter("errors")
        self._start_workers()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self._consumer_paused:
            self._consumer_paused = False
            self._start_workers()
        for _ in self._workers:
            await self._queue.put(None)
        await asyncio.gather(*self._workers)
        self._workers = []
        self.sampler.close()
        self._started = False

    @property
    def ready(self) -> bool:
        return self._started

    def records(self) -> list[dict[str, Any]]:
        return self.sampler.records()

    def reset_sampler_for_harness(self) -> None:
        if self._started:
            raise RuntimeError("stop the gateway before resetting its sampler")
        self.sampler.start()
        try:
            self.sampler.reset()
        finally:
            self.sampler.close()

    async def reset_for_run(self, run_id: str, sample_every: Any) -> None:
        if not self._started:
            raise RuntimeError("gateway is not started")
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,160}", run_id):
            raise UnsafeRequest("run_id is invalid")
        parsed = _strict_positive_int(sample_every, "validation_sample_every")
        if parsed not in (1, 5):
            raise UnsafeRequest("test reset supports validation_sample_every 1 or 5 only")
        if self._consumer_paused or self._slots_in_use or self._active_jobs or not self._queue.empty():
            raise RuntimeError("cannot reset while validation queue is not drained")
        self.sampler.reset()
        self.metrics.reset()
        self._counters = self._empty_counters()
        self._events.clear()
        self._sample_every = parsed
        self._run_id = run_id
        self._peak_depth = 0
        self._test_capacity = None
        self._fault = None
        self._event("reset", "accepted")

    @staticmethod
    def validate_sample_every(value: Any) -> bool:
        try:
            _strict_positive_int(value, "validation_sample_every")
        except ConfigurationError:
            return False
        return True

    def status(self, run_id: str | None) -> dict[str, Any]:
        if run_id != self._run_id:
            raise UnsafeRequest("unknown run_id")
        return {
            "run_id": self._run_id,
            "counters": dict(self._counters),
            "queue": {
                "depth": self._slots_in_use,
                "peak_depth": self._peak_depth,
                "capacity": self._capacity(),
                "overflow_policy": "non_blocking",
                "consumer": "paused" if self._consumer_paused else "running",
            },
            "latency_samples": {
                "generation_completion": self.metrics.histogram_count("validation_generation_completion_seconds"),
                "client_delivery": self.metrics.histogram_count("validation_client_delivery_seconds"),
                "validation": self.metrics.histogram_count("validation_latency_seconds"),
                "validation_lag": self.metrics.histogram_count("validation_lag_seconds"),
            },
        }

    def harness_records(self, run_id: str | None) -> list[dict[str, Any]]:
        if run_id != self._run_id:
            raise UnsafeRequest("unknown run_id")
        records: list[dict[str, Any]] = []
        for row in self.sampler.records(run_id):
            terminal_state = row["terminal_state"] or ("success" if row["status"] == "terminal" else "error")
            result: dict[str, Any] = {
                "validation_id": row["validation_id"],
                "response_id": row["response_id"],
                "content_digest": row["content_sha256"],
                "scheme": row["scheme"],
                "key_id": row["key_id"],
                "mode": "synchronous",
                "attempts": row["attempts"],
                "terminal_state": terminal_state,
                "delivery_outcome": row["delivery_outcome"],
            }
            if terminal_state == "success":
                managed_action = row["managed_action"]
                if managed_action not in {"blocked", "success"}:
                    raise RuntimeError("successful validation record has an invalid managed action")
                result.update({
                    "verdict": row["verdict"] == "watermarked",
                    "managed_action": managed_action,
                    "timing": {
                        "validation_latency_seconds": row["validation_latency_seconds"] or 0.0,
                        "validation_lag_seconds": row["validation_lag_seconds"] or 0.0,
                    },
                    "detector_call_id": row["detector_call_id"],
                    "guardrails_action_id": row["guardrails_action_id"],
                    # The raw managed state remains stored as managed_action;
                    # this is the stable acceptance action vocabulary.
                    "guardrails_action": "block" if row["verdict"] == "watermarked" else "pass",
                })
            records.append(result)
        return records

    async def configure_fault(self, run_id: str, scenario: Any, max_attempts: Any) -> None:
        if not self.config.test_controls:
            raise UnsafeRequest("test controls are disabled")
        if run_id != self._run_id:
            raise UnsafeRequest("unknown run_id")
        if scenario not in {"retry_then_success", "retry_exhausted", "malformed_success"}:
            raise UnsafeRequest("unknown fault scenario")
        attempts = _strict_positive_int(max_attempts, "max_attempts")
        if attempts != 3:
            raise UnsafeRequest("test fault max_attempts must be 3")
        if self._fault is not None:
            raise RuntimeError("a test fault is already armed")
        self._fault = (scenario, attempts)
        self._event("fault", "armed")

    async def set_consumer_state(self, run_id: str, state: Any, capacity: Any = None) -> None:
        if not self.config.test_controls:
            raise UnsafeRequest("test controls are disabled")
        if run_id != self._run_id:
            raise UnsafeRequest("unknown run_id")
        if state == "paused":
            parsed_capacity = _strict_positive_int(capacity, "capacity")
            if parsed_capacity != 2:
                raise UnsafeRequest("test consumer capacity must be 2")
            if self._active_jobs or self._slots_in_use or not self._queue.empty():
                raise RuntimeError("consumer can only pause with a drained queue")
            for worker in self._workers:
                worker.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = []
            self._consumer_paused = True
            self._test_capacity = parsed_capacity
            self._event("consumer", "paused")
            return
        if state == "running":
            if not self._consumer_paused:
                raise UnsafeRequest("consumer is already running")
            self._consumer_paused = False
            self._start_workers()
            self._event("consumer", "running")
            return
        raise UnsafeRequest("consumer state must be paused or running")

    async def proxy(self, endpoint: str, body: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._started:
            raise RuntimeError("gateway is not started")
        metadata = self._validate_request(body)
        upstream_body = self._strip_gateway_metadata(body)
        generation_started = time.monotonic()
        self.metrics.inc("validation_requests", outcome="started")
        self._counter("started")
        try:
            response = await self.upstream.complete(endpoint, upstream_body)
            response_id, content = self._extract_completed_text(endpoint, response)
            response_id = _response_id(response_id)
        except asyncio.CancelledError:
            self.metrics.inc("validation_requests", outcome="cancelled")
            self._counter("cancelled")
            raise
        except Exception:
            self.metrics.inc("validation_requests", outcome="failed")
            self._counter("failed")
            raise
        generated_at = time.monotonic()
        self.metrics.inc("validation_requests", outcome="completed")
        self.metrics.inc("validation_responses", outcome="completed")
        self._counter("completed")
        generation_seconds = generated_at - generation_started
        self.metrics.observe("validation_generation_completion_seconds", generation_seconds)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        validation_id = str(uuid.uuid4())
        request = ValidationRequest(
            validation_id=validation_id,
            response_id=response_id,
            content=content,
            content_sha256=digest,
            expected_enabled=metadata.expected_enabled,
            scheme=metadata.scheme,
            key_id=metadata.key_id,
            # This ID is propagated in the broker request and NeMo context;
            # do not manufacture component IDs that no downstream saw.
            detector_call_id=validation_id,
            guardrails_action_id=validation_id,
            generation_completion_seconds=generation_seconds,
        )
        ordinal, selected = self.sampler.claim(request, self._sample_every, self._run_id)
        if not selected:
            self.metrics.inc("validation_unsampled")
            self._counter("unsampled")
            return self._deliver_response(response, request, ordinal, False, generation_started, delivery_outcome="delivered")
        self.metrics.inc("validation_selected")
        self._counter("selected")
        future: asyncio.Future[ValidationOutcome] = asyncio.get_running_loop().create_future()
        job = _Job(request=request, future=future, enqueued_at=time.monotonic())
        if self._slots_in_use >= self._capacity() or self._queue.full():
            self.metrics.inc("validation_capacity", reason="queue_full")
            self.metrics.inc("validation_dropped_items", reason="queue_full")
            self.metrics.inc("validation_terminal", verdict="error")
            self._counter("queue_overflow")
            self._counter("dropped")
            self._counter("terminal")
            self._counter("errors")
            self.sampler.finish_record(
                request.validation_id, status="error", terminal_state="queue_overflow", attempts=0,
                validation_latency_seconds=0.0, validation_lag_seconds=0.0,
            )
            return self._failure_response(response, request, ordinal, generation_started)
        self._slots_in_use += 1
        self._peak_depth = max(self._peak_depth, self._slots_in_use)
        self.metrics.gauge("validation_queue_depth", float(self._slots_in_use))
        self._queue.put_nowait(job)
        try:
            outcome = await future
        except asyncio.CancelledError:
            # Generation already completed and claimed an ordinal.  Preserve one
            # validation record, make the client outcome explicit, and let the
            # worker finish/cancel its broker task independently.
            self.sampler.set_delivery_outcome(request.validation_id, "client_cancelled", time.monotonic() - generation_started)
            self.metrics.inc("validation_requests", outcome="cancelled")
            self._counter("cancelled")
            raise
        except Exception:
            return self._failure_response(response, request, ordinal, generation_started)
        if outcome.verdict == "watermarked":
            if self.config.positive_policy == "block":
                self.sampler.set_delivery_outcome(request.validation_id, "positive_blocked", time.monotonic() - generation_started)
                self.metrics.inc("validation_positive_blocks")
                raise PositiveValidationBlock(self._safe_metadata(request, ordinal, True, generation_seconds, time.monotonic() - generation_started, "positive_blocked"))
            self.sampler.set_delivery_outcome(request.validation_id, "delivered", time.monotonic() - generation_started)
            self.metrics.inc("validation_positive_flag_deliveries")
        else:
            self.sampler.set_delivery_outcome(request.validation_id, "delivered", time.monotonic() - generation_started)
        return self._deliver_response(response, request, ordinal, True, generation_started, delivery_outcome="delivered")

    def _failure_response(self, response: Mapping[str, Any], request: ValidationRequest, ordinal: int, started: float) -> Mapping[str, Any]:
        if self.config.failure_policy == "open":
            self.sampler.set_delivery_outcome(request.validation_id, "fail_open", time.monotonic() - started)
            self.metrics.inc("validation_fail_open_deliveries")
            return self._deliver_response(response, request, ordinal, True, started, delivery_outcome="fail_open")
        elapsed = time.monotonic() - started
        self.sampler.set_delivery_outcome(request.validation_id, "fail_closed", elapsed)
        self.metrics.inc("validation_fail_closed_blocks")
        raise ValidationDeliveryBlocked(self._safe_metadata(request, ordinal, True, request.generation_completion_seconds, elapsed, "fail_closed"))

    def _safe_metadata(self, request: ValidationRequest, ordinal: int, selected: bool, generation_seconds: float, delivery_seconds: float, delivery_outcome: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "response_id": request.response_id,
            "validation_id": request.validation_id,
            "content_digest": request.content_sha256,
            "ordinal": ordinal,
            "selected": selected,
            "generation_completion_latency_seconds": generation_seconds,
            "client_delivery_latency_seconds": delivery_seconds,
        }
        if delivery_outcome is not None:
            result["delivery_outcome"] = delivery_outcome
        return result

    def _deliver_response(
        self,
        response: Mapping[str, Any],
        request: ValidationRequest,
        ordinal: int,
        selected: bool,
        started: float,
        *,
        delivery_outcome: str | None = None,
        already_observed: bool = False,
    ) -> Mapping[str, Any]:
        delivery_seconds = time.monotonic() - started
        if not already_observed:
            self.metrics.observe("validation_client_delivery_seconds", delivery_seconds)
        result = dict(response)
        result["watermark_validation"] = self._safe_metadata(
            request, ordinal, selected, request.generation_completion_seconds, delivery_seconds, delivery_outcome,
        )
        return result

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            if job is None:
                self._queue.task_done()
                return
            attempts = 0
            started = time.monotonic()
            self._active_jobs += 1
            fault = self._fault
            self._fault = None
            maximum_attempts = fault[1] if fault else self.config.max_attempts
            try:
                if self.broker is not None:
                    self.broker.register(job.request)
                while True:
                    attempts += 1
                    self.sampler.update_attempts(job.request.validation_id, attempts)
                    self.metrics.inc("validation_attempts", component="managed")
                    try:
                        if fault is not None and fault[0] == "retry_exhausted":
                            raise RetryableValidationError("test retry exhaustion")
                        if fault is not None and fault[0] == "retry_then_success" and attempts == 1:
                            raise RetryableValidationError("test first retry")
                        if fault is not None and fault[0] == "malformed_success":
                            raise TerminalValidationError("test malformed response")
                        self._counter("detector_attempts")
                        self._counter("guardrails_attempts")
                        outcome = await asyncio.wait_for(
                            self.validator.validate(job.request),
                            timeout=self.config.attempt_timeout_seconds,
                        )
                    except (RetryableValidationError, TimeoutError) as exc:
                        if attempts >= maximum_attempts or time.monotonic() - started >= self.config.max_retry_elapsed_seconds:
                            if isinstance(exc, TimeoutError):
                                raise RetryableValidationError("validation attempt timed out") from exc
                            raise
                        self.metrics.inc("validation_retries")
                        self._counter("retries")
                        remaining = self.config.max_retry_elapsed_seconds - (time.monotonic() - started)
                        await asyncio.sleep(min(self.config.retry_backoff_seconds, max(remaining, 0.0)))
                        if time.monotonic() - started >= self.config.max_retry_elapsed_seconds:
                            if isinstance(exc, TimeoutError):
                                raise RetryableValidationError("validation retry deadline elapsed") from exc
                            raise
                        continue
                    latency = time.monotonic() - started
                    lag = started - job.enqueued_at
                    self.sampler.finish_record(
                        job.request.validation_id, status="terminal", terminal_state="success", attempts=attempts,
                        outcome=outcome, validation_latency_seconds=latency, validation_lag_seconds=lag,
                    )
                    self.metrics.inc("validation_terminal", verdict=outcome.verdict)
                    self._counter("terminal")
                    self._counter(outcome.verdict)
                    self.metrics.observe("validation_latency_seconds", latency)
                    self.metrics.observe("validation_lag_seconds", lag)
                    if not job.future.done():
                        job.future.set_result(outcome)
                    break
            except Exception as exc:
                terminal_state = "retry_exhausted" if isinstance(exc, RetryableValidationError) else "malformed_response" if fault and fault[0] == "malformed_success" else "error"
                latency = time.monotonic() - started
                lag = started - job.enqueued_at
                self.sampler.finish_record(
                    job.request.validation_id, status="error", terminal_state=terminal_state, attempts=attempts,
                    validation_latency_seconds=latency, validation_lag_seconds=lag,
                )
                self.metrics.inc("validation_terminal", verdict="error")
                self._counter("terminal")
                self._counter("errors")
                self.metrics.observe("validation_latency_seconds", latency)
                self.metrics.observe("validation_lag_seconds", lag)
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                if self.broker is not None:
                    await self.broker.unregister(job.request.validation_id)
                self._active_jobs -= 1
                self._slots_in_use -= 1
                self.metrics.gauge("validation_queue_depth", float(self._slots_in_use))
                self._queue.task_done()

    def _validate_request(self, body: Mapping[str, Any]) -> ResponseMetadata:
        if not isinstance(body, Mapping):
            raise UnsafeRequest("request body must be a JSON object")
        if body.get("stream", False) is not False:
            raise UnsafeRequest("streaming is unsupported by synchronous validation")
        if "n" in body and body["n"] != 1:
            raise UnsafeRequest("only n=1 is supported")
        for key in ("best_of", "tools", "tool_choice"):
            if key in body:
                raise UnsafeRequest(f"{key} is unsupported by this gateway")
        args = body.get("vllm_xargs", {})
        if not isinstance(args, Mapping):
            raise UnsafeRequest("vllm_xargs must be a JSON object")
        enabled = self.config.default_watermark_enabled if "watermark" not in args else _watermark_flag(args["watermark"])
        if "watermark_scheme" not in args or "watermark_key_id" not in args:
            raise UnsafeRequest("vllm_xargs.watermark_scheme and vllm_xargs.watermark_key_id are required")
        scheme = _scheme(args["watermark_scheme"])
        key_id = _key_id(args["watermark_key_id"])
        return ResponseMetadata(enabled, scheme, key_id)

    @staticmethod
    def _strip_gateway_metadata(body: Mapping[str, Any]) -> Mapping[str, Any]:
        """Do not forward gateway-only harness correlation metadata to vLLM."""
        result = dict(body)
        metadata = result.get("metadata")
        if metadata is None:
            return result
        if not isinstance(metadata, Mapping):
            raise UnsafeRequest("metadata must be a JSON object")
        gateway_key = "watermark_validation_request_id"
        unexpected_gateway_keys = {key for key in metadata if key.startswith("watermark_validation_")} - {gateway_key}
        if unexpected_gateway_keys:
            raise UnsafeRequest("unknown gateway metadata field")
        if "validation_request_id" in metadata:
            raise UnsafeRequest("use watermark_validation_request_id for gateway metadata")
        copied = dict(metadata)
        if gateway_key in copied:
            request_id = copied.pop(gateway_key)
            if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{1,160}", request_id):
                raise UnsafeRequest("watermark_validation_request_id is invalid")
        if copied:
            result["metadata"] = copied
        else:
            result.pop("metadata", None)
        return result

    @staticmethod
    def _extract_completed_text(endpoint: str, response: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(response, Mapping) or not isinstance(response.get("id"), str):
            raise TerminalValidationError("upstream response lacks a string id")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise TerminalValidationError("upstream response must contain exactly one choice")
        choice = choices[0]
        if endpoint == "/v1/completions":
            content = choice.get("text")
        elif endpoint == "/v1/chat/completions":
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
        else:
            raise UnsafeRequest("unsupported OpenAI endpoint")
        if not isinstance(content, str):
            raise TerminalValidationError("upstream response has no generated text content")
        return response["id"], content


def create_app(service: GatewayService) -> FastAPI:
    """Create the minimal gateway ASGI app with readiness and metrics."""
    if FastAPI is Any:  # pragma: no cover
        raise RuntimeError("fastapi is required to create the gateway app")
    if not service.config.broker_token:
        raise ConfigurationError("VALIDATION_BROKER_TOKEN is required to expose the broker route")
    if not service.config.admin_token:
        raise ConfigurationError("VALIDATION_ADMIN_TOKEN is required to expose admin routes")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    async def handle(endpoint: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise UnsafeRequest("request body must be a JSON object")
            return JSONResponse(await service.proxy(endpoint, body))
        except UnsafeRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PositiveValidationBlock as exc:
            return JSONResponse(
                {
                    "error": {"message": "watermarked response blocked", "type": "watermark_validation_block"},
                    "watermark_validation": exc.metadata,
                },
                status_code=403,
            )
        except ValidationDeliveryBlocked as exc:
            return JSONResponse(
                {
                    "error": {"message": "response withheld by validation", "type": "watermark_validation_error"},
                    "watermark_validation": exc.metadata,
                },
                status_code=503,
            )
        except TerminalValidationError as exc:
            raise HTTPException(status_code=503, detail="validation unavailable") from exc

    def require_admin(request: Request) -> None:
        authorization = request.headers.get("authorization")
        expected = f"Bearer {service.config.admin_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="unauthorized")

    def exact_object(body: Any, keys: set[str]) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) != keys:
            raise UnsafeRequest("invalid admin request")
        return body

    @app.post("/v1/completions")
    async def completions(request: Request) -> JSONResponse:
        return await handle("/v1/completions", request)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        return await handle("/v1/chat/completions", request)

    @app.get("/ready")
    async def ready() -> JSONResponse:
        if service.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not-ready"}, status_code=503)

    @app.get("/health")
    async def health() -> JSONResponse:
        # Liveness deliberately does not report sampler/worker readiness.
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(service.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/continuous-validation/admin/reset")
    async def admin_reset(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            body = exact_object(await request.json(), {"validation_sample_every", "run_id"})
            await service.reset_for_run(body["run_id"], body["validation_sample_every"])
            return JSONResponse({"accepted": True})
        except UnsafeRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError:
            raise HTTPException(status_code=409, detail="queue not drained")

    @app.get("/v1/continuous-validation/status")
    async def validation_status(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            return JSONResponse(service.status(request.query_params.get("run_id")))
        except UnsafeRequest as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/continuous-validation/records")
    async def validation_records(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            return JSONResponse({"records": service.harness_records(request.query_params.get("run_id"))})
        except UnsafeRequest as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/continuous-validation/config/validate")
    async def config_validate(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            body = exact_object(await request.json(), {"validation_sample_every"})
        except UnsafeRequest:
            return JSONResponse({"valid": False}, status_code=400)
        return JSONResponse({"valid": service.validate_sample_every(body["validation_sample_every"])})

    @app.post("/v1/continuous-validation/admin/faults")
    async def admin_faults(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            body = exact_object(await request.json(), {"run_id", "scenario", "max_attempts"})
            await service.configure_fault(body["run_id"], body["scenario"], body["max_attempts"])
            return JSONResponse({"accepted": True})
        except UnsafeRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError:
            raise HTTPException(status_code=409, detail="fault unavailable")

    @app.post("/v1/continuous-validation/admin/consumer")
    async def admin_consumer(request: Request) -> JSONResponse:
        require_admin(request)
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) not in ({"run_id", "state"}, {"run_id", "state", "capacity"}):
                raise UnsafeRequest("invalid consumer request")
            await service.set_consumer_state(body["run_id"], body["state"], body.get("capacity"))
            return JSONResponse({"accepted": True})
        except UnsafeRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError:
            raise HTTPException(status_code=409, detail="consumer unavailable")

    @app.get("/v1/continuous-validation/admin/redacted-events")
    async def redacted_events(request: Request) -> JSONResponse:
        require_admin(request)
        return JSONResponse({"events": list(service._events)})

    @app.post("/internal/v1/guardrail-action")
    async def guardrail_action(request: Request) -> JSONResponse:
        authorization = request.headers.get("authorization")
        expected_authorization = f"Bearer {service.config.broker_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected_authorization):
            raise HTTPException(status_code=401, detail="unauthorized")
        if service.broker is None:
            raise HTTPException(status_code=503, detail="guardrail action broker is not configured")
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise UnsafeRequest("guardrail-action body must be a JSON object")
            return JSONResponse(await service.broker.resolve(body))
        except UnsafeRequest as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
