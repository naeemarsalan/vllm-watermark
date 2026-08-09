#!/usr/bin/env python3
"""Content-redacting D10 continuous-validation acceptance harness.

This program drives the gateway contract documented in deploy/phase5/README.md.
It intentionally does not write request bodies, generated text, HTTP response
bodies, Authorization values, or configured marker values.  Generated text is
held only long enough to recompute its SHA-256 digest.

The command is an acceptance harness, not evidence that an endpoint has run.
Record command lines and the redacted JSON report in EXPERIMENTS.md only after
an actual RHOAI execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


DEFAULT_REQUIRED_METRICS = (
    "validation_responses_total",
    "validation_selected_total",
    "validation_terminal_total",
    "validation_attempts_total",
    "validation_queue_depth",
    "validation_generation_completion_seconds_count",
    "validation_client_delivery_seconds_count",
    "validation_latency_seconds_count",
    "validation_lag_seconds_count",
    "validation_positive_flag_deliveries_total",
)
DEFAULT_LABELS = frozenset({"scheme", "verdict", "mode", "outcome", "reason", "component", "le"})
HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+[-+0-9.eE]+(?:\s+\d+)?$")
LABEL_PAIR = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="(?:\\.|[^"\\])*"\s*(?:,|$)')


class ContractError(RuntimeError):
    """A gateway response violates the D10 contract without exposing its body."""


class CheckFailure(RuntimeError):
    """An acceptance condition was not satisfied."""


@dataclass(frozen=True)
class Case:
    enabled: bool
    scheme: str
    key_id: str

    @property
    def expected_verdict(self) -> bool:
        return self.enabled


@dataclass
class Generated:
    response_id: str
    digest: str
    ordinal: int
    selected: bool
    generation_latency: float
    delivery_latency: float


@dataclass
class RunEvidence:
    generated: list[Generated] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def as_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def as_str(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where} must be a non-empty string")
    return value


def as_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ContractError(f"{where} must be a finite non-negative number")
    return float(value)


def percentile(values: Iterable[float], q: float) -> float:
    """Linear-interpolated percentile, defined for p50/p95/p99 evidence."""
    ordered = sorted(values)
    require(bool(ordered), "cannot calculate a percentile for no samples")
    require(0 <= q <= 100, "percentile is outside 0..100")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    materialized = list(values)
    return {
        "count": len(materialized),
        "p50_seconds": percentile(materialized, 50),
        "p95_seconds": percentile(materialized, 95),
        "p99_seconds": percentile(materialized, 99),
    }


def case_matrix(key_id: str) -> list[Case]:
    return (
        [Case(True, "kgw", key_id) for _ in range(5)]
        + [Case(True, "synthid", key_id) for _ in range(5)]
        + [Case(False, "kgw", key_id) for _ in range(5)]
        + [Case(False, "synthid", key_id) for _ in range(5)]
    )


class Gateway:
    """Small JSON client that never includes body data in raised errors."""

    def __init__(self, args: argparse.Namespace):
        self.base_url = args.gateway_url.rstrip("/")
        self.timeout = args.timeout_seconds
        self.paths = {
            "generate": args.generate_path,
            "reset": args.reset_path,
            "status": args.status_path,
            "records": args.records_path,
            "config_validate": args.config_validate_path,
            "fault": args.fault_path,
            "consumer": args.consumer_path,
            "metrics": args.metrics_path,
            "logs": args.logs_path,
        }
        self.headers = {"Accept": "application/json"}
        if args.auth_token_env:
            token = os.environ.get(args.auth_token_env)
            if not token:
                raise CheckFailure(f"authentication environment variable {args.auth_token_env!r} is empty")
            self.headers["Authorization"] = f"Bearer {token}"
        self.admin_headers = dict(self.headers)
        if args.admin_token_env:
            admin_token = os.environ.get(args.admin_token_env)
            if not admin_token:
                raise CheckFailure(f"authentication environment variable {args.admin_token_env!r} is empty")
            self.admin_headers["Authorization"] = f"Bearer {admin_token}"

    def _url(self, name: str, query: Mapping[str, str] | None = None) -> str:
        path = self.paths[name]
        if not path.startswith("/"):
            raise CheckFailure(f"--{name.replace('_', '-')} must start with '/'")
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def request(
        self,
        name: str,
        body: Mapping[str, Any] | None = None,
        *,
        query: Mapping[str, str] | None = None,
        expected: tuple[int, ...] = (200,),
        raw: bool = False,
        return_status: bool = False,
        discard_body: bool = False,
        admin: bool = False,
        timeout: float | None = None,
    ) -> Any:
        encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers = dict(self.admin_headers if admin else self.headers)
        if encoded is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._url(name, query), data=encoded, headers=headers, method="POST" if body is not None else "GET")
        effective_timeout = self.timeout if timeout is None else min(self.timeout, timeout)
        if not math.isfinite(effective_timeout) or effective_timeout <= 0:
            raise CheckFailure(f"{name} request timeout is exhausted")
        try:
            with urllib.request.urlopen(request, timeout=effective_timeout) as response:
                status = response.status
                payload = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            payload = exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise CheckFailure(f"{name} transport failure ({type(exc).__name__})") from exc
        if status not in expected:
            # Bodies often echo invalid requests. Never surface them.
            raise CheckFailure(f"{name} returned HTTP {status}, expected {expected}")
        if discard_body and status != 200:
            # The transport must consume the response to release the socket,
            # but no error response content is parsed or retained by the
            # harness.
            result: Any = None
        elif raw:
            result = payload.decode("utf-8", errors="replace")
        else:
            try:
                result = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(f"{name} did not return JSON") from exc
        return (status, result) if return_status else result

    def reset(self, sample_every: int, run_id: str) -> None:
        result = as_dict(self.request("reset", {"validation_sample_every": sample_every, "run_id": run_id}, admin=True), "reset response")
        require(result.get("accepted") is True, "reset response did not acknowledge configuration")

    def status(self, run_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        return as_dict(self.request("status", query={"run_id": run_id}, admin=True, timeout=timeout), "status response")

    def records(self, run_id: str, *, timeout: float | None = None) -> list[dict[str, Any]]:
        response = as_dict(self.request("records", query={"run_id": run_id}, admin=True, timeout=timeout), "records response")
        records = response.get("records")
        if not isinstance(records, list):
            raise ContractError("records.records must be a list")
        return [as_dict(item, "record") for item in records]

    def generate(
        self,
        case: Case,
        request_id: str,
        *,
        expected_error_statuses: tuple[int, ...] = (),
        timeout: float | None = None,
    ) -> Generated | None:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
            "vllm_xargs": {
                "watermark": "on" if case.enabled else "off",
                "watermark_scheme": case.scheme,
                "watermark_key_id": case.key_id,
            },
        }
        expected = (200, *expected_error_statuses)
        status, payload = self.request(
            "generate",
            body,
            expected=expected,
            return_status=True,
            discard_body=bool(expected_error_statuses),
            timeout=timeout,
        )
        if status != 200:
            # Fail-closed positive-policy and validation failures are expected
            # in D10 runs.  Correlation comes from the run-scoped records
            # endpoint; the error body is intentionally never parsed.
            return None
        response = as_dict(payload, "generation response")
        validation = as_dict(response.get("watermark_validation"), "generation response.watermark_validation")
        text = self._text(response)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        reported_digest = as_str(validation.get("content_digest"), "generation content_digest")
        require(reported_digest == digest, "generation content digest did not match transient response text")
        response_id = as_str(validation.get("response_id"), "generation response_id")
        ordinal = validation.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ContractError("generation ordinal must be a positive integer")
        selected = validation.get("selected")
        if not isinstance(selected, bool):
            raise ContractError("generation selected must be a boolean")
        return Generated(
            response_id=response_id,
            digest=digest,
            ordinal=ordinal,
            selected=selected,
            generation_latency=as_number(validation.get("generation_completion_latency_seconds"), "generation completion latency"),
            delivery_latency=as_number(validation.get("client_delivery_latency_seconds"), "client delivery latency"),
        )

    @staticmethod
    def _text(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ContractError("generation choices[0] is missing")
        choice = choices[0]
        text = choice.get("text")
        if isinstance(text, str):
            return text
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        raise ContractError("generation choice has no text or message.content")


def configure_generation(gateway: Gateway, args: argparse.Namespace) -> None:
    gateway.model = args.model
    gateway.prompt = args.prompt
    gateway.max_tokens = args.max_tokens
    gateway.temperature = args.temperature


def counter(status: Mapping[str, Any], name: str) -> int:
    counters = as_dict(status.get("counters"), "status.counters")
    value = counters.get(name)
    if not isinstance(value, int) or value < 0:
        raise ContractError(f"status.counters.{name} must be a non-negative integer")
    return value


def require_counters(status: Mapping[str, Any], expected: Mapping[str, int]) -> None:
    for name, value in expected.items():
        actual = counter(status, name)
        require(actual == value, f"counter {name} was {actual}, expected {value}")


def normalise_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("validation_id", "response_id", "content_digest", "scheme", "key_id", "verdict", "mode", "attempts", "timing", "detector_call_id", "guardrails_action_id", "managed_action", "guardrails_action", "delivery_outcome")
    for key in required:
        if key not in record:
            raise ContractError(f"validation record lacks {key}")
    require(not {"content", "text", "prompt", "messages", "request_body"}.intersection(record), "validation record contains plaintext fields")
    result = dict(record)
    validation_id = as_str(result["validation_id"], "record.validation_id")
    try:
        parsed_validation_id = uuid.UUID(validation_id)
    except (AttributeError, ValueError) as exc:
        raise ContractError("record.validation_id must be a canonical lowercase UUID") from exc
    if str(parsed_validation_id) != validation_id:
        raise ContractError("record.validation_id must be a canonical lowercase UUID")
    result["validation_id"] = validation_id
    result["response_id"] = as_str(result["response_id"], "record.response_id")
    result["content_digest"] = as_str(result["content_digest"], "record.content_digest")
    if not HEX_DIGEST.fullmatch(result["content_digest"]):
        raise ContractError("record.content_digest must be lower-case SHA-256")
    result["scheme"] = as_str(result["scheme"], "record.scheme")
    result["key_id"] = as_str(result["key_id"], "record.key_id")
    if not isinstance(result["verdict"], bool):
        raise ContractError("record.verdict must be boolean")
    result["mode"] = as_str(result["mode"], "record.mode")
    if not isinstance(result["attempts"], int) or result["attempts"] < 1:
        raise ContractError("record.attempts must be a positive integer")
    timing = as_dict(result["timing"], "record.timing")
    for key in ("validation_latency_seconds", "validation_lag_seconds"):
        as_number(timing.get(key), f"record.timing.{key}")
    result["detector_call_id"] = as_str(result["detector_call_id"], "record.detector_call_id")
    result["guardrails_action_id"] = as_str(result["guardrails_action_id"], "record.guardrails_action_id")
    result["managed_action"] = as_str(result["managed_action"], "record.managed_action")
    if result["managed_action"] not in {"blocked", "success"}:
        raise ContractError("record.managed_action must be blocked or success")
    result["guardrails_action"] = as_str(result["guardrails_action"], "record.guardrails_action")
    result["delivery_outcome"] = as_str(result["delivery_outcome"], "record.delivery_outcome")
    if result["delivery_outcome"] != "delivered":
        raise ContractError("successful fixed-run record.delivery_outcome must be delivered")
    return result


def reconcile_run(
    evidence: RunEvidence,
    cases: list[Case],
    *,
    n: int,
    total: int,
    positive_action: str,
    clean_action: str,
    expected_mode: str,
) -> dict[str, Any]:
    require(len(evidence.generated) == total, "generation result count did not match submission count")
    expected_ordinals = list(range(n, total + 1, n))
    selected = [item for item in evidence.generated if item.selected]
    require([item.ordinal for item in evidence.generated] == list(range(1, total + 1)), "gateway ordinals were not reset and sequential")
    require([item.ordinal for item in selected] == expected_ordinals, "gateway selected the wrong ordinals")
    require(len(selected) == len(cases), "selected response count did not match case matrix")
    records = [normalise_record(record) for record in evidence.records]
    require(len(records) == len(cases), "terminal record count did not match selected cases")
    selected_by_id = {item.response_id: item for item in selected}
    require(len(selected_by_id) == len(selected), "selected response IDs were not unique")
    records_by_id = {record["response_id"]: record for record in records}
    require(len(records_by_id) == len(records), "terminal records contain duplicate response IDs")
    require(set(records_by_id) == set(selected_by_id), "terminal records do not exactly match selected responses")
    detector_ids: set[str] = set()
    guardrail_ids: set[str] = set()
    for selected_item, case in zip(selected, cases, strict=True):
        record = records_by_id[selected_item.response_id]
        require(
            record["validation_id"] == record["detector_call_id"] == record["guardrails_action_id"],
            "validation, detector, and guardrails IDs were not exactly correlated",
        )
        require(record["content_digest"] == selected_item.digest, "record digest did not match transient response digest")
        require(record["scheme"] == case.scheme and record["key_id"] == case.key_id, "record scheme/key ID did not match the pre-registered case")
        require(record["mode"] == expected_mode, "record delivery mode did not match the configured mode")
        require(record["verdict"] is case.expected_verdict, "detector verdict did not match the pre-registered case")
        require(
            record["managed_action"] == ("blocked" if case.expected_verdict else "success"),
            "managed action did not match the detector verdict",
        )
        require(record["guardrails_action"] == (positive_action if case.enabled else clean_action), "guardrails action did not match detector verdict")
        require(
            record["delivery_outcome"] == "delivered",
            "gateway delivery outcome did not match the configured positive flag policy",
        )
        detector_ids.add(record["detector_call_id"])
        guardrail_ids.add(record["guardrails_action_id"])
    require(len(detector_ids) == len(cases), "detector call IDs were not unique")
    require(len(guardrail_ids) == len(cases), "guardrails action IDs were not unique")
    expected_counts = {
        "started": total,
        "completed": total,
        "selected": len(cases),
        "unsampled": total - len(cases),
        "terminal": len(cases),
        "watermarked": 10,
        "clean": 10,
        "errors": 0,
        "failed": 0,
        "cancelled": 0,
        "detector_attempts": len(cases),
        "guardrails_attempts": len(cases),
        "retries": 0,
        "queue_overflow": 0,
        "dropped": 0,
    }
    require_counters(evidence.status, expected_counts)
    queue = as_dict(evidence.status.get("queue"), "status.queue")
    require(queue.get("depth") == 0, "queue depth was not zero at run completion")
    latency_samples = as_dict(evidence.status.get("latency_samples"), "status.latency_samples")
    expected_samples = {
        "generation_completion": total,
        "client_delivery": total,
        "validation": len(cases),
        "validation_lag": len(cases),
    }
    for name, expected in expected_samples.items():
        actual = latency_samples.get(name)
        require(actual == expected, f"latency sample count {name} was {actual}, expected {expected}")
    return {
        "responses": total,
        "selected": len(cases),
        "terminal": len(records),
        "counters": {name: counter(evidence.status, name) for name in expected_counts},
        "queue_depth": queue.get("depth"),
        "latency_samples": {name: latency_samples[name] for name in expected_samples},
        "record_evidence": [
            {
                "validation_id": record["validation_id"],
                "response_id": record["response_id"],
                "content_digest": record["content_digest"],
                "scheme": record["scheme"],
                "key_id": record["key_id"],
                "verdict": record["verdict"],
                "mode": record["mode"],
                "attempts": record["attempts"],
                "timing": record["timing"],
                "detector_call_id": record["detector_call_id"],
                "guardrails_action_id": record["guardrails_action_id"],
                "managed_action": record["managed_action"],
                "guardrails_action": record["guardrails_action"],
                "delivery_outcome": record["delivery_outcome"],
                "ids_correlated": True,
            }
            for record in records
        ],
        "generation_completion_latency": latency_summary(item.generation_latency for item in evidence.generated),
        "client_delivery_latency": latency_summary(item.delivery_latency for item in evidence.generated),
        "validation_latency": latency_summary(as_number(record["timing"]["validation_latency_seconds"], "record validation latency") for record in records),
        "validation_lag": latency_summary(as_number(record["timing"]["validation_lag_seconds"], "record validation lag") for record in records),
    }


def fixed_run(gateway: Gateway, args: argparse.Namespace, *, n: int, total: int, label: str) -> dict[str, Any]:
    safe_label = re.sub(r"[^A-Za-z0-9-]+", "-", label).strip("-")
    run_id = f"d10-{safe_label}-{uuid.uuid4()}"
    cases = case_matrix(args.key_id)
    gateway.reset(n, run_id)
    evidence = RunEvidence()
    selected_cases = iter(cases)
    fallback = Case(False, "kgw", args.key_id)
    for ordinal in range(1, total + 1):
        case = next(selected_cases) if ordinal % n == 0 else fallback
        evidence.generated.append(gateway.generate(case, f"{run_id}-{ordinal}"))
    evidence.status = gateway.status(run_id)
    evidence.records = gateway.records(run_id)
    return reconcile_run(
        evidence,
        cases,
        n=n,
        total=total,
        positive_action=args.positive_action,
        clean_action=args.clean_action,
        expected_mode=args.expected_mode,
    )


def unsampled_baseline(gateway: Gateway, args: argparse.Namespace) -> dict[str, Any]:
    """Measure four pre-selection requests with ``N=5`` and no validation.

    Using ordinals 1 through 4 after a reset gives a gateway-path latency
    baseline without inventing a disabled sampler value outside the supported
    positive-integer contract.  Watermark generation is also disabled so this
    run measures the ordinary unsampled delivery path.
    """

    total = 4
    run_id = f"d10-unsampled-baseline-{uuid.uuid4()}"
    gateway.reset(5, run_id)
    case = Case(False, "kgw", args.key_id)
    generated: list[Generated] = []
    for ordinal in range(1, total + 1):
        item = gateway.generate(case, f"{run_id}-{ordinal}")
        require(item is not None, "unsampled baseline request was not delivered")
        generated.append(item)

    require([item.ordinal for item in generated] == list(range(1, total + 1)), "unsampled baseline ordinals were not reset and sequential")
    require(all(not item.selected for item in generated), "unsampled baseline unexpectedly selected a response")
    status = gateway.status(run_id)
    require_counters(status, {
        "started": total,
        "completed": total,
        "selected": 0,
        "unsampled": total,
        "terminal": 0,
        "watermarked": 0,
        "clean": 0,
        "errors": 0,
        "failed": 0,
        "cancelled": 0,
        "detector_attempts": 0,
        "guardrails_attempts": 0,
        "retries": 0,
        "queue_overflow": 0,
        "dropped": 0,
    })
    require(gateway.records(run_id) == [], "unsampled baseline created validation records")
    require(as_dict(status.get("queue"), "unsampled baseline queue").get("depth") == 0, "unsampled baseline queue did not drain")
    latency_samples = as_dict(status.get("latency_samples"), "unsampled baseline latency_samples")
    require(latency_samples.get("generation_completion") == total, "unsampled baseline generation latency sample count did not match")
    require(latency_samples.get("client_delivery") == total, "unsampled baseline delivery latency sample count did not match")
    require(latency_samples.get("validation") == 0, "unsampled baseline invented validation latency samples")
    require(latency_samples.get("validation_lag") == 0, "unsampled baseline invented validation lag samples")
    return {
        "responses": total,
        "sample_every": 5,
        "selected": 0,
        "counters": {
            name: counter(status, name)
            for name in (
                "started", "completed", "selected", "unsampled", "terminal",
                "watermarked", "clean", "errors", "failed", "cancelled",
                "detector_attempts", "guardrails_attempts", "retries",
                "queue_overflow", "dropped",
            )
        },
        "queue_depth": 0,
        "latency_samples": {
            "generation_completion": latency_samples["generation_completion"],
            "client_delivery": latency_samples["client_delivery"],
            "validation": latency_samples["validation"],
            "validation_lag": latency_samples["validation_lag"],
        },
        "generation_completion_latency": latency_summary(item.generation_latency for item in generated),
        "client_delivery_latency": latency_summary(item.delivery_latency for item in generated),
    }


def validate_config(gateway: Gateway) -> dict[str, Any]:
    results: dict[str, str] = {}
    cases: list[tuple[str, Any, bool]] = [("1", 1, True), ("5", 5, True), ("0", 0, False), ("negative", -1, False), ("fraction", 1.5, False), ("empty", "", False), ("nonnumeric", "five", False)]
    for name, value, valid in cases:
        result = gateway.request("config_validate", {"validation_sample_every": value}, expected=(200, 400, 422), admin=True)
        result_object = as_dict(result, "config validation response")
        accepted = result_object.get("valid") is True
        require(accepted is valid, f"configuration validation case {name} was not {'accepted' if valid else 'rejected'}")
        results[name] = "accepted" if accepted else "rejected"
    return results


def fault_run(gateway: Gateway, args: argparse.Namespace, name: str, expected_attempts: int, expected_retries: int, expected_error: str) -> dict[str, Any]:
    safe_name = re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-")
    run_id = f"d10-fault-{safe_name}-{uuid.uuid4()}"
    gateway.reset(1, run_id)
    response = gateway.request("fault", {"run_id": run_id, "scenario": name, "max_attempts": 3}, admin=True)
    require(as_dict(response, "fault response").get("accepted") is True, f"fault {name} was not accepted")
    # A successful validation is still an HTTP 403 when the configured
    # positive policy is ``block``.  Exhausted/malformed validation is an
    # HTTP 503 under fail-closed policy.  In both cases the body is discarded
    # and the run-scoped records endpoint is the only correlation source.
    generated = gateway.generate(
        Case(True, "kgw", args.key_id),
        f"{run_id}-1",
        expected_error_statuses=(403, 503),
    )
    status = gateway.status(run_id)
    records = [normalise_fault_record(record) for record in gateway.records(run_id)]
    require(len(records) == 1, f"fault {name} did not yield one unique terminal record")
    # A 200 response carries transient response metadata; a fail-closed
    # non-2xx response deliberately does not.  Never correlate by parsing an
    # error body: the run ID plus the single terminal record is sufficient.
    if generated is not None:
        require(records[0]["response_id"] == generated.response_id, f"fault {name} response ID did not match its terminal record")
    record = records[0]
    require(record["attempts"] == expected_attempts, f"fault {name} attempts did not match")
    require(counter(status, "retries") == expected_retries, f"fault {name} retries did not match")
    require(counter(status, "terminal") == 1 and counter(status, "selected") == 1, f"fault {name} did not preserve selected/terminal uniqueness")
    if expected_error == "success":
        require(record.get("terminal_state") == "success", f"fault {name} did not terminate successfully")
    else:
        require(record.get("terminal_state") == expected_error, f"fault {name} terminal state did not match")
        outcome = as_str(record.get("delivery_outcome"), "fault delivery outcome")
        if args.expected_mode == "asynchronous":
            allowed = {"fail_open"}
        elif args.expected_failure_policy == "closed":
            allowed = {"fail_closed"}
        else:
            allowed = {"fail_open"}
        require(record.get("mode") == args.expected_mode, f"fault {name} delivery mode did not match configuration")
        require(outcome in allowed, f"fault {name} had an invalid failure-policy outcome")
    return {"attempts": expected_attempts, "retries": expected_retries, "terminal_state": record.get("terminal_state")}


def normalise_fault_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Fault errors legitimately have no detector verdict or guardrails call."""
    result = dict(as_dict(record, "fault record"))
    require(not {"content", "text", "prompt", "messages", "request_body"}.intersection(result), "fault record contains plaintext fields")
    terminal_state = as_str(result.get("terminal_state"), "fault record.terminal_state")
    if terminal_state == "success":
        return normalise_record(result)
    result["response_id"] = as_str(result.get("response_id"), "fault record.response_id")
    digest = as_str(result.get("content_digest"), "fault record.content_digest")
    if not HEX_DIGEST.fullmatch(digest):
        raise ContractError("fault record.content_digest must be lower-case SHA-256")
    attempts = result.get("attempts")
    if not isinstance(attempts, int) or attempts < 1:
        raise ContractError("fault record.attempts must be a positive integer")
    return result


def queue_run(gateway: Gateway, args: argparse.Namespace) -> dict[str, Any]:
    run_id = f"d10-queue-{uuid.uuid4()}"
    gateway.reset(1, run_id)
    submitted: list[Generated | None] = []
    submit_errors: list[BaseException] = []
    third: list[Generated | None] = []
    third_error: list[BaseException] = []
    first_threads: list[threading.Thread] = []
    third_thread: threading.Thread | None = None
    consumer_paused = False
    # This is the sole queue assertion deadline.  It covers queue filling, the
    # completed-counter synchronization, the short policy grace, and all
    # normal-path post-resume joins.  A separate bounded cleanup grace below
    # exists only to restore the consumer and reap workers after a failure.
    deadline = time.monotonic() + args.timeout_seconds
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    result: dict[str, Any] | None = None

    def remaining(until: float) -> float:
        return max(0.001, until - time.monotonic())

    def resume_consumer(until: float) -> None:
        nonlocal consumer_paused
        if not consumer_paused:
            return
        response = as_dict(
            gateway.request(
                "consumer",
                {"run_id": run_id, "state": "running"},
                admin=True,
                timeout=remaining(until),
            ),
            "consumer resume response",
        )
        require(response.get("accepted") is True, "consumer did not resume")
        consumer_paused = False

    def join_until(thread: threading.Thread, until: float) -> None:
        if not thread.is_alive():
            return
        try:
            thread.join(max(0.0, until - time.monotonic()))
        except BaseException as exc:
            # Cleanup failures are retained separately so that they cannot
            # replace a queue assertion or transport exception from the body.
            cleanup_errors.append(exc)

    def submit_first(ordinal: int) -> None:
        try:
            submitted.append(
                gateway.generate(
                    Case(True, "kgw", args.key_id),
                    f"{run_id}-{ordinal}",
                    expected_error_statuses=(403, 503),
                    timeout=remaining(deadline),
                )
            )
        except BaseException as exc:  # Returned below without exposing HTTP bodies.
            submit_errors.append(exc)

    def submit_third() -> None:
        try:
            third.append(
                gateway.generate(
                    Case(True, "kgw", args.key_id),
                    f"{run_id}-3",
                    expected_error_statuses=(403, 503),
                    timeout=remaining(deadline),
                )
            )
        except BaseException as exc:  # Returned below without exposing HTTP bodies.
            third_error.append(exc)

    try:
        # A failed or timed-out pause request is ambiguous: the server may
        # have applied the mutation before the client observed the failure.
        # Mark it potentially paused before issuing the request so every exit
        # path sends an idempotent resume.
        consumer_paused = True
        paused_raw = gateway.request(
            "consumer",
            {"run_id": run_id, "state": "paused", "capacity": 2},
            admin=True,
            timeout=remaining(deadline),
        )
        paused_response = as_dict(paused_raw, "consumer pause response")
        require(paused_response.get("accepted") is True, "consumer did not pause")

        # Pausing the consumer intentionally leaves selected requests waiting
        # for a worker.  Submit the first two concurrently; sequential
        # submissions deadlock because the first request cannot return while
        # the consumer is paused.
        first_threads = [threading.Thread(target=submit_first, args=(i,), daemon=True) for i in (1, 2)]
        for first_thread in first_threads:
            first_thread.start()

        paused: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            paused = gateway.status(run_id, timeout=remaining(deadline))
            queue = as_dict(paused.get("queue"), "paused queue")
            if queue.get("depth") == 2:
                break
            if all(not first_thread.is_alive() for first_thread in first_threads):
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        require(paused is not None, "queue status was not returned while paused")
        queue = as_dict(paused.get("queue"), "paused queue")
        require(queue.get("depth") == 2, "two paused requests did not occupy the queue")
        require(
            not submit_errors,
            f"paused queue submission failed ({type(submit_errors[0]).__name__})"
            if submit_errors
            else "paused queue submission failed",
        )
        policy = as_str(queue.get("overflow_policy"), "queue overflow policy")
        require(queue.get("capacity") == 2, "queue capacity was not two")
        require(queue.get("peak_depth") == 2, "queue peak depth was not two")

        # Use the counter value observed immediately before the third request
        # as a baseline and accept any monotonic overshoot (>=), since a
        # status sample can include more than one completion.
        generation_completed_before = counter(paused, "completed")
        third_thread = threading.Thread(target=submit_third, daemon=True)
        third_thread.start()
        third_completed = False
        while time.monotonic() < deadline:
            paused = gateway.status(run_id, timeout=remaining(deadline))
            if counter(paused, "completed") >= generation_completed_before + 1:
                third_completed = True
                break
            if not third_thread.is_alive():
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        require(
            third_completed or not third_thread.is_alive(),
            "third generation did not complete before the queue check deadline",
        )

        # This is only the short, policy-specific scheduling/return grace once
        # generation has completed.  It consumes the same absolute deadline.
        if third_thread.is_alive():
            third_thread.join(min(args.queue_pending_check_seconds, max(0.0, deadline - time.monotonic())))
        paused = gateway.status(run_id, timeout=remaining(deadline))
        if policy == "non_blocking":
            require(not third_thread.is_alive(), "non-blocking queue left the overflow submission pending")
            require(counter(paused, "queue_overflow") == 1, "non-blocking queue did not record exactly one overflow")
        elif policy == "blocking":
            require(third_thread.is_alive(), "blocking queue did not keep the third submission pending")
            require(
                counter(paused, "queue_overflow") == 0 and counter(paused, "dropped") == 0,
                "blocking queue recorded an overflow or drop",
            )
        else:
            raise ContractError("queue overflow_policy must be non_blocking or blocking")

        resume_consumer(deadline)
        for first_thread in first_threads:
            join_until(first_thread, deadline)
            require(not first_thread.is_alive(), "paused queue submission did not finish after consumer resume")
        join_until(third_thread, deadline)
        require(not third_thread.is_alive(), "third queue submission did not finish after consumer resume")

        if submit_errors or third_error:
            error = submit_errors[0] if submit_errors else third_error[0]
            raise CheckFailure(f"queue submission failed ({type(error).__name__})")
        require(len(third) == 1, "third queue submission did not return exactly once")
        submitted.extend(third)
        require(len(submitted) == 3, "queue run did not return exactly three submissions")
        final = gateway.status(run_id, timeout=remaining(deadline))
        raw_records = gateway.records(run_id, timeout=remaining(deadline))
        records = [normalise_queue_record(record) for record in raw_records]
        response_ids = [item.response_id for item in submitted if item is not None]
        require(len(response_ids) == len(set(response_ids)), "queue run generation response IDs were not unique")
        require(len({record["response_id"] for record in records}) == 3, "queue run lacks three unique terminal records")
        require(as_dict(final.get("queue"), "final queue").get("depth") == 0, "queue did not drain")
        if policy == "non_blocking":
            require(counter(final, "queue_overflow") == 1, "non-blocking queue final overflow count was not one")
            verdicts = sum(record.get("terminal_state") == "success" for record in records)
            require(verdicts == 2, "non-blocking queue did not validate exactly two accepted items")
            overflow = [record for record in records if record.get("terminal_state") != "success"]
            require(len(overflow) == 1, "non-blocking queue did not produce one explicit terminal overflow record")
            require("verdict" not in overflow[0] and "detector_call_id" not in overflow[0], "overflow record invented a detector verdict or call")
            expected_overflow_delivery = "fail_closed" if args.expected_failure_policy == "closed" else "fail_open"
            require(
                overflow[0].get("delivery_outcome") == expected_overflow_delivery,
                "queue overflow did not honor the configured failure policy",
            )
        else:
            require(all(record.get("terminal_state") == "success" for record in records), "blocking queue did not validate all three items")
        result = {
            "overflow_policy": policy,
            "terminal_records": len(records),
            "peak_depth": as_dict(final.get("queue"), "final queue").get("peak_depth"),
            "queue_depth": as_dict(final.get("queue"), "final queue").get("depth"),
            "queue_overflow": counter(final, "queue_overflow"),
            "validated_records": sum(record.get("terminal_state") == "success" for record in records),
        }
    except BaseException as exc:
        # Delay raising until after cleanup, while retaining the original
        # exception so cleanup failures never mask the actual queue failure.
        primary_error = exc
    finally:
        # The operation timeout must not leave a paused consumer or daemon
        # submissions behind.  Cleanup gets one explicit, bounded grace
        # period rather than restarting the full operation timeout per join.
        cleanup_deadline = time.monotonic() + min(5.0, max(0.1, args.timeout_seconds))
        if consumer_paused:
            try:
                resume_consumer(cleanup_deadline)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for first_thread in first_threads:
            join_until(first_thread, cleanup_deadline)
        if third_thread is not None:
            join_until(third_thread, cleanup_deadline)
        if any(first_thread.is_alive() for first_thread in first_threads) or (
            third_thread is not None and third_thread.is_alive()
        ):
            cleanup_errors.append(CheckFailure("queue worker cleanup incomplete"))

    if primary_error is not None:
        if cleanup_errors:
            raise CheckFailure("queue run failed and cleanup did not complete") from primary_error
        raise primary_error
    if cleanup_errors:
        # Do not include transport/HTTP exception text: the harness report is
        # intentionally content-free even when cleanup fails.
        raise CheckFailure("queue cleanup failed") from cleanup_errors[0]
    require(result is not None, "queue run produced no result")
    return result


def normalise_queue_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a queue terminal record without inventing a dropped verdict."""
    result = dict(as_dict(record, "queue record"))
    require(not {"content", "text", "prompt", "messages", "request_body"}.intersection(result), "queue record contains plaintext fields")
    result["response_id"] = as_str(result.get("response_id"), "queue record.response_id")
    result["terminal_state"] = as_str(result.get("terminal_state"), "queue record.terminal_state")
    if result["terminal_state"] == "success":
        return normalise_record(result)
    digest = as_str(result.get("content_digest"), "queue overflow content_digest")
    if not HEX_DIGEST.fullmatch(digest):
        raise ContractError("queue overflow content_digest must be lower-case SHA-256")
    attempts = result.get("attempts")
    if not isinstance(attempts, int) or attempts != 0:
        raise ContractError("unaccepted queue overflow record must have zero attempts")
    return result


def scan_observability(gateway: Gateway, args: argparse.Namespace) -> dict[str, Any]:
    markers = list(args.forbidden_marker)
    for env_name in args.secret_marker_env:
        value = os.environ.get(env_name)
        if not value:
            raise CheckFailure(f"secret marker environment variable {env_name!r} is empty")
        markers.append(value)
    require(markers, "at least one --forbidden-marker or --secret-marker-env is required")
    sources = {"metrics": gateway.request("metrics", raw=True), "logs": gateway.request("logs", raw=True, admin=True)}
    for source, content in sources.items():
        for marker in markers:
            require(marker not in content, f"forbidden marker was found in {source}")
    metric_names: set[str] = set()
    unexpected_labels: set[str] = set()
    for line in sources["metrics"].splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.match(line)
        if not match:
            continue
        metric_names.add(match.group(1))
        labels = match.group(2) or ""
        offset = 0
        while offset < len(labels):
            label = LABEL_PAIR.match(labels, offset)
            if not label:
                raise ContractError("metrics exposition contains an unparsable label set")
            if label.group(1) not in set(args.allowed_metric_label):
                unexpected_labels.add(label.group(1))
            offset = label.end()
    missing = sorted(set(args.required_metric) - metric_names)
    require(not missing, "metrics exposition lacks required series")
    require(not unexpected_labels, "metrics exposition uses labels outside the bounded allowlist")
    return {"required_metric_count": len(args.required_metric), "marker_count": len(markers)}


def synthetic_run_evidence(key_id: str, *, n: int, total: int) -> tuple[RunEvidence, list[Case]]:
    """Build hash-only contract fixtures for the local self-test.

    This exercises the exact acceptance cardinalities without making network
    calls or fabricating generated text in a report.  The digest is derived
    from an in-memory fixture and only the digest enters the synthetic record.
    """
    cases = case_matrix(key_id)
    evidence = RunEvidence()
    selected_cases = iter(cases)
    fallback = Case(False, "kgw", key_id)
    records: list[dict[str, Any]] = []
    for ordinal in range(1, total + 1):
        case = next(selected_cases) if ordinal % n == 0 else fallback
        text = f"self-test-generated-{ordinal}"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        selected = ordinal % n == 0
        response_id = f"self-{ordinal}"
        evidence.generated.append(Generated(response_id, digest, ordinal, selected, 0.001, 0.002))
        if selected:
            validation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"vllm-watermark-self-test-{ordinal}"))
            records.append({
                "validation_id": validation_id,
                "response_id": response_id,
                "content_digest": digest,
                "scheme": case.scheme,
                "key_id": case.key_id,
                "verdict": case.expected_verdict,
                "mode": "synchronous",
                "attempts": 1,
                "timing": {"validation_latency_seconds": 0.003, "validation_lag_seconds": 0.004},
                "detector_call_id": validation_id,
                "guardrails_action_id": validation_id,
                "managed_action": "blocked" if case.expected_verdict else "success",
                "guardrails_action": "block" if case.expected_verdict else "pass",
                "delivery_outcome": "delivered",
            })
    selected_count = len(cases)
    evidence.records = records
    evidence.status = {
        "counters": {
            "started": total,
            "completed": total,
            "selected": selected_count,
            "unsampled": total - selected_count,
            "terminal": selected_count,
            "watermarked": 10,
            "clean": 10,
            "errors": 0,
            "failed": 0,
            "cancelled": 0,
            "detector_attempts": selected_count,
            "guardrails_attempts": selected_count,
            "retries": 0,
            "queue_overflow": 0,
            "dropped": 0,
        },
        "queue": {"depth": 0},
        "latency_samples": {
            "generation_completion": total,
            "client_delivery": total,
            "validation": selected_count,
            "validation_lag": selected_count,
        },
    }
    return evidence, cases


def self_test() -> int:
    require([case.scheme for case in case_matrix("test-key")[:6]] == ["kgw"] * 5 + ["synthid"], "matrix order")
    require(percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5, "p50 interpolation")
    require(percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.85, "p95 interpolation")
    try:
        normalise_record({})
    except ContractError:
        pass
    else:
        raise AssertionError("record schema rejection")
    n1_evidence, n1_cases = synthetic_run_evidence("test-key", n=1, total=20)
    n1 = reconcile_run(n1_evidence, n1_cases, n=1, total=20, positive_action="block", clean_action="pass", expected_mode="synchronous")
    n5_evidence, n5_cases = synthetic_run_evidence("test-key", n=5, total=100)
    n5 = reconcile_run(n5_evidence, n5_cases, n=5, total=100, positive_action="block", clean_action="pass", expected_mode="synchronous")
    fault_error = normalise_fault_record({
        "response_id": "fault-response",
        "content_digest": "0" * 64,
        "attempts": 3,
        "terminal_state": "retry_exhausted",
    })
    require("content" not in fault_error and "text" not in fault_error, "fault self-test retained plaintext")
    queue_overflow = normalise_queue_record({
        "response_id": "overflow-response",
        "content_digest": "1" * 64,
        "attempts": 0,
        "terminal_state": "queue_overflow",
    })
    require("content" not in queue_overflow and "text" not in queue_overflow, "queue self-test retained plaintext")
    print(json.dumps({
        "self_test": "passed",
        "content_logged": False,
        "n1": {"responses": n1["responses"], "selected": n1["selected"]},
        "n5": {"responses": n5["responses"], "selected": n5["selected"]},
        "fault_cases": 3,
        "queue_overflow_checked": True,
    }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    result.add_argument("--gateway-url", help="gateway origin, e.g. https://gateway.example")
    result.add_argument("--model", help="model value sent to the configured generation endpoint")
    result.add_argument("--key-id", help="non-secret pre-registered watermark key ID")
    result.add_argument("--auth-token-env", default="", help="environment variable containing a bearer token; never printed")
    result.add_argument("--admin-token-env", default="", help="environment variable containing the separate admin bearer token; never printed")
    result.add_argument("--prompt", default="Produce a concise factual paragraph about a neutral technical topic.", help="request content; never printed or written")
    result.add_argument("--max-tokens", type=int, default=256)
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--timeout-seconds", type=float, default=90.0)
    result.add_argument("--queue-pending-check-seconds", type=float, default=0.25)
    result.add_argument("--positive-action", choices=("block", "flag"), default="block")
    result.add_argument("--clean-action", default="pass")
    result.add_argument("--expected-mode", choices=("synchronous", "asynchronous"), default="synchronous")
    result.add_argument("--expected-failure-policy", choices=("open", "closed"), default="closed")
    result.add_argument("--generate-path", default="/v1/chat/completions")
    result.add_argument("--reset-path", default="/v1/continuous-validation/admin/reset")
    result.add_argument("--status-path", default="/v1/continuous-validation/status")
    result.add_argument("--records-path", default="/v1/continuous-validation/records")
    result.add_argument("--config-validate-path", default="/v1/continuous-validation/config/validate")
    result.add_argument("--fault-path", default="/v1/continuous-validation/admin/faults")
    result.add_argument("--consumer-path", default="/v1/continuous-validation/admin/consumer")
    result.add_argument("--metrics-path", default="/metrics")
    result.add_argument("--logs-path", default="/v1/continuous-validation/admin/redacted-events")
    result.add_argument("--forbidden-marker", action="append", default=[])
    result.add_argument("--secret-marker-env", action="append", default=[])
    result.add_argument("--required-metric", action="append", default=list(DEFAULT_REQUIRED_METRICS))
    result.add_argument("--allowed-metric-label", action="append", default=sorted(DEFAULT_LABELS))
    result.add_argument("--self-test", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        return self_test()
    for field_name in ("gateway_url", "model", "key_id"):
        if not getattr(args, field_name):
            parser().error(f"--{field_name.replace('_', '-')} is required unless --self-test is used")
    require(args.max_tokens > 0, "--max-tokens must be positive")
    require(math.isfinite(args.temperature), "--temperature must be finite")
    require(args.timeout_seconds > 0 and math.isfinite(args.timeout_seconds), "--timeout-seconds must be positive and finite")
    require(args.queue_pending_check_seconds > 0 and math.isfinite(args.queue_pending_check_seconds), "--queue-pending-check-seconds must be positive and finite")
    try:
        gateway = Gateway(args)
        configure_generation(gateway, args)
        report = {
            "contract": "phase5-v1",
            "content_logged": False,
            "latency_semantics": {
                "generation_completion": "request_start_to_upstream_completion",
                # The gateway emits this under the retained client_delivery
                # field name, but stops its timer when the response mapping is
                # ready, before FastAPI serialization or socket delivery.
                "client_delivery": "request_start_to_gateway_response_ready",
                "validation": "validation_attempt_window",
                "validation_lag": "validation_queue_wait_to_attempt_start",
            },
            "policy_semantics": {
                "mode": args.expected_mode,
                "validation_failure": args.expected_failure_policy,
                "managed_guardrails_positive_action": args.positive_action,
                "gateway_positive_delivery": "flag",
            },
            "configuration": validate_config(gateway),
            "unsampled_baseline": unsampled_baseline(gateway, args),
            "n1": fixed_run(gateway, args, n=1, total=20, label="n1"),
            "n5": fixed_run(gateway, args, n=5, total=100, label="n5"),
            "faults": {
                "retry_then_success": fault_run(gateway, args, "retry_then_success", 2, 1, "success"),
                "retry_exhausted": fault_run(gateway, args, "retry_exhausted", 3, 2, "retry_exhausted"),
                "malformed_success": fault_run(gateway, args, "malformed_success", 1, 0, "malformed_response"),
            },
            "queue": queue_run(gateway, args),
            "observability": scan_observability(gateway, args),
        }
    except (CheckFailure, ContractError) as exc:
        print(json.dumps({"passed": False, "reason": str(exc), "content_logged": False}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"passed": True, **report}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
