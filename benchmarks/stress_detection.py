#!/usr/bin/env python3
"""Deterministic, content-free stress harness for token-id detectors.

The parent process expands a matrix and launches one fresh worker process per
cell.  A cell therefore has an independent timeout and peak-RSS measurement;
an expensive or broken cell cannot poison the remaining matrix.  Workers emit
only protocol events containing counts and timings.  Token ids, detector
scores, synthetic key values, raw/generated text, environment variables, and
exception messages never cross the worker boundary or appear in the report.

Defaults intentionally form a small local smoke/stress matrix.  Larger runs
are explicit, for example::

    PYTHONPATH=src python3 benchmarks/stress_detection.py \
      --vocab-sizes 128,4096,151936 \
      --lengths 1,2,5,32,256,2048 \
      --synthid-depths 1,8,30 \
      --repeats 10 --timeout-seconds 120 --max-cells 10000

The report is one JSON document on stdout (or at ``--output``).  A non-zero
exit means at least one valid cell timed out, returned a non-finite result, or
otherwise violated its expected contract.  A too-short sequence raising
``ValueError`` is an expected outcome and is counted separately.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = 1
HARNESS_VERSION = "stress-detection-v1"
PATTERNS = (
    "uniform_random",
    "constant",
    "alternating",
    "repeated_blocks",
    "distinct_modulo_vocab",
)
SCHEMES = ("kgw", "synthid")
SYNTHID_SCORERS = ("mean", "weighted_mean")
_WILSON_Z_95 = 1.959963984540054


def _json_dumps(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def _emit_worker_event(event: dict[str, Any]) -> None:
    event.setdefault("protocol_version", PROTOCOL_VERSION)
    print(_json_dumps(event), flush=True)


def _stable_uint(label: str, cell_seed: int, index: int, bits: int) -> int:
    """Derive a public, reproducible synthetic integer; never key material."""
    material = f"{HARNESS_VERSION}:{label}:{cell_seed}:{index}".encode("ascii")
    digest = hashlib.sha256(material).digest()
    value = int.from_bytes(digest[:8], "big")
    return value & ((1 << bits) - 1)


def _make_tokens(
    pattern: str,
    length: int,
    vocab_size: int,
    cell_seed: int,
    attempt: int,
    block_size: int,
) -> list[int]:
    """Build a deterministic synthetic token sequence without returning it."""
    token_seed = _stable_uint("tokens", cell_seed, attempt, 64)
    rng = random.Random(token_seed)

    if pattern == "uniform_random":
        return [rng.randrange(vocab_size) for _ in range(length)]

    if pattern == "constant":
        token = rng.randrange(vocab_size)
        return [token] * length

    if pattern == "alternating":
        first = rng.randrange(vocab_size)
        second = first if vocab_size == 1 else (first + 1 + rng.randrange(vocab_size - 1)) % vocab_size
        return [first if i % 2 == 0 else second for i in range(length)]

    if pattern == "repeated_blocks":
        width = min(max(1, block_size), max(1, length))
        block = [rng.randrange(vocab_size) for _ in range(width)]
        return [block[i % width] for i in range(length)]

    if pattern == "distinct_modulo_vocab":
        offset = rng.randrange(vocab_size)
        return [(offset + i) % vocab_size for i in range(length)]

    raise ValueError(f"unsupported pattern {pattern!r}")


def _validate_result(result: Any, scheme: str, expected_depth: int | None, max_scored: int) -> tuple[bool, int]:
    """Return (is_finite_and_coherent, scored_tokens), without serializing scores."""
    if scheme == "kgw":
        scored = result.num_tokens_scored
        numeric = (result.z_score, result.p_value)
        coherent = (
            isinstance(scored, int)
            and scored > 0
            and scored <= max_scored
            and isinstance(result.num_green, int)
            and 0 <= result.num_green <= scored
            and 0.0 <= result.p_value <= 1.0
            and isinstance(result.prediction, bool)
        )
    else:
        scored = result.num_scored
        numeric = (result.mean_g, result.score, result.z_score, result.p_value)
        coherent = (
            isinstance(scored, int)
            and scored > 0
            and scored <= max_scored
            and result.depth == expected_depth
            and 0.0 <= result.mean_g <= 1.0
            and 0.0 <= result.score <= 1.0
            and 0.0 <= result.p_value <= 1.0
            and isinstance(result.prediction, bool)
        )
    finite = all(math.isfinite(float(value)) for value in numeric)
    return bool(coherent and finite), int(scored) if isinstance(scored, int) else 0


def _worker_peak_rss_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS/BSD report bytes.
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, OSError, ValueError):
        return None


def _protocol_errors(events: list[dict[str, Any]], repeats: int, timed_out: bool) -> int:
    """Count structural/schema violations without trusting worker payloads."""
    known = {
        "worker_started": {"event", "protocol_version"},
        "attempt_started": {"event", "protocol_version", "attempt"},
        "attempt_result": {
            "event", "protocol_version", "attempt", "category", "error_type",
            "input_tokens", "latency_ns", "scored_tokens",
        },
        "worker_summary": {"event", "protocol_version", "peak_rss_bytes", "unexpected_outcomes"},
        "worker_fatal": {"event", "protocol_version", "error_type", "peak_rss_bytes"},
    }
    required = {
        "worker_started": ("protocol_version",),
        "attempt_started": ("protocol_version", "attempt"),
        "attempt_result": ("protocol_version", "attempt", "category", "input_tokens", "latency_ns", "scored_tokens"),
        "worker_summary": ("protocol_version", "peak_rss_bytes", "unexpected_outcomes"),
        "worker_fatal": ("protocol_version", "error_type", "peak_rss_bytes"),
    }
    errors = 0
    for event in events:
        kind = event.get("event")
        if kind not in known:
            errors += 1
            continue
        if event.get("protocol_version") != PROTOCOL_VERSION:
            errors += 1
        if set(event) - known[kind] or any(field not in event for field in required[kind]):
            errors += 1
        if kind == "attempt_started" and (
            not isinstance(event.get("attempt"), int) or isinstance(event.get("attempt"), bool)
        ):
            errors += 1
        if kind == "attempt_result":
            if not isinstance(event.get("attempt"), int) or isinstance(event.get("attempt"), bool):
                errors += 1
            if not isinstance(event.get("category"), str):
                errors += 1
            for field in ("input_tokens", "latency_ns", "scored_tokens"):
                if not isinstance(event.get(field), int) or isinstance(event.get(field), bool) or event[field] < 0:
                    errors += 1
    if events and events[0].get("event") != "worker_started":
        errors += 1
    terminals = {"worker_summary", "worker_fatal"}
    if any(event.get("event") in terminals for event in events[:-1]):
        errors += 1
    starts = [e for e in events if e.get("event") == "attempt_started"]
    results = [e for e in events if e.get("event") == "attempt_result"]
    summaries = [e for e in events if e.get("event") == "worker_summary"]
    fatals = [e for e in events if e.get("event") == "worker_fatal"]
    ids = [e.get("attempt") for e in starts]
    valid_ids = [i for i in ids if type(i) is int]
    if (
        len(valid_ids) != len(ids)
        or len(valid_ids) != len(set(valid_ids))
        or any(i < 0 or i >= repeats for i in valid_ids)
    ):
        errors += 1
    result_ids = [e.get("attempt") for e in results]
    valid_result_ids = [i for i in result_ids if type(i) is int]
    if (
        len(valid_result_ids) != len(result_ids)
        or any(i not in valid_ids for i in valid_result_ids)
        or len(valid_result_ids) != len(set(valid_result_ids))
    ):
        errors += 1
    worker_starts = [e for e in events if e.get("event") == "worker_started"]
    if len(worker_starts) != (0 if timed_out and not events else 1):
        errors += 1
    if not timed_out and sum(bool(x) for x in (summaries, fatals)) != 1:
        errors += 1
    if not timed_out:
        # The worker protocol is intentionally ordered: each attempt starts,
        # then completes, and only then emits its terminal event.
        expected = ["worker_started"]
        for _ in range(repeats):
            expected.extend(("attempt_started", "attempt_result"))
        expected.append("worker_summary" if summaries else "worker_fatal")
        if [event.get("event") for event in events] != expected:
            errors += 1
    return errors


def _worker_main(encoded_spec: str) -> int:
    """Internal worker entry point. Its stdout is a private JSONL protocol."""
    _emit_worker_event({"event": "worker_started", "protocol_version": PROTOCOL_VERSION})
    unexpected = 0
    try:
        spec_bytes = base64.urlsafe_b64decode(encoded_spec.encode("ascii"))
        spec = json.loads(spec_bytes.decode("utf-8"))

        repo_src = Path(__file__).resolve().parent.parent / "src"
        if str(repo_src) not in sys.path:
            sys.path.insert(0, str(repo_src))

        # Keep local timing reproducible and avoid nested CPU oversubscription
        # when a parent runs several harnesses concurrently.
        import torch

        torch.set_num_threads(spec["torch_threads"])
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A fresh worker normally permits this exactly once. If a future
            # torch build initializes the pool earlier, intra-op control still
            # provides the important bound and the environment records it.
            pass

        from vllm_watermark.kgw.core import KGWConfig
        from vllm_watermark.kgw.detector import score_token_ids as kgw_score_token_ids
        from vllm_watermark.synthid.core import SynthIDConfig
        from vllm_watermark.synthid.detector import (
            score_token_ids_mean,
            score_token_ids_weighted_mean,
        )

        scheme = spec["scheme"]
        if scheme == "kgw":
            # This value is deterministic public test data derived from the
            # report's cell_seed; it is neither loaded from nor usable as a
            # deployment secret.
            hash_key = _stable_uint("kgw-hash-key", spec["cell_seed"], 0, 64)
            cfg: Any = KGWConfig(
                vocab_size=spec["vocab_size"],
                hash_key=hash_key,
                gamma=spec["kgw_gamma"],
            )
            score_fn = lambda ids: kgw_score_token_ids(  # noqa: E731
                ids,
                cfg,
                ignore_repeated_ngrams=spec["kgw_ignore_repeated_ngrams"],
                z_threshold=spec["z_threshold"],
            )
            expected_depth = None
        else:
            synthid_keys = tuple(
                _stable_uint("synthid-subkey", spec["cell_seed"], i, 63)
                for i in range(spec["synthid_depth"])
            )
            cfg = SynthIDConfig(
                vocab_size=spec["vocab_size"],
                keys=synthid_keys,
                ngram_len=spec["synthid_ngram_len"],
                sampling_table_size=spec["synthid_sampling_table_size"],
                sampling_table_seed=spec["synthid_sampling_table_seed"],
                context_history_size=spec["synthid_context_history_size"],
            )
            scorer = spec["synthid_scorer"]
            score_fn = (
                (lambda ids: score_token_ids_mean(ids, cfg, z_threshold=spec["z_threshold"]))
                if scorer == "mean"
                else (
                    lambda ids: score_token_ids_weighted_mean(
                        ids, cfg, z_threshold=spec["z_threshold"]
                    )
                )
            )
            expected_depth = spec["synthid_depth"]

        expected_too_short = spec["expected_outcome"] == "too_short_error"
        for attempt in range(spec["repeats"]):
            _emit_worker_event({"event": "attempt_started", "attempt": attempt})
            tokens = _make_tokens(
                spec["pattern"],
                spec["length"],
                spec["vocab_size"],
                spec["cell_seed"],
                attempt,
                spec["repeated_block_size"],
            )
            started_ns = time.perf_counter_ns()
            try:
                result = score_fn(tokens)
            except ValueError:
                latency_ns = time.perf_counter_ns() - started_ns
                category = "expected_too_short" if expected_too_short else "technical_failure"
                unexpected += int(category == "technical_failure")
                _emit_worker_event(
                    {
                        "event": "attempt_result",
                        "attempt": attempt,
                        "category": category,
                        "error_type": "ValueError",
                        "input_tokens": len(tokens),
                        "latency_ns": latency_ns,
                        "scored_tokens": 0,
                    }
                )
                continue
            except Exception as exc:  # noqa: BLE001 - the harness classifies arbitrary detector failures
                latency_ns = time.perf_counter_ns() - started_ns
                unexpected += 1
                _emit_worker_event(
                    {
                        "event": "attempt_result",
                        "attempt": attempt,
                        "category": "technical_failure",
                        "error_type": type(exc).__name__,
                        "input_tokens": len(tokens),
                        "latency_ns": latency_ns,
                        "scored_tokens": 0,
                    }
                )
                continue

            latency_ns = time.perf_counter_ns() - started_ns
            max_scored = max(0, len(tokens) - (1 if scheme == "kgw" else spec["synthid_ngram_len"] - 1))
            coherent, scored_tokens = _validate_result(result, scheme, expected_depth, max_scored)
            if expected_too_short:
                category = "unexpected_success"
                unexpected += 1
            elif not coherent:
                numeric_values = (
                    (result.z_score, result.p_value)
                    if scheme == "kgw"
                    else (result.mean_g, result.score, result.z_score, result.p_value)
                )
                category = (
                    "nonfinite"
                    if not all(math.isfinite(float(value)) for value in numeric_values)
                    else "technical_failure"
                )
                unexpected += 1
            else:
                category = "success"

            _emit_worker_event(
                {
                    "event": "attempt_result",
                    "attempt": attempt,
                    "category": category,
                    "error_type": None,
                    "input_tokens": len(tokens),
                    "latency_ns": latency_ns,
                    "scored_tokens": scored_tokens,
                }
            )

        _emit_worker_event(
            {
                "event": "worker_summary",
                "peak_rss_bytes": _worker_peak_rss_bytes(),
                "unexpected_outcomes": unexpected,
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - fatal worker failures must become data, not leaked traces
        _emit_worker_event(
            {
                "event": "worker_fatal",
                "error_type": type(exc).__name__,
                "peak_rss_bytes": _worker_peak_rss_bytes(),
            }
        )
        return 3


class _ProcessMemoryMonitor:
    """Best-effort Linux RSS/HWM monitor, including workers killed on timeout."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rss-monitor-{pid}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _sample(self) -> None:
        status_path = Path(f"/proc/{self.pid}/status")
        try:
            lines = status_path.read_text(encoding="ascii", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            if line.startswith(("VmRSS:", "VmHWM:")):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    value = int(parts[1]) * 1024
                    self.peak_rss_bytes = max(self.peak_rss_bytes or 0, value)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.01)
        self._sample()


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - rank) + sorted_values[upper] * (rank - lower)


def _latency_summary(latencies_ns: Iterable[int]) -> dict[str, float | int | None]:
    values = sorted(value / 1_000_000_000.0 for value in latencies_ns)
    return {
        "count": len(values),
        "p50_seconds": _percentile(values, 50),
        "p95_seconds": _percentile(values, 95),
        "p99_seconds": _percentile(values, 99),
        "max_seconds": values[-1] if values else None,
    }


def _wilson_95(successes: int, denominator: int) -> dict[str, float | int | None]:
    if denominator <= 0:
        return {
            "count": successes,
            "denominator": denominator,
            "rate": None,
            "lower": None,
            "upper": None,
        }
    proportion = successes / denominator
    z2 = _WILSON_Z_95**2
    scale = 1.0 + z2 / denominator
    center = (proportion + z2 / (2.0 * denominator)) / scale
    half = (
        _WILSON_Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator + z2 / (4.0 * denominator**2)
        )
        / scale
    )
    return {
        "count": successes,
        "denominator": denominator,
        "rate": proportion,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
    }


def _encode_spec(spec: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(_json_dumps(spec).encode("utf-8")).decode("ascii")


def _safe_stream_metadata(stream: str) -> dict[str, Any]:
    encoded = stream.encode("utf-8", errors="replace")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _parse_worker_events(stdout: str) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(value, dict) or not isinstance(value.get("event"), str):
            malformed += 1
            continue
        events.append(value)
    return events, malformed


def _run_cell(script_path: Path, spec: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    command = [sys.executable, str(script_path), "--_worker-spec", _encode_spec(spec)]
    wall_started = time.perf_counter()
    process = subprocess.Popen(  # noqa: S603 - command is fixed; payload is generated locally
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    monitor = _ProcessMemoryMonitor(process.pid)
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        monitor.stop()
    subprocess_wall_seconds = time.perf_counter() - wall_started

    events, malformed_lines = _parse_worker_events(stdout)
    start_events = [event for event in events if event["event"] == "attempt_started"]
    starts = {
        event["attempt"]
        for event in start_events
        if type(event.get("attempt")) is int
    }
    result_events = [event for event in events if event["event"] == "attempt_result"]
    valid_result_ids = [
        event["attempt"]
        for event in result_events
        if type(event.get("attempt")) is int
    ]
    result_attempts = set(valid_result_ids)
    duplicate_results = len(valid_result_ids) - len(result_attempts)
    protocol_errors = malformed_lines + duplicate_results + _protocol_errors(events, spec["repeats"], timed_out)
    expected_attempts = set(range(spec["repeats"]))
    if not timed_out and (starts != expected_attempts or result_attempts - expected_attempts):
        protocol_errors += 1
    if any(
        not isinstance(event.get("input_tokens"), int)
        or event.get("input_tokens", -1) != spec["length"]
        or not isinstance(event.get("scored_tokens"), int)
        or event.get("scored_tokens", -1) < 0
        or event.get("scored_tokens", 0) > max(0, spec["length"] - (1 if spec["scheme"] == "kgw" else spec["synthid_ngram_len"] - 1))
        for event in result_events
    ):
        protocol_errors += 1

    categories = {
        name: sum(1 for event in result_events if event.get("category") == name)
        for name in (
            "success",
            "expected_too_short",
            "technical_failure",
            "nonfinite",
            "unexpected_success",
        )
    }
    technical_failure_types: dict[str, int] = {}
    for event in result_events:
        if event.get("category") != "technical_failure":
            continue
        error_type = event.get("error_type")
        safe_type = error_type if isinstance(error_type, str) and error_type else "InvariantViolation"
        technical_failure_types[safe_type] = technical_failure_types.get(safe_type, 0) + 1
    unknown_categories = sum(
        1
        for event in result_events
        if event.get("category") not in categories
    )
    protocol_errors += unknown_categories

    worker_summaries = [event for event in events if event["event"] == "worker_summary"]
    worker_fatals = [event for event in events if event["event"] == "worker_fatal"]
    for event in worker_fatals:
        error_type = event.get("error_type")
        safe_type = error_type if isinstance(error_type, str) and error_type else "WorkerFatal"
        technical_failure_types[safe_type] = technical_failure_types.get(safe_type, 0) + 1
    worker_started = any(event["event"] == "worker_started" for event in events)
    if len(worker_summaries) > 1 or len(worker_fatals) > 1:
        protocol_errors += 1
    if not timed_out and not worker_started:
        protocol_errors += 1
    if not timed_out and not worker_fatals and not worker_summaries:
        protocol_errors += 1

    worker_peak = None
    for event in worker_summaries + worker_fatals:
        value = event.get("peak_rss_bytes")
        if isinstance(value, int):
            worker_peak = max(worker_peak or 0, value)
    peak_rss_bytes = max(
        (value for value in (monitor.peak_rss_bytes, worker_peak) if value is not None),
        default=None,
    )

    attempts_started = len(starts)
    attempts_completed = len(result_attempts)
    incomplete_attempts = len(starts - result_attempts)
    contract_successes = categories["success"] + categories["expected_too_short"]
    cell_timeouts = int(timed_out)
    attempt_timeouts = int(timed_out and incomplete_attempts > 0)
    worker_failures = int(bool(worker_fatals))
    if not timed_out and process.returncode not in (0, None) and not worker_fatals:
        worker_failures += 1

    all_latencies = [
        int(event["latency_ns"])
        for event in result_events
        if isinstance(event.get("latency_ns"), int) and event["latency_ns"] >= 0
    ]
    success_events = [event for event in result_events if event.get("category") == "success"]
    success_latency_ns = sum(
        int(event["latency_ns"])
        for event in success_events
        if isinstance(event.get("latency_ns"), int) and event["latency_ns"] > 0
    )
    total_input_tokens = sum(
        event["input_tokens"] for event in success_events
        if type(event.get("input_tokens")) is int and event["input_tokens"] >= 0
    )
    total_scored_tokens = sum(
        event["scored_tokens"] for event in success_events
        if type(event.get("scored_tokens")) is int and event["scored_tokens"] >= 0
    )
    throughput_seconds = success_latency_ns / 1_000_000_000.0

    technical_failures = categories["technical_failure"] + worker_failures + protocol_errors
    # A timeout already explains its in-flight and unstarted attempts; count
    # that cell-level outcome once rather than inflating it with consequences
    # of the kill. Without a timeout, missing attempt events are independent
    # protocol failures and remain visible.
    missing_attempts = 0 if timed_out else max(0, spec["repeats"] - attempts_started)
    unexplained_incomplete = 0 if timed_out else incomplete_attempts
    unexpected_outcomes = (
        technical_failures
        + categories["nonfinite"]
        + categories["unexpected_success"]
        + cell_timeouts
        + unexplained_incomplete
        + missing_attempts
    )

    attempt_denominator = attempts_started
    rates = {
        "contract_success": _wilson_95(contract_successes, attempt_denominator),
        "scoring_success": _wilson_95(categories["success"], attempt_denominator),
        "expected_too_short": _wilson_95(categories["expected_too_short"], attempt_denominator),
        # Protocol/worker failures are cell-level counters and can make the
        # aggregate `technical_failures` exceed attempt count. Wilson bounds
        # here therefore use only mutually exclusive attempt outcomes.
        "technical_failure_attempt": _wilson_95(
            categories["technical_failure"], attempt_denominator
        ),
        "nonfinite": _wilson_95(categories["nonfinite"], attempt_denominator),
        "attempt_timeout": _wilson_95(attempt_timeouts, attempt_denominator),
        "cell_timeout": _wilson_95(cell_timeouts, 1),
    }

    return {
        "cell_id": spec["cell_id"],
        "config": {key: value for key, value in spec.items() if key != "cell_id"},
        "attempts_requested": spec["repeats"],
        "attempts_started": attempts_started,
        "attempts_completed": attempts_completed,
        "incomplete_attempts": incomplete_attempts,
        "contract_successes": contract_successes,
        "scoring_successes": categories["success"],
        "expected_too_short_errors": categories["expected_too_short"],
        "technical_failure_attempts": categories["technical_failure"],
        "technical_failures": technical_failures,
        "technical_failure_types": dict(sorted(technical_failure_types.items())),
        "timeouts": cell_timeouts,
        "attempt_timeouts": attempt_timeouts,
        "nonfinite_results": categories["nonfinite"],
        "unexpected_successes": categories["unexpected_success"],
        "protocol_errors": protocol_errors,
        "worker_failures": worker_failures,
        "unexpected_outcomes": unexpected_outcomes,
        "latency": _latency_summary(all_latencies),
        "tokens_per_second": {
            "input_tokens": (total_input_tokens / throughput_seconds) if throughput_seconds > 0 else None,
            "scored_tokens": (total_scored_tokens / throughput_seconds) if throughput_seconds > 0 else None,
            "timed_seconds": throughput_seconds,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "subprocess_wall_seconds": subprocess_wall_seconds,
        "rates_wilson95": rates,
        # Stderr/stdout contents are deliberately excluded. Hash+length are
        # sufficient for diagnosing nondeterministic worker noise without
        # risking text or environment leakage into the report.
        "worker_streams": {
            "stderr": _safe_stream_metadata(stderr),
            "unparsed_stdout_lines": malformed_lines,
        },
        "worker_exit_code": process.returncode,
    }


def _parse_csv_ints(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        item = raw.strip().replace("_", "")
        if not item:
            raise argparse.ArgumentTypeError("comma-separated integer lists cannot contain empty items")
        try:
            items.append(int(item, 0))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {raw!r}") from exc
    return items


def _parse_csv_choices(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid {name} {invalid!r}; choose from {allowed!r}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schemes", default="kgw,synthid", help="comma-separated: kgw,synthid")
    parser.add_argument(
        "--patterns",
        default="uniform_random",
        help=f"comma-separated: {','.join(PATTERNS)}",
    )
    parser.add_argument("--lengths", default="1,32", help="comma-separated input token counts, each >=0")
    parser.add_argument("--vocab-sizes", default="128", help="comma-separated vocabulary sizes, each >0")
    parser.add_argument("--repeats", type=int, default=2, help="attempts per isolated cell")
    parser.add_argument("--seed", type=int, default=20260809, help="public deterministic harness seed")
    parser.add_argument("--timeout-seconds", type=float, default=30.0, help="hard wall timeout per cell")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=512,
        help="matrix-size safety guard; raise explicitly for extremes",
    )
    parser.add_argument("--torch-threads", type=int, default=1, help="intra-op torch threads per worker")
    parser.add_argument("--repeated-block-size", type=int, default=8)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--kgw-gamma", type=float, default=0.25)
    parser.add_argument(
        "--kgw-ignore-repeated-ngrams",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--synthid-depths", default="4", help="comma-separated positive tournament depths")
    parser.add_argument("--synthid-scorers", default="weighted_mean", help="comma-separated: mean,weighted_mean")
    parser.add_argument("--synthid-ngram-len", type=int, default=5)
    parser.add_argument("--synthid-sampling-table-size", type=int, default=1 << 12)
    parser.add_argument("--synthid-sampling-table-seed", type=int, default=0)
    parser.add_argument("--synthid-context-history-size", type=int, default=1024)
    parser.add_argument("--output", default="-", help="JSON path, or '-' for stdout")
    parser.add_argument("--compact", action="store_true", help="emit compact rather than indented JSON")
    # Private worker transport. The parent is the only supported caller.
    parser.add_argument("--_worker-spec", default=None, help=argparse.SUPPRESS)
    return parser


def _validated_matrix_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict[str, Any]:
    try:
        schemes = _parse_csv_choices(args.schemes, SCHEMES, "schemes")
        patterns = _parse_csv_choices(args.patterns, PATTERNS, "patterns")
        scorers = _parse_csv_choices(args.synthid_scorers, SYNTHID_SCORERS, "synthid scorers")
        lengths = _parse_csv_ints(args.lengths)
        vocab_sizes = _parse_csv_ints(args.vocab_sizes)
        depths = _parse_csv_ints(args.synthid_depths)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if any(value < 0 for value in lengths):
        parser.error("all --lengths values must be >= 0")
    if len(set(lengths)) != len(lengths) or len(set(vocab_sizes)) != len(vocab_sizes) or len(set(depths)) != len(depths):
        parser.error("matrix values must be unique")
    if any(value <= 0 for value in vocab_sizes):
        parser.error("all --vocab-sizes values must be > 0")
    if any(value <= 0 for value in depths):
        parser.error("all --synthid-depths values must be > 0")
    if not 1 <= args.repeats <= 1000:
        parser.error("--repeats must be in [1,1000]")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and > 0")
    if not 1 <= args.max_cells <= 1000:
        parser.error("--max-cells must be in [1,1000]")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be > 0")
    if not 1 <= args.repeated_block_size <= 65_536:
        parser.error("--repeated-block-size must be in [1,65536]")
    if not math.isfinite(args.z_threshold):
        parser.error("--z-threshold must be finite")
    if not math.isfinite(args.kgw_gamma) or not 0.0 < args.kgw_gamma < 1.0:
        parser.error("--kgw-gamma must be finite and in (0,1)")
    if not 1 <= args.synthid_ngram_len <= 1024:
        parser.error("--synthid-ngram-len must be in [1,1024]")
    if not 1 <= args.synthid_sampling_table_size <= 1 << 21:
        parser.error("--synthid-sampling-table-size must be in [1,2**21]")
    if not -(1 << 63) <= args.synthid_sampling_table_seed <= (1 << 64) - 1:
        parser.error("--synthid-sampling-table-seed is outside torch's accepted range")
    if not 0 <= args.synthid_context_history_size <= 65_536:
        parser.error("--synthid-context-history-size must be in [0,65536]")
    if any(v > (1 << 20) for v in vocab_sizes) or any(v > 65_536 for v in lengths) or any(d > 256 for d in depths):
        parser.error("matrix values exceed deployment safety caps")

    return {
        "schemes": schemes,
        "patterns": patterns,
        "lengths": lengths,
        "vocab_sizes": vocab_sizes,
        "depths": depths,
        "scorers": scorers,
    }


def _derive_cell_seed(base_seed: int, identity: dict[str, Any]) -> int:
    material = _json_dumps({"base_seed": base_seed, "identity": identity}).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _build_specs(args: argparse.Namespace, matrix: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for scheme in matrix["schemes"]:
        for vocab_size in matrix["vocab_sizes"]:
            for length in matrix["lengths"]:
                for pattern in matrix["patterns"]:
                    variants = (
                        [(None, None)]
                        if scheme == "kgw"
                        else [
                            (depth, scorer)
                            for depth in matrix["depths"]
                            for scorer in matrix["scorers"]
                        ]
                    )
                    for depth, scorer in variants:
                        expected_outcome = (
                            "too_short_error"
                            if length < (2 if scheme == "kgw" else args.synthid_ngram_len)
                            else "valid_score"
                        )
                        identity = {
                            "scheme": scheme,
                            "vocab_size": vocab_size,
                            "length": length,
                            "pattern": pattern,
                            "synthid_depth": depth,
                            "synthid_scorer": scorer,
                        }
                        spec = {
                            **identity,
                            "repeats": args.repeats,
                            "cell_seed": _derive_cell_seed(args.seed, identity),
                            "synthetic_key_derivation": f"{HARNESS_VERSION}:sha256-v1",
                            "minimum_scoreable_tokens": (
                                2 if scheme == "kgw" else args.synthid_ngram_len
                            ),
                            "expected_outcome": expected_outcome,
                            "torch_threads": args.torch_threads,
                            "repeated_block_size": args.repeated_block_size,
                            "z_threshold": args.z_threshold,
                            "kgw_gamma": args.kgw_gamma,
                            "kgw_effective_greenlist_size": int(vocab_size * args.kgw_gamma),
                            "kgw_ignore_repeated_ngrams": args.kgw_ignore_repeated_ngrams,
                            "synthid_ngram_len": args.synthid_ngram_len,
                            "synthid_sampling_table_size": args.synthid_sampling_table_size,
                            "synthid_sampling_table_seed": args.synthid_sampling_table_seed,
                            "synthid_context_history_size": args.synthid_context_history_size,
                        }
                        specs.append(spec)

    for index, spec in enumerate(specs, start=1):
        spec["cell_id"] = f"cell-{index:05d}"
    return specs


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _environment_metadata(repo_root: Path) -> dict[str, Any]:
    # Import only version surfaces. Never enumerate or serialize environment
    # variables: the process may contain live watermark/cluster credentials.
    try:
        import torch

        torch_metadata: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "default_num_threads": torch.get_num_threads(),
        }
    except ImportError:
        torch_metadata = {"version": None, "cuda_available": False, "cuda_version": None}
    try:
        import vllm_watermark

        package_version = vllm_watermark.__version__
    except ImportError:
        package_version = None
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness_version": HARNESS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "torch": torch_metadata,
        "vllm_watermark_version": package_version,
        "git": _git_metadata(repo_root),
    }


def _aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    summed_fields = (
        "attempts_requested",
        "attempts_started",
        "attempts_completed",
        "incomplete_attempts",
        "contract_successes",
        "scoring_successes",
        "expected_too_short_errors",
        "technical_failure_attempts",
        "technical_failures",
        "timeouts",
        "attempt_timeouts",
        "nonfinite_results",
        "unexpected_successes",
        "protocol_errors",
        "worker_failures",
        "unexpected_outcomes",
    )
    totals = {field: sum(int(cell[field]) for cell in cells) for field in summed_fields}
    denominator = totals["attempts_started"]
    totals["cells"] = len(cells)
    totals["cells_with_unexpected_outcomes"] = sum(cell["unexpected_outcomes"] > 0 for cell in cells)
    totals["peak_rss_bytes_max"] = max(
        (cell["peak_rss_bytes"] for cell in cells if cell["peak_rss_bytes"] is not None),
        default=None,
    )
    totals["rates_wilson95"] = {
        "contract_success": _wilson_95(totals["contract_successes"], denominator),
        "scoring_success": _wilson_95(totals["scoring_successes"], denominator),
        "expected_too_short": _wilson_95(totals["expected_too_short_errors"], denominator),
        "technical_failure_attempt": _wilson_95(
            totals["technical_failure_attempts"], denominator
        ),
        # Cell-level worker/protocol failures remain exact counters above and
        # are intentionally not forced into an attempt-rate denominator.
        "timeout": _wilson_95(totals["attempt_timeouts"], denominator),
        "cell_timeout": _wilson_95(totals["timeouts"], len(cells)),
        "nonfinite": _wilson_95(totals["nonfinite_results"], denominator),
    }
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args._worker_spec is not None:
        return _worker_main(args._worker_spec)

    matrix = _validated_matrix_args(args, parser)
    base = len(matrix["vocab_sizes"]) * len(matrix["lengths"]) * len(matrix["patterns"])
    per_scheme = (int("kgw" in matrix["schemes"]) +
                  int("synthid" in matrix["schemes"]) * len(matrix["depths"]) * len(matrix["scorers"]))
    estimated = base * per_scheme
    if estimated > args.max_cells:
        parser.error(f"matrix expands to {estimated} cells, exceeding --max-cells={args.max_cells}")
    specs = _build_specs(args, matrix)
    if len(specs) > args.max_cells:
        parser.error(
            f"matrix expands to {len(specs)} cells, exceeding --max-cells={args.max_cells}; "
            "raise --max-cells explicitly after reviewing expected CPU/RSS cost"
        )

    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    cells = []
    run_started = time.perf_counter()
    for spec in specs:
        cells.append(_run_cell(script_path, spec, args.timeout_seconds))
    wall_seconds = time.perf_counter() - run_started

    report = {
        "schema_version": 1,
        "environment": _environment_metadata(repo_root),
        "run_config": {
            "schemes": matrix["schemes"],
            "patterns": matrix["patterns"],
            "lengths": matrix["lengths"],
            "vocab_sizes": matrix["vocab_sizes"],
            "repeats_per_cell": args.repeats,
            "base_seed": args.seed,
            "timeout_seconds_per_cell": args.timeout_seconds,
            "torch_threads_per_worker": args.torch_threads,
            "repeated_block_size": args.repeated_block_size,
            "z_threshold": args.z_threshold,
            "kgw_gamma": args.kgw_gamma,
            "kgw_ignore_repeated_ngrams": args.kgw_ignore_repeated_ngrams,
            "synthid_depths": matrix["depths"],
            "synthid_scorers": matrix["scorers"],
            "synthid_ngram_len": args.synthid_ngram_len,
            "synthid_sampling_table_size": args.synthid_sampling_table_size,
            "synthid_sampling_table_seed": args.synthid_sampling_table_seed,
            "synthid_context_history_size": args.synthid_context_history_size,
            "synthetic_inputs_only": True,
            "raw_text_or_token_ids_emitted": False,
            "deployment_keys_or_environment_read": False,
        },
        "matrix_cells": len(cells),
        "wall_seconds": wall_seconds,
        "summary": _aggregate(cells),
        "cells": cells,
    }
    payload = _json_dumps(report, pretty=not args.compact) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        Path(args.output).write_text(payload, encoding="utf-8")

    return 1 if report["summary"]["unexpected_outcomes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
