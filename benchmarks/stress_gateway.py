#!/usr/bin/env python3
"""High-count, deterministic soak test for the synchronous gateway.

The harness uses in-process synthetic upstream/validator adapters and the real
``GatewayService``/SQLite sampler. It emits aggregate JSON only: no response
text, request identifiers, hashes, or Secret material. This is an operational
state-machine stress test, not an end-to-end vLLM or detector result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Direct ``python benchmarks/stress_gateway.py`` starts with benchmarks/ on
# sys.path; make the repository package importable without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.gateway import GatewayConfig, GatewayService, ValidationOutcome, ValidationRequest


_Z_95 = 1.959963984540054
_MAX_MATRIX_VALUES = 16
_MAX_MATRIX_CELLS = 64


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def rate(events: int, total: int) -> dict[str, Any]:
    interval = None
    if total:
        proportion = events / total
        z2 = _Z_95 * _Z_95
        denominator = 1.0 + z2 / total
        center = (proportion + z2 / (2.0 * total)) / denominator
        margin = (
            _Z_95
            * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total)
            / denominator
        )
        interval = [max(0.0, center - margin), min(1.0, center + margin)]
    return {
        "events": events,
        "total": total,
        "rate": events / total if total else None,
        "wilson_95": interval,
    }


class SyntheticUpstream:
    def __init__(self) -> None:
        self.count = 0

    async def complete(self, endpoint: str, request: dict[str, object]) -> dict[str, object]:
        self.count += 1
        sequence = self.count
        # Yield at deterministic positions so completion order varies under
        # concurrency without wall-clock sleeps making the run flaky.
        if sequence % 7 == 0:
            await asyncio.sleep(0)
        return {
            "id": f"cmpl-{sequence}",
            "choices": [{"text": f"synthetic-output-{sequence}"}],
        }


class SyntheticValidator:
    async def validate(self, request: ValidationRequest) -> ValidationOutcome:
        if int(request.response_id.rsplit("-", 1)[1]) % 11 == 0:
            await asyncio.sleep(0)
        return ValidationOutcome(
            "watermarked" if request.expected_enabled else "clean",
            "blocked" if request.expected_enabled else "success",
        )


def request_body(index: int) -> dict[str, object]:
    enabled = index % 2 == 0
    return {
        "model": "synthetic-model",
        "n": 1,
        "vllm_xargs": {
            "watermark": "on" if enabled else "off",
            "watermark_scheme": "kgw" if index % 4 < 2 else "synthid",
            "watermark_key_id": "synthetic-key",
        },
    }


async def run_cell(total: int, concurrency: int, sample_every: int, root: Path) -> dict[str, Any]:
    service = GatewayService(
        GatewayConfig(
            sample_every=sample_every,
            sqlite_path=root / f"gateway-n{sample_every}-c{concurrency}.sqlite",
            positive_policy="flag",
            queue_capacity=max(32, concurrency * 2),
            max_inflight=max(4, concurrency),
            retry_backoff_seconds=0.0,
        ),
        SyntheticUpstream(),
        SyntheticValidator(),
    )
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    exceptions: list[str] = []
    responses: dict[int, dict[str, Any]] = {}
    status: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    wall_seconds = 0.0
    primary_exception: str | None = None
    cleanup_exception: str | None = None

    async def one(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await service.proxy("/v1/completions", request_body(index))
                if isinstance(response, dict):
                    responses[index] = response
            except Exception as exc:  # counted and sanitized below
                exceptions.append(type(exc).__name__)
            finally:
                latencies.append(time.perf_counter() - started)

    wall_started = time.perf_counter()
    try:
        await service.start()
        # Task creation is chunked so an extreme count does not create every
        # coroutine/future at once.
        for start in range(0, total, max(concurrency * 4, 1)):
            await asyncio.gather(*(one(i) for i in range(start, min(total, start + concurrency * 4))))
        wall_seconds = time.perf_counter() - wall_started
        await service._queue.join()
        status = service.status(None)
        records = service.harness_records(None)
    except Exception as exc:  # preserve a safe, aggregate-only failed cell
        primary_exception = type(exc).__name__
        wall_seconds = time.perf_counter() - wall_started
    finally:
        try:
            await service.stop()
        except Exception as exc:
            cleanup_exception = type(exc).__name__

    if primary_exception is None and cleanup_exception is not None:
        primary_exception = cleanup_exception

    if primary_exception is not None or status is None:
        # Cleanup errors are deliberately ignored above so they cannot mask the
        # primary failure. Keep the cell serializable and free of exception text.
        return {
            "config": {"requests": total, "concurrency": concurrency, "sample_every": sample_every},
            "counts": {
                "expected_selected": total // sample_every,
                "records": len(records),
                "unique_validation_ids": 0,
                "unexpected_exceptions": total,
                "invariant_failures": 1,
            },
            "request_failure_rate": rate(total, total),
            "invariant_failures": 1,
            "invariants_passed": False,
            "invariant_failure_categories": ["cell_exception"],
            "exception_categories": [primary_exception or "CellError"],
            "wall_seconds": wall_seconds,
            "requests_per_second": total / wall_seconds if wall_seconds else None,
            "latency_seconds": {
                "mean": statistics.fmean(latencies) if latencies else None,
                "p50": percentile(latencies, 50), "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99), "p99_9": percentile(latencies, 99.9),
                "max": max(latencies) if latencies else None,
            },
            "queue_peak_depth": None,
            "counter_snapshot": None,
        }

    expected_selected = total // sample_every
    successful_records = [row for row in records if row["terminal_state"] == "success"]
    validation_ids = [row["validation_id"] for row in records]
    observed_failures = len(exceptions)
    invariant_failures: list[str] = []
    counters = status["counters"]
    expected = {
        "started": total,
        "completed": total,
        "selected": expected_selected,
        "unsampled": total - expected_selected,
        "terminal": expected_selected,
        "errors": 0,
        "failed": 0,
        "cancelled": 0,
        "retries": 0,
        "queue_overflow": 0,
        "dropped": 0,
    }
    for name, value in expected.items():
        if counters[name] != value:
            invariant_failures.append(f"counter_{name}")
    if len(records) != expected_selected:
        invariant_failures.append("record_count")
    if len(successful_records) != expected_selected:
        invariant_failures.append("successful_record_count")
    if len(validation_ids) != len(set(validation_ids)):
        invariant_failures.append("duplicate_validation_id")
    response_ids = [row["response_id"] for row in records]
    if len(response_ids) != len(set(response_ids)) or any(
        not isinstance(row.get("response_id"), str) or not row["response_id"].startswith("cmpl-")
        or row.get("attempts") != 1 or row.get("delivery_outcome") != "delivered"
        or row.get("terminal_state") != "success" or not row.get("validation_id")
        for row in records
    ):
        invariant_failures.append("record_correlation")
    if len(responses) != total:
        invariant_failures.append("response_count")
    selected_meta: dict[str, dict[str, Any]] = {}
    required_meta = ("ordinal", "validation_id", "response_id", "content_digest", "scheme", "key_id")
    for response in responses.values():
        meta = response.get("watermark_validation")
        if isinstance(meta, dict) and meta.get("selected") is True:
            validation_id = meta.get("validation_id")
            if isinstance(validation_id, str):
                selected_meta[validation_id] = meta
    if len(selected_meta) != expected_selected:
        invariant_failures.append("selected_metadata_count")
    record_by_validation = {row.get("validation_id"): row for row in records}
    for validation_id, meta in selected_meta.items():
        if any(field not in meta for field in required_meta):
            invariant_failures.append("selected_metadata_fields")
            continue
        if not isinstance(meta["ordinal"], int) or meta["ordinal"] <= 0 or meta["ordinal"] % sample_every != 0:
            invariant_failures.append("selected_metadata_ordinal")
        row = record_by_validation.get(validation_id)
        if row is None or any(row.get(field) != meta[field] for field in ("validation_id", "response_id", "content_digest", "scheme", "key_id")):
            invariant_failures.append("selected_metadata_correlation")
    if status["queue"]["depth"] != 0:
        invariant_failures.append("queue_not_drained")
    if status["latency_samples"] != {
        "generation_completion": total,
        "client_delivery": total,
        "validation": expected_selected,
        "validation_lag": expected_selected,
    }:
        invariant_failures.append("latency_sample_counts")

    return {
        "config": {
            "requests": total,
            "concurrency": concurrency,
            "sample_every": sample_every,
        },
        "counts": {
            "expected_selected": expected_selected,
            "records": len(records),
            "unique_validation_ids": len(set(validation_ids)),
            "unexpected_exceptions": observed_failures,
            "invariant_failures": len(invariant_failures),
        },
        "request_failure_rate": rate(observed_failures, total),
        "invariant_failures": len(invariant_failures),
        "invariants_passed": not invariant_failures,
        "invariant_failure_categories": sorted(set(invariant_failures)),
        "exception_categories": sorted(set(exceptions)),
        "wall_seconds": wall_seconds,
        "requests_per_second": total / wall_seconds,
        "latency_seconds": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "p99_9": percentile(latencies, 99.9),
            "max": max(latencies),
        },
        "queue_peak_depth": status["queue"]["peak_depth"],
        "counter_snapshot": counters,
    }


def parse_positive_csv(value: str, name: str) -> list[int]:
    if value.count(",") + 1 > _MAX_MATRIX_VALUES:
        raise argparse.ArgumentTypeError(f"{name} list exceeds {_MAX_MATRIX_VALUES} values")
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be positive")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be unique")
    if len(parsed) > _MAX_MATRIX_VALUES:
        raise argparse.ArgumentTypeError(f"{name} list exceeds {_MAX_MATRIX_VALUES} values")
    if name == "--concurrency" and any(item > 256 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} values must be at most 256")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1_000, help="requests per matrix cell")
    parser.add_argument("--concurrency", default="1,16,64")
    parser.add_argument("--sample-every", default="1,5,97")
    parser.add_argument("--out", type=Path)
    return parser


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    concurrencies = parse_positive_csv(args.concurrency, "--concurrency")
    sample_rates = parse_positive_csv(args.sample_every, "--sample-every")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="vllm-watermark-gateway-stress-") as directory:
        root = Path(directory).resolve()
        cells = []
        for sample_every in sample_rates:
            for concurrency in concurrencies:
                if len(cells) >= _MAX_MATRIX_CELLS:
                    raise argparse.ArgumentTypeError(
                        f"matrix exceeds {_MAX_MATRIX_CELLS} cells"
                    )
                cells.append(await run_cell(args.requests, concurrency, sample_every, root))
    total_attempts = sum(cell["config"]["requests"] for cell in cells)
    total_request_failures = sum(cell["request_failure_rate"]["events"] for cell in cells)
    total_invariant_failures = sum(cell["invariant_failures"] for cell in cells)
    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024
    return {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scope": "synthetic adapters plus real GatewayService and SQLiteSampler",
        "matrix": {
            "requests_per_cell": args.requests,
            "concurrency": concurrencies,
            "sample_every": sample_rates,
            "cells": len(cells),
        },
        "aggregate": {
            "attempts": total_attempts,
            "request_failures": total_request_failures,
            "request_failure_rate": rate(total_request_failures, total_attempts),
            "invariant_failures": total_invariant_failures,
            "invariants_passed": total_invariant_failures == 0,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "cells": cells,
        "passed": total_request_failures == 0 and total_invariant_failures == 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.requests <= 100_000:
        print("error: --requests must be in [1,100000]", file=sys.stderr)
        return 2
    try:
        report = asyncio.run(async_main(args))
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        print(f"wrote {args.out}")
    else:
        print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
