#!/usr/bin/env python3
"""Aggregate repeated serving benchmarks without exposing request content.

Each input must be produced by ``bench_serving.py`` with ``--condition``
and ``--trial``. Results are paired by trial within an identical load cell
(model, request count, max tokens, temperature, and concurrency). The report
keeps raw numerator/denominator counts, Wilson intervals, per-trial dispersion,
and paired throughput overhead relative to a named baseline condition.

This analyzer never reads or emits per-request prompt identifiers or errors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


_Z_95 = 1.959963984540054


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def wilson_interval(events: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    proportion = events / total
    z2 = _Z_95 * _Z_95
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        _Z_95
        * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def rate(events: int, total: int) -> dict[str, Any]:
    return {
        "events": events,
        "total": total,
        "rate": events / total if total else None,
        "wilson_95": wilson_interval(events, total),
    }


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def bootstrap_mean_ci(values: list[float], samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1 or samples <= 0:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(values) for _ in range(len(values)))
        for _ in range(samples)
    ]
    return [percentile(means, 2.5), percentile(means, 97.5)]  # type: ignore[list-item]


def _positive_finite(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return parsed


def _nonnegative_finite(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report["config"]
        summary = report["summary"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid serving report {path}: {type(exc).__name__}") from exc
    if report.get("schema_version") != 2:
        raise ValueError(f"{path}: expected schema_version=2")
    condition = config.get("condition")
    trial = config.get("trial")
    if not isinstance(condition, str) or not condition:
        raise ValueError(f"{path}: config.condition is required")
    _nonnegative_int(trial, f"{path}: config.trial")
    total = _nonnegative_int(summary.get("requests_total"), f"{path}: requests_total")
    ok = _nonnegative_int(summary.get("requests_ok"), f"{path}: requests_ok")
    failed = _nonnegative_int(summary.get("requests_failed"), f"{path}: requests_failed")
    if ok + failed != total:
        raise ValueError(f"{path}: requests_ok + requests_failed != requests_total")
    _positive_finite(summary.get("wall_time_s"), f"{path}: wall_time_s")
    _nonnegative_finite(summary.get("output_tokens_per_s"), f"{path}: output_tokens_per_s")
    _nonnegative_finite(summary.get("requests_per_s"), f"{path}: requests_per_s")
    return report


def cell_key(config: dict[str, Any]) -> tuple[Any, ...]:
    return (
        config.get("model"),
        config.get("n"),
        config.get("max_tokens"),
        config.get("temperature"),
        config.get("concurrency"),
    )


def summarize_condition(reports: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(report["summary"]["requests_total"] for report in reports)
    failures = sum(report["summary"]["requests_failed"] for report in reports)
    first_failures = sum(
        report["summary"]["first_attempt_failure"]["events"] for report in reports
    )
    throughputs = [float(report["summary"]["output_tokens_per_s"]) for report in reports]
    request_rates = [float(report["summary"]["requests_per_s"]) for report in reports]
    return {
        "trials": len(reports),
        "trial_ids": sorted(report["config"]["trial"] for report in reports),
        "requests": total,
        "request_failure": rate(failures, total),
        "first_attempt_failure": rate(first_failures, total),
        "output_tokens_per_s": describe(throughputs),
        "successful_requests_per_s": describe(request_rates),
        "per_trial": [
            {
                "trial": report["config"]["trial"],
                "requests_total": report["summary"]["requests_total"],
                "requests_failed": report["summary"]["requests_failed"],
                "output_tokens_per_s": report["summary"]["output_tokens_per_s"],
                "requests_per_s": report["summary"]["requests_per_s"],
                "latency_s": report["summary"]["latency_s"],
            }
            for report in sorted(reports, key=lambda item: item["config"]["trial"])
        ],
    }


def analyze(
    reports: list[dict[str, Any]], baseline: str, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for report in reports:
        config = report["config"]
        key = cell_key(config)
        condition = config["condition"]
        trial = config["trial"]
        if trial in grouped[key][condition]:
            raise ValueError(f"duplicate cell/condition/trial: {key!r} {condition!r} {trial}")
        grouped[key][condition][trial] = report

    cells = []
    for index, (key, conditions) in enumerate(sorted(grouped.items(), key=lambda item: repr(item[0]))):
        if baseline not in conditions:
            raise ValueError(f"cell {key!r} has no baseline condition {baseline!r}")
        condition_summaries = {
            name: summarize_condition(list(by_trial.values()))
            for name, by_trial in sorted(conditions.items())
        }
        paired: dict[str, Any] = {}
        baseline_trials = conditions[baseline]
        for condition, trial_reports in sorted(conditions.items()):
            if condition == baseline:
                continue
            common = sorted(set(baseline_trials) & set(trial_reports))
            if not common:
                raise ValueError(f"cell {key!r}: {condition!r} has no paired baseline trials")
            slowdown = []
            overhead = []
            for trial in common:
                base = float(baseline_trials[trial]["summary"]["output_tokens_per_s"])
                active = float(trial_reports[trial]["summary"]["output_tokens_per_s"])
                if base > 0.0 and active > 0.0:
                    slowdown.append(base / active)
                    overhead.append(100.0 * (base - active) / base)
            paired[condition] = {
                "baseline": baseline,
                "paired_trials": common,
                "slowdown_ratio_baseline_over_condition": describe(slowdown),
                "slowdown_mean_bootstrap_95": bootstrap_mean_ci(
                    slowdown, bootstrap_samples, seed + index * 1009 + len(paired)
                ),
                "throughput_overhead_percent": describe(overhead),
                "overhead_mean_bootstrap_95": bootstrap_mean_ci(
                    overhead, bootstrap_samples, seed + index * 1013 + len(paired)
                ),
            }
        cells.append(
            {
                "load": {
                    "model": key[0],
                    "requests_per_trial": key[1],
                    "max_tokens": key[2],
                    "temperature": key[3],
                    "concurrency": key[4],
                },
                "conditions": condition_summaries,
                "paired_overhead": paired,
            }
        )
    return {
        "schema_version": 1,
        "method": {
            "failure_interval": "two-sided Wilson 95%",
            "paired_bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "baseline_condition": baseline,
        },
        "input_reports": len(reports),
        "cells": cells,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--baseline-condition", default="off")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-request-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.bootstrap_samples <= 100_000:
        print("error: --bootstrap-samples must be in [0,100000]", file=sys.stderr)
        return 2
    try:
        reports = [load_report(path) for path in args.reports]
        result = analyze(reports, args.baseline_condition, args.bootstrap_samples, args.seed)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(rendered)
    failures = sum(
        condition["request_failure"]["events"]
        for cell in result["cells"]
        for condition in cell["conditions"].values()
    )
    return 1 if args.fail_on_request_error and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
