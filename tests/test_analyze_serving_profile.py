"""Tests for repeated, paired serving-profile analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "analyze_serving_profile.py"
_SPEC = importlib.util.spec_from_file_location("analyze_serving_profile_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
profile = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = profile
_SPEC.loader.exec_module(profile)


def report(condition: str, trial: int, throughput: float, failures: int = 0) -> dict:
    total = 1_000
    return {
        "schema_version": 2,
        "config": {
            "condition": condition,
            "trial": trial,
            "model": "model",
            "n": total,
            "max_tokens": 64,
            "temperature": 0.7,
            "concurrency": 4,
        },
        "summary": {
            "requests_total": total,
            "requests_ok": total - failures,
            "requests_failed": failures,
            "wall_time_s": 10.0,
            "output_tokens_per_s": throughput,
            "requests_per_s": (total - failures) / 10.0,
            "first_attempt_failure": {"events": failures},
            "latency_s": {"p50": 1.0, "p95": 1.1, "p99": 1.2, "p99_9": 1.3},
        },
    }


def test_paired_overhead_and_aggregate_failure_rates() -> None:
    reports = []
    for trial, off, kgw in ((0, 1_000.0, 800.0), (1, 900.0, 720.0), (2, 1_100.0, 880.0)):
        reports.append(report("off", trial, off))
        reports.append(report("kgw", trial, kgw, failures=1 if trial == 2 else 0))
    result = profile.analyze(reports, "off", bootstrap_samples=100, seed=7)
    cell = result["cells"][0]
    assert cell["conditions"]["kgw"]["request_failure"]["events"] == 1
    assert cell["conditions"]["kgw"]["request_failure"]["total"] == 3_000
    paired = cell["paired_overhead"]["kgw"]
    assert paired["paired_trials"] == [0, 1, 2]
    assert paired["slowdown_ratio_baseline_over_condition"]["mean"] == pytest.approx(1.25)
    assert paired["throughput_overhead_percent"]["mean"] == pytest.approx(20.0)


def test_duplicate_and_missing_baseline_are_rejected() -> None:
    duplicate = report("off", 0, 1_000.0)
    with pytest.raises(ValueError, match="duplicate"):
        profile.analyze([duplicate, duplicate], "off", 10, 1)
    with pytest.raises(ValueError, match="no baseline"):
        profile.analyze([report("kgw", 0, 800.0)], "off", 10, 1)


def test_load_report_validates_reconciliation(tmp_path) -> None:
    path = tmp_path / "bad.json"
    bad = report("off", 0, 1_000.0)
    bad["summary"]["requests_ok"] = 999
    bad["summary"]["requests_failed"] = 2
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="requests_ok"):
        profile.load_report(path)


def test_zero_throughput_is_non_comparable_not_division_error() -> None:
    reports = [report("off", 0, 0.0), report("kgw", 0, 100.0)]
    result = profile.analyze(reports, "off", bootstrap_samples=10, seed=1)
    paired = result["cells"][0]["paired_overhead"]["kgw"]
    assert paired["slowdown_ratio_baseline_over_condition"]["count"] == 0
    assert paired["slowdown_mean_bootstrap_95"] is None
