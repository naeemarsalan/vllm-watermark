"""Focused tests for the deterministic numeric fuzz/profile harness."""

from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "benchmarks" / "fuzz_watermark.py"
_SPEC = importlib.util.spec_from_file_location("fuzz_watermark_under_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
fuzz = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = fuzz
_SPEC.loader.exec_module(fuzz)

_apply_profile_defaults = fuzz._apply_profile_defaults
_json_bytes = fuzz._json_bytes
build_parser = fuzz.build_parser
latency_summary_ms = fuzz.latency_summary_ms
percentile = fuzz.percentile
wilson_interval = fuzz.wilson_interval


def test_percentiles_and_latency_summary_are_defined_for_small_samples() -> None:
    assert percentile([4.0], 99.0) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    summary = latency_summary_ms([1.0, 2.0, 3.0, 4.0])
    assert summary == {
        "count": 4,
        "p50_ms": 2.5,
        "p95_ms": pytest.approx(3.85),
        "p99_ms": pytest.approx(3.97),
        "max_ms": 4.0,
    }


def test_wilson_failure_interval_handles_zero_and_balanced_failures() -> None:
    assert wilson_interval(0, 0) is None
    zero_low, zero_high = wilson_interval(0, 100)
    assert zero_low == 0.0
    assert 0.03 < zero_high < 0.04
    balanced_low, balanced_high = wilson_interval(50, 100)
    assert balanced_low == pytest.approx(1.0 - balanced_high)
    assert balanced_low < 0.5 < balanced_high
    with pytest.raises(ValueError):
        wilson_interval(2, 1)


def test_parser_profile_defaults_and_strict_json_encoding() -> None:
    args = build_parser().parse_args([])
    _apply_profile_defaults(args)
    assert len(args.kgw_profile_vocab) >= 2
    assert len(args.synthid_process_profile) >= 2
    assert len(args.synthid_detect_profile) >= 2
    assert json.loads(_json_bytes({"finite": 1.0})) == {"finite": 1.0}
    with pytest.raises(ValueError):
        _json_bytes({"not_finite": float("nan")})


def test_cli_smoke_is_deterministic_numeric_json() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(_REPO_ROOT / "src")
    command = [
        sys.executable,
        str(_SCRIPT),
        "--seed",
        "7",
        "--kgw-equivalence-cases",
        "1",
        "--kgw-invariant-cases",
        "1",
        "--synthid-equivalence-cases",
        "1",
        "--detector-cases",
        "2",
        "--profile-iterations",
        "1",
        "--profile-warmup",
        "0",
        "--kgw-profile-vocab",
        "16",
        "--synthid-process-profile",
        "16:2",
        "--synthid-detect-profile",
        "16:5:2",
    ]
    completed = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "passed"
    assert report["determinism"]["seed"] == 7
    assert report["aggregate"]["total_cases"] == 5
    assert report["aggregate"]["failures"] == 0
    assert report["content_in_report"] is False
    assert report["secrets_in_report"] is False
    assert len(report["profiles"]["measurements"]) == 3
