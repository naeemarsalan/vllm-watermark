"""Bound and protocol regressions for the isolated detector stress harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).parents[1] / "benchmarks" / "stress_detection.py"
spec = importlib.util.spec_from_file_location("stress_detection_under_test", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_matrix_values_and_expansion_guard() -> None:
    parser = module._build_parser()
    args = parser.parse_args(["--lengths", "1,1"])
    with pytest.raises(SystemExit):
        module._validated_matrix_args(args, parser)


def test_result_scored_tokens_cannot_exceed_input_bound() -> None:
    result = SimpleNamespace(
        num_tokens_scored=4,
        num_green=2,
        z_score=0.0,
        p_value=0.5,
        prediction=False,
    )
    coherent, scored = module._validate_result(result, "kgw", None, max_scored=3)
    assert not coherent
    assert scored == 4


def test_allocation_caps_reject_extreme_length() -> None:
    parser = module._build_parser()
    args = parser.parse_args(["--lengths", "65537"])
    with pytest.raises(SystemExit):
        module._validated_matrix_args(args, parser)


def _event_stream(*, version=module.PROTOCOL_VERSION, order=None):
    events = [
        {"event": "worker_started", "protocol_version": version},
        {"event": "attempt_started", "protocol_version": version, "attempt": 0},
        {
            "event": "attempt_result", "protocol_version": version, "attempt": 0,
            "category": "success", "error_type": None, "input_tokens": 4,
            "latency_ns": 1, "scored_tokens": 3,
        },
        {"event": "worker_summary", "protocol_version": version,
         "peak_rss_bytes": 1, "unexpected_outcomes": 0},
    ]
    return [events[index] for index in (order or range(len(events)))]


def test_protocol_rejects_unknown_wrong_version_and_order() -> None:
    assert module._protocol_errors(_event_stream(), 1, False) == 0
    assert module._protocol_errors(_event_stream(version=99), 1, False) > 0
    unknown = _event_stream()
    unknown.insert(1, {"event": "noise", "protocol_version": module.PROTOCOL_VERSION})
    assert module._protocol_errors(unknown, 1, False) > 0
    assert module._protocol_errors(_event_stream(order=[0, 2, 1, 3]), 1, False) > 0


def test_protocol_rejects_missing_and_duplicate_attempt_events() -> None:
    stream = _event_stream()
    assert module._protocol_errors(stream[:1] + stream[1:2] + stream[3:], 1, False) > 0
    duplicate = _event_stream()
    duplicate.insert(2, duplicate[1].copy())
    assert module._protocol_errors(duplicate, 1, False) > 0


def test_protocol_validation_does_not_convert_malformed_numeric_fields() -> None:
    stream = _event_stream()
    stream[2]["input_tokens"] = "not-an-int"
    # Structural validation must classify the event; it must not raise while
    # handling hostile worker JSON.
    assert module._protocol_errors(stream, 1, False) > 0


def test_protocol_validation_rejects_unhashable_attempt_ids_without_raising() -> None:
    stream = _event_stream()
    stream[1]["attempt"] = [0]
    stream[2]["attempt"] = {"value": 0}
    assert module._protocol_errors(stream, 1, False) > 0
