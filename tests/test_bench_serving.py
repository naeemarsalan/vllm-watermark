"""Focused tests for the audit-safe serving benchmark."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_serving.py"
_SPEC = importlib.util.spec_from_file_location("bench_serving_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bench = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bench
_SPEC.loader.exec_module(bench)


class _Response:
    def __init__(self, status_code: int, payload: object, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> object:
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload


def test_wilson_interval_keeps_zero_event_uncertainty() -> None:
    interval = bench.wilson_interval(0, 3_000)
    assert interval is not None
    assert interval[0] == pytest.approx(0.0)
    assert 0.001 < interval[1] < 0.002
    assert bench.wilson_interval(0, 0) is None


def test_run_one_hashes_prompt_and_never_retains_response_body(monkeypatch) -> None:
    marker = "DO-NOT-RETAIN-THIS-RESPONSE-BODY"
    monkeypatch.setattr(
        bench,
        "_post_with_retry",
        lambda *_args, **_kwargs: (_Response(200, ValueError("bad"), marker), None, 1),
    )
    result = bench.run_one(
        0,
        "sensitive prompt marker",
        "http://example.invalid/v1/completions",
        {},
        "model",
        4,
        0.7,
        None,
        1.0,
    )
    assert result.ok is False
    assert result.prompt_sha256 == "7566d021060ae2c0c9a5bf18e0208afc61189f251b2c97493434d12519134661"
    assert result.first_attempt_failed is True
    assert marker not in repr(result)
    assert "sensitive prompt" not in repr(result)


@pytest.mark.parametrize("payload", [{"choices": [{}]}, {"choices": [{"text": 7}]}])
def test_run_one_rejects_missing_or_non_string_completion_text(monkeypatch, payload) -> None:
    monkeypatch.setattr(
        bench,
        "_post_with_retry",
        lambda *_args, **_kwargs: (_Response(200, payload), None, 1),
    )
    result = bench.run_one(0, "prompt", "https://example.invalid/v1/completions", {}, "model", 4, 0.7, None, 1.0)
    assert result.ok is False
    assert result.error.startswith("malformed_response_")


def test_redact_url_discards_credentials_query_and_path_without_raising() -> None:
    safe = bench.redact_url("https://user:secret@example.invalid/private/API?token=secret#fragment")
    assert safe == "https://example.invalid"
    assert "secret" not in safe
    assert bench.redact_url("not a URL") == "<redacted-url>"


@pytest.mark.parametrize("payload", [None, {"choices": None}, {"choices": [{}], "usage": []}, {"choices": [{}], "usage": {"completion_tokens": -1}}])
def test_run_one_classifies_malformed_success_payloads(monkeypatch, payload) -> None:
    monkeypatch.setattr(bench, "_post_with_retry", lambda *_a, **_k: (_Response(200, payload), None, 1))
    result = bench.run_one(0, "prompt", "https://user:secret@example.test/v1/completions?token=secret", {}, "m", 1, 0.0, None, 1.0)
    assert not result.ok
    assert result.error.startswith("malformed_response_")


def test_redact_url_removes_credentials_and_query() -> None:
    assert bench.redact_url("https://user:secret@example.test/v1?token=secret#x") == "https://example.test/v1"


def test_main_reports_rates_and_uses_bounded_submission(tmp_path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one\ntwo\n", encoding="utf-8")
    output = tmp_path / "result.json"

    def fake_run_one(index, prompt, *_args, **_kwargs):
        failed = index in {3, 17}
        return bench.RequestResult(
            index=index,
            prompt_sha256=f"digest-{prompt}",
            ok=not failed,
            status_code=503 if failed else 200,
            latency_s=0.01 + index / 10_000,
            completion_tokens=None if failed else 8,
            error="http_503" if failed else None,
            first_attempt_failed=failed,
        )

    monkeypatch.setattr(bench, "run_one", fake_run_one)
    rc = bench.main(
        [
            "--model", "model",
            "--prompts-file", str(prompts),
            "--n", "25",
            "--concurrency", "4",
            "--submission-window", "4",
            "--condition", "kgw-on",
            "--trial", "2",
            "--out", str(output),
            "--fail-on-request-error",
        ]
    )
    assert rc == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["config"]["submission_window"] == 4
    assert report["config"]["condition"] == "kgw-on"
    assert report["config"]["trial"] == 2
    assert report["summary"]["request_failure"]["events"] == 2
    assert report["summary"]["request_failure"]["total"] == 25
    assert report["summary"]["first_attempt_failure"]["events"] == 2
    assert report["summary"]["latency_s"]["p99_9"] is not None
    assert len(report["requests"]) == 25


@pytest.mark.parametrize(
    "args",
    [
        ["--n", "0"],
        ["--concurrency", "0"],
        ["--temperature", "nan"],
        ["--timeout", "inf"],
        ["--concurrency", "4", "--submission-window", "3"],
        ["--condition", "unsafe/value"],
        ["--trial", "-1"],
    ],
)
def test_main_rejects_invalid_load_controls(tmp_path, args) -> None:
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("one\n", encoding="utf-8")
    base = ["--model", "model", "--prompts-file", str(prompts)]
    assert bench.main(base + args) == 2
