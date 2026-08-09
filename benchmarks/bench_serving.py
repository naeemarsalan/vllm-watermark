#!/usr/bin/env python3
"""Serving benchmark against an OpenAI-compatible vLLM endpoint.

Fires --n requests at POST {OPENAI_BASE_URL}/completions (default
OPENAI_BASE_URL: http://vllm:8000/v1) using a thread pool for
concurrency, and reports throughput/latency. Designed to run inside a
lightweight bench pod (python:3.12-slim + `pip install requests`) --
stdlib + requests only, no torch/transformers.

Prompts are read one-per-line from --prompts-file and cycled (modulo) up
to --n requests -- the same prompt can be sent more than once if --n
exceeds the number of lines in the file; that's intentional for a
throughput benchmark (unlike gen_corpus.py, which varies repeats because
it's building a text corpus, not measuring serving speed).

Usage:
    python3 benchmarks/bench_serving.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompts-file benchmarks/prompts.txt \\
        --n 100 --max-tokens 256 --temperature 0.7 --concurrency 4 \\
        --out results.json

    # Watermarked run, via vllm_xargs:
    python3 benchmarks/bench_serving.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompts-file benchmarks/prompts.txt \\
        --extra-body '{"vllm_xargs": {"watermark": "on"}}' \\
        --out results_watermarked.json

Env:
    OPENAI_BASE_URL   base URL including the /v1 suffix
                       (default: http://vllm:8000/v1)
    OPENAI_API_KEY    optional bearer token (default: "EMPTY", the
                       conventional vLLM placeholder -- vLLM's own OpenAI
                       server does not require a real key by default)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = "http://vllm:8000/v1"
DEFAULT_TIMEOUT_S = 120.0
MAX_REQUESTS = 100_000
MAX_DETAILS = 10_000
MAX_PROMPTS = 10_000
MAX_PROMPT_CHARS = 65_536
MAX_PROMPT_BYTES = 16 * 1024 * 1024


def load_prompts(path: str) -> list[str]:
    lines: list[str] = []
    total_bytes = 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            encoded_bytes = len(line.encode("utf-8"))
            if len(line) > MAX_PROMPT_CHARS or len(lines) >= MAX_PROMPTS or total_bytes + encoded_bytes > MAX_PROMPT_BYTES:
                raise ValueError("prompt file exceeds bounded prompt count/size")
            lines.append(line)
            total_bytes += encoded_bytes
    if not lines:
        raise ValueError(f"no prompts found in {path}")
    return lines


def cycle_prompts(prompts: list[str], n: int) -> list[str]:
    return [prompts[i % len(prompts)] for i in range(n)]


def redact_url(url: str) -> str:
    """Return only a safe endpoint label; never preserve URL credentials/query/path."""
    try:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            return "<redacted-url>"
        host = parts.hostname
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
        # Keep only the conventional API prefix for diagnostics; arbitrary
        # paths may contain tenant IDs or credentials.
        path = parts.path if parts.path in {"", "/v1"} else ""
        return f"{parts.scheme}://{host}{port}{path}"
    except (TypeError, ValueError):
        return "<redacted-url>"


def percentile(sorted_values: list[float], p: float) -> "float | None":
    """Linear-interpolation percentile (matches numpy's default 'linear'
    method), robust for any sample size >= 1 -- unlike
    statistics.quantiles(), which needs at least 2 data points and a
    chosen bucket count that doesn't map cleanly onto small samples."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] * (hi - k) + sorted_values[hi] * (k - lo)


@dataclass
class RequestResult:
    index: int
    prompt_sha256: str
    ok: bool
    status_code: "int | None" = None
    latency_s: "float | None" = None
    completion_tokens: "int | None" = None
    prompt_tokens: "int | None" = None
    finish_reason: "str | None" = None
    error: "str | None" = None
    retried: bool = False
    attempts: int = 1
    first_attempt_failed: bool = False


def _post_with_retry(
    url: str, headers: dict, body: dict, timeout: float
) -> "tuple[requests.Response | None, str | None, int]":
    """POST once; on a connection-level failure (not an HTTP error status),
    retry exactly once. Returns (response_or_None, error_str_or_None,
    attempts). A non-200 HTTP response is NOT retried here -- it is
    returned to the caller, which records only a bounded error category
    (never the response body; see run_one)."""
    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            return resp, None, attempt + 1
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Keep the exception class only. Exception messages can contain
            # URLs, headers, or echoed request data and are unnecessary for
            # aggregate reliability accounting.
            last_err = type(exc).__name__
    return None, last_err, 2


def run_one(
    index: int,
    prompt: str,
    url: str,
    headers: dict,
    model: str,
    max_tokens: int,
    temperature: float,
    extra_body: "dict | None",
    timeout: float,
) -> RequestResult:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra_body:
        body.update(extra_body)

    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    start = time.perf_counter()
    resp, err, attempts = _post_with_retry(url, headers, body, timeout)
    latency = time.perf_counter() - start
    retried = attempts > 1

    if resp is None:
        return RequestResult(
            index=index,
            prompt_sha256=prompt_sha256,
            ok=False,
            error=err,
            retried=retried,
            attempts=attempts,
            first_attempt_failed=retried,
        )

    if resp.status_code != 200:
        return RequestResult(
            index=index,
            prompt_sha256=prompt_sha256,
            ok=False,
            status_code=resp.status_code,
            latency_s=latency,
            error=f"http_{resp.status_code}",
            retried=retried,
            attempts=attempts,
            first_attempt_failed=True,
        )

    try:
        data = resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise TypeError("choices must be a non-empty list")
        choice = data["choices"][0]
        usage = data.get("usage", {})
        if not isinstance(choice, dict) or not isinstance(choice.get("text"), str) or not isinstance(usage, dict):
            raise TypeError("choice/usage must be objects")
        for field in ("completion_tokens", "prompt_tokens"):
            value = usage.get(field)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise TypeError(f"{field} must be a non-negative integer")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        return RequestResult(
            index=index,
            prompt_sha256=prompt_sha256,
            ok=False,
            status_code=resp.status_code,
            latency_s=latency,
            error=f"malformed_response_{type(exc).__name__}",
            retried=retried,
            attempts=attempts,
            first_attempt_failed=True,
        )

    return RequestResult(
        index=index,
        prompt_sha256=prompt_sha256,
        ok=True,
        status_code=resp.status_code,
        latency_s=latency,
        completion_tokens=(usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int)
                           and not isinstance(usage.get("completion_tokens"), bool)
                           and usage.get("completion_tokens") >= 0 else None),
        prompt_tokens=(usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int)
                       and not isinstance(usage.get("prompt_tokens"), bool)
                       and usage.get("prompt_tokens") >= 0 else None),
        finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
        retried=retried,
        attempts=attempts,
        first_attempt_failed=retried,
    )


def wilson_interval(events: int, total: int, z: float = 1.959963984540054) -> "list[float] | None":
    """Two-sided Wilson score interval for a binomial proportion.

    The numerator and denominator remain present beside this interval in
    every result. A zero observed event rate is therefore never presented
    as proof that the population rate is exactly zero.
    """
    if total <= 0:
        return None
    proportion = events / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * total)) / total)
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def rate_summary(events: int, total: int) -> dict[str, "int | float | list[float] | None"]:
    return {
        "events": events,
        "total": total,
        "rate": (events / total) if total else None,
        "wilson_95": wilson_interval(events, total),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model id as registered with the server")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--n", type=int, default=100, help="total number of requests to send")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--extra-body",
        default=None,
        help='JSON object string merged into every request body, e.g. '
        '\'{"vllm_xargs": {"watermark": "on"}}\'',
    )
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per-request HTTP timeout (s)")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--submission-window",
        type=int,
        default=0,
        help="maximum queued+running requests (default: 4 * concurrency)",
    )
    parser.add_argument(
        "--omit-request-details",
        action="store_true",
        help="omit per-request rows from JSON (aggregate metrics remain complete)",
    )
    parser.add_argument(
        "--fail-on-request-error",
        action="store_true",
        help="return exit status 1 if any logical request fails after retry",
    )
    parser.add_argument("--condition", default=None, help="bounded benchmark condition label")
    parser.add_argument("--trial", type=int, default=None, help="repetition index for paired analysis")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    if args.n <= 0:
        print("error: --n must be positive", file=sys.stderr)
        return 2
    if args.n > MAX_REQUESTS:
        print(f"error: --n must be <= {MAX_REQUESTS}", file=sys.stderr)
        return 2
    if args.max_tokens <= 0:
        print("error: --max-tokens must be positive", file=sys.stderr)
        return 2
    if args.max_tokens > 4096 or args.concurrency > 256:
        print("error: --max-tokens <= 4096 and --concurrency <= 256", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print("error: --concurrency must be positive", file=sys.stderr)
        return 2
    if not math.isfinite(args.temperature) or args.temperature < 0:
        print("error: --temperature must be finite and non-negative", file=sys.stderr)
        return 2
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        print("error: --timeout must be finite and positive", file=sys.stderr)
        return 2
    submission_window = args.submission_window or (4 * args.concurrency)
    if submission_window < args.concurrency:
        print("error: --submission-window must be >= --concurrency", file=sys.stderr)
        return 2
    if submission_window > 1024:
        print("error: --submission-window must be <= 1024", file=sys.stderr)
        return 2
    if not args.omit_request_details and args.n > MAX_DETAILS:
        print(f"error: omit request details for --n > {MAX_DETAILS}", file=sys.stderr)
        return 2
    if args.condition is not None and not args.condition.replace("-", "").replace("_", "").isalnum():
        print("error: --condition must contain only letters, digits, '-' or '_'", file=sys.stderr)
        return 2
    if args.trial is not None and args.trial < 0:
        print("error: --trial must be non-negative", file=sys.stderr)
        return 2

    extra_body = None
    if args.extra_body:
        try:
            extra_body = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            print(f"error: --extra-body is not valid JSON: {exc}", file=sys.stderr)
            return 2
        if not isinstance(extra_body, dict):
            print("error: --extra-body must decode to a JSON object", file=sys.stderr)
            return 2

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/completions"

    try:
        prompts = load_prompts(args.prompts_file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"Serving benchmark: {args.n} requests -> {redact_url(url)}")
    print(
        f"model={args.model} max_tokens={args.max_tokens} temperature={args.temperature} "
        f"concurrency={args.concurrency} extra_body_keys={sorted(extra_body) if extra_body else []}"
    )

    results: list["RequestResult | None"] = [None] * args.n
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures: dict[concurrent.futures.Future[RequestResult], int] = {}
        next_index = 0

        def submit_one(index: int) -> None:
            future = pool.submit(
                run_one,
                index,
                prompts[index % len(prompts)],
                url,
                headers,
                args.model,
                args.max_tokens,
                args.temperature,
                extra_body,
                args.timeout,
            )
            futures[future] = index

        while next_index < args.n and len(futures) < submission_window:
            submit_one(next_index)
            next_index += 1

        done = 0
        while futures:
            completed, _ = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in completed:
                futures.pop(future)
                result = future.result()
                results[result.index] = result
                done += 1
                if next_index < args.n:
                    submit_one(next_index)
                    next_index += 1
                if done % 10 == 0 or done == args.n:
                    print(f"  {done}/{args.n} done", file=sys.stderr)
    wall_time_s = time.perf_counter() - wall_start

    all_results: list[RequestResult] = [r for r in results if r is not None]
    ok_results = [r for r in all_results if r.ok]
    failed_results = [r for r in all_results if not r.ok]
    first_attempt_failures = sum(r.first_attempt_failed for r in all_results)
    recovered_after_retry = sum(r.ok and r.first_attempt_failed for r in all_results)
    total_attempts = sum(r.attempts for r in all_results)

    latencies = sorted(r.latency_s for r in ok_results if r.latency_s is not None)
    completion_tokens_list = [r.completion_tokens for r in ok_results if r.completion_tokens is not None]
    total_completion_tokens = sum(completion_tokens_list)

    summary = {
        "requests_total": len(all_results),
        "requests_ok": len(ok_results),
        "requests_failed": len(failed_results),
        "request_failure": rate_summary(len(failed_results), len(all_results)),
        "first_attempt_failure": rate_summary(first_attempt_failures, len(all_results)),
        "recovered_after_retry": rate_summary(recovered_after_retry, len(all_results)),
        "http_attempts_total": total_attempts,
        "wall_time_s": wall_time_s,
        "requests_per_s": (len(ok_results) / wall_time_s) if wall_time_s > 0 else None,
        "total_completion_tokens": total_completion_tokens,
        "output_tokens_per_s": (total_completion_tokens / wall_time_s) if wall_time_s > 0 else None,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "p99_9": percentile(latencies, 99.9),
            "min": latencies[0] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
    }

    output = {
        "schema_version": 2,
        "config": {
            "base_url": redact_url(base_url),
            "url": redact_url(url),
            "model": args.model,
            "prompts_file": args.prompts_file,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "extra_body_keys": sorted(extra_body) if extra_body else [],
            "timeout_s": args.timeout,
            "submission_window": submission_window,
            "condition": args.condition,
            "trial": args.trial,
        },
        "summary": summary,
        "requests": [] if args.omit_request_details else [dataclasses.asdict(r) for r in all_results],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print("=== Summary ===")
    print(
        f"requests: {summary['requests_ok']} ok / {summary['requests_failed']} failed "
        f"/ {summary['requests_total']} total"
    )
    print(f"wall time: {wall_time_s:.2f}s")
    if summary["output_tokens_per_s"] is not None:
        print(f"total completion tokens: {total_completion_tokens}")
        print(f"aggregate output tokens/sec: {summary['output_tokens_per_s']:.2f}")
        print(f"successful requests/sec: {summary['requests_per_s']:.2f}")
    lat = summary["latency_s"]
    if lat["p50"] is not None:
        print(
            f"latency (s): mean={lat['mean']:.3f} p50={lat['p50']:.3f} "
            f"p95={lat['p95']:.3f} p99={lat['p99']:.3f} p99.9={lat['p99_9']:.3f} "
            f"min={lat['min']:.3f} max={lat['max']:.3f}"
        )
    failure = summary["request_failure"]
    first_failure = summary["first_attempt_failure"]
    print(
        "logical request failure: "
        f"{failure['events']}/{failure['total']} rate={failure['rate']:.6f} "
        f"wilson95={failure['wilson_95']}"
    )
    print(
        "first-attempt failure: "
        f"{first_failure['events']}/{first_failure['total']} rate={first_failure['rate']:.6f} "
        f"wilson95={first_failure['wilson_95']} total_http_attempts={total_attempts}"
    )
    if failed_results:
        print(f"failures: {len(failed_results)} (see {args.out} for per-request error detail)")
        for r in failed_results[:5]:
            print(f"  [{r.index}] status={r.status_code} error={r.error!r}"[:300])
    print(f"wrote {args.out}")

    return 1 if args.fail_on_request_error and failed_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
