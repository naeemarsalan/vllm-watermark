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
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass

import requests

DEFAULT_BASE_URL = "http://vllm:8000/v1"
DEFAULT_TIMEOUT_S = 120.0


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        raise ValueError(f"no prompts found in {path}")
    return lines


def cycle_prompts(prompts: list[str], n: int) -> list[str]:
    return [prompts[i % len(prompts)] for i in range(n)]


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
    prompt_preview: str
    ok: bool
    status_code: "int | None" = None
    latency_s: "float | None" = None
    completion_tokens: "int | None" = None
    prompt_tokens: "int | None" = None
    finish_reason: "str | None" = None
    error: "str | None" = None
    retried: bool = False


def _post_with_retry(
    url: str, headers: dict, body: dict, timeout: float
) -> "tuple[requests.Response | None, str | None, bool]":
    """POST once; on a connection-level failure (not an HTTP error status),
    retry exactly once. Returns (response_or_None, error_str_or_None,
    retried_bool). A non-200 HTTP response is NOT retried here -- it is
    returned to the caller, which records it as a failure with the body
    captured (see run_one)."""
    last_err = None
    for attempt in range(2):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            return resp, None, attempt == 1
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err, True


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

    preview = prompt[:60].replace("\n", " ")

    start = time.perf_counter()
    resp, err, retried = _post_with_retry(url, headers, body, timeout)
    latency = time.perf_counter() - start

    if resp is None:
        return RequestResult(
            index=index, prompt_preview=preview, ok=False, error=err, retried=retried
        )

    if resp.status_code != 200:
        return RequestResult(
            index=index,
            prompt_preview=preview,
            ok=False,
            status_code=resp.status_code,
            latency_s=latency,
            error=resp.text[:2000],
            retried=retried,
        )

    try:
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
    except (ValueError, KeyError, IndexError) as exc:
        return RequestResult(
            index=index,
            prompt_preview=preview,
            ok=False,
            status_code=resp.status_code,
            latency_s=latency,
            error=f"malformed response body ({exc}): {resp.text[:500]}",
            retried=retried,
        )

    return RequestResult(
        index=index,
        prompt_preview=preview,
        ok=True,
        status_code=resp.status_code,
        latency_s=latency,
        completion_tokens=usage.get("completion_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        finish_reason=choice.get("finish_reason"),
        retried=retried,
    )


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
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

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
    work_prompts = cycle_prompts(prompts, args.n)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print(f"Serving benchmark: {len(work_prompts)} requests -> {url}")
    print(
        f"model={args.model} max_tokens={args.max_tokens} temperature={args.temperature} "
        f"concurrency={args.concurrency} extra_body={extra_body}"
    )

    results: list["RequestResult | None"] = [None] * len(work_prompts)
    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                run_one,
                i,
                prompt,
                url,
                headers,
                args.model,
                args.max_tokens,
                args.temperature,
                extra_body,
                args.timeout,
            ): i
            for i, prompt in enumerate(work_prompts)
        }
        done = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results[result.index] = result
            done += 1
            if done % 10 == 0 or done == len(work_prompts):
                print(f"  {done}/{len(work_prompts)} done", file=sys.stderr)
    wall_time_s = time.perf_counter() - wall_start

    all_results: list[RequestResult] = [r for r in results if r is not None]
    ok_results = [r for r in all_results if r.ok]
    failed_results = [r for r in all_results if not r.ok]

    latencies = sorted(r.latency_s for r in ok_results if r.latency_s is not None)
    completion_tokens_list = [r.completion_tokens for r in ok_results if r.completion_tokens is not None]
    total_completion_tokens = sum(completion_tokens_list)

    summary = {
        "requests_total": len(all_results),
        "requests_ok": len(ok_results),
        "requests_failed": len(failed_results),
        "wall_time_s": wall_time_s,
        "total_completion_tokens": total_completion_tokens,
        "output_tokens_per_s": (total_completion_tokens / wall_time_s) if wall_time_s > 0 else None,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "min": latencies[0] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
    }

    output = {
        "config": {
            "base_url": base_url,
            "url": url,
            "model": args.model,
            "prompts_file": args.prompts_file,
            "n": args.n,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": args.concurrency,
            "extra_body": extra_body,
            "timeout_s": args.timeout,
        },
        "summary": summary,
        "requests": [dataclasses.asdict(r) for r in all_results],
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
    lat = summary["latency_s"]
    if lat["p50"] is not None:
        print(
            f"latency (s): mean={lat['mean']:.3f} p50={lat['p50']:.3f} "
            f"p95={lat['p95']:.3f} p99={lat['p99']:.3f} min={lat['min']:.3f} max={lat['max']:.3f}"
        )
    if failed_results:
        print(f"failures: {len(failed_results)} (see {args.out} for per-request error detail)")
        for r in failed_results[:5]:
            print(f"  [{r.index}] status={r.status_code} error={r.error!r}"[:300])
    print(f"wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
