#!/usr/bin/env python3
"""Generate a labeled watermarked/unwatermarked text corpus via an
OpenAI-compatible vLLM /v1/completions endpoint.

Runs sequentially (no concurrency arg -- this is a corpus-building tool,
not a throughput benchmark; see bench_serving.py for that) and writes one
JSON row per successful completion to --out as it goes, so a partial run
still leaves a usable, valid JSONL file. stdlib + requests only -- no
torch/transformers -- so this can run inside a lightweight bench pod.

Per CLAUDE.md task spec, the request body's `vllm_xargs` is set as:
    --watermark on   -> {"watermark": "on", "watermark_key_id": KEY}
    --watermark off  -> {"watermark": "off"}
KEY is --key-id if given, else "default" (matching vllm_watermark.keys'
own fallback key_id name) -- watermark_key_id is always present for the
"on" case, never for "off". This is a client-side default only:
vllm_watermark.kgw.processor's own contract treats `watermark_key_id` as
optional and resolves a request-side omission the same way (WATERMARK_KEY_ID
env or "default") -- see src/vllm_watermark/kgw/processor.py module
docstring, "Per-request vllm_xargs / SamplingParams.extra_args keys this
processor recognizes". Sending it explicitly here just makes every corpus
row self-documenting about which key_id it was generated with.

Optional --scheme {kgw,synthid} adds a `watermark_scheme` key to
`vllm_xargs` (per src/vllm_watermark/request_args.py's
KNOWN_WATERMARK_XARGS / SCHEME-COORDINATION DESIGN) so the corpus is
labeled with which loaded processor should bias it. When --scheme is
omitted, `vllm_xargs` is built EXACTLY as before (no `watermark_scheme`
key at all) -- byte-identical to pre-existing behavior, letting the
server's VLLM_WATERMARK_SCHEME default apply. `--scheme` is only sent
with --watermark on (a `watermark_scheme` value is meaningless -- and
would be silently ignored server-side -- when watermark is off), mirroring
--key-id's existing on-only behavior. The resulting row also records
`"scheme"` (the --scheme value, or null if omitted) for downstream
comparison tooling (see benchmarks/compare_schemes.py).

Usage:
    python3 benchmarks/gen_corpus.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompts-file benchmarks/prompts.txt \\
        --n 120 --max-tokens 256 --temperature 0.7 \\
        --watermark on --key-id default \\
        --out benchmarks/data/corpus_watermarked.jsonl

    python3 benchmarks/gen_corpus.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --prompts-file benchmarks/prompts.txt \\
        --n 120 --watermark off \\
        --out benchmarks/data/corpus_unwatermarked.jsonl

Env:
    OPENAI_BASE_URL   base URL including the /v1 suffix
                       (default: http://vllm:8000/v1)
    OPENAI_API_KEY    optional bearer token (default: "EMPTY")
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

DEFAULT_BASE_URL = "http://vllm:8000/v1"
DEFAULT_TIMEOUT_S = 120.0


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        raise ValueError(f"no prompts found in {path}")
    return lines


def build_prompts(prompts: list[str], n: int) -> list[str]:
    """Cycle `prompts` (modulo) up to n items. Per CLAUDE.md task spec:
    "Cycles prompts if n > #prompts (append ' (variation N)' to repeated
    prompts to vary)." N is the 1-indexed cycle/pass number; the first
    pass through the prompt list is left unmodified so byte-identical
    duplicates only ever occur if a caller repeats a full pass with an
    identical prompt list AND the server happens to sample identically at
    temperature 0 -- at temperature 0.7 (the corpus default) even
    unmodified repeats normally diverge in text."""
    out = []
    for i in range(n):
        base = prompts[i % len(prompts)]
        pass_num = i // len(prompts) + 1
        out.append(base if pass_num == 1 else f"{base} (variation {pass_num})")
    return out


def request_completion(
    url: str, headers: dict, body: dict, timeout: float
) -> "tuple[requests.Response | None, str | None]":
    """POST once; retry exactly once on a connection-level failure (not on
    an HTTP error status -- those are returned to the caller as-is)."""
    last_err = None
    for _attempt in range(2):
        try:
            return requests.post(url, headers=headers, json=body, timeout=timeout), None
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--watermark", choices=["on", "off"], required=True)
    parser.add_argument(
        "--key-id",
        default=None,
        help="watermark_key_id to request (default: 'default'); only sent with --watermark on",
    )
    parser.add_argument(
        "--scheme",
        choices=["kgw", "synthid"],
        default=None,
        help=(
            "watermark_scheme to request; only sent with --watermark on. "
            "Omitted by default -- vllm_xargs then carries no watermark_scheme "
            "key at all, preserving prior byte-identical behavior."
        ),
    )
    parser.add_argument("--out", default="corpus.jsonl")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)

    if args.key_id and args.watermark == "off":
        print(
            "warning: --key-id given with --watermark off; it will not be sent "
            "(a key only applies when watermark is on)",
            file=sys.stderr,
        )

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/completions"

    try:
        prompts = load_prompts(args.prompts_file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    work_prompts = build_prompts(prompts, args.n)

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    # Per CLAUDE.md task spec: the "on" case's vllm_xargs body always
    # carries watermark_key_id (not only when --key-id was explicitly
    # passed). "default" mirrors vllm_watermark.keys' own fallback key_id
    # name (WATERMARK_KEY_ID env or "default" -- see src/vllm_watermark/
    # keys.py), so an omitted --key-id still resolves to a real,
    # reproducibly-labeled key on the server side.
    if args.scheme and args.watermark == "off":
        print(
            "warning: --scheme given with --watermark off; it will not be sent "
            "(a scheme only applies when watermark is on)",
            file=sys.stderr,
        )

    effective_key_id = args.key_id or "default"
    vllm_xargs: dict = {"watermark": args.watermark}
    if args.watermark == "on":
        vllm_xargs["watermark_key_id"] = effective_key_id
        if args.scheme:
            vllm_xargs["watermark_scheme"] = args.scheme

    print(
        f"Generating corpus: {len(work_prompts)} completions -> {url} "
        f"(watermark={args.watermark} key_id={effective_key_id if args.watermark == 'on' else None} "
        f"scheme={args.scheme if args.watermark == 'on' else None})"
    )

    n_ok = 0
    n_failed = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, prompt in enumerate(work_prompts):
            body = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "vllm_xargs": vllm_xargs,
            }
            start = time.perf_counter()
            resp, err = request_completion(url, headers, body, args.timeout)
            request_ms = (time.perf_counter() - start) * 1000.0

            if resp is None:
                n_failed += 1
                print(f"  [{i + 1}/{len(work_prompts)}] FAILED (connection): {err}", file=sys.stderr)
                continue
            if resp.status_code != 200:
                n_failed += 1
                print(
                    f"  [{i + 1}/{len(work_prompts)}] FAILED (HTTP {resp.status_code}): {resp.text[:500]}",
                    file=sys.stderr,
                )
                continue
            try:
                data = resp.json()
                choice = data["choices"][0]
                text = choice["text"]
                finish_reason = choice.get("finish_reason")
                completion_tokens = (data.get("usage") or {}).get("completion_tokens")
            except (ValueError, KeyError, IndexError) as exc:
                n_failed += 1
                print(f"  [{i + 1}/{len(work_prompts)}] FAILED (malformed body): {exc}", file=sys.stderr)
                continue

            row = {
                "prompt": prompt,
                "text": text,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "watermark": args.watermark,
                "key_id": effective_key_id if args.watermark == "on" else None,
                "scheme": args.scheme if args.watermark == "on" else None,
                "model": args.model,
                "temperature": args.temperature,
                "request_ms": request_ms,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n_ok += 1
            if (i + 1) % 10 == 0 or (i + 1) == len(work_prompts):
                print(f"  [{i + 1}/{len(work_prompts)}] ok={n_ok} failed={n_failed}", file=sys.stderr)

    print(f"done: {n_ok} ok, {n_failed} failed -> {args.out}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
