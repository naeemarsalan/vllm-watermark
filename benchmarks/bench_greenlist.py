"""Micro-benchmark: per-call latency of kgw.core.greenlist_ids on CPU.

This bounds the per-token overhead the KGW logits-processor approach adds
at detection time (and, structurally, at generation time -- the
processor's hot path is the same torch.randperm call).

Run: PYTHONPATH=src /usr/bin/python3 benchmarks/bench_greenlist.py
"""

from __future__ import annotations

import statistics
import time

import torch

from vllm_watermark.kgw.core import KGWConfig, greenlist_ids

VOCAB_SIZE = 151936  # Qwen/Qwen2.5-0.5B-Instruct vocab_size (Task A target model)
N_CALLS = 1000


def main() -> None:
    cfg = KGWConfig(vocab_size=VOCAB_SIZE, hash_key=15485863, gamma=0.25)
    torch.manual_seed(0)
    prev_tokens = torch.randint(0, VOCAB_SIZE, (N_CALLS,)).tolist()

    # Warm-up (first call sometimes pays one-off allocator/cache costs).
    for t in prev_tokens[:10]:
        greenlist_ids(t, cfg)

    per_call_ms = []
    for t in prev_tokens:
        start = time.perf_counter()
        greenlist_ids(t, cfg)
        per_call_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = sum(per_call_ms)
    print(f"vocab_size={VOCAB_SIZE} n_calls={N_CALLS}")
    print(f"total: {total_ms:.2f} ms")
    print(f"mean:  {statistics.mean(per_call_ms):.4f} ms/call")
    print(f"median:{statistics.median(per_call_ms):.4f} ms/call")
    print(f"p95:   {statistics.quantiles(per_call_ms, n=100)[94]:.4f} ms/call")
    print(f"min:   {min(per_call_ms):.4f} ms/call")
    print(f"max:   {max(per_call_ms):.4f} ms/call")


if __name__ == "__main__":
    main()
