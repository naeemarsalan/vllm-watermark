#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deterministic core-algorithm fuzzing and CPU micro-profiling.

This harness never generates, decodes, accepts, or writes text.  Inputs are
bounded numeric configurations and token IDs.  The JSON report deliberately
omits watermark keys, token IDs, exception messages, hostnames, and environment
variables; a failing case is reproducible from the public seed, campaign name,
and case index.

The equivalence oracles are the installed Apache-2.0 ``transformers`` KGW and
SynthID-Text logits processors.  KGW equivalence cases are restricted to
``hashing_key * previous_token < 2**64 - 1``.  In that domain, this repository's
``% 2**64`` seed rule and transformers' ``% (2**64 - 1)`` rule are both no-ops,
so exact permutation equality is meaningful.  Separate KGW invariant cases
exercise the repository's full unsigned-64-bit key domain without claiming
transformers equivalence where the seed rules intentionally differ.

Example (fast smoke):

    PYTHONPATH=src python3 benchmarks/fuzz_watermark.py \
      --kgw-equivalence-cases 10 --kgw-invariant-cases 10 \
      --synthid-equivalence-cases 10 --detector-cases 10 \
      --profile-iterations 2 --profile-warmup 1

Example (larger deterministic run):

    PYTHONPATH=src python3 benchmarks/fuzz_watermark.py \
      --seed 20260809 --kgw-equivalence-cases 10000 \
      --kgw-invariant-cases 10000 --synthid-equivalence-cases 10000 \
      --detector-cases 10000 --profile-iterations 100

Exit status is non-zero if any property case fails or an unexpected exception
occurs.  Expected detector errors for deliberately too-short token sequences
are counted separately and do not count as failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import transformers
from transformers import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vllm_watermark.kgw.core import KGWConfig, greenlist_ids  # noqa: E402
from vllm_watermark.kgw.detector import score_token_ids as kgw_score_token_ids  # noqa: E402
from vllm_watermark.synthid.core import SynthIDConfig, g_values, process_scores_row  # noqa: E402
from vllm_watermark.synthid.detector import (  # noqa: E402
    score_token_ids_mean,
    score_token_ids_weighted_mean,
)


_U64_MODULUS = 1 << 64
_TRANSFORMERS_SEED_MODULUS = _U64_MODULUS - 1
_WILSON_95_Z = 1.959963984540054
_DEFAULT_SEED = 20260809
_CAMPAIGN_SEED_OFFSETS = {
    "kgw_transformers_equivalence": 0x0A11CE,
    "kgw_u64_invariants": 0x0B0A7D,
    "synthid_transformers_equivalence": 0x051D,
    "detector_outputs": 0x0DE7EC7,
    "profiles": 0x0F10F11E,
}

_DEFAULT_KGW_PROFILE_VOCABS = (1_000, 8_192, 50_257)
_DEFAULT_SYNTHID_PROCESS_PROFILES = ((256, 4), (1_000, 8), (4_096, 16))
_DEFAULT_SYNTHID_DETECT_PROFILES = (
    (256, 64, 4),
    (1_000, 256, 8),
    (4_096, 512, 16),
)


class PropertyFailure(AssertionError):
    """A property failed, identified only by a fixed non-content code."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PropertyFailure(code)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, defined for one or more values."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def latency_summary_ms(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "p50_ms": percentile(values, 50.0),
        "p95_ms": percentile(values, 95.0),
        "p99_ms": percentile(values, 99.0),
        "max_ms": max(values),
    }


def wilson_interval(successes: int, total: int, z: float = _WILSON_95_Z) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("counts must be integers")
    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("counts must be integers")
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("counts must satisfy 0 <= successes <= total")
    if total == 0:
        return None
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, centre - half_width)
    high = 1.0 if successes == total else min(1.0, centre + half_width)
    return (low, high)


def peak_rss_bytes() -> int:
    """Return the process high-water RSS using only the standard library."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB.  macOS and the BSDs report bytes.
    return value if sys.platform == "darwin" else value * 1024


class LatencyReservoir:
    """Bounded deterministic reservoir for case-latency percentiles."""

    def __init__(self, limit: int, seed: int) -> None:
        if limit < 1:
            raise ValueError("latency sample limit must be positive")
        self.limit = limit
        self.seen = 0
        self.values: list[float] = []
        self._rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.limit:
            self.values.append(value)
            return
        index = self._rng.randrange(self.seen)
        if index < self.limit:
            self.values[index] = value


@dataclass
class Campaign:
    name: str
    configured_cases: int
    latency_sample_limit: int
    latency_seed: int
    max_recorded_failures: int
    total_cases: int = 0
    failures: int = 0
    expected_errors: Counter[str] = field(default_factory=Counter)
    failure_kinds: Counter[str] = field(default_factory=Counter)
    failure_examples: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    _latencies: LatencyReservoir = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._latencies = LatencyReservoir(self.latency_sample_limit, self.latency_seed)

    def execute(
        self,
        case_index: int,
        case: Callable[[], str | None],
        metadata: dict[str, int | float | str],
    ) -> None:
        started = time.perf_counter()
        failure_kind: str | None = None
        try:
            expected_error = case()
            if expected_error is not None:
                self.expected_errors[expected_error] += 1
        except PropertyFailure as exc:
            failure_kind = exc.code
        except Exception as exc:  # noqa: BLE001 - fuzz harness must account for every exception.
            failure_kind = f"unexpected_{type(exc).__name__}"
        finally:
            latency_ms = (time.perf_counter() - started) * 1_000.0
            self._latencies.add(latency_ms)
            self.total_cases += 1

        if failure_kind is not None:
            self.failures += 1
            self.failure_kinds[failure_kind] += 1
            if len(self.failure_examples) < self.max_recorded_failures:
                self.failure_examples.append(
                    {
                        "case_index": case_index,
                        "kind": failure_kind,
                        "metadata": metadata,
                    }
                )

    def report(self) -> dict[str, Any]:
        interval = wilson_interval(self.failures, self.total_cases)
        return {
            "name": self.name,
            "configured_cases": self.configured_cases,
            "total_cases": self.total_cases,
            "failures": self.failures,
            "failure_rate": self.failures / self.total_cases if self.total_cases else None,
            "failure_rate_wilson_95": interval,
            "expected_errors": dict(sorted(self.expected_errors.items())),
            "failure_kinds": dict(sorted(self.failure_kinds.items())),
            "failure_examples": self.failure_examples,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_cases_per_second": (
                self.total_cases / self.elapsed_seconds if self.elapsed_seconds > 0.0 else 0.0
            ),
            "case_latency": {
                **latency_summary_ms(self._latencies.values),
                "observations": self._latencies.seen,
                "sampled": len(self._latencies.values),
                "sampling": (
                    "all" if self._latencies.seen <= self._latencies.limit else "reservoir"
                ),
            },
            "peak_rss_bytes_after": peak_rss_bytes(),
        }


def _campaign_rng(seed: int, name: str) -> random.Random:
    return random.Random((seed + _CAMPAIGN_SEED_OFFSETS[name]) % _U64_MODULUS)


def _run_kgw_equivalence(
    count: int,
    seed: int,
    latency_sample_limit: int,
    max_recorded_failures: int,
) -> Campaign:
    name = "kgw_transformers_equivalence"
    rng = _campaign_rng(seed, name)
    campaign = Campaign(
        name,
        count,
        latency_sample_limit,
        seed ^ _CAMPAIGN_SEED_OFFSETS[name],
        max_recorded_failures,
    )
    vocab_choices = (8, 17, 64, 257, 1_000, 2_048)
    gamma_choices = (0.125, 0.25, 0.5, 0.75)
    started = time.perf_counter()
    for case_index in range(count):
        vocab_size = rng.choice(vocab_choices)
        gamma = rng.choice(gamma_choices)
        hashing_key = rng.randrange(1, 1 << 31)
        previous_token = rng.randrange(vocab_size)

        def case(
            vocab_size: int = vocab_size,
            gamma: float = gamma,
            hashing_key: int = hashing_key,
            previous_token: int = previous_token,
        ) -> None:
            require(
                hashing_key * previous_token < _TRANSFORMERS_SEED_MODULUS,
                "kgw_equivalence_domain_violation",
            )
            config = KGWConfig(
                vocab_size=vocab_size,
                hash_key=hashing_key,
                gamma=gamma,
            )
            reference = WatermarkLogitsProcessor(
                vocab_size=vocab_size,
                device="cpu",
                greenlist_ratio=gamma,
                bias=2.0,
                hashing_key=hashing_key,
                seeding_scheme="lefthash",
                context_width=1,
            )
            ours = greenlist_ids(previous_token, config)
            theirs = reference._get_greenlist_ids(  # noqa: SLF001 - explicit oracle API.
                torch.tensor([previous_token], dtype=torch.int64)
            )
            require(torch.equal(ours, theirs.cpu()), "kgw_exact_permutation_mismatch")
            require(ours.numel() == int(vocab_size * gamma), "kgw_equivalent_size_mismatch")
            return None

        campaign.execute(
            case_index,
            case,
            {
                "vocab_size": vocab_size,
                "gamma": gamma,
                "seed_product_bits": (hashing_key * previous_token).bit_length(),
            },
        )
    campaign.elapsed_seconds = time.perf_counter() - started
    return campaign


def _run_kgw_u64_invariants(
    count: int,
    seed: int,
    latency_sample_limit: int,
    max_recorded_failures: int,
) -> Campaign:
    name = "kgw_u64_invariants"
    rng = _campaign_rng(seed, name)
    campaign = Campaign(
        name,
        count,
        latency_sample_limit,
        seed ^ _CAMPAIGN_SEED_OFFSETS[name],
        max_recorded_failures,
    )
    boundary_keys = (0, 1, (1 << 63) - 1, 1 << 63, _U64_MODULUS - 1)
    # vocab_size=1 cannot form a non-empty green list while gamma remains
    # strictly below 1, so the smallest valid boundary is 2.
    vocab_choices = (2, 7, 31, 257, 1_024, 4_096)
    gamma_choices = (0.001, 0.125, 0.25, 0.5, 0.999)
    started = time.perf_counter()
    for case_index in range(count):
        vocab_size = rng.choice(vocab_choices)
        gamma = rng.choice(gamma_choices)
        if int(vocab_size * gamma) < 1:
            # Keep this campaign inside KGWConfig's valid domain while
            # still exercising every unsigned-64-bit hash-key boundary.
            gamma = 0.5
        hashing_key = (
            boundary_keys[case_index]
            if case_index < len(boundary_keys)
            else rng.getrandbits(64)
        )
        previous_token = rng.randrange(vocab_size)

        def case(
            vocab_size: int = vocab_size,
            gamma: float = gamma,
            hashing_key: int = hashing_key,
            previous_token: int = previous_token,
        ) -> None:
            config = KGWConfig(
                vocab_size=vocab_size,
                hash_key=hashing_key,
                gamma=gamma,
            )
            first = greenlist_ids(previous_token, config)
            second = greenlist_ids(previous_token, config)
            expected_size = int(vocab_size * gamma)
            require(torch.equal(first, second), "kgw_u64_nondeterministic")
            require(first.device.type == "cpu", "kgw_u64_non_cpu_output")
            require(first.dtype == torch.int64, "kgw_u64_wrong_dtype")
            require(first.dim() == 1, "kgw_u64_wrong_rank")
            require(first.numel() == expected_size, "kgw_u64_wrong_size")
            require(first.unique().numel() == expected_size, "kgw_u64_duplicate_ids")
            if expected_size:
                require(int(first.min()) >= 0, "kgw_u64_negative_id")
                require(int(first.max()) < vocab_size, "kgw_u64_out_of_range_id")
            return None

        campaign.execute(
            case_index,
            case,
            {
                "vocab_size": vocab_size,
                "gamma": gamma,
                "hash_key_bits": hashing_key.bit_length(),
            },
        )
    campaign.elapsed_seconds = time.perf_counter() - started
    return campaign


def _run_synthid_equivalence(
    count: int,
    seed: int,
    latency_sample_limit: int,
    max_recorded_failures: int,
) -> Campaign:
    name = "synthid_transformers_equivalence"
    rng = _campaign_rng(seed, name)
    campaign = Campaign(
        name,
        count,
        latency_sample_limit,
        seed ^ _CAMPAIGN_SEED_OFFSETS[name],
        max_recorded_failures,
    )
    vocab_choices = (2, 7, 31, 128, 257, 512)
    table_size_choices = (16, 32, 64, 128, 256, 512, 1_024)
    started = time.perf_counter()
    for case_index in range(count):
        vocab_size = rng.choice(vocab_choices)
        ngram_len = rng.randint(1, 6)
        depth = rng.randint(1, 8)
        table_size = rng.choice(table_size_choices)
        table_seed = rng.randrange(0, 1 << 31)
        keys = tuple(rng.randrange(0, 1 << 31) for _ in range(depth))
        context = [rng.randrange(vocab_size) for _ in range(ngram_len - 1)]
        candidate_count = rng.randint(1, min(16, vocab_size + 4))
        candidates = [rng.randrange(vocab_size) for _ in range(candidate_count)]

        def case(
            vocab_size: int = vocab_size,
            ngram_len: int = ngram_len,
            depth: int = depth,
            table_size: int = table_size,
            table_seed: int = table_seed,
            keys: tuple[int, ...] = keys,
            context: list[int] = context,
            candidates: list[int] = candidates,
        ) -> None:
            config = SynthIDConfig(
                vocab_size=vocab_size,
                keys=keys,
                ngram_len=ngram_len,
                sampling_table_size=table_size,
                sampling_table_seed=table_seed,
                context_history_size=0,
            )
            candidate_tensor = torch.tensor(candidates, dtype=torch.int64)
            ours = g_values(context, candidate_tensor, config)
            repeat = g_values(context, candidate_tensor, config)
            reference = SynthIDTextWatermarkLogitsProcessor(
                ngram_len=ngram_len,
                keys=list(keys),
                sampling_table_size=table_size,
                sampling_table_seed=table_seed,
                context_history_size=0,
                device=torch.device("cpu"),
            )
            reference_inputs = torch.tensor(
                [context + [candidate] for candidate in candidates],
                dtype=torch.int64,
            )
            theirs = reference.compute_g_values(reference_inputs)[:, 0, :]
            require(torch.equal(ours, theirs.cpu()), "synthid_exact_g_mismatch")
            require(torch.equal(ours, repeat), "synthid_g_nondeterministic")
            require(tuple(ours.shape) == (len(candidates), depth), "synthid_g_wrong_shape")
            require(ours.dtype == torch.int64, "synthid_g_wrong_dtype")
            require(ours.device.type == "cpu", "synthid_g_non_cpu_output")
            require(bool(torch.all((ours == 0) | (ours == 1))), "synthid_g_non_binary")
            return None

        campaign.execute(
            case_index,
            case,
            {
                "vocab_size": vocab_size,
                "ngram_len": ngram_len,
                "depth": depth,
                "sampling_table_size": table_size,
                "candidate_count": candidate_count,
            },
        )
    campaign.elapsed_seconds = time.perf_counter() - started
    return campaign


def _assert_common_detector_result(result: Any) -> None:
    require(math.isfinite(float(result.z_score)), "detector_nonfinite_z")
    require(math.isfinite(float(result.p_value)), "detector_nonfinite_p")
    require(0.0 <= float(result.p_value) <= 1.0, "detector_p_out_of_range")
    require(isinstance(result.prediction, bool), "detector_prediction_not_bool")


def _run_detector_outputs(
    count: int,
    seed: int,
    latency_sample_limit: int,
    max_recorded_failures: int,
) -> Campaign:
    name = "detector_outputs"
    rng = _campaign_rng(seed, name)
    campaign = Campaign(
        name,
        count,
        latency_sample_limit,
        seed ^ _CAMPAIGN_SEED_OFFSETS[name],
        max_recorded_failures,
    )
    vocab_choices = (8, 31, 128, 257)
    started = time.perf_counter()
    for case_index in range(count):
        scheme = "kgw" if case_index % 2 == 0 else "synthid"
        deliberate_short = case_index % 5 == 0
        vocab_size = rng.choice(vocab_choices)
        sequence_kind = rng.choice(("random", "repeated", "alternating"))
        if scheme == "kgw":
            gamma = rng.choice((0.125, 0.25, 0.5, 0.75))
            config: KGWConfig | SynthIDConfig = KGWConfig(
                vocab_size=vocab_size,
                hash_key=rng.getrandbits(64),
                gamma=gamma,
            )
            length = rng.randint(0, 1) if deliberate_short else rng.randint(2, 128)
            depth = 0
            ngram_len = 2
        else:
            ngram_len = rng.randint(1, 6)
            depth = rng.randint(1, 8)
            keys = tuple(rng.randrange(0, 1 << 31) for _ in range(depth))
            config = SynthIDConfig(
                vocab_size=vocab_size,
                keys=keys,
                ngram_len=ngram_len,
                sampling_table_size=rng.choice((32, 64, 128, 256)),
                sampling_table_seed=rng.randrange(0, 1 << 31),
                context_history_size=rng.choice((0, 1, 4, 16)),
            )
            length = (
                rng.randrange(ngram_len)
                if deliberate_short
                else rng.randint(ngram_len, max(ngram_len, 128))
            )

        if sequence_kind == "random":
            token_ids = [rng.randrange(vocab_size) for _ in range(length)]
        elif sequence_kind == "repeated":
            value = rng.randrange(vocab_size)
            token_ids = [value] * length
        else:
            left = rng.randrange(vocab_size)
            right = rng.randrange(vocab_size)
            token_ids = [left if index % 2 == 0 else right for index in range(length)]

        def case(
            scheme: str = scheme,
            deliberate_short: bool = deliberate_short,
            config: KGWConfig | SynthIDConfig = config,
            token_ids: list[int] = token_ids,
        ) -> str | None:
            if deliberate_short:
                try:
                    if scheme == "kgw":
                        kgw_score_token_ids(token_ids, config)  # type: ignore[arg-type]
                    else:
                        score_token_ids_mean(token_ids, config)  # type: ignore[arg-type]
                except ValueError:
                    return f"{scheme}_short_input"
                raise PropertyFailure("detector_short_input_was_accepted")

            if scheme == "kgw":
                result = kgw_score_token_ids(
                    token_ids,
                    config,  # type: ignore[arg-type]
                    ignore_repeated_ngrams=case_index % 4 == 0,
                )
                _assert_common_detector_result(result)
                expected_upper = len(token_ids) - 1
                require(1 <= result.num_tokens_scored <= expected_upper, "kgw_detector_bad_count")
                require(0 <= result.num_green <= result.num_tokens_scored, "kgw_detector_bad_green_count")
            else:
                scorer = (
                    score_token_ids_weighted_mean
                    if case_index % 4 == 1
                    else score_token_ids_mean
                )
                result = scorer(token_ids, config)  # type: ignore[arg-type]
                _assert_common_detector_result(result)
                expected_upper = len(token_ids) - config.ngram_len + 1  # type: ignore[union-attr]
                require(1 <= result.num_scored <= expected_upper, "synthid_detector_bad_count")
                require(result.depth == config.depth, "synthid_detector_bad_depth")  # type: ignore[union-attr]
                require(math.isfinite(float(result.mean_g)), "synthid_detector_nonfinite_mean")
                require(math.isfinite(float(result.score)), "synthid_detector_nonfinite_score")
                require(0.0 <= float(result.mean_g) <= 1.0, "synthid_detector_mean_out_of_range")
                require(0.0 <= float(result.score) <= 1.0, "synthid_detector_score_out_of_range")
            return None

        campaign.execute(
            case_index,
            case,
            {
                "scheme": scheme,
                "vocab_size": vocab_size,
                "sequence_length": length,
                "sequence_kind": sequence_kind,
                "deliberate_short": int(deliberate_short),
                "ngram_len": ngram_len,
                "depth": depth,
            },
        )
    campaign.elapsed_seconds = time.perf_counter() - started
    return campaign


def profile_callable(
    operation: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    logical_units_per_operation: int | None = None,
) -> dict[str, Any]:
    """Profile a synchronous callable, retaining only safe aggregate timing."""
    if warmup < 0 or iterations < 1:
        raise ValueError("profile counts must satisfy warmup >= 0 and iterations >= 1")
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        timings_ms: list[float] = []
        wall_started = time.perf_counter()
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            timings_ms.append((time.perf_counter() - started) * 1_000.0)
        elapsed_seconds = time.perf_counter() - wall_started
    result: dict[str, Any] = {
        "iterations": iterations,
        "warmup_iterations": warmup,
        "elapsed_seconds": elapsed_seconds,
        "throughput_operations_per_second": (
            iterations / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "latency": latency_summary_ms(timings_ms),
        "peak_rss_bytes_after": peak_rss_bytes(),
    }
    if logical_units_per_operation is not None:
        result["logical_units_per_operation"] = logical_units_per_operation
        result["logical_units_per_second"] = (
            iterations * logical_units_per_operation / elapsed_seconds
            if elapsed_seconds > 0.0
            else 0.0
        )
    return result


def _run_profiles(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = _campaign_rng(args.seed, "profiles")
    reports: list[dict[str, Any]] = []

    for vocab_size in args.kgw_profile_vocab:
        config = KGWConfig(
            vocab_size=vocab_size,
            hash_key=rng.getrandbits(64),
            gamma=0.25,
        )
        previous_tokens = [rng.randrange(vocab_size) for _ in range(args.profile_iterations + args.profile_warmup)]
        cursor = 0

        def operation() -> torch.Tensor:
            nonlocal cursor
            result = greenlist_ids(previous_tokens[cursor % len(previous_tokens)], config)
            cursor += 1
            return result

        reports.append(
            {
                "operation": "kgw_greenlist_ids",
                "vocab_size": vocab_size,
                "gamma": config.gamma,
                **profile_callable(
                    operation,
                    warmup=args.profile_warmup,
                    iterations=args.profile_iterations,
                    logical_units_per_operation=vocab_size,
                ),
            }
        )

    for vocab_size, depth in args.synthid_process_profile:
        ngram_len = 5
        config = SynthIDConfig(
            vocab_size=vocab_size,
            keys=tuple(rng.randrange(0, 1 << 31) for _ in range(depth)),
            ngram_len=ngram_len,
            sampling_table_size=1 << 12,
            sampling_table_seed=rng.randrange(0, 1 << 31),
            context_history_size=0,
        )
        torch_generator = torch.Generator().manual_seed(rng.randrange(0, 1 << 63))
        scores = torch.randn(vocab_size, generator=torch_generator, dtype=torch.float32)
        context = [rng.randrange(vocab_size) for _ in range(ngram_len - 1)]

        def operation() -> torch.Tensor:
            return process_scores_row(scores, context, config, context_seen=False)

        reports.append(
            {
                "operation": "synthid_process_scores_row",
                "vocab_size": vocab_size,
                "depth": depth,
                "ngram_len": ngram_len,
                **profile_callable(
                    operation,
                    warmup=args.profile_warmup,
                    iterations=args.profile_iterations,
                    logical_units_per_operation=vocab_size,
                ),
            }
        )

    for vocab_size, sequence_length, depth in args.synthid_detect_profile:
        ngram_len = min(5, sequence_length)
        config = SynthIDConfig(
            vocab_size=vocab_size,
            keys=tuple(rng.randrange(0, 1 << 31) for _ in range(depth)),
            ngram_len=ngram_len,
            sampling_table_size=1 << 12,
            sampling_table_seed=rng.randrange(0, 1 << 31),
            context_history_size=0,
        )
        token_ids = [rng.randrange(vocab_size) for _ in range(sequence_length)]
        scored_positions = sequence_length - ngram_len + 1

        def operation() -> Any:
            return score_token_ids_weighted_mean(token_ids, config)

        reports.append(
            {
                "operation": "synthid_weighted_mean_detection",
                "vocab_size": vocab_size,
                "sequence_length": sequence_length,
                "scored_positions": scored_positions,
                "depth": depth,
                "ngram_len": ngram_len,
                **profile_callable(
                    operation,
                    warmup=args.profile_warmup,
                    iterations=args.profile_iterations,
                    logical_units_per_operation=scored_positions,
                ),
            }
        )
    return reports


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _kgw_profile_vocab(value: str) -> int:
    return _positive_int(value)


def _synthid_process_profile(value: str) -> tuple[int, int]:
    try:
        vocab, depth = value.split(":", 1)
        return (_positive_int(vocab), _positive_int(depth))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError("expected VOCAB_SIZE:DEPTH with positive integers") from exc


def _synthid_detect_profile(value: str) -> tuple[int, int, int]:
    try:
        vocab, length, depth = value.split(":", 2)
        return (_positive_int(vocab), _positive_int(length), _positive_int(depth))
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "expected VOCAB_SIZE:SEQUENCE_LENGTH:DEPTH with positive integers"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--kgw-equivalence-cases", type=_nonnegative_int, default=200)
    parser.add_argument("--kgw-invariant-cases", type=_nonnegative_int, default=500)
    parser.add_argument("--synthid-equivalence-cases", type=_nonnegative_int, default=200)
    parser.add_argument("--detector-cases", type=_nonnegative_int, default=500)
    parser.add_argument("--profile-iterations", type=_positive_int, default=20)
    parser.add_argument("--profile-warmup", type=_nonnegative_int, default=3)
    parser.add_argument("--latency-sample-limit", type=_positive_int, default=100_000)
    parser.add_argument("--max-recorded-failures", type=_nonnegative_int, default=20)
    parser.add_argument("--torch-threads", type=_positive_int, default=1)
    parser.add_argument(
        "--kgw-profile-vocab",
        action="append",
        type=_kgw_profile_vocab,
        default=None,
        metavar="VOCAB_SIZE",
    )
    parser.add_argument(
        "--synthid-process-profile",
        action="append",
        type=_synthid_process_profile,
        default=None,
        metavar="VOCAB_SIZE:DEPTH",
    )
    parser.add_argument(
        "--synthid-detect-profile",
        action="append",
        type=_synthid_detect_profile,
        default=None,
        metavar="VOCAB_SIZE:SEQUENCE_LENGTH:DEPTH",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    return parser


def _apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.kgw_profile_vocab is None:
        args.kgw_profile_vocab = list(_DEFAULT_KGW_PROFILE_VOCABS)
    if args.synthid_process_profile is None:
        args.synthid_process_profile = list(_DEFAULT_SYNTHID_PROCESS_PROFILES)
    if args.synthid_detect_profile is None:
        args.synthid_detect_profile = list(_DEFAULT_SYNTHID_DETECT_PROFILES)


def _aggregate_campaigns(campaigns: Iterable[Campaign], elapsed_seconds: float) -> dict[str, Any]:
    materialized = list(campaigns)
    total_cases = sum(campaign.total_cases for campaign in materialized)
    failures = sum(campaign.failures for campaign in materialized)
    expected_errors: Counter[str] = Counter()
    sampled_latencies: list[float] = []
    observations = 0
    for campaign in materialized:
        expected_errors.update(campaign.expected_errors)
        sampled_latencies.extend(campaign._latencies.values)  # noqa: SLF001 - same-module aggregate.
        observations += campaign._latencies.seen  # noqa: SLF001 - same-module aggregate.
    interval = wilson_interval(failures, total_cases)
    return {
        "total_cases": total_cases,
        "failures": failures,
        "failure_rate": failures / total_cases if total_cases else None,
        "failure_rate_wilson_95": interval,
        "expected_errors": dict(sorted(expected_errors.items())),
        "elapsed_seconds": elapsed_seconds,
        "throughput_cases_per_second": total_cases / elapsed_seconds if elapsed_seconds > 0.0 else 0.0,
        "latency_note": "omitted at aggregate level; campaign reservoirs are not count-weighted",
    }


def _json_bytes(report: dict[str, Any]) -> bytes:
    """Strict JSON encoding: NaN/Infinity are rejected, never serialized."""
    return (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bounded = {
        "kgw_equivalence_cases": (args.kgw_equivalence_cases, 100_000),
        "kgw_invariant_cases": (args.kgw_invariant_cases, 100_000),
        "synthid_equivalence_cases": (args.synthid_equivalence_cases, 100_000),
        "detector_cases": (args.detector_cases, 100_000),
        "profile_iterations": (args.profile_iterations, 10_000),
        "profile_warmup": (args.profile_warmup, 1_000),
        "latency_sample_limit": (args.latency_sample_limit, 100_000),
    }
    if any(value > limit for value, limit in bounded.values()):
        print("error: fuzz allocation knobs exceed safety caps", file=sys.stderr)
        return 2
    if any(v > (1 << 20) for v in args.kgw_profile_vocab):
        print("error: profile vocab exceeds deployment cap", file=sys.stderr)
        return 2
    if any(v > (1 << 20) or d > 256 for v, d in args.synthid_process_profile):
        print("error: SynthID profile exceeds deployment caps", file=sys.stderr)
        return 2
    if any(v > (1 << 20) or length > 65_536 or d > 256 for v, length, d in args.synthid_detect_profile):
        print("error: SynthID detection profile exceeds deployment caps", file=sys.stderr)
        return 2
    _apply_profile_defaults(args)
    torch.set_num_threads(args.torch_threads)

    wall_started = time.perf_counter()
    campaigns = [
        _run_kgw_equivalence(
            args.kgw_equivalence_cases,
            args.seed,
            args.latency_sample_limit,
            args.max_recorded_failures,
        ),
        _run_kgw_u64_invariants(
            args.kgw_invariant_cases,
            args.seed,
            args.latency_sample_limit,
            args.max_recorded_failures,
        ),
        _run_synthid_equivalence(
            args.synthid_equivalence_cases,
            args.seed,
            args.latency_sample_limit,
            args.max_recorded_failures,
        ),
        _run_detector_outputs(
            args.detector_cases,
            args.seed,
            args.latency_sample_limit,
            args.max_recorded_failures,
        ),
    ]
    fuzz_elapsed_seconds = time.perf_counter() - wall_started

    profile_started = time.perf_counter()
    profiles = _run_profiles(args)
    profile_elapsed_seconds = time.perf_counter() - profile_started
    wall_elapsed_seconds = time.perf_counter() - wall_started
    aggregate = _aggregate_campaigns(campaigns, fuzz_elapsed_seconds)
    failures = int(aggregate["failures"])

    report = {
        "schema_version": 1,
        "status": "passed" if failures == 0 else "failed",
        "content_in_report": False,
        "secrets_in_report": False,
        "determinism": {
            "seed": args.seed,
            "campaign_substreams": "fixed_numeric_offsets",
            "torch_threads": args.torch_threads,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": sys.platform,
            "device": "cpu",
        },
        "configuration": {
            "kgw_equivalence_cases": args.kgw_equivalence_cases,
            "kgw_invariant_cases": args.kgw_invariant_cases,
            "synthid_equivalence_cases": args.synthid_equivalence_cases,
            "detector_cases": args.detector_cases,
            "profile_iterations": args.profile_iterations,
            "profile_warmup": args.profile_warmup,
            "latency_sample_limit": args.latency_sample_limit,
            "kgw_profile_vocabs": args.kgw_profile_vocab,
            "synthid_process_profiles": args.synthid_process_profile,
            "synthid_detect_profiles": args.synthid_detect_profile,
        },
        "aggregate": aggregate,
        "campaigns": [campaign.report() for campaign in campaigns],
        "profiles": {
            "elapsed_seconds": profile_elapsed_seconds,
            "measurements": profiles,
        },
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    encoded = _json_bytes(report)
    if args.output is not None:
        args.output.write_bytes(encoded)
    os.write(sys.stdout.fileno(), encoded)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
