"""Equivalence tests for the SynthID-Text port (Task A2), run against the
locally installed transformers==4.57.6 as the reference oracle for
generation-side math, plus google-deepmind/synthid-text's detector_mean.py
formulas (fetched, transliterated to torch -- this repo has no jax
dependency; see vllm_watermark.synthid.detector's module docstring for
exact citations/URLs).

Run with: /usr/bin/python3 -m pytest tests/test_synthid_equivalence.py -v
(needs PYTHONPATH to include src/, or `pip install -e .` first -- same as
tests/test_kgw_equivalence.py)

Section (a) -- statefulness note (read before touching the warm-up helper)
------------------------------------------------------------------------------
transformers' SynthIDTextWatermarkLogitsProcessor.__call__ is genuinely
stateful and has an easy-to-miss quirk: on its VERY FIRST call, `self.state`
is `None`, so `_init_state` builds `SynthIDTextWatermarkState.context` as
`torch.zeros((batch, ngram_len - 1))` and the call's `input_ids` argument is
NOT folded into that context at all (see logits_process.py `__call__`,
lines 2720-2731: the `else` branch that appends `input_ids[:, -1:]` only
runs when `self.state is not None`). Every SUBSEQUENT call appends exactly
one new token (the last one) and drops the oldest, so it takes exactly
`ngram_len` total `__call__`s (1 init + `ngram_len - 1` appends) before
`state.context` no longer contains any of those initial zeros -- verified
directly: for `ngram_len=5`, after 4 total calls `state.context` was still
`[0, r1, r2, r3]` (one zero sentinel remaining), only becoming fully real
after a 5th call.

`vllm_watermark.synthid.core.process_scores_row` deliberately does NOT
replicate this quirk -- it always takes an explicit, real `ngram_context`
argument (see core.py module docstring; a real generation-time caller is
expected to supply the actual preceding tokens, never an artificial
zero-padded context). So the fair, apples-to-apples equivalence check is:
drive the reference through exactly `ngram_len` warm-up calls with real
(non-zero) token ids, confirm by direct inspection that no sentinel zero
remains in `state.context`, and only start comparing outputs from the next
call onward. `_warm_up_reference` below does exactly this and asserts the
zero-flush post-condition; all `test_*_matches_reference` tests in this
section reuse it. We additionally disable repeated-context masking on both
sides (`context_history_size=0`) for these particular comparisons so a
same-context collision (small `vocab_size`, ngram-based hashing) can never
make the reference silently skip watermarking on one side while our
`process_scores_row(..., context_seen=False)` call unconditionally applies
it on the other -- repeated-context masking itself is checked separately,
directly against `compute_context_repetition_mask`, in
`test_repeated_context_masking_matches_reference` below (item (a)'s "on/off"
requirement).
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import statistics
import time

import pytest
import torch

from vllm_watermark.keys import WatermarkKey
from vllm_watermark.synthid.core import (
    DEFAULT_SYNTHID_DEPTH,
    SynthIDConfig,
    _accumulate_hash,
    _sampling_table,
    expected_mean_g_value,
    g_values,
    process_scores_row,
)
from vllm_watermark.synthid.detector import (
    DEFAULT_Z_THRESHOLD,
    _context_windows,
    _repeated_context_mask,
    score_token_ids_mean,
    score_token_ids_weighted_mean,
)

transformers = pytest.importorskip("transformers")
from transformers import SynthIDTextWatermarkLogitsProcessor  # noqa: E402


def _make_ref(ngram_len, keys, sampling_table_size, sampling_table_seed, context_history_size):
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=ngram_len,
        keys=list(keys),
        sampling_table_size=sampling_table_size,
        sampling_table_seed=sampling_table_seed,
        context_history_size=context_history_size,
        device="cpu",
    )


def _warm_up_reference(ref, ngram_len, vocab_size, rng) -> torch.Tensor:
    """Drive `ref` through exactly `ngram_len` calls with real (non-zero)
    token ids so `ref.state.context` no longer contains any of
    SynthIDTextWatermarkState's initial zero-sentinel values. See module
    docstring "Section (a) -- statefulness note". Tokens are drawn from
    `[1, vocab_size)` (never 0) specifically so the post-condition assert
    below is unambiguous."""
    input_ids = torch.tensor([[rng.randrange(1, vocab_size)]])
    ref(input_ids, torch.randn(1, vocab_size))  # call 1: state=None -> zero-init, ignores input_ids
    for _ in range(ngram_len - 1):  # calls 2..ngram_len: each appends one real token
        input_ids = torch.cat([input_ids, torch.tensor([[rng.randrange(1, vocab_size)]])], dim=1)
        ref(input_ids, torch.randn(1, vocab_size))
    assert 0 not in ref.state.context[0].tolist(), (
        "zero-init sentinel should be fully flushed after ngram_len calls"
    )
    return input_ids


# ---------------------------------------------------------------------------
# (a) process_scores_row / g_values equivalence vs transformers' __call__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [1000, 50257])
def test_process_scores_row_matches_reference_call(vocab_size):
    """100 random (context, scores) pairs, driving the REAL reference
    __call__ (not just its stateless internals) -- see module docstring."""
    ngram_len = 5
    keys = [11, 22, 33, 44, 55, 66, 77]
    cfg = SynthIDConfig(
        vocab_size=vocab_size,
        keys=tuple(keys),
        ngram_len=ngram_len,
        sampling_table_size=2**12,
        sampling_table_seed=3,
        context_history_size=0,  # see module docstring: masking checked separately
    )
    ref = _make_ref(ngram_len, keys, cfg.sampling_table_size, cfg.sampling_table_seed, 0)

    rng = random.Random(1000 + vocab_size)
    input_ids = _warm_up_reference(ref, ngram_len, vocab_size, rng)

    mismatches = []
    max_abs_diff = 0.0
    for trial in range(100):
        next_tok = rng.randrange(1, vocab_size)
        input_ids = torch.cat([input_ids, torch.tensor([[next_tok]])], dim=1)
        scores_row = torch.randn(vocab_size)

        out_ref = ref(input_ids, scores_row.unsqueeze(0))[0]
        context_used = ref.state.context[0].tolist()
        assert len(context_used) == ngram_len - 1

        out_ours = process_scores_row(scores_row.clone(), context_used, cfg, context_seen=False)

        diff = (out_ref - out_ours).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        if not torch.allclose(out_ref, out_ours, atol=1e-5, rtol=1e-5):
            mismatches.append((trial, diff))

    assert not mismatches, f"vocab_size={vocab_size}: {len(mismatches)} mismatches, e.g. {mismatches[:5]}"
    assert max_abs_diff < 1e-5


def test_repeated_context_masking_matches_reference():
    """item (a) 'on/off': verify our exact-tuple-equality bounded-history
    masking (_repeated_context_mask) agrees with transformers'
    compute_context_repetition_mask on a sequence with real repeats
    (a tiny vocab_size=4 / context-width=2 -- only 16 possible ngram
    contexts -- against a 10-entry history window and a 100-token sequence
    all but guarantees repeated contexts actually occur, so this is a
    meaningful equivalence check, not a vacuous one; verified below via
    `our_mask.count(False) > 0`)."""
    vocab_size = 4
    ngram_len = 3
    keys = [11, 22, 33]
    context_history_size = 10
    ref = _make_ref(ngram_len, keys, 2**8, 3, context_history_size)

    rng = random.Random(1)
    seq_len = 100
    full_seq = [rng.randrange(vocab_size) for _ in range(seq_len)]

    ref_mask = ref.compute_context_repetition_mask(torch.tensor([full_seq]))[0].bool().tolist()
    contexts = [c for c, _ in _context_windows(full_seq, ngram_len)]
    our_mask = _repeated_context_mask(contexts, context_history_size)

    assert len(ref_mask) == len(our_mask) == seq_len - (ngram_len - 1)
    mismatches = [i for i, (a, b) in enumerate(zip(ref_mask, our_mask)) if a != b]
    assert not mismatches, f"masking mismatches at positions {mismatches[:10]}"

    # Sanity: this sequence actually exercises repeats (small vocab), and no
    # context hashed to exactly 0 (the transformers zero-sentinel quirk our
    # implementation doesn't have -- see detector.py docstring "Repeated-
    # context masking deviation"), so this is a meaningful equivalence
    # check, not a vacuous one.
    assert our_mask.count(False) > 0, "expected at least one real repeated context in this simulation"
    zero_hash_contexts = sum(
        1
        for c in contexts
        if _accumulate_hash(torch.ones((), dtype=torch.int64), torch.tensor(c, dtype=torch.int64)).item() == 0
    )
    assert zero_hash_contexts == 0


# ---------------------------------------------------------------------------
# (b) g-value equivalence vs transformers' compute_g_values
# ---------------------------------------------------------------------------


def test_g_values_matches_reference_compute_g_values():
    """500 random ngram contexts (context + one candidate token, i.e. a
    full ngram_len-token window) vs transformers'
    SynthIDTextWatermarkLogitsProcessor.compute_g_values -- its stateless,
    non-vmap-batched-only-by-us-directly internal g-value computation
    (logits_process.py lines 2907-2921)."""
    vocab_size = 2000
    ngram_len = 6
    keys = [3, 5, 7, 11, 13]
    cfg = SynthIDConfig(
        vocab_size=vocab_size,
        keys=tuple(keys),
        ngram_len=ngram_len,
        sampling_table_size=2**13,
        sampling_table_seed=2,
        context_history_size=32,
    )
    ref = _make_ref(ngram_len, keys, cfg.sampling_table_size, cfg.sampling_table_seed, 32)

    rng = random.Random(7)
    mismatches = []
    for trial in range(500):
        ngram = [rng.randrange(vocab_size) for _ in range(ngram_len)]
        context, candidate = ngram[:-1], ngram[-1]

        ours = g_values(context, torch.tensor([candidate]), cfg)[0]  # (depth,)
        theirs = ref.compute_g_values(torch.tensor([ngram]))[0, 0]  # (depth,)

        if not torch.equal(ours, theirs):
            mismatches.append(trial)

    assert not mismatches, f"g-value mismatches at trials {mismatches[:10]} (of 500)"


# ---------------------------------------------------------------------------
# (c) device-independence
# ---------------------------------------------------------------------------


def test_sampling_table_has_no_device_param():
    sig = inspect.signature(_sampling_table)
    assert "device" not in sig.parameters


def test_g_values_has_no_device_param():
    sig = inspect.signature(g_values)
    assert "device" not in sig.parameters


def test_sampling_table_deterministic_and_cpu():
    t1 = _sampling_table(4096, 123)
    t2 = _sampling_table(4096, 123)
    assert torch.equal(t1, t2)
    assert t1.device.type == "cpu"
    assert t1.dtype == torch.int64
    assert set(t1.unique().tolist()) <= {0, 1}


def test_g_values_deterministic():
    cfg = SynthIDConfig(vocab_size=200, keys=(1, 2, 3), ngram_len=4, sampling_table_size=512, sampling_table_seed=9)
    g1 = g_values([5, 6, 7], torch.arange(50), cfg)
    g2 = g_values([5, 6, 7], torch.arange(50), cfg)
    assert torch.equal(g1, g2)


def test_int64_overflow_wraparound_is_c_style_two_complement():
    """Executed check backing core.py's module docstring "int64 overflow
    wraparound" claim -- accumulate_hash's LCG multiplier
    (6364136223846793005) routinely overflows signed int64 during folding;
    torch CPU int64 arithmetic must wrap the same way transformers' own
    tensor-based implementation relies on (implicitly) for correctness."""
    import ctypes

    a = torch.tensor([2**62], dtype=torch.int64) * 4
    assert a.item() == 0  # 2**64 wraps to 0

    b = torch.tensor([2**63 - 1], dtype=torch.int64) + 1
    assert b.item() == -(2**63)
    assert b.item() == ctypes.c_int64(2**63).value  # matches C two's-complement wraparound

    # torch's integer `%` is floor-mod (Python-`%`-compatible, non-negative
    # for a positive divisor), NOT C-style truncating `%` -- required for
    # `ngram_keys % sampling_table_size` (frequently negative after
    # wraparound) to always land in [0, sampling_table_size).
    neg = torch.tensor([-1], dtype=torch.int64)
    assert (neg % 65536).item() == 65535


def test_accumulate_hash_matches_composition_property():
    """accumulate_hash's own documented property:
    f(x, data[T]) = f(f(x, data[:T-1]), data[T]) -- i.e. folding a context
    then a candidate token must equal hashing the concatenated full ngram
    in one pass. This is the identity g_values() relies on internally
    (fold context once, then fold each candidate) -- verify it directly."""
    context = torch.tensor([3, 7, 9], dtype=torch.int64)
    candidate = torch.tensor([5], dtype=torch.int64)
    full = torch.tensor([3, 7, 9, 5], dtype=torch.int64)

    step1 = _accumulate_hash(torch.ones((), dtype=torch.int64), context)
    step2 = _accumulate_hash(step1, candidate)
    combined = _accumulate_hash(torch.ones((), dtype=torch.int64), full)
    assert step2.item() == combined.item()


# ---------------------------------------------------------------------------
# (e) keys.py digest-derivation tests
# ---------------------------------------------------------------------------

_DUMMY_SECRET = b"test-dummy-secret-not-production"  # obviously-dummy, never a real key


def _dummy_watermark_key(secret: bytes = _DUMMY_SECRET) -> WatermarkKey:
    digest = hashlib.sha256(secret).digest()
    return WatermarkKey(key_id="test", hash_key=int.from_bytes(digest[:8], "big"), secret_digest=digest)


def test_derive_subkeys_deterministic():
    key = _dummy_watermark_key()
    a = key.derive_subkeys(DEFAULT_SYNTHID_DEPTH, b"synthid-keys")
    b = key.derive_subkeys(DEFAULT_SYNTHID_DEPTH, b"synthid-keys")
    assert a == b
    assert len(a) == DEFAULT_SYNTHID_DEPTH


def test_derive_subkeys_distinct_per_layer():
    key = _dummy_watermark_key()
    subkeys = key.derive_subkeys(DEFAULT_SYNTHID_DEPTH, b"synthid-keys")
    assert len(set(subkeys)) == len(subkeys), "all layer subkeys must be distinct"
    assert all(0 <= k < 2**32 for k in subkeys), "subkeys must be 32-bit unsigned ints"


def test_derive_subkeys_label_namespaces_independently():
    key = _dummy_watermark_key()
    a = key.derive_subkeys(8, b"synthid-keys")
    b = key.derive_subkeys(8, b"some-other-purpose")
    assert a != b


def test_derive_subkeys_different_secret_different_keys():
    key1 = _dummy_watermark_key(b"test-dummy-secret-not-production")
    key2 = _dummy_watermark_key(b"test-dummy-secret-not-production-2")
    assert key1.derive_subkeys(8, b"synthid-keys") != key2.derive_subkeys(8, b"synthid-keys")


def test_watermark_key_repr_redacts_hash_key_and_digest():
    key = _dummy_watermark_key()
    text = repr(key)
    assert str(key) == text
    assert "<redacted>" in text
    assert str(key.hash_key) not in text
    assert key.secret_digest.hex() not in text
    assert _DUMMY_SECRET.decode() not in text


def test_derive_subkeys_rejects_bad_args():
    key = _dummy_watermark_key()
    with pytest.raises(ValueError):
        key.derive_subkeys(0, b"x")
    with pytest.raises(TypeError):
        key.derive_subkeys(4, "not-bytes")  # type: ignore[arg-type]


def test_secret_digest_is_sha256_of_secret_not_the_secret():
    key = _dummy_watermark_key()
    assert key.secret_digest == hashlib.sha256(_DUMMY_SECRET).digest()
    assert len(key.secret_digest) == 32
    assert key.secret_digest != _DUMMY_SECRET


def test_load_key_populates_secret_digest(monkeypatch):
    """End-to-end: keys.py's real env-var loading path populates
    secret_digest identically to the hand-rolled _dummy_watermark_key
    helper above."""
    from vllm_watermark.keys import load_key

    hex_secret = _DUMMY_SECRET.hex()
    monkeypatch.setenv("WATERMARK_KEY", hex_secret)
    monkeypatch.delenv("WATERMARK_KEYS", raising=False)
    key = load_key()
    assert key.secret_digest == hashlib.sha256(_DUMMY_SECRET).digest()
    assert key.derive_subkeys(4, b"x") == _dummy_watermark_key().derive_subkeys(4, b"x")


# ---------------------------------------------------------------------------
# (d) generation<->detection self-consistency
# ---------------------------------------------------------------------------
#
# Slow (~6 minutes total, executed and timed -- see the reported numbers in
# this test's own printed summary and in the task write-up): 200 watermarked
# + 200 random sequences of 256 tokens each, at toy vocab_size=1000. depth=8
# (not the production DEFAULT_SYNTHID_DEPTH=30) purely to keep this
# simulation's wall-clock reasonable for a test suite -- process_scores_row
# is called once per generated token per sequence (200*256 = 51200 calls),
# and depth barely moves the per-call cost at this vocab_size (measured
# separately: ~1.5-5 ms/call for depth in [4, 16] at vocab_size=1000, vs.
# ~350 ms/call at vocab_size=151936/depth=30 -- see this repo's micro-
# benchmark results in the task write-up). A smaller depth does not change
# the qualitative claim under test (watermarked vs. random separability);
# it only changes how many independent per-position g-values the CLT-based
# z-score in detector.py effectively averages over.
#
# Assertions here check STRUCTURAL separation with a margin (e.g.
# `min(watermarked_z) > max(random_z) + margin`), not exact reproduced
# numbers: torch's multi-threaded reduction (`.sum()`) does not guarantee a
# bit-identical accumulation order run-to-run even for a fixed
# `torch.Generator` seed, so exact float equality across runs would be
# flaky. The margin used below (a full z-score point) is tiny relative to
# the actual observed gap between distributions (watermarked z clustered
# ~18-26; random z clustered ~-2..+4 -- a ~15+ standard-deviation gap, see
# this test's printed summary), so it is not a weakened check in practice.


def _dummy_synthid_cfg(vocab_size: int, depth: int, ngram_len: int = 5) -> SynthIDConfig:
    key = _dummy_watermark_key(b"synthid-self-consistency-dummy-secret")
    keys = key.derive_subkeys(depth, b"synthid-keys")
    return SynthIDConfig(
        vocab_size=vocab_size,
        keys=keys,
        ngram_len=ngram_len,
        sampling_table_size=2**14,
        sampling_table_seed=0,
        context_history_size=64,
    )


def _generate_watermarked_sequence(cfg: SynthIDConfig, seed: int, gen_len: int, temperature: float) -> "list[int]":
    width = cfg.ngram_len - 1
    gen = torch.Generator().manual_seed(seed)
    tokens = [int(t) for t in torch.randint(1, cfg.vocab_size, (width,), generator=gen)]
    for _ in range(gen_len):
        context = tokens[-width:]
        scores_row = torch.randn(cfg.vocab_size, generator=gen) / temperature
        log_probs = process_scores_row(scores_row, context, cfg, context_seen=False)
        probs = torch.exp(log_probs)
        next_tok = int(torch.multinomial(probs, 1, generator=gen).item())
        tokens.append(next_tok)
    return tokens


def _generate_random_sequence(vocab_size: int, length: int, seed: int) -> "list[int]":
    gen = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(1, vocab_size, (length,), generator=gen)]


def test_generation_detection_self_consistency():
    vocab_size = 1000
    depth = 8
    ngram_len = 5
    gen_len = 256
    temperature = 0.7
    num_seq = 200
    width = ngram_len - 1

    cfg = _dummy_synthid_cfg(vocab_size, depth, ngram_len)

    t0 = time.perf_counter()
    watermarked = [
        _generate_watermarked_sequence(cfg, seed=10_000 + s, gen_len=gen_len, temperature=temperature)
        for s in range(num_seq)
    ]
    t_gen = time.perf_counter() - t0
    random_seqs = [
        _generate_random_sequence(vocab_size, width + gen_len, seed=90_000 + s) for s in range(num_seq)
    ]

    def z_scores(seqs, truncate=None):
        mean_z, wmean_z = [], []
        for tokens in seqs:
            t = tokens if truncate is None else tokens[:truncate]
            mean_z.append(score_token_ids_mean(t, cfg).z_score)
            wmean_z.append(score_token_ids_weighted_mean(t, cfg).z_score)
        return mean_z, wmean_z

    t0 = time.perf_counter()
    wm_mean_z, wm_wmean_z = z_scores(watermarked)
    rnd_mean_z, rnd_wmean_z = z_scores(random_seqs)
    t_det = time.perf_counter() - t0

    def summarize(name, xs):
        return (
            f"{name}: n={len(xs)} mean={statistics.mean(xs):.3f} "
            f"std={statistics.pstdev(xs):.3f} min={min(xs):.3f} max={max(xs):.3f}"
        )

    print(f"\n[(d) self-consistency] generation: {t_gen:.1f}s, detection: {t_det:.1f}s "
          f"(200x256 tok, vocab={vocab_size}, depth={depth})")
    print(summarize("watermarked mean_z (256tok)", wm_mean_z))
    print(summarize("random mean_z (256tok)", rnd_mean_z))
    print(summarize("watermarked wmean_z (256tok)", wm_wmean_z))
    print(summarize("random wmean_z (256tok)", rnd_wmean_z))

    margin = 1.0  # see module docstring "(d) ..." -- tiny relative to the observed gap
    assert min(wm_mean_z) > max(rnd_mean_z) + margin, "mean scorer: watermarked/random z overlap at 256 tok"
    assert min(wm_wmean_z) > max(rnd_wmean_z) + margin, "weighted-mean scorer: watermarked/random z overlap at 256 tok"

    # DEFAULT_Z_THRESHOLD achieves 0 classification errors on this simulation.
    tpr = sum(1 for z in wm_mean_z if z >= DEFAULT_Z_THRESHOLD) / num_seq
    fpr = sum(1 for z in rnd_mean_z if z >= DEFAULT_Z_THRESHOLD) / num_seq
    print(f"DEFAULT_Z_THRESHOLD={DEFAULT_Z_THRESHOLD}: mean scorer TPR={tpr:.3f} FPR={fpr:.3f}")
    assert tpr == 1.0
    assert fpr == 0.0

    # 200-token truncation (reuses the same generated sequences, no extra
    # generation cost -- see task brief "also run at 200-token truncation").
    truncate_at = width + 200
    wm_mean_z_t, wm_wmean_z_t = z_scores(watermarked, truncate=truncate_at)
    rnd_mean_z_t, rnd_wmean_z_t = z_scores(random_seqs, truncate=truncate_at)
    print(summarize("watermarked mean_z (200tok)", wm_mean_z_t))
    print(summarize("random mean_z (200tok)", rnd_mean_z_t))

    assert min(wm_mean_z_t) > max(rnd_mean_z_t) + margin, "mean scorer: overlap at 200-tok truncation"
    assert min(wm_wmean_z_t) > max(rnd_wmean_z_t) + margin, "weighted-mean scorer: overlap at 200-tok truncation"
