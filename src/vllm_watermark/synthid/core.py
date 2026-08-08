# SPDX-License-Identifier: Apache-2.0
"""SynthID-Text tournament-sampling watermark primitives (vLLM-free).

Attribution / upstream sources
-------------------------------
1. transformers.generation.logits_process.SynthIDTextWatermarkLogitsProcessor
   and SynthIDTextWatermarkState
       (file: generation/logits_process.py, class SynthIDTextWatermarkState
        lines 2540-2568, class SynthIDTextWatermarkLogitsProcessor lines
        2571-3010 -- __init__ 2643-2674, update_scores 2685-2707, __call__
        2710-2770, accumulate_hash 2772-2807, compute_ngram_keys 2809-2840,
        _compute_keys 2842-2880, sample_g_values 2882-2900, compute_g_values
        2907-2921, compute_context_repetition_mask 2923-2967,
        expected_mean_g_value 2995-3010)
   Installed version verified: transformers==4.57.6
       (python3 -c "import transformers; print(transformers.__file__)"
        -> .../site-packages/transformers/generation/logits_process.py)
   License: Apache-2.0 (huggingface/transformers). See
       https://github.com/huggingface/transformers/blob/main/LICENSE
2. transformers.generation.configuration_utils.SynthIDTextWatermarkingConfig
       (file: generation/configuration_utils.py, lines 1339-1420ish)
   -- field *defaults* only (sampling_table_size=2**16, sampling_table_seed=0,
   context_history_size=1024, skip_first_ngram_calls=False) and its
   `validate()` bound `sampling_table_size <= 2**24` (ported below verbatim).
3. google-deepmind/synthid-text, tag 0.2.1, commit
   8f2e2316904ea7291ac96e30eb394c453dcc577b (verified via
   `gh api repos/google-deepmind/synthid-text/tags` / raw.githubusercontent.com,
   Apache-2.0, "Copyright 2024 DeepMind Technologies Limited"):
       src/synthid_text/hashing_function.py -- `accumulate_hash`. Diffed
         byte-for-byte against transformers' own copy (only a docstring
         line-wrap differs); confirms both projects ship the identical LCG.
       src/synthid_text/g_value_expectations.py -- `expected_mean_g_value`
         (ported below as `expected_mean_g_value`, reference/documentation
         use in detector.py, not load-bearing for our detector's threshold).
   The mean / weighted-mean *scorer* formulas
   (src/synthid_text/detector_mean.py) are ported into detector.py, not
   here -- see that module's docstring.

NOT ported: google-deepmind/synthid-text's own `SynthIDLogitsProcessor`
(src/synthid_text/logits_processing.py) is a DIFFERENT, richer processor
(explicit `temperature`, `top_k`, `num_leaves`, a `hash_iv` derived from the
keys themselves, and a "distortionary" tournament variant) than the class we
port here. Per the task brief, our reference is transformers'
SynthIDTextWatermarkLogitsProcessor specifically (fixed `num_leaves=2`
implicitly, i.e. one Bernoulli(0.5) g-value per depth layer, no temperature/
top_k pre-filtering). Do not confuse the two when reading the DeepMind repo.

Convention: g-values are 0/1 ints (`torch.randint(low=0, high=2, ...)` builds
the sampling table transformers/DeepMind both use; see `_sampling_table`
below) -- a Bernoulli(0.5) coin per (ngram, depth-layer) combination, not a
{-1,+1} or continuous convention.

Repeated-context masking: transformers' own `__call__` skips watermarking
entirely (returns the original, un-reweighted scores) for any step whose
(ngram_len-1)-token context hash matches one already present in a
fixed-size ring buffer (`context_history_size`, most-recent-first, oldest
evicted). Per the Task interface contract, THIS module does not own that
history -- it takes the yes/no decision as an already-computed
`context_seen: bool` argument to `process_scores_row`. See the module
docstring of whichever caller owns the history (this repo's
`vllm_watermark.synthid.detector` for offline scoring; a future vLLM-plugin
Task owns per-row history during generation) for how that history is kept.
detector.py's caller-side implementation deliberately uses exact Python
tuple equality for context comparison instead of replicating transformers'
64-bit-hash-based ring buffer -- see detector.py docstring "Repeated-context
masking deviation" for why, and why it is provably at least as accurate.

Device independence (verified by execution, not assumed -- same discipline
as `vllm_watermark.kgw.core`)
------------------------------------------------------------------------------
SynthID has exactly one source of torch randomness: the sampling table
(`torch.randint(low=0, high=2, size=(sampling_table_size,), generator=...)`,
built once from `sampling_table_seed`). Everything else -- `accumulate_hash`
(pure add/mul/add integer arithmetic), the `% sampling_table_size` reduction,
and `update_scores`'s softmax/reweight loop -- involves no RNG at all, so it
is deterministic given identical inputs regardless of which device those
ops happen to run on (elementwise/reduction float and int ops are not
device-RNG-dependent the way `torch.Generator`-seeded sampling is). Applying
the exact same "always build with a device-less (CPU) `torch.Generator()`"
rule `vllm_watermark.kgw.core.greenlist_ids` uses for its permutation (see
that module's "CRITICAL DEVIATION" docstring section) makes the sampling
table -- and therefore every g-value derived from it -- byte-identical
whether the caller ultimately wants it on CPU or GPU: `_sampling_table`
below has no `device` parameter, exactly like `greenlist_ids`, and is moved
with `.to(device)` by the caller if needed. See `test_synthid_equivalence.py`
(c) for the executed check.

int64 overflow wraparound (verified by execution): `accumulate_hash`'s
multiplier (6364136223846793005) is close to int64-max and both `+`/`*`
routinely overflow signed 64-bit range during folding. Confirmed locally
that torch CPU int64 add/mul silently wrap using standard two's-complement
semantics (`torch.tensor([2**62]) * 4 == tensor([0])`;
`torch.tensor([2**63 - 1]) + 1 == tensor([-2**63])`, matching
`ctypes.c_int64(2**63).value == -2**63`) -- i.e. identical to what CPython's
`int` arithmetic would give if explicitly reduced mod 2**64 into signed
range at each step. We rely on torch's tensor int64 ops for this (not plain
Python ints) specifically so this wraparound happens automatically and
identically to transformers' own tensor-based implementation, rather than
us having to hand-reproduce two's-complement bit-twiddling in Python.
Likewise confirmed `tensor([-1], dtype=torch.int64) % 65536 == 65535` (torch's
integer `%` is floor-mod / same-sign-as-divisor, i.e. Python-`%`-compatible,
not C-style truncating `%`) -- required for `ngram_keys % sampling_table_size`
to always land in `[0, sampling_table_size)` even though `ngram_keys` is
frequently negative after wraparound.

`SynthIDConfig.vocab_size` is carried (like `KGWConfig.vocab_size`) for
parity/documentation and for `expected_mean_g_value(cfg.vocab_size)`, but
unlike KGW's greenlist permutation, no function in this module actually
*needs* it: `g_values`/`process_scores_row` derive their candidate set from
whatever `candidate_token_ids`/`scores_row` the caller passes in, not from
`cfg.vocab_size`. There is therefore no KGW-style "narrower vocab_size than
logits width" failure mode here to guard against.

Temperature independence (documented, not a config field)
------------------------------------------------------------------------------
Unlike DeepMind's own richer `SynthIDLogitsProcessor` (see "NOT ported"
above), the class ported here has no `temperature` field or parameter.
`process_scores_row` calls `torch.softmax` on whatever `scores_row` tensor
it is handed and applies the depth-layer tournament reweighting to that
probability distribution; the reweighting math
(`probs * (1 + g - g_mass)` per layer) is well-defined for any valid
probability distribution and has no free parameter that depends on
temperature. What *does* change with temperature is the resulting bias's
practical strength (a sharper, lower-temperature distribution is dominated
by fewer high-probability candidates, changing what `g_mass` ends up being
relative to), but that is an emergent property of the math applied to an
already-scaled distribution, not an input to this module. We confirmed by
reading (not assuming) transformers' own `.generate()` pipeline order:
`generation/utils.py`, `_get_logits_processor`, line 1267 appends
`TemperatureLogitsWarper` (inside the `do_sample` branch) strictly BEFORE
line 1299's `# Watermarking should be after all logits processing is
finished (see #34630)` / line 1300's append of the watermarking processor --
i.e. transformers' own reference pipeline always runs SynthID on
ALREADY-temperature-scaled probabilities. Whatever integrates this module
with vLLM's own logits-processor pipeline must decide, and document, its
own ordering relative to temperature -- that decision is out of scope here
(this module is vLLM-free and takes `scores_row` as given).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch

# One key per tournament layer/depth. 30 is the depth this repo's key
# derivation (`vllm_watermark.keys.WatermarkKey.derive_subkeys`) defaults
# to when building a SynthID key list from a single secret -- see that
# method's docstring. Not read by anything in this module (keys' length
# *is* the depth -- see SynthIDConfig.__post_init__); exposed here purely
# as the documented, single source of truth for that default so callers
# (tests, and whatever builds SynthIDConfig from a WatermarkKey) don't have
# to hardcode "30" independently.
DEFAULT_SYNTHID_DEPTH = 30

# Canonical subkey-derivation label for SynthID layer keys. GENERATION AND
# DETECTION MUST BOTH USE THIS CONSTANT with the same depth when calling
# WatermarkKey.derive_subkeys() — a mismatch makes every g-value silently
# disagree (scores near zero, no error raised). Promoted here from
# synthid/processor.py so the Phase 3 detection service imports the same
# constant instead of hardcoding its own copy.
SYNTHID_KEY_LABEL = b"vllm-watermark:synthid-subkeys:v1"

# LCG constants, byte-for-byte identical between transformers'
# accumulate_hash and google-deepmind/synthid-text's hashing_function.py
# (see module docstring attribution #1/#3) -- "adapted linear congruential
# generator with newlib/musl parameters".
_LCG_MULTIPLIER = 6364136223846793005
_LCG_INCREMENT = 1

# DeepMind SynthIDTextWatermarkingConfig.validate() bound (configuration_utils.py
# ~line 1407-1414): "sampling_table_size should be < 2**24".
_MAX_SAMPLING_TABLE_SIZE = 1 << 24


@dataclass(frozen=True)
class SynthIDConfig:
    """Configuration for SynthID-Text tournament-sampling watermarking.

    Attributes:
        vocab_size: carried for parity/documentation only -- see module
            docstring "SynthIDConfig.vocab_size is carried...". REQUIRED,
            explicit, matching KGWConfig's convention (never infer from a
            tokenizer).
        keys: one integer key per tournament layer/depth (transformers'
            `keys: list[int]` __init__ arg, stored here as an immutable
            tuple -- REQUIRED, non-empty). `len(keys)` IS the tournament
            depth; there is no separate `depth` __init__ field (matches
            transformers, which also derives depth from `len(keys)` via
            `g_values.shape[-1]` rather than storing it). See
            `vllm_watermark.keys.WatermarkKey.derive_subkeys` for how to
            derive a `DEFAULT_SYNTHID_DEPTH`-length key tuple from one
            secret.
        ngram_len: n-gram length (context + the token being scored/biased).
            transformers/DeepMind default 5 in their example configs; that
            default is reproduced here.
        sampling_table_size: size of the precomputed Bernoulli(0.5) g-value
            lookup table. transformers default `2**16` (65536).
        sampling_table_seed: seed for building that table. transformers
            default 0. NOT secret -- the table itself carries no
        watermark-identifying information on its own (see
        `_sampling_table`); the `keys` are what makes the scheme keyed.
        Deployments use the recorded public/default seed `0`; secret
        per-layer keys are derived separately from `WatermarkKey`.
        context_history_size: size of the repeated-context history window a
            caller (not this module -- see module docstring) should keep.
            transformers default 1024. Carried here purely so one config
            value can drive a caller's history bound the same way
            `KGWConfig.gamma`/`delta` are carried for callers that don't
            need them for greenlist math itself.
        skip_first_ngram_calls: advisory only -- see module docstring
            "temperature independence" sibling note: this module has no
            per-row call counter (`process_scores_row` is stateless and has
            no notion of "the Nth call for this row"), so it cannot itself
            skip early positions. A stateful caller tracking how many
            tokens a row has generated may consult this flag to decide
            whether to invoke `process_scores_row` at all before
            `ngram_len - 1` real context tokens exist. transformers default
            False.
    """

    vocab_size: int
    keys: "tuple[int, ...]"
    ngram_len: int = 5
    sampling_table_size: int = 1 << 16
    sampling_table_seed: int = 0
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False
    depth: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.keys, tuple):
            object.__setattr__(self, "keys", tuple(self.keys))
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if not self.keys:
            raise ValueError("keys must be a non-empty sequence (one per tournament layer)")
        for k in self.keys:
            if not isinstance(k, int) or not (0 <= k < (1 << 63)):
                raise ValueError(
                    f"each key must be an int in [0, 2**63) (fits signed int64), got {k!r}"
                )
        if self.ngram_len < 1:
            raise ValueError(f"ngram_len must be >= 1, got {self.ngram_len}")
        if self.sampling_table_size <= 0:
            raise ValueError(f"sampling_table_size must be positive, got {self.sampling_table_size}")
        if self.sampling_table_size > _MAX_SAMPLING_TABLE_SIZE:
            # Ported bound: transformers' SynthIDTextWatermarkingConfig.validate()
            # (generation/configuration_utils.py) rejects sampling_table_size > 2**24.
            raise ValueError(
                f"sampling_table_size must be <= {_MAX_SAMPLING_TABLE_SIZE} (2**24), "
                f"got {self.sampling_table_size}"
            )
        if self.context_history_size < 0:
            raise ValueError(
                f"context_history_size must be >= 0, got {self.context_history_size}"
            )
        object.__setattr__(self, "depth", len(self.keys))


def _accumulate_hash(
    current_hash: torch.Tensor,
    data: torch.Tensor,
    multiplier: int = _LCG_MULTIPLIER,
    increment: int = _LCG_INCREMENT,
) -> torch.Tensor:
    """Faithful port of transformers'/DeepMind's `accumulate_hash` (see
    module docstring attribution #1/#3 -- byte-for-byte identical LCG in
    both upstream sources).

    Unlike the upstream versions (written for a `(batch_size, ...)` leading
    dimension driven through `torch.vmap`), this port has no batch
    dimension and no vmap: `current_hash` and `data` just need to be
    broadcastable the ordinary torch way (`current_hash.shape` broadcastable
    against `data.shape[:-1]`), which every call site below arranges via
    `[:, None]` / `[None, :, None]` reshapes exactly like the upstream calls
    do for their batch dimension. This is a shape-broadcasting
    simplification only -- the arithmetic performed is identical.

    `current_hash` and `data` must be `int64` tensors (see module docstring
    "int64 overflow wraparound" for why this must run as torch tensor ops,
    not Python `int` arithmetic).
    """
    for i in range(data.shape[-1]):
        current_hash = current_hash + data[..., i]
        current_hash = current_hash * multiplier
        current_hash = current_hash + increment
    return current_hash


@functools.lru_cache(maxsize=32)
def _sampling_table(sampling_table_size: int, sampling_table_seed: int) -> torch.Tensor:
    """The precomputed Bernoulli(0.5) g-value lookup table (transformers
    `SynthIDTextWatermarkLogitsProcessor.__init__`, logits_process.py lines
    2657-2669). ALWAYS built with a device-less (CPU) `torch.Generator()` --
    see module docstring "Device independence" -- so this function has no
    `device` parameter by construction, exactly like
    `vllm_watermark.kgw.core.greenlist_ids`. Callers move the result with
    `.to(device)` if needed and MUST NOT mutate the returned tensor (it is
    shared, memoized by `(sampling_table_size, sampling_table_seed)` --
    every `SynthIDConfig` with the same two values gets back the identical
    tensor object).
    """
    generator = torch.Generator()  # no device arg -> CPU, always
    generator.manual_seed(sampling_table_seed)
    return torch.randint(
        low=0, high=2, size=(sampling_table_size,), generator=generator, dtype=torch.int64
    )


@functools.lru_cache(maxsize=32)
def _keys_tensor(keys: "tuple[int, ...]") -> torch.Tensor:
    """`torch.tensor(cfg.keys, dtype=torch.int64)`, memoized by the keys
    tuple itself (cheap either way at typical depth ~30, but avoided
    per-call to keep `process_scores_row` allocation-free on the hot path
    at high `vocab_size` -- see the micro-benchmark in
    test_synthid_equivalence.py)."""
    return torch.tensor(keys, dtype=torch.int64)


@functools.lru_cache(maxsize=32)
def _keys_tensor_on(keys: "tuple[int, ...]", device_str: str) -> torch.Tensor:
    """Per-device cached copy of `_keys_tensor(keys)`. Content is identical
    on every device — the tensor is always BUILT on CPU first (see the
    module docstring "Device independence") and only MOVED here."""
    return _keys_tensor(keys).to(device=device_str)


@functools.lru_cache(maxsize=8)
def _sampling_table_on(
    sampling_table_size: int, sampling_table_seed: int, device_str: str
) -> torch.Tensor:
    """Per-device cached copy of `_sampling_table(...)`. The table is
    always BUILT with the CPU generator (device-independent content) and
    only MOVED to `device_str` — int64 values copy bit-exactly, so a table
    lookup on CUDA returns the identical g-value the CPU lookup returns.
    This is what lets the generation-side hot path run on the GPU while a
    CPU detector recomputes identical g-values (the whole g-value path is
    integer arithmetic + this table lookup; no device RNG is involved
    after construction)."""
    return _sampling_table(sampling_table_size, sampling_table_seed).to(device=device_str)


def g_values(
    ngram_context: "list[int]", candidate_token_ids: torch.Tensor, cfg: SynthIDConfig
) -> torch.Tensor:
    """Compute g-values for a set of candidate continuation tokens given a
    single (ngram_len - 1)-token context.

    Faithful, single-row port of transformers' `_compute_keys` +
    `sample_g_values` (logits_process.py lines 2842-2900) -- verified
    equal to calling those methods directly on a batch-of-1 (see
    test_synthid_equivalence.py (a)/(b); also cross-checked that
    `_compute_keys(context, [c])` and `compute_ngram_keys(context + [c])`
    agree, i.e. folding a context then a candidate is equivalent to hashing
    the full `ngram_len`-token ngram in one pass -- this is exactly the
    `accumulate_hash` composition property its own docstring states:
    `f(x, data[T]) = f(f(x, data[:T-1]), data[T])`).

    Args:
        ngram_context: the `cfg.ngram_len - 1` token ids immediately
            preceding the position being scored/biased, oldest-first (same
            order as transformers' `SynthIDTextWatermarkState.context`).
        candidate_token_ids: 1-D int64 (or int) tensor of candidate token
            ids to score -- typically `torch.arange(vocab_size)` when
            biasing a full logits row (see `process_scores_row`), or a
            single observed token id when detecting (see `detector.py`).
        cfg: SynthIDConfig.

    Returns:
        int64 tensor of shape `(len(candidate_token_ids), cfg.depth)`,
        entries in `{0, 1}` (Bernoulli(0.5) convention -- see module
        docstring "Convention"). Row i, layer d is the g-value for
        `ngram_context + [candidate_token_ids[i]]` at tournament layer d.

    Raises:
        ValueError: `len(ngram_context) != cfg.ngram_len - 1`, or
            `candidate_token_ids` is not 1-D.
    """
    if len(ngram_context) != cfg.ngram_len - 1:
        raise ValueError(
            f"ngram_context must have exactly cfg.ngram_len - 1 = {cfg.ngram_len - 1} "
            f"tokens, got {len(ngram_context)}"
        )
    if candidate_token_ids.dim() != 1:
        raise ValueError(
            f"candidate_token_ids must be 1-D (num_candidates,), got shape "
            f"{tuple(candidate_token_ids.shape)}"
        )

    candidates = candidate_token_ids.to(dtype=torch.int64)

    # All arithmetic below runs on candidates' device. Device independence
    # is preserved by construction: the sampling table and keys tensor are
    # BUILT on CPU (bit-exact when moved), and everything else is int64
    # add/mul/mod — exact, identical semantics on CPU and CUDA (C-style
    # two's-complement wraparound; pinned by executed tests). This is what
    # makes the GPU hot path safe for CPU-side detection (measured 291 ms/
    # call CPU at vocab 151936/depth 30 — the GPU path exists because of
    # that number; see EXPERIMENTS.md 2026-08-08 Phase 2).
    device = candidates.device
    device_str = str(device)

    # Fold the context tokens (accumulate_hash(ones(), context)).
    context_tensor = torch.tensor(ngram_context, dtype=torch.int64, device=device)
    context_hash = _accumulate_hash(
        torch.ones((), dtype=torch.int64, device=device), context_tensor
    )

    # Fold in each candidate token: one more LCG step, vectorized over all
    # candidates at once (matches _compute_keys folding `indices[:, :, None]`
    # against the batch's single context hash).
    per_candidate_hash = _accumulate_hash(context_hash, candidates[:, None])  # (num_candidates,)

    # Fold in each of the `depth` keys, one per tournament layer (matches
    # _compute_keys folding `self.keys[None, None, :, None]`).
    keys_tensor = _keys_tensor_on(cfg.keys, device_str)  # (depth,)
    per_layer_hash = _accumulate_hash(
        per_candidate_hash[:, None], keys_tensor[None, :, None]
    )  # (num_candidates, depth)

    table = _sampling_table_on(cfg.sampling_table_size, cfg.sampling_table_seed, device_str)
    idx = per_layer_hash % cfg.sampling_table_size
    return table[idx]


def _update_scores(scores_row: torch.Tensor, g_values_row: torch.Tensor) -> torch.Tensor:
    """Faithful, single-row port of transformers'
    `SynthIDTextWatermarkLogitsProcessor.update_scores`
    (logits_process.py lines 2685-2707). `scores_row` is assumed to be in
    log-space (raw logits), matching the upstream docstring ("We assume
    that the scores are in the log space.").

    Performance note (measured, not guessed -- see the
    `process_scores_row` micro-benchmark in test_synthid_equivalence.py):
    `g_values_row` arrives as `(vocab_size, depth)`, so a naive per-layer
    `g_values_row[:, i]` is a stride-`depth` (non-contiguous) column slice
    at every one of the `depth` loop iterations -- `.to(dtype=...)` on that
    view was measured to dominate this function's cost at vocab_size
    ~150k (cache-unfriendly strided gather, repeated `depth` times). We
    instead convert dtype and transpose to `(depth, vocab_size)` ONCE up
    front, so each loop iteration reads a contiguous row. Numerically this
    is the exact same arithmetic as indexing `g_values_row[:, i]` directly
    -- only the memory layout changes.
    """
    depth = g_values_row.shape[-1]
    g_t = g_values_row.to(dtype=scores_row.dtype).transpose(0, 1).contiguous()  # (depth, vocab_size)
    probs = torch.softmax(scores_row, dim=-1)
    for i in range(depth):
        g_i = g_t[i]
        g_mass = (g_i * probs).sum()
        probs = probs * (1.0 + g_i - g_mass)
    log_probs = torch.log(probs)
    log_probs = torch.where(
        torch.isfinite(log_probs), log_probs, torch.finfo(log_probs.dtype).min
    )
    return log_probs


def process_scores_row(
    scores_row: torch.Tensor,
    ngram_context: "list[int]",
    cfg: SynthIDConfig,
    context_seen: bool,
) -> torch.Tensor:
    """Apply the SynthID tournament-sampling logits warp to ONE row.

    Semantically identical, for that row, to transformers'
    `SynthIDTextWatermarkLogitsProcessor.__call__` (logits_process.py lines
    2710-2770) EXCEPT: (1) no internal state / no batch dimension -- the
    caller supplies `ngram_context` and `context_seen` explicitly every
    call (see module docstring on why repeated-context history is a
    caller concern, not this module's); (2) `debug_mode`
    (uniform-scores test hook) is not ported -- irrelevant to production
    behavior; (3) `skip_first_ngram_calls`'s early-call skip is not applied
    here -- it is advisory config a stateful caller may consult before
    deciding whether to call this function at all (see SynthIDConfig
    docstring).

    Args:
        scores_row: 1-D float tensor of shape `(vocab_size,)`, in log-space
            (raw logits) -- see `_update_scores` docstring.
        ngram_context: see `g_values`.
        cfg: SynthIDConfig.
        context_seen: True if `ngram_context` has already been used at an
            earlier position for this row within the caller's bounded
            history window (i.e. transformers' `is_repeated_context`). When
            True, `scores_row` is returned UNCHANGED (matches
            `torch.where(is_repeated_context, input=scores, other=updated_scores)`
            -- transformers skips watermarking entirely for a repeated
            context rather than applying a weaker/different bias).

    Returns:
        1-D float tensor of shape `(vocab_size,)`: `scores_row` itself
        (same object, not a copy) if `context_seen`, else the
        tournament-reweighted log-probabilities from `_update_scores`.
    """
    if scores_row.dim() != 1:
        raise ValueError(
            f"scores_row must be 1-D (vocab_size,), got shape {tuple(scores_row.shape)}"
        )
    if context_seen:
        return scores_row

    vocab_size = scores_row.shape[0]
    # Candidates on scores_row's device: the whole computation (integer
    # g-value path + float reweighting) then runs where the logits live —
    # on GPU in the vLLM serving path, on CPU everywhere locally. See
    # g_values() for why this cannot change any g-value.
    candidates = torch.arange(vocab_size, dtype=torch.int64, device=scores_row.device)
    g = g_values(ngram_context, candidates, cfg)
    return _update_scores(scores_row, g)


def expected_mean_g_value(vocab_size: int, coinflip_prob: float = 0.5) -> float:
    """Theoretical expected mean g-value after ONE layer of tournament
    watermarking, assuming a uniform LM distribution over `vocab_size`
    candidates (`num_leaves=2`, i.e. the single-layer-per-key scheme this
    module implements -- see module docstring "NOT ported" for the
    distinction from DeepMind's own multi-leaf variant).

    Faithful port of transformers'
    `SynthIDTextWatermarkLogitsProcessor.expected_mean_g_value`
    (logits_process.py lines 2995-3010), cross-checked against
    `google-deepmind/synthid-text`'s `g_value_expectations.expected_mean_g_value`
    (`num_leaves=2` branch: `0.5 + 0.25 * (1 - 1/vocab_size)`, same formula,
    module docstring attribution #3).

    Reference/documentation use only -- NOT used by this module's own
    `g_values`/`process_scores_row` math, and NOT the basis for
    `detector.py`'s detection threshold (that threshold is derived from the
    null-hypothesis (unwatermarked) distribution instead -- see detector.py
    docstring). Exposed here so callers who want the textbook "how far
    above 0.5 should watermarked mean-g land" reference point don't have to
    duplicate this formula.
    """
    return coinflip_prob + coinflip_prob * (1 - coinflip_prob) * (1 - (1 / vocab_size))
