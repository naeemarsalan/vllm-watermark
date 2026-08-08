# SPDX-License-Identifier: Apache-2.0
"""SynthID-Text detector (vLLM-free): mean / weighted-mean g-value scoring
over token ids.

Attribution / upstream sources
-------------------------------
1. transformers.generation.logits_process.SynthIDTextWatermarkLogitsProcessor
   methods `compute_g_values` (generation/logits_process.py lines 2907-2921)
   and `compute_context_repetition_mask` (lines 2923-2967) -- the *shape* of
   what gets scored (sliding `ngram_len`-token windows; repeated-context
   masking) is ported from these, NOT their exact tensor mechanics (see
   "Repeated-context masking deviation" below). This repo has no Bayesian
   detector (`transformers.generation.watermarking.SynthIDTextWatermarkDetector`
   / `BayesianDetectorModel`) -- per the task brief, we implement the
   untrained mean and weighted-mean scorers instead.
   Installed version verified: transformers==4.57.6.
   License: Apache-2.0 (huggingface/transformers).
2. google-deepmind/synthid-text, tag 0.2.1, commit
   8f2e2316904ea7291ac96e30eb394c453dcc577b (Apache-2.0,
   "Copyright 2024 DeepMind Technologies Limited"):
       src/synthid_text/detector_mean.py -- `mean_score` and
       `weighted_mean_score`, the exact formulas `score_token_ids_mean` /
       `score_token_ids_weighted_mean` below reproduce (transliterated from
       jax.numpy to torch; this repo does not depend on jax). Fetched:
       https://raw.githubusercontent.com/google-deepmind/synthid-text/8f2e2316904ea7291ac96e30eb394c453dcc577b/src/synthid_text/detector_mean.py

    mean_score(g_values, mask):
        watermarking_depth = g_values.shape[-1]
        num_unmasked = sum(mask, axis=1)
        return sum(g_values * expand_dims(mask, 2), axis=(1, 2)) / (watermarking_depth * num_unmasked)

    weighted_mean_score(g_values, mask, weights=None):
        watermarking_depth = g_values.shape[-1]
        if weights is None: weights = linspace(10, 1, watermarking_depth)
        weights *= watermarking_depth / sum(weights)
        g_values *= expand_dims(weights, axis=(0, 1))
        num_unmasked = sum(mask, axis=1)
        return sum(g_values * expand_dims(mask, 2), axis=(1, 2)) / (watermarking_depth * num_unmasked)

   `score_token_ids_mean`/`score_token_ids_weighted_mean` below compute
   exactly these two formulas (verified algebraically: for a single
   sequence, "sum over unmasked (position, depth) g-values, divided by
   depth * num_unmasked_positions" is precisely `g[mask].mean()` for the
   unweighted case, and the analogous weighted reduction for the weighted
   case -- both implemented directly on a pre-filtered `(num_scored, depth)`
   tensor here rather than via an explicit `mask` array, since we drop
   masked-out rows entirely instead of zeroing and dividing).

Not ported from either source: neither `mean_score`/`weighted_mean_score`
NOR `SynthIDTextWatermarkDetector` return a z-score, p-value, or boolean
prediction -- they return a bare scalar score meant to be fed into a
SEPARATELY TRAINED Bayesian classifier (`BayesianDetectorModel`). Per the
Task brief ("we implement the untrained mean + weighted-mean scorers, not
the Bayesian one"), `DetectionResult.z_score`/`.p_value`/`.prediction`
below are THIS REPO's own addition: a closed-form null-hypothesis
significance test layered on top of the ported score, giving KGW-detector-
shaped (`z_score`/`p_value`/`prediction`, see `vllm_watermark.kgw.detector`)
ergonomics without needing a trained model. Derivation:

  Under H0 (`token_ids` was NOT produced by this watermark's sampler --
  e.g. human text, or another model's output), each scored g-value is
  drawn from the same Bernoulli(0.5) `_sampling_table` (see core.py
  "Convention") indexed by a hash of (context, token) pairs the detector
  has no reason to correlate with -- i.e. each scored g-value behaves as
  an independent fair coin flip. (This is the same "uniform LM
  distribution" null assumption `core.expected_mean_g_value`'s own
  docstring describes for the WATERMARKED-side theoretical expectation;
  we use the unwatermarked/null side of that same assumption.) Given
  `num_scored` positions and `depth` independent per-position g-values:

    mean scorer:     n_eff = num_scored * depth
                      SE    = sqrt(0.25 / n_eff)
                      z     = (mean_g - 0.5) / SE

    weighted scorer:  Var(weighted_score) = 0.25 * sum(weights**2) / (depth**2 * num_scored)
                       (standard variance-of-a-weighted-sum-of-i.i.d.-Bernoulli(0.5)
                        terms; weights already normalized to sum to `depth`,
                        matching `weighted_mean_score`'s own normalization)
                       z = (weighted_score - 0.5) / sqrt(Var(weighted_score))

  p_value is the exact one-sided upper-tail normal p-value
  (`0.5 * erfc(z / sqrt(2))`), same textbook formula as
  `vllm_watermark.kgw.detector._compute_p_value`, for the same reason (KGW
  green-token counts and SynthID g-values are both approximately-normal-
  under-CLT proportions under their respective null hypotheses). This is a
  DERIVED, EXECUTED-validated-by-simulation approximation, not a citation
  from the SynthID paper -- see test_synthid_equivalence.py (d) for the
  empirical check (simulated watermarked vs. random-sequence score
  distributions, and the threshold that actually separates them, reported
  alongside this formula's DEFAULT_Z_THRESHOLD=4.0 default).

Repeated-context masking deviation
------------------------------------
transformers' `compute_context_repetition_mask` hashes each
`(ngram_len - 1)`-token context to an int64 (via `accumulate_hash`) and
checks it against a FIXED-SIZE ring buffer of the `context_history_size`
most-recently-seen context hashes, initialized to `context_history_size`
zero-valued slots (`SynthIDTextWatermarkState.__init__`,
logits_process.py lines 2558-2567: `torch.zeros(...)`, not empty). Two
consequences we deliberately do NOT reproduce here:

  1. Hash collisions: two DIFFERENT context n-grams that happen to hash to
     the same int64 would be (incorrectly) treated as "the same context"
     by transformers. Negligible in practice (64-bit hash space), but
     avoidable outright: this module's `_repeated_context_mask` compares
     context tuples directly (Python tuple equality, exact by
     construction), never hashing them at all. This is strictly more
     accurate than the upstream approach, never less.
  2. Zero-sentinel false-positive: because transformers' ring buffer starts
     pre-filled with `context_history_size` copies of the hash value 0
     (not "empty"), if any REAL context's `accumulate_hash` output is
     exactly 0 -- astronomically unlikely but not provably impossible for
     an adversarial or pathological ngram -- transformers would flag its
     very first (real, non-repeated) occurrence as "repeated" and skip it.
     `_repeated_context_mask` here starts from a genuinely empty history
     instead, so this quirk cannot occur.

Both deviations only ever make our masking MORE permissive/accurate
relative to transformers' documented intent ("mask contexts that have
truly already occurred"), matching the same engineering judgment
`vllm_watermark.kgw.detector`'s module docstring makes for its own
`ignore_repeated_ngrams` deviation (ship the documented/intended behavior,
not a byte-for-byte replication of an upstream implementation quirk).
The *history capacity* itself (`context_history_size`, a sliding window of
that many most-recent context occurrences, oldest evicted first) IS
faithfully reproduced -- see `_repeated_context_mask`.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass

import torch

from vllm_watermark.synthid.core import SynthIDConfig, g_values

DEFAULT_Z_THRESHOLD = 4.0

_NULL_G_MEAN = 0.5  # Bernoulli(0.5) g-value convention, see core.py "Convention"
_NULL_G_VAR = 0.25  # Var(Bernoulli(0.5))


@dataclass(frozen=True)
class DetectionResult:
    num_scored: int
    depth: int
    mean_g: float
    score: float
    z_score: float
    p_value: float
    prediction: bool


def _upper_tail_p_value(z_score: float) -> float:
    """Exact one-sided (upper-tail) normal p-value: P(Z >= z_score). Same
    formula as vllm_watermark.kgw.detector._compute_p_value -- see this
    module's docstring for why."""
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _context_windows(token_ids: "list[int]", ngram_len: int):
    """Yield (context_tuple, target_token) for every scoreable position:
    context = the ngram_len-1 tokens immediately preceding position i,
    target = token_ids[i]. Same sliding window transformers'
    compute_g_values / compute_context_repetition_mask take via
    input_ids.unfold(dimension=1, size=ngram_len[-or-1], step=1)."""
    width = ngram_len - 1
    for i in range(width, len(token_ids)):
        yield tuple(token_ids[i - width : i]), token_ids[i]


def _repeated_context_mask(contexts: "list[tuple[int, ...]]", context_history_size: int) -> "list[bool]":
    """True at position i means context[i] was NOT already present in the
    trailing `context_history_size`-entry history at the time position i
    was reached (i.e. this position SHOULD be scored) -- transformers'
    `torch.logical_not(are_repeated_contexts)` convention
    (compute_context_repetition_mask docstring: "0 and 1 stand for repeated
    and not repeated ... respectively", returned negated). See module
    docstring "Repeated-context masking deviation" for how this differs
    from (and is more precise than) transformers' own hash-ring-buffer
    implementation, while faithfully reproducing its bounded-window
    capacity semantics: every context (repeated or not) is pushed into the
    window, and the oldest entry is evicted once the window is full --
    i.e. a context repeated MORE than context_history_size positions apart
    is correctly NOT flagged as repeated (it has already scrolled out of
    the window), exactly like the reference.
    """
    if context_history_size <= 0:
        # Degenerate but well-defined: an always-empty history window ->
        # nothing is ever "already seen" -> every position is scored.
        # (Reference: SynthIDTextWatermarkState's history tensor would have
        # a size-0 last dim; scoring every position is the only sane
        # behavior for that config, not a crash.)
        return [True] * len(contexts)

    history: "deque[tuple[int, ...]]" = deque()
    counts: "Counter[tuple[int, ...]]" = Counter()
    mask: "list[bool]" = []
    for context in contexts:
        seen_before = counts[context] > 0
        mask.append(not seen_before)
        history.append(context)
        counts[context] += 1
        if len(history) > context_history_size:
            evicted = history.popleft()
            counts[evicted] -= 1
            if counts[evicted] == 0:
                del counts[evicted]
    return mask


def _collect_g_values(token_ids: "list[int]", cfg: SynthIDConfig) -> torch.Tensor:
    """Compute g-values for every non-repeated-context position in
    token_ids, via core.g_values() called once per position (each call's
    candidate set is the single observed token at that position -- see
    core.g_values docstring "candidate_token_ids ... a single observed
    token id when detecting").

    Returns an int64 tensor of shape (num_scored, cfg.depth), one row per
    KEPT (non-repeated-context) scored position, in sequence order.
    """
    if len(token_ids) < cfg.ngram_len:
        raise ValueError(
            f"need at least cfg.ngram_len={cfg.ngram_len} tokens to score "
            f"(one full ngram), got {len(token_ids)}"
        )

    windows = list(_context_windows(token_ids, cfg.ngram_len))
    contexts = [c for c, _ in windows]
    keep_mask = _repeated_context_mask(contexts, cfg.context_history_size)

    rows = []
    for (context, target), keep in zip(windows, keep_mask):
        if not keep:
            continue
        candidate = torch.tensor([target], dtype=torch.int64)
        g = g_values(list(context), candidate, cfg)  # shape (1, depth)
        rows.append(g[0])

    if not rows:
        return torch.zeros((0, cfg.depth), dtype=torch.int64)
    return torch.stack(rows, dim=0)


def score_token_ids_mean(
    token_ids: "list[int]",
    cfg: SynthIDConfig,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> DetectionResult:
    """Score token ids with the (unweighted) mean g-value scorer -- ports
    detector_mean.mean_score (see module docstring). z_score/p_value/
    prediction are this module's own addition (see module docstring
    "Not ported from either source").

    Args:
        token_ids: generated (completion-only) token ids, in order.
        cfg: the SynthIDConfig used at generation time (keys, ngram_len,
            sampling_table_size/seed, context_history_size must match
            exactly, or the scored g-values are meaningless).
        z_threshold: prediction = z_score >= z_threshold.

    Raises:
        ValueError: fewer than cfg.ngram_len tokens, or every scoreable
            position was masked out by repeated-context filtering.
    """
    g = _collect_g_values(token_ids, cfg)
    num_scored = g.shape[0]
    if num_scored == 0:
        raise ValueError(
            "no positions left to score: every ngram window's context was "
            "a repeated context (see cfg.context_history_size) -- cannot "
            "compute a score"
        )
    depth = cfg.depth
    mean_g = g.to(torch.float64).mean().item()
    n_eff = num_scored * depth
    se = math.sqrt(_NULL_G_VAR / n_eff)
    z = (mean_g - _NULL_G_MEAN) / se
    p = _upper_tail_p_value(z)
    return DetectionResult(
        num_scored=num_scored,
        depth=depth,
        mean_g=mean_g,
        score=mean_g,
        z_score=z,
        p_value=p,
        prediction=z >= z_threshold,
    )


def score_token_ids_weighted_mean(
    token_ids: "list[int]",
    cfg: SynthIDConfig,
    weights: "torch.Tensor | list[float] | None" = None,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> DetectionResult:
    """Score token ids with the weighted-mean g-value scorer -- ports
    detector_mean.weighted_mean_score (see module docstring). Default
    weights (when `weights=None`) are `linspace(10, 1, depth)`, matching
    the reference's own default (favoring earlier tournament layers).
    z_score/p_value/prediction are this module's own addition (see module
    docstring "Not ported from either source").

    `.mean_g` on the returned DetectionResult is the UNWEIGHTED mean (for
    direct comparability with `score_token_ids_mean`'s output on the same
    input); `.score` is the weighted value the z-score/prediction are
    actually computed from.

    Args:
        token_ids, cfg, z_threshold: see score_token_ids_mean.
        weights: non-negative floats, length cfg.depth. None -> reference
            default (`linspace(10, 1, cfg.depth)`).

    Raises:
        ValueError: as score_token_ids_mean, or `weights` has the wrong
            length.
    """
    g = _collect_g_values(token_ids, cfg)
    num_scored = g.shape[0]
    if num_scored == 0:
        raise ValueError(
            "no positions left to score: every ngram window's context was "
            "a repeated context (see cfg.context_history_size) -- cannot "
            "compute a score"
        )
    depth = cfg.depth

    if weights is None:
        w = torch.linspace(10.0, 1.0, depth, dtype=torch.float64)
    else:
        w = torch.as_tensor(weights, dtype=torch.float64)
        if tuple(w.shape) != (depth,):
            raise ValueError(f"weights must have shape ({depth},), got {tuple(w.shape)}")
    # Normalize so weights sum to `depth`, matching weighted_mean_score's
    # `weights *= watermarking_depth / sum(weights)`.
    w = w * (depth / w.sum())

    g_f = g.to(torch.float64)
    mean_g = g_f.mean().item()
    weighted_score = (g_f * w[None, :]).sum().item() / (depth * num_scored)

    se = math.sqrt(_NULL_G_VAR * (w**2).sum().item()) / (depth * math.sqrt(num_scored))
    z = (weighted_score - _NULL_G_MEAN) / se
    p = _upper_tail_p_value(z)
    return DetectionResult(
        num_scored=num_scored,
        depth=depth,
        mean_g=mean_g,
        score=weighted_score,
        z_score=z,
        p_value=p,
        prediction=z >= z_threshold,
    )


def detect_text(
    text: str,
    tokenizer,
    cfg: SynthIDConfig,
    scorer: str = "mean",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    weights: "torch.Tensor | list[float] | None" = None,
) -> DetectionResult:
    """Tokenize `text` and score it. `tokenizer` is any HF-tokenizer-like
    object exposing `.encode(text, add_special_tokens=False) -> list[int]`
    (e.g. a transformers PreTrainedTokenizerBase). transformers itself is
    NOT imported by this module -- pass in an already-constructed
    tokenizer so this module stays usable without transformers installed
    (matches vllm_watermark.kgw.detector.detect_text).

    Args:
        scorer: "mean" -> score_token_ids_mean, "weighted_mean" ->
            score_token_ids_weighted_mean (weights forwarded).
    """
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if scorer == "mean":
        return score_token_ids_mean(token_ids, cfg, z_threshold=z_threshold)
    if scorer == "weighted_mean":
        return score_token_ids_weighted_mean(
            token_ids, cfg, weights=weights, z_threshold=z_threshold
        )
    raise ValueError(f"scorer must be 'mean' or 'weighted_mean', got {scorer!r}")
