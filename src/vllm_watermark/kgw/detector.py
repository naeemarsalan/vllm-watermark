"""KGW watermark detector (vLLM-free): z-score / p-value over token ids.

Attribution / upstream source
------------------------------
Ported from:
    transformers.generation.watermarking.WatermarkDetector
    (file: generation/watermarking.py, class WatermarkDetector,
     methods _score_ngrams_in_passage and _compute_z_score, restricted to
     the "lefthash" seeding scheme with context_width=1)
Installed version verified: transformers==4.57.6
License: Apache-2.0 (huggingface/transformers). See
    https://github.com/huggingface/transformers/blob/main/LICENSE

Scoring window: for lefthash with context_width=1, transformers scores
n=context_width+1-selfhash=2-token windows (prefix=token[i-1],
target=token[i]) for i in 1..T-1, i.e. every token is judged against the
green-list seeded by its immediately preceding token. That is exactly what
this module does.

DEVIATION 1 -- ignore_repeated_ngrams: a real bug found by EXECUTING the
installed transformers 4.57.6 code, not by reading it
------------------------------------------------------------------------
transformers' `_score_ngrams_in_passage` builds
`collections.Counter(ngram_tensors[batch_idx])`, i.e. a Counter keyed by
*torch.Tensor row objects*. torch.Tensor.__hash__ is identity-based
(`id()`), not value-based, so two ngram windows with IDENTICAL token
values are NEVER counted as the same Counter key -- every ngram window is
its own dict entry with count 1, regardless of repeats. We verified this
directly:

    >>> import torch, collections
    >>> t = torch.tensor([[1,2],[1,2],[3,4]])
    >>> collections.Counter(list(t))
    Counter({tensor([1, 2]): 1, tensor([1, 2]): 1, tensor([3, 4]): 1})
    >>> len(_)   # 3, not 2 -- the two [1,2] rows were NOT merged

Consequence: in transformers 4.57.6, `ignore_repeated_ngrams=True` and
`ignore_repeated_ngrams=False` produce IDENTICAL detector output --
"ignore_repeated_ngrams" is a no-op in practice, contradicting its
docstring ("Whether to count every unique ngram only once or not.").

This module implements the DOCUMENTED/INTENDED semantics instead (value
-based dedup of (prev_token, target_token) pairs when
ignore_repeated_ngrams=True), because that is the documented contract we
were asked to port ("port faithfully" the *unique-ngram logic*, i.e. the
logic the docstring describes) and because shipping a knowingly-broken
flag would be a worse choice for a production detector. This is verified
NOT to break equivalence with transformers' actual (buggy) runtime output
under the test conditions used here: at vocab_size >= 50257 and sequence
lengths <= 300, the probability of two identical adjacent-token pairs
occurring by chance in a uniformly random sequence is negligible, so
transformers' (buggy, always-count-every-window) output and our
(correct, dedup-when-requested) output coincide for both flag values in
test_kgw_equivalence.py. See that file for the empirical check.

DEVIATION 2 -- p_value: exact one-sided normal tail via math.erfc, not
transformers' approximation
------------------------------------------------------------------------
transformers' `_compute_pval` computes
`0.5 * exp(-2*z**2/pi)` for z>=0 (algebraically simplified from its
sign()-based formula) -- a fast approximation to the erf function, not the
exact normal-distribution survival function. Per the Task A spec, this
module instead computes the exact one-sided (upper-tail) p-value of a
standard normal z-score using `math.erfc` (no scipy dependency):

    p_value = 0.5 * erfc(z / sqrt(2))

This is the textbook-exact P(Z >= z) for Z ~ N(0,1), appropriate for a
one-sided test (watermarked text has an excess of green tokens, not a
deficit). z_score itself (not p_value) is what test_kgw_equivalence.py
checks against transformers for exact agreement; p_value is intentionally
not required to match transformers' approximation.

DEVIATION 3 -- prediction threshold: z >= 4.0 (>=, not >; default 4.0, not
transformers' 3.0). Per Task A spec, not a transformers equivalence claim.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from dataclasses import dataclass

from vllm_watermark.kgw.core import KGWConfig, greenlist_ids

DEFAULT_Z_THRESHOLD = 4.0
_MAX_GREENLIST_CACHE_ENTRIES = 32
_GREENLIST_CACHE_BUDGET_BYTES = 64 << 20
_PY_SET_BYTES_PER_TOKEN = 96  # conservative Python int/set accounting


def _greenlist_cache_capacity(cfg: KGWConfig) -> int:
    per_entry = max(1, cfg.greenlist_size * _PY_SET_BYTES_PER_TOKEN)
    return max(1, min(_MAX_GREENLIST_CACHE_ENTRIES, _GREENLIST_CACHE_BUDGET_BYTES // per_entry))


@dataclass(frozen=True)
class DetectionResult:
    num_tokens_scored: int
    num_green: int
    z_score: float
    p_value: float
    prediction: bool


def _compute_z_score(num_green: int, num_tokens_scored: int, gamma: float) -> float:
    expected = gamma * num_tokens_scored
    denom = math.sqrt(num_tokens_scored * gamma * (1.0 - gamma))
    return (num_green - expected) / denom


def _compute_p_value(z_score: float) -> float:
    """Exact one-sided (upper-tail) normal p-value: P(Z >= z_score)."""
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def score_token_ids(
    token_ids: list[int],
    cfg: KGWConfig,
    ignore_repeated_ngrams: bool = False,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> DetectionResult:
    """Score a sequence of token ids for the KGW green-list watermark.

    Each token at position i (1 <= i < len(token_ids)) is judged against
    the green-list seeded by token_ids[i-1] (lefthash, context_width=1).
    Position 0 is never scored -- it has no predecessor to seed from.

    Args:
        token_ids: generated (completion-only) token ids, in order. Do NOT
            include a BOS/prompt prefix that wasn't itself watermarked --
            unlike transformers' WatermarkDetector this function does not
            special-case stripping a leading bos_token_id; callers pass
            exactly the span they want scored.
        cfg: the KGWConfig used at generation time (must match exactly --
            especially vocab_size and hash_key -- or scores are meaningless
            silently near-zero; see core.py module docstring).
        ignore_repeated_ngrams: if True, each distinct (prev_token,
            target_token) pair is counted at most once (see DEVIATION 1
            above for why this differs from transformers' actual runtime
            behavior while matching its documented intent).
        z_threshold: prediction = z_score >= z_threshold.

    Returns:
        DetectionResult.

    Raises:
        ValueError: if token_ids has fewer than 2 tokens (nothing to score).
    """
    if len(token_ids) < cfg.context_width + 1:
        raise ValueError(
            f"need at least {cfg.context_width + 1} tokens to score "
            f"(context_width={cfg.context_width}), got {len(token_ids)}"
        )

    pairs = [(token_ids[i - 1], token_ids[i]) for i in range(1, len(token_ids))]

    # Cache the greenlist SET per distinct prev_token: seeding only depends
    # on prev_token (lefthash, context_width=1), so this is both correct
    # and avoids recomputing a torch.randperm per token when prev_token
    # repeats -- true regardless of ignore_repeated_ngrams.
    greenlist_cache: "OrderedDict[int, set[int]]" = OrderedDict()
    cache_capacity = _greenlist_cache_capacity(cfg)

    def is_green(prev_token: int, target_token: int) -> bool:
        cached = greenlist_cache.get(prev_token)
        if cached is None:
            cached = set(greenlist_ids(prev_token, cfg).tolist())
            greenlist_cache[prev_token] = cached
            if len(greenlist_cache) > cache_capacity:
                greenlist_cache.popitem(last=False)
        else:
            greenlist_cache.move_to_end(prev_token)
        return target_token in cached

    if ignore_repeated_ngrams:
        unique_pairs = Counter(pairs)
        num_tokens_scored = len(unique_pairs)
        num_green = sum(1 for pair in unique_pairs if is_green(*pair))
    else:
        num_tokens_scored = len(pairs)
        num_green = sum(1 for pair in pairs if is_green(*pair))

    z_score = _compute_z_score(num_green, num_tokens_scored, cfg.gamma)
    p_value = _compute_p_value(z_score)
    prediction = z_score >= z_threshold

    return DetectionResult(
        num_tokens_scored=num_tokens_scored,
        num_green=num_green,
        z_score=z_score,
        p_value=p_value,
        prediction=prediction,
    )


def detect_text(
    text: str,
    tokenizer,
    cfg: KGWConfig,
    ignore_repeated_ngrams: bool = False,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> DetectionResult:
    """Tokenize `text` and score it. `tokenizer` is any HF-tokenizer-like
    object exposing `.encode(text, add_special_tokens=False) -> list[int]`
    (e.g. a transformers PreTrainedTokenizerBase). transformers itself is
    NOT imported by this module -- pass in an already-constructed
    tokenizer so this module stays usable without transformers installed.
    """
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return score_token_ids(
        list(token_ids), cfg, ignore_repeated_ngrams=ignore_repeated_ngrams, z_threshold=z_threshold
    )
