"""Equivalence tests for the KGW port (Task A), run against the locally
installed transformers==4.57.6 as the reference oracle.

Run with: /usr/bin/python3 -m pytest tests/test_kgw_equivalence.py -v
(needs PYTHONPATH to include src/, or `pip install -e .` first)

Test (b) constructs transformers.WatermarkDetector directly via a bare
PretrainedConfig(vocab_size=..., bos_token_id=-1) -- confirmed by execution
that this needs no tokenizer/model download (WatermarkDetector.__init__
only reads model_config.bos_token_id / is_encoder_decoder and builds a
WatermarkLogitsProcessor from vocab_size); see kgw/detector.py module
docstring for the ignore_repeated_ngrams caveat this test's design works
around.
"""

from __future__ import annotations

import hashlib
import math
import random

import pytest
import torch

from vllm_watermark.kgw.core import KGWConfig, greenlist_ids
from vllm_watermark.kgw.detector import score_token_ids

transformers = pytest.importorskip("transformers")
from transformers import PretrainedConfig, WatermarkDetector, WatermarkLogitsProcessor  # noqa: E402

TRANSFORMERS_DEFAULT_HASHING_KEY = 15485863  # read from installed
# transformers/generation/logits_process.py WatermarkLogitsProcessor.__init__
# default (and configuration_utils.WatermarkingConfig default) -- "the
# millionth prime", per that file's own docstring.


# ---------------------------------------------------------------------------
# (a) greenlist_ids equivalence vs transformers.WatermarkLogitsProcessor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [50257, 151936])
def test_greenlist_ids_matches_transformers(vocab_size):
    cfg = KGWConfig(
        vocab_size=vocab_size,
        hash_key=TRANSFORMERS_DEFAULT_HASHING_KEY,
        gamma=0.25,
    )
    processor = WatermarkLogitsProcessor(
        vocab_size=vocab_size,
        device="cpu",
        greenlist_ratio=0.25,
        hashing_key=TRANSFORMERS_DEFAULT_HASHING_KEY,
        seeding_scheme="lefthash",
        context_width=1,
    )

    rng = random.Random(1234)
    mismatches = []
    for _ in range(200):
        prev_token = rng.randrange(vocab_size)
        ours = set(greenlist_ids(prev_token, cfg).tolist())
        theirs = set(processor._get_greenlist_ids(torch.tensor([prev_token], device="cpu")).tolist())
        if ours != theirs:
            mismatches.append(prev_token)

    assert not mismatches, f"greenlist mismatch for prev_token values: {mismatches[:10]}"


# ---------------------------------------------------------------------------
# (b) detector z_score equivalence vs transformers.WatermarkDetector
# ---------------------------------------------------------------------------


def _make_transformers_detector(vocab_size: int, ignore_repeated_ngrams: bool) -> "WatermarkDetector":
    # bos_token_id=-1: never equals a real (>=0) token id, so
    # WatermarkDetector's "strip a leading bos" branch never fires --
    # keeps this an apples-to-apples comparison against score_token_ids,
    # which has no BOS-stripping behavior at all (see detector.py docstring).
    model_config = PretrainedConfig(vocab_size=vocab_size, is_encoder_decoder=False, bos_token_id=-1)
    watermarking_config = {
        "greenlist_ratio": 0.25,
        "bias": 2.0,
        "hashing_key": TRANSFORMERS_DEFAULT_HASHING_KEY,
        "seeding_scheme": "lefthash",
        "context_width": 1,
    }
    return WatermarkDetector(
        model_config=model_config,
        device="cpu",
        watermarking_config=watermarking_config,
        ignore_repeated_ngrams=ignore_repeated_ngrams,
    )


@pytest.mark.parametrize("ignore_repeated_ngrams", [False, True])
def test_detector_z_score_matches_transformers(ignore_repeated_ngrams):
    vocab_size = 50257
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=TRANSFORMERS_DEFAULT_HASHING_KEY, gamma=0.25)
    detector = _make_transformers_detector(vocab_size, ignore_repeated_ngrams)

    rng = random.Random(5678)
    max_abs_diff = 0.0
    for trial in range(50):
        length = rng.randint(30, 300)
        token_ids = [rng.randrange(vocab_size) for _ in range(length)]

        theirs = detector(
            torch.tensor([token_ids], device="cpu"), z_threshold=4.0, return_dict=True
        )
        their_z = float(theirs.z_score[0])

        ours = score_token_ids(token_ids, cfg, ignore_repeated_ngrams=ignore_repeated_ngrams)

        assert ours.z_score == pytest.approx(their_z, rel=1e-6, abs=1e-9), (
            f"trial={trial} length={length} ignore_repeated_ngrams={ignore_repeated_ngrams} "
            f"ours={ours.z_score} theirs={their_z}"
        )
        max_abs_diff = max(max_abs_diff, abs(ours.z_score - their_z))

    # Sanity: the loop actually executed and compared real numbers.
    assert max_abs_diff < 1e-6


# ---------------------------------------------------------------------------
# (c) generation<->detection self-consistency with a derived 64-bit hash_key
# ---------------------------------------------------------------------------


def _derive_hash_key(secret: bytes) -> int:
    # Mirrors keys.py's _derive_hash_key -- inlined here so this test does
    # not depend on env vars (keys.py's public API requires WATERMARK_KEYS
    # or WATERMARK_KEY to be set in the environment).
    return int.from_bytes(hashlib.sha256(secret).digest()[:8], "big")


def test_generation_detection_self_consistency():
    dummy_secret = b"test-dummy-secret-not-production"  # obviously-dummy test key, not a real secret
    hash_key = _derive_hash_key(dummy_secret)
    assert hash_key >= 2**32  # sanity: this is a real 64-bit-derived value, not a small transformers-range int

    vocab_size = 2000
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=hash_key, gamma=0.25, delta=4.0)

    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(42)

    # Watermarked greedy generation over a toy vocab: bias green-list
    # logits by cfg.delta at every step, then argmax.
    tokens = [int(torch.randint(0, vocab_size, (1,), generator=gen).item())]
    for _ in range(256):
        base_logits = torch.randn(vocab_size, generator=gen)
        green = greenlist_ids(tokens[-1], cfg)
        base_logits[green] += cfg.delta
        tokens.append(int(torch.argmax(base_logits).item()))

    watermarked_result = score_token_ids(tokens, cfg)
    assert watermarked_result.z_score > 4.0, (
        f"watermarked z_score={watermarked_result.z_score} did not clear 4.0"
    )

    # Random (unwatermarked) token sequence of the same length: should not
    # show a watermark signal.
    random_tokens = [int(t) for t in torch.randint(0, vocab_size, (257,), generator=gen)]
    random_result = score_token_ids(random_tokens, cfg)
    assert random_result.z_score < 1.5, (
        f"random z_score={random_result.z_score} unexpectedly high"
    )


# ---------------------------------------------------------------------------
# (d) device-independence
# ---------------------------------------------------------------------------


def test_greenlist_ids_device_independent():
    import inspect

    # greenlist_ids has no device parameter at all -- device-independence
    # by construction, per core.py's documented deviation from transformers.
    sig = inspect.signature(greenlist_ids)
    assert "device" not in sig.parameters

    cfg = KGWConfig(vocab_size=1000, hash_key=999_999_937)
    ids_first = greenlist_ids(7, cfg)
    ids_second = greenlist_ids(7, cfg)
    assert torch.equal(ids_first, ids_second)
    assert ids_first.device.type == "cpu"

    # Only CPU is available on this workstation (AGENTS.md: no local GPU);
    # moving CPU->CPU is still a meaningful check that the returned tensor
    # is a plain, device-movable LongTensor and that moving it doesn't
    # mutate/resample anything.
    moved = ids_first.to("cpu")
    assert torch.equal(moved, ids_first)
    assert ids_first.dtype == torch.int64
