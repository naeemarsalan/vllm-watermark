"""Static unit tests for the Phase 2 SynthIDLogitsProcessor wrapper.

vLLM is not installed on this workstation -- see `tests/conftest.py`'s
docstring for the stub it installs (shared with
`test_processor_static.py`, not duplicated here) before this file is
collected, so `vllm_watermark.synthid.processor`'s module-level
`from vllm.v1.sample.logits_processor import (...)` succeeds. Everything
else this file exercises (`SynthIDConfig`, `process_scores_row`,
`WatermarkKey.derive_subkeys`) is real, non-stubbed code: this
file is a wiring test for `synthid/processor.py`, not a reimplementation of
the Phase 2 algorithm tests.

Run with:
    PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_synthid_processor_static.py -v
(conftest.py self-inserts src/ onto sys.path so plain
`pytest tests/test_synthid_processor_static.py` works without PYTHONPATH
too)

All key material below is an obviously-dummy test value (AGENTS.md #3
secrets policy). The small `_Fake*` fixture classes and dummy
secrets below are intentionally duplicated (not shared via conftest.py)
from `test_processor_static.py`'s own copies -- they are ~20 lines of
trivial per-file test scaffolding, not the substantial vllm-stub-install
logic that WAS moved to conftest.py; keeping each processor's test file
independently readable/self-contained outweighs de-duplicating this much
code (matches the existing repo convention of two independently-complete
`kgw/processor.py`-shaped and `synthid/processor.py`-shaped files).

Section map:
  - init / graceful degradation
  - validate_params (incl. scheme validation)
  - update_state: added / removed / moved bookkeeping
  - scheme coordination (row absent when watermark_scheme != "synthid")
  - _row_context(): prompt/output boundary, zero-padding vs.
    skip_first_ngram_calls (Task brief: "investigate what transformers
    does for the first calls ... and mirror it" -- see
    synthid/processor.py module docstring "DESIGN DECISION 1/2" for the
    citation-backed writeup this test section pins down in code)
  - _ContextHistory: repeated-context masking + LRU-style (FIFO) bounding
  - apply(): end-to-end row selection + bias application, cross-checked
    against vllm_watermark.synthid.core.process_scores_row called directly
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

# Import via the canonical path: tests/conftest.py has already installed
# the v0.18.0-accurate stub into sys.modules (or real vllm exists) by the
# time pytest imports this module, so this works under any import mode and
# combined-suite collection (from-conftest module imports do not).
from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality  # noqa: E402

# Import order matters: conftest.py's stub (if any) must already be
# installed before this import -- guaranteed by pytest always collecting
# conftest.py before test files in its directory.
from vllm_watermark.synthid.processor import (  # noqa: E402
    SynthIDLogitsProcessor,
    RowState,
    _ContextHistory,
    _SYNTHID_KEY_LABEL,
)
from vllm_watermark.synthid.core import (  # noqa: E402
    DEFAULT_SYNTHID_DEPTH,
    SynthIDConfig,
    process_scores_row,
)
from vllm_watermark.keys import load_key  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / fakes (see module docstring for why these are duplicated, not
# shared, with test_processor_static.py's copies)
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        return self._vocab_size


class _FakeVllmConfig:
    def __init__(self, vocab_size: int):
        self.model_config = _FakeModelConfig(vocab_size)


@dataclass
class _FakeSamplingParams:
    """Only exposes what processor.py actually reads (`extra_args`)."""

    extra_args: "dict | None" = None


# Obviously-dummy hex secrets -- never real key material (AGENTS.md #3).
_DUMMY_SECRET = "aa" * 8  # for WATERMARK_KEY (singular) -> key_id "default"
_DUMMY_SECRET_K1 = "11" * 8  # for WATERMARK_KEYS entries -> key_id "k1"
_DUMMY_SECRET_K2 = "22" * 8  # -> key_id "k2"


def _make_processor(vocab_size=50) -> SynthIDLogitsProcessor:
    return SynthIDLogitsProcessor(_FakeVllmConfig(vocab_size), device="cpu", is_pin_memory=False)


def _empty_history(maxlen=1024) -> _ContextHistory:
    return _ContextHistory(maxlen=maxlen)


# ---------------------------------------------------------------------------
# __init__ / graceful degradation
# ---------------------------------------------------------------------------


def test_init_graceful_when_no_keys_configured():
    processor = _make_processor(vocab_size=1000)
    assert processor._keys == {}
    assert processor._default_key is None
    assert processor._rows == {}
    assert processor.is_argmax_invariant() is False


def test_init_default_config_matches_synthid_config_defaults():
    """No SynthID-specific env set -> processor's hyperparameters match
    SynthIDConfig's own field defaults (synthid/core.py)."""
    processor = _make_processor(vocab_size=1000)
    assert processor._ngram_len == 5
    assert processor._sampling_table_size == 1 << 16
    assert processor._sampling_table_seed == 0
    assert processor._context_history_size == 1024
    assert processor._skip_first_ngram_calls is False
    assert processor._key_depth == DEFAULT_SYNTHID_DEPTH == 30


def test_init_synthid_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE", "1024")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED", "7")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE", "8")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS", "on")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_KEY_DEPTH", "4")
    processor = _make_processor(vocab_size=1000)
    assert processor._ngram_len == 3
    assert processor._sampling_table_size == 1024
    assert processor._sampling_table_seed == 7
    assert processor._context_history_size == 8
    assert processor._skip_first_ngram_calls is True
    assert processor._key_depth == 4


def test_init_rejects_non_positive_key_depth(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_KEY_DEPTH", "0")
    with pytest.raises(ValueError, match="VLLM_WATERMARK_SYNTHID_KEY_DEPTH"):
        _make_processor(vocab_size=1000)


def test_init_loads_configured_keys_but_no_implicit_default(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1},k2:{_DUMMY_SECRET_K2}")
    processor = _make_processor(vocab_size=1000)
    assert set(processor._keys) == {"k1", "k2"}
    assert processor._default_key is None


def test_init_single_watermark_key_env_becomes_default(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=1000)
    assert set(processor._keys) == {"default"}
    assert processor._default_key is not None
    assert processor._default_key.key_id == "default"


# ---------------------------------------------------------------------------
# validate_params
# ---------------------------------------------------------------------------


def test_validate_params_accepts_no_watermark_args():
    SynthIDLogitsProcessor.validate_params(_FakeSamplingParams(extra_args=None))
    SynthIDLogitsProcessor.validate_params(_FakeSamplingParams(extra_args={}))


def test_validate_params_rejects_unknown_watermark_key():
    with pytest.raises(ValueError, match="Unknown watermark_"):
        SynthIDLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_bogus": 1})
        )


def test_validate_params_rejects_malformed_watermark_flag():
    with pytest.raises(ValueError, match="watermark must be"):
        SynthIDLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": "maybe"})
        )


def test_validate_params_rejects_empty_key_id():
    with pytest.raises(ValueError, match="non-empty string"):
        SynthIDLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_key_id": "   "})
        )


def test_validate_params_rejects_watermark_on_without_any_keys():
    with pytest.raises(ValueError, match="no watermark keys are configured"):
        SynthIDLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": "on"})
        )


def test_validate_params_accepts_watermark_on_with_configured_key(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    SynthIDLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"})
    )


@pytest.mark.parametrize("scheme", ["kgw", "synthid", "SYNTHID", " synthid "])
def test_validate_params_accepts_valid_watermark_scheme_values(monkeypatch, scheme):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    SynthIDLogitsProcessor.validate_params(
        _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k1", "watermark_scheme": scheme}
        )
    )


@pytest.mark.parametrize("bad_scheme", ["bogus", "synthid2", 3, ""])
def test_validate_params_rejects_invalid_watermark_scheme_values(bad_scheme):
    with pytest.raises(ValueError, match="watermark_scheme"):
        SynthIDLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_scheme": bad_scheme})
        )


def test_validate_params_and_kgw_agree_on_the_same_extra_args(monkeypatch):
    """The whole point of vllm_watermark.request_args: both processors'
    validate_params() must reach the identical accept/reject verdict for
    the identical extra_args -- see that module's docstring."""
    from vllm_watermark.kgw.processor import KGWLogitsProcessor

    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    agreeing_cases = [
        {},
        {"watermark": "off"},
        {"watermark": "on", "watermark_key_id": "k1"},
        {"watermark_scheme": "kgw"},
        {"watermark_scheme": "synthid"},
    ]
    for extra_args in agreeing_cases:
        KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args=extra_args))
        SynthIDLogitsProcessor.validate_params(_FakeSamplingParams(extra_args=extra_args))

    disagreeing_cases = [
        {"watermark": "bogus"},
        {"watermark_scheme": "not-a-scheme"},
        {"watermark_bogus": 1},
        {"watermark_key_id": ""},
    ]
    for extra_args in disagreeing_cases:
        for cls in (KGWLogitsProcessor, SynthIDLogitsProcessor):
            with pytest.raises(ValueError):
                cls.validate_params(_FakeSamplingParams(extra_args=extra_args))


# ---------------------------------------------------------------------------
# update_state: added / removed / moved bookkeeping
# ---------------------------------------------------------------------------


def test_update_state_none_is_noop(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    processor = _make_processor()
    processor.update_state(None)
    assert processor._rows == {}


def test_update_state_add_only_watermarked_rows_appear(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k1", "watermark_scheme": "synthid"}
        ), None, [1, 2]),
        (1, _FakeSamplingParams(extra_args={"watermark": "off"}), None, [3]),
        (2, _FakeSamplingParams(extra_args=None), [4], []),  # no override -> default off -> absent
    ]
    processor.update_state(BatchUpdate(batch_size=3, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {0}
    row0 = processor._rows[0]
    assert isinstance(row0, RowState)
    assert row0.enabled is True
    assert row0.key_id == "k1"
    assert row0.output_tok_ids == [1, 2]
    key = load_key(key_id="k1")
    assert row0.keys == key.derive_subkeys(processor._key_depth, _SYNTHID_KEY_LABEL)


def test_update_state_default_on_enables_unmarked_rows(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_DEFAULT", "on")
    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "synthid")
    processor = _make_processor()

    added = [(0, _FakeSamplingParams(extra_args=None), None, [1])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {0}
    assert processor._rows[0].key_id == "default"


def test_update_state_output_tok_ids_is_same_object_reference(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    live_output_ids = [1, 2, 3]
    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, live_output_ids)]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert processor._rows[0].output_tok_ids is live_output_ids
    live_output_ids.append(4)
    assert processor._rows[0].output_tok_ids == [1, 2, 3, 4]


def test_update_state_removed(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "synthid"}), None, [1]),
        (1, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "synthid"}), None, [2]),
    ]
    processor.update_state(BatchUpdate(batch_size=2, removed=[], added=added, moved=[]))
    assert set(processor._rows) == {0, 1}

    processor.update_state(BatchUpdate(batch_size=1, removed=[0], added=[], moved=[]))
    assert set(processor._rows) == {1}


def test_update_state_moved_swap(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1},k2:{_DUMMY_SECRET_K2}")
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k1", "watermark_scheme": "synthid"}
        ), None, [1]),
        (3, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k2", "watermark_scheme": "synthid"}
        ), None, [2]),
    ]
    processor.update_state(BatchUpdate(batch_size=4, removed=[], added=added, moved=[]))
    state_a, state_b = processor._rows[0], processor._rows[3]
    assert state_a.key_id == "k1" and state_b.key_id == "k2"

    processor.update_state(
        BatchUpdate(batch_size=4, removed=[], added=[], moved=[(0, 3, MoveDirectionality.SWAP)])
    )
    assert set(processor._rows) == {0, 3}
    assert processor._rows[3] is state_a
    assert processor._rows[0] is state_b


# ---------------------------------------------------------------------------
# Scheme coordination
# ---------------------------------------------------------------------------


def test_scheme_class_attribute():
    assert SynthIDLogitsProcessor.SCHEME == "synthid"


def test_init_default_scheme_env(monkeypatch):
    processor = _make_processor(vocab_size=1000)
    assert processor._default_scheme == "kgw"

    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "synthid")
    processor = _make_processor(vocab_size=1000)
    assert processor._default_scheme == "synthid"


def test_new_row_state_scheme_mismatch_row_absent(monkeypatch):
    """watermark_scheme="kgw" (explicit) must NOT activate a row in
    SynthIDLogitsProcessor."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "kgw"}), None, [1]),
        (1, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "synthid"}), None, [2]),
        (2, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [3]),  # no scheme -> default "kgw"
    ]
    processor.update_state(BatchUpdate(batch_size=3, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {1}, "only the explicit watermark_scheme=synthid row activates"


def test_new_row_state_scheme_default_from_env(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "synthid")
    processor = _make_processor()

    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {0}, "default scheme is synthid -> this processor claims the row"


# ---------------------------------------------------------------------------
# _row_context(): prompt/output boundary, zero-padding vs.
# skip_first_ngram_calls -- see synthid/processor.py "DESIGN DECISION 1/2"
# ---------------------------------------------------------------------------


def _row(prompt_tok_ids, output_tok_ids, keys=(1, 2, 3)) -> RowState:
    return RowState(
        enabled=True,
        key_id="x",
        keys=keys,
        prompt_tok_ids=prompt_tok_ids,
        output_tok_ids=output_tok_ids,
        context_history=_empty_history(),
    )


def test_row_context_uses_output_when_sufficient(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "4")  # needed=3
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=[100, 101], output_tok_ids=[7, 8, 9, 10])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == [8, 9, 10]


def test_row_context_mixes_prompt_and_output_at_boundary(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "4")  # needed=3
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=[100, 101, 102, 103], output_tok_ids=[55])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == [102, 103, 55], "last 2 prompt tokens + the 1 output token, in order"


def test_row_context_zero_pads_when_insufficient_and_not_skipping(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "5")  # needed=4
    processor = _make_processor(vocab_size=50)
    assert processor._skip_first_ngram_calls is False
    row = _row(prompt_tok_ids=[9], output_tok_ids=[])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == [0, 0, 0, 9]


def test_row_context_handles_none_prompt(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")  # needed=2
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=None, output_tok_ids=[])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == [0, 0]


def test_row_context_skip_first_ngram_calls_skips_when_insufficient(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "5")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS", "on")
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=[9], output_tok_ids=[])
    ctx, skip = processor._row_context(row)
    assert skip
    assert ctx == []


def test_row_context_skip_flag_does_not_skip_once_context_sufficient(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS", "on")
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=[9, 10], output_tok_ids=[])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == [9, 10]


def test_row_context_ngram_len_one_has_empty_context(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "1")  # needed=0
    processor = _make_processor(vocab_size=50)
    row = _row(prompt_tok_ids=[9, 10], output_tok_ids=[11])
    ctx, skip = processor._row_context(row)
    assert not skip
    assert ctx == []


# ---------------------------------------------------------------------------
# _ContextHistory: repeated-context masking + LRU-style (FIFO) bounding
# ---------------------------------------------------------------------------


def test_context_history_membership_before_and_after_push():
    history = _ContextHistory(maxlen=10)
    assert history.push((1, 2)) is False, "never seen before -> not a repeat"
    assert history.push((1, 2)) is True, "same context again -> is a repeat"
    assert history.push((3, 4)) is False, "different context -> not a repeat"
    assert len(history) == 3


def test_context_history_fifo_bounding_evicts_oldest():
    history = _ContextHistory(maxlen=3)
    assert history.push((1,)) is False
    assert history.push((2,)) is False
    assert history.push((3,)) is False
    assert history.push((1,)) is True, "(1,) still within the last-3 window"
    assert history.push((2,)) is True, "(2,) still within the last-3 window"
    assert len(history) == 3

    # Push enough new, distinct contexts to fully scroll the original
    # window [(2,), (3,), (1,)] (in this order after the pushes above) out.
    assert history.push((4,)) is False
    assert history.push((5,)) is False
    assert history.push((6,)) is False
    assert len(history) == 3

    assert history.push((2,)) is False, "(2,) must have been evicted by now"
    assert history.push((3,)) is False, "(3,) must have been evicted by now"


def test_context_history_handles_duplicate_values_correctly_on_eviction():
    """Regression guard for a naive set-based implementation: pushing the
    SAME context repeatedly, then evicting past the point where the first
    occurrence scrolls out, must not forget the value is still present
    (still-live duplicate occurrences must keep it a "seen" member)."""
    history = _ContextHistory(maxlen=2)
    assert history.push((9,)) is False  # window: [(9,)]
    assert history.push((9,)) is True  # window: [(9,), (9,)]
    assert history.push((9,)) is True  # push #3 evicts the OLDEST (9,), one (9,) remains
    assert len(history) == 2
    assert history.push((9,)) is True, "one (9,) is still live in the window"


def test_context_history_maxlen_zero_never_flags_repeat():
    history = _ContextHistory(maxlen=0)
    assert history.push((1,)) is False
    assert history.push((1,)) is False
    assert len(history) == 0


# ---------------------------------------------------------------------------
# apply(): row selection + bias application
# ---------------------------------------------------------------------------


def test_apply_noop_when_no_rows_active():
    processor = _make_processor(vocab_size=50)
    logits = torch.randn(4, 50)
    original = logits.clone()
    out = processor.apply(logits)
    assert out is logits
    assert torch.equal(logits, original)


def test_apply_skips_out_of_range_row_index(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=20)

    added = [(5, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, [1])]
    processor.update_state(BatchUpdate(batch_size=6, removed=[], added=added, moved=[]))
    assert 5 in processor._rows

    logits = torch.zeros(2, 20)
    out = processor.apply(logits)  # must not raise
    assert torch.equal(out, torch.zeros(2, 20))


def test_apply_narrower_logits_than_vocab_size_asserts(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=100)

    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, [3])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.zeros(1, 50)  # narrower than vocab_size=100
    with pytest.raises(AssertionError):
        processor.apply(logits)


def test_apply_wider_logits_than_vocab_size_leaves_padding_untouched(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=40)

    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, [3, 4, 5, 6])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.randn(1, 64)  # padded wider than vocab_size=40
    original_padding = logits[0, 40:].clone()
    out = processor.apply(logits)
    assert out.shape == (1, 64)
    assert torch.equal(out[0, 40:], original_padding), "padding columns beyond vocab_size must never be touched"


def test_apply_matches_process_scores_row_called_directly(monkeypatch):
    """Whitebox equivalence: apply()'s output for the one active row must
    equal calling vllm_watermark.synthid.core.process_scores_row() directly
    with the SAME context/keys/cfg this processor derives."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")  # needed=2
    vocab_size = 40
    processor = _make_processor(vocab_size=vocab_size)

    prompt = [5, 6, 7]
    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), prompt, [])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    torch.manual_seed(0)
    logits = torch.randn(1, vocab_size)
    original_row = logits[0].clone()

    out = processor.apply(logits)
    assert out is logits

    key = load_key(key_id="default")
    keys = key.derive_subkeys(processor._key_depth, _SYNTHID_KEY_LABEL)
    cfg = SynthIDConfig(
        vocab_size=vocab_size,
        keys=keys,
        ngram_len=3,
        sampling_table_size=processor._sampling_table_size,
        sampling_table_seed=processor._sampling_table_seed,
        context_history_size=processor._context_history_size,
        skip_first_ngram_calls=processor._skip_first_ngram_calls,
    )
    ngram_context = prompt[-2:]  # needed=2, output empty -> prompt tail
    expected = process_scores_row(original_row.clone(), ngram_context, cfg, context_seen=False)
    assert torch.allclose(logits[0], expected)


def test_apply_zero_pads_and_still_biases_on_first_call(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")  # needed=2, default: no skip
    vocab_size = 30
    processor = _make_processor(vocab_size=vocab_size)

    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, [])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.randn(1, vocab_size)
    original = logits.clone()
    processor.apply(logits)
    assert not torch.equal(logits[0], original[0]), (
        "even with 0 real context tokens, the default (skip_first_ngram_calls=off) "
        "zero-pads and biases immediately"
    )


def test_apply_skip_first_ngram_calls_leaves_logits_untouched_until_enough_context(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "3")  # needed=2
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS", "on")
    vocab_size = 30
    processor = _make_processor(vocab_size=vocab_size)

    output_tok_ids: list[int] = []
    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), None, output_tok_ids)]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits1 = torch.randn(1, vocab_size)
    original1 = logits1.clone()
    processor.apply(logits1)
    assert torch.equal(logits1[0], original1[0]), "0 real context tokens -> must be skipped"

    output_tok_ids.append(5)
    logits2 = torch.randn(1, vocab_size)
    original2 = logits2.clone()
    processor.apply(logits2)
    assert torch.equal(logits2[0], original2[0]), "1 real context token (< needed=2) -> still skipped"

    output_tok_ids.append(6)
    logits3 = torch.randn(1, vocab_size)
    original3 = logits3.clone()
    processor.apply(logits3)
    assert not torch.equal(logits3[0], original3[0]), "2 real context tokens (== needed) -> now watermarked"


def test_apply_repeated_context_leaves_row_unchanged(monkeypatch):
    """End-to-end version of the _ContextHistory unit tests above: a
    row whose context repeats must come back from apply() byte-identical
    to its pre-apply() input (process_scores_row's context_seen=True
    branch returns scores_row unchanged)."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", "2")  # needed=1
    vocab_size = 30
    processor = _make_processor(vocab_size=vocab_size)

    output_tok_ids: list[int] = []
    added = [(0, _FakeSamplingParams(
        extra_args={"watermark": "on", "watermark_scheme": "synthid"}
    ), [3], output_tok_ids)]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    torch.manual_seed(1)
    first_input = torch.randn(1, vocab_size)
    first_copy = first_input.clone()
    processor.apply(first_input)
    assert not torch.equal(first_input[0], first_copy[0]), "sanity: first call actually biases the row"

    output_tok_ids.append(3)  # same token value as the prompt -> context (3,) repeats
    second_input = torch.randn(1, vocab_size)
    second_copy = second_input.clone()
    processor.apply(second_input)
    assert torch.equal(second_input[0], second_copy[0]), "repeated context must leave the row untouched"


def test_apply_multiple_rows_only_active_ones_change(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    vocab_size = 25
    processor = _make_processor(vocab_size=vocab_size)

    added = [
        (0, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k1", "watermark_scheme": "synthid"}
        ), None, [7, 8]),
        (1, _FakeSamplingParams(extra_args={"watermark": "off"}), None, [9]),
    ]
    processor.update_state(BatchUpdate(batch_size=2, removed=[], added=added, moved=[]))

    logits = torch.randn(2, vocab_size)
    original = logits.clone()
    processor.apply(logits)

    assert torch.equal(logits[1], original[1]), "inactive row must be untouched"
    assert not torch.equal(logits[0], original[0]), "active row must be biased"
