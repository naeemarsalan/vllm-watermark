"""Static unit tests for KGWLogitsProcessor (Task B: vLLM plugin wrapper).

vLLM is not installed on this workstation (AGENTS.md environment facts;
docs/api-notes-vllm-v0.18.0.md), so `vllm_watermark.kgw.processor` cannot be
imported as-is: it does `from vllm.v1.sample.logits_processor import
(BatchUpdate, LogitsProcessor, MoveDirectionality)` at module level.
`tests/conftest.py` installs a minimal stub of exactly that import surface
into `sys.modules` (or leaves real vLLM in place if it's actually
installed) BEFORE this file is collected -- see that module's docstring for
the full rationale (shared, not duplicated, with
`test_synthid_processor_static.py`). Everything else in this file
(KGWConfig, greenlist_ids, WatermarkKey, load_key/load_keys) is Task A's
real, non-stubbed code: this file is a wiring test for processor.py, not a
reimplementation of Task A's algorithm tests (see
tests/test_kgw_equivalence.py for those).

Run with:
    PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_processor_static.py -v
(conftest.py self-inserts src/ onto sys.path so plain
`pytest tests/test_processor_static.py` works without PYTHONPATH too)

All key material below is an obviously-dummy test value (AGENTS.md #3 /
CLAUDE.md secrets policy: "Test/demo keys in tests must be obviously-dummy
values").

Env-var pattern used throughout (important -- see keys.py, Task A):
  * `WATERMARK_KEY=<hex>` (singular) auto-registers key_id "default", so it
    is used whenever a test wants a row with NO explicit `watermark_key_id`
    in extra_args to resolve via KGWLogitsProcessor's default-key logic.
  * `WATERMARK_KEYS=<id>:<hex>,...` (plural, multi-key) does NOT
    auto-designate any key as default -- `load_key(key_id=None)` falls
    back to literal key_id "default" (or `WATERMARK_KEY_ID` if set), which
    will not match custom ids like "k1"/"k2" unless `WATERMARK_KEY_ID` is
    also set. Tests using `WATERMARK_KEYS` therefore always pass an
    explicit `watermark_key_id` in extra_args, except
    `test_init_default_key_id_env_resolves` and
    `test_init_loads_configured_keys`, which exist specifically to pin down
    that distinction.

SCHEME-COORDINATION tests (KGW-vs-SynthID row routing, `watermark_scheme`
validation) live in this file too, not just in
`test_synthid_processor_static.py`, because they exercise
KGWLogitsProcessor's own `validate_params()`/`_new_row_state()` -- see the
"scheme coordination" section near the bottom. The cross-processor
"both loaded together, each claims only its own rows" test additionally
imports `SynthIDLogitsProcessor` and needs `vllm_watermark.synthid.core`/
`vllm_watermark.synthid.detector` (Task A2) importable; it is skipped
(not failed) if that package is not yet present, so this file remains
runnable standalone regardless of Task A2/B2 landing order -- see
`_synthid_available()` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from conftest import BatchUpdate, MoveDirectionality  # noqa: E402

# Import order matters: conftest.py's stub (if any) must already be
# installed before this import, since processor.py does
# `from vllm.v1.sample.logits_processor import ...` at module scope --
# guaranteed by pytest always collecting conftest.py before test files in
# its directory.
from vllm_watermark.kgw.processor import KGWLogitsProcessor, RowState  # noqa: E402
from vllm_watermark.kgw.core import KGWConfig, greenlist_ids  # noqa: E402
from vllm_watermark.keys import load_key  # noqa: E402


def _synthid_available() -> bool:
    """True iff vllm_watermark.synthid.processor (Task A2 + this task,
    B2) can be imported. Lets the cross-processor "both loaded together"
    test in this file (see "scheme coordination" section) be skipped rather
    than failed if it runs before Task A2 has landed."""
    try:
        import vllm_watermark.synthid.processor  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeModelConfig:
    def __init__(self, vocab_size: int):
        self._vocab_size = vocab_size

    def get_vocab_size(self) -> int:
        # Mirrors vllm/config/model.py:1119-1120 get_vocab_size() -- see
        # docs/api-notes-vllm-v0.18.0.md §8.
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


def _make_processor(vocab_size=50) -> KGWLogitsProcessor:
    return KGWLogitsProcessor(_FakeVllmConfig(vocab_size), device="cpu", is_pin_memory=False)


# ---------------------------------------------------------------------------
# __init__ / graceful degradation
# ---------------------------------------------------------------------------


def test_init_graceful_when_no_keys_configured(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_DEFAULT", "off")
    processor = _make_processor(vocab_size=1000)
    assert processor._keys == {}
    assert processor._default_key is None
    assert processor._rows == {}
    assert processor.is_argmax_invariant() is False


def test_init_loads_configured_keys_but_no_implicit_default(monkeypatch):
    """WATERMARK_KEYS with custom ids and no WATERMARK_KEY_ID: both keys
    load, but neither is auto-selected as the default (keys.py falls back
    to literal key_id "default", which matches neither "k1" nor "k2")."""
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1},k2:{_DUMMY_SECRET_K2}")
    processor = _make_processor(vocab_size=1000)
    assert set(processor._keys) == {"k1", "k2"}
    assert processor._default_key is None


def test_init_default_key_id_env_resolves(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1},k2:{_DUMMY_SECRET_K2}")
    monkeypatch.setenv("WATERMARK_KEY_ID", "k2")
    processor = _make_processor(vocab_size=1000)
    assert processor._default_key is not None
    assert processor._default_key.key_id == "k2"


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
    KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args=None))
    KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args={}))


def test_validate_params_rejects_unknown_watermark_key():
    with pytest.raises(ValueError, match="Unknown watermark_"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_bogus": 1})
        )


@pytest.mark.parametrize("bad_value", ["maybe", "2"])
def test_validate_params_rejects_malformed_watermark_flag(bad_value):
    with pytest.raises(ValueError, match="watermark must be"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": bad_value})
        )


def test_validate_params_rejects_empty_key_id():
    with pytest.raises(ValueError, match="non-empty string"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_key_id": "   "})
        )


def test_validate_params_rejects_watermark_on_without_any_keys():
    with pytest.raises(ValueError, match="no watermark keys are configured"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": "on"})
        )


def test_validate_params_rejects_unresolvable_key_id(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    with pytest.raises(ValueError):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "does-not-exist"})
        )


def test_validate_params_accepts_watermark_on_with_configured_key(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"})
    )


def test_validate_params_accepts_bool_and_string_forms(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    for value in (True, "on", "true", "1", "yes"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark": value, "watermark_key_id": "k1"})
        )
    for value in (False, "off", "false", "0", "no"):
        KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args={"watermark": value}))


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
    monkeypatch.setenv("VLLM_WATERMARK_DEFAULT", "off")
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"}), None, [1, 2]),
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
    assert row0.hash_key == key.hash_key


def test_update_state_default_on_enables_unmarked_rows(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)  # -> key_id "default"
    monkeypatch.setenv("VLLM_WATERMARK_DEFAULT", "on")
    processor = _make_processor()

    added = [(0, _FakeSamplingParams(extra_args=None), None, [1])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {0}
    assert processor._rows[0].key_id == "default"


def test_update_state_output_tok_ids_is_same_object_reference(monkeypatch):
    """BatchUpdate contract (interface.py): output_tok_ids is a *reference*
    to the request's live list, not a copy -- see docs/api-notes §1."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    live_output_ids = [1, 2, 3]
    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, live_output_ids)]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert processor._rows[0].output_tok_ids is live_output_ids
    live_output_ids.append(4)
    assert processor._rows[0].output_tok_ids == [1, 2, 3, 4]


def test_update_state_removed(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1]),
        (1, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [2]),
    ]
    processor.update_state(BatchUpdate(batch_size=2, removed=[], added=added, moved=[]))
    assert set(processor._rows) == {0, 1}

    processor.update_state(BatchUpdate(batch_size=1, removed=[0], added=[], moved=[]))
    assert set(processor._rows) == {1}


def test_update_state_moved_unidirectional(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))
    original_state = processor._rows[0]

    processor.update_state(
        BatchUpdate(batch_size=1, removed=[], added=[], moved=[(0, 5, MoveDirectionality.UNIDIRECTIONAL)])
    )
    assert set(processor._rows) == {5}
    assert processor._rows[5] is original_state


def test_update_state_moved_unidirectional_clears_source_and_dest(monkeypatch):
    """a->b unidirectional: b ends up with whatever a had (here: nothing),
    even if b previously had an entry -- matches builtin.py's
    process_dict_updates semantics (docs/api-notes §2)."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [(1, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1])]
    processor.update_state(BatchUpdate(batch_size=2, removed=[], added=added, moved=[]))
    assert set(processor._rows) == {1}

    # Move unwatermarked row 0 (absent from _rows) -> row 1 (present).
    processor.update_state(
        BatchUpdate(batch_size=2, removed=[], added=[], moved=[(0, 1, MoveDirectionality.UNIDIRECTIONAL)])
    )
    assert processor._rows == {}


def test_update_state_moved_swap(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1},k2:{_DUMMY_SECRET_K2}")
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"}), None, [1]),
        (3, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k2"}), None, [2]),
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
# apply(): row selection math
# ---------------------------------------------------------------------------


def test_apply_noop_when_no_rows_active(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_DEFAULT", "off")
    processor = _make_processor(vocab_size=50)
    logits = torch.randn(4, 50)
    original = logits.clone()
    out = processor.apply(logits)
    assert out is logits
    assert torch.equal(logits, original)


def test_apply_biases_only_active_rows_by_exact_greenlist(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    monkeypatch.setenv("VLLM_WATERMARK_GAMMA", "0.25")
    monkeypatch.setenv("VLLM_WATERMARK_DELTA", "3.0")
    vocab_size = 50
    processor = _make_processor(vocab_size=vocab_size)

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"}), None, [7]),
        (1, _FakeSamplingParams(extra_args={"watermark": "off"}), None, [9]),
    ]
    processor.update_state(BatchUpdate(batch_size=2, removed=[], added=added, moved=[]))

    logits = torch.zeros(2, vocab_size)
    original = logits.clone()
    out = processor.apply(logits)
    assert out is logits

    # Inactive row untouched.
    assert torch.equal(logits[1], original[1])

    # Active row: bias must land on exactly the greenlist computed
    # independently via Task A's real greenlist_ids, given the real
    # hash_key for "k1" and prev_token=7 (output_tok_ids[-1]).
    key = load_key(key_id="k1")
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=key.hash_key, gamma=0.25, delta=3.0)
    expected_green = set(greenlist_ids(7, cfg).tolist())
    assert expected_green, "sanity: greenlist should be non-empty for gamma=0.25"

    for tok in range(vocab_size):
        expected = 3.0 if tok in expected_green else 0.0
        assert logits[0, tok].item() == pytest.approx(expected), f"token {tok}"


def test_apply_uses_prompt_tok_ids_when_output_empty(monkeypatch):
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    vocab_size = 30
    processor = _make_processor(vocab_size=vocab_size)

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on"}), [11, 22], []),  # empty output_tok_ids
    ]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.zeros(1, vocab_size)
    processor.apply(logits)

    key = load_key(key_id="default")
    cfg = KGWConfig(vocab_size=vocab_size, hash_key=key.hash_key, gamma=processor._gamma, delta=processor._delta)
    expected_green = set(greenlist_ids(22, cfg).tolist())  # prompt_tok_ids[-1] == 22
    biased = {tok for tok in range(vocab_size) if logits[0, tok].item() != 0.0}
    assert biased == expected_green


def test_apply_skips_out_of_range_row_index(monkeypatch):
    """Defensive guard: a stale/out-of-range row index in self._rows must
    not crash apply() -- it should just be skipped."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=20)

    added = [(5, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1])]
    processor.update_state(BatchUpdate(batch_size=6, removed=[], added=added, moved=[]))
    assert 5 in processor._rows

    # Logits tensor only has 2 rows -- row 5 is out of range.
    logits = torch.zeros(2, 20)
    out = processor.apply(logits)  # must not raise
    assert torch.equal(out, torch.zeros(2, 20))


def test_apply_wider_logits_than_vocab_size_is_fine(monkeypatch):
    """Padded-vocab case (logits width > configured vocab_size): expected
    and safe, no exception, greenlist ids are always < vocab_size so always
    in-bounds for the wider tensor."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=40)  # narrower than the logits tensor below

    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [3])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.zeros(1, 64)  # padded wider than vocab_size=40
    out = processor.apply(logits)  # must not raise
    assert out.shape == (1, 64)
    assert (out[0, 40:] == 0.0).all(), "padding columns beyond vocab_size must never be biased"


def test_apply_narrower_logits_than_vocab_size_asserts(monkeypatch):
    """Genuine misconfiguration (logits width < configured vocab_size):
    must fail loudly, not silently write out-of-range/wrong-signal bias."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=100)

    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [3])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    logits = torch.zeros(1, 50)  # narrower than vocab_size=100
    with pytest.raises(AssertionError):
        processor.apply(logits)


def test_check_vocab_width_only_warns_once(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor(vocab_size=40)
    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [3])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    with caplog.at_level(logging.WARNING, logger="vllm_watermark.kgw.processor"):
        processor.apply(torch.zeros(1, 64))
        processor.apply(torch.zeros(1, 64))
        processor.apply(torch.zeros(1, 64))

    mismatch_warnings = [r for r in caplog.records if "logits width" in r.message]
    assert len(mismatch_warnings) == 1, "vocab-width mismatch must be logged exactly once, not per apply() call"


def test_greenlist_cache_identical_and_lru(monkeypatch):
    """The LRU memo must return bit-identical greenlists vs the uncached path,
    and evict oldest entries beyond VLLM_WATERMARK_CACHE_SIZE."""
    import torch

    from vllm_watermark.kgw.core import KGWConfig, greenlist_ids

    monkeypatch.setenv("VLLM_WATERMARK_CACHE_SIZE", "4")
    proc = _make_processor(vocab_size=1000)
    hash_key = 0xDEADBEEF12345678 % (1 << 64)
    for prev_token in [7, 11, 7, 999, 0, 42, 7]:
        cached = proc._greenlist_ids_cached(hash_key, prev_token)
        direct = greenlist_ids(
            prev_token,
            KGWConfig(vocab_size=proc._vocab_size, hash_key=hash_key,
                      gamma=proc._gamma, delta=proc._delta),
        )
        assert torch.equal(cached, direct), prev_token
    assert len(proc._greenlist_cache) <= 4

    monkeypatch.setenv("VLLM_WATERMARK_CACHE_SIZE", "0")
    proc0 = _make_processor(vocab_size=1000)
    out = proc0._greenlist_ids_cached(hash_key, 7)
    assert torch.equal(out, greenlist_ids(
        7, KGWConfig(vocab_size=proc0._vocab_size, hash_key=hash_key,
                     gamma=proc0._gamma, delta=proc0._delta)))
    assert len(proc0._greenlist_cache) == 0


# ---------------------------------------------------------------------------
# Scheme coordination (watermark_scheme / VLLM_WATERMARK_SCHEME) -- see
# vllm_watermark.request_args module docstring "SCHEME-COORDINATION DESIGN".
# ---------------------------------------------------------------------------


def test_scheme_class_attribute():
    assert KGWLogitsProcessor.SCHEME == "kgw"


def test_validate_params_accepts_no_watermark_scheme_unchanged(monkeypatch):
    """A request that never mentions watermark_scheme at all must validate
    exactly as before this task -- both with keys configured and without,
    both watermark on and off. Pins down the "WITHOUT changing any existing
    behavior for requests that don't pass watermark_scheme when
    VLLM_WATERMARK_SCHEME is unset/kgw" requirement."""
    KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args=None))
    KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args={"watermark": "off"}))
    with pytest.raises(ValueError, match="no watermark keys are configured"):
        KGWLogitsProcessor.validate_params(_FakeSamplingParams(extra_args={"watermark": "on"}))

    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark": "on", "watermark_key_id": "k1"})
    )


@pytest.mark.parametrize("scheme", ["kgw", "synthid", "KGW", "SynthID", " kgw "])
def test_validate_params_accepts_valid_watermark_scheme_values(monkeypatch, scheme):
    monkeypatch.setenv("WATERMARK_KEYS", f"k1:{_DUMMY_SECRET_K1}")
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_key_id": "k1", "watermark_scheme": scheme}
        )
    )


@pytest.mark.parametrize("bad_scheme", ["bogus", "KGW2", 3, 1.5, ""])
def test_validate_params_rejects_invalid_watermark_scheme_values(bad_scheme):
    with pytest.raises(ValueError, match="watermark_scheme"):
        KGWLogitsProcessor.validate_params(
            _FakeSamplingParams(extra_args={"watermark_scheme": bad_scheme})
        )


def test_validate_params_explicit_none_watermark_scheme_means_omitted():
    """extra_args={"watermark_scheme": None} is indistinguishable from
    omitting the key entirely (dict.get returns None either way) -- must
    NOT raise, and resolves to the default scheme, not a rejection."""
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark_scheme": None})
    )


def test_validate_params_now_recognizes_watermark_scheme_as_known_key():
    """Before this task, `watermark_scheme` would have been rejected as an
    unknown watermark_* key (KNOWN_WATERMARK_XARGS used to be just
    {"watermark", "watermark_key_id"}). It must now be accepted as a known
    key (subject to its own value validation, covered above)."""
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark_scheme": "kgw"})
    )
    KGWLogitsProcessor.validate_params(
        _FakeSamplingParams(extra_args={"watermark_scheme": "synthid"})
    )


def test_init_default_scheme_env(monkeypatch):
    processor = _make_processor(vocab_size=1000)
    assert processor._default_scheme == "kgw", "VLLM_WATERMARK_SCHEME unset -> default 'kgw'"

    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "synthid")
    processor = _make_processor(vocab_size=1000)
    assert processor._default_scheme == "synthid"


def test_init_bad_scheme_env_raises(monkeypatch):
    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "not-a-scheme")
    with pytest.raises(ValueError, match="watermark_scheme"):
        _make_processor(vocab_size=1000)


def test_new_row_state_scheme_mismatch_row_absent(monkeypatch):
    """A request explicitly asking for watermark=on but watermark_scheme=
    "synthid" must NOT activate a row in KGWLogitsProcessor -- see class
    docstring "SCHEME-COORDINATION DESIGN"."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    processor = _make_processor()

    added = [
        (0, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_scheme": "synthid"}
        ), None, [1]),
        (1, _FakeSamplingParams(
            extra_args={"watermark": "on", "watermark_scheme": "kgw"}
        ), None, [2]),
        (2, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [3]),  # no scheme -> default "kgw"
    ]
    processor.update_state(BatchUpdate(batch_size=3, removed=[], added=added, moved=[]))

    assert set(processor._rows) == {1, 2}, "row 0 (scheme=synthid) must be absent from KGW's rows"


def test_new_row_state_scheme_default_from_env(monkeypatch):
    """VLLM_WATERMARK_SCHEME=synthid means a request with NO explicit
    watermark_scheme resolves to "synthid" and therefore does NOT activate
    in KGWLogitsProcessor, even though `watermark=on` is explicit."""
    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    monkeypatch.setenv("VLLM_WATERMARK_SCHEME", "synthid")
    processor = _make_processor()

    added = [(0, _FakeSamplingParams(extra_args={"watermark": "on"}), None, [1])]
    processor.update_state(BatchUpdate(batch_size=1, removed=[], added=added, moved=[]))

    assert processor._rows == {}, "default scheme is synthid -> KGW must not claim this row"


@pytest.mark.skipif(not _synthid_available(), reason="vllm_watermark.synthid.processor not present yet")
def test_both_processors_loaded_each_claims_only_its_own_rows(monkeypatch):
    """The end-to-end scheme-coordination scenario: KGWLogitsProcessor and
    SynthIDLogitsProcessor loaded into the same "engine" (two independent
    instances, as build_logitsprocs() would construct -- see
    docs/api-notes-vllm-v0.18.0.md §3), fed the IDENTICAL BatchUpdate. Each
    must end up with exactly its own scheme's rows in self._rows, the two
    sets are disjoint, and together they cover every enabled row."""
    from vllm_watermark.synthid.processor import SynthIDLogitsProcessor

    monkeypatch.setenv("WATERMARK_KEY", _DUMMY_SECRET)
    kgw = _make_processor(vocab_size=1000)
    synthid = SynthIDLogitsProcessor(_FakeVllmConfig(1000), device="cpu", is_pin_memory=False)

    added = [
        (0, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "kgw"}), None, [1]),
        (1, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "synthid"}), None, [2]),
        (2, _FakeSamplingParams(extra_args={"watermark": "off"}), None, [3]),
        (3, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "kgw"}), None, [4]),
        (4, _FakeSamplingParams(extra_args={"watermark": "on", "watermark_scheme": "synthid"}), None, [5]),
    ]
    batch_update = BatchUpdate(batch_size=5, removed=[], added=added, moved=[])

    kgw.update_state(batch_update)
    synthid.update_state(batch_update)

    assert set(kgw._rows) == {0, 3}
    assert set(synthid._rows) == {1, 4}
    assert set(kgw._rows) & set(synthid._rows) == set(), "no row may be claimed by both processors"
    assert set(kgw._rows) | set(synthid._rows) == {0, 1, 3, 4}, "every enabled row must be claimed by exactly one"
