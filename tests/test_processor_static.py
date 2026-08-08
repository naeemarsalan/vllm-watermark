"""Static unit tests for KGWLogitsProcessor (Task B: vLLM plugin wrapper).

vLLM is not installed on this workstation (AGENTS.md environment facts;
docs/api-notes-vllm-v0.18.0.md), so `vllm_watermark.kgw.processor` cannot be
imported as-is: it does `from vllm.v1.sample.logits_processor import
(BatchUpdate, LogitsProcessor, MoveDirectionality)` at module level. Before
importing it, this file installs a minimal stub of exactly that import
surface into `sys.modules`, built to match the *actual* v0.18.0 source
verbatim (field names, `MoveDirectionality` member names, `LogitsProcessor`
method signatures) -- see docs/api-notes-vllm-v0.18.0.md §1 for the fetched
source this is checked against.

If real `vllm` IS importable (e.g. this file runs in a cluster CI
environment where vLLM 0.18.0 is actually installed), the stub is skipped
and the *real* vllm.v1.sample.logits_processor classes are used instead --
a strictly stronger test. Everything else in this file (KGWConfig,
greenlist_ids, WatermarkKey, load_key/load_keys) is Task A's real,
non-stubbed code: this file is a wiring test for processor.py, not a
reimplementation of Task A's algorithm tests (see
tests/test_kgw_equivalence.py for those).

Run with:
    PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_processor_static.py -v
(this file also self-inserts src/ onto sys.path so plain
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
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import vllm  # noqa: F401  -- real vLLM, if ever present, wins over the stub

    _USING_REAL_VLLM = True
except ImportError:
    _USING_REAL_VLLM = False

    from enum import Enum, auto

    class MoveDirectionality(Enum):
        # Verified against vllm/v1/sample/logits_processor/interface.py
        # v0.18.0, lines 17-21: two members, this exact naming.
        UNIDIRECTIONAL = auto()
        SWAP = auto()

    @dataclass(frozen=True)
    class BatchUpdate:
        # Verified against interface.py v0.18.0, lines 36-57: exactly these
        # four fields, this exact order (we always construct with keyword
        # args in this file, so order does not matter to us either way).
        batch_size: int
        removed: list
        added: list
        moved: list

    class LogitsProcessor:
        """Minimal stand-in for the abstract base -- interface.py lines
        60-106. Not actually abstract here (no enforcement needed for a
        unit test that always overrides every method)."""

        @classmethod
        def validate_params(cls, sampling_params):
            return None

        def __init__(self, vllm_config, device, is_pin_memory) -> None:
            raise NotImplementedError

        def apply(self, logits):
            raise NotImplementedError

        def is_argmax_invariant(self) -> bool:
            raise NotImplementedError

        def update_state(self, batch_update) -> None:
            raise NotImplementedError

    _vllm = types.ModuleType("vllm")
    _vllm_v1 = types.ModuleType("vllm.v1")
    _vllm_v1_sample = types.ModuleType("vllm.v1.sample")
    _vllm_lp = types.ModuleType("vllm.v1.sample.logits_processor")
    _vllm_lp.BatchUpdate = BatchUpdate
    _vllm_lp.LogitsProcessor = LogitsProcessor
    _vllm_lp.MoveDirectionality = MoveDirectionality
    _vllm_v1_sample.logits_processor = _vllm_lp
    _vllm_v1.sample = _vllm_v1_sample
    _vllm.v1 = _vllm_v1

    sys.modules["vllm"] = _vllm
    sys.modules["vllm.v1"] = _vllm_v1
    sys.modules["vllm.v1.sample"] = _vllm_v1_sample
    sys.modules["vllm.v1.sample.logits_processor"] = _vllm_lp

# Import order matters: the stub (if any) must be installed above before
# this import, since processor.py does `from vllm.v1.sample.logits_processor
# import ...` at module scope.
from vllm_watermark.kgw.processor import KGWLogitsProcessor, RowState  # noqa: E402
from vllm_watermark.kgw.core import KGWConfig, greenlist_ids  # noqa: E402
from vllm_watermark.keys import load_key  # noqa: E402

if _USING_REAL_VLLM:
    from vllm.v1.sample.logits_processor import (  # noqa: E402
        BatchUpdate,
        MoveDirectionality,
    )


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

_WATERMARK_ENV_VARS = (
    "WATERMARK_KEYS",
    "WATERMARK_KEY",
    "WATERMARK_KEY_ID",
    "VLLM_WATERMARK_DEFAULT",
    "VLLM_WATERMARK_GAMMA",
    "VLLM_WATERMARK_DELTA",
)


@pytest.fixture(autouse=True)
def _clean_watermark_env(monkeypatch):
    """Every test starts from a blank slate regardless of ambient env."""
    for var in _WATERMARK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


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
