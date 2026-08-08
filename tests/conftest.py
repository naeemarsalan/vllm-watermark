"""Shared pytest setup for every `test_*_processor_static.py` file in this
directory (currently `test_processor_static.py` for
`vllm_watermark.kgw.processor` and `test_synthid_processor_static.py` for
`vllm_watermark.synthid.processor`).

Two things live here so neither test file has to duplicate them (moved out
of `test_processor_static.py`, which used to own this logic alone, per the
Task B2 instruction "reuse the existing vllm-stub pattern (import it,
don't duplicate)"):

1. `src/`-on-`sys.path` bootstrap -- makes the src-layout package importable
   when running pytest from the repo root without an editable install (the
   local workstation cannot pip-install vllm, and an editable install is
   not required just to test the vLLM-free modules).

2. A minimal stub of `vllm.v1.sample.logits_processor`'s public surface
   (`BatchUpdate`, `LogitsProcessor`, `MoveDirectionality`), installed into
   `sys.modules` BEFORE any test file is collected, so that
   `vllm_watermark.kgw.processor` / `vllm_watermark.synthid.processor` --
   which both do `from vllm.v1.sample.logits_processor import (...)` at
   module level -- can be imported at all. Built to match the *actual*
   v0.18.0 source verbatim (field names, `MoveDirectionality` member names,
   `LogitsProcessor` method signatures) -- see
   `docs/api-notes-vllm-v0.18.0.md` §1 for the fetched source this is
   checked against.

   If real `vllm` IS importable (e.g. this runs in a cluster CI environment
   where vLLM 0.18.0 is actually installed), the stub is skipped and the
   *real* `vllm.v1.sample.logits_processor` classes are used instead -- a
   strictly stronger test. `USING_REAL_VLLM` records which case applied,
   for tests that want to know.

   Because this module (not each test file) does the
   `sys.modules["vllm.v1.sample.logits_processor"] = ...` installation,
   every test file can just write a normal-looking
   `from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor,
   MoveDirectionality` at its own module top -- by the time pytest imports
   any test file, this conftest.py has already run (pytest always collects
   `conftest.py` before the test modules in its directory) so that import
   always succeeds, real or stubbed.

Also defines the `_clean_watermark_env` autouse fixture: every test in this
directory starts from a blank slate regardless of ambient host environment
variables, for every `WATERMARK_*`/`VLLM_WATERMARK_*` name either processor
module reads (see `vllm_watermark.kgw.processor` and
`vllm_watermark.synthid.processor` module docstrings "Env vars this module
reads directly").

All key material anywhere in the test suite is an obviously-dummy test
value (AGENTS.md #3 / CLAUDE.md secrets policy).
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import vllm  # noqa: F401  -- real vLLM, if ever present, wins over the stub

    USING_REAL_VLLM = True
except ImportError:
    USING_REAL_VLLM = False

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
        # args in test files, so order does not matter to us either way).
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

# Re-exported so test files can do `from conftest import BatchUpdate, ...`
# if they prefer that over importing from `vllm.v1.sample.logits_processor`
# directly (both resolve to the identical objects -- real or stubbed,
# installed into sys.modules above).
from vllm.v1.sample.logits_processor import (  # noqa: E402
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

__all__ = [
    "USING_REAL_VLLM",
    "BatchUpdate",
    "LogitsProcessor",
    "MoveDirectionality",
]

# Every WATERMARK_*/VLLM_WATERMARK_* env var either processor module reads
# -- see vllm_watermark.kgw.processor / vllm_watermark.synthid.processor
# module docstrings "Env vars this module reads directly".
_WATERMARK_ENV_VARS = (
    "WATERMARK_KEYS",
    "WATERMARK_KEY",
    "WATERMARK_KEY_ID",
    "VLLM_WATERMARK_DEFAULT",
    "VLLM_WATERMARK_SCHEME",
    "VLLM_WATERMARK_GAMMA",
    "VLLM_WATERMARK_DELTA",
    "VLLM_WATERMARK_CACHE_SIZE",
    "VLLM_WATERMARK_SYNTHID_NGRAM_LEN",
    "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE",
    "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED",
    "VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE",
    "VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS",
    "VLLM_WATERMARK_SYNTHID_KEY_DEPTH",
)


@pytest.fixture(autouse=True)
def _clean_watermark_env(monkeypatch):
    """Every test in this directory starts from a blank slate regardless of
    ambient host env -- applies to all test files (kgw's and synthid's
    alike), see module docstring."""
    for var in _WATERMARK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield
