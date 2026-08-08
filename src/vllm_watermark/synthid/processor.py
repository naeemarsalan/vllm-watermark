# SPDX-License-Identifier: Apache-2.0
"""SynthID-Text watermark vLLM V1 LogitsProcessor plugin (Task B2: plugin
wrapper).

Structured to mirror `vllm_watermark.kgw.processor` (Task B, KGW's plugin
wrapper) as closely as the two algorithms allow: same `LogitsProcessor`
ABC, same sparse per-row `dict[int, RowState]` bookkeeping, same
added/removed/moved `update_state()` protocol (adapted from
`vllm/v1/sample/logits_processor/builtin.py`'s `process_dict_updates`, see
that module's docstring for the full citation -- not repeated here), same
`VLLM_WATERMARK_DEFAULT` env convention, same defensive
"validate_params() should have already rejected this, but fail closed here
too" pattern, same vLLM V1 citations (`docs/api-notes-vllm-v0.18.0.md`).
This is the ONLY module in this package that imports `vllm` (see
`kgw/processor.py` module docstring for why -- same reasoning applies
verbatim; the actual SynthID tournament-sampling math lives in vllm-free
`vllm_watermark.synthid.core` / `vllm_watermark.synthid.detector`, Task A2).

Unlike `kgw/processor.py`, THIS file does not port any algorithm logic from
an upstream source itself -- it is pure wiring, plus several DESIGN
DECISIONS of our own (documented below, each flagged as such per AGENTS.md
"no assumptions presented as facts" -- these are OUR engineering choices
for the vLLM integration, not claims about what transformers/DeepMind do)
that `vllm_watermark.synthid.core`'s module docstring explicitly leaves to
"whatever integrates this module with vLLM's own logits-processor
pipeline" (that module's words, "Temperature independence" section) or to
"a stateful caller" (same module, `SynthIDConfig.skip_first_ngram_calls`
docstring).

SCHEME-COORDINATION DESIGN
---------------------------
See `vllm_watermark.request_args` module docstring for the full rationale.
This class sets `SCHEME = "synthid"`; `KGWLogitsProcessor` sets
`SCHEME = "kgw"`. Both can be loaded into the same engine
(`pyproject.toml`'s `vllm.logits_processors` entry-point group lists both);
a batch row is only ever added to THIS processor's `self._rows` if the
request's resolved scheme (per-request `watermark_scheme` extra_arg, or
`VLLM_WATERMARK_SCHEME` env default) equals `"synthid"` -- see
`_new_row_state()`.

DESIGN DECISION 1 -- context extraction across the prompt/output boundary
---------------------------------------------------------------------------
The Task interface contract (shared design doc, not an upstream source)
specifies: "the last (ngram_len-1)-token context comes from
prompt_tok_ids/output_tok_ids refs at apply() time (same pattern as KGW's
prev_token)". This is a DELIBERATE DEVIATION from how
`transformers.generation.logits_process.SynthIDTextWatermarkLogitsProcessor`
itself builds context during `.generate()`: that class's
`SynthIDTextWatermarkState.__init__` (logits_process.py lines 2558-2568)
initializes `self.context = torch.zeros((batch_size, ngram_len - 1))` --
i.e. it discards the prompt entirely for context purposes and always
starts from `ngram_len - 1` zero-tokens, filling in real (generated) tokens
only as `__call__` is invoked once per decode step
(`self.state.context = concat((self.state.context, input_ids[:, -1:]))[:, 1:]`,
lines 2727-2731). A vLLM row, by contrast, very often already has a prompt
at least `ngram_len - 1` tokens long by the time the FIRST token is
generated -- throwing that real context away in favor of zero-padding would
be strictly worse (weaker, more collision-prone watermarking on the
earliest generated tokens of every request) with no compensating benefit,
so `_row_context()` below instead pulls real tokens from
`prompt_tok_ids`/`output_tok_ids` (falling back to the prompt only for
however many of the needed `ngram_len - 1` tokens the output list doesn't
yet have -- exactly KGW's `prev_token` fallback, generalized from a window
of 1 token to a window of `ngram_len - 1` tokens).

DESIGN DECISION 2 -- insufficient real context: skip_first_ngram_calls vs.
zero-padding
---------------------------------------------------------------------------
`vllm_watermark.synthid.core.g_values()` requires `len(ngram_context) ==
cfg.ngram_len - 1` EXACTLY (raises ValueError otherwise) -- there is no
"partial context" mode. So the very first apply() call(s) for a row whose
combined prompt+output history is still shorter than `ngram_len - 1` (only
possible when the prompt itself is shorter than `ngram_len - 1`, since
`output_tok_ids` only grows) must do ONE of:
  (a) skip calling `process_scores_row` entirely for this row this step
      (leave `logits` untouched) -- this is what
      `SynthIDConfig.skip_first_ngram_calls` is FOR, per that field's own
      docstring ("a stateful caller ... may consult this flag to decide
      whether to invoke process_scores_row at all"). This is the
      `VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS=on` behavior below.
  (b) left-pad the short real context with token-id `0` up to exactly
      `ngram_len - 1` tokens -- mirroring transformers' OWN zero-padding
      choice for its (entirely-discarded-prompt) initial context, just
      applied to a genuinely-too-short real-token window instead of an
      always-empty one. This is the DEFAULT (`skip_first_ngram_calls=off`
      matches transformers' own documented default -- see
      `SynthIDConfig.skip_first_ngram_calls` docstring, "transformers
      default False").
Both choices are implemented in `_row_context()`. Note this only ever
matters for the first `ngram_len - 1` tokens of a request whose PROMPT is
also shorter than `ngram_len - 1` tokens (e.g. `ngram_len=5` and a
2-token prompt) -- once `output_tok_ids` alone reaches `ngram_len - 1`
entries, or if the prompt was long enough to begin with, every context is
a full real-token window and neither branch above is ever taken again for
that row.

DESIGN DECISION 3 -- repeated-context history representation
---------------------------------------------------------------------------
Per the interface contract, `context_history` is "a per-row bounded
set/deque in RowState", and per `vllm_watermark.synthid.core`'s module
docstring, THIS caller (not core.py) owns computing the `context_seen: bool`
argument `process_scores_row()` takes. `_ContextHistory` below is the
online (one-context-pushed-per-apply()-call) counterpart of
`vllm_watermark.synthid.detector._repeated_context_mask` (that module's
offline, whole-sequence-known-upfront version) -- same `Counter`-backed
multiset + `deque` FIFO-eviction idiom, same exact-Python-tuple-equality
comparison (never a hash -- see `detector.py` module docstring "Repeated-
context masking deviation" for why this is strictly more accurate than
transformers' own int64-hash ring buffer, which this package deliberately
does not replicate anywhere). Using the identical idiom in both places
means "was this context already seen" means exactly the same thing whether
asked at generation time (here) or at detection time (`detector.py`),
which matters because a real detector will eventually need to reconstruct
comparable masking during offline scoring of vLLM-generated text (Phase 3,
not built as of this task).

DESIGN DECISION 4 -- CPU/GPU device transfer (a real correctness fix, not
just documentation)
---------------------------------------------------------------------------
`vllm_watermark.synthid.core.g_values()` (and the `_sampling_table()` /
`_keys_tensor()` / `_accumulate_hash()` helpers it calls) build every
tensor WITHOUT a `device=` argument -- i.e. always CPU, by the same
"device-independent by construction" design `kgw/core.py` uses (see
`synthid/core.py` module docstring "Device independence"). KGW's
`apply()` copes with this by moving only a small green-list INDEX tensor
to `logits.device` (`ids.to(device=logits.device, ...)`) while `logits`
itself never moves. SynthID's `process_scores_row()` cannot use that
trick: internally it does `g_i * probs` where `probs = softmax(scores_row)`
lives on WHATEVER device `scores_row` was passed on and `g_i` (derived
from `g_values()`) is always CPU -- multiplying tensors on different
devices raises a `RuntimeError` in torch. So, UNLIKE KGW, this processor
moves the affected logits ROW to CPU before calling `process_scores_row()`
and moves the (small, `(vocab_size,)`-shaped) result back to
`logits.device` afterward -- see `apply()`. `Tensor.to(device)` is a no-op
(returns the same object, no copy) when the tensor is already on that
device, so this costs nothing extra on a CPU-only deployment (e.g. these
static tests) and is exercised there, but it IS a real per-active-row,
per-decode-step host<->device transfer of a full `(vocab_size,)` tensor on
a real GPU deployment -- a materially different (and likely more
expensive) cost profile than KGW's sparse index-only transfer. This has
NOT been measured (no GPU available on this workstation -- see AGENTS.md
environment facts); flagged for Phase 2 benchmarking
(`docs/implementation.md` Phase 2 accept criteria already calls for a
KGW-vs-SynthID overhead comparison table).

DESIGN DECISION 5 -- per-tournament-layer key derivation label
---------------------------------------------------------------------------
`vllm_watermark.keys.WatermarkKey.derive_subkeys(n, label)` derives `n`
tournament-layer keys from one configured secret, namespaced by an
arbitrary `label` (see that method's docstring -- "different subkey
purposes drawn from the same secret ... never collide"). This module fixes
`_SYNTHID_KEY_LABEL` below as ITS label. CROSS-TASK COORDINATION NOTE: any
future detection-side caller (a Phase 3 detection service, not built as of
this task -- see `docs/implementation.md` Phase 3) that reconstructs a
`SynthIDConfig` from a `WatermarkKey` to verify text this processor
generated MUST call `derive_subkeys()` with the IDENTICAL `(depth, label)`
-- i.e. `depth = VLLM_WATERMARK_SYNTHID_KEY_DEPTH` (default
`DEFAULT_SYNTHID_DEPTH` = 30) and `label = _SYNTHID_KEY_LABEL` below -- or
generation and detection will derive DIFFERENT tournament keys and every
g-value will silently disagree, exactly the failure mode `kgw/core.py`'s
module docstring warns about for a `vocab_size` mismatch between
generation and detection.

Env vars this module reads directly (never logged/printed as values):
    VLLM_WATERMARK_DEFAULT   "on"/"off" (default "off") -- shared with
        vllm_watermark.kgw.processor (identical name/default/parsing, via
        vllm_watermark.request_args.resolve_default_on()).
    VLLM_WATERMARK_SCHEME    "kgw"/"synthid" (default "kgw") -- shared with
        vllm_watermark.kgw.processor (identical name/default/parsing, via
        vllm_watermark.request_args.resolve_default_scheme()). A row only
        ever activates in THIS processor if the resolved scheme ==
        "synthid" -- see class docstring "SCHEME-COORDINATION DESIGN".
    VLLM_WATERMARK_SYNTHID_NGRAM_LEN               int >= 1, default "5"
        (matches SynthIDConfig.ngram_len's own default).
    VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE     int, default "65536"
        (2**16, matches SynthIDConfig's own default).
    VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED     int, default "0"
        (matches SynthIDConfig's own default; NOT secret -- see that
        field's docstring in synthid/core.py).
    VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE    int >= 0, default
        "1024" -- per-row repeated-context history window size (see
        DESIGN DECISION 3).
    VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS  "on"/"off", default
        "off" (matches SynthIDConfig's own transformers-sourced default --
        see DESIGN DECISION 2).
    VLLM_WATERMARK_SYNTHID_KEY_DEPTH   int > 0, default "30"
        (vllm_watermark.synthid.core.DEFAULT_SYNTHID_DEPTH) -- number of
        tournament-layer subkeys to derive per configured secret (see
        DESIGN DECISION 5).
    (key material itself is loaded via vllm_watermark.keys -- WATERMARK_KEYS
    / WATERMARK_KEY / WATERMARK_KEY_ID -- never read directly by this file,
    same as kgw/processor.py)

Per-request `vllm_xargs` / `SamplingParams.extra_args` keys this processor
recognizes: `watermark`, `watermark_key_id`, `watermark_scheme` -- see
`vllm_watermark.request_args` module docstring (identical set/semantics to
`kgw/processor.py`, parsed by the identical shared implementation so the
two `validate_params()` methods cannot disagree).
"""

from __future__ import annotations

import logging
import os
from collections import Counter, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from vllm_watermark.keys import WatermarkKey, load_key, load_keys
from vllm_watermark.request_args import (
    parse_watermark_flag,
    resolve_default_on,
    resolve_default_scheme,
    resolve_key_or_raise,
    resolve_request,
)
from vllm_watermark.synthid.core import DEFAULT_SYNTHID_DEPTH, SynthIDConfig, process_scores_row, SYNTHID_KEY_LABEL

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

__all__ = ["SynthIDLogitsProcessor", "RowState"]

_DEFAULT_NGRAM_LEN_ENV = "5"
_DEFAULT_SAMPLING_TABLE_SIZE_ENV = str(1 << 16)
_DEFAULT_SAMPLING_TABLE_SEED_ENV = "0"
_DEFAULT_CONTEXT_HISTORY_SIZE_ENV = "1024"
_DEFAULT_SKIP_FIRST_NGRAM_CALLS_ENV = "off"
_DEFAULT_KEY_DEPTH_ENV = str(DEFAULT_SYNTHID_DEPTH)

# See module docstring "DESIGN DECISION 5". Fixed, versioned label so a
# future label change (e.g. a v2 derivation) cannot silently collide with
# this one.
# Promoted to vllm_watermark.synthid.core.SYNTHID_KEY_LABEL (canonical home,
# shared with the Phase 3 detection service). Kept as a module alias so
# existing imports/tests keep working.
_SYNTHID_KEY_LABEL = SYNTHID_KEY_LABEL


class _ContextHistory:
    """Online, per-row counterpart of
    `vllm_watermark.synthid.detector._repeated_context_mask` -- see this
    module's docstring "DESIGN DECISION 3" for the full rationale. Pushes
    one `(ngram_len - 1)`-token context tuple per `apply()` call for a
    given row and reports whether that exact tuple was already present in
    the trailing `maxlen`-entry window (oldest evicted first).
    """

    __slots__ = ("_history", "_counts", "_maxlen")

    def __init__(self, maxlen: int) -> None:
        self._maxlen = maxlen
        self._history: "deque[tuple[int, ...]]" = deque()
        self._counts: "Counter[tuple[int, ...]]" = Counter()

    def __len__(self) -> int:
        return len(self._history)

    def push(self, context: "tuple[int, ...]") -> bool:
        """Record `context` as seen at the current position and return
        True iff it was ALREADY present in the (bounded) history before
        this call -- i.e. transformers' `is_repeated_context` /
        `detector.py`'s `seen_before`, computed incrementally instead of
        over a whole known-upfront sequence."""
        if self._maxlen <= 0:
            # Degenerate but well-defined: an always-empty window -> never
            # "already seen" -- matches detector.py's
            # _repeated_context_mask same-guard behavior for
            # context_history_size <= 0.
            return False
        seen_before = self._counts[context] > 0
        self._history.append(context)
        self._counts[context] += 1
        if len(self._history) > self._maxlen:
            evicted = self._history.popleft()
            self._counts[evicted] -= 1
            if self._counts[evicted] == 0:
                del self._counts[evicted]
        return seen_before


@dataclass
class RowState:
    """Per-batch-row watermark state. Only rows with the watermark enabled
    for this request AND resolved to the "synthid" scheme ever get an
    entry -- see update_state()/apply() and class docstring
    "SCHEME-COORDINATION DESIGN".

    `prompt_tok_ids` / `output_tok_ids` are the *same list objects* vLLM's
    BatchUpdate.added tuples carry -- see kgw/processor.py's RowState
    docstring (identical reasoning, not repeated here) and
    `docs/api-notes-vllm-v0.18.0.md` §1.

    `keys` is this row's derived tournament-layer key tuple (see class
    docstring "DESIGN DECISION 5") -- resolved once at row-creation time in
    `_new_row_state()`, not re-derived per `apply()` call.

    `context_history` is this row's own `_ContextHistory` (bounded to
    `VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE` entries) -- see class
    docstring "DESIGN DECISION 3".
    """

    enabled: bool
    key_id: str
    keys: "tuple[int, ...]"
    prompt_tok_ids: "list[int] | None"
    output_tok_ids: "list[int]"
    context_history: _ContextHistory


class SynthIDLogitsProcessor(LogitsProcessor):
    """vLLM V1 custom LogitsProcessor applying SynthID-Text tournament-
    sampling bias.

    Subclasses `vllm.v1.sample.logits_processor.LogitsProcessor` -- see
    `vllm_watermark.kgw.processor.KGWLogitsProcessor`'s class docstring for
    the full citation (identical ABC, identical loading mechanisms). Loaded
    either via the `vllm.logits_processors` entry-point group (see
    pyproject.toml, `synthid = ...`) or via `--logits-processors
    vllm_watermark.synthid.processor:SynthIDLogitsProcessor`.

    SCHEME-COORDINATION DESIGN: see module docstring and
    `vllm_watermark.request_args`. `SCHEME = "synthid"`; a row only ever
    activates here if its resolved scheme equals that -- see
    `_new_row_state()`. `KGWLogitsProcessor` (SCHEME = "kgw") can be loaded
    in the same engine; each row is claimed by at most one of the two.
    """

    SCHEME = "synthid"

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Reject malformed/unresolvable watermark_* extra_args.

        Identical structure and behavior to
        `KGWLogitsProcessor.validate_params()` (see that method's docstring
        for the full "why a classmethod can still see env / why ValueError
        surfaces as HTTP 400" discussion -- not repeated here) -- both
        route through `vllm_watermark.request_args.resolve_request()` so
        they cannot disagree about the same `extra_args` dict. Key
        resolvability is checked the same way regardless of
        `watermark_scheme`: `vllm_watermark.keys` key material is shared
        across both schemes (see module docstring "DESIGN DECISION 5"),
        so whether THIS processor's row will actually end up "synthid" is
        irrelevant to whether the requested key_id resolves.
        """
        extra_args = sampling_params.extra_args or {}
        # default_on=False reproduces the same "only check resolvability
        # for an EXPLICIT watermark=on" behavior KGWLogitsProcessor uses --
        # see that method's validate_params() docstring for why.
        enabled, _scheme, key_id = resolve_request(
            extra_args, default_on=False, default_scheme=cls.SCHEME
        )

        if key_id is not None:
            resolve_key_or_raise(
                key_id, context=f"watermark_key_id={key_id!r} given but not resolvable"
            )
        elif enabled:
            resolve_key_or_raise(
                None, context="watermark=on requested but no watermark keys are configured"
            )

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.is_pin_memory = is_pin_memory

        # Same citation as KGWLogitsProcessor.__init__ -- see that method's
        # comment (vllm/config/model.py:1119-1120).
        self._vocab_size: int = vllm_config.model_config.get_vocab_size()

        self._ngram_len = int(
            os.environ.get("VLLM_WATERMARK_SYNTHID_NGRAM_LEN", _DEFAULT_NGRAM_LEN_ENV)
        )
        self._sampling_table_size = int(
            os.environ.get(
                "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SIZE", _DEFAULT_SAMPLING_TABLE_SIZE_ENV
            )
        )
        self._sampling_table_seed = int(
            os.environ.get(
                "VLLM_WATERMARK_SYNTHID_SAMPLING_TABLE_SEED", _DEFAULT_SAMPLING_TABLE_SEED_ENV
            )
        )
        self._context_history_size = int(
            os.environ.get(
                "VLLM_WATERMARK_SYNTHID_CONTEXT_HISTORY_SIZE", _DEFAULT_CONTEXT_HISTORY_SIZE_ENV
            )
        )
        self._skip_first_ngram_calls = parse_watermark_flag(
            os.environ.get(
                "VLLM_WATERMARK_SYNTHID_SKIP_FIRST_NGRAM_CALLS",
                _DEFAULT_SKIP_FIRST_NGRAM_CALLS_ENV,
            )
        )
        self._key_depth = int(
            os.environ.get("VLLM_WATERMARK_SYNTHID_KEY_DEPTH", _DEFAULT_KEY_DEPTH_ENV)
        )
        if self._key_depth <= 0:
            raise ValueError(
                f"VLLM_WATERMARK_SYNTHID_KEY_DEPTH must be positive, got {self._key_depth}"
            )

        # Both env vars are shared with vllm_watermark.kgw.processor (same
        # names, same defaults, same parsing) -- see
        # vllm_watermark.request_args module docstring.
        self._default_on = resolve_default_on()
        self._default_scheme = resolve_default_scheme()

        # Keys are read from env ONCE here, at engine init -- identical
        # graceful-degradation behavior to KGWLogitsProcessor.__init__ (see
        # that method's comment: __init__ must not raise if no keys are
        # configured; validate_params() is what rejects an unresolvable
        # watermark=on request at request-validation time instead).
        try:
            self._keys: dict[str, WatermarkKey] = load_keys()
        except RuntimeError:
            self._keys = {}
        try:
            self._default_key: "WatermarkKey | None" = (
                load_key(key_id=None) if self._keys else None
            )
        except (RuntimeError, KeyError):
            self._default_key = None

        # Sparse per-row state -- same rationale as KGWLogitsProcessor.
        self._rows: "dict[int, RowState]" = {}

        # Logged once, not per apply() call -- see _check_vocab_width().
        self._vocab_width_checked = False

        logger.info(
            "SynthIDLogitsProcessor initialized: vocab_size=%d ngram_len=%d "
            "sampling_table_size=%d sampling_table_seed=%d "
            "context_history_size=%d skip_first_ngram_calls=%s key_depth=%d "
            "default=%s scheme_default=%s configured_key_ids=%s default_key_id=%s",
            self._vocab_size,
            self._ngram_len,
            self._sampling_table_size,
            self._sampling_table_seed,
            self._context_history_size,
            self._skip_first_ngram_calls,
            self._key_depth,
            self._default_on,
            self._default_scheme,
            sorted(self._keys),
            self._default_key.key_id if self._default_key else None,
        )

    def is_argmax_invariant(self) -> bool:
        """SynthID's tournament reweighting (`probs *= (1 + g - g_mass)`
        per depth layer, see synthid/core.py `_update_scores`) can and
        does shift which token has the highest resulting probability -- it
        is non-distortionary only in EXPECTATION over the sampling
        distribution, not per-token-deterministically -- so this is never
        argmax-invariant (same conclusion as KGWLogitsProcessor, different
        underlying reason)."""
        return False

    def _new_row_state(
        self,
        params: "SamplingParams",
        prompt_tok_ids: "list[int] | None",
        output_tok_ids: "list[int]",
    ) -> "RowState | None":
        """Return the RowState for a newly-added request, or None if the
        watermark should not be applied to it BY THIS PROCESSOR -- see
        `KGWLogitsProcessor._new_row_state()`'s docstring for the identical
        "disabled, or enabled-but-other-scheme" distinction (not repeated
        here)."""
        extra_args = params.extra_args or {}
        enabled, scheme, key_id = resolve_request(
            extra_args, self._default_on, self._default_scheme
        )
        if not enabled or scheme != self.SCHEME:
            return None

        key = self._keys.get(key_id) if key_id is not None else self._default_key
        if key is None:
            # Defensive only -- see KGWLogitsProcessor._new_row_state()'s
            # identical comment for why this should not be reachable in
            # practice.
            logger.warning(
                "SynthIDLogitsProcessor: watermark requested but key_id=%r did "
                "not resolve against configured keys %s; treating this row as "
                "unwatermarked. This should have been rejected by "
                "validate_params().",
                key_id,
                sorted(self._keys),
            )
            return None

        keys = key.derive_subkeys(self._key_depth, _SYNTHID_KEY_LABEL)

        return RowState(
            enabled=True,
            key_id=key.key_id,
            keys=keys,
            prompt_tok_ids=prompt_tok_ids,
            output_tok_ids=output_tok_ids,
            context_history=_ContextHistory(maxlen=self._context_history_size),
        )

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        """Maintain self._rows following the added/removed/moved protocol.

        Identical logic (and identical "added, then removed, then moved"
        ordering rationale) to
        `KGWLogitsProcessor.update_state()` -- see that method's docstring
        for the full citation (`vllm/v1/sample/logits_processor/builtin.py`
        `process_dict_updates`, and the documented added-vs-removed-first
        prose/code discrepancy) and `docs/api-notes-vllm-v0.18.0.md` §2.
        Reproduced here (rather than shared via inheritance) to keep both
        processor files independently complete and readable, matching the
        existing repo convention.
        """
        if not batch_update:
            return

        for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
            state = self._new_row_state(params, prompt_tok_ids, output_tok_ids)
            if state is not None:
                self._rows[index] = state
            else:
                self._rows.pop(index, None)

        if self._rows:
            for index in batch_update.removed:
                self._rows.pop(index, None)

            for a_index, b_index, direct in batch_update.moved:
                a_entry = self._rows.pop(a_index, None)
                b_entry = self._rows.pop(b_index, None)
                if a_entry is not None:
                    self._rows[b_index] = a_entry
                if b_entry is not None and direct == MoveDirectionality.SWAP:
                    self._rows[a_index] = b_entry

    def _row_context(self, row: "RowState") -> "tuple[list[int], bool]":
        """Compute this row's `(ngram_len - 1)`-token context at the
        current apply() call, and whether this row should be SKIPPED this
        call. See class docstring "DESIGN DECISION 1" (prompt/output
        boundary) and "DESIGN DECISION 2" (insufficient real context:
        skip vs. zero-pad).

        Returns:
            `(ngram_context, skip)`. If `skip` is True, `ngram_context` is
            `[]` and must be ignored -- caller must not invoke
            `process_scores_row()` this call for this row. Otherwise
            `ngram_context` has EXACTLY `self._ngram_len - 1` entries
            (`vllm_watermark.synthid.core.g_values()` requires this exact
            length, see that function's docstring).
        """
        needed = self._ngram_len - 1
        if needed <= 0:
            return [], False

        tail = row.output_tok_ids[-needed:] if row.output_tok_ids else []
        if len(tail) < needed:
            remaining = needed - len(tail)
            prompt = row.prompt_tok_ids or []
            prompt_tail = prompt[-remaining:] if prompt else []
            tail = prompt_tail + tail

        if len(tail) < needed:
            if self._skip_first_ngram_calls:
                return [], True
            # Zero-pad on the left -- mirrors transformers' own
            # zero-initialized context for insufficient history (see class
            # docstring "DESIGN DECISION 2").
            tail = [0] * (needed - len(tail)) + tail

        return tail, False

    def _check_vocab_width(self, vocab_width: int) -> None:
        """Warn (once) if logits.shape[-1] disagrees with the configured
        vocab_size, and assert the direction that would be genuinely
        unsafe. Identical rationale to
        `KGWLogitsProcessor._check_vocab_width()` (see that method's
        docstring -- not repeated here); the "narrower than configured"
        case is unsafe here for a different reason than KGW's (KGW: green-
        list ids could reference out-of-range columns; here: `apply()`
        slices `logits[index, :self._vocab_size]` to build the row
        `process_scores_row()` biases, so a narrower actual tensor would
        silently bias fewer columns than intended -- see `apply()`).
        """
        if self._vocab_width_checked:
            return
        self._vocab_width_checked = True
        if vocab_width != self._vocab_size:
            logger.warning(
                "SynthIDLogitsProcessor: logits width (%d) != configured "
                "vocab_size (%d) from model_config.get_vocab_size(); this is "
                "expected when the model pads its embedding matrix, but if "
                "width < vocab_size the watermark signal will be weakened. "
                "Logged once per process.",
                vocab_width,
                self._vocab_size,
            )
        assert self._vocab_size <= vocab_width, (
            f"SynthIDLogitsProcessor: configured vocab_size ({self._vocab_size}) "
            f"exceeds logits width ({vocab_width}); model_config.get_vocab_size() "
            "disagrees with the actual logits tensor width."
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._rows:
            return logits

        vocab_width = logits.shape[-1]
        self._check_vocab_width(vocab_width)
        num_rows = logits.shape[0]
        width = min(vocab_width, self._vocab_size)

        for index, row in self._rows.items():
            if index >= num_rows:
                # Defensive: see KGWLogitsProcessor.apply()'s identical
                # comment.
                continue

            ngram_context, skip = self._row_context(row)
            if skip:
                continue

            context_seen = row.context_history.push(tuple(ngram_context))

            cfg = SynthIDConfig(
                vocab_size=self._vocab_size,
                keys=row.keys,
                ngram_len=self._ngram_len,
                sampling_table_size=self._sampling_table_size,
                sampling_table_seed=self._sampling_table_seed,
                context_history_size=self._context_history_size,
                skip_first_ngram_calls=self._skip_first_ngram_calls,
            )

            # Runs ON logits.device (GPU in the vLLM serving path): the
            # g-value path is exact integer arithmetic + a bit-exact moved
            # lookup table, so GPU generation and CPU detection compute
            # identical g-values (see core.g_values device notes). Compute
            # in float32 for numerical parity with the CPU-verified
            # reference tests regardless of the sampler's logits dtype.
            # Rationale: CPU compute measured 291 ms/row/step at
            # vocab 151936 / depth 30 (EXPERIMENTS.md 2026-08-08) — the
            # device-native path is what makes SynthID servable at all.
            scores_row = logits[index, :width].to(dtype=torch.float32)
            updated = process_scores_row(scores_row, ngram_context, cfg, context_seen)
            logits[index, :width] = updated.to(dtype=logits.dtype)

        return logits
