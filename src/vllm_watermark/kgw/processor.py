# SPDX-License-Identifier: Apache-2.0
"""KGW watermark vLLM V1 LogitsProcessor plugin (Phase 1 wrapper).

This module imports `vllm`; the package top level and algorithm modules stay
vllm-free so local detector execution does not require vLLM. `vllm` is
not installed on the local dev workstation (see docs/cluster.md). The actual
KGW hashing/greenlist math lives in vllm-free `vllm_watermark.kgw.core`
(the Phase 1 algorithm); key loading lives in vllm-free
`vllm_watermark.keys`.
This module is pure wiring: vLLM's V1 custom-LogitsProcessor ABC <->
KGWConfig/greenlist_ids/WatermarkKey.

Everything below was verified by fetching vLLM's actual v0.18.0 tagged
source (RHOAI 3.4 target per docs/facts.md C1), not from memory:

    git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
    -> resolved commit bcf2be96120005e9aea171927f85055a6a5c0cf6
       (verified: gh api repos/vllm-project/vllm/git/refs/tags/v0.18.0)

Full citation list with line numbers: docs/api-notes-vllm-v0.18.0.md (STATIC,
this whole file's design decisions are cross-referenced there).

Quick-reference citations (file paths relative to the vLLM repo root at the
v0.18.0 tag above; all under Apache-2.0, "Copyright contributors to the vLLM
project"):

  * LogitsProcessor ABC, BatchUpdate, MoveDirectionality:
      vllm/v1/sample/logits_processor/interface.py
  * process_dict_updates() sparse-dict added/removed/moved reference
    pattern (ADAPTED below with attribution, not imported -- see
    update_state() docstring for why):
      vllm/v1/sample/logits_processor/builtin.py, lines 294-332
  * Entry-point group name "vllm.logits_processors", FQCN "module:Class"
    loading format:
      vllm/v1/sample/logits_processor/__init__.py
      (LOGITSPROCS_GROUP = "vllm.logits_processors", line 47;
       `module_path, qualname = logitproc.split(":")`, ~line 128)
  * ModelConfig.get_vocab_size() -> int (returns
    self.model_arch_config.vocab_size):
      vllm/config/model.py, lines 1119-1120
  * validate_params() is called per-request via SamplingParams.verify() ->
    _validate_logits_processors() -> validate_logits_processors_parameters(),
    itself invoked from InputProcessor._validate_params() during request
    processing; a plain ValueError raised there is turned into an HTTP 400
    ("BadRequestError") by entrypoints/utils.py's create_error_response():
      vllm/sampling_params.py, lines 609-620 (verify()), 672-677
        (_validate_logits_processors())
      vllm/v1/engine/input_processor.py, lines 83-123 (_validate_params()),
        line 201 (call site)
      vllm/entrypoints/utils.py, lines 300-345 (create_error_response(); the
        `isinstance(exc, (ValueError, TypeError, OverflowError))` branch ->
        BadRequestError / HTTPStatus.BAD_REQUEST)
    See docs/api-notes-vllm-v0.18.0.md "Per-request error surfacing" for the
    full chain and why this resolves the "validate_params is a @classmethod,
    can it see env-derived config?" question from the task brief.
  * vllm_xargs -> SamplingParams.extra_args:
      vllm/entrypoints/openai/chat_completion/protocol.py, line 339
        (`vllm_xargs` field), line 488 (`to_sampling_params()` copies it into
        `extra_args`)
      vllm/entrypoints/openai/completion/protocol.py, line 162 / line 292
        (same pattern for the legacy /v1/completions request)
  * Custom logits processors are hard-incompatible with speculative
    decoding (matches docs/facts.md B7):
      vllm/v1/sample/logits_processor/__init__.py, lines 43-45
        (STR_SPEC_DEC_REJECTS_LOGITSPROCS), lines 200-209 (raised in
        build_logitsprocs())

Docs page cross-checked (fetched 2026-08-08):
    https://docs.vllm.ai/en/latest/features/custom_logitsprocs/
Its `DummyLogitsProcessor` example implements update_state() in
added-then-removed-then-moved order, confirming the order actually used by
the shipped builtin.py reference implementation -- see the "batch-update
order" note in update_state()'s docstring below for a documented
discrepancy between that and BatchUpdate's own prose docstring.

Env vars this module reads directly (never logged/printed as values):
    VLLM_WATERMARK_DEFAULT  "on"/"off" (default "off") -- whether requests
        that don't pass a `watermark` extra_arg get watermarked. Shared
        (same name, same default, same parsing -- via
        vllm_watermark.request_args) with
        vllm_watermark.synthid.processor.
    VLLM_WATERMARK_SCHEME   "kgw"/"synthid" (default "kgw") -- which
        scheme requests that don't pass a `watermark_scheme` extra_arg
        resolve to. A row only ever activates in THIS processor if the
        resolved scheme == "kgw" -- see class docstring
        "SCHEME-COORDINATION DESIGN". Shared with
        vllm_watermark.synthid.processor (same name/default/parsing).
    VLLM_WATERMARK_GAMMA    float in (0, 1), default "0.25"
    VLLM_WATERMARK_DELTA    float, default "2.0"
    VLLM_WATERMARK_CACHE_SIZE  int >= 0, default "1024" -- LRU size of the
        greenlist memo cache, keyed (hash_key, prev_token). Pure
        memoization of greenlist_ids() (identical outputs, measured-equal
        statistics); exists because torch.randperm(vocab_size) on CPU costs
        ~7 ms at vocab 151936 and dominated decode latency in the Phase 1
        benchmark (3.2x slowdown uncached, see EXPERIMENTS.md 2026-08-08).
        Zipfian token frequency makes a small LRU highly effective. Memory
        bound: one CPU int64 tensor of gamma*vocab_size ids per entry
        (~300 KB at gamma 0.25 / vocab 151936 -> ~300 MB at 1024 entries).
        0 disables caching.
    (key material itself is loaded via vllm_watermark.keys -- WATERMARK_KEYS
    / WATERMARK_KEY / WATERMARK_KEY_ID -- never read directly by this file)

Per-request `vllm_xargs` / `SamplingParams.extra_args` keys this processor
recognizes:
    watermark          "on"/"off" (or a JSON bool) -- overrides
                        VLLM_WATERMARK_DEFAULT for this request
    watermark_key_id   str, non-empty -- which configured key to use
                        (default: vllm_watermark.keys' own default-key
                        resolution, i.e. WATERMARK_KEY_ID env or "default")
    watermark_scheme   "kgw" | "synthid" -- overrides VLLM_WATERMARK_SCHEME
                        for this request; this processor only ever biases
                        a row whose resolved scheme == SCHEME ("kgw") --
                        see _new_row_state() and
                        vllm_watermark.request_args module docstring
                        "SCHEME-COORDINATION DESIGN".
Any other `watermark*`-prefixed key is rejected by validate_params() as an
unknown argument (fail loud on typos rather than silently ignoring them).
The parsing/validation of all three keys above is shared with
vllm_watermark.synthid.processor.SynthIDLogitsProcessor via
vllm_watermark.request_args (single implementation, so the two
validate_params() methods cannot silently disagree about the same
extra_args -- see that module's docstring for why this matters given vLLM
calls validate_params() on every loaded logits processor for every
request, per docs/api-notes-vllm-v0.18.0.md §6).
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

from vllm_watermark.keys import WatermarkKey, load_key, load_keys
from vllm_watermark.kgw.core import KGWConfig, greenlist_ids
from vllm_watermark.request_args import (
    resolve_default_on,
    resolve_default_scheme,
    resolve_key_or_raise,
    resolve_request,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

__all__ = ["KGWLogitsProcessor", "RowState"]

_DEFAULT_GAMMA_ENV = "0.25"
_DEFAULT_DELTA_ENV = "2.0"


@dataclass
class RowState:
    """Per-batch-row watermark state. Only rows with the watermark enabled
    for this request ever get an entry -- see update_state()/apply().

    `prompt_tok_ids` / `output_tok_ids` are the *same list objects* vLLM's
    BatchUpdate.added tuples carry (see interface.py BatchUpdate docstring:
    "the `output_tok_ids` list ... is a reference to the request's running
    output tokens list"). We deliberately store the reference, not a copy,
    so apply() always sees the latest generated tokens without us needing
    to be told about every new token explicitly.
    """

    enabled: bool
    key_id: str
    hash_key: int
    prompt_tok_ids: "list[int] | None"
    output_tok_ids: "list[int]"


class KGWLogitsProcessor(LogitsProcessor):
    """vLLM V1 custom LogitsProcessor applying KGW green-list bias.

    Subclasses `vllm.v1.sample.logits_processor.LogitsProcessor` (the public
    plugin ABC re-exported from `vllm.v1.sample.logits_processor.interface`;
    see vllm/v1/sample/logits_processor/__init__.py `__all__`). Loaded either
    via the `vllm.logits_processors` entry-point group (see pyproject.toml)
    or via `--logits-processors vllm_watermark.kgw.processor:KGWLogitsProcessor`.

    SCHEME-COORDINATION DESIGN (see vllm_watermark.request_args module
    docstring for the full rationale): both this class and
    vllm_watermark.synthid.processor.SynthIDLogitsProcessor can be loaded
    into the same engine at once. Each sets a class attribute `SCHEME`; a
    batch row is only ever added to THIS processor's self._rows if the
    request's resolved scheme (per-request `watermark_scheme` extra_arg, or
    VLLM_WATERMARK_SCHEME env default) equals `SCHEME` -- see
    _new_row_state(). Exactly one of the two loaded processors ever biases
    any given row, even though both receive update_state()/apply() calls
    for every row in the batch.
    """

    SCHEME = "kgw"

    @classmethod
    def validate_params(cls, sampling_params: "SamplingParams") -> None:
        """Reject malformed/unresolvable watermark_* extra_args.

        Called once per incoming request (before it ever reaches
        update_state()/apply()) via SamplingParams.verify() -- see module
        docstring "Per-request error surfacing" for the full call chain and
        why a ValueError raised here becomes an HTTP 400 to the API caller.

        This is a @classmethod (mandated by
        vllm.v1.sample.logits_processor.interface.LogitsProcessor;
        interface.py lines 61-67), so it has no access to a *specific*
        instance's already-loaded key cache (self._keys / self._default_key,
        set in __init__ below). That's fine: watermark key material is
        process-global env-var configuration (see vllm_watermark.keys), not
        per-instance state, and there is exactly one KGWLogitsProcessor
        instance per engine in practice (build_logitsprocs() constructs each
        configured logits-processor class exactly once -- vllm/v1/sample/
        logits_processor/__init__.py build_logitsprocs()). So this method
        re-resolves key configuration directly via vllm_watermark.keys
        (which itself reads env) rather than trying to reach into an
        instance. The cost (env read + hex-decode + sha256 per configured
        key) is paid once per incoming request, not per token/per step, so
        it is negligible.
        """
        extra_args = sampling_params.extra_args or {}

        # default_on=False here (not self._default_on -- this is a
        # classmethod, there is no self) reproduces exactly the prior
        # "requested_on" semantics: `enabled` comes out True iff the
        # request's `watermark` extra_arg is explicitly present and parses
        # truthy, regardless of VLLM_WATERMARK_DEFAULT. See
        # vllm_watermark.request_args.resolve_request() docstring
        # "default_on" for why passing False here is exactly this, and
        # module docstring "Per-request error surfacing" above for why an
        # implicit (env-defaulted) enable is deliberately NOT
        # resolved-key-checked at request-validation time -- unchanged
        # behavior from before this function was routed through
        # resolve_request(). default_scheme is irrelevant to what this
        # method validates (key resolvability does not depend on scheme --
        # both schemes share the same vllm_watermark.keys store) beyond
        # resolve_request() rejecting a malformed `watermark_scheme` value,
        # so `cls.SCHEME` is passed only for that validation, not consumed
        # further here.
        enabled, _scheme, key_id = resolve_request(
            extra_args, default_on=False, default_scheme=cls.SCHEME
        )

        # Fail loudly at request-validation time (before generation starts)
        # rather than silently degrading to unwatermarked output later in
        # update_state(). This is the mechanism the task brief asked us to
        # identify: "a request explicitly asking watermark=on gets a clear
        # ValueError at validate time."
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

        # ModelConfig.get_vocab_size() -> self.model_arch_config.vocab_size;
        # see module docstring citation (vllm/config/model.py:1119-1120).
        # Deliberately NOT inferred from a logits tensor width or tokenizer
        # length -- see vllm_watermark.kgw.core module docstring
        # "KGWConfig.vocab_size is REQUIRED and must be passed explicitly"
        # for why that distinction matters for detector agreement.
        self._vocab_size: int = vllm_config.model_config.get_vocab_size()

        self._gamma = float(os.environ.get("VLLM_WATERMARK_GAMMA", _DEFAULT_GAMMA_ENV))
        self._delta = float(os.environ.get("VLLM_WATERMARK_DELTA", _DEFAULT_DELTA_ENV))
        # Both env vars are shared with vllm_watermark.synthid.processor
        # (same names, same defaults, same parsing) -- see
        # vllm_watermark.request_args module docstring.
        self._default_on = resolve_default_on()
        self._default_scheme = resolve_default_scheme()

        # Keys are read from env ONCE here, at engine init, and cached for
        # the life of this instance -- consistent with the documented fact
        # that "the processor set is immutable after engine init" (see
        # docs/facts.md B4). validate_params() (a classmethod, see above)
        # re-reads env per-request instead, since it cannot see this cache.
        #
        # Graceful degradation (task requirement): if no keys are
        # configured at all, __init__ must NOT raise -- the engine must
        # still start (with every request treated as unwatermarked unless
        # it explicitly asks for watermark=on, which validate_params()
        # rejects with a clear per-request 400 instead, per above).
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

        # Sparse per-row state: rows with the watermark disabled for this
        # request are never inserted, so apply()/update_state() do zero
        # work when nothing in the batch is watermarked (task requirement;
        # also vLLM's own documented best practice for infrequently-used
        # logits processors -- see docs/api-notes "sparse representation").
        self._rows: "dict[int, RowState]" = {}

        # Greenlist LRU memo cache -- see module docstring
        # VLLM_WATERMARK_CACHE_SIZE. Pure memoization: values are exactly
        # greenlist_ids(prev_token, cfg) results (CPU LongTensor), keyed by
        # everything the result depends on besides process-global gamma/
        # vocab_size (constant for this instance): (hash_key, prev_token).
        self._cache_size = int(os.environ.get("VLLM_WATERMARK_CACHE_SIZE", "1024"))
        if self._cache_size < 0:
            raise ValueError(
                f"VLLM_WATERMARK_CACHE_SIZE must be >= 0, got {self._cache_size}"
            )
        self._greenlist_cache: "OrderedDict[tuple[int, int], torch.Tensor]" = OrderedDict()

        # Logged once, not per apply() call -- see _check_vocab_width().
        self._vocab_width_checked = False

        logger.info(
            "KGWLogitsProcessor initialized: vocab_size=%d gamma=%.4f delta=%.4f "
            "default=%s configured_key_ids=%s default_key_id=%s",
            self._vocab_size,
            self._gamma,
            self._delta,
            self._default_on,
            sorted(self._keys),
            self._default_key.key_id if self._default_key else None,
        )

    def is_argmax_invariant(self) -> bool:
        """Adding delta to green-list logits can change which token has the
        highest logit, so this is never argmax-invariant."""
        return False

    def _new_row_state(
        self,
        params: "SamplingParams",
        prompt_tok_ids: "list[int] | None",
        output_tok_ids: "list[int]",
    ) -> "RowState | None":
        """Return the RowState for a newly-added request, or None if the
        watermark should not be applied to it BY THIS PROCESSOR (row is
        then absent from self._rows -- see class docstring "sparse" note).

        "Not applied by this processor" covers two cases: watermarking is
        disabled for the request, OR it is enabled but resolved to the
        OTHER scheme (e.g. `watermark_scheme=synthid`) -- see class
        docstring "SCHEME-COORDINATION DESIGN". In the second case
        SynthIDLogitsProcessor's own _new_row_state() (if that processor is
        also loaded) is what adds the row to ITS self._rows instead.
        """
        extra_args = params.extra_args or {}
        enabled, scheme, key_id = resolve_request(
            extra_args, self._default_on, self._default_scheme
        )
        if not enabled or scheme != self.SCHEME:
            return None

        key = self._keys.get(key_id) if key_id is not None else self._default_key
        if key is None:
            # Defensive only: validate_params() (called for every request
            # before it can reach here -- see module docstring "Per-request
            # error surfacing") should already have rejected an
            # unresolvable watermark_key_id or a watermark=on request with
            # no keys configured. If we get here anyway (e.g. keys were
            # reconfigured between validate_params() and update_state(),
            # which should not happen since the processor set -- and this
            # instance's key cache -- is immutable after engine init per
            # docs/facts.md B4), fail closed: treat the row as
            # unwatermarked rather than raising deep inside the batch-update
            # path, where a raised exception would break the whole batch.
            logger.warning(
                "KGWLogitsProcessor: watermark requested but key_id=%r did not "
                "resolve against configured keys %s; treating this row as "
                "unwatermarked. This should have been rejected by "
                "validate_params().",
                key_id,
                sorted(self._keys),
            )
            return None

        return RowState(
            enabled=True,
            key_id=key.key_id,
            hash_key=key.hash_key,
            prompt_tok_ids=prompt_tok_ids,
            output_tok_ids=output_tok_ids,
        )

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        """Maintain self._rows following the added/removed/moved protocol.

        Adapted (not imported) from vllm.v1.sample.logits_processor.builtin.
        process_dict_updates (v0.18.0, vllm/v1/sample/logits_processor/
        builtin.py lines 294-332, Apache-2.0, "Copyright contributors to the
        vLLM project") -- the reference pattern vLLM's own sparse-state
        builtins (LogitBiasLogitsProcessor, MinTokensLogitsProcessor) use.
        Adapted rather than imported because process_dict_updates is not
        re-exported from vllm.v1.sample.logits_processor's public __all__
        (vllm/v1/sample/logits_processor/__init__.py) and so is not part of
        the documented stable plugin surface; the *behavior* is copied
        faithfully with attribution per AGENTS.md licensing rules, the
        *dependency* on an unstable internal symbol is not.

        Batch-update processing order -- a documented discrepancy:
        BatchUpdate's own docstring (interface.py, "NOTE" block) states
        operations "should be processed in the following order: removed,
        added, moved", and vLLM's docs page (https://docs.vllm.ai/en/latest/
        features/custom_logitsprocs/, "Notes" list) repeats that same
        prose. However BOTH vLLM's actual shipped reference implementation
        (process_dict_updates, cited above) AND that same docs page's own
        `DummyLogitsProcessor` example code process ADDED first, then
        REMOVED, then MOVED. We follow the code (added, removed, moved),
        not the prose, because it is what the tested/shipped reference
        processors (LogitBiasLogitsProcessor, MinTokensLogitsProcessor,
        MinPLogitsProcessor) actually do, and matches the officially
        documented example verbatim. See docs/api-notes-vllm-v0.18.0.md for
        the full discussion.
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

    def _greenlist_ids_cached(self, hash_key: int, prev_token: int) -> torch.Tensor:
        """greenlist_ids() with an LRU memo -- identical outputs by
        construction (the cached value IS a previous greenlist_ids() return
        for the same (hash_key, prev_token) and the same process-global
        gamma/vocab_size). Motivation + sizing: module docstring
        VLLM_WATERMARK_CACHE_SIZE. Callers must not mutate the returned
        tensor (apply() only reads it / copies it to the GPU)."""
        if self._cache_size == 0:
            return greenlist_ids(
                prev_token,
                KGWConfig(
                    vocab_size=self._vocab_size,
                    hash_key=hash_key,
                    gamma=self._gamma,
                    delta=self._delta,
                ),
            )
        cache_key = (hash_key, prev_token)
        ids = self._greenlist_cache.get(cache_key)
        if ids is not None:
            self._greenlist_cache.move_to_end(cache_key)
            return ids
        ids = greenlist_ids(
            prev_token,
            KGWConfig(
                vocab_size=self._vocab_size,
                hash_key=hash_key,
                gamma=self._gamma,
                delta=self._delta,
            ),
        )
        self._greenlist_cache[cache_key] = ids
        if len(self._greenlist_cache) > self._cache_size:
            self._greenlist_cache.popitem(last=False)
        return ids

    def _check_vocab_width(self, vocab_width: int) -> None:
        """Warn (once) if logits.shape[-1] disagrees with the configured
        vocab_size, and assert the direction that would be genuinely unsafe.

        A wider logits tensor than vocab_size (padded embedding matrix) is
        expected/harmless: green-list ids are always < self._vocab_size, so
        they are always in-bounds for a wider tensor -- see
        vllm_watermark.kgw.core module docstring on why generation and
        detection must agree on the *model* vocab_size, not a padded width.
        A NARROWER logits tensor than vocab_size would be a genuine
        misconfiguration (green-list ids could reference out-of-range
        columns) -- assert catches that in dev/test; apply() also caps ids
        defensively at runtime regardless (assert is compiled out under
        `python -O`, so it cannot be the only guard -- see apply()).
        """
        if self._vocab_width_checked:
            return
        self._vocab_width_checked = True
        if vocab_width != self._vocab_size:
            logger.warning(
                "KGWLogitsProcessor: logits width (%d) != configured "
                "vocab_size (%d) from model_config.get_vocab_size(); this is "
                "expected when the model pads its embedding matrix, but if "
                "width < vocab_size the watermark signal will be weakened "
                "by id capping. Logged once per process.",
                vocab_width,
                self._vocab_size,
            )
        assert self._vocab_size <= vocab_width, (
            f"KGWLogitsProcessor: configured vocab_size ({self._vocab_size}) "
            f"exceeds logits width ({vocab_width}); green-list ids would "
            "reference out-of-range columns. model_config.get_vocab_size() "
            "disagrees with the actual logits tensor width."
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._rows:
            return logits

        vocab_width = logits.shape[-1]
        self._check_vocab_width(vocab_width)
        num_rows = logits.shape[0]

        for index, row in self._rows.items():
            if index >= num_rows:
                # Defensive: should not happen if update_state() tracked the
                # batch correctly (see docstring above), but never index
                # past a batch that shrank underneath us.
                continue

            if row.output_tok_ids:
                prev_token = row.output_tok_ids[-1]
            elif row.prompt_tok_ids:
                prev_token = row.prompt_tok_ids[-1]
            else:
                # No token history at all to seed the greenlist from
                # (e.g. an empty prompt with prompt_tok_ids=None -- see
                # interface.py AddedRequest typing). Skip defensively.
                continue

            # Built fresh per row per step rather than cached per key_id:
            # KGWConfig's __post_init__ is pure-Python validation + one int
            # multiply (no tensor ops), so the cost is negligible next to
            # greenlist_ids()'s torch.randperm below; caching would risk
            # staleness if gamma/delta/vocab_size ever became per-request
            # instead of process-global. See vllm_watermark.kgw.core for the
            # KGWConfig contract.
            ids = self._greenlist_ids_cached(row.hash_key, prev_token)  # CPU LongTensor

            if vocab_width < self._vocab_size:
                # Only reachable if the _check_vocab_width() assert above
                # was compiled out (python -O). Defense in depth, not the
                # primary guard -- see _check_vocab_width() docstring.
                ids = ids[ids < vocab_width]
            if ids.numel() == 0:
                continue

            logits[index, ids.to(device=logits.device, non_blocking=True)] += self._delta

        return logits
