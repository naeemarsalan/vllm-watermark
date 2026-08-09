# vLLM v0.18.0 custom-LogitsProcessor API notes

**Mixed `STATIC` / `OFFICIAL-SRC` / `EXECUTED`.** API descriptions come
from the v0.18.0-tagged source and official vLLM documentation fetched
2026-08-08. Plugin loading, HTTP-400 request validation, and speculative-
decoding rejection were also executed; those claims link to preserved
commands and raw output in [`EXPERIMENTS.md`](../EXPERIMENTS.md).

```
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
-> resolved commit: bcf2be96120005e9aea171927f85055a6a5c0cf6
   (confirmed separately: gh api repos/vllm-project/vllm/git/refs/tags/v0.18.0
    -> object.sha == bcf2be96120005e9aea171927f85055a6a5c0cf6)
```

All `vllm/...` file paths and line numbers below are relative to that
commit. All cited vLLM files carry `# SPDX-License-Identifier: Apache-2.0`
/ `# SPDX-FileCopyrightText: Copyright contributors to the vLLM project`.

This page documents the vLLM-facing processor interface. It does not cover
the KGW detector algorithm or the detection-service contract.

---

## 1. `LogitsProcessor` ABC

Source: `vllm/v1/sample/logits_processor/interface.py` (106 lines), re-exported
publicly from `vllm.v1.sample.logits_processor`.

```python
class MoveDirectionality(Enum):
    UNIDIRECTIONAL = auto()   # one-way i1->i2 move within batch
    SWAP = auto()             # two-way i1<->i2 swap within batch

RemovedRequest = int
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]
MovedRequest = tuple[int, int, MoveDirectionality]

@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]

class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.
        Raise ValueError for invalid ones."""
        return None

    @abstractmethod
    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool) -> None: ...

    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def is_argmax_invariant(self) -> bool: ...

    @abstractmethod
    def update_state(self, batch_update: "BatchUpdate | None") -> None: ...
```

`AddedRequest` carries the request's `prompt_tok_ids` (may be `None`) and
`output_tok_ids` **by reference** — the `BatchUpdate` docstring states this
explicitly:

> "the `output_tok_ids` list ... is an element of each tuple in `added`) is
> a reference to the request's running output tokens list; via this
> reference, the logits processors always see the latest list of generated
> output tokens."

`KGWLogitsProcessor.RowState` relies on exactly this: it stores the list
objects, not copies, so `apply()` always sees the latest generated tokens
without needing per-token notifications.

## 2. Batch-update processing order — a documented discrepancy

`BatchUpdate`'s own docstring (`interface.py`) says:

> "Operations should be processed in the following order: — removed,
> added, moved"

The live docs page (§5 below) repeats this prose verbatim in its "Notes"
list: *"A logits processor `update_state()` method must process batch
update operations in the following order: removes, adds, moves"*.

**However**, both of the following actually process **added, then removed,
then moved** — the opposite order for added/removed:

- `vllm/v1/sample/logits_processor/builtin.py`'s `process_dict_updates()`
  (lines 294-332), the shared helper `LogitBiasLogitsProcessor` and
  `MinTokensLogitsProcessor` both use for their sparse per-request dict
  state:
  ```python
  for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
      ...
  if req_entries:
      for index in batch_update.removed:
          ...
      for a_index, b_index, direct in batch_update.moved:
          ...
  ```
- `MinPLogitsProcessor.update_state()` (also `builtin.py`): "Process added
  requests." first, then "Process removed requests.", then moved.
- The docs page's own `DummyLogitsProcessor` and
  `WrappedPerReqLogitsProcessor` example code (§5): identical
  added→removed→moved order.

No builtin or documented example inspected used removed-before-added.
`KGWLogitsProcessor.update_state()` follows the shipped code order: added,
removed, moved (`STATIC`; pinned-source comparison). The source does not
explicitly state whether `removed` and `added` can target the same index in
one update, so safety under that case remains `OPEN`; no missing invariant is
presented as fact.

## 3. Loading a plugin: entry-point group name and FQCN format

Source: `vllm/v1/sample/logits_processor/__init__.py` (357 lines).

```python
LOGITSPROCS_GROUP = "vllm.logits_processors"   # line 47
```

This is the exact `[project.entry-points."..."]` group name used in this
repo's `pyproject.toml`.

FQCN loading (`_load_logitsprocs_by_fqcns`, ~line 86-155), the function
`--logits-processors` values are resolved through:

```python
module_path, qualname = logitproc.split(":")   # ~line 128
...
module = importlib.import_module(module_path)
obj = module
for attr in qualname.split("."):
    obj = getattr(obj, attr)
if not isinstance(obj, type): raise ValueError(...)
if not issubclass(obj, LogitsProcessor): raise ValueError(...)
```

**FQCN format is `<dotted.module.path>:<ClassName>` — a colon before the
class name, not a dot.** This repo's plugin FQCN is:

```
vllm_watermark.kgw.processor:KGWLogitsProcessor
```

CLI flag: the actual `argparse` registration in the pinned v0.18.0
`vllm/engine/arg_utils.py`, line 749, is (`STATIC`; pinned source revision
registered at the top of this note):

```python
model_group.add_argument("--logits-processors", **model_kwargs["logits_processors"])
```

i.e. **`--logits-processors`** (hyphenated). The live docs page (§5) shows
`--logits_processors` (underscored) in its `vllm serve` examples in two
places — a documentation inconsistency with the actual registered flag name.
No underscore-flag alias appears in the pinned `arg_utils.py` source
(`STATIC`). Use the hyphenated form.

`build_logitsprocs()` (same file, ~line 184-217) constructs one instance
per entry in the returned processor-class list at engine init:
`BUILTIN_LOGITS_PROCESSORS` (`MinTokensLogitsProcessor`,
`LogitBiasLogitsProcessor`, `MinPLogitsProcessor`) plus
`custom_logitsprocs_classes` (from `_load_custom_logitsprocs`, entry
points + FQCNs), each called as `ctor(vllm_config, device, is_pin_memory)`
(`STATIC`). The same class can occur more than once when entry-point and
FQCN loading are combined; §10 records the executed double-load.
Environment access is process-global, so `validate_params()` can read the
configured key material without relying on a processor instance.

## 4. Speculative-decoding incompatibility (matches `docs/facts.md` B7)

Same file, lines 43-45:

```python
STR_SPEC_DEC_REJECTS_LOGITSPROCS = (
    "Custom logits processors are not supported when speculative decoding is enabled."
)
```

Raised in `build_logitsprocs()`, ~lines 200-209:

```python
if vllm_config.speculative_config:
    if custom_logitsprocs:
        raise ValueError(STR_SPEC_DEC_REJECTS_LOGITSPROCS)
    ...
```

i.e. this is an engine-*startup*-time `ValueError` (config validation, not
per-request), raised only if speculative decoding is configured **and**
any custom logits processor (entry-point or FQCN) is present (`STATIC`).
The exact rejection was also observed on vLLM 0.18.0 (`EXECUTED`; [raw
record](../EXPERIMENTS.md#spec-decode-incompatibility-executed--b7-upgraded)).

## 5. Docs page cross-check

`https://docs.vllm.ai/en/latest/features/custom_logitsprocs/` (fetched
2026-08-08; page carries a banner: *"Some logits processors design changes
are still in progress and the API may change in the near future."*).

Confirms, in the docs' own prose:

> `validate_params(cls, sampling_params: SamplingParams)`: "Raise
> `ValueError` if `SamplingParams` has invalid arguments (especially custom
> arguments) used by logits processor. **When request is sent to
> entrypoint, `validate_params()` will validate `SamplingParams` and refuse
> request with invalid arguments.**"

This is the docs-level confirmation (`OFFICIAL-SRC`; fetched vLLM page above)
of the mechanism traced at the source
level in §6 below. The page's example `DummyLogitsProcessor` /
`WrappedPerReqLogitsProcessor` snippets, `vllm_xargs` REST/SDK examples,
and FQCN/entry-point loading instructions are otherwise consistent with
the pinned source (§1-§4), except for the two
discrepancies flagged above (§2 processing order, §3 CLI flag spelling).

## 6. Per-request error surfacing: how a `validate_params()` `ValueError`
becomes an HTTP 400

**Finding:** source tracing shows that `validate_params()` raising a plain
`ValueError` maps to HTTP 400 (`STATIC`), and the repository's malformed
watermark arguments were rejected with HTTP 400 (`EXECUTED`; [Phase 1
record](../EXPERIMENTS.md#per-request-control--validation-executed)). No builtin
processor defers this kind of rejection to `update_state()` — searching
`builtin.py`, none of `MinPLogitsProcessor`, `LogitBiasLogitsProcessor`,
`MinTokensLogitsProcessor` override `validate_params()` at all (they rely
on the no-op base-class default, since their arguments — `min_p`,
`logit_bias`, `min_tokens` — are plain `SamplingParams` fields already
validated elsewhere); the docs page's own worked examples (§5) both
implement `validate_params()` to raise `ValueError` for a bad
`extra_args["target_token"]`, which is the pattern `KGWLogitsProcessor`
follows. `os.environ`
(and, in this repository, `vllm_watermark.keys.load_key`/`load_keys`, which reads
`os.environ`) is process-global state, readable from anywhere including a
`@classmethod` — the classmethod just can't reach a *specific instance's*
cached state, which is not needed for this validation.

Full call chain, traced from the source (`vllm/sampling_params.py`,
`vllm/v1/engine/input_processor.py`, `vllm/entrypoints/...`), all in this
v0.18.0 checkout:

1. **`SamplingParams.verify(model_config, speculative_config,
   structured_outputs_config, tokenizer)`** — `vllm/sampling_params.py`,
   lines 609-620 — calls, among others:
   ```python
   self._validate_logits_processors(model_config)   # line 618
   ```
2. **`SamplingParams._validate_logits_processors()`** — same file, lines
   672-677:
   ```python
   def _validate_logits_processors(self, model_config: ModelConfig) -> None:
       from vllm.v1.sample.logits_processor import (
           validate_logits_processors_parameters,
       )
       validate_logits_processors_parameters(model_config.logits_processors, self)
   ```
3. **`validate_logits_processors_parameters()`** —
   `vllm/v1/sample/logits_processor/__init__.py`, lines 223-231:
   ```python
   def validate_logits_processors_parameters(logits_processors, sampling_params):
       logits_processors = tuple(logits_processors) if logits_processors is not None else None
       for logits_procs in cached_load_custom_logitsprocs(logits_processors):
           logits_procs.validate_params(sampling_params)
   ```
   This is the call site of the repository's `KGWLogitsProcessor.validate_params(cls,
   sampling_params)`.
4. **`SamplingParams.verify()` is called per-request** from
   `InputProcessor._validate_params()` —
   `vllm/v1/engine/input_processor.py`, lines 83-107 (docstring: *"Raise
   `ValueError` if `SamplingParams` or `PoolingParams` is not valid."*):
   ```python
   params.verify(self.model_config, self.speculative_config,
                 self.structured_outputs_config, self.tokenizer)   # line 96
   ```
   called from `InputProcessor.process_inputs()` at line 201, which runs
   for every incoming request before it is scheduled/generated.
5. A `ValueError` raised anywhere in that chain propagates up out of
   request processing into the async engine / OpenAI-server request
   handler, where it is caught by
   **`create_error_response()`** — `vllm/entrypoints/utils.py`, lines
   300-345:
   ```python
   elif isinstance(exc, (ValueError, TypeError, OverflowError)):
       # Common validation errors from user input
       err_type = "BadRequestError"
       status_code = HTTPStatus.BAD_REQUEST
       param = None
   ```
   returned as `JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)`
   (i.e. HTTP **400**) by the registered exception handler
   (`vllm/entrypoints/openai/server_utils.py`, `exception_handler`, lines
   371-381).

   `vllm.exceptions.VLLMValidationError` (a `ValueError` subclass, defined
   `vllm/exceptions.py` line 9) gets a slightly richer response (carries a
   `parameter` field), but a **plain `ValueError`** — which is exactly
   what `KGWLogitsProcessor.validate_params()` raises — is independently
   handled by the same `isinstance(exc, (ValueError, ...))` branch and
   also produces `BadRequestError` / HTTP 400. There is no need to raise
   the vLLM-internal `VLLMValidationError` subclass specifically.

**Conclusion applied in `processor.py`:** `KGWLogitsProcessor.validate_params()`
independently calls `vllm_watermark.keys.load_key()` (re-reading env; see
processor.py docstring for why this is cheap and correct despite being a
classmethod) and raises plain `ValueError` when a request explicitly asks
`watermark=on` (or gives an unresolvable `watermark_key_id`) but no
matching key is configured. The traced chain predicts a `BadRequestError`
before generation (`STATIC`), and the malformed, unknown-field, and unknown-
key cases in the Phase 1 matrix behaved that way (`EXECUTED`). Other
untested exception paths are not claimed ([Phase 1 validation
record](../EXPERIMENTS.md#per-request-control--validation-executed)).

## 7. `vllm_xargs` → `SamplingParams.extra_args`

Two OpenAI-compatible request schemas both expose `vllm_xargs` and copy it
into `extra_args` the same way:

- **Chat Completions** — `vllm/entrypoints/openai/chat_completion/protocol.py`,
  class `ChatCompletionRequest` (starts line 150):
  ```python
  vllm_xargs: dict[str, str | int | float | list[str | int | float]] | None = Field(
      default=None, ...)   # line 339
  ```
  `to_sampling_params()` (starts line 424):
  ```python
  extra_args: dict[str, Any] = self.vllm_xargs if self.vllm_xargs else {}   # line 488
  if self.kv_transfer_params:
      extra_args["kv_transfer_params"] = self.kv_transfer_params
  return SamplingParams.from_optional(
      ...,
      extra_args=extra_args or None,   # line 519
      ...
  )
  ```
- **Legacy Completions** — `vllm/entrypoints/openai/completion/protocol.py`:
  `vllm_xargs` field at line 162, identical `extra_args` copy at line 292.

`SamplingParams.extra_args: dict[str, Any] | None = None` —
`vllm/sampling_params.py`, line 271 (field), line 320 (constructor param),
line 360 (`from_optional` passthrough).

Note `vllm.entrypoints.openai...` may **add** to `extra_args` itself
(`kv_transfer_params`, above) — this is why
`KGWLogitsProcessor.validate_params()` only rejects unrecognized keys that
start with the literal prefix `"watermark"`, not every unrecognized
`extra_args` key; other logits processors / vLLM internals may legitimately
use `extra_args` for their own purposes.

## 8. `ModelConfig.get_vocab_size()`

Source: `vllm/config/model.py`, lines 1119-1120:

```python
def get_vocab_size(self) -> int:
    return self.model_arch_config.vocab_size
```

This is the API `KGWLogitsProcessor.__init__` calls
(`vllm_config.model_config.get_vocab_size()`) rather than any tokenizer-
derived length. It is also what vLLM's own `SamplingParams` validation uses
for `logprobs`/`logit_bias` vocab bounds checks (`_validate_logprobs`,
`_validate_logit_bias`, same file `vllm/sampling_params.py` — both call
`model_config.get_vocab_size()`), so it is the same notion of "vocab size"
vLLM uses for logits bounds (`STATIC`). The detector must be configured
consistently with generation; the runtime effect of a mismatch is not
quantified here (`OPEN`).

## 9. Sources fetched (exact URLs)

- `https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/sample/logits_processor/interface.py`
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/sample/logits_processor/builtin.py`
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/sample/logits_processor/__init__.py`
- `https://raw.githubusercontent.com/vllm-project/vllm/v0.18.0/vllm/v1/sample/logits_processor/state.py`
- Full shallow clone at tag `v0.18.0` (commit `bcf2be96120005e9aea171927f85055a6a5c0cf6`)
  of `https://github.com/vllm-project/vllm.git`, from which the following
  were read directly (not fetched as individual raw URLs):
  `vllm/config/model.py`, `vllm/sampling_params.py`,
  `vllm/v1/engine/input_processor.py`, `vllm/entrypoints/utils.py`,
  `vllm/entrypoints/openai/server_utils.py`,
  `vllm/entrypoints/openai/chat_completion/protocol.py`,
  `vllm/entrypoints/openai/completion/protocol.py`,
  `vllm/engine/arg_utils.py`, `vllm/exceptions.py`.
- `https://docs.vllm.ai/en/latest/features/custom_logitsprocs/` (rendered
  page, fetched 2026-08-08)
- `https://api.github.com/repos/vllm-project/vllm/contents/vllm/v1/sample/logits_processor?ref=v0.18.0`
  (directory listing) and
  `https://api.github.com/repos/vllm-project/vllm/git/refs/tags/v0.18.0`
  (tag → commit SHA), both via `gh api`.

## 10. Plugin loading has no deduplication — entry points + FQCN flag double-load

`_load_custom_logitsprocs()` (vllm/v1/sample/logits_processor/__init__.py, ~line 160)
returns `_load_logitsprocs_plugins() + _load_logitsprocs_by_fqcns(logits_processors)`:

- `_load_logitsprocs_plugins()` loads **every** installed entry point in group
  `vllm.logits_processors` **unconditionally** — a pip-installed plugin package is
  active with no CLI flag at all.
- `_load_logitsprocs_by_fqcns()` appends every `--logits-processors` FQCN.
- The two lists are concatenated with **no dedup** — a class present both as an entry
  point and as a flag value is instantiated **twice**, and both instances run in
  `apply()` — for a bias-style watermark this doubles the effective delta
  (`STATIC`; pinned v0.18.0 source).

The class-list result was independently re-executed in a fresh CPU-only pod
using the v0.18.0 image digest and installed wheel (`EXECUTED`; [command and
raw output](../EXPERIMENTS.md#raw-evidence-vllm-plugin-double-load-independent-re-execution)):

```
>>> _load_custom_logitsprocs(["vllm_watermark.kgw.processor:KGWLogitsProcessor"])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']   # KGW twice!
>>> _load_custom_logitsprocs([])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor']                          # correct
```

**Rule for this repository: install the wheel and pass no
`--logits-processors` flag.** The Phase 1 watermark-on signal, overhead, and
temperature-0 interpretation from the flag-plus-entry-point window (effective
delta approximately 4) are superseded; the corrected single-instance run is
the evidence of record ([correction](../EXPERIMENTS.md#2026-08-08--correction-phase-1-ran-two-kgw-processor-instances-effective-delta-40),
[rerun](../EXPERIMENTS.md#2026-08-08--phase-1-corrected--phase-2-synthid-through-vllm-serve-closes-d8)).
