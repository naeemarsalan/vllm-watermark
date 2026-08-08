# vLLM v0.18.0 custom-LogitsProcessor API notes

**STATIC.** Everything in this page was verified by fetching the actual
v0.18.0-tagged vLLM source and the live vLLM docs page, not from memory or
from `docs/facts.md`'s prior (also STATIC, `main`-branch-sourced) entries.
Fetched 2026-08-08.

```
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git
-> resolved commit: bcf2be96120005e9aea171927f85055a6a5c0cf6
   (confirmed separately: gh api repos/vllm-project/vllm/git/refs/tags/v0.18.0
    -> object.sha == bcf2be96120005e9aea171927f85055a6a5c0cf6)
```

All `vllm/...` file paths and line numbers below are relative to that
commit. All cited vLLM files carry `# SPDX-License-Identifier: Apache-2.0`
/ `# SPDX-FileCopyrightText: Copyright contributors to the vLLM project`.

This page supports `src/vllm_watermark/kgw/processor.py` (Task B). It does
**not** cover the KGW algorithm itself (Task A: `kgw/core.py`, `kgw/detector.py`,
`keys.py`) or the detection-service contract (Phase 3).

---

## 1. `LogitsProcessor` ABC

Source: `vllm/v1/sample/logits_processor/interface.py` (106 lines), re-exported
publicly from `vllm.v1.sample.logits_processor` (see §3 `__all__`).

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

We could not find any builtin or documented-example processor that
actually implements removed-before-added. `KGWLogitsProcessor.update_state()`
follows the code (added, removed, moved), not the prose, because that is
what every shipped/tested reference implementation and the docs' own
worked example do. This is very likely safe either way for genuinely
disjoint indices, and the reason added-before-removed doesn't break replace
-in-place ("Added or moved requests may replace existing requests with the
same index" per the `BatchUpdate` docstring's own NOTE) is that an index
reused within one step appears in `added` but is *not* separately listed in
`removed` for that step — i.e. `removed` and `added` are not expected to
target the same index within a single `BatchUpdate`. We did not find an
explicit statement of that invariant in the source; it is inferred from the
NOTE plus the fact that every reference implementation relies on it.

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

CLI flag: the actual `argparse` registration is (verified in a separate
v0.18.0 checkout of `vllm/engine/arg_utils.py`, line 749):

```python
model_group.add_argument("--logits-processors", **model_kwargs["logits_processors"])
```

i.e. **`--logits-processors`** (hyphenated). The live docs page (§5) shows
`--logits_processors` (underscored) in its `vllm serve` examples in two
places — this looks like a docs prose/typo inconsistency with the actual
registered flag name; we did not find an underscore-flag alias registered
anywhere in `arg_utils.py`. Use the hyphenated form.

`build_logitsprocs()` (same file, ~line 184-217) is what actually
constructs one instance per configured logits-processor class at engine
init: `BUILTIN_LOGITS_PROCESSORS` (`MinTokensLogitsProcessor`,
`LogitBiasLogitsProcessor`, `MinPLogitsProcessor`) plus
`custom_logitsprocs_classes` (from `_load_custom_logitsprocs`, entry
points + FQCNs), each called as `ctor(vllm_config, device, is_pin_memory)`
— exactly one instance per class per engine, which is why
`KGWLogitsProcessor.validate_params()` (a `@classmethod`, no `self`) not
being able to see a *specific instance's* cached key config is a
non-issue in practice: there is only ever one instance to have cached
anything, and it's simplest for `validate_params()` to just re-derive the
same (process-global, env-sourced) answer independently. See §4.

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
any custom logits processor (entry-point or FQCN) is present.

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

This is the docs-level confirmation of the mechanism traced at the source
level in §6 below. The page's example `DummyLogitsProcessor` /
`WrappedPerReqLogitsProcessor` snippets, `vllm_xargs` REST/SDK examples,
and FQCN/entry-point loading instructions are otherwise consistent with
what we independently verified from source (§1-§4), except for the two
discrepancies flagged above (§2 processing order, §3 CLI flag spelling).

## 6. Per-request error surfacing: how a `validate_params()` `ValueError`
becomes an HTTP 400

This is the specific question the Task B brief raised: *"validate_params
can't see env? it's a classmethod — if so, do the rejection in
update_state->raise? Check what the builtin processors do for
request-time errors and pick the mechanism that actually surfaces a 4xx to
the API caller."*

**Finding: `validate_params()` raising a plain `ValueError` is exactly the
mechanism vLLM uses, and it does surface as an HTTP 400.** No builtin
processor defers this kind of rejection to `update_state()` — searching
`builtin.py`, none of `MinPLogitsProcessor`, `LogitBiasLogitsProcessor`,
`MinTokensLogitsProcessor` override `validate_params()` at all (they rely
on the no-op base-class default, since their arguments — `min_p`,
`logit_bias`, `min_tokens` — are plain `SamplingParams` fields already
validated elsewhere); the docs page's own worked examples (§5) both
implement `validate_params()` to raise `ValueError` for a bad
`extra_args["target_token"]`, which is the pattern `KGWLogitsProcessor`
follows. The classmethod-vs-env question resolves simply: `os.environ`
(and, in our case, `vllm_watermark.keys.load_key`/`load_keys`, which reads
`os.environ`) is process-global state, readable from anywhere including a
`@classmethod` — the classmethod just can't reach a *specific instance's*
cached state, which is not needed here (§3, one instance per engine).

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
   This is the call site of our `KGWLogitsProcessor.validate_params(cls,
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
matching key is configured. That `ValueError` is guaranteed by the chain
above to reach the API caller as an HTTP 400 with a `BadRequestError` body,
*before* generation starts — not a silent fall-through to unwatermarked
output, and not a failure buried inside `update_state()`/`apply()` where it
would abort the whole batch rather than just the one bad request.

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
vLLM itself uses to size the logits tensor's last dimension — consistent
with the `vllm_watermark.kgw.core` requirement (Task A) that generation and
detection use the identical `vocab_size` value or scores silently
degrade to near-zero.

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

## 8. Plugin loading has NO deduplication — entry points + FQCN flag double-load (EXECUTED)

`_load_custom_logitsprocs()` (vllm/v1/sample/logits_processor/__init__.py, ~line 160)
returns `_load_logitsprocs_plugins() + _load_logitsprocs_by_fqcns(logits_processors)`:

- `_load_logitsprocs_plugins()` loads **every** installed entry point in group
  `vllm.logits_processors` **unconditionally** — a pip-installed plugin package is
  active with no CLI flag at all.
- `_load_logitsprocs_by_fqcns()` appends every `--logits-processors` FQCN.
- The two lists are concatenated with **no dedup** — a class present both as an entry
  point and as a flag value is instantiated **twice**, and both instances run in
  `apply()` — for a bias-style watermark this silently doubles the effective delta.

EXECUTED in the serving pod (2026-08-08, vLLM v0.18.0):

```
>>> _load_custom_logitsprocs(["vllm_watermark.kgw.processor:KGWLogitsProcessor"])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor', 'KGWLogitsProcessor']   # KGW twice!
>>> _load_custom_logitsprocs([])
['KGWLogitsProcessor', 'SynthIDLogitsProcessor']                          # correct
```

**Rule for this repo: install the wheel and pass NO `--logits-processors` flag.**
The Phase 1 measurements taken with flag+entry-point (effective delta 4.0) were
re-taken single-instance — see EXPERIMENTS.md 2026-08-08 correction entry.
