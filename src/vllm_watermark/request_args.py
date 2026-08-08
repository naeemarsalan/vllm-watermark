# SPDX-License-Identifier: Apache-2.0
"""Shared `vllm_xargs` / `SamplingParams.extra_args` parsing and validation
for every watermark-scheme `LogitsProcessor` plugin in this package
(currently `vllm_watermark.kgw.processor.KGWLogitsProcessor` and
`vllm_watermark.synthid.processor.SynthIDLogitsProcessor`).

Why this module exists (SCHEME-COORDINATION DESIGN)
-----------------------------------------------------
Two independent `LogitsProcessor` classes can both be loaded into the same
vLLM engine (`docs/api-notes-vllm-v0.18.0.md` §3: `build_logitsprocs()`
constructs one instance per configured class), and vLLM calls
`validate_params()` on **every** configured class for **every** incoming
request (`docs/api-notes-vllm-v0.18.0.md` §6:
`validate_logits_processors_parameters()` iterates
`cached_load_custom_logitsprocs(logits_processors)` and calls
`logits_procs.validate_params(sampling_params)` for each).

Both classes' `validate_params()` must therefore parse and validate the
exact same `watermark` / `watermark_key_id` / `watermark_scheme`
`vllm_xargs` keys IDENTICALLY. If the two implementations diverged even
slightly (e.g. one accepting `"yes"` as an alias for `watermark=on` and the
other not, or one recognizing `watermark_scheme` as a known key before the
other was updated to), the SAME request could be accepted by one
processor's `validate_params()` and rejected by the other's -- and vLLM has
no mechanism to reconcile that (each `validate_params()` call independently
determines whether the request 400s; see `docs/api-notes-vllm-v0.18.0.md`
§6 for the full call chain). This module is the single implementation both
classes import for that shared surface, so that failure mode is
structurally impossible rather than merely "tested to currently agree".

This module -- like `kgw/core.py` and `keys.py` -- stays vllm-free (no
`import vllm` at module scope); both `kgw/processor.py` and
`synthid/processor.py` (the vllm-facing wiring modules) import it.

Per-request `vllm_xargs` / `SamplingParams.extra_args` keys every
watermark-scheme processor in this package understands
-----------------------------------------------------------------
    watermark          "on"/"off" (or a JSON bool) -- overrides
                        VLLM_WATERMARK_DEFAULT for this request.
    watermark_key_id   str, non-empty -- which configured key to use
                        (default: `vllm_watermark.keys`' own default-key
                        resolution, i.e. WATERMARK_KEY_ID env or "default").
    watermark_scheme   "kgw" | "synthid" -- overrides VLLM_WATERMARK_SCHEME
                        for this request; selects which loaded processor
                        (if any) actually biases this request's logits.
                        Each processor module sets a class attribute
                        `SCHEME` ("kgw" / "synthid"); a row activates in
                        processor P iff (watermark enabled for the
                        request) AND (resolved scheme == P.SCHEME) -- see
                        `KGWLogitsProcessor._new_row_state()` /
                        `SynthIDLogitsProcessor._new_row_state()`.
Any other `watermark*`-prefixed key is rejected by `resolve_request()`
(called from every processor's `validate_params()`) as an unrecognized
argument -- fail loud on typos rather than silently ignoring them. This
mirrors `docs/api-notes-vllm-v0.18.0.md` §7's note that
`vllm.entrypoints.openai...` may itself add unrelated keys to `extra_args`
(e.g. `kv_transfer_params`), which is why only the `watermark`-prefixed
subset is policed here, not every unrecognized `extra_args` key.
"""

from __future__ import annotations

import os
from typing import Any

from vllm_watermark.keys import WatermarkKey, load_key

__all__ = [
    "KNOWN_WATERMARK_XARGS",
    "VALID_SCHEMES",
    "parse_watermark_flag",
    "parse_scheme",
    "resolve_request",
    "resolve_default_on",
    "resolve_default_scheme",
    "resolve_key_or_raise",
]

# The complete set of `watermark*`-prefixed extra_args keys ANY watermark
# processor in this package understands. See module docstring.
KNOWN_WATERMARK_XARGS = frozenset({"watermark", "watermark_key_id", "watermark_scheme"})

# Scheme identifiers this package implements a LogitsProcessor for. Each
# processor module sets a class attribute SCHEME to one of these.
VALID_SCHEMES = ("kgw", "synthid")

_DEFAULT_ON_ENV = "VLLM_WATERMARK_DEFAULT"
_DEFAULT_ON_ENV_DEFAULT = "off"
_DEFAULT_SCHEME_ENV = "VLLM_WATERMARK_SCHEME"
_DEFAULT_SCHEME_ENV_DEFAULT = "kgw"


def parse_watermark_flag(value: Any) -> bool:
    """Parse the `watermark` extra_arg / VLLM_WATERMARK_DEFAULT env value.

    Accepts a real bool (JSON `true`/`false` deserializes to Python bool via
    vllm_xargs' `dict[str, str | int | float | list[...]]` typing -- see
    `docs/api-notes-vllm-v0.18.0.md` §7) or a string in {"on","off","true",
    "false","1","0","yes","no"} (case-insensitive). Anything else raises
    ValueError, which is exactly the error type `validate_params()` and
    `__init__()` need to surface a clear rejection (a plain `ValueError`
    raised from `validate_params()` becomes an HTTP 400 -- see
    `docs/api-notes-vllm-v0.18.0.md` §6).

    Moved here (from `kgw/processor.py`, formerly module-private
    `_parse_watermark_flag`) so `synthid/processor.py` uses the identical
    implementation -- see module docstring "SCHEME-COORDINATION DESIGN".
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("on", "true", "1", "yes"):
            return True
        if v in ("off", "false", "0", "no"):
            return False
    raise ValueError(
        f"watermark must be 'on'/'off' (or a boolean), got {value!r}"
    )


def parse_scheme(value: Any) -> str:
    """Parse/validate the `watermark_scheme` extra_arg / VLLM_WATERMARK_SCHEME
    env value.

    Must be a string (case-insensitive, surrounding whitespace stripped)
    equal to one of VALID_SCHEMES ("kgw", "synthid"). Raises ValueError
    otherwise -- see `parse_watermark_flag()` docstring for why ValueError
    is the right type here too.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in VALID_SCHEMES:
            return v
    raise ValueError(
        f"watermark_scheme must be one of {VALID_SCHEMES!r}, got {value!r}"
    )


def resolve_request(
    extra_args: "dict[str, Any] | None",
    default_on: bool,
    default_scheme: str,
) -> "tuple[bool, str, str | None]":
    """Parse+validate one request's `watermark_*` extra_args in one pass.

    This is the single source of truth for the known-keys set
    (`KNOWN_WATERMARK_XARGS`) and the unknown-`watermark*`-key rejection --
    see module docstring. Both `KGWLogitsProcessor` and
    `SynthIDLogitsProcessor` call this from BOTH `validate_params()`
    (classmethod, request-validation time) and `_new_row_state()`
    (instance method, batch-update time) so the two call sites can never
    disagree about what a given `extra_args` dict means.

    Args:
        extra_args: `SamplingParams.extra_args` (or None).
        default_on: effective watermark on/off when the request omits the
            `watermark` key. Callers doing pure *validation* (no
            keys-material side effects wanted for the implicit-default
            case -- see `kgw/processor.py` `validate_params()` docstring
            "Per-request error surfacing" for why the implicit-default case
            is deliberately NOT resolved-key-checked at request-validation
            time) pass `False` here regardless of the processor's actual
            configured default, which reproduces exactly that "only check
            resolvability for an *explicit* watermark=on" behavior.
        default_scheme: effective scheme when the request omits
            `watermark_scheme`. Expected to already be a valid scheme
            (e.g. obtained via `resolve_default_scheme()`) -- this function
            does not re-validate `default_scheme` itself, only an
            explicitly-provided `watermark_scheme` extra_arg.

    Returns:
        `(enabled, scheme, key_id)`:
          enabled -- bool, effective watermark on/off for this request.
          scheme  -- "kgw" or "synthid", effective scheme for this request.
          key_id  -- explicit `watermark_key_id` extra_arg, or None (caller
              resolves None against its own default-key logic -- see
              `vllm_watermark.keys.load_key(key_id=None)`).

    Raises:
        ValueError: unrecognized `watermark*`-prefixed key; malformed
            `watermark` value; malformed `watermark_scheme` value; empty or
            non-string `watermark_key_id`. Callers that only need to
            VALIDATE (e.g. `validate_params()`) can call this and discard
            the returned tuple -- the raising behavior alone is the
            validation.
    """
    extra_args = extra_args or {}

    watermark_keys_present = {k for k in extra_args if k.startswith("watermark")}
    unknown = watermark_keys_present - KNOWN_WATERMARK_XARGS
    if unknown:
        raise ValueError(
            f"Unknown watermark_* extra_args key(s) {sorted(unknown)}; "
            f"known keys are {sorted(KNOWN_WATERMARK_XARGS)}"
        )

    watermark_flag = extra_args.get("watermark")
    enabled = default_on if watermark_flag is None else parse_watermark_flag(watermark_flag)

    scheme_value = extra_args.get("watermark_scheme")
    scheme = default_scheme if scheme_value is None else parse_scheme(scheme_value)

    key_id = extra_args.get("watermark_key_id")
    if key_id is not None and (not isinstance(key_id, str) or not key_id.strip()):
        raise ValueError(f"watermark_key_id must be a non-empty string, got {key_id!r}")

    return enabled, scheme, key_id


def resolve_default_on(
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> bool:
    """VLLM_WATERMARK_DEFAULT env var -> bool, default "off".

    Shared by both processors' `__init__()` so their process-global default
    matches exactly given the same environment -- see module docstring.
    """
    if env is None:
        env = os.environ
    return parse_watermark_flag(env.get(_DEFAULT_ON_ENV, _DEFAULT_ON_ENV_DEFAULT))


def resolve_default_scheme(
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> str:
    """VLLM_WATERMARK_SCHEME env var -> "kgw"/"synthid", default "kgw".

    Shared by both processors' `__init__()` (process-global default scheme)
    and available to `validate_params()` implementations as the
    `default_scheme` argument to `resolve_request()` -- using this instead
    of reading the env var ad hoc means a misconfigured env value
    (`VLLM_WATERMARK_SCHEME=bogus`) is rejected identically everywhere it
    is read, and a request that omits `watermark_scheme` resolves to the
    same effective scheme regardless of which processor is asked.
    """
    if env is None:
        env = os.environ
    return parse_scheme(env.get(_DEFAULT_SCHEME_ENV, _DEFAULT_SCHEME_ENV_DEFAULT))


def resolve_key_or_raise(key_id: "str | None", *, context: str) -> WatermarkKey:
    """Thin wrapper around `vllm_watermark.keys.load_key()` that normalizes
    its failure modes (RuntimeError: nothing configured at all; KeyError:
    key_id not among configured keys; ValueError: malformed
    WATERMARK_KEYS/WATERMARK_KEY env value) to a single ValueError,
    suitable for both `validate_params()` (must raise ValueError per the
    vLLM contract -- see `parse_watermark_flag()` docstring) and `__init__()`
    defensive checks. `context` is prepended to the message so the API
    caller sees *why* the key lookup was attempted, not just keys.py's
    generic "no key configured" text.

    Moved here (from `kgw/processor.py`, formerly module-private
    `_resolve_key`) so `synthid/processor.py` shares the identical
    normalization -- see module docstring "SCHEME-COORDINATION DESIGN".
    """
    try:
        return load_key(key_id=key_id)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise ValueError(f"{context}: {exc}") from exc
